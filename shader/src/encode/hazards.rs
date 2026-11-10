//! Hardware hazard workarounds for gfx11 and gfx12.
//!
//! Every rule follows the PS5 LLVM fork (commit 3c1efa15): `GCNHazardRecognizer::fixHazards`
//! and, for gfx12, `AMDGPUWaitSGPRHazards`, including which hardware needs each one:
//!
//! | LLVM rule                        | gfx11.0 | gfx11.5  | gfx12 | fix                              |
//! |----------------------------------|---------|----------|-------|----------------------------------|
//! | `fixVcmpxPermlaneHazards`        | yes     | yes      | yes   | `v_mov_b32 src0, src0` before    |
//! | `fixLdsDirectVMEMHazard`         | yes     | yes      | (1)   | `s_waitcnt_depctr vm_vsrc(0)`    |
//! | `fixVALUPartialForwardingHazard` | wave64  | wave64   | no    | `s_waitcnt_depctr va_vdst(0)`    |
//! | `fixVALUTransUseHazard`          | yes     | no       | no    | `s_waitcnt_depctr va_vdst(0)`    |
//! | `fixVALUMaskWriteHazard`         | wave64  | wave64   | no    | `s_waitcnt_depctr sa_sdst(0)` after |
//! | `fixRequiredExportPriority`      | no      | graphics | no    | `s_setprio` around export runs   |
//! | `AMDGPUWaitSGPRHazards`          | no      | no       | yes   | `s_wait_alu`                     |
//!
//! (1) gfx12 `ds_param_load` waits through its own `wait_vm_vsrc` field, which the encoder leaves
//! at 0 (wait). `lds_param_load` always encodes `wait_vdst` 0, the safe value of
//! `fixLdsDirectVALUHazard`.
//!
//! Searches walk backward from an instruction through predecessor blocks like LLVM's
//! `getWaitStatesSince` and `hasHazard`, seeing the fixes already inserted before it.

use std::cell::Cell;

use crate::ir::opcodes::flag;
use crate::ir::{
    Fields, Instruction, InstructionFamily as F, Operand, RegFile, RegRange, VopExt,
    SCALAR_EXEC_HI, SCALAR_EXEC_LO, SCALAR_M0, SCALAR_VCC_HI, SCALAR_VCC_LO,
};
use crate::lowering::{derive, LoweredProgram};
use crate::regalloc::scalar_mask;
use crate::{ShaderStage, TargetIsa};

pub const SOPP_GFX11_WAITCNT_DEPCTR: u16 = 0x08;
const SOPP_GFX11_NOP: u16 = 0x00;
const SOPP_GFX11_WAITCNT: u16 = 0x09;
const SOPP_GFX11_ENDPGM: u16 = 0x30;
const SOPP_GFX11_ENDPGM_SAVED: u16 = 0x31;
const SOPP_GFX11_ENDPGM_ORDERED_PS_DONE: u16 = 0x32;
const SOPP_GFX11_SETPRIO: u16 = 0x35;
const SOPK_GFX11_WAITCNT_EXPCNT: u16 = 0x1a;
const SOP1_GFX11_SETPC_B64: u16 = 0x48;
const SOP1_GFX11_SWAPPC_B64: u16 = 0x49;
const SOP1_GFX11_RFE_B64: u16 = 0x4a;
const SOPK_GFX11_CALL_B64: u16 = 0x14;
const VOP1_NOP: u16 = 0x00;
const VOP1_MOV_B32: u16 = 0x01;
const VOP3_VOP1_NOP: u16 = 0x180;
const VOP3_GFX11_PERMLANE16_B32: u16 = 0x25b;
const VOP3_GFX11_PERMLANEX16_B32: u16 = 0x25c;
const DS_NOP: u16 = 0x14;

/// `s_waitcnt_depctr` immediates, each 0xffff with one field cleared, as LLVM inserts them.
pub const DEPCTR_VM_VSRC_0: u16 = 0xffe3;
pub const DEPCTR_VA_VDST_0: u16 = 0x0fff;
pub const DEPCTR_SA_SDST_0: u16 = 0xfffe;
pub const DEPCTR_VA_SDST_0: u16 = 0xf1ff;
pub const DEPCTR_VA_VCC_0: u16 = 0xfffd;
const DEPCTR_NONE: u16 = 0xffff;

const NORMAL_PRIORITY: u16 = 2;
const POST_EXPORT_PRIORITY: u16 = 0;
const MAX_PRIORITY: u16 = 3;

pub fn insert(program: &mut LoweredProgram) {
    if !program.target.is_gfx11_plus() || program.blocks.is_empty() {
        return;
    }
    let rules = Rules {
        target: program.target,
        wave64: program.wave64,
    };
    if program.target == TargetIsa::Gfx115 && program.stage != ShaderStage::Compute {
        export_priority(program);
    }
    let succ = program.successors();
    let preds = predecessors(&succ);
    let mut blocks: Vec<Vec<Instruction>> = program
        .blocks
        .iter_mut()
        .map(|b| std::mem::take(&mut b.instructions))
        .collect();
    let visits = Visits::new(blocks.len());
    for id in 0..blocks.len() {
        let mut at = 0;
        while at < blocks[id].len() {
            let inst = blocks[id][at].clone();
            for rule in BEFORE {
                let view = View {
                    blocks: &blocks,
                    preds: &preds,
                    block: id,
                    at,
                    visits: &visits,
                };
                if let Some(fix) = rule(&rules, &view, &inst) {
                    blocks[id].insert(at, fix);
                    at += 1;
                }
            }
            let view = View {
                blocks: &blocks,
                preds: &preds,
                block: id,
                at,
                visits: &visits,
            };
            let after = rules.mask_write_fix(&view, &inst);
            at += 1;
            if let Some(fix) = after {
                blocks[id].insert(at, fix);
            }
        }
    }
    if program.target.is_gfx12() {
        sgpr_hazards(&mut blocks, &succ, program.wave64);
    }
    for (block, instructions) in program.blocks.iter_mut().zip(blocks) {
        block.instructions = instructions;
    }
}

type Rule = fn(&Rules, &View, &Instruction) -> Option<Instruction>;

/// The rules that insert a fix before the instruction, in `GCNHazardRecognizer::fixHazards`
/// order; each sees the fixes of the rules before it.
const BEFORE: [Rule; 4] = [
    Rules::vcmpx_permlane_fix,
    Rules::lds_direct_vmem_fix,
    Rules::partial_forwarding_fix,
    Rules::trans_use_fix,
];

/// The instructions before position `at` of `block`, as the walk has rewritten them so far.
struct View<'a> {
    blocks: &'a [Vec<Instruction>],
    preds: &'a [Vec<usize>],
    block: usize,
    at: usize,
    visits: &'a Visits,
}

/// The blocks a search has walked: those whose mark equals the search's generation, so a new
/// search starts with none walked without clearing the marks.
struct Visits {
    marks: Vec<Cell<u32>>,
    generation: Cell<u32>,
}

impl Visits {
    fn new(blocks: usize) -> Self {
        Self {
            marks: vec![Cell::new(0); blocks],
            generation: Cell::new(0),
        }
    }

    fn start(&self) -> u32 {
        let generation = self.generation.get().wrapping_add(1).max(1);
        if generation == 1 {
            self.marks.iter().for_each(|m| m.set(0));
        }
        self.generation.set(generation);
        generation
    }

    /// Marks `block` walked; false when this search has walked it already.
    fn first(&self, block: usize, generation: u32) -> bool {
        self.marks[block].replace(generation) != generation
    }
}

enum Step {
    Found,
    Expired,
    Continue,
}

impl View<'_> {
    /// LLVM's `getWaitStatesSince` as a yes/no question: walking backward, does any path reach
    /// a `hazard` instruction before an `expired` one?
    fn hazard_before(
        &self,
        hazard: &dyn Fn(&Instruction) -> bool,
        expired: &dyn Fn(&Instruction) -> bool,
    ) -> bool {
        self.search_before((), &|_, inst| {
            if hazard(inst) {
                Step::Found
            } else if expired(inst) {
                Step::Expired
            } else {
                Step::Continue
            }
        })
    }

    /// LLVM's `hasHazard`: `step` classifies each instruction, walking backward, and updates
    /// the path state when the search continues past it.
    fn search_before<S: Clone>(
        &self,
        state: S,
        step: &dyn Fn(&mut S, &Instruction) -> Step,
    ) -> bool {
        let generation = self.visits.start();
        self.walk(self.block, self.at, state, step, generation)
    }

    fn walk<S: Clone>(
        &self,
        block: usize,
        end: usize,
        mut state: S,
        step: &dyn Fn(&mut S, &Instruction) -> Step,
        generation: u32,
    ) -> bool {
        for inst in self.blocks[block][..end].iter().rev() {
            match step(&mut state, inst) {
                Step::Found => return true,
                Step::Expired => return false,
                Step::Continue => {}
            }
        }
        self.preds[block].iter().any(|&pred| {
            self.visits.first(pred, generation)
                && self.walk(
                    pred,
                    self.blocks[pred].len(),
                    state.clone(),
                    step,
                    generation,
                )
        })
    }
}

struct Rules {
    target: TargetIsa,
    wave64: bool,
}

impl Rules {
    /// `fixVcmpxPermlaneHazards` (every gfx11+): a `v_cmpx` writing EXEC before
    /// `v_permlane*16` with no VALU other than `v_nop` in between. The fix is a VALU that the
    /// SQ keeps: `v_mov_b32` of the permlane's own source onto itself.
    fn vcmpx_permlane_fix(&self, view: &View, inst: &Instruction) -> Option<Instruction> {
        let Fields::Vop3(v) = inst.fields else {
            return None;
        };
        let permlane = matches!(
            inst.opcode,
            VOP3_GFX11_PERMLANE16_B32 | VOP3_GFX11_PERMLANEX16_B32
        );
        let expired = |i: &Instruction| is_valu(i) && !is_vnop(i);
        if !permlane || !v.src[0].is_vgpr() || !view.hazard_before(&is_vcmpx, &expired) {
            return None;
        }
        let fields = Fields::Vop1 {
            vdst: v.src[0],
            src0: v.src[0],
            ext: VopExt::None,
        };
        Some(derive(inst, F::Vop1, VOP1_MOV_B32, 0, fields))
    }

    fn lds_direct_vmem_fix(&self, view: &View, inst: &Instruction) -> Option<Instruction> {
        (self.target.is_gfx11() && lds_direct_vmem_hazard(view, inst))
            .then(|| depctr(inst, DEPCTR_VM_VSRC_0))
    }

    fn partial_forwarding_fix(&self, view: &View, inst: &Instruction) -> Option<Instruction> {
        (self.target.is_gfx11() && self.wave64 && partial_forwarding_hazard(view, inst))
            .then(|| depctr(inst, DEPCTR_VA_VDST_0))
    }

    fn trans_use_fix(&self, view: &View, inst: &Instruction) -> Option<Instruction> {
        (self.target == TargetIsa::Gfx11 && trans_use_hazard(view, inst))
            .then(|| depctr(inst, DEPCTR_VA_VDST_0))
    }

    /// The `fixVALUMaskWriteHazard` wait goes after the SALU write.
    fn mask_write_fix(&self, view: &View, inst: &Instruction) -> Option<Instruction> {
        (self.target.is_gfx11() && self.wave64 && mask_write_hazard(view, inst))
            .then(|| depctr(inst, DEPCTR_SA_SDST_0))
    }
}

/// `fixLdsDirectVMEMHazard`: `lds_param_load` writing a VGPR that an earlier VMEM, FLAT or DS
/// instruction may still read or write.
fn lds_direct_vmem_hazard(view: &View, inst: &Instruction) -> bool {
    let Fields::Ldsdir(l) = inst.fields else {
        return false;
    };
    let vdst = u16::from(l.vdst);
    let touches = move |i: &Instruction| {
        if !(is_vmem(i) || is_ds(i)) {
            return false;
        }
        let mut touched = false;
        let mut check = |r: RegRange| touched |= r.contains(RegFile::Vector, vdst);
        i.each_use(false, &mut check);
        i.each_def(false, &mut check);
        touched
    };
    view.hazard_before(&touches, &|i: &Instruction| {
        is_valu(i)
            || i.family == F::Exp
            || is_waitcnt_zero(i)
            || depctr_of(i).is_some_and(|d| d & 0x1c == 0)
    })
}

/// `fixVALUTransUseHazard` (gfx11.0): a VALU reading a transcendental's result with at most
/// five VALUs, one of them transcendental, in between.
fn trans_use_hazard(view: &View, inst: &Instruction) -> bool {
    if !is_valu(inst) {
        return false;
    }
    let sources = vgpr_uses(inst);
    if sources == [0; 2] {
        return false;
    }
    #[derive(Clone, Copy)]
    struct Window {
        valus: u32,
        trans: u32,
    }
    view.search_before(Window { valus: 0, trans: 0 }, &|w, i| {
        if w.valus > 5 || w.trans > 1 {
            return Step::Expired;
        }
        if is_vmem(i) || is_ds(i) || i.family == F::Exp || depctr_of(i) == Some(DEPCTR_VA_VDST_0) {
            return Step::Expired;
        }
        if i.has_flag(flag::TRANS) && overlaps(vgpr_defs(i), sources) {
            return Step::Found;
        }
        if is_valu(i) {
            w.valus += 1;
        }
        if i.has_flag(flag::TRANS) {
            w.trans += 1;
        }
        Step::Continue
    })
}

/// `fixVALUPartialForwardingHazard` (gfx11 wave64): a VALU reading Va and Vb where
///   Va <- VALU; intv1; EXEC <- SALU; intv2; Vb <- VALU; intv3; VALU reads Va, Vb
/// with intv1 + intv2 <= 2 VALUs and intv3 <= 4 VALUs.
fn partial_forwarding_hazard(view: &View, inst: &Instruction) -> bool {
    if !is_valu(inst) {
        return false;
    }
    let mut sources: Vec<RegRange> = Vec::new();
    for range in explicit_vgpr_uses(inst) {
        if !sources.contains(&range) {
            sources.push(range);
        }
    }
    if sources.len() <= 1 {
        return false;
    }
    #[derive(Clone)]
    struct Positions {
        defs: Vec<Option<u32>>,
        exec: Option<u32>,
        valus: u32,
    }
    let start = Positions {
        defs: vec![None; sources.len()],
        exec: None,
        valus: 0,
    };
    return view.search_before(start, &|s, i| {
        let verdict = forwarding_step(s, i, &sources);
        if matches!(verdict, Step::Continue) && is_valu(i) {
            s.valus += 1;
        }
        verdict
    });

    /// LLVM's `IsHazardFn` for this rule; `s.valus` counts the VALUs between `i` and the reader.
    fn forwarding_step(s: &mut Positions, i: &Instruction, sources: &[RegRange]) -> Step {
        const INTV1_PLUS_2_MAX_VALUS: u32 = 2;
        const INTV3_MAX_VALUS: u32 = 4;
        const NO_HAZARD_VALU_WAIT_STATES: u32 = 6 + 2;
        if s.valus > NO_HAZARD_VALU_WAIT_STATES {
            return Step::Expired;
        }
        if is_vmem(i)
            || is_ds(i)
            || i.family == F::Exp
            || depctr_of(i).is_some_and(|d| d >> 12 == 0)
        {
            return Step::Expired;
        }
        let mut changed = false;
        if is_valu(i) {
            let valus = s.valus;
            i.each_def(true, |d| {
                for (slot, source) in s.defs.iter_mut().zip(sources) {
                    if slot.is_none() && ranges_overlap(d, *source) {
                        *slot = Some(valus);
                        changed = true;
                    }
                }
            });
        } else if is_salu(i)
            && s.exec.is_none()
            && s.defs.iter().any(Option::is_some)
            && writes_exec(i)
        {
            s.exec = Some(s.valus);
            changed = true;
        }
        if s.valus > INTV3_MAX_VALUS && s.defs.iter().all(Option::is_none) {
            return Step::Expired;
        }
        let Some(exec) = s.exec.filter(|_| changed) else {
            return Step::Continue;
        };
        let pre = s
            .defs
            .iter()
            .flatten()
            .filter(|&&p| p >= exec)
            .min()
            .copied();
        let Some(post) = s
            .defs
            .iter()
            .flatten()
            .filter(|&&p| p < exec)
            .min()
            .copied()
        else {
            return Step::Continue;
        };
        let intv2 = exec - post - 1;
        if post > INTV3_MAX_VALUS || intv2 > INTV1_PLUS_2_MAX_VALUS {
            return Step::Expired;
        }
        let Some(pre) = pre else {
            return Step::Continue;
        };
        let intv1 = pre - exec;
        if intv1 > INTV1_PLUS_2_MAX_VALUS || intv1 + intv2 > INTV1_PLUS_2_MAX_VALUS {
            return Step::Expired;
        }
        Step::Found
    }
}

/// `fixVALUMaskWriteHazard` (gfx11 wave64): an SALU writing an SGPR that an earlier VALU read
/// as its lane mask (`v_cndmask`, carry-in adds and subtracts, `v_div_fmas`).
fn mask_write_hazard(view: &View, inst: &Instruction) -> bool {
    let Some(written) = salu_destination(inst) else {
        return false;
    };
    let mask_read =
        move |i: &Instruction| lane_mask_source(i).is_some_and(|m| ranges_overlap(m, written));
    view.hazard_before(&mask_read, &|i: &Instruction| {
        depctr_of(i).is_some_and(|d| d & 1 == 0) || (is_valu(i) && reads_sgpr_or_literal(i))
    })
}

/// `fixRequiredExportPriority` (gfx11.5 graphics): runs at normal priority 2 from the entry,
/// and after each run of exports drops to priority 0 until the exports leave the wave.
fn export_priority(program: &mut LoweredProgram) {
    let has_calls = program.instructions().any(is_call);
    let mut raise_entry = false;
    for block in &mut program.blocks {
        let mut out = Vec::with_capacity(block.instructions.len());
        let mut iter = block.instructions.iter().peekable();
        while let Some(inst) = iter.next() {
            let mut inst = inst.clone();
            let after_export = out.last().is_some_and(|p: &Instruction| p.family == F::Exp);
            if let Some(prio) = setprio_of(&inst) {
                let in_workaround = prio == POST_EXPORT_PRIORITY && after_export;
                if !in_workaround && prio < NORMAL_PRIORITY {
                    inst.fields = Fields::Sopp {
                        simm16: (prio + NORMAL_PRIORITY).min(MAX_PRIORITY),
                    };
                }
            }
            let ends = inst.family == F::Sopp
                && matches!(
                    inst.opcode,
                    SOPP_GFX11_ENDPGM | SOPP_GFX11_ENDPGM_SAVED | SOPP_GFX11_ENDPGM_ORDERED_PS_DONE
                );
            raise_entry |= ends && has_calls;
            let export = inst.family == F::Exp;
            out.push(inst);
            if !export {
                continue;
            }
            raise_entry = true;
            let next = iter.peek().copied();
            let applied = next
                .is_some_and(|n| n.family == F::Exp || setprio_of(n) == Some(POST_EXPORT_PRIORITY));
            if applied {
                continue;
            }
            let origin = out.last().cloned().expect("pushed above");
            let end_of_shader =
                next.is_some_and(|n| n.family == F::Sopp && n.opcode == SOPP_GFX11_ENDPGM);
            out.push(sopp(&origin, SOPP_GFX11_SETPRIO, POST_EXPORT_PRIORITY));
            if !end_of_shader {
                let wait = Fields::Sopk {
                    sdst: Operand::Null,
                    simm16: 0,
                    imm32: None,
                };
                out.push(derive(
                    &origin,
                    F::Sopk,
                    SOPK_GFX11_WAITCNT_EXPCNT,
                    flag::NO_DST | flag::READS_DST,
                    wait,
                ));
            }
            out.push(sopp(&origin, SOPP_GFX11_NOP, 0));
            out.push(sopp(&origin, SOPP_GFX11_NOP, 0));
            if !end_of_shader {
                out.push(sopp(&origin, SOPP_GFX11_SETPRIO, NORMAL_PRIORITY));
            }
        }
        block.instructions = out;
    }
    let entry = &mut program.blocks[0].instructions;
    let raised = entry
        .first()
        .and_then(setprio_of)
        .is_some_and(|p| p >= NORMAL_PRIORITY);
    if raise_entry && !raised {
        if let Some(first) = entry.first().cloned() {
            entry.insert(0, sopp(&first, SOPP_GFX11_SETPRIO, NORMAL_PRIORITY));
        }
    }
}

/// gfx12 `AMDGPUWaitSGPRHazards`: once a VALU has read an SGPR pair, a later SALU write to it
/// must commit before any read (`sa_sdst`), a VALU write before a VALU read (`va_sdst`), and a
/// VALU write of VCC before a VALU read of VCC (`va_vcc`).
fn sgpr_hazards(blocks: &mut [Vec<Instruction>], succ: &[Vec<usize>], wave64: bool) {
    let mut entry = vec![SgprState::default(); blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for id in 0..blocks.len() {
            let mut state = entry[id].clone();
            for inst in &blocks[id] {
                state.step(inst, wave64);
            }
            for &s in &succ[id] {
                let merged = entry[s].merge(&state);
                if merged != entry[s] {
                    entry[s] = merged;
                    changed = true;
                }
            }
        }
    }
    for (id, block) in blocks.iter_mut().enumerate() {
        let mut state = entry[id].clone();
        let mut out: Vec<Instruction> = Vec::with_capacity(block.len());
        for inst in block.drain(..) {
            let wait = state.step(&inst, wave64);
            if wait != DEPCTR_NONE {
                match out.last_mut() {
                    Some(prev) if depctr_of(prev).is_some() => {
                        if let Fields::Sopp { simm16 } = &mut prev.fields {
                            *simm16 &= wait;
                        }
                    }
                    _ => out.push(depctr(&inst, wait)),
                }
            }
            out.push(inst);
        }
        *block = out;
    }
}

const VCC_SALU: u8 = 1;
const VCC_VALU: u8 = 2;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct SgprState {
    /// SGPR pairs a VALU has read.
    tracked: u64,
    /// SGPRs with uncommitted SALU writes.
    salu: u128,
    /// SGPRs with uncommitted VALU writes.
    valu: u128,
    /// Uncommitted writers of VCC.
    vcc: u8,
    ds_nops: u32,
}

impl SgprState {
    fn merge(&self, other: &Self) -> Self {
        Self {
            tracked: self.tracked | other.tracked,
            salu: self.salu | other.salu,
            valu: self.valu | other.valu,
            vcc: self.vcc | other.vcc,
            ds_nops: 0,
        }
    }

    /// Applies `inst` and returns the `s_wait_alu` immediate it needs first (0xffff for none).
    fn step(&mut self, inst: &Instruction, wave64: bool) -> u16 {
        if inst.family == F::Ds && inst.opcode == DS_NOP {
            self.ds_nops += 1;
            if self.ds_nops >= if wave64 { 8 } else { 4 } {
                self.tracked = 0;
            }
            return DEPCTR_NONE;
        }
        self.ds_nops = 0;
        if matches!(inst.family, F::Mubuf | F::Mtbuf | F::Mimg | F::Smem) {
            self.vcc = 0;
            self.salu = 0;
            self.valu = 0;
            return DEPCTR_NONE;
        }
        if let Some(d) = depctr_of(inst) {
            if d & 0x2 == 0 {
                self.vcc &= !VCC_VALU;
            }
            if d & 0x1 == 0 {
                self.salu = 0;
                self.vcc &= !VCC_SALU;
            }
            if d & 0x0e00 == 0 {
                self.valu = 0;
            }
            return DEPCTR_NONE;
        }
        let (valu, salu) = (is_valu(inst), is_salu(inst));
        if !valu && !salu {
            return DEPCTR_NONE;
        }
        let mut wait = DEPCTR_NONE;
        if is_call_or_return(inst) {
            if self.vcc & VCC_VALU != 0 {
                wait &= DEPCTR_VA_VCC_0;
            }
            if self.salu != 0 || self.vcc & VCC_SALU != 0 {
                wait &= DEPCTR_SA_SDST_0;
            }
            if self.valu != 0 {
                wait &= DEPCTR_VA_SDST_0;
            }
        } else {
            for range in sgpr_operands(&inst.uses(wave64)) {
                wait &= self.read(range, valu, salu);
            }
        }
        if wait & 0x2 == 0 {
            self.vcc &= !VCC_VALU;
        }
        if wait & 0x1 == 0 {
            self.salu = 0;
            self.vcc &= !VCC_SALU;
        }
        if wait & 0x0e00 == 0 {
            self.valu = 0;
        }
        if is_call(inst) {
            self.tracked = u64::MAX;
        }
        for range in sgpr_operands(&inst.defs(wave64)) {
            self.write(range, valu);
        }
        wait
    }

    fn read(&mut self, range: RegRange, valu: bool, salu: bool) -> u16 {
        let pair = 1u64 << ((range.first >> 1) & 0x3f);
        if self.tracked & pair == 0 {
            if valu {
                self.tracked |= pair;
            }
            return DEPCTR_NONE;
        }
        let vcc = is_vcc(range);
        if salu {
            if vcc {
                if self.vcc & VCC_VALU != 0 {
                    self.vcc = 0;
                }
            } else {
                self.valu = 0;
            }
        }
        let bits = scalar_mask(range.first, range.count);
        let mut wait = DEPCTR_NONE;
        if self.salu & bits != 0 {
            wait &= DEPCTR_SA_SDST_0;
        }
        if valu && self.valu & bits != 0 {
            wait &= DEPCTR_VA_SDST_0;
        }
        if vcc && self.vcc & VCC_SALU != 0 {
            wait &= DEPCTR_SA_SDST_0;
        }
        if vcc && self.vcc & VCC_VALU != 0 {
            wait &= DEPCTR_VA_VCC_0;
        }
        wait
    }

    fn write(&mut self, range: RegRange, valu: bool) {
        let pair = 1u64 << ((range.first >> 1) & 0x3f);
        if self.tracked & pair == 0 {
            return;
        }
        if is_vcc(range) {
            self.vcc = if valu { VCC_VALU } else { VCC_SALU };
        } else if valu {
            self.valu |= scalar_mask(range.first, range.count);
        } else {
            self.salu |= scalar_mask(range.first, range.count);
        }
    }
}

// ---- instruction classes (LLVM `SIInstrInfo` predicates) -----------------------------------

fn is_valu(inst: &Instruction) -> bool {
    inst.family.is_vector_alu()
}

fn is_salu(inst: &Instruction) -> bool {
    inst.family.is_scalar_alu()
}

/// `SIInstrInfo::isVMEM`: buffer, image and FLAT instructions.
fn is_vmem(inst: &Instruction) -> bool {
    matches!(inst.family, F::Mubuf | F::Mtbuf | F::Mimg | F::Flat)
}

fn is_ds(inst: &Instruction) -> bool {
    inst.family == F::Ds
}

fn is_vnop(inst: &Instruction) -> bool {
    (inst.family == F::Vop1 && inst.opcode == VOP1_NOP)
        || (inst.family == F::Vop3 && inst.opcode == VOP3_VOP1_NOP)
}

/// A VOPC or VOP3 compare that writes EXEC.
fn is_vcmpx(inst: &Instruction) -> bool {
    matches!(inst.family, F::Vopc | F::Vop3) && inst.has_flag(flag::CMPX)
}

fn is_call(inst: &Instruction) -> bool {
    (inst.family == F::Sop1 && inst.opcode == SOP1_GFX11_SWAPPC_B64)
        || (inst.family == F::Sopk && inst.opcode == SOPK_GFX11_CALL_B64)
}

/// Calls, returns and indirect branches, whose targets cannot see pending SGPR writes.
fn is_call_or_return(inst: &Instruction) -> bool {
    is_call(inst)
        || (inst.family == F::Sop1
            && matches!(inst.opcode, SOP1_GFX11_SETPC_B64 | SOP1_GFX11_RFE_B64))
}

fn is_waitcnt_zero(inst: &Instruction) -> bool {
    inst.family == F::Sopp
        && inst.opcode == SOPP_GFX11_WAITCNT
        && inst.fields == Fields::Sopp { simm16: 0 }
}

fn depctr_of(inst: &Instruction) -> Option<u16> {
    match inst.fields {
        Fields::Sopp { simm16 } if inst.opcode == SOPP_GFX11_WAITCNT_DEPCTR => Some(simm16),
        _ => None,
    }
}

fn setprio_of(inst: &Instruction) -> Option<u16> {
    match inst.fields {
        Fields::Sopp { simm16 } if inst.opcode == SOPP_GFX11_SETPRIO => Some(simm16),
        _ => None,
    }
}

fn writes_exec(inst: &Instruction) -> bool {
    let mut writes = false;
    inst.each_def(true, |r| {
        writes |= r.contains(RegFile::Scalar, SCALAR_EXEC_LO)
            || r.contains(RegFile::Scalar, SCALAR_EXEC_HI);
    });
    writes
}

/// The SGPRs an SALU writes through its `sdst` field, unless EXEC or M0.
fn salu_destination(inst: &Instruction) -> Option<RegRange> {
    let sdst = match inst.fields {
        Fields::Sop1 { sdst, .. } | Fields::Sop2 { sdst, .. } | Fields::Sopk { sdst, .. } => sdst,
        _ => return None,
    };
    if inst.has_flag(flag::NO_DST) {
        return None;
    }
    let first = sdst.scalar_index()?;
    if matches!(first, SCALAR_EXEC_LO | SCALAR_EXEC_HI | SCALAR_M0) {
        return None;
    }
    Some(RegRange {
        file: RegFile::Scalar,
        first,
        count: if inst.has_flag(flag::D2) { 2 } else { 1 },
    })
}

/// The SGPRs a VALU reads as its wave64 lane mask: VCC for the VOP2 and DPP forms and
/// `v_div_fmas`, src2 for the VOP3 forms.
fn lane_mask_source(inst: &Instruction) -> Option<RegRange> {
    let vcc = RegRange {
        file: RegFile::Scalar,
        first: SCALAR_VCC_LO,
        count: 2,
    };
    if inst.has_flag(flag::READS_VCC) {
        return Some(vcc);
    }
    if !inst.has_flag(flag::CARRY_IN) {
        return None;
    }
    match inst.fields {
        Fields::Vop3(v) => v.src[2].scalar_index().map(|first| RegRange {
            file: RegFile::Scalar,
            first,
            count: 2,
        }),
        Fields::Vop2 { .. } => Some(vcc),
        _ => None,
    }
}

/// Whether a VALU reads an SGPR other than EXEC and M0, or a literal.
fn reads_sgpr_or_literal(inst: &Instruction) -> bool {
    let mut sgpr = false;
    inst.each_use(true, |r| {
        sgpr |= r.file == RegFile::Scalar
            && !r.contains(RegFile::Scalar, SCALAR_EXEC_LO)
            && !r.contains(RegFile::Scalar, SCALAR_M0);
    });
    let literal = match &inst.fields {
        Fields::Vop1 { src0, .. } | Fields::Vopc { src0, .. } => src0.literal().is_some(),
        Fields::Vop2 { src0, constant, .. } => src0.literal().is_some() || constant.is_some(),
        Fields::Vop3(v) => v.src.iter().any(|s| s.literal().is_some()),
        Fields::Vop3p(v) => v.src.iter().any(|s| s.literal().is_some()),
        _ => false,
    };
    sgpr || literal
}

fn explicit_vgpr_uses(inst: &Instruction) -> Vec<RegRange> {
    inst.uses(true)
        .into_iter()
        .filter(|r| r.file == RegFile::Vector)
        .collect()
}

/// SGPR operands for `AMDGPUWaitSGPRHazards`: every scalar register except EXEC and M0, once.
fn sgpr_operands(ranges: &[RegRange]) -> Vec<RegRange> {
    let mut out: Vec<RegRange> = Vec::new();
    for &r in ranges {
        let special = matches!(r.first, SCALAR_EXEC_LO | SCALAR_EXEC_HI | SCALAR_M0);
        if r.file == RegFile::Scalar && r.first < 128 && !special && !out.contains(&r) {
            out.push(r);
        }
    }
    out
}

fn is_vcc(range: RegRange) -> bool {
    matches!(range.first, SCALAR_VCC_LO | SCALAR_VCC_HI)
}

fn ranges_overlap(a: RegRange, b: RegRange) -> bool {
    a.file == b.file && a.first < b.first + b.count.max(1) && b.first < a.first + a.count.max(1)
}

fn add_vgpr_bits(bits: &mut [u128; 2], r: RegRange) {
    if r.file == RegFile::Vector {
        for reg in r.first..(r.first + r.count).min(256) {
            bits[usize::from(reg / 128)] |= 1 << (reg % 128);
        }
    }
}

/// The VGPRs `inst` reads, as bits.
fn vgpr_uses(inst: &Instruction) -> [u128; 2] {
    let mut bits = [0u128; 2];
    inst.each_use(false, |r| add_vgpr_bits(&mut bits, r));
    bits
}

/// The VGPRs `inst` writes, as bits.
fn vgpr_defs(inst: &Instruction) -> [u128; 2] {
    let mut bits = [0u128; 2];
    inst.each_def(false, |r| add_vgpr_bits(&mut bits, r));
    bits
}

fn overlaps(a: [u128; 2], b: [u128; 2]) -> bool {
    a[0] & b[0] != 0 || a[1] & b[1] != 0
}

fn depctr(origin: &Instruction, simm16: u16) -> Instruction {
    sopp(origin, SOPP_GFX11_WAITCNT_DEPCTR, simm16)
}

fn sopp(origin: &Instruction, opcode: u16, simm16: u16) -> Instruction {
    derive(origin, F::Sopp, opcode, 0, Fields::Sopp { simm16 })
}

fn predecessors(succ: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); succ.len()];
    for (id, targets) in succ.iter().enumerate() {
        for &t in targets {
            if !preds[t].contains(&id) {
                preds[t].push(id);
            }
        }
    }
    preds
}
