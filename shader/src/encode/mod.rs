pub mod hazards;
mod layout;
mod vmem_waits;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};

use layout::{Layout, Unit};

use crate::decode::{DPP16_MARKER, DPP8_FI_MARKER, DPP8_MARKER, SDWA_MARKER};
use crate::ir::opcodes::{flag, SOPP_BRANCH, SOPP_ENDPGM};
use crate::ir::{
    memops, Buffer, Ds, Export, Fields, Flat, FlatSegment, Image, Instruction,
    InstructionFamily as F, MemKind, Operand, Sdwa, Smem, VopExt,
};
use crate::lowering::{DispatchTable, LoweredProgram, DISPATCH_LABEL, MISS_LABEL, LOCAL_LABEL_BASE, PROLOGUE_LABEL};
use crate::{CompileError, CompileOptions, TargetIsa};

pub const S_CODE_END: u32 = 0xbf9f_0000;
pub const PREFETCH_PADDING_LINES: usize = 3;
const SOPP_BASE: u32 = 0xbf80_0000;
const SOP1_BASE: u32 = 0xbe80_0000;
const SOPC_BASE: u32 = 0xbf00_0000;
const SOPK_BASE: u32 = 0xb000_0000;
const SOP2_BASE: u32 = 0x8000_0000;
const VOP1_BASE: u32 = 0x7e00_0000;
const VOPC_BASE: u32 = 0x7c00_0000;
const VOP3_BASE: u32 = 0xd400_0000;
const VOP3P_BASE: u32 = 0xcc00_0000;
const VINTRP_BASE: u32 = 0xc800_0000;
const LDSDIR_BASE: u32 = 0xce00_0000;
const VINTERP_BASE: u32 = 0xcd00_0000;
const SMEM_BASE: u32 = 0xf400_0000;
const MUBUF_BASE: u32 = 0xe000_0000;
const MTBUF_BASE: u32 = 0xe800_0000;
const VBUFFER_BASE: u32 = 0xc400_0000;
const FLAT_BASE: u32 = 0xdc00_0000;
const VFLAT_BASE: u32 = 0xec00_0000;
const DS_BASE: u32 = 0xd800_0000;
const MIMG_BASE: u32 = 0xf000_0000;
const VIMAGE_BASE: u32 = 0xd000_0000;
const VSAMPLE_BASE: u32 = 0xe400_0000;
const EXP_BASE: u32 = 0xf800_0000;
const SDWA_S0: u32 = 1 << 23;
const GFX10_FLAT_SADDR_NULL: u32 = 0x7d;
const FLAT_SADDR_OFF: u32 = 0x7f;
/// gfx11 encodes a FLAT/global/scratch instruction without an SGPR base as SADDR = NULL.
const GFX11_NULL: u32 = 124;
const GFX12_NULL: u32 = 124;
const GFX12_MTBUF_OP_BASE: u16 = 0x80;
const GFX12_SCOPE_DEV: u32 = 2;
const GFX12_TH_RETURN: u32 = 1;
const GFX12_TH_NONTEMPORAL: u32 = 1;
const GFX12_VIMAGE_ADDRESS_SLOTS: usize = 5;
const GFX12_VSAMPLE_ADDRESS_SLOTS: usize = 4;
/// `image_bvh_intersect_ray` and `image_bvh64_intersect_ray` on gfx11 and gfx12.
const HOST_BVH: u16 = 0x19;
const HOST_BVH64: u16 = 0x1a;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Encoded {
    pub code: Vec<u32>,
    pub program_words: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Encoder {
    options: CompileOptions,
}

impl Encoder {
    pub fn new(options: CompileOptions) -> Self {
        Self { options }
    }

    pub fn encode(&self, program: &LoweredProgram) -> Result<Encoded, CompileError> {
        let mut program = program.clone();
        vmem_waits::insert(&mut program);
        hazards::insert(&mut program);
        let target = self.options.target;
        let mut layout = Layout::new(target_opcode_branch(target));
        let end = || Unit {
            words: vec![sopp(target_opcode_endpgm(target), 0)],
            target: None,
            pc: 0,
            falls_through: false,
            site: false,
        };
        let mut falls_through = true;
        let mut stub = DispatchStub::default();
        for block in &program.blocks {
            let dispatch = block.guest_pc == DISPATCH_LABEL;
            if dispatch && falls_through {
                layout.push(end());
            }
            layout.label(block.guest_pc);
            // The dispatch and miss stubs stay whole: the table is addressed from inside them.
            let stubs = matches!(block.guest_pc, DISPATCH_LABEL | MISS_LABEL);
            let mut previous: Option<&Instruction> = None;
            for inst in &block.instructions {
                if (LOCAL_LABEL_BASE..PROLOGUE_LABEL).contains(&inst.pc) {
                    layout.label(inst.pc);
                }
                let site = !stubs
                    && previous.is_none_or(|before| target.is_gfx11_plus() && separable(before, inst));
                let unit = layout.push(Unit {
                    words: encode_instruction(inst, target)?,
                    target: inst.branch_target,
                    pc: inst.pc,
                    falls_through: inst.falls_through(),
                    site,
                });
                if dispatch {
                    stub.note(inst, unit);
                }
                falls_through = inst.falls_through();
                previous = Some(inst);
            }
        }
        if falls_through {
            layout.push(end());
        }
        let placed = layout.place()?;
        let mut code = placed.code;
        let program_words = code.len();
        if let Some(table) = &program.dispatch {
            stub.append_table(&program, table, &placed.units, &placed.labels, &mut code, target)?;
        }
        let line = target.instruction_cache_line_bytes() / 4;
        while code.len() % line != 0 {
            code.push(S_CODE_END);
        }
        code.extend(std::iter::repeat_n(
            S_CODE_END,
            PREFETCH_PADDING_LINES * line,
        ));
        Ok(Encoded {
            code,
            program_words,
        })
    }
}

/// The dispatch stub's PC anchor and table load, by the layout unit each one is.
#[derive(Default)]
struct DispatchStub {
    /// The stub's `s_getpc_b64`: table entries are relative to the word after it.
    getpc: Option<usize>,
    /// The table load.
    load: Option<(usize, Instruction)>,
}

impl DispatchStub {
    fn note(&mut self, inst: &Instruction, unit: usize) {
        let getpc = inst.family == F::Sop1
            && inst.has_flag(flag::PC_RELATIVE)
            && !inst.has_flag(flag::CALL);
        if getpc && self.getpc.is_none() {
            self.getpc = Some(unit);
        }
        if matches!(inst.fields, Fields::Smem(_)) && self.load.is_none() {
            self.load = Some((unit, inst.clone()));
        }
    }

    /// Appends the guest-to-host table after the program and points the stub's load at it.
    /// `units` and `block_start` are the words the layout put each unit and label at.
    fn append_table(
        &self,
        program: &LoweredProgram,
        table: &DispatchTable,
        units: &[usize],
        block_start: &BTreeMap<u32, usize>,
        code: &mut Vec<u32>,
        target: TargetIsa,
    ) -> Result<(), CompileError> {
        let malformed = CompileError::InvalidOperand {
            pc: DISPATCH_LABEL,
            reason: "the dispatch stub lacks its PC anchor or table load",
        };
        let (Some(getpc), Some((load_unit, load))) = (self.getpc, &self.load) else {
            return Err(malformed);
        };
        let (anchor, at) = (units[getpc] + 1, units[*load_unit]);
        let miss = *block_start.get(&MISS_LABEL).ok_or(malformed)?;
        let landing: HashMap<u32, usize> = program
            .blocks
            .iter()
            .filter(|b| b.dynamic_entry)
            .filter_map(|b| Some((b.guest_pc, *block_start.get(&b.guest_pc)?)))
            .collect();
        let relative = |host: usize| (4 * (host as i64 - anchor as i64)) as u32;
        let table_at = code.len();
        for word in 0..table.words {
            let host = landing.get(&(4 * word)).copied().unwrap_or(miss);
            code.push(relative(host));
        }
        code.push(relative(miss));
        let offset = 4 * (table_at as i64 - anchor as i64);
        let limit = if target.is_gfx12() { 1 << 23 } else { 1 << 20 };
        if offset >= limit {
            return Err(CompileError::BranchOutOfRange { pc: load.pc });
        }
        let mut load = load.clone();
        if let Fields::Smem(s) = &mut load.fields {
            s.offset = offset as i32;
        }
        let words = encode_instruction(&load, target)?;
        code[at..at + words.len()].copy_from_slice(&words);
        Ok(())
    }
}

fn target_opcode_endpgm(target: TargetIsa) -> u16 {
    if target.is_gfx11_plus() {
        0x30
    } else {
        SOPP_ENDPGM
    }
}

fn target_opcode_branch(target: TargetIsa) -> u16 {
    if target.is_gfx11_plus() {
        0x20
    } else {
        SOPP_BRANCH
    }
}

/// Whether a pool of branch islands may lie between two instructions of a block
/// ([`layout`]): they translate different guest instructions, so no expansion is split, and
/// neither is an export.
fn separable(before: &Instruction, after: &Instruction) -> bool {
    before.pc != after.pc
        && before.pc < LOCAL_LABEL_BASE
        && after.pc < LOCAL_LABEL_BASE
        && before.family != F::Exp
        && after.family != F::Exp
}

pub fn sopp(op: u16, simm16: u16) -> u32 {
    SOPP_BASE | (u32::from(op) << 16) | u32::from(simm16)
}

fn invalid(inst: &Instruction, reason: &'static str) -> CompileError {
    CompileError::InvalidOperand {
        pc: inst.pc,
        reason,
    }
}

struct Enc<'a> {
    inst: &'a Instruction,
    target: TargetIsa,
}

impl Enc<'_> {
    fn code(&self, op: Operand) -> Result<u32, CompileError> {
        op.code(self.target)
            .map(u32::from)
            .ok_or_else(|| invalid(self.inst, "operand is not encodable on the target"))
    }

    fn vgpr8(&self, op: Operand) -> Result<u32, CompileError> {
        match op {
            Operand::Vgpr(n) => Ok(u32::from(n)),
            Operand::Raw(v) => Ok(u32::from(v & 0xff)),
            other => Ok(self.code(other)? & 0xff),
        }
    }

    fn literal(&self, words: &mut Vec<u32>, ops: &[Operand]) -> Result<(), CompileError> {
        let mut value = None;
        for v in ops.iter().filter_map(|op| op.literal()) {
            if value.is_some_and(|existing| existing != v) {
                return Err(invalid(
                    self.inst,
                    "instruction needs two different literals",
                ));
            }
            value = Some(v);
        }
        words.extend(value);
        Ok(())
    }
}

pub fn encode_instruction(inst: &Instruction, target: TargetIsa) -> Result<Vec<u32>, CompileError> {
    let e = Enc { inst, target };
    let op = u32::from(inst.opcode);
    let gfx11 = target.is_gfx11_plus();
    let mut w = Vec::with_capacity(3);
    match &inst.fields {
        Fields::Sop1 { sdst, src0 } => {
            w.push(SOP1_BASE | (e.code(*sdst)? << 16) | (op << 8) | e.code(*src0)?);
            if !inst.has_flag(flag::NO_SRC) {
                e.literal(&mut w, &[*src0])?;
            }
        }
        Fields::Sop2 { sdst, src0, src1 } => {
            w.push(
                SOP2_BASE
                    | (op << 23)
                    | (e.code(*sdst)? << 16)
                    | (e.code(*src1)? << 8)
                    | e.code(*src0)?,
            );
            e.literal(&mut w, &[*src0, *src1])?;
        }
        Fields::Sopk {
            sdst,
            simm16,
            imm32,
        } => {
            w.push(SOPK_BASE | (op << 23) | (e.code(*sdst)? << 16) | u32::from(*simm16));
            w.extend(*imm32);
        }
        Fields::Sopc { src0, src1 } => {
            w.push(SOPC_BASE | (op << 16) | (e.code(*src1)? << 8) | e.code(*src0)?);
            e.literal(&mut w, &[*src0, *src1])?;
        }
        Fields::Sopp { simm16 } => w.push(sopp(inst.opcode, *simm16)),
        Fields::Vop1 { vdst, src0, ext } => {
            let head = VOP1_BASE | (e.vgpr8(*vdst)? << 17) | (op << 9);
            vop_ext(&e, &mut w, head, *src0, None, *ext, None)?;
        }
        Fields::Vop2 {
            vdst,
            src0,
            vsrc1,
            constant,
            ext,
        } => {
            let head = (op << 25) | (e.vgpr8(*vdst)? << 17);
            vop_ext(&e, &mut w, head, *src0, Some(*vsrc1), *ext, None)?;
            w.extend(*constant);
        }
        Fields::Vopc {
            sdst,
            src0,
            vsrc1,
            ext,
        } => {
            let sd = if inst.has_flag(flag::CMPX) || *sdst == Operand::VccLo {
                None
            } else {
                Some(*sdst)
            };
            if sd.is_some() && !matches!(ext, VopExt::Sdwa(_)) {
                return Err(invalid(inst, "VOPC e32 writes only VCC"));
            }
            vop_ext(
                &e,
                &mut w,
                VOPC_BASE | (op << 17),
                *src0,
                Some(*vsrc1),
                *ext,
                sd,
            )?;
        }
        Fields::Vop3(v) => {
            let mut w0 = VOP3_BASE | (op << 16) | (u32::from(v.clamp) << 15) | e.vgpr8(v.vdst)?;
            match v.sdst {
                Some(sdst) => w0 |= (e.code(sdst)? & 0x7f) << 8,
                None => w0 |= (u32::from(v.opsel & 0xf) << 11) | (u32::from(v.abs & 0x7) << 8),
            }
            w.push(w0);
            w.push(
                (u32::from(v.neg & 7) << 29)
                    | (u32::from(v.omod & 3) << 27)
                    | (e.code(v.src[2])? << 18)
                    | (e.code(v.src[1])? << 9)
                    | e.code(v.src[0])?,
            );
            e.literal(&mut w, &v.src)?;
        }
        Fields::Vop3p(v) => {
            w.push(
                VOP3P_BASE
                    | (op << 16)
                    | (u32::from(v.clamp) << 15)
                    | (u32::from((v.opsel_hi >> 2) & 1) << 14)
                    | (u32::from(v.opsel & 7) << 11)
                    | (u32::from(v.neg_hi & 7) << 8)
                    | e.vgpr8(v.vdst)?,
            );
            w.push(
                (u32::from(v.neg & 7) << 29)
                    | (u32::from(v.opsel_hi & 3) << 27)
                    | (e.code(v.src[2])? << 18)
                    | (e.code(v.src[1])? << 9)
                    | e.code(v.src[0])?,
            );
            e.literal(&mut w, &v.src)?;
        }
        Fields::Vintrp(i) => {
            if gfx11 {
                return Err(invalid(inst, "VINTRP does not exist on the target"));
            }
            w.push(
                VINTRP_BASE
                    | (u32::from(i.vdst) << 18)
                    | (op << 16)
                    | (u32::from(i.attr & 0x3f) << 10)
                    | (u32::from(i.chan & 3) << 8)
                    | u32::from(i.vsrc),
            );
        }
        Fields::Ldsdir(l) => w.push(
            LDSDIR_BASE
                | (op << 20)
                | (u32::from(l.wait_vdst & 0xf) << 16)
                | (u32::from(l.attr & 0x3f) << 10)
                | (u32::from(l.chan & 3) << 8)
                | u32::from(l.vdst),
        ),
        Fields::Vinterp(v) => {
            w.push(
                VINTERP_BASE
                    | (op << 16)
                    | (u32::from(v.clamp) << 15)
                    | (u32::from(v.opsel & 0xf) << 11)
                    | (u32::from(v.wait_exp & 7) << 8)
                    | u32::from(v.vdst),
            );
            w.push(
                (u32::from(v.neg & 7) << 29)
                    | (e.code(v.src[2])? << 18)
                    | (e.code(v.src[1])? << 9)
                    | e.code(v.src[0])?,
            );
        }
        Fields::Smem(s) => smem(&e, &mut w, s)?,
        Fields::Mubuf(b) => buffer(&e, &mut w, b, false)?,
        Fields::Mtbuf(b) => buffer(&e, &mut w, b, true)?,
        Fields::Flat(f) => flat(&e, &mut w, f)?,
        Fields::Ds(d) => ds(&mut w, op, d),
        Fields::Mimg(m) => image(&e, &mut w, m)?,
        Fields::Exp(x) => export(&mut w, x, gfx11),
    }
    Ok(w)
}

fn sdwa_word(
    e: &Enc<'_>,
    s: &Sdwa,
    src0: Operand,
    src1: Option<Operand>,
    sdst: Option<Operand>,
) -> Result<u32, CompileError> {
    let src0_bits = match src0 {
        Operand::Vgpr(n) => u32::from(n),
        other => (e.code(other)? & 0xff) | SDWA_S0,
    };
    let src1_scalar = src1.is_some_and(|op| !op.is_vgpr());
    let mut w = src0_bits
        | (u32::from(src1_scalar) << 31)
        | (u32::from(s.src0_sel & 7) << 16)
        | (u32::from(s.src0_sext) << 19)
        | (u32::from(s.src0_neg) << 20)
        | (u32::from(s.src0_abs) << 21)
        | (u32::from(s.src1_sel & 7) << 24)
        | (u32::from(s.src1_sext) << 27)
        | (u32::from(s.src1_neg) << 28)
        | (u32::from(s.src1_abs) << 29);
    match (e.inst.family, sdst) {
        (F::Vopc, Some(sdst)) => w |= ((e.code(sdst)? & 0x7f) << 8) | (1 << 15),
        (F::Vopc, None) => {}
        _ => {
            w |= (u32::from(s.dst_sel & 7) << 8)
                | (u32::from(s.dst_unused & 3) << 11)
                | (u32::from(s.clamp) << 13)
                | (u32::from(s.omod & 3) << 14);
        }
    }
    Ok(w)
}

fn vop_ext(
    e: &Enc<'_>,
    w: &mut Vec<u32>,
    head: u32,
    src0: Operand,
    vsrc1: Option<Operand>,
    ext: VopExt,
    sdst: Option<Operand>,
) -> Result<(), CompileError> {
    let vsrc1_bits = |code: u32| (code & 0xff) << 9;
    match ext {
        VopExt::None => {
            let v1 = match vsrc1 {
                Some(Operand::Vgpr(n)) => vsrc1_bits(u32::from(n)),
                Some(Operand::Raw(v)) => vsrc1_bits(u32::from(v)),
                Some(_) => return Err(invalid(e.inst, "VOP2/VOPC vsrc1 must be a VGPR")),
                None => 0,
            };
            w.push(head | v1 | e.code(src0)?);
            e.literal(w, &[src0])?;
        }
        VopExt::Sdwa(s) => {
            if e.target.is_gfx11_plus() {
                return Err(invalid(e.inst, "SDWA does not exist on the target"));
            }
            let v1 = match vsrc1 {
                Some(Operand::Vgpr(n)) => vsrc1_bits(u32::from(n)),
                Some(other) => vsrc1_bits(e.code(other)?),
                None => 0,
            };
            w.push(head | v1 | u32::from(SDWA_MARKER));
            w.push(sdwa_word(e, &s, src0, vsrc1, sdst)?);
        }
        VopExt::Dpp16(d) => {
            let v1 = vsrc1.map_or(Ok(0), |op| e.vgpr8(op).map(vsrc1_bits))?;
            w.push(head | v1 | u32::from(DPP16_MARKER));
            w.push(
                e.vgpr8(src0)?
                    | (u32::from(d.ctrl & 0x1ff) << 8)
                    | (u32::from(d.fetch_inactive) << 18)
                    | (u32::from(d.bound_ctrl) << 19)
                    | (u32::from(d.src0_neg) << 20)
                    | (u32::from(d.src0_abs) << 21)
                    | (u32::from(d.src1_neg) << 22)
                    | (u32::from(d.src1_abs) << 23)
                    | (u32::from(d.bank_mask & 0xf) << 24)
                    | (u32::from(d.row_mask & 0xf) << 28),
            );
        }
        VopExt::Dpp8 {
            selects,
            fetch_inactive,
        } => {
            let v1 = vsrc1.map_or(Ok(0), |op| e.vgpr8(op).map(vsrc1_bits))?;
            let marker = if fetch_inactive {
                DPP8_FI_MARKER
            } else {
                DPP8_MARKER
            };
            w.push(head | v1 | u32::from(marker));
            w.push(e.vgpr8(src0)? | (selects << 8));
        }
    }
    Ok(())
}

fn gfx12_policy(kind: MemKind, glc: bool, slc: bool) -> (u32, u32) {
    match kind {
        MemKind::Atomic => (if glc { GFX12_TH_RETURN } else { 0 }, GFX12_SCOPE_DEV),
        _ => (
            if slc { GFX12_TH_NONTEMPORAL } else { 0 },
            if glc { GFX12_SCOPE_DEV } else { 0 },
        ),
    }
}

fn smem(e: &Enc<'_>, w: &mut Vec<u32>, s: &Smem) -> Result<(), CompileError> {
    let op = u32::from(e.inst.opcode);
    let sdata = e.code(s.sdata)? & 0x7f;
    let base = SMEM_BASE | (sdata << 6) | u32::from(s.sbase / 2);
    let soffset = (e.code(s.soffset)? & 0x7f) << 25;
    match e.target {
        TargetIsa::Gfx1013 | TargetIsa::Gfx103 => {
            w.push(base | (op << 18) | (u32::from(s.glc) << 16) | (u32::from(s.dlc) << 14));
            w.push(soffset | (s.offset as u32 & 0x1f_ffff));
        }
        TargetIsa::Gfx11 | TargetIsa::Gfx115 => {
            w.push(base | (op << 18) | (u32::from(s.glc) << 14) | (u32::from(s.dlc) << 13));
            w.push(soffset | (s.offset as u32 & 0x1f_ffff));
        }
        TargetIsa::Gfx12 => {
            let scope = if s.glc { GFX12_SCOPE_DEV } else { 0 };
            w.push(base | ((op & 0x3f) << 13) | (scope << 21));
            w.push(soffset | (s.offset as u32 & 0xff_ffff));
        }
    }
    Ok(())
}

fn buffer(e: &Enc<'_>, w: &mut Vec<u32>, b: &Buffer, typed: bool) -> Result<(), CompileError> {
    let op = u32::from(e.inst.opcode);
    let soffset = match b.soffset {
        Operand::InlineInt(0) if e.target.is_gfx12() => GFX12_NULL,
        other => e.code(other)?,
    };
    if e.target.is_gfx12() && soffset > 0x7f {
        return Err(invalid(
            e.inst,
            "buffer offset must be an SGPR on the target",
        ));
    }
    let regs = (u32::from(b.srsrc / 4) << 16) | (u32::from(b.vdata) << 8) | u32::from(b.vaddr);
    let offset = u32::from(b.offset & 0xfff);
    let (glc, dlc, slc) = (u32::from(b.glc), u32::from(b.dlc), u32::from(b.slc));
    let (offen, idxen, tfe) = (u32::from(b.offen), u32::from(b.idxen), u32::from(b.tfe));
    match e.target {
        TargetIsa::Gfx1013 | TargetIsa::Gfx103 => {
            let w0 = if typed {
                MTBUF_BASE | (u32::from(b.format & 0x7f) << 19) | ((op & 7) << 16)
            } else {
                MUBUF_BASE | ((op & 0x7f) << 18) | ((op >> 7) << 25) | (u32::from(b.lds) << 16)
            };
            w.push(w0 | (dlc << 15) | (glc << 14) | (idxen << 13) | (offen << 12) | offset);
            let op_hi = if typed { (op >> 3) << 21 } else { 0 };
            w.push((soffset << 24) | (tfe << 23) | (slc << 22) | op_hi | regs);
        }
        TargetIsa::Gfx11 | TargetIsa::Gfx115 => {
            let w0 = if typed {
                MTBUF_BASE | (u32::from(b.format & 0x7f) << 19) | ((op & 0xf) << 15)
            } else {
                MUBUF_BASE | ((op & 0xff) << 18)
            };
            w.push(w0 | (glc << 14) | (dlc << 13) | (slc << 12) | offset);
            w.push((soffset << 24) | (idxen << 23) | (offen << 22) | (tfe << 21) | regs);
        }
        TargetIsa::Gfx12 => {
            let op = if typed {
                u32::from(GFX12_MTBUF_OP_BASE | e.inst.opcode)
            } else {
                op
            };
            let (th, scope) = gfx12_policy(b.kind, b.glc, b.slc);
            let format = if typed {
                u32::from(b.format & 0x7f) << 23
            } else {
                0
            };
            w.push(VBUFFER_BASE | (tfe << 22) | ((op & 0xff) << 14) | (soffset & 0x7f));
            w.push(
                (idxen << 31)
                    | (offen << 30)
                    | format
                    | (th << 20)
                    | (scope << 18)
                    | (u32::from(b.srsrc) << 9)
                    | u32::from(b.vdata),
            );
            w.push((offset << 8) | u32::from(b.vaddr));
        }
    }
    Ok(())
}

fn flat(e: &Enc<'_>, w: &mut Vec<u32>, f: &Flat) -> Result<(), CompileError> {
    let op = u32::from(e.inst.opcode);
    let seg = match f.segment {
        FlatSegment::Flat => 0,
        FlatSegment::Scratch => 1,
        FlatSegment::Global => 2,
    };
    let (glc, dlc, slc) = (u32::from(f.glc), u32::from(f.dlc), u32::from(f.slc));
    let sve = u32::from(f.segment == FlatSegment::Scratch && f.uses_vaddr);
    let saddr = |off: u32, null: u32| -> Result<u32, CompileError> {
        Ok(match f.saddr {
            None => off,
            Some(Operand::Null) => null,
            Some(reg) => e.code(reg)? & 0x7f,
        })
    };
    let regs = (u32::from(f.vdst) << 24) | (u32::from(f.vdata) << 8) | u32::from(f.vaddr);
    match e.target {
        TargetIsa::Gfx1013 | TargetIsa::Gfx103 => {
            w.push(
                FLAT_BASE
                    | (op << 18)
                    | (slc << 17)
                    | (glc << 16)
                    | (seg << 14)
                    | (u32::from(f.lds) << 13)
                    | (dlc << 12)
                    | (f.offset as u32 & 0xfff),
            );
            w.push(regs | (saddr(FLAT_SADDR_OFF, GFX10_FLAT_SADDR_NULL)? << 16));
        }
        TargetIsa::Gfx11 | TargetIsa::Gfx115 => {
            w.push(
                FLAT_BASE
                    | (op << 18)
                    | (seg << 16)
                    | (slc << 15)
                    | (glc << 14)
                    | (dlc << 13)
                    | (f.offset as u32 & 0x1fff),
            );
            w.push(regs | (sve << 23) | (saddr(GFX11_NULL, GFX11_NULL)? << 16));
        }
        TargetIsa::Gfx12 => {
            let (th, scope) = gfx12_policy(f.kind, f.glc, f.slc);
            let saddr = match f.saddr {
                None | Some(Operand::Null) => GFX12_NULL,
                Some(reg) => e.code(reg)? & 0x7f,
            };
            w.push(VFLAT_BASE | (seg << 24) | ((op & 0xff) << 14) | saddr);
            w.push(
                (u32::from(f.vdata) << 23)
                    | (th << 20)
                    | (scope << 18)
                    | (sve << 17)
                    | u32::from(f.vdst),
            );
            w.push(((f.offset as u32 & 0xff_ffff) << 8) | u32::from(f.vaddr));
        }
    }
    Ok(())
}

fn ds(w: &mut Vec<u32>, op: u32, d: &Ds) {
    w.push(
        DS_BASE
            | (op << 18)
            | (u32::from(d.gds) << 17)
            | (u32::from(d.offset1) << 8)
            | u32::from(d.offset0),
    );
    w.push(
        (u32::from(d.vdst) << 24)
            | (u32::from(d.data1) << 16)
            | (u32::from(d.data0) << 8)
            | u32::from(d.addr),
    );
}

fn image(e: &Enc<'_>, w: &mut Vec<u32>, m: &Image) -> Result<(), CompileError> {
    let op = u32::from(e.inst.opcode);
    let b = |v: u8| u32::from(v);
    let f = |v: bool| u32::from(v);
    let regs = (b(m.srsrc / 4) << 16) | (b(m.vdata) << 8) | b(m.addr[0]);
    let ssamp = if m.uses_sampler { b(m.ssamp / 4) } else { 0 };
    match e.target {
        TargetIsa::Gfx1013 | TargetIsa::Gfx103 => {
            w.push(
                MIMG_BASE
                    | ((op & 0x7f) << 18)
                    | (f(m.slc) << 25)
                    | (f(m.lwe) << 17)
                    | (f(m.tfe) << 16)
                    | (f(m.r128) << 15)
                    | (f(m.glc) << 13)
                    | (f(m.unorm) << 12)
                    | (b(m.dmask & 0xf) << 8)
                    | (f(m.dlc) << 7)
                    | (b(m.dim & 7) << 3)
                    | (b(m.nsa_words & 3) << 1)
                    | (op >> 7),
            );
            w.push((f(m.d16) << 31) | (f(m.a16) << 30) | (ssamp << 21) | regs);
            for word in 0..usize::from(m.nsa_words) {
                let bytes = [0, 1, 2, 3].map(|i| m.addr[1 + word * 4 + i]);
                w.push(u32::from_le_bytes(bytes));
            }
        }
        TargetIsa::Gfx11 | TargetIsa::Gfx115 => {
            let nsa = m.nsa && m.addr_dwords > 1;
            w.push(
                MIMG_BASE
                    | (op << 18)
                    | (f(m.d16) << 17)
                    | (f(m.a16) << 16)
                    | (f(m.r128) << 15)
                    | (f(m.glc) << 14)
                    | (f(m.dlc) << 13)
                    | (f(m.slc) << 12)
                    | (b(m.dmask & 0xf) << 8)
                    | (f(m.unorm) << 7)
                    | (b(m.dim & 7) << 2)
                    | f(nsa),
            );
            w.push((ssamp << 26) | (f(m.lwe) << 22) | (f(m.tfe) << 21) | regs);
            if nsa {
                let bytes = match ray_groups(e.inst.opcode, m) {
                    Some(groups) => {
                        let slots = address_slots(e, m, &groups)?;
                        [1, 2, 3, 4].map(|i| slots[i] as u8)
                    }
                    None => [1, 2, 3, 4].map(|i| {
                        if i < usize::from(m.addr_dwords) {
                            m.addr[i]
                        } else {
                            0
                        }
                    }),
                };
                w.push(u32::from_le_bytes(bytes));
            }
        }
        TargetIsa::Gfx12 => {
            let addr = gfx12_address_slots(e, m)?;
            let (th, scope) = gfx12_policy(m.kind, m.glc, m.slc);
            let common = (b(m.dmask & 0xf) << 22)
                | ((op & 0xff) << 14)
                | (f(m.a16) << 6)
                | (f(m.d16) << 5)
                | (f(m.r128) << 4)
                | b(m.dim & 7);
            let policy = (th << 20) | (scope << 18) | (b(m.srsrc) << 9) | b(m.vdata);
            if memops::gfx12_vsample(e.inst.opcode, m.uses_sampler) {
                let samp = if m.uses_sampler { b(m.ssamp) } else { 0 };
                w.push(VSAMPLE_BASE | common | (f(m.unorm) << 13) | (f(m.tfe) << 3));
                w.push((samp << 23) | policy | (f(m.lwe) << 8));
            } else {
                w.push(VIMAGE_BASE | common);
                w.push((addr[4] << 24) | (f(m.tfe) << 23) | policy);
            }
            w.push(addr[0] | (addr[1] << 8) | (addr[2] << 16) | (addr[3] << 24));
        }
    }
    Ok(())
}

/// The address groups of a BVH intersection on gfx11 and gfx12, in dwords: the node pointer (two
/// for the 64-bit form), the ray extent, and the origin, direction and inverse-direction triples
/// (with A16 the direction and inverse direction share one triple). `None` for other images.
fn ray_groups(opcode: u16, m: &Image) -> Option<Vec<usize>> {
    if !matches!(opcode, HOST_BVH | HOST_BVH64) || m.uses_sampler {
        return None;
    }
    let mut groups = vec![if opcode == HOST_BVH64 { 2 } else { 1 }, 1, 3, 3];
    if !m.a16 {
        groups.push(3);
    }
    Some(groups)
}

/// gfx12 VIMAGE/VSAMPLE name each address operand by its first VGPR. Each slot holds one dword
/// and the last slot every remaining dword, except for BVH intersections ([`ray_groups`]).
fn gfx12_address_slots(
    e: &Enc<'_>,
    m: &Image,
) -> Result<[u32; GFX12_VIMAGE_ADDRESS_SLOTS], CompileError> {
    let dwords = usize::from(m.addr_dwords);
    let widths = ray_groups(e.inst.opcode, m).unwrap_or_else(|| {
        let slots = if memops::gfx12_vsample(e.inst.opcode, m.uses_sampler) {
            GFX12_VSAMPLE_ADDRESS_SLOTS
        } else {
            GFX12_VIMAGE_ADDRESS_SLOTS
        };
        let singles = dwords.min(slots).saturating_sub(1);
        let mut widths = vec![1; singles];
        widths.push(dwords - singles);
        widths
    });
    address_slots(e, m, &widths)
}

/// The first VGPR of each address group of `widths` dwords, which must lie in consecutive
/// registers.
fn address_slots(
    e: &Enc<'_>,
    m: &Image,
    widths: &[usize],
) -> Result<[u32; GFX12_VIMAGE_ADDRESS_SLOTS], CompileError> {
    let mut addr = [0u32; GFX12_VIMAGE_ADDRESS_SLOTS];
    let mut next = 0;
    for (slot, &width) in addr.iter_mut().zip(widths) {
        let first = m.address_register(next);
        if (1..width).any(|k| m.address_register(next + k) != first.wrapping_add(k as u8)) {
            return Err(invalid(
                e.inst,
                "image address group is not in consecutive VGPRs",
            ));
        }
        *slot = u32::from(first);
        next += width;
    }
    Ok(addr)
}

fn export(w: &mut Vec<u32>, x: &Export, gfx11: bool) {
    let flags = if gfx11 {
        (u32::from(x.row_en) << 13) | (u32::from(x.done) << 11)
    } else {
        (u32::from(x.vm) << 12) | (u32::from(x.done) << 11) | (u32::from(x.compr) << 10)
    };
    w.push(EXP_BASE | flags | (u32::from(x.target & 0x3f) << 4) | u32::from(x.en & 0xf));
    w.push(u32::from_le_bytes(x.vsrc));
}
