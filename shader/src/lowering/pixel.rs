//! Pixel shaders on gfx11 and later: parameter loads, the null-export target and dual-source
//! exports.
//!
//! gfx11 interpolation reads parameters that `lds_param_load` leaves in lanes 0-2 of each quad,
//! so a load must cover whole quads, while the guest's VINTRP worked lane by lane under any
//! EXEC. When an interpolation's M0 provably holds the wave's `prim_mask` (the SGPR after the
//! user SGPRs, unchanged since entry), it reads a VGPR the prologue loaded with every lane
//! enabled. Any other interpolation enables every lane around its own load. Lanes outside the
//! guest's EXEC only ever write translation VGPRs.

use std::collections::{BTreeMap, BTreeSet};

use super::{derive, Ctx, SOP1_MOV_B32, SOP1_MOV_B64};
use crate::ir::opcodes::{
    flag, LDSDIR_PARAM_LOAD, VOP3_INTERP_MOV_F32, VOP3_INTERP_P1LL_F16, VOP3_INTERP_P1LV_F16,
    VOP3_INTERP_P1_F32, VOP3_INTERP_P2_F16,
};
use crate::ir::{
    Fields, Instruction, InstructionFamily as F, LdsDir, Operand, RegFile, ShaderIr, SCALAR_M0,
};
use crate::regalloc::RegisterAllocator;
use crate::CompileError;

/// Programs interpolating more attribute channels than this load each where it is used.
const HOISTED_CHANNELS: usize = 32;

pub(super) const EXP_MRT0: u8 = 0;
pub(super) const EXP_MRT1: u8 = 1;
pub(super) const EXP_MRTZ: u8 = 8;
/// gfx11 dual-source blending exports MRT0 and MRT1 swizzled to these targets.
pub(super) const EXP_DUAL_SOURCE: [u8; 2] = [21, 22];
/// SPI_SHADER_COL_FORMAT/Z_FORMAT: no export memory.
const SPI_SHADER_ZERO: u32 = 0;
/// SPI_SHADER_COL_FORMAT: one 32-bit channel.
pub const SPI_SHADER_32_R: u32 = 1;

/// The attribute and channel an interpolation instruction reads.
pub(super) fn interpolated(inst: &Instruction) -> Option<(u8, u8)> {
    match (inst.family, &inst.fields) {
        (F::Vintrp, Fields::Vintrp(i)) => Some((i.attr, i.chan)),
        (F::Vop3, Fields::Vop3(v)) if is_vop3_interp(inst.opcode) => match v.src[0] {
            Operand::Raw(select) => Some(((select & 0x3f) as u8, ((select >> 6) & 3) as u8)),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn is_vop3_interp(opcode: u16) -> bool {
    matches!(
        opcode,
        VOP3_INTERP_P1_F32
            ..=VOP3_INTERP_MOV_F32
                | VOP3_INTERP_P1LL_F16
                | VOP3_INTERP_P1LV_F16
                | VOP3_INTERP_P2_F16
    )
}

/// What the translation of one pixel shader shares.
#[derive(Clone, Debug, Default)]
pub(crate) struct PixelPlan {
    /// The SGPR the hardware loads with `prim_mask`, when the user SGPR count is known.
    pub prim_mask: Option<u8>,
    /// Attribute channels the prologue loads, and the VGPR each one lands in.
    pub hoisted: BTreeMap<(u8, u8), u8>,
    /// Interpolations that read the hoisted loads.
    pub from_prologue: BTreeSet<u32>,
    /// Where a guest null export goes: MRT0, or MRTZ when only depth has export memory.
    pub null_target: u8,
    /// Dual-source blending: four VGPRs holding the first of the MRT0/MRT1 exports.
    pub dual_source: Option<u8>,
}

/// The pixel-shader inputs of [`super::Lowerer`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PixelInterface {
    /// The user SGPR count is known, so `prim_mask` follows it.
    pub prim_mask: Option<u8>,
    pub col_format: Option<u32>,
    pub z_format: Option<u32>,
    pub dual_source: bool,
}

impl PixelPlan {
    /// Plans the translation and reserves its VGPRs, before any transient register is used.
    pub fn new(
        ir: &ShaderIr,
        regs: &mut RegisterAllocator,
        wave64: bool,
        interface: PixelInterface,
    ) -> Result<Self, CompileError> {
        let mut plan = Self {
            prim_mask: interface.prim_mask,
            null_target: null_target(interface.col_format, interface.z_format),
            ..Self::default()
        };
        if let Some(prim_mask) = interface.prim_mask {
            let at_entry = m0_holds_prim_mask(ir, wave64, prim_mask);
            let channels: BTreeSet<(u8, u8)> = ir
                .instructions
                .iter()
                .filter(|i| at_entry.contains(&i.pc))
                .filter_map(interpolated)
                .collect();
            if !channels.is_empty() && channels.len() <= HOISTED_CHANNELS {
                let base = regs.reserve_program_vgprs(channels.len() as u8)?;
                plan.hoisted = channels.into_iter().zip(base..).collect();
                plan.from_prologue = at_entry;
            }
        }
        if interface.dual_source {
            plan.dual_source = Some(regs.reserve_program_vgprs(4)?);
        }
        Ok(plan)
    }

    /// The prologue's VGPR for the interpolation at `pc`, if it reads one.
    pub fn hoisted(&self, pc: u32, attr: u8, chan: u8) -> Option<u8> {
        self.from_prologue
            .contains(&pc)
            .then(|| self.hoisted.get(&(attr, chan)).copied())
            .flatten()
    }
}

/// gfx11 has no null export target: the guest's null export becomes an export without data to
/// a target with export memory, so EXEC still kills pixels (the hardware ignores it when no
/// export memory is allocated). That is MRT0, unless only depth has memory.
fn null_target(col_format: Option<u32>, z_format: Option<u32>) -> u8 {
    let mrt0_zero = col_format.is_some_and(|f| f & 0xf == SPI_SHADER_ZERO);
    let has_depth = z_format.is_some_and(|f| f != SPI_SHADER_ZERO);
    if mrt0_zero && has_depth {
        EXP_MRTZ
    } else {
        EXP_MRT0
    }
}

/// SPI_SHADER_COL_FORMAT for a translation whose null exports go to MRT0: MRT0 needs export
/// memory for them to kill pixels, so a ZERO format becomes 32_R. `None` when nothing changes
/// or the guest format is unknown.
pub(crate) fn col_format_for(
    col_format: Option<u32>,
    z_format: Option<u32>,
    null_exports: bool,
) -> Option<u32> {
    let col = col_format?;
    let needs_memory = null_exports
        && null_target(col_format, z_format) == EXP_MRT0
        && col & 0xf == SPI_SHADER_ZERO;
    needs_memory.then_some(col | SPI_SHADER_32_R)
}

/// What is known to hold the entry `prim_mask` at one point of the program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Facts {
    m0: bool,
    sgprs: u128,
}

impl Facts {
    const NOTHING: Self = Self {
        m0: false,
        sgprs: 0,
    };
    const UNVISITED: Self = Self {
        m0: true,
        sgprs: u128::MAX,
    };

    fn meet(self, other: Self) -> Self {
        Self {
            m0: self.m0 && other.m0,
            sgprs: self.sgprs & other.sgprs,
        }
    }

    fn holds(self, operand: Operand) -> bool {
        match operand {
            Operand::M0 => self.m0,
            Operand::Sgpr(r) => self.sgprs & (1 << r) != 0,
            _ => false,
        }
    }

    fn step(&mut self, inst: &Instruction, wave64: bool) {
        if inst.has_flag(flag::CALL | flag::RELATIVE) {
            *self = Self::NOTHING;
            return;
        }
        let copied = match inst.fields {
            Fields::Sop1 { sdst, src0 }
                if inst.family == F::Sop1 && inst.opcode == SOP1_MOV_B32 =>
            {
                self.holds(src0).then_some(sdst)
            }
            _ => None,
        };
        for range in inst.defs(wave64) {
            if range.file != RegFile::Scalar {
                continue;
            }
            for reg in range.first..range.first + range.count.max(1) {
                match reg {
                    SCALAR_M0 => self.m0 = false,
                    0..=127 => self.sgprs &= !(1u128 << reg),
                    _ => {}
                }
            }
        }
        match copied {
            Some(Operand::M0) => self.m0 = true,
            Some(Operand::Sgpr(r)) => self.sgprs |= 1 << r,
            _ => {}
        }
    }
}

/// Interpolations whose M0 holds the entry value of the `prim_mask` SGPR on every path: a
/// forward must-analysis of copies through `s_mov_b32`. Calls, relative moves and blocks that
/// computed jumps enter know nothing.
fn m0_holds_prim_mask(ir: &ShaderIr, wave64: bool, prim_mask: u8) -> BTreeSet<u32> {
    let count = ir.blocks.len();
    let mut entry = vec![Facts::UNVISITED; count];
    let mut seen = vec![false; count];
    for block in ir.blocks.iter().filter(|b| b.reachable) {
        if block.id == 0 {
            entry[0] = Facts {
                m0: false,
                sgprs: 1 << prim_mask,
            };
            seen[0] = true;
        }
        if ir.flow.entries.contains(&block.start_pc) {
            entry[block.id] = Facts::NOTHING;
            seen[block.id] = true;
        }
    }
    let exit = |facts: Facts, id: usize| {
        let mut facts = facts;
        for inst in &ir.instructions[ir.blocks[id].instructions()] {
            facts.step(inst, wave64);
        }
        facts
    };
    let mut work: Vec<usize> = (0..count).filter(|&id| seen[id]).collect();
    while let Some(id) = work.pop() {
        let out = exit(entry[id], id);
        for &next in &ir.blocks[id].successors {
            let merged = if seen[next] {
                entry[next].meet(out)
            } else {
                out
            };
            if !seen[next] || merged != entry[next] {
                entry[next] = merged;
                seen[next] = true;
                work.push(next);
            }
        }
    }
    let mut found = BTreeSet::new();
    for block in ir.blocks.iter().filter(|b| b.reachable && seen[b.id]) {
        let mut facts = entry[block.id];
        for inst in &ir.instructions[block.instructions()] {
            if facts.m0 && interpolated(inst).is_some() {
                found.insert(inst.pc);
            }
            facts.step(inst, wave64);
        }
    }
    found
}

pub(super) fn param_load(origin: &Instruction, vdst: u8, attr: u8, chan: u8) -> Instruction {
    let load = LdsDir {
        vdst,
        attr,
        chan,
        wait_vdst: 0,
    };
    derive(
        origin,
        F::Ldsdir,
        LDSDIR_PARAM_LOAD,
        flag::READS_M0,
        Fields::Ldsdir(load),
    )
}

impl Ctx<'_> {
    /// Copies EXEC to `saved` (one SGPR, or an aligned pair for wave64) and enables every lane.
    pub(super) fn enable_all_lanes(
        &mut self,
        origin: &Instruction,
        saved: u8,
    ) -> Result<(), CompileError> {
        let mov = if self.wave64 {
            SOP1_MOV_B64
        } else {
            SOP1_MOV_B32
        };
        let copy = |sdst, src0| Fields::Sop1 { sdst, src0 };
        self.lower_gfx1013(
            origin,
            F::Sop1,
            mov,
            copy(Operand::Sgpr(saved), Operand::ExecLo),
        )?;
        self.lower_gfx1013(
            origin,
            F::Sop1,
            mov,
            copy(Operand::ExecLo, Operand::InlineInt(-1)),
        )
    }

    pub(super) fn restore_lanes(
        &mut self,
        origin: &Instruction,
        saved: u8,
    ) -> Result<(), CompileError> {
        let mov = if self.wave64 {
            SOP1_MOV_B64
        } else {
            SOP1_MOV_B32
        };
        let fields = Fields::Sop1 {
            sdst: Operand::ExecLo,
            src0: Operand::Sgpr(saved),
        };
        self.lower_gfx1013(origin, F::Sop1, mov, fields)
    }

    /// The prologue's parameter loads: M0 = `prim_mask`, every lane enabled, one load per
    /// hoisted channel, then a wait for all of them.
    pub(super) fn load_hoisted_parameters(
        &mut self,
        origin: &Instruction,
    ) -> Result<(), CompileError> {
        let Some(plan) = self.pixel else {
            return Ok(());
        };
        let Some(prim_mask) = plan.prim_mask.filter(|_| !plan.hoisted.is_empty()) else {
            return Ok(());
        };
        let fields = Fields::Sop1 {
            sdst: Operand::M0,
            src0: Operand::Sgpr(prim_mask),
        };
        self.lower_gfx1013(origin, F::Sop1, SOP1_MOV_B32, fields)?;
        let saved = self.transient_sgprs(origin, 2)?;
        self.enable_all_lanes(origin, saved)?;
        for (&(attr, chan), &vgpr) in &plan.hoisted {
            self.push(param_load(origin, vgpr, attr, chan));
        }
        self.restore_lanes(origin, saved)?;
        self.wait_for_parameters(origin);
        Ok(())
    }
}
