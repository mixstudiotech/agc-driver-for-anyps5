//! Primitive-ordered pixel shading (POPS) on gfx11 and later.
//!
//! Before gfx11 a wave waits for the waves it overlaps itself; ACO's
//! `pops_await_overlapped_waves`, PAL and the PS5 compiler emit the same code. The SPI loads a
//! collision word into the SGPR after `prim_mask` (bit 31: the wave overlaps an earlier one,
//! 29:28 its packer, 25:16 the newest overlapped wave's ID, 9:0 its own). A wave that overlaps
//! writes its packer to `HW_REG_POPS_PACKER` and sleeps until `src_pops_exiting_wave_id` has
//! passed the overlapped wave, runs its ordered section and ends it with
//! `s_sendmsg MSG_ORDERED_PS_DONE`. With LOAD_INTRAWAVE_COLLISION a second SGPR marks the quads
//! that overlap earlier quads of the same wave, and the program runs the section once per layer.
//!
//! gfx11 has neither the SGPRs nor the operand. A wave waits with `s_wait_event` and its `done`
//! export ends the ordered section (radv and ACO: `nir_intrinsic_begin_invocation_interlock`,
//! `is_pops_end_export`); the rasterizer keeps overlapping quads out of one wave.
//!
//! The translation keeps the guest's code and gives it the inputs that take it that way:
//! - the prologue sets the collision SGPR to "this is wave 1 and it overlaps wave 0" and the
//!   intrawave SGPR to "no quad overlaps another",
//! - the write of the packer register becomes `s_wait_event`,
//! - `src_pops_exiting_wave_id` reads as 1, this wave: every earlier wave has left, so the
//!   guest's own wait ends at its first check,
//! - MSG_ORDERED_PS_DONE is dropped, as the `done` export after it ends the section.
//!
//! The last point holds only if no `done` export can run before the ordered section ends; a
//! program where one can is refused.

use std::collections::BTreeSet;

use super::{derive, unsupported, Ctx, HWREG_ID_MASK, SOP1_MOV_B32, SOPK_SETREG, SOPK_SETREG_IMM32};
use crate::ir::{Fields, Instruction, InstructionFamily as F, Operand, ShaderIr};
use crate::{CompileError, ShaderStage, TargetIsa};

pub(super) const HWREG_POPS_PACKER: u16 = 25;
pub(super) const MSG_ORDERED_PS_DONE: u16 = 7;
/// The wave ID the translation gives the wave. Any earlier wave is then one the guest can wait for.
const WAVE_ID: u32 = 1;
/// The collision word: the wave overlaps (bit 31) wave 0 (bits 25:16), on packer 0, and is
/// wave [`WAVE_ID`] (bits 9:0).
const COLLISION: u32 = 1 << 31 | WAVE_ID;
const SOPP_SENDMSG: u16 = 0x10;
const SOPP_SENDMSGHALT: u16 = 0x11;
const SOPP_GFX11_WAIT_EVENT: u16 = 0x0b;
/// `s_wait_event` immediates that wait for `export_ready`: gfx11 waits unless bit 0 is set,
/// gfx12 waits when bit 1 is (ACO's `wait_event_imm`).
const GFX11_WAIT_EXPORT_READY: u16 = 0;
const GFX12_WAIT_EXPORT_READY: u16 = 2;

/// Whether the instruction is part of the gfx10 ordering protocol: a write of the packer
/// register, the end-of-section message or a read of the exiting wave ID.
fn orders(inst: &Instruction) -> bool {
    let reads_exiting = |operands: &[Operand]| operands.contains(&Operand::PopsExitingWaveId);
    match inst.fields {
        Fields::Sopk { simm16, .. } => {
            matches!(inst.opcode, SOPK_SETREG | SOPK_SETREG_IMM32) && simm16 & HWREG_ID_MASK == HWREG_POPS_PACKER
        }
        Fields::Sopp { simm16 } => {
            matches!(inst.opcode, SOPP_SENDMSG | SOPP_SENDMSGHALT) && simm16 & 0xf == MSG_ORDERED_PS_DONE
        }
        Fields::Sop1 { src0, .. } => inst.family == F::Sop1 && reads_exiting(&[src0]),
        Fields::Sop2 { src0, src1, .. } | Fields::Sopc { src0, src1 } => reads_exiting(&[src0, src1]),
        _ => false,
    }
}

/// Refuses a pixel program in which a `done` export can run before the ordered section has
/// ended: on gfx11 that export would end the section early.
pub(super) fn check(ir: &ShaderIr, target: TargetIsa, stage: ShaderStage) -> Result<(), CompileError> {
    if stage != ShaderStage::Fragment || !target.is_gfx11_plus() {
        return Ok(());
    }
    let blocks = || ir.blocks.iter().filter(|b| b.reachable);
    let ordering: Vec<(usize, u32)> = blocks()
        .flat_map(|b| ir.instructions[b.instructions()].iter().filter(|i| orders(i)).map(|i| (b.id, i.pc)))
        .collect();
    if ordering.is_empty() {
        return Ok(());
    }
    for block in blocks() {
        for export in &ir.instructions[block.instructions()] {
            if !matches!(export.fields, Fields::Exp(e) if e.done) {
                continue;
            }
            // Blocks control can reach after the export.
            let mut after = BTreeSet::new();
            let mut stack = block.successors.clone();
            while let Some(id) = stack.pop() {
                if after.insert(id) {
                    stack.extend(ir.blocks[id].successors.iter().copied());
                }
            }
            let late = ordering.iter().any(|&(id, pc)| after.contains(&id) || (id == block.id && pc > export.pc));
            if late {
                return Err(unsupported(
                    export,
                    target,
                    "the done export can run before the ordered section (POPS) ends, which on the target would end it",
                ));
            }
        }
    }
    Ok(())
}

/// The instruction with every read of `src_pops_exiting_wave_id` replaced by this wave's ID, if
/// it has one.
pub(super) fn without_exiting_wave(inst: &Instruction) -> Option<Instruction> {
    let mut rewritten = inst.clone();
    let mut found = false;
    for operand in super::source_operands(&mut rewritten.fields) {
        if *operand == Operand::PopsExitingWaveId {
            *operand = Operand::constant(WAVE_ID);
            found = true;
        }
    }
    found.then_some(rewritten)
}

impl Ctx<'_> {
    /// The prologue's part: the collision inputs the guest reads, which gfx11 does not load.
    pub(super) fn pops_inputs(&mut self, origin: &Instruction, live_in_scalar: u128) -> Result<(), CompileError> {
        if self.stage != ShaderStage::Fragment || !self.target.is_gfx11_plus() {
            return Ok(());
        }
        let (collision, intrawave) = self.regs.collision_sgprs();
        for (sgpr, value) in [(collision, COLLISION), (intrawave, 0)] {
            let Some(sgpr) = sgpr.filter(|&r| live_in_scalar & (1 << r) != 0) else { continue };
            let fields = Fields::Sop1 {
                sdst: Operand::Sgpr(sgpr),
                src0: Operand::constant(value),
            };
            self.lower_gfx1013(origin, F::Sop1, SOP1_MOV_B32, fields)?;
        }
        Ok(())
    }

    /// A write of `HW_REG_POPS_PACKER`, which starts the guest's wait for the waves it overlaps:
    /// on gfx11 and later it is the wait itself. (gfx10.3 has the register and keeps the write.)
    pub(super) fn pops_packer(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        if !matches!(inst.opcode, SOPK_SETREG | SOPK_SETREG_IMM32) {
            return Err(unsupported(inst, self.target, "the POPS packer register cannot be read on the target"));
        }
        let wait = if self.target.is_gfx12() { GFX12_WAIT_EXPORT_READY } else { GFX11_WAIT_EXPORT_READY };
        self.push(derive(inst, F::Sopp, SOPP_GFX11_WAIT_EVENT, 0, Fields::Sopp { simm16: wait }));
        Ok(())
    }
}
