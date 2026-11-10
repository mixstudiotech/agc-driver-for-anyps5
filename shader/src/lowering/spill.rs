//! SGPR spilling to VGPR lanes, as ACO spills SGPRs to a linear VGPR. A program that leaves the
//! translation too few SGPRs above its own gets the transient SGPRs of an expansion from an
//! aligned block of the guest's SGPRs that the instruction being lowered neither reads nor
//! writes: the block is saved to lanes of a reserved VGPR before the expansion and restored
//! after the instruction (`v_writelane_b32`/`v_readlane_b32`, which ignore EXEC). Blocks never
//! reach the SGPRs the translation owns ([`crate::regalloc::RegisterAllocator::translation_sgprs`]), which other
//! expansions of the same instruction may be using.

use super::{unsupported, waitcnt, Ctx};
use crate::ir::{Fields, Instruction, InstructionFamily as F, Operand, RegFile, Vop3, SGPR_LIMIT};
use crate::CompileError;

const VOP3_READLANE_B32: u16 = 0x360;
const VOP3_WRITELANE_B32: u16 = 0x361;
/// Blocks start at multiples of 4, as descriptors held in SGPRs must.
const BLOCK_ALIGN: u16 = 4;

/// The reserved VGPR and the SGPR blocks saved to its lanes for the instruction being lowered.
/// Its top `persistent` lanes hold values the whole program keeps (an NGG plan's, which found
/// no SGPRs: [`super::ngg::NggPlan::persistent_lanes`]); blocks use the lanes below them.
#[derive(Clone, Debug)]
pub(crate) struct SpillLanes {
    pub vgpr: u8,
    persistent: u8,
    /// First SGPR, SGPRs and first lane of each block saved for the current instruction.
    saved: Vec<(u8, u8, u8)>,
}

impl SpillLanes {
    pub fn new(vgpr: u8, persistent: u8) -> Self {
        Self {
            vgpr,
            persistent,
            saved: Vec::new(),
        }
    }
}

impl Ctx<'_> {
    /// `count` SGPRs (a block aligned to 4) for an expansion of `origin`: SGPRs `origin` does not
    /// touch and the translation does not reserve, saved to lanes of the spill VGPR until
    /// [`Self::restore_spills`]. Blocks of one instruction never overlap.
    pub(crate) fn spill_block(
        &mut self,
        origin: &Instruction,
        count: u8,
    ) -> Result<u8, CompileError> {
        let Some(spill) = self.spill.as_ref() else {
            return Err(unsupported(
                origin,
                self.target,
                "the translation needs SGPRs above the shader's own",
            ));
        };
        let lanes = if self.wave64 { 64 } else { 32 } - u16::from(spill.persistent);
        let size = u16::from(count).next_multiple_of(BLOCK_ALIGN);
        let used_lanes: u16 = spill.saved.iter().map(|&(_, n, _)| u16::from(n)).sum();
        if used_lanes + size > lanes {
            return Err(unsupported(
                origin,
                self.target,
                "the SGPRs one instruction spills exceed the lanes of a VGPR",
            ));
        }
        let mut touched = origin.uses(self.wave64);
        touched.extend(origin.defs(self.wave64));
        touched.retain(|r| r.file == RegFile::Scalar);
        let owned = self.regs.translation_sgprs();
        let taken: Vec<(u16, u16)> = spill
            .saved
            .iter()
            .map(|&(b, n, _)| (u16::from(b), u16::from(n)))
            .collect();
        let overlaps = |base: u16| {
            let end = base + size;
            touched
                .iter()
                .any(|r| r.first < end && base < r.first + r.count.max(1))
                || (owned.start < end && base < owned.end)
                || taken.iter().any(|&(b, n)| b < end && base < b + n)
        };
        let mut base = (SGPR_LIMIT - size) / BLOCK_ALIGN * BLOCK_ALIGN;
        while overlaps(base) {
            if base < BLOCK_ALIGN {
                return Err(unsupported(
                    origin,
                    self.target,
                    "no SGPR block the instruction leaves alone can be spilled",
                ));
            }
            base -= BLOCK_ALIGN;
        }
        let (vgpr, first_lane) = (spill.vgpr, used_lanes as u8);
        // An earlier SMEM load may still be writing this borrowed block. The guest did
        // not read it here, so its own wait can be later than our save and overwrite.
        self.lower_gfx1013(
            origin,
            F::Sopp,
            crate::ir::opcodes::SOPP_WAITCNT,
            Fields::Sopp {
                simm16: waitcnt::encode_gfx10(waitcnt::GFX10_VMCNT_MAX, waitcnt::GFX10_EXPCNT_MAX, 0),
            },
        )?;
        for k in 0..size as u8 {
            let write = Vop3::new(
                Operand::Vgpr(vgpr),
                [
                    Operand::Sgpr(base as u8 + k),
                    Operand::InlineInt((first_lane + k) as i8),
                    Operand::Raw(0),
                ],
            );
            self.lower_gfx1013(origin, F::Vop3, VOP3_WRITELANE_B32, Fields::Vop3(write))?;
        }
        if let Some(spill) = self.spill.as_mut() {
            spill.saved.push((base as u8, size as u8, first_lane));
        }
        Ok(base as u8)
    }

    /// `v_readlane_b32 sdst, spill, lane`: an SGPR from a lane of the spill VGPR.
    pub(crate) fn read_lane(
        &mut self,
        origin: &Instruction,
        sdst: u8,
        lane: u8,
    ) -> Result<(), CompileError> {
        let vgpr = self.spill_vgpr(origin)?;
        let read = Vop3::new(
            Operand::Sgpr(sdst),
            [Operand::Vgpr(vgpr), Operand::InlineInt(lane as i8), Operand::Raw(0)],
        );
        self.lower_gfx1013(origin, F::Vop3, VOP3_READLANE_B32, Fields::Vop3(read))
    }

    /// `v_writelane_b32 spill, src, lane`: an SGPR or constant into a lane of the spill VGPR.
    pub(crate) fn write_lane(
        &mut self,
        origin: &Instruction,
        src: Operand,
        lane: u8,
    ) -> Result<(), CompileError> {
        let vgpr = self.spill_vgpr(origin)?;
        let write = Vop3::new(
            Operand::Vgpr(vgpr),
            [src, Operand::InlineInt(lane as i8), Operand::Raw(0)],
        );
        self.lower_gfx1013(origin, F::Vop3, VOP3_WRITELANE_B32, Fields::Vop3(write))
    }

    fn spill_vgpr(&self, origin: &Instruction) -> Result<u8, CompileError> {
        self.spill.as_ref().map(|spill| spill.vgpr).ok_or_else(|| {
            unsupported(
                origin,
                self.target,
                "the translation needs SGPRs above the shader's own",
            )
        })
    }

    /// Restores every block [`Self::spill_block`] saved for `origin`, after its instructions.
    pub(crate) fn restore_spills(&mut self, origin: &Instruction) -> Result<(), CompileError> {
        let Some(spill) = self.spill.as_mut() else {
            return Ok(());
        };
        if spill.saved.is_empty() {
            return Ok(());
        }
        let (vgpr, saved) = (spill.vgpr, std::mem::take(&mut spill.saved));
        for (base, size, first_lane) in saved {
            for k in 0..size {
                let read = Vop3::new(
                    Operand::Sgpr(base + k),
                    [
                        Operand::Vgpr(vgpr),
                        Operand::InlineInt((first_lane + k) as i8),
                        Operand::Raw(0),
                    ],
                );
                self.lower_gfx1013(origin, F::Vop3, VOP3_READLANE_B32, Fields::Vop3(read))?;
            }
        }
        Ok(())
    }
}
