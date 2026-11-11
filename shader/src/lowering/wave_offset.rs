//! What a program does with the gfx10 scratch wave offset, on targets that have none.
//!
//! gfx10 loads an SGPR with the offset of the wave's scratch in the scratch ring. Programs use
//! it in two ways. As the SOFFSET of a buffer access through the scratch V# it addresses the
//! private segment, which the target reaches through its own scratch
//! ([`super::Ctx::private_buffer_access`] and its flat form). And the entry code adds it to the
//! ring's address and writes the sum to `HW_REG_FLAT_SCR_LO/HI`, which the target's hardware
//! sets for every wave, so those writes are dropped.
//!
//! The translation gives the SGPR the value 0, which a guest wave can have too, so nothing the
//! guest computes or branches on becomes invalid by it. What 0 cannot stand for is the place
//! of this wave's scratch: a value computed from the offset must not address memory anywhere
//! but through those two uses. A forward dataflow follows the offset through scalar
//! instructions, and SCC as the carry between them, and refuses the program when a value
//! computed from it reaches anything else. A buffer access is private only where its SOFFSET
//! register still holds the offset itself on every path: the guest reuses the SGPR.

use std::collections::BTreeSet;

use super::{unsupported, HWREG_FLAT_SCR_HI, HWREG_FLAT_SCR_LO, HWREG_ID_MASK, SOPK_SETREG, SOPK_SETREG_IMM32};
use crate::ir::opcodes::flag;
use crate::ir::{Fields, Instruction, InstructionFamily as F, Operand, RegFile, ShaderIr};
use crate::{CompileError, TargetIsa};

/// What a program does with the scratch wave offset.
#[derive(Clone, Debug, Default)]
pub(super) struct WaveOffsetUses {
    /// Buffer accesses whose SOFFSET is the offset itself: the private segment.
    pub private: BTreeSet<u32>,
    /// Scalar instructions compute with the offset, so its SGPR needs a value.
    pub computed: bool,
}

/// What is known at one point of the program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Flow {
    /// Scalar registers that may hold the offset or a value computed from it.
    derived: u128,
    /// SCC may come from such a value.
    scc: bool,
    /// The offset's SGPR holds the offset itself on every path.
    exact: bool,
    reached: bool,
}

impl Flow {
    const UNREACHED: Self = Self {
        derived: 0,
        scc: false,
        exact: true,
        reached: false,
    };

    fn join(self, other: Self) -> Self {
        match (self.reached, other.reached) {
            (false, _) => other,
            (_, false) => self,
            _ => Self {
                derived: self.derived | other.derived,
                scc: self.scc || other.scc,
                exact: self.exact && other.exact,
                reached: true,
            },
        }
    }
}

/// What one instruction does with the offset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Use {
    None,
    /// A buffer access addressed by the offset itself.
    Private,
    /// A scalar instruction computing with it.
    Computed,
    /// A write of the flat scratch registers, which the target drops.
    FlatScratch,
    /// Anything else: not translatable.
    Stray,
}

fn scalar_mask(inst: &Instruction, wave64: bool, defs: bool) -> u128 {
    let mut mask = 0u128;
    let mut add = |range: crate::ir::RegRange| {
        if range.file == RegFile::Scalar {
            mask |= crate::regalloc::scalar_mask(range.first, range.count.max(1));
        }
    };
    if defs {
        inst.each_def(wave64, &mut add);
    } else {
        inst.each_use(wave64, &mut add);
    }
    mask
}

/// Scalar instructions whose result depends on SCC: `s_addc_u32`, `s_subb_u32`, the selects
/// and conditional moves, and any instruction with SCC as a source operand. (Branches on SCC
/// are not listed: with an offset of 0 they go a way the guest can go.)
fn reads_scc(inst: &Instruction) -> bool {
    let operand = |operands: &[Operand]| operands.contains(&Operand::Scc);
    match (inst.family, inst.opcode, inst.fields) {
        (F::Sop2, 0x04 | 0x05 | 0x0a | 0x0b, _) | (F::Sop1, 0x05 | 0x06, _) | (F::Sopk, 0x02, _) => true,
        (_, _, Fields::Sop1 { src0, .. }) => operand(&[src0]),
        (_, _, Fields::Sop2 { src0, src1, .. } | Fields::Sopc { src0, src1 }) => operand(&[src0, src1]),
        _ => false,
    }
}

/// Whether the scalar instruction sets SCC: `Some(false)` for those that leave it alone, `None`
/// where the opcode tables do not say (SOP1), which is taken as "may".
fn writes_scc(inst: &Instruction) -> Option<bool> {
    match (inst.family, inst.opcode) {
        (F::Sopc, _) => Some(true),
        // s_cselect, s_bfm, s_mul_i32, s_pack_*, s_mul_hi_* leave SCC alone.
        (F::Sop2, 0x0a | 0x0b | 0x24 | 0x25 | 0x26 | 0x32..=0x36) => Some(false),
        (F::Sop2, _) => Some(true),
        // s_cmpk_* and s_addk_i32.
        (F::Sopk, 0x03..=0x0f) => Some(true),
        (F::Sopk, _) => Some(false),
        _ => None,
    }
}

fn is_scalar_alu(inst: &Instruction) -> bool {
    matches!(inst.family, F::Sop1 | F::Sop2 | F::Sopc) || (inst.family == F::Sopk && !matches!(inst.opcode, SOPK_SETREG | SOPK_SETREG_IMM32))
}

fn writes_flat_scratch(inst: &Instruction) -> bool {
    matches!(inst.fields, Fields::Sopk { simm16, .. }
        if inst.family == F::Sopk
            && inst.opcode == SOPK_SETREG
            && matches!(simm16 & HWREG_ID_MASK, HWREG_FLAT_SCR_LO | HWREG_FLAT_SCR_HI))
}

/// Applies `inst` to `flow` and says what it does with the offset held in `sgpr`. With
/// `forget` off, registers and SCC never lose what they may hold (for programs whose control
/// flow is not all known).
fn step(inst: &Instruction, flow: &mut Flow, sgpr: u8, wave64: bool, forget: bool) -> Use {
    let offset = 1u128 << sgpr;
    let reads = scalar_mask(inst, wave64, false);
    let defs = scalar_mask(inst, wave64, true);
    let from_scc = flow.scc && reads_scc(inst);
    let derived_read = reads & flow.derived != 0 || from_scc;
    let soffset = match inst.fields {
        Fields::Mubuf(b) | Fields::Mtbuf(b) => Some(b.soffset),
        _ => None,
    };
    let used = if !derived_read {
        Use::None
    } else if soffset == Some(Operand::Sgpr(sgpr)) && flow.exact && reads & flow.derived == offset {
        Use::Private
    } else if writes_flat_scratch(inst) {
        Use::FlatScratch
    } else if is_scalar_alu(inst) && !inst.has_flag(flag::JUMP | flag::CALL) {
        Use::Computed
    } else {
        Use::Stray
    };
    if used == Use::Computed {
        flow.derived |= defs;
    } else if forget {
        flow.derived &= !defs;
    }
    if defs & offset != 0 {
        flow.exact = false;
    }
    if is_scalar_alu(inst) {
        match writes_scc(inst) {
            Some(true) if forget => flow.scc = derived_read,
            Some(true) | None => flow.scc |= derived_read,
            Some(false) => {}
        }
    }
    used
}

impl WaveOffsetUses {
    /// Follows the offset loaded in `sgpr` through the program.
    pub fn find(ir: &ShaderIr, wave64: bool, sgpr: u8, target: TargetIsa) -> Result<Self, CompileError> {
        let entry = Flow {
            derived: 1 << sgpr,
            scc: false,
            exact: true,
            reached: true,
        };
        let blocks = ir.blocks.len();
        let mut at_entry = vec![Flow::UNREACHED; blocks];
        if ir.flow.has_dynamic_jumps() {
            // Control can enter blocks in ways the block graph does not show: one state for
            // the whole program, in which nothing is ever forgotten.
            let mut all = entry;
            loop {
                let before = all;
                for inst in ir.blocks.iter().filter(|b| b.reachable).flat_map(|b| &ir.instructions[b.instructions()]) {
                    step(inst, &mut all, sgpr, wave64, false);
                }
                if all == before {
                    break;
                }
            }
            at_entry.fill(all);
        } else {
            if let Some(first) = at_entry.first_mut() {
                *first = entry;
            }
            let mut work: Vec<usize> = vec![0];
            while let Some(id) = work.pop() {
                let mut flow = at_entry[id];
                for inst in &ir.instructions[ir.blocks[id].instructions()] {
                    step(inst, &mut flow, sgpr, wave64, true);
                }
                for &next in &ir.blocks[id].successors {
                    let joined = at_entry[next].join(flow);
                    if joined != at_entry[next] {
                        at_entry[next] = joined;
                        work.push(next);
                    }
                }
            }
        }
        let forget = !ir.flow.has_dynamic_jumps();
        let mut uses = Self::default();
        for block in ir.blocks.iter().filter(|b| b.reachable && at_entry[b.id].reached) {
            let mut flow = at_entry[block.id];
            for inst in &ir.instructions[block.instructions()] {
                match step(inst, &mut flow, sgpr, wave64, forget) {
                    Use::Private => {
                        uses.private.insert(inst.pc);
                    }
                    Use::Computed => uses.computed = true,
                    Use::Stray => {
                        return Err(unsupported(
                            inst,
                            target,
                            "a value computed from the scratch wave offset is used outside the private segment and the flat scratch setup",
                        ))
                    }
                    Use::None | Use::FlatScratch => {}
                }
            }
        }
        Ok(uses)
    }
}
