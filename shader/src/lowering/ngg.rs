//! NGG vertex programs on gfx11+: parameter exports become attribute-ring stores (see
//! [`crate::ngg`]), and the position export that ends the wave (`done`) waits for them first:
//! on gfx11 the hardware may reuse a culled subgroup's ring space while its stores are still in
//! flight.
//!
//! When no parameter export can follow a done position export, the stores stay where the guest
//! exports and `s_waitcnt_vscnt null, 0` precedes the done export. Otherwise the done export
//! is deferred: where the guest issues it, its data and EXEC are saved, and the epilogue, which
//! every `s_endpgm` branches to, waits for the stores and exports.

use super::{
    derive, unsupported, Ctx, SOP1_MOV_B32, SOP1_MOV_B64, SOP2_BFE_U32, SOPC_CMP_EQ_U32,
    SOPC_CMP_EQ_U64, SOPC_CMP_LG_U32,
};
use crate::ir::opcodes::{self, flag, Xlat};
use crate::ir::{
    memops, Buffer, Export, Fields, Instruction, InstructionFamily as F, JumpTarget, MemKind,
    Operand, ShaderIr, Vop3,
};
use crate::regalloc::RegisterAllocator;
use crate::{AttributeRingAbi, CompileError};

/// Block labels of the epilogue (see [`super::PROLOGUE_LABEL`]); above every guest pc and
/// [`super::DISPATCH_LABEL`], so dropped-block redirection leaves branches to them alone.
pub const EPILOGUE_LABEL: u32 = 0xffff_fff8;
pub const EPILOGUE_END_LABEL: u32 = 0xffff_fffc;

pub(super) const EXP_POS_FIRST: u8 = 12;
pub(super) const EXP_POS_LAST: u8 = 16;
pub(super) const EXP_PARAM_FIRST: u8 = 32;
const EXP_PARAM_LAST: u8 = 63;
/// gfx1013 opcodes.
const MUBUF_STORE_DWORDX2: u16 = 0x1d;
const MUBUF_STORE_DWORDX4: u16 = 0x1e;
const SOP2_LSHL_B32: u16 = 0x1e;
const SOP2_OR_B32: u16 = 0x10;
const SOP2_OR_B64: u16 = 0x11;
const SOPK_WAITCNT_VSCNT: u16 = 0x17;
const SOPP_CBRANCH_SCC1: u16 = 0x05;
const VOP3_MBCNT_LO: u16 = 0x365;
const VOP3_MBCNT_HI: u16 = 0x366;
/// `s_bfe_u32` controls: s5 bits 14:0 (`gs_attr_offset`), s3 bits 27:24 (wave index).
const ATTR_OFFSET_FIELD: u32 = 15 << 16;
const WAVE_INDEX_FIELD: u32 = 24 | (4 << 16);
const ATTR_OFFSET_UNIT_SHIFT: u32 = 9;
const GS_ATTR_OFFSET_SGPR: u8 = 5;
const MERGED_WAVE_INFO_SGPR: u8 = 3;

pub(super) fn is_param(target: u8) -> bool {
    (EXP_PARAM_FIRST..=EXP_PARAM_LAST).contains(&target)
}

fn is_done_position(e: &Export) -> bool {
    e.done && (EXP_POS_FIRST..=EXP_POS_LAST).contains(&e.target)
}

fn export(inst: &Instruction) -> Option<&Export> {
    match &inst.fields {
        Fields::Exp(e) => Some(e),
        _ => None,
    }
}

/// Where a value the whole program keeps lives: SGPRs reserved above the program's, or, for a
/// program that leaves none, lanes of the spill VGPR ([`super::spill`]) counted from its top.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kept {
    Sgpr(u8),
    Lane(u8),
}

/// The attribute ring's registers: its descriptor, this wave's byte offset into it, and the
/// lane's vertex index in the subgroup. With no free SGPRs, the descriptor and offset are
/// initialized once in persistent spill lanes and read into a transient block at each store.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ring {
    pub descriptor: Kept,
    pub offset: Kept,
    pub vertex: u8,
    pub words: [u32; 4],
}

/// The deferred done position export: its target and enables, and the registers holding its
/// data and the lanes that issued it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Deferred {
    pub target: u8,
    pub en: u8,
    pub data: u8,
    pub exec: Kept,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NggPlan {
    pub ring: Option<Ring>,
    pub deferred: Option<Deferred>,
    /// Lanes at the top of the spill VGPR the plan keeps values in.
    pub persistent_lanes: u8,
}

/// The lane `index` from the top of a wave's spill VGPR.
fn top_lane(wave64: bool, index: u8) -> u8 {
    (if wave64 { 63 } else { 31 }) - index
}

impl NggPlan {
    /// Plans the translation and reserves its registers, before any transient register is used.
    /// With `spill` (a spill VGPR is reserved), values that find no SGPRs above the program's
    /// are kept in lanes of the spill VGPR instead.
    pub fn new(
        ir: &ShaderIr,
        regs: &mut RegisterAllocator,
        wave64: bool,
        ring: Option<AttributeRingAbi>,
        spill: bool,
    ) -> Result<Self, CompileError> {
        let param_count = ir.metadata.exports.param_count();
        let mut plan = Self::default();
        let (Some(ring), true) = (ring, param_count > 0) else {
            return Ok(plan);
        };
        let first = ir.instructions.first().ok_or(CompileError::EmptyProgram)?;
        let target = regs.target();
        let no_sgprs = || {
            unsupported(
                first,
                target,
                "attribute-ring stores need SGPRs above the shader's own",
            )
        };
        let (descriptor, offset) = match regs.reserve_sgprs(5) {
            Some(descriptor) => (Kept::Sgpr(descriptor), Kept::Sgpr(descriptor + 4)),
            None if spill => {
                plan.persistent_lanes += 5;
                (
                    Kept::Lane(top_lane(wave64, 4)),
                    Kept::Lane(top_lane(wave64, 0)),
                )
            }
            _ => return Err(no_sgprs()),
        };
        let vertex = regs.reserve_program_vgprs(1)?;
        plan.ring = Some(Ring {
            descriptor,
            offset,
            vertex,
            words: crate::ngg::ring_descriptor(ring, param_count),
        });
        if params_follow_done_position(ir) {
            let done: Vec<&Export> = ir
                .blocks
                .iter()
                .filter(|b| b.reachable)
                .flat_map(|b| &ir.instructions[b.instructions()])
                .filter_map(export)
                .filter(|e| is_done_position(e))
                .collect();
            let Some(&&first_done) = done.first() else {
                return Ok(plan);
            };
            let leaves = ir.flow.jumps.values().any(|t| *t == JumpTarget::External);
            if leaves {
                return Err(unsupported(
                    first,
                    target,
                    "a deferred done position export cannot follow a jump out of the program",
                ));
            }
            if done.iter().any(|e| {
                (e.target, e.en, e.compr) != (first_done.target, first_done.en, first_done.compr)
            }) {
                return Err(unsupported(
                    first,
                    target,
                    "deferring done position exports with different targets or channels",
                ));
            }
            let data = regs.reserve_program_vgprs(4)?;
            let exec = match regs.reserve_sgprs(if wave64 { 2 } else { 1 }) {
                Some(sgpr) => Kept::Sgpr(sgpr),
                // The mask's low dword, and its high one in the lane below for wave64.
                None if spill => {
                    let lane = top_lane(wave64, plan.persistent_lanes + u8::from(wave64));
                    plan.persistent_lanes += if wave64 { 2 } else { 1 };
                    Kept::Lane(lane)
                }
                None => return Err(no_sgprs()),
            };
            plan.deferred = Some(Deferred {
                target: first_done.target,
                en: first_done.en,
                data,
                exec,
            });
        }
        Ok(plan)
    }
}

/// Whether a parameter export can run after a done position export: later in its block or in
/// a block reachable from it. Computed jumps make every block reachable.
fn params_follow_done_position(ir: &ShaderIr) -> bool {
    let has_param =
        |range: &[Instruction]| range.iter().filter_map(export).any(|e| is_param(e.target));
    let mut seen = vec![false; ir.blocks.len()];
    let mut work: Vec<usize> = Vec::new();
    for block in ir.blocks.iter().filter(|b| b.reachable) {
        let insts = &ir.instructions[block.instructions()];
        if let Some(at) = insts
            .iter()
            .position(|i| export(i).is_some_and(is_done_position))
        {
            if ir.flow.has_dynamic_jumps() || has_param(&insts[at + 1..]) {
                return true;
            }
            work.extend(&block.successors);
        }
    }
    while let Some(id) = work.pop() {
        if std::mem::replace(&mut seen[id], true) {
            continue;
        }
        let block = &ir.blocks[id];
        if has_param(&ir.instructions[block.instructions()]) {
            return true;
        }
        work.extend(&block.successors);
    }
    false
}

impl Ctx<'_> {
    /// The prologue's part: the wave's ring offset from s5 and the lane's vertex index from
    /// s3, both read before guest code can overwrite them, the ring descriptor, and the
    /// deferred export's empty lane mask.
    pub(super) fn ngg_prologue(&mut self, origin: &Instruction) -> Result<(), CompileError> {
        let Some(plan) = self.ngg else {
            return Ok(());
        };
        if let Some(ring) = plan.ring {
            let sop2 = |sdst, src0, src1| Fields::Sop2 { sdst, src0, src1 };
            // The offset and the first lane's index, computed in transient SGPRs where the
            // offset is kept in a lane.
            let (offset_sgpr, first_lane) = match ring.offset {
                Kept::Sgpr(offset) => (offset, self.transient_sgprs(origin, 1)?),
                Kept::Lane(_) => {
                    let t = self.transient_sgprs(origin, 2)?;
                    (t, t + 1)
                }
            };
            let offset = Operand::Sgpr(offset_sgpr);
            let field = Operand::Literal(ATTR_OFFSET_FIELD);
            let s5 = Operand::Sgpr(GS_ATTR_OFFSET_SGPR);
            self.lower_gfx1013(origin, F::Sop2, SOP2_BFE_U32, sop2(offset, s5, field))?;
            let unit = Operand::constant(ATTR_OFFSET_UNIT_SHIFT);
            self.lower_gfx1013(origin, F::Sop2, SOP2_LSHL_B32, sop2(offset, offset, unit))?;
            if let Kept::Lane(lane) = ring.offset {
                self.write_lane(origin, offset, lane)?;
            }
            let first_lane = Operand::Sgpr(first_lane);
            let s3 = Operand::Sgpr(MERGED_WAVE_INFO_SGPR);
            let field = Operand::Literal(WAVE_INDEX_FIELD);
            self.lower_gfx1013(origin, F::Sop2, SOP2_BFE_U32, sop2(first_lane, s3, field))?;
            let lanes = Operand::constant(if self.wave64 { 6 } else { 5 });
            let fields = sop2(first_lane, first_lane, lanes);
            self.lower_gfx1013(origin, F::Sop2, SOP2_LSHL_B32, fields)?;
            let vertex = Operand::Vgpr(ring.vertex);
            let all = Operand::InlineInt(-1);
            let mbcnt = Vop3::new(vertex, [all, first_lane, Operand::Raw(0)]);
            self.lower_gfx1013(origin, F::Vop3, VOP3_MBCNT_LO, Fields::Vop3(mbcnt))?;
            if self.wave64 {
                let mbcnt = Vop3::new(vertex, [all, vertex, Operand::Raw(0)]);
                self.lower_gfx1013(origin, F::Vop3, VOP3_MBCNT_HI, Fields::Vop3(mbcnt))?;
            }
            match ring.descriptor {
                Kept::Sgpr(descriptor) => {
                    self.ring_descriptor(origin, descriptor, ring.words)?;
                }
                Kept::Lane(lane) => {
                    let descriptor = self.transient_sgprs(origin, 4)?;
                    self.ring_descriptor(origin, descriptor, ring.words)?;
                    for k in 0..4 {
                        self.write_lane(origin, Operand::Sgpr(descriptor + k), lane + k)?;
                    }
                }
            }
        }
        if let Some(deferred) = plan.deferred {
            match deferred.exec {
                Kept::Sgpr(exec) => {
                    let fields = Fields::Sop1 {
                        sdst: Operand::Sgpr(exec),
                        src0: Operand::InlineInt(0),
                    };
                    self.lower_gfx1013(origin, F::Sop1, self.lane_mask_mov(), fields)?;
                }
                Kept::Lane(lane) => {
                    for k in 0..if self.wave64 { 2 } else { 1 } {
                        self.write_lane(origin, Operand::InlineInt(0), lane + k)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The ring descriptor's constant words into the SGPRs from `sgpr`.
    fn ring_descriptor(
        &mut self,
        origin: &Instruction,
        sgpr: u8,
        words: [u32; 4],
    ) -> Result<(), CompileError> {
        for (k, word) in words.into_iter().enumerate() {
            let fields = Fields::Sop1 {
                sdst: Operand::Sgpr(sgpr + k as u8),
                src0: Operand::Literal(word),
            };
            self.lower_gfx1013(origin, F::Sop1, SOP1_MOV_B32, fields)?;
        }
        Ok(())
    }

    /// The deferred export's lane mask from the lanes it is kept in into the SGPRs from `sgpr`
    /// (an even pair for wave64).
    fn read_mask(&mut self, origin: &Instruction, sgpr: u8, lane: u8) -> Result<(), CompileError> {
        for k in 0..if self.wave64 { 2 } else { 1 } {
            self.read_lane(origin, sgpr + k, lane + k)?;
        }
        Ok(())
    }

    /// Merged ES/GS and LS/HS waves start with an EXEC the hardware does not set: every lane on
    /// before anything else, as ACO, radeonsi and radv's NGG shaders begin. The guest's own
    /// code then masks lanes from the merged wave info.
    pub(super) fn init_full_exec(&mut self, origin: &Instruction) -> Result<(), CompileError> {
        let fields = Fields::Sop1 {
            sdst: Operand::ExecLo,
            src0: Operand::InlineInt(-1),
        };
        self.lower_gfx1013(origin, F::Sop1, self.lane_mask_mov(), fields)
    }

    fn lane_mask_mov(&self) -> u16 {
        if self.wave64 {
            SOP1_MOV_B64
        } else {
            SOP1_MOV_B32
        }
    }

    /// A guest `exp paramN`: the enabled channels (zero for the others) to the lane's vertex
    /// slot of parameter N, two dwords for a packed export.
    pub(super) fn ring_store(
        &mut self,
        inst: &Instruction,
        e: &Export,
        ring: Ring,
    ) -> Result<(), CompileError> {
        let (dwords, enabled): (u8, [bool; 4]) = if e.compr {
            (2, [e.en & 0x3 != 0, e.en & 0xc != 0, false, false])
        } else {
            (4, [0, 1, 2, 3].map(|c| e.en & (1 << c) != 0))
        };
        let contiguous = (0..dwords).all(|c| {
            enabled[usize::from(c)] && e.vsrc[usize::from(c)] == e.vsrc[0].wrapping_add(c)
        });
        let data = if contiguous {
            e.vsrc[0]
        } else {
            let data = self.regs.scratch_vgprs(dwords)?;
            for c in 0..dwords {
                let src = if enabled[usize::from(c)] {
                    Operand::Vgpr(e.vsrc[usize::from(c)])
                } else {
                    Operand::InlineInt(0)
                };
                let fields = Fields::Vop1 {
                    vdst: Operand::Vgpr(data + c),
                    src0: src,
                    ext: crate::ir::VopExt::None,
                };
                self.lower_gfx1013(inst, F::Vop1, opcodes::VOP1_MOV_B32, fields)?;
            }
            data
        };
        let opcode = if dwords == 2 {
            MUBUF_STORE_DWORDX2
        } else {
            MUBUF_STORE_DWORDX4
        };
        let info = memops::mubuf(opcode).expect("buffer stores are in the table");
        let Xlat::Op(host) = info.translate(self.target) else {
            return Err(unsupported(inst, self.target, "buffer stores are missing"));
        };
        let (srsrc, soffset) = match (ring.descriptor, ring.offset) {
            (Kept::Sgpr(descriptor), Kept::Sgpr(offset)) => (descriptor, offset),
            // Both values stay in persistent lanes, outside the transient spill area.
            (Kept::Lane(descriptor), Kept::Lane(offset)) => {
                let block = self.spill_block(inst, 5)?;
                for k in 0..4 {
                    self.read_lane(inst, block + k, descriptor + k)?;
                }
                self.read_lane(inst, block + 4, offset)?;
                (block, block + 4)
            }
            _ => {
                return Err(unsupported(
                    inst,
                    self.target,
                    "an attribute ring with mixed storage",
                ))
            }
        };
        let store = Buffer {
            vdata: data,
            vaddr: ring.vertex,
            srsrc,
            soffset: Operand::Sgpr(soffset),
            offset: 16 * u16::from(e.target - EXP_PARAM_FIRST),
            offen: false,
            idxen: true,
            glc: false,
            dlc: false,
            slc: false,
            tfe: false,
            lds: false,
            format: 0,
            kind: MemKind::Store,
            data_dwords: dwords,
        };
        self.push(derive(
            inst,
            F::Mubuf,
            host,
            info.flags,
            Fields::Mubuf(store),
        ));
        Ok(())
    }

    /// `s_waitcnt_vscnt null, 0`: every store, the ring's included, has completed.
    pub(super) fn wait_for_stores(&mut self, origin: &Instruction) -> Result<(), CompileError> {
        let fields = Fields::Sopk {
            sdst: Operand::Null,
            simm16: 0,
            imm32: None,
        };
        self.lower_gfx1013(origin, F::Sopk, SOPK_WAITCNT_VSCNT, fields)
    }

    /// A guest done position export under deferral: its data into the reserved VGPRs (under
    /// the current EXEC) and EXEC into the lane mask, with SCC preserved.
    pub(super) fn defer_position(
        &mut self,
        inst: &Instruction,
        e: &Export,
        deferred: Deferred,
    ) -> Result<(), CompileError> {
        for c in (0..4u8).filter(|c| e.en & (1 << c) != 0) {
            let fields = Fields::Vop1 {
                vdst: Operand::Vgpr(deferred.data + c),
                src0: Operand::Vgpr(e.vsrc[usize::from(c)]),
                ext: crate::ir::VopExt::None,
            };
            self.lower_gfx1013(inst, F::Vop1, opcodes::VOP1_MOV_B32, fields)?;
        }
        // SCC's copy, and the mask's pair (even, for wave64) where it is kept in lanes.
        let t = self.transient_sgprs(inst, 4)?;
        let saved_scc = t;
        self.scalar_select(inst, Operand::Sgpr(saved_scc))?;
        let or = if self.wave64 {
            SOP2_OR_B64
        } else {
            SOP2_OR_B32
        };
        let mask_sgpr = match deferred.exec {
            Kept::Sgpr(exec) => exec,
            Kept::Lane(lane) => {
                self.read_mask(inst, t + 2, lane)?;
                t + 2
            }
        };
        let mask = Operand::Sgpr(mask_sgpr);
        let fields = Fields::Sop2 {
            sdst: mask,
            src0: mask,
            src1: Operand::ExecLo,
        };
        self.lower_gfx1013(inst, F::Sop2, or, fields)?;
        if let Kept::Lane(lane) = deferred.exec {
            for k in 0..if self.wave64 { 2 } else { 1 } {
                self.write_lane(inst, Operand::Sgpr(mask_sgpr + k), lane + k)?;
            }
        }
        let restore = Fields::Sopc {
            src0: Operand::Sgpr(saved_scc),
            src1: Operand::InlineInt(0),
        };
        self.lower_gfx1013(inst, F::Sopc, SOPC_CMP_LG_U32, restore)
    }

    /// The epilogue: wait for the ring stores, then issue the deferred export for the lanes
    /// that reached it (none: skip it). The block at [`EPILOGUE_END_LABEL`] ends the wave.
    pub(super) fn ngg_epilogue(
        &mut self,
        origin: &Instruction,
        deferred: Deferred,
    ) -> Result<(), CompileError> {
        self.wait_for_stores(origin)?;
        let mask_sgpr = match deferred.exec {
            Kept::Sgpr(exec) => exec,
            // The wave ends here: transient SGPRs need no restoring.
            Kept::Lane(lane) => {
                let t = self.transient_sgprs(origin, 2)?;
                self.read_mask(origin, t, lane)?;
                t
            }
        };
        let mask = Operand::Sgpr(mask_sgpr);
        let compare = if self.wave64 {
            SOPC_CMP_EQ_U64
        } else {
            SOPC_CMP_EQ_U32
        };
        let fields = Fields::Sopc {
            src0: mask,
            src1: Operand::InlineInt(0),
        };
        self.lower_gfx1013(origin, F::Sopc, compare, fields)?;
        let mut skip = derive(
            origin,
            F::Sopp,
            SOPP_CBRANCH_SCC1,
            flag::BRANCH,
            Fields::Sopp { simm16: 0 },
        );
        skip.branch_target = Some(EPILOGUE_END_LABEL);
        self.lower(&skip)?;
        let fields = Fields::Sop1 {
            sdst: Operand::ExecLo,
            src0: mask,
        };
        self.lower_gfx1013(origin, F::Sop1, self.lane_mask_mov(), fields)?;
        let data = deferred.data;
        let export = Export {
            target: deferred.target,
            en: deferred.en,
            compr: false,
            done: true,
            vm: false,
            row_en: false,
            vsrc: [data, data + 1, data + 2, data + 3],
        };
        self.push(derive(origin, F::Exp, 0, 0, Fields::Exp(export)));
        Ok(())
    }
}
