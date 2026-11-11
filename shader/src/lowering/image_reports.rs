//! Reports of the images a program finds at run time ([`crate::image_reports`]).
//!
//! The report rides on the conversion of the T# ([`super::descriptors`]): where the conversion
//! loads its table entries, the bucket the runtime would have the image in is loaded too, and
//! one wait serves all of them; after the conversion the bucket's two hashes are compared with
//! the T#'s. Only an unknown image goes on, to one lane storing the T# into its report slot.
//! A block that uses a T# again does neither. Nothing here waits for vector memory, so the
//! program's own samples still overlap.
//!
//! Six SGPRs next to the conversion's own are all it takes, so that a program with none to
//! spare, whose conversion borrows a block of the guest's ([`super::spill`]), reports too:
//! most of a title's programs are of that kind.

use super::{Ctx, LOCAL_LABEL_BASE};
use crate::image_reports::{CHECK, KNOWN_BUCKETS, REPORTS_OFFSET, REPORT_CHECK_OFFSET, REPORT_SLOTS, SPREAD};
use crate::ir::{memops, Fields, Flat, FlatSegment, Instruction, InstructionFamily as F, Operand, Smem, VopExt};
use crate::CompileError;

const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP1_FF1_I32_B32: u16 = 0x13;
const SOP1_FF1_I32_B64: u16 = 0x14;
const SOP2_ADD_U32: u16 = 0x00;
const SOP2_MIN_U32: u16 = 0x07;
const SOP2_AND_B32: u16 = 0x0e;
const SOP2_XOR_B32: u16 = 0x12;
const SOP2_LSHL_B32: u16 = 0x1e;
const SOP2_LSHL_B64: u16 = 0x1f;
const SOP2_LSHR_B32: u16 = 0x20;
const SOP2_MUL_I32: u16 = 0x26;
const SOPC_CMP_LT_I32: u16 = 0x04;
const SOPC_CMP_EQ_U32: u16 = 0x06;
const SOPP_NOP: u16 = 0x00;
const SOPP_CBRANCH_SCC1: u16 = 0x05;
const SMEM_LOAD_DWORDX2: u16 = 0x01;
const VOP1_MOV_B32: u16 = 0x01;
const GLOBAL_STORE_DWORD: u16 = 0x1c;

// A report's SGPRs, from an even base. KNOWN is a bucket's two hashes, and holds EXEC while one
// lane stores; ADDRESS is the table's.
const KNOWN: u8 = 0;
const ADDRESS: u8 = 2;
const HASH: u8 = 4;
const OFFSET: u8 = 5;
/// SGPRs of a report.
pub(super) const SGPRS: u8 = 6;

// The scratch VGPRs.
const V_OFFSET: u8 = 0;
const V_DATA: u8 = 1;
const VGPRS: u8 = 2;

/// The label a report's ways out meet at. Other expansions of an instruction label
/// `LOCAL_LABEL_BASE + pc` (pcs are multiples of four).
const LABEL: u32 = LOCAL_LABEL_BASE + 2;

/// A report under way: the registers it works in, between [`Report::begin`] and
/// [`Report::finish`].
#[derive(Clone, Copy)]
pub(super) struct Report {
    table: u64,
    /// The T#'s first SGPR and its dwords.
    source: u8,
    dwords: u8,
    /// The first of the report's SGPRs and VGPRs.
    s: u8,
    v: u8,
}

impl Report {
    /// A report of the T# at `source` through the table at `table`, in the [`SGPRS`] SGPRs from
    /// `sgprs` (even). `None` for a program with no VGPRs left for the one lane's stores.
    pub fn new(ctx: &mut Ctx<'_>, table: u64, source: u8, r128: bool, sgprs: u8) -> Option<Self> {
        let v = ctx.regs.scratch_vgprs(VGPRS).ok()?;
        Some(Self { table, source, dwords: if r128 { 4 } else { 8 }, s: sgprs, v })
    }

    fn sgpr(&self, offset: u8) -> Operand {
        Operand::Sgpr(self.s + offset)
    }

    /// Dword `k` of the guest's T#.
    fn t(&self, k: u8) -> Operand {
        if k < self.dwords { Operand::Sgpr(self.source + k) } else { Operand::InlineInt(0) }
    }

    fn sop2(&self, ctx: &mut Ctx<'_>, origin: &Instruction, opcode: u16, sdst: Operand, src0: Operand, src1: Operand) -> Result<(), CompileError> {
        ctx.lower_gfx1013(origin, F::Sop2, opcode, Fields::Sop2 { sdst, src0, src1 })
    }

    /// `sdst` = dwords 1 to 5 of the T# in one ([`crate::image_reports::fold`]).
    fn fold(&self, ctx: &mut Ctx<'_>, origin: &Instruction, sdst: Operand) -> Result<(), CompileError> {
        self.sop2(ctx, origin, SOP2_XOR_B32, sdst, self.t(1), self.t(2))?;
        for k in 3..6.min(self.dwords) {
            self.sop2(ctx, origin, SOP2_XOR_B32, sdst, sdst, self.t(k))?;
        }
        Ok(())
    }

    /// Stores `value` at `offset` in the report slot scratch `V_OFFSET` addresses.
    fn store(&self, ctx: &mut Ctx<'_>, origin: &Instruction, value: Operand, offset: u32) -> Result<(), CompileError> {
        let data = self.v + V_DATA;
        let mov = Fields::Vop1 { vdst: Operand::Vgpr(data), src0: value, ext: VopExt::None };
        ctx.lower_gfx1013(origin, F::Vop1, VOP1_MOV_B32, mov)?;
        let info = memops::flat(GLOBAL_STORE_DWORD).expect("the global store of the memops table");
        let fields = Fields::Flat(Flat {
            segment: FlatSegment::Global,
            vdst: 0,
            vdata: data,
            vaddr: self.v + V_OFFSET,
            saddr: Some(self.sgpr(ADDRESS)),
            uses_vaddr: true,
            offset: offset as i32,
            glc: false,
            dlc: false,
            slc: false,
            lds: false,
            kind: info.kind,
            data_dwords: info.dwords,
        });
        ctx.lower_gfx1013(origin, F::Flat, GLOBAL_STORE_DWORD, fields)
    }

    /// Before the conversion waits for its loads: the T#'s hash, and the load of the bucket the
    /// runtime has the image in if it knows it. SCC is the caller's to keep.
    pub fn begin(&self, ctx: &mut Ctx<'_>, origin: &Instruction) -> Result<(), CompileError> {
        let (hash, offset) = (self.sgpr(HASH), self.sgpr(OFFSET));
        let literal = Operand::Literal;
        self.fold(ctx, origin, hash)?;
        self.sop2(ctx, origin, SOP2_MUL_I32, offset, self.t(0), literal(SPREAD))?;
        self.sop2(ctx, origin, SOP2_XOR_B32, hash, hash, offset)?;
        self.sop2(ctx, origin, SOP2_MUL_I32, hash, hash, literal(SPREAD))?;
        self.sop2(ctx, origin, SOP2_LSHR_B32, offset, hash, Operand::InlineInt((32 - KNOWN_BUCKETS.trailing_zeros()) as i8))?;
        self.sop2(ctx, origin, SOP2_LSHL_B32, offset, offset, Operand::InlineInt(3))?;
        for (half, value) in [(0, self.table as u32), (1, (self.table >> 32) as u32)] {
            ctx.lower_gfx1013(origin, F::Sop1, SOP1_MOV_B32, Fields::Sop1 { sdst: self.sgpr(ADDRESS + half), src0: literal(value) })?;
        }
        let load = Fields::Smem(Smem { sdata: self.sgpr(KNOWN), sbase: self.s + ADDRESS, offset: 0, soffset: offset, glc: false, dlc: false, dwords: 2, buffer: false });
        ctx.lower_gfx1013(origin, F::Smem, SMEM_LOAD_DWORDX2, load)
    }

    /// After the conversion's wait: nothing more for an image the bucket holds; one lane stores
    /// any other into its report slot.
    pub fn finish(&self, ctx: &mut Ctx<'_>, origin: &Instruction) -> Result<(), CompileError> {
        let (mov, ff1, lshl) = if ctx.wave64 {
            (SOP1_MOV_B64, SOP1_FF1_I32_B64, SOP2_LSHL_B64)
        } else {
            (SOP1_MOV_B32, SOP1_FF1_I32_B32, SOP2_LSHL_B32)
        };
        let done = LABEL + origin.pc;
        let literal = Operand::Literal;
        let (first, second, hash, offset) = (self.sgpr(KNOWN), self.sgpr(KNOWN + 1), self.sgpr(HASH), self.sgpr(OFFSET));
        let branch_to_done = |ctx: &mut Ctx<'_>, opcode: u16| -> Result<(), CompileError> {
            ctx.lower_gfx1013(origin, F::Sopp, opcode, Fields::Sopp { simm16: 0 })?;
            ctx.out.last_mut().expect("just pushed").branch_target = Some(done);
            Ok(())
        };
        // What each of the bucket's hashes differs from the T#'s in: nothing, for a known image.
        self.sop2(ctx, origin, SOP2_XOR_B32, first, first, hash)?;
        self.sop2(ctx, origin, SOP2_XOR_B32, second, second, hash)?;
        self.sop2(ctx, origin, SOP2_MIN_U32, first, first, second)?;
        ctx.lower_gfx1013(origin, F::Sopc, SOPC_CMP_EQ_U32, Fields::Sopc { src0: first, src1: Operand::InlineInt(0) })?;
        branch_to_done(ctx, SOPP_CBRANCH_SCC1)?;
        // One lane stores the T# and its check word into the image's report slot. EXEC waits in
        // the bucket's registers meanwhile.
        ctx.lower_gfx1013(origin, F::Sop1, ff1, Fields::Sop1 { sdst: offset, src0: Operand::ExecLo })?;
        ctx.lower_gfx1013(origin, F::Sopc, SOPC_CMP_LT_I32, Fields::Sopc { src0: offset, src1: Operand::InlineInt(0) })?;
        branch_to_done(ctx, SOPP_CBRANCH_SCC1)?;
        ctx.lower_gfx1013(origin, F::Sop1, mov, Fields::Sop1 { sdst: first, src0: Operand::ExecLo })?;
        self.sop2(ctx, origin, lshl, Operand::ExecLo, Operand::InlineInt(1), offset)?;
        self.sop2(ctx, origin, SOP2_MUL_I32, offset, self.t(0), literal(SPREAD))?;
        self.sop2(ctx, origin, SOP2_LSHR_B32, offset, offset, Operand::InlineInt(14))?;
        self.sop2(ctx, origin, SOP2_AND_B32, offset, offset, literal((REPORT_SLOTS - 1) << 6))?;
        self.sop2(ctx, origin, SOP2_ADD_U32, offset, offset, literal(REPORTS_OFFSET))?;
        let slot = Fields::Vop1 { vdst: Operand::Vgpr(self.v + V_OFFSET), src0: offset, ext: VopExt::None };
        ctx.lower_gfx1013(origin, F::Vop1, VOP1_MOV_B32, slot)?;
        for k in 0..8 {
            self.store(ctx, origin, self.t(k), u32::from(k) * 4)?;
        }
        // Last, so that a slot whose word checks holds all of the T#.
        self.fold(ctx, origin, hash)?;
        self.sop2(ctx, origin, SOP2_XOR_B32, hash, hash, literal(CHECK))?;
        self.store(ctx, origin, hash, REPORT_CHECK_OFFSET)?;
        ctx.lower_gfx1013(origin, F::Sop1, mov, Fields::Sop1 { sdst: Operand::ExecLo, src0: first })?;
        // The ways out meet on a NOP: what follows belongs to the conversion.
        ctx.lower_gfx1013(origin, F::Sopp, SOPP_NOP, Fields::Sopp { simm16: 0 })?;
        ctx.out.last_mut().expect("just pushed").pc = done;
        Ok(())
    }
}
