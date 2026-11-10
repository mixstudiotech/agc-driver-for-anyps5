//! Texture LOD statistics, measured by the translated pixel shaders.
//!
//! The PS5 GPU keeps, for each of 256 counters a T# can name, the finest mip its samples asked
//! for and how many samples the T#'s level clamp held back; `sceAgcDcbGetLodStats` reports them
//! and texture streaming decides from that what to load. gfx11 and gfx12 have no such counters.
//!
//! A sample of a T# with counting enabled therefore measures itself first, in one quad per wave:
//! the LOD follows from the coordinates' differences across the quad, the T#'s extent and the
//! S#'s anisotropy and bias, with VALU work only, so no sample has to wait for a texture
//! operation it did not need. The result goes into [`crate::mip_stats`]'s table with two global
//! atomics: slot dword 1 keeps the minimum of `mip << 24 | 1` (finest mip, and "sampled"), slot
//! dword 0 adds the wave's pixels when the level was clamped. A T# without counting costs a
//! test and a branch.

use super::{Ctx, LOCAL_LABEL_BASE};
use crate::ir::{
    memops, Fields, Flat, FlatSegment, Image, Instruction, InstructionFamily as F, Operand, Vop3,
    VopExt,
};
use crate::CompileError;

const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP1_BCNT1_I32_B32: u16 = 0x0f;
const SOP1_BCNT1_I32_B64: u16 = 0x10;
const SOP1_FF1_I32_B32: u16 = 0x13;
const SOP1_FF1_I32_B64: u16 = 0x14;
const SOP2_ADD_U32: u16 = 0x00;
const SOP2_MAX_U32: u16 = 0x09;
const SOP2_CSELECT_B32: u16 = 0x0a;
const SOP2_OR_B32: u16 = 0x10;
const SOP2_ANDN2_B32: u16 = 0x14;
const SOP2_LSHL_B32: u16 = 0x1e;
const SOP2_LSHL_B64: u16 = 0x1f;
const SOP2_BFE_U32: u16 = 0x27;
const SOP2_BFE_I32: u16 = 0x28;
const SOPC_CMP_LT_I32: u16 = 0x04;
const SOPC_CMP_LG_U32: u16 = 0x07;
const SOPC_CMP_LT_U32: u16 = 0x0a;
const SOPC_BITCMP1_B32: u16 = 0x0d;
const SOPP_CBRANCH_SCC0: u16 = 0x04;
const SOPP_CBRANCH_SCC1: u16 = 0x05;
const VOP1_MOV_B32: u16 = 0x01;
const VOP1_READFIRSTLANE_B32: u16 = 0x02;
const VOP1_CVT_F32_I32: u16 = 0x05;
const VOP1_CVT_F32_U32: u16 = 0x06;
const VOP1_CVT_FLR_I32_F32: u16 = 0x0d;
const VOP1_LOG_F32: u16 = 0x27;
const VOP2_ADD_F32: u16 = 0x03;
const VOP2_SUB_F32: u16 = 0x04;
const VOP2_MUL_F32: u16 = 0x08;
const VOP2_MIN_F32: u16 = 0x0f;
const VOP2_MAX_F32: u16 = 0x10;
const VOP2_MAX_I32: u16 = 0x12;
const VOP2_LSHLREV_B32: u16 = 0x1a;
const VOP2_AND_B32: u16 = 0x1b;
const VOP2_OR_B32: u16 = 0x1c;
const VOP3_READLANE_B32: u16 = 0x360;
const GLOBAL_ATOMIC_ADD: u16 = 0x32;
const GLOBAL_ATOMIC_UMIN: u16 = 0x36;

/// T# dword 5: this texture's samples are counted.
const COUNTER_ENABLE_BIT: i8 = 25;
/// S# LOD_BIAS is signed 5.8 fixed point: the value of one step, 1/256, as an IEEE single.
const LOD_BIAS_STEP: u32 = 0x3b80_0000;
const ABS_MASK: u32 = 0x7fff_ffff;

// The scratch SGPRs, from an even base. EXEC and ADDRESS are pairs.
const EXEC: u8 = 0;
const SCC: u8 = 2;
const LANE: u8 = 3;
const LANE1: u8 = 4;
const LANE2: u8 = 5;
/// The top-left, top-right and bottom-left values of up to two coordinates.
const COORDS: u8 = 6;
const WIDTH: u8 = 12;
const HEIGHT: u8 = 13;
const TMP: u8 = 14;
const TRACKED: u8 = 15;
const PIXELS: u8 = 16;
const CLAMPED: u8 = 17;
const MIP: u8 = 18;
const OFFSET: u8 = 19;
const ADDRESS: u8 = 20;
const SAMPLER_BIAS: u8 = 22;
const MAX_ANISO: u8 = 23;
const INSTRUCTION_BIAS: u8 = 24;
const SGPRS: u8 = 25;

// The scratch VGPRs.
const V_WIDTH: u8 = 0;
const V_HEIGHT: u8 = 1;
const V_D0: u8 = 2;
const V_D1: u8 = 3;
const V_D2: u8 = 4;
const V_D3: u8 = 5;
const V_OFFSET: u8 = 6;
const V_DATA: u8 = 7;
const VGPRS: u8 = 8;

fn field(offset: u32, width: u32) -> Operand {
    Operand::Literal(offset | (width << 16))
}

/// Emits the measurement of one sample.
struct Measure<'c, 'r> {
    ctx: &'c mut Ctx<'r>,
    origin: Instruction,
    /// The first scratch SGPR and VGPR.
    s: u8,
    v: u8,
}

impl Measure<'_, '_> {
    fn op(&mut self, family: F, opcode: u16, fields: Fields) -> Result<(), CompileError> {
        self.ctx.lower_gfx1013(&self.origin, family, opcode, fields)
    }

    fn sgpr(&self, offset: u8) -> Operand {
        Operand::Sgpr(self.s + offset)
    }

    fn vgpr(&self, offset: u8) -> u8 {
        self.v + offset
    }

    fn sop1(&mut self, opcode: u16, sdst: Operand, src0: Operand) -> Result<(), CompileError> {
        self.op(F::Sop1, opcode, Fields::Sop1 { sdst, src0 })
    }

    fn sop2(
        &mut self,
        opcode: u16,
        sdst: Operand,
        src0: Operand,
        src1: Operand,
    ) -> Result<(), CompileError> {
        self.op(F::Sop2, opcode, Fields::Sop2 { sdst, src0, src1 })
    }

    fn sopc(&mut self, opcode: u16, src0: Operand, src1: Operand) -> Result<(), CompileError> {
        self.op(F::Sopc, opcode, Fields::Sopc { src0, src1 })
    }

    fn vop1(&mut self, opcode: u16, vdst: u8, src0: Operand) -> Result<(), CompileError> {
        let fields = Fields::Vop1 {
            vdst: Operand::Vgpr(self.vgpr(vdst)),
            src0,
            ext: VopExt::None,
        };
        self.op(F::Vop1, opcode, fields)
    }

    /// `vdst = src0 <op> vsrc1`, on scratch VGPRs.
    fn vop2(&mut self, opcode: u16, vdst: u8, src0: Operand, vsrc1: u8) -> Result<(), CompileError> {
        let fields = Fields::Vop2 {
            vdst: Operand::Vgpr(self.vgpr(vdst)),
            src0,
            vsrc1: Operand::Vgpr(self.vgpr(vsrc1)),
            constant: None,
            ext: VopExt::None,
        };
        self.op(F::Vop2, opcode, fields)
    }

    /// `vop2` with a scratch VGPR as the first source too.
    fn vop2v(&mut self, opcode: u16, vdst: u8, src0: u8, vsrc1: u8) -> Result<(), CompileError> {
        let src0 = Operand::Vgpr(self.vgpr(src0));
        self.vop2(opcode, vdst, src0, vsrc1)
    }

    fn read_lane(&mut self, sdst: u8, vgpr: u8, lane: u8) -> Result<(), CompileError> {
        let read = Vop3::new(
            self.sgpr(sdst),
            [Operand::Vgpr(vgpr), self.sgpr(lane), Operand::Raw(0)],
        );
        self.op(F::Vop3, VOP3_READLANE_B32, Fields::Vop3(read))
    }

    /// A branch to `label`, inside the block being lowered.
    fn branch_to(&mut self, opcode: u16, label: u32) -> Result<(), CompileError> {
        self.op(F::Sopp, opcode, Fields::Sopp { simm16: 0 })?;
        self.ctx.out.last_mut().expect("just pushed").branch_target = Some(label);
        Ok(())
    }

    /// `vdst = |(after - before) * extent|`: a coordinate's change across the quad, in texels.
    fn footprint(&mut self, vdst: u8, before: u8, after: u8, extent: u8) -> Result<(), CompileError> {
        let (before, after) = (self.sgpr(before), self.sgpr(after));
        self.vop1(VOP1_MOV_B32, vdst, before)?;
        self.vop2(VOP2_SUB_F32, vdst, after, vdst)?;
        self.vop2v(VOP2_MUL_F32, vdst, extent, vdst)?;
        self.vop2(VOP2_AND_B32, vdst, Operand::Literal(ABS_MASK), vdst)
    }

    /// The global atomic `opcode` on the slot at scratch `V_OFFSET` plus `offset`.
    fn atomic(&mut self, opcode: u16, offset: i32, data: u8) -> Result<(), CompileError> {
        let info = memops::flat(opcode).expect("a global atomic of the memops table");
        let fields = Fields::Flat(Flat {
            segment: FlatSegment::Global,
            vdst: 0,
            vdata: self.vgpr(data),
            vaddr: self.vgpr(V_OFFSET),
            saddr: Some(self.sgpr(ADDRESS)),
            uses_vaddr: true,
            offset,
            glc: false,
            dlc: false,
            slc: false,
            lds: false,
            kind: info.kind,
            data_dwords: info.dwords,
        });
        self.op(F::Flat, opcode, fields)
    }

    /// The LOD of the quad into `V_D1`, from `coords` coordinates at the SGPRs [`COORDS`].
    fn level_of_detail(
        &mut self,
        image: &Image,
        coords: u8,
        instruction_bias: bool,
    ) -> Result<(), CompileError> {
        let t = |k: u8| Operand::Sgpr(image.srsrc + k);
        let sampler = |k: u8| Operand::Sgpr(image.ssamp + k);
        let (width, height, scratch) = (self.sgpr(WIDTH), self.sgpr(HEIGHT), self.sgpr(TMP));
        self.sop2(SOP2_BFE_U32, width, t(1), field(30, 2))?;
        self.sop2(SOP2_BFE_U32, scratch, t(2), field(0, 12))?;
        self.sop2(SOP2_LSHL_B32, scratch, scratch, Operand::InlineInt(2))?;
        self.sop2(SOP2_OR_B32, width, width, scratch)?;
        self.sop2(SOP2_ADD_U32, width, width, Operand::InlineInt(1))?;
        self.sop2(SOP2_BFE_U32, height, t(2), field(14, 14))?;
        self.sop2(SOP2_ADD_U32, height, height, Operand::InlineInt(1))?;
        self.vop1(VOP1_CVT_F32_U32, V_WIDTH, width)?;
        self.vop1(VOP1_CVT_F32_U32, V_HEIGHT, height)?;
        self.footprint(V_D0, COORDS, COORDS + 1, V_WIDTH)?;
        self.footprint(V_D2, COORDS, COORDS + 2, V_WIDTH)?;
        if coords == 2 {
            self.footprint(V_D1, COORDS + 3, COORDS + 4, V_HEIGHT)?;
            self.footprint(V_D3, COORDS + 3, COORDS + 5, V_HEIGHT)?;
        } else {
            self.vop1(VOP1_MOV_B32, V_D1, Operand::InlineInt(0))?;
            self.vop1(VOP1_MOV_B32, V_D3, Operand::InlineInt(0))?;
        }
        // The footprint along screen x and y.
        self.vop2v(VOP2_MAX_F32, V_D0, V_D1, V_D0)?;
        self.vop2v(VOP2_MAX_F32, V_D2, V_D3, V_D2)?;
        // Major and minor axis. With anisotropy the hardware walks the major axis in up to
        // 2^MAX_ANISO_RATIO steps: LOD = max(log2(minor), log2(major) - MAX_ANISO_RATIO).
        let (d0, d2) = (Operand::Vgpr(self.vgpr(V_D0)), Operand::Vgpr(self.vgpr(V_D2)));
        self.vop1(VOP1_MOV_B32, V_D1, d0)?;
        self.vop2v(VOP2_MAX_F32, V_D1, V_D2, V_D1)?;
        self.vop1(VOP1_MOV_B32, V_D3, d2)?;
        self.vop2v(VOP2_MIN_F32, V_D3, V_D0, V_D3)?;
        let (d1, d3) = (Operand::Vgpr(self.vgpr(V_D1)), Operand::Vgpr(self.vgpr(V_D3)));
        self.vop1(VOP1_LOG_F32, V_D1, d1)?;
        self.vop1(VOP1_LOG_F32, V_D3, d3)?;
        let (aniso, bias) = (self.sgpr(MAX_ANISO), self.sgpr(SAMPLER_BIAS));
        self.sop2(SOP2_BFE_U32, aniso, sampler(0), field(9, 3))?;
        self.vop1(VOP1_CVT_F32_U32, V_HEIGHT, aniso)?;
        self.vop2v(VOP2_SUB_F32, V_D1, V_D1, V_HEIGHT)?;
        self.vop2v(VOP2_MAX_F32, V_D1, V_D3, V_D1)?;
        self.sop2(SOP2_BFE_I32, bias, sampler(2), field(0, 14))?;
        self.vop1(VOP1_CVT_F32_I32, V_WIDTH, bias)?;
        self.vop2(VOP2_MUL_F32, V_WIDTH, Operand::Literal(LOD_BIAS_STEP), V_WIDTH)?;
        self.vop2v(VOP2_ADD_F32, V_D1, V_WIDTH, V_D1)?;
        if instruction_bias {
            let bias = self.sgpr(INSTRUCTION_BIAS);
            self.vop2(VOP2_ADD_F32, V_D1, bias, V_D1)?;
        }
        Ok(())
    }
}

/// Before the sample `inst` of `image`: counts it in the table at `table` when its T# asks for
/// that. A sample whose LOD cannot be derived here counts as asking for mip 0, so that texture
/// streaming does not hold back what it may need.
pub(super) fn count(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    image: &Image,
    flags: u32,
    table: u64,
) -> Result<(), CompileError> {
    use memops::img;
    if image.r128 || image.a16 {
        return Ok(());
    }
    let Some(base) = ctx.regs.scratch_sgprs(SGPRS + 1) else {
        return Ok(());
    };
    let s = (base + 1) & !1;
    let Ok(v) = ctx.regs.scratch_vgprs(VGPRS) else {
        return Ok(());
    };
    let wave64 = ctx.wave64;
    let (mov, ff1, bcnt, lshl) = if wave64 {
        (SOP1_MOV_B64, SOP1_FF1_I32_B64, SOP1_BCNT1_I32_B64, SOP2_LSHL_B64)
    } else {
        (SOP1_MOV_B32, SOP1_FF1_I32_B32, SOP1_BCNT1_I32_B32, SOP2_LSHL_B32)
    };
    let coords = match image.dim {
        _ if image.unorm || flags & (img::LOD | img::LZ | img::DERIV) != 0 => None,
        0 | 4 => Some(1),
        1 | 5 => Some(2),
        _ => None,
    };
    let offset_operands = usize::from(flags & img::OFFSET != 0);
    let bias_operands = usize::from(flags & img::BIAS != 0);
    let compare_operands = usize::from(flags & img::COMPARE != 0);
    let first_coord = offset_operands + bias_operands + compare_operands;
    let done = LOCAL_LABEL_BASE + inst.pc;
    let t = |k: u8| Operand::Sgpr(image.srsrc + k);

    let mut m = Measure { ctx, origin: inst.clone(), s, v };
    let origin = inst;
    m.ctx.scalar_select(origin, m.sgpr(SCC))?;
    m.sopc(SOPC_BITCMP1_B32, t(5), Operand::InlineInt(COUNTER_ENABLE_BIT))?;
    m.branch_to(SOPP_CBRANCH_SCC0, done)?;
    let saved = m.sgpr(EXEC);
    m.sop1(mov, saved, Operand::ExecLo)?;
    let lane = m.sgpr(LANE);
    m.sop1(ff1, lane, Operand::ExecLo)?;
    m.sopc(SOPC_CMP_LT_I32, lane, Operand::InlineInt(0))?;
    m.branch_to(SOPP_CBRANCH_SCC1, done)?;
    m.sop2(SOP2_ANDN2_B32, lane, lane, Operand::InlineInt(3))?;
    if let Some(coords) = coords {
        let (lane1, lane2) = (m.sgpr(LANE1), m.sgpr(LANE2));
        m.sop2(SOP2_ADD_U32, lane1, lane, Operand::InlineInt(1))?;
        m.sop2(SOP2_ADD_U32, lane2, lane, Operand::InlineInt(2))?;
        for coord in 0..coords {
            let register = image.address_register(first_coord + usize::from(coord));
            for (position, lane) in [LANE, LANE1, LANE2].into_iter().enumerate() {
                m.read_lane(COORDS + coord * 3 + position as u8, register, lane)?;
            }
        }
        if bias_operands != 0 {
            m.read_lane(INSTRUCTION_BIAS, image.address_register(offset_operands), LANE)?;
        }
    }
    // One lane from here: the arithmetic and the atomics run once per wave.
    m.sop2(lshl, Operand::ExecLo, Operand::InlineInt(1), lane)?;
    if let Some(coords) = coords {
        m.level_of_detail(image, coords, bias_operands != 0)?;
        let lod = Operand::Vgpr(m.vgpr(V_D1));
        m.vop1(VOP1_CVT_FLR_I32_F32, V_D0, lod)?;
        m.vop2(VOP2_MAX_I32, V_D0, Operand::InlineInt(0), V_D0)?;
    } else {
        m.vop1(VOP1_MOV_B32, V_D0, Operand::InlineInt(0))?;
    }
    let read_mip = Fields::Vop1 {
        vdst: m.sgpr(MIP),
        src0: Operand::Vgpr(m.vgpr(V_D0)),
        ext: VopExt::None,
    };
    m.op(F::Vop1, VOP1_READFIRSTLANE_B32, read_mip)?;
    // The tracked level is max(MIN_LOD, BASE_LEVEL), MIN_LOD in 4.8 fixed point: a mip finer than
    // it was clamped.
    let (tracked, scratch, pixels, clamped) =
        (m.sgpr(TRACKED), m.sgpr(TMP), m.sgpr(PIXELS), m.sgpr(CLAMPED));
    m.sop2(SOP2_BFE_U32, tracked, t(1), field(16, 4))?;
    m.sop2(SOP2_BFE_U32, scratch, t(3), field(12, 4))?;
    m.sop2(SOP2_MAX_U32, tracked, tracked, scratch)?;
    m.sop1(bcnt, pixels, saved)?;
    let mip = m.sgpr(MIP);
    m.sopc(SOPC_CMP_LT_U32, mip, tracked)?;
    m.sop2(SOP2_CSELECT_B32, clamped, pixels, Operand::InlineInt(0))?;
    let offset = m.sgpr(OFFSET);
    m.sop2(SOP2_BFE_U32, offset, t(6), field(0, 8))?;
    m.sop2(SOP2_LSHL_B32, offset, offset, Operand::InlineInt(3))?;
    let (low, high) = (m.sgpr(ADDRESS), m.sgpr(ADDRESS + 1));
    m.sop1(SOP1_MOV_B32, low, Operand::Literal(table as u32))?;
    m.sop1(SOP1_MOV_B32, high, Operand::Literal((table >> 32) as u32))?;
    m.vop1(VOP1_MOV_B32, V_OFFSET, offset)?;
    m.vop1(VOP1_MOV_B32, V_DATA, clamped)?;
    m.atomic(GLOBAL_ATOMIC_ADD, 0, V_DATA)?;
    m.vop2(VOP2_LSHLREV_B32, V_D0, Operand::InlineInt(24), V_D0)?;
    m.vop2(VOP2_OR_B32, V_D0, Operand::InlineInt(1), V_D0)?;
    m.atomic(GLOBAL_ATOMIC_UMIN, 4, V_D0)?;
    m.sop1(mov, Operand::ExecLo, saved)?;
    // Both ways out come here: the guest's SCC is back.
    let scc = m.sgpr(SCC);
    m.sopc(SOPC_CMP_LG_U32, scc, Operand::InlineInt(0))?;
    m.ctx.out.last_mut().expect("just pushed").pc = done;
    Ok(())
}
