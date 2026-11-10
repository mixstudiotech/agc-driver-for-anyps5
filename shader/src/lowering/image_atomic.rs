//! Float image min/max on gfx11 through image integer compare-and-swap. Image coordinates and
//! descriptors stay with the texture unit, including tiled addressing and bounds handling.

use super::{unsupported, Ctx, LOCAL_LABEL_BASE};
use crate::ir::{opcodes::flag, Fields, Instruction, InstructionFamily as F, Operand, VopExt};
use crate::CompileError;

const MOV_B64: u16 = 0x04;
const AND_B32: u16 = 0x0e;
const AND_B64: u16 = 0x0f;
const CSELECT_B64: u16 = 0x0b;
const ANDN2_B32: u16 = 0x14;
const ANDN2_B64: u16 = 0x15;
const CMP_LT_U32: u16 = 0xc1;
const CMP_EQ_U32: u16 = 0xc2;
const CMP_GT_U32: u16 = 0xc4;
const V_MOV: u16 = 0x01;
const V_CNDMASK: u16 = 0x01;
const V_ASHRREV: u16 = 0x18;
const V_AND: u16 = 0x1b;
const V_OR: u16 = 0x1c;
const V_XOR: u16 = 0x1d;
const V_ADD: u16 = 0x25;

pub(super) fn lower(ctx: &mut Ctx<'_>, origin: &Instruction, mut image: Instruction) -> Result<(), CompileError> {
    let Fields::Mimg(guest) = origin.fields else { return Err(unsupported(origin, ctx.target, "malformed float image atomic")) };
    if guest.dmask != 1 || guest.data_dwords != 1 || guest.d16 || guest.tfe || guest.lwe {
        return Err(unsupported(origin, ctx.target, "float image atomic emulation requires a 32-bit operand without D16, TFE or LWE"));
    }
    let s = ctx.transient_sgprs(origin, 8)?;
    let v = ctx.regs.scratch_vgprs(7)?;
    let mut seq = Sequence { ctx, origin, s, v };
    seq.sop1(MOV_B64, Operand::Sgpr(s), Operand::ExecLo)?;
    seq.sop1(MOV_B64, Operand::Sgpr(s + 2), Operand::VccLo)?;
    seq.ctx.scalar_select(origin, Operand::Sgpr(s + 4))?;
    // Only input-denorm MODE controls the comparison. The selected original bits are stored;
    // VALU min/max would also flush the selected value and has different signaling-NaN rules.
    seq.ctx.hwreg(origin, 0x12, Operand::Sgpr(s + 5), None)?;
    seq.sop2(AND_B32, Operand::Sgpr(s + 5), Operand::InlineInt(1), Operand::Sgpr(s + 5))?;
    seq.sop2(CSELECT_B64, Operand::Sgpr(s + 6), Operand::InlineInt(0), Operand::InlineInt(-1))?;
    seq.mov(0, Operand::Vgpr(guest.vdata))?;
    seq.mov(2, Operand::InlineInt(0))?;
    let loop_at = seq.ctx.out.len();
    let label = LOCAL_LABEL_BASE + origin.pc;
    // Keep the label on a NOP: hazard insertion must not move the loop target past a wait.
    seq.op(F::Sopp, 0x00, Fields::Sopp { simm16: 0 })?;
    seq.ctx.out[loop_at].pc = label;
    seq.key(0, 4)?;
    seq.key(2, 5)?;
    seq.cmp(if origin.opcode == 0x1f { CMP_GT_U32 } else { CMP_LT_U32 }, seq.vgpr(4), 5)?;
    seq.select(1, 2, 0)?;
    // Quiet NaNs lose to numbers. A signaling NaN is quieted and propagated, independent of
    // IEEE_MODE. The input is source 0 and memory is source 1 for the selection priority.
    seq.nan(2, 0)?;
    seq.nan(0, 2)?;
    seq.signaling_nan(2)?;
    seq.signaling_nan(0)?;
    let Fields::Mimg(cas) = &mut image.fields else { unreachable!() };
    cas.vdata = v + 1;
    cas.dmask = 3;
    cas.data_dwords = 2;
    cas.glc = true;
    image.flags |= flag::CMPSWAP;
    seq.ctx.push(image);
    seq.op(F::Sopp, 0x0c, Fields::Sopp { simm16: super::waitcnt::encode_gfx10(0, 7, 63) })?;
    seq.mov(3, seq.vgpr(1))?;
    seq.cmp(CMP_EQ_U32, seq.vgpr(1), 2)?;
    seq.mov(2, seq.vgpr(1))?;
    seq.sop2(if seq.ctx.wave64 { ANDN2_B64 } else { ANDN2_B32 }, Operand::ExecLo, Operand::ExecLo, Operand::VccLo)?;
    let at = seq.ctx.out.len();
    seq.op(F::Sopp, 0x09, Fields::Sopp { simm16: 0 })?;
    seq.ctx.out[at].branch_target = Some(label);
    seq.sop1(MOV_B64, Operand::ExecLo, Operand::Sgpr(s))?;
    if guest.glc {
        seq.op(F::Vop1, V_MOV, Fields::Vop1 { vdst: Operand::Vgpr(guest.vdata), src0: seq.vgpr(3), ext: VopExt::None })?;
    }
    seq.sop1(MOV_B64, Operand::VccLo, Operand::Sgpr(s + 2))?;
    seq.op(F::Sopc, 0x07, Fields::Sopc { src0: Operand::Sgpr(s + 4), src1: Operand::InlineInt(0) })
}

struct Sequence<'c, 'r, 'i> {
    ctx: &'c mut Ctx<'r>,
    origin: &'i Instruction,
    s: u8,
    v: u8,
}

impl Sequence<'_, '_, '_> {
    fn vgpr(&self, n: u8) -> Operand { Operand::Vgpr(self.v + n) }

    fn op(&mut self, family: F, opcode: u16, fields: Fields) -> Result<(), CompileError> {
        self.ctx.lower_gfx1013(self.origin, family, opcode, fields)
    }

    fn sop1(&mut self, opcode: u16, sdst: Operand, src0: Operand) -> Result<(), CompileError> {
        self.op(F::Sop1, opcode, Fields::Sop1 { sdst, src0 })
    }

    fn sop2(&mut self, opcode: u16, sdst: Operand, src0: Operand, src1: Operand) -> Result<(), CompileError> {
        self.op(F::Sop2, opcode, Fields::Sop2 { sdst, src0, src1 })
    }

    fn mov(&mut self, dst: u8, src0: Operand) -> Result<(), CompileError> {
        self.op(F::Vop1, V_MOV, Fields::Vop1 { vdst: self.vgpr(dst), src0, ext: VopExt::None })
    }

    fn valu(&mut self, opcode: u16, dst: u8, src0: Operand, src1: u8) -> Result<(), CompileError> {
        self.op(F::Vop2, opcode, Fields::Vop2 { vdst: self.vgpr(dst), src0, vsrc1: self.vgpr(src1), constant: None, ext: VopExt::None })
    }

    fn cmp(&mut self, opcode: u16, src0: Operand, src1: u8) -> Result<(), CompileError> {
        self.op(F::Vopc, opcode, Fields::Vopc { sdst: Operand::VccLo, src0, vsrc1: self.vgpr(src1), ext: VopExt::None })
    }

    fn select(&mut self, dst: u8, no: u8, yes: u8) -> Result<(), CompileError> {
        self.valu(V_CNDMASK, dst, self.vgpr(no), yes)
    }

    fn key(&mut self, raw: u8, key: u8) -> Result<(), CompileError> {
        self.valu(V_AND, key, Operand::Literal(0x7fff_ffff), raw)?;
        self.valu(V_AND, 6, Operand::Literal(0x8000_0000), raw)?;
        self.cmp(CMP_GT_U32, Operand::Literal(0x0080_0000), key)?;
        self.sop2(if self.ctx.wave64 { AND_B64 } else { AND_B32 }, Operand::VccLo, Operand::VccLo, Operand::Sgpr(self.s + 6))?;
        self.select(key, raw, 6)?;
        self.valu(V_ASHRREV, 6, Operand::InlineInt(31), key)?;
        self.valu(V_OR, 6, Operand::Literal(0x8000_0000), 6)?;
        self.valu(V_XOR, key, self.vgpr(6), key)
    }

    fn nan(&mut self, raw: u8, other: u8) -> Result<(), CompileError> {
        self.valu(V_AND, 6, Operand::Literal(0x7fff_ffff), raw)?;
        self.cmp(CMP_LT_U32, Operand::Literal(0x7f80_0000), 6)?;
        self.select(1, 1, other)
    }

    fn signaling_nan(&mut self, raw: u8) -> Result<(), CompileError> {
        self.valu(V_AND, 6, Operand::Literal(0x7fff_ffff), raw)?;
        self.valu(V_ADD, 6, Operand::Literal(0x807f_ffff), 6)?;
        self.cmp(CMP_GT_U32, Operand::Literal(0x003f_ffff), 6)?;
        self.valu(V_OR, 6, Operand::Literal(0x0040_0000), raw)?;
        self.select(1, 1, 6)
    }
}
