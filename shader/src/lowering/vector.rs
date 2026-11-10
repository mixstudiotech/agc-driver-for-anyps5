use super::{derive, interp, retarget, sdwa, unsupported, Ctx};
use crate::ir::opcodes::{
    self, flag, OpInfo, Rule, VopcInfo, Xlat, VOP2_ADD_F32, VOP2_FMAAK_F32, VOP2_FMAC_F32,
    VOP2_FMAMK_F32, VOP2_MUL_F32, VOP2_MUL_LEGACY_F32, VOP3P_DOT2_F32_F16, VOP3P_DOT4_I32_I8,
    VOP3_FMA_F32, VOP3_VOP1_BASE, VOP3_VOP2_BASE,
};
use crate::ir::{Fields, Instruction, InstructionFamily as F, Operand, Vop3, Vop3p, VopExt};
use crate::{CompileError, PrecisionMode, TargetIsa};

pub const SOP1_GFX11_MOV_B32: u16 = 0x00;
const VOP3_MAD_U64_U32: u16 = 0x176;
const VOP3_MAD_I64_I32: u16 = 0x177;
pub const SOP1_GFX11_MOV_B64: u16 = 0x01;
const PACKED_HI_ALL: u8 = 0b111;
const SIGNED_SOURCES: u8 = 0b011;
const HIGH_VGPR: u8 = 128;

pub(crate) fn vop3_only(target: TargetIsa, gfx10_op: u16) -> u16 {
    match opcodes::lookup(F::Vop3, gfx10_op).map(|i| i.translate(target)) {
        Some(Xlat::Op(op)) => op,
        _ => gfx10_op,
    }
}

pub(crate) fn vop2_op(target: TargetIsa, gfx10_op: u16) -> u16 {
    match opcodes::lookup(F::Vop2, gfx10_op).map(|i| i.translate(target)) {
        Some(Xlat::Op(op)) => op,
        _ => gfx10_op,
    }
}

pub(crate) fn vop3_inst(origin: &Instruction, opcode: u16, flags: u64, v: Vop3) -> Instruction {
    derive(origin, F::Vop3, opcode, flags, Fields::Vop3(v))
}

pub(crate) fn vop2_inst(
    origin: &Instruction,
    opcode: u16,
    vdst: Operand,
    src0: Operand,
    vsrc1: Operand,
) -> Instruction {
    let fields = Fields::Vop2 {
        vdst,
        src0,
        vsrc1,
        constant: None,
        ext: VopExt::None,
    };
    derive(origin, F::Vop2, opcode, 0, fields)
}

fn range_base(family: F) -> u16 {
    match family {
        F::Vop1 => VOP3_VOP1_BASE,
        F::Vop2 => VOP3_VOP2_BASE,
        _ => 0,
    }
}

pub(super) fn lower(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), CompileError> {
    match inst.family {
        F::Vop3p => vop3p(ctx, inst),
        F::Vopc => vopc(ctx, inst, inst.opcode),
        F::Vop3 => match opcodes::vop3_base(inst.opcode) {
            Some((F::Vopc, base)) => vopc(ctx, inst, base),
            Some((family, base)) => alu(ctx, inst, family, base),
            None => vop3_only_op(ctx, inst),
        },
        family => alu(ctx, inst, family, inst.opcode),
    }
}

fn lookup(
    ctx: &Ctx<'_>,
    inst: &Instruction,
    family: F,
    op: u16,
) -> Result<&'static OpInfo, CompileError> {
    opcodes::lookup(family, op)
        .ok_or_else(|| unsupported(inst, ctx.target, "unknown vector opcode"))
}

fn alu(ctx: &mut Ctx<'_>, inst: &Instruction, family: F, op: u16) -> Result<(), CompileError> {
    let info = lookup(ctx, inst, family, op)?;
    match info.translate(ctx.target) {
        Xlat::Op(target_op) => emit(ctx, inst, family, target_op),
        Xlat::Drop | Xlat::Same => Ok(()),
        Xlat::No => Err(unsupported(
            inst,
            ctx.target,
            "instruction was removed on the target",
        )),
        Xlat::Rule(rule) => alu_rule(ctx, inst, rule),
    }
}

fn high_vgpr(ops: &[Operand]) -> bool {
    ops.iter()
        .any(|op| matches!(op, Operand::Vgpr(n) if *n >= HIGH_VGPR))
}

fn emit(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    family: F,
    target_op: u16,
) -> Result<(), CompileError> {
    let vop3_op = range_base(family) + target_op;
    let (vdst, src0, vsrc1, ext) = match inst.fields {
        Fields::Vop3(_) => {
            ctx.push(retarget(inst, vop3_op));
            return Ok(());
        }
        Fields::Vop1 { vdst, src0, ext } => (vdst, src0, Operand::Raw(0), ext),
        Fields::Vop2 {
            vdst,
            src0,
            vsrc1,
            ext,
            ..
        } => (vdst, src0, vsrc1, ext),
        _ => {
            return Err(unsupported(
                inst,
                ctx.target,
                "malformed vector instruction",
            ))
        }
    };
    if ctx.target.is_gfx11_plus() {
        if let VopExt::Sdwa(_) = ext {
            return sdwa::expand(ctx, inst, vop3_op);
        }
        if inst.has_flag(flag::T16) && high_vgpr(&[vdst, src0, vsrc1]) {
            if ext != VopExt::None || inst.has_flag(flag::LITK) {
                return Err(unsupported(
                    inst,
                    ctx.target,
                    "16-bit operands above v127 need a VOP3 form",
                ));
            }
            let src2 = if inst.has_flag(flag::CARRY_IN) {
                Operand::VccLo
            } else {
                Operand::Raw(0)
            };
            let mut v = Vop3::new(vdst, [src0, vsrc1, src2]);
            if inst.has_flag(flag::CARRY_OUT) {
                v.sdst = Some(Operand::VccLo);
            }
            ctx.push(vop3_inst(inst, vop3_op, inst.flags, v));
            return Ok(());
        }
    }
    ctx.push(retarget(inst, target_op));
    Ok(())
}

struct Mac {
    vdst: Operand,
    src0: Operand,
    src1: Operand,
    src2: Operand,
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
}

fn mac_operands(ctx: &Ctx<'_>, inst: &Instruction) -> Result<Mac, CompileError> {
    match inst.fields {
        Fields::Vop2 {
            vdst,
            src0,
            vsrc1,
            constant,
            ext: VopExt::None,
        } => Ok(Mac {
            vdst,
            src0,
            src1: vsrc1,
            src2: constant.map_or(vdst, Operand::Literal),
            abs: 0,
            neg: 0,
            clamp: false,
            omod: 0,
        }),
        Fields::Vop3(v) => Ok(Mac {
            vdst: v.vdst,
            src0: v.src[0],
            src1: v.src[1],
            src2: v.vdst,
            abs: v.abs,
            neg: v.neg,
            clamp: v.clamp,
            omod: v.omod,
        }),
        _ => Err(unsupported(
            inst,
            ctx.target,
            "MAC with DPP or SDWA has no fused equivalent",
        )),
    }
}

fn alu_rule(ctx: &mut Ctx<'_>, inst: &Instruction, rule: Rule) -> Result<(), CompileError> {
    match rule {
        Rule::MacLegacy => {
            let m = mac_operands(ctx, inst)?;
            legacy_mad(ctx, inst, m)
        }
        Rule::Mac => {
            if ctx.precision == PrecisionMode::Exact {
                let m = mac_operands(ctx, inst)?;
                return unfused_mad(ctx, inst, VOP2_MUL_F32, m);
            }
            emit_fused_vop2(ctx, inst, VOP2_FMAC_F32)
        }
        Rule::Madmk | Rule::Madak => {
            let Fields::Vop2 {
                vdst,
                src0,
                vsrc1,
                constant: Some(k),
                ext: VopExt::None,
            } = inst.fields
            else {
                return Err(unsupported(
                    inst,
                    ctx.target,
                    "malformed inline-constant MAD",
                ));
            };
            if ctx.precision == PrecisionMode::Exact {
                let (a, b, c) = if rule == Rule::Madmk {
                    (src0, Operand::Literal(k), vsrc1)
                } else {
                    (src0, vsrc1, Operand::Literal(k))
                };
                let m = Mac {
                    vdst,
                    src0: a,
                    src1: b,
                    src2: c,
                    abs: 0,
                    neg: 0,
                    clamp: false,
                    omod: 0,
                };
                return unfused_mad(ctx, inst, VOP2_MUL_F32, m);
            }
            let op = if rule == Rule::Madmk {
                VOP2_FMAMK_F32
            } else {
                VOP2_FMAAK_F32
            };
            emit_fused_vop2(ctx, inst, op)
        }
        Rule::Dot2c | Rule::Dot4c => {
            let (vdst, src0, src1, neg, clamp) = match inst.fields {
                Fields::Vop2 {
                    vdst,
                    src0,
                    vsrc1,
                    ext: VopExt::None,
                    ..
                } => (vdst, src0, vsrc1, 0, false),
                Fields::Vop3(v) if v.abs == 0 && v.omod == 0 && v.opsel == 0 => {
                    (v.vdst, v.src[0], v.src[1], v.neg, v.clamp)
                }
                _ => {
                    return Err(unsupported(
                        inst,
                        ctx.target,
                        "dot product modifiers have no VOP3P form",
                    ))
                }
            };
            let (opcode, neg, neg_hi) = if rule == Rule::Dot2c {
                (VOP3P_DOT2_F32_F16, neg & 0b011, neg & 0b011)
            } else if neg == 0 {
                (VOP3P_DOT4_I32_I8, SIGNED_SOURCES, 0)
            } else {
                return Err(unsupported(inst, ctx.target, "negated integer dot sources"));
            };
            let v = Vop3p {
                vdst,
                src: [src0, src1, vdst],
                neg,
                neg_hi,
                opsel: 0,
                opsel_hi: PACKED_HI_ALL,
                clamp,
            };
            ctx.push(derive(
                inst,
                F::Vop3p,
                opcode,
                flag::SRC3 | flag::READS_DST,
                Fields::Vop3p(v),
            ));
            Ok(())
        }
        _ => Err(unsupported(
            inst,
            ctx.target,
            "no vector lowering for this rule",
        )),
    }
}

fn emit_fused_vop2(ctx: &mut Ctx<'_>, inst: &Instruction, vop2: u16) -> Result<(), CompileError> {
    let op = vop2_op(ctx.target, vop2);
    match inst.fields {
        Fields::Vop3(_) => ctx.push(retarget(inst, VOP3_VOP2_BASE + op)),
        Fields::Vop2 {
            ext: VopExt::Sdwa(_),
            ..
        } => {
            return Err(unsupported(
                inst,
                ctx.target,
                "SDWA MAC has no fused equivalent",
            ))
        }
        _ => ctx.push(retarget(inst, op)),
    }
    Ok(())
}

fn fused_mad(ctx: &mut Ctx<'_>, inst: &Instruction, vop3: u16, m: Mac) -> Result<(), CompileError> {
    let v = Vop3 {
        vdst: m.vdst,
        sdst: None,
        src: [m.src0, m.src1, m.src2],
        abs: m.abs,
        neg: m.neg,
        opsel: 0,
        clamp: m.clamp,
        omod: m.omod,
    };
    ctx.push(vop3_inst(inst, vop3, flag::SRC3, v));
    Ok(())
}

/// PS5 `v_mad_legacy_f32` (VOP3 0x140) and `v_mac_legacy_f32` (VOP2 0x06) are unfused: a zero
/// factor yields +0 even against inf or NaN (DX9 rule), the product and the sum round
/// separately, and, like every gfx10 MAD, they flush f32 denormals whatever the mode.
/// AnyPS5's FmaLegacy/MacLegacy tables pin this: (0x3f800800, 0x3f800800, 0xbf800000) gives
/// 0x3a000000, where a fused multiply-add would give 0x3a000400. No target keeps the
/// instruction (gfx10.3 reused 0x140 for the fused `v_fma_legacy_f32`, gfx11 and gfx12 only
/// have the fused `v_fma_dx9_zero_f32`), so every target runs `v_mul_legacy_f32`
/// (`v_mul_dx9_zero_f32`) and `v_add_f32` with f32 denormals flushed. `exec` implements the
/// same semantics.
fn legacy_mad(ctx: &mut Ctx<'_>, inst: &Instruction, m: Mac) -> Result<(), CompileError> {
    let restore = ctx.enter_f32_flush(inst)?;
    unfused_mad(ctx, inst, VOP2_MUL_LEGACY_F32, m)?;
    ctx.leave_f32_flush(inst, restore)
}

/// `vdst = src0 * src1 + src2` with the product rounded on its own, through the gfx1013 VOP2
/// multiply `mul` and `v_add_f32`.
fn unfused_mad(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    mul: u16,
    m: Mac,
) -> Result<(), CompileError> {
    let tmp = Operand::Vgpr(ctx.regs.scratch_vgprs(1)?);
    let mul_op = VOP3_VOP2_BASE + vop2_op(ctx.target, mul);
    let add_op = VOP3_VOP2_BASE + vop2_op(ctx.target, VOP2_ADD_F32);
    let product = Vop3 {
        vdst: tmp,
        sdst: None,
        src: [m.src0, m.src1, Operand::Raw(0)],
        abs: m.abs & 0b011,
        neg: m.neg & 0b011,
        opsel: 0,
        clamp: false,
        omod: 0,
    };
    ctx.push(vop3_inst(inst, mul_op, 0, product));
    let sum = Vop3 {
        vdst: m.vdst,
        sdst: None,
        src: [tmp, m.src2, Operand::Raw(0)],
        abs: (m.abs >> 1) & 0b010,
        neg: (m.neg >> 1) & 0b010,
        opsel: 0,
        clamp: m.clamp,
        omod: m.omod,
    };
    ctx.push(vop3_inst(inst, add_op, 0, sum));
    Ok(())
}

fn vop3_only_op(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), CompileError> {
    let info = lookup(ctx, inst, F::Vop3, inst.opcode)?;
    match info.translate(ctx.target) {
        Xlat::Op(op) if needs_mad64_scratch(ctx.target, inst) => {
            mad64_through_scratch(ctx, inst, op)
        }
        Xlat::Op(op) => {
            ctx.push(retarget(inst, op));
            Ok(())
        }
        Xlat::Drop | Xlat::Same => Ok(()),
        Xlat::No => Err(unsupported(
            inst,
            ctx.target,
            "instruction was removed on the target",
        )),
        Xlat::Rule(Rule::Interp) => interp::lower_vop3(ctx, inst),
        Xlat::Rule(rule @ (Rule::Mad | Rule::MadLegacy)) => {
            let Fields::Vop3(v) = inst.fields else {
                return Err(unsupported(inst, ctx.target, "malformed MAD"));
            };
            let m = Mac {
                vdst: v.vdst,
                src0: v.src[0],
                src1: v.src[1],
                src2: v.src[2],
                abs: v.abs,
                neg: v.neg,
                clamp: v.clamp,
                omod: v.omod,
            };
            if rule == Rule::MadLegacy {
                return legacy_mad(ctx, inst, m);
            }
            if ctx.precision == PrecisionMode::Exact {
                return unfused_mad(ctx, inst, VOP2_MUL_F32, m);
            }
            fused_mad(ctx, inst, vop3_only(ctx.target, VOP3_FMA_F32), m)
        }
        Xlat::Rule(_) => Err(unsupported(
            inst,
            ctx.target,
            "no VOP3 lowering for this rule",
        )),
    }
}

/// gfx11.0 forwards results inside `v_mad_u64_u32`/`v_mad_i64_i32` too early
/// (`FeatureMADIntraFwdBug`), so LLVM never lets their destination overlap a source; guest code
/// that accumulates in place computes into scratch VGPRs instead.
fn needs_mad64_scratch(target: TargetIsa, inst: &Instruction) -> bool {
    let Fields::Vop3(v) = inst.fields else {
        return false;
    };
    let Operand::Vgpr(dst) = v.vdst else {
        return false;
    };
    let dst = u16::from(dst)..u16::from(dst) + 2;
    let widths = [1, 1, 2];
    let overlaps = v.src.iter().zip(widths).any(|(src, width)| match src {
        Operand::Vgpr(first) => {
            let first = u16::from(*first);
            first < dst.end && dst.start < first + width
        }
        _ => false,
    });
    target == TargetIsa::Gfx11
        && matches!(inst.opcode, VOP3_MAD_U64_U32 | VOP3_MAD_I64_I32)
        && overlaps
}

fn mad64_through_scratch(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    op: u16,
) -> Result<(), CompileError> {
    let Fields::Vop3(v) = inst.fields else {
        return Err(unsupported(inst, ctx.target, "malformed 64-bit MAD"));
    };
    let Operand::Vgpr(dst) = v.vdst else {
        return Err(unsupported(inst, ctx.target, "malformed 64-bit MAD"));
    };
    let tmp = ctx.regs.scratch_vgprs(2)?;
    let mut mad = retarget(inst, op);
    mad.fields = Fields::Vop3(Vop3 {
        vdst: Operand::Vgpr(tmp),
        ..v
    });
    ctx.push(mad);
    for k in 0..2 {
        let fields = Fields::Vop1 {
            vdst: Operand::Vgpr(dst + k),
            src0: Operand::Vgpr(tmp + k),
            ext: VopExt::None,
        };
        ctx.push(derive(inst, F::Vop1, opcodes::VOP1_MOV_B32, 0, fields));
    }
    Ok(())
}

fn vopc(ctx: &mut Ctx<'_>, inst: &Instruction, op: u16) -> Result<(), CompileError> {
    let info =
        VopcInfo::decode(op).ok_or_else(|| unsupported(inst, ctx.target, "unknown compare"))?;
    if ctx.target.is_gfx12() {
        if let Some(value) = info.is_constant_result() {
            return constant_compare(ctx, inst, info.exec, value);
        }
    }
    let target_op = if ctx.target.is_gfx11_plus() {
        info.gfx11_opcode()
    } else {
        op
    };
    let Fields::Vopc {
        sdst,
        src0,
        vsrc1,
        ext,
    } = inst.fields
    else {
        ctx.push(retarget(inst, target_op));
        return Ok(());
    };
    if ctx.target.is_gfx11_plus() {
        if let VopExt::Sdwa(_) = ext {
            return sdwa::expand(ctx, inst, target_op);
        }
        if inst.has_flag(flag::T16) && high_vgpr(&[src0, vsrc1]) {
            if ext != VopExt::None {
                return Err(unsupported(
                    inst,
                    ctx.target,
                    "16-bit DPP compare above v127",
                ));
            }
            let v = Vop3::new(sdst, [src0, vsrc1, Operand::Raw(0)]);
            ctx.push(vop3_inst(inst, target_op, inst.flags, v));
            return Ok(());
        }
    }
    ctx.push(retarget(inst, target_op));
    Ok(())
}

fn constant_compare(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    exec: bool,
    value: bool,
) -> Result<(), CompileError> {
    let dst = match inst.fields {
        _ if exec => Operand::ExecLo,
        Fields::Vopc { sdst, .. } => sdst,
        Fields::Vop3(v) => v.vdst,
        _ => return Err(unsupported(inst, ctx.target, "malformed compare")),
    };
    if exec && value {
        return Ok(());
    }
    let src0 = if value {
        Operand::ExecLo
    } else {
        Operand::InlineInt(0)
    };
    let (opcode, mut flags) = if ctx.wave64 {
        (SOP1_GFX11_MOV_B64, flag::D2 | flag::S0_2)
    } else {
        (SOP1_GFX11_MOV_B32, 0)
    };
    if dst == Operand::ExecLo {
        flags |= flag::WRITES_EXEC;
    }
    ctx.push(derive(
        inst,
        F::Sop1,
        opcode,
        flags,
        Fields::Sop1 { sdst: dst, src0 },
    ));
    Ok(())
}

fn vop3p(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), CompileError> {
    let info = lookup(ctx, inst, F::Vop3p, inst.opcode)?;
    match info.translate(ctx.target) {
        Xlat::Op(op) => ctx.push(retarget(inst, op)),
        Xlat::Drop | Xlat::Same => {}
        Xlat::No => {
            return Err(unsupported(
                inst,
                ctx.target,
                "packed instruction was removed on the target",
            ))
        }
        Xlat::Rule(Rule::Dot2(signed)) => return super::dot2::lower(ctx, inst, signed),
        Xlat::Rule(Rule::DotSigned(op)) => {
            let Fields::Vop3p(mut v) = inst.fields else {
                return Err(unsupported(inst, ctx.target, "malformed packed dot"));
            };
            if v.neg != 0 || v.neg_hi != 0 {
                return Err(unsupported(inst, ctx.target, "negated integer dot sources"));
            }
            v.neg = SIGNED_SOURCES;
            ctx.push(derive(inst, F::Vop3p, op, inst.flags, Fields::Vop3p(v)));
        }
        Xlat::Rule(_) => {
            return Err(unsupported(
                inst,
                ctx.target,
                "no VOP3P lowering for this rule",
            ))
        }
    }
    Ok(())
}
