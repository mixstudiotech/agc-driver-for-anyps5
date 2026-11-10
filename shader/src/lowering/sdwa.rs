use super::vector::{vop2_inst, vop2_op, vop3_inst, vop3_only};
use super::{unsupported, Ctx};
use crate::ir::opcodes::{flag, VOP2_LSHLREV_B32, VOP3_BFE_I32, VOP3_BFE_U32, VOP3_BFI_B32};
use crate::ir::{
    Fields, Instruction, Operand, Sdwa, Vop3, VopExt, SDWA_SEL_DWORD, SDWA_UNUSED_PAD,
    SDWA_UNUSED_PRESERVE, SDWA_UNUSED_SEXT,
};
use crate::CompileError;

const SCRATCH_VGPRS: u8 = 3;
const RESULT_SLOT: u8 = 2;

pub fn select_bits(sel: u8) -> Option<Option<(u32, u32)>> {
    match sel {
        0..=3 => Some(Some((8 * u32::from(sel), 8))),
        4 => Some(Some((0, 16))),
        5 => Some(Some((16, 16))),
        SDWA_SEL_DWORD => Some(None),
        _ => None,
    }
}

pub(super) fn expand(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    vop3_op: u16,
) -> Result<(), CompileError> {
    let (dst, sources, sdwa, compare) = match inst.fields {
        Fields::Vop1 {
            vdst,
            src0,
            ext: VopExt::Sdwa(s),
        } => (vdst, vec![src0], s, false),
        Fields::Vop2 {
            vdst,
            src0,
            vsrc1,
            ext: VopExt::Sdwa(s),
            ..
        } => (vdst, vec![src0, vsrc1], s, false),
        Fields::Vopc {
            sdst,
            src0,
            vsrc1,
            ext: VopExt::Sdwa(s),
        } => (sdst, vec![src0, vsrc1], s, true),
        _ => return Err(unsupported(inst, ctx.target, "malformed SDWA instruction")),
    };
    if inst.has_flag(flag::READS_DST) {
        return Err(unsupported(
            inst,
            ctx.target,
            "SDWA accumulation has no gfx11 equivalent",
        ));
    }
    let scratch = ctx.regs.scratch_vgprs(SCRATCH_VGPRS)?;
    let bfe_u = vop3_only(ctx.target, VOP3_BFE_U32);
    let bfe_i = vop3_only(ctx.target, VOP3_BFE_I32);
    let selects = [
        (sdwa.src0_sel, sdwa.src0_sext, sdwa.src0_neg, sdwa.src0_abs),
        (sdwa.src1_sel, sdwa.src1_sext, sdwa.src1_neg, sdwa.src1_abs),
    ];
    let mut src = [Operand::Raw(0); 3];
    let (mut neg, mut abs) = (0u8, 0u8);
    for (i, (&source, &(sel, sext, n, a))) in sources.iter().zip(&selects).enumerate() {
        let bits = select_bits(sel)
            .ok_or_else(|| unsupported(inst, ctx.target, "invalid SDWA source select"))?;
        src[i] = match bits {
            Some((offset, width)) => {
                let tmp = Operand::Vgpr(scratch + i as u8);
                let extract = Vop3::new(
                    tmp,
                    [source, Operand::constant(offset), Operand::constant(width)],
                );
                ctx.push(vop3_inst(
                    inst,
                    if sext { bfe_i } else { bfe_u },
                    flag::SRC3,
                    extract,
                ));
                tmp
            }
            None => source,
        };
        neg |= u8::from(n) << i;
        abs |= u8::from(a) << i;
    }
    if inst.has_flag(flag::CARRY_IN) {
        src[2] = Operand::VccLo;
    }
    let dst_bits = if compare {
        None
    } else {
        select_bits(sdwa.dst_sel)
            .ok_or_else(|| unsupported(inst, ctx.target, "invalid SDWA destination select"))?
    };
    let result = Operand::Vgpr(scratch + RESULT_SLOT);
    let op = Vop3 {
        vdst: if dst_bits.is_some() { result } else { dst },
        sdst: inst.has_flag(flag::CARRY_OUT).then_some(Operand::VccLo),
        src,
        abs,
        neg,
        opsel: 0,
        clamp: sdwa.clamp,
        omod: sdwa.omod,
    };
    ctx.push(vop3_inst(inst, vop3_op, inst.flags & !flag::T16, op));
    if let Some((offset, width)) = dst_bits {
        merge(ctx, inst, &sdwa, dst, result, offset, width)?;
    }
    Ok(())
}

fn merge(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    sdwa: &Sdwa,
    dst: Operand,
    result: Operand,
    offset: u32,
    width: u32,
) -> Result<(), CompileError> {
    let lshl = vop2_op(ctx.target, VOP2_LSHLREV_B32);
    let shift = |ctx: &mut Ctx<'_>, to: Operand| {
        ctx.push(vop2_inst(inst, lshl, to, Operand::constant(offset), result));
    };
    match sdwa.dst_unused {
        SDWA_UNUSED_PRESERVE => {
            if offset > 0 {
                shift(ctx, result);
            }
            let mask = (((1u64 << width) - 1) as u32) << offset;
            let insert = Vop3::new(dst, [Operand::constant(mask), result, dst]);
            ctx.push(vop3_inst(
                inst,
                vop3_only(ctx.target, VOP3_BFI_B32),
                flag::SRC3,
                insert,
            ));
        }
        SDWA_UNUSED_PAD | SDWA_UNUSED_SEXT => {
            let extract = if sdwa.dst_unused == SDWA_UNUSED_SEXT {
                VOP3_BFE_I32
            } else {
                VOP3_BFE_U32
            };
            let to = if offset == 0 { dst } else { result };
            let bfe = Vop3::new(
                to,
                [result, Operand::InlineInt(0), Operand::constant(width)],
            );
            ctx.push(vop3_inst(
                inst,
                vop3_only(ctx.target, extract),
                flag::SRC3,
                bfe,
            ));
            if offset > 0 {
                shift(ctx, dst);
            }
        }
        _ => {
            return Err(unsupported(
                inst,
                ctx.target,
                "invalid SDWA unused-bits policy",
            ))
        }
    }
    Ok(())
}
