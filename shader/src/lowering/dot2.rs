//! Integer DOT2 on gfx11+: two exact MAD64 operations, then one final truncation or
//! saturation. Each packed source half is selected before its signed/unsigned extension.

use super::{Ctx, unsupported};
use crate::CompileError;
use crate::ir::{Fields, Instruction, InstructionFamily as F, Operand, Vop3, VopExt};

pub(super) fn lower(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    signed: bool,
) -> Result<(), CompileError> {
    let Fields::Vop3p(dot) = inst.fields else {
        return Err(unsupported(inst, ctx.target, "malformed integer DOT2"));
    };
    if dot.neg != 0 || dot.neg_hi != 0 || dot.opsel & 4 != 0 || dot.opsel_hi & 4 == 0 {
        return Err(unsupported(
            inst,
            ctx.target,
            "integer DOT2 requires unmodified full-width accumulator and legal half selectors",
        ));
    }
    let base = ctx.regs.scratch_vgprs(8)?;
    let v = |n| Operand::Vgpr(base + n);
    // Copy every scalar source before borrowing scratch SGPRs: a preceding lowering may
    // itself have materialized a source there. Original VGPR destinations may alias sources.
    for half in 0..2 {
        for source in 0..2 {
            let selectors = if half == 0 { dot.opsel } else { dot.opsel_hi };
            let offset = if selectors & (1 << source) != 0 {
                16
            } else {
                0
            };
            op3(
                ctx,
                inst,
                if signed { 0x149 } else { 0x148 },
                v(half * 2 + source),
                [
                    dot.src[source as usize],
                    Operand::InlineInt(offset),
                    Operand::InlineInt(16),
                ],
            )?;
        }
    }
    mov(ctx, inst, v(4), dot.src[2])?;
    if signed {
        op3(
            ctx,
            inst,
            0x100 + 0x18,
            v(5),
            [Operand::InlineInt(31), v(4), Operand::Raw(0)],
        )?;
    } else {
        mov(ctx, inst, v(5), Operand::InlineInt(0))?;
    }
    let mask = Operand::Sgpr(ctx.transient_sgprs(inst, 2)?);
    // Destinations never overlap sources, including on gfx11.0 with MADIntraFwdBug.
    for (dst, a, b, sum) in [(6, 0, 1, 4), (4, 2, 3, 6)] {
        let mut mad = Vop3::new(v(dst), [v(a), v(b), v(sum)]);
        mad.sdst = Some(mask);
        ctx.lower_gfx1013(
            inst,
            F::Vop3,
            if signed { 0x177 } else { 0x176 },
            Fields::Vop3(mad),
        )?;
    }
    if !dot.clamp {
        return mov(ctx, inst, dot.vdst, v(4));
    }
    let (compare, saturated) = if signed {
        // An i32 result fits iff its sign extension agrees with the 64-bit high word.
        op3(
            ctx,
            inst,
            0x100 + 0x18,
            v(0),
            [Operand::InlineInt(31), v(4), Operand::Raw(0)],
        )?;
        op3(
            ctx,
            inst,
            0x100 + 0x18,
            v(1),
            [Operand::InlineInt(31), v(5), Operand::Raw(0)],
        )?;
        op3(
            ctx,
            inst,
            0x100 + 0x1d,
            v(1),
            [Operand::Literal(0x7fff_ffff), v(1), Operand::Raw(0)],
        )?;
        (v(0), v(1))
    } else {
        (Operand::InlineInt(0), Operand::InlineInt(-1))
    };
    op3(ctx, inst, 0xc5, mask, [compare, v(5), Operand::Raw(0)])?;
    // CMP and MAD carry outputs use our own scalar mask; guest VCC and SCC stay untouched.
    op3(ctx, inst, 0x101, dot.vdst, [v(4), saturated, mask])
}

fn mov(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    dst: Operand,
    src: Operand,
) -> Result<(), CompileError> {
    ctx.lower_gfx1013(
        inst,
        F::Vop1,
        1,
        Fields::Vop1 {
            vdst: dst,
            src0: src,
            ext: VopExt::None,
        },
    )
}

fn op3(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    opcode: u16,
    dst: Operand,
    src: [Operand; 3],
) -> Result<(), CompileError> {
    ctx.lower_gfx1013(inst, F::Vop3, opcode, Fields::Vop3(Vop3::new(dst, src)))
}
