//! gfx11/gfx12 encode only the first few NSA addresses separately. The last encoded
//! address starts a consecutive tuple (Mesa ACO's MIMG operand grouping).

use super::Ctx;
use crate::ir::{Fields, Instruction, InstructionFamily as F, Operand, VopExt, memops, opcodes};
use crate::{CompileError, TargetIsa};

pub(super) fn pack(
    ctx: &mut Ctx<'_>,
    origin: &Instruction,
    image: &mut Instruction,
) -> Result<(), CompileError> {
    let Fields::Mimg(m) = &mut image.fields else {
        return Ok(());
    };
    if !ctx.target.is_gfx11_plus() || !m.nsa {
        return Ok(());
    }
    let slots =
        if ctx.target == TargetIsa::Gfx12 && memops::gfx12_vsample(image.opcode, m.uses_sampler) {
            4
        } else {
            5
        };
    let count = usize::from(m.addr_dwords);
    if count <= slots {
        return Ok(());
    }
    let tail = slots - 1;
    if (tail..count).all(|i| usize::from(m.addr[i]) == usize::from(m.addr[tail]) + i - tail) {
        return Ok(());
    }
    let base = ctx.regs.scratch_vgprs((count - tail) as u8)?;
    for i in tail..count {
        let dst = base + (i - tail) as u8;
        ctx.lower_gfx1013(
            origin,
            F::Vop1,
            opcodes::VOP1_MOV_B32,
            Fields::Vop1 {
                vdst: Operand::Vgpr(dst),
                src0: Operand::Vgpr(m.addr[i]),
                ext: VopExt::None,
            },
        )?;
        m.addr[i] = dst;
    }
    Ok(())
}
