//! gfx10 interpolation (`v_interp_*`) on gfx11+: `lds_param_load` plus VINTERP or a quad
//! broadcast. The parameter VGPR comes from the prologue when [`super::pixel`] hoisted it;
//! otherwise it is loaded here with every lane enabled, because VINTERP reads P0, P10 and P20
//! from lanes 0, 1 and 2 of each quad.

use super::pixel::param_load;
use super::waitcnt::{encode_gfx11, GFX10_LGKMCNT_MAX, GFX10_VMCNT_MAX, GFX12_WAIT_EXPCNT};
use super::{derive, retarget, unsupported, Ctx, SOPP_GFX11_WAITCNT};
use crate::ir::opcodes::{
    VINTERP_P10_F16_F32, VINTERP_P10_F32, VINTERP_P2_F16_F32, VINTERP_P2_F32, VINTRP_MOV,
    VINTRP_P1, VINTRP_P2, VOP1_MOV_B32, VOP3_INTERP_P1LL_F16, VOP3_INTERP_P1LV_F16,
    VOP3_INTERP_P1_F32, VOP3_INTERP_P2_F16,
};
use crate::ir::{
    Dpp16, Fields, Instruction, InstructionFamily as F, Operand, VInterp, Vop3p, VopExt,
};
use crate::CompileError;

/// Quad lane of each `v_interp_mov_f32` parameter: P10, P20, P0.
const PARAM_LANE: [u16; 3] = [1, 2, 0];
const P10_LANE: u16 = 1;
const ATTR_MASK: u16 = 0x3f;
const CHAN_SHIFT: u16 = 6;
const HIGH_HALF: u16 = 1 << 8;
/// VINTERP `wait_exp` that waits for nothing: the prologue already waited for its loads.
const NO_EXP_WAIT: u8 = 7;
/// VOP3 opsel bits: source 2's high half, the destination's high half.
const OPSEL_SRC2_HI: u8 = 1 << 2;
const OPSEL_DST_HI: u8 = 1 << 3;
const VOP3P_FMA_MIX_F32: u16 = 0x20;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    P1,
    P2,
    Mov,
    P1llF16,
    P1lvF16,
    P2F16,
}

struct Request {
    kind: Kind,
    vdst: u8,
    /// The barycentric coordinate, or the parameter (`Raw`) of `v_interp_mov_f32`.
    coord: Operand,
    /// Source 2 of `v_interp_p1lv_f16` (P0) and `v_interp_p2_f16` (the p1 result).
    addend: Operand,
    attr: u8,
    chan: u8,
    /// The upper f16 of the attribute's packed pair.
    high: bool,
    /// Negation per source as VINTERP encodes it (source 1 is the coordinate).
    neg: u8,
    opsel: u8,
    clamp: bool,
}

pub(super) fn lower(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), CompileError> {
    if !ctx.target.is_gfx11_plus() {
        ctx.push(retarget(inst, inst.opcode));
        return Ok(());
    }
    let Fields::Vintrp(i) = inst.fields else {
        return Err(unsupported(inst, ctx.target, "malformed interpolation"));
    };
    let (kind, coord) = match inst.opcode {
        VINTRP_P1 => (Kind::P1, Operand::Vgpr(i.vsrc)),
        VINTRP_P2 => (Kind::P2, Operand::Vgpr(i.vsrc)),
        VINTRP_MOV => (Kind::Mov, Operand::Raw(u16::from(i.vsrc))),
        _ => {
            return Err(unsupported(
                inst,
                ctx.target,
                "unknown interpolation opcode",
            ))
        }
    };
    emit(
        ctx,
        inst,
        Request {
            kind,
            vdst: i.vdst,
            coord,
            addend: Operand::Null,
            attr: i.attr,
            chan: i.chan,
            high: false,
            neg: 0,
            opsel: 0,
            clamp: false,
        },
    )
}

pub(super) fn lower_vop3(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), CompileError> {
    let Fields::Vop3(v) = inst.fields else {
        return Err(unsupported(inst, ctx.target, "malformed interpolation"));
    };
    let Operand::Raw(select) = v.src[0] else {
        return Err(unsupported(
            inst,
            ctx.target,
            "malformed interpolation attribute",
        ));
    };
    let Operand::Vgpr(vdst) = v.vdst else {
        return Err(unsupported(
            inst,
            ctx.target,
            "malformed interpolation destination",
        ));
    };
    let kind = match inst.opcode {
        VOP3_INTERP_P1LL_F16 => Kind::P1llF16,
        VOP3_INTERP_P1LV_F16 => Kind::P1lvF16,
        VOP3_INTERP_P2_F16 => Kind::P2F16,
        op => match op - VOP3_INTERP_P1_F32 {
            VINTRP_P1 => Kind::P1,
            VINTRP_P2 => Kind::P2,
            _ => Kind::Mov,
        },
    };
    let f16 = matches!(kind, Kind::P1llF16 | Kind::P1lvF16 | Kind::P2F16);
    let opsel_used = match kind {
        Kind::P1lvF16 => OPSEL_SRC2_HI,
        Kind::P2F16 => OPSEL_DST_HI,
        _ => 0,
    };
    if v.abs != 0 || v.omod != 0 || v.opsel & !opsel_used != 0 || (!f16 && select & HIGH_HALF != 0)
    {
        return Err(unsupported(
            inst,
            ctx.target,
            "interpolation modifiers have no VINTERP form",
        ));
    }
    let neg_used = match kind {
        Kind::P1lvF16 | Kind::P2F16 => 0b110,
        _ => 0b010,
    };
    emit(
        ctx,
        inst,
        Request {
            kind,
            vdst,
            coord: v.src[1],
            addend: v.src[2],
            attr: (select & ATTR_MASK) as u8,
            chan: ((select >> CHAN_SHIFT) & 3) as u8,
            high: select & HIGH_HALF != 0,
            neg: v.neg & neg_used,
            opsel: v.opsel,
            clamp: v.clamp,
        },
    )
}

fn emit(ctx: &mut Ctx<'_>, inst: &Instruction, req: Request) -> Result<(), CompileError> {
    let hoisted = ctx
        .pixel
        .and_then(|plan| plan.hoisted(inst.pc, req.attr, req.chan));
    let scratch = ctx.regs.scratch_vgprs(2)?;
    let (param, wait_exp) = match hoisted {
        Some(vgpr) => (vgpr, NO_EXP_WAIT),
        None => {
            let saved = ctx.transient_sgprs(inst, 2)?;
            ctx.enable_all_lanes(inst, saved)?;
            ctx.push(param_load(inst, scratch, req.attr, req.chan));
            ctx.restore_lanes(inst, saved)?;
            (scratch, 0)
        }
    };
    let p = Operand::Vgpr(param);
    let high = u8::from(req.high);
    let vinterp = |opcode, src2, opsel| {
        let v = VInterp {
            vdst: req.vdst,
            src: [p, req.coord, src2],
            neg: req.neg,
            opsel,
            clamp: req.clamp,
            wait_exp,
        };
        derive(inst, F::Vinterp, opcode, 0, Fields::Vinterp(v))
    };
    match req.kind {
        Kind::P1 => ctx.push(vinterp(VINTERP_P10_F32, p, 0)),
        Kind::P2 => ctx.push(vinterp(VINTERP_P2_F32, Operand::Vgpr(req.vdst), 0)),
        Kind::P1llF16 => ctx.push(vinterp(VINTERP_P10_F16_F32, p, high * 0b101)),
        Kind::P2F16 => {
            let opsel = high | (req.opsel & OPSEL_DST_HI);
            ctx.push(vinterp(VINTERP_P2_F16_F32, req.addend, opsel));
        }
        Kind::Mov => {
            let Operand::Raw(slot) = req.coord else {
                return Err(unsupported(
                    inst,
                    ctx.target,
                    "malformed interpolation parameter",
                ));
            };
            let lane = *PARAM_LANE
                .get(usize::from(slot))
                .ok_or_else(|| unsupported(inst, ctx.target, "invalid interpolation parameter"))?;
            if wait_exp != NO_EXP_WAIT {
                ctx.wait_for_parameters(inst);
            }
            ctx.push(quad_broadcast(inst, req.vdst, p, lane));
        }
        Kind::P1lvF16 => {
            // P0 comes from a VGPR of the lane itself, which VINTERP's quad-relative source 2
            // cannot read: broadcast P10 and use a mixed-precision FMA.
            if wait_exp != NO_EXP_WAIT {
                ctx.wait_for_parameters(inst);
            }
            let p10 = scratch + 1;
            ctx.push(quad_broadcast(inst, p10, p, P10_LANE));
            let mix = Vop3p {
                vdst: Operand::Vgpr(req.vdst),
                src: [Operand::Vgpr(p10), req.coord, req.addend],
                neg: req.neg,
                neg_hi: 0,
                opsel: high | (req.opsel & OPSEL_SRC2_HI),
                opsel_hi: 0b101,
                clamp: req.clamp,
            };
            ctx.lower_gfx1013(inst, F::Vop3p, VOP3P_FMA_MIX_F32, Fields::Vop3p(mix))?;
        }
    }
    Ok(())
}

/// `v_mov_b32_dpp vdst, src quad_perm:[lane,lane,lane,lane] fi:1`.
fn quad_broadcast(origin: &Instruction, vdst: u8, src: Operand, lane: u16) -> Instruction {
    let dpp = Dpp16 {
        ctrl: lane | (lane << 2) | (lane << 4) | (lane << 6),
        fetch_inactive: true,
        bound_ctrl: false,
        src0_neg: false,
        src0_abs: false,
        src1_neg: false,
        src1_abs: false,
        bank_mask: 0xf,
        row_mask: 0xf,
    };
    let fields = Fields::Vop1 {
        vdst: Operand::Vgpr(vdst),
        src0: src,
        ext: VopExt::Dpp16(dpp),
    };
    derive(origin, F::Vop1, VOP1_MOV_B32, 0, fields)
}

impl Ctx<'_> {
    /// Waits for every `lds_param_load` in flight (they count on EXPcnt).
    pub(super) fn wait_for_parameters(&mut self, origin: &Instruction) {
        let (opcode, simm16) = if self.target.is_gfx12() {
            (GFX12_WAIT_EXPCNT, 0)
        } else {
            (
                SOPP_GFX11_WAITCNT,
                encode_gfx11(GFX10_VMCNT_MAX, 0, GFX10_LGKMCNT_MAX),
            )
        };
        self.push(derive(origin, F::Sopp, opcode, 0, Fields::Sopp { simm16 }));
    }
}
