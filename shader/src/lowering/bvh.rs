//! `image_bvh_intersect_ray` / `image_bvh64_intersect_ray` for hosts other than gfx1013.
//!
//! No other target is known to reproduce the gfx1013 ray unit bit for bit (gfx10.3 shares the
//! encoding, node formats and descriptor but not a verified datapath; gfx11 repacks A16 operands
//! and reinterprets descriptor and pointer bits; gfx12 replaced the node formats), so the
//! instruction is expanded into gfx1013 vector/scalar code that evaluates the reference model of
//! the ray unit operation for operation and then follows the normal per-target lowering.
//!
//! The expansion is branch-free. It saves SCC and EXEC, decodes the descriptor with scalar
//! instructions, fetches the node with masked `global_load_dwordx4` (the second half only for
//! box32 lanes), widens box16 bounds in place, runs the box and triangle tests for every lane and
//! selects each lane's result by node type before writing VDATA under the original EXEC. The
//! arithmetic needs f32 denormal flushing, preserved f16 denormals and round-to-nearest-even; a
//! different mode is switched around the expansion and restored.

use super::{derive, unsupported, Ctx};
use crate::ir::memops::{self, img};
use crate::ir::opcodes::{self, flag};
use crate::ir::{
    Fields, Flat, FlatSegment, Image, Instruction, InstructionFamily as F, MemKind, Operand, Vop3,
    VopExt,
};
use crate::lowering::waitcnt;
use crate::CompileError;

const F32_INF: u32 = 0x7f80_0000;
const F32_NEG_INF: u32 = 0xff80_0000;
/// 2^-24: one box-grow step.
const F32_GROW_STEP: u32 = 0x3380_0000;
const F32_CANONICAL_NAN: u32 = 0xffc0_0000;
/// Box sorting network (pairs compared in order); ties never swap.
const SORT_NETWORK: [(usize, usize); 5] = [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)];
/// Vertex slots (`a, b, c`) of triangles 0-3 within a triangle node.
const TRIANGLE_VERTICES: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]];
const TRIANGLE_ID_DWORD: usize = 15;
const BOX32_DWORDS: usize = 28;

/// Instruction fields the ray unit requires; anything else is undefined on the hardware.
fn encoding_error(m: &Image) -> Option<&'static str> {
    if m.dmask != 0xf || !m.r128 || m.d16 || m.tfe || m.lwe {
        return Some("BVH intersection requires DMASK=0xF, R128=1 and no D16, TFE or LWE");
    }
    None
}

/// FLOAT_MODE (MODE bits 7:0) the expansion computes in: round to nearest even, f32 denormals
/// flushed, f16/f64 denormals preserved.
pub const FLOAT_MODE: u8 = 0xc0;
pub const SCRATCH_VGPRS: u8 = T + T_COUNT;
pub const SCRATCH_SGPRS: u8 = 24;

const HWREG_MODE: u16 = 1;
const HWREG_MODE_FLOAT: u16 = HWREG_MODE | (7 << 11);
const SOPP_WAITCNT: u16 = 0x0c;
const SOPP_ROUND_MODE: u16 = 0x24;
const SOPP_DENORM_MODE: u16 = 0x25;
const SOPK_GETREG: u16 = 0x12;
const SOPK_SETREG: u16 = 0x13;
const SOPK_SETREG_IMM32: u16 = 0x15;

const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP2_CSELECT_B32: u16 = 0x0a;
const SOP2_CSELECT_B64: u16 = 0x0b;
const SOP2_AND_B32: u16 = 0x0e;
const SOP2_AND_B64: u16 = 0x0f;
const SOP2_OR_B32: u16 = 0x10;
const SOP2_OR_B64: u16 = 0x11;
const SOP2_ANDN2_B32: u16 = 0x14;
const SOP2_ANDN2_B64: u16 = 0x15;
const SOP2_LSHL_B64: u16 = 0x1f;
const SOP2_BFE_U32: u16 = 0x27;
const SOPC_CMP_LG_U32: u16 = 0x07;
const SOPC_BITCMP1_B32: u16 = 0x0d;

const VOP1_NOP: u16 = 0x00;
const VOP1_MOV_B32: u16 = 0x01;
const VOP1_CVT_F32_U32: u16 = 0x06;
const VOP1_CVT_F32_F16: u16 = 0x0b;
const VOP2_ADD_F32: u16 = 0x03;
const VOP2_SUB_F32: u16 = 0x04;
const VOP2_MUL_F32: u16 = 0x08;
const VOP2_LSHRREV_B32: u16 = 0x16;
const VOP2_LSHLREV_B32: u16 = 0x1a;
const VOP2_AND_B32: u16 = 0x1b;
const VOP3_CNDMASK_B32: u16 = 0x101;
const VOP3_ADD_CO_CI_U32: u16 = 0x128;
const VOP3_BFE_U32: u16 = 0x148;
const VOP3_LSHLREV_B64: u16 = 0x2ff;
const VOP3_LSHRREV_B64: u16 = 0x300;
const VOP3_ADD_CO_U32: u16 = 0x30f;
const FLAT_LOAD_DWORDX4: u16 = 0x0e;

const CMP_LT_F32: u16 = 0x01;
const CMP_LE_F32: u16 = 0x03;
const CMP_GT_F32: u16 = 0x04;
const CMP_LG_F32: u16 = 0x05;
const CMP_GE_F32: u16 = 0x06;
const CMP_U_F32: u16 = 0x08;
const CMP_EQ_U32: u16 = 0xc2;
const CMP_LE_U32: u16 = 0xc3;
const CMP_NE_U32: u16 = 0xc5;
const CMP_GE_U32: u16 = 0xc6;
const CMP_LT_U64: u16 = 0xe1;
const CMP_LE_U64: u16 = 0xe3;

const ZERO: Operand = Operand::InlineInt(0);
const ALL_ONES: Operand = Operand::InlineInt(-1);
const F32_ONE_INLINE: Operand = Operand::InlineFloat(242);

// Scratch VGPRs, relative to the allocated base.
const DATA: u8 = 0;
const ADDR: u8 = DATA + BOX32_DWORDS as u8;
const INDEX: u8 = ADDR + 2;
const TYPE: u8 = INDEX + 2;
const EXTENT: u8 = TYPE + 1;
const FACTOR: u8 = EXTENT + 1;
const DIR: u8 = FACTOR + 1;
const INV: u8 = DIR + 3;
const CHILD: u8 = INV + 3;
const KEY: u8 = CHILD + 4;
const SPARE: u8 = KEY + 4;
const T: u8 = SPARE + 2;
const T_COUNT: u8 = 52;

// Scratch SGPR pairs (lane masks), relative to the allocated base; wave32 uses the low halves.
const P_EXEC: u8 = 0;
const P_SORT: u8 = 1;
const P_IJ: u8 = 2;
const P_VALID: u8 = 3;
const P_TRI: u8 = 4;
const P_A: u8 = 5;
const P_B: u8 = 6;
const P_C: u8 = 7;
const P_M: u8 = 8;
const P_N: u8 = 9;
const P_O: u8 = 10;
const S_SCC: u8 = 22;
const S_MODE: u8 = 23;

/// How the FP mode at the instruction is known.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModeState {
    /// The program never writes MODE: the mode is the shader's initial FLOAT_MODE.
    Static(u8),
    /// The program writes MODE; the expansion saves and restores it.
    Dynamic,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub wave64: bool,
    pub vgpr_base: u8,
    pub sgpr_base: u8,
    pub mode: ModeState,
}

pub(crate) fn writes_mode(inst: &Instruction) -> bool {
    match inst.fields {
        Fields::Sopp { .. } => matches!(inst.opcode, SOPP_ROUND_MODE | SOPP_DENORM_MODE),
        Fields::Sopk { simm16, .. } => {
            matches!(inst.opcode, SOPK_SETREG | SOPK_SETREG_IMM32) && simm16 & 0x3f == HWREG_MODE
        }
        _ => false,
    }
}

pub(super) fn lower(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), CompileError> {
    let Fields::Mimg(m) = inst.fields else {
        return Err(unsupported(inst, ctx.target, "malformed MIMG"));
    };
    if let Some(reason) = encoding_error(&m) {
        return Err(unsupported(inst, ctx.target, reason));
    }
    let vgpr_base = ctx.regs.scratch_vgprs(SCRATCH_VGPRS)?;
    let sgpr_base = ctx.regs.scratch_sgprs(SCRATCH_SGPRS).ok_or_else(|| {
        unsupported(
            inst,
            ctx.target,
            "BVH emulation needs 24 SGPRs above the shader's own",
        )
    })?;
    let layout = Layout {
        wave64: ctx.wave64,
        vgpr_base,
        sgpr_base,
        mode: ctx.mode,
    };
    for expanded in expand(inst, &layout) {
        ctx.lower(&expanded)?;
    }
    Ok(())
}

/// The gfx1013 instruction sequence that replaces one legacy BVH intersection.
pub(crate) fn expand(inst: &Instruction, layout: &Layout) -> Vec<Instruction> {
    let Fields::Mimg(m) = inst.fields else {
        return Vec::new();
    };
    let wide = memops::image(inst.opcode).is_some_and(|info| info.flags & img::BVH64 != 0);
    let mut g = Gen {
        origin: inst,
        m,
        wide,
        wave64: layout.wave64,
        v: layout.vgpr_base,
        s: layout.sgpr_base,
        out: Vec::new(),
    };
    g.enter_mode(layout.mode);
    g.sop2(
        SOP2_CSELECT_B32,
        g.single(S_SCC),
        Operand::InlineInt(1),
        ZERO,
    );
    g.mov_mask(g.pair(P_EXEC), Operand::ExecLo);
    g.descriptor();
    g.ray_inputs();
    g.fetch();
    let child = g.boxes();
    let result = g.triangle();
    g.write_back(child, result);
    g.sopc(SOPC_CMP_LG_U32, g.single(S_SCC), ZERO);
    g.leave_mode(layout.mode);
    g.out
}

struct Gen<'a> {
    origin: &'a Instruction,
    m: Image,
    wide: bool,
    wave64: bool,
    v: u8,
    s: u8,
    out: Vec<Instruction>,
}

impl Gen<'_> {
    // ---- operands -----------------------------------------------------------------------

    fn r(&self, offset: u8) -> Operand {
        Operand::Vgpr(self.v + offset)
    }

    fn pair(&self, slot: u8) -> Operand {
        Operand::Sgpr(self.s + 2 * slot)
    }

    fn single(&self, offset: u8) -> Operand {
        Operand::Sgpr(self.s + offset)
    }

    fn hi(&self, slot: u8) -> Operand {
        Operand::Sgpr(self.s + 2 * slot + 1)
    }

    fn address(&self, i: usize) -> Operand {
        Operand::Vgpr(self.m.address_register(i))
    }

    fn first_ray_dword(&self) -> usize {
        if self.wide {
            2
        } else {
            1
        }
    }

    fn extent(&self) -> Operand {
        self.address(self.first_ray_dword())
    }

    fn origin(&self, k: usize) -> Operand {
        self.address(self.first_ray_dword() + 1 + k)
    }

    fn dir(&self, k: usize) -> Operand {
        if self.m.a16 {
            self.r(DIR + k as u8)
        } else {
            self.address(self.first_ray_dword() + 4 + k)
        }
    }

    fn inv(&self, k: usize) -> Operand {
        if self.m.a16 {
            self.r(INV + k as u8)
        } else {
            self.address(self.first_ray_dword() + 7 + k)
        }
    }

    fn descriptor_word(&self, k: u8) -> Operand {
        Operand::Sgpr(self.m.srsrc + k)
    }

    fn data(&self, dword: usize) -> Operand {
        self.r(DATA + dword as u8)
    }

    // ---- instruction builders (gfx1013 opcodes) -------------------------------------------

    fn emit(&mut self, family: F, opcode: u16, flags: u64, fields: Fields) {
        self.out
            .push(derive(self.origin, family, opcode, flags, fields));
    }

    fn table_flags(family: F, opcode: u16) -> u64 {
        opcodes::lookup(family, opcode).map_or(0, |info| info.flags)
    }

    fn sop1(&mut self, opcode: u16, sdst: Operand, src0: Operand) {
        let flags = Self::table_flags(F::Sop1, opcode);
        self.emit(F::Sop1, opcode, flags, Fields::Sop1 { sdst, src0 });
    }

    fn sop2(&mut self, opcode: u16, sdst: Operand, src0: Operand, src1: Operand) {
        let flags = Self::table_flags(F::Sop2, opcode);
        self.emit(F::Sop2, opcode, flags, Fields::Sop2 { sdst, src0, src1 });
    }

    fn sopc(&mut self, opcode: u16, src0: Operand, src1: Operand) {
        let flags = Self::table_flags(F::Sopc, opcode);
        self.emit(F::Sopc, opcode, flags, Fields::Sopc { src0, src1 });
    }

    fn sopp(&mut self, opcode: u16, simm16: u16) {
        let flags = Self::table_flags(F::Sopp, opcode);
        self.emit(F::Sopp, opcode, flags, Fields::Sopp { simm16 });
    }

    fn sopk(&mut self, opcode: u16, sdst: Operand, simm16: u16) {
        let flags = Self::table_flags(F::Sopk, opcode);
        let fields = Fields::Sopk {
            sdst,
            simm16,
            imm32: None,
        };
        self.emit(F::Sopk, opcode, flags, fields);
    }

    fn mask_op(&self, b32: u16, b64: u16) -> u16 {
        if self.wave64 {
            b64
        } else {
            b32
        }
    }

    fn mov_mask(&mut self, dst: Operand, src: Operand) {
        self.sop1(self.mask_op(SOP1_MOV_B32, SOP1_MOV_B64), dst, src);
    }

    fn and(&mut self, dst: u8, a: u8, b: u8) {
        let op = self.mask_op(SOP2_AND_B32, SOP2_AND_B64);
        self.sop2(op, self.pair(dst), self.pair(a), self.pair(b));
    }

    fn or(&mut self, dst: u8, a: u8, b: u8) {
        let op = self.mask_op(SOP2_OR_B32, SOP2_OR_B64);
        self.sop2(op, self.pair(dst), self.pair(a), self.pair(b));
    }

    /// `dst = a & !b`.
    fn andn2(&mut self, dst: u8, a: u8, b: u8) {
        let op = self.mask_op(SOP2_ANDN2_B32, SOP2_ANDN2_B64);
        self.sop2(op, self.pair(dst), self.pair(a), self.pair(b));
    }

    fn set_exec(&mut self, slot: u8) {
        self.mov_mask(Operand::ExecLo, self.pair(slot));
    }

    /// Uniform mask: all lanes when descriptor bit `bit` of dword `word` is set.
    fn descriptor_flag(&mut self, dst: u8, word: u8, bit: i8) {
        self.sopc(
            SOPC_BITCMP1_B32,
            self.descriptor_word(word),
            Operand::InlineInt(bit),
        );
        let op = self.mask_op(SOP2_CSELECT_B32, SOP2_CSELECT_B64);
        self.sop2(op, self.pair(dst), ALL_ONES, ZERO);
    }

    fn vop1(&mut self, opcode: u16, vdst: Operand, src0: Operand) {
        let flags = Self::table_flags(F::Vop1, opcode);
        let fields = Fields::Vop1 {
            vdst,
            src0,
            ext: VopExt::None,
        };
        self.emit(F::Vop1, opcode, flags, fields);
    }

    fn vop2(&mut self, opcode: u16, vdst: Operand, src0: Operand, vsrc1: Operand) {
        let flags = Self::table_flags(F::Vop2, opcode);
        let fields = Fields::Vop2 {
            vdst,
            src0,
            vsrc1,
            constant: None,
            ext: VopExt::None,
        };
        self.emit(F::Vop2, opcode, flags, fields);
    }

    fn vop3(&mut self, opcode: u16, vdst: Operand, sdst: Option<Operand>, src: [Operand; 3]) {
        let flags = opcodes::vop3_flags(opcode).unwrap_or(0);
        let v = Vop3 {
            sdst,
            ..Vop3::new(vdst, src)
        };
        self.emit(F::Vop3, opcode, flags, Fields::Vop3(v));
    }

    fn nop(&mut self) {
        let fields = Fields::Vop1 {
            vdst: Operand::Vgpr(0),
            src0: Operand::Sgpr(0),
            ext: VopExt::None,
        };
        self.emit(F::Vop1, VOP1_NOP, flag::NO_OPERANDS, fields);
    }

    fn mov(&mut self, dst: Operand, src: Operand) {
        self.vop1(VOP1_MOV_B32, dst, src);
    }

    fn sub(&mut self, dst: Operand, a: Operand, b: Operand) {
        self.vop2(VOP2_SUB_F32, dst, a, b);
    }

    fn mul(&mut self, dst: Operand, a: Operand, b: Operand) {
        self.vop2(VOP2_MUL_F32, dst, a, b);
    }

    fn add(&mut self, dst: Operand, a: Operand, b: Operand) {
        self.vop2(VOP2_ADD_F32, dst, a, b);
    }

    /// Lane mask `dst` = compare(a, b) with a VOPC opcode.
    fn cmp(&mut self, opcode: u16, dst: u8, a: Operand, b: Operand) {
        self.vop3(opcode, self.pair(dst), None, [a, b, Operand::Raw(0)]);
    }

    /// `dst = mask ? yes : no`.
    fn select(&mut self, dst: Operand, mask: u8, yes: Operand, no: Operand) {
        self.vop3(VOP3_CNDMASK_B32, dst, None, [no, yes, self.pair(mask)]);
    }

    fn load4(&mut self, dword: usize) {
        let info = memops::flat(FLAT_LOAD_DWORDX4);
        let fields = Fields::Flat(Flat {
            segment: FlatSegment::Global,
            vdst: self.v + DATA + dword as u8,
            vdata: 0,
            vaddr: self.v + ADDR,
            saddr: Some(Operand::Null),
            uses_vaddr: true,
            offset: 4 * dword as i32,
            glc: false,
            dlc: false,
            slc: false,
            lds: false,
            kind: MemKind::Load,
            data_dwords: 4,
        });
        let flags = info.map_or(0, |i| i.flags);
        self.emit(F::Flat, FLAT_LOAD_DWORDX4, flags, fields);
    }

    // ---- FP mode ----------------------------------------------------------------------------

    fn enter_mode(&mut self, mode: ModeState) {
        match mode {
            ModeState::Static(current) => self.switch_mode(current, FLOAT_MODE),
            ModeState::Dynamic => {
                self.sopk(SOPK_GETREG, self.single(S_MODE), HWREG_MODE_FLOAT);
                self.sopp(SOPP_ROUND_MODE, u16::from(FLOAT_MODE & 0xf));
                self.sopp(SOPP_DENORM_MODE, u16::from(FLOAT_MODE >> 4));
            }
        }
    }

    fn leave_mode(&mut self, mode: ModeState) {
        match mode {
            ModeState::Static(current) => self.switch_mode(FLOAT_MODE, current),
            ModeState::Dynamic => self.sopk(SOPK_SETREG, self.single(S_MODE), HWREG_MODE_FLOAT),
        }
    }

    fn switch_mode(&mut self, from: u8, to: u8) {
        if from & 0xf != to & 0xf {
            self.sopp(SOPP_ROUND_MODE, u16::from(to & 0xf));
        }
        if from >> 4 != to >> 4 {
            self.sopp(SOPP_DENORM_MODE, u16::from(to >> 4));
        }
    }

    // ---- inputs ---------------------------------------------------------------------------

    /// Box sort and barycentric masks, the grow factor, and (in `P_A`/`P_B`) base and size.
    fn descriptor(&mut self) {
        let (base, size) = (P_A, P_B);
        self.sop1(SOP1_MOV_B32, self.pair(base), self.descriptor_word(0));
        self.sop2(
            SOP2_AND_B32,
            self.hi(base),
            self.descriptor_word(1),
            Operand::Literal(0xff),
        );
        self.sop2(
            SOP2_LSHL_B64,
            self.pair(base),
            self.pair(base),
            Operand::InlineInt(8),
        );
        self.sop1(SOP1_MOV_B32, self.pair(size), self.descriptor_word(2));
        self.sop2(
            SOP2_AND_B32,
            self.hi(size),
            self.descriptor_word(3),
            Operand::Literal(0x3ff),
        );
        self.descriptor_flag(P_SORT, 1, 31);
        self.descriptor_flag(P_IJ, 3, 24);
        let grow = self.pair(P_C);
        self.sop2(
            SOP2_BFE_U32,
            grow,
            self.descriptor_word(1),
            Operand::Literal((8 << 16) | 23),
        );
        let factor = self.r(FACTOR);
        self.vop1(VOP1_CVT_F32_U32, factor, grow);
        self.mul(factor, Operand::Literal(F32_GROW_STEP), factor);
        self.add(factor, F32_ONE_INLINE, factor);
    }

    /// A16 halves, the sanitised extent, node type, index and address, and the lane classes.
    fn ray_inputs(&mut self) {
        if self.m.a16 {
            for h in 0..6 {
                let word = self.address(self.first_ray_dword() + 4 + h / 2);
                let dst = if h < 3 {
                    self.r(DIR + h as u8)
                } else {
                    self.r(INV + h as u8 - 3)
                };
                if h % 2 == 1 {
                    self.vop2(VOP2_LSHRREV_B32, dst, Operand::InlineInt(16), word);
                    self.vop1(VOP1_CVT_F32_F16, dst, dst);
                } else {
                    self.vop1(VOP1_CVT_F32_F16, dst, word);
                }
            }
        }
        let extent = self.extent();
        self.cmp(CMP_U_F32, P_M, extent, extent);
        self.select(self.r(EXTENT), P_M, Operand::Literal(F32_INF), extent);

        let (ty, index, addr) = (self.r(TYPE), self.r(INDEX), self.r(ADDR));
        self.vop2(VOP2_AND_B32, ty, Operand::InlineInt(7), self.address(0));
        if self.wide {
            self.mov(index, self.address(0));
            self.mov(self.r(INDEX + 1), self.address(1));
            self.vop3(
                VOP3_LSHRREV_B64,
                index,
                None,
                [Operand::InlineInt(3), index, Operand::Raw(0)],
            );
        } else {
            self.vop2(
                VOP2_LSHRREV_B32,
                index,
                Operand::InlineInt(3),
                self.address(0),
            );
            self.mov(self.r(INDEX + 1), ZERO);
        }
        self.vop3(
            VOP3_LSHLREV_B64,
            addr,
            None,
            [Operand::InlineInt(6), index, Operand::Raw(0)],
        );
        let (base, size) = (P_A, P_B);
        self.vop3(
            VOP3_ADD_CO_U32,
            addr,
            Some(self.pair(P_M)),
            [self.pair(base), addr, Operand::Raw(0)],
        );
        self.vop3(
            VOP3_ADD_CO_CI_U32,
            self.r(ADDR + 1),
            Some(self.pair(P_N)),
            [self.hi(base), self.r(ADDR + 1), self.pair(P_M)],
        );

        // Valid: types 0-4 with index <= size, type 5 with index < size.
        self.cmp(CMP_LE_U64, P_M, index, self.pair(size));
        self.cmp(CMP_LT_U64, P_N, index, self.pair(size));
        self.cmp(CMP_EQ_U32, P_O, ty, Operand::InlineInt(5));
        self.andn2(P_VALID, P_M, P_O);
        self.cmp(CMP_GE_U32, P_M, ty, Operand::InlineInt(6));
        self.andn2(P_VALID, P_VALID, P_M);
        self.and(P_N, P_N, P_O);
        self.or(P_VALID, P_VALID, P_N);
        self.cmp(CMP_LE_U32, P_TRI, ty, Operand::InlineInt(3));
    }

    /// Masked node loads; box16 lanes then widen their bounds into the box32 layout.
    fn fetch(&mut self) {
        let box32 = P_N;
        self.set_exec(P_VALID);
        for dword in (0..16).step_by(4) {
            self.load4(dword);
        }
        // A VALU between VMEM and an SALU EXEC write (gfx10 VMEMtoScalarWriteHazard).
        self.nop();
        self.set_exec(box32);
        for dword in (16..BOX32_DWORDS).step_by(4) {
            self.load4(dword);
        }
        self.sopp(SOPP_WAITCNT, waitcnt::encode_gfx10(0, 7, 63));
        self.nop();
        self.set_exec(P_EXEC);

        self.cmp(CMP_EQ_U32, P_M, self.r(TYPE), Operand::InlineInt(4));
        self.set_exec(P_M);
        let shifted = self.r(T);
        for child in (0..4).rev() {
            for source in (0..3).rev() {
                let word = self.data(4 + 3 * child + source);
                let lo = self.data(4 + 6 * child + 2 * source);
                let hi = self.data(4 + 6 * child + 2 * source + 1);
                self.vop2(VOP2_LSHRREV_B32, shifted, Operand::InlineInt(16), word);
                self.vop1(VOP1_CVT_F32_F16, hi, shifted);
                self.vop1(VOP1_CVT_F32_F16, lo, word);
            }
        }
        self.set_exec(P_EXEC);
    }

    // ---- box test -------------------------------------------------------------------------

    /// Returns the registers holding the four (sorted) child results.
    fn boxes(&mut self) -> [u8; 4] {
        let positive = [P_A, P_B, P_C];
        for (k, &mask) in positive.iter().enumerate() {
            self.cmp(CMP_GE_F32, mask, self.inv(k), ZERO);
        }
        let near = |k: usize| T + k as u8;
        let far = |k: usize| T + 3 + k as u8;
        let (lo, hi, entry, exit) = (T + 6, T + 7, T + 8, T + 9);
        let (m, all_near, all_far) = (P_M, P_N, P_O);
        for c in 0..4 {
            for (k, &pos) in positive.iter().enumerate() {
                let (min, max) = (self.data(4 + 6 * c + k), self.data(4 + 6 * c + 3 + k));
                self.sub(self.r(lo), min, self.origin(k));
                self.mul(self.r(lo), self.r(lo), self.inv(k));
                self.sub(self.r(hi), max, self.origin(k));
                self.mul(self.r(hi), self.r(hi), self.inv(k));
                self.select(self.r(near(k)), pos, self.r(lo), self.r(hi));
                self.select(self.r(far(k)), pos, self.r(hi), self.r(lo));
            }
            for k in 0..3 {
                for (reg, all, fill) in
                    [(near(k), all_near, F32_NEG_INF), (far(k), all_far, F32_INF)]
                {
                    let t = self.r(reg);
                    self.cmp(CMP_U_F32, m, t, t);
                    self.select(t, m, Operand::Literal(fill), t);
                    if k == 0 {
                        self.mov_mask(self.pair(all), self.pair(m));
                    } else {
                        self.and(all, all, m);
                    }
                }
            }
            self.cmp(CMP_GT_F32, m, self.r(near(0)), self.r(near(1)));
            self.select(self.r(entry), m, self.r(near(0)), self.r(near(1)));
            self.cmp(CMP_GT_F32, m, self.r(entry), self.r(near(2)));
            self.select(self.r(entry), m, self.r(entry), self.r(near(2)));
            self.cmp(CMP_LT_F32, m, self.r(far(0)), self.r(far(1)));
            self.select(self.r(exit), m, self.r(far(0)), self.r(far(1)));
            self.cmp(CMP_LT_F32, m, self.r(exit), self.r(far(2)));
            self.select(self.r(exit), m, self.r(exit), self.r(far(2)));

            let key = self.r(KEY + c as u8);
            self.cmp(CMP_GT_F32, m, self.r(entry), ZERO);
            self.select(key, m, self.r(entry), ZERO);
            self.cmp(CMP_LT_F32, m, self.r(exit), self.r(EXTENT));
            self.select(self.r(exit), m, self.r(exit), self.r(EXTENT));
            self.mul(self.r(exit), self.r(exit), self.r(FACTOR));
            self.cmp(CMP_LE_F32, m, key, self.r(exit));
            self.or(all_near, all_near, all_far);
            self.andn2(m, m, all_near);
            self.select(self.r(CHILD + c as u8), m, self.data(c), ALL_ONES);
        }

        let mut child: [u8; 4] = std::array::from_fn(|c| CHILD + c as u8);
        let mut key: [u8; 4] = std::array::from_fn(|c| KEY + c as u8);
        let (mut spare_child, mut spare_key) = (SPARE, SPARE + 1);
        let (swap, other) = (P_M, P_N);
        for (a, b) in SORT_NETWORK {
            self.cmp(CMP_NE_U32, swap, self.r(child[b]), ALL_ONES);
            self.cmp(CMP_LT_F32, other, self.r(key[b]), self.r(key[a]));
            self.and(swap, swap, other);
            self.cmp(CMP_EQ_U32, other, self.r(child[a]), ALL_ONES);
            self.or(swap, swap, other);
            self.and(swap, swap, P_SORT);
            for (regs, spare) in [(&mut child, &mut spare_child), (&mut key, &mut spare_key)] {
                let (ra, rb) = (self.r(regs[a]), self.r(regs[b]));
                self.select(self.r(*spare), swap, rb, ra);
                self.select(rb, swap, ra, rb);
                std::mem::swap(&mut regs[a], spare);
            }
        }
        child
    }

    // ---- triangle test --------------------------------------------------------------------

    /// Returns the registers holding the four triangle results.
    fn triangle(&mut self) -> [u8; 4] {
        let vertex = |v: usize, k: usize| T + 3 * v as u8 + k as u8;
        let mag = |k: usize| T + 9 + k as u8;
        let (dkx, dky, dkz, sz, sx, sy) = (T + 12, T + 13, T + 14, T + 15, T + 16, T + 17);
        let perm = |v: usize, axis: usize| T + 18 + 3 * v as u8 + axis as u8;
        let sheared = |v: usize, axis: usize| T + 27 + 2 * v as u8 + axis as u8;
        let depth = |v: usize| T + 33 + v as u8;
        let (u, vw, w, det, t) = (T + 36, T + 37, T + 38, T + 39, T + 40);
        let (control, isrc, jsrc, bi, bj) = (T + 41, T + 42, T + 43, T + 44, T + 45);
        let result = [T + 46, T + 47, T + 48, T + 49];
        let (x0, x1) = (T + 50, T + 51);

        // Vertex slots by type bits, relative to the origin.
        let (bit0, bit1) = (P_A, P_B);
        let ty = self.r(TYPE);
        self.vop2(VOP2_AND_B32, self.r(x0), Operand::InlineInt(1), ty);
        self.cmp(CMP_NE_U32, bit0, self.r(x0), ZERO);
        self.vop2(VOP2_AND_B32, self.r(x0), Operand::InlineInt(2), ty);
        self.cmp(CMP_NE_U32, bit1, self.r(x0), ZERO);
        let slots_by_vertex: [[usize; 4]; 3] =
            std::array::from_fn(|v| TRIANGLE_VERTICES.map(|slots| slots[v]));
        for (v, slots) in slots_by_vertex.into_iter().enumerate() {
            for k in 0..3 {
                let coord: [Operand; 5] = std::array::from_fn(|slot| self.data(3 * slot + k));
                let dst = self.r(vertex(v, k));
                let lo = if slots[0] == slots[1] {
                    coord[slots[0]]
                } else {
                    self.select(self.r(x0), bit0, coord[slots[1]], coord[slots[0]]);
                    self.r(x0)
                };
                let hi = if slots[2] == slots[3] {
                    coord[slots[2]]
                } else {
                    self.select(dst, bit0, coord[slots[3]], coord[slots[2]]);
                    dst
                };
                self.select(dst, bit1, hi, lo);
                self.sub(dst, dst, self.origin(k));
            }
        }

        // kz = argmax |dir| (first on ties), kx/ky follow it and swap for a negative dir[kz].
        let (kz0, kz1, swapped) = (P_A, P_B, P_C);
        for k in 0..3 {
            self.vop2(
                VOP2_AND_B32,
                self.r(mag(k)),
                Operand::Literal(0x7fff_ffff),
                self.dir(k),
            );
        }
        self.cmp(CMP_GE_F32, P_M, self.r(mag(0)), self.r(mag(1)));
        self.cmp(CMP_GE_F32, P_N, self.r(mag(0)), self.r(mag(2)));
        self.cmp(CMP_GE_F32, P_O, self.r(mag(1)), self.r(mag(2)));
        self.and(kz0, P_M, P_N);
        self.andn2(kz1, P_O, P_M);
        let d = [self.dir(0), self.dir(1), self.dir(2)];
        self.permute(d, [dkx, dky, dkz], kz0, kz1);
        self.cmp(CMP_LT_F32, swapped, self.r(dkz), ZERO);
        self.swap_xy(dkx, dky, x0, swapped);
        let inv = [self.inv(0), self.inv(1), self.inv(2)];
        self.select(self.r(sz), kz1, inv[1], inv[2]);
        self.select(self.r(sz), kz0, inv[0], self.r(sz));
        self.mul(self.r(sx), self.r(dkx), self.r(sz));
        self.mul(self.r(sy), self.r(dky), self.r(sz));
        for v in 0..3 {
            let p = std::array::from_fn(|k| self.r(vertex(v, k)));
            let out = [perm(v, 0), perm(v, 1), perm(v, 2)];
            self.permute(p, out, kz0, kz1);
            self.swap_xy(out[0], out[1], x0, swapped);
            for (axis, s) in [(0, sx), (1, sy)] {
                self.mul(self.r(x0), self.r(s), self.r(out[2]));
                self.sub(self.r(sheared(v, axis)), self.r(out[axis]), self.r(x0));
            }
            self.mul(self.r(depth(v)), self.r(sz), self.r(out[2]));
        }

        // Edge functions u (vertices b, c), v (c, a), w (a, b).
        for (dst, p, q) in [(u, 2, 1), (vw, 0, 2), (w, 1, 0)] {
            self.mul(self.r(x0), self.r(sheared(p, 0)), self.r(sheared(q, 1)));
            self.mul(self.r(x1), self.r(sheared(p, 1)), self.r(sheared(q, 0)));
            self.sub(self.r(dst), self.r(x0), self.r(x1));
        }
        self.add(self.r(det), self.r(u), self.r(vw));
        self.add(self.r(det), self.r(det), self.r(w));
        self.mul(self.r(t), self.r(u), self.r(depth(0)));
        self.mul(self.r(x0), self.r(vw), self.r(depth(1)));
        self.add(self.r(t), self.r(t), self.r(x0));
        self.mul(self.r(x0), self.r(w), self.r(depth(2)));
        self.add(self.r(t), self.r(t), self.r(x0));

        let (mixed, hit) = (P_A, P_B);
        for (dst, opcode) in [(P_M, CMP_LT_F32), (P_N, CMP_GT_F32)] {
            self.cmp(opcode, dst, self.r(u), ZERO);
            for e in [vw, w] {
                self.cmp(opcode, P_O, self.r(e), ZERO);
                self.or(dst, dst, P_O);
            }
        }
        self.and(mixed, P_M, P_N);
        self.cmp(CMP_LG_F32, hit, self.r(det), ZERO);
        self.andn2(hit, hit, mixed);
        self.cmp(CMP_U_F32, P_O, self.r(t), self.r(t));
        self.andn2(hit, hit, P_O);
        for (det_cmp, t_cmp) in [(CMP_GT_F32, CMP_LT_F32), (CMP_LT_F32, CMP_GT_F32)] {
            self.cmp(det_cmp, P_O, self.r(det), ZERO);
            self.cmp(t_cmp, P_M, self.r(t), ZERO);
            self.and(P_O, P_O, P_M);
            self.andn2(hit, hit, P_O);
        }
        self.select(self.r(result[0]), hit, self.r(t), Operand::Literal(F32_INF));
        self.select(self.r(result[1]), hit, self.r(det), F32_ONE_INLINE);

        // Barycentric numerators chosen by the triangle's I/J source fields.
        let id = self.data(TRIANGLE_ID_DWORD);
        self.vop2(VOP2_LSHLREV_B32, self.r(x0), Operand::InlineInt(3), ty);
        self.vop2(VOP2_LSHRREV_B32, self.r(control), self.r(x0), id);
        self.vop2(
            VOP2_AND_B32,
            self.r(isrc),
            Operand::InlineInt(3),
            self.r(control),
        );
        self.vop3(
            VOP3_BFE_U32,
            self.r(jsrc),
            None,
            [
                self.r(control),
                Operand::InlineInt(2),
                Operand::InlineInt(2),
            ],
        );
        for (dst, source) in [(bi, isrc), (bj, jsrc)] {
            let out = self.r(dst);
            self.cmp(CMP_EQ_U32, P_M, self.r(source), Operand::InlineInt(2));
            self.select(out, P_M, self.r(w), self.r(u));
            self.cmp(CMP_EQ_U32, P_M, self.r(source), Operand::InlineInt(1));
            self.select(out, P_M, self.r(vw), out);
            self.cmp(CMP_U_F32, P_M, out, out);
            self.select(out, P_M, Operand::Literal(F32_CANONICAL_NAN), out);
        }
        self.select(self.r(x0), hit, Operand::InlineInt(1), ZERO);
        self.select(self.r(result[2]), P_IJ, self.r(bi), id);
        self.select(self.r(result[3]), P_IJ, self.r(bj), self.r(x0));
        result
    }

    /// `out = (dir[kx], dir[ky], dir[kz])` for kz selected by `kz0`/`kz1` (else 2), before the
    /// sign swap of kx and ky.
    fn permute(&mut self, src: [Operand; 3], out: [u8; 3], kz0: u8, kz1: u8) {
        for (i, &dst) in out.iter().enumerate() {
            let pick = |kz: usize| src[(kz + 1 + i) % 3];
            let dst = self.r(dst);
            self.select(dst, kz1, pick(1), pick(2));
            self.select(dst, kz0, pick(0), dst);
        }
    }

    fn swap_xy(&mut self, x: u8, y: u8, scratch: u8, swapped: u8) {
        let (x, y, t) = (self.r(x), self.r(y), self.r(scratch));
        self.select(t, swapped, y, x);
        self.select(y, swapped, x, y);
        self.mov(x, t);
    }

    // ---- results ----------------------------------------------------------------------------

    fn write_back(&mut self, child: [u8; 4], triangle: [u8; 4]) {
        for k in 0..4 {
            let (c, t) = (self.r(child[k]), self.r(triangle[k]));
            self.select(c, P_TRI, t, c);
            let vdata = Operand::Vgpr(self.m.vdata + k as u8);
            self.select(vdata, P_VALID, c, ALL_ONES);
        }
    }
}
