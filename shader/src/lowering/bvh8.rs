//! PS5 Pro `image_bvh8_intersect_ray` (MIMG 0xe5) expanded in software for gfx1013 hosts.
//!
//! The expansion is branch-free and runs on 86 scratch VGPRs and 24 scratch SGPRs.
//!
//! Entry and exit:
//! - SCC, EXEC and MODE are saved on entry and restored on exit.
//! - All eleven logical inputs are snapshotted before anything is written.
//!
//! Lane classes (computed under the original EXEC):
//! - early-cull/user lanes: never touch memory and return all INVALID.
//! - internal (box) lanes, triangle lanes and instance lanes.
//!
//! Each class is processed under its own EXEC mask, so lanes of different classes can reuse the
//! same scratch registers without clobbering each other. Big box nodes are processed as two
//! sequential four-child passes. Triangle-pair nodes run the exact reference emulation once per
//! triangle, refetching the first 64 bytes for the second triangle.
//!
//! Final writes, in order:
//! 1. Transformed origin/direction, under the instance EXEC.
//! 2. All ten VDATA registers, under the original EXEC.
//!
//! Node layouts and arithmetic follow the SDK 10 Pro reference. Descriptor sorting and
//! maximum-plane growth extend the reference using the documented AMD descriptor controls.

use super::bvh::{FLOAT_MODE, ModeState};
use super::{Ctx, derive, unsupported};
use crate::CompileError;
use crate::ir::memops;
use crate::ir::opcodes::{self, flag};
use crate::ir::{
    Fields, Flat, FlatSegment, Image, Instruction, InstructionFamily as F, MemKind, Operand, Vop3,
    VopExt,
};
use crate::lowering::waitcnt;

pub const SCRATCH_VGPRS: u8 = 86;
pub const SCRATCH_SGPRS: u8 = 24;

// ---- node kinds ------------------------------------------------------------------------------
const NODE_TRI_1: u32 = 1;
// SDK Pro internal node kinds.
const NODE_TRINITY_BIG_FP32: u32 = 2;
const NODE_TRINITY_BIG_SHARED_EXPONENT: u32 = 3;
const NODE_TRINITY_SHARED_EXPONENT: u32 = 4;
const NODE_FP32: u32 = 5;
const NODE_TRINITY_INSTANCE: u32 = 6;
const NODE_USER: u32 = 7;

const INVALID: u32 = 0xffff_ffff;
const F32_INF: u32 = 0x7f80_0000;
const F32_ONE: u32 = 0x3f80_0000;
const F32_HALF: u32 = 0x3f00_0000;
const F32_GROW_STEP: u32 = 0x3380_0000;
const SIGN: u32 = 0x8000_0000;

/// SDK 8-wide sorting network; the first ten pairs sort each group of four.
const SORT_NETWORK: [(usize, usize); 19] = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (1, 2),
    (4, 5),
    (6, 7),
    (4, 6),
    (5, 7),
    (5, 6),
    (0, 4),
    (1, 5),
    (1, 4),
    (2, 6),
    (3, 7),
    (3, 6),
    (2, 4),
    (3, 5),
    (3, 4),
];

// ---- hardware encodings ----------------------------------------------------------------------
const HWREG_MODE: u16 = 1;
const HWREG_MODE_FLOAT: u16 = HWREG_MODE | (7 << 11);
const SOPP_WAITCNT: u16 = 0x0c;
const SOPP_ROUND_MODE: u16 = 0x24;
const SOPP_DENORM_MODE: u16 = 0x25;
const SOPK_GETREG: u16 = 0x12;
const SOPK_SETREG: u16 = 0x13;

const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP2_CSELECT_B32: u16 = 0x0a;
const SOP2_CSELECT_B64: u16 = 0x0b;
const SOP2_AND_B32: u16 = 0x0e;
const SOP2_AND_B64: u16 = 0x0f;
const SOP2_OR_B32: u16 = 0x10;
const SOP2_OR_B64: u16 = 0x11;
const SOP2_XOR_B32: u16 = 0x12;
const SOP2_XOR_B64: u16 = 0x13;
const SOP2_ANDN2_B32: u16 = 0x14;
const SOP2_ANDN2_B64: u16 = 0x15;
const SOP2_BFE_U32: u16 = 0x27;
const SOPC_CMP_EQ_U32: u16 = 0x06;
const SOPC_CMP_LG_U32: u16 = 0x07;
const SOPC_BITCMP1_B32: u16 = 0x0d;

const VOP1_NOP: u16 = 0x00;
const VOP1_MOV_B32: u16 = 0x01;
const VOP1_CVT_F32_U32: u16 = 0x06;
const VOP1_RCP_F32: u16 = 0x2a;
const VOP2_ADD_F32: u16 = 0x03;
const VOP2_SUB_F32: u16 = 0x04;
const VOP2_MUL_F32: u16 = 0x08;
const VOP2_MIN_U32: u16 = 0x13;
const VOP2_LSHRREV_B32: u16 = 0x16;
const VOP2_LSHLREV_B32: u16 = 0x1a;
const VOP2_AND_B32: u16 = 0x1b;
const VOP2_OR_B32: u16 = 0x1c;
const VOP2_XOR_B32: u16 = 0x1d;
const VOP2_ADD_NC_U32: u16 = 0x25;
const VOP2_SUB_NC_U32: u16 = 0x26;
const VOP3_CNDMASK_B32: u16 = 0x101;
const VOP3_LSHLREV_B32: u16 = 0x11a;
const VOP3_ADD_CO_CI_U32: u16 = 0x128;
const VOP3_BFE_U32: u16 = 0x148;
const VOP3_LSHLREV_B64: u16 = 0x2ff;
const VOP3_ADD_CO_U32: u16 = 0x30f;
const FLAT_LOAD_DWORDX4: u16 = 0x0e;

const CMP_LT_F32: u16 = 0x01;
const CMP_EQ_F32: u16 = 0x02;
const CMP_GT_F32: u16 = 0x04;
const CMP_GE_F32: u16 = 0x06;
const CMP_U_F32: u16 = 0x08;
const CMP_EQ_U32: u16 = 0xc2;
const CMP_LE_U32: u16 = 0xc3;
const CMP_GT_U32: u16 = 0xc4;
const CMP_NE_U32: u16 = 0xc5;
const CMP_GE_U32: u16 = 0xc6;

// ---- scratch VGPRs (relative) ----------------------------------------------------------------
const IN_ENC: u8 = 0; // 0,1
const IN_EXTENT: u8 = 2;
const IN_MASK: u8 = 3;
const IN_ORG: u8 = 4; // 4..6
const IN_DIR: u8 = 7; // 7..9
const IN_INDEX: u8 = 10;
const PFLAGS: u8 = 11;
const KIND: u8 = 12;
const TRI_FLAG: u8 = 13;
const ADDR: u8 = 14; // 14,15
const OUT: u8 = 16; // 16..25
const KEY: u8 = 26; // 26..33
const DATA: u8 = 34; // 34..65
const WORK: u8 = 66; // 66..85

// Box work.
const BMIN: u8 = WORK; // 66..68 (later near planes)
const BMAX: u8 = WORK + 3; // 69..71 (later far planes)
const BIDX: u8 = WORK + 6; // 72
const BFLAGS: u8 = WORK + 7; // 73
const BT: u8 = WORK + 8; // 74..81
const RDIR: u8 = WORK + 16; // 82..84
const FACTOR: u8 = WORK + 19; // 85

// Triangle temps (26..77, valid only under triangle EXEC).
const TREL: u8 = 26; // 26..34
const TP: u8 = 35; // 35..43
const TD: u8 = 44; // 44..46
const AX: u8 = 47;
const AY: u8 = 48;
const BX: u8 = 49;
const BY: u8 = 50;
const CX: u8 = 51;
const CY: u8 = 52;
const TU: u8 = 53;
const TV: u8 = 54;
const TW: u8 = 55;
const TUZ: u8 = 56;
const TVZ: u8 = 57;
const TWZ: u8 = 58;
const TNUM: u8 = 59;
const TDEN: u8 = 60;
const RZ: u8 = 61;
const RW: u8 = 62;
const TI: u8 = 63;
const TJ: u8 = 64;
const PROC: u8 = 67;
const DSIDE: u8 = 68;
const OPQ: u8 = 69;
const FLIP: u8 = 70;
const FRONT: u8 = 72;
const S0: u8 = 73;
const S1: u8 = 74;
const S2: u8 = 75;

// ---- scratch SGPR pairs ----------------------------------------------------------------------
const P_EXEC: u8 = 0;
const P_VALID: u8 = 1;
const P_INST: u8 = 2;
const P_TRI: u8 = 3;
const P_A: u8 = 4;
const P_B: u8 = 5;
const P_C: u8 = 6;
const P_D: u8 = 7;
const P_M: u8 = 8;
const P_N: u8 = 9;
const P_O: u8 = 10;
const S_SCC: u8 = 22;
const S_MODE: u8 = 23;

fn encoding_error(m: &Image) -> Option<&'static str> {
    if m.dmask != 0xf || !m.r128 || m.d16 || m.tfe || m.lwe || m.a16 {
        return Some("BVH8 intersection requires DMASK=0xF, R128=1 and no A16, D16, TFE or LWE");
    }
    None
}

pub(super) fn lower(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), CompileError> {
    let Fields::Mimg(m) = inst.fields else {
        return Err(unsupported(inst, ctx.target, "malformed MIMG"));
    };
    if let Some(reason) = encoding_error(&m) {
        return Err(unsupported(inst, ctx.target, reason));
    }
    let v = ctx.regs.scratch_vgprs(SCRATCH_VGPRS)?;
    let s = ctx.transient_sgprs(inst, SCRATCH_SGPRS)?;
    let mut g = Gen {
        origin: inst,
        m,
        wave64: ctx.wave64,
        v,
        s,
        out: Vec::new(),
    };
    g.expand(ctx.mode);
    for expanded in std::mem::take(&mut g.out) {
        ctx.lower(&expanded)?;
    }
    Ok(())
}

fn k(x: u32) -> Operand {
    Operand::constant(x)
}

struct Gen<'a> {
    origin: &'a Instruction,
    m: Image,
    wave64: bool,
    v: u8,
    s: u8,
    out: Vec<Instruction>,
}

impl Gen<'_> {
    // ---- operands ----------------------------------------------------------------------------

    fn r(&self, o: u8) -> Operand {
        Operand::Vgpr(self.v + o)
    }

    fn pair(&self, slot: u8) -> Operand {
        Operand::Sgpr(self.s + 2 * slot)
    }

    fn single(&self, o: u8) -> Operand {
        Operand::Sgpr(self.s + o)
    }

    fn guest(&self, i: usize) -> Operand {
        Operand::Vgpr(self.m.address_register(i))
    }

    fn desc(&self, word: u8) -> Operand {
        Operand::Sgpr(self.m.srsrc + word)
    }

    // ---- builders ----------------------------------------------------------------------------

    fn emit(&mut self, family: F, opcode: u16, flags: u64, fields: Fields) {
        self.out
            .push(derive(self.origin, family, opcode, flags, fields));
    }

    fn tflags(family: F, opcode: u16) -> u64 {
        opcodes::lookup(family, opcode).map_or(0, |i| i.flags)
    }

    fn sop1(&mut self, op: u16, sdst: Operand, src0: Operand) {
        let f = Self::tflags(F::Sop1, op);
        self.emit(F::Sop1, op, f, Fields::Sop1 { sdst, src0 });
    }

    fn sop2(&mut self, op: u16, sdst: Operand, src0: Operand, src1: Operand) {
        let f = Self::tflags(F::Sop2, op);
        self.emit(F::Sop2, op, f, Fields::Sop2 { sdst, src0, src1 });
    }

    fn sopc(&mut self, op: u16, src0: Operand, src1: Operand) {
        let f = Self::tflags(F::Sopc, op);
        self.emit(F::Sopc, op, f, Fields::Sopc { src0, src1 });
    }

    fn sopp(&mut self, op: u16, simm16: u16) {
        let f = Self::tflags(F::Sopp, op);
        self.emit(F::Sopp, op, f, Fields::Sopp { simm16 });
    }

    fn sopk(&mut self, op: u16, sdst: Operand, simm16: u16) {
        let f = Self::tflags(F::Sopk, op);
        let fields = Fields::Sopk {
            sdst,
            simm16,
            imm32: None,
        };
        self.emit(F::Sopk, op, f, fields);
    }

    fn mop(&self, b32: u16, b64: u16) -> u16 {
        if self.wave64 { b64 } else { b32 }
    }

    fn mov_mask(&mut self, dst: Operand, src: Operand) {
        self.sop1(self.mop(SOP1_MOV_B32, SOP1_MOV_B64), dst, src);
    }

    fn and(&mut self, d: u8, a: u8, b: u8) {
        let op = self.mop(SOP2_AND_B32, SOP2_AND_B64);
        self.sop2(op, self.pair(d), self.pair(a), self.pair(b));
    }

    fn or(&mut self, d: u8, a: u8, b: u8) {
        let op = self.mop(SOP2_OR_B32, SOP2_OR_B64);
        self.sop2(op, self.pair(d), self.pair(a), self.pair(b));
    }

    fn xor(&mut self, d: u8, a: u8, b: u8) {
        let op = self.mop(SOP2_XOR_B32, SOP2_XOR_B64);
        self.sop2(op, self.pair(d), self.pair(a), self.pair(b));
    }

    /// `d = a & !b`.
    fn andn2(&mut self, d: u8, a: u8, b: u8) {
        let op = self.mop(SOP2_ANDN2_B32, SOP2_ANDN2_B64);
        self.sop2(op, self.pair(d), self.pair(a), self.pair(b));
    }

    fn set_exec(&mut self, slot: u8) {
        self.sop2(
            self.mop(SOP2_AND_B32, SOP2_AND_B64),
            Operand::ExecLo,
            self.pair(P_EXEC),
            self.pair(slot),
        );
    }

    /// Uniform all-lanes mask when descriptor bit `bit` of dword `word` is set (clobbers SCC).
    fn desc_flag(&mut self, dst: u8, word: u8, bit: i8) {
        self.sopc(SOPC_BITCMP1_B32, self.desc(word), Operand::InlineInt(bit));
        let op = self.mop(SOP2_CSELECT_B32, SOP2_CSELECT_B64);
        self.sop2(
            op,
            self.pair(dst),
            Operand::InlineInt(-1),
            Operand::InlineInt(0),
        );
    }

    /// Uniform mask: descriptor box sorting heuristic == `value`. Uses the low SGPR of `tmp`.
    fn heuristic_flag(&mut self, dst: u8, tmp: u8, value: u32) {
        self.sop2(
            SOP2_BFE_U32,
            self.pair(tmp),
            self.desc(1),
            k((2 << 16) | 21),
        );
        self.sopc(SOPC_CMP_EQ_U32, self.pair(tmp), k(value));
        let op = self.mop(SOP2_CSELECT_B32, SOP2_CSELECT_B64);
        self.sop2(
            op,
            self.pair(dst),
            Operand::InlineInt(-1),
            Operand::InlineInt(0),
        );
    }

    fn vop1(&mut self, op: u16, vdst: Operand, src0: Operand) {
        let f = Self::tflags(F::Vop1, op);
        let fields = Fields::Vop1 {
            vdst,
            src0,
            ext: VopExt::None,
        };
        self.emit(F::Vop1, op, f, fields);
    }

    /// `vsrc1` must be a VGPR.
    fn vop2(&mut self, op: u16, vdst: Operand, src0: Operand, vsrc1: Operand) {
        let f = Self::tflags(F::Vop2, op);
        let fields = Fields::Vop2 {
            vdst,
            src0,
            vsrc1,
            constant: None,
            ext: VopExt::None,
        };
        self.emit(F::Vop2, op, f, fields);
    }

    fn vop3(&mut self, op: u16, vdst: Operand, sdst: Option<Operand>, src: [Operand; 3]) {
        let f = opcodes::vop3_flags(op).unwrap_or(0);
        let v = Vop3 {
            sdst,
            ..Vop3::new(vdst, src)
        };
        self.emit(F::Vop3, op, f, Fields::Vop3(v));
    }

    fn nop(&mut self) {
        let fields = Fields::Vop1 {
            vdst: Operand::Vgpr(0),
            src0: Operand::Sgpr(0),
            ext: VopExt::None,
        };
        self.emit(F::Vop1, VOP1_NOP, flag::NO_OPERANDS, fields);
    }

    fn mov(&mut self, d: Operand, s: Operand) {
        self.vop1(VOP1_MOV_B32, d, s);
    }

    fn fadd(&mut self, d: u8, a: u8, b: u8) {
        self.vop2(VOP2_ADD_F32, self.r(d), self.r(a), self.r(b));
    }

    fn fsub(&mut self, d: u8, a: u8, b: u8) {
        self.vop2(VOP2_SUB_F32, self.r(d), self.r(a), self.r(b));
    }

    fn fmul(&mut self, d: u8, a: u8, b: u8) {
        self.vop2(VOP2_MUL_F32, self.r(d), self.r(a), self.r(b));
    }

    /// Integer op with a constant or VGPR `a` and VGPR `b`.
    fn iop(&mut self, op: u16, d: u8, a: Operand, b: u8) {
        self.vop2(op, self.r(d), a, self.r(b));
    }

    fn iand(&mut self, d: u8, a: Operand, b: u8) {
        self.iop(VOP2_AND_B32, d, a, b);
    }

    fn ior(&mut self, d: u8, a: Operand, b: u8) {
        self.iop(VOP2_OR_B32, d, a, b);
    }

    fn ixor(&mut self, d: u8, a: Operand, b: u8) {
        self.iop(VOP2_XOR_B32, d, a, b);
    }

    fn shr(&mut self, d: u8, amount: Operand, b: u8) {
        self.iop(VOP2_LSHRREV_B32, d, amount, b);
    }

    fn shl(&mut self, d: u8, amount: Operand, b: u8) {
        self.iop(VOP2_LSHLREV_B32, d, amount, b);
    }

    fn bfe(&mut self, d: u8, src: Operand, offset: u32, width: u32) {
        self.vop3(VOP3_BFE_U32, self.r(d), None, [src, k(offset), k(width)]);
    }

    fn cmp(&mut self, op: u16, dst: u8, a: Operand, b: Operand) {
        self.vop3(op, self.pair(dst), None, [a, b, Operand::Raw(0)]);
    }

    /// `d = mask ? yes : no`.
    fn select(&mut self, d: Operand, mask: u8, yes: Operand, no: Operand) {
        self.vop3(VOP3_CNDMASK_B32, d, None, [no, yes, self.pair(mask)]);
    }

    fn sel(&mut self, d: u8, mask: u8, yes: Operand, no: Operand) {
        self.select(self.r(d), mask, yes, no);
    }

    /// `dwords` dwords from the node address plus `byte_offset` into relative VGPR `vdst`.
    fn load(&mut self, vdst: u8, dwords: usize, byte_offset: i32) {
        let info = memops::flat(FLAT_LOAD_DWORDX4);
        let flags = info.map_or(0, |i| i.flags);
        for q in (0..dwords).step_by(4) {
            let fields = Fields::Flat(Flat {
                segment: FlatSegment::Global,
                vdst: self.v + vdst + q as u8,
                vdata: 0,
                vaddr: self.v + ADDR,
                saddr: Some(Operand::Null),
                uses_vaddr: true,
                offset: byte_offset + 4 * q as i32,
                glc: false,
                dlc: false,
                slc: false,
                lds: false,
                kind: MemKind::Load,
                data_dwords: 4,
            });
            self.emit(F::Flat, FLAT_LOAD_DWORDX4, flags, fields);
        }
    }

    /// Waits for all loads, then a VALU before any later SALU EXEC write.
    fn wait_loads(&mut self) {
        self.nop();
        self.sopp(SOPP_WAITCNT, waitcnt::encode_gfx10(0, 7, 63));
        self.nop();
    }

    /// `dst = (kind == any of kinds)`, evaluated under the current EXEC; clobbers `tmp`.
    fn kind_is(&mut self, dst: u8, tmp: u8, kinds: &[u32]) {
        for (i, &kv) in kinds.iter().enumerate() {
            let target = if i == 0 { dst } else { tmp };
            self.cmp(CMP_EQ_U32, target, self.r(KIND), k(kv));
            if i != 0 {
                self.or(dst, dst, tmp);
            }
        }
    }

    // ---- FP mode -----------------------------------------------------------------------------

    fn enter_mode(&mut self, mode: ModeState) {
        match mode {
            ModeState::Static(cur) => self.switch_mode(cur, FLOAT_MODE),
            ModeState::Dynamic => {
                self.sopk(SOPK_GETREG, self.single(S_MODE), HWREG_MODE_FLOAT);
                self.sopp(SOPP_ROUND_MODE, u16::from(FLOAT_MODE & 0xf));
                self.sopp(SOPP_DENORM_MODE, u16::from(FLOAT_MODE >> 4));
            }
        }
    }

    fn leave_mode(&mut self, mode: ModeState) {
        match mode {
            ModeState::Static(cur) => self.switch_mode(FLOAT_MODE, cur),
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

    // ---- top level ---------------------------------------------------------------------------

    fn expand(&mut self, mode: ModeState) {
        self.enter_mode(mode);
        self.sop2(
            SOP2_CSELECT_B32,
            self.single(S_SCC),
            Operand::InlineInt(1),
            Operand::InlineInt(0),
        );
        self.mov_mask(self.pair(P_EXEC), Operand::ExecLo);
        self.prologue();
        self.classify();
        self.boxes();
        self.set_exec(P_EXEC);
        self.triangles();
        self.set_exec(P_EXEC);
        self.instance();
        self.write_back();
        self.sopc(SOPC_CMP_LG_U32, self.single(S_SCC), Operand::InlineInt(0));
        self.leave_mode(mode);
    }

    /// Snapshot inputs, initialise outputs, decode pointer flags and node address.
    fn prologue(&mut self) {
        for i in 0..11u8 {
            self.mov(self.r(IN_ENC + i), self.guest(i as usize));
        }
        for i in 0..10u8 {
            self.mov(self.r(OUT + i), k(INVALID));
        }
        // Pointer flags: bits 54..63 of the encoded address.
        self.shr(PFLAGS, k(22), IN_ENC + 1);
        // disable_cull clears both face culls; otherwise both face culls skip triangles.
        self.iand(WORK, k(4), PFLAGS);
        self.cmp(CMP_NE_U32, P_A, self.r(WORK), k(0));
        self.iand(WORK, k(0xffff_ffcf), PFLAGS);
        self.sel(PFLAGS, P_A, self.r(WORK), self.r(PFLAGS));
        self.iand(WORK, k(0x30), PFLAGS);
        self.cmp(CMP_EQ_U32, P_B, self.r(WORK), k(0x30));
        self.ior(WORK, k(0x100), PFLAGS);
        self.sel(PFLAGS, P_B, self.r(WORK), self.r(PFLAGS));

        // Node address = ((encoded & (2^54-1)) << 3) + (index >> 3) * 64.
        self.mov(self.r(WORK), self.r(IN_ENC));
        self.iand(WORK + 1, k(0x003f_ffff), IN_ENC + 1);
        self.vop3(
            VOP3_LSHLREV_B64,
            self.r(ADDR),
            None,
            [k(3), self.r(WORK), Operand::Raw(0)],
        );
        self.iand(WORK, k(0xffff_fff8), IN_INDEX);
        self.shl(WORK, k(3), WORK);
        self.shr(WORK + 1, k(29), IN_INDEX);
        self.vop3(
            VOP3_ADD_CO_U32,
            self.r(ADDR),
            Some(self.pair(P_A)),
            [self.r(ADDR), self.r(WORK), Operand::Raw(0)],
        );
        self.vop3(
            VOP3_ADD_CO_CI_U32,
            self.r(ADDR + 1),
            Some(self.pair(P_B)),
            [self.r(ADDR + 1), self.r(WORK + 1), self.pair(P_A)],
        );
        self.iand(KIND, k(7), IN_INDEX);
    }

    /// P_VALID (not early culled), P_INST, P_TRI; all within the original EXEC.
    fn classify(&mut self) {
        self.cmp(CMP_EQ_U32, P_A, self.r(KIND), k(NODE_USER));
        self.cmp(CMP_EQ_U32, P_B, self.r(KIND), k(NODE_TRINITY_INSTANCE));
        self.desc_flag(P_C, 3, 22);
        self.and(P_INST, P_B, P_C);
        // Instance kind with instancing disabled is treated as USER.
        self.andn2(P_D, P_B, P_C);
        self.or(P_A, P_A, P_D);
        self.cmp(CMP_LE_U32, P_TRI, self.r(KIND), k(NODE_TRI_1));
        // Triangle node with skip_triangles.
        self.iand(WORK, k(0x100), PFLAGS);
        self.cmp(CMP_NE_U32, P_B, self.r(WORK), k(0));
        self.and(P_B, P_B, P_TRI);
        self.or(P_A, P_A, P_B);
        // force_opaque&cull_opaque, force_non_opaque&cull_non_opaque, cull_opaque&cull_non_opaque.
        for m in [0x41u32, 0x82, 0xc0] {
            self.iand(WORK, k(m), PFLAGS);
            self.cmp(CMP_EQ_U32, P_B, self.r(WORK), k(m));
            self.or(P_A, P_A, P_B);
        }
        self.andn2(P_VALID, P_EXEC, P_A);
        self.and(P_INST, P_INST, P_VALID);
        self.and(P_TRI, P_TRI, P_VALID);
    }

    // ---- box nodes ---------------------------------------------------------------------------

    /// `dst` = valid internal lanes (kinds 2..5); EXEC must be the original EXEC.
    fn box_mask(&mut self, dst: u8, tmp: u8) {
        self.cmp(CMP_GE_U32, dst, self.r(KIND), k(2));
        self.cmp(CMP_LE_U32, tmp, self.r(KIND), k(5));
        self.and(dst, dst, tmp);
        self.and(dst, dst, P_VALID);
    }

    fn boxes(&mut self) {
        self.set_exec(P_EXEC);
        self.box_mask(P_M, P_A);
        self.set_exec(P_M);
        // Reciprocal direction, grow factor and miss-initialised keys.
        for a in 0..3u8 {
            self.vop1(VOP1_RCP_F32, self.r(RDIR + a), self.r(IN_DIR + a));
        }
        self.sop2(
            SOP2_BFE_U32,
            self.pair(P_A),
            self.desc(1),
            k((8 << 16) | 23),
        );
        self.vop1(VOP1_CVT_F32_U32, self.r(FACTOR), self.pair(P_A));
        self.vop2(
            VOP2_MUL_F32,
            self.r(FACTOR),
            k(F32_GROW_STEP),
            self.r(FACTOR),
        );
        self.vop2(VOP2_ADD_F32, self.r(FACTOR), k(F32_ONE), self.r(FACTOR));
        for c in 0..8u8 {
            self.mov(self.r(KEY + c), k(F32_INF));
        }
        self.box_pass(0);
        self.box_pass(1);
        self.sort();
    }

    /// One four-child pass: `pass` 0 is the first node, 1 the second node of big kinds.
    fn box_pass(&mut self, pass: u8) {
        // P_M: pass lanes, P_N: FP32 lanes, P_O: shared-exponent lanes.
        self.set_exec(P_EXEC);
        self.box_mask(P_M, P_A);
        if pass == 1 {
            self.kind_is(
                P_B,
                P_A,
                &[NODE_TRINITY_BIG_FP32, NODE_TRINITY_BIG_SHARED_EXPONENT],
            );
            self.and(P_M, P_M, P_B);
        }
        self.kind_is(P_N, P_A, &[NODE_FP32, NODE_TRINITY_BIG_FP32]);
        self.and(P_N, P_N, P_M);
        self.kind_is(
            P_O,
            P_A,
            &[
                NODE_TRINITY_SHARED_EXPONENT,
                NODE_TRINITY_BIG_SHARED_EXPONENT,
            ],
        );
        self.and(P_O, P_O, P_M);

        if pass == 0 {
            self.set_exec(P_M);
            self.load(DATA, 16, 0);
            self.nop();
            self.set_exec(P_N);
            self.load(DATA + 16, 12, 64);
        } else {
            self.set_exec(P_N);
            self.load(DATA, 28, 128);
            self.nop();
            self.set_exec(P_O);
            self.load(DATA, 16, 64);
        }
        self.wait_loads();

        for c in 0..4u8 {
            self.child(pass * 4 + c, c);
        }
        self.set_exec(P_EXEC);
    }

    fn d(&self, w: u8) -> Operand {
        self.r(DATA + w)
    }

    fn child(&mut self, slot: u8, c: u8) {
        self.set_exec(P_EXEC);
        if slot < 4 {
            self.box_mask(P_M, P_A);
        } else {
            self.kind_is(
                P_M,
                P_A,
                &[NODE_TRINITY_BIG_FP32, NODE_TRINITY_BIG_SHARED_EXPONENT],
            );
            self.and(P_M, P_M, P_VALID);
        }
        self.kind_is(P_N, P_A, &[NODE_FP32, NODE_TRINITY_BIG_FP32]);
        self.and(P_N, P_N, P_M);
        self.kind_is(
            P_O,
            P_A,
            &[
                NODE_TRINITY_SHARED_EXPONENT,
                NODE_TRINITY_BIG_SHARED_EXPONENT,
            ],
        );
        self.and(P_O, P_O, P_M);
        self.set_exec(P_N);
        self.mov(self.r(BIDX), self.d(c));
        self.mov(self.r(BFLAGS), k(0));
        for i in 0..6u8 {
            self.mov(self.r(BMIN + i), self.d(4 + 6 * c + i));
        }
        self.set_exec(P_O);
        self.se_index(c);
        self.bfe(
            BFLAGS,
            self.d(1),
            9 + 4 * u32::from(c),
            if c < 3 { 4 } else { 3 },
        );
        for max in [false, true] {
            for axis in 0..3u8 {
                self.se_bound(if max { BMAX + axis } else { BMIN + axis }, c, axis, max);
            }
        }
        self.set_exec(P_M);
        self.slab(slot);
        self.set_exec(P_EXEC);
    }

    /// Extract one packed field; all shared-exponent fields span at most two words.
    fn packed(&mut self, d: u8, offset: u32, width: u32) {
        let word = (offset / 32) as u8;
        let shift = offset % 32;
        self.shr(d, k(shift), DATA + word);
        if shift + width > 32 {
            self.shl(BT + 7, k(32 - shift), DATA + word + 1);
            self.ior(d, self.r(BT + 7), d);
        }
        self.iand(d, k((1u32 << width) - 1), d);
    }

    fn se_index(&mut self, c: u8) {
        self.packed(BIDX, 0, 29);
        for j in 0..c {
            self.packed(BT, 29 + 3 * u32::from(j), 3);
            self.mov(self.r(BT + 1), k(1));
            self.cmp(CMP_EQ_U32, P_A, self.r(BT), k(2));
            self.sel(BT + 1, P_A, k(4), self.r(BT + 1));
            self.cmp(CMP_EQ_U32, P_A, self.r(BT), k(3));
            self.cmp(CMP_EQ_U32, P_B, self.r(BT), k(5));
            self.or(P_A, P_A, P_B);
            self.cmp(CMP_EQ_U32, P_B, self.r(BT), k(6));
            self.or(P_A, P_A, P_B);
            self.sel(BT + 1, P_A, k(2), self.r(BT + 1));
            self.iop(VOP2_ADD_NC_U32, BIDX, self.r(BIDX), BT + 1);
        }
        self.packed(BT, 29 + 3 * u32::from(c), 3);
        self.shl(BIDX, k(3), BIDX);
        self.ior(BIDX, self.r(BT), BIDX);
    }

    /// Outward-rounded binary32 reconstruction of a signed 18-bit mantissa.
    /// BT..BT+7 are reused; reciprocal direction and the grow factor stay live above them.
    fn se_bound(&mut self, d: u8, c: u8, axis: u8, max: bool) {
        let (exp, sign, mant, diff, result_exp, result_mant, trail) =
            (BT, BT + 1, BT + 2, BT + 4, BT + 5, BT + 6, BT + 7);
        self.packed(
            d,
            80 + 18 * (6 * u32::from(c) + if max { 3 } else { 0 } + u32::from(axis)),
            18,
        );
        self.packed(exp, 56 + 8 * u32::from(axis), 8);
        self.bfe(sign, self.r(d), 17, 1);
        self.shl(sign, k(31), sign);
        self.iand(mant, k(0x1ffff), d);
        self.vop1(0x39, self.r(BT + 3), self.r(mant));
        self.iop(VOP2_ADD_NC_U32, diff, k((-15i32) as u32), BT + 3);
        self.iop(VOP2_MIN_U32, diff, self.r(exp), diff);
        self.iop(VOP2_SUB_NC_U32, result_exp, self.r(exp), diff);
        self.iop(VOP2_ADD_NC_U32, diff, k(7), diff);
        self.shl(result_mant, self.r(diff), mant);
        self.vop3(
            VOP3_LSHLREV_B32,
            self.r(trail),
            None,
            [self.r(diff), k(1), Operand::Raw(0)],
        );
        self.iop(VOP2_ADD_NC_U32, trail, k(u32::MAX), trail);
        self.cmp(CMP_EQ_U32, P_A, self.r(sign), k(if max { 0 } else { SIGN }));
        self.cmp(CMP_NE_U32, P_B, self.r(result_exp), k(255));
        self.and(P_B, P_A, P_B);
        self.sel(trail, P_B, self.r(trail), k(0));
        self.ior(result_mant, self.r(trail), result_mant);
        self.iand(result_mant, k(0x7fffff), result_mant);
        // A zero mantissa has its own outward-rounded representation.
        self.iop(VOP2_ADD_NC_U32, BT + 3, k((-17i32) as u32), exp);
        self.cmp(CMP_GT_U32, P_B, self.r(exp), k(17));
        self.and(P_B, P_A, P_B);
        self.sel(BT + 3, P_B, self.r(BT + 3), k(0));
        self.sel(trail, P_A, k(0x7fffff), k(0));
        self.cmp(CMP_EQ_U32, P_C, self.r(mant), k(0));
        self.sel(result_exp, P_C, self.r(BT + 3), self.r(result_exp));
        self.sel(result_mant, P_C, self.r(trail), self.r(result_mant));
        self.shl(result_exp, k(23), result_exp);
        self.ior(result_mant, self.r(result_exp), result_mant);
        self.ior(d, self.r(sign), result_mant);
    }

    fn slab(&mut self, slot: u8) {
        for axis in 0..3u8 {
            self.vop2(
                VOP2_SUB_F32,
                self.r(BMIN + axis),
                self.r(BMIN + axis),
                self.r(IN_ORG + axis),
            );
            self.fmul(BMIN + axis, BMIN + axis, RDIR + axis);
            self.vop2(
                VOP2_SUB_F32,
                self.r(BMAX + axis),
                self.r(BMAX + axis),
                self.r(IN_ORG + axis),
            );
            self.fmul(BMAX + axis, BMAX + axis, RDIR + axis);
            self.cmp(CMP_GE_F32, P_A, self.r(RDIR + axis), k(0));
            self.sel(BT, P_A, self.r(BMIN + axis), self.r(BMAX + axis));
            self.sel(BMAX + axis, P_A, self.r(BMAX + axis), self.r(BMIN + axis));
            self.mov(self.r(BMIN + axis), self.r(BT));
        }
        self.mov(self.r(BT), self.r(BMIN));
        self.mov(self.r(BT + 1), self.r(BMAX));
        for axis in 1..3u8 {
            self.cmp(CMP_LT_F32, P_A, self.r(BT), self.r(BMIN + axis));
            self.sel(BT, P_A, self.r(BMIN + axis), self.r(BT));
            self.cmp(CMP_LT_F32, P_A, self.r(BT + 1), self.r(BMAX + axis));
            self.sel(BT + 1, P_A, self.r(BT + 1), self.r(BMAX + axis));
        }
        self.fmul(BT + 1, BT + 1, FACTOR);
        self.cmp(CMP_GE_F32, P_A, self.r(BT + 1), self.r(BT));
        self.cmp(CMP_GE_F32, P_B, self.r(BT + 1), k(0));
        self.and(P_D, P_A, P_B);
        self.cmp(CMP_LT_F32, P_A, self.r(BT), self.r(IN_EXTENT));
        self.and(P_D, P_D, P_A);
        self.cmp(CMP_LT_F32, P_A, self.r(BT), k(0));
        self.sel(KEY + slot, P_A, k(0), self.r(BT));
        self.fsub(BT + 2, BT + 1, BT);
        self.vop2(VOP2_SUB_F32, self.r(BT + 2), k(0), self.r(BT + 2));
        self.heuristic_flag(P_A, P_B, 1);
        self.sel(KEY + slot, P_A, self.r(BT + 2), self.r(KEY + slot));
        self.fadd(BT + 2, BT, BT + 1);
        self.vop2(VOP2_MUL_F32, self.r(BT + 2), k(F32_HALF), self.r(BT + 2));
        self.heuristic_flag(P_A, P_B, 2);
        self.sel(KEY + slot, P_A, self.r(BT + 2), self.r(KEY + slot));
        self.subtree_cull();
        self.sel(OUT + slot, P_D, self.r(BIDX), k(INVALID));
        self.sel(KEY + slot, P_D, self.r(KEY + slot), k(F32_INF));
    }

    /// Remove lanes whose entire subtree is excluded by the inherited pointer flags.
    fn subtree_cull(&mut self) {
        for (node_bit, force, opposite, cull) in [(1u32, 1u32, 2u32, 64u32), (2, 2, 1, 128)] {
            self.iand(BT + 2, k(node_bit), BFLAGS);
            self.iand(BT + 3, k(force), PFLAGS);
            self.ior(BT + 2, self.r(BT + 3), BT + 2);
            self.cmp(CMP_NE_U32, P_A, self.r(BT + 2), k(0));
            self.iand(BT + 2, k(opposite), PFLAGS);
            self.cmp(CMP_NE_U32, P_B, self.r(BT + 2), k(0));
            self.andn2(P_A, P_A, P_B);
            self.iand(BT + 2, k(cull), PFLAGS);
            self.cmp(CMP_NE_U32, P_B, self.r(BT + 2), k(0));
            self.and(P_A, P_A, P_B);
            self.andn2(P_D, P_D, P_A);
        }
        for (node_bit, cull) in [(4u32, 256u32), (8, 512)] {
            self.iand(BT + 2, k(node_bit), BFLAGS);
            self.cmp(CMP_NE_U32, P_A, self.r(BT + 2), k(0));
            self.iand(BT + 2, k(cull), PFLAGS);
            self.cmp(CMP_NE_U32, P_B, self.r(BT + 2), k(0));
            self.and(P_A, P_A, P_B);
            self.andn2(P_D, P_D, P_A);
        }
    }

    fn sort(&mut self) {
        self.set_exec(P_EXEC);
        self.box_mask(P_M, P_A);
        for (step, (a, b)) in SORT_NETWORK.into_iter().enumerate() {
            self.set_exec(P_M);
            if step >= 10 {
                self.desc_flag(P_C, 3, 21);
                self.and(P_C, P_C, P_M);
                self.set_exec(P_C);
            }
            let (a, b) = (a as u8, b as u8);
            self.cmp(CMP_GT_U32, P_A, self.r(KEY + a), self.r(KEY + b));
            self.cmp(CMP_GT_F32, P_B, self.r(KEY + a), self.r(KEY + b));
            self.heuristic_flag(P_C, P_D, 0);
            self.and(P_A, P_A, P_C);
            self.andn2(P_B, P_B, P_C);
            self.or(P_A, P_A, P_B);
            self.desc_flag(P_C, 1, 20);
            self.iand(BT, k(7), OUT + a);
            self.iand(BT + 1, k(7), OUT + b);
            self.cmp(CMP_LE_U32, P_B, self.r(BT), k(1));
            self.cmp(CMP_LE_U32, P_D, self.r(BT + 1), k(1));
            self.xor(P_N, P_B, P_D);
            self.and(P_N, P_N, P_C);
            self.andn2(P_O, P_D, P_B);
            self.and(P_O, P_O, P_C);
            self.andn2(P_A, P_A, P_N);
            self.or(P_A, P_A, P_O);
            self.desc_flag(P_C, 1, 31);
            self.and(P_A, P_A, P_C);
            for (base, tmp) in [(OUT, BT), (KEY, BT + 1)] {
                self.sel(tmp, P_A, self.r(base + b), self.r(base + a));
                self.sel(base + b, P_A, self.r(base + a), self.r(base + b));
                self.mov(self.r(base + a), self.r(tmp));
            }
        }
        self.set_exec(P_EXEC);
    }

    fn triangles(&mut self) {
        for triangle in 0..2u8 {
            self.set_exec(P_TRI);
            self.load(DATA, 16, 0);
            self.wait_loads();
            self.mov(self.r(TRI_FLAG), self.d(15));
            if triangle == 0 {
                self.mov(self.r(OUT + 8), self.d(14));
            } else {
                self.cmp(CMP_EQ_U32, P_A, self.r(KIND), k(NODE_TRI_1));
                self.sel(OUT + 9, P_A, self.d(14), k(INVALID));
            }
            self.triangle_math(triangle);
            self.triangle_flags(triangle);
            for (i, r) in [TNUM, TDEN, RZ, RW].into_iter().enumerate() {
                self.mov(self.r(OUT + 4 * triangle + i as u8), self.r(r));
            }
        }
        self.set_exec(P_EXEC);
    }

    fn permute(&mut self, src: [Operand; 3], out: [u8; 3], x: u8, y: u8) {
        for (axis, dst) in out.into_iter().enumerate() {
            self.sel(dst, y, src[(2 + axis) % 3], src[axis]);
            self.sel(dst, x, src[(1 + axis) % 3], self.r(dst));
        }
    }

    fn triangle_math(&mut self, triangle: u8) {
        let vertices = if triangle == 0 {
            [0u8, 1, 2]
        } else {
            [1, 3, 2]
        };
        // The final destination aliases only an already-consumed source coordinate.
        for (v, slot) in vertices.into_iter().enumerate() {
            for axis in 0..3u8 {
                self.vop2(
                    VOP2_SUB_F32,
                    self.r(TREL + 3 * v as u8 + axis),
                    self.d(3 * slot + axis),
                    self.r(IN_ORG + axis),
                );
            }
        }
        for axis in 0..3u8 {
            self.iand(S0 + axis, k(0x7fff_ffff), IN_DIR + axis);
        }
        self.cmp(CMP_GE_F32, P_M, self.r(S0), self.r(S1));
        self.cmp(CMP_GE_F32, P_N, self.r(S0), self.r(S2));
        self.cmp(CMP_GE_F32, P_O, self.r(S1), self.r(S2));
        self.and(P_A, P_M, P_N);
        self.andn2(P_B, P_O, P_M);
        self.permute(
            [self.r(IN_DIR), self.r(IN_DIR + 1), self.r(IN_DIR + 2)],
            [TD, TD + 1, TD + 2],
            P_A,
            P_B,
        );
        for v in 0..3u8 {
            self.permute(
                [
                    self.r(TREL + 3 * v),
                    self.r(TREL + 3 * v + 1),
                    self.r(TREL + 3 * v + 2),
                ],
                [TP + 3 * v, TP + 3 * v + 1, TP + 3 * v + 2],
                P_A,
                P_B,
            );
        }
        // Unnormalised projection: preserve the reference's numerator/denominator scale.
        for (v, x, y) in [(0u8, AX, AY), (1, BX, BY), (2, CX, CY)] {
            self.fmul(x, TP + 3 * v, TD + 2);
            self.fmul(S0, TD, TP + 3 * v + 2);
            self.fsub(x, x, S0);
            self.fmul(y, TP + 3 * v + 1, TD + 2);
            self.fmul(S0, TD + 1, TP + 3 * v + 2);
            self.fsub(y, y, S0);
        }
        for (dst, px, py, qx, qy) in [
            (TU, CX, CY, BX, BY),
            (TV, AX, AY, CX, CY),
            (TW, BX, BY, AX, AY),
        ] {
            self.fmul(S0, px, qy);
            self.fmul(S1, py, qx);
            self.fsub(dst, S0, S1);
        }
        for (dst, edge) in [(TUZ, TU), (TVZ, TV), (TWZ, TW)] {
            self.fmul(dst, edge, TD + 2);
        }
        self.fadd(TDEN, TUZ, TVZ);
        self.fadd(TDEN, TDEN, TWZ);
        self.fmul(TNUM, TU, TP + 2);
        self.fmul(S0, TV, TP + 5);
        self.fadd(TNUM, TNUM, S0);
        self.fmul(S0, TW, TP + 8);
        self.fadd(TNUM, TNUM, S0);

        // P_D accumulates the geometric miss mask.
        self.cmp(CMP_LT_F32, P_A, self.r(TU), k(0));
        self.cmp(CMP_GT_F32, P_B, self.r(TU), k(0));
        for edge in [TV, TW] {
            self.cmp(CMP_LT_F32, P_C, self.r(edge), k(0));
            self.or(P_A, P_A, P_C);
            self.cmp(CMP_GT_F32, P_C, self.r(edge), k(0));
            self.or(P_B, P_B, P_C);
        }
        self.and(P_D, P_A, P_B);
        self.cmp(CMP_EQ_F32, P_A, self.r(TDEN), k(0));
        self.or(P_D, P_D, P_A);
        self.cmp(CMP_U_F32, P_A, self.r(TNUM), self.r(TNUM));
        self.or(P_D, P_D, P_A);
        for (den_cmp, num_cmp) in [(CMP_GT_F32, CMP_LT_F32), (CMP_LT_F32, CMP_GT_F32)] {
            self.cmp(den_cmp, P_A, self.r(TDEN), k(0));
            self.cmp(num_cmp, P_B, self.r(TNUM), k(0));
            self.and(P_A, P_A, P_B);
            self.or(P_D, P_D, P_A);
        }
        for (edge, y0, y1, opposite) in [(TU, BY, CY, AY), (TV, CY, AY, BY), (TW, AY, BY, CY)] {
            self.cmp(CMP_EQ_F32, P_M, self.r(edge), k(0));
            self.cmp(CMP_EQ_F32, P_A, self.r(y0), k(0));
            self.cmp(CMP_EQ_F32, P_B, self.r(y1), k(0));
            self.and(P_N, P_A, P_B);
            self.cmp(CMP_GT_F32, P_A, self.r(opposite), k(0));
            self.and(P_A, P_A, P_N);
            self.cmp(CMP_LT_F32, P_B, self.r(y0), k(0));
            self.cmp(CMP_EQ_F32, P_C, self.r(y0), k(0));
            self.cmp(CMP_GT_F32, P_O, self.r(y1), k(0));
            self.and(P_C, P_C, P_O);
            self.or(P_B, P_B, P_C);
            self.cmp(CMP_GT_F32, P_O, self.r(TDEN), k(0));
            self.xor(P_B, P_B, P_O);
            self.andn2(P_B, P_B, P_N);
            self.or(P_A, P_A, P_B);
            self.and(P_A, P_A, P_M);
            self.or(P_D, P_D, P_A);
        }

        self.bfe(TI, self.r(TRI_FLAG), 8 * u32::from(triangle), 2);
        self.bfe(TJ, self.r(TRI_FLAG), 8 * u32::from(triangle) + 2, 2);
        for (dst, index) in [(RZ, TI), (RW, TJ)] {
            self.mov(self.r(dst), self.r(TUZ));
            self.cmp(CMP_EQ_U32, P_B, self.r(index), k(1));
            self.sel(dst, P_B, self.r(TVZ), self.r(dst));
            self.cmp(CMP_EQ_U32, P_B, self.r(index), k(2));
            self.sel(dst, P_B, self.r(TWZ), self.r(dst));
        }
        self.desc_flag(P_A, 3, 24);
        self.iop(VOP2_ADD_NC_U32, S0, k(u32::from(triangle)), TRI_FLAG);
        self.sel(RZ, P_A, self.r(RZ), self.r(S0));
        self.sel(S0, P_D, k(0), k(1));
        self.sel(RW, P_A, self.r(RW), self.r(S0));
        self.sel(TNUM, P_D, k(F32_INF), self.r(TNUM));
        self.sel(TDEN, P_D, k(F32_ONE), self.r(TDEN));
    }

    fn pointer_bit(&mut self, mask: u8, bit: u32) {
        self.bfe(S0, self.r(PFLAGS), bit, 1);
        self.cmp(CMP_NE_U32, mask, self.r(S0), k(0));
    }

    fn triangle_flags(&mut self, triangle: u8) {
        let shift = 8 * u32::from(triangle);
        for (dst, bit) in [(DSIDE, 4), (FLIP, 5), (PROC, 6), (OPQ, 7)] {
            self.bfe(dst, self.r(TRI_FLAG), shift + bit, 1);
        }
        self.desc_flag(P_A, 3, 23);
        for r in [DSIDE, FLIP, PROC, OPQ] {
            self.sel(r, P_A, self.r(r), k(0));
        }
        self.bfe(S0, self.r(PFLAGS), 3, 1);
        self.ixor(FLIP, self.r(S0), FLIP);
        self.bfe(S0, self.r(PFLAGS), 0, 1);
        self.ior(OPQ, self.r(S0), OPQ);
        self.bfe(S0, self.r(PFLAGS), 1, 1);
        self.ixor(S0, k(1), S0);
        self.iand(OPQ, self.r(S0), OPQ);
        self.cmp(CMP_NE_U32, P_A, self.r(PROC), k(0));
        self.sel(TNUM, P_A, k(0), self.r(TNUM));
        self.sel(TDEN, P_A, k(F32_ONE), self.r(TDEN));
        self.sel(RZ, P_A, k(F32_ONE), self.r(RZ));
        self.sel(RW, P_A, k(F32_ONE), self.r(RW));
        if triangle == 1 {
            self.cmp(CMP_EQ_U32, P_B, self.r(KIND), k(NODE_TRI_1));
            self.desc_flag(P_A, 3, 24);
            self.sel(S0, P_A, k(0), k(INVALID));
            self.sel(TNUM, P_B, self.r(TNUM), k(F32_INF));
            self.sel(TDEN, P_B, self.r(TDEN), k(F32_ONE));
            self.sel(RZ, P_B, self.r(RZ), self.r(S0));
            self.sel(RW, P_B, self.r(RW), k(0));
        }
        self.cmp(CMP_EQ_U32, P_D, self.r(TNUM), k(F32_INF));
        self.cmp(CMP_EQ_U32, P_A, self.r(PROC), k(0));
        self.cmp(CMP_NE_U32, P_B, self.r(FLIP), k(0));
        self.and(P_A, P_A, P_B);
        self.andn2(P_A, P_A, P_D);
        for r in [TNUM, TDEN, RZ, RW] {
            self.ixor(S0, k(SIGN), r);
            self.sel(r, P_A, self.r(S0), self.r(r));
        }
        self.cmp(CMP_GT_F32, P_A, self.r(TDEN), k(0));
        self.sel(FRONT, P_A, k(1), k(0));

        for (opaque_cmp, bit) in [(CMP_NE_U32, 6), (CMP_EQ_U32, 7)] {
            self.cmp(opaque_cmp, P_A, self.r(OPQ), k(0));
            self.pointer_bit(P_B, bit);
            self.and(P_A, P_A, P_B);
            self.or(P_D, P_D, P_A);
        }
        for (front_cmp, bit) in [(CMP_NE_U32, 5), (CMP_EQ_U32, 4)] {
            self.cmp(front_cmp, P_A, self.r(FRONT), k(0));
            self.pointer_bit(P_B, bit);
            self.and(P_A, P_A, P_B);
            self.cmp(CMP_EQ_U32, P_B, self.r(DSIDE), k(0));
            self.and(P_A, P_A, P_B);
            self.cmp(CMP_EQ_U32, P_B, self.r(PROC), k(0));
            self.and(P_A, P_A, P_B);
            self.or(P_D, P_D, P_A);
        }
        for (proc_cmp, bit) in [(CMP_EQ_U32, 8), (CMP_NE_U32, 9)] {
            self.cmp(proc_cmp, P_A, self.r(PROC), k(0));
            self.pointer_bit(P_B, bit);
            self.and(P_A, P_A, P_B);
            self.or(P_D, P_D, P_A);
        }
        self.desc_flag(P_A, 3, 24);
        self.sel(S0, P_A, k(0), k(INVALID));
        self.sel(TNUM, P_D, k(F32_INF), self.r(TNUM));
        self.sel(TDEN, P_D, k(F32_ONE), self.r(TDEN));
        self.sel(RZ, P_D, self.r(S0), self.r(RZ));
        self.sel(RW, P_D, k(0), self.r(RW));
        // Any-hit annotations apply only to surviving intersections.
        self.desc_flag(P_A, 3, 23);
        self.andn2(P_A, P_A, P_D);
        self.ixor(S0, k(1), OPQ);
        self.ior(S0, self.r(PROC), S0);
        self.shl(S0, k(31), S0);
        self.iand(S1, k(0x7fff_ffff), RZ);
        self.ior(S1, self.r(S0), S1);
        self.sel(RZ, P_A, self.r(S1), self.r(RZ));
        self.ixor(S0, k(1), OPQ);
        self.shl(S0, k(31), S0);
        self.iand(S1, k(0x7fff_ffff), RW);
        self.ior(S1, self.r(S0), S1);
        self.sel(RW, P_A, self.r(S1), self.r(RW));
    }

    fn instance(&mut self) {
        self.set_exec(P_INST);
        self.load(DATA, 16, 0);
        self.wait_loads();
        for row in 0..3u8 {
            for (input, dst, position) in
                [(IN_ORG, WORK + row, true), (IN_DIR, WORK + 3 + row, false)]
            {
                self.vop2(VOP2_MUL_F32, self.r(dst), self.d(4 * row), self.r(input));
                for axis in 1..3u8 {
                    self.vop2(
                        VOP2_MUL_F32,
                        self.r(WORK + 16),
                        self.d(4 * row + axis),
                        self.r(input + axis),
                    );
                    self.fadd(dst, dst, WORK + 16);
                }
                if position {
                    self.vop2(VOP2_ADD_F32, self.r(dst), self.r(dst), self.d(4 * row + 3));
                }
            }
        }
        let (child, merged, a, b, c) = (WORK + 6, WORK + 7, WORK + 8, WORK + 9, WORK + 10);
        self.shr(child, k(22), DATA + 13);
        self.mov(self.r(merged), k(0));
        for bit in 0..10u32 {
            self.bfe(a, self.r(child), bit, 1);
            match bit {
                0 | 1 => {
                    self.bfe(b, self.r(PFLAGS), 1 - bit, 1);
                    self.ixor(b, k(1), b);
                    self.iand(a, self.r(b), a);
                    self.bfe(b, self.r(PFLAGS), bit, 1);
                    self.ior(a, self.r(b), a);
                }
                2 => {}
                3 => {
                    self.bfe(b, self.r(PFLAGS), bit, 1);
                    self.ixor(a, self.r(b), a);
                }
                _ => {
                    self.bfe(b, self.r(PFLAGS), bit, 1);
                    self.ior(a, self.r(b), a);
                }
            }
            self.shl(a, k(bit), a);
            self.ior(merged, self.r(a), merged);
        }
        self.mov(self.r(OUT + 2), self.d(12));
        self.iand(c, k(0x3f_ffff), DATA + 13);
        self.shl(merged, k(22), merged);
        self.ior(OUT + 3, self.r(c), merged);
        self.iand(OUT + 6, k(0xff_ffff), DATA + 15);
        self.shr(a, k(24), DATA + 15);
        self.iand(a, self.r(IN_MASK), a);
        self.cmp(CMP_NE_U32, P_A, self.r(a), k(0));
        self.sel(OUT + 7, P_A, self.d(14), k(INVALID));
        // Mask rejection only changes the child index; it does not suppress the transform.
        for axis in 0..3usize {
            self.mov(self.guest(4 + axis), self.r(WORK + axis as u8));
            self.mov(self.guest(7 + axis), self.r(WORK + 3 + axis as u8));
        }
        self.set_exec(P_EXEC);
    }

    fn write_back(&mut self) {
        self.set_exec(P_EXEC);
        for i in 0..10u8 {
            self.select(
                Operand::Vgpr(self.m.vdata + i),
                P_VALID,
                self.r(OUT + i),
                k(INVALID),
            );
        }
    }
}
