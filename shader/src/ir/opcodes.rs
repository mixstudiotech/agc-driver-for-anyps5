use super::{memops, Fields, Instruction, InstructionFamily};
use crate::TargetIsa;

pub mod flag {
    pub const D2: u64 = 1 << 0;
    pub const S0_2: u64 = 1 << 1;
    pub const S1_2: u64 = 1 << 2;
    pub const S2_2: u64 = 1 << 3;
    pub const BRANCH: u64 = 1 << 4;
    pub const COND_BRANCH: u64 = 1 << 5;
    pub const END: u64 = 1 << 6;
    pub const JUMP: u64 = 1 << 7;
    pub const CALL: u64 = 1 << 8;
    pub const NO_DST: u64 = 1 << 9;
    pub const NO_SRC: u64 = 1 << 10;
    pub const NO_OPERANDS: u64 = 1 << 11;
    pub const WRITES_EXEC: u64 = 1 << 12;
    pub const READS_EXEC: u64 = 1 << 13;
    pub const READS_VCC: u64 = 1 << 14;
    pub const WRITES_VCC: u64 = 1 << 15;
    pub const READS_M0: u64 = 1 << 16;
    pub const READS_DST: u64 = 1 << 17;
    pub const CARRY_IN: u64 = 1 << 18;
    pub const CARRY_OUT: u64 = 1 << 19;
    pub const SDST: u64 = 1 << 20;
    pub const MASK_SRC2: u64 = 1 << 21;
    pub const SWAP: u64 = 1 << 22;
    pub const TRANS: u64 = 1 << 23;
    pub const T16: u64 = 1 << 24;
    pub const LITK: u64 = 1 << 25;
    pub const CMPX: u64 = 1 << 26;
    pub const CMPSWAP: u64 = 1 << 27;
    pub const IMM_U: u64 = 1 << 28;
    pub const VOP3B: u64 = 1 << 29;
    pub const SMEM_ADDR: u64 = 1 << 30;
    pub const SRC1: u64 = 1 << 31;
    pub const SRC2: u64 = 1 << 32;
    pub const SRC3: u64 = 1 << 33;
    pub const RELATIVE: u64 = 1 << 34;
    pub const PC_RELATIVE: u64 = 1 << 35;
    pub const LANE_SELECT: u64 = 1 << 36;
    /// A VOP1/VOP2 instruction without a VOP3 (`_e64`) form.
    pub const NO_VOP3: u64 = 1 << 38;
    pub const W64: u64 = D2 | S0_2 | S1_2 | S2_2;
}

use flag::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Rule {
    Bvh8,
    BvhStack8,
    Mac,
    MacLegacy,
    Mad,
    MadLegacy,
    Madmk,
    Madak,
    Dot2c,
    Dot4c,
    DotSigned(u16),
    Cmpk(u16),
    Hwreg(u16),
    Sendmsg(u16),
    Waitcnt,
    WaitcntCounter,
    Barrier,
    MemTime,
    Interp,
    /// `s_get_waveid_in_workgroup`: TG_SIZE bits 24:20.
    WaveId,
    FindFirstZero,
    Dot2(bool),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Xlat {
    Same,
    Op(u16),
    Rule(Rule),
    Drop,
    No,
}

#[derive(Clone, Copy, Debug)]
pub struct OpInfo {
    pub op: u16,
    pub name: &'static str,
    pub flags: u64,
    pub gfx103: Xlat,
    pub gfx11: Xlat,
    pub gfx12: Xlat,
}

impl OpInfo {
    pub fn translate(&self, target: TargetIsa) -> Xlat {
        let x = match target {
            TargetIsa::Gfx1013 => Xlat::Same,
            TargetIsa::Gfx103 => self.gfx103,
            TargetIsa::Gfx11 | TargetIsa::Gfx115 => self.gfx11,
            TargetIsa::Gfx12 => self.gfx12,
        };
        match x {
            Xlat::Same => Xlat::Op(self.op),
            other => other,
        }
    }
}

const S: Xlat = Xlat::Same;
const N: Xlat = Xlat::No;
const D: Xlat = Xlat::Drop;

const fn o(op: u16) -> Xlat {
    Xlat::Op(op)
}

const fn r(rule: Rule) -> Xlat {
    Xlat::Rule(rule)
}

const fn e(
    op: u16,
    name: &'static str,
    flags: u64,
    gfx103: Xlat,
    gfx11: Xlat,
    gfx12: Xlat,
) -> OpInfo {
    OpInfo {
        op,
        name,
        flags,
        gfx103,
        gfx11,
        gfx12,
    }
}

pub const VINTRP_P1: u16 = 0;
pub const VINTRP_P2: u16 = 1;
pub const VINTRP_MOV: u16 = 2;
/// gfx11 LDSDIR and VINTERP opcodes (gfx11 numbering; these families exist only on the host).
pub const LDSDIR_PARAM_LOAD: u16 = 0;
pub const VINTERP_P10_F32: u16 = 0;
pub const VINTERP_P2_F32: u16 = 1;
pub const VINTERP_P10_F16_F32: u16 = 2;
pub const VINTERP_P2_F16_F32: u16 = 3;
/// gfx10 VOP3 f16 interpolation: `v_interp_p1ll_f16`, `v_interp_p1lv_f16`, `v_interp_p2_f16`.
pub const VOP3_INTERP_P1LL_F16: u16 = 0x342;
pub const VOP3_INTERP_P1LV_F16: u16 = 0x343;
pub const VOP3_INTERP_P2_F16: u16 = 0x35a;
pub const VOP3_VOPC_LAST: u16 = 0xff;
pub const VOP3_VOP2_BASE: u16 = 0x100;
pub const VOP3_VOP2_LAST: u16 = 0x13f;
pub const VOP3_VOP1_BASE: u16 = 0x180;
pub const VOP3_VOP1_LAST: u16 = 0x1ff;

const SAVEEXEC64: u64 = D2 | S0_2 | WRITES_EXEC;

pub static SOP1: &[OpInfo] = &[
    e(0x03, "s_mov_b32", 0, S, o(0x00), o(0x00)),
    e(0x04, "s_mov_b64", D2 | S0_2, S, o(0x01), o(0x01)),
    e(0x05, "s_cmov_b32", READS_DST, S, o(0x02), o(0x02)),
    e(
        0x06,
        "s_cmov_b64",
        D2 | S0_2 | READS_DST,
        S,
        o(0x03),
        o(0x03),
    ),
    e(0x07, "s_not_b32", 0, S, o(0x1e), o(0x1e)),
    e(0x08, "s_not_b64", D2 | S0_2, S, o(0x1f), o(0x1f)),
    e(0x09, "s_wqm_b32", 0, S, o(0x1c), o(0x1c)),
    e(0x0a, "s_wqm_b64", D2 | S0_2, S, o(0x1d), o(0x1d)),
    e(0x0b, "s_brev_b32", 0, S, o(0x04), o(0x04)),
    e(0x0c, "s_brev_b64", D2 | S0_2, S, o(0x05), o(0x05)),
    e(0x0d, "s_bcnt0_i32_b32", 0, S, o(0x16), o(0x16)),
    e(0x0e, "s_bcnt0_i32_b64", S0_2, S, o(0x17), o(0x17)),
    e(0x0f, "s_bcnt1_i32_b32", 0, S, o(0x18), o(0x18)),
    e(0x10, "s_bcnt1_i32_b64", S0_2, S, o(0x19), o(0x19)),
    e(0x11, "s_ff0_i32_b32", 0, S, r(Rule::FindFirstZero), r(Rule::FindFirstZero)),
    e(0x12, "s_ff0_i32_b64", S0_2, S, r(Rule::FindFirstZero), r(Rule::FindFirstZero)),
    e(0x13, "s_ff1_i32_b32", 0, S, o(0x08), o(0x08)),
    e(0x14, "s_ff1_i32_b64", S0_2, S, o(0x09), o(0x09)),
    e(0x15, "s_flbit_i32_b32", 0, S, o(0x0a), o(0x0a)),
    e(0x16, "s_flbit_i32_b64", S0_2, S, o(0x0b), o(0x0b)),
    e(0x17, "s_flbit_i32", 0, S, o(0x0c), o(0x0c)),
    e(0x18, "s_flbit_i32_i64", S0_2, S, o(0x0d), o(0x0d)),
    e(0x19, "s_sext_i32_i8", 0, S, o(0x0e), o(0x0e)),
    e(0x1a, "s_sext_i32_i16", 0, S, o(0x0f), o(0x0f)),
    e(0x1b, "s_bitset0_b32", READS_DST, S, o(0x10), o(0x10)),
    e(0x1c, "s_bitset0_b64", D2 | READS_DST, S, o(0x11), o(0x11)),
    e(0x1d, "s_bitset1_b32", READS_DST, S, o(0x12), o(0x12)),
    e(0x1e, "s_bitset1_b64", D2 | READS_DST, S, o(0x13), o(0x13)),
    e(
        0x1f,
        "s_getpc_b64",
        D2 | NO_SRC | PC_RELATIVE,
        S,
        o(0x47),
        o(0x47),
    ),
    e(
        0x20,
        "s_setpc_b64",
        NO_DST | S0_2 | JUMP,
        S,
        o(0x48),
        o(0x48),
    ),
    e(
        0x21,
        "s_swappc_b64",
        D2 | S0_2 | CALL | PC_RELATIVE,
        S,
        o(0x49),
        o(0x49),
    ),
    e(0x22, "s_rfe_b64", NO_DST | S0_2 | JUMP, S, o(0x4a), o(0x4a)),
    e(0x24, "s_and_saveexec_b64", SAVEEXEC64, S, o(0x21), o(0x21)),
    e(0x25, "s_or_saveexec_b64", SAVEEXEC64, S, o(0x23), o(0x23)),
    e(0x26, "s_xor_saveexec_b64", SAVEEXEC64, S, o(0x25), o(0x25)),
    e(
        0x27,
        "s_andn2_saveexec_b64",
        SAVEEXEC64,
        S,
        o(0x31),
        o(0x31),
    ),
    e(0x28, "s_orn2_saveexec_b64", SAVEEXEC64, S, o(0x33), o(0x33)),
    e(0x29, "s_nand_saveexec_b64", SAVEEXEC64, S, o(0x27), o(0x27)),
    e(0x2a, "s_nor_saveexec_b64", SAVEEXEC64, S, o(0x29), o(0x29)),
    e(0x2b, "s_xnor_saveexec_b64", SAVEEXEC64, S, o(0x2b), o(0x2b)),
    e(0x2c, "s_quadmask_b32", 0, S, o(0x1a), o(0x1a)),
    e(0x2d, "s_quadmask_b64", D2 | S0_2, S, o(0x1b), o(0x1b)),
    e(
        0x2e,
        "s_movrels_b32",
        READS_M0 | RELATIVE,
        S,
        o(0x40),
        o(0x40),
    ),
    e(
        0x2f,
        "s_movrels_b64",
        D2 | S0_2 | READS_M0 | RELATIVE,
        S,
        o(0x41),
        o(0x41),
    ),
    e(
        0x30,
        "s_movreld_b32",
        READS_M0 | RELATIVE,
        S,
        o(0x42),
        o(0x42),
    ),
    e(
        0x31,
        "s_movreld_b64",
        D2 | S0_2 | READS_M0 | RELATIVE,
        S,
        o(0x43),
        o(0x43),
    ),
    e(0x34, "s_abs_i32", 0, S, o(0x15), o(0x15)),
    e(
        0x37,
        "s_andn1_saveexec_b64",
        SAVEEXEC64,
        S,
        o(0x2d),
        o(0x2d),
    ),
    e(0x38, "s_orn1_saveexec_b64", SAVEEXEC64, S, o(0x2f), o(0x2f)),
    e(0x39, "s_andn1_wrexec_b64", SAVEEXEC64, S, o(0x35), o(0x35)),
    e(0x3a, "s_andn2_wrexec_b64", SAVEEXEC64, S, o(0x37), o(0x37)),
    e(0x3b, "s_bitreplicate_b64_b32", D2, S, o(0x14), o(0x14)),
    e(0x3c, "s_and_saveexec_b32", WRITES_EXEC, S, o(0x20), o(0x20)),
    e(0x3d, "s_or_saveexec_b32", WRITES_EXEC, S, o(0x22), o(0x22)),
    e(0x3e, "s_xor_saveexec_b32", WRITES_EXEC, S, o(0x24), o(0x24)),
    e(
        0x3f,
        "s_andn2_saveexec_b32",
        WRITES_EXEC,
        S,
        o(0x30),
        o(0x30),
    ),
    e(
        0x40,
        "s_orn2_saveexec_b32",
        WRITES_EXEC,
        S,
        o(0x32),
        o(0x32),
    ),
    e(
        0x41,
        "s_nand_saveexec_b32",
        WRITES_EXEC,
        S,
        o(0x26),
        o(0x26),
    ),
    e(0x42, "s_nor_saveexec_b32", WRITES_EXEC, S, o(0x28), o(0x28)),
    e(
        0x43,
        "s_xnor_saveexec_b32",
        WRITES_EXEC,
        S,
        o(0x2a),
        o(0x2a),
    ),
    e(
        0x44,
        "s_andn1_saveexec_b32",
        WRITES_EXEC,
        S,
        o(0x2c),
        o(0x2c),
    ),
    e(
        0x45,
        "s_orn1_saveexec_b32",
        WRITES_EXEC,
        S,
        o(0x2e),
        o(0x2e),
    ),
    e(0x46, "s_andn1_wrexec_b32", WRITES_EXEC, S, o(0x34), o(0x34)),
    e(0x47, "s_andn2_wrexec_b32", WRITES_EXEC, S, o(0x36), o(0x36)),
    e(
        0x49,
        "s_movrelsd_2_b32",
        READS_M0 | RELATIVE,
        S,
        o(0x44),
        o(0x44),
    ),
];

const L64: u64 = D2 | S0_2 | S1_2;

pub static SOP2: &[OpInfo] = &[
    e(0x00, "s_add_u32", 0, S, o(0x00), o(0x00)),
    e(0x01, "s_sub_u32", 0, S, o(0x01), o(0x01)),
    e(0x02, "s_add_i32", 0, S, o(0x02), o(0x02)),
    e(0x03, "s_sub_i32", 0, S, o(0x03), o(0x03)),
    e(0x04, "s_addc_u32", 0, S, o(0x04), o(0x04)),
    e(0x05, "s_subb_u32", 0, S, o(0x05), o(0x05)),
    e(0x06, "s_min_i32", 0, S, o(0x12), o(0x12)),
    e(0x07, "s_min_u32", 0, S, o(0x13), o(0x13)),
    e(0x08, "s_max_i32", 0, S, o(0x14), o(0x14)),
    e(0x09, "s_max_u32", 0, S, o(0x15), o(0x15)),
    e(0x0a, "s_cselect_b32", 0, S, o(0x30), o(0x30)),
    e(0x0b, "s_cselect_b64", L64, S, o(0x31), o(0x31)),
    e(0x0e, "s_and_b32", 0, S, o(0x16), o(0x16)),
    e(0x0f, "s_and_b64", L64, S, o(0x17), o(0x17)),
    e(0x10, "s_or_b32", 0, S, o(0x18), o(0x18)),
    e(0x11, "s_or_b64", L64, S, o(0x19), o(0x19)),
    e(0x12, "s_xor_b32", 0, S, o(0x1a), o(0x1a)),
    e(0x13, "s_xor_b64", L64, S, o(0x1b), o(0x1b)),
    e(0x14, "s_andn2_b32", 0, S, o(0x22), o(0x22)),
    e(0x15, "s_andn2_b64", L64, S, o(0x23), o(0x23)),
    e(0x16, "s_orn2_b32", 0, S, o(0x24), o(0x24)),
    e(0x17, "s_orn2_b64", L64, S, o(0x25), o(0x25)),
    e(0x18, "s_nand_b32", 0, S, o(0x1c), o(0x1c)),
    e(0x19, "s_nand_b64", L64, S, o(0x1d), o(0x1d)),
    e(0x1a, "s_nor_b32", 0, S, o(0x1e), o(0x1e)),
    e(0x1b, "s_nor_b64", L64, S, o(0x1f), o(0x1f)),
    e(0x1c, "s_xnor_b32", 0, S, o(0x20), o(0x20)),
    e(0x1d, "s_xnor_b64", L64, S, o(0x21), o(0x21)),
    e(0x1e, "s_lshl_b32", 0, S, o(0x08), o(0x08)),
    e(0x1f, "s_lshl_b64", D2 | S0_2, S, o(0x09), o(0x09)),
    e(0x20, "s_lshr_b32", 0, S, o(0x0a), o(0x0a)),
    e(0x21, "s_lshr_b64", D2 | S0_2, S, o(0x0b), o(0x0b)),
    e(0x22, "s_ashr_i32", 0, S, o(0x0c), o(0x0c)),
    e(0x23, "s_ashr_i64", D2 | S0_2, S, o(0x0d), o(0x0d)),
    e(0x24, "s_bfm_b32", 0, S, o(0x2a), o(0x2a)),
    e(0x25, "s_bfm_b64", D2, S, o(0x2b), o(0x2b)),
    e(0x26, "s_mul_i32", 0, S, o(0x2c), o(0x2c)),
    e(0x27, "s_bfe_u32", 0, S, o(0x26), o(0x26)),
    e(0x28, "s_bfe_i32", 0, S, o(0x27), o(0x27)),
    e(0x29, "s_bfe_u64", D2 | S0_2, S, o(0x28), o(0x28)),
    e(0x2a, "s_bfe_i64", D2 | S0_2, S, o(0x29), o(0x29)),
    e(0x2c, "s_absdiff_i32", 0, S, o(0x06), o(0x06)),
    e(0x2e, "s_lshl1_add_u32", 0, S, o(0x0e), o(0x0e)),
    e(0x2f, "s_lshl2_add_u32", 0, S, o(0x0f), o(0x0f)),
    e(0x30, "s_lshl3_add_u32", 0, S, o(0x10), o(0x10)),
    e(0x31, "s_lshl4_add_u32", 0, S, o(0x11), o(0x11)),
    e(0x32, "s_pack_ll_b32_b16", 0, S, o(0x32), o(0x32)),
    e(0x33, "s_pack_lh_b32_b16", 0, S, o(0x33), o(0x33)),
    e(0x34, "s_pack_hh_b32_b16", 0, S, o(0x34), o(0x34)),
    e(0x35, "s_mul_hi_u32", 0, S, o(0x2d), o(0x2d)),
    e(0x36, "s_mul_hi_i32", 0, S, o(0x2e), o(0x2e)),
];

const CMPK: u64 = NO_DST | READS_DST;
const WAITK: u64 = NO_DST | IMM_U;

pub static SOPK: &[OpInfo] = &[
    e(0x00, "s_movk_i32", 0, S, o(0x00), o(0x00)),
    e(0x01, "s_version", NO_DST | IMM_U, S, o(0x01), o(0x01)),
    e(0x02, "s_cmovk_i32", READS_DST, S, o(0x02), o(0x02)),
    e(0x03, "s_cmpk_eq_i32", CMPK, S, o(0x03), r(Rule::Cmpk(0x00))),
    e(0x04, "s_cmpk_lg_i32", CMPK, S, o(0x04), r(Rule::Cmpk(0x01))),
    e(0x05, "s_cmpk_gt_i32", CMPK, S, o(0x05), r(Rule::Cmpk(0x02))),
    e(0x06, "s_cmpk_ge_i32", CMPK, S, o(0x06), r(Rule::Cmpk(0x03))),
    e(0x07, "s_cmpk_lt_i32", CMPK, S, o(0x07), r(Rule::Cmpk(0x04))),
    e(0x08, "s_cmpk_le_i32", CMPK, S, o(0x08), r(Rule::Cmpk(0x05))),
    e(
        0x09,
        "s_cmpk_eq_u32",
        CMPK | IMM_U,
        S,
        o(0x09),
        r(Rule::Cmpk(0x06)),
    ),
    e(
        0x0a,
        "s_cmpk_lg_u32",
        CMPK | IMM_U,
        S,
        o(0x0a),
        r(Rule::Cmpk(0x07)),
    ),
    e(
        0x0b,
        "s_cmpk_gt_u32",
        CMPK | IMM_U,
        S,
        o(0x0b),
        r(Rule::Cmpk(0x08)),
    ),
    e(
        0x0c,
        "s_cmpk_ge_u32",
        CMPK | IMM_U,
        S,
        o(0x0c),
        r(Rule::Cmpk(0x09)),
    ),
    e(
        0x0d,
        "s_cmpk_lt_u32",
        CMPK | IMM_U,
        S,
        o(0x0d),
        r(Rule::Cmpk(0x0a)),
    ),
    e(
        0x0e,
        "s_cmpk_le_u32",
        CMPK | IMM_U,
        S,
        o(0x0e),
        r(Rule::Cmpk(0x0b)),
    ),
    e(0x0f, "s_addk_i32", READS_DST, S, o(0x0f), o(0x0f)),
    e(0x10, "s_mulk_i32", READS_DST, S, o(0x10), o(0x10)),
    e(
        0x12,
        "s_getreg_b32",
        IMM_U,
        S,
        r(Rule::Hwreg(0x11)),
        r(Rule::Hwreg(0x11)),
    ),
    e(
        0x13,
        "s_setreg_b32",
        CMPK | IMM_U,
        S,
        r(Rule::Hwreg(0x12)),
        r(Rule::Hwreg(0x12)),
    ),
    e(
        0x15,
        "s_setreg_imm32_b32",
        NO_DST | IMM_U,
        S,
        r(Rule::Hwreg(0x13)),
        r(Rule::Hwreg(0x13)),
    ),
    e(0x16, "s_call_b64", D2 | CALL, S, o(0x14), o(0x14)),
    e(
        0x17,
        "s_waitcnt_vscnt",
        WAITK | READS_DST,
        S,
        o(0x18),
        r(Rule::WaitcntCounter),
    ),
    e(
        0x18,
        "s_waitcnt_vmcnt",
        WAITK | READS_DST,
        S,
        o(0x19),
        r(Rule::WaitcntCounter),
    ),
    e(
        0x19,
        "s_waitcnt_expcnt",
        WAITK | READS_DST,
        S,
        o(0x1a),
        r(Rule::WaitcntCounter),
    ),
    e(
        0x1a,
        "s_waitcnt_lgkmcnt",
        WAITK | READS_DST,
        S,
        o(0x1b),
        r(Rule::WaitcntCounter),
    ),
    e(
        0x1b,
        "s_subvector_loop_begin",
        COND_BRANCH | READS_DST,
        S,
        o(0x16),
        N,
    ),
    e(
        0x1c,
        "s_subvector_loop_end",
        COND_BRANCH | READS_DST,
        S,
        o(0x17),
        N,
    ),
];

pub static SOPC: &[OpInfo] = &[
    e(0x00, "s_cmp_eq_i32", 0, S, S, S),
    e(0x01, "s_cmp_lg_i32", 0, S, S, S),
    e(0x02, "s_cmp_gt_i32", 0, S, S, S),
    e(0x03, "s_cmp_ge_i32", 0, S, S, S),
    e(0x04, "s_cmp_lt_i32", 0, S, S, S),
    e(0x05, "s_cmp_le_i32", 0, S, S, S),
    e(0x06, "s_cmp_eq_u32", 0, S, S, S),
    e(0x07, "s_cmp_lg_u32", 0, S, S, S),
    e(0x08, "s_cmp_gt_u32", 0, S, S, S),
    e(0x09, "s_cmp_ge_u32", 0, S, S, S),
    e(0x0a, "s_cmp_lt_u32", 0, S, S, S),
    e(0x0b, "s_cmp_le_u32", 0, S, S, S),
    e(0x0c, "s_bitcmp0_b32", 0, S, S, S),
    e(0x0d, "s_bitcmp1_b32", 0, S, S, S),
    e(0x0e, "s_bitcmp0_b64", S0_2, S, S, S),
    e(0x0f, "s_bitcmp1_b64", S0_2, S, S, S),
    e(0x12, "s_cmp_eq_u64", S0_2 | S1_2, S, o(0x10), o(0x10)),
    e(0x13, "s_cmp_lg_u64", S0_2 | S1_2, S, o(0x11), o(0x11)),
];

pub const SOPP_ENDPGM: u16 = 0x01;
pub const SOPP_BRANCH: u16 = 0x02;
pub const SOPP_WAITCNT: u16 = 0x0c;
pub const SOPP_CODE_END: u16 = 0x1f;
pub const SOPP_WAITCNT_DEPCTR: u16 = 0x23;

pub static SOPP: &[OpInfo] = &[
    e(0x00, "s_nop", 0, S, S, S),
    e(0x01, "s_endpgm", END, S, o(0x30), o(0x30)),
    e(0x02, "s_branch", BRANCH, S, o(0x20), o(0x20)),
    e(0x03, "s_wakeup", 0, S, o(0x34), o(0x34)),
    e(0x04, "s_cbranch_scc0", COND_BRANCH, S, o(0x21), o(0x21)),
    e(0x05, "s_cbranch_scc1", COND_BRANCH, S, o(0x22), o(0x22)),
    e(
        0x06,
        "s_cbranch_vccz",
        COND_BRANCH | READS_VCC,
        S,
        o(0x23),
        o(0x23),
    ),
    e(
        0x07,
        "s_cbranch_vccnz",
        COND_BRANCH | READS_VCC,
        S,
        o(0x24),
        o(0x24),
    ),
    e(
        0x08,
        "s_cbranch_execz",
        COND_BRANCH | READS_EXEC,
        S,
        o(0x25),
        o(0x25),
    ),
    e(
        0x09,
        "s_cbranch_execnz",
        COND_BRANCH | READS_EXEC,
        S,
        o(0x26),
        o(0x26),
    ),
    e(0x0a, "s_barrier", 0, S, o(0x3d), r(Rule::Barrier)),
    e(0x0b, "s_setkill", 0, S, o(0x01), o(0x01)),
    e(0x0c, "s_waitcnt", 0, S, r(Rule::Waitcnt), r(Rule::Waitcnt)),
    e(0x0d, "s_sethalt", 0, S, o(0x02), o(0x02)),
    e(0x0e, "s_sleep", 0, S, o(0x03), o(0x03)),
    e(0x0f, "s_setprio", 0, S, o(0x35), o(0x35)),
    e(
        0x10,
        "s_sendmsg",
        READS_M0,
        S,
        r(Rule::Sendmsg(0x36)),
        r(Rule::Sendmsg(0x36)),
    ),
    e(
        0x11,
        "s_sendmsghalt",
        READS_M0,
        S,
        r(Rule::Sendmsg(0x37)),
        r(Rule::Sendmsg(0x37)),
    ),
    e(0x12, "s_trap", 0, S, o(0x10), o(0x10)),
    e(0x13, "s_icache_inv", 0, S, o(0x3c), o(0x3c)),
    e(0x14, "s_incperflevel", 0, S, o(0x38), o(0x38)),
    e(0x15, "s_decperflevel", 0, S, o(0x39), o(0x39)),
    e(0x16, "s_ttracedata", READS_M0, S, o(0x3a), o(0x3a)),
    // gfx12 has no conditional-debug branches. They branch only while a debugger sets the
    // wave's CDBG bits, so outside a debugger (as `exec` models them) they are never taken.
    e(0x17, "s_cbranch_cdbgsys", COND_BRANCH, S, o(0x27), D),
    e(0x18, "s_cbranch_cdbguser", COND_BRANCH, S, o(0x28), D),
    e(
        0x19,
        "s_cbranch_cdbgsys_or_user",
        COND_BRANCH,
        S,
        o(0x29),
        D,
    ),
    e(
        0x1a,
        "s_cbranch_cdbgsys_and_user",
        COND_BRANCH,
        S,
        o(0x2a),
        D,
    ),
    e(0x1b, "s_endpgm_saved", END, S, o(0x31), o(0x31)),
    e(0x1e, "s_endpgm_ordered_ps_done", END, S, o(0x32), N),
    e(0x1f, "s_code_end", END, S, S, S),
    e(0x20, "s_inst_prefetch", 0, S, D, D),
    e(0x21, "s_clause", 0, S, D, D),
    e(0x22, "s_wait_idle", 0, S, o(0x0a), o(0x0a)),
    e(0x23, "s_waitcnt_depctr", 0, S, o(0x08), o(0x08)),
    e(0x24, "s_round_mode", 0, S, o(0x11), o(0x11)),
    e(0x25, "s_denorm_mode", 0, S, o(0x12), o(0x12)),
    e(0x28, "s_ttracedata_imm", 0, S, o(0x3b), o(0x3b)),
];

pub const VOP1_NOP: u16 = 0x00;
pub const VOP1_MOV_B32: u16 = 0x01;

const TR: u64 = TRANS;
const T16T: u64 = T16 | TRANS;
const F64: u64 = D2 | S0_2;
const MOVREL: u64 = READS_M0 | RELATIVE;

pub static VOP1: &[OpInfo] = &[
    e(0x00, "v_nop", NO_OPERANDS, S, S, S),
    e(0x01, "v_mov_b32", 0, S, S, S),
    e(0x02, "v_readfirstlane_b32", SDST | NO_VOP3, S, S, S),
    e(0x03, "v_cvt_i32_f64", S0_2, S, S, S),
    e(0x04, "v_cvt_f64_i32", D2, S, S, S),
    e(0x05, "v_cvt_f32_i32", 0, S, S, S),
    e(0x06, "v_cvt_f32_u32", 0, S, S, S),
    e(0x07, "v_cvt_u32_f32", 0, S, S, S),
    e(0x08, "v_cvt_i32_f32", 0, S, S, S),
    e(0x0a, "v_cvt_f16_f32", T16, S, S, S),
    e(0x0b, "v_cvt_f32_f16", T16, S, S, S),
    e(0x0c, "v_cvt_rpi_i32_f32", 0, S, S, S),
    e(0x0d, "v_cvt_flr_i32_f32", 0, S, S, S),
    e(0x0e, "v_cvt_off_f32_i4", 0, S, S, S),
    e(0x0f, "v_cvt_f32_f64", S0_2, S, S, S),
    e(0x10, "v_cvt_f64_f32", D2, S, S, S),
    e(0x11, "v_cvt_f32_ubyte0", 0, S, S, S),
    e(0x12, "v_cvt_f32_ubyte1", 0, S, S, S),
    e(0x13, "v_cvt_f32_ubyte2", 0, S, S, S),
    e(0x14, "v_cvt_f32_ubyte3", 0, S, S, S),
    e(0x15, "v_cvt_u32_f64", S0_2, S, S, S),
    e(0x16, "v_cvt_f64_u32", D2, S, S, S),
    e(0x17, "v_trunc_f64", F64, S, S, S),
    e(0x18, "v_ceil_f64", F64, S, S, S),
    e(0x19, "v_rndne_f64", F64, S, S, S),
    e(0x1a, "v_floor_f64", F64, S, S, S),
    e(0x1b, "v_pipeflush", NO_OPERANDS, S, S, S),
    e(0x20, "v_fract_f32", 0, S, S, S),
    e(0x21, "v_trunc_f32", 0, S, S, S),
    e(0x22, "v_ceil_f32", 0, S, S, S),
    e(0x23, "v_rndne_f32", 0, S, S, S),
    e(0x24, "v_floor_f32", 0, S, S, S),
    e(0x25, "v_exp_f32", TR, S, S, S),
    e(0x27, "v_log_f32", TR, S, S, S),
    e(0x2a, "v_rcp_f32", TR, S, S, S),
    e(0x2b, "v_rcp_iflag_f32", TR, S, S, S),
    e(0x2e, "v_rsq_f32", TR, S, S, S),
    e(0x2f, "v_rcp_f64", F64 | TR, S, S, S),
    e(0x31, "v_rsq_f64", F64 | TR, S, S, S),
    e(0x33, "v_sqrt_f32", TR, S, S, S),
    e(0x34, "v_sqrt_f64", F64 | TR, S, S, S),
    e(0x35, "v_sin_f32", TR, S, S, S),
    e(0x36, "v_cos_f32", TR, S, S, S),
    e(0x37, "v_not_b32", 0, S, S, S),
    e(0x38, "v_bfrev_b32", 0, S, S, S),
    e(0x39, "v_ffbh_u32", 0, S, S, S),
    e(0x3a, "v_ffbl_b32", 0, S, S, S),
    e(0x3b, "v_ffbh_i32", 0, S, S, S),
    e(0x3c, "v_frexp_exp_i32_f64", S0_2, S, S, S),
    e(0x3d, "v_frexp_mant_f64", F64, S, S, S),
    e(0x3e, "v_fract_f64", F64, S, S, S),
    e(0x3f, "v_frexp_exp_i32_f32", 0, S, S, S),
    e(0x40, "v_frexp_mant_f32", 0, S, S, S),
    e(0x41, "v_clrexcp", NO_OPERANDS, S, D, D),
    e(0x42, "v_movreld_b32", MOVREL, S, S, S),
    e(0x43, "v_movrels_b32", MOVREL, S, S, S),
    e(0x44, "v_movrelsd_b32", MOVREL, S, S, S),
    e(0x48, "v_movrelsd_2_b32", MOVREL, S, S, S),
    e(0x50, "v_cvt_f16_u16", T16, S, S, S),
    e(0x51, "v_cvt_f16_i16", T16, S, S, S),
    e(0x52, "v_cvt_u16_f16", T16, S, S, S),
    e(0x53, "v_cvt_i16_f16", T16, S, S, S),
    e(0x54, "v_rcp_f16", T16T, S, S, S),
    e(0x55, "v_sqrt_f16", T16T, S, S, S),
    e(0x56, "v_rsq_f16", T16T, S, S, S),
    e(0x57, "v_log_f16", T16T, S, S, S),
    e(0x58, "v_exp_f16", T16T, S, S, S),
    e(0x59, "v_frexp_mant_f16", T16, S, S, S),
    e(0x5a, "v_frexp_exp_i16_f16", T16, S, S, S),
    e(0x5b, "v_floor_f16", T16, S, S, S),
    e(0x5c, "v_ceil_f16", T16, S, S, S),
    e(0x5d, "v_trunc_f16", T16, S, S, S),
    e(0x5e, "v_rndne_f16", T16, S, S, S),
    e(0x5f, "v_fract_f16", T16, S, S, S),
    e(0x60, "v_sin_f16", T16T, S, S, S),
    e(0x61, "v_cos_f16", T16T, S, S, S),
    e(0x62, "v_sat_pk_u8_i16", T16, S, S, S),
    e(0x63, "v_cvt_norm_i16_f16", T16, S, S, S),
    e(0x64, "v_cvt_norm_u16_f16", T16, S, S, S),
    e(0x65, "v_swap_b32", SWAP | NO_VOP3, S, S, S),
    e(0x68, "v_swaprel_b32", SWAP | MOVREL | NO_VOP3, S, S, S),
];

pub const VOP2_CNDMASK: u16 = 0x01;
pub const VOP2_ADD_F32: u16 = 0x03;
pub const VOP2_MUL_LEGACY_F32: u16 = 0x07;
pub const VOP2_MUL_F32: u16 = 0x08;
pub const VOP2_LSHLREV_B32: u16 = 0x1a;
pub const VOP2_AND_B32: u16 = 0x1b;
pub const VOP2_OR_B32: u16 = 0x1c;
pub const VOP2_XOR_B32: u16 = 0x1d;
pub const VOP2_FMAC_F32: u16 = 0x2b;
pub const VOP2_FMAMK_F32: u16 = 0x2c;
pub const VOP2_FMAAK_F32: u16 = 0x2d;
pub const VOP2_MAC_LEGACY_F32: u16 = 0x06;

const CARRY: u64 = CARRY_IN | CARRY_OUT;
const ACC: u64 = READS_DST;

pub static VOP2: &[OpInfo] = &[
    e(0x01, "v_cndmask_b32", CARRY_IN, S, S, S),
    e(0x02, "v_dot2c_f32_f16", ACC | NO_VOP3, S, S, r(Rule::Dot2c)),
    e(0x03, "v_add_f32", 0, S, S, S),
    e(0x04, "v_sub_f32", 0, S, S, S),
    e(0x05, "v_subrev_f32", 0, S, S, S),
    e(
        0x06,
        "v_mac_legacy_f32",
        ACC,
        r(Rule::MacLegacy),
        r(Rule::MacLegacy),
        r(Rule::MacLegacy),
    ),
    e(0x07, "v_mul_legacy_f32", 0, S, S, S),
    e(0x08, "v_mul_f32", 0, S, S, S),
    e(0x09, "v_mul_i32_i24", 0, S, S, S),
    e(0x0a, "v_mul_hi_i32_i24", 0, S, S, S),
    e(0x0b, "v_mul_u32_u24", 0, S, S, S),
    e(0x0c, "v_mul_hi_u32_u24", 0, S, S, S),
    e(
        0x0d,
        "v_dot4c_i32_i8",
        ACC | NO_VOP3,
        S,
        r(Rule::Dot4c),
        r(Rule::Dot4c),
    ),
    // gfx12 replaced the float min/max family (VOP2 f32/f16, VOP3 min3/max3/med3 and f64,
    // VOP3P packed f16) with the IEEE 754-2019 minimumNumber/maximumNumber `*_num_*` forms.
    // They select like the gfx10 instructions for every input that is
    // not a signaling NaN: -0 < +0, a quiet NaN against a number gives the number, and two
    // quiet NaNs give src0. Signaling NaNs, which arithmetic never produces (they only arrive as
    // raw memory bits), are the one difference: against a number, gfx10 in IEEE mode returns
    // the quieted NaN where gfx12 returns the number; against another NaN, gfx10 outside IEEE
    // mode returns src0's bits unquieted where gfx12 quiets them.
    e(0x0f, "v_min_f32", 0, S, S, o(0x15)),
    e(0x10, "v_max_f32", 0, S, S, o(0x16)),
    e(0x11, "v_min_i32", 0, S, S, S),
    e(0x12, "v_max_i32", 0, S, S, S),
    e(0x13, "v_min_u32", 0, S, S, S),
    e(0x14, "v_max_u32", 0, S, S, S),
    e(0x16, "v_lshrrev_b32", 0, S, o(0x19), o(0x19)),
    e(0x18, "v_ashrrev_i32", 0, S, o(0x1a), o(0x1a)),
    e(0x1a, "v_lshlrev_b32", 0, S, o(0x18), o(0x18)),
    e(0x1b, "v_and_b32", 0, S, S, S),
    e(0x1c, "v_or_b32", 0, S, S, S),
    e(0x1d, "v_xor_b32", 0, S, S, S),
    e(0x1e, "v_xnor_b32", 0, S, S, S),
    e(
        0x1f,
        "v_mac_f32",
        ACC,
        r(Rule::Mac),
        r(Rule::Mac),
        r(Rule::Mac),
    ),
    e(
        0x20,
        "v_madmk_f32",
        LITK | NO_VOP3,
        r(Rule::Madmk),
        r(Rule::Madmk),
        r(Rule::Madmk),
    ),
    e(
        0x21,
        "v_madak_f32",
        LITK | NO_VOP3,
        r(Rule::Madak),
        r(Rule::Madak),
        r(Rule::Madak),
    ),
    e(0x25, "v_add_nc_u32", 0, S, S, S),
    e(0x26, "v_sub_nc_u32", 0, S, S, S),
    e(0x27, "v_subrev_nc_u32", 0, S, S, S),
    e(0x28, "v_add_co_ci_u32", CARRY, S, o(0x20), o(0x20)),
    e(0x29, "v_sub_co_ci_u32", CARRY, S, o(0x21), o(0x21)),
    e(0x2a, "v_subrev_co_ci_u32", CARRY, S, o(0x22), o(0x22)),
    e(0x2b, "v_fmac_f32", ACC, S, S, S),
    e(0x2c, "v_fmamk_f32", LITK | NO_VOP3, S, S, S),
    e(0x2d, "v_fmaak_f32", LITK | NO_VOP3, S, S, S),
    e(0x2f, "v_cvt_pkrtz_f16_f32", 0, S, S, S),
    e(0x32, "v_add_f16", T16, S, S, S),
    e(0x33, "v_sub_f16", T16, S, S, S),
    e(0x34, "v_subrev_f16", T16, S, S, S),
    e(0x35, "v_mul_f16", T16, S, S, S),
    e(0x36, "v_fmac_f16", T16 | ACC, S, S, S),
    e(0x37, "v_fmamk_f16", T16 | LITK | NO_VOP3, S, S, S),
    e(0x38, "v_fmaak_f16", T16 | LITK | NO_VOP3, S, S, S),
    e(0x39, "v_max_f16", T16, S, S, o(0x31)),
    e(0x3a, "v_min_f16", T16, S, S, o(0x30)),
    e(0x3b, "v_ldexp_f16", T16, S, S, S),
    e(0x3c, "v_pk_fmac_f16", ACC | NO_VOP3, S, S, S),
];

pub const VOP3_MAD_LEGACY_F32: u16 = 0x140;
pub const VOP3_MAD_F32: u16 = 0x141;
pub const VOP3_BFE_U32: u16 = 0x148;
pub const VOP3_BFE_I32: u16 = 0x149;
pub const VOP3_BFI_B32: u16 = 0x14a;
pub const VOP3_FMA_F32: u16 = 0x14b;
pub const VOP3_INTERP_P1_F32: u16 = 0x200;
pub const VOP3_INTERP_P2_F32: u16 = 0x201;
pub const VOP3_INTERP_MOV_F32: u16 = 0x202;

const T2: u64 = SRC2;
const T3: u64 = SRC3;
const INTERP: u64 = SRC3 | READS_M0;

pub static VOP3: &[OpInfo] = &[
    e(
        0x140,
        "v_mad_legacy_f32",
        T3,
        r(Rule::MadLegacy),
        r(Rule::MadLegacy),
        r(Rule::MadLegacy),
    ),
    e(
        0x141,
        "v_mad_f32",
        T3,
        r(Rule::Mad),
        r(Rule::Mad),
        r(Rule::Mad),
    ),
    e(0x142, "v_mad_i32_i24", T3, S, o(0x20a), o(0x20a)),
    e(0x143, "v_mad_u32_u24", T3, S, o(0x20b), o(0x20b)),
    e(0x144, "v_cubeid_f32", T3, S, o(0x20c), o(0x20c)),
    e(0x145, "v_cubesc_f32", T3, S, o(0x20d), o(0x20d)),
    e(0x146, "v_cubetc_f32", T3, S, o(0x20e), o(0x20e)),
    e(0x147, "v_cubema_f32", T3, S, o(0x20f), o(0x20f)),
    e(0x148, "v_bfe_u32", T3, S, o(0x210), o(0x210)),
    e(0x149, "v_bfe_i32", T3, S, o(0x211), o(0x211)),
    e(0x14a, "v_bfi_b32", T3, S, o(0x212), o(0x212)),
    e(0x14b, "v_fma_f32", T3, S, o(0x213), o(0x213)),
    e(0x14c, "v_fma_f64", T3 | W64, S, o(0x214), o(0x214)),
    e(0x14d, "v_lerp_u8", T3, S, o(0x215), o(0x215)),
    e(0x14e, "v_alignbit_b32", T3, S, o(0x216), o(0x216)),
    e(0x14f, "v_alignbyte_b32", T3, S, o(0x217), o(0x217)),
    e(0x150, "v_mullit_f32", T3, S, o(0x218), o(0x218)),
    e(0x151, "v_min3_f32", T3, S, o(0x219), o(0x229)),
    e(0x152, "v_min3_i32", T3, S, o(0x21a), o(0x21a)),
    e(0x153, "v_min3_u32", T3, S, o(0x21b), o(0x21b)),
    e(0x154, "v_max3_f32", T3, S, o(0x21c), o(0x22a)),
    e(0x155, "v_max3_i32", T3, S, o(0x21d), o(0x21d)),
    e(0x156, "v_max3_u32", T3, S, o(0x21e), o(0x21e)),
    e(0x157, "v_med3_f32", T3, S, o(0x21f), o(0x231)),
    e(0x158, "v_med3_i32", T3, S, o(0x220), o(0x220)),
    e(0x159, "v_med3_u32", T3, S, o(0x221), o(0x221)),
    e(0x15a, "v_sad_u8", T3, S, o(0x222), o(0x222)),
    e(0x15b, "v_sad_hi_u8", T3, S, o(0x223), o(0x223)),
    e(0x15c, "v_sad_u16", T3, S, o(0x224), o(0x224)),
    e(0x15d, "v_sad_u32", T3, S, o(0x225), o(0x225)),
    e(0x15e, "v_cvt_pk_u8_f32", T3, S, o(0x226), o(0x226)),
    e(0x15f, "v_div_fixup_f32", T3, S, o(0x227), o(0x227)),
    e(0x160, "v_div_fixup_f64", T3 | W64, S, o(0x228), o(0x228)),
    e(0x164, "v_add_f64", T2 | W64, S, o(0x327), o(0x102)),
    e(0x165, "v_mul_f64", T2 | W64, S, o(0x328), o(0x106)),
    e(0x166, "v_min_f64", T2 | W64, S, o(0x329), o(0x10d)),
    e(0x167, "v_max_f64", T2 | W64, S, o(0x32a), o(0x10e)),
    e(0x168, "v_ldexp_f64", T2 | D2 | S0_2, S, o(0x32b), o(0x32b)),
    e(0x169, "v_mul_lo_u32", T2, S, o(0x32c), o(0x32c)),
    e(0x16a, "v_mul_hi_u32", T2, S, o(0x32d), o(0x32d)),
    e(0x16b, "v_mul_lo_i32", T2, o(0x169), o(0x32c), o(0x32c)),
    e(0x16c, "v_mul_hi_i32", T2, S, o(0x32e), o(0x32e)),
    e(0x16d, "v_div_scale_f32", T3 | VOP3B, S, o(0x2fc), o(0x2fc)),
    e(
        0x16e,
        "v_div_scale_f64",
        T3 | VOP3B | W64,
        S,
        o(0x2fd),
        o(0x2fd),
    ),
    e(
        0x16f,
        "v_div_fmas_f32",
        T3 | READS_VCC,
        S,
        o(0x237),
        o(0x237),
    ),
    e(
        0x170,
        "v_div_fmas_f64",
        T3 | W64 | READS_VCC,
        S,
        o(0x238),
        o(0x238),
    ),
    e(0x171, "v_msad_u8", T3, S, o(0x239), o(0x239)),
    e(
        0x172,
        "v_qsad_pk_u16_u8",
        T3 | D2 | S0_2 | S2_2,
        S,
        o(0x23a),
        o(0x23a),
    ),
    e(
        0x173,
        "v_mqsad_pk_u16_u8",
        T3 | D2 | S0_2 | S2_2,
        S,
        o(0x23b),
        o(0x23b),
    ),
    e(
        0x174,
        "v_trig_preop_f64",
        T2 | D2 | S0_2,
        S,
        o(0x32f),
        o(0x32f),
    ),
    e(
        0x175,
        "v_mqsad_u32_u8",
        T3 | D2 | S0_2 | S2_2,
        S,
        o(0x23d),
        o(0x23d),
    ),
    e(
        0x176,
        "v_mad_u64_u32",
        T3 | VOP3B | D2 | S2_2,
        S,
        o(0x2fe),
        o(0x2fe),
    ),
    e(
        0x177,
        "v_mad_i64_i32",
        T3 | VOP3B | D2 | S2_2,
        S,
        o(0x2ff),
        o(0x2ff),
    ),
    e(0x178, "v_xor3_b32", T3, S, o(0x240), o(0x240)),
    e(
        0x200,
        "v_interp_p1_f32",
        INTERP,
        S,
        r(Rule::Interp),
        r(Rule::Interp),
    ),
    e(
        0x201,
        "v_interp_p2_f32",
        INTERP | ACC,
        S,
        r(Rule::Interp),
        r(Rule::Interp),
    ),
    e(
        0x202,
        "v_interp_mov_f32",
        INTERP,
        S,
        r(Rule::Interp),
        r(Rule::Interp),
    ),
    e(
        0x2ff,
        "v_lshlrev_b64",
        T2 | D2 | S1_2,
        S,
        o(0x33c),
        o(0x11f),
    ),
    e(
        0x300,
        "v_lshrrev_b64",
        T2 | D2 | S1_2,
        S,
        o(0x33d),
        o(0x33d),
    ),
    e(
        0x301,
        "v_ashrrev_i64",
        T2 | D2 | S1_2,
        S,
        o(0x33e),
        o(0x33e),
    ),
    e(0x303, "v_add_nc_u16", T2, S, S, S),
    e(0x304, "v_sub_nc_u16", T2, S, S, S),
    e(0x305, "v_mul_lo_u16", T2, S, S, S),
    e(0x307, "v_lshrrev_b16", T2, S, o(0x339), o(0x339)),
    e(0x308, "v_ashrrev_i16", T2, S, o(0x33a), o(0x33a)),
    e(0x309, "v_max_u16", T2, S, S, S),
    e(0x30a, "v_max_i16", T2, S, S, S),
    e(0x30b, "v_min_u16", T2, S, S, S),
    e(0x30c, "v_min_i16", T2, S, S, S),
    e(0x30d, "v_add_nc_i16", T2, S, S, S),
    e(0x30e, "v_sub_nc_i16", T2, S, S, S),
    e(0x30f, "v_add_co_u32", T2 | VOP3B, S, o(0x300), o(0x300)),
    e(0x310, "v_sub_co_u32", T2 | VOP3B, S, o(0x301), o(0x301)),
    e(0x311, "v_pack_b32_f16", T2, S, S, S),
    e(0x312, "v_cvt_pknorm_i16_f16", T2, S, S, S),
    e(0x313, "v_cvt_pknorm_u16_f16", T2, S, S, S),
    e(0x314, "v_lshlrev_b16", T2, S, o(0x338), o(0x338)),
    e(0x319, "v_subrev_co_u32", T2 | VOP3B, S, o(0x302), o(0x302)),
    e(0x340, "v_mad_u16", T3, S, o(0x241), o(0x241)),
    e(
        0x342,
        "v_interp_p1ll_f16",
        INTERP,
        S,
        r(Rule::Interp),
        r(Rule::Interp),
    ),
    e(
        0x343,
        "v_interp_p1lv_f16",
        INTERP,
        S,
        r(Rule::Interp),
        r(Rule::Interp),
    ),
    e(0x344, "v_perm_b32", T3, S, o(0x244), o(0x244)),
    e(0x345, "v_xad_u32", T3, S, o(0x245), o(0x245)),
    e(0x346, "v_lshl_add_u32", T3, S, o(0x246), o(0x246)),
    e(0x347, "v_add_lshl_u32", T3, S, o(0x247), o(0x247)),
    e(0x34b, "v_fma_f16", T3, S, o(0x248), o(0x248)),
    e(0x351, "v_min3_f16", T3, S, o(0x249), o(0x22b)),
    e(0x352, "v_min3_i16", T3, S, o(0x24a), o(0x24a)),
    e(0x353, "v_min3_u16", T3, S, o(0x24b), o(0x24b)),
    e(0x354, "v_max3_f16", T3, S, o(0x24c), o(0x22c)),
    e(0x355, "v_max3_i16", T3, S, o(0x24d), o(0x24d)),
    e(0x356, "v_max3_u16", T3, S, o(0x24e), o(0x24e)),
    e(0x357, "v_med3_f16", T3, S, o(0x24f), o(0x232)),
    e(0x358, "v_med3_i16", T3, S, o(0x250), o(0x250)),
    e(0x359, "v_med3_u16", T3, S, o(0x251), o(0x251)),
    e(
        0x35a,
        "v_interp_p2_f16",
        INTERP,
        S,
        r(Rule::Interp),
        r(Rule::Interp),
    ),
    e(0x35e, "v_mad_i16", T3, S, o(0x253), o(0x253)),
    e(0x35f, "v_div_fixup_f16", T3, S, o(0x254), o(0x254)),
    e(0x360, "v_readlane_b32", T2 | SDST | LANE_SELECT, S, S, S),
    e(0x361, "v_writelane_b32", T2 | ACC | LANE_SELECT, S, S, S),
    e(0x362, "v_ldexp_f32", T2, S, o(0x31c), o(0x31c)),
    e(0x363, "v_bfm_b32", T2, S, o(0x31d), o(0x31d)),
    e(0x364, "v_bcnt_u32_b32", T2, S, o(0x31e), o(0x31e)),
    e(0x365, "v_mbcnt_lo_u32_b32", T2, S, o(0x31f), o(0x31f)),
    e(0x366, "v_mbcnt_hi_u32_b32", T2, S, o(0x320), o(0x320)),
    e(0x368, "v_cvt_pknorm_i16_f32", T2, S, o(0x321), o(0x321)),
    e(0x369, "v_cvt_pknorm_u16_f32", T2, S, o(0x322), o(0x322)),
    e(0x36a, "v_cvt_pk_u16_u32", T2, S, o(0x323), o(0x323)),
    e(0x36b, "v_cvt_pk_i16_i32", T2, S, o(0x324), o(0x324)),
    e(0x36d, "v_add3_u32", T3, S, o(0x255), o(0x255)),
    e(0x36f, "v_lshl_or_b32", T3, S, o(0x256), o(0x256)),
    e(0x371, "v_and_or_b32", T3, S, o(0x257), o(0x257)),
    e(0x372, "v_or3_b32", T3, S, o(0x258), o(0x258)),
    e(0x373, "v_mad_u32_u16", T3, S, o(0x259), o(0x259)),
    e(0x375, "v_mad_i32_i16", T3, S, o(0x25a), o(0x25a)),
    e(0x376, "v_sub_nc_i32", T2, S, o(0x325), o(0x325)),
    e(0x377, "v_permlane16_b32", T3 | ACC, S, o(0x25b), o(0x25b)),
    e(0x378, "v_permlanex16_b32", T3 | ACC, S, o(0x25c), o(0x25c)),
    e(0x37f, "v_add_nc_i32", T2, S, o(0x326), o(0x326)),
];

pub const VOP3P_DOT2_F32_F16: u16 = 0x13;
pub const VOP3P_DOT4_I32_I8: u16 = 0x16;

pub static VOP3P: &[OpInfo] = &[
    e(0x00, "v_pk_mad_i16", T3, S, S, S),
    e(0x01, "v_pk_mul_lo_u16", T2, S, S, S),
    e(0x02, "v_pk_add_i16", T2, S, S, S),
    e(0x03, "v_pk_sub_i16", T2, S, S, S),
    e(0x04, "v_pk_lshlrev_b16", T2, S, S, S),
    e(0x05, "v_pk_lshrrev_b16", T2, S, S, S),
    e(0x06, "v_pk_ashrrev_i16", T2, S, S, S),
    e(0x07, "v_pk_max_i16", T2, S, S, S),
    e(0x08, "v_pk_min_i16", T2, S, S, S),
    e(0x09, "v_pk_mad_u16", T3, S, S, S),
    e(0x0a, "v_pk_add_u16", T2, S, S, S),
    e(0x0b, "v_pk_sub_u16", T2, S, S, S),
    e(0x0c, "v_pk_max_u16", T2, S, S, S),
    e(0x0d, "v_pk_min_u16", T2, S, S, S),
    e(0x0e, "v_pk_fma_f16", T3, S, S, S),
    e(0x0f, "v_pk_add_f16", T2, S, S, S),
    e(0x10, "v_pk_mul_f16", T2, S, S, S),
    e(0x11, "v_pk_min_f16", T2, S, S, o(0x1b)),
    e(0x12, "v_pk_max_f16", T2, S, S, o(0x1c)),
    e(0x13, "v_dot2_f32_f16", T3, S, S, S),
    e(0x14, "v_dot2_i32_i16", T3, S, r(Rule::Dot2(true)), r(Rule::Dot2(true))),
    e(0x15, "v_dot2_u32_u16", T3, S, r(Rule::Dot2(false)), r(Rule::Dot2(false))),
    e(
        0x16,
        "v_dot4_i32_i8",
        T3,
        S,
        r(Rule::DotSigned(0x16)),
        r(Rule::DotSigned(0x16)),
    ),
    e(0x17, "v_dot4_u32_u8", T3, S, S, S),
    e(
        0x18,
        "v_dot8_i32_i4",
        T3,
        S,
        r(Rule::DotSigned(0x18)),
        r(Rule::DotSigned(0x18)),
    ),
    e(0x19, "v_dot8_u32_u4", T3, S, S, S),
    e(0x20, "v_fma_mix_f32", T3, S, S, S),
    e(0x21, "v_fma_mixlo_f16", T3 | ACC, S, S, S),
    e(0x22, "v_fma_mixhi_f16", T3 | ACC, S, S, S),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CmpType {
    F16,
    F32,
    F64,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

impl CmpType {
    fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    fn is_64(self) -> bool {
        matches!(self, Self::F64 | Self::I64 | Self::U64)
    }

    fn gfx11_base(self) -> u16 {
        match self {
            Self::F16 => 0x00,
            Self::F32 => 0x10,
            Self::F64 => 0x20,
            Self::I16 => 0x30,
            Self::U16 => 0x38,
            Self::I32 => 0x40,
            Self::U32 => 0x48,
            Self::I64 => 0x50,
            Self::U64 => 0x58,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CmpCond {
    Index(u8),
    Class,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VopcInfo {
    pub ty: CmpType,
    pub cond: CmpCond,
    pub exec: bool,
}

const FLOAT_CONDS: [&str; 16] = [
    "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt",
    "tru",
];
const INT_CONDS: [&str; 8] = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"];

impl VopcInfo {
    pub fn decode(op: u16) -> Option<Self> {
        let exec = op & 0x10 != 0;
        let low = (op & 0x0f) as u8;
        let (ty, cond) = match op {
            0x00..=0x1f => (CmpType::F32, CmpCond::Index(low)),
            0x20..=0x3f => (CmpType::F64, CmpCond::Index(low)),
            0x80..=0x9f => match low {
                0..=7 => (CmpType::I32, CmpCond::Index(low)),
                8 => (CmpType::F32, CmpCond::Class),
                9..=14 => (CmpType::I16, CmpCond::Index(low - 8)),
                _ => (CmpType::F16, CmpCond::Class),
            },
            0xa0..=0xbf => match low {
                0..=7 => (CmpType::I64, CmpCond::Index(low)),
                8 => (CmpType::F64, CmpCond::Class),
                9..=14 => (CmpType::U16, CmpCond::Index(low - 8)),
                _ => return None,
            },
            0xc0..=0xdf => match low {
                0..=7 => (CmpType::U32, CmpCond::Index(low)),
                _ => (CmpType::F16, CmpCond::Index(low - 8)),
            },
            0xe0..=0xff => match low {
                0..=7 => (CmpType::U64, CmpCond::Index(low)),
                _ => (CmpType::F16, CmpCond::Index(low)),
            },
            _ => return None,
        };
        Some(Self { ty, cond, exec })
    }

    pub fn name(self) -> String {
        let prefix = if self.exec { "v_cmpx" } else { "v_cmp" };
        match self.cond {
            CmpCond::Class => format!("{prefix}_class_{}", self.ty.name()),
            CmpCond::Index(i) => {
                let cond = if self.ty.is_float() {
                    FLOAT_CONDS[usize::from(i)]
                } else {
                    INT_CONDS[usize::from(i)]
                };
                format!("{prefix}_{cond}_{}", self.ty.name())
            }
        }
    }

    pub fn flags(self) -> u64 {
        let mut flags = 0;
        let wide = match self.cond {
            CmpCond::Class => self.ty.is_64() && self.ty != CmpType::F16,
            CmpCond::Index(_) => self.ty.is_64(),
        };
        if wide {
            flags |= S0_2;
            if !matches!(self.cond, CmpCond::Class) {
                flags |= S1_2;
            }
        }
        if matches!(self.ty, CmpType::F16 | CmpType::I16 | CmpType::U16) {
            flags |= T16;
        }
        if self.exec {
            flags |= CMPX | WRITES_EXEC;
        }
        flags
    }

    pub fn is_constant_result(self) -> Option<bool> {
        match self.cond {
            CmpCond::Index(0) => Some(false),
            CmpCond::Index(15) if self.ty.is_float() => Some(true),
            CmpCond::Index(7) if !self.ty.is_float() => Some(true),
            _ => None,
        }
    }

    pub fn gfx11_opcode(self) -> u16 {
        let exec = if self.exec { 0x80 } else { 0 };
        match self.cond {
            CmpCond::Class => {
                let class = match self.ty {
                    CmpType::F16 => 0x7d,
                    CmpType::F64 => 0x7f,
                    _ => 0x7e,
                };
                class | exec
            }
            CmpCond::Index(i) => self.ty.gfx11_base() + u16::from(i) + exec,
        }
    }
}

fn table(family: InstructionFamily) -> &'static [OpInfo] {
    match family {
        InstructionFamily::Sop1 => SOP1,
        InstructionFamily::Sop2 => SOP2,
        InstructionFamily::Sopk => SOPK,
        InstructionFamily::Sopc => SOPC,
        InstructionFamily::Sopp => SOPP,
        InstructionFamily::Vop1 => VOP1,
        InstructionFamily::Vop2 => VOP2,
        InstructionFamily::Vop3 => VOP3,
        InstructionFamily::Vop3p => VOP3P,
        _ => &[],
    }
}

pub fn lookup(family: InstructionFamily, op: u16) -> Option<&'static OpInfo> {
    table(family).iter().find(|info| info.op == op)
}

pub fn vop3_base(op: u16) -> Option<(InstructionFamily, u16)> {
    match op {
        0..=VOP3_VOPC_LAST => Some((InstructionFamily::Vopc, op)),
        VOP3_VOP2_BASE..=VOP3_VOP2_LAST => Some((InstructionFamily::Vop2, op - VOP3_VOP2_BASE)),
        VOP3_VOP1_BASE..=VOP3_VOP1_LAST => Some((InstructionFamily::Vop1, op - VOP3_VOP1_BASE)),
        _ => None,
    }
}

pub fn vop3_flags(op: u16) -> Option<u64> {
    match vop3_base(op) {
        Some((InstructionFamily::Vopc, base)) => VopcInfo::decode(base).map(VopcInfo::flags),
        Some((family, base)) => lookup(family, base)
            .filter(|i| i.flags & NO_VOP3 == 0)
            .map(|i| i.flags),
        None => lookup(InstructionFamily::Vop3, op).map(|i| i.flags),
    }
}

pub fn mnemonic(family: InstructionFamily, op: u16) -> String {
    let named = |info: Option<&OpInfo>| info.map(|i| i.name.to_string());
    let name = match family {
        InstructionFamily::Vopc => VopcInfo::decode(op).map(VopcInfo::name),
        InstructionFamily::Vop3 => match vop3_base(op) {
            Some((InstructionFamily::Vopc, base)) => VopcInfo::decode(base).map(VopcInfo::name),
            Some((base_family, base)) => named(lookup(base_family, base)),
            None => named(lookup(family, op)),
        },
        InstructionFamily::Vintrp => match op {
            VINTRP_P1 => Some("v_interp_p1_f32".into()),
            VINTRP_P2 => Some("v_interp_p2_f32".into()),
            VINTRP_MOV => Some("v_interp_mov_f32".into()),
            _ => None,
        },
        InstructionFamily::Exp => Some("exp".into()),
        _ => named(lookup(family, op)).or_else(|| memops::name(family, op)),
    };
    name.unwrap_or_else(|| format!("{}_op_{op:#x}", family.name().to_ascii_lowercase()))
}

pub fn flag_source_count(flags: u64) -> Option<usize> {
    if flags & SRC3 != 0 {
        Some(3)
    } else if flags & SRC2 != 0 {
        Some(2)
    } else if flags & SRC1 != 0 {
        Some(1)
    } else {
        None
    }
}

pub fn vop3_source_count(op: u16, flags: u64) -> usize {
    flag_source_count(flags).unwrap_or(match vop3_base(op) {
        Some((InstructionFamily::Vop1, _)) => usize::from(flags & NO_OPERANDS == 0),
        Some((InstructionFamily::Vop2, _)) if flags & CARRY_IN != 0 => 3,
        Some(_) => 2,
        None => 3,
    })
}

pub fn source_count(inst: &Instruction) -> usize {
    match inst.fields {
        Fields::Vop3(_) => vop3_source_count(inst.opcode, inst.flags),
        _ => flag_source_count(inst.flags).unwrap_or(match inst.fields {
            Fields::Vop3p(_) => 3,
            _ => 2,
        }),
    }
}
