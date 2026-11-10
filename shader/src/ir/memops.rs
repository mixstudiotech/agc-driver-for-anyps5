use super::opcodes::{flag, Rule, Xlat};
use super::{InstructionFamily, MemKind};
use crate::TargetIsa;

#[derive(Clone, Copy, Debug)]
pub struct MemOp {
    pub op: u16,
    pub name: &'static str,
    pub kind: MemKind,
    pub dwords: u8,
    pub flags: u64,
    pub gfx103: Xlat,
    pub gfx11: Xlat,
    pub gfx12: Xlat,
}

impl MemOp {
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
const LOAD: MemKind = MemKind::Load;
const STORE: MemKind = MemKind::Store;
const ATOMIC: MemKind = MemKind::Atomic;
const CONTROL: MemKind = MemKind::Control;

const fn o(op: u16) -> Xlat {
    Xlat::Op(op)
}

#[allow(clippy::too_many_arguments)]
const fn m(
    op: u16,
    name: &'static str,
    kind: MemKind,
    dwords: u8,
    flags: u64,
    gfx103: Xlat,
    gfx11: Xlat,
    gfx12: Xlat,
) -> MemOp {
    MemOp {
        op,
        name,
        kind,
        dwords,
        flags,
        gfx103,
        gfx11,
        gfx12,
    }
}

const fn c(op: u16, name: &'static str, kind: MemKind, dwords: u8, flags: u64, next: u16) -> MemOp {
    m(op, name, kind, dwords, flags, S, o(next), o(next))
}

const BUF: u64 = flag::IMM_U;
const NOSRC: u64 = flag::NO_SRC;
const HI: u64 = flag::READS_DST;
const CS: u64 = flag::CMPSWAP;

pub fn is_buffer_smem(op: &MemOp) -> bool {
    op.flags & BUF != 0
}

pub static SMEM: &[MemOp] = &[
    m(0x00, "s_load_dword", LOAD, 1, 0, S, S, S),
    m(0x01, "s_load_dwordx2", LOAD, 2, 0, S, S, S),
    m(0x02, "s_load_dwordx4", LOAD, 4, 0, S, S, S),
    m(0x03, "s_load_dwordx8", LOAD, 8, 0, S, S, S),
    m(0x04, "s_load_dwordx16", LOAD, 16, 0, S, S, S),
    m(0x05, "s_scratch_load_dword", LOAD, 1, 0, N, N, N),
    m(0x06, "s_scratch_load_dwordx2", LOAD, 2, 0, N, N, N),
    m(0x07, "s_scratch_load_dwordx4", LOAD, 4, 0, N, N, N),
    m(0x08, "s_buffer_load_dword", LOAD, 1, BUF, S, S, o(0x10)),
    m(0x09, "s_buffer_load_dwordx2", LOAD, 2, BUF, S, S, o(0x11)),
    m(0x0a, "s_buffer_load_dwordx4", LOAD, 4, BUF, S, S, o(0x12)),
    m(0x0b, "s_buffer_load_dwordx8", LOAD, 8, BUF, S, S, o(0x13)),
    m(0x0c, "s_buffer_load_dwordx16", LOAD, 16, BUF, S, S, o(0x14)),
    m(0x1f, "s_gl1_inv", CONTROL, 0, NOSRC, S, o(0x20), D),
    m(0x20, "s_dcache_inv", CONTROL, 0, NOSRC, S, o(0x21), o(0x21)),
    // Write-back and discard only act on lines dirtied by scalar stores, which gfx10.3 and later
    // do not have, so dropping them preserves every observable value.
    m(0x21, "s_dcache_wb", CONTROL, 0, NOSRC, D, D, D),
    m(
        0x24,
        "s_memtime",
        LOAD,
        2,
        NOSRC,
        S,
        Xlat::Rule(Rule::MemTime),
        Xlat::Rule(Rule::MemTime),
    ),
    m(
        0x25,
        "s_memrealtime",
        LOAD,
        2,
        NOSRC,
        S,
        Xlat::Rule(Rule::MemTime),
        Xlat::Rule(Rule::MemTime),
    ),
    m(
        0x26,
        "s_atc_probe",
        CONTROL,
        0,
        flag::SMEM_ADDR,
        S,
        o(0x22),
        o(0x22),
    ),
    m(
        0x27,
        "s_atc_probe_buffer",
        CONTROL,
        0,
        flag::SMEM_ADDR | BUF,
        S,
        o(0x23),
        o(0x23),
    ),
    m(
        0x28,
        "s_dcache_discard",
        CONTROL,
        0,
        flag::SMEM_ADDR,
        D,
        D,
        D,
    ),
    m(
        0x29,
        "s_dcache_discard_x2",
        CONTROL,
        0,
        flag::SMEM_ADDR,
        D,
        D,
        D,
    ),
    m(
        0x2a,
        "s_get_waveid_in_workgroup",
        LOAD,
        1,
        NOSRC,
        N,
        Xlat::Rule(Rule::WaveId),
        N,
    ),
];

pub static VMEM_COMMON: &[MemOp] = &[
    c(0x08, "load_ubyte", LOAD, 1, 0, 0x10),
    c(0x09, "load_sbyte", LOAD, 1, 0, 0x11),
    c(0x0a, "load_ushort", LOAD, 1, 0, 0x12),
    c(0x0b, "load_sshort", LOAD, 1, 0, 0x13),
    c(0x0c, "load_dword", LOAD, 1, 0, 0x14),
    c(0x0d, "load_dwordx2", LOAD, 2, 0, 0x15),
    c(0x0e, "load_dwordx4", LOAD, 4, 0, 0x17),
    c(0x0f, "load_dwordx3", LOAD, 3, 0, 0x16),
    c(0x18, "store_byte", STORE, 1, 0, 0x18),
    c(0x19, "store_byte_d16_hi", STORE, 1, 0, 0x24),
    c(0x1a, "store_short", STORE, 1, 0, 0x19),
    c(0x1b, "store_short_d16_hi", STORE, 1, 0, 0x25),
    c(0x1c, "store_dword", STORE, 1, 0, 0x1a),
    c(0x1d, "store_dwordx2", STORE, 2, 0, 0x1b),
    c(0x1e, "store_dwordx4", STORE, 4, 0, 0x1d),
    c(0x1f, "store_dwordx3", STORE, 3, 0, 0x1c),
    c(0x20, "load_ubyte_d16", LOAD, 1, HI, 0x1e),
    c(0x21, "load_ubyte_d16_hi", LOAD, 1, HI, 0x21),
    c(0x22, "load_sbyte_d16", LOAD, 1, HI, 0x1f),
    c(0x23, "load_sbyte_d16_hi", LOAD, 1, HI, 0x22),
    c(0x24, "load_short_d16", LOAD, 1, HI, 0x20),
    c(0x25, "load_short_d16_hi", LOAD, 1, HI, 0x23),
    c(0x30, "atomic_swap", ATOMIC, 1, 0, 0x33),
    c(0x31, "atomic_cmpswap", ATOMIC, 2, CS, 0x34),
    c(0x32, "atomic_add", ATOMIC, 1, 0, 0x35),
    c(0x33, "atomic_sub", ATOMIC, 1, 0, 0x36),
    c(0x34, "atomic_csub", ATOMIC, 1, 0, 0x37),
    c(0x35, "atomic_smin", ATOMIC, 1, 0, 0x38),
    c(0x36, "atomic_umin", ATOMIC, 1, 0, 0x39),
    c(0x37, "atomic_smax", ATOMIC, 1, 0, 0x3a),
    c(0x38, "atomic_umax", ATOMIC, 1, 0, 0x3b),
    c(0x39, "atomic_and", ATOMIC, 1, 0, 0x3c),
    c(0x3a, "atomic_or", ATOMIC, 1, 0, 0x3d),
    c(0x3b, "atomic_xor", ATOMIC, 1, 0, 0x3e),
    c(0x3c, "atomic_inc", ATOMIC, 1, 0, 0x3f),
    c(0x3d, "atomic_dec", ATOMIC, 1, 0, 0x40),
    m(0x3e, "atomic_fcmpswap", ATOMIC, 2, CS, S, o(0x50), N),
    c(0x3f, "atomic_fmin", ATOMIC, 1, 0, 0x51),
    c(0x40, "atomic_fmax", ATOMIC, 1, 0, 0x52),
    c(0x50, "atomic_swap_x2", ATOMIC, 2, 0, 0x41),
    c(0x51, "atomic_cmpswap_x2", ATOMIC, 4, CS, 0x42),
    c(0x52, "atomic_add_x2", ATOMIC, 2, 0, 0x43),
    c(0x53, "atomic_sub_x2", ATOMIC, 2, 0, 0x44),
    c(0x55, "atomic_smin_x2", ATOMIC, 2, 0, 0x45),
    c(0x56, "atomic_umin_x2", ATOMIC, 2, 0, 0x46),
    c(0x57, "atomic_smax_x2", ATOMIC, 2, 0, 0x47),
    c(0x58, "atomic_umax_x2", ATOMIC, 2, 0, 0x48),
    c(0x59, "atomic_and_x2", ATOMIC, 2, 0, 0x49),
    c(0x5a, "atomic_or_x2", ATOMIC, 2, 0, 0x4a),
    c(0x5b, "atomic_xor_x2", ATOMIC, 2, 0, 0x4b),
    c(0x5c, "atomic_inc_x2", ATOMIC, 2, 0, 0x4c),
    c(0x5d, "atomic_dec_x2", ATOMIC, 2, 0, 0x4d),
    m(0x5e, "atomic_fcmpswap_x2", ATOMIC, 4, CS, S, N, N),
    m(0x5f, "atomic_fmin_x2", ATOMIC, 2, 0, S, N, N),
    m(0x60, "atomic_fmax_x2", ATOMIC, 2, 0, S, N, N),
];

pub static MUBUF_ONLY: &[MemOp] = &[
    c(0x00, "load_format_x", LOAD, 1, 0, 0x00),
    c(0x01, "load_format_xy", LOAD, 2, 0, 0x01),
    c(0x02, "load_format_xyz", LOAD, 3, 0, 0x02),
    c(0x03, "load_format_xyzw", LOAD, 4, 0, 0x03),
    c(0x04, "store_format_x", STORE, 1, 0, 0x04),
    c(0x05, "store_format_xy", STORE, 2, 0, 0x05),
    c(0x06, "store_format_xyz", STORE, 3, 0, 0x06),
    c(0x07, "store_format_xyzw", STORE, 4, 0, 0x07),
    c(0x26, "load_format_d16_hi_x", LOAD, 1, HI, 0x26),
    c(0x27, "store_format_d16_hi_x", STORE, 1, 0, 0x27),
    m(0x71, "gl0_inv", CONTROL, 0, 0, S, o(0x2b), N),
    m(0x72, "gl1_inv", CONTROL, 0, 0, S, o(0x2c), N),
    c(0x80, "load_format_d16_x", LOAD, 1, HI, 0x08),
    c(0x81, "load_format_d16_xy", LOAD, 1, 0, 0x09),
    c(0x82, "load_format_d16_xyz", LOAD, 2, HI, 0x0a),
    c(0x83, "load_format_d16_xyzw", LOAD, 2, 0, 0x0b),
    c(0x84, "store_format_d16_x", STORE, 1, 0, 0x0c),
    c(0x85, "store_format_d16_xy", STORE, 1, 0, 0x0d),
    c(0x86, "store_format_d16_xyz", STORE, 2, 0, 0x0e),
    c(0x87, "store_format_d16_xyzw", STORE, 2, 0, 0x0f),
];

pub static MTBUF: &[MemOp] = &[
    c(0x00, "tbuffer_load_format_x", LOAD, 1, 0, 0x00),
    c(0x01, "tbuffer_load_format_xy", LOAD, 2, 0, 0x01),
    c(0x02, "tbuffer_load_format_xyz", LOAD, 3, 0, 0x02),
    c(0x03, "tbuffer_load_format_xyzw", LOAD, 4, 0, 0x03),
    c(0x04, "tbuffer_store_format_x", STORE, 1, 0, 0x04),
    c(0x05, "tbuffer_store_format_xy", STORE, 2, 0, 0x05),
    c(0x06, "tbuffer_store_format_xyz", STORE, 3, 0, 0x06),
    c(0x07, "tbuffer_store_format_xyzw", STORE, 4, 0, 0x07),
    c(0x08, "tbuffer_load_format_d16_x", LOAD, 1, HI, 0x08),
    c(0x09, "tbuffer_load_format_d16_xy", LOAD, 1, 0, 0x09),
    c(0x0a, "tbuffer_load_format_d16_xyz", LOAD, 2, HI, 0x0a),
    c(0x0b, "tbuffer_load_format_d16_xyzw", LOAD, 2, 0, 0x0b),
    c(0x0c, "tbuffer_store_format_d16_x", STORE, 1, 0, 0x0c),
    c(0x0d, "tbuffer_store_format_d16_xy", STORE, 1, 0, 0x0d),
    c(0x0e, "tbuffer_store_format_d16_xyz", STORE, 2, 0, 0x0e),
    c(0x0f, "tbuffer_store_format_d16_xyzw", STORE, 2, 0, 0x0f),
];

/// FLAT-encoding operations that exist only in the GLOBAL segment (as `atomic_csub` does).
pub static FLAT_ONLY: &[MemOp] = &[
    c(0x16, "load_dword_addtid", LOAD, 1, 0, 0x28),
    c(0x17, "store_dword_addtid", STORE, 1, 0, 0x29),
];

pub const FLAT_ATOMIC_CSUB: u16 = 0x34;

#[derive(Clone, Copy, Debug)]
pub struct DsOp {
    pub op: u16,
    pub name: &'static str,
    pub dst: u8,
    pub data0: u8,
    pub data1: u8,
    pub flags: u8,
}

pub const DS_NO_ADDR: u8 = 1 << 0;
pub const DS_GWS: u8 = 1 << 1;
/// The second operand is LDS at an address derived from the first (removed in gfx10.3).
pub const DS_SRC2: u8 = 1 << 2;
pub const DS_GDS_ONLY: u8 = 1 << 3;
/// Compare-store: gfx11 renamed `ds_cmpst_*` to `ds_cmpstore_*` and swapped DATA0 (the compare
/// value on gfx10) with DATA1 (the new value).
pub const DS_CMPST: u8 = 1 << 4;
/// Removed in gfx12.
pub const DS_NOT_GFX12: u8 = 1 << 5;
const CMPST: u8 = DS_CMPST;
const CMPST_F: u8 = DS_CMPST | DS_NOT_GFX12;
pub const DS_ADD_RTN_F32_GFX10: u16 = 0x55;
pub const DS_ADD_RTN_F32_GFX11: u16 = 0x79;

const fn ds(op: u16, name: &'static str, dst: u8, data0: u8, data1: u8, flags: u8) -> DsOp {
    DsOp {
        op,
        name,
        dst,
        data0,
        data1,
        flags,
    }
}

pub static DS: &[DsOp] = &[
    ds(0x00, "ds_add_u32", 0, 1, 0, 0),
    ds(0x01, "ds_sub_u32", 0, 1, 0, 0),
    ds(0x02, "ds_rsub_u32", 0, 1, 0, 0),
    ds(0x03, "ds_inc_u32", 0, 1, 0, 0),
    ds(0x04, "ds_dec_u32", 0, 1, 0, 0),
    ds(0x05, "ds_min_i32", 0, 1, 0, 0),
    ds(0x06, "ds_max_i32", 0, 1, 0, 0),
    ds(0x07, "ds_min_u32", 0, 1, 0, 0),
    ds(0x08, "ds_max_u32", 0, 1, 0, 0),
    ds(0x09, "ds_and_b32", 0, 1, 0, 0),
    ds(0x0a, "ds_or_b32", 0, 1, 0, 0),
    ds(0x0b, "ds_xor_b32", 0, 1, 0, 0),
    ds(0x0c, "ds_mskor_b32", 0, 1, 1, 0),
    ds(0x0d, "ds_write_b32", 0, 1, 0, 0),
    ds(0x0e, "ds_write2_b32", 0, 1, 1, 0),
    ds(0x0f, "ds_write2st64_b32", 0, 1, 1, 0),
    ds(0x10, "ds_cmpst_b32", 0, 1, 1, CMPST),
    ds(0x11, "ds_cmpst_f32", 0, 1, 1, CMPST_F),
    ds(0x12, "ds_min_f32", 0, 1, 0, 0),
    ds(0x13, "ds_max_f32", 0, 1, 0, 0),
    ds(0x14, "ds_nop", 0, 0, 0, DS_NO_ADDR),
    ds(0x15, "ds_add_f32", 0, 1, 0, 0),
    ds(
        0x18,
        "ds_gws_sema_release_all",
        0,
        0,
        0,
        DS_NO_ADDR | DS_GWS,
    ),
    ds(0x19, "ds_gws_init", 0, 1, 0, DS_NO_ADDR | DS_GWS),
    ds(0x1a, "ds_gws_sema_v", 0, 0, 0, DS_NO_ADDR | DS_GWS),
    ds(0x1b, "ds_gws_sema_br", 0, 1, 0, DS_NO_ADDR | DS_GWS),
    ds(0x1c, "ds_gws_sema_p", 0, 0, 0, DS_NO_ADDR | DS_GWS),
    ds(0x1d, "ds_gws_barrier", 0, 1, 0, DS_NO_ADDR | DS_GWS),
    ds(0x1e, "ds_write_b8", 0, 1, 0, 0),
    ds(0x1f, "ds_write_b16", 0, 1, 0, 0),
    ds(0x20, "ds_add_rtn_u32", 1, 1, 0, 0),
    ds(0x21, "ds_sub_rtn_u32", 1, 1, 0, 0),
    ds(0x22, "ds_rsub_rtn_u32", 1, 1, 0, 0),
    ds(0x23, "ds_inc_rtn_u32", 1, 1, 0, 0),
    ds(0x24, "ds_dec_rtn_u32", 1, 1, 0, 0),
    ds(0x25, "ds_min_rtn_i32", 1, 1, 0, 0),
    ds(0x26, "ds_max_rtn_i32", 1, 1, 0, 0),
    ds(0x27, "ds_min_rtn_u32", 1, 1, 0, 0),
    ds(0x28, "ds_max_rtn_u32", 1, 1, 0, 0),
    ds(0x29, "ds_and_rtn_b32", 1, 1, 0, 0),
    ds(0x2a, "ds_or_rtn_b32", 1, 1, 0, 0),
    ds(0x2b, "ds_xor_rtn_b32", 1, 1, 0, 0),
    ds(0x2c, "ds_mskor_rtn_b32", 1, 1, 1, 0),
    ds(0x2d, "ds_wrxchg_rtn_b32", 1, 1, 0, 0),
    ds(0x2e, "ds_wrxchg2_rtn_b32", 2, 1, 1, 0),
    ds(0x2f, "ds_wrxchg2st64_rtn_b32", 2, 1, 1, 0),
    ds(0x30, "ds_cmpst_rtn_b32", 1, 1, 1, CMPST),
    ds(0x31, "ds_cmpst_rtn_f32", 1, 1, 1, CMPST_F),
    ds(0x32, "ds_min_rtn_f32", 1, 1, 0, 0),
    ds(0x33, "ds_max_rtn_f32", 1, 1, 0, 0),
    ds(0x34, "ds_wrap_rtn_b32", 1, 1, 1, DS_NOT_GFX12),
    ds(0x35, "ds_swizzle_b32", 1, 0, 0, 0),
    ds(0x36, "ds_read_b32", 1, 0, 0, 0),
    ds(0x37, "ds_read2_b32", 2, 0, 0, 0),
    ds(0x38, "ds_read2st64_b32", 2, 0, 0, 0),
    ds(0x39, "ds_read_i8", 1, 0, 0, 0),
    ds(0x3a, "ds_read_u8", 1, 0, 0, 0),
    ds(0x3b, "ds_read_i16", 1, 0, 0, 0),
    ds(0x3c, "ds_read_u16", 1, 0, 0, 0),
    ds(0x3d, "ds_consume", 1, 0, 0, DS_NO_ADDR),
    ds(0x3e, "ds_append", 1, 0, 0, DS_NO_ADDR),
    ds(0x3f, "ds_ordered_count", 1, 0, 0, DS_GDS_ONLY),
    ds(0x40, "ds_add_u64", 0, 2, 0, 0),
    ds(0x41, "ds_sub_u64", 0, 2, 0, 0),
    ds(0x42, "ds_rsub_u64", 0, 2, 0, 0),
    ds(0x43, "ds_inc_u64", 0, 2, 0, 0),
    ds(0x44, "ds_dec_u64", 0, 2, 0, 0),
    ds(0x45, "ds_min_i64", 0, 2, 0, 0),
    ds(0x46, "ds_max_i64", 0, 2, 0, 0),
    ds(0x47, "ds_min_u64", 0, 2, 0, 0),
    ds(0x48, "ds_max_u64", 0, 2, 0, 0),
    ds(0x49, "ds_and_b64", 0, 2, 0, 0),
    ds(0x4a, "ds_or_b64", 0, 2, 0, 0),
    ds(0x4b, "ds_xor_b64", 0, 2, 0, 0),
    ds(0x4c, "ds_mskor_b64", 0, 2, 2, 0),
    ds(0x4d, "ds_write_b64", 0, 2, 0, 0),
    ds(0x4e, "ds_write2_b64", 0, 2, 2, 0),
    ds(0x4f, "ds_write2st64_b64", 0, 2, 2, 0),
    ds(0x50, "ds_cmpst_b64", 0, 2, 2, CMPST),
    ds(0x51, "ds_cmpst_f64", 0, 2, 2, CMPST_F),
    ds(0x52, "ds_min_f64", 0, 2, 0, 0),
    ds(0x53, "ds_max_f64", 0, 2, 0, 0),
    ds(0x55, "ds_add_rtn_f32", 1, 1, 0, 0),
    ds(0x60, "ds_add_rtn_u64", 2, 2, 0, 0),
    ds(0x61, "ds_sub_rtn_u64", 2, 2, 0, 0),
    ds(0x62, "ds_rsub_rtn_u64", 2, 2, 0, 0),
    ds(0x63, "ds_inc_rtn_u64", 2, 2, 0, 0),
    ds(0x64, "ds_dec_rtn_u64", 2, 2, 0, 0),
    ds(0x65, "ds_min_rtn_i64", 2, 2, 0, 0),
    ds(0x66, "ds_max_rtn_i64", 2, 2, 0, 0),
    ds(0x67, "ds_min_rtn_u64", 2, 2, 0, 0),
    ds(0x68, "ds_max_rtn_u64", 2, 2, 0, 0),
    ds(0x69, "ds_and_rtn_b64", 2, 2, 0, 0),
    ds(0x6a, "ds_or_rtn_b64", 2, 2, 0, 0),
    ds(0x6b, "ds_xor_rtn_b64", 2, 2, 0, 0),
    ds(0x6c, "ds_mskor_rtn_b64", 2, 2, 2, 0),
    ds(0x6d, "ds_wrxchg_rtn_b64", 2, 2, 0, 0),
    ds(0x6e, "ds_wrxchg2_rtn_b64", 4, 2, 2, 0),
    ds(0x6f, "ds_wrxchg2st64_rtn_b64", 4, 2, 2, 0),
    ds(0x70, "ds_cmpst_rtn_b64", 2, 2, 2, CMPST),
    ds(0x71, "ds_cmpst_rtn_f64", 2, 2, 2, CMPST_F),
    ds(0x72, "ds_min_rtn_f64", 2, 2, 0, 0),
    ds(0x73, "ds_max_rtn_f64", 2, 2, 0, 0),
    ds(0x76, "ds_read_b64", 2, 0, 0, 0),
    ds(0x77, "ds_read2_b64", 4, 0, 0, 0),
    ds(0x78, "ds_read2st64_b64", 4, 0, 0, 0),
    ds(0x7e, "ds_condxchg32_rtn_b64", 2, 2, 0, 0),
    ds(0x80, "ds_add_src2_u32", 0, 0, 0, DS_SRC2),
    ds(0x81, "ds_sub_src2_u32", 0, 0, 0, DS_SRC2),
    ds(0x82, "ds_rsub_src2_u32", 0, 0, 0, DS_SRC2),
    ds(0x83, "ds_inc_src2_u32", 0, 0, 0, DS_SRC2),
    ds(0x84, "ds_dec_src2_u32", 0, 0, 0, DS_SRC2),
    ds(0x85, "ds_min_src2_i32", 0, 0, 0, DS_SRC2),
    ds(0x86, "ds_max_src2_i32", 0, 0, 0, DS_SRC2),
    ds(0x87, "ds_min_src2_u32", 0, 0, 0, DS_SRC2),
    ds(0x88, "ds_max_src2_u32", 0, 0, 0, DS_SRC2),
    ds(0x89, "ds_and_src2_b32", 0, 0, 0, DS_SRC2),
    ds(0x8a, "ds_or_src2_b32", 0, 0, 0, DS_SRC2),
    ds(0x8b, "ds_xor_src2_b32", 0, 0, 0, DS_SRC2),
    ds(0x8d, "ds_write_src2_b32", 0, 0, 0, DS_SRC2),
    ds(0x92, "ds_min_src2_f32", 0, 0, 0, DS_SRC2),
    ds(0x93, "ds_max_src2_f32", 0, 0, 0, DS_SRC2),
    ds(0x95, "ds_add_src2_f32", 0, 0, 0, DS_SRC2),
    ds(0xa0, "ds_write_b8_d16_hi", 0, 1, 0, 0),
    ds(0xa1, "ds_write_b16_d16_hi", 0, 1, 0, 0),
    ds(0xa2, "ds_read_u8_d16", 1, 0, 0, 0),
    ds(0xa3, "ds_read_u8_d16_hi", 1, 0, 0, 0),
    ds(0xa4, "ds_read_i8_d16", 1, 0, 0, 0),
    ds(0xa5, "ds_read_i8_d16_hi", 1, 0, 0, 0),
    ds(0xa6, "ds_read_u16_d16", 1, 0, 0, 0),
    ds(0xa7, "ds_read_u16_d16_hi", 1, 0, 0, 0),
    ds(0xb0, "ds_write_addtid_b32", 0, 1, 0, DS_NO_ADDR),
    ds(0xb1, "ds_read_addtid_b32", 1, 0, 0, DS_NO_ADDR),
    ds(0xb2, "ds_permute_b32", 1, 1, 0, 0),
    ds(0xb3, "ds_bpermute_b32", 1, 1, 0, 0),
    ds(0xde, "ds_write_b96", 0, 3, 0, 0),
    ds(0xdf, "ds_write_b128", 0, 4, 0, 0),
    ds(0xfe, "ds_read_b96", 3, 0, 0, 0),
    ds(0xff, "ds_read_b128", 4, 0, 0, 0),
];

impl DsOp {
    pub fn translate(&self, target: TargetIsa) -> Xlat {
        let removed = match target {
            TargetIsa::Gfx1013 => false,
            TargetIsa::Gfx103 | TargetIsa::Gfx11 | TargetIsa::Gfx115 => self.flags & DS_SRC2 != 0,
            TargetIsa::Gfx12 => self.flags & (DS_SRC2 | DS_GWS | DS_GDS_ONLY | DS_NOT_GFX12) != 0,
        };
        if removed {
            return Xlat::No;
        }
        if target.is_gfx11_plus() && self.op == DS_ADD_RTN_F32_GFX10 {
            return Xlat::Op(DS_ADD_RTN_F32_GFX11);
        }
        Xlat::Op(self.op)
    }

    /// Whether the target encoding swaps DATA0 and DATA1 relative to gfx1013.
    pub fn swaps_data(&self, target: TargetIsa) -> bool {
        self.flags & DS_CMPST != 0 && target.is_gfx11_plus()
    }
}

pub mod img {
    pub const SAMPLER: u32 = 1 << 0;
    pub const GATHER: u32 = 1 << 1;
    pub const OFFSET: u32 = 1 << 2;
    pub const BIAS: u32 = 1 << 3;
    pub const COMPARE: u32 = 1 << 4;
    pub const LOD: u32 = 1 << 5;
    pub const CLAMP: u32 = 1 << 6;
    pub const DERIV: u32 = 1 << 7;
    pub const LZ: u32 = 1 << 8;
    pub const G16: u32 = 1 << 9;
    pub const MIP: u32 = 1 << 10;
    pub const ATOMIC_OP: u32 = 1 << 11;
    pub const STORE_OP: u32 = 1 << 12;
    pub const RESINFO: u32 = 1 << 13;
    pub const MSAA: u32 = 1 << 14;
    pub const BVH: u32 = 1 << 15;
    pub const BVH64: u32 = 1 << 16;
    pub const CMPSWAP: u32 = 1 << 17;
    pub const PACKED_GATHER: u32 = 1 << 18;
    /// A PS5 `_a` sample or gather: the plain form's opcode plus 0x80, with the same address
    /// operands. Guest code stores lane data in sampler dword 3 bits 12-29
    /// ([`SAMPLER_ADJUST_BITS`]), which gfx10 reserves; with them cleared the instruction is the
    /// plain form, which is how it executes and translates.
    pub const ADJUST: u32 = 1 << 19;
    pub const ADJUST_OPCODE_BIT: u16 = 0x80;
    pub const SAMPLER_ADJUST_BITS: u32 = 0x3fff_f000;
}

use img::*;

#[derive(Clone, Copy, Debug)]
pub struct ImageOp {
    pub op: u16,
    pub name: &'static str,
    pub flags: u32,
    pub gfx103: Xlat,
    pub gfx11: Xlat,
    pub gfx12: Xlat,
}

impl ImageOp {
    pub fn translate(&self, target: TargetIsa) -> Xlat {
        if self.flags & ADJUST != 0 && target != TargetIsa::Gfx1013 {
            return self
                .plain()
                .map_or(Xlat::No, |plain| plain.translate(target));
        }
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

    pub fn kind(&self) -> MemKind {
        if self.flags & ATOMIC_OP != 0 {
            MemKind::Atomic
        } else if self.flags & STORE_OP != 0 {
            MemKind::Store
        } else {
            MemKind::Load
        }
    }

    pub fn uses_sampler(&self) -> bool {
        self.flags & (SAMPLER | GATHER) != 0
    }

    /// The plain form of an [`ADJUST`] instruction.
    pub fn plain(&self) -> Option<&'static ImageOp> {
        (self.flags & ADJUST != 0)
            .then(|| image(self.op & !ADJUST_OPCODE_BIT))
            .flatten()
    }
}

/// An image instruction gfx10.3 keeps at its gfx1013 opcode and gfx11/gfx12 share at `next`.
const fn im(op: u16, name: &'static str, flags: u32, next: Xlat) -> ImageOp {
    im12(op, name, flags, next, next)
}

const fn im12(op: u16, name: &'static str, flags: u32, gfx11: Xlat, gfx12: Xlat) -> ImageOp {
    ImageOp {
        op,
        name,
        flags,
        gfx103: S,
        gfx11,
        gfx12,
    }
}

pub(crate) const fn ps5(op: u16, name: &'static str, flags: u32) -> ImageOp {
    ImageOp {
        op,
        name,
        flags,
        gfx103: N,
        gfx11: N,
        gfx12: N,
    }
}

/// An [`ADJUST`] form; [`ImageOp::translate`] follows its plain form.
const fn adjusted(op: u16, name: &'static str, flags: u32) -> ImageOp {
    ImageOp {
        op,
        name,
        flags: flags | ADJUST,
        gfx103: S,
        gfx11: S,
        gfx12: S,
    }
}

const SMP: u32 = SAMPLER;
const GTH: u32 = GATHER | SAMPLER;
const DV: u32 = DERIV;
const AT: u32 = ATOMIC_OP;

pub static MIMG: &[ImageOp] = &[
    im(0x00, "image_load", 0, o(0x00)),
    im(0x01, "image_load_mip", MIP, o(0x01)),
    im(0x02, "image_load_pck", 0, o(0x02)),
    im(0x03, "image_load_pck_sgn", 0, o(0x03)),
    im(0x04, "image_load_mip_pck", MIP, o(0x04)),
    im(0x05, "image_load_mip_pck_sgn", MIP, o(0x05)),
    im(0x08, "image_store", STORE_OP, o(0x06)),
    im(0x09, "image_store_mip", STORE_OP | MIP, o(0x07)),
    im(0x0a, "image_store_pck", STORE_OP, o(0x08)),
    im(0x0b, "image_store_mip_pck", STORE_OP | MIP, o(0x09)),
    im(0x0e, "image_get_resinfo", RESINFO, o(0x17)),
    im(0x0f, "image_atomic_swap", AT, o(0x0a)),
    im(0x10, "image_atomic_cmpswap", AT | CMPSWAP, o(0x0b)),
    im(0x11, "image_atomic_add", AT, o(0x0c)),
    im(0x12, "image_atomic_sub", AT, o(0x0d)),
    im(0x14, "image_atomic_smin", AT, o(0x0e)),
    im(0x15, "image_atomic_umin", AT, o(0x0f)),
    im(0x16, "image_atomic_smax", AT, o(0x10)),
    im(0x17, "image_atomic_umax", AT, o(0x11)),
    im(0x18, "image_atomic_and", AT, o(0x12)),
    im(0x19, "image_atomic_or", AT, o(0x13)),
    im(0x1a, "image_atomic_xor", AT, o(0x14)),
    im(0x1b, "image_atomic_inc", AT, o(0x15)),
    im(0x1c, "image_atomic_dec", AT, o(0x16)),
    im(0x1d, "image_atomic_fcmpswap", AT | CMPSWAP, N),
    im12(0x1e, "image_atomic_fmin", AT, N, o(0x84)),
    im12(0x1f, "image_atomic_fmax", AT, N, o(0x85)),
    im(0x20, "image_sample", SMP, o(0x1b)),
    im(0x21, "image_sample_cl", SMP | CLAMP, o(0x40)),
    im(0x22, "image_sample_d", SMP | DV, o(0x1c)),
    im(0x23, "image_sample_d_cl", SMP | DV | CLAMP, o(0x41)),
    im(0x24, "image_sample_l", SMP | LOD, o(0x1d)),
    im(0x25, "image_sample_b", SMP | BIAS, o(0x1e)),
    im(0x26, "image_sample_b_cl", SMP | BIAS | CLAMP, o(0x42)),
    im(0x27, "image_sample_lz", SMP | LZ, o(0x1f)),
    im(0x28, "image_sample_c", SMP | COMPARE, o(0x20)),
    im(0x29, "image_sample_c_cl", SMP | COMPARE | CLAMP, o(0x43)),
    im(0x2a, "image_sample_c_d", SMP | COMPARE | DV, o(0x21)),
    im(
        0x2b,
        "image_sample_c_d_cl",
        SMP | COMPARE | DV | CLAMP,
        o(0x44),
    ),
    im(0x2c, "image_sample_c_l", SMP | COMPARE | LOD, o(0x22)),
    im(0x2d, "image_sample_c_b", SMP | COMPARE | BIAS, o(0x23)),
    im(
        0x2e,
        "image_sample_c_b_cl",
        SMP | COMPARE | BIAS | CLAMP,
        o(0x45),
    ),
    im(0x2f, "image_sample_c_lz", SMP | COMPARE | LZ, o(0x24)),
    im(0x30, "image_sample_o", SMP | OFFSET, o(0x25)),
    im(0x31, "image_sample_cl_o", SMP | CLAMP | OFFSET, o(0x46)),
    im(0x32, "image_sample_d_o", SMP | DV | OFFSET, o(0x26)),
    im(
        0x33,
        "image_sample_d_cl_o",
        SMP | DV | CLAMP | OFFSET,
        o(0x47),
    ),
    im(0x34, "image_sample_l_o", SMP | LOD | OFFSET, o(0x27)),
    im(0x35, "image_sample_b_o", SMP | BIAS | OFFSET, o(0x28)),
    im(
        0x36,
        "image_sample_b_cl_o",
        SMP | BIAS | CLAMP | OFFSET,
        o(0x48),
    ),
    im(0x37, "image_sample_lz_o", SMP | LZ | OFFSET, o(0x29)),
    im(0x38, "image_sample_c_o", SMP | COMPARE | OFFSET, o(0x2a)),
    im(
        0x39,
        "image_sample_c_cl_o",
        SMP | COMPARE | CLAMP | OFFSET,
        o(0x49),
    ),
    im(
        0x3a,
        "image_sample_c_d_o",
        SMP | COMPARE | DV | OFFSET,
        o(0x2b),
    ),
    im(
        0x3b,
        "image_sample_c_d_cl_o",
        SMP | COMPARE | DV | CLAMP | OFFSET,
        o(0x4a),
    ),
    im(
        0x3c,
        "image_sample_c_l_o",
        SMP | COMPARE | LOD | OFFSET,
        o(0x2c),
    ),
    im(
        0x3d,
        "image_sample_c_b_o",
        SMP | COMPARE | BIAS | OFFSET,
        o(0x2d),
    ),
    im(
        0x3e,
        "image_sample_c_b_cl_o",
        SMP | COMPARE | BIAS | CLAMP | OFFSET,
        o(0x4b),
    ),
    im(
        0x3f,
        "image_sample_c_lz_o",
        SMP | COMPARE | LZ | OFFSET,
        o(0x2e),
    ),
    im(0x40, "image_gather4", GTH, o(0x2f)),
    im(0x41, "image_gather4_cl", GTH | CLAMP, o(0x60)),
    ps5(0x42, "image_load_by2", 0),
    ps5(0x43, "image_load_by4", 0),
    im(0x44, "image_gather4_l", GTH | LOD, o(0x30)),
    im(0x45, "image_gather4_b", GTH | BIAS, o(0x31)),
    im(0x46, "image_gather4_b_cl", GTH | BIAS | CLAMP, o(0x61)),
    im(0x47, "image_gather4_lz", GTH | LZ, o(0x32)),
    im(0x48, "image_gather4_c", GTH | COMPARE, o(0x33)),
    im(0x49, "image_gather4_c_cl", GTH | COMPARE | CLAMP, o(0x62)),
    ps5(0x4a, "image_load_mip_by2", MIP),
    ps5(0x4b, "image_load_mip_by4", MIP),
    im(0x4c, "image_gather4_c_l", GTH | COMPARE | LOD, o(0x63)),
    im(0x4d, "image_gather4_c_b", GTH | COMPARE | BIAS, o(0x64)),
    im(
        0x4e,
        "image_gather4_c_b_cl",
        GTH | COMPARE | BIAS | CLAMP,
        o(0x65),
    ),
    im(0x4f, "image_gather4_c_lz", GTH | COMPARE | LZ, o(0x34)),
    im(0x50, "image_gather4_o", GTH | OFFSET, o(0x35)),
    im(0x51, "image_gather4_cl_o", GTH | CLAMP | OFFSET, N),
    ps5(0x52, "image_store_by2", STORE_OP),
    ps5(0x53, "image_store_by4", STORE_OP),
    im(0x54, "image_gather4_l_o", GTH | LOD | OFFSET, N),
    im(0x55, "image_gather4_b_o", GTH | BIAS | OFFSET, N),
    im(0x56, "image_gather4_b_cl_o", GTH | BIAS | CLAMP | OFFSET, N),
    im(0x57, "image_gather4_lz_o", GTH | LZ | OFFSET, o(0x36)),
    im(0x58, "image_gather4_c_o", GTH | COMPARE | OFFSET, N),
    im(
        0x59,
        "image_gather4_c_cl_o",
        GTH | COMPARE | CLAMP | OFFSET,
        N,
    ),
    ps5(0x5a, "image_store_mip_by2", STORE_OP | MIP),
    ps5(0x5b, "image_store_mip_by4", STORE_OP | MIP),
    im(0x5c, "image_gather4_c_l_o", GTH | COMPARE | LOD | OFFSET, N),
    im(
        0x5d,
        "image_gather4_c_b_o",
        GTH | COMPARE | BIAS | OFFSET,
        N,
    ),
    im(
        0x5e,
        "image_gather4_c_b_cl_o",
        GTH | COMPARE | BIAS | CLAMP | OFFSET,
        N,
    ),
    im(
        0x5f,
        "image_gather4_c_lz_o",
        GTH | COMPARE | LZ | OFFSET,
        o(0x37),
    ),
    im(0x60, "image_get_lod", SMP, o(0x38)),
    im(0x61, "image_gather4h", GTH, o(0x90)),
    im(0x62, "image_gather4h_pck", GTH | PACKED_GATHER, N),
    im(0x63, "image_gather8h_pck", GTH | PACKED_GATHER, N),
    im(0x68, "image_sample_cd", SMP | DV, N),
    im(0x69, "image_sample_cd_cl", SMP | DV | CLAMP, N),
    im(0x6a, "image_sample_c_cd", SMP | COMPARE | DV, N),
    im(0x6b, "image_sample_c_cd_cl", SMP | COMPARE | DV | CLAMP, N),
    im(0x6c, "image_sample_cd_o", SMP | DV | OFFSET, N),
    im(0x6d, "image_sample_cd_cl_o", SMP | DV | CLAMP | OFFSET, N),
    im(0x6e, "image_sample_c_cd_o", SMP | COMPARE | DV | OFFSET, N),
    im(
        0x6f,
        "image_sample_c_cd_cl_o",
        SMP | COMPARE | DV | CLAMP | OFFSET,
        N,
    ),
    ps5(0x70, "image_load_pck2", 0),
    ps5(0x71, "image_load_pck4", 0),
    ps5(0x73, "image_load_mip_pck2", MIP),
    ps5(0x74, "image_load_mip_pck4", MIP),
    ps5(0x76, "image_store_pck2", STORE_OP),
    ps5(0x77, "image_store_pck4", STORE_OP),
    ps5(0x79, "image_store_mip_pck2", STORE_OP | MIP),
    ps5(0x7a, "image_store_mip_pck4", STORE_OP | MIP),
    im(0x80, "image_msaa_load", MSAA, o(0x18)),
    adjusted(0xa0, "image_sample_a", SMP),
    adjusted(0xa1, "image_sample_cl_a", SMP | CLAMP),
    im(0xa2, "image_sample_d_g16", SMP | DV | G16, o(0x39)),
    im(
        0xa3,
        "image_sample_d_cl_g16",
        SMP | DV | CLAMP | G16,
        o(0x5f),
    ),
    adjusted(0xa5, "image_sample_b_a", SMP | BIAS),
    adjusted(0xa8, "image_sample_c_a", SMP | COMPARE),
    im(
        0xaa,
        "image_sample_c_d_g16",
        SMP | COMPARE | DV | G16,
        o(0x3a),
    ),
    im(
        0xab,
        "image_sample_c_d_cl_g16",
        SMP | COMPARE | DV | CLAMP | G16,
        o(0x54),
    ),
    adjusted(0xb0, "image_sample_o_a", SMP | OFFSET),
    im(
        0xb2,
        "image_sample_d_o_g16",
        SMP | DV | OFFSET | G16,
        o(0x3b),
    ),
    im(
        0xb3,
        "image_sample_d_cl_o_g16",
        SMP | DV | CLAMP | OFFSET | G16,
        o(0x55),
    ),
    adjusted(0xb8, "image_sample_c_o_a", SMP | COMPARE | OFFSET),
    im(
        0xba,
        "image_sample_c_d_o_g16",
        SMP | COMPARE | DV | OFFSET | G16,
        o(0x3c),
    ),
    im(
        0xbb,
        "image_sample_c_d_cl_o_g16",
        SMP | COMPARE | DV | CLAMP | OFFSET | G16,
        o(0x56),
    ),
    adjusted(0xc8, "image_gather4_c_a", GTH | COMPARE),
    adjusted(0xd8, "image_gather4_c_o_a", GTH | COMPARE | OFFSET),
    im(0xe6, "image_bvh_intersect_ray", BVH, o(0x19)),
    im(0xe7, "image_bvh64_intersect_ray", BVH | BVH64, o(0x1a)),
    im(0xe8, "image_sample_cd_g16", SMP | DV | G16, N),
    im(0xe9, "image_sample_cd_cl_g16", SMP | DV | CLAMP | G16, N),
    im(0xea, "image_sample_c_cd_g16", SMP | COMPARE | DV | G16, N),
    im(
        0xeb,
        "image_sample_c_cd_cl_g16",
        SMP | COMPARE | DV | CLAMP | G16,
        N,
    ),
    im(0xec, "image_sample_cd_o_g16", SMP | DV | OFFSET | G16, N),
    im(
        0xed,
        "image_sample_cd_cl_o_g16",
        SMP | DV | CLAMP | OFFSET | G16,
        N,
    ),
    im(
        0xee,
        "image_sample_c_cd_o_g16",
        SMP | COMPARE | DV | OFFSET | G16,
        N,
    ),
    im(
        0xef,
        "image_sample_c_cd_cl_o_g16",
        SMP | COMPARE | DV | CLAMP | OFFSET | G16,
        N,
    ),
];

pub const GFX12_MSAA_LOAD: u16 = 0x18;

/// Whether gfx12 encodes the image instruction with gfx12 opcode `opcode` as VSAMPLE and
/// counts it in SAMPLE_CNT: sampling instructions and `image_msaa_load`, which reads no
/// sampler.
pub fn gfx12_vsample(opcode: u16, uses_sampler: bool) -> bool {
    uses_sampler || opcode == GFX12_MSAA_LOAD
}

pub fn smem(op: u16) -> Option<&'static MemOp> {
    SMEM.iter().find(|e| e.op == op)
}

pub fn mubuf(op: u16) -> Option<&'static MemOp> {
    VMEM_COMMON.iter().chain(MUBUF_ONLY).find(|e| e.op == op)
}

pub fn mtbuf(op: u16) -> Option<&'static MemOp> {
    MTBUF.iter().find(|e| e.op == op)
}

pub fn flat(op: u16) -> Option<&'static MemOp> {
    VMEM_COMMON.iter().chain(FLAT_ONLY).find(|e| e.op == op)
}

pub fn ds_op(op: u16) -> Option<&'static DsOp> {
    DS.iter().find(|e| e.op == op)
}

pub fn image(op: u16) -> Option<&'static ImageOp> {
    MIMG.iter().find(|e| e.op == op)
}

pub fn name(family: InstructionFamily, op: u16) -> Option<String> {
    match family {
        InstructionFamily::Smem => smem(op).map(|e| e.name.to_string()),
        InstructionFamily::Mubuf => mubuf(op).map(|e| format!("buffer_{}", e.name)),
        InstructionFamily::Mtbuf => mtbuf(op).map(|e| e.name.to_string()),
        InstructionFamily::Flat => flat(op).map(|e| format!("flat_{}", e.name)),
        InstructionFamily::Ds => ds_op(op).map(|e| e.name.to_string()),
        InstructionFamily::Mimg => image(op).map(|e| e.name.to_string()),
        _ => None,
    }
}

pub fn image_coordinates(dim: u8) -> u8 {
    match dim {
        0 => 1,
        1 | 4 => 2,
        2 | 3 | 5 | 6 => 3,
        _ => 4,
    }
}

fn image_gradients(dim: u8) -> u8 {
    match dim {
        0 | 4 => 1,
        2 => 3,
        _ => 2,
    }
}

pub fn image_address_dwords(flags: u32, dim: u8, a16: bool) -> u8 {
    if flags & BVH != 0 {
        let base = if a16 { 8 } else { 11 };
        return base + u8::from(flags & BVH64 != 0);
    }
    if flags & RESINFO != 0 {
        return 1;
    }
    let mut dwords = u8::from(flags & OFFSET != 0)
        + u8::from(flags & BIAS != 0)
        + u8::from(flags & COMPARE != 0);
    if flags & DERIV != 0 {
        let grads = image_gradients(dim);
        dwords += if flags & G16 != 0 || a16 {
            2 * grads.div_ceil(2)
        } else {
            2 * grads
        };
    }
    let trailing = image_coordinates(dim)
        + u8::from(flags & LOD != 0)
        + u8::from(flags & CLAMP != 0)
        + u8::from(flags & MIP != 0);
    dwords + if a16 { trailing.div_ceil(2) } else { trailing }
}

pub fn image_data_dwords(flags: u32, dmask: u8, d16: bool) -> u8 {
    if flags & BVH != 0 {
        return 4;
    }
    let components = if flags & (GATHER | MSAA) != 0 && flags & PACKED_GATHER == 0 {
        4
    } else {
        dmask.count_ones() as u8
    };
    if d16 {
        components.div_ceil(2)
    } else {
        components
    }
}

const GFX10_REMOVED_FORMATS: &[u8] = &[30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 46, 47];

pub fn remap_buffer_format(format: u8, target: TargetIsa) -> Option<u8> {
    if !target.is_gfx11_plus() || format <= 29 {
        return (format <= 77).then_some(format);
    }
    if GFX10_REMOVED_FORMATS.contains(&format) {
        return None;
    }
    Some(match format {
        36 => 30,
        43 => 31,
        44 | 45 => format - 12,
        48..=77 => format - 14,
        _ => return None,
    })
}
