//! NGG (merged ES/GS) vertex programs on gfx11+: what the runtime programs for a translation.
//!
//! gfx11 removed parameter exports: a translated program stores them to the attribute ring
//! (see [`crate::AttributeRingAbi`]) from which the SPI feeds pixel shaders. Each wave finds
//! its part of the ring in s5 (`gs_attr_offset`, 512-byte units in bits 14:0), where gfx10
//! loaded the scratch wave offset; vertex `v` of the subgroup stores parameter `n` at
//! `ring + offset + (v / 32) * 512 * P + n * 512 + (v % 32) * 16`, the swizzled layout of
//! [`ring_descriptor`].

use crate::{AttributeRingAbi, TargetIsa};

/// SH register byte offsets.
pub const SPI_SHADER_PGM_LO_GS: u32 = 0xb220;
pub const SPI_SHADER_PGM_HI_GS: u32 = 0xb224;
pub const SPI_SHADER_PGM_LO_ES: u32 = 0xb320;
pub const SPI_SHADER_PGM_HI_ES: u32 = 0xb324;

const VS_EXPORT_COUNT_SHIFT: u32 = 1;
const NO_PC_EXPORT: u32 = 1 << 7;
/// Ring descriptor word 3: DST_SEL XYZW, FORMAT 32_32_32_32_FLOAT, INDEX_STRIDE 32 (2),
/// OOB_SELECT structured with offset (0).
const RING_WORD3: u32 = 0x0043_ffac;
/// SWIZZLE_ENABLE = 3: 16-byte elements.
const RING_SWIZZLE_16_BYTES: u32 = 3 << 30;

/// Where the value of a register the runtime programs comes from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RegisterSource {
    /// The GPU address of [`crate::ShaderArtifact::code`]: bits 39:8 for a PGM_LO register,
    /// bits 47:40 for a PGM_HI one.
    CodeAddress,
    /// The guest's SPI_SHADER_USER_DATA_ADDR_LO_GS / HI_GS (SH 0xb208 / 0xb20c). gfx11 loads
    /// PGM_LO/HI_GS into `s[0:1]`, where the guest program expects that pointer.
    GuestUserDataAddress,
}

/// A register the runtime programs from a value only it knows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RegisterRelocation {
    /// SH register byte offset.
    pub register: u32,
    pub source: RegisterSource,
}

/// The program registers of an NGG translation: it runs in the ES slot.
pub const PROGRAM_REGISTERS: [RegisterRelocation; 4] = [
    RegisterRelocation {
        register: SPI_SHADER_PGM_LO_ES,
        source: RegisterSource::CodeAddress,
    },
    RegisterRelocation {
        register: SPI_SHADER_PGM_HI_ES,
        source: RegisterSource::CodeAddress,
    },
    RegisterRelocation {
        register: SPI_SHADER_PGM_LO_GS,
        source: RegisterSource::GuestUserDataAddress,
    },
    RegisterRelocation {
        register: SPI_SHADER_PGM_HI_GS,
        source: RegisterSource::GuestUserDataAddress,
    },
];

/// What the runtime programs for a gfx11+ NGG translation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct NggState {
    /// Parameters per vertex in the attribute ring (P): the highest exported index plus one.
    pub param_count: u32,
    /// SPI_VS_OUT_CONFIG for P ([`spi_vs_out_config`]). The ring descriptor's stride comes
    /// from the same P, so the two always agree.
    pub spi_vs_out_config: u32,
    /// Dword offsets in [`crate::ShaderArtifact::code`] of the two ring-address literals: the
    /// one holding the address' bits 31:0, and the one holding bits 47:32 in its low half (the
    /// stride and swizzle mode in the high half). A cached binary can be patched there for
    /// another ring address. `None` without parameter stores.
    pub attribute_ring_relocations: Option<[u32; 2]>,
    /// SPI_SHADER_PGM_LO/HI_ES and SPI_SHADER_PGM_LO/HI_GS ([`PROGRAM_REGISTERS`]).
    pub registers: [RegisterRelocation; 4],
}

/// SPI_VS_OUT_CONFIG for `param_count` parameters: VS_EXPORT_COUNT = max(P, 1) - 1, and
/// NO_PC_EXPORT when P is 0.
pub fn spi_vs_out_config(param_count: u32) -> u32 {
    let count = param_count.max(1) - 1;
    let none = if param_count == 0 { NO_PC_EXPORT } else { 0 };
    (count << VS_EXPORT_COUNT_SHIFT) | none
}

/// The attribute ring's buffer descriptor (gfx11 layout) for programs with `param_count`
/// parameters: stride 16 × max(P, 1), 16-byte swizzle elements, 32 indices per swizzle row.
pub fn ring_descriptor(ring: AttributeRingAbi, param_count: u32) -> [u32; 4] {
    let stride = 16 * param_count.max(1);
    [
        ring.address as u32,
        ((ring.address >> 32) as u32 & 0xffff) | (stride << 16) | RING_SWIZZLE_16_BYTES,
        ring.bytes,
        RING_WORD3,
    ]
}

const GE_CNTL_PRIMS_PER_SUBGRP_SHIFT: u32 = 0;
const GE_CNTL_VERTS_PER_SUBGRP_SHIFT: u32 = 9;
const GE_CNTL_PRIM_GRP_SIZE_SHIFT: u32 = 21;
const GE_CNTL_GRP_MASK: u32 = 0x1ff;
/// gfx10 GE_CNTL BREAK_WAVE_AT_EOI. gfx11 has BREAK_SUBGRP_AT_EOI at this bit, which radv and
/// radeonsi leave clear.
const GFX10_BREAK_WAVE_AT_EOI: u32 = 1 << 18;
/// PACKET_TO_ONE_PA, at the same bit on gfx10 and gfx11.
const GE_CNTL_PACKET_TO_ONE_PA: u32 = 1 << 19;
/// gfx11 GE_CNTL BREAK_PRIMGRP_AT_EOI, where radv and radeonsi put gfx10's BREAK_WAVE_AT_EOI.
const GFX11_BREAK_PRIMGRP_AT_EOI: u32 = 1 << 20;
/// gfx12 GE_CNTL DIS_PG_SIZE_ADJUST_FOR_STRIP, which radv and radeonsi set there.
const GFX12_DIS_PG_SIZE_ADJUST_FOR_STRIP: u32 = 1 << 31;
const ONCHIP_ES_VERTS_SHIFT: u32 = 0;
const ONCHIP_GS_PRIMS_SHIFT: u32 = 11;
const ONCHIP_FIELD_MASK: u32 = 0x7ff;

/// GE_CNTL (gfx11+) for the guest's subgroup limits, which culling programs lay out LDS for:
/// PRIMS_PER_SUBGRP and VERTS_PER_SUBGRP from the guest's VGT_GS_ONCHIP_CNTL
/// (GS_PRIMS_PER_SUBGRP, ES_VERTS_PER_SUBGRP), or from its GE_CNTL (PRIM_GRP_SIZE,
/// VERT_GRP_SIZE) when VGT_GS_ONCHIP_CNTL is zero; the guest's BREAK_WAVE_AT_EOI as
/// BREAK_PRIMGRP_AT_EOI (bit 20) and its PACKET_TO_ONE_PA (bit 19); PRIM_GRP_SIZE 252 (256 and
/// DIS_PG_SIZE_ADJUST_FOR_STRIP on gfx12), as radv and radeonsi program it.
pub fn ge_cntl(target: TargetIsa, vgt_gs_onchip_cntl: u32, guest_ge_cntl: u32) -> u32 {
    let (prims, verts) = if vgt_gs_onchip_cntl != 0 {
        (
            (vgt_gs_onchip_cntl >> ONCHIP_GS_PRIMS_SHIFT) & ONCHIP_FIELD_MASK,
            (vgt_gs_onchip_cntl >> ONCHIP_ES_VERTS_SHIFT) & ONCHIP_FIELD_MASK,
        )
    } else {
        (
            guest_ge_cntl & GE_CNTL_GRP_MASK,
            (guest_ge_cntl >> GE_CNTL_VERTS_PER_SUBGRP_SHIFT) & GE_CNTL_GRP_MASK,
        )
    };
    let (prim_grp_size, strip_adjust) = if target.is_gfx12() {
        (256, GFX12_DIS_PG_SIZE_ADJUST_FOR_STRIP)
    } else {
        (252, 0)
    };
    let break_at_eoi = if guest_ge_cntl & GFX10_BREAK_WAVE_AT_EOI != 0 {
        GFX11_BREAK_PRIMGRP_AT_EOI
    } else {
        0
    };
    ((prims & GE_CNTL_GRP_MASK) << GE_CNTL_PRIMS_PER_SUBGRP_SHIFT)
        | ((verts & GE_CNTL_GRP_MASK) << GE_CNTL_VERTS_PER_SUBGRP_SHIFT)
        | (guest_ge_cntl & GE_CNTL_PACKET_TO_ONE_PA)
        | break_at_eoi
        | ((prim_grp_size & GE_CNTL_GRP_MASK) << GE_CNTL_PRIM_GRP_SIZE_SHIFT)
        | strip_adjust
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vs_out_config_and_the_ring_stride_come_from_one_parameter_count() {
        assert_eq!(spi_vs_out_config(0), NO_PC_EXPORT);
        assert_eq!(spi_vs_out_config(1), 0);
        assert_eq!(spi_vs_out_config(4), 3 << 1);
        let ring = AttributeRingAbi {
            address: 0x1_0100_0000,
            bytes: 0x2c_0000,
        };
        // The interface note's example: VA bits 47:32 = 1 and 4 parameters.
        assert_eq!(
            ring_descriptor(ring, 4),
            [0x0100_0000, 0xc040_0001, 0x2c_0000, 0x0043_ffac]
        );
        assert_eq!(ring_descriptor(ring, 0)[1] >> 16 & 0x3fff, 16);
    }

    #[test]
    fn the_ring_descriptor_has_the_gfx12_layout_too() {
        // Mesa's ac_build_attr_ring_descriptor builds the same words for gfx11 and gfx12: word1
        // is unchanged, and gfx12's word3 keeps FORMAT (bits 17:12, the gfx11 format table's
        // 63 for 32_32_32_32_FLOAT), INDEX_STRIDE (22:21) and OOB_SELECT (29:28) in place, with
        // its new STRIDE_SCALE (19:18) and compression bits (27:24) zero.
        let ring = AttributeRingAbi {
            address: 0x2_0000_0000,
            bytes: 0x10_0000,
        };
        let [_, word1, _, word3] = ring_descriptor(ring, 2);
        assert_eq!((word1 >> 30, word1 >> 16 & 0x3fff), (3, 32));
        assert_eq!(word3 >> 12 & 0x3f, 63, "FORMAT");
        assert_eq!(word3 >> 21 & 3, 2, "INDEX_STRIDE");
        assert_eq!(word3 >> 28 & 3, 0, "OOB_SELECT");
        assert_eq!(
            word3 & (3 << 18 | 0xf << 24),
            0,
            "STRIDE_SCALE and compression"
        );
    }

    #[test]
    fn ge_cntl_keeps_the_guest_subgroup_limits() {
        // ES_VERTS 100, GS_PRIMS 120 (GS_INST_PRIMS 120); the guest GE_CNTL sets
        // BREAK_WAVE_AT_EOI (bit 18) and PACKET_TO_ONE_PA (bit 19).
        let onchip = 100 | (120 << 11) | (120 << 22);
        let guest = 120 | (100 << 9) | (1 << 18) | (1 << 19);
        let host = ge_cntl(TargetIsa::Gfx115, onchip, guest);
        // Mesa's gfx11 GE_CNTL: PRIMS_PER_SUBGRP 8:0, VERTS_PER_SUBGRP 17:9,
        // BREAK_SUBGRP_AT_EOI 18, PACKET_TO_ONE_PA 19, BREAK_PRIMGRP_AT_EOI 20,
        // PRIM_GRP_SIZE 29:21, DIS_PG_SIZE_ADJUST_FOR_STRIP 31. radv and radeonsi put gfx10's
        // BREAK_WAVE_AT_EOI in BREAK_PRIMGRP_AT_EOI and leave BREAK_SUBGRP_AT_EOI clear.
        assert_eq!(host & 0x1ff, 120, "PRIMS_PER_SUBGRP");
        assert_eq!((host >> 9) & 0x1ff, 100, "VERTS_PER_SUBGRP");
        assert_eq!(host & (1 << 18), 0, "BREAK_SUBGRP_AT_EOI");
        assert_eq!(host & (1 << 19), 1 << 19, "PACKET_TO_ONE_PA");
        assert_eq!(host & (1 << 20), 1 << 20, "BREAK_PRIMGRP_AT_EOI");
        assert_eq!(host >> 21 & 0x1ff, 252, "PRIM_GRP_SIZE");
        assert_eq!(host, 120 | (100 << 9) | (3 << 19) | (252 << 21));
        let gfx12 = ge_cntl(TargetIsa::Gfx12, onchip, guest);
        assert_eq!(
            gfx12,
            120 | (100 << 9) | (3 << 19) | (256 << 21) | (1 << 31)
        );
        // Without the break the bit stays clear.
        let plain = ge_cntl(TargetIsa::Gfx115, onchip, 120 | (100 << 9));
        assert_eq!(plain & (7 << 18), 0);
        // Without VGT_GS_ONCHIP_CNTL the guest GE_CNTL group sizes stand in.
        let fallback = ge_cntl(TargetIsa::Gfx115, 0, 64 | (96 << 9));
        assert_eq!((fallback & 0x1ff, (fallback >> 9) & 0x1ff), (64, 96));
    }
}
