//! Field-level translations of guest gfx10 register values into their gfx11.5 layouts
//! (`s2-regstate/report.md` §2.3-§2.4, `s2-drawstate/REPORT.md` §3.3). Each function takes the
//! guest value and returns the host value, or the reason the value has no gfx11.5 encoding.

fn field(value: u32, lo: u32, width: u32) -> u32 {
    (value >> lo) & ((1 << width) - 1)
}

fn replace(value: u32, lo: u32, width: u32, field_value: u32) -> u32 {
    let mask = ((1u32 << width) - 1) << lo;
    (value & !mask) | ((field_value << lo) & mask)
}

pub type Translated = Result<u32, &'static str>;

/// DB_RENDER_CONTROL: keep [12:0], PS_INVOKE_DISABLE [13] to [14], and keep host depth and
/// stencil uncompressed (no host HTILE). With no HTILE, resummarize and decompress passes have
/// nothing to do, so those bits are dropped and the pass draws as the CPU executor draws it;
/// depth/stencil-to-colour copies move data and need a shader pass.
pub fn db_render_control(guest: u32) -> Translated {
    const DEPTH_COPY: u32 = 1 << 2;
    const STENCIL_COPY: u32 = 1 << 3;
    const RESUMMARIZE_ENABLE: u32 = 1 << 4;
    const STENCIL_COMPRESS_DISABLE: u32 = 1 << 5;
    const DEPTH_COMPRESS_DISABLE: u32 = 1 << 6;
    const DECOMPRESS_ENABLE: u32 = 1 << 12;
    if guest & (DEPTH_COPY | STENCIL_COPY) != 0 {
        return Err("DB_RENDER_CONTROL depth/stencil copy passes need a shader copy");
    }
    Ok((guest & 0x0fff & !RESUMMARIZE_ENABLE & !DECOMPRESS_ENABLE)
        | (field(guest, 13, 1) << 14)
        | STENCIL_COMPRESS_DISABLE
        | DEPTH_COMPRESS_DISABLE)
}

/// PA_SC_FSR_ENABLE (`sce::Agc::CxPaScFsrEnable`, PS5 only): flexible scaled rasterization of
/// viewports [15:0], their any-enabled summary [31]. gfx11 rasterizes without it, so only the
/// value that enables none is honoured.
pub fn fsr_disabled(guest: u32) -> Translated {
    if guest == 0 {
        Ok(0)
    } else {
        Err("flexible scaled rasterization (PS5 FSR distortion viewports) has no gfx11 counterpart")
    }
}

/// DB_COUNT_CONTROL: ZPASS_INCREMENT_DISABLE [0] is gone.
pub fn db_count_control(guest: u32) -> Translated {
    Ok(guest & !1)
}

/// DB_RENDER_OVERRIDE: FORCE_QC_SMASK_CONFLICT [15] is gone.
pub fn db_render_override(guest: u32) -> Translated {
    Ok(guest & !(1 << 15))
}

/// gfx11 depth swizzles are 64KB_Z_X and 256KB_Z_X (banned on APUs); host-layout depth copies
/// use 64KB_Z_X.
const SW_64KB_Z_X: u32 = 24;

/// DB_Z_INFO: no HTILE on the host (TILE_SURFACE_ENABLE [29] clear), host copy in 64KB_Z_X. A
/// null depth target gets NUM_SAMPLES 3 so the DB never clamps the exposed sample count.
pub fn db_z_info(guest: u32) -> Translated {
    const TILE_SURFACE_ENABLE: u32 = 1 << 29;
    let host = guest & !TILE_SURFACE_ENABLE;
    Ok(if field(host, 0, 2) == 0 {
        replace(host, 2, 2, 3)
    } else {
        replace(host, 4, 5, SW_64KB_Z_X)
    })
}

/// DB_STENCIL_INFO: TILE_STENCIL_DISABLE [29] set (no HTILE), host copy in 64KB_Z_X.
pub fn db_stencil_info(guest: u32) -> Translated {
    const TILE_STENCIL_DISABLE: u32 = 1 << 29;
    let host = guest | TILE_STENCIL_DISABLE;
    Ok(if field(host, 0, 1) == 0 {
        host
    } else {
        replace(host, 4, 5, SW_64KB_Z_X)
    })
}

/// PA_SU_HARDWARE_SCREEN_OFFSET: gfx11 needs 32-pixel alignment; odd 16-pixel fields round down
/// (the caller must then recompute the guardband).
pub fn hardware_screen_offset(guest: u32) -> (u32, bool) {
    let x = field(guest, 0, 9);
    let y = field(guest, 16, 9);
    ((x & !1) | ((y & !1) << 16), (x | y) & 1 != 0)
}

/// SPI_PS_INPUT_CNTL_n: CYL_WRAP [16:13] is gone (it would need shader emulation); OFFSET,
/// DEFAULT_VAL, FLAT_SHADE, the point-sprite overrides and FP16 fields keep their bits.
pub fn spi_ps_input_cntl(guest: u32) -> Translated {
    if field(guest, 13, 4) != 0 {
        return Err("SPI_PS_INPUT_CNTL CYL_WRAP has no gfx11 field; emulate in the pixel shader");
    }
    Ok(guest & !(0xf << 13) & !(1 << 12) & !(1 << 11))
}

/// SPI_PS_IN_CONTROL: keep NUM_INTERP [5:0] and PS_W32_EN [15]; no off-chip parameters or
/// primitive interpolants on the host.
pub fn spi_ps_in_control(guest: u32) -> Translated {
    Ok(field(guest, 0, 6) | (field(guest, 15, 1) << 15))
}

/// SPI_TMPRING_SIZE / COMPUTE_TMPRING_SIZE: WAVESIZE [24:12] in 1 KiB units becomes [26:12] in
/// 256-byte units, and WAVES counts per shader engine.
pub fn tmpring_size(guest: u32, shader_engines: u32) -> Translated {
    let waves = field(guest, 0, 12);
    let wave_kib = field(guest, 12, 13);
    let wave_256 = wave_kib * 4;
    if wave_256 >= 1 << 15 {
        return Err("TMPRING_SIZE wave size does not fit the gfx11 field");
    }
    Ok(waves.div_ceil(shader_engines.max(1)) | (wave_256 << 12))
}

/// One gfx10 blend factor (`BlendOp`): 0-10 unchanged, 13-20 shift down by two, 11/12
/// (BOTH_SRC_ALPHA, BOTH_INV_SRC_ALPHA) have no gfx11 encoding.
pub fn blend_factor(guest: u32) -> Translated {
    match guest {
        0..=10 => Ok(guest),
        11 | 12 => Err("blend factor BOTH_SRC_ALPHA/BOTH_INV_SRC_ALPHA has no gfx11 encoding"),
        13..=20 => Ok(guest - 2),
        _ => Err("blend factor is out of range"),
    }
}

/// CB_BLENDn_CONTROL: translate the four factor enums, keep comb functions and enables.
/// CB_BLENDn_CONTROL with each factor renumbered. BOTH_SRC_ALPHA / BOTH_INV_SRC_ALPHA, which
/// gfx11 lacks, are emulated: as a source factor they become SRC_ALPHA / ONE_MINUS_SRC_ALPHA and
/// force the destination factor to the opposite one (D3D9 BOTHSRCALPHA semantics, as the CPU
/// executor applies them); as a destination factor they read as the plain alpha factor.
pub fn cb_blend_control(guest: u32) -> Translated {
    let mut host = guest;
    for (source_lo, destination_lo) in [(0, 8), (16, 24)] {
        let source = field(guest, source_lo, 5);
        let destination = destination_factor(
            source,
            field(guest, destination_lo, 5),
        );
        let plain = |factor: u32| match factor {
            11 => Ok(4),
            12 => Ok(5),
            other => blend_factor(other),
        };
        host = replace(host, source_lo, 5, plain(source)?);
        host = replace(host, destination_lo, 5, plain(destination)?);
    }
    Ok(host)
}

/// CB_COLOR_CONTROL: MODE DISABLE/NORMAL unchanged. Host colour surfaces are never compressed,
/// so ELIMINATE_FAST_CLEAR (2), DECOMPRESS (4), FMASK_DECOMPRESS (5) and DCC_DECOMPRESS (6)
/// passes write nothing, as in the CPU executor: they become DISABLE. RESOLVE has no gfx11
/// mode and needs a shader resolve. Keeps DISABLE_DUAL_QUAD, DEGAMMA_ENABLE and ROP3.
pub fn cb_color_control(guest: u32) -> Translated {
    let mode = match field(guest, 4, 3) {
        mode @ (0 | 1) => mode,
        2 | 4..=6 => 0,
        3 => return Err("CB_COLOR_CONTROL RESOLVE has no gfx11 mode; resolve in a shader"),
        _ => return Err("CB_COLOR_CONTROL reserved mode"),
    };
    Ok(replace(guest & 0x00ff_0009, 4, 3, mode))
}

/// DB_SHADER_CONTROL: EXEC_IF_OVERLAPPED [17] and POPS_OVERLAP_NUM_SAMPLES [22:20] are gone,
/// and COVERAGE_TO_MASK_ENABLE [7] stays clear (gfx11 takes MRTZ.a by default).
pub fn db_shader_control(guest: u32) -> Translated {
    Ok(guest & !(1 << 17) & !(7 << 20) & !(1 << 7))
}

/// PA_CL_VS_OUT_CNTL: USE_VTX_GS_CUT_FLAG [25] and USE_VTX_SHD_OBJPRIM_ID [26] are gone.
pub fn pa_cl_vs_out_cntl(guest: u32) -> Translated {
    Ok(guest & !(3 << 25))
}

/// PA_SU_LINE_STIPPLE_CNTL: DIAMOND_ADJUST [4] is gone.
pub fn pa_su_line_stipple_cntl(guest: u32) -> Translated {
    Ok(guest & !(1 << 4))
}

/// PA_SC_MODE_CNTL_0: SCALE_LINE_WIDTH_PAD [4] is gone.
pub fn pa_sc_mode_cntl_0(guest: u32) -> Translated {
    Ok(guest & !(1 << 4))
}

/// VGT_DRAW_PAYLOAD_CNTL: OBJPRIM_ID_EN [0] and OBJECT_ID_INST_EN [2] are gone.
pub fn vgt_draw_payload_cntl(guest: u32) -> Translated {
    Ok(guest & !0b101)
}

/// VGT_SHADER_STAGES_EN: drop DISPATCH_DRAW_EN and the dealloc-accum bits, clear VS_W32_EN (no
/// hardware VS) and require PRIMGEN_EN; legacy routings must be re-expressed as NGG. The NGG
/// vertex stage runs as a domain shader with tessellation (ES_EN = DS) and as a real ES
/// without (ES_EN = REAL), as gfx11 requires.
pub fn vgt_shader_stages_en(guest: u32) -> Translated {
    const HS_EN: u32 = 1 << 2;
    const ES_STAGE_DS: u32 = 1;
    const ES_STAGE_REAL: u32 = 2;
    const PRIMGEN_EN: u32 = 1 << 13;
    const KEEP: u32 = 0x07ff_f1ff;
    const VS_W32_EN: u32 = 1 << 23;
    if guest & PRIMGEN_EN == 0 {
        return Err(
            "VGT_SHADER_STAGES_EN without PRIMGEN_EN: legacy vertex routing has no gfx11 path",
        );
    }
    let host = guest & KEEP & !VS_W32_EN;
    let es = if guest & HS_EN == 0 {
        ES_STAGE_REAL
    } else {
        ES_STAGE_DS
    };
    Ok(replace(host, 3, 2, es))
}

/// VGT_TF_PARAM: DETECT_ONE [19] and DETECT_ZERO [20] are one bit on both generations, but
/// both gfx10 values (pre- and post-clamp detection) mean "detect", which is gfx11 value 0
/// (ENABLE_TF*_OPT); gfx11 value 1 disables the optimization. Bits 8 and 9 must be 0 on gfx11.
pub fn vgt_tf_param(guest: u32) -> Translated {
    Ok(guest & !(0b11 << 19) & !(0b11 << 8))
}

/// VGT_TF_RING_SIZE: SIZE [16:0] in dwords covers the whole tessellation factor ring on gfx10
/// and each shader engine's share of it on gfx11, so the guest ring is split across the host's
/// `shader_engines` (its footprint unchanged, at least one dword each).
pub fn vgt_tf_ring_size(guest: u32, shader_engines: u32) -> Translated {
    Ok((field(guest, 0, 17) / shader_engines.max(1)).clamp(1, 0x1ffff))
}

/// VGT_HS_OFFCHIP_PARAM: gfx10's OFFCHIP_BUFFERING [8:0] and OFFCHIP_GRANULARITY [10:9] are
/// gfx11's [9:0] and [11:10].
pub fn vgt_hs_offchip_param(guest: u32) -> Translated {
    Ok(field(guest, 0, 9) | (field(guest, 9, 2) << 10))
}

/// VGT_PRIMITIVE_TYPE PRIM_TYPE [5:0]: DI_PT_2D_RECTANGLE (7), which the guest's full-screen
/// passes draw with three corners per rectangle, becomes DI_PT_RECTLIST (17), the rectangle list
/// radv draws on gfx11 with a RECT_2D output primitive ([`vgt_gs_out_prim_type`]); every other
/// type passes.
pub fn vgt_primitive_type(guest: u32) -> u32 {
    const DI_PT_2D_RECTANGLE: u32 = 7;
    const DI_PT_RECTLIST: u32 = 17;
    if guest & 0x3f == DI_PT_2D_RECTANGLE {
        (guest & !0x3f) | DI_PT_RECTLIST
    } else {
        guest
    }
}

/// VGT_GS_OUT_PRIM_TYPE (moved to user config 0x30998): OUTPRIM_TYPE [5:0] keeps its value.
/// gfx10's RECTLIST (3) is gfx11's RECT_2D, the output primitive radv and radeonsi draw
/// rectangle lists with on gfx11 NGG (Mesa dddaef6f8c9: both rectangle types are meant to behave
/// the same); neither ever uses gfx11's new RECTLIST (4), which this used to translate 3 into.
pub fn vgt_gs_out_prim_type(guest: u32) -> Translated {
    Ok(field(guest, 0, 6))
}

/// GE_MULTI_PRIM_IB_RESET_EN: keep RESET_EN and MATCH_ALL_BITS, add DISABLE_FOR_AUTO_INDEX.
pub fn ge_multi_prim_ib_reset_en(guest: u32) -> Translated {
    Ok((guest & 0b11) | 0b100)
}

/// GE_CNTL from the gfx10 GE_CNTL and VGT_GS_ONCHIP_CNTL (removed on gfx11), as the compiler
/// computes it for its translations ([`shader::ngg::ge_cntl`]). A guest that sizes no
/// subgroups at all gets radv's 128 primitives and vertices for a passthrough VS.
pub fn ge_cntl(
    target: shader::TargetIsa,
    guest_ge_cntl: u32,
    guest_gs_onchip_cntl: u32,
) -> Translated {
    const PASSTHROUGH_SUBGROUP: u32 = 128 | (128 << 11);
    const GROUP_SIZES: u32 = 0x3_ffff;
    let (prims, verts) = (
        field(guest_gs_onchip_cntl, 11, 11),
        field(guest_gs_onchip_cntl, 0, 11),
    );
    if prims > 0x1ff || verts > 0x1ff {
        return Err("NGG subgroup sizes exceed the gfx11 GE_CNTL fields");
    }
    let onchip = if guest_gs_onchip_cntl == 0 && guest_ge_cntl & GROUP_SIZES == 0 {
        PASSTHROUGH_SUBGROUP
    } else {
        guest_gs_onchip_cntl
    };
    Ok(shader::ngg::ge_cntl(target, onchip, guest_ge_cntl))
}

/// CB_COLORn_INFO: FORMAT [6:2] to [4:0]; keep LINEAR_GENERAL, NUMBER_TYPE, COMP_SWAP, the
/// blend and rounding bits and BLEND_OPT_*; drop ENDIAN, fast clear, compression, CMASK/FMASK
/// and DCC_ENABLE (which moves to FDCC_CONTROL).
pub fn cb_color_info(guest: u32) -> Translated {
    let format = field(guest, 2, 5);
    if format == 31 {
        return Err("CB_COLOR_INFO format 2_10_10_10_6E4 has no gfx11 encoding");
    }
    Ok(format
        | (field(guest, 7, 1) << 7)
        | (field(guest, 8, 3) << 8)
        | (field(guest, 11, 2) << 11)
        | (field(guest, 15, 4) << 15)
        | (field(guest, 20, 6) << 20))
}

/// SX_MRTn_BLEND_OPT for guest CB_BLENDn_CONTROL, as radv's `radv_translate_blend_equation` and
/// `radv_emit_color_blend` compute it: every MRT needs a value, a blend-disabled one included,
/// because RB+ drops the colour of an MRT left at OPT_COMB_NONE.
pub fn sx_mrt_blend_opt(blend: u32) -> u32 {
    const ENABLE: u32 = 1 << 30;
    const SEPARATE_ALPHA: u32 = 1 << 29;
    const OPT_COMB_BLEND_DISABLED: u32 = 6;
    const COLOR_COMB_SHIFT: u32 = 8;
    const ALPHA_COMB_SHIFT: u32 = 24;
    if blend & ENABLE == 0 {
        return (OPT_COMB_BLEND_DISABLED << COLOR_COMB_SHIFT) | (OPT_COMB_BLEND_DISABLED << ALPHA_COMB_SHIFT);
    }
    // BLEND_OPT_PRESERVE_* values of the SRC and DST fields.
    const NONE_IGNORE_ALL: u32 = 0;
    const ALL_IGNORE_NONE: u32 = 1;
    const C1_IGNORE_C0: u32 = 2;
    const C0_IGNORE_C1: u32 = 3;
    const A1_IGNORE_A0: u32 = 4;
    const A0_IGNORE_A1: u32 = 5;
    const NONE_IGNORE_A0: u32 = 6;
    const NONE_IGNORE_NONE: u32 = 7;
    // BLEND_* factors of CB_BLENDn_CONTROL.
    const ZERO: u32 = 0;
    const ONE: u32 = 1;
    const SRC_COLOR: u32 = 2;
    const ONE_MINUS_SRC_COLOR: u32 = 3;
    const SRC_ALPHA: u32 = 4;
    const ONE_MINUS_SRC_ALPHA: u32 = 5;
    const DST_ALPHA: u32 = 6;
    const ONE_MINUS_DST_ALPHA: u32 = 7;
    const DST_COLOR: u32 = 8;
    const ONE_MINUS_DST_COLOR: u32 = 9;
    const SRC_ALPHA_SATURATE: u32 = 10;
    let opt_factor = |factor: u32, is_alpha: bool| match factor {
        ZERO => NONE_IGNORE_ALL,
        ONE => ALL_IGNORE_NONE,
        SRC_COLOR => if is_alpha { A1_IGNORE_A0 } else { C1_IGNORE_C0 },
        ONE_MINUS_SRC_COLOR => if is_alpha { A0_IGNORE_A1 } else { C0_IGNORE_C1 },
        SRC_ALPHA => A1_IGNORE_A0,
        ONE_MINUS_SRC_ALPHA => A0_IGNORE_A1,
        SRC_ALPHA_SATURATE => if is_alpha { ALL_IGNORE_NONE } else { NONE_IGNORE_A0 },
        _ => NONE_IGNORE_NONE,
    };
    let uses_dst = |factor: u32| matches!(factor, DST_ALPHA | ONE_MINUS_DST_ALPHA | DST_COLOR | ONE_MINUS_DST_COLOR | SRC_ALPHA_SATURATE);
    // CB COMB_FCN (ADD, SUBTRACT, MIN, MAX, REVSUBTRACT) to OPT_COMB_*.
    let opt_function = |function: u32| match function {
        0 => 1,
        1 => 2,
        2 => 3,
        3 => 4,
        _ => 5,
    };
    let (src_rgb, function_rgb, dst_rgb) = (field(blend, 0, 5), field(blend, 5, 3), field(blend, 8, 5));
    let (src_a, function_a, dst_a) = if blend & SEPARATE_ALPHA != 0 { (field(blend, 16, 5), field(blend, 21, 3), field(blend, 24, 5)) } else { (src_rgb, function_rgb, dst_rgb) };
    // MIN and MAX ignore the factors: radv's normalization makes them ONE.
    let (src_rgb, dst_rgb) = if matches!(function_rgb, 2 | 3) { (ONE, ONE) } else { (src_rgb, dst_rgb) };
    let (src_a, dst_a) = if matches!(function_a, 2 | 3) { (ONE, ONE) } else { (src_a, dst_a) };
    let mut dst_rgb_opt = opt_factor(dst_rgb, false);
    let mut dst_a_opt = opt_factor(dst_a, true);
    if uses_dst(src_rgb) {
        dst_rgb_opt = NONE_IGNORE_NONE;
    }
    if uses_dst(src_a) {
        dst_a_opt = NONE_IGNORE_NONE;
    }
    if src_rgb == SRC_ALPHA_SATURATE && matches!(dst_rgb, ZERO | SRC_ALPHA | SRC_ALPHA_SATURATE) {
        dst_rgb_opt = NONE_IGNORE_A0;
    }
    opt_factor(src_rgb, false)
        | (dst_rgb_opt << 4)
        | (opt_function(function_rgb) << COLOR_COMB_SHIFT)
        | (opt_factor(src_a, true) << 16)
        | (dst_a_opt << 20)
        | (opt_function(function_a) << ALPHA_COMB_SHIFT)
}

/// CB_COLORn_ATTRIB: NUM_FRAGMENTS [16:15] to [1:0], FORCE_DST_ALPHA_1 [17] to [2],
/// LIMIT_COLOR_FETCH_TO_256B_MAX [19] to [4]; tile-mode, FMASK and sample fields are gone.
pub fn cb_color_attrib(guest: u32) -> Translated {
    Ok(field(guest, 15, 2) | (field(guest, 17, 1) << 2) | (field(guest, 19, 1) << 4))
}

/// CB_COLORn_DCC_CONTROL to FDCC_CONTROL. Host colour surfaces stay uncompressed, so
/// FDCC_ENABLE is never set; the other fields keep their meaning.
pub fn cb_color_fdcc_control(guest: u32) -> Translated {
    Ok((guest & 0x000c_03fc) | (field(guest, 20, 1) << 10))
}

/// CB_COLORn_ATTRIB3: keep MIP0_DEPTH, META_LINEAR, RESOURCE_TYPE and DCC_PIPE_ALIGNED; drop
/// FMASK_SW_MODE, CMASK_PIPE_ALIGNED and RESOURCE_LEVEL. COLOR_SW_MODE becomes the mode of the
/// host-layout copy the target renders into (gfx11 has no 2D S modes).
pub fn cb_color_attrib3(guest: u32) -> Translated {
    use crate::amd::surface::tiling::{host_swizzle_mode, SurfaceUse};
    const RESOURCE_TYPE_3D: u32 = 2;
    let usage = if field(guest, 24, 2) == RESOURCE_TYPE_3D {
        SurfaceUse::Volume
    } else {
        SurfaceUse::Thin2d
    };
    let mode = host_swizzle_mode(field(guest, 14, 5), usage);
    Ok(replace(
        guest & (0x1fff | (1 << 13) | (3 << 24) | (1 << 30)),
        14,
        5,
        mode,
    ))
}

/// The destination factor a source factor leaves in force: BOTH_SRC_ALPHA (11) and
/// BOTH_INV_SRC_ALPHA (12), D3D9's legacy source-only factors, also set the destination to the
/// opposite alpha factor.
fn destination_factor(source: u32, destination: u32) -> u32 {
    match source {
        11 => 5,
        12 => 4,
        _ => destination,
    }
}
