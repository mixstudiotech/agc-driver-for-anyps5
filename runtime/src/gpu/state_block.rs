//! The per-submission gfx11.5 state block. Every host submission starts with it and assumes
//! nothing from earlier submissions: no CLEAR_STATE, no register shadowing (CONTEXT_CONTROL
//! with only the update-enable bits), plain SET_CONTEXT_REG on this APU, and gfx11.5 values
//! where radv-on-WDDM would program gfx11.0 ones (binning mode 3, no ping-pong bin order).
//!
//! The order and values follow `native-gfx-research/s1-submission/preamble_gfx1151.md`, which is
//! the GFX11 branch of Mesa's `ac_init_graphics_preamble_state` plus radv's queue preamble,
//! evaluated for gfx1151 (2 SE, 8 RB, 10 good CUs per SA, 32 MiB MALL).

use super::{
    packets::{event, gcr, PacketStream},
    regs::*,
};
use crate::{Error, Result};

/// gfx1151 topology as the KMD reports it on the 8060S.
pub const SHADER_ENGINES: u32 = 2;
pub const RENDER_BACKENDS: u32 = 8;
/// Attribute ring bytes per shader engine: 1400 KiB aligned up to 64 KiB, the size Mesa picks
/// when the chip has a MALL.
pub const ATTRIBUTE_RING_BYTES_PER_SE: u32 = 1408 * 1024;
pub const ATTRIBUTE_RING_BYTES: u32 = ATTRIBUTE_RING_BYTES_PER_SE * SHADER_ENGINES;
/// Mesa places the ring on a 2 MiB boundary; the register itself needs 64 KiB.
pub const ATTRIBUTE_RING_ALIGNMENT: u64 = 2 << 20;
const ATTRIBUTE_RING_BASE_SHIFT: u32 = 16;

/// SPI_SHADER_PGM_RSRC3_PS: CU_EN = the 10 good CUs per SA, WAVE_LIMIT 0x3f, LDS_GROUP_SIZE 1.
const RSRC3_PS: u32 = 0x007f_03ff;
/// SPI_SHADER_PGM_RSRC4_PS: CU_EN for CUs 16-31, none on gfx1151.
const RSRC4_PS: u32 = 0;
const RSRC3_HS: u32 = 0xffff_003f;
const RSRC4_HS: u32 = 0x0000_ffff;
/// Compute may use every CU of the first two SEs.
const ALL_CUS: u32 = 0xffff_ffff;

/// Where the attribute ring lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttributeRing {
    pub address: u64,
    pub bytes_per_se: u32,
}

impl AttributeRing {
    pub fn new(address: u64) -> Result<Self> {
        if address == 0 || address & ((1 << ATTRIBUTE_RING_BASE_SHIFT) - 1) != 0 {
            return Err(Error::invalid(format!(
                "attribute ring 0x{address:x} is not 64 KiB aligned"
            )));
        }
        Ok(Self {
            address,
            bytes_per_se: ATTRIBUTE_RING_BYTES_PER_SE,
        })
    }

    pub fn total_bytes(&self) -> u32 {
        self.bytes_per_se * SHADER_ENGINES
    }

    /// SPI_ATTRIBUTE_RING_BASE.
    pub fn base_register(&self) -> u32 {
        (self.address >> ATTRIBUTE_RING_BASE_SHIFT) as u32
    }

    /// SPI_ATTRIBUTE_RING_SIZE: MEM_SIZE[7:0] in 64 KiB units minus one, BIG_PAGE[16] 0 (APU),
    /// L1_POLICY[18:17] 1.
    pub fn size_register(&self) -> u32 {
        const L1_POLICY: u32 = 1 << 17;
        ((self.bytes_per_se >> ATTRIBUTE_RING_BASE_SHIFT) - 1) | L1_POLICY
    }

    /// The gfx11 V# translated NGG shaders store parameters through: 16-byte swizzle,
    /// index stride 32, 32_32_32_32_FLOAT, STRIDE = 16 bytes per exported parameter.
    pub fn descriptor(&self, parameters: u32) -> [u32; 4] {
        const SWIZZLE_16_BYTES: u32 = 3 << 30;
        const DST_SEL_XYZW: u32 = 0xfac;
        const FORMAT_32_32_32_32_FLOAT: u32 = 63 << 12;
        const INDEX_STRIDE_32: u32 = 2 << 21;
        let stride = 16u32.saturating_mul(parameters.max(1));
        [
            self.address as u32,
            crate::amd::descriptors::buffer_word1(self.address, stride) | SWIZZLE_16_BYTES,
            self.total_bytes(),
            DST_SEL_XYZW | FORMAT_32_32_32_32_FLOAT | INDEX_STRIDE_32,
        ]
    }
}

/// PA_SC_TILE_STEERING_OVERRIDE for gfx11/gfx11.5. Mesa writes the kernel's value
/// (`ac_cmdbuf.c:499` `info->pa_sc_tile_steering_override`, copied at `ac_gpu_info.c:1130`);
/// Linux sets it to 0 for every gfx11 part (`gfx_v11_0.c` `gfx_v11_0_constants_init`), Mesa's
/// recorded GFX11.5 device (`amdgpu_devices.c`, "strix1") reports 0, and the WDDM winsys in
/// `references/mesa-wddm` never fills it. Nothing derives from chip topology.
pub const TILE_STEERING_OVERRIDE: u32 = 0;

/// PA_RATE_CNTL (VERTEX_RATE [3:0], PRIM_RATE [7:4]). radv, radeonsi and `ac_*` never write it
/// on any gfx level, so Mesa runs with the gfx11 clear-state value the kernel loads at queue
/// start: 0 (`clearstate_gfx11.h`, `gfx11_SECT_CONTEXT_def_2` entry for 0xa188, emitted by
/// `gfx_v11_0_cp_gfx_start`). RS64 firmware does not shadow it (`ac_shadowed_regs.c:652`), so
/// every submission writes it.
pub const RATE_CNTL: u32 = 0;

/// Inputs of the state block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateBlockConfig {
    pub attribute_ring: AttributeRing,
    /// TA_BC_BASE_ADDR / TA_CS_BC_BASE_ADDR (0 without a border-colour table).
    pub border_color_address: u64,
    /// PA_SC_TILE_STEERING_OVERRIDE, [`TILE_STEERING_OVERRIDE`] unless a KMD reports another.
    pub tile_steering_override: u32,
    /// PA_RATE_CNTL to write ([`RATE_CNTL`]), or `None` to keep the KMD's value.
    pub rate_cntl: Option<u32>,
    /// EVENT_WRITE PIXEL_PIPE_STAT_CONTROL for occlusion queries.
    pub occlusion_counters: bool,
    /// TA_CS_BC_BASE_ADDR(_HI), user-config 0x380/0x381.
    pub compute_border_color: bool,
}

impl StateBlockConfig {
    pub fn new(attribute_ring: AttributeRing) -> Self {
        Self {
            attribute_ring,
            border_color_address: 0,
            tile_steering_override: TILE_STEERING_OVERRIDE,
            rate_cntl: None,
            occlusion_counters: true,
            compute_border_color: true,
        }
    }

}

/// Builds the state block. The draw that follows must still program its own pipeline,
/// viewport, scissor and render-target registers.
pub fn build(config: &StateBlockConfig) -> Result<PacketStream> {
    let mut s = PacketStream::new();
    s.context_control_no_shadowing()?;
    emit_shader_defaults(&mut s)?;
    emit_context_defaults(&mut s, config)?;
    emit_ge_defaults(&mut s)?;
    if config.occlusion_counters {
        s.pixel_pipe_stat_control(RENDER_BACKENDS)?;
    }
    emit_late_defaults(&mut s)?;
    emit_compute_defaults(&mut s, config)?;
    emit_attribute_ring(&mut s, &config.attribute_ring)?;
    emit_user_data_pointers(&mut s)?;
    emit_cache_invalidate(&mut s)?;
    Ok(s)
}

/// Rows 1-17: SH defaults (CU masks with plain SET_SH_REG, as radv-on-WDDM does without a
/// kernel CU mask).
fn emit_shader_defaults(s: &mut PacketStream) -> Result<()> {
    s.set(SPI_SHADER_PGM_RSRC3_PS, RSRC3_PS)?;
    s.set(SPI_SHADER_REQ_CTRL_PS, 0x7)?;
    s.set_run(SPI_SHADER_USER_ACCUM_PS, &[0; 4])?;
    s.set_run(SPI_SHADER_USER_ACCUM_ESGS, &[0; 4])?;
    s.set(SPI_SHADER_PGM_HI_ES, 0)?;
    s.set(SPI_SHADER_PGM_RSRC3_HS, RSRC3_HS)?;
    s.set_run(SPI_SHADER_USER_ACCUM_LSHS, &[0; 4])?;
    s.set(SPI_SHADER_PGM_HI_LS, 0)
}

/// Rows 18-49: context registers CLEAR_STATE would have set.
fn emit_context_defaults(s: &mut PacketStream, config: &StateBlockConfig) -> Result<()> {
    s.set(PA_SC_SCREEN_SCISSOR_TL, 0)?;
    s.set_run(PA_SC_GENERIC_SCISSOR_TL, &[0x8000_0000, 0x4000_4000])?;
    s.set(PA_SC_TILE_STEERING_OVERRIDE, config.tile_steering_override)?;
    s.set(PA_SC_VRS_RATE_CACHE_CNTL, 0)?;
    s.set(CB_COVERAGE_OUT_CONTROL, 0)?;
    s.set(SPI_BARYC_SSAA_CNTL, 0)?;
    s.set_run(PA_CL_POINT_X_RAD, &[0; 4])?;
    s.set_run(PA_CL_NANINF_CNTL, &[0, 0])?;
    s.set_run(PA_SU_OVER_RASTERIZATION_CNTL, &[0, 0])?;
    s.set(VGT_ENHANCE, 0)?;
    s.set(VGT_PRIMITIVEID_RESET, 0)?;
    s.set(VGT_REUSE_OFF, 0)?;
    s.set(PA_SC_SHADER_CONTROL, 0)?;
    // Z/S writes stream through GL2, reads use NOA (DB does not cache in GL2 here).
    s.set(DB_RMI_L2_CACHE_CONTROL, 0x002a_0055)?;
    s.set_run(
        TA_BC_BASE_ADDR,
        &[
            (config.border_color_address >> 8) as u32,
            (config.border_color_address >> 40) as u32 & 0xff,
        ],
    )?;
    // COLOR/DCC writes stream, reads NOA. radv's value also carries gfx10-only CMASK/FMASK
    // bits; this is the clean gfx11 encoding.
    s.set(CB_RMI_GL2_CACHE_CONTROL, 0x00a0_0005)?;
    s.set(SX_PS_DOWNCONVERT_CONTROL, 0xff)?;
    s.set_run(VGT_HOS_MAX_TESS_LEVEL, &[0x4280_0000, 0])?;
    s.set(VGT_ESGS_RING_ITEMSIZE, 1)?;
    s.set(VGT_TESS_DISTRIBUTION, 0xd880_8080)?;
    // MAX_ALLOC_COUNT 16 (gfx11 has no -1), MAX_PRIM_PER_BATCH 1023.
    s.set(PA_SC_BINNER_CNTL_1, 0x03ff_0010)?;
    // gfx11.5: ping-pong bin order stays off.
    s.set(PA_SC_BINNER_CNTL_2, 0)?;
    s.set(PA_SC_NGG_MODE_CNTL, 0x10)?;
    match config.rate_cntl {
        Some(value) => s.set(PA_RATE_CNTL, value),
        None => Ok(()),
    }
}

/// Rows 50-58: GE and line-stipple user-config defaults.
fn emit_ge_defaults(s: &mut PacketStream) -> Result<()> {
    // GE_MULTI_PRIM_IB_RESET_EN.DISABLE_FOR_AUTO_INDEX.
    s.set_run(GE_MIN_VTX_INDX, &[0, 0, 0x4])?;
    s.set_run(GE_MAX_VTX_INDX, &[0xffff_ffff, 0])?;
    s.set(GE_STEREO_CNTL, 0)?;
    s.set(GE_USER_VGPR_EN, 0)?;
    // The RS64 firmware does not shadow PA_SC_LINE_STIPPLE_STATE; it is rewritten every time.
    s.set_run(PA_SU_LINE_STIPPLE_VALUE, &[0, 0])
}

/// Rows 60-103: viewport depth ranges, edge rule, HS/PS CU masks and fixed PA state.
fn emit_late_defaults(s: &mut PacketStream) -> Result<()> {
    let depth_ranges: Vec<u32> = (0..16).flat_map(|_| [0, 1.0f32.to_bits()]).collect();
    s.set_run(PA_SC_VPORT_ZMIN, &depth_ranges)?;
    s.set(PA_SC_EDGERULE, 0xaaaa_aaaa)?;
    s.set(SPI_SHADER_PGM_RSRC4_HS, RSRC4_HS)?;
    s.set(SPI_SHADER_PGM_RSRC4_PS, RSRC4_PS)?;
    s.set(PA_CL_NGG_CNTL, 0x78)?;
    s.set(PA_CL_VRS_CNTL, 0x200)?;
    s.set_run(PA_SU_POINT_SIZE, &[0x0008_0008, 0xffff_0000])?;
    // SMALL_PRIM_FILTER_ENABLE, SC_1XMSAA_COMPATIBLE_DISABLE (gfx11.5 field).
    s.set(PA_SU_SMALL_PRIM_FILTER_CNTL, 0x41)?;
    s.set(SPI_INTERP_CONTROL_0, 0x86b)?;
    // PIX_CENTER 0.5, round to even, 16.8 fixed point.
    s.set(PA_SU_VTX_CNTL, 0x2d)?;
    s.set(PA_CL_VTE_CNTL, 0x43f)?;
    s.set(PA_SU_LINE_STIPPLE_SCALE, 1.0f32.to_bits())
}

/// The block for a compute queue: the compute defaults and a cache invalidation, without the
/// context, user-config and graphics-pipe state a compute queue refuses.
pub fn build_compute(config: &StateBlockConfig) -> Result<PacketStream> {
    let mut s = PacketStream::new_compute();
    emit_compute_defaults(&mut s, config)?;
    s.event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
    s.acquire_mem_global(gcr::INVALIDATE_ALL)?;
    Ok(s)
}

/// Rows 105-124: compute defaults.
fn emit_compute_defaults(s: &mut PacketStream, config: &StateBlockConfig) -> Result<()> {
    if config.compute_border_color {
        s.set_run(
            TA_CS_BC_BASE_ADDR,
            &[
                (config.border_color_address >> 8) as u32,
                (config.border_color_address >> 40) as u32 & 0xff,
            ],
        )?;
    }
    emit_compute_registers(s)
}

/// The COMPUTE_* SH defaults of rows 105-124.
fn emit_compute_registers(s: &mut PacketStream) -> Result<()> {
    s.set(COMPUTE_PGM_HI, 0)?;
    s.set_run(COMPUTE_STATIC_THREAD_MGMT_SE0, &[ALL_CUS, ALL_CUS])?;
    s.set_run(COMPUTE_STATIC_THREAD_MGMT_SE2, &[0, 0])?;
    s.set_run(COMPUTE_USER_ACCUM, &[0; 4])?;
    s.set_run(COMPUTE_STATIC_THREAD_MGMT_SE4, &[0; 4])?;
    s.set(COMPUTE_DISPATCH_INTERLEAVE, 0x40)?;
    s.set(COMPUTE_DISPATCH_TUNNEL, 0)?;
    s.set_run(COMPUTE_START, &[0, 0, 0])
}

/// Rows 131-136: wait for idle, then point the SPI at the attribute ring.
fn emit_attribute_ring(s: &mut PacketStream, ring: &AttributeRing) -> Result<()> {
    s.release_mem_pws_bottom_of_pipe()?;
    s.acquire_mem_pws_wait()?;
    s.set_run(
        SPI_GS_THROTTLE_CNTL1,
        &[
            0x1235_5123,
            0x0001_544d,
            ring.base_register(),
            ring.size_register(),
        ],
    )
}

/// Rows 137-139 load radv's descriptor pointer into the merged-stage s[0:1] registers. The
/// native path has no such table yet; the pointers are cleared so no earlier value survives.
fn emit_user_data_pointers(s: &mut PacketStream) -> Result<()> {
    s.set_run(SPI_SHADER_PGM_LO_GS, &[0, 0])?;
    s.set_run(SPI_SHADER_PGM_LO_HS, &[0, 0])
}

/// Rows 148-151: drain, invalidate every cache the CPU uploads bypassed, start statistics.
fn emit_cache_invalidate(s: &mut PacketStream) -> Result<()> {
    s.event_write(event::PS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
    s.event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
    s.acquire_mem_global(gcr::INVALIDATE_ALL)?;
    s.event_write(event::PIPELINESTAT_START, 0)?;
    Ok(())
}
