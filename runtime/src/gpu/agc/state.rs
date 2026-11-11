//! Host state of a draw or dispatch from the guest registers and programs (decisions D12, D20,
//! D27), by what the registers mean: the guest's registers are read by name, and the host's are
//! written, most of them unchanged, some recomputed from the translated programs and the
//! surfaces' host copies. The result is pre-built PM4 kept by fingerprint: a draw's footprint
//! calls a block with its registers, or only the difference to the state before.

use super::arena::Bump;
use super::fields;
use super::link::LinkInputs;
use super::program::{PixelInterface, Program, ProgramRequest};
use super::runtime::{ScratchRing, TargetRuntime};
use crate::agc::cache::{self, Table};
use crate::agc::header;
use crate::agc::state::GuestRegisters;
use crate::amd::defs::{gfx10, gfx115, pkt3};
use crate::amd::surface::tiling::split_pipe_bank_xor;
use crate::backend::va;
use crate::gpu::regs::{self as host, HostRegister};
use crate::gpu::state_block::SHADER_ENGINES;
use crate::ir::RegisterBank::{self, Context, Shader, UserConfig};
use crate::Result;
use shader::{ShaderStage, VertexStage};
use std::collections::BTreeMap;
use std::sync::Arc;

const fn cx(offset: u32) -> u32 {
    (offset - 0x2_8000) / 4
}

const fn shader(offset: u32) -> u32 {
    (offset - 0xb000) / 4
}

const fn uc(offset: u32) -> u32 {
    (offset - 0x3_0000) / 4
}

/// VGT_SHADER_STAGES_EN fields of the guest.
const HS_EN: u32 = 1 << 2;
const PRIMGEN_EN: u32 = 1 << 13;
const HS_W32_EN: u32 = 1 << 21;
const GS_W32_EN: u32 = 1 << 22;
const STREAMOUT_ENABLES: u32 = 0x7f;

const NGG_USER_DATA_SGPR: u8 = 8;

/// The no-op pixel program of draws without one (`exp null off, off, off, off done vm; s_endpgm`),
/// radv's no-op fragment shader.
const NULL_PIXEL_WORDS: [u32; 3] = [0xf800_1890, 0, 0xbf81_0000];
const NULL_PIXEL_RSRC1: u32 = 0xc0 << 12;
const NULL_PIXEL_INPUT: u32 = 1 << 1;
const NULL_PIXEL_DB_EXPORTS: u32 = 0x7 | (1 << 6) | (1 << 7) | (1 << 8);

/// Host values for state the guest may not set, written before the translated guest values.
const DRAW_DEFAULTS: &[(HostRegister, u32)] = &[
    (host::GE_CNTL, 0x1f81_0080),
    (host::GE_PC_ALLOC, 0x7fe),
    (host::GE_MAX_OUTPUT_PER_SUBGROUP, 0x80),
    (host::GE_NGG_SUBGRP_CNTL, 1),
    (host::VGT_GS_INSTANCE_CNT, 4),
    (host::VGT_GS_OUT_PRIM_TYPE, 2),
    (host::SPI_SHADER_IDX_FORMAT, 1),
    (host::SPI_SHADER_POS_FORMAT, 4),
    (host::SPI_SHADER_PGM_RSRC3_GS, 0x003f_ffff),
    (host::SPI_SHADER_PGM_RSRC4_GS, 1),
    (host::PA_SC_BINNER_CNTL_0, 0x1004_0123),
    (host::SX_PS_DOWNCONVERT, 0),
    (host::SX_BLEND_OPT_EPSILON, 0),
    (host::SX_BLEND_OPT_CONTROL, 0),
    (host::PA_CL_GB_VERT_CLIP_ADJ, 0x3f80_0000),
    (host::PA_CL_GB_VERT_DISC_ADJ, 0x3f80_0000),
    (host::PA_CL_GB_HORZ_CLIP_ADJ, 0x3f80_0000),
    (host::PA_CL_GB_HORZ_DISC_ADJ, 0x3f80_0000),
    (host::PA_SC_MODE_CNTL_1, 0),
    (host::DB_Z_INFO, 0xc),
    (host::DB_STENCIL_INFO, 1 << 29),
    (host::DB_COUNT_CONTROL, 0),
    (host::PA_SU_PRIM_FILTER_CNTL, 0xc000_0000),
    (host::PA_SU_LINE_CNTL, 8),
    (host::PA_SC_LINE_STIPPLE, 0x4000_0000),
    (host::VGT_DRAW_PAYLOAD_CNTL, 0),
    (host::VGT_PRIMITIVEID_EN, 0),
    (host::PA_SC_CENTROID_PRIORITY_0, 0),
    (host::PA_SC_CENTROID_PRIORITY_1, 0),
    (host::PA_SC_LINE_CNTL, 0),
    (host::PA_SC_VRS_OVERRIDE_CNTL, 0),
    (host::SPI_BARYC_CNTL, 0),
    (host::CB_COLOR_FDCC_CONTROL, 0),
    (host::CB_COLOR_DCC_BASE, 0),
    (host::CB_COLOR_DCC_BASE_EXT, 0),
];

/// NGG draw fields no driver leaves at 0, with their mask and host value: the geometry engine
/// waits forever on each of the first three at 0, and MAX_PRIMGRP_IN_WAVE is 2 in radv.
const GEOMETRY_ENGINE_LIMITS: [(HostRegister, u32, u32); 4] = [
    (host::GE_MAX_OUTPUT_PER_SUBGROUP, 0x3ff, 0x80),
    (host::GE_NGG_SUBGRP_CNTL, 0x1ff, 1),
    (host::SPI_SHADER_POS_FORMAT, 0xf, 4),
    (host::VGT_SHADER_STAGES_EN, 0xf << 15, 2 << 15),
];

macro_rules! pass_through {
    ($(($name:ident, $count:expr, $stride:expr)),* $(,)?) => {
        /// Guest context registers the host keeps at the same offset with the same meaning:
        /// first dword, elements, dwords between elements.
        const PASS_CONTEXT: &[(u32, u32, u32)] = &[$((cx(gfx10::$name::OFFSET), $count, $stride)),*];
        const _: () = { $(assert!(gfx10::$name::OFFSET == gfx115::$name::OFFSET);)* };
    };
}

pass_through![
    (DB_DEPTH_VIEW, 1, 1),
    (DB_RENDER_OVERRIDE2, 1, 1),
    (DB_DEPTH_SIZE_XY, 1, 1),
    (DB_DEPTH_BOUNDS_MIN, 1, 1),
    (DB_DEPTH_BOUNDS_MAX, 1, 1),
    (DB_STENCIL_CLEAR, 1, 1),
    (DB_DEPTH_CLEAR, 1, 1),
    (PA_SC_SCREEN_SCISSOR_TL, 1, 1),
    (PA_SC_SCREEN_SCISSOR_BR, 1, 1),
    (TA_BC_BASE_ADDR, 1, 1),
    (TA_BC_BASE_ADDR_HI, 1, 1),
    (PA_SC_WINDOW_OFFSET, 1, 1),
    (PA_SC_WINDOW_SCISSOR_TL, 1, 1),
    (PA_SC_WINDOW_SCISSOR_BR, 1, 1),
    (PA_SC_CLIPRECT_RULE, 1, 1),
    (PA_SC_EDGERULE, 1, 1),
    (CB_TARGET_MASK, 1, 1),
    (CB_SHADER_MASK, 1, 1),
    (PA_SC_GENERIC_SCISSOR_TL, 1, 1),
    (PA_SC_GENERIC_SCISSOR_BR, 1, 1),
    (PA_SC_VPORT_SCISSOR_0_TL, 16, 2),
    (PA_SC_VPORT_SCISSOR_0_BR, 16, 2),
    (PA_SC_VPORT_ZMIN_0, 16, 2),
    (PA_SC_VPORT_ZMAX_0, 16, 2),
    (VGT_MULTI_PRIM_IB_RESET_INDX, 1, 1),
    (CB_BLEND_RED, 1, 1),
    (CB_BLEND_GREEN, 1, 1),
    (CB_BLEND_BLUE, 1, 1),
    (CB_BLEND_ALPHA, 1, 1),
    (DB_STENCIL_CONTROL, 1, 1),
    (DB_STENCILREFMASK, 1, 1),
    (DB_STENCILREFMASK_BF, 1, 1),
    (PA_CL_VPORT_XSCALE, 16, 6),
    (PA_CL_VPORT_XOFFSET, 16, 6),
    (PA_CL_VPORT_YSCALE, 16, 6),
    (PA_CL_VPORT_YOFFSET, 16, 6),
    (PA_CL_VPORT_ZSCALE, 16, 6),
    (PA_CL_VPORT_ZOFFSET, 16, 6),
    (SPI_PS_INPUT_ENA, 1, 1),
    (SPI_PS_INPUT_ADDR, 1, 1),
    (SPI_INTERP_CONTROL_0, 1, 1),
    (SPI_BARYC_CNTL, 1, 1),
    (SPI_SHADER_IDX_FORMAT, 1, 1),
    (SPI_SHADER_POS_FORMAT, 1, 1),
    (SPI_SHADER_Z_FORMAT, 1, 1),
    (SPI_SHADER_COL_FORMAT, 1, 1),
    (PA_CL_POINT_X_RAD, 1, 1),
    (PA_CL_POINT_Y_RAD, 1, 1),
    (PA_CL_POINT_SIZE, 1, 1),
    (PA_CL_POINT_CULL_RAD, 1, 1),
    (GE_MAX_OUTPUT_PER_SUBGROUP, 1, 1),
    (DB_DEPTH_CONTROL, 1, 1),
    (DB_EQAA, 1, 1),
    (PA_CL_CLIP_CNTL, 1, 1),
    (PA_SU_SC_MODE_CNTL, 1, 1),
    (PA_CL_VTE_CNTL, 1, 1),
    (PA_CL_NANINF_CNTL, 1, 1),
    (PA_SU_LINE_STIPPLE_SCALE, 1, 1),
    (PA_SU_PRIM_FILTER_CNTL, 1, 1),
    (PA_SU_OVER_RASTERIZATION_CNTL, 1, 1),
    (PA_SU_POINT_SIZE, 1, 1),
    (PA_SU_POINT_MINMAX, 1, 1),
    (PA_SU_LINE_CNTL, 1, 1),
    (PA_SC_LINE_STIPPLE, 1, 1),
    (VGT_HOS_MAX_TESS_LEVEL, 1, 1),
    (VGT_HOS_MIN_TESS_LEVEL, 1, 1),
    (PA_SC_MODE_CNTL_1, 1, 1),
    (VGT_PRIMITIVEID_EN, 1, 1),
    (VGT_PRIMITIVEID_RESET, 1, 1),
    (VGT_ESGS_RING_ITEMSIZE, 1, 1),
    (VGT_GS_MAX_VERT_OUT, 1, 1),
    (GE_NGG_SUBGRP_CNTL, 1, 1),
    (VGT_LS_HS_CONFIG, 1, 1),
    (DB_ALPHA_TO_MASK, 1, 1),
    (PA_SU_POLY_OFFSET_DB_FMT_CNTL, 1, 1),
    (PA_SU_POLY_OFFSET_CLAMP, 1, 1),
    (PA_SU_POLY_OFFSET_FRONT_SCALE, 1, 1),
    (PA_SU_POLY_OFFSET_FRONT_OFFSET, 1, 1),
    (PA_SU_POLY_OFFSET_BACK_SCALE, 1, 1),
    (PA_SU_POLY_OFFSET_BACK_OFFSET, 1, 1),
    (VGT_GS_INSTANCE_CNT, 1, 1),
    (PA_SC_CENTROID_PRIORITY_0, 1, 1),
    (PA_SC_CENTROID_PRIORITY_1, 1, 1),
    (PA_SC_LINE_CNTL, 1, 1),
    (PA_SC_AA_CONFIG, 1, 1),
    (PA_SU_VTX_CNTL, 1, 1),
    (PA_CL_GB_VERT_CLIP_ADJ, 1, 1),
    (PA_CL_GB_VERT_DISC_ADJ, 1, 1),
    (PA_CL_GB_HORZ_CLIP_ADJ, 1, 1),
    (PA_CL_GB_HORZ_DISC_ADJ, 1, 1),
    (PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0, 16, 1),
    (PA_SC_AA_MASK_X0Y0_X1Y0, 1, 1),
    (PA_SC_AA_MASK_X0Y1_X1Y1, 1, 1),
    (PA_SC_CONSERVATIVE_RASTERIZATION_CNTL, 1, 1),
    (CB_COLOR0_VIEW, 8, 15),
    (CB_COLOR0_ATTRIB2, 8, 1),
];

/// User-config registers the host keeps unchanged.
const PASS_USER_CONFIG: &[u32] = &[
    uc(gfx10::GE_STEREO_CNTL::OFFSET),
    uc(gfx10::GE_INDX_OFFSET::OFFSET),
    uc(gfx10::VGT_TF_MEMORY_BASE::OFFSET),
    uc(gfx10::VGT_TF_MEMORY_BASE_HI::OFFSET),
    uc(gfx10::GE_USER_VGPR_EN::OFFSET),
];

/// Registers a pure field translation takes to the host: guest dword (context), elements, dwords
/// between elements, host register, translation.
type FieldRule = (u32, u32, u32, HostRegister, fn(u32) -> fields::Translated);

const FIELD_RULES: &[FieldRule] = &[
    (cx(gfx10::DB_RENDER_CONTROL::OFFSET), 1, 1, host::DB_RENDER_CONTROL, fields::db_render_control),
    (cx(gfx10::DB_COUNT_CONTROL::OFFSET), 1, 1, host::DB_COUNT_CONTROL, fields::db_count_control),
    (cx(gfx10::DB_RENDER_OVERRIDE::OFFSET), 1, 1, host::DB_RENDER_OVERRIDE, fields::db_render_override),
    (cx(gfx10::DB_Z_INFO::OFFSET), 1, 1, host::DB_Z_INFO, fields::db_z_info),
    (cx(gfx10::DB_STENCIL_INFO::OFFSET), 1, 1, host::DB_STENCIL_INFO, fields::db_stencil_info),
    (cx(gfx10::SPI_PS_INPUT_CNTL_0::OFFSET), 32, 1, host::SPI_PS_INPUT_CNTL, fields::spi_ps_input_cntl),
    (cx(gfx10::SPI_PS_IN_CONTROL::OFFSET), 1, 1, host::SPI_PS_IN_CONTROL, fields::spi_ps_in_control),
    (cx(gfx10::CB_BLEND0_CONTROL::OFFSET), 8, 1, host::CB_BLEND_CONTROL, fields::cb_blend_control),
    (cx(gfx10::CB_COLOR_CONTROL::OFFSET), 1, 1, host::CB_COLOR_CONTROL, fields::cb_color_control),
    (cx(gfx10::DB_SHADER_CONTROL::OFFSET), 1, 1, host::DB_SHADER_CONTROL, fields::db_shader_control),
    (cx(gfx10::PA_SC_SHADER_CONTROL::OFFSET), 1, 1, host::PA_SC_SHADER_CONTROL, fields::pa_sc_shader_control),
    (cx(gfx10::PA_CL_VS_OUT_CNTL::OFFSET), 1, 1, host::PA_CL_VS_OUT_CNTL, fields::pa_cl_vs_out_cntl),
    (cx(gfx10::PA_SU_LINE_STIPPLE_CNTL::OFFSET), 1, 1, host::PA_SU_LINE_STIPPLE_CNTL, fields::pa_su_line_stipple_cntl),
    (cx(gfx10::PA_SC_MODE_CNTL_0::OFFSET), 1, 1, host::PA_SC_MODE_CNTL_0, fields::pa_sc_mode_cntl_0),
    (cx(gfx10::VGT_DRAW_PAYLOAD_CNTL::OFFSET), 1, 1, host::VGT_DRAW_PAYLOAD_CNTL, fields::vgt_draw_payload_cntl),
    (cx(gfx10::VGT_SHADER_STAGES_EN::OFFSET), 1, 1, host::VGT_SHADER_STAGES_EN, fields::vgt_shader_stages_en),
    (cx(gfx10::VGT_TF_PARAM::OFFSET), 1, 1, host::VGT_TF_PARAM, fields::vgt_tf_param),
    (cx(gfx10::VGT_GS_OUT_PRIM_TYPE::OFFSET), 1, 1, host::VGT_GS_OUT_PRIM_TYPE, fields::vgt_gs_out_prim_type),
    (cx(gfx10::CB_COLOR0_INFO::OFFSET), 8, 15, host::CB_COLOR_INFO, fields::cb_color_info),
    (cx(gfx10::CB_COLOR0_ATTRIB::OFFSET), 8, 15, host::CB_COLOR_ATTRIB, fields::cb_color_attrib),
    (cx(gfx10::CB_COLOR0_DCC_CONTROL::OFFSET), 8, 15, host::CB_COLOR_FDCC_CONTROL, fields::cb_color_fdcc_control),
    (cx(gfx10::CB_COLOR0_ATTRIB3::OFFSET), 8, 1, host::CB_COLOR_ATTRIB3, fields::cb_color_attrib3),
];

/// What a render target clears to, for when the title clears its metadata.
pub struct ClearTarget {
    pub surface: u64,
    /// The surface as the draw bound it: the memory may hold another one later.
    pub desc: crate::amd::surface::tiling::SurfaceDesc,
    /// The element the host copy is filled with.
    pub texel: Vec<u8>,
}

/// The fast clears the targets of a draw can have: the address of each target's metadata (HTILE,
/// DCC keys), the target, and the element a clear leaves in it, from the draw's clear registers.
pub fn clear_values(targets: &super::surface::RenderTargets, registers: &[(RegisterBank, u32, u32)]) -> Vec<(u64, ClearTarget)> {
    let mut values = Vec::new();
    if let Some(depth) = &targets.depth {
        if let Some(htile) = depth.htile {
            let dword = (gfx10::DB_DEPTH_CLEAR::OFFSET - 0x2_8000) / 4;
            let clear = registers.iter().find(|&&(bank, at, _)| bank == Context && at == dword).map_or(0, |&(_, _, value)| value);
            let texel = if depth.layout.desc().bytes_per_element == 2 { ((f32::from_bits(clear) * 65535.0).round() as u16).to_le_bytes().to_vec() } else { clear.to_le_bytes().to_vec() };
            values.push((htile, ClearTarget { surface: depth.write_base, desc: *depth.layout.desc(), texel }));
        }
    }
    for color in &targets.colors {
        let Some(dcc) = &color.dcc else { continue };
        let bytes = color.layout.desc().bytes_per_element as usize;
        if bytes > 8 {
            continue;
        }
        let mut value = [0u8; 8];
        value[..4].copy_from_slice(&dcc.clear_words[0].to_le_bytes());
        value[4..].copy_from_slice(&dcc.clear_words[1].to_le_bytes());
        values.push((dcc.base, ClearTarget { surface: color.base, desc: *color.layout.desc(), texel: value[..bytes].to_vec() }));
    }
    values
}

/// Remembers which addresses hold the metadata of `targets`, and what they last cleared to: a
/// title clears a target by writing its HTILE or DCC keys from a compute shader, which leaves the
/// surface alone, and the host copies have no metadata, so the clear is carried out on the copy
/// (D29). The address may hold something else by the time it is written again (titles reuse the
/// memory of transient targets): the clear is carried out where a draw binds a target with that
/// metadata, or a texture with it is read while its last target is still what is tracked (D41).
fn note_clear_targets(runtime: &TargetRuntime, targets: &super::surface::RenderTargets, registers: &[(RegisterBank, u32, u32)]) {
    for (metadata, target) in clear_values(targets, registers) {
        runtime.clear_targets.insert(metadata, Arc::new(target));
    }
}

/// A range of PM4 in runtime memory: an IB2 target.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IbRange {
    pub address: u64,
    pub dwords: u32,
}

impl IbRange {
    /// `words` padded to 8 dwords, as radv pads every CP IB, in `memory`.
    fn place(memory: &Bump, words: &[u32]) -> Result<Self> {
        let mut padded = words.to_vec();
        padded.resize((words.len() + 7) & !7, 0);
        if padded.len() != words.len() {
            let gap = padded.len() - words.len();
            let at = words.len();
            super::nop(&mut padded[at..at + gap]);
        }
        let address = memory.put(&padded, 32)?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        Ok(Self { address, dwords: padded.len() as u32 })
    }
}

/// Host registers of a state: each once, sorted by bank and dword.
#[derive(Clone, Debug, Default)]
pub struct HostRegisters {
    values: BTreeMap<(RegisterBank, u32), u32>,
}

impl HostRegisters {
    fn put(&mut self, register: HostRegister, index: u32, value: u32) {
        self.values.insert((register.bank, register.at(index)), value);
    }

    fn set(&mut self, register: HostRegister, value: u32) {
        self.put(register, 0, value);
    }

    fn get(&self, register: HostRegister, index: u32) -> Option<u32> {
        self.values.get(&(register.bank, register.at(index))).copied()
    }

    fn sorted(&self) -> Box<[(RegisterBank, u32, u32)]> {
        self.values.iter().map(|(&(bank, dword), &value)| (bank, dword, value)).collect()
    }

    /// SET_*_REG packets of the registers, in runs, into `out`.
    fn emit(entries: &[(RegisterBank, u32, u32)], out: &mut Vec<u32>) -> Result<()> {
        let mut index = 0;
        while index < entries.len() {
            let (bank, first, _) = entries[index];
            let mut end = index + 1;
            while end < entries.len() && entries[end].0 == bank && entries[end].1 == first + (end - index) as u32 {
                end += 1;
            }
            out.push(header(bank.set_opcode(), (end - index) as u32 + 2, 0)?);
            out.push(first);
            out.extend(entries[index..end].iter().map(|&(_, _, value)| value));
            index = end;
        }
        Ok(())
    }
}

/// A translated program's place in a draw or dispatch: the stage's user data starts at guest SH
/// dword `user_base`, which is input SGPR `user_sgpr`.
#[derive(Clone, Debug)]
pub struct Stage {
    pub program: Arc<Program>,
    pub user_base: u32,
    pub user_sgpr: u8,
    pub user_count: u32,
}

/// A cached draw or dispatch state.
#[derive(Debug)]
pub struct StateRecord {
    pub key: u64,
    /// All of a draw's host registers in one IB2 target; `None` for a dispatch.
    pub block: Option<IbRange>,
    pub registers: Box<[(RegisterBank, u32, u32)]>,
    /// A dispatch's state as packets: copied whole into its footprint.
    pub compute_packets: Option<[u32; super::draw::dwords::COMPUTE_STATE as usize]>,
    pub primitive_type: u32,
    pub stages: Vec<Stage>,
    /// Guest SH dwords whose values reach the programs' descriptors.
    pub watched: Box<[u32]>,
    pub targets: Option<Arc<super::surface::RenderTargets>>,
}

/// A cached plan, or why the work cannot run.
pub type Cached<T> = std::result::Result<T, String>;

/// The user-data registers a state's programs read as code addresses, per stage: the guest SH
/// dword of the stage's first user-data register and a mask of its registers.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Linkage(pub Vec<(u32, u32)>);

impl Linkage {
    /// The state key of `fingerprint` with this linkage's user-data values in `registers`.
    fn key(&self, fingerprint: u64, registers: &GuestRegisters) -> u64 {
        if self.0.iter().all(|&(_, mask)| mask == 0) {
            return fingerprint;
        }
        let values = self.0.iter().flat_map(|&(first, mask)| {
            (0..32).filter(move |bit| mask >> bit & 1 != 0).map(move |bit| {
                let dword = first + bit;
                u64::from(dword) << 32 | u64::from(registers.get_or_zero(Shader, dword))
            })
        });
        cache::key(std::iter::once(fingerprint).chain(values))
    }
}

/// DISPATCH_INITIATOR CS_W32_EN.
const CS_W32_EN: u32 = 1 << 15;

/// The caches every recording shares (D20).
pub struct StateCache {
    draw_linkage: Table<Linkage>,
    draws: Table<Cached<Arc<StateRecord>>>,
    dispatch_linkage: Table<Linkage>,
    dispatches: Table<Cached<Arc<StateRecord>>>,
    transitions: Table<IbRange>,
    sequences: Table<IbRange>,
}

impl Default for StateCache {
    fn default() -> Self {
        Self {
            draw_linkage: Table::with_capacity(1 << 14),
            draws: Table::with_capacity(1 << 16),
            dispatch_linkage: Table::with_capacity(1 << 12),
            dispatches: Table::with_capacity(1 << 14),
            transitions: Table::with_capacity(1 << 18),
            sequences: Table::with_capacity(1 << 10),
        }
    }
}

impl StateCache {
    /// The state of a draw with `registers`, planned now when no cached state has it.
    pub fn draw(&self, runtime: &TargetRuntime, registers: &GuestRegisters) -> Cached<Arc<StateRecord>> {
        let mut fingerprint = registers.graphics_fingerprint();
        if registers.get_or_zero(Context, cx(gfx10::VGT_SHADER_STAGES_EN::OFFSET)) & HS_EN != 0 {
            // A tessellating draw also depends on the rings the title gave the driver.
            let rings = crate::agc::driver::Driver::get().tessellation.rings();
            let (address, bytes) = rings.factor.unwrap_or_default();
            fingerprint = cache::key([fingerprint, address, u64::from(bytes), rings.offchip.map_or(u64::MAX, u64::from)]);
        }
        if let Some(linkage) = self.draw_linkage.get(fingerprint) {
            let key = linkage.key(fingerprint, registers);
            if let Some(state) = self.draws.get(key) {
                return (*state).clone();
            }
        }
        let (linkage, planned) = match GraphicsPlan::new(runtime, registers).plan() {
            Ok((plan, linkage)) => (linkage, Ok(plan)),
            Err(reason) => (Linkage::default(), Err(reason)),
        };
        let linkage = self.draw_linkage.insert(fingerprint, Arc::new(linkage));
        let key = linkage.key(fingerprint, registers);
        let state = planned.and_then(|plan| plan.record(runtime, key));
        (*self.draws.insert(key, Arc::new(state))).clone()
    }

    /// The state of a dispatch with `registers` and DISPATCH_INITIATOR `initiator` on a compute
    /// or graphics queue.
    pub fn dispatch(&self, runtime: &TargetRuntime, registers: &GuestRegisters, initiator: u32, compute: bool) -> Cached<Arc<StateRecord>> {
        let fingerprint = cache::key([registers.compute_fingerprint(), u64::from(initiator & CS_W32_EN), u64::from(compute)]);
        if let Some(linkage) = self.dispatch_linkage.get(fingerprint) {
            let key = linkage.key(fingerprint, registers);
            if let Some(state) = self.dispatches.get(key) {
                return (*state).clone();
            }
        }
        let (linkage, planned) = match ComputePlan::new(runtime, registers, initiator, compute).plan() {
            Ok((plan, linkage)) => (linkage, Ok(plan)),
            Err(reason) => (Linkage::default(), Err(reason)),
        };
        let linkage = self.dispatch_linkage.insert(fingerprint, Arc::new(linkage));
        let key = linkage.key(fingerprint, registers);
        let state = planned.and_then(|plan| plan.record(key));
        (*self.dispatches.insert(key, Arc::new(state))).clone()
    }

    /// The registers that change between `from` and `to`, as an IB2 target; zero dwords when none
    /// do.
    pub fn transition(&self, runtime: &TargetRuntime, from: &StateRecord, to: &StateRecord) -> Result<IbRange> {
        let key = cache::key([from.key, to.key, 0x7472_616e]);
        if let Some(range) = self.transitions.get(key) {
            return Ok(*range);
        }
        let mut changed = Vec::new();
        let mut old = from.registers.iter().peekable();
        for &(bank, dword, value) in to.registers.iter() {
            while old.peek().is_some_and(|&&(b, d, _)| (b, d) < (bank, dword)) {
                old.next();
            }
            let same = old.peek().is_some_and(|&&(b, d, v)| (b, d) == (bank, dword) && v == value);
            if !same {
                changed.push((bank, dword, value));
            }
        }
        let mut words = Vec::with_capacity(changed.len() * 3);
        HostRegisters::emit(&changed, &mut words)?;
        let range = if words.is_empty() { IbRange::default() } else { IbRange::place(&runtime.commands, &words)? };
        Ok(*self.transitions.insert(key, Arc::new(range)))
    }

    /// PM4 sequence `key` (built by `build` on a miss), padded to 8 dwords, as an IB2 target.
    pub fn sequence(&self, runtime: &TargetRuntime, key: u64, build: impl FnOnce() -> Result<Vec<u32>>) -> Result<IbRange> {
        if let Some(range) = self.sequences.get(key) {
            return Ok(*range);
        }
        let range = IbRange::place(&runtime.commands, &build()?)?;
        Ok(*self.sequences.insert(key, Arc::new(range)))
    }
}

/// A draw's host state before it has a key.
struct GraphicsState {
    registers: HostRegisters,
    primitive_type: u32,
    stages: Vec<Stage>,
    targets: Option<Arc<super::surface::RenderTargets>>,
}

impl GraphicsState {
    fn record(self, runtime: &TargetRuntime, key: u64) -> Cached<Arc<StateRecord>> {
        let sorted = self.registers.sorted();
        let mut words = Vec::with_capacity(sorted.len() * 3);
        HostRegisters::emit(&sorted, &mut words).map_err(|error| error.message)?;
        let block = IbRange::place(&runtime.commands, &words).map_err(|error| error.message)?;
        if let Some(targets) = &self.targets {
            note_clear_targets(runtime, targets, &sorted);
        }
        let watched = watched_registers(&self.stages);
        Ok(Arc::new(StateRecord {
            key,
            block: Some(block),
            registers: sorted,
            compute_packets: None,
            primitive_type: self.primitive_type,
            stages: self.stages,
            watched,
            targets: self.targets,
        }))
    }
}

/// The guest SH dwords that hold a stage's descriptors or the pointers leading to them.
fn watched_registers(stages: &[Stage]) -> Box<[u32]> {
    use shader::ir::DescriptorSource as D;
    let mut dwords = std::collections::BTreeSet::new();
    for stage in stages {
        let register = |sgpr: u8| sgpr.checked_sub(stage.user_sgpr).map(|element| stage.user_base + u32::from(element)).filter(|&dword| dword < stage.user_base + stage.user_count);
        let mut watch = |source: &D, count: u8| match source {
            D::UserSgprs { first } => dwords.extend((0..count).filter_map(|element| register(first + element))),
            D::Table(table) => dwords.extend([table.root, table.root + 1].into_iter().filter_map(register)),
            D::TableChoices(tables) => {
                dwords.extend(tables.iter().flat_map(|table| [table.root, table.root + 1]).filter_map(register))
            }
            D::Dynamic => {}
        };
        for image in &stage.program.images {
            watch(&image.source, image.dword_count);
        }
        for buffer in &stage.program.buffers {
            watch(&buffer.source, if buffer.pointer { 2 } else { 4 });
        }
    }
    dwords.into_iter().collect()
}

struct GraphicsPlan<'a> {
    runtime: &'a TargetRuntime,
    guest: &'a GuestRegisters,
    out: HostRegisters,
}

fn address_of(low: u32, high: u32) -> u64 {
    (u64::from(low) << 8) | (u64::from(high & 0xff) << 40)
}

impl<'a> GraphicsPlan<'a> {
    fn new(runtime: &'a TargetRuntime, guest: &'a GuestRegisters) -> Self {
        Self { runtime, guest, out: HostRegisters::default() }
    }

    fn context(&self, dword: u32) -> Option<u32> {
        self.guest.get(Context, dword)
    }

    fn context_or_zero(&self, dword: u32) -> u32 {
        self.guest.get_or_zero(Context, dword)
    }

    fn sh(&self, dword: u32) -> u32 {
        self.guest.get_or_zero(Shader, dword)
    }

    fn program_address(&self, low: u32) -> u64 {
        address_of(self.sh(low), self.sh(low + 1))
    }

    fn defaults(&mut self) {
        for &(register, value) in DRAW_DEFAULTS {
            self.out.set(register, value);
        }
    }

    /// Registers that keep their offset and meaning.
    fn pass_through(&mut self) {
        for &(first, count, stride) in PASS_CONTEXT {
            for index in 0..count {
                let dword = first + index * stride;
                if let Some(value) = self.guest.get(Context, dword) {
                    self.out.values.insert((Context, dword), value);
                }
            }
        }
        for &dword in PASS_USER_CONFIG {
            if let Some(value) = self.guest.get(UserConfig, dword) {
                self.out.values.insert((UserConfig, dword), value);
            }
        }
    }

    /// Registers whose fields move or change meaning.
    fn field_translations(&mut self) {
        for &(first, count, stride, register, translate) in FIELD_RULES {
            for index in 0..count {
                let Some(value) = self.context(first + index * stride) else { continue };
                match translate(value) {
                    Ok(translated) => self.out.put(register, index, translated),
                    Err(reason) => report_once(&format!("guest register {register:?}[{index}]: {reason}"), first),
                }
            }
        }
        if let Some(value) = self.guest.get(UserConfig, uc(gfx10::GE_MULTI_PRIM_IB_RESET_EN::OFFSET)) {
            self.out.set(host::GE_MULTI_PRIM_IB_RESET_EN, (value & 0b11) | 0b100);
        }
        if let Some(value) = self.guest.get(UserConfig, uc(gfx10::VGT_HS_OFFCHIP_PARAM::OFFSET)) {
            self.out.set(host::VGT_HS_OFFCHIP_PARAM, fields::vgt_hs_offchip_param(value, SHADER_ENGINES).unwrap_or(0));
        }
        if let Some(value) = self.guest.get(UserConfig, uc(gfx10::VGT_TF_RING_SIZE::OFFSET)) {
            self.out.set(host::VGT_TF_RING_SIZE, fields::vgt_tf_ring_size(value, SHADER_ENGINES).unwrap_or(1));
        }
        // The merged stages' s[0:1] user-data pointers moved to the old PGM_LO/HI registers.
        for (low, high, register_low, register_high) in [
            (gfx10::SPI_SHADER_USER_DATA_ADDR_LO_GS::OFFSET, gfx10::SPI_SHADER_USER_DATA_ADDR_HI_GS::OFFSET, host::SPI_SHADER_PGM_LO_GS, host::SPI_SHADER_PGM_HI_GS),
            (gfx10::SPI_SHADER_USER_DATA_ADDR_LO_HS::OFFSET, gfx10::SPI_SHADER_USER_DATA_ADDR_HI_HS::OFFSET, host::SPI_SHADER_PGM_LO_HS, host::SPI_SHADER_PGM_HI_HS),
        ] {
            if let (Some(lo), Some(hi)) = (self.guest.get(Shader, shader(low)), self.guest.get(Shader, shader(high))) {
                self.out.set(register_low, lo);
                self.out.set(register_high, hi);
            }
        }
    }

    /// PA_SU_HARDWARE_SCREEN_OFFSET aligned to 32 pixels, the scratch size per shader engine and
    /// the removed VGT_GS_ONCHIP_CNTL folded into GE_CNTL.
    fn special_registers(&mut self) -> std::result::Result<(), String> {
        if let Some(value) = self.context(cx(gfx10::PA_SU_HARDWARE_SCREEN_OFFSET::OFFSET)) {
            self.out.set(host::PA_SU_HARDWARE_SCREEN_OFFSET, fields::hardware_screen_offset(value).0);
        }
        if let Some(value) = self.context(cx(gfx10::SPI_TMPRING_SIZE::OFFSET)) {
            self.out.set(host::SPI_TMPRING_SIZE, fields::tmpring_size(value, SHADER_ENGINES)?);
        }
        if let Some(value) = self.guest.get(UserConfig, uc(gfx10::GE_CNTL::OFFSET)) {
            let onchip = self.context_or_zero(cx(gfx10::VGT_GS_ONCHIP_CNTL::OFFSET));
            self.out.set(host::GE_CNTL, fields::ge_cntl(self.runtime.isa, value, onchip)?);
        }
        Ok(())
    }

    /// Renders tiled colour targets and depth through their host-layout copies at guest VA + 2^40.
    fn relocate_surfaces(&mut self) {
        for slot in 0..8u32 {
            let block = slot * 15;
            let info = self.context_or_zero(cx(gfx10::CB_COLOR0_INFO::OFFSET) + block);
            if (info >> 2) & 0x1f == 0 {
                continue;
            }
            let swizzle_mode = (self.context_or_zero(cx(gfx10::CB_COLOR0_ATTRIB3::OFFSET) + slot) >> 14) & 0x1f;
            let guest_base = address_of(
                self.context_or_zero(cx(gfx10::CB_COLOR0_BASE::OFFSET) + block),
                self.context_or_zero(cx(gfx10::CB_COLOR0_BASE_EXT::OFFSET) + slot),
            );
            let host_base = if swizzle_mode == 0 {
                guest_base
            } else {
                let (base, _xor) = split_pipe_bank_xor(swizzle_mode, guest_base);
                if base == 0 { 0 } else { base + va::HOST_COPY_OFFSET }
            };
            self.out.put(host::CB_COLOR_BASE, slot, (host_base >> 8) as u32);
            self.out.put(host::CB_COLOR_BASE_EXT, slot, (host_base >> 40) as u32 & 0xff);
        }
        let z_info = self.context_or_zero(cx(gfx10::DB_Z_INFO::OFFSET));
        let stencil_info = self.context_or_zero(cx(gfx10::DB_STENCIL_INFO::OFFSET));
        let planes = [
            (z_info & 3 != 0, z_info, gfx10::DB_Z_READ_BASE::OFFSET, gfx10::DB_Z_READ_BASE_HI::OFFSET, host::DB_Z_READ_BASE, host::DB_Z_READ_BASE_HI),
            (z_info & 3 != 0, z_info, gfx10::DB_Z_WRITE_BASE::OFFSET, gfx10::DB_Z_WRITE_BASE_HI::OFFSET, host::DB_Z_WRITE_BASE, host::DB_Z_WRITE_BASE_HI),
            (stencil_info & 1 != 0, stencil_info, gfx10::DB_STENCIL_READ_BASE::OFFSET, gfx10::DB_STENCIL_READ_BASE_HI::OFFSET, host::DB_STENCIL_READ_BASE, host::DB_STENCIL_READ_BASE_HI),
            (stencil_info & 1 != 0, stencil_info, gfx10::DB_STENCIL_WRITE_BASE::OFFSET, gfx10::DB_STENCIL_WRITE_BASE_HI::OFFSET, host::DB_STENCIL_WRITE_BASE, host::DB_STENCIL_WRITE_BASE_HI),
        ];
        for (enabled, info, low, high, register_low, register_high) in planes {
            if !enabled {
                continue;
            }
            let guest_base = address_of(self.context_or_zero(cx(low)), self.context_or_zero(cx(high)));
            let (base, _xor) = split_pipe_bank_xor((info >> 4) & 0x1f, guest_base);
            let host_base = if base == 0 { 0 } else { base + va::HOST_COPY_OFFSET };
            self.out.set(register_low, (host_base >> 8) as u32);
            self.out.set(register_high, (host_base >> 40) as u32 & 0xff);
        }
    }

    /// A depth or stencil plane the draw neither tests, writes nor clears stays invalid on the
    /// host, as the guest registers say: its host copy is only made for draws that use it.
    fn disable_unused_depth_stencil(&mut self) {
        let control = self.context_or_zero(cx(gfx10::DB_DEPTH_CONTROL::OFFSET));
        let render = self.context_or_zero(cx(gfx10::DB_RENDER_CONTROL::OFFSET));
        for (used, register, format) in [(control & 0xe != 0 || render & 1 != 0, host::DB_Z_INFO, 3), (control & 1 != 0, host::DB_STENCIL_INFO, 1)] {
            if let Some(value) = self.out.get(register, 0) {
                if !used && value & format != 0 {
                    self.out.set(register, value & !format);
                }
            }
        }
    }

    fn scratch_per_lane(&self, tmpring: u32, wave: u8, rsrc2: u32) -> u32 {
        if rsrc2 & 1 != 0 { ((tmpring >> 12 & 0x1fff) * 1024 / u32::from(wave)) & !3 } else { 0 }
    }

    fn link_inputs(&self, user_data: u32, registers: u32, sgpr: u8, rsrc2: u32) -> LinkInputs {
        let mut values = [0u32; 32];
        for (index, slot) in values.iter_mut().take(registers as usize).enumerate() {
            *slot = self.sh(user_data + index as u32);
        }
        // USER_SGPR [5:1]; the graphics stages' 32 registers need USER_SGPR_MSB [27] too.
        let msb = if registers > 16 { ((rsrc2 >> 27) & 1) << 5 } else { 0 };
        let loaded = (((rsrc2 >> 1) & 0x1f) | msb).min(registers);
        LinkInputs { second_half: 0, user_data: values, user_data_sgpr: sgpr, user_sgprs: loaded as u8, unknown_user_data: 0 }
    }

    fn pixel_interface(&self) -> PixelInterface {
        const ENABLE: u32 = 1 << 30;
        const SEPARATE_ALPHA: u32 = 1 << 29;
        let blend = self.context_or_zero(cx(gfx10::CB_BLEND0_CONTROL::OFFSET));
        let second_source = |factor: u32| (15..=18).contains(&(factor & 0x1f));
        let colour = second_source(blend) || second_source(blend >> 8);
        let alpha = blend & SEPARATE_ALPHA != 0 && (second_source(blend >> 16) || second_source(blend >> 24));
        PixelInterface {
            input_ena: self.context_or_zero(cx(gfx10::SPI_PS_INPUT_ENA::OFFSET)),
            input_addr: self.context_or_zero(cx(gfx10::SPI_PS_INPUT_ADDR::OFFSET)),
            col_format: self.context_or_zero(cx(gfx10::SPI_SHADER_COL_FORMAT::OFFSET)),
            z_format: self.context_or_zero(cx(gfx10::SPI_SHADER_Z_FORMAT::OFFSET)),
            dual_source_blend: blend & ENABLE != 0 && (colour || alpha),
        }
    }

    /// The rings of a tessellating draw (radv `ac_emit_cp_tess_rings`): where the geometry
    /// engine reads the factors, and how much of the factor ring and of the off-chip ring each
    /// shader engine gets. They come from what the title told the driver
    /// ([`crate::agc::driver::Tessellation`]); a register the title wrote itself was translated
    /// with the others and stays.
    fn tessellation_rings(&mut self) -> std::result::Result<(), String> {
        let rings = crate::agc::driver::Driver::get().tessellation.rings();
        let written = |dword: u32| self.guest.get(UserConfig, uc(dword)).is_some();
        let (base, size, param) = (
            written(gfx10::VGT_TF_MEMORY_BASE::OFFSET),
            written(gfx10::VGT_TF_RING_SIZE::OFFSET),
            written(gfx10::VGT_HS_OFFCHIP_PARAM::OFFSET),
        );
        if !base || !size {
            let (address, bytes) = rings.factor.ok_or("draw skipped: it tessellates and the title set no tessellation factor ring")?;
            if !base {
                self.out.set(host::VGT_TF_MEMORY_BASE, (address >> 8) as u32);
                self.out.set(host::VGT_TF_MEMORY_BASE_HI, (address >> 40) as u32 & 0xff);
            }
            if !size {
                self.out.set(host::VGT_TF_RING_SIZE, fields::vgt_tf_ring_size(bytes / 4, SHADER_ENGINES).unwrap_or(1));
            }
        }
        if !param {
            let guest = rings.offchip.ok_or("draw skipped: it tessellates and the title set no off-chip tessellation ring size")?;
            self.out.set(host::VGT_HS_OFFCHIP_PARAM, fields::vgt_hs_offchip_param(guest, SHADER_ENGINES).unwrap_or(0));
        }
        Ok(())
    }

    fn plan(mut self) -> std::result::Result<(GraphicsState, Linkage), String> {
        let stages_en = self.context_or_zero(cx(gfx10::VGT_SHADER_STAGES_EN::OFFSET));
        if stages_en & PRIMGEN_EN == 0 {
            return Err("draw without NGG (PRIMGEN_EN): legacy vertex routing has no gfx11 path".into());
        }
        if self.context_or_zero(cx(gfx10::VGT_STRMOUT_CONFIG::OFFSET)) & STREAMOUT_ENABLES != 0 {
            return Err("draw skipped: streamout (VGT_STRMOUT_CONFIG) is not supported".into());
        }
        self.defaults();
        self.pass_through();
        self.field_translations();
        self.special_registers()?;
        self.relocate_surfaces();
        self.disable_unused_depth_stencil();
        for index in 0..host::SX_MRT_BLEND_OPT.count {
            let blend = self.context_or_zero(cx(gfx10::CB_BLEND0_CONTROL::OFFSET) + index);
            self.out.put(host::SX_MRT_BLEND_OPT, index, fields::sx_mrt_blend_opt(blend));
        }
        let programs = &self.runtime.programs;
        let tmpring = self.context_or_zero(cx(gfx10::SPI_TMPRING_SIZE::OFFSET));
        let vertex_wave: u8 = if stages_en & GS_W32_EN != 0 { 32 } else { 64 };
        let vertex_rsrc2 = self.sh(shader(gfx10::SPI_SHADER_PGM_RSRC2_GS::OFFSET));
        let user_gs = shader(gfx10::SPI_SHADER_USER_DATA_GS_0::OFFSET);
        let vertex = programs.resolve(&ProgramRequest {
            guest_address: self.program_address(shader(gfx10::SPI_SHADER_PGM_LO_ES::OFFSET)),
            stage: ShaderStage::Vertex,
            wave_size: vertex_wave,
            guest_rsrc1: self.sh(shader(gfx10::SPI_SHADER_PGM_RSRC1_GS::OFFSET)),
            guest_rsrc2: vertex_rsrc2,
            pixel: None,
            link: LinkInputs {
                second_half: self.program_address(shader(gfx10::SPI_SHADER_PGM_LO_GS::OFFSET)),
                ..self.link_inputs(user_gs, 32, NGG_USER_DATA_SGPR, vertex_rsrc2)
            },
            attribute_ring: Some(self.runtime.attribute_ring),
            vertex_stage: VertexStage::EsGs,
            scratch_bytes_per_lane: self.scratch_per_lane(tmpring, vertex_wave, vertex_rsrc2),
        });
        let vertex = vertex.map_err(|reason| format!("draw skipped: its vertex program: {reason}"))?;
        // Tessellation: the merged LS-HS program.
        let hull_wave: u8 = if stages_en & HS_W32_EN != 0 { 32 } else { 64 };
        let user_hs = shader(gfx10::SPI_SHADER_USER_DATA_HS_0::OFFSET);
        let hull = if stages_en & HS_EN == 0 {
            None
        } else {
            let rsrc2 = self.sh(shader(gfx10::SPI_SHADER_PGM_RSRC2_HS::OFFSET));
            let request = ProgramRequest {
                guest_address: self.program_address(shader(gfx10::SPI_SHADER_PGM_LO_LS::OFFSET)),
                stage: ShaderStage::Vertex,
                wave_size: hull_wave,
                guest_rsrc1: self.sh(shader(gfx10::SPI_SHADER_PGM_RSRC1_HS::OFFSET)),
                guest_rsrc2: rsrc2,
                pixel: None,
                link: LinkInputs {
                    second_half: self.program_address(shader(gfx10::SPI_SHADER_PGM_LO_HS::OFFSET)),
                    ..self.link_inputs(user_hs, 32, NGG_USER_DATA_SGPR, rsrc2)
                },
                attribute_ring: None,
                vertex_stage: VertexStage::LsHs,
                scratch_bytes_per_lane: self.scratch_per_lane(tmpring, hull_wave, rsrc2),
            };
            self.tessellation_rings()?;
            Some(programs.resolve(&request).map_err(|reason| format!("draw skipped: its LS-HS program: {reason}"))?)
        };
        let ps_control = self.context_or_zero(cx(gfx10::SPI_PS_IN_CONTROL::OFFSET));
        let pixel_wave: u8 = if ps_control & (1 << 15) != 0 { 32 } else { 64 };
        let pixel_address = self.program_address(shader(gfx10::SPI_SHADER_PGM_LO_PS::OFFSET));
        let user_ps = shader(gfx10::SPI_SHADER_USER_DATA_PS_0::OFFSET);
        let interface = self.pixel_interface();
        let pixel = if pixel_address == 0 {
            let request = ProgramRequest {
                guest_address: 0,
                stage: ShaderStage::Fragment,
                wave_size: pixel_wave,
                guest_rsrc1: NULL_PIXEL_RSRC1,
                guest_rsrc2: 0,
                pixel: Some(PixelInterface { input_ena: NULL_PIXEL_INPUT, input_addr: NULL_PIXEL_INPUT, dual_source_blend: false, ..interface }),
                link: LinkInputs::default(),
                attribute_ring: None,
                vertex_stage: VertexStage::EsGs,
                scratch_bytes_per_lane: 0,
            };
            programs.resolve_runtime(&NULL_PIXEL_WORDS, &request)
        } else {
            let rsrc2 = self.sh(shader(gfx10::SPI_SHADER_PGM_RSRC2_PS::OFFSET));
            programs.resolve(&ProgramRequest {
                guest_address: pixel_address,
                stage: ShaderStage::Fragment,
                wave_size: pixel_wave,
                guest_rsrc1: self.sh(shader(gfx10::SPI_SHADER_PGM_RSRC1_PS::OFFSET)),
                guest_rsrc2: rsrc2,
                pixel: Some(interface),
                link: self.link_inputs(user_ps, 32, 0, rsrc2),
                attribute_ring: None,
                vertex_stage: VertexStage::EsGs,
                scratch_bytes_per_lane: self.scratch_per_lane(tmpring, pixel_wave, rsrc2),
            })
        };
        let pixel = pixel.map_err(|reason| format!("draw skipped: its pixel program: {reason}"))?;
        if vertex.gs_alloc_req == Some(false) {
            return Err("draw skipped: the NGG vertex program never sends GS_ALLOC_REQ, so the geometry engine would wait for it forever".into());
        }
        if vertex.drops_parameters {
            return Err("draw skipped: the translated vertex program dropped its parameter exports, which had no attribute ring".into());
        }
        if let Some(format) = pixel.spi_shader_col_format {
            self.out.set(host::SPI_SHADER_COL_FORMAT, format);
        }
        if pixel_address == 0 {
            // A draw without a pixel program writes depth and stencil only.
            let db = fields::db_shader_control(self.context_or_zero(cx(gfx10::DB_SHADER_CONTROL::OFFSET))).unwrap_or(0);
            self.out.set(host::SPI_PS_INPUT_ENA, NULL_PIXEL_INPUT);
            self.out.set(host::SPI_PS_INPUT_ADDR, NULL_PIXEL_INPUT);
            self.out.set(host::CB_SHADER_MASK, 0);
            self.out.set(host::DB_SHADER_CONTROL, db & !NULL_PIXEL_DB_EXPORTS);
        }
        let scratch_programs = [
            (vertex.scratch_bytes_per_lane, vertex_wave),
            (pixel.scratch_bytes_per_lane, pixel_wave),
            (hull.as_ref().map_or(0, |program| program.scratch_bytes_per_lane), hull_wave),
        ];
        size_scratch(&mut self.out, host::SPI_TMPRING_SIZE, &scratch_programs, self.runtime.scratch_graphics);
        scratch_base(&mut self.out, host::SPI_TMPRING_SIZE, [host::SPI_GFX_SCRATCH_BASE_LO, host::SPI_GFX_SCRATCH_BASE_HI], self.runtime.scratch_graphics)
            .ok_or("draw skipped: its scratch size needs a larger host scratch ring")?;
        let code = |address: u64| [(address >> 8) as u32, (address >> 40) as u32 & 0xff];
        let [es_lo, es_hi] = code(vertex.host_address);
        let [ps_lo, ps_hi] = code(pixel.host_address);
        self.out.set(host::SPI_SHADER_PGM_LO_ES, es_lo);
        self.out.set(host::SPI_SHADER_PGM_HI_ES, es_hi);
        self.out.set(host::SPI_SHADER_PGM_RSRC1_GS, vertex.rsrc1);
        self.out.set(host::SPI_SHADER_PGM_RSRC2_GS, vertex.rsrc2);
        self.out.set(host::SPI_SHADER_PGM_LO_PS, ps_lo);
        self.out.set(host::SPI_SHADER_PGM_HI_PS, ps_hi);
        self.out.set(host::SPI_SHADER_PGM_RSRC1_PS, pixel.rsrc1);
        self.out.set(host::SPI_SHADER_PGM_RSRC2_PS, pixel.rsrc2);
        let parameters = vertex.parameters;
        self.out.set(host::SPI_VS_OUT_CONFIG, (parameters.max(1) - 1) << 1 | u32::from(parameters == 0) << 7);
        if let Some(hull) = &hull {
            let [ls_lo, ls_hi] = code(hull.host_address);
            self.out.set(host::SPI_SHADER_PGM_LO_LS, ls_lo);
            self.out.set(host::SPI_SHADER_PGM_HI_LS, ls_hi);
            self.out.set(host::SPI_SHADER_PGM_RSRC1_HS, hull.rsrc1);
            self.out.set(host::SPI_SHADER_PGM_RSRC2_HS, hull.rsrc2);
        }
        self.clamp_screen_scissor();
        self.repair_geometry_engine_limits();
        let mut stages = vec![
            Stage { program: vertex, user_base: user_gs, user_sgpr: NGG_USER_DATA_SGPR, user_count: 32 },
            Stage { program: pixel, user_base: user_ps, user_sgpr: 0, user_count: 32 },
        ];
        if let Some(hull) = hull {
            stages.push(Stage { program: hull, user_base: user_hs, user_sgpr: NGG_USER_DATA_SGPR, user_count: 32 });
        }
        let linkage = Linkage(stages.iter().map(|stage| (stage.user_base, stage.program.pointer_user_data)).collect());
        let primitive_type = fields::vgt_primitive_type(self.guest.get_or_zero(UserConfig, uc(gfx10::VGT_PRIMITIVE_TYPE::OFFSET)));
        let targets = super::surface::RenderTargets::of(self.guest).map(Arc::new);
        Ok((GraphicsState { registers: self.out, primitive_type, stages, targets: targets.ok() }, linkage))
    }

    /// PA_SC_SCREEN_SCISSOR_BR clamped to the smallest surface the draw binds, as radv does, so
    /// no pixel lands outside one.
    fn clamp_screen_scissor(&mut self) {
        let value = |plan: &Self, register: HostRegister, index: u32| plan.out.get(register, index);
        let mut extent: Option<(u32, u32)> = None;
        let mut bound = |width: u32, height: u32| {
            extent = Some(extent.map_or((width, height), |(w, h)| (w.min(width), h.min(height))));
        };
        let mode = (value(self, host::CB_COLOR_CONTROL, 0).unwrap_or(0) >> 4) & 7;
        let mask = value(self, host::CB_TARGET_MASK, 0).unwrap_or(0);
        if mode != 0 {
            for slot in 0..host::CB_COLOR_INFO.count {
                let info = value(self, host::CB_COLOR_INFO, slot).unwrap_or(0);
                if (mask >> (4 * slot)) & 0xf == 0 || info & 0x1f == 0 {
                    continue;
                }
                let attrib2 = value(self, host::CB_COLOR_ATTRIB2, slot).unwrap_or(0);
                bound(((attrib2 >> 14) & 0x3fff) + 1, (attrib2 & 0x3fff) + 1);
            }
        }
        let depth = value(self, host::DB_Z_INFO, 0).unwrap_or(0) & 3;
        let stencil = value(self, host::DB_STENCIL_INFO, 0).unwrap_or(0) & 1;
        if depth != 0 || stencil != 0 {
            let size = value(self, host::DB_DEPTH_SIZE_XY, 0).unwrap_or(0);
            bound((size & 0x3fff) + 1, ((size >> 16) & 0x3fff) + 1);
        }
        let Some((width, height)) = extent else {
            return;
        };
        const RESET_BR: u32 = 0x4000_4000;
        let corner = value(self, host::PA_SC_SCREEN_SCISSOR_BR, 0).unwrap_or(RESET_BR);
        self.out.set(host::PA_SC_SCREEN_SCISSOR_BR, (corner & 0xffff).min(width) | ((corner >> 16).min(height) << 16));
    }

    /// Each [`GEOMETRY_ENGINE_LIMITS`] field the state leaves at 0 set to its host value.
    fn repair_geometry_engine_limits(&mut self) {
        for (register, field, host_value) in GEOMETRY_ENGINE_LIMITS {
            let value = self.out.get(register, 0).unwrap_or(0);
            if value & field == 0 {
                self.out.set(register, value | host_value);
            }
        }
    }
}

/// A dispatch's host state from the guest registers and program.
struct ComputePlan<'a> {
    runtime: &'a TargetRuntime,
    guest: &'a GuestRegisters,
    initiator: u32,
    compute: bool,
}

struct ComputeState {
    packets: [u32; super::draw::dwords::COMPUTE_STATE as usize],
    stage: Stage,
}

impl ComputeState {
    fn record(self, key: u64) -> Cached<Arc<StateRecord>> {
        let watched = watched_registers(std::slice::from_ref(&self.stage));
        Ok(Arc::new(StateRecord {
            key,
            block: None,
            registers: Box::new([]),
            compute_packets: Some(self.packets),
            primitive_type: 0,
            stages: vec![self.stage],
            watched,
            targets: None,
        }))
    }
}

impl<'a> ComputePlan<'a> {
    fn new(runtime: &'a TargetRuntime, guest: &'a GuestRegisters, initiator: u32, compute: bool) -> Self {
        Self { runtime, guest, initiator, compute }
    }

    fn sh(&self, dword: u32) -> u32 {
        self.guest.get_or_zero(Shader, dword)
    }

    fn plan(self) -> std::result::Result<(ComputeState, Linkage), String> {
        let wave: u8 = if self.initiator & CS_W32_EN != 0 { 32 } else { 64 };
        let rsrc2 = self.sh(shader(gfx10::COMPUTE_PGM_RSRC2::OFFSET));
        let user_cs = shader(gfx10::COMPUTE_USER_DATA_0::OFFSET);
        let tmpring = self.sh(shader(gfx10::COMPUTE_TMPRING_SIZE::OFFSET));
        let mut values = [0u32; 32];
        for (index, slot) in values.iter_mut().take(16).enumerate() {
            *slot = self.sh(user_cs + index as u32);
        }
        let loaded = ((rsrc2 >> 1) & 0x1f).min(16);
        let per_lane = if rsrc2 & 1 != 0 { ((tmpring >> 12 & 0x1fff) * 1024 / u32::from(wave)) & !3 } else { 0 };
        let program = self.runtime.programs.resolve(&ProgramRequest {
            guest_address: address_of(self.sh(shader(gfx10::COMPUTE_PGM_LO::OFFSET)), self.sh(shader(gfx10::COMPUTE_PGM_HI::OFFSET))),
            stage: ShaderStage::Compute,
            wave_size: wave,
            guest_rsrc1: self.sh(shader(gfx10::COMPUTE_PGM_RSRC1::OFFSET)),
            guest_rsrc2: rsrc2,
            pixel: None,
            link: LinkInputs { second_half: 0, user_data: values, user_data_sgpr: 0, user_sgprs: loaded as u8, unknown_user_data: 0 },
            attribute_ring: None,
            vertex_stage: VertexStage::EsGs,
            scratch_bytes_per_lane: per_lane,
        });
        let program = program.map_err(|reason| format!("dispatch skipped: its compute program: {reason}"))?;
        let ring = if self.compute { self.runtime.scratch_compute } else { self.runtime.scratch_graphics };
        let mut out = HostRegisters::default();
        out.set(host::COMPUTE_TMPRING_SIZE, fields::tmpring_size(tmpring, SHADER_ENGINES)?);
        size_scratch(&mut out, host::COMPUTE_TMPRING_SIZE, &[(program.scratch_bytes_per_lane, wave)], ring);
        let scratch = scratch_base(&mut out, host::COMPUTE_TMPRING_SIZE, [host::COMPUTE_DISPATCH_SCRATCH_BASE_LO, host::COMPUTE_DISPATCH_SCRATCH_BASE_HI], ring)
            .ok_or("dispatch skipped: its scratch size needs a larger host scratch ring")?;
        let tmpring = out.get(host::COMPUTE_TMPRING_SIZE, 0).unwrap_or(0);
        let (limits, start, threads) = (
            self.sh(shader(gfx10::COMPUTE_RESOURCE_LIMITS::OFFSET)),
            shader(gfx10::COMPUTE_START_X::OFFSET),
            shader(gfx10::COMPUTE_NUM_THREAD_X::OFFSET),
        );
        let mut packets = [0u32; super::draw::dwords::COMPUTE_STATE as usize];
        let mut at = 0;
        let mut set = |register: HostRegister, values: &[u32]| -> Result<()> {
            packets[at] = header(pkt3::SET_SH_REG, values.len() as u32 + 2, 0)?;
            packets[at + 1] = register.dword;
            packets[at + 2..at + 2 + values.len()].copy_from_slice(values);
            at += values.len() + 2;
            Ok(())
        };
        let code = program.host_address;
        let encode = |result: Result<()>| result.map_err(|error| error.message);
        encode(set(host::COMPUTE_PGM_LO, &[(code >> 8) as u32, (code >> 40) as u32 & 0xff]))?;
        encode(set(host::COMPUTE_PGM_RSRC1, &[program.rsrc1, program.rsrc2]))?;
        encode(set(host::COMPUTE_PGM_RSRC3, &[0]))?;
        encode(set(host::COMPUTE_RESOURCE_LIMITS, &[limits]))?;
        encode(set(host::COMPUTE_START, &[self.sh(start), self.sh(start + 1), self.sh(start + 2)]))?;
        encode(set(host::COMPUTE_NUM_THREAD, &[self.sh(threads), self.sh(threads + 1), self.sh(threads + 2)]))?;
        encode(set(host::COMPUTE_TMPRING_SIZE, &[tmpring]))?;
        encode(set(host::COMPUTE_DISPATCH_SCRATCH_BASE_LO, &scratch))?;
        debug_assert_eq!(at, packets.len());
        let linkage = Linkage(vec![(user_cs, program.pointer_user_data)]);
        Ok((ComputeState { packets, stage: Stage { program, user_base: user_cs, user_sgpr: 0, user_count: 16 } }, linkage))
    }
}

/// Raises TMPRING_SIZE.WAVESIZE in `out` to what the translated programs' flat scratch needs
/// (`(bytes per lane, wave size)` each). WAVES stays the guest's; without guest scratch it is as
/// many waves per shader engine as the ring holds.
fn size_scratch(out: &mut HostRegisters, tmpring: HostRegister, programs: &[(u32, u8)], ring: ScratchRing) {
    let needed = programs.iter().map(|&(bytes, wave)| (u64::from(bytes) * u64::from(wave)).div_ceil(256)).max().unwrap_or(0);
    let current = out.get(tmpring, 0).unwrap_or(0);
    let (waves, wave_size) = (current & 0xfff, (current >> 12) & 0x7fff);
    if needed <= u64::from(wave_size) {
        return;
    }
    let wave_size = needed.min(0x7fff);
    let waves = if waves != 0 {
        waves
    } else {
        (ring.bytes / (wave_size * 256 * u64::from(SHADER_ENGINES))).clamp(1, 0xfff) as u32
    };
    out.set(tmpring, waves | ((wave_size as u32) << 12));
}

/// The scratch base of the last TMPRING_SIZE in `out`: zeros without scratch, `None` when the ring
/// is too small for what it asks.
fn scratch_base(out: &mut HostRegisters, tmpring: HostRegister, registers: [HostRegister; 2], ring: ScratchRing) -> Option<[u32; 2]> {
    let current = out.get(tmpring, 0).unwrap_or(0);
    let needed = u64::from(current & 0xfff) * u64::from((current >> 12) & 0x7fff) * 256 * u64::from(SHADER_ENGINES);
    let base = if needed == 0 {
        [0, 0]
    } else {
        if ring.bytes < needed || ring.address % 256 != 0 {
            return None;
        }
        [(ring.address >> 8) as u32, (ring.address >> 40) as u32]
    };
    out.set(registers[0], base[0]);
    out.set(registers[1], base[1]);
    Some(base)
}

/// Prints each reason `what` once per process.
pub(super) fn report_once(what: &str, seed: u32) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: [AtomicU32; 256] = [const { AtomicU32::new(0) }; 256];
    let tag = (what.bytes().fold(seed, |hash, byte| hash.wrapping_mul(31).wrapping_add(u32::from(byte))) | 1).max(1);
    let index = (tag % 256) as usize;
    if SEEN[index].swap(tag, Ordering::Relaxed) != tag {
        eprintln!("[agc] {what}");
    }
}
