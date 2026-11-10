use crate::ir::opcodes::flag;
use crate::ir::{
    Fields, FlatSegment, MemKind, Operand, RegFile, RegRange, ResourceBinding, ResourceKind,
    ShaderIr, ShaderMetadata, SGPR_LIMIT, VGPR_LIMIT,
};
use crate::{CompileError, CompileOptions, ShaderInterface, ShaderStage, TargetIsa, VertexStage};

#[cfg(test)]
mod tests;

pub const RSRC1_VGPRS_MASK: u32 = 0x3f;
pub const RSRC1_SGPRS_MASK: u32 = 0xf << 6;
pub const RSRC1_FLOAT_MODE_SHIFT: u32 = 12;
/// Round to nearest even, f32 denormals flushed, f16/f64 denormals kept: the usual PS5 and radv
/// setting, used when the guest RSRC1 is unknown.
pub const FLOAT_MODE_DEFAULT: u32 = 0xc0;
pub const RSRC1_DX10_CLAMP: u32 = 1 << 21;
pub const RSRC1_IEEE_MODE: u32 = 1 << 23;
/// COMPUTE_PGM_RSRC1 positions.
pub const RSRC1_WGP_MODE: u32 = 1 << 29;
pub const RSRC1_MEM_ORDERED: u32 = 1 << 30;
/// SPI_SHADER_PGM_RSRC1_PS/GS position (bits 29-30 of RSRC1_GS are GS_VGPR_COMP_CNT).
pub const RSRC1_GRAPHICS_MEM_ORDERED: u32 = 1 << 25;
pub const RSRC2_SCRATCH_EN: u32 = 1;
pub const RSRC2_USER_SGPR_SHIFT: u32 = 1;
pub const RSRC2_USER_SGPR_MASK: u32 = 0x1f << RSRC2_USER_SGPR_SHIFT;
pub const RSRC2_TRAP_PRESENT: u32 = 1 << 6;
pub const RSRC2_TGID_X_EN: u32 = 1 << 7;
pub const RSRC2_TG_SIZE_EN: u32 = 1 << 10;
pub const RSRC2_TIDIG_SHIFT: u32 = 11;
pub const RSRC2_TIDIG_MASK: u32 = 3 << RSRC2_TIDIG_SHIFT;
pub const RSRC2_LDS_SHIFT: u32 = 15;
pub const RSRC2_LDS_MASK: u32 = 0x1ff << RSRC2_LDS_SHIFT;
/// SPI_SHADER_PGM_RSRC2_GS LDS_SIZE.
pub const RSRC2_GS_LDS_SHIFT: u32 = 19;
pub const RSRC2_GS_LDS_MASK: u32 = 0xff << RSRC2_GS_LDS_SHIFT;
/// SPI_SHADER_PGM_RSRC2_HS LDS_SIZE, bits 26:18 in 512-byte blocks.
pub const RSRC2_HS_LDS_SHIFT: u32 = 18;
pub const RSRC2_HS_LDS_MASK: u32 = 0x1ff << RSRC2_HS_LDS_SHIFT;
/// SPI_SHADER_PGM_RSRC1_HS LS_VGPR_COMP_CNT: how many VGPRs after the vertex ID (v2) the
/// local part receives.
const RSRC1_HS_LS_VGPR_COMP_CNT_SHIFT: u32 = 28;
/// SPI_SHADER_PGM_RSRC2_PS EXTRA_LDS_SIZE: 512-byte units on gfx10, 1 KiB on gfx11 and later.
pub const RSRC2_PS_EXTRA_LDS_SHIFT: u32 = 8;
pub const RSRC2_PS_EXTRA_LDS_MASK: u32 = 0xff << RSRC2_PS_EXTRA_LDS_SHIFT;
pub const RSRC2_USER_SGPR_MSB: u32 = 1 << 27;
pub const LDS_GRANULE_BYTES: u32 = 512;
pub const MAX_COMPUTE_USER_SGPRS: u8 = 16;
pub const MAX_GRAPHICS_USER_SGPRS: u8 = 32;
pub const SYSTEM_SGPR_ENABLES: usize = 4;
const SCRATCH_SGPR_ALIGN: u16 = 4;

const SMEM_SCRATCH_LOADS: std::ops::RangeInclusive<u16> = 0x05..=0x07;
const SMEM_GET_WAVEID: u16 = 0x2a;
const EXP_MRT_LAST: u8 = 7;
const EXP_MRTZ: u8 = 8;
const EXP_POS_FIRST: u8 = 12;
const EXP_POS_LAST: u8 = 16;
const EXP_PRIM: u8 = 20;
const EXP_PARAM_FIRST: u8 = 32;

pub fn vgpr_granule(wave64: bool) -> u16 {
    if wave64 {
        4
    } else {
        8
    }
}

pub fn encode_vgpr_blocks(count: u16, wave64: bool) -> u32 {
    let granule = vgpr_granule(wave64);
    u32::from(count.max(1).div_ceil(granule) - 1)
}

pub fn decode_vgpr_blocks(rsrc1: u32, wave64: bool) -> u16 {
    ((rsrc1 & RSRC1_VGPRS_MASK) as u16 + 1) * vgpr_granule(wave64)
}

pub fn analyze(ir: &mut ShaderIr, options: CompileOptions, interface: &ShaderInterface) {
    let wave64 = options.wave64();
    let mut meta = ShaderMetadata {
        user_data_tail_sgpr: ir.metadata.user_data_tail_sgpr,
        stage: Some(options.stage),
        ..ShaderMetadata::default()
    };
    let mut max_sgpr: Option<u16> = None;
    let mut max_vgpr: Option<u16> = None;
    let mut bindings: Vec<ResourceBinding> = Vec::new();
    let mut bind = |kind, base: u8, dword_count: u8, writable: bool| match bindings
        .iter_mut()
        .find(|b| b.kind == kind && b.base_sgpr == base && b.dword_count == dword_count)
    {
        Some(existing) => existing.writable |= writable,
        None => bindings.push(ResourceBinding {
            kind,
            base_sgpr: base,
            dword_count,
            writable,
        }),
    };

    for block in ir.blocks.iter().filter(|b| b.reachable) {
        for inst in &ir.instructions[block.instructions()] {
            let mut track = |range: RegRange| match range.file {
                RegFile::Vector => {
                    max_vgpr = max_vgpr.max(Some(range.last().min(VGPR_LIMIT - 1)));
                }
                RegFile::Scalar if range.first < SGPR_LIMIT => {
                    max_sgpr = max_sgpr.max(Some(range.last().min(SGPR_LIMIT - 1)));
                }
                RegFile::Scalar => {
                    meta.uses_vcc |= range.first <= crate::ir::SCALAR_VCC_HI;
                }
            };
            inst.each_def(wave64, &mut track);
            inst.each_use(wave64, &mut track);
            meta.uses_pc_relative |= inst.has_flag(flag::PC_RELATIVE);
            meta.uses_relative_indexing |= inst.has_flag(flag::RELATIVE);
            let writes = |kind: MemKind| kind != MemKind::Load;
            match &inst.fields {
                Fields::Smem(s) if s.dwords > 0 && !inst.has_flag(flag::NO_SRC) => {
                    if s.buffer {
                        bind(ResourceKind::BufferDescriptor, s.sbase, 4, false);
                    } else {
                        bind(ResourceKind::ScalarMemoryPair, s.sbase, 2, false);
                    }
                    meta.uses_scratch |= SMEM_SCRATCH_LOADS.contains(&inst.opcode);
                }
                Fields::Smem(_) => meta.uses_wave_id |= inst.opcode == SMEM_GET_WAVEID,
                Fields::Mubuf(b) | Fields::Mtbuf(b) if b.kind != MemKind::Control => {
                    bind(ResourceKind::BufferDescriptor, b.srsrc, 4, writes(b.kind));
                    meta.uses_lds |= b.lds;
                }
                Fields::Mimg(m) => {
                    let dwords = if m.r128 { 4 } else { 8 };
                    bind(
                        ResourceKind::ImageDescriptor,
                        m.srsrc,
                        dwords,
                        writes(m.kind),
                    );
                    if m.uses_sampler {
                        bind(ResourceKind::SamplerDescriptor, m.ssamp, 4, false);
                    }
                }
                Fields::Flat(f) => {
                    if let Some(Operand::Sgpr(base)) = f.saddr {
                        if f.segment == FlatSegment::Global {
                            bind(ResourceKind::ScalarMemoryPair, base, 2, writes(f.kind));
                        }
                    }
                    meta.uses_scratch |= f.segment == FlatSegment::Scratch;
                    meta.uses_lds |= f.lds;
                }
                Fields::Ds(d) => {
                    meta.uses_lds |= !d.gds && d.uses_addr;
                    meta.uses_gds |= d.gds || inst.has_flag(flag::READS_M0);
                }
                Fields::Vintrp(_) => meta.uses_interpolation = true,
                Fields::Vop3(v) => {
                    meta.uses_interpolation |= crate::lowering::pixel::is_vop3_interp(inst.opcode);
                    meta.uses_scratch |= v
                        .src
                        .iter()
                        .any(|s| matches!(s, Operand::PrivateBase | Operand::PrivateLimit));
                }
                Fields::Exp(e) => record_export(&mut meta, e.target, e.en, e.compr, e.vm),
                _ => {}
            }
        }
    }

    bindings.sort_by_key(|b| (b.base_sgpr, b.kind));
    meta.bindings = bindings;
    meta.max_sgpr = max_sgpr.map(|v| v as u8);
    meta.max_vgpr = max_vgpr.map(|v| v as u8);
    meta.sgpr_count = max_sgpr.map_or(0, |v| v + 1);
    meta.vgpr_count = max_vgpr.map_or(0, |v| v + 1);
    (meta.live_in_scalar, meta.live_in_vector) = live_in(ir, wave64);
    meta.user_sgpr_count = user_sgpr_count(&meta, options, interface);
    let user_data = match options.stage {
        ShaderStage::Vertex => {
            NGG_SYSTEM_SGPRS..NGG_SYSTEM_SGPRS + u16::from(MAX_GRAPHICS_USER_SGPRS)
        }
        _ => 0..u16::from(meta.user_sgpr_count),
    };
    let uses = crate::cfg::sources::descriptor_uses(ir, wave64, user_data);
    meta.descriptor_uses = uses.descriptors;
    meta.pointer_uses = uses.pointers;
    if let Some(rsrc2) = interface.guest_pgm_rsrc2 {
        meta.uses_scratch |= rsrc2 & RSRC2_SCRATCH_EN != 0;
    }
    meta.uses_scratch |= interface.scratch_bytes_per_lane > 0;
    ir.metadata = meta;
}

fn record_export(meta: &mut ShaderMetadata, target: u8, en: u8, compr: bool, vm: bool) {
    let exports = &mut meta.exports;
    exports.compressed |= compr;
    exports.valid_mask |= vm;
    match target {
        0..=EXP_MRT_LAST => exports.color_mask |= u32::from(en & 0xf) << (4 * u32::from(target)),
        EXP_MRTZ => exports.depth_mask |= en & 0xf,
        EXP_POS_FIRST..=EXP_POS_LAST => exports.position_mask |= 1 << (target - EXP_POS_FIRST),
        EXP_PRIM => exports.primitive = true,
        EXP_PARAM_FIRST.. => exports.param_mask |= 1 << (target - EXP_PARAM_FIRST),
        _ => {}
    }
}

/// The scalar and vector registers live into the program.
fn live_in(ir: &ShaderIr, wave64: bool) -> (u128, u128) {
    let file = |range: &RegRange| usize::from(range.file == RegFile::Vector);
    // Per block and file, the registers read before any write, and those written.
    let summaries: Vec<[(u128, u128); 2]> = ir
        .blocks
        .iter()
        .map(|block| {
            let mut summary = [(0u128, 0u128); 2];
            for inst in &ir.instructions[block.instructions()] {
                inst.each_use(wave64, |range| {
                    let (uses, defs) = &mut summary[file(&range)];
                    *uses |= scalar_mask(range.first, range.count) & !*defs;
                });
                inst.each_def(wave64, |range| {
                    summary[file(&range)].1 |= scalar_mask(range.first, range.count);
                });
            }
            summary
        })
        .collect();
    let mut live_in = vec![[0u128; 2]; ir.blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (id, block) in ir.blocks.iter().enumerate().rev() {
            for (f, &(uses, defs)) in summaries[id].iter().enumerate() {
                let live_out = block
                    .successors
                    .iter()
                    .fold(0u128, |acc, &s| acc | live_in[s][f]);
                let value = uses | (live_out & !defs);
                if value != live_in[id][f] {
                    live_in[id][f] = value;
                    changed = true;
                }
            }
        }
    }
    live_in
        .first()
        .map_or((0, 0), |&[scalar, vector]| (scalar, vector))
}

pub fn scalar_mask(first: u16, count: u16) -> u128 {
    (first..first.saturating_add(count))
        .filter(|&r| r < 128)
        .fold(0u128, |acc, r| acc | (1u128 << r))
}

fn user_sgpr_count(
    meta: &ShaderMetadata,
    options: CompileOptions,
    interface: &ShaderInterface,
) -> u8 {
    let limit = if options.stage == ShaderStage::Compute {
        MAX_COMPUTE_USER_SGPRS
    } else {
        MAX_GRAPHICS_USER_SGPRS
    };
    if let Some(count) = interface.user_sgpr_count {
        return count.min(limit);
    }
    if let Some(rsrc2) = interface.guest_pgm_rsrc2 {
        let msb = if options.stage != ShaderStage::Compute && rsrc2 & RSRC2_USER_SGPR_MSB != 0 {
            32
        } else {
            0
        };
        return (((rsrc2 & RSRC2_USER_SGPR_MASK) >> RSRC2_USER_SGPR_SHIFT) as u8 + msb).min(limit);
    }
    let sgprs = meta.live_in_scalar & ((1u128 << SGPR_LIMIT) - 1);
    let highest = (u128::BITS - sgprs.leading_zeros()) as u8;
    highest.min(limit)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShaderConfig {
    pub pgm_rsrc1: u32,
    pub pgm_rsrc2: u32,
    pub num_sgprs: u16,
    pub num_vgprs: u16,
    pub lds_bytes: u32,
    pub scratch_bytes_per_lane: u32,
    pub user_sgpr_count: u8,
}

#[derive(Clone, Debug)]
pub struct RegisterAllocator {
    target: TargetIsa,
    wave64: bool,
    scratch_base: u16,
    vgpr_high_water: u16,
    sgpr_high_water: u16,
    scratch_sgpr_base: Option<u16>,
    /// The first SGPR above the guest's: reservations run from here to `scratch_sgpr_base`.
    reserved_from: Option<u16>,
    float_mode: u8,
    thread_id_components: u8,
    vgpr_floor: u16,
    scratch_used: bool,
    system: SystemSgprs,
}

const NGG_SCRATCH_WAVE_OFFSET_SGPR: u8 = 5;
/// Merged NGG waves receive system values in s0-s7; user SGPRs follow.
const NGG_SYSTEM_SGPRS: u16 = 8;

/// Where a compute wave's system SGPRs land: after the user SGPRs, the enabled TGID X/Y/Z, then
/// TG_SIZE, then (gfx10 only) the scratch wave offset. Merged NGG waves have the scratch wave
/// offset in s5.
#[derive(Clone, Copy, Debug, Default)]
struct SystemSgprs {
    tg_size: Option<u8>,
    scratch_wave_offset: Option<u8>,
}

impl SystemSgprs {
    fn new(meta: &ShaderMetadata, options: CompileOptions, interface: &ShaderInterface) -> Self {
        let scratch_en = interface
            .guest_pgm_rsrc2
            .is_some_and(|r| r & RSRC2_SCRATCH_EN != 0);
        match options.stage {
            // Merged NGG waves: gfx10 loads the scratch wave offset into s5, where gfx11
            // loads the attribute ring offset.
            ShaderStage::Vertex => {
                return Self {
                    tg_size: None,
                    scratch_wave_offset: scratch_en.then_some(NGG_SCRATCH_WAVE_OFFSET_SGPR),
                }
            }
            // The ordinary PS ABI loads prim_mask after user data, then the gfx10 scratch
            // wave offset when SCRATCH_EN is set. gfx11 keeps prim_mask and initializes
            // scratch in hardware, so the latter input must be translated just as on CS.
            ShaderStage::Fragment => {
                return Self {
                    tg_size: None,
                    scratch_wave_offset: scratch_en.then_some(meta.user_sgpr_count + 1),
                }
            }
            ShaderStage::Compute => {}
        }
        let user = meta.user_sgpr_count;
        let rsrc2 = interface.guest_pgm_rsrc2;
        let tgids = match rsrc2 {
            Some(r) => ((r >> 7) & 7).count_ones() as u8,
            None => implied_tgids(meta),
        };
        let guest_tg_size = rsrc2.is_some_and(|r| r & RSRC2_TG_SIZE_EN != 0);
        let tg_size = (guest_tg_size || meta.uses_wave_id).then_some(user + tgids);
        let scratch_wave_offset = rsrc2
            .filter(|r| r & RSRC2_SCRATCH_EN != 0)
            .map(|_| user + tgids + u8::from(guest_tg_size));
        Self {
            tg_size,
            scratch_wave_offset,
        }
    }
}

/// TGID components a compute program without a known RSRC2 reads: the live-in SGPRs above the
/// user SGPRs.
fn implied_tgids(meta: &ShaderMetadata) -> u8 {
    let sgprs = meta.live_in_scalar & ((1u128 << SGPR_LIMIT) - 1);
    let highest = (u128::BITS - sgprs.leading_zeros()) as u8;
    highest
        .saturating_sub(meta.user_sgpr_count)
        .min(SYSTEM_SGPR_ENABLES as u8 - 1)
}

impl RegisterAllocator {
    pub fn new(
        meta: &ShaderMetadata,
        options: CompileOptions,
        interface: &ShaderInterface,
    ) -> Self {
        let wave64 = options.wave64();
        let guest_allocation = interface
            .guest_pgm_rsrc1
            .map(|rsrc1| decode_vgpr_blocks(rsrc1, wave64));
        let mut scratch_base = meta.vgpr_count;
        if meta.uses_relative_indexing {
            scratch_base = scratch_base.max(guest_allocation.unwrap_or(VGPR_LIMIT));
        }
        let float_mode = interface
            .guest_pgm_rsrc1
            .map_or(FLOAT_MODE_DEFAULT, |rsrc1| {
                (rsrc1 >> RSRC1_FLOAT_MODE_SHIFT) & 0xff
            }) as u8;
        let thread_id_components = match options.stage {
            ShaderStage::Compute => interface.guest_pgm_rsrc2.map_or_else(
                || compute_thread_id_components(meta),
                |rsrc2| ((rsrc2 & RSRC2_TIDIG_MASK) >> RSRC2_TIDIG_SHIFT) as u8,
            ),
            _ => 0,
        };
        let system = SystemSgprs::new(meta, options, interface);
        // The translation reads hardware-loaded SGPRs the guest may never touch: TG_SIZE (wave
        // ID), the NGG ring offset and wave index (s5, s3), `prim_mask` (hoisted parameter
        // loads). Its own SGPRs start above them.
        let loaded = match options.stage {
            ShaderStage::Compute => system.tg_size.map_or(0, |r| u16::from(r) + 1),
            ShaderStage::Vertex => NGG_SYSTEM_SGPRS,
            ShaderStage::Fragment => u16::from(meta.user_sgpr_count) + 1,
        };
        let scratch_sgpr_base = (!meta.uses_relative_indexing).then(|| {
            meta.sgpr_count
                .max(u16::from(meta.user_sgpr_count))
                .max(loaded)
                .next_multiple_of(SCRATCH_SGPR_ALIGN)
        });
        Self {
            target: options.target,
            wave64,
            scratch_base,
            vgpr_high_water: meta.vgpr_count,
            sgpr_high_water: meta.sgpr_count,
            scratch_sgpr_base,
            reserved_from: scratch_sgpr_base,
            float_mode,
            thread_id_components,
            vgpr_floor: input_vgpr_floor(options, interface, thread_id_components),
            scratch_used: false,
            system,
        }
    }

    /// The SGPR holding TG_SIZE (wave index in bits 24:20) once the host enables it.
    pub fn tg_size_sgpr(&self) -> Option<u8> {
        self.system.tg_size
    }

    /// The SGPR gfx10 loads with the wave's scratch offset (compute with SCRATCH_EN); gfx11
    /// leaves it unset.
    pub fn scratch_wave_offset_sgpr(&self) -> Option<u8> {
        self.system.scratch_wave_offset
    }

    /// Thread ID components the dispatch loads (COMPUTE_PGM_RSRC2 TIDIG_COMP_CNT).
    pub fn thread_id_components(&self) -> u8 {
        self.thread_id_components
    }

    /// One more than the highest VGPR the hardware loads before the shader starts; the VGPR
    /// allocation never ends below it.
    pub fn vgpr_floor(&self) -> u16 {
        self.vgpr_floor
    }

    /// The SGPRs the translation owns above the guest's: its reservations
    /// ([`Self::reserve_sgprs`]) and the transient [`Self::scratch_sgprs`] above them. Empty
    /// when the guest indexes SGPRs relatively, which leaves the translation no SGPR of its own.
    pub fn translation_sgprs(&self) -> std::ops::Range<u16> {
        match self.reserved_from {
            Some(from) => from..SGPR_LIMIT,
            None => 0..0,
        }
    }

    /// SGPRs above the guest's and every reservation: 0 when the guest indexes SGPRs relatively.
    pub fn free_sgprs(&self) -> u16 {
        self.scratch_sgpr_base
            .map_or(0, |base| SGPR_LIMIT.saturating_sub(base))
    }

    /// SGPRs that keep their value for the whole program (a prologue's saved inputs), above
    /// the guest's SGPRs and below the transient [`Self::scratch_sgprs`]. All reservations come
    /// before the first transient use.
    pub fn reserve_sgprs(&mut self, count: u8) -> Option<u8> {
        debug_assert!(!self.scratch_used, "reservations follow transient scratch");
        let base = self.scratch_sgpr_base?;
        let end = base + u16::from(count);
        if end > SGPR_LIMIT {
            return None;
        }
        self.scratch_sgpr_base = Some(end.next_multiple_of(SCRATCH_SGPR_ALIGN));
        self.sgpr_high_water = self.sgpr_high_water.max(end);
        Some(base as u8)
    }

    /// VGPRs that keep their value for the whole program, below the transient
    /// [`Self::scratch_vgprs`].
    pub fn reserve_program_vgprs(&mut self, count: u8) -> Result<u8, CompileError> {
        debug_assert!(!self.scratch_used, "reservations follow transient scratch");
        let base = self.scratch_base;
        let end = base + u16::from(count);
        if end > VGPR_LIMIT {
            return Err(CompileError::OutOfRegisters { vgprs: end });
        }
        self.scratch_base = end;
        self.vgpr_high_water = self.vgpr_high_water.max(end);
        Ok(base as u8)
    }

    /// FLOAT_MODE (MODE bits 7:0) the translated shader starts with.
    pub fn initial_float_mode(&self) -> u8 {
        self.float_mode
    }

    /// SGPRs above every SGPR the guest uses, shared by expansions like `scratch_vgprs`.
    /// `None` when the guest indexes SGPRs relatively or the SGPRs would exceed the file.
    pub fn scratch_sgprs(&mut self, count: u8) -> Option<u8> {
        self.scratch_used = true;
        let base = self.scratch_sgpr_base?;
        let end = base + u16::from(count);
        if end > SGPR_LIMIT {
            return None;
        }
        self.sgpr_high_water = self.sgpr_high_water.max(end);
        Some(base as u8)
    }

    pub fn target(&self) -> TargetIsa {
        self.target
    }

    pub fn wave64(&self) -> bool {
        self.wave64
    }

    pub fn scratch_base(&self) -> u16 {
        self.scratch_base
    }

    pub fn vgpr_high_water(&self) -> u16 {
        self.vgpr_high_water
    }

    pub fn scratch_vgprs(&mut self, count: u8) -> Result<u8, CompileError> {
        self.scratch_used = true;
        let end = self.scratch_base + u16::from(count);
        if end > VGPR_LIMIT {
            return Err(CompileError::OutOfRegisters { vgprs: end });
        }
        self.vgpr_high_water = self.vgpr_high_water.max(end);
        Ok(self.scratch_base as u8)
    }

    pub fn reserve_vgprs(&mut self, count: u16) {
        self.vgpr_high_water = self.vgpr_high_water.max(count);
    }

    pub fn finish(
        &self,
        meta: &ShaderMetadata,
        options: CompileOptions,
        interface: &ShaderInterface,
    ) -> ShaderConfig {
        let num_vgprs = self.vgpr_high_water.max(self.vgpr_floor).max(1);
        let lds_bytes = if interface.lds_bytes > 0 {
            interface.lds_bytes
        } else if options.stage == ShaderStage::Compute {
            interface.guest_pgm_rsrc2.map_or(0, |r| {
                ((r & RSRC2_LDS_MASK) >> RSRC2_LDS_SHIFT) * LDS_GRANULE_BYTES
            })
        } else {
            0
        };
        ShaderConfig {
            pgm_rsrc1: self.rsrc1(num_vgprs, options, interface),
            pgm_rsrc2: self.rsrc2(meta, lds_bytes, options, interface),
            num_sgprs: self.sgpr_high_water.max(u16::from(meta.user_sgpr_count)),
            num_vgprs,
            lds_bytes,
            scratch_bytes_per_lane: interface.scratch_bytes_per_lane,
            user_sgpr_count: meta.user_sgpr_count,
        }
    }

    fn rsrc1(&self, num_vgprs: u16, options: CompileOptions, interface: &ShaderInterface) -> u32 {
        let mut value = interface
            .guest_pgm_rsrc1
            .unwrap_or_else(|| default_pgm_rsrc1(options.stage))
            & !(RSRC1_VGPRS_MASK | RSRC1_SGPRS_MASK);
        if self.target.is_gfx12() {
            value &= !(RSRC1_DX10_CLAMP | RSRC1_IEEE_MODE);
        }
        value | encode_vgpr_blocks(num_vgprs, self.wave64)
    }

    fn rsrc2(
        &self,
        meta: &ShaderMetadata,
        lds_bytes: u32,
        options: CompileOptions,
        interface: &ShaderInterface,
    ) -> u32 {
        let user = u32::from(meta.user_sgpr_count);
        let mut value = interface.guest_pgm_rsrc2.unwrap_or(0)
            & !(RSRC2_SCRATCH_EN | RSRC2_USER_SGPR_MASK | RSRC2_USER_SGPR_MSB);
        if meta.uses_scratch {
            value |= RSRC2_SCRATCH_EN;
        }
        value |= (user & 0x1f) << RSRC2_USER_SGPR_SHIFT;
        if options.stage == ShaderStage::Compute {
            if meta.uses_wave_id && self.target.is_gfx11_plus() {
                value |= RSRC2_TG_SIZE_EN;
            }
            if interface.guest_pgm_rsrc2.is_none() {
                let sgprs = meta.live_in_scalar & ((1u128 << SGPR_LIMIT) - 1);
                let highest = (u128::BITS - sgprs.leading_zeros()) as usize;
                let system = highest.saturating_sub(user as usize);
                for i in 0..system.min(SYSTEM_SGPR_ENABLES) {
                    value |= RSRC2_TGID_X_EN << i;
                }
                value |= u32::from(compute_thread_id_components(meta)) << RSRC2_TIDIG_SHIFT;
            }
            if lds_bytes > 0 {
                let blocks = self.lds_blocks(lds_bytes.div_ceil(LDS_GRANULE_BYTES));
                value &= !RSRC2_LDS_MASK;
                value |= blocks.min(0x1ff) << RSRC2_LDS_SHIFT;
            }
        } else {
            if user >= 32 {
                value |= RSRC2_USER_SGPR_MSB;
            }
            let (shift, mask) = match options.stage {
                ShaderStage::Vertex if interface.vertex_stage == VertexStage::LsHs => {
                    (RSRC2_HS_LDS_SHIFT, RSRC2_HS_LDS_MASK)
                }
                ShaderStage::Vertex => (RSRC2_GS_LDS_SHIFT, RSRC2_GS_LDS_MASK),
                _ => (RSRC2_PS_EXTRA_LDS_SHIFT, RSRC2_PS_EXTRA_LDS_MASK),
            };
            let guest = (value & mask) >> shift;
            let host = match options.stage {
                ShaderStage::Fragment if self.target.is_gfx11_plus() => guest.div_ceil(2),
                _ => self.lds_blocks(guest),
            };
            value = (value & !mask) | (host.min(0xff) << shift);
        }
        if self.target.is_gfx12() {
            value &= !RSRC2_TRAP_PRESENT;
        }
        value
    }

    /// LDS_SIZE in 512-byte blocks: gfx11 and later allocate LDS in 1 KiB granules, so an odd
    /// count would describe less than the wave receives.
    fn lds_blocks(&self, blocks: u32) -> u32 {
        if self.target.is_gfx11_plus() {
            blocks.next_multiple_of(2)
        } else {
            blocks
        }
    }
}

/// PGM_RSRC1 for a shader whose guest RSRC1 is unknown: [`FLOAT_MODE_DEFAULT`], DX10_CLAMP and
/// MEM_ORDERED at the stage's bit (plus WGP mode for compute). The VGPR and SGPR counts are
/// filled in later.
pub fn default_pgm_rsrc1(stage: ShaderStage) -> u32 {
    let common = (FLOAT_MODE_DEFAULT << RSRC1_FLOAT_MODE_SHIFT) | RSRC1_DX10_CLAMP;
    match stage {
        ShaderStage::Compute => common | RSRC1_MEM_ORDERED | RSRC1_WGP_MODE,
        ShaderStage::Vertex | ShaderStage::Fragment => common | RSRC1_GRAPHICS_MEM_ORDERED,
    }
}

/// VGPR slots of the pixel shader inputs, in SPI_PS_INPUT_ADDR bit order.
const PS_INPUT_VGPRS: [u16; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
/// The ES part of a merged (NGG) shader receives its VGPRs from v5 on: vertex ID, then up to
/// three more (SPI_SHADER_PGM_RSRC2_GS ES_VGPR_COMP_CNT), ending with the instance ID in v8.
const NGG_ES_FIRST_VGPR: u16 = 5;
const RSRC2_GS_ES_VGPR_COMP_CNT_SHIFT: u32 = 16;

/// One more than the highest VGPR the hardware writes before the program starts.
fn input_vgpr_floor(options: CompileOptions, interface: &ShaderInterface, thread_ids: u8) -> u16 {
    match options.stage {
        ShaderStage::Compute if options.target.is_gfx11_plus() => 1,
        ShaderStage::Compute => u16::from(thread_ids) + 1,
        ShaderStage::Fragment => match (interface.ps_input_ena, interface.ps_input_addr) {
            (Some(ena), Some(addr)) => ps_input_vgprs(ena, addr),
            _ => 0,
        },
        // Merged LS-HS: v0-v1 for the hull part, then the vertex ID (v2) and up to three more.
        ShaderStage::Vertex if interface.vertex_stage == VertexStage::LsHs => {
            interface.guest_pgm_rsrc1.map_or(6, |rsrc1| {
                let extra = (rsrc1 >> RSRC1_HS_LS_VGPR_COMP_CNT_SHIFT) & 3;
                3 + extra as u16
            })
        }
        ShaderStage::Vertex => interface.guest_pgm_rsrc2.map_or(0, |rsrc2| {
            let extra = (rsrc2 >> RSRC2_GS_ES_VGPR_COMP_CNT_SHIFT) & 3;
            NGG_ES_FIRST_VGPR + extra as u16 + 1
        }),
    }
}

/// The VGPRs a pixel shader's enabled inputs (`ena`) occupy in the layout `addr` describes.
pub fn ps_input_vgprs(ena: u32, addr: u32) -> u16 {
    let layout = addr | ena;
    let mut next = 0;
    let mut end = 0;
    for (bit, size) in PS_INPUT_VGPRS.iter().enumerate() {
        if layout & (1 << bit) != 0 {
            if ena & (1 << bit) != 0 {
                end = next + size;
            }
            next += size;
        }
    }
    end
}

pub fn compute_thread_id_components(meta: &ShaderMetadata) -> u8 {
    if meta.live_in_vector & 0b100 != 0 {
        2
    } else if meta.live_in_vector & 0b010 != 0 {
        1
    } else {
        0
    }
}
