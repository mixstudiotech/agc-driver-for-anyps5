//! Guest programs and their translations (decisions D4, D20): a program is translated the first
//! time a draw or dispatch uses it, on the recording thread, placed in the runtime's code arena
//! and cached by everything the translation depends on. Only misses take the lock.

use super::arena::Bump;
use super::link::{self, LinkInputs, Linked};
use crate::agc::guest::GuestMemoryReader;
use crate::gpu::state_block::AttributeRing;
use shader::ir::{BufferAccess, DescriptorAccess, DescriptorSource};
use shader::{
    cache::hash_words, AttributeRingAbi, CompileError, CompileOptions, Compiler, Linkage, ResourceKind, ShaderArtifact, ShaderInterface, ShaderStage,
    TargetIsa, VertexStage,
};
use std::collections::BTreeMap;
use std::sync::atomic::{fence, Ordering};
use std::sync::{Arc, Mutex};

/// Guest program dwords read when a program is not registered, before growing the window.
const PROGRAM_WINDOW_DWORDS: usize = 256;
const PROGRAM_LIMIT_DWORDS: usize = 1 << 18;

/// Guest SPI_PS_INPUT_ENA/ADDR, SPI_SHADER_COL_FORMAT/Z_FORMAT and whether MRT0 blends with a
/// second source, at the draw.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PixelInterface {
    pub input_ena: u32,
    pub input_addr: u32,
    pub col_format: u32,
    pub z_format: u32,
    pub dual_source_blend: bool,
}

/// A guest program the host needs translated code for.
#[derive(Clone, Copy, Debug)]
pub struct ProgramRequest {
    pub guest_address: u64,
    pub stage: ShaderStage,
    pub wave_size: u8,
    pub guest_rsrc1: u32,
    pub guest_rsrc2: u32,
    pub pixel: Option<PixelInterface>,
    pub link: LinkInputs,
    pub attribute_ring: Option<AttributeRing>,
    pub vertex_stage: VertexStage,
    pub scratch_bytes_per_lane: u32,
}

/// An image a translated program samples, loads or stores, and where its T# comes from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageUse {
    pub source: DescriptorSource,
    pub dword_count: u8,
    pub written: bool,
}

/// A buffer descriptor or pointer a translated program uses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferUse {
    pub source: DescriptorSource,
    pub access: BufferAccess,
    pub written: bool,
    pub pointer: bool,
}

/// Translated code placed in the code arena, and what the runtime needs to know about it.
#[derive(Debug)]
pub struct Program {
    pub host_address: u64,
    pub rsrc1: u32,
    pub rsrc2: u32,
    /// Parameter exports of a vertex program.
    pub parameters: u32,
    pub scratch_bytes_per_lane: u32,
    /// The SPI_SHADER_COL_FORMAT to program instead of the guest's.
    pub spi_shader_col_format: Option<u32>,
    /// Whether the program sends GS_ALLOC_REQ; `None` when unknown.
    pub gs_alloc_req: Option<bool>,
    /// A vertex program whose translation dropped its parameter exports.
    pub drops_parameters: bool,
    /// The user-data registers (bit `n` for the `n`th) linking read as code addresses.
    pub pointer_user_data: u32,
    pub images: Vec<ImageUse>,
    pub buffers: Vec<BufferUse>,
    /// The program does nothing but fill a buffer with user data.
    pub fill: Option<shader::ir::BufferFill>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CacheKey {
    address: u64,
    hash: u64,
    stage: u8,
    wave_size: u8,
    rsrc1: u32,
    rsrc2: u32,
    pixel: Option<PixelInterface>,
    regions: Vec<(u64, u64)>,
    second_half: Option<u64>,
    hull: bool,
    scratch_bytes_per_lane: u32,
}

/// The fixed addresses every translation is built against.
#[derive(Clone, Copy, Debug)]
pub struct TranslationTargets {
    pub isa: TargetIsa,
    pub descriptor_lut: u64,
    pub gds_buffer: u64,
    /// The texture LOD statistics table pixel shaders update.
    pub mip_stats_table: u64,
    /// The table programs report the images they find at run time through; `None` with
    /// `APS5_AGC_NO_IMAGE_REPORTS`, to measure what reporting costs.
    pub image_reports: Option<u64>,
}

#[derive(Default)]
struct Translator {
    analyses: link::Analyses,
    /// Guest code read before, by address: reused while memory still holds the same words.
    loaded: BTreeMap<u64, Vec<u32>>,
    cache: BTreeMap<CacheKey, Arc<Program>>,
}

/// Every translated program of the process.
pub struct ProgramCache {
    targets: TranslationTargets,
    code: Bump,
    memory: Arc<dyn GuestMemoryReader + Send + Sync>,
    translator: Mutex<Translator>,
}

fn stage_code(stage: ShaderStage) -> u8 {
    match stage {
        ShaderStage::Compute => 0,
        ShaderStage::Vertex => 1,
        ShaderStage::Fragment => 2,
    }
}

impl ProgramCache {
    pub fn new(targets: TranslationTargets, code: Bump, memory: Arc<dyn GuestMemoryReader + Send + Sync>) -> Self {
        Self { targets, code, memory, translator: Mutex::default() }
    }

    fn interface(&self, request: &ProgramRequest) -> ShaderInterface {
        let pixel = request.pixel;
        ShaderInterface {
            guest_pgm_rsrc1: Some(request.guest_rsrc1),
            guest_pgm_rsrc2: Some(request.guest_rsrc2),
            guest_code_address: (request.guest_address != 0).then_some(request.guest_address),
            descriptor_lut: Some(self.targets.descriptor_lut),
            ps_input_ena: pixel.map(|p| p.input_ena),
            ps_input_addr: pixel.map(|p| p.input_addr),
            ps_col_format: pixel.map(|p| p.col_format),
            ps_z_format: pixel.map(|p| p.z_format),
            dual_source_blend: pixel.is_some_and(|p| p.dual_source_blend),
            gds_buffer: Some(self.targets.gds_buffer),
            mip_stats_table: pixel.map(|_| self.targets.mip_stats_table),
            image_reports: self.targets.image_reports,
            attribute_ring: request.attribute_ring.map(|ring| AttributeRingAbi { address: ring.address, bytes: ring.total_bytes() }),
            vertex_stage: request.vertex_stage,
            scratch_bytes_per_lane: request.scratch_bytes_per_lane,
            ..ShaderInterface::default()
        }
    }

    /// Guest program dwords at `address`, to the end of everything the program can reach.
    fn read_program(memory: &dyn GuestMemoryReader, address: u64) -> Result<Vec<u32>, String> {
        if address == 0 {
            return Err("shader program address is null".into());
        }
        let interface = ShaderInterface { guest_code_address: Some(address), ..ShaderInterface::default() };
        let mut window = PROGRAM_WINDOW_DWORDS;
        let mut words: Vec<u32> = Vec::new();
        loop {
            if words.len() < window {
                let at = address + (words.len() * 4) as u64;
                match memory.read_words(at, window - words.len()) {
                    Ok(more) => words.extend(more),
                    Err(error) if words.is_empty() => return Err(error.message),
                    Err(_) => {}
                }
            }
            let extent = Compiler::new(TargetIsa::Gfx1013).analyze(&words, CompileOptions::new(TargetIsa::Gfx1013, ShaderStage::Compute, 64), &interface);
            match extent {
                Ok(ir) => {
                    words.truncate(ir.code_words);
                    return Ok(words);
                }
                Err(CompileError::Truncated { .. } | CompileError::MissingEndProgram | CompileError::InvalidBranch { .. })
                    if words.len() == window && window < PROGRAM_LIMIT_DWORDS =>
                {
                    window *= 2;
                }
                Err(error) => return Err(format!("guest program at 0x{address:x}: {error}")),
            }
        }
    }

    /// Guest code at `address`: the copy read before while memory still holds those words.
    fn load_code(memory: &dyn GuestMemoryReader, loaded: &mut BTreeMap<u64, Vec<u32>>, address: u64) -> Result<Vec<u32>, String> {
        if let Some(words) = loaded.get(&address) {
            if memory.read_words(address, words.len()).ok().as_ref() == Some(words) {
                return Ok(words.clone());
            }
        }
        let words = Self::read_program(memory, address)?;
        loaded.insert(address, words.clone());
        Ok(words)
    }

    /// Loads, links and translates the guest program `request` names.
    pub fn resolve(&self, request: &ProgramRequest) -> Result<Arc<Program>, String> {
        let mut guard = self.translator.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let translator = &mut *guard;
        let words = Self::load_code(&*self.memory, &mut translator.loaded, request.guest_address)?;
        let interface = self.interface(request);
        let program = link::Program {
            address: request.guest_address,
            words: &words,
            hash: None,
            options: CompileOptions::new(self.targets.isa, request.stage, request.wave_size),
            interface: &interface,
        };
        let memory = self.memory.clone();
        let loaded = &mut translator.loaded;
        let linked = link::link(&Compiler::new(self.targets.isa), &program, &request.link, &mut translator.analyses, &mut |address| {
            Self::load_code(&*memory, loaded, address)
        });
        self.place(&mut translator.cache, &words, request, &linked)
    }

    /// Host code for a runtime-owned gfx1013 program of `request`'s stage and interface, translated
    /// as guest programs are (the no-op pixel program of draws without one).
    pub fn resolve_runtime(&self, words: &[u32], request: &ProgramRequest) -> Result<Arc<Program>, String> {
        let mut translator = self.translator.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.place(&mut translator.cache, words, request, &Linked::default())
    }

    /// Host code for a runtime-owned gfx1013 compute kernel.
    pub fn resolve_kernel(&self, words: &[u32], rsrc1: u32, rsrc2: u32, wave_size: u8) -> Result<Arc<Program>, String> {
        let request = ProgramRequest {
            guest_address: 0,
            stage: ShaderStage::Compute,
            wave_size,
            guest_rsrc1: rsrc1,
            guest_rsrc2: rsrc2,
            pixel: None,
            link: LinkInputs::default(),
            attribute_ring: None,
            vertex_stage: VertexStage::EsGs,
            scratch_bytes_per_lane: 0,
        };
        self.resolve_runtime(words, &request)
    }

    fn place(&self, cache: &mut BTreeMap<CacheKey, Arc<Program>>, words: &[u32], request: &ProgramRequest, linked: &Linked) -> Result<Arc<Program>, String> {
        let (regions, second_half) = linked.code.key();
        let key = CacheKey {
            address: request.guest_address,
            hash: hash_words(words),
            stage: stage_code(request.stage),
            wave_size: request.wave_size,
            rsrc1: request.guest_rsrc1,
            rsrc2: request.guest_rsrc2,
            pixel: request.pixel,
            regions,
            second_half,
            hull: request.vertex_stage == VertexStage::LsHs,
            scratch_bytes_per_lane: request.scratch_bytes_per_lane,
        };
        if let Some(program) = cache.get(&key) {
            return Ok(program.clone());
        }
        let interface = self.interface(request);
        let compiler = Compiler::new(self.targets.isa);
        let options = CompileOptions::new(self.targets.isa, request.stage, request.wave_size);
        let code_regions = linked.code.code_regions();
        let linkage = Linkage { regions: &code_regions, second_half: linked.code.second_half };
        let mut used = linked.code.clone();
        let artifact = match compiler.compile_linked(words, options, &interface, &linkage) {
            Err(CompileError::InvalidOptions(_)) if !linked.code.is_empty() => {
                used = link::LinkedCode::default();
                compiler.compile_with_interface(words, options, &interface)
            }
            compiled => compiled,
        }
        .map_err(|error| error.to_string())?;
        if link::coverage(&artifact.external_targets, &request.link, &used) != artifact.external_targets.len() {
            return Err("a shader call/fetch target is not proven by CPU-known user data and linked code".into());
        }
        let host_address = self.code.put(&artifact.code, 256).map_err(|error| error.message)?;
        // The arena is CPU write-combined memory the GPU reads as code once a submission refers
        // to it: drain the write-combining buffers now.
        fence(Ordering::SeqCst);
        if let Some(directory) = std::env::var_os("APS5_AGC_SHADER_DUMP") {
            let directory = std::path::PathBuf::from(directory);
            let dump = || -> std::io::Result<()> {
                std::fs::create_dir_all(&directory)?;
                let stem = directory.join(format!("{host_address:x}"));
                let guest: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                std::fs::write(stem.with_extension("guest.bin"), guest)?;
                std::fs::write(stem.with_extension("host.bin"), artifact.code_bytes())?;
                let metadata = format!("guest=0x{:x}\nhost=0x{host_address:x}\nrsrc1=0x{:x}\nrsrc2=0x{:x}\noptions={:#?}\nlinked_region_count={}\nguest_bin_contains_entry_only=true\nrequest={request:#?}\ninterface={interface:#?}\ndescriptors={:#?}\npointers={:#?}\n",
                    request.guest_address, artifact.pgm_rsrc1, artifact.pgm_rsrc2, artifact.options, used.code_regions().len(), artifact.descriptor_uses, artifact.pointer_uses);
                std::fs::write(stem.with_extension("txt"), metadata)
            };
            if let Err(error) = dump() {
                eprintln!("[shader-map] dump failed for 0x{host_address:x}: {error}");
            }
            eprintln!("[shader-map] host=0x{host_address:x} guest=0x{:x} stage={:?} wave={} code_dwords={} user_sgprs={} rsrc1=0x{:x} rsrc2=0x{:x}",
                request.guest_address, request.stage, request.wave_size, artifact.code.len(), artifact.user_sgpr_count, artifact.pgm_rsrc1, artifact.pgm_rsrc2);
        }
        let program = Arc::new(Program {
            host_address,
            rsrc1: artifact.pgm_rsrc1,
            rsrc2: artifact.pgm_rsrc2,
            parameters: artifact.exports.param_count(),
            scratch_bytes_per_lane: artifact.scratch_bytes_per_lane,
            spi_shader_col_format: artifact.spi_shader_col_format,
            gs_alloc_req: linked.gs_alloc_req,
            drops_parameters: artifact.ngg.as_ref().is_some_and(|ngg| ngg.param_count > 0 && ngg.attribute_ring_relocations.is_none()),
            pointer_user_data: linked
                .pointer_sgprs
                .iter()
                .filter_map(|&sgpr| sgpr.checked_sub(request.link.user_data_sgpr))
                .filter(|&register| register < 32)
                .fold(0u64, |mask, register| mask | 3 << register) as u32,
            images: image_uses(&artifact),
            buffers: buffer_uses(&artifact),
            fill: artifact.buffer_fill,
        });
        cache.insert(key, program.clone());
        Ok(program)
    }
}

/// The image uses of an artifact, each once; `image_get_resinfo` reads only the descriptor.
fn image_uses(artifact: &ShaderArtifact) -> Vec<ImageUse> {
    let mut images: Vec<ImageUse> = Vec::new();
    for used in &artifact.descriptor_uses {
        if used.kind != ResourceKind::ImageDescriptor || used.access == DescriptorAccess::Queried {
            continue;
        }
        let image = ImageUse { source: used.source.clone(), dword_count: used.dword_count, written: used.access == DescriptorAccess::Written };
        if !images.contains(&image) {
            images.push(image);
        }
    }
    images
}

fn buffer_uses(artifact: &ShaderArtifact) -> Vec<BufferUse> {
    artifact
        .descriptor_uses
        .iter()
        .filter(|used| used.kind == ResourceKind::BufferDescriptor)
        .filter_map(|used| Some(BufferUse { source: used.source.clone(), access: used.buffer_access?, written: used.access == DescriptorAccess::Written, pointer: false }))
        .chain(artifact.pointer_uses.iter().filter_map(|used| {
            Some(BufferUse { source: used.source.clone(), access: used.buffer_access?, written: used.access == DescriptorAccess::Written, pointer: true })
        }))
        .collect()
}
