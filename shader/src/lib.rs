#![forbid(unsafe_code)]

pub mod cache;
pub mod cfg;
pub mod decode;
pub mod descriptors;
pub mod encode;
pub mod gds;
pub mod ir;
pub mod isa;
pub mod lowering;
pub mod mip_stats;
pub mod ngg;
pub mod regalloc;

use std::fmt;

pub use cache::{ArtifactCache, CacheKey};
pub use ir::{ExportInfo, InstructionFamily, ResourceBinding, ResourceKind, ShaderIr};

pub const COMPILER_VERSION: u32 = 1;

/// Host instruction set a shader is translated for.
///
/// gfx11.0 and gfx11.5 share every encoding but need different hardware-bug workarounds (see
/// [`encode::hazards`]), so they are separate targets.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum TargetIsa {
    Gfx1013,
    Gfx103,
    /// gfx11.0 (gfx1100-gfx1103).
    Gfx11,
    /// gfx11.5 (gfx1150-gfx1153), e.g. the Radeon 8060S (gfx1151).
    Gfx115,
    Gfx12,
}

impl TargetIsa {
    pub const ALL: [Self; 5] = [
        Self::Gfx1013,
        Self::Gfx103,
        Self::Gfx11,
        Self::Gfx115,
        Self::Gfx12,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Gfx1013 => "gfx1013",
            Self::Gfx103 => "gfx10.3",
            Self::Gfx11 => "gfx11",
            Self::Gfx115 => "gfx11.5",
            Self::Gfx12 => "gfx12",
        }
    }

    pub fn from_gfx_ip(major: u32, minor: u32, stepping: u32) -> Option<Self> {
        match (major, minor, stepping) {
            (10, 1, 3) => Some(Self::Gfx1013),
            (10, 3, _) => Some(Self::Gfx103),
            (11, 5, _) => Some(Self::Gfx115),
            (11, _, _) => Some(Self::Gfx11),
            (12, _, _) => Some(Self::Gfx12),
            _ => None,
        }
    }

    pub fn is_gfx11_plus(self) -> bool {
        matches!(self, Self::Gfx11 | Self::Gfx115 | Self::Gfx12)
    }

    /// The gfx11 encoding family: gfx11.0 and gfx11.5.
    pub fn is_gfx11(self) -> bool {
        matches!(self, Self::Gfx11 | Self::Gfx115)
    }

    pub fn is_gfx12(self) -> bool {
        self == Self::Gfx12
    }

    pub fn instruction_cache_line_bytes(self) -> usize {
        if self.is_gfx11_plus() {
            128
        } else {
            64
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ShaderStage {
    Compute,
    Vertex,
    Fragment,
}

/// What runs the gfx1013 ray intersections (`image_bvh_intersect_ray`,
/// `image_bvh64_intersect_ray`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum RayUnit {
    /// The target's ray unit: it reads the same nodes and descriptor as gfx1013.
    #[default]
    Host,
    /// The gfx1013 reference model, expanded into vector and scalar code that reproduces the
    /// gfx1013 unit bit for bit: what a target's unit is checked against.
    Reference,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CompileOptions {
    pub target: TargetIsa,
    pub stage: ShaderStage,
    pub wave_size: u8,
    pub ray_unit: RayUnit,
}

impl CompileOptions {
    pub fn new(target: TargetIsa, stage: ShaderStage, wave_size: u8) -> Self {
        Self {
            target,
            stage,
            wave_size,
            ray_unit: RayUnit::default(),
        }
    }

    pub fn with_ray_unit(mut self, ray_unit: RayUnit) -> Self {
        self.ray_unit = ray_unit;
        self
    }

    pub fn wave64(self) -> bool {
        self.wave_size == 64
    }

    pub fn validate(self) -> Result<(), CompileError> {
        if self.wave_size != 32 && self.wave_size != 64 {
            return Err(CompileError::InvalidOptions("wave_size must be 32 or 64"));
        }
        Ok(())
    }
}

/// How unfused gfx10 multiply-adds (`v_mad_f32`, `v_mac_f32`, `v_madmk/madak_f32`) translate on
/// targets without them: `Exact` reproduces them (multiply, then add), `Fused` uses one FMA,
/// which rounds once and differs in the last bit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum PrecisionMode {
    Fused,
    #[default]
    Exact,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ShaderInterface {
    pub user_sgpr_count: Option<u8>,
    pub guest_pgm_rsrc1: Option<u32>,
    pub guest_pgm_rsrc2: Option<u32>,
    pub lds_bytes: u32,
    pub scratch_bytes_per_lane: u32,
    /// Guest virtual address of the first code word. Guest memory stays mapped at the same
    /// addresses on the GPU, so translated code hands the program guest PCs (`s_getpc_b64`,
    /// call return addresses) and PC-relative data loads read the original bytes. Required by
    /// `s_getpc_b64`; without it, call and jump targets are numbered from zero.
    pub guest_code_address: Option<u64>,
    /// Pixel shaders: guest SPI_PS_INPUT_ENA and SPI_PS_INPUT_ADDR. The hardware loads the
    /// inputs ENA enables into the VGPR slots ADDR lays out (sizes 2,2,2,3,2,2,2,1 then 1 each),
    /// the same on gfx10 and gfx11.
    pub ps_input_ena: Option<u32>,
    pub ps_input_addr: Option<u32>,
    /// Pixel shaders: guest SPI_SHADER_COL_FORMAT and SPI_SHADER_Z_FORMAT.
    pub ps_col_format: Option<u32>,
    pub ps_z_format: Option<u32>,
    /// Pixel shaders feeding dual-source blending (a separate pipeline variant).
    pub dual_source_blend: bool,
    /// NGG (vertex) shaders: the number of vertex parameters, SPI_VS_OUT_CONFIG
    /// VS_EXPORT_COUNT + 1 (0 when NO_PC_EXPORT is set).
    pub param_exports: Option<u8>,
    /// GPU address of the runtime's copy of [`descriptors::lut`]. With it the translation
    /// converts every guest V#, T# and S# where a gfx11 or gfx11.5 instruction uses it;
    /// without it descriptors must already be in the target's format.
    pub descriptor_lut: Option<u64>,
    /// With [`Self::descriptor_lut`]: the runtime converts the user-SGPR descriptors listed in
    /// [`ShaderArtifact::preconverted_descriptors`] before the dispatch or draw, and the
    /// translation uses them as they are.
    pub preconvert_user_descriptors: bool,
    /// NGG vertex programs on gfx11+: the attribute ring their parameter exports are stored to.
    /// Without it, parameter exports are dropped.
    pub attribute_ring: Option<AttributeRingAbi>,
    /// GPU address of the runtime's GDS buffer ([`gds::BUFFER_BYTES`]): with it, GDS accesses
    /// become global memory accesses to it on every host target ([`gds`]); without it they stay
    /// GDS instructions, which gfx12 rejects.
    pub gds_buffer: Option<u64>,
    /// GPU address of the runtime's texture LOD statistics table ([`mip_stats::TABLE_BYTES`]):
    /// with it, pixel shaders count the samples of textures whose T# asks for it.
    pub mip_stats_table: Option<u64>,
    /// [`ShaderStage::Vertex`] programs: the merged hardware stage they run in.
    pub vertex_stage: VertexStage,
}

/// The merged hardware stage of a [`ShaderStage::Vertex`] program. Both have the same eight
/// system SGPRs before the user data (s8 on); they differ in the rest of their inputs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum VertexStage {
    /// ES and GS in one NGG program: a vertex shader, or a domain shader when the draw
    /// tessellates (VGT_SHADER_STAGES_EN.ES_EN = DS), with or without a geometry shader.
    #[default]
    EsGs,
    /// LS and HS in one program ahead of the tessellator: a vertex shader (its local part)
    /// and a hull shader. s2 holds the off-chip ring offset, s3 the merged wave info, s4 the
    /// tessellation factor ring offset and s5 the scratch wave offset (gfx11: the wave's index
    /// in its threadgroup); v0 the patch ID, v1 the relative patch ID (bits 7:0) and control
    /// point ID (bits 12:8), v2 the vertex ID, v3 the LS thread's index in its threadgroup
    /// (which gfx11 does not load; the translation computes it) and v5 the instance ID.
    LsHs,
}

/// Guest code outside a shader's own words that its translation includes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CodeRegion<'a> {
    /// Guest address of the first word.
    pub address: u64,
    pub words: &'a [u32],
}

/// What [`Compiler::compile_linked`] translates beyond the shader's own words.
///
/// - A merged-stage front half (the program at guest SPI_SHADER_PGM_LO_ES that ends with
///   `s_setpc_b64 s[6:7]` or `s_swappc_b64 null, s[6:7]`, reported as
///   [`ShaderArtifact::user_data_tail_sgpr`] 6) continues into its back half: link the code at
///   guest SPI_SHADER_PGM_LO/HI_GS and set [`Self::second_half`] to its address.
/// - A jump or call to code outside the program ([`ir::ComputedFlow::external`], e.g. a fetch
///   subroutine whose address is in user data) reaches a linked region through the dispatch
///   stub, so the region must hold the address the wave computes at run time.
///
/// Regions must not overlap each other or the shader's words, and neither may cross a 4 GiB
/// boundary. The shader's guest address ([`ShaderInterface::guest_code_address`]) is required.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Linkage<'a> {
    pub regions: &'a [CodeRegion<'a>],
    /// The guest address PS5 hardware gives a merged-stage front half in `s[6:7]`: its back
    /// half's. gfx11 leaves `s[6:7]` undefined, so the translation branches to the linked back
    /// half and sets `s[6:7]` for guest code that reads them. Must be a region's address.
    pub second_half: Option<u64>,
}

/// The NGG attribute ring as the runtime allocates it and programs SPI_ATTRIBUTE_RING_BASE and
/// SPI_ATTRIBUTE_RING_SIZE. Translated programs build the ring descriptor from it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AttributeRingAbi {
    /// GPU address of the ring, 64 KiB aligned.
    pub address: u64,
    /// The ring's size over all shader engines in bytes (the descriptor's NUM_RECORDS).
    pub bytes: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    Unsupported,
    InvalidOptions(&'static str),
    TargetMismatch {
        compiler: TargetIsa,
        requested: TargetIsa,
    },
    EmptyProgram,
    Truncated {
        pc: u32,
    },
    InvalidEncoding {
        pc: u32,
        word: u32,
    },
    UnknownOpcode {
        pc: u32,
        family: InstructionFamily,
        opcode: u16,
    },
    InvalidOperand {
        pc: u32,
        reason: &'static str,
    },
    InvalidBranch {
        pc: u32,
        target: u32,
    },
    MissingEndProgram,
    UnsupportedInstruction {
        pc: u32,
        family: InstructionFamily,
        opcode: u16,
        target: TargetIsa,
        reason: &'static str,
    },
    BranchOutOfRange {
        pc: u32,
    },
    OutOfRegisters {
        vgprs: u16,
    },
}

impl CompileError {
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::Unsupported | Self::UnsupportedInstruction { .. } | Self::UnknownOpcode { .. }
        )
    }

    pub fn pc(&self) -> Option<u32> {
        match *self {
            Self::Truncated { pc }
            | Self::InvalidEncoding { pc, .. }
            | Self::UnknownOpcode { pc, .. }
            | Self::InvalidOperand { pc, .. }
            | Self::InvalidBranch { pc, .. }
            | Self::UnsupportedInstruction { pc, .. }
            | Self::BranchOutOfRange { pc } => Some(pc),
            _ => None,
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("unsupported shader"),
            Self::InvalidOptions(reason) => write!(f, "invalid compile options: {reason}"),
            Self::TargetMismatch {
                compiler,
                requested,
            } => write!(
                f,
                "compiler targets {} but {} was requested",
                compiler.name(),
                requested.name()
            ),
            Self::EmptyProgram => f.write_str("shader code is empty"),
            Self::Truncated { pc } => write!(f, "instruction at {pc:#x} is truncated"),
            Self::InvalidEncoding { pc, word } => {
                write!(f, "invalid encoding {word:#010x} at {pc:#x}")
            }
            Self::UnknownOpcode { pc, family, opcode } => {
                write!(f, "unknown {} opcode {opcode:#x} at {pc:#x}", family.name())
            }
            Self::InvalidOperand { pc, reason } => {
                write!(f, "invalid operand at {pc:#x}: {reason}")
            }
            Self::InvalidBranch { pc, target } => {
                write!(f, "branch at {pc:#x} targets invalid address {target:#x}")
            }
            Self::MissingEndProgram => f.write_str("code ends before a reachable s_endpgm"),
            Self::UnsupportedInstruction {
                pc,
                family,
                opcode,
                target,
                reason,
            } => write!(
                f,
                "{} at {pc:#x} is unsupported on {}: {reason}",
                ir::opcodes::mnemonic(*family, *opcode),
                target.name()
            ),
            Self::BranchOutOfRange { pc } => {
                write!(f, "branch at {pc:#x} is out of range after lowering")
            }
            Self::OutOfRegisters { vgprs } => {
                write!(
                    f,
                    "lowering needs {vgprs} VGPRs, exceeding the register file"
                )
            }
        }
    }
}

impl std::error::Error for CompileError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderArtifact {
    pub code: Vec<u32>,
    pub options: CompileOptions,
    pub precision: PrecisionMode,
    pub pgm_rsrc1: u32,
    pub pgm_rsrc2: u32,
    pub num_sgprs: u16,
    pub num_vgprs: u16,
    pub lds_bytes: u32,
    pub scratch_bytes_per_lane: u32,
    pub user_sgpr_count: u8,
    pub program_words: usize,
    pub guest_words: usize,
    pub guest_hash: u64,
    pub bindings: Vec<ResourceBinding>,
    pub exports: ExportInfo,
    pub user_data_tail_sgpr: Option<u8>,
    pub uses_pc_relative: bool,
    /// With [`ShaderInterface::preconvert_user_descriptors`]: the user-SGPR descriptors the
    /// program uses unconverted. The runtime writes their gfx11 form
    /// ([`descriptors::convert_vsharp`] and siblings) before the dispatch or draw.
    pub preconverted_descriptors: Vec<ResourceBinding>,
    /// gfx11+ pixel shaders: the SPI_SHADER_COL_FORMAT to program instead of the guest's. Set
    /// when a guest null export became an MRT0 export without data while MRT0 has no export
    /// memory (format ZERO, and no depth export memory to take it instead): MRT0 becomes 32_R
    /// so EXEC still kills pixels. `None` keeps the guest's value.
    pub spi_shader_col_format: Option<u32>,
    /// gfx11+ NGG vertex programs: what the runtime programs for the translation.
    pub ngg: Option<ngg::NggState>,
    /// Jumps and calls to code outside the program and the linked regions: at run time they
    /// end the wave unless their target lies in a region linked into the translation (see
    /// [`Linkage`]). [`ir::ExternalTarget::sgpr`] names the input SGPRs holding the target.
    pub external_targets: Vec<ir::ExternalTarget>,
    /// Every descriptor use with its access and where the guest descriptor came from (user
    /// SGPRs, a chain of table loads from user data, or dynamic): the guest surfaces and
    /// buffers a draw or dispatch reads and writes.
    pub descriptor_uses: Vec<ir::DescriptorUse>,
    pub pointer_uses: Vec<ir::DescriptorUse>,
}

impl ShaderArtifact {
    pub fn code_bytes(&self) -> Vec<u8> {
        self.code.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    pub fn program(&self) -> &[u32] {
        &self.code[..self.program_words]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Compiler {
    target: TargetIsa,
    precision: PrecisionMode,
}

impl Compiler {
    pub fn new(target: TargetIsa) -> Self {
        Self {
            target,
            precision: PrecisionMode::default(),
        }
    }

    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision = precision;
        self
    }

    pub fn target(&self) -> TargetIsa {
        self.target
    }

    pub fn precision(&self) -> PrecisionMode {
        self.precision
    }

    pub fn compile(
        &self,
        words: &[u32],
        options: CompileOptions,
    ) -> Result<ShaderArtifact, CompileError> {
        self.compile_with_interface(words, options, &ShaderInterface::default())
    }

    pub fn analyze(
        &self,
        words: &[u32],
        options: CompileOptions,
        interface: &ShaderInterface,
    ) -> Result<ShaderIr, CompileError> {
        self.analyze_linked(words, options, interface, &Linkage::default())
    }

    /// [`Self::analyze`] with linked code regions; instruction pcs past the shader's own
    /// words address the regions in order (see [`ir::CodeLayout`]).
    pub fn analyze_linked(
        &self,
        words: &[u32],
        options: CompileOptions,
        interface: &ShaderInterface,
        linkage: &Linkage<'_>,
    ) -> Result<ShaderIr, CompileError> {
        self.check(options)?;
        let (words, layout) = link_layout(words, interface, linkage)?;
        let mut ir = decode_program(&words, &layout)?;
        regalloc::analyze(&mut ir, options, interface);
        Ok(ir)
    }

    pub fn compile_with_interface(
        &self,
        words: &[u32],
        options: CompileOptions,
        interface: &ShaderInterface,
    ) -> Result<ShaderArtifact, CompileError> {
        self.compile_linked(words, options, interface, &Linkage::default())
    }

    /// Translates `words` together with the guest code `linkage` names, as one program: a
    /// merged-stage front half continues into its back half, and calls into a linked fetch
    /// subroutine run its translation instead of ending the wave.
    pub fn compile_linked(
        &self,
        words: &[u32],
        options: CompileOptions,
        interface: &ShaderInterface,
        linkage: &Linkage<'_>,
    ) -> Result<ShaderArtifact, CompileError> {
        let ir = self.analyze_linked(words, options, interface, linkage)?;
        let mut registers = regalloc::RegisterAllocator::new(&ir.metadata, options, interface);
        let user_sgprs_known =
            interface.user_sgpr_count.is_some() || interface.guest_pgm_rsrc2.is_some();
        let program = lowering::Lowerer::new(options, self.precision)
            .with_guest_code_address(interface.guest_code_address)
            .with_descriptor_conversion(
                interface.descriptor_lut,
                interface.preconvert_user_descriptors,
            )
            .with_pixel_interface(
                user_sgprs_known.then_some(ir.metadata.user_sgpr_count),
                interface.ps_col_format,
                interface.ps_z_format,
                interface.dual_source_blend,
            )
            .with_attribute_ring(interface.attribute_ring)
            .with_gds_buffer(interface.gds_buffer)
            .with_mip_stats_table(interface.mip_stats_table)
            .with_vertex_stage(interface.vertex_stage)
            .lower(&ir, &mut registers)?;
        let null_exports = ir
            .instructions
            .iter()
            .any(|i| matches!(i.fields, ir::Fields::Exp(e) if e.target == ir::EXP_TARGET_NULL));
        let spi_shader_col_format = match options.stage {
            ShaderStage::Fragment if options.target.is_gfx11_plus() => {
                lowering::pixel::col_format_for(
                    interface.ps_col_format,
                    interface.ps_z_format,
                    null_exports,
                )
            }
            _ => None,
        };
        let encoded = encode::Encoder::new(options).encode(&program)?;
        let ngg = match options.stage {
            ShaderStage::Vertex
                if options.target.is_gfx11_plus()
                    && interface.vertex_stage == VertexStage::EsGs =>
            {
                let param_count = ir.metadata.exports.param_count();
                let program_code = &encoded.code[..encoded.program_words];
                let relocations = match program.attribute_ring {
                    Some(ring) => Some(ring_relocations(&program, program_code, ring)?),
                    None => None,
                };
                Some(ngg::NggState {
                    param_count,
                    spi_vs_out_config: ngg::spi_vs_out_config(param_count),
                    attribute_ring_relocations: relocations,
                    registers: ngg::PROGRAM_REGISTERS,
                })
            }
            _ => None,
        };
        let config = registers.finish(&ir.metadata, options, interface);
        Ok(ShaderArtifact {
            code: encoded.code,
            options,
            precision: self.precision,
            pgm_rsrc1: config.pgm_rsrc1,
            pgm_rsrc2: config.pgm_rsrc2,
            num_sgprs: config.num_sgprs,
            num_vgprs: config.num_vgprs,
            lds_bytes: config.lds_bytes,
            scratch_bytes_per_lane: config.scratch_bytes_per_lane,
            user_sgpr_count: config.user_sgpr_count,
            program_words: encoded.program_words,
            guest_words: ir.code_words.min(words.len()),
            guest_hash: cache::hash_words(&words[..ir.code_words.min(words.len())]),
            bindings: ir.metadata.bindings.clone(),
            exports: ir.metadata.exports,
            user_data_tail_sgpr: ir.metadata.user_data_tail_sgpr,
            uses_pc_relative: ir.metadata.uses_pc_relative,
            preconverted_descriptors: program.preconverted,
            spi_shader_col_format,
            ngg,
            external_targets: ir.flow.external.clone(),
            descriptor_uses: ir.metadata.descriptor_uses.clone(),
            pointer_uses: ir.metadata.pointer_uses.clone(),
        })
    }

    fn check(&self, options: CompileOptions) -> Result<(), CompileError> {
        options.validate()?;
        if options.target != self.target {
            return Err(CompileError::TargetMismatch {
                compiler: self.target,
                requested: options.target,
            });
        }
        Ok(())
    }
}

/// Dword offsets in the encoded program of the literals the prologue moves into words 0 and 1
/// of the attribute ring descriptor.
fn ring_relocations(
    program: &lowering::LoweredProgram,
    code: &[u32],
    (sgpr, words): (Option<u8>, [u32; 4]),
) -> Result<[u32; 2], CompileError> {
    let mut offsets = [0; 2];
    for (k, offset) in offsets.iter_mut().enumerate() {
        // The prologue initializes the descriptor once, into reserved SGPRs or persistent
        // spill lanes. Every store reads that same descriptor, so one relocation pair suffices.
        let wanted = |fields: &ir::Fields| match (fields, sgpr) {
            (ir::Fields::Sop1 { sdst, src0 }, Some(sgpr)) => {
                *sdst == ir::Operand::Sgpr(sgpr + k as u8)
                    && *src0 == ir::Operand::Literal(words[k])
            }
            (ir::Fields::Sop1 { src0, .. }, None) => *src0 == ir::Operand::Literal(words[k]),
            _ => false,
        };
        let found = program
            .instructions()
            .find(|i| wanted(&i.fields))
            .ok_or(CompileError::Unsupported)?;
        let encoded = encode::encode_instruction(found, program.target)?;
        let at = code
            .windows(encoded.len())
            .position(|w| w == encoded.as_slice())
            .ok_or(CompileError::Unsupported)?;
        *offset = (at + encoded.len() - 1) as u32;
    }
    Ok(offsets)
}

/// The shader's own words followed by every linked region, and where they lie in guest memory.
fn link_layout(
    words: &[u32],
    interface: &ShaderInterface,
    linkage: &Linkage<'_>,
) -> Result<(Vec<u32>, ir::CodeLayout), CompileError> {
    let own_bytes = 4 * words.len() as u64;
    let mut layout = ir::CodeLayout {
        base: interface.guest_code_address,
        own_bytes: own_bytes as u32,
        regions: Vec::new(),
        second_half: linkage.second_half,
    };
    if linkage.regions.is_empty() && linkage.second_half.is_none() {
        return Ok((words.to_vec(), layout));
    }
    let base = interface
        .guest_code_address
        .ok_or(CompileError::InvalidOptions(
            "linked code needs ShaderInterface::guest_code_address",
        ))?;
    let mut spans = vec![(base, own_bytes)];
    let mut linked = words.to_vec();
    for region in linkage.regions {
        let bytes = 4 * region.words.len() as u64;
        let end = region.address.checked_add(bytes);
        if region.words.is_empty() || region.address % 4 != 0 || end.is_none() {
            return Err(CompileError::InvalidOptions(
                "a linked region must be non-empty, dword aligned words",
            ));
        }
        layout.regions.push(ir::LinkedRegion {
            address: region.address,
            start: 4 * linked.len() as u32,
            bytes: bytes as u32,
        });
        linked.extend_from_slice(region.words);
        spans.push((region.address, bytes));
    }
    for (k, &(address, bytes)) in spans.iter().enumerate() {
        if address >> 32 != (address + bytes - 1) >> 32 {
            return Err(CompileError::InvalidOptions(
                "linked code must not cross a 4 GiB boundary",
            ));
        }
        let overlaps = spans[..k]
            .iter()
            .any(|&(other, len)| address < other + len && other < address + bytes);
        if overlaps {
            return Err(CompileError::InvalidOptions("linked code regions overlap"));
        }
    }
    let half_linked = linkage
        .second_half
        .is_none_or(|h| layout.regions.iter().any(|r| r.address == h));
    if !half_linked {
        return Err(CompileError::InvalidOptions(
            "Linkage::second_half must be the address of a linked region",
        ));
    }
    Ok((linked, layout))
}

/// Decodes a program with its blocks and computed control flow. Computed jumps can reveal
/// more code (callees after `s_endpgm`, jump-table cases after `s_setpc_b64`), so decoding,
/// block building and jump classification repeat until the classification settles. Linked
/// regions are decoded from their first words on.
fn decode_program(words: &[u32], layout: &ir::CodeLayout) -> Result<ShaderIr, CompileError> {
    const ROUNDS: usize = 8;
    let decoder = decode::Decoder::new();
    let mut flow = ir::ComputedFlow::default();
    let mut last_jump = 0;
    for _ in 0..ROUNDS {
        let mut scope = cfg::computed::decode_scope(&flow);
        scope.entries.extend(layout.regions.iter().map(|r| r.start));
        let mut ir = decoder.decode_scoped(words, &scope)?;
        ir.layout = layout.clone();
        cfg::computed::apply(&mut ir, &flow)?;
        cfg::ControlFlowBuilder::new().build(&mut ir)?;
        let next = cfg::computed::analyze(&ir, layout.base, words.len());
        if next == flow {
            return Ok(ir);
        }
        last_jump = next.jumps.keys().next_back().copied().unwrap_or(0);
        flow = next;
    }
    Err(CompileError::InvalidOperand {
        pc: last_jump,
        reason: "computed jump targets do not settle",
    })
}
