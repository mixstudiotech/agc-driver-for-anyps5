pub(crate) mod bvh;
pub mod computed;
mod descriptors;
mod gds;
mod interp;
mod image_atomic;
mod image_addresses;
mod dot2;
mod mip_stats;
pub(crate) mod ngg;
pub(crate) mod pixel;
mod sdwa;
mod spill;
mod vector;
pub mod waitcnt;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::collections::HashMap;

use crate::ir::memops;
use crate::ir::opcodes::{self, flag, Rule, Xlat};
use crate::ir::{
    CodeLayout, ComputedFlow, Dpp16, Export, Fields, FlatSegment, Instruction,
    InstructionFamily as F, MemKind, Operand, ResourceBinding, ResourceKind, ShaderIr, Vop3,
    VopExt,
};
use crate::regalloc::RegisterAllocator;
use crate::{CompileError, CompileOptions, PrecisionMode, ShaderStage, TargetIsa};
pub use computed::{DISPATCH_LABEL, DISPATCH_SGPRS, MISS_LABEL};
/// SGPRs an expansion takes at most as transient registers (a T# conversion's temporaries).
const SPILL_HEADROOM_SGPRS: u16 = 8;
/// SGPRs the translation's other reservations take at most (an NGG attribute ring's descriptor
/// and offset, a saved wave index).
const RESERVED_SGPR_ESTIMATE: u16 = 8;
use descriptors::{DescriptorCache, DescriptorSlots};
use ngg::{Kept, NggPlan};
pub use ngg::{EPILOGUE_END_LABEL, EPILOGUE_LABEL};
use pixel::{PixelInterface, PixelPlan, EXP_DUAL_SOURCE, EXP_MRT0, EXP_MRT1};
use spill::SpillLanes;
/// Label of the block of code the translation runs before guest pc 0.
pub const PROLOGUE_LABEL: u32 = 0xffff_ffe0;
/// Labels inside a block start here, below [`PROLOGUE_LABEL`]: an instruction whose `pc` is one
/// is that label's position.
pub const LOCAL_LABEL_BASE: u32 = 0xf000_0000;
use waitcnt::WaitTracker;

pub const SOPP_GFX11_WAITCNT: u16 = 0x09;
pub const SOPP_GFX12_BARRIER_WAIT: u16 = 0x14;
pub const SOP1_GFX12_BARRIER_SIGNAL: u16 = 0x4e;
pub const SOP1_GFX11_SENDMSG_RTN_B64: u16 = 0x4d;
pub const MSG_RTN_GET_REALTIME: u16 = 0x83;
const MSG_INTERRUPT: u16 = 1;
const MSG_EARLY_PRIM_DEALLOC: u16 = 8;
const MSG_GS_ALLOC_REQ: u16 = 9;
const HWREG_ID_MASK: u16 = 0x3f;
/// Hardware registers with the same layout on the target as on gfx10. GPR_ALLOC (5) changed on
/// gfx11.
const PORTABLE_HWREGS: [u16; 6] = [1, 2, 5, 6, 23, 24];
const HWREG_GPR_ALLOC: u16 = 5;
const EXP_NULL: u8 = crate::ir::EXP_TARGET_NULL;
const DS_ORDERED_COUNT: u16 = 0x3f;
const EXP_PARAM_FIRST: u8 = 32;
const PACKED_TID_BITS: u32 = 10;
const SOPP_DENORM_MODE: u16 = 0x25;
const SOPK_GETREG: u16 = 0x12;
const SOPK_SETREG: u16 = 0x13;
const SOPK_SETREG_IMM32: u16 = 0x15;
const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP2_ADD_U32: u16 = 0x00;
const SOP2_CSELECT_B32: u16 = 0x0a;
const SOP2_AND_B32: u16 = 0x0e;
const SOP2_OR_B32: u16 = 0x10;
const SOP2_LSHR_B64: u16 = 0x21;
const SOP2_MUL_I32: u16 = 0x26;
const SOP2_BFE_U32: u16 = 0x27;
const SOP2_MUL_HI_U32: u16 = 0x35;
const VOP3_MBCNT_LO: u16 = 0x365;
const VOP3_MBCNT_HI: u16 = 0x366;
const SMEM_MEMTIME: u16 = 0x24;
/// PS5 shader clock ticks per 32 realtime (100 MHz) ticks: 2.23 GHz / 100 MHz = 22.3 = 715/32.
const SHADER_CLOCKS_PER_32_REALTIME: u32 = 715;
/// `s_bfe_u32` control for TG_SIZE bits 24:20, the wave's index in its workgroup.
const TG_SIZE_WAVE_ID: u32 = 20 | (5 << 16);
const HWREG_FLAT_SCR_LO: u16 = 20;
const HWREG_FLAT_SCR_HI: u16 = 21;
const SOPK_GETREG_B32: u16 = 0x12;
/// gfx11's `s_getreg_b32`, for hardware registers the guest's has no counterpart of.
const GFX11_SOPK_GETREG_B32: u16 = 0x11;
const SOPC_CMP_EQ_U32: u16 = 0x06;
const SOPC_CMP_LG_U32: u16 = 0x07;
const SOPC_CMP_EQ_U64: u16 = 0x12;
/// FLOAT_MODE bits 5:4: f32 input and output denormals.
const F32_DENORM_MODE: u8 = 0x30;
/// `hwreg(HW_REG_MODE, 4, 2)`: the f32 denormal bits.
const HWREG_MODE_F32_DENORM: u16 = 1 | (4 << 6) | (1 << 11);

/// How [`Ctx::enter_f32_flush`] changed the FP mode, for [`Ctx::leave_f32_flush`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum ModeRestore {
    Unchanged,
    /// FP_DENORM (MODE bits 7:4) to put back with `s_denorm_mode`.
    Denorm(u16),
    /// SGPR holding the saved f32 denormal bits.
    Saved(u8),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LoweredBlock {
    /// The guest pc the block starts at, or [`PROLOGUE_LABEL`]/[`DISPATCH_LABEL`]/[`MISS_LABEL`].
    pub guest_pc: u32,
    pub instructions: Vec<Instruction>,
    /// Computed jumps may enter the block through the dispatch stub.
    pub dynamic_entry: bool,
}

/// The guest-to-host table behind the dispatch stub (see [`computed`]).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DispatchTable {
    /// Guest code words the table maps; entry `i` maps guest pc `4 * i`, entry `words` misses.
    pub words: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoweredProgram {
    pub target: TargetIsa,
    pub stage: ShaderStage,
    pub wave64: bool,
    pub blocks: Vec<LoweredBlock>,
    pub dispatch: Option<DispatchTable>,
    /// User-SGPR descriptors the translation uses unconverted: the runtime converts them.
    pub preconverted: Vec<ResourceBinding>,
    /// NGG with parameter stores: the first SGPR of the attribute ring descriptor the
    /// prologue builds (none where it keeps the descriptor in persistent spill lanes), and its four
    /// words.
    pub attribute_ring: Option<(Option<u8>, [u32; 4])>,
}

impl LoweredProgram {
    pub fn instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.blocks.iter().flat_map(|b| b.instructions.iter())
    }

    /// Control-flow successors of each block, by index; the dispatch stub leads to every
    /// dynamic entry and the miss stub.
    pub fn successors(&self) -> Vec<Vec<usize>> {
        let index: HashMap<u32, usize> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.guest_pc, i))
            .collect();
        let count = self.blocks.len();
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, block)| {
                let mut next = Vec::new();
                if block.guest_pc == DISPATCH_LABEL {
                    next.extend(
                        self.blocks
                            .iter()
                            .enumerate()
                            .filter(|(_, b)| b.dynamic_entry || b.guest_pc == MISS_LABEL)
                            .map(|(id, _)| id),
                    );
                    return next;
                }
                let falls = match block.instructions.last() {
                    Some(last) => {
                        if let Some(&target) = last.branch_target.and_then(|t| index.get(&t)) {
                            next.push(target);
                        }
                        last.falls_through()
                    }
                    None => true,
                };
                if falls && i + 1 < count && !next.contains(&(i + 1)) {
                    next.push(i + 1);
                }
                next
            })
            .collect()
    }
}

pub(crate) fn unsupported(
    inst: &Instruction,
    target: TargetIsa,
    reason: &'static str,
) -> CompileError {
    CompileError::UnsupportedInstruction {
        pc: inst.pc,
        family: inst.family,
        opcode: inst.opcode,
        target,
        reason,
    }
}

pub(crate) fn derive(
    origin: &Instruction,
    family: F,
    opcode: u16,
    flags: u64,
    fields: Fields,
) -> Instruction {
    Instruction {
        pc: origin.pc,
        family,
        opcode,
        flags,
        fields,
        branch_target: None,
        words: Vec::new(),
    }
}

pub(crate) fn retarget(inst: &Instruction, opcode: u16) -> Instruction {
    Instruction {
        opcode,
        words: Vec::new(),
        ..inst.clone()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Lowerer {
    options: CompileOptions,
    precision: PrecisionMode,
    guest_code_address: Option<u64>,
    descriptor_lut: Option<u64>,
    preconvert_user_descriptors: bool,
    pixel: PixelInterface,
    attribute_ring: Option<crate::AttributeRingAbi>,
    gds_buffer: Option<u64>,
    mip_stats_table: Option<u64>,
    /// A merged LS-HS program ([`crate::VertexStage::LsHs`]).
    hull: bool,
}

/// What the blocks of one program share.
#[derive(Clone, Copy)]
struct Shared<'a> {
    dispatch: Option<u8>,
    /// The VGPR whose lanes hold spilled SGPRs ([`spill`]).
    spill_vgpr: Option<u8>,
    descriptors: Option<DescriptorSlots>,
    preconverted: &'a [ResourceBinding],
    pixel: Option<&'a PixelPlan>,
    ngg: Option<NggPlan>,
    wave_index: Option<Kept>,
    layout: &'a CodeLayout,
    private_buffer: bool,
}

pub(crate) struct Ctx<'r> {
    pub target: TargetIsa,
    pub stage: ShaderStage,
    pub wave64: bool,
    pub precision: PrecisionMode,
    pub regs: &'r mut RegisterAllocator,
    pub out: Vec<Instruction>,
    pub waits: WaitTracker,
    pub mode: bvh::ModeState,
    /// Guest address of the first code word (see [`crate::ShaderInterface`]).
    pub guest_base: Option<u64>,
    /// Where the program's words, linked regions included, lie in guest memory.
    pub layout: &'r CodeLayout,
    pub flow: &'r ComputedFlow,
    /// First of the [`DISPATCH_SGPRS`] SGPRs, when the program has dynamic jumps.
    pub dispatch: Option<u8>,
    /// gfx11+ compute: the SGPR gfx10 loads with the scratch wave offset, when the program
    /// reads it.
    pub scratch_wave_offset: Option<u8>,
    /// Converted-descriptor slots, when guest descriptors are converted where used.
    pub descriptors: Option<DescriptorSlots>,
    pub descriptor_cache: DescriptorCache,
    /// User-SGPR descriptors the runtime converts before the dispatch or draw.
    pub preconverted: &'r [ResourceBinding],
    /// gfx11+ pixel shaders.
    pub pixel: Option<&'r PixelPlan>,
    /// Dual-source blending: the first of the block's MRT0/MRT1 exports, held until the second.
    pub pending_dual_source: Option<Export>,
    /// gfx11+ NGG vertex programs.
    pub ngg: Option<NggPlan>,
    /// `s_endpgm` branches to the epilogue ([`EPILOGUE_LABEL`]), which ends the wave.
    pub epilogue: bool,
    /// gfx11+ compute: the SGPR holding the wave index the prologue extracted from TG_SIZE.
    pub wave_index: Option<Kept>,
    /// The runtime's GDS buffer, which GDS accesses become global accesses to ([`gds`]).
    pub gds: Option<u64>,
    /// The texture LOD statistics table pixel shaders count samples in ([`mip_stats`]).
    pub mip_stats: Option<u64>,
    /// gfx11 programs with indexed private-segment accesses: every private access goes
    /// through the wave's scratch with a buffer instruction
    /// ([`Ctx::private_buffer_access`]) instead of flat scratch.
    pub private_buffer: bool,
    /// SGPR spilling to VGPR lanes, for programs that leave too few SGPRs above their own.
    pub spill: Option<SpillLanes>,
}

impl Lowerer {
    pub fn new(options: CompileOptions, precision: PrecisionMode) -> Self {
        Self {
            options,
            precision,
            guest_code_address: None,
            descriptor_lut: None,
            preconvert_user_descriptors: false,
            pixel: PixelInterface::default(),
            attribute_ring: None,
            gds_buffer: None,
            mip_stats_table: None,
            hull: false,
        }
    }

    /// The program is a merged LS-HS (hull) program rather than an NGG one
    /// ([`crate::ShaderInterface::vertex_stage`]).
    pub fn with_vertex_stage(mut self, stage: crate::VertexStage) -> Self {
        self.hull = stage == crate::VertexStage::LsHs;
        self
    }

    /// GDS accesses become global accesses to the runtime's GDS buffer at `address`
    /// ([`crate::ShaderInterface::gds_buffer`]).
    pub fn with_gds_buffer(mut self, address: Option<u64>) -> Self {
        self.gds_buffer = address;
        self
    }

    /// Pixel shaders count the samples of textures whose T# asks for it in the table at `address`
    /// ([`crate::ShaderInterface::mip_stats_table`]).
    pub fn with_mip_stats_table(mut self, address: Option<u64>) -> Self {
        self.mip_stats_table = address;
        self
    }

    /// NGG vertex programs: the attribute ring parameter exports are stored to
    /// ([`crate::ShaderInterface::attribute_ring`]).
    pub fn with_attribute_ring(mut self, ring: Option<crate::AttributeRingAbi>) -> Self {
        self.attribute_ring = ring;
        self
    }

    /// Pixel shaders: the `prim_mask` SGPR when the user SGPR count is known (not guessed),
    /// SPI_SHADER_COL_FORMAT/Z_FORMAT, and whether the pipeline blends with two sources.
    pub fn with_pixel_interface(
        mut self,
        prim_mask: Option<u8>,
        col_format: Option<u32>,
        z_format: Option<u32>,
        dual_source: bool,
    ) -> Self {
        self.pixel = PixelInterface {
            prim_mask,
            col_format,
            z_format,
            dual_source,
        };
        self
    }

    /// The guest address of the program's first code word; PC values the translated program
    /// hands out are guest PCs relative to it.
    pub fn with_guest_code_address(mut self, address: Option<u64>) -> Self {
        self.guest_code_address = address;
        self
    }

    /// Converts guest descriptors where instructions use them, reading the table at `lut`
    /// ([`crate::ShaderInterface::descriptor_lut`]); with `preconvert`, user-SGPR descriptors
    /// used only as descriptors are left to the runtime.
    pub fn with_descriptor_conversion(mut self, lut: Option<u64>, preconvert: bool) -> Self {
        self.descriptor_lut = lut;
        self.preconvert_user_descriptors = preconvert;
        self
    }

    pub fn lower(
        &self,
        ir: &ShaderIr,
        regs: &mut RegisterAllocator,
    ) -> Result<LoweredProgram, CompileError> {
        let target = self.options.target;
        self.reject_streamout(ir)?;
        let mode = if ir.instructions.iter().any(bvh::writes_mode) {
            bvh::ModeState::Dynamic
        } else {
            bvh::ModeState::Static(regs.initial_float_mode())
        };
        // A program whose SGPRs leave too little room above them for the descriptor slots, the
        // translation's other reservations and an expansion's transient SGPRs spills SGPRs to
        // the lanes of a reserved VGPR instead (as ACO does); descriptors are then converted in
        // spilled blocks where used.
        let mut spill_vgpr = None;
        if target.is_gfx11_plus()
            && regs.free_sgprs()
                < self.descriptor_slot_sgprs(ir) + RESERVED_SGPR_ESTIMATE + SPILL_HEADROOM_SGPRS
        {
            spill_vgpr = regs.reserve_program_vgprs(1).ok();
        }
        let descriptors = self.reserve_descriptor_slots(ir, regs, spill_vgpr.is_some())?;
        let preconverted = match descriptors {
            Some(_) if self.preconvert_user_descriptors => {
                descriptors::pure_user_descriptors(ir, self.options.wave64())
            }
            _ => Vec::new(),
        };
        let pixel = match self.options.stage {
            ShaderStage::Fragment if target.is_gfx11_plus() => {
                Some(PixelPlan::new(ir, regs, self.options.wave64(), self.pixel)?)
            }
            _ => None,
        };
        let ngg = match self.options.stage {
            ShaderStage::Vertex if self.hull => {
                if target.is_gfx12() {
                    return Err(match ir.instructions.first() {
                        Some(first) => unsupported(
                            first,
                            target,
                            "merged LS-HS (hull) programs are not supported on gfx12",
                        ),
                        None => CompileError::Unsupported,
                    });
                }
                None
            }
            ShaderStage::Vertex if target.is_gfx11_plus() => Some(NggPlan::new(
                ir,
                regs,
                self.options.wave64(),
                self.attribute_ring,
                spill_vgpr.is_some(),
            )?),
            _ => None,
        };
        let wave_index = match self.options.stage {
            ShaderStage::Compute if target.is_gfx11_plus() && ir.metadata.uses_wave_id => {
                Some(match regs.reserve_sgprs(1) {
                    Some(sgpr) => Kept::Sgpr(sgpr),
                    None if spill_vgpr.is_some() => {
                        Kept::Lane(if self.options.wave64() { 63 } else { 31 })
                    }
                    None => {
                        return Err(match ir.instructions.first() {
                            Some(first) => unsupported(
                                first,
                                target,
                                "the wave index needs an SGPR or a spill VGPR",
                            ),
                            None => CompileError::Unsupported,
                        })
                    }
                })
            }
            _ => None,
        };
        // The reservations took more than estimated: spill too, before any transient SGPR.
        if spill_vgpr.is_none()
            && target.is_gfx11_plus()
            && regs.free_sgprs() < SPILL_HEADROOM_SGPRS
        {
            spill_vgpr = regs.reserve_program_vgprs(1).ok();
        }
        let dispatch = match ir.flow.has_dynamic_jumps() {
            true => regs.scratch_sgprs(DISPATCH_SGPRS),
            false => None,
        };
        let layout = CodeLayout {
            base: ir.layout.base.or(self.guest_code_address),
            own_bytes: if ir.layout.own_bytes == 0 {
                4 * ir.code_words as u32
            } else {
                ir.layout.own_bytes
            },
            ..ir.layout.clone()
        };
        let shared = Shared {
            dispatch,
            spill_vgpr,
            descriptors,
            preconverted: &preconverted,
            pixel: pixel.as_ref(),
            ngg,
            wave_index,
            layout: &layout,
            private_buffer: indexes_private_segment(ir, regs, target),
        };
        let static_reach = statically_reachable(ir);
        let mut prologue = self.context(regs, WaitTracker::new(true), mode, ir, shared);
        self.prologue(ir, &mut prologue)?;
        let prologue = prologue.out;
        let mut blocks: Vec<LoweredBlock> = Vec::new();
        let mut lowered: Vec<usize> = Vec::new();
        for block in ir.blocks.iter().filter(|b| b.reachable) {
            let entry = block.id == 0 && block.predecessors.is_empty();
            let mut ctx = self.context(regs, WaitTracker::new(entry), mode, ir, shared);
            let result = ir.instructions[block.instructions()]
                .iter()
                .try_for_each(|inst| ctx.lower_guest(&inline_scalar_literals(inst)))
                .and_then(|()| match ctx.pending_dual_source {
                    Some(_) => Err(unsupported(
                        &ir.instructions[block.end - 1],
                        target,
                        "dual-source blending needs MRT0 and MRT1 exported in one block",
                    )),
                    None => Ok(()),
                });
            if !static_reach[block.id] {
                // Only an open computed jump reaches the block, which may be data decoded as
                // code: keep it only if it translates, so its table entries miss otherwise.
                let encodable = ctx
                    .out
                    .iter()
                    .all(|i| crate::encode::encode_instruction(i, target).is_ok());
                if result.is_err() || !encodable {
                    continue;
                }
            }
            result?;
            lowered.push(block.id);
            let mut instructions = ctx.out;
            drop_broken_clauses(target, &mut instructions);
            blocks.push(LoweredBlock {
                guest_pc: block.start_pc,
                instructions,
                dynamic_entry: dispatch.is_some() && ir.flow.entries.contains(&block.start_pc),
            });
        }
        if let (Some(deferred), Some(origin)) =
            (ngg.and_then(|p| p.deferred), ir.instructions.last())
        {
            let mut epilogue = self.context(regs, WaitTracker::new(false), mode, ir, shared);
            epilogue.epilogue = false;
            epilogue.ngg_epilogue(origin, deferred)?;
            let epilogue = epilogue.out;
            let mut end = self.context(regs, WaitTracker::new(false), mode, ir, shared);
            end.epilogue = false;
            end.lower_gfx1013(
                origin,
                F::Sopp,
                opcodes::SOPP_ENDPGM,
                Fields::Sopp { simm16: 0 },
            )?;
            for (guest_pc, instructions) in
                [(EPILOGUE_LABEL, epilogue), (EPILOGUE_END_LABEL, end.out)]
            {
                blocks.push(LoweredBlock {
                    guest_pc,
                    instructions,
                    dynamic_entry: false,
                });
            }
        }
        let mut program = LoweredProgram {
            target,
            stage: self.options.stage,
            wave64: self.options.wave64(),
            blocks,
            dispatch: None,
            preconverted: preconverted.clone(),
            attribute_ring: ngg.and_then(|p| p.ring).map(|ring| {
                let descriptor = match ring.descriptor {
                    ngg::Kept::Sgpr(sgpr) => Some(sgpr),
                    ngg::Kept::Lane(_) => None,
                };
                (descriptor, ring.words)
            }),
        };
        if let Some(origin) = ir.instructions.first().filter(|_| dispatch.is_some()) {
            self.redirect_dropped_blocks(ir, &lowered, &mut program, regs, mode, shared)?;
            let code_bytes = 4 * ir.code_words as u32;
            let mut stub = self.context(regs, WaitTracker::new(false), mode, ir, shared);
            stub.dispatch_stub(origin, code_bytes)?;
            let stub = stub.out;
            let mut miss = self.context(regs, WaitTracker::new(false), mode, ir, shared);
            miss.miss_stub(origin)?;
            program.blocks.extend([
                LoweredBlock {
                    guest_pc: DISPATCH_LABEL,
                    instructions: stub,
                    dynamic_entry: false,
                },
                LoweredBlock {
                    guest_pc: MISS_LABEL,
                    instructions: miss.out,
                    dynamic_entry: false,
                },
            ]);
            program.dispatch = Some(DispatchTable {
                words: ir.code_words as u32,
            });
        } else if ir.flow.has_dynamic_jumps() {
            let jump = ir.flow.jumps.keys().next().copied().unwrap_or(0);
            let inst = ir.instruction_at(jump).map(|i| &ir.instructions[i]);
            return Err(match inst {
                Some(inst) => unsupported(
                    inst,
                    target,
                    "computed jumps need SGPRs above the shader's own",
                ),
                None => CompileError::Unsupported,
            });
        }
        if !prologue.is_empty() {
            program.blocks.insert(
                0,
                LoweredBlock {
                    guest_pc: PROLOGUE_LABEL,
                    instructions: prologue,
                    dynamic_entry: false,
                },
            );
        }
        Ok(program)
    }

    fn context<'r>(
        &self,
        regs: &'r mut RegisterAllocator,
        waits: WaitTracker,
        mode: bvh::ModeState,
        ir: &'r ShaderIr,
        shared: Shared<'r>,
    ) -> Ctx<'r> {
        let scratch_wave_offset = regs.scratch_wave_offset_sgpr().filter(|&r| {
            self.options.target.is_gfx11_plus() && ir.metadata.live_in_scalar & (1 << r) != 0
        });
        Ctx {
            target: self.options.target,
            stage: self.options.stage,
            wave64: self.options.wave64(),
            precision: self.precision,
            regs,
            out: Vec::new(),
            waits,
            mode,
            guest_base: shared.layout.base,
            layout: shared.layout,
            flow: &ir.flow,
            dispatch: shared.dispatch,
            scratch_wave_offset,
            descriptors: shared.descriptors,
            descriptor_cache: DescriptorCache::default(),
            preconverted: shared.preconverted,
            pixel: shared.pixel,
            pending_dual_source: None,
            ngg: shared.ngg,
            epilogue: shared.ngg.is_some_and(|p| p.deferred.is_some()),
            wave_index: shared.wave_index,
            gds: self
                .gds_buffer
                .filter(|_| self.options.target != TargetIsa::Gfx1013),
            mip_stats: self
                .mip_stats_table
                .filter(|_| self.options.target != TargetIsa::Gfx1013),
            private_buffer: shared.private_buffer,
            spill: shared.spill_vgpr.map(|vgpr| {
                let persistent = shared.ngg.map_or(0, |plan| plan.persistent_lanes);
                let wave_id = u8::from(matches!(shared.wave_index, Some(Kept::Lane(_))));
                SpillLanes::new(vgpr, persistent.max(wave_id))
            }),
        }
    }

    /// PS5 NGG streamout orders its buffer writes with GDS ordered counters
    /// (`ds_ordered_count`), which no other vertex program needs. It is not supported: the
    /// runtime has no streamout, so the program is rejected rather than translated without it.
    fn reject_streamout(&self, ir: &ShaderIr) -> Result<(), CompileError> {
        if self.options.stage != ShaderStage::Vertex || self.options.target == TargetIsa::Gfx1013 {
            return Ok(());
        }
        let ordered = ir
            .blocks
            .iter()
            .filter(|b| b.reachable)
            .flat_map(|b| &ir.instructions[b.instructions()])
            .find(|i| i.family == F::Ds && i.opcode == DS_ORDERED_COUNT);
        match ordered {
            Some(inst) => Err(unsupported(
                inst,
                self.options.target,
                "streamout (GDS ordered counters in a vertex program) is not supported",
            )),
            None => Ok(()),
        }
    }

    /// Slots for converted descriptors, reserved before any transient SGPR: the conversion
    /// targets gfx11 and gfx11.5 (gfx10.3 reads PS5 descriptors as they are).
    /// SGPRs the descriptor slots of [`Self::reserve_descriptor_slots`] take, aligned as
    /// reservations are: those of the program's resource bindings.
    fn descriptor_slot_sgprs(&self, ir: &ShaderIr) -> u16 {
        if self.descriptor_lut.is_none() {
            return 0;
        }
        let uses = |kind: ResourceKind| ir.metadata.bindings.iter().any(|b| b.kind == kind);
        let (images, samplers, buffers) = (
            uses(ResourceKind::ImageDescriptor),
            uses(ResourceKind::SamplerDescriptor),
            uses(ResourceKind::BufferDescriptor),
        );
        [
            (images, 8),
            (buffers, 4),
            (samplers, 4),
            (images || buffers, 4),
        ]
        .iter()
        .filter(|(used, _)| *used)
        .map(|(_, sgprs)| sgprs)
        .sum()
    }

    fn reserve_descriptor_slots(
        &self,
        ir: &ShaderIr,
        regs: &mut RegisterAllocator,
        spill: bool,
    ) -> Result<Option<DescriptorSlots>, CompileError> {
        let target = self.options.target;
        let Some(address) = self.descriptor_lut.filter(|_| target.is_gfx11_plus()) else {
            return Ok(None);
        };
        let failed = |reason| match ir.instructions.first() {
            Some(first) => unsupported(first, target, reason),
            None => CompileError::Unsupported,
        };
        if target.is_gfx12() {
            return Err(failed("descriptor conversion covers gfx11 and gfx11.5"));
        }
        if spill {
            return Ok(Some(DescriptorSlots {
                tsharp: descriptors::NO_SLOT,
                vsharp: descriptors::NO_SLOT,
                ssharp: descriptors::NO_SLOT,
                table: descriptors::NO_SLOT,
                address,
            }));
        }
        // Only the slots the program's resource bindings use: a program with high SGPR
        // pressure keeps room for the transient conversion SGPRs above them.
        let uses = |kind: ResourceKind| ir.metadata.bindings.iter().any(|b| b.kind == kind);
        let (images, samplers, buffers) = (
            uses(ResourceKind::ImageDescriptor),
            uses(ResourceKind::SamplerDescriptor),
            uses(ResourceKind::BufferDescriptor),
        );
        let (sgprs, relative) = (ir.metadata.sgpr_count, ir.metadata.uses_relative_indexing);
        let mut reserve = |wanted: bool, count: u8| -> Result<u8, CompileError> {
            if !wanted {
                return Ok(descriptors::NO_SLOT);
            }
            regs.reserve_sgprs(count).ok_or_else(|| {
                failed(if relative {
                    "descriptor conversion needs SGPRs above the shader's own: it indexes SGPRs relatively"
                } else if sgprs > 96 {
                    "descriptor conversion needs SGPRs above the shader's own: it uses more than 96"
                } else if sgprs > 88 {
                    "descriptor conversion needs SGPRs above the shader's own: it uses 89 to 96"
                } else {
                    "descriptor conversion needs SGPRs above the shader's own: it uses at most 88"
                })
            })
        };
        Ok(Some(DescriptorSlots {
            tsharp: reserve(images, 8)?,
            vsharp: reserve(buffers, 4)?,
            ssharp: reserve(samplers, 4)?,
            table: reserve(images || buffers, 2)?,
            address,
        }))
    }

    /// Branches that reach a dropped block (or fall into one) go to the miss stub instead.
    fn redirect_dropped_blocks(
        &self,
        ir: &ShaderIr,
        lowered: &[usize],
        program: &mut LoweredProgram,
        regs: &mut RegisterAllocator,
        mode: bvh::ModeState,
        shared: Shared<'_>,
    ) -> Result<(), CompileError> {
        let mut kept: Vec<u32> = program.blocks.iter().map(|b| b.guest_pc).collect();
        kept.sort_unstable();
        let mut is_lowered = vec![false; ir.blocks.len() + 1];
        for &id in lowered {
            is_lowered[id] = true;
        }
        for (position, &id) in lowered.iter().enumerate() {
            let block = &ir.blocks[id];
            let falls_into_dropped = ir.instructions[block.end - 1].falls_through()
                && block.successors.contains(&(id + 1))
                && !is_lowered[id + 1];
            for inst in program.blocks[position].instructions.iter_mut() {
                if let Some(target) = inst.branch_target {
                    if target < LOCAL_LABEL_BASE && kept.binary_search(&target).is_err() {
                        inst.branch_target = Some(MISS_LABEL);
                    }
                }
            }
            if falls_into_dropped {
                let origin = &ir.instructions[block.end - 1];
                let mut ctx = self.context(regs, WaitTracker::new(false), mode, ir, shared);
                ctx.branch(origin, MISS_LABEL)?;
                program.blocks[position].instructions.extend(ctx.out);
            }
        }
        Ok(())
    }

    /// Code that runs once before guest pc 0. It forms its own block (see [`PROLOGUE_LABEL`]),
    /// so a branch to pc 0 does not repeat it.
    fn prologue(&self, ir: &ShaderIr, ctx: &mut Ctx<'_>) -> Result<(), CompileError> {
        // The prologue reads system SGPRs itself (the NGG ring offset in s5, where gfx10 had
        // the scratch wave offset), which guest instructions must not.
        ctx.scratch_wave_offset = None;
        if self.options.stage == ShaderStage::Compute {
            self.unpack_thread_ids(ir, ctx);
        }
        if let Some(first) = ir.instructions.first() {
            if self.options.stage == ShaderStage::Vertex {
                ctx.init_full_exec(first)?;
            }
            if self.hull {
                ctx.hull_prologue(first, ir.metadata.live_in_vector)?;
            }
            ctx.save_wave_index(first)?;
            ctx.set_second_half(first, ir.metadata.live_in_scalar)?;
            ctx.load_descriptor_table(first)?;
            ctx.load_hoisted_parameters(first)?;
            ctx.ngg_prologue(first)?;
            ctx.restore_spills(first)?;
        }
        Ok(())
    }

    /// gfx10 loads the enabled thread IDs into v0-v2; gfx11 packs them into v0 (x `[9:0]`, y
    /// `[19:10]`, z `[29:20]`). Y and Z are extracted before v0 is masked, and v0 is masked
    /// whenever Y or Z is enabled, because their bits are set even if only v0 is read.
    fn unpack_thread_ids(&self, ir: &ShaderIr, ctx: &mut Ctx<'_>) {
        let components = ctx.regs.thread_id_components();
        if !ctx.target.is_gfx11_plus() || components == 0 {
            return;
        }
        let Some(first) = ir.instructions.first() else {
            return;
        };
        let live = ir.metadata.live_in_vector;
        let bfe = vector::vop3_only(ctx.target, opcodes::VOP3_BFE_U32);
        let mut highest = 0;
        for component in [2u8, 1] {
            if component <= components && live & (1 << component) != 0 {
                highest = highest.max(component);
                let src = [
                    Operand::Vgpr(0),
                    Operand::constant(PACKED_TID_BITS * u32::from(component)),
                    Operand::constant(PACKED_TID_BITS),
                ];
                ctx.push(vector::vop3_inst(
                    first,
                    bfe,
                    flag::SRC3,
                    Vop3::new(Operand::Vgpr(component), src),
                ));
            }
        }
        if live & 1 != 0 {
            let and = vector::vop2_op(ctx.target, opcodes::VOP2_AND_B32);
            let mask = Operand::Literal((1 << PACKED_TID_BITS) - 1);
            ctx.push(vector::vop2_inst(
                first,
                and,
                Operand::Vgpr(0),
                mask,
                Operand::Vgpr(0),
            ));
        }
        ctx.regs.reserve_vgprs(u16::from(highest) + 1);
    }
}

impl Ctx<'_> {
    pub fn push(&mut self, inst: Instruction) {
        self.waits.record(self.target, &inst);
        self.out.push(inst);
    }

    /// Flushes f32 denormals until [`Ctx::leave_f32_flush`], for expansions of gfx1013
    /// instructions that do not support denormals.
    pub(crate) fn enter_f32_flush(
        &mut self,
        origin: &Instruction,
    ) -> Result<ModeRestore, CompileError> {
        match self.mode {
            bvh::ModeState::Static(mode) if mode & F32_DENORM_MODE == 0 => {
                Ok(ModeRestore::Unchanged)
            }
            bvh::ModeState::Static(mode) => {
                let denorm = u16::from(mode >> 4);
                let flushed = denorm & !u16::from(F32_DENORM_MODE >> 4);
                let fields = Fields::Sopp { simm16: flushed };
                self.lower(&derive(origin, F::Sopp, SOPP_DENORM_MODE, 0, fields))?;
                Ok(ModeRestore::Denorm(denorm))
            }
            bvh::ModeState::Dynamic => {
                let saved = self.regs.scratch_sgprs(1).ok_or_else(|| {
                    unsupported(
                        origin,
                        self.target,
                        "saving the FP mode needs an SGPR above the shader's own",
                    )
                })?;
                self.hwreg(origin, SOPK_GETREG, Operand::Sgpr(saved), None)?;
                self.hwreg(origin, SOPK_SETREG_IMM32, Operand::Raw(0), Some(0))?;
                Ok(ModeRestore::Saved(saved))
            }
        }
    }

    pub(crate) fn leave_f32_flush(
        &mut self,
        origin: &Instruction,
        restore: ModeRestore,
    ) -> Result<(), CompileError> {
        match restore {
            ModeRestore::Unchanged => Ok(()),
            ModeRestore::Denorm(denorm) => {
                let fields = Fields::Sopp { simm16: denorm };
                self.lower(&derive(origin, F::Sopp, SOPP_DENORM_MODE, 0, fields))
            }
            ModeRestore::Saved(sgpr) => self.hwreg(origin, SOPK_SETREG, Operand::Sgpr(sgpr), None),
        }
    }

    /// Lowers a gfx1013 `s_getreg`/`s_setreg` of the f32 denormal bits.
    fn hwreg(
        &mut self,
        origin: &Instruction,
        opcode: u16,
        sdst: Operand,
        imm32: Option<u32>,
    ) -> Result<(), CompileError> {
        let fields = Fields::Sopk {
            sdst,
            simm16: HWREG_MODE_F32_DENORM,
            imm32,
        };
        self.lower_gfx1013(origin, F::Sopk, opcode, fields)
    }

    /// Lowers the guest instruction `inst`, then restores the SGPRs its expansions spilled
    /// ([`spill`]). Expansions lower the instructions they derive with [`Self::lower`], which
    /// leaves the spilled blocks in place until the guest instruction is done.
    fn lower_guest(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        self.lower(inst)?;
        self.restore_spills(inst)
    }

    fn lower(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        self.descriptor_cache.invalidate(inst, self.wave64);
        if self.epilogue && inst.family == F::Sopp && inst.opcode == opcodes::SOPP_ENDPGM {
            return self.branch(inst, EPILOGUE_LABEL);
        }
        if let Some(lowered) = self.lower_pc_relative(inst) {
            return lowered;
        }
        if self.target == TargetIsa::Gfx1013 {
            self.push(retarget(inst, inst.opcode));
            return Ok(());
        }
        if self.target.is_gfx11_plus() {
            if let Some(rewritten) = self.materialize_zero_flags(inst)? {
                return self.lower(&rewritten);
            }
        }
        if let Some(wave_offset) = self.scratch_wave_offset {
            let private = match inst.fields {
                Fields::Mubuf(b) => b.soffset == Operand::Sgpr(wave_offset),
                Fields::Mtbuf(b) => self.private_buffer && b.soffset == Operand::Sgpr(wave_offset),
                _ => false,
            };
            let mut reads = false;
            inst.each_use(self.wave64, |r| {
                reads |= r.contains(crate::ir::RegFile::Scalar, u16::from(wave_offset));
            });
            if reads && !private {
                return Err(unsupported(
                    inst,
                    self.target,
                    "the scratch wave offset SGPR does not exist on the target",
                ));
            }
        }
        match inst.family {
            F::Sop1 | F::Sop2 | F::Sopk | F::Sopc | F::Sopp => self.scalar(inst),
            F::Vop1 | F::Vop2 | F::Vopc | F::Vop3 | F::Vop3p => vector::lower(self, inst),
            F::Vintrp => interp::lower(self, inst),
            F::Smem => self.smem(inst),
            F::Mubuf | F::Mtbuf => self.buffer(inst),
            F::Flat => self.flat(inst),
            F::Ds => self.ds(inst),
            F::Mimg => self.image(inst),
            F::Exp => self.export(inst),
            F::Ldsdir | F::Vinterp => {
                Err(unsupported(inst, self.target, "not a gfx1013 instruction"))
            }
        }
    }

    /// gfx11 removed the `src_vccz`/`src_execz` source operands (the `s_cbranch_vccz/execz`
    /// branches remain). Each use reads a scratch SGPR set to `VCC == 0` / `EXEC == 0` just
    /// before the instruction, with SCC saved and restored around the computation. Returns the
    /// instruction rewritten to read those SGPRs, or `None` when it uses neither operand.
    fn materialize_zero_flags(
        &mut self,
        inst: &Instruction,
    ) -> Result<Option<Instruction>, CompileError> {
        let mut rewritten = inst.clone();
        let flags: Vec<Operand> = source_operands(&mut rewritten.fields)
            .into_iter()
            .map(|op| *op)
            .filter(|op| matches!(op, Operand::VccZ | Operand::ExecZ))
            .collect();
        if flags.is_empty() {
            return Ok(None);
        }
        let base = self.transient_sgprs(inst, 3)?;
        let (saved_scc, vccz, execz) = (
            Operand::Sgpr(base),
            Operand::Sgpr(base + 1),
            Operand::Sgpr(base + 2),
        );
        let mask_compare = if self.wave64 {
            SOPC_CMP_EQ_U64
        } else {
            SOPC_CMP_EQ_U32
        };
        self.scalar_select(inst, saved_scc)?;
        for (flag, mask, dst) in [
            (Operand::VccZ, Operand::VccLo, vccz),
            (Operand::ExecZ, Operand::ExecLo, execz),
        ] {
            if flags.contains(&flag) {
                let fields = Fields::Sopc {
                    src0: mask,
                    src1: Operand::InlineInt(0),
                };
                self.lower_gfx1013(inst, F::Sopc, mask_compare, fields)?;
                self.scalar_select(inst, dst)?;
            }
        }
        let restore = Fields::Sopc {
            src0: saved_scc,
            src1: Operand::InlineInt(0),
        };
        self.lower_gfx1013(inst, F::Sopc, SOPC_CMP_LG_U32, restore)?;
        for op in source_operands(&mut rewritten.fields) {
            match *op {
                Operand::VccZ => *op = vccz,
                Operand::ExecZ => *op = execz,
                _ => {}
            }
        }
        Ok(Some(rewritten))
    }

    /// `count` transient SGPRs for one expansion (aligned, so a pair can hold a 64-bit mask).
    pub(crate) fn transient_sgprs(
        &mut self,
        origin: &Instruction,
        count: u8,
    ) -> Result<u8, CompileError> {
        match self.regs.scratch_sgprs(count) {
            Some(base) => Ok(base),
            None => self.spill_block(origin, count),
        }
    }

    /// `dst = SCC ? 1 : 0` without changing SCC.
    fn scalar_select(&mut self, origin: &Instruction, dst: Operand) -> Result<(), CompileError> {
        let fields = Fields::Sop2 {
            sdst: dst,
            src0: Operand::InlineInt(1),
            src1: Operand::InlineInt(0),
        };
        self.lower_gfx1013(origin, F::Sop2, SOP2_CSELECT_B32, fields)
    }

    /// Lowers a gfx1013 instruction built by an expansion, with the opcode table's flags.
    pub(crate) fn lower_gfx1013(
        &mut self,
        origin: &Instruction,
        family: F,
        opcode: u16,
        fields: Fields,
    ) -> Result<(), CompileError> {
        let flags = if family == F::Vop3 {
            opcodes::vop3_flags(opcode).unwrap_or(0)
        } else {
            opcodes::lookup(family, opcode).map_or(0, |info| info.flags)
        };
        self.lower(&derive(origin, family, opcode, flags, fields))
    }

    fn scalar(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let info = opcodes::lookup(inst.family, inst.opcode)
            .ok_or_else(|| unsupported(inst, self.target, "unknown scalar opcode"))?;
        match info.translate(self.target) {
            Xlat::Op(op) => self.push(retarget(inst, op)),
            Xlat::Drop | Xlat::Same => {}
            Xlat::No => {
                return Err(unsupported(
                    inst,
                    self.target,
                    "instruction was removed on the target",
                ))
            }
            Xlat::Rule(rule) => self.scalar_rule(inst, rule)?,
        }
        Ok(())
    }

    fn scalar_rule(&mut self, inst: &Instruction, rule: Rule) -> Result<(), CompileError> {
        let simm16 = match inst.fields {
            Fields::Sopk { simm16, .. } | Fields::Sopp { simm16 } => simm16,
            _ => 0,
        };
        match rule {
            Rule::FindFirstZero => {
                let Fields::Sop1 { sdst, src0 } = inst.fields else {
                    return Err(unsupported(inst, self.target, "malformed first-zero instruction"));
                };
                let wide = inst.has_flag(flag::S0_2);
                let count = if wide { 2 } else { 1 };
                let temp = self.transient_sgprs(inst, count + 1)?;
                let saved_scc = Operand::Sgpr(temp + count);
                self.scalar_select(inst, saved_scc)?;
                self.lower_gfx1013(inst, F::Sop1, if wide { 0x08 } else { 0x07 }, Fields::Sop1 {
                    sdst: Operand::Sgpr(temp), src0,
                })?;
                self.lower_gfx1013(inst, F::Sop1, if wide { 0x14 } else { 0x13 }, Fields::Sop1 {
                    sdst, src0: Operand::Sgpr(temp),
                })?;
                self.lower_gfx1013(inst, F::Sopc, 0x07, Fields::Sopc {
                    src0: saved_scc, src1: Operand::InlineInt(0),
                })?;
            }
            Rule::Cmpk(sopc) => {
                let Fields::Sopk { sdst, .. } = inst.fields else {
                    return Err(unsupported(inst, self.target, "malformed SOPK compare"));
                };
                let value = if inst.has_flag(flag::IMM_U) {
                    u32::from(simm16)
                } else {
                    simm16 as i16 as i32 as u32
                };
                let fields = Fields::Sopc {
                    src0: sdst,
                    src1: Operand::constant(value),
                };
                self.push(derive(inst, F::Sopc, sopc, 0, fields));
            }
            Rule::Hwreg(op) => {
                let id = simm16 & HWREG_ID_MASK;
                let flat_scratch = matches!(id, HWREG_FLAT_SCR_LO | HWREG_FLAT_SCR_HI);
                if flat_scratch && self.target.is_gfx11_plus() && inst.opcode != SOPK_GETREG_B32 {
                    // gfx11 hardware initialises flat scratch for each wave.
                    return Ok(());
                }
                let portable = PORTABLE_HWREGS.contains(&id)
                    && !(id == HWREG_GPR_ALLOC && self.target.is_gfx11_plus());
                if !portable {
                    return Err(unsupported(
                        inst,
                        self.target,
                        "hardware register layout differs on the target",
                    ));
                }
                self.push(retarget(inst, op));
            }
            Rule::Sendmsg(op) => match simm16 & 0xf {
                MSG_INTERRUPT | MSG_GS_ALLOC_REQ => self.push(retarget(inst, op)),
                MSG_EARLY_PRIM_DEALLOC => {}
                _ => {
                    return Err(unsupported(
                        inst,
                        self.target,
                        "message is not available on the target",
                    ))
                }
            },
            Rule::Waitcnt => {
                let (vm, exp, lgkm) = waitcnt::decode_gfx10(simm16);
                if self.target.is_gfx12() {
                    self.emit_gfx12_waits(inst, Some(vm), Some(exp), Some(lgkm), None);
                } else {
                    let fields = Fields::Sopp {
                        simm16: waitcnt::encode_gfx11(vm, exp, lgkm),
                    };
                    self.push(derive(inst, F::Sopp, SOPP_GFX11_WAITCNT, 0, fields));
                }
            }
            Rule::WaitcntCounter => {
                let Fields::Sopk { sdst, .. } = inst.fields else {
                    return Err(unsupported(inst, self.target, "malformed counter wait"));
                };
                if sdst != Operand::Null {
                    return Err(unsupported(
                        inst,
                        self.target,
                        "counter waits read from an SGPR",
                    ));
                }
                let count = simm16.min(u16::from(u8::MAX)) as u8;
                match inst.opcode {
                    0x17 => self.emit_gfx12_waits(inst, None, None, None, Some(count)),
                    0x18 => self.emit_gfx12_waits(inst, Some(count), None, None, None),
                    0x19 => self.emit_gfx12_waits(inst, None, Some(count), None, None),
                    _ => self.emit_gfx12_waits(inst, None, None, Some(count), None),
                }
            }
            Rule::Barrier => {
                let signal = Fields::Sop1 {
                    sdst: Operand::Raw(0),
                    src0: Operand::InlineInt(-1),
                };
                self.push(derive(
                    inst,
                    F::Sop1,
                    SOP1_GFX12_BARRIER_SIGNAL,
                    flag::NO_DST,
                    signal,
                ));
                let wait = Fields::Sopp { simm16: 0xffff };
                self.push(derive(inst, F::Sopp, SOPP_GFX12_BARRIER_WAIT, 0, wait));
            }
            _ => {
                return Err(unsupported(
                    inst,
                    self.target,
                    "no scalar lowering for this rule",
                ))
            }
        }
        Ok(())
    }

    fn emit_gfx12_waits(
        &mut self,
        inst: &Instruction,
        vm: Option<u8>,
        exp: Option<u8>,
        lgkm: Option<u8>,
        vs: Option<u8>,
    ) {
        let waits = self.waits.gfx12_waits(vm, exp, lgkm, vs);
        for (opcode, count) in waits {
            let fields = Fields::Sopp {
                simm16: u16::from(count),
            };
            self.push(derive(inst, F::Sopp, opcode, 0, fields));
        }
    }

    fn smem(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let info = memops::smem(inst.opcode)
            .ok_or_else(|| unsupported(inst, self.target, "unknown SMEM opcode"))?;
        match info.translate(self.target) {
            Xlat::Op(op) => self.push(retarget(inst, op)),
            Xlat::Drop | Xlat::Same => {}
            Xlat::No => {
                return Err(unsupported(
                    inst,
                    self.target,
                    "scalar memory operation was removed",
                ))
            }
            Xlat::Rule(Rule::MemTime) => {
                let Fields::Smem(s) = inst.fields else {
                    return Err(unsupported(inst, self.target, "malformed SMEM"));
                };
                let fields = Fields::Sop1 {
                    sdst: s.sdata,
                    src0: Operand::Raw(MSG_RTN_GET_REALTIME),
                };
                let flags = flag::D2 | flag::NO_SRC;
                self.push(derive(
                    inst,
                    F::Sop1,
                    SOP1_GFX11_SENDMSG_RTN_B64,
                    flags,
                    fields,
                ));
                if inst.opcode == SMEM_MEMTIME {
                    self.scale_realtime_to_shader_clock(inst, s.sdata)?;
                }
            }
            Xlat::Rule(Rule::WaveId) => self.wave_id(inst)?,
            Xlat::Rule(_) => {
                return Err(unsupported(
                    inst,
                    self.target,
                    "no SMEM lowering for this rule",
                ))
            }
        }
        Ok(())
    }

    /// `s_memtime` counts the PS5 shader clock (2.23 GHz); the target only offers the 100 MHz
    /// realtime counter, so its value is scaled by 715/32 (22.34). The 64-bit product stays
    /// exact for seven years of uptime.
    fn scale_realtime_to_shader_clock(
        &mut self,
        inst: &Instruction,
        dst: Operand,
    ) -> Result<(), CompileError> {
        let Operand::Sgpr(lo) = dst else {
            return Err(unsupported(
                inst,
                self.target,
                "s_memtime into a special register",
            ));
        };
        let base = self.regs.scratch_sgprs(2).ok_or_else(|| {
            unsupported(
                inst,
                self.target,
                "s_memtime scaling needs SGPRs above the shader's own",
            )
        })?;
        let (hi, carry, saved_scc) = (
            Operand::Sgpr(lo + 1),
            Operand::Sgpr(base),
            Operand::Sgpr(base + 1),
        );
        let (lo, factor) = (
            Operand::Sgpr(lo),
            Operand::Literal(SHADER_CLOCKS_PER_32_REALTIME),
        );
        if self.target.is_gfx12() {
            self.emit_gfx12_waits(inst, None, None, Some(0), None);
        } else {
            let wait = Fields::Sopp {
                simm16: waitcnt::encode_gfx11(
                    waitcnt::GFX10_VMCNT_MAX,
                    waitcnt::GFX10_EXPCNT_MAX,
                    0,
                ),
            };
            self.push(derive(inst, F::Sopp, SOPP_GFX11_WAITCNT, 0, wait));
        }
        self.scalar_select(inst, saved_scc)?;
        let sop2 = |op: u16, sdst: Operand, src0: Operand, src1: Operand| {
            (op, Fields::Sop2 { sdst, src0, src1 })
        };
        for (op, fields) in [
            sop2(SOP2_MUL_HI_U32, carry, lo, factor),
            sop2(SOP2_MUL_I32, lo, lo, factor),
            sop2(SOP2_MUL_I32, hi, hi, factor),
            sop2(SOP2_ADD_U32, hi, hi, carry),
            sop2(SOP2_LSHR_B64, lo, lo, Operand::InlineInt(5)),
        ] {
            self.lower_gfx1013(inst, F::Sop2, op, fields)?;
        }
        let restore = Fields::Sopc {
            src0: saved_scc,
            src1: Operand::InlineInt(0),
        };
        self.lower_gfx1013(inst, F::Sopc, SOPC_CMP_LG_U32, restore)
    }

    /// `s_get_waveid_in_workgroup` (gone on gfx11): the index the prologue extracted, copied
    /// without touching SCC.
    fn wave_id(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let Fields::Smem(s) = inst.fields else {
            return Err(unsupported(inst, self.target, "malformed SMEM"));
        };
        let index = self.wave_index.ok_or_else(|| {
            unsupported(
                inst,
                self.target,
                "the wave index needs the compute TG_SIZE SGPR",
            )
        })?;
        match index {
            Kept::Sgpr(index) => {
                let copy = Fields::Sop1 {
                    sdst: s.sdata,
                    src0: Operand::Sgpr(index),
                };
                self.lower_gfx1013(inst, F::Sop1, SOP1_MOV_B32, copy)
            }
            Kept::Lane(lane) => {
                let Operand::Sgpr(destination) = s.sdata else {
                    return Err(unsupported(
                        inst,
                        self.target,
                        "a wave index destination is not an SGPR",
                    ));
                };
                self.read_lane(inst, destination, lane)
            }
        }
    }

    /// gfx11 merged LS-HS waves do not load gfx10's v3, the LS thread's index in its
    /// threadgroup: when the program reads it, it is the wave's index (s5 bits 2:0, which gfx11
    /// loads there instead of the scratch wave offset) times the wave size plus the lane's.
    pub(super) fn hull_prologue(
        &mut self,
        origin: &Instruction,
        live_in_vector: u128,
    ) -> Result<(), CompileError> {
        const LOCAL_INDEX_VGPR: u8 = 3;
        const TCS_WAVE_ID_SGPR: u8 = 5;
        const TCS_WAVE_ID_FIELD: u32 = 3 << 16;
        if !self.target.is_gfx11() || live_in_vector & (1 << LOCAL_INDEX_VGPR) == 0 {
            return Ok(());
        }
        let first_lane = self.transient_sgprs(origin, 1)?;
        let sop2 = |src0, src1| Fields::Sop2 {
            sdst: Operand::Sgpr(first_lane),
            src0,
            src1,
        };
        let wave = sop2(
            Operand::Sgpr(TCS_WAVE_ID_SGPR),
            Operand::Literal(TCS_WAVE_ID_FIELD),
        );
        self.lower_gfx1013(origin, F::Sop2, SOP2_BFE_U32, wave)?;
        let lanes = sop2(
            Operand::Sgpr(first_lane),
            Operand::constant(if self.wave64 { 64 } else { 32 }),
        );
        self.lower_gfx1013(origin, F::Sop2, SOP2_MUL_I32, lanes)?;
        let index = Operand::Vgpr(LOCAL_INDEX_VGPR);
        let all = Operand::InlineInt(-1);
        let lo = crate::ir::Vop3::new(index, [all, Operand::Sgpr(first_lane), Operand::Raw(0)]);
        self.lower_gfx1013(origin, F::Vop3, VOP3_MBCNT_LO, Fields::Vop3(lo))?;
        if self.wave64 {
            let hi = crate::ir::Vop3::new(index, [all, index, Operand::Raw(0)]);
            self.lower_gfx1013(origin, F::Vop3, VOP3_MBCNT_HI, Fields::Vop3(hi))?;
        }
        Ok(())
    }

    /// The prologue's part for `s_get_waveid_in_workgroup`: the wave index from TG_SIZE (bits
    /// 24:20). The translation enables TG_SIZE where the guest may not have, so the guest can
    /// use that SGPR as its own before asking for the index; it is read before any guest code.
    fn save_wave_index(&mut self, origin: &Instruction) -> Result<(), CompileError> {
        let Some(index) = self.wave_index else {
            return Ok(());
        };
        let tg_size = self.regs.tg_size_sgpr().ok_or_else(|| {
            unsupported(
                origin,
                self.target,
                "the wave index needs the compute TG_SIZE SGPR",
            )
        })?;
        let destination = match index {
            Kept::Sgpr(sgpr) => sgpr,
            Kept::Lane(_) => self.transient_sgprs(origin, 1)?,
        };
        let extract = Fields::Sop2 {
            sdst: Operand::Sgpr(destination),
            src0: Operand::Sgpr(tg_size),
            src1: Operand::Literal(TG_SIZE_WAVE_ID),
        };
        self.lower_gfx1013(origin, F::Sop2, SOP2_BFE_U32, extract)?;
        if let Kept::Lane(lane) = index {
            self.write_lane(origin, Operand::Sgpr(destination), lane)?;
        }
        Ok(())
    }

    /// A buffer access addressed by the scratch wave offset SGPR: the gfx10 private segment.
    fn is_private(&self, inst: &Instruction) -> bool {
        matches!(
            (self.scratch_wave_offset, inst.fields),
            (Some(wave_offset), Fields::Mubuf(b) | Fields::Mtbuf(b))
                if b.soffset == Operand::Sgpr(wave_offset)
        )
    }

    /// gfx11 has no scratch wave offset SGPR: a private-segment access becomes the same access
    /// through flat scratch, which addresses the wave's lane-private memory directly. A TFE
    /// load's status dword is written only when a fetch fails, which private memory never
    /// does, so it is left alone here too. Other buffer accesses are untouched.
    fn scratch_buffer(&mut self, inst: &Instruction) -> Result<Option<Instruction>, CompileError> {
        let Fields::Mubuf(b) = inst.fields else {
            return Ok(None);
        };
        if !self.is_private(inst) {
            return Ok(None);
        }
        if b.idxen || b.lds {
            return Err(unsupported(
                inst,
                self.target,
                "indexed and LDS-direct private-segment accesses are not supported on the target",
            ));
        }
        let info = memops::flat(inst.opcode)
            .filter(|op| op.kind != MemKind::Control)
            .ok_or_else(|| unsupported(inst, self.target, "no flat scratch form"))?;
        let fields = Fields::Flat(crate::ir::Flat {
            segment: crate::ir::FlatSegment::Scratch,
            vdst: b.vdata,
            vdata: b.vdata,
            vaddr: b.vaddr,
            saddr: b.offen.then_some(Operand::Null),
            uses_vaddr: b.offen,
            offset: i32::from(b.offset),
            glc: b.glc,
            dlc: b.dlc,
            slc: b.slc,
            lds: false,
            kind: info.kind,
            data_dwords: info.dwords,
        });
        Ok(Some(derive(inst, F::Flat, inst.opcode, info.flags, fields)))
    }

    /// Buffer mode, for programs with indexed private-segment accesses on gfx11, where flat
    /// scratch cannot reach another lane's private memory: every private access keeps its
    /// buffer instruction, through a copy of its V# whose base is the wave's scratch
    /// (`HW_REG_FLAT_SCR`), with the index stride set to the wave size so the wave's lanes fit
    /// its scratch, and no scratch wave offset. The guest's swizzled, indexed addressing then
    /// works as on gfx10, within the wave's scratch.
    fn private_buffer_access(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let (Fields::Mubuf(b) | Fields::Mtbuf(b)) = inst.fields else {
            return Err(unsupported(
                inst,
                self.target,
                "malformed buffer instruction",
            ));
        };
        let guest = if self.descriptors.is_some()
            && !self.is_preconverted(ResourceKind::BufferDescriptor, b.srsrc, 4)
        {
            self.converted_vsharp(inst, b.srsrc)?
        } else {
            b.srsrc
        };
        let t = self.transient_sgprs(inst, 6)?;
        let (temp, saved_scc) = (t + 4, t + 5);
        self.scalar_select(inst, Operand::Sgpr(saved_scc))?;
        for (k, id) in [(0, HWREG_FLAT_SCR_LO), (1, HWREG_FLAT_SCR_HI)] {
            let fields = Fields::Sopk {
                sdst: Operand::Sgpr(t + k),
                simm16: id | (31 << 11),
                imm32: None,
            };
            self.push(derive(inst, F::Sopk, GFX11_SOPK_GETREG_B32, 0, fields));
        }
        let index_stride: u32 = if self.wave64 { 3 } else { 2 };
        let s = Operand::Sgpr;
        let words = [
            (SOP2_AND_B32, t + 1, s(t + 1), Operand::constant(0xffff)),
            (
                SOP2_AND_B32,
                temp,
                s(guest + 1),
                Operand::constant(0xffff_0000),
            ),
            (SOP2_OR_B32, t + 1, s(t + 1), s(temp)),
            (SOP2_OR_B32, t + 2, s(guest + 2), Operand::InlineInt(0)),
            (
                SOP2_AND_B32,
                t + 3,
                s(guest + 3),
                Operand::constant(!(3u32 << 21)),
            ),
            (
                SOP2_OR_B32,
                t + 3,
                s(t + 3),
                Operand::constant(index_stride << 21),
            ),
        ];
        for (opcode, sdst, src0, src1) in words {
            let fields = Fields::Sop2 {
                sdst: s(sdst),
                src0,
                src1,
            };
            self.lower_gfx1013(inst, F::Sop2, opcode, fields)?;
        }
        let restore = Fields::Sopc {
            src0: s(saved_scc),
            src1: Operand::InlineInt(0),
        };
        self.lower_gfx1013(inst, F::Sopc, SOPC_CMP_LG_U32, restore)?;
        let mut access = inst.clone();
        if let Fields::Mubuf(a) | Fields::Mtbuf(a) = &mut access.fields {
            a.srsrc = t;
            a.soffset = Operand::InlineInt(0);
        }
        self.buffer_access(&access, false)
    }

    fn buffer(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        if self.private_buffer && self.is_private(inst) {
            return self.private_buffer_access(inst);
        }
        if let Some(scratch) = self.scratch_buffer(inst)? {
            return self.lower(&scratch);
        }
        self.buffer_access(inst, true)
    }

    /// A buffer instruction on the target; with `convert`, its V# is converted where the
    /// translation converts descriptors.
    fn buffer_access(&mut self, inst: &Instruction, convert: bool) -> Result<(), CompileError> {
        let info = if inst.family == F::Mubuf {
            memops::mubuf(inst.opcode)
        } else {
            memops::mtbuf(inst.opcode)
        }
        .ok_or_else(|| unsupported(inst, self.target, "unknown buffer opcode"))?;
        let op = match info.translate(self.target) {
            Xlat::Op(op) => op,
            Xlat::Drop | Xlat::Same => return Ok(()),
            _ => {
                return Err(unsupported(
                    inst,
                    self.target,
                    "buffer operation was removed",
                ))
            }
        };
        let mut lowered = retarget(inst, op);
        if let Fields::Mubuf(b) | Fields::Mtbuf(b) = &mut lowered.fields {
            if self.target.is_gfx11_plus() && b.lds {
                return Err(unsupported(
                    inst,
                    self.target,
                    "LDS-direct buffer loads use different opcodes",
                ));
            }
            if inst.family == F::Mtbuf {
                b.format = memops::remap_buffer_format(b.format, self.target).ok_or_else(|| {
                    unsupported(inst, self.target, "buffer data format was removed")
                })?;
            }
        }
        if let (Some(_), Fields::Mubuf(b) | Fields::Mtbuf(b), true) =
            (self.descriptors, inst.fields, convert)
        {
            if !self.is_preconverted(ResourceKind::BufferDescriptor, b.srsrc, 4) {
                let converted = self.converted_vsharp(inst, b.srsrc)?;
                if let Fields::Mubuf(b) | Fields::Mtbuf(b) = &mut lowered.fields {
                    b.srsrc = converted;
                }
            }
        }
        self.push(lowered);
        Ok(())
    }

    fn is_preconverted(&self, kind: ResourceKind, base: u8, dwords: u8) -> bool {
        self.preconverted
            .iter()
            .any(|b| b.kind == kind && b.base_sgpr == base && b.dword_count == dwords)
    }

    fn flat(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let info = memops::flat(inst.opcode)
            .ok_or_else(|| unsupported(inst, self.target, "unknown FLAT opcode"))?;
        let Xlat::Op(op) = info.translate(self.target) else {
            return Err(unsupported(inst, self.target, "flat operation was removed"));
        };
        if let Fields::Flat(f) = &inst.fields {
            if self.target.is_gfx11_plus() && f.lds {
                return Err(unsupported(
                    inst,
                    self.target,
                    "LDS-direct global loads use different opcodes",
                ));
            }
        }
        self.push(retarget(inst, op));
        Ok(())
    }

    fn ds(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let info = memops::ds_op(inst.opcode)
            .ok_or_else(|| unsupported(inst, self.target, "unknown DS opcode"))?;
        if info.flags & memops::DS_GWS != 0 {
            return Err(unsupported(inst, self.target, gds::GWS_NOT_EMULATED));
        }
        if let (Some(buffer), Fields::Ds(d)) = (self.gds, &inst.fields) {
            if d.gds || info.flags & (memops::DS_GWS | memops::DS_GDS_ONLY) != 0 {
                return gds::lower(self, inst, buffer);
            }
        }
        let Xlat::Op(op) = info.translate(self.target) else {
            let reason = if info.flags & memops::DS_SRC2 != 0 {
                "LDS src2 forms were removed in gfx10.3"
            } else if info.flags & memops::DS_NOT_GFX12 != 0 {
                "gfx12 removed the float compare-stores and ds_wrap_rtn_b32"
            } else {
                "GWS and GDS-only operations were removed in gfx12"
            };
            return Err(unsupported(inst, self.target, reason));
        };
        let mut lowered = retarget(inst, op);
        if let Fields::Ds(d) = &mut lowered.fields {
            if self.target.is_gfx12() && d.gds {
                return Err(unsupported(inst, self.target, "GDS was removed"));
            }
            if info.swaps_data(self.target) {
                std::mem::swap(&mut d.data0, &mut d.data1);
            }
        }
        self.push(lowered);
        Ok(())
    }

    fn image(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let info = memops::image(inst.opcode)
            .ok_or_else(|| unsupported(inst, self.target, "unknown MIMG opcode"))?;
        if info.flags & memops::img::BVH != 0 {
            return bvh::lower(self, inst);
        }
        let float_atomic = self.target.is_gfx11() && matches!(inst.opcode, 0x1e | 0x1f);
        let op = match info.translate(self.target) {
            Xlat::Op(op) => op,
            _ if float_atomic => 0x0b,
            _ => return Err(unsupported(inst, self.target, "image operation is not available on the target")),
        };
        let mut lowered = retarget(inst, op);
        let Fields::Mimg(guest) = inst.fields else {
            return Err(unsupported(inst, self.target, "malformed MIMG"));
        };
        if let Some(table) = self.mip_stats {
            let sampled = info.flags & memops::img::SAMPLER != 0 && info.flags & memops::img::GATHER == 0;
            if self.stage == ShaderStage::Fragment && sampled {
                mip_stats::count(self, inst, &guest, info.flags, table)?;
            }
        }
        let image_dwords = if guest.r128 { 4 } else { 8 };
        let convert_image = self.descriptors.is_some()
            && !self.is_preconverted(ResourceKind::ImageDescriptor, guest.srsrc, image_dwords);
        let convert_sampler = self.descriptors.is_some()
            && guest.uses_sampler
            && !self.is_preconverted(ResourceKind::SamplerDescriptor, guest.ssamp, 4);
        let srsrc = match convert_image {
            true => Some(self.converted_tsharp(inst, guest.srsrc, guest.r128)?),
            false => None,
        };
        let ssamp = if convert_sampler {
            Some(self.converted_ssharp(inst, guest.ssamp)?)
        } else if info.flags & memops::img::ADJUST != 0 {
            Some(self.adjusted_sampler(inst)?)
        } else {
            None
        };
        if let Fields::Mimg(m) = &mut lowered.fields {
            if let Some(srsrc) = srsrc {
                m.srsrc = srsrc;
                m.r128 = false;
            }
            if let Some(ssamp) = ssamp {
                m.ssamp = ssamp;
            }
        }
        image_addresses::pack(self, inst, &mut lowered)?;
        if float_atomic {
            image_atomic::lower(self, inst, lowered)
        } else {
            self.push(lowered);
            Ok(())
        }
    }

    /// Copies the sampler of a PS5 `_a` instruction into scratch SGPRs with the lane bits of
    /// dword 3 cleared (the plain form the target runs must not see them: gfx11 and gfx12 keep
    /// BORDER_COLOR_PTR there), preserving SCC. Returns the copy's first SGPR.
    fn adjusted_sampler(&mut self, inst: &Instruction) -> Result<u8, CompileError> {
        let Fields::Mimg(m) = inst.fields else {
            return Err(unsupported(inst, self.target, "malformed MIMG"));
        };
        let base = self.regs.scratch_sgprs(5).ok_or_else(|| {
            unsupported(
                inst,
                self.target,
                "an _a sample needs SGPRs above the shader's own for its sampler copy",
            )
        })?;
        let (samp, saved_scc) = (m.ssamp, Operand::Sgpr(base + 4));
        self.scalar_select(inst, saved_scc)?;
        let pair = Fields::Sop1 {
            sdst: Operand::Sgpr(base),
            src0: Operand::Sgpr(samp),
        };
        self.lower_gfx1013(inst, F::Sop1, SOP1_MOV_B64, pair)?;
        let third = Fields::Sop1 {
            sdst: Operand::Sgpr(base + 2),
            src0: Operand::Sgpr(samp + 2),
        };
        self.lower_gfx1013(inst, F::Sop1, SOP1_MOV_B32, third)?;
        let masked = Fields::Sop2 {
            sdst: Operand::Sgpr(base + 3),
            src0: Operand::Sgpr(samp + 3),
            src1: Operand::Literal(!memops::img::SAMPLER_ADJUST_BITS),
        };
        self.lower_gfx1013(inst, F::Sop2, SOP2_AND_B32, masked)?;
        let restore = Fields::Sopc {
            src0: saved_scc,
            src1: Operand::InlineInt(0),
        };
        self.lower_gfx1013(inst, F::Sopc, SOPC_CMP_LG_U32, restore)?;
        Ok(base)
    }

    fn export(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let mut lowered = retarget(inst, inst.opcode);
        if let Fields::Exp(e) = &mut lowered.fields {
            if self.target.is_gfx11_plus() {
                if e.target >= EXP_PARAM_FIRST {
                    // gfx11+ removed parameter exports: vertex attributes travel through the
                    // NGG attribute ring. Without one only the reference executor's path (which
                    // consumes the guest exports itself) can drop them; other stages cannot
                    // emit them at all.
                    if let Some(ring) = self.ngg.and_then(|p| p.ring) {
                        let e = *e;
                        return self.ring_store(inst, &e, ring);
                    }
                    if self.stage == ShaderStage::Vertex {
                        return Ok(());
                    }
                    return Err(unsupported(
                        inst,
                        self.target,
                        "parameter exports require the attribute ring",
                    ));
                }
                if e.target == EXP_NULL {
                    e.target = self.pixel.map_or(EXP_MRT0, |p| p.null_target);
                    e.en = 0;
                }
                if e.compr {
                    e.en = u8::from(e.en & 0x3 != 0) | (u8::from(e.en & 0xc != 0) << 1);
                    e.vsrc = [e.vsrc[0], e.vsrc[1], 0, 0];
                    e.compr = false;
                }
                e.vm = false;
                let done_position =
                    e.done && (ngg::EXP_POS_FIRST..=ngg::EXP_POS_LAST).contains(&e.target);
                if let (Some(plan), true) = (self.ngg, done_position) {
                    if let Some(deferred) = plan.deferred {
                        let e = *e;
                        return self.defer_position(inst, &e, deferred);
                    }
                    if plan.ring.is_some() {
                        self.wait_for_stores(inst)?;
                    }
                }
                let dual_source = self.pixel.and_then(|p| p.dual_source);
                if let (Some(held), EXP_MRT0 | EXP_MRT1) = (dual_source, e.target) {
                    let e = *e;
                    return self.dual_source_export(inst, e, held);
                }
            }
        }
        self.push(lowered);
        Ok(())
    }

    /// gfx11 dual-source blending exports MRT0 (src0) and MRT1 (src1) as targets 21 and 22,
    /// each lane pair swizzled: 21 = {even: src0, odd: src1 of the even lane}, 22 = {even:
    /// src0 of the odd lane, odd: src1}. The first export of the pair is copied to `held` (the
    /// guest may overwrite its VGPRs before the second); the swizzle runs with every lane
    /// enabled, as the hardware reads a pixel's data from both lanes of its pair.
    fn dual_source_export(
        &mut self,
        inst: &Instruction,
        export: Export,
        held: u8,
    ) -> Result<(), CompileError> {
        let Some(first) = self.pending_dual_source.take() else {
            for c in (0..4u8).filter(|c| export.en & (1 << c) != 0) {
                let src = Operand::Vgpr(export.vsrc[usize::from(c)]);
                self.lower_gfx1013(
                    inst,
                    F::Vop1,
                    opcodes::VOP1_MOV_B32,
                    Fields::Vop1 {
                        vdst: Operand::Vgpr(held + c),
                        src0: src,
                        ext: VopExt::None,
                    },
                )?;
            }
            self.pending_dual_source = Some(export);
            return Ok(());
        };
        if first.target == export.target {
            return Err(unsupported(
                inst,
                self.target,
                "dual-source blending exports MRT0 and MRT1 once each",
            ));
        }
        let copies = [held, held + 1, held + 2, held + 3];
        let (src0, src1) = if first.target == EXP_MRT0 {
            (copies, export.vsrc)
        } else {
            (export.vsrc, copies)
        };
        let en = first.en | export.en;
        let swizzled = self.regs.scratch_vgprs(8)?;
        let saved = self.transient_sgprs(inst, 4)?;
        let (saved_exec, saved_vcc) = (saved, saved + 2);
        self.enable_all_lanes(inst, saved_exec)?;
        let mov = if self.wave64 {
            SOP1_MOV_B64
        } else {
            SOP1_MOV_B32
        };
        let copy = |sdst, src0| Fields::Sop1 { sdst, src0 };
        self.lower_gfx1013(
            inst,
            F::Sop1,
            mov,
            copy(Operand::Sgpr(saved_vcc), Operand::VccLo),
        )?;
        for (half, mask, sources) in [
            (0u8, 0x5555_5555, (src1, src0)),
            (4, 0xaaaa_aaaa, (src0, src1)),
        ] {
            for dst in [Operand::VccLo, Operand::VccHi]
                .iter()
                .take(if self.wave64 { 2 } else { 1 })
            {
                let fields = copy(*dst, Operand::Literal(mask));
                self.lower_gfx1013(inst, F::Sop1, SOP1_MOV_B32, fields)?;
            }
            // v_cndmask_b32_dpp d, a, b, vcc row_xmask:1 = VCC ? b : a of the other lane of the pair.
            for c in (0..4u8).filter(|c| en & (1 << c) != 0) {
                let (other, own) = (sources.0[usize::from(c)], sources.1[usize::from(c)]);
                let fields = Fields::Vop2 {
                    vdst: Operand::Vgpr(swizzled + half + c),
                    src0: Operand::Vgpr(other),
                    vsrc1: Operand::Vgpr(own),
                    constant: None,
                    ext: VopExt::Dpp16(Dpp16 {
                        ctrl: DPP_ROW_XMASK_1,
                        fetch_inactive: true,
                        bound_ctrl: false,
                        src0_neg: false,
                        src0_abs: false,
                        src1_neg: false,
                        src1_abs: false,
                        bank_mask: 0xf,
                        row_mask: 0xf,
                    }),
                };
                self.lower_gfx1013(inst, F::Vop2, opcodes::VOP2_CNDMASK, fields)?;
            }
        }
        self.lower_gfx1013(
            inst,
            F::Sop1,
            mov,
            copy(Operand::VccLo, Operand::Sgpr(saved_vcc)),
        )?;
        self.restore_lanes(inst, saved_exec)?;
        for (k, target) in EXP_DUAL_SOURCE.into_iter().enumerate() {
            let base = swizzled + 4 * k as u8;
            let fields = Fields::Exp(Export {
                target,
                en,
                compr: false,
                done: k == 1 && (first.done || export.done),
                vm: false,
                row_en: false,
                vsrc: [base, base + 1, base + 2, base + 3],
            });
            self.push(derive(inst, F::Exp, inst.opcode, inst.flags, fields));
        }
        Ok(())
    }
}

/// DPP16 `row_xmask:1`: the other lane of each even/odd pair.
const DPP_ROW_XMASK_1: u16 = 0x161;

const SOPP_CLAUSE: u16 = 0x21;
const SOPP_NOP: u16 = 0x00;

/// A gfx10 hard clause (`s_clause`, kept only on gfx10 targets) covers the instructions after
/// it, one more than `simm16[5:0]`, which must be memory instructions of one type with nothing
/// but `s_nop` between them. Lowering can expand or drop a member (BVH intersections become
/// ALU code, cache write-backs disappear), so a clause that no longer fits is removed; clauses
/// only affect cache locality.
fn drop_broken_clauses(target: TargetIsa, instructions: &mut Vec<Instruction>) {
    if target.is_gfx11_plus() {
        return;
    }
    let clause_type = |inst: &Instruction| match &inst.fields {
        Fields::Mubuf(_) | Fields::Mtbuf(_) | Fields::Mimg(_) => Some(0),
        Fields::Flat(f) if f.segment != FlatSegment::Flat => Some(0),
        Fields::Flat(_) => Some(1),
        Fields::Smem(_) => Some(2),
        Fields::Ds(_) => Some(3),
        _ => None,
    };
    let is_nop = |inst: &Instruction| inst.family == F::Sopp && inst.opcode == SOPP_NOP;
    let mut k = 0;
    while k < instructions.len() {
        let inst = &instructions[k];
        if let (F::Sopp, SOPP_CLAUSE, Fields::Sopp { simm16 }) =
            (inst.family, inst.opcode, &inst.fields)
        {
            let members = instructions.get(k + 1..k + 2 + usize::from(simm16 & 0x3f));
            let first = members.and_then(|m| m.first()).and_then(clause_type);
            let fits =
                members.is_some_and(|m| m.iter().all(|i| clause_type(i) == first || is_nop(i)));
            if first.is_none() || !fits {
                instructions.remove(k);
                continue;
            }
        }
        k += 1;
    }
}

/// Blocks reachable from the entry without a dynamic jump.
/// A guest SALU instruction with each literal on a 32-bit source replaced by the inline
/// constant that reads the same ([`Operand::inline_32`]), 4 bytes shorter. Only guest
/// instructions: the lowering's own literals can be relocation slots.
fn inline_scalar_literals(inst: &Instruction) -> Cow<'_, Instruction> {
    let inline = |op: Operand, wide: bool| match op {
        Operand::Literal(value) if !wide => Operand::inline_32(value),
        _ => None,
    };
    let fields = match inst.fields {
        Fields::Sop1 { sdst, src0 } if !inst.has_flag(flag::NO_SRC) => {
            inline(src0, inst.has_flag(flag::S0_2)).map(|src0| Fields::Sop1 { sdst, src0 })
        }
        Fields::Sop2 { sdst, src0, src1 } => {
            let (a, b) = (
                inline(src0, inst.has_flag(flag::S0_2)),
                inline(src1, inst.has_flag(flag::S1_2)),
            );
            (a.is_some() || b.is_some()).then(|| Fields::Sop2 {
                sdst,
                src0: a.unwrap_or(src0),
                src1: b.unwrap_or(src1),
            })
        }
        Fields::Sopc { src0, src1 } => {
            let (a, b) = (
                inline(src0, inst.has_flag(flag::S0_2)),
                inline(src1, inst.has_flag(flag::S1_2)),
            );
            (a.is_some() || b.is_some()).then(|| Fields::Sopc {
                src0: a.unwrap_or(src0),
                src1: b.unwrap_or(src1),
            })
        }
        _ => None,
    };
    match fields {
        Some(fields) => Cow::Owned(Instruction {
            fields,
            ..inst.clone()
        }),
        None => Cow::Borrowed(inst),
    }
}

/// gfx11 programs whose private-segment accesses include an indexed one, which flat scratch
/// cannot express (see [`Ctx::private_buffer_access`]).
fn indexes_private_segment(ir: &ShaderIr, regs: &RegisterAllocator, target: TargetIsa) -> bool {
    let Some(wave_offset) = regs.scratch_wave_offset_sgpr() else {
        return false;
    };
    target.is_gfx11()
        && ir.metadata.live_in_scalar & (1 << wave_offset) != 0
        && ir
            .blocks
            .iter()
            .filter(|b| b.reachable)
            .flat_map(|b| &ir.instructions[b.instructions()])
            .any(|inst| {
                matches!(inst.fields, Fields::Mubuf(b) | Fields::Mtbuf(b)
                    if b.idxen && b.soffset == Operand::Sgpr(wave_offset))
            })
}

fn statically_reachable(ir: &ShaderIr) -> Vec<bool> {
    let mut seen = vec![false; ir.blocks.len()];
    let mut stack = vec![0usize];
    while let Some(id) = stack.pop() {
        if id < seen.len() && !seen[id] {
            seen[id] = true;
            stack.extend(ir.blocks[id].successors.iter().copied());
        }
    }
    seen
}

/// Every operand an instruction reads through a source field.
fn source_operands(fields: &mut Fields) -> Vec<&mut Operand> {
    match fields {
        Fields::Sop1 { src0, .. } | Fields::Vop1 { src0, .. } => vec![src0],
        Fields::Sop2 { src0, src1, .. } | Fields::Sopc { src0, src1 } => vec![src0, src1],
        Fields::Vop2 { src0, vsrc1, .. } | Fields::Vopc { src0, vsrc1, .. } => vec![src0, vsrc1],
        Fields::Vop3(v) => v.src.iter_mut().collect(),
        Fields::Vop3p(v) => v.src.iter_mut().collect(),
        Fields::Smem(s) => vec![&mut s.soffset],
        Fields::Mubuf(b) | Fields::Mtbuf(b) => vec![&mut b.soffset],
        _ => Vec::new(),
    }
}

pub(crate) fn is_vmem_load(inst: &Instruction) -> bool {
    match &inst.fields {
        Fields::Mubuf(b) | Fields::Mtbuf(b) => {
            b.kind == MemKind::Load || (b.kind == MemKind::Atomic && b.glc)
        }
        Fields::Flat(f) => f.kind == MemKind::Load || (f.kind == MemKind::Atomic && f.glc),
        Fields::Mimg(m) => m.kind == MemKind::Load || (m.kind == MemKind::Atomic && m.glc),
        _ => false,
    }
}
