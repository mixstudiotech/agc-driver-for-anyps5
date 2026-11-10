//! What recordings and the device thread share once the device has opened: the memory state
//! blocks and programs live in, and the caches over them (decisions D13, D27). Built on the
//! device thread by `Gpu::open`; recording threads only allocate from it.

use super::arena::{Arena, Bump};
use super::program::ProgramCache;
use super::side_table::Chunk;
use super::state::StateCache;
use crate::gpu::state_block::AttributeRing;
use shader::TargetIsa;
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// Flat scratch memory for guest work with a scratch size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScratchRing {
    /// 256-byte aligned GPU address.
    pub address: u64,
    pub bytes: u64,
}

/// Everything recordings need from the device (D27).
pub struct TargetRuntime {
    pub isa: TargetIsa,
    /// State blocks, transition blocks and cached sequences: GL2-bypass, CPU-written.
    pub commands: Bump,
    pub programs: ProgramCache,
    pub states: StateCache,
    /// Per-recording transient data the GPU reads through recorded packets.
    pub transient: Arc<Arena>,
    pub attribute_ring: AttributeRing,
    pub scratch_graphics: ScratchRing,
    pub scratch_compute: ScratchRing,
    /// New chunks of command memory, for the device thread's index.
    pub chunks: Sender<Arc<Chunk>>,
    /// Where a release writes while its address is still to be patched in.
    pub scribble: u64,
    /// The guest's GDS as the translation emulates it (`shader::gds`): guest GDS byte `i` is the
    /// buffer's byte `i`.
    pub gds: u64,
    /// The texture LOD statistics pixel shaders update, and its contents with nothing counted.
    pub mip_stats_table: u64,
    pub mip_stats_template: u64,
    /// The title's debug markers, with timestamps (`APS5_AGC_GPU_TIMING`).
    pub stage_log: Option<Arc<super::gpu_timing::StageLog>>,
    /// The render targets with fast-clear metadata, by the metadata's address.
    pub clear_targets: crate::agc::cache::Table<super::state::ClearTarget>,
}
