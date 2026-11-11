//! The AMD target's engine on WDDM (decisions D13): the device, its guest window and
//! GPU queues, driven by the device thread (`agc::driver::device`).

use super::arena::{self, Arena, Bump};
use super::chain::{self, Stretch};
use super::program::{ProgramCache, TranslationTargets};
use super::runtime::{ScratchRing, TargetRuntime};
use super::side_table::{ChunkIndex, Root, SideTable, WaitCondition};
use super::state::{clear_values, report_once, ClearTarget, Stage, StateCache};
use super::submit::{self, AwaitedLabels, Segment, Settled, WaitLedger};
use super::surface::{host_view, CopyDirection, Fill, ImageView, SurfaceTable};
use super::{Amd, Baseline, Generation};
use crate::agc::driver::device::{Engine, Submission};
use crate::agc::driver::flips::{Event, Flips, Wait};
use crate::amd::descriptors::VSharpDescriptor;
use crate::amd::surface::formats::TexelFormat;
use crate::amd::surface::tiling::{SurfaceDesc, SurfaceUse, SW_LINEAR};
use super::gpu_timing::{GpuTiming, Stamp};
use super::image_reports::ImageReports;
use super::surface::CopyPass;
use crate::amd::surface::tiling::SurfaceLayout;
use crate::backend::{wddm::QueueKind, MemoryKind};
use crate::gpu::packets::PacketStream;
use crate::gpu::state_block::{self, AttributeRing, StateBlockConfig};
use crate::present::DisplayBufferDesc;
use crate::residency::{GuestRange, Queue, Reach, Residents};
use crate::scheduler::gpu::HostQueues;
use crate::agc::driver::anyps5::{Anyps5, GuestArena};
use crate::agc::driver::interface::{FlipInfo, GuestSparseRange};
use crate::agc::driver::memory::{GuestAccess, ImportRecord, Imports};
use crate::scheduler::gpu::{GpuQueues, LiveQueues};
use crate::residency::{GuestResidency, ResidencyDevice, ResidencyOptions, WddmWindow};
use crate::scheduler::{SubmitEvent, SubmitOutcome};
use crate::{Buffer, Device, Error, ErrorKind, Result};
use shader::ir::{DescriptorSource, DescriptorTable};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;


/// The guest's sparse (partially resident) ranges as the device has them mapped.
#[derive(Default)]
struct SparseMappings {
    ranges: Vec<GuestSparseRange>,
    generation: u64,
    mappings: u64,
    synchronized: bool,
}

impl SparseMappings {
    fn snapshot(arena: &Anyps5) -> (Vec<GuestSparseRange>, u64) {
        let mut ranges = Vec::new();
        loop {
            let (count, generation) = arena.sparse_ranges(&mut ranges);
            if count <= ranges.len() {
                ranges.truncate(count);
                return (ranges, generation);
            }
            ranges.resize(count, GuestSparseRange::default());
        }
    }

    fn range(sparse: &GuestSparseRange) -> GuestRange {
        GuestRange::new(sparse.address as u64, sparse.bytes as u64)
    }

    /// The parts of the mapped ranges `latest` no longer covers (both in address order).
    fn removed(&self, latest: &[GuestSparseRange]) -> Vec<GuestRange> {
        let mut removed = Vec::new();
        for previous in &self.ranges {
            let mut at = previous.address;
            let end = at + previous.bytes;
            for range in latest {
                if range.address >= end {
                    break;
                }
                if range.address + range.bytes <= at {
                    continue;
                }
                if at < range.address {
                    removed.push(GuestRange::new(at as u64, (range.address - at) as u64));
                }
                at = at.max(end.min(range.address + range.bytes));
            }
            if at < end {
                removed.push(GuestRange::new(at as u64, (end - at) as u64));
            }
        }
        removed
    }

    /// Whether the mapped ranges already cover all of `range`.
    fn registered(&self, range: &GuestSparseRange) -> bool {
        let mut at = range.address;
        let end = at + range.bytes;
        for previous in &self.ranges {
            if previous.address > at {
                break;
            }
            at = at.max(end.min(previous.address + previous.bytes));
            if at == end {
                return true;
            }
        }
        false
    }

    /// Applies the guest's latest ranges. A range the GPU still uses defers the whole change,
    /// checked before any page table changes: the next submission tries again.
    fn synchronize(&mut self, queues: &mut LiveQueues, arena: &Anyps5) -> Result<()> {
        let (latest, generation) = Self::snapshot(arena);
        let mappings = arena.generation();
        if self.synchronized && self.generation == generation && self.mappings == mappings {
            return Ok(());
        }
        let removed = self.removed(&latest);
        let changing = latest.iter().filter(|range| range.resident == 0 || !self.registered(range)).map(Self::range);
        for range in removed.iter().copied().chain(changing) {
            match queues.residency.require_range_idle(&mut queues.device, range) {
                Ok(()) => {}
                Err(error) if error.kind == ErrorKind::Busy => return Ok(()),
                Err(error) => return Err(error),
            }
        }
        for &range in &removed {
            queues.residency.sparse_reserve(&mut queues.device, range, false)?;
        }
        for range in &latest {
            queues.residency.sparse_reserve(&mut queues.device, Self::range(range), true)?;
        }
        for range in &latest {
            if range.resident != 0 {
                queues.residency.sparse_bind(&mut queues.device, Self::range(range))?;
            } else {
                queues.residency.sparse_unbind(&mut queues.device, Self::range(range))?;
            }
        }
        *self = Self { ranges: latest, generation, mappings, synchronized: true };
        Ok(())
    }
}

/// The memory a device's submissions use besides guest memory.
const CODE_BYTES: usize = 128 << 20;
const COMMANDS_BYTES: usize = 32 << 20;
const TRANSIENT_BYTES: usize = 256 << 20;
const TRANSIENT_BLOCK_BYTES: usize = 64 << 10;
const GDS_BYTES: usize = 0x2_0000;
const SCRATCH_BYTES: usize = 16 << 20;
const COPY_PARAM_BYTES: usize = 16 << 20;
const _: () = assert!(shader::gds::BUFFER_BYTES as usize <= GDS_BYTES);

fn range_of(buffer: &Buffer) -> Result<arena::Range> {
    let (cpu, bytes) = buffer.mapping().ok_or_else(|| Error::unsupported("runtime memory has no CPU mapping"))?;
    Ok(arena::Range { cpu, gpu: buffer.address(), bytes })
}

fn host_queue(queue: u32) -> Queue {
    Queue::from_guest(queue)
}

fn queue_kind(queue: Queue) -> QueueKind {
    match queue {
        Queue::Graphics => QueueKind::Graphics,
        Queue::Compute => QueueKind::Compute,
        Queue::ComputeQueue(index) => QueueKind::ComputeQueue(index),
    }
}

/// A submission on its way to the GPU, one segment at a time.
struct Running {
    serial: u64,
    /// Every recording its commands come from.
    tables: Vec<Arc<SideTable>>,
    segments: Vec<Segment>,
    next: usize,
    /// What the recording's first segment made resident; the later ones reuse it.
    residents: Option<Residents>,
    _leases: Vec<arena::Lease>,
    busy: Option<ResidencyWait>,
    /// The queue and serial the next segment waits for on the GPU (D25), for the stall report.
    after: Option<(Queue, u64)>,
}

/// A segment handed to the KMD, kept for the report of segments that stop finishing.
struct QueuedSegment {
    sequence: u64,
    queue: Queue,
    serial: u64,
    start: u64,
    end: u64,
    wait: Option<WaitCondition>,
    after: Option<(Queue, u64)>,
}

/// Segments kept for the stall report.
const QUEUED_LOG: usize = 1024;
/// How long no segment may finish, with some in flight, before the stall report.
const STALL_REPORT: std::time::Duration = std::time::Duration::from_secs(5);

struct ResidencyWait {
    since: std::time::Instant,
    attempts: u64,
    message: String,
}

/// A queued submission's commands, through the tail jumps they take, held until it runs: the
/// recordings' notes and the transient memory they reach (D35).
struct RecordingUse {
    stretches: Vec<Stretch>,
    tables: Vec<Arc<SideTable>>,
    leases: Vec<arena::Lease>,
}

#[derive(Default)]
struct ResidencyTimings {
    prepares: u64,
    prepare_nanos: u128,
    copy_checks: u64,
    copy_nanos: u128,
    target_nanos: u128,
    queued_nanos: u128,
}

/// Frozen source packets for a diagnostic submission, retained after guest memory is reused.
struct TracedSegment {
    sequence: u64,
    queue: Queue,
    words: Vec<u32>,
    ranges: Vec<super::trace::SourceRange>,
}

/// How many recordings may go by without a CPU write check when the title does not flip.
const RECORDINGS_PER_WRITE_CHECK: u32 = 16;

/// Bytes the host copy of a 3D image takes at most: whole blocks of 64 elements along each axis,
/// doubled for a mip chain.
fn volume_copy_bytes(desc: &SurfaceDesc) -> u64 {
    let axis = |extent: u32| u64::from(extent.next_multiple_of(64));
    let first = axis(desc.width) * axis(desc.height) * axis(desc.depth) * u64::from(desc.bytes_per_element);
    if desc.mip_levels > 1 { first * 2 } else { first }
}

/// What preparing a segment's surfaces gathers: the copy passes that run ahead of its commands,
/// the targets cleared with them, and what the segment's work then does to the surfaces.
#[derive(Default)]
struct Binds {
    touched: Vec<Touch>,
    passes: Vec<CopyPass>,
    clears: Vec<Arc<ClearTarget>>,
}

/// An image programs reported and the report table marks known: the view its T# describes.
struct KnownImage {
    base: u64,
    desc: SurfaceDesc,
    usage: SurfaceUse,
    first_level: u32,
    /// Bytes of the view's guest memory, and how many the runtime retiled to keep the copy
    /// current since a program last reported the image.
    bytes: u64,
    refreshed: u64,
}

/// What the runtime retiles of a known image without hearing of it again. A program that still
/// uses the image never reports it while it is marked known, so past this the mark is taken
/// back: an image still in use is reported again within a frame, and one no program uses any
/// more stops costing copies. Small enough that a large view of memory the title renders to
/// every frame is asked about after a few frames, large enough that a small texture the title
/// rewrites every frame practically never is.
const KNOWN_REFRESH_BYTES: u64 = 256 << 20;

/// What a segment's work does to the memory of a surface, in the order of its draws and
/// dispatches.
enum Touch {
    /// Renders to the host copy of the surface at this guest base.
    Rendered(u64),
    /// Stores to guest memory through a buffer: address and bytes.
    Stored(u64, u64),
    /// Fills guest memory through a buffer with one record.
    Filled(u64, u64, Fill),
    /// The host copy of the surface at this guest base was filled ahead of the segment, with
    /// what the fills before this point of it leave there.
    Consumed(u64),
}

/// Guest descriptors as a recording's submission reads them: a table a thousand draws share
/// is read once.
#[derive(Default)]
struct DescriptorMemo {
    pointers: HashMap<u64, Option<[u32; 2]>>,
    descriptors: HashMap<u64, Option<[u32; 8]>>,
}

/// The engine on a real device (decisions D13, D27).
pub struct Gpu {
    queues: LiveQueues,
    access: GuestAccess<Anyps5>,
    arena: Arc<Anyps5>,
    imports: ImportRecord<Anyps5>,
    flips: Arc<Flips>,
    sparse: SparseMappings,
    runtime: Arc<TargetRuntime>,
    chunks: ChunkIndex,
    pending: BTreeMap<u64, Result<RecordingUse>>,
    generation: &'static Generation,
    /// What each queue's guest registers hold after the submissions planned on it (D38).
    queue_states: BTreeMap<Queue, Baseline>,
    /// Stretches planned, draws or dispatches rewritten or left out, and state changes read that
    /// their recordings never read, since the last report.
    timed_stretches: u64,
    timed_rewrites: u64,
    timed_rejected: u64,
    timed_unread: u64,
    /// Buffer stores into surface memory, and views checked for CPU writes before they were
    /// bound as the GPU left their memory, since the last report.
    timed_stores: u64,
    timed_write_queries: u64,
    /// Images programs reported, and those of them whose copies took a copy pass.
    timed_reports: u64,
    timed_reports_bound: u64,
    timed_reports_unmapped: u64,
    timed_reports_refreshed: u64,
    /// The table translated programs report the images they find at run time through, the
    /// images it marks known, by T#, and the flip their reports were last taken at.
    image_reports: ImageReports,
    known_images: HashMap<[u32; 8], KnownImage>,
    reports_flip: u64,
    /// The latest segments handed to the KMD, when one last finished, and whether the segments
    /// that stopped finishing since were reported.
    queued_log: VecDeque<QueuedSegment>,
    last_outcome: std::time::Instant,
    stall_reported: bool,
    /// Stretches reported for works whose footprints hold other commands.
    stale_reports: u32,
    /// Buffers every submission uses: code, state and transient memory, rings, parameters.
    memory: Vec<Buffer>,
    preamble_graphics: Vec<u32>,
    preamble_compute: Vec<u32>,
    surfaces: SurfaceTable,
    awaited: AwaitedLabels,
    ledger: WaitLedger,
    running: BTreeMap<Queue, Running>,
    /// Rendering waits of submissions in progress: `None` once satisfied.
    rendering_waits: HashMap<(u64, u32, u32), Option<Wait>>,
    /// Segments handed to the KMD so far, in the order they were.
    sequence: u64,
    in_flight: BTreeSet<u64>,
    advanced: bool,
    /// The generation of the last batched collection of CPU writes over tracked surfaces.
    written_generation: u64,
    /// Flips seen, and the count the last CPU write check ran at.
    flip_count: u64,
    flips_checked: u64,
    /// Recordings prepared since the last CPU write check.
    recordings_since_check: u32,
    /// Whether the plan of the parked submissions was printed.
    segments_dumped: bool,
    /// The latest submissions received: serial, queue, address, dwords.
    history: VecDeque<(u64, u32, u64, u32)>,
    plans_logged: u32,
    /// What the last prepare made resident, with the mapping generation it saw.
    prepared: Option<(u64, Residents)>,
    /// Host copies known to be allocated: guest base and bytes.
    ensured_copies: HashSet<(u64, u64)>,
    /// The generation of the tracker's copies that `ensured_copies` covers; `None` when a 3D
    /// image's copy changed since.
    copies_ensured: Option<u64>,
    written_nanos: u128,
    scan_nanos: u128,
    /// Host copies of 3D images no copy pass reaches: guest base and bytes.
    volume_copies: BTreeMap<u64, u64>,
    /// Fast-clear metadata the title wrote whose clear no copy has had yet, by address, with the
    /// flip count then: carried out where a draw binds a target with that metadata (D41).
    cleared: HashMap<u64, u64>,
    /// Time spent since the last report: making memory resident, preparing surfaces, handing
    /// segments to the KMD, in nanoseconds; and the moment each step started being measured.
    resident_nanos: u128,
    resident_details: ResidencyTimings,
    surfaces_nanos: u128,
    queue_nanos: u128,
    timed_since: std::time::Instant,
    timed_segments: u64,
    timed_outcomes: u64,
    /// Reports in a row without a new segment submitted.
    stalled_reports: u32,
    dumped_at: u64,
    /// When the state of the queues was last printed.
    reported: std::time::Instant,
    submitted: u64,
    /// The hang trace's memory (`APS5_AGC_TRACE`): a slot per queue, two dwords each.
    trace: Option<Buffer>,
    trace_children: bool,
    /// Diagnostic submissions, including the packets actually handed to the GPU.
    traced: VecDeque<TracedSegment>,
    trace_reported: std::time::Instant,
    /// GPU time of each segment's parts (`APS5_AGC_GPU_TIMING`).
    gpu_timing: Option<GpuTiming>,
    display: Option<super::present::NativeDisplay>,
    /// The frames whose passes are listed (`APS5_AGC_PASSES=<first flip>,<flips>`).
    listed_frames: std::ops::Range<u64>,
    /// Guest memory whose image views are reported as the draws that name them are prepared
    /// (`APS5_AGC_WATCH=<address>,<bytes>`): what the driver knew of a place a fault hit.
    watched_memory: Option<(u64, u64)>,
    /// Surfaces shown in the display buffer's place, ten seconds each in turn from the first
    /// present (`APS5_AGC_SHOW=<address>,...`).
    shown: Vec<u64>,
    shown_since: Option<std::time::Instant>,
    /// The surface last reported, and what happened to it.
    shown_reported: (u64, u8),
}

/// The hexadecimal or decimal numbers of a comma-separated environment variable.
fn numbers(variable: &str) -> Vec<u64> {
    let parse = |number: &str| match number.trim().strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => number.trim().parse().ok(),
    };
    std::env::var(variable).map(|value| value.split(',').filter_map(parse).collect()).unwrap_or_default()
}

impl Gpu {
    /// Opens the device and its guest window, and publishes the runtime recordings use.
    pub fn open(amd: &Amd, flips: Arc<Flips>, arena_handle: Arc<Anyps5>) -> Result<Self> {
        let mut device = Device::open()?;
        let isa = crate::gpu::target::host_target(device.identity());
        amd.adopt(isa)?;
        let mut residency = GuestResidency::<WddmWindow>::open(&mut device, &ResidencyOptions::default())?;
        let imports = Imports::new(arena_handle.clone());
        let record = imports.record();
        residency.set_memory(Some(Box::new(imports)))?;
        device.use_coherent_host_imports();
        if let Err(error) = device.open_compute_queue() {
            eprintln!("[agc] no host compute queue: {}", error.message);
        }
        let code = device.allocate_executable(CODE_BYTES)?;
        let commands = device.allocate_kind(COMMANDS_BYTES, MemoryKind::Commands)?;
        let transient = device.allocate_kind(TRANSIENT_BYTES, MemoryKind::Commands)?;
        let gds = device.allocate_kind(GDS_BYTES, MemoryKind::Device)?;
        let scratch_graphics = device.allocate_kind(SCRATCH_BYTES, MemoryKind::Device)?;
        let scratch_compute = device.allocate_kind(SCRATCH_BYTES, MemoryKind::Device)?;
        let copy_params = device.allocate_kind(COPY_PARAM_BYTES, MemoryKind::Commands)?;
        let ring = device.allocate_attribute_ring()?;
        let attribute_ring = AttributeRing::new(ring.address)?;
        let config = StateBlockConfig::new(attribute_ring);
        let preamble_graphics = state_block::build(&config)?.into_words();
        let preamble_compute = state_block::build_compute(&config)?.into_words();
        let code_arena = Bump::new(range_of(&code)?);
        // The T# and V# conversions read this table.
        let lut: Vec<u32> = shader::descriptors::lut().chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().expect("chunks of 4"))).collect();
        let descriptor_lut = code_arena.put(&lut, 256)?;
        let scribble_arena = Bump::new(range_of(&commands)?);
        let scribble = scribble_arena.allocate(256, 256)?.1;
        // Shaders update the table with global atomics, through GL2 like the GDS buffer.
        let empty_stats: Vec<u8> = shader::mip_stats::empty_table().iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut mip_stats = device.allocate_kind(empty_stats.len() * 2, MemoryKind::Device)?;
        mip_stats.write(0, &empty_stats)?;
        mip_stats.write(empty_stats.len(), &empty_stats)?;
        let mip_stats_table = mip_stats.address();
        let mip_stats_template = mip_stats_table + empty_stats.len() as u64;
        let image_reports = ImageReports::new(&mut device)?;
        let reported = std::env::var_os("APS5_AGC_NO_IMAGE_REPORTS").is_none().then(|| image_reports.address());
        let targets = TranslationTargets { isa, descriptor_lut, gds_buffer: gds.address(), mip_stats_table, image_reports: reported };
        let programs = ProgramCache::new(targets, code_arena, Arc::new(crate::agc::guest::ProcessMemory));
        let (chunks, sender) = ChunkIndex::new();
        let gpu_timing = GpuTiming::from_environment(&mut device)?;
        let runtime = Arc::new(TargetRuntime {
            isa,
            commands: scribble_arena,
            programs,
            states: StateCache::default(),
            transient: Arc::new(Arena::new(range_of(&transient)?, TRANSIENT_BLOCK_BYTES)?),
            attribute_ring,
            scratch_graphics: ScratchRing { address: scratch_graphics.address(), bytes: SCRATCH_BYTES as u64 },
            scratch_compute: ScratchRing { address: scratch_compute.address(), bytes: SCRATCH_BYTES as u64 },
            chunks: sender,
            scribble,
            gds: gds.address(),
            mip_stats_table,
            mip_stats_template,
            stage_log: gpu_timing.as_ref().map(GpuTiming::stage_log),
            clear_targets: crate::agc::cache::Table::with_capacity(1 << 10),
        });
        let surfaces = SurfaceTable::new((copy_params.address(), COPY_PARAM_BYTES as u64));
        amd.shared().runtime.set(runtime.clone()).map_err(|_| Error::invalid("libSceAgcDriver: the target runtime is already published"))?;
        let mut queues = GpuQueues::new(residency, device);
        let trace = match std::env::var_os("APS5_AGC_TRACE") {
            Some(_) => {
                let mut buffer = queues.device.allocate_kind(4096, MemoryKind::Commands)?;
                buffer.fill(0)?;
                Some(buffer)
            }
            None => None,
        };
        if trace.is_some() {
            // Diagnostic readers of this process need the CPU alias of native IB2/code memory;
            // unlike guest imports, these allocations do not use the same CPU and GPU VA.
            for (name, buffer) in [("code", &code), ("commands", &commands), ("transient", &transient)] {
                if let Some((cpu, bytes)) = buffer.mapping() {
                    eprintln!("[trace-map] {name} gpu=0x{:x} cpu=0x{:x} bytes=0x{bytes:x}", buffer.address(), cpu as usize);
                }
            }
        }
        Ok(Self {
            queues,
            access: GuestAccess::new(arena_handle.clone()),
            arena: arena_handle,
            imports: record,
            flips,
            sparse: SparseMappings::default(),
            generation: Generation::of(isa)?,
            runtime,
            chunks,
            pending: BTreeMap::new(),
            queue_states: BTreeMap::new(),
            timed_stretches: 0,
            timed_rewrites: 0,
            timed_rejected: 0,
            timed_unread: 0,
            timed_stores: 0,
            timed_write_queries: 0,
            timed_reports: 0,
            timed_reports_bound: 0,
            timed_reports_unmapped: 0,
            timed_reports_refreshed: 0,
            image_reports,
            known_images: HashMap::new(),
            reports_flip: 0,
            queued_log: VecDeque::new(),
            last_outcome: std::time::Instant::now(),
            stall_reported: false,
            stale_reports: 0,
            memory: vec![code, commands, transient, gds, scratch_graphics, scratch_compute, copy_params, ring.buffer, mip_stats],
            preamble_graphics,
            preamble_compute,
            surfaces,
            awaited: AwaitedLabels::default(),
            ledger: WaitLedger::default(),
            running: BTreeMap::new(),
            rendering_waits: HashMap::new(),
            sequence: 0,
            in_flight: BTreeSet::new(),
            advanced: false,
            written_generation: 0,
            flip_count: 0,
            flips_checked: 0,
            recordings_since_check: 0,
            segments_dumped: false,
            history: VecDeque::new(),
            plans_logged: 0,
            prepared: None,
            ensured_copies: HashSet::new(),
            copies_ensured: None,
            written_nanos: 0,
            scan_nanos: 0,
            volume_copies: BTreeMap::new(),
            cleared: HashMap::new(),
            resident_nanos: 0,
            resident_details: ResidencyTimings::default(),
            surfaces_nanos: 0,
            queue_nanos: 0,
            timed_since: std::time::Instant::now(),
            timed_segments: 0,
            timed_outcomes: 0,
            stalled_reports: 0,
            dumped_at: 0,
            reported: std::time::Instant::now(),
            submitted: 0,
            trace,
            trace_children: std::env::var_os("APS5_AGC_TRACE_CHILDREN").is_some(),
            traced: VecDeque::new(),
            trace_reported: std::time::Instant::now(),
            gpu_timing,
            display: None,
            watched_memory: match numbers("APS5_AGC_WATCH")[..] {
                [address, bytes] => Some((address, bytes)),
                _ => None,
            },
            listed_frames: match numbers("APS5_AGC_PASSES")[..] {
                [first] => first..first + 1,
                [first, frames, ..] => first..first + frames,
                [] => 0..0,
            },
            shown: numbers("APS5_AGC_SHOW"),
            shown_since: None,
            shown_reported: (0, 0),
        })
    }

    fn events(outcome: SubmitOutcome, events: &mut Vec<Event>) {
        for event in outcome.events {
            if let Some(flow) = super::flow::diagnostics() {
                flow.completed(&event);
            }
            match event {
                SubmitEvent::Flip(flip) => {
                    events.push(Event::Flip(FlipInfo { handle: flip.handle, index: flip.index, mode: flip.mode, reserved: 0, argument: flip.argument }))
                }
                SubmitEvent::EopFlip(eop) | SubmitEvent::Eop(eop) => events.push(Event::Eop { queue: eop.queue, context: eop.context }),
                // The GPU wrote it: nothing to tell.
                SubmitEvent::Label(_) => {}
            }
        }
    }

    /// `(address, bytes)` of guest memory as a value, when that much is mapped.
    fn read_label(access: &mut GuestAccess<Anyps5>, address: u64, bytes: usize) -> Option<u64> {
        let mut value = [0u8; 8];
        access.read(address, &mut value[..bytes.min(8)]).ok()?;
        Some(u64::from_le_bytes(value))
    }

    /// Whether the guest's wait for the display buffers `waits` is over.
    fn flips_ready(&mut self, serial: u64, waits: &[(u32, i32)]) -> Result<bool> {
        for &(handle, index) in waits {
            let (handle, index) = (handle, index as u32);
            if self.queues.flip_pending(handle, index as i32) {
                return Ok(false);
            }
            let key = (serial, handle, index);
            if !self.rendering_waits.contains_key(&key) {
                let wait = self.flips.rendering_wait(handle, index).map_err(Error::from)?;
                self.rendering_waits.insert(key, Some(wait));
            }
            let Some(wait) = self.rendering_waits.get(&key).and_then(Option::as_ref) else {
                continue;
            };
            if !wait.wait_for(0).map_err(Error::from)? {
                return Ok(false);
            }
            self.rendering_waits.insert(key, None);
        }
        Ok(true)
    }

    /// CPU writes to guest memory make the host copies of the surfaces they touch stale. Asking
    /// about a range costs time for every page of it, so this runs once per flip and asks about
    /// the surfaces the CPU wrote recently and a slice of the others
    /// ([`SurfaceTracker::write_check_ranges`]).
    fn collect_guest_writes(&mut self) -> Result<()> {
        let mut ranges = self.surfaces.tracker.write_check_ranges();
        if ranges.is_empty() {
            return Ok(());
        }
        ranges.sort_unstable();
        let mut merged: Vec<GuestRange> = Vec::with_capacity(ranges.len());
        for (address, bytes) in ranges {
            let end = address + bytes;
            match merged.last_mut() {
                Some(last) if address <= last.end() => {
                    let end = end.max(last.end());
                    *last = GuestRange::new(last.address, end - last.address);
                }
                _ => merged.push(GuestRange::new(address, bytes)),
            }
        }
        if let Some((generation, written)) = self.queues.residency.written_since(&merged, self.written_generation)? {
            for range in written {
                self.surfaces.tracker.guest_written(range.address, range.bytes);
            }
            self.written_generation = generation;
        }
        Ok(())
    }

    /// A view about to be bound as the GPU left its memory, with nothing uploaded
    /// ([`SurfaceTracker::taken_as_rendered`]): what the CPU wrote there since it was last asked
    /// about is reported first, so a title that fills reused memory from the CPU gets its upload.
    fn confirm_rendered(&mut self, base: u64, desc: &SurfaceDesc, usage: SurfaceUse) -> Result<()> {
        let Some((address, bytes)) = self.surfaces.tracker.taken_as_rendered(base, desc, usage) else {
            return Ok(());
        };
        let started = std::time::Instant::now();
        self.timed_write_queries += 1;
        // Only what this collect finds: what earlier ones found was reported when they did.
        let found = match self.queues.residency.written_since(&[], 0)? {
            Some((generation, _)) => self.queues.residency.written_since(&[GuestRange::new(address, bytes)], generation)?,
            None => None,
        };
        match found {
            Some((_, written)) => {
                for range in written {
                    self.surfaces.tracker.guest_written(range.address, range.bytes);
                }
            }
            // No write watch to ask: the memory may hold anything.
            None => self.surfaces.tracker.guest_written(address, bytes),
        }
        self.written_nanos += started.elapsed().as_nanos();
        Ok(())
    }

    /// Brings the host copies a segment's work uses up to date: render targets and images it
    /// names, in a compute stream ahead of its commands. The first segment of a submission also
    /// collects the CPU's writes to guest memory.
    fn prepare_surfaces(&mut self, segment: &Segment, compute: bool, first: bool) -> Result<Vec<u32>> {
        if first {
            self.recordings_since_check += 1;
            if self.flip_count != self.flips_checked || self.recordings_since_check >= RECORDINGS_PER_WRITE_CHECK {
                self.flips_checked = self.flip_count;
                self.recordings_since_check = 0;
                let written_started = std::time::Instant::now();
                self.collect_guest_writes()?;
                self.written_nanos += written_started.elapsed().as_nanos();
            }
        }
        let scan_started = std::time::Instant::now();
        // A listed frame also says why each view it binds copies (`APS5_AGC_PASSES`).
        let listing = self.listed_frames.contains(&self.flip_count);
        let frame = self.flip_count;
        let runtime = self.runtime.clone();
        let mut stream = if compute { PacketStream::new_compute() } else { PacketStream::new() };
        let mut binds = Binds::default();
        let mut memo = DescriptorMemo::default();
        // Buffer descriptors are four dwords: read apart from the images' eight.
        let mut buffers = DescriptorMemo::default();
        let mut plain_stores: HashSet<(u64, u64)> = HashSet::new();
        let mut seen_records = HashSet::new();
        let mut seen_views: HashSet<[u32; 8]> = HashSet::new();
        if first && self.flip_count != self.reports_flip {
            self.reports_flip = self.flip_count;
            self.bind_reported_images(listing.then_some((frame, segment.start)), &mut binds)?;
        }
        for root in &segment.roots {
            let record = &root.record;
            let targets = if seen_records.insert(record.key) { record.targets.as_ref() } else { None };
            if let Some(targets) = targets {
                // A target whose metadata the title cleared since: the draw says what the target
                // is and what it clears to, so its copy is filled now (D41).
                if !self.cleared.is_empty() {
                    for (metadata, target) in clear_values(targets, &record.registers) {
                        if self.take_clear(metadata) {
                            binds.clears.push(Arc::new(target));
                        }
                    }
                }
                for color in &targets.colors {
                    let desc = *color.layout.desc();
                    if desc.swizzle_mode == SW_LINEAR {
                        continue;
                    }
                    let usage = if desc.volume { SurfaceUse::Volume } else { SurfaceUse::Thin2d };
                    self.confirm_rendered(color.base, &desc, usage)?;
                    if listing {
                        if let Some(why) = self.surfaces.tracker.explain(color.base, &desc, usage, 0) {
                            eprintln!("[copy] frame {frame} 0x{:x} color target 0x{:x}: {why}", segment.start, color.base);
                        }
                    }
                    let bound = self.surfaces.tracker.bind_passes(color.base, &desc, usage)?;
                    binds.touched.extend(bound.iter().filter(|pass| pass.direction == CopyDirection::Fill).map(|pass| Touch::Consumed(pass.guest_base)));
                    binds.passes.extend(bound);
                    binds.touched.push(Touch::Rendered(color.base));
                }
                if let Some(depth) = &targets.depth {
                    let desc = *depth.layout.desc();
                    self.confirm_rendered(depth.write_base, &desc, SurfaceUse::Depth)?;
                    if listing {
                        if let Some(why) = self.surfaces.tracker.explain(depth.write_base, &desc, SurfaceUse::Depth, 0) {
                            eprintln!("[copy] frame {frame} 0x{:x} depth target 0x{:x}: {why}", segment.start, depth.write_base);
                        }
                    }
                    let bound = self.surfaces.tracker.bind_passes(depth.write_base, &desc, SurfaceUse::Depth)?;
                    binds.touched.extend(bound.iter().filter(|pass| pass.direction == CopyDirection::Fill).map(|pass| Touch::Consumed(pass.guest_base)));
                    binds.passes.extend(bound);
                    if targets.depth_write {
                        binds.touched.push(Touch::Rendered(depth.write_base));
                    }
                }
            }
            for stage in &record.stages {
                for image in &stage.program.images {
                    // No copy can be prepared here for an image the program finds at run time:
                    // the program reports it, and its copy is ready from the next frame on
                    // ([`Self::bind_reported_images`]).
                    if matches!(image.source, DescriptorSource::Dynamic) {
                        report_once(&format!("program 0x{:x} reads an image it finds at run time", stage.program.host_address), 21);
                    }
                    for words in Self::image_words(&mut self.access, &mut memo, root, stage, &image.source, image.dword_count) {
                        if ImageView::is_null(&words) || (!image.written && !seen_views.insert(words)) {
                            continue;
                        }
                        let Ok(view) = ImageView::decode(&words) else { continue };
                        if self.watched_memory.is_some_and(|(address, bytes)| view.base >= address && view.base - address < bytes) {
                            report_once(
                                &format!(
                                    "watched: program 0x{:x} {} image 0x{:x} type {} format {} mode {} {}x{}x{} levels {}..={}, in place {}, source {}",
                                    stage.program.host_address, if image.written { "writes" } else { "reads" }, view.base, view.ty, view.format, view.swizzle_mode, view.width, view.height, view.depth,
                                    view.base_level, view.last_level, host_view(&view).0,
                                    match &image.source { DescriptorSource::UserSgprs { .. } => "user data", DescriptorSource::Table(_) => "a table", DescriptorSource::TableChoices(_) => "one of several tables", DescriptorSource::Dynamic => "found at run time" }
                                ),
                                22,
                            );
                        }
                        self.bind_image(&view, image.written, listing.then_some((frame, segment.start)), &mut binds)?;
                    }
                }
                // A store through a buffer into the memory of a surface is in guest layout (a
                // title clears its targets this way): the surface uploads it when next bound. It
                // is noted here for the views bound later in this segment, whose retile then
                // reads the memory as the store before left it, and again in order below.
                for buffer in stage.program.buffers.iter().filter(|buffer| buffer.written && !buffer.pointer) {
                    for words in Self::image_words(&mut self.access, &mut buffers, root, stage, &buffer.source, 4) {
                        let view = VSharpDescriptor::from_words([words[0], words[1], words[2], words[3]]);
                        let (address, stride) = (view.base_address, u64::from(view.stride));
                        let bytes = u64::from(view.num_records) * stride.max(1);
                        if address == 0 || bytes == 0 || plain_stores.contains(&(address, bytes)) {
                            continue;
                        }
                        // Records are filled only by a format as wide as they are. The clear of
                        // a compressed target writes its colour into the first element of every
                        // 256-byte block (radv's comp-to-single clear): that fills no surface,
                        // and the clear itself reaches the copy through the metadata (D41).
                        let fills = stride <= 1 || TexelFormat::from_gfx10(u32::from(view.format)).is_ok_and(|format| u64::from(format.bytes_per_element()) >= stride);
                        if !fills {
                            plain_stores.insert((address, bytes));
                            continue;
                        }
                        // A program that only fills, dispatched over every record with raw
                        // 32-bit components: the record it stores is in its user data, and the
                        // surfaces it fills need no upload to hold it.
                        let record = stage.program.fill.filter(|fill| matches!(buffer.source, DescriptorSource::UserSgprs { first } if first == fill.buffer)).and_then(|fill| {
                            let covered = matches!(root.groups, Some([groups, 1, 1]) if u64::from(groups) * 64 >= u64::from(view.num_records));
                            // 32, 32_32 and 32_32_32_32 formats store the dwords as they are.
                            let raw = stride == match view.format { 20..=22 => 4, 62..=64 => 8, 75..=77 => 16, _ => 0 };
                            let value = Self::image_words(&mut self.access, &mut buffers, root, stage, &DescriptorSource::UserSgprs { first: fill.value }, 4);
                            let &[dwords] = value.as_slice() else { return None };
                            (covered && raw).then(|| Fill::new([dwords[0], dwords[1], dwords[2], dwords[3]], stride)).flatten()
                        });
                        let noted = match record {
                            Some(fill) => self.surfaces.tracker.guest_filled(address, bytes, fill),
                            None => self.surfaces.tracker.guest_stored(address, bytes),
                        };
                        if !noted {
                            plain_stores.insert((address, bytes));
                            continue;
                        }
                        if listing {
                            eprintln!("[copy] frame {frame} 0x{:x} buffer {} 0x{address:x}+0x{bytes:x} into surface memory", segment.start, if record.is_some() { "fill" } else { "store" });
                        }
                        self.timed_stores += 1;
                        binds.touched.push(record.map_or(Touch::Stored(address, bytes), |fill| Touch::Filled(address, bytes, fill)));
                    }
                }
            }
        }
        self.scan_nanos += scan_started.elapsed().as_nanos();
        if let Some(timing) = &mut self.gpu_timing {
            timing.copies_planned(binds.passes.len(), binds.passes.iter().map(|pass| pass.dispatches.len()).sum());
        }
        for pass in &binds.passes {
            self.surfaces.emit_pass(&runtime, &mut stream, compute, pass)?;
            // What the CPU wrote so far to the levels this retile copies is in the memory it reads.
            if pass.direction == CopyDirection::ToHost {
                for (address, bytes) in self.surfaces.tracker.guest_level_ranges(pass.guest_base, pass.levels.clone()) {
                    self.queues.residency.absorb_writes(GuestRange::new(address, bytes));
                }
            }
        }
        // In the order the segment's work runs: a target a store cleared and a draw then rendered
        // to has its newest contents in the copy, one rendered to and then stored over has them
        // in guest memory.
        for touch in binds.touched {
            match touch {
                Touch::Rendered(base) => self.surfaces.tracker.host_written(base),
                Touch::Stored(address, bytes) => {
                    self.surfaces.tracker.guest_stored(address, bytes);
                }
                Touch::Filled(address, bytes, fill) => {
                    self.surfaces.tracker.guest_filled(address, bytes, fill);
                }
                Touch::Consumed(base) => self.surfaces.tracker.fill_applied(base),
            }
        }
        for target in &binds.clears {
            self.clear_copy(&runtime, &mut stream, compute, target)?;
        }
        // The segment's DMA writes guest memory: a host copy they overlap is stale from now on.
        for &(address, bytes) in &segment.dma_writes {
            self.surfaces.tracker.guest_written(address, bytes);
        }
        Ok(stream.into_words())
    }

    /// Brings the host copy of the image `view` up to date for work that reads it, or with
    /// `written` writes it: the copy passes and clears go to `binds`. A view the host addresses
    /// in place needs nothing. `listed` is the frame and segment of a listing
    /// (`APS5_AGC_PASSES`). `false` for a view left as it is because the guest memory it would
    /// be retiled from is not mapped.
    fn bind_image(&mut self, view: &ImageView, written: bool, listed: Option<(u64, u64)>, binds: &mut Binds) -> Result<bool> {
        if host_view(view).0 {
            return Ok(true);
        }
        // A program is given every image it can read, and may read none of some: the T# of a
        // texture a material leaves out names a type and no format.
        let desc = match view.surface_desc() {
            Ok(desc) => desc,
            Err(error) => {
                report_once(&format!("image at 0x{:x} type {} format {}: {}", view.base, view.ty, view.format, error.message), 24);
                return Ok(true);
            }
        };
        let first_level = view.first_reachable_level();
        self.confirm_rendered(view.base, &desc, view.surface_use())?;
        if let Some((frame, segment)) = listed {
            if let Some(why) = self.surfaces.tracker.explain(view.base, &desc, view.surface_use(), first_level) {
                let access = if written { "written image" } else { "read image" };
                eprintln!("[copy] frame {frame} 0x{segment:x} {access} 0x{:x} type {} format {}: {why}", view.base, view.ty, view.format);
            }
        }
        let residency = &self.queues.residency;
        let mapped = |ranges: &[(u64, u64)]| ranges.iter().all(|&(address, bytes)| residency.guest_backings(address, bytes).is_some());
        let bound = match self.surfaces.tracker.bind_readable_texture(view.base, &desc, view.surface_use(), first_level, mapped) {
            Ok(Some(bound)) => bound,
            Ok(None) => {
                report_once(&format!("image at 0x{:x} type {} format {} mode {}: the memory it would be retiled from is not mapped", view.base, view.ty, view.format, view.swizzle_mode), 25);
                return Ok(false);
            }
            // A 3D image in a thick swizzle mode: the conversion keeps the mode at the host
            // copy's address, the copy kernel cannot address it. Work on the GPU fills and reads
            // it there (the froxel grids of a clustered renderer), so the copy needs memory and
            // no copy pass.
            Err(_) if view.ty == 10 => {
                let bytes = volume_copy_bytes(&desc);
                if self.queues.residency.guest_backings(view.base, bytes).is_none() {
                    report_once(&format!("3D image at 0x{:x} format {} mode {}: its memory is not mapped", view.base, view.format, view.swizzle_mode), 25);
                    return Ok(false);
                }
                if self.volume_copies.insert(view.base, bytes) != Some(bytes) {
                    self.copies_ensured = None;
                }
                return Ok(true);
            }
            Err(error) => {
                report_once(&format!("image at 0x{:x}: {}", view.base, error.message), 3);
                return Ok(true);
            }
        };
        binds.touched.extend(bound.iter().filter(|pass| pass.direction == CopyDirection::Fill).map(|pass| Touch::Consumed(pass.guest_base)));
        binds.passes.extend(bound);
        if written {
            binds.touched.push(Touch::Rendered(view.base));
        }
        // A compressed image read before any draw bound it again: as its last target left it
        // cleared, when that target is what the view reads.
        if let Some(target) = view.meta.filter(|meta| self.cleared.contains_key(meta)).and_then(|meta| self.runtime.clear_targets.get(meta).map(|target| (meta, target))) {
            if target.1.surface == view.base && self.take_clear(target.0) {
                binds.clears.push(target.1);
            }
        }
        Ok(true)
    }

    /// The images programs found at run time and reported since the last flip
    /// ([`ImageReports`]): no work names them, so their host copies are brought up to date
    /// here, a frame after their first use, and kept so from then on: a known image whose
    /// guest memory changed is retiled again at the next flip.
    ///
    /// A program must not be left to report an image it uses in every wave: its waves cannot
    /// see each other's reports (a scalar load reads what its cache held when the dispatch
    /// began), so every one of them would store the report, into the same cache line. On mix
    /// that doubled the title's GPU time for 24 small textures it rewrites every frame. So a
    /// mark is only taken back when the image's memory is gone or [`KNOWN_REFRESH_BYTES`] were
    /// retiled for it unasked.
    fn bind_reported_images(&mut self, listed: Option<(u64, u64)>, binds: &mut Binds) -> Result<()> {
        let tracker = &self.surfaces.tracker;
        let stale: Vec<[u32; 8]> = self.known_images.iter().filter(|(_, known)| !tracker.view_current(known.base, &known.desc, known.usage, known.first_level)).map(|(words, _)| *words).collect();
        for words in stale {
            let known = self.known_images.get_mut(&words).expect("listed above");
            known.refreshed += known.bytes;
            let kept = match ImageView::decode(&words) {
                Ok(view) if known.refreshed <= KNOWN_REFRESH_BYTES => self.bind_image(&view, false, listed, binds)?,
                _ => false,
            };
            if kept {
                self.timed_reports_refreshed += 1;
            } else {
                self.known_images.remove(&words);
                self.image_reports.unmark(&words);
            }
        }
        for words in self.image_reports.take() {
            self.timed_reports += 1;
            // What needs no copy is marked all the same, and so is what describes no image (a
            // program loads the T# of a light it then finds switched off): a program has nothing
            // to wait for.
            let view = (!ImageView::is_null(&words)).then(|| ImageView::decode(&words).ok()).flatten().filter(|view| !host_view(view).0);
            let view = view.and_then(|view| view.surface_desc().ok().map(|desc| (view, desc)));
            if let Some((view, desc)) = &view {
                if self.watched_memory.is_some_and(|(address, bytes)| view.base >= address && view.base - address < bytes) {
                    report_once(
                        &format!(
                            "watched: a program found image 0x{:x} type {} format {} mode {} {}x{}x{} levels {}..={} at run time",
                            view.base, view.ty, view.format, view.swizzle_mode, view.width, view.height, view.depth, view.base_level, view.last_level
                        ),
                        23,
                    );
                }
                // What a reported image copies, said once: one that copies every frame shares
                // its memory with a target rendered every frame.
                if let Some(why) = self.surfaces.tracker.explain(view.base, desc, view.surface_use(), view.first_reachable_level()) {
                    report_once(&format!("a program found image 0x{:x} type {} format {} at run time: {why}", view.base, view.ty, view.format), 26);
                }
                let planned = binds.passes.len();
                // The report is a frame old, and the title may have given the image's memory up
                // since. Left unmarked, the image is reported again if a program still uses it.
                if !self.bind_image(view, false, listed, binds)? {
                    self.timed_reports_unmapped += 1;
                    continue;
                }
                self.timed_reports_bound += u64::from(binds.passes.len() != planned);
                let bytes = SurfaceLayout::new(desc).map_or(0, |layout| layout.size_bytes());
                self.known_images.insert(words, KnownImage { base: view.base, desc: *desc, usage: view.surface_use(), first_level: view.first_reachable_level(), bytes, refreshed: 0 });
            }
            self.image_reports.mark(&words);
        }
        Ok(())
    }

    /// Prints what each draw and dispatch of `segment` renders to, reads and writes: a frame's
    /// data flow, to find where a picture goes missing (`APS5_AGC_PASSES`).
    fn list_passes(&mut self, segment: &Segment, queue: Queue) {
        use std::fmt::Write as _;
        let frame = self.flip_count;
        let mut memo = DescriptorMemo::default();
        let unnamed = |source: &DescriptorSource| if matches!(source, DescriptorSource::Dynamic) { "found by the program" } else { "not bound" };
        for root in &segment.roots {
            let record = &root.record;
            let mut line = format!("[pass] frame {frame} {queue:?} 0x{:x}:", segment.start);
            match &record.targets {
                Some(targets) => {
                    for color in &targets.colors {
                        let desc = color.layout.desc();
                        let _ = write!(line, " color 0x{:x} {}x{} {:?} mask {:x}", color.base, desc.width, desc.height, color.format, color.write_mask);
                    }
                    if let Some(depth) = &targets.depth {
                        let desc = depth.layout.desc();
                        let _ = write!(line, " depth 0x{:x} {}x{}{}", depth.write_base, desc.width, desc.height, if targets.depth_write { " written" } else { "" });
                    }
                    if targets.colors.is_empty() && targets.depth.is_none() {
                        line.push_str(" no target");
                    }
                }
                None => line.push_str(" dispatch"),
            }
            for stage in &record.stages {
                let _ = write!(line, " | 0x{:x}", stage.program.host_address);
                for image in &stage.program.images {
                    let access = if image.written { "writes" } else { "reads" };
                    let views = Self::image_words(&mut self.access, &mut memo, root, stage, &image.source, image.dword_count);
                    if views.is_empty() {
                        let _ = write!(line, " {access} an image {}", unnamed(&image.source));
                    }
                    for words in views.iter().filter(|words| !ImageView::is_null(words)) {
                        match ImageView::decode(words) {
                            Ok(view) => {
                                let _ = write!(line, " {access} 0x{:x} type {} format {} {}x{}", view.base, view.ty, view.format, view.width, view.height);
                            }
                            Err(error) => {
                                let _ = write!(line, " {access} an image that does not decode ({})", error.message);
                            }
                        }
                    }
                }
                for buffer in stage.program.buffers.iter().filter(|buffer| buffer.written) {
                    let views = Self::image_words(&mut self.access, &mut memo, root, stage, &buffer.source, if buffer.pointer { 2 } else { 4 });
                    if views.is_empty() {
                        let _ = write!(line, " writes a buffer {}", unnamed(&buffer.source));
                    }
                    for words in views {
                        if buffer.pointer {
                            let _ = write!(line, " writes through pointer 0x{:x}", u64::from(words[0]) | u64::from(words[1]) << 32);
                        } else {
                            let base = u64::from(words[0]) | u64::from(words[1] & 0xffff) << 32;
                            let _ = write!(line, " writes buffer 0x{base:x} stride {} records {}", words[1] >> 16 & 0x3fff, words[2]);
                        }
                    }
                }
            }
            eprintln!("{line}");
        }
        for &(address, bytes) in &segment.dma_writes {
            eprintln!("[pass] frame {frame} {queue:?} 0x{:x}: DMA writes 0x{address:x}+0x{bytes:x}", segment.start);
        }
        for event in &segment.events {
            if let SubmitEvent::Flip(flip) = event {
                eprintln!("[pass] frame {frame} {queue:?} 0x{:x}: flip of buffer {} on handle {}", segment.start, flip.index, flip.handle);
            }
        }
    }

    /// The guest T#s `source` names, read from the root's user data and guest memory.
    fn image_words(access: &mut GuestAccess<Anyps5>, memo: &mut DescriptorMemo, root: &Root, stage: &Stage, source: &DescriptorSource, dword_count: u8) -> Vec<[u32; 8]> {
        let record = &root.record;
        let user = |sgpr: u8| -> Option<u32> {
            let element = u32::from(sgpr.checked_sub(stage.user_sgpr)?);
            if element >= stage.user_count {
                return None;
            }
            let dword = stage.user_base + element;
            let index = record.watched.binary_search(&dword).ok()?;
            root.values.get(index).copied()
        };
        let mut read_table = |table: &DescriptorTable| -> Option<[u32; 8]> {
            if user(table.root).is_none() || table.root.checked_add(1).and_then(user).is_none() {
                return None;
            }
            let address = table.bound_address(
                |sgpr| user(sgpr).unwrap_or(0),
                |at| {
                    *memo.pointers.entry(at).or_insert_with(|| {
                        let mut pair = [0u8; 8];
                        access.read(at, &mut pair).ok()?;
                        Some([u32::from_le_bytes(pair[..4].try_into().ok()?), u32::from_le_bytes(pair[4..].try_into().ok()?)])
                    })
                },
            )?;
            *memo.descriptors.entry(address).or_insert_with(|| {
                let mut bytes = [0u8; 32];
                access.read(address, &mut bytes[..usize::from(dword_count.min(8)) * 4]).ok()?;
                let mut words = [0u32; 8];
                for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(4)) {
                    *word = u32::from_le_bytes(chunk.try_into().ok()?);
                }
                Some(words)
            })
        };
        match source {
            DescriptorSource::UserSgprs { first } => {
                let mut words = [0u32; 8];
                for index in 0..dword_count.min(8) {
                    match user(*first + index) {
                        Some(value) => words[usize::from(index)] = value,
                        None => return Vec::new(),
                    }
                }
                vec![words]
            }
            DescriptorSource::Table(table) => read_table(table).into_iter().collect(),
            DescriptorSource::TableChoices(tables) => tables.iter().filter_map(|table| read_table(table)).collect(),
            DescriptorSource::Dynamic => Vec::new(),
        }
    }

    /// Prints the segments on the GPU when none has finished for a while: where each starts, what
    /// it waited for before it was queued (the label and what memory holds there now, the queue it
    /// waits for on the GPU), and how far each queue has run.
    fn report_stall(&mut self) {
        let mut lines = Vec::new();
        let mut queues = BTreeSet::new();
        for queued in self.queued_log.iter().filter(|queued| self.in_flight.contains(&queued.sequence)) {
            queues.insert(queued.queue);
            // SAFETY: guest command memory of a submission the title made and keeps until it ran.
            let packet: [u32; 4] = unsafe { std::ptr::read_volatile(queued.start as *const [u32; 4]) };
            let wait = queued.wait.map(|wait| {
                let mut value = [0u8; 8];
                let held = self.access.read(wait.address, &mut value[..usize::from(wait.bytes)]).map(|()| u64::from_le_bytes(value)).ok();
                format!("label 0x{:x} func {} ref 0x{:x} holds {held:x?}", wait.address, wait.function, wait.reference)
            });
            lines.push(format!(
                "#{} {:?} serial {} 0x{:x}+{:x} starts {packet:08x?} wait {} gpu-wait {:?}",
                queued.sequence, queued.queue, queued.serial, queued.start, (queued.end - queued.start) / 4, wait.unwrap_or_else(|| "none".into()), queued.after
            ));
        }
        for queue in queues.iter().chain(self.queued_log.iter().filter_map(|queued| queued.after.map(|(producer, _)| producer)).collect::<BTreeSet<_>>().iter()) {
            lines.push(format!("{queue:?} completed serial {}", ResidencyDevice::completed(&mut self.queues.device, *queue)));
        }
        let (state, address) = (self.queues.device.fault_state(), self.queues.device.fault_address());
        eprintln!(
            "[stall] no segment finished for {:.1} s with {} in flight (device state: {state}; fault address {address:x?}): {}",
            self.last_outcome.elapsed().as_secs_f64(), self.in_flight.len(), lines.join(" | ")
        );
        self.report_trace();
    }

    /// Prints what every parked queue waits for and which recordings can write it.
    fn report_parked(&mut self) {
        let mut lines = Vec::new();
        for (queue, running) in &self.running {
            if let Some(busy) = &running.busy {
                lines.push(format!("{queue:?}: submission {} segment {} residency busy for {:.1} s, {} attempts: {}", running.serial, running.next, busy.since.elapsed().as_secs_f64(), busy.attempts, busy.message));
                continue;
            }
            let Some(segment) = running.segments.get(running.next) else { continue };
            let Some(wait) = segment.wait else {
                lines.push(format!("{queue:?}: submission {} segment {}/{} waits only for flips {:?}", running.serial, running.next, running.segments.len(), segment.flip_waits));
                continue;
            };
            let mut value = [0u8; 8];
            let held = self.access.read(wait.address, &mut value[..usize::from(wait.bytes)]).map(|()| u64::from_le_bytes(value));
            let writers: Vec<String> = self
                .chunks
                .chunks()
                .flat_map(|chunk| chunk.table.writes.iter().filter(move |write| write.address == wait.address).map(move |write| format!("chunk 0x{:x} at 0x{:x} value 0x{:x}", chunk.start, write.at, write.value)))
                .collect();
            // SAFETY: guest command memory of a submission the title made.
            let packet: [u32; 7] = unsafe { std::ptr::read_volatile(segment.start as *const [u32; 7]) };
            lines.push(format!(
                "{queue:?}: submission {} segment {}/{} waits for 0x{:x} func {} ref 0x{:x}; the packet there now: {packet:08x?}; memory holds {held:x?}; {} recorded writes of it: {}",
                running.serial, running.next, running.segments.len(), wait.address, wait.function, wait.reference, writers.len(),
                writers.iter().rev().take(4).cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        if !lines.is_empty() {
            eprintln!("[agc] parked: {} ({} segments submitted, {} pending)", lines.join(" | "), self.submitted, self.queues.pending());
            if self.dumped_at != self.submitted {
                self.stalled_reports = 0;
                self.dumped_at = self.submitted;
            }
            self.stalled_reports += 1;
            if self.stalled_reports == 3 && !self.segments_dumped {
                self.segments_dumped = true;
                let history: Vec<String> = self
                    .history
                    .iter()
                    .map(|&(serial, queue, address, dwords)| {
                        let chunk = self.chunks.chunks().filter(|chunk| chunk.start <= address).last().map_or(0, |chunk| chunk.start);
                        format!("#{serial} q{queue:x} 0x{address:x}+{dwords:x} chunk 0x{chunk:x}")
                    })
                    .collect();
                eprintln!("[segments] history: {}", history.join(", "));
                for (queue, running) in &self.running {
                    for (index, segment) in running.segments.iter().enumerate() {
                        let writes: Vec<String> = segment.writes.iter().map(|write| format!("0x{:x}=0x{:x}@+{:x}", write.address, write.value, (write.at - segment.start) / 4)).collect();
                        eprintln!(
                            "[segments] {queue:?} {} [{index}{}] 0x{:x}+{:x} wait {:x?} flips {:?} writes {}",
                            running.serial,
                            if index == running.next { " NEXT" } else { "" },
                            segment.start,
                            (segment.end - segment.start) / 4,
                            segment.wait.map(|wait| (wait.address, wait.reference)),
                            segment.flip_waits,
                            writes.join(" ")
                        );
                    }
                }
            }
        }
    }

    /// Reads a diagnostic source from a native CPU alias or same-address guest memory.
    fn trace_words(memory: &[Buffer], access: &mut GuestAccess<Anyps5>, address: u64, dwords: usize) -> Result<Vec<u32>> {
        let bytes = dwords.checked_mul(4).ok_or_else(|| Error::invalid("trace source size overflows"))?;
        let end = address.checked_add(bytes as u64).ok_or_else(|| Error::invalid("trace source address overflows"))?;
        for buffer in memory {
            if address >= buffer.address() && end <= buffer.address() + buffer.size() as u64 {
                let (cpu, mapped) = buffer.mapping().ok_or_else(|| Error::unsupported("trace source has no native CPU alias"))?;
                let offset = (address - buffer.address()) as usize;
                if offset + bytes > mapped { return Err(Error::invalid("trace source exceeds its CPU alias")); }
                // SAFETY: the live allocation and checked mapped range; command memory is dword aligned.
                return Ok(unsafe { std::slice::from_raw_parts(cpu.add(offset).cast::<u32>(), dwords) }.to_vec());
            }
        }
        let mut bytes = vec![0u8; bytes];
        access.read(address, &mut bytes)?;
        Ok(bytes.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().expect("four bytes"))).collect())
    }

    /// A GPU page fault in the host copies: which surface is tracked at the address, which side
    /// of it was current, and whether its copy had memory. A shader turns every image descriptor
    /// it reads into a copy address, also one no draw's bindings named.
    fn report_fault(&mut self) {
        let Some(address) = self.queues.device.fault_address() else { return };
        let offset = crate::backend::va::HOST_COPY_OFFSET;
        if address < offset || address - offset >= offset {
            eprintln!("[fault] 0x{address:x} is not a host copy address");
            return;
        }
        let guest = address - offset;
        let allocated = self.queues.residency.host_copies(guest & !0xfff, 0x1000).is_some();
        let imported = self.queues.residency.guest_backings(guest & !0xfff, 0x1000).is_some();
        // 3D images have copy memory without being tracked.
        let volumes: Vec<String> = self.volume_copies.range(..=guest).next_back().into_iter().chain(self.volume_copies.range(guest + 1..).next())
            .map(|(&base, &bytes)| format!("0x{base:x}+0x{bytes:x}{}", if base <= guest && guest < base + bytes { " (holds the address)" } else { "" }))
            .collect();
        eprintln!(
            "[fault] 0x{address:x} is the host copy of guest 0x{guest:x}: {}; copy memory allocated there {allocated}; the guest memory there is imported {imported}; nearest 3D image copies {volumes:?}; {} copies ensured, for generation {:?} of {}",
            self.surfaces.tracker.describe(guest), self.ensured_copies.len(), self.copies_ensured, self.surfaces.tracker.copies_generation()
        );
    }

    /// Prints where the CP's progress marker is. A marker precedes packet issue; asynchronous
    /// shader work may fault later, and predication may skip the packet itself.
    fn report_trace(&mut self) {
        let Some(trace) = &self.trace else { return };
        let Some((memory, _)) = trace.mapping() else { return };
        // Guest compute queues include indices above seven (TBDR uses queue 12).
        for slot in 0..64usize {
            // SAFETY: the buffer's persistent mapping, which the runtime holds; read while the GPU
            // may still be writing it, hence volatile.
            let (at, sequence) = unsafe { (memory.cast::<u32>().add(slot * 2).read_volatile(), memory.cast::<u32>().add(slot * 2 + 1).read_volatile()) };
            if sequence == 0 {
                continue;
            }
            let Some(segment) = self.traced.iter().find(|segment| segment.sequence == u64::from(sequence)) else {
                eprintln!("[trace] slot {slot}: segment {sequence} at dword {at} (segment no longer kept)");
                continue;
            };
            let at = at as usize;
            if at >= segment.words.len() {
                eprintln!("[trace] slot {slot}: segment {sequence} has an invalid packet offset {at}");
                continue;
            }
            let queue = segment.queue;
            let Some(range) = segment.ranges.iter().find(|range| at >= range.first && at < range.first + range.dwords) else {
                eprintln!("[trace] slot {slot}: marker {at} has no frozen source range");
                continue;
            };
            let address = range.address + (at - range.first) as u64 * 4;
            let packet = &segment.words[at..(at + 8).min(range.first + range.dwords)];
            eprintln!(
                "[trace] {queue:?}: segment {sequence} last reached before source packet 0x{address:x} (snapshot dword {at}): opcode 0x{:02x} header 0x{:08x} source words {:08x?}; {} segments submitted, {} pending",
                packet[0] >> 8 & 0xff, packet[0], &packet[1..], self.submitted, self.queues.pending()
            );
            // A memory wait the queue may be stuck in: what it waits for, and what memory holds.
            if packet[0] >> 8 & 0xff == crate::amd::defs::pkt3::WAIT_REG_MEM && packet.len() >= 6 {
                let label = (u64::from(packet[2]) & !3) | u64::from(packet[3]) << 32;
                let mut value = [0u8; 4];
                let held = self.access.read(label, &mut value).map(|()| u32::from_le_bytes(value)).ok();
                eprintln!("[trace] {queue:?}: the wait is for 0x{label:x} (function {}, reference 0x{:x}, mask 0x{:x}); memory holds {held:x?}", packet[1] & 7, packet[4], packet[5]);
            }
            // Static source order in this IB, not execution history across calls or predicates.
            let mut walk = range.first;
            let mut listed = VecDeque::new();
            while walk <= at {
                let header = segment.words[walk];
                let Some(dwords) = super::packet_dwords(header) else { break };
                if walk + dwords > segment.words.len() { break; }
                if dwords > 1 {
                    let body = &segment.words[walk + 1..walk + dwords.min(5)];
                    listed.push_back(format!("{walk:x}:{:02x}{body:08x?}", header >> 8 & 0xff));
                    if listed.len() > 48 { listed.pop_front(); }
                }
                walk += dwords;
            }
            eprintln!("[trace] {queue:?}: source packets in IB 0x{:x} before marker: {}", range.address, listed.into_iter().collect::<Vec<_>>().join(" "));
            if segment.ranges.len() > 1 {
                match Self::trace_words(&self.memory, &mut self.access, range.address, range.dwords) {
                    Ok(current) => {
                        let saved = &segment.words[range.first..range.first + range.dwords];
                        if current != saved {
                            eprintln!("[trace] {queue:?}: original IB 0x{:x} changed since the diagnostic snapshot; copied execution may differ", range.address);
                        }
                    }
                    Err(error) => eprintln!("[trace] {queue:?}: original IB no longer readable: {}", error.message),
                }
            }
        }
    }

    /// The submission's commands through every tail jump they take, with the recordings that wrote
    /// them and leases on the transient memory those reach (D35).
    fn retain(&mut self, submission: &Submission) -> Result<RecordingUse> {
        let stretches = chain::follow(&mut self.chunks, submission.address, submission.dwords).map_err(|error| Error::new(error.kind, format!("submission {}: {}", submission.serial, error.message)))?;
        let mut tables: Vec<Arc<SideTable>> = Vec::new();
        for table in stretches.iter().flat_map(Stretch::tables) {
            if !tables.iter().any(|known| Arc::ptr_eq(known, table)) {
                tables.push(table.clone());
            }
        }
        let leases = tables.iter().flat_map(|table| table.blocks.iter()).map(|&index| self.runtime.transient.lease(index)).collect::<Result<Vec<_>>>()?;
        Ok(RecordingUse { stretches, tables, leases })
    }

    /// Plans the submission retained when the title queued it: works out the guest state along its
    /// commands from what the queue's registers hold (D38), and cuts it into segments.
    fn begin(&mut self, submission: &Submission) -> Result<Running> {
        let retained = match self.pending.remove(&submission.serial) {
            Some(retained) => retained?,
            None => self.retain(submission)?,
        };
        let RecordingUse { stretches, tables, leases } = retained;
        // Diagnostic runs check that each stretch is whole packets, before and after the walk.
        let checked = self.trace.is_some();
        let breaks: Vec<Option<(u64, u32)>> = if checked { stretches.iter().map(|stretch| chain::break_in(stretch.start, stretch.end)).collect() } else { Vec::new() };
        let state = self.queue_states.entry(host_queue(submission.queue)).or_default();
        let access = &mut self.access;
        let read = |address: u64, words: &mut [u32]| -> Result<()> {
            let mut bytes = vec![0u8; words.len() * 4];
            access.read(address, &mut bytes)?;
            for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(4)) {
                *word = u32::from_le_bytes(chunk.try_into().expect("four bytes"));
            }
            Ok(())
        };
        let mut walk = chain::Walk::new(state, &self.runtime, self.generation, read);
        walk.inspect = self.listed_frames.contains(&self.flip_count);
        let resolved = walk.resolve(&stretches);
        for (stretch, resolved) in stretches.iter().zip(&resolved).filter(|(_, resolved)| !resolved.stale.is_empty()) {
            if self.stale_reports < 8 {
                self.stale_reports += 1;
                let at = resolved.stale[0];
                eprintln!(
                    "[chain] submission {} stretch 0x{:x}..0x{:x}: works recorded at {:x?} hold other commands now; owners: {}; chunks near 0x{at:x}: {}",
                    submission.serial, stretch.start, stretch.end, resolved.stale,
                    stretch.owners.iter().map(|owner| format!("0x{:x}..0x{:x} table {:p}", owner.from, owner.to, Arc::as_ptr(&owner.table))).collect::<Vec<_>>().join(", "),
                    self.chunks.describe_near(at)
                );
            }
        }
        if checked {
            for (stretch, before) in stretches.iter().zip(&breaks) {
                if let Some((at, header)) = chain::break_in(stretch.start, stretch.end) {
                    eprintln!(
                        "[chain] submission {} stretch 0x{:x}..0x{:x} is not whole packets at 0x{at:x} (dword 0x{header:08x}); before the state walk: {before:x?}; {}",
                        submission.serial, stretch.start, stretch.end, chain::describe_break(stretch, at)
                    );
                }
            }
        }
        self.timed_stretches += resolved.len() as u64;
        self.timed_rewrites += resolved.iter().map(|stretch| stretch.rewritten as u64).sum::<u64>();
        self.timed_rejected += resolved.iter().map(|stretch| stretch.rejected as u64).sum::<u64>();
        self.timed_unread += resolved.iter().map(|stretch| stretch.unread as u64).sum::<u64>();
        let segments = submit::plan(&resolved, submission.queue, &self.awaited);
        if let Some(flow) = super::flow::diagnostics() {
            flow.planned(&resolved, submission.queue, submission.address, submission.dwords, &segments, &leases);
        }
        if self.plans_logged < 40 {
            self.plans_logged += 1;
            let labels: std::collections::BTreeSet<u64> = segments.iter().flat_map(|segment| segment.writes.iter().map(|write| write.address & !0xffff)).collect();
            eprintln!(
                "[plan] #{} q{:x} 0x{:x}+{:x}: {} stretches, {} segments, {} rewritten, {} stale; first wait {:x?}; label pages {:x?}",
                submission.serial, submission.queue, submission.address, submission.dwords, resolved.len(), segments.len(),
                resolved.iter().map(|stretch| stretch.rewritten).sum::<usize>(), resolved.iter().map(|stretch| stretch.stale.len()).sum::<usize>(),
                segments.iter().find_map(|segment| segment.wait.map(|wait| (wait.address, wait.reference))), labels
            );
        }
        Ok(Running { serial: submission.serial, tables, segments, next: 0, residents: None, _leases: leases, busy: None, after: None })
    }

    /// Whether the title cleared the metadata at `metadata` recently and no copy has had the
    /// clear: taken, so that it is carried out once. A clear no draw followed within a frame is
    /// dropped: the address has gone to other uses.
    fn take_clear(&mut self, metadata: u64) -> bool {
        self.cleared.remove(&metadata).is_some_and(|flip| self.flip_count <= flip + 1)
    }

    /// Fills the host copy of `target` with what it clears to. A clear that does not fit the
    /// surface tracked there is reported and left out: the address holds another surface now.
    fn clear_copy(&mut self, runtime: &TargetRuntime, stream: &mut PacketStream, compute: bool, target: &ClearTarget) -> Result<()> {
        match self.surfaces.tracker.clear_pass(target.surface, &target.texel) {
            Ok(Some(pass)) => {
                self.surfaces.emit_pass(runtime, stream, compute, &pass)?;
                self.surfaces.tracker.host_written(target.surface);
                if let Some(timing) = &mut self.gpu_timing {
                    timing.clear_planned();
                }
            }
            Ok(None) => {}
            Err(error) => report_once(&format!("fast clear of 0x{:x} left out: {}", target.surface, error.message), 14),
        }
        Ok(())
    }

    /// The fast clears the title made just before this segment and no draw of the segment took
    /// (`prepare_surfaces`): carried out now on the target the metadata last belonged to, while
    /// that target is still the surface tracked there; otherwise they wait for the draw that binds
    /// a target with that metadata. An address inside a tracked surface is no metadata any more.
    fn target_clears(&mut self, segment: &Segment, compute: bool) -> Result<Vec<u32>> {
        let runtime = self.runtime.clone();
        let mut stream = if compute { PacketStream::new_compute() } else { PacketStream::new() };
        for &metadata in &segment.clears {
            if !self.cleared.contains_key(&metadata) {
                continue;
            }
            if self.surfaces.tracker.holds_address(metadata) {
                self.cleared.remove(&metadata);
                continue;
            }
            let Some(target) = runtime.clear_targets.get(metadata) else { continue };
            if self.surfaces.tracker.tracks(target.surface, &target.desc) && self.take_clear(metadata) {
                self.clear_copy(&runtime, &mut stream, compute, &target)?;
            }
        }
        Ok(stream.into_words())
    }

    /// What the display conversion reads for `desc`: the current host render surface the GPU drew
    /// it into, or else the guest memory, in the guest's display layout (the CPU, a DMA or a buffer
    /// store wrote it). The GPU reads either; nothing goes through the CPU (D28).
    fn display_source(&self, desc: DisplayBufferDesc) -> Result<super::present::DisplaySource> {
        if let Some((address, surface)) = self.surfaces.tracker.display_surface(desc.address) {
            return Ok(super::present::DisplaySource { desc, address, layout: crate::present::host_kernel_layout(&desc, surface)?, bytes: surface.size_bytes(), in_guest: false });
        }
        report_once(
            &format!(
                "native display: no current host render surface for 0x{:x} ({}x{}, pixel format 0x{:x}, tiling mode {}, DCC at 0x{:x}); converting the guest memory",
                desc.address, desc.width, desc.height, desc.pixel_format, desc.tiling_mode, desc.dcc_address
            ),
            7,
        );
        let bytes = crate::present::display_buffer_size(&desc)? as u64;
        Ok(super::present::DisplaySource { desc, address: desc.address, layout: crate::present::guest_kernel_layout(&desc)?, bytes, in_guest: true })
    }

    /// `desc`, or in its place the surface `APS5_AGC_SHOW` names for now: what the title rendered
    /// somewhere else, seen in the window.
    fn shown_instead(&mut self, desc: DisplayBufferDesc) -> DisplayBufferDesc {
        if self.shown.is_empty() {
            return desc;
        }
        let since = *self.shown_since.get_or_insert_with(std::time::Instant::now);
        let address = self.shown[(since.elapsed().as_secs() / 10) as usize % self.shown.len()];
        let shown = DisplayBufferDesc { address, ..desc };
        let fits = self.surfaces.tracker.display_surface(address).map(|(_, surface)| crate::present::host_kernel_layout(&shown, surface));
        let (outcome, what) = match &fits {
            Some(Ok(_)) => (1, "shown in the display buffer's place".to_string()),
            Some(Err(error)) => (2, format!("cannot be shown ({}); showing the display buffer", error.message)),
            None => (3, "is not a current render surface; showing the display buffer".to_string()),
        };
        // Once for each surface's turn, and again when what happens to it changes.
        if self.shown_reported != (address, outcome) {
            self.shown_reported = (address, outcome);
            eprintln!("[agc] native display: the surface at 0x{address:x} {what}");
        }
        if outcome == 1 { shown } else { desc }
    }

    /// The buffers the display conversion of `source` reads: the host copies, or the guest
    /// memory's backings.
    fn display_targets<'a>(residency: &'a GuestResidency<WddmWindow>, source: &super::present::DisplaySource) -> Result<Vec<&'a Buffer>> {
        let targets = if source.in_guest { residency.guest_backings(source.desc.address, source.bytes) } else { residency.host_copies(source.desc.address, source.bytes) };
        targets.ok_or_else(|| Error::invalid(format!("native display: the source at 0x{:x} is not resident", source.desc.address)))
    }

    /// Gives every tracked surface and 3D image its host-copy memory. Nothing to do until the set
    /// of copies changes, which it rarely does from one segment to the next.
    fn ensure_copies(&mut self) -> Result<()> {
        let generation = self.surfaces.tracker.copies_generation();
        if self.copies_ensured == Some(generation) {
            return Ok(());
        }
        let surfaces = self.surfaces.tracker.copies().into_iter().map(|(base, _, bytes)| (base, bytes));
        let volumes = self.volume_copies.iter().map(|(&base, &bytes)| (base, bytes));
        for copy in surfaces.chain(volumes) {
            self.resident_details.copy_checks += 1;
            if !self.ensured_copies.contains(&copy) {
                self.queues.residency.ensure_host_copy(&mut self.queues.device, copy.0, copy.1)?;
                self.ensured_copies.insert(copy);
            }
        }
        self.copies_ensured = Some(generation);
        Ok(())
    }

    /// Hands `segment` to the KMD.
    fn submit_segment(&mut self, running: &mut Running, queue: Queue) -> Result<()> {
        let segment = running.segments[running.next].clone();
        let segment = &segment;
        let resident_started = std::time::Instant::now();
        // Everything GPU-visible stays resident until the guest's mappings change: a recording
        // reuses what the last prepare made, and a segment what its recording's first did. An
        // import released since (over the budget, a remap) makes it prepare again.
        let mapping_generation = self.arena.generation();
        let usable = |residents: &Residents, residency: &GuestResidency<WddmWindow>| residents.mapping_generation() == mapping_generation && residency.valid(residents);
        if !running.residents.as_ref().is_some_and(|residents| usable(residents, &self.queues.residency)) {
            let generation = self.arena.generation();
            let cached = self.prepared.take().filter(|(prepared, residents)| *prepared == generation && usable(residents, &self.queues.residency));
            let residents = match cached {
                Some((_, residents)) => residents,
                None => {
                    let ranges = [GuestRange::new(segment.start, segment.end - segment.start)];
                    self.resident_details.prepares += 1;
                    let started = std::time::Instant::now();
                    let residents = self.queues.residency.prepare(&mut self.queues.device, Reach { ranges: &ranges, computed_addresses: true });
                    self.resident_details.prepare_nanos += started.elapsed().as_nanos();
                    residents?
                }
            };
            self.prepared = Some((generation, residents.clone()));
            running.residents = Some(residents);
        }
        self.queues.residency.protect(running.residents.as_ref().expect("prepared above"))?;
        self.resident_nanos += resident_started.elapsed().as_nanos();
        self.submit_prepared_segment(running, queue).map_err(|error| {
            if error.kind == ErrorKind::Busy {
                Error::new(ErrorKind::System, format!("submission {} segment {} cannot yield after surface planning: {}", running.serial, running.next, error.message))
            } else {
                error
            }
        })
    }

    fn submit_prepared_segment(&mut self, running: &mut Running, queue: Queue) -> Result<()> {
        let segment = running.segments[running.next].clone();
        let segment = &segment;
        let compute = queue != Queue::Graphics;
        // The surfaces a segment uses are brought up to date when it is queued, after whatever it
        // waits for: work on other queues that fills a surface it uses has its state in the
        // tracker by then, and the retile here does not overwrite it.
        // The fast clears the title made before this segment wait for a draw that binds their
        // target; the segment's own draws come first (D41).
        for &metadata in &segment.clears {
            self.cleared.insert(metadata, self.flip_count);
        }
        let mut copies = if segment.roots.is_empty() && segment.dma_writes.is_empty() && running.next != 0 {
            Vec::new()
        } else {
            let surfaces_started = std::time::Instant::now();
            let copies = self.prepare_surfaces(segment, compute, running.next == 0)?;
            self.surfaces_nanos += surfaces_started.elapsed().as_nanos();
            copies
        };
        if !segment.clears.is_empty() {
            copies.extend(self.target_clears(segment, compute)?);
        }
        let listed = self.listed_frames.contains(&self.flip_count).then(|| (self.flip_count, format!("{queue:?}"), segment.start));
        if listed.is_some() {
            self.list_passes(segment, queue);
        }
        self.flip_count += segment.events.iter().filter(|event| matches!(event, SubmitEvent::Flip(_))).count() as u64;
        if let Some(timing) = &mut self.gpu_timing {
            for event in &segment.events {
                if let SubmitEvent::Flip(flip) = event {
                    timing.note_flip(flip.handle);
                }
            }
        }
        let trailer: Option<Vec<u32>> = None;
        let timing_slot = self.gpu_timing.as_mut().map(GpuTiming::open);
        if let (Some(timing), Some(slot)) = (&self.gpu_timing, timing_slot) {
            copies = timing.around_copies(slot, copies, compute)?;
        }
        let preamble = if compute { &self.preamble_compute } else { &self.preamble_graphics };
        let image = match &self.trace {
            Some(trace) => {
                let sequence = self.sequence + 1;
                let marker = trace.address() + queue.index() as u64 * 8;
                let mut recorded = submit::traced_ib(&copies, preamble, segment.user_data.as_ref(), compute, segment, marker, sequence, &running.tables)?;
                if self.trace_children && !compute && super::trace::has_root_chain(&recorded.source) {
                    // Consume the one-shot before reading/allocating child copies. IB2-only
                    // segments never accumulate snapshots in the non-reclaiming command arena.
                    self.trace_children = false;
                    let memory = &self.memory;
                    let access = &mut self.access;
                    recorded = submit::traced_tree_ib(&copies, preamble, segment.user_data.as_ref(), compute, segment, marker, sequence,
                        &self.runtime.commands, |address, dwords| Self::trace_words(memory, access, address, dwords))?;
                    eprintln!("[trace-tree] captured {} frozen-source IB snapshots for segment {sequence}; originals assumed immutable; markers report packet issue, not a faulting instruction", recorded.ranges.len());
                }
                self.traced.push_back(TracedSegment { sequence, queue, words: recorded.source, ranges: recorded.ranges });
                if self.traced.len() > 4096 {
                    self.traced.pop_front();
                }
                recorded.image
            }
            None => submit::host_ib(&copies, preamble, segment.user_data.as_ref(), compute, segment)?,
        };
        let mut images = vec![image];
        let has_trailer = trailer.is_some();
        if let (Some(timing), Some(slot)) = (&self.gpu_timing, timing_slot) {
            images.push(timing.stamp_image(slot, Stamp::AfterTitle, compute)?);
        }
        images.extend(trailer);
        if let (Some(timing), Some(slot), true) = (&self.gpu_timing, timing_slot, has_trailer) {
            images.push(timing.stamp_image(slot, Stamp::AfterWriteBack, compute)?);
        }
        let resident_started = std::time::Instant::now();
        self.ensure_copies()?;
        let residents = running.residents.as_ref().expect("set above");
        let elapsed = resident_started.elapsed().as_nanos();
        self.resident_nanos += elapsed;
        self.resident_details.copy_nanos += elapsed;
        let targeted = std::time::Instant::now();
        let targets = self.queues.residency.shared_targets(&self.queues.device, residents, &self.memory)?;
        self.resident_details.target_nanos += targeted.elapsed().as_nanos();
        let queued_at = std::time::Instant::now();
        let serial = match self.queues.device.queue_images(queue_kind(queue), &images, targets) {
            Ok(serial) => serial,
            Err(error) => {
                let state = self.queues.device.fault_state();
                let address = self.queues.device.fault_address();
                self.report_fault();
                self.report_trace();
                return Err(Error::new(error.kind, format!("{} (device state: {state}; fault address {address:x?}; {} segments submitted)", error.message, self.submitted)));
            }
        };
        self.queue_nanos += queued_at.elapsed().as_nanos();
        self.timed_segments += 1;
        let residency = &mut self.queues.residency;
        let recorded = std::time::Instant::now();
        residency.queued(residents, crate::residency::Submission { queue, serial });
        self.resident_details.queued_nanos += recorded.elapsed().as_nanos();
        self.ledger.note(segment, queue, serial);
        self.sequence += 1;
        let sequence = self.sequence;
        self.in_flight.insert(sequence);
        while self.queued_log.front().is_some_and(|queued| queued.sequence < sequence && !self.in_flight.contains(&queued.sequence)) || self.queued_log.len() == QUEUED_LOG {
            self.queued_log.pop_front();
        }
        self.queued_log.push_back(QueuedSegment { sequence, queue, serial, start: segment.start, end: segment.end, wait: segment.wait, after: running.after.take() });
        if let (Some(timing), Some(slot)) = (&mut self.gpu_timing, timing_slot) {
            timing.queued(sequence, slot, listed);
        }
        for block in running.tables.iter().flat_map(|table| table.blocks.iter()) {
            self.runtime.transient.submitted(*block, sequence);
        }
        self.queues.queued(queue, serial, sequence, segment.events.clone());
        self.submitted += 1;
        running.busy = None;
        Ok(())
    }
}

impl Engine for Gpu {
    fn adapter(&self) -> (u32, i32) {
        let identity = self.queues.device.identity();
        (identity.luid_low, identity.luid_high)
    }

    fn execute(&mut self, submission: &Submission, _events: &mut Vec<Event>) -> Result<bool> {
        self.advanced = false;
        let queue = host_queue(submission.queue);
        self.queues.device.open_queue(queue)?;
        if !self.queues.device.available(queue) {
            return Ok(false);
        }
        let mut running = match self.running.remove(&queue) {
            Some(running) => {
                if running.serial != submission.serial {
                    return Err(Error::invalid(format!("submission {} runs while {} is half way through its queue", submission.serial, running.serial)));
                }
                running
            }
            None => self.begin(submission)?,
        };
        while running.next < running.segments.len() {
            let segment = running.segments[running.next].clone();
            let ready = if segment.flip_waits.is_empty() { true } else { self.flips_ready(running.serial, &segment.flip_waits)? };
            if !ready {
                self.running.insert(queue, running);
                return Ok(false);
            }
            if let Some(wait) = &segment.wait {
                let access = &mut self.access;
                let device = &mut self.queues.device;
                let settled = self.ledger.settle(wait, queue, |address, bytes| Self::read_label(access, address, bytes), |writer| device.completed(writer));
                match settled {
                    Settled::Ready => {}
                    Settled::Parked => {
                        self.running.insert(queue, running);
                        return Ok(false);
                    }
                    Settled::After(producer, serial) => {
                        self.queues.device.wait_for_queue(queue_kind(queue), queue_kind(producer), serial)?;
                        running.after = Some((producer, serial));
                    }
                }
            }
            if !self.queues.device.available(queue) {
                self.running.insert(queue, running);
                return Ok(false);
            }
            if let Err(error) = self.submit_segment(&mut running, queue) {
                if error.kind == ErrorKind::Busy {
                    let busy = running.busy.get_or_insert_with(|| ResidencyWait { since: std::time::Instant::now(), attempts: 0, message: String::new() });
                    busy.attempts += 1;
                    busy.message = error.message;
                    if busy.since.elapsed() >= std::time::Duration::from_secs(60) {
                        return Err(Error::new(ErrorKind::System, format!("submission {} segment {} made no residency progress for 60 seconds after {} attempts: {}", running.serial, running.next, busy.attempts, busy.message)));
                    }
                    self.running.insert(queue, running);
                    return Ok(false);
                }
                let around: Vec<String> = running.segments.iter().enumerate().skip(running.next.saturating_sub(3)).take(6)
                    .map(|(index, segment)| format!("[{index}{}] 0x{:x}..0x{:x} wait {:x?} clears {:x?}", if index == running.next { " FAILED" } else { "" }, segment.start, segment.end, segment.wait.map(|wait| wait.address), segment.clears))
                    .collect();
                eprintln!("[agc] submission {} segment {}/{} failed; segments around it: {}", running.serial, running.next, running.segments.len(), around.join(" | "));
                return Err(error);
            }
            running.next += 1;
            self.advanced = true;
        }
        self.rendering_waits.retain(|&(serial, _, _), _| serial != submission.serial);
        Ok(true)
    }

    fn advanced(&self) -> bool {
        self.advanced
    }

    fn note(&mut self, submission: &Submission) {
        let retained = self.retain(submission);
        if let Ok(recording) = &retained {
            for table in &recording.tables {
                self.awaited.learn(table);
            }
        }
        self.pending.insert(submission.serial, retained);
        if self.history.len() == 48 {
            self.history.pop_front();
        }
        self.history.push_back((submission.serial, submission.queue, submission.address, submission.dwords));
    }

    fn poll(&mut self, events: &mut Vec<Event>) -> Result<()> {
        if self.trace.is_some() && self.trace_reported.elapsed() > std::time::Duration::from_secs(2) {
            self.trace_reported = std::time::Instant::now();
            self.report_trace();
        }
        if self.reported.elapsed() > std::time::Duration::from_secs(2) {
            self.reported = std::time::Instant::now();
            self.report_parked();
        }
        if self.timed_since.elapsed() > std::time::Duration::from_secs(1) {
            if let Some(flow) = super::flow::diagnostics() {
                flow.report();
            }
            let seconds = self.timed_since.elapsed().as_secs_f64();
            let queue = self.queues.device.take_queue_timings();
            let ms = |nanos: u128| nanos as f64 / 1e6 / seconds;
            let resident = std::mem::take(&mut self.resident_details);
            eprintln!("[residency-cpu] {seconds:.3} s: {} prepares {:.2} ms/s, cache/protection {:.2}, {} copy checks {:.2}, target lists {:.2}, queued bookkeeping {:.2}", resident.prepares, ms(resident.prepare_nanos), ms(self.resident_nanos.saturating_sub(resident.prepare_nanos + resident.copy_nanos)), resident.copy_checks, ms(resident.copy_nanos), ms(resident.target_nanos), ms(resident.queued_nanos));
            eprintln!("[submit-cpu] {seconds:.3} s: {} calls, {} target sets, {} targets, {} retired; hold {:.2} ms/s, queue poll {:.2}, IB {:.2}, encode {:.2}, KMT call {:.2}, mark {:.2}; all retirement {:.2}, batched paging {:.2}", queue.submissions, queue.target_sets, queue.targets, queue.retired_targets, ms(queue.hold_nanos), ms(queue.poll_nanos), ms(queue.ib_nanos), ms(queue.encode_nanos), ms(queue.kernel_nanos), ms(queue.mark_nanos), ms(queue.retire_nanos), ms(queue.paging_nanos));
            let bindings = self.queues.residency.take_binding_stats();
            eprintln!("[amm] {} physical imports ({} KiB new), {} KiB reused, {} KiB unmapped, {} KiB evicted; {} physical allocations cached", bindings.created, bindings.imported_bytes / 1024, bindings.reused_bytes / 1024, bindings.unmapped_bytes / 1024, bindings.evicted_bytes / 1024, self.queues.residency.import_count());
            eprintln!(
                "[agc] {seconds:.3} s: {} segments queued, {} finished; residency {:.1} ms/s, surfaces {:.1} ms/s (cpu writes {:.1}, scan {:.1}), KMD submit {:.1} ms/s; {} segments in flight; {} stretches planned, {} works rewritten for their state, {} left out, {} state changes read after their recordings",
                self.timed_segments, self.timed_outcomes, ms(self.resident_nanos), ms(self.surfaces_nanos), ms(self.written_nanos), ms(self.scan_nanos), ms(self.queue_nanos), self.queues.pending(),
                self.timed_stretches, self.timed_rewrites, self.timed_rejected, self.timed_unread
            );
            let counts = self.surfaces.tracker.take_counts();
            let mib = |bytes: u64| bytes as f64 / f64::from(1u32 << 20);
            eprintln!(
                "[surface] {seconds:.3} s: {} uploads ({:.1} MiB), {} write-backs ({:.1} MiB), {} views bound as rendered, {} buffer stores into surface memory, {} CPU write queries, {} fills ({:.1} MiB); {} images reported by programs, {} of them copied, {} no longer mapped, {} known, {} of those retiled again",
                counts.uploads, mib(counts.upload_bytes), counts.write_backs, mib(counts.write_back_bytes), counts.rendered_views, self.timed_stores, self.timed_write_queries, counts.fills, mib(counts.fill_bytes),
                self.timed_reports, self.timed_reports_bound, self.timed_reports_unmapped, self.known_images.len(), self.timed_reports_refreshed
            );
            self.timed_reports_refreshed = 0;
            self.timed_stores = 0;
            self.timed_write_queries = 0;
            self.timed_reports = 0;
            self.timed_reports_bound = 0;
            self.timed_reports_unmapped = 0;
            self.timed_stretches = 0;
            self.timed_rewrites = 0;
            self.timed_rejected = 0;
            self.timed_unread = 0;
            if let Some(timing) = &mut self.gpu_timing {
                let times = timing.take();
                let span = times.last.saturating_sub(times.first).max(1) as f64;
                // The clock's rate, from the span the timestamps of this interval cover.
                let ticks_per_second = span / seconds;
                let ms_per_second = |ticks: u64| ticks as f64 / ticks_per_second * 1e3 / seconds;
                eprintln!(
                    "[gpu-time] {seconds:.3} s: {} segments; surface work {:.1} ms/s in {} segments ({} copy passes, {} dispatches, {} target clears), title {:.1}, flip write-back {:.1}; clock {:.1} MHz",
                    times.segments, ms_per_second(times.copies), times.work_segments, times.copy_passes, times.copy_dispatches, times.clears, ms_per_second(times.title), ms_per_second(times.write_back), ticks_per_second / 1e6
                );
                timing.report_stages();
            }
            self.written_nanos = 0;
            self.scan_nanos = 0;
            self.resident_nanos = 0;
            self.surfaces_nanos = 0;
            self.queue_nanos = 0;
            self.timed_segments = 0;
            self.timed_outcomes = 0;
            self.timed_since = std::time::Instant::now();
        }
        let outcomes = match self.queues.poll() {
            Ok(outcomes) => outcomes,
            Err(error) => {
                self.report_fault();
                self.report_trace();
                return Err(error);
            }
        };
        if !outcomes.is_empty() || self.in_flight.is_empty() {
            self.last_outcome = std::time::Instant::now();
            self.stall_reported = false;
        } else if !self.stall_reported && self.last_outcome.elapsed() >= STALL_REPORT {
            self.stall_reported = true;
            self.report_stall();
        }
        for outcome in outcomes {
            self.timed_outcomes += 1;
            self.in_flight.remove(&outcome.serial);
            if let Some(timing) = &mut self.gpu_timing {
                timing.finished(outcome.serial);
            }
            Self::events(outcome, events);
        }
        let completed = self.in_flight.first().map_or(self.sequence, |oldest| oldest - 1);
        self.runtime.transient.advance(completed);
        Ok(())
    }

    fn busy(&self) -> bool {
        self.queues.pending() != 0
    }

    fn release(&mut self, range: GuestRange) -> bool {
        match self.queues.residency.release_range(&mut self.queues.device, range) {
            Ok(released) => {
                self.surfaces.tracker.remove_range(range.address, range.bytes);
                self.ensured_copies.clear();
                released != 0
            }
            // A lost device keeps the pages pinned; the guest's mapping change goes ahead.
            Err(error) if error.kind == ErrorKind::DeviceLost => {
                self.imports.abandon(range);
                true
            }
            Err(error) => {
                eprintln!("[agc] import release of {range} failed: {}", error.message);
                false
            }
        }
    }

    fn wait_idle(&mut self) -> Result<()> {
        self.queues.wait_idle()
    }

    fn synchronize(&mut self) -> Result<()> {
        self.sparse.synchronize(&mut self.queues, &self.arena)
    }

    fn close(&mut self) {
        let queues = &mut self.queues;
        if let Err(error) = queues.residency.trim(&mut queues.device, 0) {
            eprintln!("[agc] releasing imports at exit: {}", error.message);
        }
    }

    fn capture_overlay(&mut self, buffer: Option<DisplayBufferDesc>) -> Result<()> {
        let source = buffer.map(|desc| self.display_source(desc)).transpose()?;
        let mut display = match self.display.take() {
            Some(display) => display,
            None => super::present::NativeDisplay::new(&mut self.queues.device, self.runtime.isa)?,
        };
        let mut targets = match &source {
            Some(source) => Self::display_targets(&self.queues.residency, source)?,
            None => Vec::new(),
        };
        targets.extend(self.memory.iter());
        let result = display.capture_overlay(&mut self.queues.device, source, &targets);
        self.display = Some(display);
        result
    }

    fn present(&mut self, target: Option<crate::present::handoff::NativeTarget>, frame: crate::present::handoff::NativeFrame) -> Result<()> {
        if let Some(flow) = super::flow::diagnostics() {
            flow.present_requested();
        }
        let source = frame.buffer.map(|desc| self.shown_instead(desc)).map(|desc| self.display_source(desc)).transpose()?;
        let mut display = match self.display.take() {
            Some(display) => display,
            None => super::present::NativeDisplay::new(&mut self.queues.device, self.runtime.isa)?,
        };
        let mut targets = match &source {
            Some(source) => Self::display_targets(&self.queues.residency, source)?,
            None => Vec::new(),
        };
        targets.extend(self.memory.iter());
        let result = display.present(&mut self.queues.device, target, frame, source, &targets);
        self.display = Some(display);
        result
    }
}
