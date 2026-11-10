//! The AMD target of the Agc front end (decisions D12–D14, D16, D17): Agc commands become
//! gfx11.5 (RDNA3.5) or gfx12 (RDNA4) PM4 in the title's command buffers, with registers and
//! packets from the definitions generated from Mesa (`amd::defs`).
//!
//! Register writes go into the buffer's guest state (D12); a write of graphics or compute user
//! data, which passes to the host unchanged at the same offsets, is also written in place. Every
//! footprint is filled exactly (D14).

pub mod arena;
pub mod chain;
pub mod commands;
pub mod draw;
#[cfg(all(feature = "wddm", target_os = "windows"))]
pub mod engine;
#[cfg(all(feature = "wddm", target_os = "windows"))]
mod present;
pub mod fields;
mod flow;
pub mod gpu_timing;
pub mod link;
pub mod patch;
pub mod program;
pub mod runtime;
pub mod side_table;
pub mod state;
pub mod submit;
mod trace;
pub mod surface;
pub mod sync;

use crate::agc::header;
use crate::agc::target::{DmaEnd, Draw, DrawKind, DrawModifier, Kind, Notice, Op, Patch, Queue, Recorder, Target};
use arena::Blocks;
use side_table::{ChangeKind, SideTable, StateChange};
use state::{IbRange, StateRecord};
use commands::{dwords, HostQueue};
use crate::amd::defs::{gfx10, gfx115, gfx12, pkt3};
use crate::agc::state::GuestRegisters;
use crate::ir::RegisterBank;
use draw::DrawParameters;
use crate::agc::guest::GuestMemoryReader;
use crate::{Error, Result};
use shader::TargetIsa;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

/// A one-dword NOP: PKT3 NOP with the reserved count 0x3fff, as radv pads command buffers.
pub const NOP_PAD: u32 = 0xffff_1000;
/// The largest NOP with a body: count 0x3ffe (0x3fff means the one-dword form).
const NOP_MAX_DWORDS: usize = 0x4000;

/// SH register dword of a byte offset in the persistent SH space.
const fn sh(offset: u32) -> u32 {
    (offset - 0xb000) / 4
}

/// User data: SH runs whose values pass to the host unchanged, element by element.
#[derive(Clone, Copy, Debug)]
struct UserData {
    /// First SH dword of each stage's run: PS, GS, HS, compute.
    first: [u32; 4],
}

const USER_DATA_LENGTHS: [u32; 4] = [32, 32, 32, 16];
const GUEST_STAGES_EN: u32 = (gfx10::VGT_SHADER_STAGES_EN::OFFSET - 0x2_8000) / 4;

const GUEST_USER_DATA: UserData = UserData {
    first: [
        sh(gfx10::SPI_SHADER_USER_DATA_PS_0::OFFSET),
        sh(gfx10::SPI_SHADER_USER_DATA_GS_0::OFFSET),
        sh(gfx10::SPI_SHADER_USER_DATA_HS_0::OFFSET),
        sh(gfx10::COMPUTE_USER_DATA_0::OFFSET),
    ],
};

/// What differs between the AMD hosts the target supports (D16).
#[derive(Clone, Copy, Debug)]
pub struct Generation {
    isa: TargetIsa,
    /// RDNA4: no GL1 and no GLM cache actions, DB flushes without the meta flush.
    gfx12: bool,
    user_data: UserData,
}

static GFX115: Generation = Generation {
    isa: TargetIsa::Gfx115,
    gfx12: false,
    user_data: UserData {
        first: [
            sh(gfx115::SPI_SHADER_USER_DATA_PS_0::OFFSET),
            sh(gfx115::SPI_SHADER_USER_DATA_GS_0::OFFSET),
            sh(gfx115::SPI_SHADER_USER_DATA_HS_0::OFFSET),
            sh(gfx115::COMPUTE_USER_DATA_0::OFFSET),
        ],
    },
};

static GFX12: Generation = Generation {
    isa: TargetIsa::Gfx12,
    gfx12: true,
    user_data: UserData {
        first: [
            sh(gfx12::SPI_SHADER_USER_DATA_PS_0::OFFSET),
            sh(gfx12::SPI_SHADER_USER_DATA_GS_0::OFFSET),
            sh(gfx12::SPI_SHADER_USER_DATA_HS_0::OFFSET),
            sh(gfx12::COMPUTE_USER_DATA_0::OFFSET),
        ],
    },
};

impl Generation {
    /// The generation of host `isa`.
    pub fn of(isa: TargetIsa) -> Result<&'static Self> {
        match isa {
            TargetIsa::Gfx115 => Ok(&GFX115),
            TargetIsa::Gfx12 => Ok(&GFX12),
            other => Err(Error::unsupported(format!(
                "Agc targets gfx11.5 (RDNA3.5) and gfx12 (RDNA4) hosts, not {}",
                other.name()
            ))),
        }
    }

    /// The host SH dword of `count` guest shader registers from `offset` when they are all user
    /// data of one stage.
    pub fn host_user_data(&self, bank: RegisterBank, offset: u32, count: usize) -> Option<u32> {
        if bank != RegisterBank::Shader || count == 0 {
            return None;
        }
        (0..4).find_map(|stage| {
            let first = GUEST_USER_DATA.first[stage];
            (offset >= first && offset + count as u32 <= first + USER_DATA_LENGTHS[stage])
                .then(|| self.user_data.first[stage] + (offset - first))
        })
    }
}

/// Fills `out` with NOP packets.
pub fn nop(out: &mut [u32]) {
    let mut rest = out;
    while !rest.is_empty() {
        let chunk = rest.len().min(NOP_MAX_DWORDS);
        let (packet, tail) = rest.split_at_mut(chunk);
        if chunk == 1 {
            packet[0] = NOP_PAD;
        } else {
            packet[0] = header(pkt3::NOP, chunk as u32, 0).expect("NOP sizes are in range");
            packet[1..].fill(0);
        }
        rest = tail;
    }
}

/// Dwords of the type-3 packet with this header (1 for the one-dword NOP).
pub fn packet_dwords(header: u32) -> Option<usize> {
    if header >> 30 != 3 {
        return None;
    }
    let count = (header >> 16) & 0x3fff;
    let opcode = (header >> 8) & 0xff;
    if opcode == pkt3::NOP && count == 0x3fff {
        return Some(1);
    }
    Some(count as usize + 2)
}

/// The target for this host: gfx11.5 unless `APS5_AGC_TARGET` names `gfx12`, reading guest
/// memory in place (`agc::guest::ProcessMemory`). The device sets the real generation when it
/// opens.
pub fn default_target() -> Box<dyn Target> {
    let isa = match std::env::var("APS5_AGC_TARGET").as_deref() {
        Ok("gfx12") => TargetIsa::Gfx12,
        _ => TargetIsa::Gfx115,
    };
    let memory = Arc::new(crate::agc::guest::ProcessMemory);
    Box::new(Amd::new(isa, memory).expect("gfx11.5 and gfx12 are supported targets"))
}

/// The AMD target.
pub struct Amd {
    /// One of the generations above: the host's, once the device has opened.
    generation: AtomicPtr<Generation>,
    memory: Arc<dyn GuestMemoryReader + Send + Sync>,
    shared: Arc<AmdShared>,
}

impl Amd {
    /// The target for host `isa` (gfx11.5 or gfx12), reading guest memory through `memory`.
    pub fn new(isa: TargetIsa, memory: Arc<dyn GuestMemoryReader + Send + Sync>) -> Result<Self> {
        let generation = AtomicPtr::new(std::ptr::from_ref(Generation::of(isa)?).cast_mut());
        Ok(Self { generation, memory, shared: Arc::default() })
    }

    fn generation(&self) -> &'static Generation {
        // SAFETY: only the statics of `Generation::of` are ever stored.
        unsafe { &*self.generation.load(Ordering::Acquire) }
    }

    pub fn isa(&self) -> TargetIsa {
        self.generation().isa
    }

    pub fn shared(&self) -> &Arc<AmdShared> {
        &self.shared
    }

    /// Records for the host the device turned out to be, from here on.
    pub fn adopt(&self, isa: TargetIsa) -> Result<()> {
        self.generation.store(std::ptr::from_ref(Generation::of(isa)?).cast_mut(), Ordering::Release);
        Ok(())
    }
}

/// Dwords of a footprint of `kind` (D14).
pub fn footprint(kind: Kind) -> Result<u32> {
    Ok(match kind {
        Kind::Nop { dwords } => dwords,
        Kind::SetRegisters { count, .. } => count + 2,
        // The console's worst case, one SET_*_REG per register (`numRegs * 12` bytes).
        Kind::SetRegisterPairs { count, .. } => count
            .checked_mul(3)
            .ok_or_else(|| Error::invalid("register list size overflow"))?,
        Kind::SetRegistersIndirect { .. } => 5,
        // A NOP with the NUL-terminated text.
        Kind::Marker { text: Some(bytes) } => (3 + bytes / 4).max(gpu_timing::marker_dwords()),
        Kind::Marker { text: None } => gpu_timing::marker_dwords().max(2),
        Kind::Flip => dwords::FLIP,
        Kind::WaitFlipDone => dwords::WAIT_FLIP_DONE,
        Kind::ResetQueue => dwords::RESET_QUEUE,
        Kind::Rewind => dwords::REWIND,
        Kind::LodStats => dwords::LOD_STATS,
        Kind::Acquire => dwords::ACQUIRE,
        Kind::Release => dwords::RELEASE,
        Kind::Event { event } if event & !1 == crate::gpu::packets::event::PIXEL_PIPE_STAT_CONTROL => dwords::OCCLUSION,
        Kind::Event { event } => match sync::EventPlan::of(event) {
            sync::EventPlan::TimestampFlush { .. } => dwords::EVENT,
            sync::EventPlan::Forward { .. } | sync::EventPlan::Unsupported(_) => 2,
        },
        Kind::WriteData { dwords } => dwords + 4,
        Kind::CopyData => dwords::COPY_DATA,
        Kind::DmaData => dwords::DMA_DATA,
        Kind::AtomicMem => dwords::ATOMIC_MEM,
        Kind::Wait { bits64: false } => dwords::WAIT_32,
        Kind::Wait { bits64: true } => dwords::WAIT_64,
        Kind::CondWrite => dwords::COND_WRITE,
        Kind::Semaphore => dwords::SEMAPHORE,
        Kind::SyncParser => dwords::SYNC_PARSER,
        Kind::Jump => dwords::JUMP,
        Kind::CondExec => dwords::COND_EXEC,
        Kind::IndexBase => dwords::INDEX_BASE,
        Kind::IndexCount => dwords::INDEX_COUNT,
        Kind::IndexType => dwords::INDEX_TYPE,
        Kind::Instances => dwords::INSTANCES,
        Kind::IndirectBase => dwords::INDIRECT_BASE,
        Kind::Predication => dwords::PREDICATION,
        Kind::Draw(kind) => match kind {
            DrawKind::Indexed => draw::dwords::INDEXED,
            DrawKind::IndexedOffset => draw::dwords::INDEXED_OFFSET,
            DrawKind::Auto => draw::dwords::AUTO,
            DrawKind::Indirect => draw::dwords::INDIRECT,
            DrawKind::IndirectMulti => draw::dwords::INDIRECT_MULTI,
        },
        Kind::Dispatch => draw::dwords::DISPATCH,
        Kind::DispatchIndirect => draw::dwords::DISPATCH_INDIRECT,
    })
}

impl Target for Amd {
    fn footprint(&self, kind: Kind) -> Result<u32> {
        footprint(kind)
    }

    #[cfg(all(feature = "wddm", target_os = "windows"))]
    fn open(&self, flips: Arc<crate::agc::driver::flips::Flips>, arena: Arc<crate::agc::driver::anyps5::Anyps5>) -> Result<Box<dyn crate::agc::driver::device::Engine>> {
        Ok(Box::new(engine::Gpu::open(self, flips, arena)?))
    }

    unsafe fn command_dwords(&self, packet: *const u32) -> Option<u32> {
        // SAFETY: the caller's contract.
        packet_dwords(unsafe { *packet }).map(|dwords| dwords as u32)
    }

    fn record(&self) -> Box<dyn Recorder> {
        Box::new(AmdRecorder::with_generation(*self.generation(), self.memory.clone(), self.shared.clone()))
    }

    fn write(&self, op: &Op<'_>, out: &mut [u32]) -> Result<()> {
        let address = out.as_mut_ptr() as u64;
        let mut recorder = AmdRecorder::with_generation(*self.generation(), self.memory.clone(), self.shared.clone());
        recorder.standalone = true;
        recorder.encode(Queue::Any, op, out, address)
    }

    fn fill(&self, out: &mut [u32]) {
        nop(out);
    }

    unsafe fn patch(&self, footprint: *mut u32, patch: Patch) -> Result<()> {
        // SAFETY: the caller's contract.
        unsafe { patch::Recorded::at(footprint)?.apply(patch) }?;
        if let Patch::JumpTarget { target, dwords } = patch {
            if let Some(flow) = flow::diagnostics() {
                flow.patched_chain(footprint as u64, target, dwords);
            }
        }
        if self.shared.trace_recording() {
            eprintln!("[record-patch] command=0x{:x} {patch:?}", footprint as u64);
        }
        Ok(())
    }

    unsafe fn predicate_range(&self, start: *mut u32, end: *const u32, predicate: bool) -> Result<()> {
        // SAFETY: the caller's contract.
        unsafe { patch::Commands::between(start, end)? }.predicate(predicate)
    }
}

/// The guest state a queue's registers hold: what the last submission on it left, and what
/// submission works out at each point of the commands it runs (D38).
#[derive(Clone)]
pub struct Baseline {
    pub guest: GuestRegisters,
    pub parameters: DrawParameters,
}

impl Default for Baseline {
    fn default() -> Self {
        Self { guest: GuestRegisters::default(), parameters: DrawParameters::default() }
    }
}

/// The user data `guest` holds: PS, GS, HS and compute, in that order.
pub fn user_data(guest: &GuestRegisters) -> side_table::UserData {
    let mut values = [0; side_table::USER_DATA_REGISTERS];
    let mut at = 0;
    for (stage, first) in GUEST_USER_DATA.first.iter().enumerate() {
        for index in 0..USER_DATA_LENGTHS[stage] as usize {
            values[at] = guest.get_or_zero(RegisterBank::Shader, first + index as u32);
            at += 1;
        }
    }
    values
}

/// A guest register write; VGT_INDEX_TYPE also sets the index type draws use.
fn apply_register(guest: &mut GuestRegisters, parameters: &mut DrawParameters, bank: RegisterBank, offset: u32, value: u32) {
    guest.write(bank, offset, value);
    if bank == RegisterBank::UserConfig && offset == crate::ir::GUEST_VGT_INDEX_TYPE {
        parameters.index_type = value & 3;
    }
}

/// A guest register list of flattened (offset, value) pairs, as SET_*_REG_INDIRECT applies it.
fn apply_list(guest: &mut GuestRegisters, parameters: &mut DrawParameters, bank: RegisterBank, list: &[u32]) -> Result<()> {
    guest.write_list(bank, list)?;
    if bank == RegisterBank::UserConfig {
        for pair in list.chunks_exact(2) {
            if crate::ir::register_offset(pair[0])? == crate::ir::GUEST_VGT_INDEX_TYPE {
                parameters.index_type = pair[1] & 3;
            }
        }
    }
    Ok(())
}

/// The state slot of a draw: a NOP when `state` is what the CP already has (`bound`), a call of
/// the registers that differ when the state before is known, of them all otherwise.
fn write_slot(runtime: &runtime::TargetRuntime, bound: Option<&Arc<StateRecord>>, state: &Arc<StateRecord>, slot: &mut [u32]) -> Result<()> {
    let target = match bound {
        Some(bound) if bound.key == state.key => IbRange::default(),
        Some(bound) => runtime.states.transition(runtime, bound, state)?,
        None => state.block.ok_or_else(|| Error::invalid("a draw state without a register block"))?,
    };
    if target.dwords == 0 {
        nop(slot);
    } else {
        indirect_buffer(slot, target)?;
    }
    Ok(())
}

/// A draw's footprint with `state`, after `bound`.
fn write_draw(generation: &Generation, runtime: &runtime::TargetRuntime, bound: Option<&Arc<StateRecord>>, state: &Arc<StateRecord>, draw: Draw, modifier: &DrawModifier, parameters: &DrawParameters, out: &mut [u32]) -> Result<()> {
    let (slot, rest) = out.split_at_mut(draw::dwords::SLOT as usize);
    write_slot(runtime, bound, state, slot)?;
    draw::encode_draw(generation, rest, state.primitive_type, draw, modifier, parameters)
}

/// A dispatch's footprint with `state`, after `bound`: the compute state inline (no IB2 on MEC).
fn write_dispatch(bound: Option<&Arc<StateRecord>>, state: &Arc<StateRecord>, queue: HostQueue, groups: Option<[u32; 3]>, indirect: (u64, u32), modifier: u32, parameters: &DrawParameters, out: &mut [u32]) -> Result<()> {
    let (slot, rest) = out.split_at_mut(draw::dwords::COMPUTE_STATE as usize);
    if bound.is_some_and(|bound| bound.key == state.key) {
        nop(slot);
    } else {
        let packets = state.compute_packets.as_ref().ok_or_else(|| Error::invalid("a dispatch state without its packets"))?;
        slot.copy_from_slice(packets);
    }
    draw::encode_dispatch(queue, rest, groups, indirect, modifier, parameters)
}

/// What every recording of an `Amd` target shares: the device's runtime once it has opened,
/// and what each queue kind's last submission left in its registers.
#[derive(Default)]
pub struct AmdShared {
    pub runtime: std::sync::OnceLock<Arc<runtime::TargetRuntime>>,
    baseline: [std::sync::Mutex<Option<Arc<Baseline>>>; 2],
    trace_recording: std::sync::OnceLock<bool>,
}

impl AmdShared {
    fn trace_recording(&self) -> bool {
        *self.trace_recording.get_or_init(|| std::env::var_os("APS5_AGC_RECORD_TRACE").is_some())
    }

    fn slot(&self, queue: HostQueue) -> &std::sync::Mutex<Option<Arc<Baseline>>> {
        &self.baseline[usize::from(queue == HostQueue::Compute)]
    }
}

/// The AMD target's state for one command buffer.
pub struct AmdRecorder {
    generation: Generation,
    memory: Arc<dyn GuestMemoryReader + Send + Sync>,
    shared: Arc<AmdShared>,
    runtime: Option<Arc<runtime::TargetRuntime>>,
    /// The guest registers as the buffer's commands leave them (D12).
    guest: GuestRegisters,
    /// The index buffer, instance count and indirect bases as the commands leave them.
    parameters: DrawParameters,
    /// Register values the title writes into the buffer after recording (D12).
    deferred_values: bool,
    /// The queue the buffer is for, from its first `Dcb*` or `Acb*` command; `Cb*` commands
    /// before that one take the graphics forms.
    queue: Option<HostQueue>,
    /// How many changes, the last ones of the table's `waiting`, the next draw or dispatch
    /// applies: none until a register list is recorded, so that other state goes straight into
    /// `guest`.
    waiting: usize,
    /// A register list as `settle` read it last.
    list: Vec<u32>,
    /// The recording's notes for submission (D27).
    table: Arc<SideTable>,
    /// The chunk the commands are going into, which the device thread's index holds.
    chunk: Option<Arc<side_table::Chunk>>,
    /// Where the next command should be: another address starts a chunk.
    next_address: u64,
    /// The draw state the stream leaves the CP with, as this buffer knows it.
    bound_graphics: Option<Arc<StateRecord>>,
    bound_compute: Option<Arc<StateRecord>>,
    /// Transient arena blocks this recording allocated.
    blocks: Blocks,
    /// Descriptor roots already noted.
    roots_seen: std::collections::HashSet<u64>,
    /// Whether the guest state started from the queue's last submission yet.
    inherited: bool,
    /// A recorder of one command outside any recording: it is in no chunk.
    standalone: bool,
    /// The memory the buffer's commands go into.
    bounds: (u64, u64),
}

impl AmdRecorder {
    fn with_generation(generation: Generation, memory: Arc<dyn GuestMemoryReader + Send + Sync>, shared: Arc<AmdShared>) -> Self {
        Self {
            generation,
            memory,
            shared,
            runtime: None,
            guest: GuestRegisters::default(),
            parameters: DrawParameters::default(),
            deferred_values: false,
            queue: None,
            waiting: 0,
            list: Vec::new(),
            table: Arc::new(SideTable::default()),
            chunk: None,
            next_address: 0,
            bound_graphics: None,
            bound_compute: None,
            blocks: Blocks::default(),
            roots_seen: std::collections::HashSet::new(),
            inherited: false,
            standalone: false,
            bounds: (0, 0),
        }
    }

    /// Starts the guest state from what the queue's last submission left in its registers
    /// (`submitted`), once, at the recording's first command. This is a guess: the commands run
    /// after whatever the title submits or chains before them, which submission checks (D38).
    fn inherit(&mut self, queue: HostQueue) {
        if std::mem::replace(&mut self.inherited, true) {
            return;
        }
        let baseline = self.shared.slot(queue).lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        if let Some(baseline) = baseline {
            self.guest = baseline.guest.clone();
            self.parameters = baseline.parameters;
        }
    }

    /// Notes a change the command at `at` made, for submission to work the state out (D38).
    fn log(&self, at: u64, kind: ChangeKind) {
        self.table.changes.push(StateChange { at, kind });
    }

    /// The device's runtime, opening the device on first use.
    fn runtime(&mut self) -> Result<Arc<runtime::TargetRuntime>> {
        if let Some(runtime) = &self.runtime {
            return Ok(runtime.clone());
        }
        let runtime = match self.shared.runtime.get() {
            Some(runtime) => runtime.clone(),
            None => {
                crate::agc::driver::Driver::get().host().map_err(|failure| Error::unsupported(failure.message))?;
                self.shared.runtime.get().cloned().ok_or_else(|| Error::unsupported("the device opened without a target runtime"))?
            }
        };
        self.runtime = Some(runtime.clone());
        Ok(runtime)
    }

    /// `change` now, or behind the register lists waiting for a draw.
    #[inline]
    fn change(&mut self, at: u64, kind: ChangeKind) -> Result<()> {
        let change = StateChange { at, kind };
        if self.waiting == 0 {
            self.apply(change)
        } else {
            self.wait(change);
            Ok(())
        }
    }

    /// Notes `change` behind the register lists no draw has needed yet: `settle` applies it, or
    /// submission when the recording never does (D38).
    fn wait(&mut self, change: StateChange) {
        self.table.waiting.push(change);
        self.waiting += 1;
    }

    fn apply(&mut self, change: StateChange) -> Result<()> {
        let at = change.at;
        match change.kind {
            ChangeKind::Register { bank, offset, value } => {
                apply_register(&mut self.guest, &mut self.parameters, bank, offset, value);
                if bank == RegisterBank::Context && offset == GUEST_STAGES_EN {
                    if let Some(flow) = flow::diagnostics() {
                        flow.stage_write(&self.table, value);
                    }
                }
            }
            // Registers the title fills in the footprint later: the recording cannot know them.
            ChangeKind::InPlace { .. } => {}
            ChangeKind::IndexType(size) => self.parameters.index_type = size,
            ChangeKind::ContextStateOp(operation) => {
                self.guest.context_state_op(operation)?;
                self.bound_graphics = None;
            }
            ChangeKind::IndexBase(base) => self.parameters.index_base = base,
            ChangeKind::IndexCount(count) => self.parameters.index_buffer_size = Some(count),
            ChangeKind::Instances(count) => self.parameters.instance_count = count,
            ChangeKind::IndirectBase { dispatch: true, address } => self.parameters.dispatch_indirect_base = address,
            ChangeKind::IndirectBase { dispatch: false, address } => self.parameters.draw_indirect_base = address,
            ChangeKind::ListAt { bank } => {
                // SAFETY: a footprint this recording wrote, in a chunk of the buffer it records,
                // which the title keeps until the GPU has run it.
                return match unsafe { side_table::named_list(at) } {
                    Some((address, count)) if count != 0 => self.read_list(at, bank, address, count),
                    _ => Ok(()),
                };
            }
            ChangeKind::List { bank, address, count } => return self.read_list(at, bank, address, count),
        }
        self.log(at, change.kind);
        Ok(())
    }

    /// Applies the register list of `count` pairs at `address`, which the footprint at `at` names,
    /// and makes the footprint load its user data.
    fn read_list(&mut self, at: u64, bank: RegisterBank, address: u64, count: u32) -> Result<()> {
        self.list.resize(count as usize * 2, 0);
        self.memory.read_into(address, &mut self.list)?;
        apply_list(&mut self.guest, &mut self.parameters, bank, &self.list)?;
        if bank == RegisterBank::Context {
            if let Some(flow) = flow::diagnostics() {
                for pair in self.list.chunks_exact(2) {
                    if crate::ir::register_offset(pair[0])? == GUEST_STAGES_EN {
                        flow.stage_write(&self.table, pair[1]);
                    }
                }
            }
        }
        if bank == RegisterBank::Shader {
            self.load_user_data(at, address, count)?;
        }
        self.log(at, ChangeKind::List { bank, address, count });
        Ok(())
    }

    /// Makes the footprint of the shader register list at `at` load its user data for the GPU:
    /// `LOAD_SH_REG_INDEX` of the guest's list when every pair is user data (the console's own
    /// packet), of a filtered copy otherwise.
    fn load_user_data(&mut self, at: u64, address: u64, count: u32) -> Result<()> {
        let generation = self.generation;
        let mut host_pairs = 0usize;
        let mut plain = true;
        for pair in self.list.chunks_exact(2) {
            let offset = crate::ir::register_offset(pair[0])?;
            match generation.host_user_data(RegisterBank::Shader, offset, 1) {
                Some(host) if host == offset && offset == pair[0] => host_pairs += 1,
                Some(_) => {
                    host_pairs += 1;
                    plain = false;
                }
                None => plain = false,
            }
        }
        if host_pairs == 0 {
            return Ok(());
        }
        let (source, entries) = if plain && host_pairs == count as usize {
            (address, count)
        } else {
            let runtime = self.runtime()?;
            let (cpu, gpu) = self.allocate_transient(&runtime, host_pairs * 8, 4)?;
            let mut at_pair = 0;
            for pair in self.list.chunks_exact(2) {
                let offset = crate::ir::register_offset(pair[0])?;
                if let Some(host) = generation.host_user_data(RegisterBank::Shader, offset, 1) {
                    // SAFETY: the transient memory just allocated for `host_pairs` pairs.
                    unsafe {
                        cpu.cast::<u32>().add(at_pair * 2).write(host);
                        cpu.cast::<u32>().add(at_pair * 2 + 1).write(pair[1]);
                    }
                    at_pair += 1;
                }
            }
            (gpu, host_pairs as u32)
        };
        // SAFETY: the footprint this recording wrote at `at`; the GPU has not reached it (no
        // draw has settled it yet, and the buffer is not submitted before one).
        let footprint = unsafe { std::slice::from_raw_parts_mut(at as *mut u32, 5) };
        footprint[0] = header(pkt3::LOAD_SH_REG_INDEX, 5, 0)?;
        footprint[1] = source as u32 & !3;
        footprint[2] = (source >> 32) as u32;
        footprint[3] = 1 << 31;
        footprint[4] = entries;
        Ok(())
    }

    /// A DMA end in the guest's GDS as the place in the runtime's GDS buffer that holds it; other
    /// ends as they are. A range past the GDS's end is an error, as on the console.
    fn gds(&mut self, end: DmaEnd, bytes: u32) -> Result<DmaEnd> {
        let DmaEnd::Gds(offset) = end else { return Ok(end) };
        if u64::from(offset) + u64::from(bytes) > u64::from(shader::gds::GDS_BYTES) {
            return Err(Error::invalid(format!("a DMA of {bytes} bytes at GDS byte 0x{offset:x} leaves the {}-byte GDS", shader::gds::GDS_BYTES)));
        }
        Ok(DmaEnd::Memory(self.runtime()?.gds + u64::from(offset)))
    }

    /// `bytes` of transient memory the GPU reads through a recorded packet.
    fn allocate_transient(&mut self, runtime: &runtime::TargetRuntime, bytes: usize, align: usize) -> Result<(*mut u8, u64)> {
        let before = self.blocks.count();
        let allocated = self.blocks.allocate(&runtime.transient, bytes, align)?;
        // The memory is reached through the submissions of this recording: they mark the block.
        for index in self.blocks.indices().skip(before) {
            self.table.blocks.push(index);
        }
        Ok(allocated)
    }

    /// Brings the guest state up to the commands recorded so far, for a draw or dispatch that
    /// needs it. A register list is read here and not where it was recorded: `Core::StateBuffer`
    /// records its lists empty and appends to them, patching their address and count, until
    /// the draw (D12 point 5). What the recording notes after its last draw stays unread: the
    /// table tells submission, which reads it for the commands that run after these (D38).
    pub fn settle(&mut self) -> Result<()> {
        if self.waiting == 0 {
            return Ok(());
        }
        let noted = self.table.waiting.len();
        let result = (noted - self.waiting..noted).try_for_each(|index| {
            let change = *self.table.waiting.get(index).expect("a change this recording noted");
            self.apply(change)
        });
        self.waiting = 0;
        self.table.settled.store(noted, Ordering::Release);
        result
    }

    /// Whether the buffer holds register values the title writes after recording.
    pub fn has_deferred_values(&self) -> bool {
        self.deferred_values
    }

    /// The guest registers the recorded commands have set.
    pub fn guest(&self) -> &GuestRegisters {
        &self.guest
    }

    /// (offset, value) pairs: user data in place as SET_SH_REG runs, the other registers in the
    /// guest state alone. The console's placeholder offsets above 0xffff are skipped.
    fn encode_pairs(&mut self, bank: RegisterBank, pairs: &[crate::agc::ShaderRegister], out: &mut [u32], address: u64) -> Result<()> {
        nop(out);
        let register = |offset, value| ChangeKind::Register { bank, offset, value };
        let mut at = 0;
        let mut index = 0;
        while index < pairs.len() {
            let pair = pairs[index];
            index += 1;
            if pair.offset > 0xffff {
                continue;
            }
            self.change(address, register(pair.offset, pair.value))?;
            let Some(first) = self.generation.host_user_data(bank, pair.offset, 1) else {
                continue;
            };
            let mut end = index;
            while end < pairs.len()
                && pairs[end].offset == pair.offset + (end + 1 - index) as u32
                && pairs[end].offset <= 0xffff
                && self.generation.host_user_data(bank, pairs[end].offset, 1) == Some(first + (end + 1 - index) as u32)
            {
                self.change(address, register(pairs[end].offset, pairs[end].value))?;
                end += 1;
            }
            let run = end - index + 1;
            out[at] = header(pkt3::SET_SH_REG, run as u32 + 2, 0)?;
            out[at + 1] = first;
            out[at + 2] = pair.value;
            for (slot, later) in out[at + 3..at + 2 + run].iter_mut().zip(&pairs[index..end]) {
                *slot = later.value;
            }
            at += run + 2;
            index = end;
        }
        nop(&mut out[at..]);
        Ok(())
    }

    fn cut(&mut self, at: u64, reason: side_table::CutReason) -> Result<()> {
        self.settle()?;
        self.table.cuts.push(side_table::Cut { at, reason, changes: self.table.changes.len() });
        self.bound_graphics = None;
        self.bound_compute = None;
        Ok(())
    }

    /// A value the command at `[at, end)` writes to guest memory: submission cuts after it when
    /// something waits for the value, and a following segment starts with the state there.
    fn note_write(&mut self, at: u64, end: u64, address: u64, value: u64, bytes: u8) -> Result<()> {
        self.settle()?;
        self.table.writes.push(side_table::LabelWrite { at, end, address, value, bytes, changes: self.table.changes.len() });
        self.bound_graphics = None;
        self.bound_compute = None;
        Ok(())
    }

    /// A dispatch that writes a known render target's fast-clear metadata cuts the stream after
    /// it (D29); each state and user data is checked once.
    fn note_clears(&mut self, state: &Arc<StateRecord>) -> Result<()> {
        if state.targets.is_some() {
            return Ok(());
        }
        let values = state.watched.iter().map(|&dword| u64::from(self.guest.get_or_zero(RegisterBank::Shader, dword)));
        let key = crate::agc::cache::key(std::iter::once(state.key).chain(values));
        if self.roots_seen.insert(key) {
            self.note_target_clears(state)?;
        }
        Ok(())
    }

    /// Cuts after a dispatch that writes the fast-clear metadata of a known render target, so the
    /// clear reaches the target's host copy at that point of the stream (D29).
    fn note_target_clears(&mut self, state: &StateRecord) -> Result<()> {
        let runtime = self.runtime()?;
        if runtime.clear_targets.is_empty() {
            return Ok(());
        }
        let next = self.next_address;
        for stage in &state.stages {
            for buffer in stage.program.buffers.iter().filter(|buffer| buffer.written && !buffer.pointer) {
                let Some(words) = self.buffer_descriptor(stage, &buffer.source) else { continue };
                let metadata = u64::from(words[0]) | (u64::from(words[1] & 0xffff) << 32);
                if metadata != 0 && runtime.clear_targets.contains(metadata) {
                    self.cut(next, side_table::CutReason::TargetClear { metadata })?;
                }
            }
        }
        Ok(())
    }

    /// Optional fast-clear inspection. Static buffer uses can belong to shader branches that
    /// never execute, so unbound or unreadable tables must not fault the recording thread.
    fn buffer_descriptor(&self, stage: &state::Stage, source: &shader::ir::DescriptorSource) -> Option<[u32; 4]> {
        use shader::ir::DescriptorSource as Source;
        let user = |sgpr: u8| -> Option<u32> {
            let element = u32::from(sgpr.checked_sub(stage.user_sgpr)?);
            (element < stage.user_count).then(|| self.guest.get_or_zero(RegisterBank::Shader, stage.user_base + element))
        };
        let mut words = [0u32; 4];
        match source {
            Source::UserSgprs { first } => {
                for (index, word) in words.iter_mut().enumerate() {
                    *word = user(first + index as u8)?;
                }
            }
            Source::Table(table) => {
                user(table.root)?;
                user(table.root.checked_add(1)?)?;
                let address = table.bound_address(
                    |sgpr| user(sgpr).unwrap_or(0),
                    |at| {
                        let mut pair = [0u32; 2];
                        self.memory.probe_into(at, &mut pair).ok()?;
                        Some(pair)
                    },
                )?;
                self.memory.probe_into(address, &mut words).ok()?;
            }
            _ => return None,
        }
        Some(words)
    }

    /// Notes a draw or dispatch the recording wrote with `state` after `bound`, for submission to
    /// check against the state the commands actually run with (D38).
    fn note_work(&self, at: u64, kind: side_table::WorkKind, state: Option<&Arc<StateRecord>>, bound: Option<&Arc<StateRecord>>, first: u32) {
        let work = side_table::Work {
            at,
            changes: self.table.changes.len(),
            kind,
            state: state.cloned(),
            bound: bound.map_or(0, |bound| bound.key),
            parameters: self.parameters,
            first,
        };
        self.table.works.push(work);
    }

    fn encode_draw(&mut self, draw: Draw, modifier: &DrawModifier, out: &mut [u32], address: u64) -> Result<()> {
        self.settle()?;
        let runtime = self.runtime()?;
        let result = runtime.states.draw(&runtime, &self.guest);
        if let Some(flow) = flow::diagnostics() {
            let stages = self.guest.get(RegisterBank::Context, GUEST_STAGES_EN);
            flow.draw(&self.table, address, stages, result.is_err());
        }
        let kind = side_table::WorkKind::Draw { draw, modifier: *modifier };
        let state = match result {
            Ok(state) => state,
            Err(reason) => {
                state::report_once(&reason, 0);
                nop(out);
                self.note_work(address, kind, None, self.bound_graphics.as_ref(), out[0]);
                return Ok(());
            }
        };
        write_draw(&self.generation, &runtime, self.bound_graphics.as_ref(), &state, draw, modifier, &self.parameters, out)?;
        self.note_work(address, kind, Some(&state), self.bound_graphics.as_ref(), out[0]);
        self.bound_graphics = Some(state.clone());
        self.note_clears(&state)
    }

    fn encode_dispatch(&mut self, host: HostQueue, groups: Option<[u32; 3]>, indirect: (u64, u32), modifier: u32, out: &mut [u32], address: u64) -> Result<()> {
        self.settle()?;
        let runtime = self.runtime()?;
        let kind = side_table::WorkKind::Dispatch { queue: host, groups, indirect, modifier };
        let state = match runtime.states.dispatch(&runtime, &self.guest, modifier, host == HostQueue::Compute) {
            Ok(state) => state,
            Err(reason) => {
                state::report_once(&reason, 1);
                nop(out);
                self.note_work(address, kind, None, self.bound_compute.as_ref(), out[0]);
                return Ok(());
            }
        };
        write_dispatch(self.bound_compute.as_ref(), &state, host, groups, indirect, modifier, &self.parameters, out)?;
        self.note_work(address, kind, Some(&state), self.bound_compute.as_ref(), out[0]);
        self.bound_compute = Some(state.clone());
        self.note_clears(&state)
    }

    /// `build` as a cached IB2 sequence (`key`), called from the footprint's first four dwords.
    fn encode_call(&mut self, key: u64, capacity: usize, out: &mut [u32], build: impl FnOnce(&mut commands::Packets) -> Result<()>) -> Result<()> {
        let runtime = self.runtime()?;
        let range = runtime.states.sequence(&runtime, key, || {
            let mut buffer = vec![0u32; capacity];
            let mut packets = commands::Packets::new(&mut buffer);
            build(&mut packets)?;
            let used = packets.written();
            buffer.truncate(used);
            Ok(buffer)
        })?;
        indirect_buffer(&mut out[..4], range)?;
        nop(&mut out[4..]);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/agc/mod.rs"]
mod tests;

/// An IB2 call of `target`.
fn indirect_buffer(out: &mut [u32], target: IbRange) -> Result<()> {
    out[0] = header(pkt3::INDIRECT_BUFFER, 4, 0)?;
    out[1] = target.address as u32;
    out[2] = (target.address >> 32) as u32;
    out[3] = target.dwords;
    Ok(())
}

impl Drop for AmdRecorder {
    fn drop(&mut self) {
        if let Some(runtime) = &self.runtime {
            self.blocks.retire_all(&runtime.transient);
        }
    }
}

impl Recorder for AmdRecorder {
    fn encode_in_place(&mut self, op: &Op<'_>, out: &mut [u32], address: u64) -> Result<()> {
        // The command sits inside the recording's commands: nothing about where the recording
        // stands changes.
        let next = std::mem::replace(&mut self.next_address, address);
        let result = self.encode(Queue::Any, op, out, address);
        self.next_address = next;
        result
    }

    fn set_bounds(&mut self, bottom: u64, top: u64) {
        self.bounds = (bottom, top);
    }

    fn submitted(&mut self, queue: Queue) {
        if self.settle().is_err() {
            return;
        }
        let queue = if queue == Queue::Compute { HostQueue::Compute } else { HostQueue::Graphics };
        let baseline = Arc::new(Baseline { guest: self.guest.clone(), parameters: self.parameters });
        *self.shared.slot(queue).lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(baseline);
    }

    fn encode(&mut self, queue: Queue, op: &Op<'_>, out: &mut [u32], address: u64) -> Result<()> {
        match queue {
            Queue::Graphics => self.queue = self.queue.or(Some(HostQueue::Graphics)),
            Queue::Compute => self.queue = self.queue.or(Some(HostQueue::Compute)),
            Queue::Any => {}
        }
        let host = self.queue.unwrap_or(HostQueue::Graphics);
        self.inherit(host);
        // The title writes some packets into the buffer itself (thread-trace markers through
        // `sceAgcDriverUserData*`): a command further on in the same buffer continues the chunk,
        // which holds those packets too, so no older recording there still seems to own them.
        let buffer = self.bounds.0..self.bounds.1;
        let further = self.chunk.as_ref().is_some_and(|chunk| address >= chunk.end() && buffer.contains(&address) && buffer.contains(&chunk.start));
        if address != self.next_address && !self.standalone && !further {
            // A new chunk (the first command, or the title's callback moved the buffer).
            let runtime = self.runtime()?;
            let chunk = Arc::new(side_table::Chunk::new(address, self.bounds.1, self.table.clone()));
            let _ = runtime.chunks.send(chunk.clone());
            self.chunk = Some(chunk);
            if self.shared.trace_recording() {
                eprintln!("[record-chunk] table={:p} start=0x{address:x} bounds={:x?}", Arc::as_ptr(&self.table), self.bounds);
            }
        }
        self.next_address = address + out.len() as u64 * 4;
        let next = self.next_address;
        let result = self.encode_op(host, op, out, address, next);
        // The chunk holds the command now, notes and all (a failed one is filled with NOPs). A
        // command written inside the commands (`encode_in_place`) does not move their end.
        if let Some(chunk) = &self.chunk {
            if chunk.start <= address && next > chunk.end() {
                chunk.extend(next);
            }
        }
        result
    }
}

impl AmdRecorder {
    fn encode_op(&mut self, host: HostQueue, op: &Op<'_>, out: &mut [u32], address: u64, next: u64) -> Result<()> {
        match *op {
            Op::Nop { payload, .. } => {
                nop(out);
                if let Some(payload) = payload {
                    if out.len() < 2 || payload.len() > out.len() - 1 {
                        return Err(Error::invalid(format!(
                            "a {}-dword NOP cannot hold {} payload dwords",
                            out.len(),
                            payload.len()
                        )));
                    }
                    out[1..1 + payload.len()].copy_from_slice(payload);
                }
            }
            Op::SetRegisters { bank, offset, count, values } => {
                if let Some(values) = values {
                    for (index, &value) in values.iter().enumerate() {
                        self.change(address, ChangeKind::Register { bank, offset: offset + index as u32, value })?;
                    }
                } else {
                    // Values the title writes into the footprint later: submission reads them.
                    self.deferred_values = true;
                    self.change(address, ChangeKind::InPlace { bank, offset, count })?;
                    if bank == RegisterBank::Context && GUEST_STAGES_EN.checked_sub(offset).is_some_and(|relative| relative < count) {
                        if let Some(flow) = flow::diagnostics() {
                            flow.deferred_stage_write(&self.table, address);
                        }
                    }
                }
                if let Some(host) = self.generation.host_user_data(bank, offset, count as usize) {
                    out[0] = header(pkt3::SET_SH_REG, count + 2, 0)?;
                    out[1] = host;
                } else {
                    nop(out);
                }
                match values {
                    Some(values) => out[2..].copy_from_slice(values),
                    None => out[2..].fill(0),
                }
            }
            Op::SetRegisterPairs { bank, pairs } => self.encode_pairs(bank, pairs, out, address)?,
            Op::SetRegistersIndirect { bank, address: list, count } => {
                // The footprint names the list where the title patches it (`named_list`).
                nop(out);
                out[2] = list as u32;
                out[3] = (list >> 32) as u32;
                out[4] = count;
                self.wait(StateChange { at: address, kind: ChangeKind::ListAt { bank } });
            }
            // The guest state draws use (D12).
            Op::IndexBase(base) => {
                self.change(address, ChangeKind::IndexBase(base))?;
                commands::encode(&self.generation, host, op, out)?;
            }
            Op::IndexCount(count) => {
                self.change(address, ChangeKind::IndexCount(count))?;
                commands::encode(&self.generation, host, op, out)?;
            }
            Op::IndexType(size) => {
                self.change(address, ChangeKind::IndexType(size))?;
                commands::encode(&self.generation, host, op, out)?;
            }
            Op::Instances(count) => {
                self.change(address, ChangeKind::Instances(count))?;
                commands::encode(&self.generation, host, op, out)?;
            }
            Op::IndirectBase { dispatch, address: base } => {
                self.change(address, ChangeKind::IndirectBase { dispatch, address: base })?;
                commands::encode(&self.generation, host, op, out)?;
            }
            Op::Draw { draw, modifier } => self.encode_draw(draw, &modifier, out, address)?,
            Op::Dispatch { groups, modifier } => self.encode_dispatch(host, Some(groups), (0, 0), modifier, out, address)?,
            Op::DispatchIndirect { address: arguments, offset, modifier } => self.encode_dispatch(host, None, (arguments, offset), modifier, out, address)?,
            Op::Acquire { pfp, cb_db, gcr, .. } => {
                if host == HostQueue::Graphics {
                    let generation = self.generation;
                    let key = crate::agc::cache::key([u64::from(cb_db), u64::from(gcr), u64::from(pfp), u64::from(generation.gfx12), 0x6163_7175]);
                    self.encode_call(key, 64, out, |packets| commands::acquire(packets, &generation, host, pfp, cb_db, gcr))?;
                } else {
                    let mut packets = commands::Packets::new(out);
                    commands::acquire(&mut packets, &self.generation, host, pfp, cb_db, gcr)?;
                    packets.finish();
                }
            }
            Op::Release { event, gcr, address: target, data, interrupt, context } => {
                let (select, value) = commands::GuestRelease::data(data)?;
                // A null address is patched in later: until then the release writes where it
                // does no harm, so the patch only has the address to fill in.
                let written = if target == 0 { self.runtime()?.scribble } else { target };
                let release = commands::GuestRelease { event, gcr, select, address: written, data: value };
                let mut packets = commands::Packets::new(out);
                release.release_mem(&mut packets, &self.generation, host)?;
                packets.finish();
                if select != 0 && target != 0 {
                    let bytes = if select == 1 { 4 } else { 8 };
                    self.note_write(address, next, target, value, bytes)?;
                }
                if interrupt != 0 {
                    let event = side_table::Event::Interrupt { context, address: target, value };
                    self.table.notices.push(side_table::Notice { at: address, event });
                    if let Some(flow) = flow::diagnostics() {
                        flow.recorded(&self.table, address, event);
                    }
                }
            }
            Op::Event { event, address: target } if host == HostQueue::Graphics && matches!(sync::EventPlan::of(event), sync::EventPlan::TimestampFlush { .. }) => {
                let generation = self.generation;
                let key = crate::agc::cache::key([u64::from(event), u64::from(generation.gfx12), 0x6576_656e]);
                self.encode_call(key, 64, out, |packets| commands::event(packets, &generation, host, event, target))?;
            }
            Op::Wait { bits64, function, address: target, reference, mask, .. } => {
                if self.shared.trace_recording() {
                    eprintln!("[record-wait] table={:p} command=0x{address:x} target=0x{target:x} reference=0x{reference:x} function={function}", Arc::as_ptr(&self.table));
                }
                // A label in the buffer's own memory (`Core::gpuSyncEvent`'s temporary ones) is
                // written by the commands before this one: the CP runs them in order.
                if !(self.bounds.0..self.bounds.1).contains(&target) {
                    let condition = side_table::WaitCondition { address: target, function, reference, mask: if bits64 { mask } else { mask & 0xffff_ffff }, bytes: if bits64 { 8 } else { 4 } };
                    self.cut(address, side_table::CutReason::Wait(condition))?;
                }
                commands::encode(&self.generation, host, op, out)?;
            }
            Op::WriteData { dst, address: target, values, .. } => {
                if matches!(dst, 1 | 2 | 5) && matches!(values.len(), 1 | 2) {
                    let (value, bytes) = if values.len() == 1 { (u64::from(values[0]), 4) } else { (u64::from(values[0]) | u64::from(values[1]) << 32, 8) };
                    self.note_write(address, next, target, value, bytes)?;
                }
                commands::encode(&self.generation, host, op, out)?;
            }
            Op::DmaData { pfp, source, destination, bytes, wait_previous, confirm } if matches!(source, DmaEnd::Gds(_)) || matches!(destination, DmaEnd::Gds(_)) => {
                // The guest's GDS is a buffer of the runtime's, which translated programs reach
                // for their GDS accesses (`shader::gds`): the DMA reaches it there too.
                let redirected = Op::DmaData { pfp, source: self.gds(source, bytes)?, destination: self.gds(destination, bytes)?, bytes, wait_previous, confirm };
                if let DmaEnd::Memory(target) = destination {
                    self.table.dma_writes.push(side_table::DmaWrite { at: address, address: target, bytes: u64::from(bytes) });
                }
                commands::encode(&self.generation, host, &redirected, out)?;
            }
            Op::DmaData { source, destination: DmaEnd::Memory(target), bytes, .. } => {
                if let (DmaEnd::Data(pattern), 4 | 8) = (source, bytes) {
                    self.note_write(address, next, target, u64::from(pattern), 4)?;
                }
                self.table.dma_writes.push(side_table::DmaWrite { at: address, address: target, bytes: u64::from(bytes) });
                commands::encode(&self.generation, host, op, out)?;
            }
            Op::LodStats { address: target, bytes } => {
                if target == 0 || host != HostQueue::Graphics {
                    nop(out);
                } else {
                    // The slots, then the table back to nothing counted, then the label the
                    // title waits on: CP_SYNC keeps them in that order.
                    use shader::mip_stats::{REPORT_BYTES, REPORT_SLOTS_OFFSET, TABLE_BYTES};
                    let runtime = self.runtime()?;
                    let (table, template) = (runtime.mip_stats_table, runtime.mip_stats_template);
                    let slots = bytes.min(REPORT_BYTES).saturating_sub(REPORT_SLOTS_OFFSET) & !3;
                    let key = crate::agc::cache::key([target, u64::from(bytes), 0x6c6f_6473]);
                    self.encode_call(key, 24, out, move |packets| {
                        use crate::amd::defs::cp_gfx11::{meg::DMA_DATA as D, pfp::WRITE_DATA as W};
                        let moves = [(table, target + u64::from(REPORT_SLOTS_OFFSET), slots), (template, table, TABLE_BYTES)];
                        for (source, destination, count) in moves.into_iter().filter(|&(_, _, count)| count != 0) {
                            let packet = packets.packet(D::OPCODE, D::WORDS as u32)?;
                            D::SRC_SEL.set(packet, D::enums::src_sel::SRC_ADDR_USING_L2);
                            D::DST_SEL.set(packet, D::enums::dst_sel::DST_ADDR_USING_L2);
                            D::CP_SYNC.set(packet, 1);
                            packet[2] = source as u32;
                            packet[3] = (source >> 32) as u32;
                            packet[4] = destination as u32;
                            packet[5] = (destination >> 32) as u32;
                            D::BYTE_COUNT.set(packet, count);
                        }
                        let packet = packets.packet(W::OPCODE, 5)?;
                        W::DST_SEL.set(packet, W::enums::dst_sel::MEMORY);
                        W::WR_CONFIRM.set(packet, 1);
                        packet[2] = target as u32;
                        packet[3] = (target >> 32) as u32;
                        packet[4] = 1;
                        Ok(())
                    })?;
                }
            }
            Op::Notice(Notice::Flip { video_out, buffer, mode, argument }) => {
                nop(out);
                let event = side_table::Event::Flip { video_out, buffer, mode, argument };
                self.table.notices.push(side_table::Notice { at: address, event });
                if let Some(flow) = flow::diagnostics() {
                    flow.recorded(&self.table, address, event);
                }
                self.cut(next, side_table::CutReason::AfterFlip)?;
            }
            Op::Notice(Notice::WaitFlipDone { video_out, buffer }) => {
                nop(out);
                self.cut(address, side_table::CutReason::FlipDone { video_out, buffer })?;
            }
            Op::Notice(Notice::ResetQueue { state, .. }) if host == HostQueue::Graphics => {
                // CLEAR_STATE: clear, push, pop, or push then clear the context.
                if state <= 3 {
                    self.change(address, ChangeKind::ContextStateOp(state))?;
                    self.bound_graphics = None;
                }
                nop(out);
            }
            Op::Notice(Notice::ResetQueue { .. } | Notice::Rewind { .. }) => nop(out),
            Op::Jump { chain: false, .. } => return Err(Error::unsupported("guest indirect-buffer calls are not implemented on the GPU path yet")),
            Op::Jump { target, dwords, chain: true } => {
                commands::encode(&self.generation, host, op, out)?;
                self.table.branches.push(side_table::Branch { at: address, kind: side_table::BranchKind::Chain });
                if let Some(flow) = flow::diagnostics() {
                    flow.chain(&self.table, address, target, dwords);
                }
            }
            Op::CondExec { .. } => {
                commands::encode(&self.generation, host, op, out)?;
                self.table.branches.push(side_table::Branch { at: address, kind: side_table::BranchKind::CondExec });
            }
            Op::Marker { text } if gpu_timing::marker_dwords() != 0 => {
                let stages = self.runtime()?.stage_log.clone();
                let Some(stages) = stages else {
                    return Err(Error::invalid("marker timestamps without a stage log"));
                };
                let slot = stages.note(Arc::as_ptr(&self.table) as u64, text);
                let mut packets = commands::Packets::new(out);
                commands::release_timestamp(&mut packets, stages.address(slot))?;
                packets.finish();
            }
            _ => commands::encode(&self.generation, host, op, out)?,
        }
        Ok(())
    }
}
