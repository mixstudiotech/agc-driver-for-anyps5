//! The seam between the Agc front end and a target GPU (decisions D13, D14).
//!
//! The front end turns each Agc call into an [`Op`]: what the title asked for, in the guest's
//! terms (guest registers, guest addresses, PS5 enums), with nothing of the target in it. A
//! [`Target`] says how many dwords each op takes in a command buffer and gives every recorded
//! buffer a [`Recorder`] that writes the ops in the target's command format. AMD (gfx11.5 and
//! gfx12) is the first target; Mali and Adreno implement the same traits.

use crate::{ir::RegisterBank, Result};

/// The queue family an Agc function records for: `Dcb*` functions graphics, `Acb*` functions
/// asynchronous compute, `Cb*` functions either.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Queue {
    Graphics,
    Compute,
    Any,
}

/// What a guest release writes once its event has happened.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseData {
    None,
    Value32(u32),
    Value64(u64),
    /// The GPU clock, 64 bits.
    Timestamp,
    /// GDS dwords from `offset`.
    Gds { offset: u16, dwords: u16 },
}

/// One end of a guest DMA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmaEnd {
    Memory(u64),
    /// A 32-bit pattern (source only).
    Data(u32),
    /// A GDS byte offset.
    Gds(u32),
    /// No destination: the DMA only prefetches its source.
    Nowhere,
}

/// What the runtime does when a submission reaches the command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Notice {
    /// `sceAgc*SetFlip`: show `buffer` of the video-out port `video_out`.
    Flip { video_out: u32, buffer: i32, mode: u32, argument: i64 },
    /// `sceAgc*WaitUntilSafeForRendering`: what follows waits until `buffer` is off screen.
    WaitFlipDone { video_out: u32, buffer: i32 },
    /// `sceAgc{Acb,Dcb}ResetQueue`: the queue's state back to its defaults.
    ResetQueue { operation: u32, state: u32 },
    /// `sceAgc*Rewind`: the command processor waits here until the packet is made valid.
    Rewind { valid: bool },
}

/// Where a guest draw takes its vertices from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Draw {
    /// `count` indices from `address` (`sceAgcDcbDrawIndex`).
    Indexed { count: u32, address: u64 },
    /// `count` indices from `offset` in the index buffer `IndexBase` set
    /// (`sceAgcDcbDrawIndexOffset`).
    IndexedOffset { offset: u32, count: u32 },
    /// `count` vertices without indices (`sceAgcDcbDrawIndexAuto`).
    Auto { count: u32 },
    /// Arguments at `offset` from the draw indirect base (`sceAgcDcb{DrawIndirect,
    /// DrawIndexIndirect}`).
    Indirect { indexed: bool, offset: u32 },
    /// Up to `count` draws whose arguments start at `offset` from the base, `stride` bytes
    /// apart; the count is read from `count_address` when that is not 0
    /// (`sceAgcDcbDraw{,Index}IndirectMulti`).
    IndirectMulti { indexed: bool, offset: u32, count: u32, count_address: u64, stride: u32 },
}

/// A guest draw modifier (`ShaderDrawModifier`) decoded: the guest SGPRs (shader-bank dwords)
/// that receive the draw's vertex offset, instance offset and draw index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrawModifier {
    pub vertex_offset: Option<u32>,
    pub instance_offset: Option<u32>,
    pub draw_index: Option<u32>,
    /// User VGPRs carry the draw parameters (multi draws).
    pub user_vgprs: bool,
    /// DRAW_INITIATOR NOT_EOP: the draw may run together with the next one.
    pub fuse: bool,
}

impl DrawModifier {
    /// The modifier word as the SDK passes it; bits from 33 on are reserved.
    pub fn decode(modifier: u64) -> Option<Self> {
        if modifier >> 33 != 0 {
            return None;
        }
        let word = modifier as u32;
        // Hull-shader pipelines put the parameters in the HS user data, the others in the GS's.
        let base = if matches!(word >> 29, 3 | 5) { 0x10c } else { 0x8c };
        let at = |enabled: u32, shift: u32| (word & enabled != 0).then(|| base + (word >> shift & 0x1f));
        Some(Self {
            vertex_offset: at(1, 9),
            instance_offset: at(4, 19),
            draw_index: at(8, 24),
            user_vgprs: word & 0x10 != 0,
            fuse: modifier & 1 << 32 == 0 && word & 0x100 != 0,
        })
    }
}

/// One Agc command, as the title recorded it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op<'a> {
    /// `count` dwords the command processor skips (`sceAgc*Nop`); `payload` words, when given,
    /// fill the packet's body (data the title placed in the buffer).
    Nop { dwords: u32, payload: Option<&'a [u32]> },
    /// `count` consecutive guest registers from `offset` (`Set*RegisterRangeDirect`,
    /// `Set*RegisterDirect`). Without `values` the title fills them in the footprint later, from
    /// dword 2 on (`sceAgcGetDataPacketPayloadAddress` type 1).
    SetRegisters { bank: RegisterBank, offset: u32, count: u32, values: Option<&'a [u32]> },
    /// Guest (offset, value) pairs copied into the buffer (`sceAgcCbSet{Sh,Uc}RegistersDirect`).
    SetRegisterPairs { bank: RegisterBank, pairs: &'a [crate::agc::ShaderRegister] },
    /// Guest (offset, value) pairs the GPU reads from `address` when it executes the command
    /// (`Set*RegistersIndirect`).
    SetRegistersIndirect { bank: RegisterBank, address: u64, count: u32 },
    /// A debug marker: `Some(text)` opens a scope, `None` closes the innermost one.
    Marker { text: Option<&'a [u8]> },
    /// `sceAgcDcbGetLodStats`: the report written to `address`, when the GPU gets here (0: none).
    LodStats { address: u64, bytes: u32 },
    Notice(Notice),
    /// Guest ACQUIRE_MEM: the CB/DB flushes of `cb_db` (guest COHER_CNTL bits) and the cache
    /// actions of `gcr` (guest GCR_CNTL) before what follows; `pfp` makes the prefetch parser
    /// wait as well. `ranged` when the guest limited the actions to part of memory.
    Acquire { pfp: bool, cb_db: u32, gcr: u32, ranged: bool },
    /// Guest RELEASE_MEM: once `event` has happened, the cache actions of `gcr` (guest
    /// RELEASE_MEM_OP bits 12 to 23), then `data` at `address` and the guest interrupt
    /// `interrupt` with `context`.
    Release { event: u32, gcr: u32, address: u64, data: ReleaseData, interrupt: u32, context: u32 },
    /// Guest EVENT_WRITE of `event`, with `address` for the occlusion dumps.
    Event { event: u32, address: u64 },
    /// Guest WRITE_DATA of `values` to `address` (guest destination select `dst`).
    WriteData { dst: u32, pfp: bool, address: u64, values: &'a [u32], increment: bool, confirm: bool },
    /// Guest COPY_DATA (guest source and destination selects).
    CopyData { src: u32, dst: u32, pfp: bool, source: u64, destination: u64, item64: bool, confirm: bool },
    DmaData { pfp: bool, source: DmaEnd, destination: DmaEnd, bytes: u32, wait_previous: bool, confirm: bool },
    /// Guest ATOMIC_MEM (gfx10 GL2 atomic `op`, `command` 0 to 3).
    AtomicMem { op: u32, command: u32, address: u64, source: u64, compare: u64, interval: u32 },
    /// Guest WAIT_REG_MEM on memory: wait until `function(memory & mask, reference)` holds.
    Wait { bits64: bool, function: u32, pfp: bool, address: u64, reference: u64, mask: u64, poll: u32 },
    /// Guest COND_WRITE: write `value` to `address` when `function(memory at poll & mask,
    /// reference)` holds.
    CondWrite { function: u32, poll: u64, reference: u32, mask: u32, address: u64, value: u32 },
    /// Guest MEM_SEMAPHORE: signal (`set_one`: write 1, else increment) or wait and decrement.
    Semaphore { address: u64, signal: bool, set_one: bool },
    /// The prefetch parser waits for the micro engine (`sceAgcDcbStallCommandBufferParser`).
    SyncParser,
    /// Continue in the commands at `target` (`chain`), or call them and come back.
    Jump { target: u64, dwords: u32, chain: bool },
    /// Skip the next `dwords` dwords when the 32-bit value at `address` is 0.
    CondExec { address: u64, dwords: u32 },
    IndexBase(u64),
    IndexCount(u32),
    /// Index element size: 0 16-bit, 1 32-bit, 2 8-bit.
    IndexType(u32),
    Instances(u32),
    /// Base of the indirect draw (`dispatch` false) or dispatch arguments.
    IndirectBase { dispatch: bool, address: u64 },
    /// Guest SET_PREDICATION: `operation` (0 clears the predicate), the guest's condition and
    /// wait bits, and the predicate's memory.
    Predication { operation: u32, condition: bool, wait: bool, address: u64 },
    /// A draw with the guest state as the recording has it (D12 point 3).
    Draw { draw: Draw, modifier: DrawModifier },
    /// A compute dispatch of `groups` thread groups; `modifier` is the guest
    /// DISPATCH_INITIATOR (`sceAgcCbDispatch`).
    Dispatch { groups: [u32; 3], modifier: u32 },
    /// A dispatch whose group counts the GPU reads: at `address` (`sceAgcAcbDispatchIndirect`)
    /// or, when `address` is 0, at `offset` from the dispatch indirect base
    /// (`sceAgcDcbDispatchIndirect`).
    DispatchIndirect { address: u64, offset: u32, modifier: u32 },
}

/// The footprint class of a draw (one per draw function, D14).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DrawKind {
    Indexed,
    IndexedOffset,
    Auto,
    Indirect,
    IndirectMulti,
}

impl Draw {
    pub fn kind(&self) -> DrawKind {
        match self {
            Self::Indexed { .. } => DrawKind::Indexed,
            Self::IndexedOffset { .. } => DrawKind::IndexedOffset,
            Self::Auto { .. } => DrawKind::Auto,
            Self::Indirect { .. } => DrawKind::Indirect,
            Self::IndirectMulti { .. } => DrawKind::IndirectMulti,
        }
    }
}

/// What an op's footprint depends on (D14): the parameters of its `*GetSize` function, or of
/// the call itself for functions without one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Kind {
    Nop { dwords: u32 },
    SetRegisters { bank: RegisterBank, count: u32 },
    SetRegisterPairs { bank: RegisterBank, count: u32 },
    SetRegistersIndirect { bank: RegisterBank },
    /// A marker push with `text` bytes, or a pop.
    Marker { text: Option<u32> },
    LodStats,
    Flip,
    WaitFlipDone,
    ResetQueue,
    Rewind,
    Acquire,
    Release,
    Event { event: u32 },
    WriteData { dwords: u32 },
    CopyData,
    DmaData,
    AtomicMem,
    Wait { bits64: bool },
    CondWrite,
    Semaphore,
    SyncParser,
    Jump,
    CondExec,
    IndexBase,
    IndexCount,
    IndexType,
    Instances,
    IndirectBase,
    Predication,
    Draw(DrawKind),
    Dispatch,
    DispatchIndirect,
}

impl Op<'_> {
    pub fn kind(&self) -> Kind {
        match *self {
            Self::Nop { dwords, .. } => Kind::Nop { dwords },
            Self::SetRegisters { bank, count, .. } => Kind::SetRegisters { bank, count },
            Self::SetRegisterPairs { bank, pairs } => Kind::SetRegisterPairs { bank, count: pairs.len() as u32 },
            Self::SetRegistersIndirect { bank, .. } => Kind::SetRegistersIndirect { bank },
            Self::Marker { text } => Kind::Marker { text: text.map(|text| text.len() as u32) },
            Self::LodStats { .. } => Kind::LodStats,
            Self::Notice(Notice::Flip { .. }) => Kind::Flip,
            Self::Notice(Notice::WaitFlipDone { .. }) => Kind::WaitFlipDone,
            Self::Notice(Notice::ResetQueue { .. }) => Kind::ResetQueue,
            Self::Notice(Notice::Rewind { .. }) => Kind::Rewind,
            Self::Acquire { .. } => Kind::Acquire,
            Self::Release { .. } => Kind::Release,
            Self::Event { event, .. } => Kind::Event { event },
            Self::WriteData { values, .. } => Kind::WriteData { dwords: values.len() as u32 },
            Self::CopyData { .. } => Kind::CopyData,
            Self::DmaData { .. } => Kind::DmaData,
            Self::AtomicMem { .. } => Kind::AtomicMem,
            Self::Wait { bits64, .. } => Kind::Wait { bits64 },
            Self::CondWrite { .. } => Kind::CondWrite,
            Self::Semaphore { .. } => Kind::Semaphore,
            Self::SyncParser => Kind::SyncParser,
            Self::Jump { .. } => Kind::Jump,
            Self::CondExec { .. } => Kind::CondExec,
            Self::IndexBase(_) => Kind::IndexBase,
            Self::IndexCount(_) => Kind::IndexCount,
            Self::IndexType(_) => Kind::IndexType,
            Self::Instances(_) => Kind::Instances,
            Self::IndirectBase { .. } => Kind::IndirectBase,
            Self::Predication { .. } => Kind::Predication,
            Self::Draw { draw, .. } => Kind::Draw(draw.kind()),
            Self::Dispatch { .. } => Kind::Dispatch,
            Self::DispatchIndirect { .. } => Kind::DispatchIndirect,
        }
    }
}

/// A change to a recorded command (`sceAgc*Patch*`, D5, D14), in the guest's terms; the
/// footprint is where the Agc function that recorded the command returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Patch {
    /// The register list of a `Set*RegistersIndirect` (bank checked).
    IndirectAddress { bank: RegisterBank, address: u64 },
    /// Its register count: set, or with `add` increased.
    IndirectCount { bank: RegisterBank, count: u32, add: bool },
    CondExecAddress(u64),
    /// The end of the range a COND_EXEC guards (an address in the buffer).
    CondExecEnd(u64),
    DmaDestination(u64),
    DmaSource(u64),
    JumpTarget { target: u64, dwords: u32 },
    ReleaseAddress(u64),
    ReleaseData(u64),
    /// Guest RELEASE_MEM_OP cache actions (bits 12 to 23).
    ReleaseCaches(u32),
    /// The guest end-of-pipe event.
    ReleaseEvent(u32),
    Rewind(bool),
    WaitAddress(u64),
    WaitReference(u64),
    /// The low dword of the mask (`sceAgcWaitRegMemPatchMask`).
    WaitMask(u32),
    WaitFunction(u32),
    WriteDataAddress(u64),
    /// The guest destination select; `compute` for the async encoding.
    WriteDataDestination { dst: u32, compute: bool },
    /// The cache policy of a WRITE_DATA, which the host does not take: only checked.
    WriteDataCachePolicy,
    /// Guest ACQUIRE_MEM engine: 0 the prefetch parser, 1 the micro engine.
    AcquireEngine(u32),
    /// The command becomes one that does nothing (`sceAgcSetNop`).
    Nop,
    /// The command's work runs only under the predicate (`sceAgcSetPacketPredication`).
    Predicate(bool),
}

/// PIXEL_PIPE_STAT_DUMP: the guest event of an occlusion dump.
pub const EVENT_PIXEL_PIPE_STAT_DUMP: u32 = 0x39;

/// A target GPU: AMD today, and NVIDIA, Mali or Adreno as other implementations of this trait
/// (D13). Nothing outside the target names its host types.
pub trait Target: Send + Sync {
    /// Opens the GPU this target records for, on the device thread: what runs the title's
    /// submissions.
    fn open(&self, flips: std::sync::Arc<crate::agc::driver::flips::Flips>, arena: std::sync::Arc<crate::agc::driver::anyps5::Anyps5>) -> Result<Box<dyn crate::agc::driver::device::Engine>> {
        let _ = (flips, arena);
        Err(crate::Error::unsupported("this build has no GPU backend for the target"))
    }

    /// Dwords of the recorded command at `packet`, `None` when it does not start one
    /// (`sceAgcGetPacketSize`).
    ///
    /// # Safety
    /// `packet` points to memory the title recorded commands in.
    unsafe fn command_dwords(&self, packet: *const u32) -> Option<u32> {
        let _ = packet;
        None
    }

    /// Writes one command outside any recording (`sceAgcDriver*WritePacket`, which take a raw
    /// pointer): `out` is exactly its footprint.
    fn write(&self, op: &Op<'_>, out: &mut [u32]) -> Result<()> {
        let address = out.as_mut_ptr() as u64;
        self.record().encode(Queue::Any, op, out, address)
    }

    /// Dwords an op of `kind` takes in a command buffer (D14).
    fn footprint(&self, kind: Kind) -> Result<u32>;

    /// The target as itself, for the driver of its GPU (`agc::driver`).
    /// Applies `patch` to the command recorded at `footprint`.
    ///
    /// # Safety
    /// `footprint` is where an Agc function of this target returned, in a live buffer.
    unsafe fn patch(&self, footprint: *mut u32, patch: Patch) -> Result<()> {
        let _ = (footprint, patch);
        Err(crate::Error::unsupported("this target patches no commands"))
    }

    /// Puts the work of the commands in `[start, end)` under the predicate, or takes it out
    /// (`sceAgcSetRangePredication`).
    ///
    /// # Safety
    /// The range holds whole commands this target recorded, in a live buffer.
    unsafe fn predicate_range(&self, start: *mut u32, end: *const u32, predicate: bool) -> Result<()> {
        let _ = (start, end, predicate);
        Err(crate::Error::unsupported("this target predicates no commands"))
    }

    /// The state of a command buffer recorded from its start.
    fn record(&self) -> Box<dyn Recorder>;

    /// Fills `out` with commands that do nothing (a footprint whose encoding failed).
    fn fill(&self, out: &mut [u32]);
}

/// A target's state for one command buffer.
pub trait Recorder: Send {
    /// Writes `op` into `out`, which is exactly its footprint and lies at GPU address
    /// `address`, for a function of the `queue` family.
    fn encode(&mut self, queue: Queue, op: &Op<'_>, out: &mut [u32], address: u64) -> Result<()>;

    /// Writes `op` at `address` inside the commands this recorder wrote, for a function that
    /// writes one command by itself (`sceAgcDriver*WritePacket`): the command takes part in what
    /// the recording notes for submission, but its position does not move the recording's.
    fn encode_in_place(&mut self, op: &Op<'_>, out: &mut [u32], address: u64) -> Result<()> {
        self.encode(Queue::Any, op, out, address)
    }

    /// The memory the buffer's commands currently go into, before each command: a label inside
    /// it is the buffer's own (`Core::gpuSyncEvent` places temporary labels in the command stream).
    fn set_bounds(&mut self, _bottom: u64, _top: u64) {}

    /// The title is submitting these commands to a queue of the `queue` family: what they leave
    /// in the queue's registers is what the next recording for that queue starts from (the
    /// hardware keeps it between submissions).
    fn submitted(&mut self, _queue: Queue) {}
}
