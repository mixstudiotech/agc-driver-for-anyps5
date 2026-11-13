//! What a recording notes for submission, so that submission never parses the commands (D27):
//! where the stream must be cut, which label writes it makes, what happens at flips, the guest
//! memory its DMA writes, the transient memory it reaches, and where it jumps to other command
//! buffers. It also logs every change it makes to the guest state and every draw and dispatch with
//! the state it gave them: a buffer's commands run with the registers the buffers before it leave,
//! which only submission knows, and it works the state out from these logs (D38).
//!
//! One thread appends to a table while the device thread reads it: each log publishes its length
//! after the new element is written and never moves one, so neither side takes a lock.

use super::commands::HostQueue;
use super::draw::DrawParameters;
use super::arena::{Arena, Block, Lease};
use super::state::{CommandBlock, IbRange, StateRecord};
use crate::agc::target::{Draw, DrawModifier};
use crate::ir::RegisterBank;
use std::cell::{RefCell, UnsafeCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, OnceLock, Weak};

static COMMAND_THREADS: AtomicU64 = AtomicU64::new(1);
thread_local! {
    static COMMAND_THREAD: u64 = COMMAND_THREADS.fetch_add(1, Ordering::Relaxed);
    static COMMAND_CURSORS: RefCell<Vec<CommandCursor>> = const { RefCell::new(Vec::new()) };
}

/// One bump block per thread and arena, shared by all of that thread's recordings.
struct CommandCursor {
    arena: Weak<Arena>,
    block: Option<Block>,
    used: usize,
    packets: HashMap<u64, Weak<CommandPacket>>,
    prune_at: usize,
}

struct CommandPacket {
    range: IbRange,
    _lease: Lease,
}

impl CommandCursor {
    fn allocate(&mut self, arena: &Arena, bytes: usize, align: usize) -> crate::Result<(*mut u8, u64, u32)> {
        if bytes > arena.block_bytes() || !align.is_power_of_two() || align > 256 {
            return Err(crate::Error::invalid(format!("Agc commands: {bytes} bytes aligned to {align} in {}-byte blocks", arena.block_bytes())));
        }
        let mut offset = self.used.next_multiple_of(align);
        if self.block.is_none() || offset + bytes > arena.block_bytes() {
            if let Some(block) = self.block.take() { arena.retire(block); }
            self.block = Some(arena.take()?);
            self.used = 0;
            offset = 0;
        }
        self.used = offset + bytes;
        let block = self.block.as_ref().expect("allocated above");
        let (cpu, address) = arena.at(block, offset, bytes)?;
        Ok((cpu, address, block.index()))
    }

    fn on_thread(arena: &Arc<Arena>, bytes: usize, align: usize) -> crate::Result<(*mut u8, u64, u32)> {
        Self::with_thread(arena, |cursor| cursor.allocate(arena, bytes, align))
    }

    fn with_thread<T>(arena: &Arc<Arena>, use_cursor: impl FnOnce(&mut Self) -> crate::Result<T>) -> crate::Result<T> {
        COMMAND_CURSORS.with(|cursors| {
            let mut cursors = cursors.borrow_mut();
            cursors.retain(|cursor| cursor.arena.strong_count() != 0);
            let index = match cursors.iter().position(|cursor| cursor.arena.as_ptr() == Arc::as_ptr(arena)) {
                Some(index) => index,
                None => {
                    cursors.push(Self { arena: Arc::downgrade(arena), block: None, used: 0, packets: HashMap::new(), prune_at: 1024 });
                    cursors.len() - 1
                }
            };
            use_cursor(&mut cursors[index])
        })
    }

    fn packet(arena: &Arc<Arena>, block: &CommandBlock) -> crate::Result<Arc<CommandPacket>> {
        Self::with_thread(arena, |cursor| {
            if let Some(packet) = cursor.packets.get(&block.key).and_then(Weak::upgrade) { return Ok(packet); }
            let (cpu, address, index) = cursor.allocate(arena, block.words.len() * 4, 32)?;
            // SAFETY: this allocation is disjoint and not reachable by GPU commands yet.
            unsafe { std::ptr::copy_nonoverlapping(block.words.as_ptr().cast::<u8>(), cpu, block.words.len() * 4) };
            std::sync::atomic::fence(Ordering::SeqCst);
            let packet = Arc::new(CommandPacket { range: IbRange { address, dwords: block.words.len() as u32 }, _lease: arena.lease(index)? });
            if cursor.packets.len() >= cursor.prune_at {
                cursor.packets.retain(|_, packet| packet.strong_count() != 0);
                cursor.prune_at = cursor.packets.len() + 1024;
            }
            cursor.packets.insert(block.key, Arc::downgrade(&packet));
            Ok(packet)
        })
    }

    #[cfg(test)]
    fn retire_on_thread(arena: &Arc<Arena>) {
        COMMAND_CURSORS.with(|cursors| cursors.borrow_mut().retain(|cursor| cursor.arena.as_ptr() != Arc::as_ptr(arena)));
    }
}

impl Drop for CommandCursor {
    fn drop(&mut self) {
        if let (Some(arena), Some(block)) = (self.arena.upgrade(), self.block.take()) { arena.retire(block); }
    }
}

#[derive(Default)]
struct CommandAllocations {
    leases: Vec<Lease>,
    leased: HashSet<u32>,
    ranges: HashMap<u64, Arc<CommandPacket>>,
}

/// GPU command copies live as long as the recording that can replay their IB calls.
struct RecordedCommands {
    arena: Arc<Arena>,
    writers: AtomicPtr<CommandWriter>,
}

struct CommandWriter {
    thread: u64,
    arena: Arc<Arena>,
    allocations: UnsafeCell<CommandAllocations>,
    published: Log<u32>,
    next: *mut CommandWriter,
}

// SAFETY: only the identified thread accesses allocations; other threads read the published log.
unsafe impl Send for CommandWriter {}
unsafe impl Sync for CommandWriter {}

impl CommandWriter {
    fn allocate(&self, bytes: usize, align: usize) -> crate::Result<(*mut u8, u64)> {
        assert_eq!(self.thread, COMMAND_THREAD.with(|thread| *thread));
        // SAFETY: allocation is private to this identified thread and never reenters the writer.
        let allocations = unsafe { &mut *self.allocations.get() };
        let (cpu, address, index) = CommandCursor::on_thread(&self.arena, bytes, align)?;
        if !allocations.leased.contains(&index) {
            let lease = self.arena.lease(index)?;
            allocations.leases.push(lease);
            allocations.leased.insert(index);
            self.published.push(index);
        }
        Ok((cpu, address))
    }

    fn place(&self, block: &CommandBlock) -> crate::Result<IbRange> {
        assert_eq!(self.thread, COMMAND_THREAD.with(|thread| *thread));
        // SAFETY: this writer belongs to the calling thread, and allocation never calls back into it.
        if let Some(range) = unsafe { &*self.allocations.get() }.ranges.get(&block.key) {
            return Ok(range.range);
        }
        let packet = CommandCursor::packet(&self.arena, block)?;
        let range = packet.range;
        let index = self.arena.block_at(range.address).expect("placed command packet");
        // SAFETY: the previous immutable lookup ended before allocation; this thread alone writes.
        let allocations = unsafe { &mut *self.allocations.get() };
        if allocations.leased.insert(index) { self.published.push(index); }
        allocations.ranges.insert(block.key, packet);
        Ok(range)
    }
}

impl RecordedCommands {
    fn writer(&self) -> &CommandWriter {
        let thread = COMMAND_THREAD.with(|thread| *thread);
        let mut writer = self.writers.load(Ordering::Acquire);
        while !writer.is_null() {
            // SAFETY: published nodes remain allocated until this recording is dropped.
            let node = unsafe { &*writer };
            if node.thread == thread { return node; }
            writer = node.next;
        }
        let writer = Box::into_raw(Box::new(CommandWriter { thread, arena: self.arena.clone(), allocations: UnsafeCell::default(), published: Log::default(), next: std::ptr::null_mut() }));
        let mut head = self.writers.load(Ordering::Acquire);
        loop {
            // SAFETY: the new node is not published until the successful CAS.
            unsafe { (*writer).next = head };
            match self.writers.compare_exchange_weak(head, writer, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(current) => head = current,
            }
        }
        // SAFETY: this calling thread owns the just-published writer.
        unsafe { &*writer }
    }

    fn submitted(&self, sequence: u64) {
        let mut writer = self.writers.load(Ordering::Acquire);
        while !writer.is_null() {
            // SAFETY: nodes stay allocated, and their block log is published for concurrent readers.
            let node = unsafe { &*writer };
            for &index in node.published.iter() { self.arena.submitted(index, sequence); }
            writer = node.next;
        }
    }
}

impl Drop for RecordedCommands {
    fn drop(&mut self) {
        let mut writer = *self.writers.get_mut();
        while !writer.is_null() {
            // SAFETY: the recording has no remaining users; each node was allocated once above.
            let node = unsafe { Box::from_raw(writer) };
            writer = node.next;
        }
    }
}

/// Entries of a log's first chunk; each later chunk is twice the one before.
const FIRST_CHUNK: usize = 32;
/// Chunks a log can have: `FIRST_CHUNK << CHUNKS` entries in all.
const CHUNKS: usize = 32;

/// The chunk of a log entry and its slot there: chunk `k` holds `FIRST_CHUNK << k` entries from
/// `FIRST_CHUNK * (2^k - 1)` on.
fn locate(index: usize) -> (usize, usize) {
    let chunk = (usize::BITS - 1 - (index / FIRST_CHUNK + 1).leading_zeros()) as usize;
    (chunk, index - FIRST_CHUNK * ((1 << chunk) - 1))
}

/// An append-only log with one writer at a time and readers on other threads. A short log costs
/// one small chunk: recordings are many, and most note little.
pub struct Log<T> {
    chunks: [AtomicPtr<MaybeUninit<T>>; CHUNKS],
    len: AtomicUsize,
}

// SAFETY: elements are written once before their publication and only read afterwards.
unsafe impl<T: Send> Send for Log<T> {}
unsafe impl<T: Send + Sync> Sync for Log<T> {}

impl<T> Default for Log<T> {
    fn default() -> Self {
        Self { chunks: [const { AtomicPtr::new(std::ptr::null_mut()) }; CHUNKS], len: AtomicUsize::new(0) }
    }
}

impl<T> Log<T> {
    fn chunk(&self, index: usize) -> *mut MaybeUninit<T> {
        let slot = &self.chunks[index];
        let existing = slot.load(Ordering::Acquire);
        if !existing.is_null() {
            return existing;
        }
        let chunk = Box::into_raw(Box::<[MaybeUninit<T>]>::new_uninit_slice(FIRST_CHUNK << index)).cast::<MaybeUninit<T>>();
        slot.store(chunk, Ordering::Release);
        chunk
    }

    /// Appends `value`; only the recording's thread calls this.
    pub fn push(&self, value: T) {
        let index = self.len.load(Ordering::Relaxed);
        let (chunk, slot) = locate(index);
        let chunk = self.chunk(chunk);
        // SAFETY: the slot is inside the chunk and nothing reads it before the store below.
        unsafe { chunk.add(slot).cast::<T>().write(value) };
        self.len.store(index + 1, Ordering::Release);
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }
        let (chunk, slot) = locate(index);
        let chunk = self.chunks[chunk].load(Ordering::Acquire);
        // SAFETY: published elements are initialized and never move or change.
        Some(unsafe { &*chunk.add(slot).cast::<T>() })
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len()).map_while(move |index| self.get(index))
    }
}

impl<T> Drop for Log<T> {
    fn drop(&mut self) {
        let len = *self.len.get_mut();
        for (index, slot) in self.chunks.iter_mut().enumerate() {
            let chunk = *slot.get_mut();
            if chunk.is_null() {
                continue;
            }
            let capacity = FIRST_CHUNK << index;
            let used = len.saturating_sub(FIRST_CHUNK * ((1 << index) - 1)).min(capacity);
            // SAFETY: the first `used` elements are initialized, and `chunk` came from the boxed
            // slice `chunk` made with this capacity.
            unsafe {
                for element in 0..used {
                    chunk.add(element).cast::<T>().drop_in_place();
                }
                drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(chunk, capacity)));
            }
        }
    }
}

/// The guest's user data at a point of the stream: PS, GS, HS (32 registers each) and compute (16).
pub const USER_DATA_REGISTERS: usize = 32 * 3 + 16;
pub type UserData = [u32; USER_DATA_REGISTERS];

/// A guest wait on memory: the 8 bytes at `address` compare with `reference` under `mask`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WaitCondition {
    pub address: u64,
    /// WAIT_REG_MEM function.
    pub function: u32,
    pub reference: u64,
    pub mask: u64,
    /// Bytes compared: 4 or 8.
    pub bytes: u8,
}

impl WaitCondition {
    /// Whether a value in memory ends the wait.
    pub fn holds(&self, value: u64) -> bool {
        let value = if self.bytes == 4 { value & 0xffff_ffff } else { value };
        let (value, reference) = (value & self.mask, self.reference & self.mask);
        match self.function {
            0 => true,
            1 => value < reference,
            2 => value <= reference,
            3 => value == reference,
            4 => value != reference,
            5 => value >= reference,
            6 => value > reference,
            _ => true,
        }
    }
}

/// Why a stream is cut at `at`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutReason {
    /// The new segment waits for this guest condition.
    Wait(WaitCondition),
    /// The new segment waits until this display buffer has left the screen.
    FlipDone { video_out: u32, buffer: i32 },
    /// The segment before ends with a flip.
    AfterFlip,
    /// The commands before left user data no command of theirs loads on the GPU (a register list
    /// no draw of its recording needed): the new segment starts with it.
    UserData,
}

/// A point where the stream is cut.
#[derive(Clone, Copy, Debug)]
pub struct Cut {
    pub at: u64,
    pub reason: CutReason,
    /// State changes the recording had applied by then (an index into `SideTable::changes`).
    pub changes: usize,
}

/// A value the stream's commands write to guest memory, at or after `at`.
#[derive(Clone, Copy, Debug)]
pub struct LabelWrite {
    /// The command's footprint.
    pub at: u64,
    pub end: u64,
    pub address: u64,
    pub value: u64,
    pub bytes: u8,
    /// State changes the recording had applied by then: a segment starting after the write
    /// restores the user data there.
    pub changes: usize,
}

/// Something the runtime does when the GPU reaches `at`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    Flip { video_out: u32, buffer: i32, mode: u32, argument: i64 },
    /// A guest end-of-pipe interrupt.
    Interrupt { context: u32, address: u64, value: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Notice {
    pub at: u64,
    pub event: Event,
}

/// Guest memory a DMA command at `at` writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DmaWrite {
    pub at: u64,
    pub address: u64,
    pub bytes: u64,
    pub source: Option<(u64, u64)>,
}

/// A change the command at `at` makes to the guest state, in the order the recording applied it
/// (a register list waits for the draw after it, D12 point 5).
#[derive(Clone, Copy, Debug)]
pub struct StateChange {
    pub at: u64,
    pub kind: ChangeKind,
}

#[derive(Clone, Copy, Debug)]
pub enum ChangeKind {
    Register { bank: RegisterBank, offset: u32, value: u32 },
    /// `count` registers from `offset` whose values the title writes into the footprint at `at`
    /// after recording it, from its dword 2 on.
    InPlace { bank: RegisterBank, offset: u32, count: u32 },
    /// The guest list of `count` (offset, value) pairs at `address`, as its recording read it. The
    /// title may have appended to it since: its footprint at `at` names it as it is
    /// ([`named_list`]) unless it loads user data on the GPU now.
    List { bank: RegisterBank, address: u64, count: u32 },
    /// A register list its recording has not read: the footprint at `at` names it.
    ListAt { bank: RegisterBank },
    IndexType(u32),
    /// CLEAR_STATE: clear, push, pop, or push then clear the context.
    ContextStateOp(u32),
    IndexBase(u64),
    IndexCount(u32),
    Instances(u32),
    IndirectBase { dispatch: bool, address: u64 },
}

/// The list a `Set*RegistersIndirect` footprint at `at` names: its address and how many (offset,
/// value) pairs it holds, in dwords 2 to 4 of a five-dword NOP, where the console's packet has
/// them and `Core::StateBuffer` patches them while it appends to the list. `None` when other
/// commands are there now, or the packet that loads a shader list's user data on the GPU.
///
/// # Safety
/// `at` is in command memory the title keeps until the GPU has run it, five dwords at least.
pub unsafe fn named_list(at: u64) -> Option<(u64, u32)> {
    // SAFETY: the caller's contract.
    let footprint = unsafe { std::ptr::read_volatile(at as *const [u32; 5]) };
    let list = footprint[0] >> 30 == 3 && (footprint[0] >> 16) & 0x3fff == 3 && (footprint[0] >> 8) & 0xff == crate::amd::defs::pkt3::NOP;
    list.then(|| (u64::from(footprint[2]) | u64::from(footprint[3]) << 32, footprint[4]))
}

/// What a draw or dispatch footprint holds besides its state.
#[derive(Clone, Copy, Debug)]
pub enum WorkKind {
    Draw { draw: Draw, modifier: DrawModifier },
    Dispatch { queue: HostQueue, groups: Option<[u32; 3]>, indirect: (u64, u32), modifier: u32 },
}

/// A draw or dispatch as the recording wrote it, from what it knew of the guest state.
#[derive(Clone, Debug)]
pub struct Work {
    /// Where its footprint starts.
    pub at: u64,
    /// State changes the recording had applied by then.
    pub changes: usize,
    pub kind: WorkKind,
    /// The state it was given; `None` when the recording found none and left a NOP.
    pub state: Option<Arc<StateRecord>>,
    /// The key of the state bound before it, which its slot changes from; 0 for none.
    pub bound: u64,
    pub parameters: DrawParameters,
    /// The footprint's first dword as the recording left it: submission rewrites the footprint
    /// only while it still holds it (the title may have written other commands there since).
    pub first: u32,
    pub encoding: Arc<WorkEncoding>,
}

#[derive(Debug)]
pub struct WorkEncoding {
    pub first: AtomicU32,
    signature: AtomicU64,
}

impl WorkEncoding {
    fn signature(state: Option<u64>, bound: u64, parameters: DrawParameters) -> u64 {
        crate::agc::cache::key([state.unwrap_or(0), bound, parameters.index_base, parameters.index_buffer_size.map_or(u64::MAX, u64::from), u64::from(parameters.index_type), u64::from(parameters.instance_count), parameters.draw_indirect_base, parameters.dispatch_indirect_base])
    }

    pub fn new(first: u32, state: Option<u64>, bound: u64, parameters: DrawParameters) -> Self {
        Self { first: AtomicU32::new(first), signature: AtomicU64::new(Self::signature(state, bound, parameters)) }
    }

    pub fn matches(&self, state: Option<u64>, bound: u64, parameters: DrawParameters) -> bool {
        self.signature.load(Ordering::Acquire) == Self::signature(state, bound, parameters)
    }

    pub fn update(&self, first: u32, state: Option<u64>, bound: u64, parameters: DrawParameters) {
        self.first.store(first, Ordering::Release);
        self.signature.store(Self::signature(state, bound, parameters), Ordering::Release);
    }
}

/// A command that changes where the CP goes next.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchKind {
    /// INDIRECT_BUFFER with CHAIN: the commands continue at its target.
    Chain,
    /// COND_EXEC: the commands after it run only when its condition holds.
    CondExec,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Branch {
    pub at: u64,
    pub kind: BranchKind,
}

/// A draw or dispatch's descriptor roots: the record it ran and the user data its programs'
/// descriptors are reached from.
#[derive(Clone, Debug)]
pub struct Root {
    pub point: WorkPoint,
    pub record: Arc<StateRecord>,
    pub values: Box<[u32]>,
    /// The thread groups of a direct dispatch.
    pub groups: Option<[u32; 3]>,
}

#[derive(Clone, Debug, Default)]
pub struct WorkPoint {
    pub at: u64,
    pub end: u64,
    pub compute_user: [u32; 16],
    pub threads: [u32; 3],
    pub start: [u32; 3],
    pub predicated: bool,
    pub guarded: bool,
}

/// One recording's notes.
#[derive(Default)]
pub struct SideTable {
    commands: OnceLock<RecordedCommands>,
    pub cuts: Log<Cut>,
    pub writes: Log<LabelWrite>,
    pub notices: Log<Notice>,
    pub dma_writes: Log<DmaWrite>,
    /// Transient-arena blocks the recording allocated from.
    pub blocks: Log<u32>,
    pub changes: Log<StateChange>,
    /// The changes noted behind a register list, in stream order, and how many of them the
    /// recording has applied since, which are in `changes` then. A recording reads a list when a
    /// draw, a dispatch or a cut of its own needs the state: what it noted after its last one is
    /// still unread when its commands run, and the commands chained after them run with it.
    pub waiting: Log<StateChange>,
    pub settled: AtomicUsize,
    pub works: Log<Work>,
    pub branches: Log<Branch>,
}

impl SideTable {
    #[cfg(test)]
    pub(super) fn retire_thread_commands(arena: &Arc<Arena>) { CommandCursor::retire_on_thread(arena); }

    fn command_storage(&self, arena: &Arc<Arena>) -> &RecordedCommands {
        self.commands.get_or_init(|| RecordedCommands { arena: arena.clone(), writers: AtomicPtr::default() })
    }

    pub fn allocate_transient(&self, arena: &Arc<Arena>, bytes: usize, align: usize) -> crate::Result<(*mut u8, u64)> {
        self.command_storage(arena).writer().allocate(bytes, align)
    }

    pub fn place_commands(&self, arena: &Arc<Arena>, block: &CommandBlock) -> crate::Result<IbRange> {
        if block.words.is_empty() {
            return Ok(IbRange::default());
        }
        self.command_storage(arena).writer().place(block)
    }

    pub fn commands_submitted(&self, sequence: u64) {
        if let Some(commands) = self.commands.get() {
            commands.submitted(sequence);
        }
    }

    /// The changes the recording noted and has not applied, in stream order.
    pub fn unsettled(&self) -> impl Iterator<Item = &StateChange> {
        (self.settled.load(Ordering::Acquire)..self.waiting.len()).map_while(|index| self.waiting.get(index))
    }
}

/// Contiguous command memory a recording writes into, from `start`.
pub struct Chunk {
    pub start: u64,
    /// Where the commands written so far end.
    end: AtomicU64,
    /// The end of the memory the buffer owns: the chunk cannot grow past it.
    limit: u64,
    pub table: Arc<SideTable>,
}

const CHUNK_PRUNED: u64 = 1 << 63;

impl Chunk {
    pub fn new(start: u64, limit: u64, table: Arc<SideTable>) -> Self {
        Self { start, end: AtomicU64::new(start), limit, table }
    }

    pub fn end(&self) -> u64 {
        self.end.load(Ordering::Acquire) & !CHUNK_PRUNED
    }

    /// The recording has written commands up to `end`.
    pub fn extend(&self, end: u64) -> bool {
        let mut previous = self.end.load(Ordering::Acquire);
        loop {
            match self.end.compare_exchange_weak(previous, end.max(previous & !CHUNK_PRUNED), Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return previous & CHUNK_PRUNED != 0,
                Err(current) => previous = current,
            }
        }
    }

    fn prune_at(&self, end: u64) -> bool {
        match self.end.compare_exchange(end, end | CHUNK_PRUNED, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => true,
            Err(current) => current == end | CHUNK_PRUNED,
        }
    }
}

/// The part `[from, to)` of a range that a chunk's recording wrote.
#[derive(Clone)]
pub struct Owner {
    pub from: u64,
    pub to: u64,
    pub table: Arc<SideTable>,
}

impl Owner {
    pub fn holds(&self, at: u64) -> bool {
        (self.from..self.to).contains(&at)
    }
}

/// A chunk in the index, with the order chunks arrived in.
struct Entry {
    arrived: u64,
    chunk: Arc<Chunk>,
    /// The latest chunk started in a buffer that holds this one's start: what that one writes is
    /// no longer this chunk's (the title reused the memory).
    superseded: Option<(u64, Arc<Chunk>)>,
}

/// Which recording wrote the commands at an address. A chunk owns what its recording wrote, less
/// what newer chunks wrote over; a tail jump written by another recording after the commands is
/// such a chunk of its own.
pub struct ChunkIndex {
    incoming: Receiver<Arc<Chunk>>,
    /// By start, the latest chunk started there.
    chunks: BTreeMap<u64, Entry>,
    /// Still-written tails whose original map entry or superseder anchor was replaced.
    orphaned: BTreeMap<u64, Arc<Chunk>>,
    arrived: u64,
    pruned: std::time::Instant,
}

impl ChunkIndex {
    pub fn new() -> (Self, Sender<Arc<Chunk>>) {
        let (sender, incoming) = std::sync::mpsc::channel();
        (Self { incoming, chunks: BTreeMap::new(), orphaned: BTreeMap::new(), arrived: 0, pruned: std::time::Instant::now() }, sender)
    }

    fn refresh(&mut self) {
        while let Ok(chunk) = self.incoming.try_recv() {
            self.arrived += 1;
            let arrived = self.arrived;
            if chunk.limit > chunk.start + 1 {
                for entry in self.chunks.range_mut(chunk.start + 1..chunk.limit).map(|(_, entry)| entry) {
                    if let Some((earlier, previous)) = entry.superseded.take() { self.orphaned.insert(earlier, previous); }
                    entry.superseded = Some((arrived, chunk.clone()));
                }
            }
            let end = chunk.end();
            if let Some(previous) = self.chunks.insert(chunk.start, Entry { arrived, chunk, superseded: None }) {
                let previous_end = previous.chunk.end();
                if previous_end > end || !previous.chunk.prune_at(previous_end) { self.orphaned.insert(previous.arrived, previous.chunk); }
                if let Some((earlier, previous)) = previous.superseded { self.orphaned.insert(earlier, previous); }
            }
        }
        let orphaned = &mut self.orphaned;
        self.chunks.retain(|_, entry| {
            if let Some((arrived, newer)) = &entry.superseded {
                let end = entry.chunk.end();
                if newer.start <= entry.chunk.start && newer.end() >= end {
                    if entry.chunk.prune_at(end) {
                        orphaned.insert(*arrived, newer.clone());
                        return false;
                    }
                }
            }
            true
        });
        if self.pruned.elapsed() >= std::time::Duration::from_secs(1) { self.prune_covered(); }
    }

    /// Removes recordings whose written bytes were all overwritten, including by several chunks.
    pub fn prune_covered(&mut self) {
        let mut candidates = self.orphaned.clone();
        for entry in self.chunks.values() {
            candidates.insert(entry.arrived, entry.chunk.clone());
            if let Some((arrived, chunk)) = &entry.superseded { candidates.insert(*arrived, chunk.clone()); }
        }
        let mut covered: BTreeMap<u64, u64> = BTreeMap::new();
        let mut dead = HashSet::new();
        for (&arrived, chunk) in candidates.iter().rev() {
            let (mut from, mut to) = (chunk.start, chunk.end());
            if from == to { continue; }
            if let Some((&start, &end)) = covered.range(..=from).next_back() {
                if end >= to && chunk.prune_at(to) { dead.insert(arrived); continue; }
                if end >= from { from = start; }
            }
            while let Some((&start, &end)) = covered.range(from..).next() {
                if start > to { break; }
                to = to.max(end);
                covered.remove(&start);
            }
            covered.insert(from, to);
        }
        self.chunks.retain(|_, entry| !dead.contains(&entry.arrived));
        for entry in self.chunks.values_mut() {
            if entry.superseded.as_ref().is_some_and(|(arrived, _)| dead.contains(arrived)) { entry.superseded = None; }
        }
        let anchored: HashSet<u64> = self.chunks.values().flat_map(|entry| std::iter::once(entry.arrived).chain(entry.superseded.as_ref().map(|(arrived, _)| *arrived))).collect();
        self.orphaned = candidates.into_iter().filter(|(arrived, _)| !dead.contains(arrived) && !anchored.contains(arrived)).collect();
        self.pruned = std::time::Instant::now();
    }

    /// Every chunk the index knows, by address: the label writes in them, for a report.
    pub fn chunks(&self) -> impl Iterator<Item = &Arc<Chunk>> {
        self.chunks.values().map(|entry| &entry.chunk).chain(self.orphaned.values().filter(|chunk| !self.chunks.get(&chunk.start).is_some_and(|entry| Arc::ptr_eq(&entry.chunk, chunk))))
    }

    /// The chunks that start in the 4 MiB below `at` (the nearest twelve), for a report on whose
    /// commands an address holds.
    pub fn describe_near(&self, at: u64) -> String {
        self.chunks
            .range(at.saturating_sub(4 << 20)..=at)
            .rev()
            .take(12)
            .map(|(_, entry)| {
                let chunk = &entry.chunk;
                format!(
                    "0x{:x}..0x{:x} limit 0x{:x} #{} table {:p}{}",
                    chunk.start, chunk.end(), chunk.limit, entry.arrived, Arc::as_ptr(&chunk.table),
                    entry.superseded.as_ref().map_or(String::new(), |(arrived, newer)| format!(" superseded by #{arrived} at 0x{:x}..0x{:x}", newer.start, newer.end()))
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// The recordings that wrote the commands in `[start, end)`, in address order, each with the
    /// part it wrote. Parts no recording wrote are left out.
    pub fn owners(&mut self, start: u64, end: u64) -> Vec<Owner> {
        self.refresh();
        let mut candidates: Vec<(u64, &Arc<Chunk>)> = Vec::new();
        for entry in self.chunks.range(..end).map(|(_, entry)| entry) {
            let superseder = entry.superseded.as_ref().map(|(arrived, chunk)| (*arrived, chunk));
            for (arrived, chunk) in std::iter::once((entry.arrived, &entry.chunk)).chain(superseder) {
                let written = chunk.end();
                if chunk.start < end && written > start && written > chunk.start && !candidates.iter().any(|(_, known)| Arc::ptr_eq(known, chunk)) {
                    candidates.push((arrived, chunk));
                }
            }
        }
        for (&arrived, chunk) in &self.orphaned {
            if chunk.start < end && chunk.end() > start && !candidates.iter().any(|(_, known)| Arc::ptr_eq(known, chunk)) { candidates.push((arrived, chunk)); }
        }
        let mut owners = Vec::with_capacity(candidates.len());
        for &(arrived, chunk) in &candidates {
            let mut parts = vec![(chunk.start.max(start), chunk.end().min(end))];
            for &(_, newer) in candidates.iter().filter(|&&(later, _)| later > arrived) {
                let (over_start, over_end) = (newer.start, newer.end());
                parts = parts
                    .into_iter()
                    .flat_map(|(from, to)| [(from, to.min(over_start)), (from.max(over_end), to)])
                    .filter(|(from, to)| from < to)
                    .collect();
            }
            owners.extend(parts.into_iter().map(|(from, to)| Owner { from, to, table: chunk.table.clone() }));
        }
        owners.sort_by_key(|owner| owner.from);
        owners
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/agc/side_table.rs"]
mod tests;
