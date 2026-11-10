mod diagnostics;
mod fence;
mod guest;
mod interop;
mod private;
mod sparse;
pub use self::sparse::SparseTiles;

pub use self::diagnostics::{DeviceState, ExecutionState, PageFault};
pub use self::guest::{GuestWindow, GUEST_PAGE_BYTES, IMPORT_WINDOW_BYTES};
pub use self::interop::{OpenedFence, QueueFence, SharedFence, SharedResourceData};
pub use self::private::SharedRecords;
pub use super::MemoryKind;

use self::{
    fence::{KmtFenceWait, MonitoredFence},
    private::{AmdPrivateAbi, CachePolicy, IbEntry},
};
use crate::{
    backend::{
        host_pm4,
        va,
    },
    Error, ErrorKind, Result,
};
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    mem::size_of,
    ops::Range,
    ptr,
    rc::{Rc, Weak},
    sync::atomic::{fence, Ordering},
    time::Duration,
};
use windows_sys::{Wdk::Graphics::Direct3D::*, Win32::Foundation::LUID};

/// D3DDDIGPUVIRTUALADDRESS_RESERVATION_TYPE (d3dukmdt.h): unmapped addresses fault, or read zero.
const D3DDDIGPUVA_RESERVE_NO_ACCESS: i32 = 0;
const D3DDDIGPUVA_RESERVE_ZERO: i32 = 1;

// Public flags: Windows SDK d3dukmdt.h / d3dkmthk.h.
const CONTEXT_HW_QUEUE_SUPPORTED: u32 = 0x10;
const CREATE_ALLOCATION_RESOURCE: u32 = 0x01;
const CREATE_ALLOCATION_NON_SECURE: u32 = 0x04;
const GPU_VA_ALLOW_WRITE: u64 = 0x01;
// D3DDDIGPUVIRTUALADDRESS_PROTECTION_TYPE.Execute: required for shader instruction fetch.
const GPU_VA_ALLOW_EXECUTE: u64 = 0x02;
// D3DDDI_MAKERESIDENT_FLAGS.MustSucceed: residency may not fail for budget reasons.
const MAKE_RESIDENT_MUST_SUCCEED: u32 = 0x02;
const GPU_PAGE_BYTES: usize = 4096;
const ADAPTER_NODE_LIMIT: u32 = 256;
const FENCE_TIMEOUT: Duration = Duration::from_secs(5);
/// A paging operation: the OS pins and zeroes what a first import of several GiB needs, for far
/// longer than a GPU fence takes. radv waits for VM updates without a limit; a hung GPU is
/// caught by the queues' own wait.
const PAGING_TIMEOUT: Duration = Duration::from_secs(60);
/// Submissions that may be in flight on a queue before a new one first waits for the oldest.
const MAX_IN_FLIGHT: usize = 64;
/// Retired IBs a queue keeps for reuse; any beyond this are released.
const MAX_POOLED_IBS: usize = 256;
/// What a chained IB needs after its commands: worst-case NOP padding plus the CHAIN packet.
const CHAIN_OVERHEAD_DWORDS: usize = host_pm4::IB_ALIGNMENT_DWORDS - 1 + 4;
/// ReserveGpuVirtualAddress takes a 64 KiB-aligned base and size.
const RESERVATION_ALIGNMENT: u64 = 64 * 1024;
/// Pinned host imports are split into allocations of at most 128 MiB, as AMD's UMD does.
const HOST_IMPORT_MAX_BYTES: usize = 128 << 20;
const VERSION_COMPONENT_BITS: u32 = u16::BITS;
const VERSION_COMPONENT_MASK: u64 = u16::MAX as u64;
const VERSION_MAJOR_SHIFT: u32 = VERSION_COMPONENT_BITS * 3;
const VERSION_MINOR_SHIFT: u32 = VERSION_COMPONENT_BITS * 2;
const ABI_FIXTURE_GPU_ADDRESS: u64 = 0x1234_5678_9000;
const ABI_FIXTURE_IB_BYTES: usize = 256;

#[derive(Clone, Debug)]
pub struct AdapterIdentity {
    pub vendor_id: u32,
    pub device_id: u32,
    pub revision: u32,
    pub driver_version: u64,
    pub luid_low: u32,
    pub luid_high: i32,
}

impl AdapterIdentity {
    pub fn driver_string(&self) -> String {
        let v = self.driver_version;
        format!(
            "{}.{}.{}.{}",
            v >> VERSION_MAJOR_SHIFT,
            (v >> VERSION_MINOR_SHIFT) & VERSION_COMPONENT_MASK,
            (v >> VERSION_COMPONENT_BITS) & VERSION_COMPONENT_MASK,
            v & VERSION_COMPONENT_MASK
        )
    }
}

/// One engine node of the adapter, from NODEMETADATA.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeInfo {
    pub ordinal: u32,
    pub engine_type: i32,
    pub name: String,
}

struct Adapter(u32);
impl Drop for Adapter {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe {
                D3DKMTCloseAdapter(&D3DKMT_CLOSEADAPTER { hAdapter: self.0 });
            }
        }
    }
}

struct Kmt;
impl Kmt {
    fn status(value: i32, operation: &str) -> Result<()> {
        if value < 0 {
            Err(Error::new(
                ErrorKind::System,
                format!("{operation}: NTSTATUS=0x{:08x}", value as u32),
            ))
        } else {
            Ok(())
        }
    }

    fn query<T>(adapter: u32, kind: KMTQUERYADAPTERINFOTYPE, data: &mut T) -> Result<()> {
        let mut q = D3DKMT_QUERYADAPTERINFO {
            hAdapter: adapter,
            Type: kind,
            pPrivateDriverData: (data as *mut T).cast(),
            PrivateDriverDataSize: size_of::<T>() as u32,
        };
        Kmt::status(
            unsafe { D3DKMTQueryAdapterInfo(&mut q) },
            "QueryAdapterInfo",
        )
    }
}

struct AdapterInventory;
impl AdapterInventory {
    fn adapters(&self) -> Result<Vec<(Adapter, AdapterIdentity)>> {
        let mut e = D3DKMT_ENUMADAPTERS2::default();
        Kmt::status(
            unsafe { D3DKMTEnumAdapters2(&mut e) },
            "EnumAdapters2(count)",
        )?;
        if e.NumAdapters > ADAPTER_NODE_LIMIT {
            return Err(Error::invalid("unreasonable adapter count"));
        }
        let mut entries = vec![D3DKMT_ADAPTERINFO::default(); e.NumAdapters as usize];
        e.pAdapters = entries.as_mut_ptr();
        Kmt::status(unsafe { D3DKMTEnumAdapters2(&mut e) }, "EnumAdapters2")?;
        let opened: Vec<_> = entries
            .into_iter()
            .take(e.NumAdapters as usize)
            .map(|info| (Adapter(info.hAdapter), info.AdapterLuid))
            .collect();
        let mut result = Vec::new();
        for (adapter, luid) in opened {
            let mut ids = D3DKMT_QUERY_DEVICE_IDS::default();
            if Kmt::query(adapter.0, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS, &mut ids).is_err() {
                continue;
            }
            let mut version = D3DKMT_KMD_DRIVER_VERSION::default();
            Kmt::query(adapter.0, KMTQAITYPE_KMD_DRIVER_VERSION, &mut version)?;
            result.push((
                adapter,
                AdapterIdentity {
                    vendor_id: ids.DeviceIds.VendorID,
                    device_id: ids.DeviceIds.DeviceID,
                    revision: ids.DeviceIds.RevisionID,
                    driver_version: version.DriverVersion as u64,
                    luid_low: luid.LowPart,
                    luid_high: luid.HighPart,
                },
            ));
        }
        Ok(result)
    }
}

/// Which hardware queue a submission runs on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueKind {
    Graphics,
    Compute,
    ComputeQueue(u8),
}

impl QueueKind {
    fn index(self) -> usize {
        match self {
            QueueKind::Graphics => 0,
            QueueKind::Compute => 1,
            QueueKind::ComputeQueue(index) => 1 + usize::from(index),
        }
    }
}

/// How a host stream is laid out in IBs (FINAL-REPORT §6, Milestone 1: one IB first, then a
/// multi-entry submission with a 0xC-flagged state block, then chained IBs).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmitLayout {
    /// One IB.
    Single,
    /// Two submission entries: the first `dwords` of the stream (the state block), flagged 0xC
    /// as a preamble, then the rest as the main IB.
    Preamble { dwords: usize },
    /// IBs of at most `max_dwords` each, linked by INDIRECT_BUFFER CHAIN packets; only the
    /// first is listed in the submission.
    Chained { max_dwords: usize },
}

/// One IB of a planned submission: the stream dwords it carries and its entry flags.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Segment {
    words: Range<usize>,
    flags: u32,
}

/// Splits a stream into IB segments at packet boundaries outside COND_EXEC ranges.
fn plan_segments(words: &[u32], layout: SubmitLayout) -> Result<Vec<Segment>> {
    let main = |words: Range<usize>| Segment {
        words,
        flags: IbEntry::MAIN,
    };
    match layout {
        SubmitLayout::Single => Ok(vec![main(0..words.len())]),
        SubmitLayout::Preamble { dwords } => {
            if dwords == 0
                || dwords >= words.len()
                || !host_pm4::split_points(words)?.contains(&dwords)
            {
                return Err(Error::invalid(format!(
                    "a preamble split at dword {dwords} is not a packet boundary inside the stream and outside every COND_EXEC range"
                )));
            }
            Ok(vec![
                Segment {
                    words: 0..dwords,
                    flags: IbEntry::PREAMBLE,
                },
                main(dwords..words.len()),
            ])
        }
        SubmitLayout::Chained { max_dwords } => {
            if max_dwords <= CHAIN_OVERHEAD_DWORDS || max_dwords > host_pm4::MAX_IB_DWORDS {
                return Err(Error::invalid(format!(
                    "chained IBs of {max_dwords} dwords leave no room for commands"
                )));
            }
            let limit = max_dwords - CHAIN_OVERHEAD_DWORDS;
            let points = host_pm4::split_points(words)?;
            let mut segments = Vec::new();
            let mut start = 0;
            for pair in points.windows(2) {
                let (piece, end) = (pair[0], pair[1]);
                if end - piece > limit {
                    return Err(Error::invalid(format!(
                        "a {}-dword packet or COND_EXEC range does not fit a chained IB of {max_dwords} dwords",
                        end - piece
                    )));
                }
                if end - start > limit {
                    segments.push(main(start..piece));
                    start = piece;
                }
            }
            segments.push(main(start..words.len()));
            Ok(segments)
        }
    }
}

/// The dwords of each IB: its commands and NOP padding, plus, when chained, a CHAIN packet to the
/// next IB (at `addresses[i + 1]`) on every IB but the last. Every image is a multiple of 8
/// dwords.
fn ib_images(
    words: &[u32],
    segments: &[Segment],
    addresses: &[u64],
    chained: bool,
) -> Result<Vec<Vec<u32>>> {
    let mut images = Vec::with_capacity(segments.len());
    let mut next: Option<(u64, usize)> = None;
    for (segment, &address) in segments.iter().zip(addresses).rev() {
        let commands = &words[segment.words.clone()];
        let image = match next.filter(|_| chained) {
            Some((next_address, next_dwords)) => {
                let alignment = host_pm4::IB_ALIGNMENT_DWORDS;
                let gap = (alignment - (commands.len() + 4) % alignment) % alignment;
                let mut image = commands.to_vec();
                image.extend(host_pm4::nop_fill(gap));
                image.extend(host_pm4::chain_packet(next_address, next_dwords)?);
                image
            }
            None => host_pm4::pad(commands)?,
        };
        next = Some((address, image.len()));
        images.push(image);
    }
    images.reverse();
    Ok(images)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QueueTimings {
    pub submissions: u64,
    pub targets: u64,
    /// Target sets built ([`Device::share_targets`]); their members count in `targets`.
    pub target_sets: u64,
    pub retired_targets: u64,
    pub hold_nanos: u128,
    pub poll_nanos: u128,
    pub ib_nanos: u128,
    pub encode_nanos: u128,
    pub kernel_nanos: u128,
    pub mark_nanos: u128,
    pub retire_nanos: u128,
    pub paging_nanos: u128,
}

struct Session {
    adapter: Adapter,
    identity: AdapterIdentity,
    device: u32,
    paging: u32,
    paging_fence: MonitoredFence,
    fence_timeout: Duration,
    virtual_address_bits: u32,
    va_end: u64,
    nodes: Vec<NodeInfo>,
    poisoned: Cell<bool>,
    fault: RefCell<Option<DeviceState>>,
    live_allocations: Cell<usize>,
    queue_timings: RefCell<QueueTimings>,
}

impl Drop for Session {
    fn drop(&mut self) {
        if self.poisoned.get() {
            eprintln!("WDDM_FAULT_QUARANTINE: handles and allocations retained until process exit");
            self.adapter.0 = 0;
            return;
        }
        unsafe {
            if self.paging != 0 {
                D3DKMTDestroyPagingQueue(&mut D3DDDI_DESTROYPAGINGQUEUE {
                    hPagingQueue: self.paging,
                });
            }
            if self.device != 0 {
                D3DKMTDestroyDevice(&D3DKMT_DESTROYDEVICE {
                    hDevice: self.device,
                });
            }
        }
    }
}

impl Session {
    fn healthy(&self) -> Result<()> {
        if self.poisoned.get() {
            Err(Error::new(
                ErrorKind::DeviceLost,
                "device faulted; no further GPU/CPU buffer access",
            ))
        } else {
            Ok(())
        }
    }

    /// Quarantines the device: nothing the GPU may still use is released, every later operation
    /// fails, and the KMD's view of the device is recorded once.
    fn poison(&self, reason: &str) {
        if self.poisoned.replace(true) {
            return;
        }
        let state = DeviceState::query(self.device);
        eprintln!("WDDM_DEVICE_STATE reason=\"{reason}\" {state}");
        *self.fault.borrow_mut() = Some(state);
    }

    /// Waits for a fence value. Unless completion is confirmed the device is poisoned.
    fn wait(&self, monitored: &MonitoredFence, target: u64) -> Result<()> {
        self.reach(monitored, target, self.fence_timeout)
    }

    /// Waits for the paging fence to reach `target`, as [`Self::wait`] with the paging limit.
    fn wait_paging(&self, target: u64) -> Result<()> {
        self.reach(&self.paging_fence, target, PAGING_TIMEOUT)
    }

    fn reach(&self, monitored: &MonitoredFence, target: u64, timeout: Duration) -> Result<()> {
        let waiter = KmtFenceWait {
            device: self.device,
            fence: monitored,
        };
        match waiter.reach(target, timeout) {
            Ok(()) => {
                fence(Ordering::SeqCst);
                Ok(())
            }
            Err(error) => {
                self.poison(&error.message);
                Err(error)
            }
        }
    }
}

/// A GPU VA range reserved at a fixed base. Allocations mapped inside it keep it reserved, and
/// it owns their GPU VA: freeing it unmaps them.
struct Reservation {
    session: Rc<Session>,
    base: u64,
    size: u64,
    freed: Cell<bool>,
}

impl Reservation {
    fn new(session: &Rc<Session>, base: u64, size: u64) -> Self {
        Self {
            session: session.clone(),
            base,
            size,
            freed: Cell::new(false),
        }
    }

    /// Frees the whole range now, unmapping whatever is still mapped inside it. A failure
    /// quarantines the device.
    fn free_now(&self) -> Result<()> {
        if self.freed.replace(true) || self.session.device == 0 {
            return Ok(());
        }
        self.session.healthy()?;
        let result = unsafe {
            D3DKMTFreeGpuVirtualAddress(&D3DKMT_FREEGPUVIRTUALADDRESS {
                hAdapter: self.session.adapter.0,
                BaseAddress: self.base,
                Size: self.size,
            })
        };
        if result < 0 {
            let message = format!(
                "FreeGpuVirtualAddress(reservation 0x{:x}+0x{:x}) NTSTATUS=0x{:08x}",
                self.base, self.size, result as u32
            );
            self.session.poison(&message);
            return Err(Error::new(ErrorKind::System, message));
        }
        Ok(())
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.session.poisoned.get() {
            let _ = self.free_now();
        }
    }
}

/// A GPU VA window reserved at a fixed base (FINAL-REPORT §2-C), for mappings at chosen addresses
/// such as the guest window at the guest's own VAs.
pub struct VaReservation {
    inner: Rc<Reservation>,
}

impl VaReservation {
    pub fn base(&self) -> u64 {
        self.inner.base
    }
    pub fn size(&self) -> u64 {
        self.inner.size
    }
    pub fn contains(&self, address: u64, bytes: u64) -> bool {
        address >= self.inner.base
            && bytes <= self.inner.size
            && address - self.inner.base <= self.inner.size - bytes
    }
}

/// Where an allocation is mapped.
#[derive(Clone, Copy)]
enum Placement<'a> {
    /// Anywhere in the runtime window, below the KMD's va_end.
    RuntimeWindow,
    /// At `address`, inside `reservation`.
    Fixed {
        address: u64,
        reservation: &'a VaReservation,
    },
}

/// The last submission on each queue (by [`QueueKind::index`]) that used something.
struct QueueUses {
    last: [Cell<Option<(MonitoredFence, u64)>>; 58],
    /// The queues with an entry in `last`, one bit each.
    active: Cell<u64>,
}

impl QueueUses {
    fn new() -> Self {
        Self {
            last: std::array::from_fn(|_| Cell::new(None)),
            active: Cell::new(0),
        }
    }

    fn mark(&self, queue: QueueKind, fence: MonitoredFence, serial: u64) {
        let index = queue.index();
        self.last[index].set(Some((fence, serial)));
        self.active.set(self.active.get() | (1u64 << index));
    }

    /// Forgets a use once its submission has retired, so no fence of a destroyed queue is read.
    fn retire(&self, queue: QueueKind, completed: u64) {
        let index = queue.index();
        if self.last[index].get().is_some_and(|(_, serial)| serial <= completed) {
            self.last[index].set(None);
            self.active.set(self.active.get() & !(1u64 << index));
        }
    }

    fn wait(&self, session: &Session) -> Result<()> {
        let mut active = self.active.get();
        while active != 0 {
            let index = active.trailing_zeros() as usize;
            active &= active - 1;
            if let Some((fence, serial)) = self.last[index].get() {
                session.wait(&fence, serial)?;
            }
        }
        Ok(())
    }
}

/// Buffers a run of submissions all use, held and tracked as one, so that what a submission costs
/// does not grow with the memory the title has resident. radv does the same with one global BO
/// list that only changes when allocations do (`radv_amdgpu_global_bo_list_add`); WDDM takes no
/// list at submission, so here the set is only the buffers' lifetime and fence bookkeeping.
pub struct TargetSet {
    session: Rc<Session>,
    members: Vec<Rc<Allocation>>,
    uses: QueueUses,
}

impl TargetSet {
    fn new(session: Rc<Session>, members: Vec<Rc<Allocation>>) -> Rc<Self> {
        let set = Rc::new(Self {
            session,
            members,
            uses: QueueUses::new(),
        });
        let weak = Rc::downgrade(&set);
        for member in &set.members {
            member.join(&weak);
        }
        set
    }
}

/// Budget and usage of one memory segment group for this process.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SegmentUsage {
    pub budget: u64,
    pub usage: u64,
}

/// D3DKMTQueryVideoMemoryInfo for the local (carve-out) and non-local (system) segment groups.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VideoMemory {
    pub local: SegmentUsage,
    pub non_local: SegmentUsage,
}

struct Allocation {
    session: Rc<Session>,
    handle: u32,
    resource: u32,
    address: u64,
    size: usize,
    cpu: *mut u8,
    /// Whether `cpu` is a Lock2 mapping. A host import's CPU view is the caller's own memory.
    locked: bool,
    kind: MemoryKind,
    /// Whether the GPU mapping carries the Execute protection; normally `kind.is_executable()`.
    execute: bool,
    uses: QueueUses,
    /// The target sets this allocation is in; a set's uses are this allocation's uses too.
    sets: RefCell<Vec<Weak<TargetSet>>>,
    reservation: Option<Rc<Reservation>>,
}

impl Allocation {
    fn new(session: &Rc<Session>, kind: MemoryKind) -> Self {
        Self {
            session: session.clone(),
            handle: 0,
            resource: 0,
            address: 0,
            size: 0,
            cpu: ptr::null_mut(),
            locked: false,
            kind,
            execute: kind.is_executable(),
            uses: QueueUses::new(),
            sets: RefCell::new(Vec::new()),
            reservation: None,
        }
    }

    /// Waits until no queued submission still uses this allocation.
    fn await_gpu(&self) -> Result<()> {
        self.uses.wait(&self.session)?;
        for set in self.sets.borrow().iter().filter_map(Weak::upgrade) {
            set.uses.wait(&self.session)?;
        }
        Ok(())
    }

    /// Adds `set` to the sets this allocation is in, forgetting those already gone: a set outlives
    /// every submission that used it.
    fn join(&self, set: &Weak<TargetSet>) {
        let mut sets = self.sets.borrow_mut();
        sets.retain(|old| old.strong_count() != 0);
        sets.push(set.clone());
    }

    fn write(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.session.healthy()?;
        if self.cpu.is_null() {
            return Err(Error::invalid("buffer has no CPU mapping"));
        }
        if offset > self.size || bytes.len() > self.size - offset {
            return Err(Error::invalid("CPU write outside buffer"));
        }
        self.await_gpu()?;
        for (i, byte) in bytes.iter().enumerate() {
            unsafe {
                self.cpu.add(offset + i).write_volatile(*byte);
            }
        }
        fence(Ordering::SeqCst);
        Ok(())
    }

    fn write_words(&self, words: &[u32]) -> Result<()> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write(0, &bytes)
    }

    fn read(&self) -> Result<Vec<u8>> {
        self.session.healthy()?;
        if self.cpu.is_null() {
            return Err(Error::invalid("buffer has no CPU mapping"));
        }
        self.await_gpu()?;
        fence(Ordering::SeqCst);
        Ok((0..self.size)
            .map(|i| unsafe { self.cpu.add(i).read_volatile() })
            .collect())
    }

    /// What dropping the allocation releases, in this order.
    fn release(&self) -> Release {
        Release {
            unlock: self.locked && !self.cpu.is_null(),
            // A mapping inside a reservation is not a VA range of its own (FreeGpuVirtualAddress
            // refuses it with STATUS_INVALID_PARAMETER); destroying the allocation unmaps it.
            free_va: self.address != 0 && self.reservation.is_none(),
            destroy: self.handle != 0,
        }
    }
}

/// The OS objects an [`Allocation`] releases when dropped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Release {
    unlock: bool,
    free_va: bool,
    destroy: bool,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.session.poisoned.get() {
            return;
        }
        let release = self.release();
        unsafe {
            if release.unlock {
                let result = D3DKMTUnlock2(&D3DKMT_UNLOCK2 {
                    hDevice: self.session.device,
                    hAllocation: self.handle,
                });
                if result < 0 {
                    self.session
                        .poison(&format!("cleanup Unlock2 NTSTATUS=0x{:08x}", result as u32));
                    return;
                }
            }
            if release.free_va {
                let result = D3DKMTFreeGpuVirtualAddress(&D3DKMT_FREEGPUVIRTUALADDRESS {
                    hAdapter: self.session.adapter.0,
                    BaseAddress: self.address,
                    Size: self.size as u64,
                });
                if result < 0 {
                    self.session.poison(&format!(
                        "cleanup FreeGpuVirtualAddress NTSTATUS=0x{:08x}",
                        result as u32
                    ));
                    return;
                }
            }
            if release.destroy {
                let result = D3DKMTDestroyAllocation2(&D3DKMT_DESTROYALLOCATION2 {
                    hDevice: self.session.device,
                    hResource: self.resource,
                    phAllocationList: if self.resource == 0 {
                        &self.handle
                    } else {
                        ptr::null()
                    },
                    AllocationCount: u32::from(self.resource == 0),
                    ..Default::default()
                });
                if result < 0 {
                    self.session.poison(&format!(
                        "cleanup DestroyAllocation2 NTSTATUS=0x{:08x}",
                        result as u32
                    ));
                } else {
                    self.session
                        .live_allocations
                        .set(self.session.live_allocations.get() - 1);
                }
            }
        }
    }
}

/// A GPU buffer, usually with a CPU mapping. While a queued submission uses it its memory stays
/// allocated even if the `Buffer` is dropped, and CPU access ([`Buffer::read`],
/// [`Buffer::write`], [`Buffer::fill`]) first waits for those submissions to complete.
pub struct Buffer {
    allocation: Rc<Allocation>,
}

impl Buffer {
    pub fn address(&self) -> u64 {
        self.allocation.address
    }
    pub fn size(&self) -> usize {
        self.allocation.size
    }
    pub fn kind(&self) -> MemoryKind {
        self.allocation.kind
    }
    pub fn is_executable(&self) -> bool {
        self.allocation.kind.is_executable()
    }
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.allocation.write(offset, bytes)
    }
    pub fn read(&self) -> Result<Vec<u8>> {
        self.allocation.read()
    }
    pub fn fill(&mut self, byte: u8) -> Result<()> {
        self.write(0, &vec![byte; self.size()])
    }
    /// The persistent CPU mapping and its size, for writers on other threads that own a part
    /// of the buffer no submission has reached yet.
    pub(crate) fn mapping(&self) -> Option<(*mut u8, usize)> {
        (!self.allocation.cpu.is_null()).then(|| (self.allocation.cpu, self.allocation.size))
    }
}

/// The attribute ring of the gfx11 NGG pipeline: `bytes` at the 2 MiB-aligned `address` inside
/// `buffer`.
pub struct AttributeRing {
    pub buffer: Buffer,
    pub address: u64,
    pub bytes: u64,
}

/// What a submission keeps allocated until it retires, besides its IBs.
enum Held {
    /// Buffers only this submission lists.
    Listed(Vec<Rc<Allocation>>),
    /// A set many submissions share.
    Shared(Rc<TargetSet>),
}

impl Held {
    fn mark(&self, queue: QueueKind, fence: MonitoredFence, serial: u64) {
        match self {
            Self::Listed(allocations) => allocations.iter().for_each(|allocation| allocation.uses.mark(queue, fence, serial)),
            Self::Shared(set) => set.uses.mark(queue, fence, serial),
        }
    }

    /// Forgets the uses of a retired submission and returns how many records that took.
    fn retire(&self, queue: QueueKind, completed: u64) -> u64 {
        match self {
            Self::Listed(allocations) => {
                allocations.iter().for_each(|allocation| allocation.uses.retire(queue, completed));
                allocations.len() as u64
            }
            Self::Shared(set) => {
                set.uses.retire(queue, completed);
                1
            }
        }
    }
}

/// A queued submission and everything it uses: its IBs and the target buffers.
struct InFlight {
    serial: u64,
    ibs: Vec<Rc<Allocation>>,
    held: Held,
}

/// Creates, maps, makes resident (MustSucceed) and CPU-maps one device allocation.
fn allocate_raw(
    session: &Rc<Session>,
    requested: usize,
    kind: MemoryKind,
    placement: Placement,
) -> Result<Allocation> {
    allocate_mapped(session, requested, kind, kind.is_executable(), placement)
}

fn allocate_mapped(
    session: &Rc<Session>,
    requested: usize,
    kind: MemoryKind,
    execute: bool,
    placement: Placement,
) -> Result<Allocation> {
    session.healthy()?;
    if requested == 0 || requested > u32::MAX as usize - (AmdPrivateAbi::BUFFER_ALIGNMENT - 1) {
        return Err(Error::invalid("buffer size is zero or too large"));
    }
    let size =
        (requested + AmdPrivateAbi::BUFFER_ALIGNMENT - 1) & !(AmdPrivateAbi::BUFFER_ALIGNMENT - 1);
    let policy = if kind.bypasses_gl2() {
        CachePolicy::GL2_BYPASS
    } else {
        CachePolicy::GL2_CACHED
    };
    let (base_address, reservation, (private_data, mut records)) = match placement {
        Placement::RuntimeWindow => (0, None, AmdPrivateAbi.allocation_with(size as u32, policy)),
        Placement::Fixed {
            address,
            reservation,
        } => {
            check_fixed(session, reservation, address, size, RESERVATION_ALIGNMENT)?;
            (
                address,
                Some(reservation.inner.clone()),
                AmdPrivateAbi.allocation_at(size as u32, policy, address),
            )
        }
    };
    let mut a = Allocation::new(session, kind);
    a.size = size;
    a.execute = execute;
    a.reservation = reservation;
    let mut info = D3DDDI_ALLOCATIONINFO2 {
        pPrivateDriverData: records.as_mut_ptr().cast(),
        PrivateDriverDataSize: records.len() as u32,
        VidPnSourceId: u32::MAX,
        ..Default::default()
    };
    info.Anonymous2.Priority = D3DDDI_ALLOCATIONPRIORITY_NORMAL;
    create_allocation(session, &mut a, &private_data, &mut info)?;
    map_and_make_resident(session, &mut a, base_address)?;
    let mut lock = D3DKMT_LOCK2 {
        hDevice: session.device,
        hAllocation: a.handle,
        ..Default::default()
    };
    Kmt::status(unsafe { D3DKMTLock2(&mut lock) }, "Lock2")?;
    a.cpu = lock.pData.cast();
    a.locked = true;
    if a.cpu.is_null() {
        return Err(Error::new(ErrorKind::System, "Lock2 returned null"));
    }
    Ok(a)
}

/// Checks a fixed mapping of `bytes` at `address`: aligned, inside `reservation`, and on this
/// device.
fn check_fixed(
    session: &Rc<Session>,
    reservation: &VaReservation,
    address: u64,
    bytes: usize,
    alignment: u64,
) -> Result<()> {
    if !Rc::ptr_eq(session, &reservation.inner.session) {
        return Err(Error::invalid(
            "the VA reservation belongs to another device",
        ));
    }
    if address % alignment != 0 || !reservation.contains(address, bytes as u64) {
        return Err(Error::invalid(format!(
            "0x{address:x}+0x{bytes:x} is not {alignment}-byte aligned inside the reservation 0x{:x}+0x{:x}",
            reservation.base(),
            reservation.size()
        )));
    }
    Ok(())
}

fn create_allocation(
    session: &Rc<Session>,
    a: &mut Allocation,
    private_data: &[u8],
    info: &mut D3DDDI_ALLOCATIONINFO2,
) -> Result<()> {
    let mut request = D3DKMT_CREATEALLOCATION {
        hDevice: session.device,
        PrivateDriverDataSize: private_data.len() as u32,
        NumAllocations: 1,
        ..Default::default()
    };
    request.Anonymous1.pPrivateDriverData = private_data.as_ptr().cast();
    request.Anonymous2.pAllocationInfo2 = info;
    request.Flags._bitfield = CREATE_ALLOCATION_RESOURCE | CREATE_ALLOCATION_NON_SECURE;
    Kmt::status(
        unsafe { D3DKMTCreateAllocation2(&mut request) },
        "CreateAllocation2(AMD-v9)",
    )?;
    a.handle = info.hAllocation;
    a.resource = request.hResource;
    session
        .live_allocations
        .set(session.live_allocations.get() + 1);
    Ok(())
}

/// Maps `a` (at `base_address`, or anywhere in the runtime window when it is 0) and makes it
/// resident, waiting for both paging operations.
fn map_and_make_resident(
    session: &Rc<Session>,
    a: &mut Allocation,
    base_address: u64,
) -> Result<()> {
    let paging = map_and_queue_resident(session, a, base_address)?;
    session.wait_paging(paging)
}

/// As [`map_and_make_resident`], but only queues the paging work: returns the paging fence value
/// a later wait covers it with. Paging runs in order, so waiting for the last one is enough.
fn map_and_queue_resident(
    session: &Rc<Session>,
    a: &mut Allocation,
    base_address: u64,
) -> Result<u64> {
    let mut map = D3DDDI_MAPGPUVIRTUALADDRESS {
        hPagingQueue: session.paging,
        hAllocation: a.handle,
        SizeInPages: (a.size / GPU_PAGE_BYTES) as u64,
        ..Default::default()
    };
    if base_address != 0 {
        map.BaseAddress = base_address;
    } else {
        map.MinimumAddress = va::RUNTIME_BASE;
        map.MaximumAddress = session.va_end;
    }
    map.Protection.Anonymous.Value = if a.execute {
        GPU_VA_ALLOW_WRITE | GPU_VA_ALLOW_EXECUTE
    } else {
        GPU_VA_ALLOW_WRITE
    };
    Kmt::status(
        unsafe { D3DKMTMapGpuVirtualAddress(&mut map) },
        "MapGpuVirtualAddress",
    )?;
    a.address = map.VirtualAddress;
    if base_address != 0 && a.address != base_address {
        return Err(Error::new(
            ErrorKind::Unsupported,
            format!(
                "MapGpuVirtualAddress placed 0x{:x} instead of the requested 0x{base_address:x}",
                a.address
            ),
        ));
    }
    let mut resident = D3DDDI_MAKERESIDENT {
        hPagingQueue: session.paging,
        NumAllocations: 1,
        AllocationList: &a.handle,
        ..Default::default()
    };
    resident.Flags.Anonymous.Value = MAKE_RESIDENT_MUST_SUCCEED;
    Kmt::status(unsafe { D3DKMTMakeResident(&mut resident) }, "MakeResident")?;
    Ok(resident.PagingFenceValue.max(map.PagingFenceValue))
}

/// One hardware queue: its context and KMD queue, the queue's progress fence, the submissions in
/// flight on it and the IBs they draw from.
struct HwQueue {
    session: Rc<Session>,
    kind: QueueKind,
    context: u32,
    handle: u32,
    id: u32,
    progress_fence: MonitoredFence,
    progress_fence_gpu_va: u64,
    submitted: u64,
    completed: u64,
    free_ibs: Vec<Rc<Allocation>>,
    in_flight: VecDeque<InFlight>,
}

impl Drop for HwQueue {
    /// Waits for queued work so its IBs and buffers can be released. If completion cannot be
    /// confirmed the device is quarantined and keeps them, the queue and its context.
    fn drop(&mut self) {
        if self.submitted > self.completed && !self.session.poisoned.get() {
            let _ = self.wait(self.submitted);
        }
        if self.session.poisoned.get() {
            return;
        }
        self.in_flight.clear();
        self.free_ibs.clear();
        unsafe {
            if self.handle != 0 {
                D3DKMTDestroyHwQueue(&D3DKMT_DESTROYHWQUEUE {
                    hHwQueue: self.handle,
                });
            }
            if self.context != 0 {
                D3DKMTDestroyContext(&D3DKMT_DESTROYCONTEXT {
                    hContext: self.context,
                });
            }
        }
    }
}

impl HwQueue {
    fn new(session: &Rc<Session>, kind: QueueKind) -> Self {
        Self {
            session: session.clone(),
            kind,
            context: 0,
            handle: 0,
            id: 0,
            progress_fence: MonitoredFence::NONE,
            progress_fence_gpu_va: 0,
            submitted: 0,
            completed: 0,
            free_ibs: Vec::new(),
            in_flight: VecDeque::new(),
        }
    }

    /// Creates the context (AMD private record v9 for graphics, lfrb's compute record for
    /// compute) and the hardware queue on `node`.
    fn create(session: &Rc<Session>, kind: QueueKind, node: u32) -> Result<Self> {
        let mut queue = Self::new(session, kind);
        let mut context_data = match kind {
            QueueKind::Graphics => AmdPrivateAbi.context(),
            QueueKind::Compute | QueueKind::ComputeQueue(_) => AmdPrivateAbi.compute_context(),
        };
        let mut context = D3DKMT_CREATECONTEXTVIRTUAL {
            hDevice: session.device,
            NodeOrdinal: node,
            EngineAffinity: 1,
            pPrivateDriverData: context_data.as_mut_ptr().cast(),
            PrivateDriverDataSize: context_data.len() as u32,
            ClientHint: D3DKMT_CLIENTHINT_VULKAN,
            ..Default::default()
        };
        context.Flags.Anonymous.Value = CONTEXT_HW_QUEUE_SUPPORTED;
        Kmt::status(
            unsafe { D3DKMTCreateContextVirtual(&raw mut context) },
            "CreateContextVirtual(AMD-v9)",
        )?;
        queue.context = context.hContext;
        let mut queue_data = AmdPrivateAbi.queue();
        let mut create = D3DKMT_CREATEHWQUEUE {
            hHwContext: queue.context,
            pPrivateDriverData: queue_data.as_mut_ptr().cast(),
            PrivateDriverDataSize: queue_data.len() as u32,
            ..Default::default()
        };
        Kmt::status(
            unsafe { D3DKMTCreateHwQueue(&mut create) },
            "CreateHwQueue(AMD-v9)",
        )?;
        queue.handle = create.hHwQueue;
        queue.id = AmdPrivateAbi::queue_id(&queue_data);
        queue.progress_fence = MonitoredFence {
            object: create.hHwQueueProgressFence,
            value: create
                .HwQueueProgressFenceCPUVirtualAddress
                .cast_const()
                .cast(),
        };
        queue.progress_fence_gpu_va = create.HwQueueProgressFenceGPUVirtualAddress;
        Ok(queue)
    }

    fn wait(&mut self, serial: u64) -> Result<()> {
        if serial > self.submitted {
            return Err(Error::invalid(format!(
                "submission {serial} was never queued"
            )));
        }
        if serial > self.completed {
            self.session.healthy()?;
            self.session.wait(&self.progress_fence, serial)?;
            self.completed = serial;
        }
        self.poll();
        Ok(())
    }

    fn poll(&mut self) -> u64 {
        if let Some(value) = self.progress_fence.current() {
            self.completed = self.completed.max(value.min(self.submitted));
        }
        let retire_started = std::time::Instant::now();
        let mut retired = 0;
        while self
            .in_flight
            .front()
            .is_some_and(|f| f.serial <= self.completed)
        {
            let done = self.in_flight.pop_front().expect("checked above");
            retired += done.held.retire(self.kind, self.completed);
            for ib in done.ibs {
                ib.uses.retire(self.kind, self.completed);
                if self.free_ibs.len() < MAX_POOLED_IBS {
                    self.free_ibs.push(ib);
                }
            }
            // Targets whose `Buffer` the caller already dropped are released here.
            drop(done.held);
        }
        let mut timings = self.session.queue_timings.borrow_mut();
        timings.retire_nanos += retire_started.elapsed().as_nanos();
        timings.retired_targets += retired;
        self.completed
    }

    /// The smallest pooled IB of at least `bytes`, or a new one.
    fn take_ib(&mut self, bytes: usize) -> Result<Rc<Allocation>> {
        let best = self
            .free_ibs
            .iter()
            .enumerate()
            .filter(|(_, ib)| ib.size >= bytes)
            .min_by_key(|(_, ib)| ib.size)
            .map(|(index, _)| index);
        match best {
            Some(index) => Ok(self.free_ibs.swap_remove(index)),
            None => Ok(Rc::new(allocate_raw(
                &self.session,
                bytes,
                MemoryKind::Commands,
                Placement::RuntimeWindow,
            )?)),
        }
    }

    /// Submits a stream.
    fn submit(&mut self, words: &[u32], held: Held, layout: SubmitLayout) -> Result<u64> {
        let segments = plan_segments(words, layout)?;
        self.poll();
        if self.in_flight.len() >= MAX_IN_FLIGHT { return Err(Error::new(ErrorKind::System, "GPU queue is full; defer admission without blocking other queues")); }
        let mut ibs = Vec::with_capacity(segments.len());
        let chained = matches!(layout, SubmitLayout::Chained { .. });
        for segment in &segments {
            let dwords = segment.words.len() + CHAIN_OVERHEAD_DWORDS;
            ibs.push(self.take_ib(dwords * host_pm4::DWORD_BYTES)?);
        }
        let addresses: Vec<u64> = ibs.iter().map(|ib| ib.address).collect();
        let images = ib_images(words, &segments, &addresses, chained)?;
        for (ib, image) in ibs.iter().zip(&images) {
            ib.write_words(image)?;
        }
        let listed = if chained { 1 } else { segments.len() };
        let entries: Vec<IbEntry> = segments
            .iter()
            .zip(&images)
            .zip(&addresses)
            .take(listed)
            .map(|((segment, image), &address)| IbEntry {
                address,
                bytes: image.len() * host_pm4::DWORD_BYTES,
                flags: segment.flags,
            })
            .collect();
        self.submit_ibs(entries, ibs, held)
    }

    /// Submits IBs already written, each a multiple of 8 dwords: `images` in order.
    fn submit_images(&mut self, images: &[Vec<u32>], held: Held) -> Result<u64> {
        let polled = std::time::Instant::now();
        self.poll();
        self.session.queue_timings.borrow_mut().poll_nanos += polled.elapsed().as_nanos();
        if self.in_flight.len() >= MAX_IN_FLIGHT {
            return Err(Error::new(ErrorKind::System, "GPU queue is full; defer admission without blocking other queues"));
        }
        let prepared = std::time::Instant::now();
        let mut ibs = Vec::with_capacity(images.len());
        let mut entries = Vec::with_capacity(images.len());
        for image in images {
            let ib = self.take_ib(image.len() * host_pm4::DWORD_BYTES)?;
            ib.write_words(image)?;
            entries.push(IbEntry { address: ib.address, bytes: image.len() * host_pm4::DWORD_BYTES, flags: IbEntry::MAIN });
            ibs.push(ib);
        }
        self.session.queue_timings.borrow_mut().ib_nanos += prepared.elapsed().as_nanos();
        self.submit_ibs(entries, ibs, held)
    }

    /// Hands `entries` (the IBs in `ibs`) to the KMD.
    fn submit_ibs(&mut self, entries: Vec<IbEntry>, ibs: Vec<Rc<Allocation>>, held: Held) -> Result<u64> {
        let encoded = std::time::Instant::now();
        let mut private_data = AmdPrivateAbi.submission_entries(&entries)?;
        let serial = self
            .submitted
            .checked_add(1)
            .ok_or_else(|| Error::invalid("submission serial overflow"))?;
        let request = D3DKMT_SUBMITCOMMANDTOHWQUEUE {
            hHwQueue: self.handle,
            HwQueueProgressFenceId: serial,
            CommandBuffer: entries[0].address,
            CommandLength: entries[0].bytes as u32,
            pPrivateDriverData: private_data.as_mut_ptr().cast(),
            PrivateDriverDataSize: private_data.len() as u32,
            ..Default::default()
        };
        self.session.queue_timings.borrow_mut().encode_nanos += encoded.elapsed().as_nanos();
        let submitted = std::time::Instant::now();
        let status = Kmt::status(
            unsafe { D3DKMTSubmitCommandToHwQueue(&request) },
            "SubmitCommandToHwQueue(AMD-v9)",
        );
        {
            let mut timings = self.session.queue_timings.borrow_mut();
            timings.kernel_nanos += submitted.elapsed().as_nanos();
            timings.submissions += 1;
        }
        let marked = std::time::Instant::now();
        // The IBs and targets stay held either way: a failed submission quarantines the device,
        // which then releases nothing.
        if status.is_ok() {
            for ib in &ibs {
                ib.uses.mark(self.kind, self.progress_fence, serial);
            }
            held.mark(self.kind, self.progress_fence, serial);
        }
        self.session.queue_timings.borrow_mut().mark_nanos += marked.elapsed().as_nanos();
        self.in_flight.push_back(InFlight { serial, ibs, held });
        if let Err(error) = status {
            self.session.poison(&error.message);
            return Err(error);
        }
        self.submitted = serial;
        Ok(serial)
    }
}

pub struct Device {
    session: Rc<Session>,
    graphics: HwQueue,
    compute: Option<HwQueue>,
    compute_queues: std::collections::BTreeMap<u8, HwQueue>,
    coherent_imports: bool,
    last_health_query: std::time::Instant,
    /// The newest paging fence value imports have queued and nothing has waited for yet.
    pending_paging: u64,
}

impl Device {
    pub fn enumerate_adapters() -> Result<Vec<AdapterIdentity>> {
        Ok(AdapterInventory
            .adapters()?
            .into_iter()
            .map(|(_, id)| id)
            .collect())
    }

    pub fn open() -> Result<Self> {
        let (adapter, identity) = AdapterInventory
            .adapters()?
            .into_iter()
            .find(|(_, id)| {
                id.vendor_id == AmdPrivateAbi::VENDOR_ID
                    && id.device_id == AmdPrivateAbi::DEVICE_ID
                    && id.driver_version == AmdPrivateAbi::DRIVER_VERSION
                    && id.revision == AmdPrivateAbi::REVISION_ID
            })
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "no adapter matches verified profile {}",
                    AmdPrivateAbi::PROFILE
                ))
            })?;
        let mut caps = D3DKMT_QUERY_GPUMMU_CAPS::default();
        let virtual_address_bits =
            if Kmt::query(adapter.0, KMTQAITYPE_QUERY_GPUMMU_CAPS, &mut caps).is_ok() {
                caps.Caps.VirtualAddressBitCount
            } else {
                0
            };
        let mut s = Session {
            adapter,
            identity,
            device: 0,
            paging: 0,
            paging_fence: MonitoredFence::NONE,
            fence_timeout: FENCE_TIMEOUT,
            virtual_address_bits,
            va_end: va::va_end(virtual_address_bits),
            nodes: Vec::new(),
            poisoned: Cell::new(false),
            fault: RefCell::new(None),
            live_allocations: Cell::new(0),
            queue_timings: RefCell::new(QueueTimings::default()),
        };
        let mut create = D3DKMT_CREATEDEVICE::default();
        create.Anonymous.hAdapter = s.adapter.0;
        Kmt::status(unsafe { D3DKMTCreateDevice(&mut create) }, "CreateDevice")?;
        s.device = create.hDevice;
        let mut paging = D3DKMT_CREATEPAGINGQUEUE {
            hDevice: s.device,
            ..Default::default()
        };
        Kmt::status(
            unsafe { D3DKMTCreatePagingQueue(&mut paging) },
            "CreatePagingQueue",
        )?;
        s.paging = paging.hPagingQueue;
        s.paging_fence = MonitoredFence {
            object: paging.hSyncObject,
            value: paging.FenceValueCPUVirtualAddress.cast_const().cast(),
        };
        s.nodes = Self::query_nodes(&s)?;
        let node = s
            .nodes
            .iter()
            .find(|n| n.engine_type == DXGK_ENGINE_TYPE_3D)
            .map(|n| n.ordinal)
            .ok_or_else(|| Error::unsupported("no graphics node"))?;
        let session = Rc::new(s);
        let graphics = HwQueue::create(&session, QueueKind::Graphics, node)?;
        Ok(Self {
            session,
            graphics,
            compute: None,
            compute_queues: std::collections::BTreeMap::new(),
            coherent_imports: false,
            last_health_query: std::time::Instant::now(),
            pending_paging: 0,
        })
    }

    fn query_nodes(s: &Session) -> Result<Vec<NodeInfo>> {
        let mut statistics = D3DKMT_QUERYSTATISTICS {
            Type: D3DKMT_QUERYSTATISTICS_ADAPTER,
            AdapterLuid: LUID {
                LowPart: s.identity.luid_low,
                HighPart: s.identity.luid_high,
            },
            ..Default::default()
        };
        Kmt::status(
            unsafe { D3DKMTQueryStatistics(&raw mut statistics) },
            "QueryStatistics",
        )?;
        let count = unsafe { statistics.QueryResult.AdapterInformation.NodeCount };
        let mut nodes = Vec::new();
        for ordinal in 0..count.min(ADAPTER_NODE_LIMIT) {
            let mut metadata = D3DKMT_NODEMETADATA {
                NodeOrdinalAndAdapterIndex: ordinal,
                ..Default::default()
            };
            if Kmt::query(s.adapter.0, KMTQAITYPE_NODEMETADATA, &mut metadata).is_ok() {
                let data = metadata.NodeData;
                let name = { data.FriendlyName };
                let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
                nodes.push(NodeInfo {
                    ordinal,
                    engine_type: data.EngineType,
                    name: String::from_utf16_lossy(&name[..end]),
                });
            }
        }
        Ok(nodes)
    }

    pub fn identity(&self) -> &AdapterIdentity {
        &self.session.identity
    }
    pub fn profile(&self) -> &'static str {
        AmdPrivateAbi::PROFILE
    }
    /// The adapter's engine nodes.
    pub fn nodes(&self) -> &[NodeInfo] {
        &self.session.nodes
    }
    /// The last serial queued on the graphics queue.
    pub fn submitted(&self) -> u64 {
        self.graphics.submitted
    }
    /// The last serial known complete on the graphics queue.
    pub fn completed(&self) -> u64 {
        self.graphics.completed
    }
    /// The last serial known complete on `queue` (0 for a compute queue that is not open).
    pub fn completed_on(&self, queue: QueueKind) -> u64 {
        match queue {
            QueueKind::Graphics => self.graphics.completed,
            QueueKind::Compute => self.compute.as_ref().map_or(0, |queue| queue.completed),
            QueueKind::ComputeQueue(index) => self.compute_queues.get(&index).map_or(0, |queue| queue.completed),
        }
    }
    pub fn live_allocations(&self) -> usize {
        self.session.live_allocations.get()
    }
    pub fn is_faulted(&self) -> bool {
        self.session.poisoned.get()
    }
    /// The identifier the KMD assigned to the graphics queue.
    pub fn queue_id(&self) -> u32 {
        self.graphics.id
    }
    /// Where dxgkrnl mapped the graphics queue's progress fence in this process's GPU VA space.
    pub fn queue_fence_gpu_va(&self) -> u64 {
        self.graphics.progress_fence_gpu_va
    }
    /// Where dxgkrnl mapped the compute queue's progress fence, once the queue is open.
    pub fn compute_queue_fence_gpu_va(&self) -> Option<u64> {
        self.compute
            .as_ref()
            .map(|queue| queue.progress_fence_gpu_va)
    }
    /// The identifier of the compute queue, once [`Self::open_compute_queue`] created it.
    pub fn compute_queue_id(&self) -> Option<u32> {
        self.compute.as_ref().map(|q| q.id)
    }
    /// The GPU VA bit count from QUERY_GPUMMU_CAPS (0 when the query failed).
    pub fn virtual_address_bits(&self) -> u32 {
        self.session.virtual_address_bits
    }
    /// The highest GPU VA any mapping may use.
    pub fn va_end(&self) -> u64 {
        self.session.va_end
    }
    /// Submissions in flight on the graphics queue.
    pub fn in_flight(&self) -> usize {
        self.graphics.in_flight.len()
    }
    /// Retired IBs the graphics queue keeps for reuse.
    pub fn pooled_ibs(&self) -> usize {
        self.graphics.free_ibs.len()
    }
    /// The KMD's view of the device recorded when it was quarantined, if it was.
    pub fn fault_report(&self) -> Option<DeviceState> {
        self.session.fault.borrow().clone()
    }

    /// Asks the KMD for the device's execution state, then its page-fault state (faulted VA,
    /// error code, pipeline stage, flags), one D3DKMTGetDeviceState call each. Works on a
    /// quarantined device too.
    pub fn query_device_state(&self) -> DeviceState {
        DeviceState::query(self.session.device)
    }

    /// The same query as [`Self::query_device_state`].
    pub fn device_state(&self) -> DeviceState {
        self.query_device_state()
    }

    /// Opens a hardware queue on the adapter's compute node ("Compute 0", node 2 on the 8060S)
    /// and returns its KMD queue identifier.
    pub fn open_compute_queue(&mut self) -> Result<u32> {
        if let Some(queue) = &self.compute {
            return Ok(queue.id);
        }
        self.session.healthy()?;
        let node = self
            .session
            .nodes
            .iter()
            .find(|n| n.name.to_ascii_lowercase().contains("compute"))
            .map(|n| n.ordinal)
            .ok_or_else(|| Error::unsupported("the adapter reports no compute node"))?;
        let queue = HwQueue::create(&self.session, QueueKind::Compute, node)?;
        let id = queue.id;
        self.compute = Some(queue);
        Ok(id)
    }

    pub fn open_guest_compute_queue(&mut self, index: u8) -> Result<u32> {
        if index == 0 { return self.open_compute_queue(); }
        if index > 56 { return Err(Error::invalid("compute queue index exceeds the guest queue range")); }
        if let Some(queue) = self.compute_queues.get(&index) { return Ok(queue.id); }
        self.session.healthy()?;
        let node = self.session.nodes.iter().find(|n| n.name.to_ascii_lowercase().contains("compute")).map(|n| n.ordinal).ok_or_else(|| Error::unsupported("the adapter reports no compute node"))?;
        let queue = HwQueue::create(&self.session, QueueKind::ComputeQueue(index), node)?;
        let id = queue.id;
        self.compute_queues.insert(index, queue);
        Ok(id)
    }

    pub fn use_coherent_host_imports(&mut self) { self.coherent_imports = true; }

    pub fn check_async_health(&mut self) -> Result<()> {
        self.session.healthy()?;
        if self.last_health_query.elapsed() < std::time::Duration::from_millis(250) { return Ok(()); }
        self.last_health_query = std::time::Instant::now();
        let state = self.query_device_state();
        if state.execution.is_some_and(|state| state != ExecutionState::Active) {
            self.session.poison(&format!("asynchronous GPU execution failed: {state}"));
            return Err(Error::new(ErrorKind::DeviceLost, format!("asynchronous GPU execution failed: {state}")));
        }
        Ok(())
    }

    /// A CPU-visible buffer the GPU writes and the CPU reads back (GL2 bypass).
    pub fn allocate(&mut self, requested: usize) -> Result<Buffer> {
        self.allocate_kind(requested, MemoryKind::Readback)
    }

    /// A buffer whose GPU mapping permits instruction fetch, for shader code. The AMD private
    /// allocation records are those of a cached data buffer; executability is the OS GPU-VA
    /// protection (`Execute`), which the KMD turns into executable page-table entries.
    pub fn allocate_executable(&mut self, requested: usize) -> Result<Buffer> {
        self.allocate_kind(requested, MemoryKind::Code)
    }

    /// Experiment S5 only: shader-code memory the validator treats as code, mapped without the
    /// Execute protection, to measure whether the GPU MMU enforces no-execute. A dispatch from it
    /// is expected to page-fault when enforcement is on.
    pub fn allocate_code_without_execute(&mut self, requested: usize) -> Result<Buffer> {
        Ok(Buffer {
            allocation: Rc::new(allocate_mapped(
                &self.session,
                requested,
                MemoryKind::Code,
                false,
                Placement::RuntimeWindow,
            )?),
        })
    }

    /// This process's video-memory budget and usage in the local and non-local segment groups.
    pub fn video_memory(&self) -> Result<VideoMemory> {
        let query = |group: D3DKMT_MEMORY_SEGMENT_GROUP| -> Result<SegmentUsage> {
            let mut info = D3DKMT_QUERYVIDEOMEMORYINFO {
                hProcess: ptr::null_mut(),
                hAdapter: self.session.adapter.0,
                MemorySegmentGroup: group,
                ..Default::default()
            };
            Kmt::status(
                unsafe { D3DKMTQueryVideoMemoryInfo(&mut info) },
                "QueryVideoMemoryInfo",
            )?;
            Ok(SegmentUsage {
                budget: info.Budget,
                usage: info.CurrentUsage,
            })
        };
        Ok(VideoMemory {
            local: query(D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL)?,
            non_local: query(D3DKMT_MEMORY_SEGMENT_GROUP_NON_LOCAL)?,
        })
    }

    /// A buffer of `kind` anywhere in the runtime VA window.
    pub fn allocate_kind(&mut self, requested: usize, kind: MemoryKind) -> Result<Buffer> {
        Ok(Buffer {
            allocation: Rc::new(allocate_raw(
                &self.session,
                requested,
                kind,
                Placement::RuntimeWindow,
            )?),
        })
    }

    /// An attribute ring in the runtime window, GL2-cached, with room to start on a 2 MiB
    /// boundary.
    pub fn allocate_attribute_ring(&mut self) -> Result<AttributeRing> {
        let buffer = self.allocate_kind(
            (va::ATTRIBUTE_RING_BYTES + va::ATTRIBUTE_RING_ALIGNMENT) as usize,
            MemoryKind::Device,
        )?;
        let address = buffer
            .address()
            .next_multiple_of(va::ATTRIBUTE_RING_ALIGNMENT);
        Ok(AttributeRing {
            buffer,
            address,
            bytes: va::ATTRIBUTE_RING_BYTES,
        })
    }

    /// Reserves the GPU VA window `[base, base + size)` (64 KiB aligned) with the adapter, so
    /// later mappings can be placed at fixed addresses inside it and nothing else lands there.
    pub fn reserve_va(&mut self, base: u64, size: u64) -> Result<VaReservation> {
        self.reserve_va_with(base, size, D3DDDIGPUVA_RESERVE_NO_ACCESS)
    }

    /// As [`Self::reserve_va`], except that every address the reservation leaves unmapped reads
    /// zero and drops writes instead of faulting the device (`D3DDDIGPUVA_RESERVE_ZERO`, what
    /// tiled resources map their null tiles to).
    pub fn reserve_va_zero(&mut self, base: u64, size: u64) -> Result<VaReservation> {
        self.reserve_va_with(base, size, D3DDDIGPUVA_RESERVE_ZERO)
    }

    fn reserve_va_with(&mut self, base: u64, size: u64, kind: i32) -> Result<VaReservation> {
        self.session.healthy()?;
        let end = base.checked_add(size);
        if base == 0
            || size == 0
            || base % RESERVATION_ALIGNMENT != 0
            || size % RESERVATION_ALIGNMENT != 0
            || end.is_none_or(|end| end - 1 > self.session.va_end)
        {
            return Err(Error::invalid(format!(
                "VA window 0x{base:x}+0x{size:x} is not 64 KiB aligned below va_end 0x{:x}",
                self.session.va_end
            )));
        }
        let mut reserve = D3DDDI_RESERVEGPUVIRTUALADDRESS {
            BaseAddress: base,
            Size: size,
            ..Default::default()
        };
        // Mapping the zero page is a paging operation: such a reservation names the paging
        // queue, and its fence is waited for below.
        if kind == D3DDDIGPUVA_RESERVE_ZERO {
            reserve.Anonymous1.hPagingQueue = self.session.paging;
        } else {
            reserve.Anonymous1.hAdapter = self.session.adapter.0;
        }
        reserve.Anonymous2.ReservationType = kind;
        Kmt::status(
            unsafe { D3DKMTReserveGpuVirtualAddress(&mut reserve) },
            "ReserveGpuVirtualAddress",
        )?;
        let reservation = VaReservation {
            inner: Rc::new(Reservation::new(
                &self.session,
                reserve.VirtualAddress,
                size,
            )),
        };
        if reserve.VirtualAddress != base {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!(
                    "ReserveGpuVirtualAddress placed 0x{:x} instead of 0x{base:x}",
                    reserve.VirtualAddress
                ),
            ));
        }
        let paging = unsafe { reserve.Anonymous4.PagingFenceValue };
        if paging != 0 {
            self.session.wait_paging(paging)?;
        }
        Ok(reservation)
    }

    /// A device buffer of `kind` mapped at the fixed GPU address `address` (64 KiB aligned)
    /// inside `reservation`.
    pub fn allocate_at(
        &mut self,
        requested: usize,
        kind: MemoryKind,
        address: u64,
        reservation: &VaReservation,
    ) -> Result<Buffer> {
        Ok(Buffer {
            allocation: Rc::new(allocate_raw(
                &self.session,
                requested,
                kind,
                Placement::Fixed {
                    address,
                    reservation,
                },
            )?),
        })
    }

    /// Imports `bytes` of host memory at `memory` as a pinned AMD allocation (lfrb's
    /// host-pointer records) mapped at the fixed GPU address `address` inside `reservation`, and
    /// makes it resident. `memory` and `address` are page aligned and `bytes` is a whole number
    /// of pages, at most 128 MiB. The CPU view of the returned buffer is `memory` itself.
    ///
    /// # Safety
    ///
    /// `memory` must stay committed, readable and writable, and must not be freed or
    /// re-protected while the buffer or any submission that uses it lives.
    pub unsafe fn import_host_memory(
        &mut self,
        memory: *mut u8,
        bytes: usize,
        address: u64,
        reservation: &VaReservation,
    ) -> Result<Buffer> {
        self.session.healthy()?;
        let page = AmdPrivateAbi::HOST_ALIGNMENT;
        if memory.is_null()
            || memory as usize % page != 0
            || bytes == 0
            || bytes % page != 0
            || bytes > HOST_IMPORT_MAX_BYTES
        {
            return Err(Error::invalid(format!(
                "host import of 0x{bytes:x} bytes at {memory:p} is not whole pages of at most 128 MiB"
            )));
        }
        check_fixed(&self.session, reservation, address, bytes, page as u64)?;
        let (private_data, mut records) = if self.coherent_imports { AmdPrivateAbi.host_allocation_with_policy(bytes as u32, address, CachePolicy::GL2_BYPASS) } else { AmdPrivateAbi.host_allocation(bytes as u32, address) };
        let mut a = Allocation::new(&self.session, MemoryKind::Data);
        a.size = bytes;
        a.reservation = Some(reservation.inner.clone());
        let mut info = D3DDDI_ALLOCATIONINFO2 {
            pPrivateDriverData: records.as_mut_ptr().cast(),
            PrivateDriverDataSize: records.len() as u32,
            VidPnSourceId: u32::MAX,
            ..Default::default()
        };
        info.Anonymous1.pSystemMem = memory.cast_const().cast();
        info.Anonymous2.Priority = D3DDDI_ALLOCATIONPRIORITY_NORMAL;
        create_allocation(&self.session, &mut a, &private_data, &mut info)?;
        let paging = map_and_queue_resident(&self.session, &mut a, address)?;
        self.pending_paging = self.pending_paging.max(paging);
        a.cpu = memory;
        Ok(Buffer {
            allocation: Rc::new(a),
        })
    }

    /// Waits for the paging work imports queued since the last call.
    pub fn settle_paging(&mut self) -> Result<()> {
        let target = std::mem::take(&mut self.pending_paging);
        if target == 0 {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let settled = self.session.wait_paging(target);
        self.session.queue_timings.borrow_mut().paging_nanos += started.elapsed().as_nanos();
        settled
    }

    pub fn take_queue_timings(&self) -> QueueTimings { std::mem::take(&mut *self.session.queue_timings.borrow_mut()) }

    /// Waits until no queued submission uses `buffer`, then retires those that did. Returns
    /// whether `buffer` then holds its memory alone, so that dropping it destroys the memory.
    pub fn await_buffer(&mut self, buffer: &Buffer) -> Result<bool> {
        buffer.allocation.await_gpu()?;
        self.poll();
        Ok(Rc::strong_count(&buffer.allocation) == 1)
    }

    /// Queues host-format PM4 as one IB on the graphics queue without waiting and
    /// returns its serial; see [`Self::queue_host`]. Track completion with [`Self::wait`] or
    /// [`Self::poll`]; CPU access to the targets also waits for it.
    pub fn submit_host(&mut self, words: &[u32], targets: &[&Buffer]) -> Result<u64> {
        self.queue_host(words, targets, SubmitLayout::Single)
    }

    /// Queues host-format PM4 on the graphics queue without waiting and returns its serial for
    /// [`Self::wait`]. The targets stay allocated until the submission retires. A failed submission or an
    /// unconfirmed completion quarantines the device.
    pub fn queue_host(
        &mut self,
        words: &[u32],
        targets: &[&Buffer],
        layout: SubmitLayout,
    ) -> Result<u64> {
        self.queue_host_on(QueueKind::Graphics, words, targets, layout)
    }

    /// [`Self::queue_host`] on either queue.
    pub fn queue_host_on(
        &mut self,
        queue: QueueKind,
        words: &[u32],
        targets: &[&Buffer],
        layout: SubmitLayout,
    ) -> Result<u64> {
        self.session.healthy()?;
        self.settle_paging()?;
        let held = self.hold(targets)?;
        self.hw_queue(queue)?.submit(words, Held::Listed(held), layout)
    }

    /// Queues IBs already written, each a multiple of 8 dwords, on `queue` without waiting and
    /// returns the submission's serial. `targets` are what they use besides the IBs.
    pub fn queue_images(&mut self, queue: QueueKind, images: &[Vec<u32>], targets: &Rc<TargetSet>) -> Result<u64> {
        self.session.healthy()?;
        self.settle_paging()?;
        if !Rc::ptr_eq(&self.session, &targets.session) {
            return Err(Error::invalid("submission targets belong to another device"));
        }
        self.hw_queue(queue)?.submit_images(images, Held::Shared(targets.clone()))
    }

    /// Holds `targets` as one set a run of submissions shares ([`Self::queue_images`]). Dropping
    /// it lets a buffer go once the submissions that used it have retired.
    pub fn share_targets(&self, targets: &[&Buffer]) -> Result<Rc<TargetSet>> {
        let members = self.hold(targets)?;
        let started = std::time::Instant::now();
        let set = TargetSet::new(self.session.clone(), members);
        let mut timings = self.session.queue_timings.borrow_mut();
        timings.hold_nanos += started.elapsed().as_nanos();
        timings.target_sets += 1;
        Ok(set)
    }

    fn hw_queue(&mut self, queue: QueueKind) -> Result<&mut HwQueue> {
        match queue {
            QueueKind::Graphics => Ok(&mut self.graphics),
            QueueKind::Compute => self
                .compute
                .as_mut()
                .ok_or_else(|| Error::invalid("the compute queue is not open")),
            QueueKind::ComputeQueue(index) => self.compute_queues.get_mut(&index).ok_or_else(|| Error::invalid("the guest compute queue is not open")),
        }
    }

    pub fn can_queue_host_on(&mut self, queue: QueueKind) -> bool {
        self.poll();
        self.hw_queue(queue).is_ok_and(|queue| queue.in_flight.len() < MAX_IN_FLIGHT)
    }

    /// Waits until graphics submission `serial` has completed, then releases what finished
    /// submissions held.
    pub fn wait(&mut self, serial: u64) -> Result<()> {
        self.graphics.wait(serial)
    }

    /// [`Self::wait`] for a submission on either queue.
    pub fn wait_on(&mut self, queue: QueueKind, serial: u64) -> Result<()> {
        self.hw_queue(queue)?.wait(serial)
    }

    /// Waits until every queue is idle.
    pub fn wait_idle(&mut self) -> Result<()> {
        let serial = self.graphics.submitted;
        self.graphics.wait(serial)?;
        if let Some(compute) = &mut self.compute {
            let serial = compute.submitted;
            compute.wait(serial)?;
        }
        for compute in self.compute_queues.values_mut() { compute.wait(compute.submitted)?; }
        Ok(())
    }

    /// Retires every submission the progress fences show complete and returns the graphics
    /// queue's completed serial. Never blocks.
    pub fn poll(&mut self) -> u64 {
        for compute in self.compute_queues.values_mut() { compute.poll(); }
        if let Some(compute) = &mut self.compute {
            compute.poll();
        }
        self.graphics.poll()
    }

    fn hold(&self, targets: &[&Buffer]) -> Result<Vec<Rc<Allocation>>> {
        let started = std::time::Instant::now();
        let held = targets
            .iter()
            .map(|target| {
                if Rc::ptr_eq(&self.session, &target.allocation.session) {
                    Ok(target.allocation.clone())
                } else {
                    Err(Error::invalid(
                        "submission target belongs to another device",
                    ))
                }
            })
            .collect();
        let mut timings = self.session.queue_timings.borrow_mut();
        timings.hold_nanos += started.elapsed().as_nanos();
        timings.targets += targets.len() as u64;
        held
    }
}

impl Device {
    pub fn abi_fixture() -> Vec<u8> {
        let (a, b) = AmdPrivateAbi.allocation(AmdPrivateAbi::BUFFER_ALIGNMENT as u32);
        [
            AmdPrivateAbi.context().as_slice(),
            AmdPrivateAbi.queue().as_slice(),
            a.as_slice(),
            b.as_slice(),
            AmdPrivateAbi
                .submission(ABI_FIXTURE_GPU_ADDRESS, ABI_FIXTURE_IB_BYTES)
                .unwrap()
                .as_slice(),
        ]
        .concat()
    }
}
