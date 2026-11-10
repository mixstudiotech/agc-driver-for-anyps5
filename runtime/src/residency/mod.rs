//! GPU residency of same-VA guest memory (guest-memory-integration.md §9): the guest pages
//! imported into the device's guest window, made on demand for each submission, released when
//! AnyPS5 changes the guest mapping, and kept under an import budget.
//!
//! [`GuestResidency`] owns the guest window and, for every import, the alias AnyPS5 made for it
//! (`map_import`). It hands each alias back (`unmap_import`) exactly once, after its GPU import is
//! gone, and never on a lost device, which keeps its imports pinned until process exit.
//!
//! A submission:
//!
//! ```ignore
//! let residents = residency.prepare(&mut device, Reach { ranges: &ranges, computed_addresses })?;
//! let mut targets = residency.targets(&residents)?;
//! targets.extend(runtime_buffers);
//! let serial = device.queue_host(&words, &targets, layout)?;
//! residency.queued(&residents, Submission::graphics(serial));
//! ```
//!
//! A submission that renders into or samples host-layout copies of guest surfaces also calls
//! [`GuestResidency::ensure_host_copy`] for each and passes the copy memory as a target. A run of
//! submissions that all use the same memory passes [`GuestResidency::shared_targets`] instead of
//! listing it each time.
//!
//! Everything runs on the thread that owns the device; like the device, residency is not `Send`.

pub mod sparse;
#[cfg(all(feature = "wddm", target_os = "windows"))]
mod wddm;

#[cfg(all(feature = "wddm", target_os = "windows"))]
pub use wddm::{guest_window_reserved, WddmWindow};

use crate::{
    backend::va::{self, GUEST_PAGE_BYTES, HOST_COPY_OFFSET, IMPORT_WINDOW_BYTES},
    types::AccessRange,
    Error, ErrorKind, Result,
};
use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet, HashMap},
    ffi::c_void,
    fmt,
    ptr::NonNull,
    sync::Arc,
};

/// A range of guest addresses.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GuestRange {
    pub address: u64,
    pub bytes: u64,
}

impl GuestRange {
    pub const fn new(address: u64, bytes: u64) -> Self {
        Self { address, bytes }
    }

    /// The first address past the range. `address + bytes` must not overflow.
    pub const fn end(self) -> u64 {
        self.address + self.bytes
    }

    fn intersection(self, other: Self) -> Option<Self> {
        let start = self.address.max(other.address);
        let end = self.end().min(other.end());
        (start < end).then(|| Self::new(start, end - start))
    }
}

impl From<AccessRange> for GuestRange {
    fn from(access: AccessRange) -> Self {
        Self::new(access.address.raw(), access.bytes as u64)
    }
}

impl fmt::Display for GuestRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:x}+0x{:x}", self.address, self.bytes)
    }
}

/// The hardware queue a submission went to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Queue {
    Graphics,
    Compute,
    ComputeQueue(u8),
}

impl Queue {
    pub const ALL: [Queue; 2] = [Queue::Graphics, Queue::Compute];

    pub fn from_guest(raw: u32) -> Self {
        if raw < 0x20 { Self::Graphics } else if raw == 0x20 { Self::Compute } else { Self::ComputeQueue((raw - 0x20) as u8) }
    }

    pub fn index(self) -> usize {
        match self {
            Queue::Graphics => 0,
            Queue::Compute => 1,
            Queue::ComputeQueue(index) => 1 + usize::from(index),
        }
    }
}

/// A queued submission: its queue and its serial there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Submission {
    pub queue: Queue,
    pub serial: u64,
}

impl Submission {
    pub const fn graphics(serial: u64) -> Self {
        Self {
            queue: Queue::Graphics,
            serial,
        }
    }

    pub const fn compute(serial: u64) -> Self {
        Self {
            queue: Queue::Compute,
            serial,
        }
    }
}

/// Why AnyPS5 made no import alias: `GuestArena::ImportRefusal`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportRefusal {
    /// Not wholly registered GPU-visible guest memory, or a mapping change is under way there.
    NotGpuVisible,
    /// GPU-visible memory with no section behind it: flexible memory (anonymous, zero-page or
    /// guest-heap pages). AnyPS5 reports each such allocation once, with where it became
    /// GPU-visible.
    NotSectionMemory,
    NotWholePages,
    CrossesWindow,
    /// A host write stayed open on resident direct memory in the range past `map_import`'s wait.
    HostWriteOpen,
    /// A code this runtime does not know.
    Unknown(u32),
}

impl ImportRefusal {
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => Self::NotGpuVisible,
            2 => Self::NotSectionMemory,
            3 => Self::NotWholePages,
            4 => Self::CrossesWindow,
            5 => Self::HostWriteOpen,
            other => Self::Unknown(other),
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Self::NotGpuVisible => 1,
            Self::NotSectionMemory => 2,
            Self::NotWholePages => 3,
            Self::CrossesWindow => 4,
            Self::HostWriteOpen => 5,
            Self::Unknown(code) => code,
        }
    }

    fn reason(self) -> String {
        match self {
            Self::NotGpuVisible => "it is not wholly registered GPU-visible guest memory, or a mapping change is under way there".into(),
            Self::NotSectionMemory => "it is GPU-visible flexible memory with no section behind it, and same-VA imports need section-backed (direct) memory; AnyPS5 logged the allocation once with where it became GPU-visible".into(),
            Self::NotWholePages => "AnyPS5 found it is not whole 16 KiB guest pages".into(),
            Self::CrossesWindow => "AnyPS5 found it crosses an 8 MiB import window".into(),
            Self::HostWriteOpen => "a host write stayed open on resident direct memory there for longer than map_import waits".into(),
            Self::Unknown(code) => format!("AnyPS5 refused it with unknown refusal code {code}"),
        }
    }
}

/// AnyPS5's guest memory. Called on
/// the device's owning thread; an implementation never calls back into the residency.
pub trait GuestMemory {
    fn mapping_generation(&self) -> u64;
    /// A read-write alias of the guest pages `[guest, guest + bytes)`, whole 16 KiB pages inside
    /// one 8 MiB import window, that shares their pages until [`Self::unmap_import`].
    fn map_import(
        &mut self,
        guest: u64,
        bytes: u64,
    ) -> std::result::Result<NonNull<c_void>, ImportRefusal>;

    /// Unmaps an alias [`Self::map_import`] returned. Called once per alias, after its GPU import
    /// is gone.
    fn unmap_import(&mut self, alias: NonNull<c_void>);
    fn import_pages(&self, alias: NonNull<c_void>, bytes: u64) -> Result<Vec<u64>>;
    fn map_backing(&mut self, guest: u64, bytes: u64) -> Result<NonNull<c_void>>;
    fn unmap_backing(&mut self, alias: NonNull<c_void>);

    /// The registered GPU-visible guest memory inside `range`, clipped to it, in address order:
    /// one range per allocation (the pieces of one allocation merged).
    fn gpu_ranges(&mut self, range: GuestRange) -> Vec<GuestRange>;
    fn collect_writes(&mut self, _address: u64, _bytes: u64) -> Result<u64> { Ok(0) }
    fn unchanged_since(&mut self, _address: u64, _bytes: u64, _generation: u64) -> Result<bool> { Ok(false) }
    /// The pieces of `ranges` the CPU wrote after `since` (an earlier call's generation, or 0),
    /// collected once each, and the generation to pass next time; `None` when the callbacks do
    /// not offer it (then [`Self::unchanged_since`] per range).
    fn written_since(&mut self, _ranges: &[GuestRange], _since: u64) -> Result<Option<(u64, Vec<GuestRange>)>> { Ok(None) }
    /// Forgets the CPU writes to `range` made so far: what reads the range next already sees them.
    fn absorb_writes(&mut self, _range: GuestRange) {}
}

/// The device that owns the guest window, as residency uses it.
pub trait ResidencyDevice {
    /// The last completed serial of `queue`, after retiring what the queues finished. Never
    /// blocks.
    fn completed(&mut self, queue: Queue) -> u64;

    /// Waits until every queue is idle.
    fn wait_idle(&mut self) -> Result<()>;

    /// Whether the device is lost (quarantined). A lost device keeps everything the GPU may still
    /// use, imports included, until process exit.
    fn is_lost(&self) -> bool;

    /// Waits for the paging work the imports made so far have queued: a batch of imports waits
    /// once, after its last one.
    fn settle_paging(&mut self) -> Result<()> {
        Ok(())
    }

    /// This process's local segment group usage and budget.
    fn local_memory(&self) -> Result<LocalMemory>;
}

/// The guest window, reserved at the guest's own GPU addresses, and the imports in it.
///
/// Dropping it waits until no submission uses an import, then frees the window and destroys
/// every import in it; a lost device keeps them instead.
pub trait ResidencyWindow {
    type Device: ResidencyDevice;
    /// What a submission passes to its queue for an import (a [`crate::Buffer`] on WDDM).
    type Import;
    /// Imports held as one for a run of submissions ([`GuestResidency::shared_targets`]).
    type Targets;

    fn share_targets(&self, device: &Self::Device, targets: &[&Self::Import]) -> Result<Self::Targets>;

    fn base(&self) -> u64;

    fn end(&self) -> u64;
    fn sparse_zero(&mut self, _device: &mut Self::Device, _range: GuestRange) -> Result<()> { Err(Error::unsupported("this window cannot map sparse holes")) }

    /// Imports stable physical backing once, initially mapped at `guest`.
    ///
    /// # Safety
    ///
    /// `alias` stays mapped read-write until `release_backing` returns or the healthy window is
    /// dropped. Guest binding changes do not end this lifetime.
    unsafe fn import_backing(
        &mut self,
        device: &mut Self::Device,
        id: u64,
        guest: u64,
        bytes: u64,
        alias: NonNull<c_void>,
    ) -> Result<()>;

    fn bind_backing(&mut self, device: &mut Self::Device, guest: u64, bytes: u64, id: u64, offset: u64) -> Result<()>;
    fn release_bindings(&mut self, device: &mut Self::Device, guest: u64, bytes: u64) -> Result<()>;
    fn release_backing(&mut self, device: &mut Self::Device, id: u64) -> Result<()>;
    fn backing_at(&self, id: u64) -> Option<&Self::Import>;

    /// Allocates device memory for host-layout copies at `[address, address + bytes)` of the
    /// host-copy window: whole [`COPY_GRANULE_BYTES`] granules overlapping no other copy memory.
    /// A window without a host-copy window refuses.
    fn allocate_copy(
        &mut self,
        _device: &mut Self::Device,
        address: u64,
        bytes: u64,
    ) -> Result<()> {
        Err(Error::unsupported(format!(
            "this guest window has no host-copy window for 0x{address:x}+0x{bytes:x}"
        )))
    }

    /// Waits for every submission that uses the copy memory starting at `address`, then frees it.
    fn release_copy(&mut self, _device: &mut Self::Device, address: u64) -> Result<()> {
        Err(Error::invalid(format!(
            "no host copy starts at 0x{address:x}"
        )))
    }

    /// The copy memory that starts at `address`.
    fn copy_at(&self, _address: u64) -> Option<&Self::Import> {
        None
    }
}

/// Host-copy memory is allocated in whole granules of this size: the alignment of device
/// allocations at fixed GPU addresses.
pub const COPY_GRANULE_BYTES: u64 = 64 << 10;

/// Usage and budget of the local segment group for this process (D3DKMTQueryVideoMemoryInfo).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LocalMemory {
    pub usage: u64,
    pub budget: u64,
}

/// What `memory_info` reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryInfo {
    pub window_base: u64,
    pub window_end: u64,
    pub imported_bytes: u64,
    pub imports: u64,
    /// Host-layout copy memory, which counts against the import budget too.
    pub host_copy_bytes: u64,
    pub host_copies: u64,
    pub import_budget: u64,
    pub local: LocalMemory,
    /// AnyPS5's callbacks are installed ([`GuestResidency::set_memory`]).
    pub imports_enabled: bool,
}

/// What one submission can reach in guest memory.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reach<'a> {
    /// The guest ranges the submission names.
    pub ranges: &'a [GuestRange],
    /// A program computes addresses (flat or global accesses through pointers in memory), so it
    /// may reach every GPU-visible guest allocation in the window.
    pub computed_addresses: bool,
}

/// Where the guest window goes and how much imports may pin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyOptions {
    pub window_base: u64,
    pub window_end: u64,
    /// Bytes imports may pin; 0 means half of the local segment budget.
    pub import_budget: u64,
}

impl Default for ResidencyOptions {
    fn default() -> Self {
        Self {
            window_base: va::GUEST_WINDOW_BASE,
            window_end: va::GUEST_END,
            import_budget: 0,
        }
    }
}

/// The import budget for a requested one: `requested`, or half of the local budget for 0.
pub fn import_budget(requested: u64, local_budget: u64) -> u64 {
    if requested == 0 {
        local_budget / 2
    } else {
        requested
    }
}

/// The imports one submission uses, from [`GuestResidency::prepare`], in address order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[must_use = "pass the targets to the queue, then report the serial with GuestResidency::queued"]
pub struct Residents {
    starts: Arc<[u64]>,
    backings: Arc<[u64]>,
    imported: usize,
    mapping_generation: u64,
    binding_generation: u64,
    incarnations: Arc<[u64]>,
}

impl Residents {
    pub fn mapping_generation(&self) -> u64 { self.mapping_generation }
    /// The guest addresses where the imports start.
    pub fn starts(&self) -> &[u64] {
        &self.starts
    }

    /// How many of them [`GuestResidency::prepare`] made.
    pub fn imported(&self) -> usize {
        self.imported
    }

    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
}

/// One import: the alias it was made from, which only `unmap_import` frees, and its last use.
struct Import {
    bytes: u64,
    alias: NonNull<c_void>,
    backings: Vec<u64>,
    incarnation: u64,
    /// The least-recently-used clock when a submission last used it.
    last_use: u64,
    /// The last submission on each queue ([`Queue::index`]) that used it.
    last: BTreeMap<Queue, u64>,
}

struct PhysicalBacking {
    alias: NonNull<c_void>,
    pages: Vec<u64>,
    last_use: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BindingStats {
    pub created: u64,
    pub imported_bytes: u64,
    pub reused_bytes: u64,
    pub unmapped_bytes: u64,
    pub evicted_bytes: u64,
}

impl PhysicalBacking {
    fn bytes(&self) -> u64 { self.pages.len() as u64 * GUEST_PAGE_BYTES }
}

/// One host-copy allocation and the guest surfaces whose copies it holds.
struct HostCopy {
    bytes: u64,
    surfaces: Vec<GuestRange>,
}

/// The set [`GuestResidency::shared_targets`] last built, and the backings it holds.
struct SharedTargets<T> {
    backings: Arc<[u64]>,
    targets: T,
}

/// The submissions queued since the imports' own records were last brought up to date
/// ([`GuestResidency::settle_uses`]): every one of them used `starts`.
struct RecentUse {
    starts: Arc<[u64]>,
    /// The least-recently-used clock at the latest of them.
    tick: u64,
    /// The latest of them on each queue.
    last: BTreeMap<Queue, u64>,
}

/// Where a piece of a submission's reach came from, for the refusal message.
#[derive(Clone, Copy)]
enum Origin {
    Named(GuestRange),
    Computed,
}

/// The guest pages imported into one device's guest window (§9.1).
///
/// Dropping it without [`Self::close`] destroys the imports but never unmaps their aliases: they
/// stay mapped until process exit.
pub struct GuestResidency<W: ResidencyWindow> {
    window: W,
    memory: Option<Box<dyn GuestMemory>>,
    imports: BTreeMap<u64, Import>,
    imported_bytes: u64,
    backings: BTreeMap<u64, PhysicalBacking>,
    physical_pages: HashMap<u64, (u64, u64)>,
    backing_bytes: u64,
    next_backing: u64,
    binding_generation: u64,
    stats: BindingStats,
    /// Host-layout copy memory by host address.
    copies: BTreeMap<u64, HostCopy>,
    copy_bytes: u64,
    /// The imports the last [`Self::prepare`] returned whose submission is not queued yet: room
    /// for a host copy is never made by releasing them.
    prepared: Option<Arc<[u64]>>,
    budget: u64,
    /// The least-recently-used clock: every queued submission advances it.
    tick: u64,
    sparse: sparse::SparseRegistry,
    /// Empty whenever a backing or host copy is about to go: its release needs it held alone.
    shared: Option<SharedTargets<W::Targets>>,
    /// A run of submissions costs one record however much it uses; `None` once settled.
    recent: Option<RecentUse>,
}

impl<W: ResidencyWindow> GuestResidency<W> {
    /// Residency over a reserved guest window, with no callbacks yet ([`Self::set_memory`]).
    pub fn new(window: W, budget: u64) -> Self {
        Self {
            window,
            memory: None,
            imports: BTreeMap::new(),
            imported_bytes: 0,
            backings: BTreeMap::new(),
            physical_pages: HashMap::new(),
            backing_bytes: 0,
            next_backing: 1,
            binding_generation: 0,
            stats: BindingStats::default(),
            copies: BTreeMap::new(),
            copy_bytes: 0,
            prepared: None,
            budget,
            tick: 0,
            sparse: sparse::SparseRegistry::default(),
            shared: None,
            recent: None,
        }
    }

    pub fn window(&self) -> &W {
        &self.window
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    pub fn imported_bytes(&self) -> u64 {
        self.backing_bytes
    }

    pub fn import_count(&self) -> usize {
        self.backings.len()
    }

    pub fn take_binding_stats(&mut self) -> BindingStats { std::mem::take(&mut self.stats) }
    pub fn sparse_ranges(&self) -> Vec<GuestRange> { self.sparse.ranges() }

    pub fn host_copy_bytes(&self) -> u64 {
        self.copy_bytes
    }

    pub fn host_copy_count(&self) -> usize {
        self.copies.len()
    }

    pub fn imports_enabled(&self) -> bool {
        self.memory.is_some()
    }

    pub fn collect_writes(&mut self, address: u64, bytes: u64) -> Result<u64> {
        match self.memory.as_mut() { Some(memory) => memory.collect_writes(address, bytes), None => Ok(0) }
    }

    pub fn written_since(&mut self, ranges: &[GuestRange], since: u64) -> Result<Option<(u64, Vec<GuestRange>)>> {
        match self.memory.as_mut() { Some(memory) => memory.written_since(ranges, since), None => Ok(None) }
    }

    /// See [`GuestMemory::absorb_writes`].
    pub fn absorb_writes(&mut self, range: GuestRange) {
        if let Some(memory) = self.memory.as_mut() {
            memory.absorb_writes(range);
        }
    }

    pub fn unchanged_since(&mut self, address: u64, bytes: u64, generation: u64) -> Result<bool> {
        if generation == 0 { return Ok(false); }
        match self.memory.as_mut() { Some(memory) => memory.unchanged_since(address, bytes, generation), None => Ok(false) }
    }

    /// Installs AnyPS5's callbacks (`set_guest_memory`), or removes them with `None`. Refused
    /// while imports live: their aliases belong to the callbacks that made them, so release them
    /// first ([`Self::trim`] to 0).
    pub fn set_memory(&mut self, memory: Option<Box<dyn GuestMemory>>) -> Result<()> {
        if !self.imports.is_empty() || !self.backings.is_empty() {
            return Err(Error::invalid(format!(
                "{} guest imports are live; release them (trim_imports to 0) before changing the guest-memory callbacks",
                self.imports.len()
            )));
        }
        self.memory = memory;
        Ok(())
    }

    /// Makes everything `reach` names resident for one submission and returns the imports it
    /// uses (§9.3):
    ///
    /// 1. Each range is cut at 8 MiB import windows and rounded out to whole guest pages. A range
    ///    outside the guest window fails the submission (§7).
    /// 2. Each part no import covers grows to its GPU-visible allocation inside the import window,
    ///    up to the neighbouring imports, so later uses of the allocation hit the same import. When
    ///    that would not fit the budget beside this submission's other imports, only the part is
    ///    imported.
    /// 3. Room is made under the budget, which host copies share, by releasing the
    ///    least-recently-used imports this submission does not use, retired ones first. AnyPS5
    ///    then maps an alias and the window imports it.
    ///
    /// A refusal or a failed import fails the submission; the imports made before it stay.
    pub fn prepare(&mut self, device: &mut W::Device, reach: Reach<'_>) -> Result<Residents> {
        self.prepared = None;
        if device.is_lost() {
            return Err(device_lost("no submission can be prepared"));
        }
        let mapping_generation = self.memory()?.mapping_generation();
        let mut pieces = Vec::new();
        for &range in reach.ranges {
            if range.bytes != 0 {
                self.check_inside(range)?;
                pieces.extend(window_pieces(range).map(|piece| (piece, Origin::Named(range))));
            }
        }
        if reach.computed_addresses {
            let window =
                GuestRange::new(self.window.base(), self.window.end() - self.window.base());
            for allocation in self.memory()?.gpu_ranges(window) {
                if let Some(inside) = allocation.intersection(window) {
                    pieces.extend(window_pieces(inside).map(|piece| (piece, Origin::Computed)));
                }
            }
        }
        let mut used = BTreeSet::new();
        for &(piece, _) in &pieces {
            used.extend(self.overlapping(piece));
        }
        let mut visible: BTreeMap<u64, Vec<GuestRange>> = BTreeMap::new();
        let mut imported = 0;
        for &(piece, origin) in &pieces {
            for gap in self.gaps(piece).into_iter().flat_map(|gap| self.sparse.unregistered(gap)).collect::<Vec<_>>() {
                let window = window_of(gap.address);
                let around = match visible.entry(window) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(
                        self.memory()?
                            .gpu_ranges(GuestRange::new(window, IMPORT_WINDOW_BYTES)),
                    ),
                };
                let mut import = self.extend(gap, around);
                import = self.sparse.unregistered(import).into_iter().find(|range| range.address <= gap.address && range.end() >= gap.end()).ok_or_else(|| Error::invalid("ordinary import would cover a sparse hole"))?;
                let held = used
                    .iter()
                    .map(|start| self.imports[start].bytes)
                    .sum::<u64>()
                    + self.copy_bytes;
                if held + import.bytes > self.budget && held + gap.bytes <= self.budget {
                    import = gap;
                }
                let start = self.make_import(device, import, origin, &used)?;
                used.insert(start);
                imported += 1;
            }
        }
        device.settle_paging()?;
        if self.memory()?.mapping_generation() != mapping_generation {
            return Err(Error::new(ErrorKind::Busy, "guest mappings changed during residency preparation"));
        }
        let residents = Residents {
            backings: used.iter().flat_map(|start| self.imports[start].backings.iter().copied()).collect::<BTreeSet<_>>().into_iter().collect(),
            incarnations: used.iter().map(|start| self.imports[start].incarnation).collect(),
            starts: used.into_iter().collect(),
            imported,
            mapping_generation,
            binding_generation: self.binding_generation,
        };
        self.prepared = Some(residents.starts.clone());
        Ok(residents)
    }

    /// The imports to pass to the queue for `residents`, in address order. Fails if one was
    /// released since [`Self::prepare`].
    pub fn valid(&self, residents: &Residents) -> bool { residents.binding_generation == self.binding_generation }

    fn bindings_match(&self, residents: &Residents) -> bool {
        self.valid(residents) || residents.starts.iter().zip(residents.incarnations.iter()).all(|(start, incarnation)| self.imports.get(start).is_some_and(|import| import.incarnation == *incarnation))
    }

    pub fn protect(&mut self, residents: &Residents) -> Result<()> {
        if !self.bindings_match(residents) { return Err(Error::new(ErrorKind::Busy, "resident bindings changed before protection")); }
        self.prepared = Some(residents.starts.clone());
        Ok(())
    }

    pub fn targets(&self, residents: &Residents) -> Result<Vec<&W::Import>> {
        if !self.bindings_match(residents) {
            return Err(Error::new(ErrorKind::Busy, "resident bindings changed after preparation"));
        }
        residents.backings.iter().map(|&id| self.window.backing_at(id).ok_or_else(|| Error::invalid("a resident physical backing is missing"))).collect()
    }

    /// What the submissions of `residents` pass to their queue as one: its backings, every host
    /// copy and `others`. Built once, and again only when a copy comes or goes or `residents`
    /// holds other backings: a segment then costs the same however much memory is resident.
    /// Every host copy is in it, not just the ones a segment uses, as computed addresses can
    /// reach any of them through descriptors in memory.
    pub fn shared_targets(&mut self, device: &W::Device, residents: &Residents, others: &[W::Import]) -> Result<&W::Targets> {
        if !self.bindings_match(residents) {
            return Err(Error::new(ErrorKind::Busy, "resident bindings changed after preparation"));
        }
        let current = self.shared.as_ref().is_some_and(|shared| Arc::ptr_eq(&shared.backings, &residents.backings) || shared.backings == residents.backings);
        if !current {
            let mut targets = self.targets(residents)?;
            for &start in self.copies.keys() {
                targets.push(self.window.copy_at(start).ok_or_else(|| Error::invalid("a host copy's memory is missing"))?);
            }
            targets.extend(others);
            let built = self.window.share_targets(device, &targets)?;
            self.shared = Some(SharedTargets { backings: residents.backings.clone(), targets: built });
        }
        Ok(&self.shared.as_ref().expect("built above").targets)
    }

    /// Records that `submission` was queued with the targets of `residents` (§9.3 step 6). Only
    /// the first of a run with the same imports walks them; the rest update [`RecentUse`].
    pub fn queued(&mut self, residents: &Residents, submission: Submission) {
        self.prepared = None;
        if !self.recent.as_ref().is_some_and(|recent| Arc::ptr_eq(&recent.starts, &residents.starts) || recent.starts == residents.starts) {
            self.settle_uses();
            self.recent = Some(RecentUse { starts: residents.starts.clone(), tick: 0, last: BTreeMap::new() });
        }
        let recent = self.recent.as_mut().expect("set above");
        recent.tick = self.tick;
        recent.last.insert(submission.queue, submission.serial);
        self.tick += 1;
    }

    /// Brings the imports' and backings' last uses up to date. Called before they are read or an
    /// import goes.
    fn settle_uses(&mut self) {
        let Some(recent) = self.recent.take() else { return };
        for start in recent.starts.iter() {
            if let Some(import) = self.imports.get_mut(start) {
                import.last_use = recent.tick;
                import.last.extend(recent.last.iter().map(|(&queue, &serial)| (queue, serial)));
                for id in &import.backings {
                    self.backings.get_mut(id).expect("live binding backing").last_use = recent.tick;
                }
            }
        }
    }

    /// The host-layout copy memory for the guest surface `[guest_base, guest_base + bytes)`: at
    /// guest VA + 2^40 (`FINAL-REPORT.md` §2-D, [`va::HOST_COPY_OFFSET`]), in whole
    /// [`COPY_GRANULE_BYTES`] granules, allocated on first use. Pass the allocations
    /// [`Self::host_copies`] returns to the queue as targets.
    ///
    /// Copy memory is a set of allocations that never overlap; a surface's copy may span several.
    /// Surfaces whose copies share a granule (neighbours, and aliases of the same memory) share
    /// its memory: only the granules no allocation covers yet are allocated, one allocation per
    /// contiguous gap, so every surface keeps its copy at its fixed address and none moves.
    ///
    /// Copy memory counts against the import budget. Room is made as [`Self::prepare`] makes it,
    /// except from the imports the last `prepare` returned for a submission not yet queued, and
    /// never from copies: a copy may hold the only current contents of its surfaces, so neither
    /// the budget nor [`Self::trim`] releases one. When a gap cannot be allocated, the gaps this
    /// call allocated are freed again.
    ///
    /// An allocation is released once the guest memory of every surface it holds goes
    /// ([`Self::release_range`]), by [`Self::release_host_copy`], or at [`Self::close`].
    pub fn ensure_host_copy(
        &mut self,
        device: &mut W::Device,
        guest_base: u64,
        bytes: u64,
    ) -> Result<()> {
        if device.is_lost() {
            return Err(device_lost("no host copy can be made"));
        }
        let surface = GuestRange::new(guest_base, bytes);
        if bytes == 0 {
            return Err(Error::invalid(format!(
                "the host copy of guest surface {surface} is empty"
            )));
        }
        self.check_inside(surface)?;
        let span = copy_span(surface);
        let mut allocated = Vec::new();
        for gap in self.copy_gaps(span) {
            let keep = BTreeSet::new();
            let made = self.make_room(device, gap.bytes, &keep).and_then(|()| {
                self.window
                    .allocate_copy(device, gap.address, gap.bytes)
                    .map_err(|error| {
                        Error::new(
                            error.kind,
                            format!(
                                "allocating the host copy {gap} of guest surface {surface} failed: {error}"
                            ),
                        )
                    })
            });
            if let Err(error) = made {
                for start in allocated {
                    self.release_copy(device, start)?;
                }
                return Err(error);
            }
            self.copies.insert(
                gap.address,
                HostCopy {
                    bytes: gap.bytes,
                    surfaces: Vec::new(),
                },
            );
            self.shared = None;
            self.copy_bytes += gap.bytes;
            allocated.push(gap.address);
        }
        for start in self.copies_overlapping(span) {
            if let Some(copy) = self.copies.get_mut(&start) {
                if !copy.surfaces.contains(&surface) {
                    copy.surfaces.push(surface);
                }
            }
        }
        Ok(())
    }

    /// The host-copy allocations that together hold the copy of `[guest_base, guest_base +
    /// bytes)`, by address, if every granule of it is allocated.
    pub fn host_copies(&self, guest_base: u64, bytes: u64) -> Option<Vec<&W::Import>> {
        let span = copy_span(GuestRange::new(guest_base, bytes.max(1)));
        if !self.copy_gaps(span).is_empty() {
            return None;
        }
        self.copies_overlapping(span)
            .into_iter()
            .map(|start| self.window.copy_at(start))
            .collect()
    }

    /// The physical backings the imports of `[guest_base, guest_base + bytes)` map, when imports
    /// cover all of it: what GPU work that reads the guest memory there holds.
    pub fn guest_backings(&self, guest_base: u64, bytes: u64) -> Option<Vec<&W::Import>> {
        let range = GuestRange::new(guest_base, bytes.max(1));
        if !self.gaps(range).is_empty() {
            return None;
        }
        let mut ids: Vec<u64> = self.overlapping(range).into_iter().flat_map(|start| self.imports[&start].backings.iter().copied()).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter().map(|id| self.window.backing_at(id)).collect()
    }

    /// Frees the host-copy memory of the surfaces whose copies hold guest address `guest_base`:
    /// the allocation at that address and every allocation that holds part of one of its
    /// surfaces, each with all its surfaces, once no submission uses it. Returns the bytes.
    pub fn release_host_copy(&mut self, device: &mut W::Device, guest_base: u64) -> Result<u64> {
        let [start] = *self
            .copies_overlapping(copy_span(GuestRange::new(guest_base, 1)))
            .as_slice()
        else {
            return Err(Error::invalid(format!(
                "no host copy holds guest address 0x{guest_base:x}"
            )));
        };
        let surfaces = self.copies[&start].surfaces.clone();
        let starts: Vec<u64> = self
            .copies
            .iter()
            .filter(|&(&other, copy)| {
                other == start || copy.surfaces.iter().any(|s| surfaces.contains(s))
            })
            .map(|(&other, _)| other)
            .collect();
        let mut released = 0;
        for start in starts {
            released += self.release_copy(device, start)?;
        }
        Ok(released)
    }

    /// Releases what AnyPS5 is changing the guest mapping of at `range` (`release_guest_range`,
    /// §9.5): every import that overlaps it, and every host copy all of whose surfaces it
    /// overlaps. Each waits for the submissions that use it and is destroyed, then an import's
    /// alias is unmapped. Returns how many it released. On a lost device it releases nothing and
    /// fails with `DeviceLost`: the imports and their aliases stay until process exit.
    pub fn release_range(&mut self, device: &mut W::Device, range: GuestRange) -> Result<u64> {
        let starts = self.overlapping(range);
        for &start in &starts {
            self.release_import(device, start)?;
        }
        for sparse in self.sparse.overlaps(range) { self.window.sparse_zero(device, sparse)?; }
        let mut emptied = Vec::new();
        for (&start, copy) in &mut self.copies {
            let held = copy.surfaces.len();
            // A surface only part of whose memory goes (a streamed mip) keeps its copy.
            copy.surfaces
                .retain(|surface| !(range.address <= surface.address && surface.end() <= range.end()));
            if held != 0 && copy.surfaces.is_empty() {
                emptied.push(start);
            }
        }
        for &start in &emptied {
            self.release_copy(device, start)?;
        }
        Ok((starts.len() + emptied.len()) as u64)
    }

    /// Releases least-recently-used imports, retired ones first, until at most `target_bytes`
    /// are imported (`trim_imports`, §9.6), and returns the bytes released.
    pub fn trim(&mut self, device: &mut W::Device, target_bytes: u64) -> Result<u64> {
        self.settle_uses();
        let mut released = 0;
        while self.backing_bytes > target_bytes {
            if let Some(bytes) = self.evict_backing(device)? {
                released += bytes;
                continue;
            }
            let Some(start) = self.least_recently_used(device, &BTreeSet::new()) else {
                break;
            };
            self.release_import(device, start)?;
        }
        Ok(released)
    }

    pub fn memory_info(&self, device: &W::Device) -> Result<MemoryInfo> {
        Ok(MemoryInfo {
            window_base: self.window.base(),
            window_end: self.window.end(),
            imported_bytes: self.backing_bytes,
            imports: self.backings.len() as u64,
            host_copy_bytes: self.copy_bytes,
            host_copies: self.copies.len() as u64,
            import_budget: self.budget,
            local: device.local_memory()?,
            imports_enabled: self.memory.is_some(),
        })
    }

    /// Teardown (`close`, §9.7): waits until the device is idle, drops the window, which frees it
    /// and destroys every import and host copy, then unmaps every alias. Returns how many it
    /// unmapped. On a lost device the aliases stay mapped and it fails with `DeviceLost`.
    pub fn close(self, device: &mut W::Device) -> Result<u64> {
        let idle = device.wait_idle();
        let Self {
            window,
            memory,
            imports,
            backings,
            shared,
            ..
        } = self;
        drop(shared);
        drop(window);
        let lost = match idle {
            Err(error) => Some(error.to_string()),
            Ok(()) if device.is_lost() => Some("the device is lost".to_string()),
            Ok(()) => None,
        };
        if let Some(reason) = lost {
            return Err(Error::new(
                ErrorKind::DeviceLost,
                format!(
                    "closing guest residency: {reason}; {} import aliases stay mapped until process exit",
                    imports.len()
                ),
            ));
        }
        let mut unmapped = 0;
        if let Some(mut memory) = memory {
            for import in imports.into_values() {
                memory.unmap_import(import.alias);
                unmapped += 1;
            }
            for backing in backings.into_values() {
                memory.unmap_backing(backing.alias);
                unmapped += 1;
            }
        }
        Ok(unmapped)
    }

    fn memory(&mut self) -> Result<&mut dyn GuestMemory> {
        match self.memory.as_deref_mut() {
            Some(memory) => Ok(memory),
            None => Err(Error::unsupported(
                "same-VA guest memory needs AnyPS5's guest-memory callbacks (set_guest_memory)",
            )),
        }
    }

    fn check_inside(&self, range: GuestRange) -> Result<()> {
        let Some(end) = range.address.checked_add(range.bytes) else {
            return Err(Error::invalid(format!(
                "guest range 0x{:x}+0x{:x} overflows",
                range.address, range.bytes
            )));
        };
        if range.address < self.window.base() || end > self.window.end() {
            return Err(Error::invalid(format!(
                "guest range {range} is outside the guest window 0x{:x}..0x{:x}: the GPU reaches guest memory at its own address only inside the window, and staged copies of memory outside it do not exist yet",
                self.window.base(),
                self.window.end()
            )));
        }
        Ok(())
    }

    /// The parts of the host range `span` no host-copy allocation covers, by address.
    fn copy_gaps(&self, span: GuestRange) -> Vec<GuestRange> {
        let mut gaps = Vec::new();
        let mut cursor = span.address;
        for start in self.copies_overlapping(span) {
            if start > cursor {
                gaps.push(GuestRange::new(cursor, start - cursor));
            }
            cursor = cursor.max(start + self.copies[&start].bytes);
        }
        if cursor < span.end() {
            gaps.push(GuestRange::new(cursor, span.end() - cursor));
        }
        gaps
    }

    /// The host copies that overlap the host range `span`, by start.
    fn copies_overlapping(&self, span: GuestRange) -> Vec<u64> {
        let first = self
            .copies
            .range(..=span.address)
            .next_back()
            .map_or(span.address, |(&start, _)| start);
        self.copies
            .range(first..span.end())
            .filter(|&(&start, copy)| start + copy.bytes > span.address)
            .map(|(&start, _)| start)
            .collect()
    }

    /// Frees the host copy at `start` once no submission uses it. Returns its size.
    fn release_copy(&mut self, device: &mut W::Device, start: u64) -> Result<u64> {
        if device.is_lost() {
            return Err(device_lost(&format!(
                "the host copy at 0x{start:x} stays until process exit"
            )));
        }
        self.shared = None;
        if let Err(error) = self.window.release_copy(device, start) {
            return Err(if device.is_lost() {
                device_lost(&format!(
                    "releasing the host copy at 0x{start:x}: {error}; it stays until process exit"
                ))
            } else {
                error
            });
        }
        let Some(copy) = self.copies.remove(&start) else {
            return Ok(0);
        };
        self.copy_bytes -= copy.bytes;
        Ok(copy.bytes)
    }

    /// What the residency knows of guest `address`, for a GPU fault report: the import that
    /// covers it, whether a registered PRT range holds it, and whether guest memory reports it
    /// GPU-visible.
    pub fn describe_address(&mut self, address: u64) -> String {
        let page = GuestRange::new(address & !0xfff, 0x1000);
        let mut parts = Vec::new();
        if !(self.window.base()..self.window.end()).contains(&address) {
            parts.push("outside the guest window".to_string());
        }
        match self.overlapping(page).first() {
            Some(&start) => parts.push(format!(
                "imported in 0x{start:x}+0x{:x}",
                self.imports[&start].bytes
            )),
            None => parts.push("not imported".to_string()),
        }
        if !self.sparse.overlaps(page).is_empty() {
            parts.push("inside a registered PRT range".to_string());
        }
        let around = GuestRange::new(address & !0xffff, 0x10000);
        match self.memory() {
            Ok(memory) => {
                let visible = memory.gpu_ranges(around);
                if visible.iter().any(|range| range.address <= address && address < range.end()) {
                    parts.push("GPU-visible guest memory".to_string());
                } else {
                    let near: Vec<String> = visible.iter().map(|range| range.to_string()).collect();
                    parts.push(format!(
                        "not GPU-visible guest memory (GPU-visible in its 64 KiB: [{}])",
                        near.join(", ")
                    ));
                }
            }
            Err(_) => parts.push("no guest memory installed".to_string()),
        }
        parts.join(", ")
    }

    /// The imports that overlap `range`, by start.
    fn overlapping(&self, range: GuestRange) -> Vec<u64> {
        let first = self
            .imports
            .range(..=range.address)
            .next_back()
            .map_or(range.address, |(&start, _)| start);
        self.imports
            .range(first..range.end())
            .filter(|&(&start, import)| start + import.bytes > range.address)
            .map(|(&start, _)| start)
            .collect()
    }

    /// The parts of `piece` no import covers.
    fn gaps(&self, piece: GuestRange) -> Vec<GuestRange> {
        let mut gaps = Vec::new();
        let mut cursor = piece.address;
        for start in self.overlapping(piece) {
            if start > cursor {
                gaps.push(GuestRange::new(cursor, start - cursor));
            }
            cursor = cursor.max(start + self.imports[&start].bytes);
        }
        if cursor < piece.end() {
            gaps.push(GuestRange::new(cursor, piece.end() - cursor));
        }
        gaps
    }

    /// `gap` grown to the GPU-visible allocation that holds it, inside its import window and up to
    /// the neighbouring imports. A gap no single allocation holds stays as it is, for
    /// `map_import` to judge.
    fn extend(&self, gap: GuestRange, visible: &[GuestRange]) -> GuestRange {
        let Some(allocation) = visible
            .iter()
            .find(|allocation| allocation.address <= gap.address && gap.end() <= allocation.end())
        else {
            return gap;
        };
        let window = window_of(gap.address);
        let below = self
            .imports
            .range(..gap.address)
            .next_back()
            .map_or(0, |(&start, import)| start + import.bytes);
        let above = self
            .imports
            .range(gap.end()..)
            .next()
            .map_or(u64::MAX, |(&start, _)| start);
        let start = page_ceil(allocation.address).max(window).max(below);
        let end = page_floor(allocation.end())
            .min(window + IMPORT_WINDOW_BYTES)
            .min(above);
        GuestRange::new(start, end - start)
    }

    /// Releases least-recently-used imports outside `keep` until `bytes` more fit the budget
    /// beside the imports and host copies.
    fn make_room(
        &mut self,
        device: &mut W::Device,
        bytes: u64,
        keep: &BTreeSet<u64>,
    ) -> Result<()> {
        self.settle_uses();
        while self.backing_bytes + self.copy_bytes + bytes > self.budget {
            if self.evict_backing(device)?.is_some() { continue; }
            let Some(start) = self.least_recently_used(device, keep) else {
                return Err(Error::new(
                    ErrorKind::System,
                    format!(
                        "import budget exhausted: 0x{bytes:x} more bytes do not fit the budget of 0x{:x} beside 0x{:x} bytes of imports the submission uses and 0x{:x} bytes of host copies",
                        self.budget, self.backing_bytes, self.copy_bytes
                    ),
                ));
            };
            self.release_import(device, start)?;
        }
        Ok(())
    }

    /// The import to release first, outside `keep`: retired before pending, then least recently
    /// used.
    fn least_recently_used(&self, device: &mut W::Device, keep: &BTreeSet<u64>) -> Option<u64> {
        let completed: BTreeMap<_, _> = self.imports.values().flat_map(|import| import.last.keys().copied()).map(|queue| (queue, device.completed(queue))).collect();
        self.imports
            .iter()
            .filter(|&(start, _)| !keep.contains(start) && !self.prepared.as_ref().is_some_and(|starts| starts.binary_search(start).is_ok()) && !self.sparse.contains(GuestRange::new(*start, self.imports[start].bytes)))
            .min_by_key(|&(&start, import)| {
                let pending = import
                    .last
                    .iter()
                    .any(|(queue, serial)| *serial > completed[queue]);
                (pending, import.last_use, start)
            })
            .map(|(&start, _)| start)
    }

    fn evict_backing(&mut self, device: &mut W::Device) -> Result<Option<u64>> {
        let bound: BTreeSet<_> = self.imports.values().flat_map(|import| import.backings.iter().copied()).collect();
        let candidate = self.backings.iter().filter(|(id, _)| !bound.contains(id)).min_by_key(|(_, backing)| backing.last_use).map(|(&id, _)| id);
        let Some(id) = candidate else { return Ok(None) };
        self.shared = None;
        self.window.release_backing(device, id)?;
        let backing = self.backings.remove(&id).expect("selected physical backing");
        let bytes = backing.bytes();
        for page in &backing.pages {
            if self.physical_pages.get(page).is_some_and(|(owner, _)| *owner == id) {
                self.physical_pages.remove(page);
            }
        }
        self.backing_bytes -= bytes;
        self.stats.evicted_bytes += bytes;
        self.memory()?.unmap_backing(backing.alias);
        Ok(Some(bytes))
    }

    /// Registers the guest binding, reuses cached physical pages, and imports only uncached runs.
    fn make_import(
        &mut self,
        device: &mut W::Device,
        import: GuestRange,
        origin: Origin,
        keep: &BTreeSet<u64>,
    ) -> Result<u64> {
        let alias = self.memory()?.map_import(import.address, import.bytes).map_err(|why| {
            if why == ImportRefusal::NotGpuVisible && matches!(origin, Origin::Computed) {
                Error::new(ErrorKind::Busy, format!("guest mapping changed while preparing {import}"))
            } else {
                refused(import, origin, why)
            }
        })?;
        let pages = match self.memory()?.import_pages(alias, import.bytes) {
            Ok(pages) => pages,
            Err(error) => { self.memory()?.unmap_import(alias); return Err(error); }
        };
        let cached: BTreeSet<_> = pages.iter().filter_map(|page| self.physical_pages.get(page).map(|&(id, _)| id)).collect();
        let needed = pages.iter().filter(|page| !self.physical_pages.contains_key(page)).copied().collect::<BTreeSet<_>>().len() as u64 * GUEST_PAGE_BYTES;
        self.imports.insert(
            import.address,
            Import {
                bytes: import.bytes,
                alias,
                backings: cached.into_iter().collect(),
                incarnation: self.binding_generation + 1,
                last_use: self.tick,
                last: BTreeMap::new(),
            },
        );
        self.imported_bytes += import.bytes;
        self.binding_generation += 1;
        let mut keep = keep.clone();
        keep.insert(import.address);
        let made = (|| -> Result<()> {
            self.make_room(device, needed, &keep)?;
            let mut first = 0;
            while first < pages.len() {
                let guest = import.address + first as u64 * GUEST_PAGE_BYTES;
                let (id, offset, stop) = match self.physical_pages.get(&pages[first]).copied() {
                    Some((id, offset)) => {
                        let mut stop = first + 1;
                        while stop < pages.len() && self.physical_pages.get(&pages[stop]) == Some(&(id, offset + (stop - first) as u64 * GUEST_PAGE_BYTES)) { stop += 1; }
                        self.stats.reused_bytes += (stop - first) as u64 * GUEST_PAGE_BYTES;
                        (id, offset, stop)
                    }
                    None => {
                        let mut stop = first + 1;
                        let mut distinct = BTreeSet::from([pages[first]]);
                        while stop < pages.len() && !self.physical_pages.contains_key(&pages[stop]) && distinct.insert(pages[stop]) { stop += 1; }
                        let bytes = (stop - first) as u64 * GUEST_PAGE_BYTES;
                        let id = self.next_backing;
                        self.next_backing = id.checked_add(1).ok_or_else(|| Error::invalid("physical backing identity exhausted"))?;
                        let backing_alias = self.memory()?.map_backing(guest, bytes)?;
                        // SAFETY: this independent CPU alias stays pinned across guest binding changes.
                        if let Err(error) = unsafe { self.window.import_backing(device, id, guest, bytes, backing_alias) } {
                            if !device.is_lost() { self.memory()?.unmap_backing(backing_alias); }
                            return Err(error);
                        }
                        let owned = pages[first..stop].to_vec();
                        for (index, &page) in owned.iter().enumerate() {
                            self.physical_pages.entry(page).or_insert((id, index as u64 * GUEST_PAGE_BYTES));
                        }
                        self.backings.insert(id, PhysicalBacking { alias: backing_alias, pages: owned, last_use: self.tick });
                        self.backing_bytes += bytes;
                        self.stats.created += 1;
                        self.stats.imported_bytes += bytes;
                        self.imports.get_mut(&import.address).expect("binding under construction").backings.push(id);
                        (id, 0, stop)
                    }
                };
                self.window.bind_backing(device, guest, (stop - first) as u64 * GUEST_PAGE_BYTES, id, offset)?;
                first = stop;
            }
            Ok(())
        })();
        if let Err(error) = made {
            if !device.is_lost() { self.release_import(device, import.address)?; }
            return Err(error);
        }
        Ok(import.address)
    }

    /// §9.5 for one import: the window waits for its submissions and destroys it, then AnyPS5
    /// unmaps its alias. Returns its size.
    fn release_import(&mut self, device: &mut W::Device, start: u64) -> Result<u64> {
        if device.is_lost() {
            return Err(device_lost(&format!(
                "the guest import at 0x{start:x} and its alias stay until process exit"
            )));
        }
        let Some(bytes) = self.imports.get(&start).map(|import| import.bytes) else { return Ok(0) };
        self.settle_uses();
        if let Err(error) = self.window.release_bindings(device, start, bytes) {
            return Err(if device.is_lost() {
                device_lost(&format!(
                    "releasing the guest import at 0x{start:x}: {error}; it and its alias stay until process exit"
                ))
            } else {
                error
            });
        }
        if self.prepared.as_ref().is_some_and(|starts| starts.binary_search(&start).is_ok()) {
            self.prepared = None;
        }
        let Some(import) = self.imports.remove(&start) else {
            return Ok(0);
        };
        self.binding_generation += 1;
        self.imported_bytes -= import.bytes;
        self.stats.unmapped_bytes += import.bytes;
        if let Some(memory) = self.memory.as_deref_mut() {
            memory.unmap_import(import.alias);
        }
        Ok(import.bytes)
    }
}

fn device_lost(what: &str) -> Error {
    Error::new(ErrorKind::DeviceLost, format!("the device is lost: {what}"))
}

fn refused(import: GuestRange, origin: Origin, why: ImportRefusal) -> Error {
    let reach = match origin {
        Origin::Named(range) => format!("the submission reaches guest range {range}"),
        Origin::Computed => {
            "a program computes addresses, so every GPU-visible guest allocation in the window must be imported".to_string()
        }
    };
    Error::invalid(format!(
        "guest memory {import} cannot be imported for the GPU ({reach}): {} (map_import refusal {})",
        why.reason(),
        why.code()
    ))
}

fn window_of(address: u64) -> u64 {
    address & !(IMPORT_WINDOW_BYTES - 1)
}

/// The host range of `surface`'s copy at guest VA + 2^40, rounded out to whole granules.
fn copy_span(surface: GuestRange) -> GuestRange {
    let start = surface.address.saturating_add(HOST_COPY_OFFSET);
    let end = start.saturating_add(surface.bytes);
    let first = start & !(COPY_GRANULE_BYTES - 1);
    let last = end
        .checked_next_multiple_of(COPY_GRANULE_BYTES)
        .unwrap_or(!(COPY_GRANULE_BYTES - 1));
    GuestRange::new(first, last - first)
}

fn page_floor(address: u64) -> u64 {
    address & !(GUEST_PAGE_BYTES - 1)
}

fn page_ceil(address: u64) -> u64 {
    address.next_multiple_of(GUEST_PAGE_BYTES)
}

/// `range` cut at 8 MiB import windows, each piece rounded out to whole guest pages.
fn window_pieces(range: GuestRange) -> impl Iterator<Item = GuestRange> {
    let end = range.end();
    let mut cursor = range.address;
    std::iter::from_fn(move || {
        if cursor >= end {
            return None;
        }
        let piece_end = end.min(window_of(cursor) + IMPORT_WINDOW_BYTES);
        let start = page_floor(cursor);
        cursor = piece_end;
        Some(GuestRange::new(start, page_ceil(piece_end) - start))
    })
}
