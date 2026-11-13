//! Guest memory for the driver's device (PLAN P3.4 step 3c, decisions D15, D21): the imports the
//! residency maps through libc, and the CPU writes it asks about, which [`WriteTracker`] collects
//! from libc's write watch (GetWriteWatch over the guest arena) and stamps. Both belong to the
//! thread that owns the device, so neither takes a lock.

use super::anyps5::GuestArena;
use super::interface::{self, import};
use crate::residency::{GuestMemory, GuestRange, ImportRefusal};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The write watch's page.
const PAGE: u64 = 4096;
/// Pages one stamp block covers.
const BLOCK_PAGES: usize = 16;
const BLOCK: u64 = PAGE * BLOCK_PAGES as u64;
/// Page addresses one collect call returns at most.
const COLLECT_PAGES: usize = 4096;

/// The collect that last found each page of a block written.
#[derive(Clone, Copy, Default)]
struct Block {
    pages: [u64; BLOCK_PAGES],
    newest: u64,
}

/// CPU writes to guest memory: each collect stamps the pages written since the previous one with
/// a new generation, so a page stamped after generation `g` was written after the collect that
/// returned `g`. A range libc cannot watch counts as wholly written at every collect.
pub struct WriteTracker {
    generation: u64,
    blocks: HashMap<u64, Block>,
    pages: Vec<*mut c_void>,
    /// The write-watched arena: base and bytes.
    arena: (u64, u64),
}

impl WriteTracker {
    pub fn new(arena: (u64, u64)) -> Self {
        Self { generation: 1, blocks: HashMap::new(), pages: vec![std::ptr::null_mut(); COLLECT_PAGES], arena }
    }

    fn watched(&self, address: u64, bytes: u64) -> bool {
        let (base, size) = self.arena;
        size != 0 && address >= base && address.checked_add(bytes).is_some_and(|end| end <= base + size)
    }

    fn stamp(&mut self, page: u64, generation: u64) {
        let block = self.blocks.entry(page / BLOCK).or_default();
        block.pages[((page % BLOCK) / PAGE) as usize] = generation;
        block.newest = block.newest.max(generation);
    }

    fn stamp_range(&mut self, first: u64, stop: u64, generation: u64) {
        let mut page = first;
        while page < stop {
            self.stamp(page, generation);
            page += PAGE;
        }
    }

    /// Collects the writes in `[address, address + bytes)`: their generation, or 0 when libc cannot
    /// watch the range (no history to compare against later).
    pub fn collect<G: GuestArena + ?Sized>(&mut self, arena: &G, address: u64, bytes: u64) -> u64 {
        let Some(end) = address.checked_add(bytes).filter(|_| bytes != 0) else {
            return 0;
        };
        let first = address & !(PAGE - 1);
        let stop = end.div_ceil(PAGE) * PAGE;
        self.generation += 1;
        let generation = self.generation;
        if !self.watched(first, stop - first) {
            self.stamp_range(first, stop, generation);
            return 0;
        }
        let mut cursor = first;
        while cursor < stop {
            let mut pages = std::mem::take(&mut self.pages);
            let collected = arena.collect_writes(cursor, stop - cursor, &mut pages, true);
            let Some(count) = collected else {
                self.pages = pages;
                self.stamp_range(cursor, stop, generation);
                return 0;
            };
            for &page in &pages[..count] {
                self.stamp(page as u64 & !(PAGE - 1), generation);
            }
            let next = (count == pages.len()).then(|| pages[count - 1] as u64 + PAGE);
            self.pages = pages;
            match next {
                Some(next) => cursor = next,
                None => break,
            }
        }
        generation
    }

    /// Whether no CPU write touched the range after the collect that returned `generation`.
    pub fn unchanged_since<G: GuestArena + ?Sized>(&mut self, arena: &G, address: u64, bytes: u64, generation: u64) -> bool {
        if generation == 0 || self.collect(arena, address, bytes) == 0 {
            return false;
        }
        let first = address & !(PAGE - 1);
        let stop = (address + bytes).div_ceil(PAGE) * PAGE;
        let mut page = first;
        while page < stop {
            let block = page / BLOCK;
            match self.blocks.get(&block) {
                Some(stamps) if stamps.newest > generation => {
                    if stamps.pages[((page % BLOCK) / PAGE) as usize] > generation {
                        return false;
                    }
                    page += PAGE;
                }
                // The block has nothing newer: skip the rest of it.
                _ => page = (block + 1) * BLOCK,
            }
        }
        true
    }

    /// Forgets the CPU writes made to `[address, address + bytes)` so far: a retile of the range
    /// about to run reads them. A range libc cannot watch stays as `collect` stamped it.
    pub fn absorb<G: GuestArena + ?Sized>(&mut self, arena: &G, address: u64, bytes: u64) {
        let generation = self.collect(arena, address, bytes);
        if generation == 0 {
            return;
        }
        let stop = (address + bytes).div_ceil(PAGE) * PAGE;
        let mut page = address & !(PAGE - 1);
        while page < stop {
            let Some(block) = self.blocks.get_mut(&(page / BLOCK)) else {
                page = (page / BLOCK + 1) * BLOCK;
                continue;
            };
            let stamp = &mut block.pages[((page % BLOCK) / PAGE) as usize];
            if *stamp == generation {
                *stamp = 0;
            }
            page += PAGE;
        }
    }

    /// The pieces of `ranges` the CPU wrote after `since`, collected once each, merged, and the
    /// generation to pass next time.
    pub fn written_since<G: GuestArena + ?Sized>(&mut self, arena: &G, ranges: &[GuestRange], since: u64) -> (u64, Vec<GuestRange>) {
        let mut written: Vec<GuestRange> = Vec::new();
        let mut append = |address: u64, end: u64| match written.last_mut() {
            Some(last) if last.end() == address => last.bytes = end - last.address,
            _ => written.push(GuestRange::new(address, end - address)),
        };
        for &range in ranges {
            if range.bytes == 0 {
                continue;
            }
            if self.collect(arena, range.address, range.bytes) == 0 {
                append(range.address, range.end());
                continue;
            }
            let mut page = range.address & !(PAGE - 1);
            while page < range.end() {
                let block = page / BLOCK;
                let newer = self.blocks.get(&block).filter(|stamps| stamps.newest > since);
                let Some(stamps) = newer else {
                    page = (block + 1) * BLOCK;
                    continue;
                };
                if stamps.pages[((page % BLOCK) / PAGE) as usize] > since {
                    append(page.max(range.address), (page + PAGE).min(range.end()));
                }
                page += PAGE;
            }
        }
        (self.generation, written)
    }
}


/// How long `map_import` retries while a host write stays open on the pages, 1 ms apart.
const HOST_WRITE_WAIT: Duration = Duration::from_secs(1);
/// How many times the GPU-visible pieces are asked again with room for the count reported.
const GPU_RANGES_ATTEMPTS: usize = 3;

/// The live import aliases, by address, with their guest ranges: shared on the device thread by
/// the residency's [`Imports`] and the release path, which abandons them on a lost device.
pub struct ImportRecord<A: GuestArena + ?Sized> {
    arena: Arc<A>,
    aliases: Rc<RefCell<HashMap<usize, GuestRange>>>,
}

impl<A: GuestArena + ?Sized> ImportRecord<A> {
    /// Ends the import records of the aliases overlapping `range` without unmapping them: a lost
    /// device keeps their pages pinned until the process exits. How many.
    pub fn abandon(&self, range: GuestRange) -> usize {
        let mut aliases = self.aliases.borrow_mut();
        let abandoned: Vec<(usize, GuestRange)> = aliases
            .iter()
            .filter(|(_, alias)| alias.address < range.end() && range.address < alias.end())
            .map(|(&address, &alias)| (address, alias))
            .collect();
        for (address, alias) in &abandoned {
            aliases.remove(address);
            self.arena.forget_import(alias.address, alias.bytes);
        }
        abandoned.len()
    }

    pub fn live(&self) -> usize {
        self.aliases.borrow().len()
    }
}

/// The residency's guest memory over libc: import aliases, GPU-visible ranges and CPU writes.
pub struct Imports<A: GuestArena + ?Sized> {
    arena: Arc<A>,
    writes: WriteTracker,
    aliases: Rc<RefCell<HashMap<usize, GuestRange>>>,
}

impl<A: GuestArena + ?Sized> Imports<A> {
    pub fn new(arena: Arc<A>) -> Self {
        let watched = arena.arena();
        Self { arena, writes: WriteTracker::new(watched), aliases: Rc::default() }
    }

    /// The record of the aliases, for the device thread.
    pub fn record(&self) -> ImportRecord<A> {
        ImportRecord { arena: self.arena.clone(), aliases: self.aliases.clone() }
    }
}

impl<A: GuestArena + ?Sized> GuestMemory for Imports<A> {
    fn mapping_generation(&self) -> u64 { self.arena.gpu_generation() }
    fn map_import(&mut self, guest: u64, bytes: u64) -> Result<NonNull<c_void>, ImportRefusal> {
        let deadline = Instant::now() + HOST_WRITE_WAIT;
        loop {
            match self.arena.map_import_alias(guest, bytes) {
                Ok(alias) => {
                    self.aliases.borrow_mut().insert(alias.as_ptr() as usize, GuestRange::new(guest, bytes));
                    return Ok(alias);
                }
                Err(import::HOST_WRITE_OPEN) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(1)),
                Err(import::MADE) => return Err(ImportRefusal::NotGpuVisible),
                Err(code) => return Err(ImportRefusal::from_code(code)),
            }
        }
    }

    fn unmap_import(&mut self, alias: NonNull<c_void>) {
        self.arena.unmap_import_alias(alias);
        self.aliases.borrow_mut().remove(&(alias.as_ptr() as usize));
    }

    fn import_pages(&self, alias: NonNull<c_void>, bytes: u64) -> crate::Result<Vec<u64>> {
        let mut pages = vec![0; (bytes / crate::backend::va::GUEST_PAGE_BYTES) as usize];
        if self.arena.import_alias_pages(alias, &mut pages) != pages.len() || pages.contains(&0) {
            return Err(crate::Error::invalid("libc returned incomplete physical import page identities"));
        }
        Ok(pages)
    }

    fn map_backing(&mut self, guest: u64, bytes: u64) -> crate::Result<NonNull<c_void>> {
        self.arena.map_backing_alias(guest, bytes)
    }

    fn unmap_backing(&mut self, alias: NonNull<c_void>) {
        self.arena.unmap_backing_alias(alias);
    }

    fn gpu_ranges(&mut self, range: GuestRange) -> Vec<GuestRange> {
        let mut pieces = vec![interface::GuestRange::default(); 16];
        for attempt in 1..=GPU_RANGES_ATTEMPTS {
            let count = self.arena.gpu_visible(range.address, range.bytes, &mut pieces);
            if count <= pieces.len() {
                pieces.truncate(count);
                break;
            }
            if attempt == GPU_RANGES_ATTEMPTS {
                break;
            }
            pieces.resize(count, interface::GuestRange::default());
        }
        pieces.iter().map(|piece| GuestRange::new(piece.address as u64, piece.bytes as u64)).collect()
    }

    fn collect_writes(&mut self, address: u64, bytes: u64) -> crate::Result<u64> {
        Ok(self.writes.collect(&*self.arena, address, bytes))
    }

    fn unchanged_since(&mut self, address: u64, bytes: u64, generation: u64) -> crate::Result<bool> {
        Ok(self.writes.unchanged_since(&*self.arena, address, bytes, generation))
    }

    fn written_since(&mut self, ranges: &[GuestRange], since: u64) -> crate::Result<Option<(u64, Vec<GuestRange>)>> {
        Ok(Some(self.writes.written_since(&*self.arena, ranges, since)))
    }

    fn absorb_writes(&mut self, range: GuestRange) {
        self.writes.absorb(&*self.arena, range.address, range.bytes);
    }
}

/// One run of pages with the same access, as `VirtualQuery` describes it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Region {
    base: u64,
    end: u64,
    readable: bool,
    writable: bool,
}

/// Same-address access to guest memory (decisions D15): whether a range is mapped with the
/// access asked for, from `VirtualQuery`, cached until libc's guest mappings change.
pub struct GuestAccess<A: GuestArena + ?Sized> {
    arena: Arc<A>,
    generation: u64,
    /// Regions queried since the mappings last changed, in address order.
    regions: Vec<Region>,
}

impl<A: GuestArena + ?Sized> GuestAccess<A> {
    pub fn new(arena: Arc<A>) -> Self {
        Self { arena, generation: 0, regions: Vec::new() }
    }

    fn region(&mut self, address: u64) -> Option<Region> {
        let found = self.regions.partition_point(|region| region.end <= address);
        if let Some(&region) = self.regions.get(found).filter(|region| region.base <= address) {
            return Some(region);
        }
        let region = Self::query(address)?;
        self.regions.insert(found, region);
        Some(region)
    }

    #[cfg(all(feature = "wddm", target_os = "windows"))]
    fn query(address: u64) -> Option<Region> {
        use windows_sys::Win32::System::Memory::{
            VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY,
            PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
        };
        let mut memory: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
        // SAFETY: `memory` has room for the answer.
        if unsafe { VirtualQuery(address as usize as *const c_void, &mut memory, size) } != size {
            return None;
        }
        let base = memory.BaseAddress as u64;
        let protect = memory.Protect;
        let committed = memory.State == MEM_COMMIT && protect & (PAGE_GUARD | PAGE_NOACCESS) == 0;
        let writable = protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) != 0;
        let readable = writable || protect & (PAGE_READONLY | PAGE_EXECUTE_READ) != 0;
        Some(Region { base, end: base + memory.RegionSize as u64, readable: committed && readable, writable: committed && writable })
    }

    #[cfg(not(all(feature = "wddm", target_os = "windows")))]
    fn query(_address: u64) -> Option<Region> {
        None
    }

    /// Whether `[address, address + bytes)` is mapped guest memory with the access asked for.
    pub fn accessible(&mut self, address: u64, bytes: u64, writable: bool) -> bool {
        let Some(end) = address.checked_add(bytes).filter(|_| address != 0) else {
            return false;
        };
        let generation = self.arena.generation();
        if generation != self.generation {
            self.regions.clear();
            self.generation = generation;
        }
        let mut cursor = address;
        while cursor < end {
            match self.region(cursor) {
                Some(region) if region.readable && (region.writable || !writable) => cursor = region.end,
                _ => return false,
            }
        }
        true
    }

    pub fn read(&mut self, address: u64, out: &mut [u8]) -> crate::Result<()> {
        if !self.accessible(address, out.len() as u64, false) {
            return Err(crate::Error::invalid(format!("guest memory 0x{address:x}+0x{:x} is not readable", out.len())));
        }
        // SAFETY: mapped readable guest memory at the same address in this process.
        unsafe { std::ptr::copy_nonoverlapping(address as usize as *const u8, out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    pub fn write(&mut self, address: u64, data: &[u8]) -> crate::Result<()> {
        if !self.accessible(address, data.len() as u64, true) {
            return Err(crate::Error::invalid(format!("guest memory 0x{address:x}+0x{:x} is not writable", data.len())));
        }
        // SAFETY: mapped writable guest memory at the same address in this process.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), address as usize as *mut u8, data.len()) };
        Ok(())
    }
}
