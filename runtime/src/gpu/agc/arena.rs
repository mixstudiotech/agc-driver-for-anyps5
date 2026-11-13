//! The AMD target's own GPU memory (PLAN P3.4 2b, decisions D12, D14), without locks: state IBs
//! of draws and translated register pairs live here, referenced from the footprints in the
//! title's buffers.
//!
//! The arena is one GPU-visible, CPU-mapped range cut into equal blocks. A recording takes whole
//! blocks and allocates inside them by bumping (one thread records a buffer at a time); free
//! blocks sit on a lock-free stack whose head carries a tag against ABA. Submission marks every
//! block a submitted stream reaches with the queue's fence value; a recording that starts over
//! returns the blocks the GPU has finished with and keeps the rest until it has.

use crate::{Error, Result};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Where the arena's memory is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Range {
    /// The CPU mapping.
    pub cpu: *mut u8,
    /// The same memory's GPU virtual address.
    pub gpu: u64,
    pub bytes: usize,
}

// SAFETY: the range is plain memory the arena hands out in disjoint blocks.
unsafe impl Send for Range {}
unsafe impl Sync for Range {}

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Usage {
    pub blocks: usize,
    pub leased: usize,
    pub leases: u64,
    pub gpu: usize,
}

/// The arena.
pub struct Arena {
    range: Range,
    block_bytes: usize,
    /// Free list head: tag in the high 32 bits, block index (or [`NONE`]) in the low.
    head: AtomicU64,
    /// Each block's successor on the free list.
    next: Box<[AtomicU32]>,
    /// The fence value of the last submission that reached each block.
    fences: Box<[AtomicU64]>,
    leases: Box<[AtomicU32]>,
    /// Blocks recordings gave up while the GPU may still use them: tag and index, as `head`.
    retired: AtomicU64,
    /// Submissions up to this one have finished ([`Self::advance`]).
    completed: AtomicU64,
}

impl Arena {
    /// An arena over `range` in blocks of `block_bytes` (a power of two of at least 256 bytes).
    pub fn new(range: Range, block_bytes: usize) -> Result<Self> {
        if !block_bytes.is_power_of_two() || block_bytes < 256 || range.gpu % block_bytes as u64 != 0 {
            return Err(Error::invalid(format!(
                "Agc arena: blocks of {block_bytes} bytes over GPU 0x{:x}",
                range.gpu
            )));
        }
        let blocks = range.bytes / block_bytes;
        if blocks == 0 || blocks >= NONE as usize {
            return Err(Error::invalid(format!("Agc arena: {} bytes make {blocks} blocks", range.bytes)));
        }
        let next = (0..blocks)
            .map(|index| AtomicU32::new(if index + 1 < blocks { index as u32 + 1 } else { NONE }))
            .collect();
        let fences = (0..blocks).map(|_| AtomicU64::new(0)).collect();
        let leases = (0..blocks).map(|_| AtomicU32::new(0)).collect();
        Ok(Self { range, block_bytes, head: AtomicU64::new(0), next, fences, leases, retired: AtomicU64::new(u64::from(NONE)), completed: AtomicU64::new(0) })
    }

    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn usage(&self) -> Usage {
        let completed = self.completed.load(Ordering::Acquire);
        let mut usage = Usage { blocks: self.next.len(), leased: 0, leases: 0, gpu: 0 };
        for (leases, fence) in self.leases.iter().zip(self.fences.iter()) {
            let count = leases.load(Ordering::Acquire);
            usage.leased += usize::from(count != 0);
            usage.leases += u64::from(count);
            usage.gpu += usize::from(fence.load(Ordering::Acquire) > completed);
        }
        usage
    }

    /// Takes a free block, reclaiming retired ones the GPU has finished with when none is free.
    pub fn take(&self) -> Result<Block> {
        match self.pop_free() {
            Some(block) => Ok(block),
            None => {
                self.reclaim();
                self.pop_free().ok_or_else(|| Error::new(crate::ErrorKind::Busy, "Agc arena: no free block until the GPU finishes"))
            }
        }
    }

    fn pop_free(&self) -> Option<Block> {
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            let index = head as u32;
            if index == NONE {
                return None;
            }
            let next = self.next[index as usize].load(Ordering::Relaxed);
            let tagged = ((head >> 32).wrapping_add(1) << 32) | u64::from(next);
            match self.head.compare_exchange_weak(head, tagged, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(Block { index }),
                Err(current) => head = current,
            }
        }
    }

    /// Hands `block` back; it returns to use once the GPU has finished with it.
    pub fn retire(&self, block: Block) {
        let mut head = self.retired.load(Ordering::Acquire);
        loop {
            self.next[block.index as usize].store(head as u32, Ordering::Relaxed);
            let tagged = ((head >> 32).wrapping_add(1) << 32) | u64::from(block.index);
            match self.retired.compare_exchange_weak(head, tagged, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return,
                Err(current) => head = current,
            }
        }
    }

    /// Moves the retired blocks the GPU has finished with to the free list.
    fn reclaim(&self) {
        let completed = self.completed.load(Ordering::Acquire);
        let taken = self.retired.swap(u64::from(NONE), Ordering::AcqRel);
        let mut index = taken as u32;
        while index != NONE {
            let next = self.next[index as usize].load(Ordering::Relaxed);
            let block = Block { index };
            if self.leases[index as usize].load(Ordering::Acquire) == 0 && self.fences[index as usize].load(Ordering::Acquire) <= completed {
                self.give(block);
            } else {
                self.retire(block);
            }
            index = next;
        }
    }

    /// Submissions up to and including `sequence` have finished.
    pub fn advance(&self, sequence: u64) {
        self.completed.fetch_max(sequence, Ordering::AcqRel);
    }

    /// Returns `block` to the free list.
    pub fn give(&self, block: Block) {
        if self.leases[block.index as usize].load(Ordering::Acquire) != 0 {
            self.retire(block);
            return;
        }
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            self.next[block.index as usize].store(head as u32, Ordering::Relaxed);
            let tagged = ((head >> 32).wrapping_add(1) << 32) | u64::from(block.index);
            match self.head.compare_exchange_weak(head, tagged, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return,
                Err(current) => head = current,
            }
        }
    }

    /// The block holding GPU address `gpu`, if the arena has it.
    pub fn block_at(&self, gpu: u64) -> Option<u32> {
        let offset = gpu.checked_sub(self.range.gpu)?;
        (offset < self.range.bytes as u64).then(|| (offset / self.block_bytes as u64) as u32)
    }

    /// Notes that a submission waiting for `fence` reaches block `index`.
    pub fn submitted(&self, index: u32, fence: u64) {
        self.fences[index as usize].fetch_max(fence, Ordering::AcqRel);
    }

    /// Retains a recording's block while its side table or a pending submission can reach it.
    pub fn lease(self: &Arc<Self>, index: u32) -> Result<Lease> {
        let count = self.leases.get(index as usize).ok_or_else(|| Error::invalid("Agc arena: lease outside the arena"))?;
        count.fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| count.checked_add(1)).map_err(|_| Error::invalid("Agc arena: block lease count exhausted"))?;
        Ok(Lease { arena: self.clone(), index })
    }

    /// Whether the GPU is done with `block`, given the queue's completed fence value.
    pub fn idle(&self, block: &Block, completed: u64) -> bool {
        self.fences[block.index as usize].load(Ordering::Acquire) <= completed
    }

    /// CPU and GPU addresses of `bytes` at `offset` inside `block`.
    pub fn at(&self, block: &Block, offset: usize, bytes: usize) -> Result<(*mut u8, u64)> {
        if offset.checked_add(bytes).is_none_or(|end| end > self.block_bytes) {
            return Err(Error::invalid(format!(
                "Agc arena: {bytes} bytes at {offset} overrun a {}-byte block",
                self.block_bytes
            )));
        }
        let start = block.index as usize * self.block_bytes + offset;
        // SAFETY: inside the range, in a block its taker owns.
        Ok((unsafe { self.range.cpu.add(start) }, self.range.gpu + start as u64))
    }
}

pub struct Lease {
    arena: Arc<Arena>,
    index: u32,
}

impl Lease {
    pub fn index(&self) -> u32 { self.index }
}

impl Drop for Lease {
    fn drop(&mut self) { self.arena.leases[self.index as usize].fetch_sub(1, Ordering::Release); }
}

/// A block a recording owns until it gives it back.
#[derive(Debug, Eq, PartialEq)]
pub struct Block {
    index: u32,
}

impl Block {
    pub fn index(&self) -> u32 {
        self.index
    }
}

/// A recording's bump allocator over the blocks it takes (one thread at a time).
pub struct Blocks {
    held: Vec<Block>,
    used: usize,
}

impl Default for Blocks {
    fn default() -> Self {
        Self { held: Vec::new(), used: 0 }
    }
}

impl Blocks {
    /// `bytes` (aligned to `align`, a power of two up to 256) in the current block or a new one.
    pub fn allocate(&mut self, arena: &Arena, bytes: usize, align: usize) -> Result<(*mut u8, u64)> {
        if bytes > arena.block_bytes() || !align.is_power_of_two() || align > 256 {
            return Err(Error::invalid(format!(
                "Agc arena: {bytes} bytes aligned to {align} in {}-byte blocks",
                arena.block_bytes()
            )));
        }
        let offset = self.used.next_multiple_of(align);
        if self.held.is_empty() || offset + bytes > arena.block_bytes() {
            self.held.push(arena.take()?);
            self.used = 0;
            return self.allocate(arena, bytes, align);
        }
        self.used = offset + bytes;
        arena.at(self.held.last().unwrap(), offset, bytes)
    }

    /// Blocks held.
    pub fn count(&self) -> usize {
        self.held.len()
    }

    /// The blocks held, for the submissions that reach what was allocated in them.
    pub fn indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.held.iter().map(Block::index)
    }

    /// Retires every held block; what was allocated in them stays valid until the GPU has
    /// finished with the submissions that reached it.
    pub fn retire_all(&mut self, arena: &Arena) {
        for block in self.held.drain(..) {
            arena.retire(block);
        }
        self.used = 0;
    }
}

/// A CPU-mapped, GPU-visible range handed out once by address, from any thread and never
/// recycled: the runtime's state blocks and translated programs, which live as long as the
/// device.
pub struct Bump {
    range: Range,
    used: AtomicUsize,
}

impl Bump {
    pub fn new(range: Range) -> Self {
        Self { range, used: AtomicUsize::new(0) }
    }

    /// `bytes` aligned to `align` (a power of two): the CPU and GPU addresses.
    pub fn allocate(&self, bytes: usize, align: usize) -> Result<(*mut u8, u64)> {
        let mut used = self.used.load(Ordering::Relaxed);
        loop {
            let start = (self.range.gpu as usize + used).next_multiple_of(align) - self.range.gpu as usize;
            let end = start.checked_add(bytes).filter(|&end| end <= self.range.bytes).ok_or_else(|| {
                Error::new(crate::ErrorKind::Busy, format!("Agc: the {}-byte runtime arena is full", self.range.bytes))
            })?;
            match self.used.compare_exchange_weak(used, end, Ordering::AcqRel, Ordering::Relaxed) {
                // SAFETY: inside the range, and this thread alone won these bytes.
                Ok(_) => return Ok((unsafe { self.range.cpu.add(start) }, self.range.gpu + start as u64)),
                Err(current) => used = current,
            }
        }
    }

    /// Copies `words` into new memory aligned to `align` bytes and returns its GPU address.
    pub fn put(&self, words: &[u32], align: usize) -> Result<u64> {
        let (cpu, gpu) = self.allocate(std::mem::size_of_val(words), align)?;
        // SAFETY: the bytes `allocate` just gave this thread.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), cpu.cast::<u32>(), words.len()) };
        Ok(gpu)
    }

    pub fn gpu(&self) -> u64 {
        self.range.gpu
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }
}
