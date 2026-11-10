//! Command buffer recordings (PLAN P3.4 2a, decisions D12, D14), without locks.
//!
//! The front end keeps one [`Recording`] per title `CommandBuffer` object: where the last
//! command ended and the target's [`Recorder`] holding the buffer's state. A recording starts
//! over when the title resets the buffer (its up cursor below where the last command ended) or
//! points it at other memory; the title's allocation callback moving it to a new chunk during a
//! command continues the same recording.
//!
//! The SDK's contract is that one thread at a time records a given buffer, while different
//! buffers record concurrently. Each buffer object owns a slot of a fixed table, claimed with a
//! compare-and-swap. The thread that holds a slot has its recording without a lock, and a
//! second, concurrent writer is an error; the title's allocation callback runs inside one of
//! that thread's commands and may record on the same buffer (`Core::StateBuffer` does when its
//! segment is lost), which continues the recording. Submission does not look recordings up: it
//! follows the commands themselves.
//!
//! A recording is also looked up by the address of its commands, when the title submits them or
//! writes a packet among them by itself. The lookup reads where each recording stands without
//! holding its slot and takes only the one it is after, so it is never in the way of a thread
//! recording another buffer; a thread that starts a command on that very buffer meanwhile waits
//! the lookup out, which lasts one settling of the recording at most.

use super::target::Recorder;
use crate::{Error, Result};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// What one `CommandBuffer` has recorded since it was last reset.
pub struct Recording {
    /// The target's state for this buffer.
    pub recorder: Box<dyn Recorder>,
}

/// Where the title's buffer stands before the front end writes a command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cursor {
    pub bottom: u64,
    pub up: u64,
}

struct Slot {
    /// The title's `CommandBuffer` address; 0 while the slot is free.
    key: AtomicUsize,
    /// The slot's recording (`Box::into_raw`), null until the buffer records.
    recording: AtomicPtr<Recording>,
    /// The thread working on the recording ([`this_thread`]), with [`LOOKUP`] set when it only
    /// looks the recording up; 0 while none is.
    owner: AtomicUsize,
    /// The chunk the buffer is writing in, and where the last command written ended (the up
    /// cursor right after it): a lookup by address reads them without holding the slot.
    bottom: AtomicU64,
    cursor: AtomicU64,
}

impl Slot {
    fn stands(&self) -> (u64, u64) {
        (self.bottom.load(Ordering::Acquire), self.cursor.load(Ordering::Acquire))
    }

    fn stand(&self, bottom: u64, cursor: u64) {
        self.bottom.store(bottom, Ordering::Release);
        self.cursor.store(cursor, Ordering::Release);
    }
}

/// Set in a slot's owner while a thread has the recording for a lookup by address, not to record:
/// no thread mark has it, user-space addresses end below it.
const LOOKUP: usize = 1 << (usize::BITS - 1);

/// Every buffer object's recording.
pub struct Recordings {
    slots: Box<[Slot]>,
    /// 64 − log2(slots): Fibonacci hashing keeps the product's top bits.
    shift: u32,
    /// The claimed slots, in the order they were claimed, for finding a recording by its commands.
    claimed: Box<[AtomicUsize]>,
    claimed_count: AtomicUsize,
}

/// Buffer objects a title may use at once.
const DEFAULT_SLOTS: usize = 1 << 14;

impl Default for Recordings {
    fn default() -> Self {
        Self::with_slots(DEFAULT_SLOTS)
    }
}

/// This thread, as a number that is never 0.
fn this_thread() -> usize {
    thread_local!(static MARK: u8 = const { 0 });
    MARK.with(|mark| std::ptr::from_ref(mark) as usize)
}

/// One command's hold on its buffer's recording, until dropped.
pub struct Claim<'a> {
    slot: &'a Slot,
    /// Inside a command of the same thread: the outermost claim releases the slot.
    nested: bool,
}

impl Claim<'_> {
    /// The recording. Take it after the title's allocation callback has returned, and do not
    /// keep it across one: commands the callback records need it too.
    pub fn recording(&mut self) -> &mut Recording {
        // SAFETY: `enter` created the recording, and this thread holds the slot; the borrow of
        // the claim ends before control can return to the title.
        unsafe { &mut *self.slot.recording.load(Ordering::Acquire) }
    }

    /// Notes where the buffer stands after the command just written: its chunk, and the up
    /// cursor right after the command.
    pub fn stand(&mut self, bottom: u64, cursor: u64) {
        self.slot.stand(bottom, cursor);
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        if !self.nested {
            self.slot.owner.store(0, Ordering::Release);
        }
    }
}

impl Recordings {
    /// A table for `slots` buffer objects (rounded up to a power of two).
    pub fn with_slots(slots: usize) -> Self {
        let count = slots.next_power_of_two().max(2);
        let slots = (0..count)
            .map(|_| Slot {
                key: AtomicUsize::new(0),
                recording: AtomicPtr::new(std::ptr::null_mut()),
                owner: AtomicUsize::new(0),
                bottom: AtomicU64::new(0),
                cursor: AtomicU64::new(0),
            })
            .collect();
        let claimed = (0..count).map(|_| AtomicUsize::new(0)).collect();
        Self { slots, shift: 64 - count.trailing_zeros(), claimed, claimed_count: AtomicUsize::new(0) }
    }

    /// The slot of buffer object `key`, claimed on first use.
    fn slot(&self, key: usize) -> Result<&Slot> {
        let mask = self.slots.len() - 1;
        let mut index = ((key as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> self.shift) as usize;
        for _ in 0..self.slots.len() {
            let slot = &self.slots[index];
            match slot.key.compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    let at = self.claimed_count.fetch_add(1, Ordering::AcqRel);
                    self.claimed[at].store(index, Ordering::Release);
                    return Ok(slot);
                }
                Err(owner) if owner == key => return Ok(slot),
                Err(_) => index = (index + 1) & mask,
            }
        }
        Err(Error::invalid(format!(
            "Agc: more than {} command buffer objects record at once",
            self.slots.len()
        )))
    }

    /// Runs `with` on the recording whose commands hold `address`, if a recording does and no
    /// thread is writing to it right now. A buffer being submitted has finished recording, so
    /// its slot is free.
    pub fn with_recording_at(&self, address: u64, with: impl FnOnce(&mut Recording)) {
        let _ = self.find(|bottom, cursor| bottom <= address && address < cursor, with);
    }

    /// As [`Self::with_recording_at`], for a packet the title writes by itself: also into the
    /// recording whose commands end where the packet starts (the title writes at its cursor,
    /// `sceAgcDriverWaitUntilSafeForRendering`), so what the packet notes is not lost.
    pub fn with_recording_writing_at(&self, address: u64, with: impl FnOnce(&mut Recording)) {
        if let Err(with) = self.find(|bottom, cursor| bottom <= address && address < cursor, with) {
            let _ = self.find(|bottom, cursor| bottom <= address && address == cursor, with);
        }
    }

    /// Runs `with` on the first recording that stands where `holds` says (its chunk, its cursor)
    /// and that no thread is writing to; gives `with` back when there is none. Only that
    /// recording's slot is taken: a thread recording another buffer never meets the lookup.
    fn find<F: FnOnce(&mut Recording)>(&self, holds: impl Fn(u64, u64) -> bool, with: F) -> std::result::Result<(), F> {
        let lookup = this_thread() | LOOKUP;
        let count = self.claimed_count.load(Ordering::Acquire).min(self.claimed.len());
        for claimed in &self.claimed[..count] {
            let slot = &self.slots[claimed.load(Ordering::Acquire)];
            let (bottom, cursor) = slot.stands();
            if !holds(bottom, cursor) || slot.owner.compare_exchange(0, lookup, Ordering::Acquire, Ordering::Relaxed).is_err() {
                continue;
            }
            // Asked again now that no thread can move it.
            let (bottom, cursor) = slot.stands();
            // SAFETY: holding the slot gives this thread the only access to its recording.
            let found = unsafe { slot.recording.load(Ordering::Acquire).as_mut() }.filter(|_| holds(bottom, cursor));
            if let Some(recording) = found {
                with(recording);
                slot.owner.store(0, Ordering::Release);
                return Ok(());
            }
            slot.owner.store(0, Ordering::Release);
        }
        Err(with)
    }

    /// A command's hold on the recording of buffer object `key` standing at `cursor`, started
    /// anew with `start`'s recorder when the title reset the buffer or gave it other memory. A
    /// command the title records from the allocation callback of another continues the
    /// recording, wherever the callback has moved the buffer by then.
    pub fn enter(&self, key: usize, cursor: Cursor, start: impl FnOnce() -> Box<dyn Recorder>) -> Result<Claim<'_>> {
        let slot = self.slot(key)?;
        let thread = this_thread();
        loop {
            match slot.owner.compare_exchange(0, thread, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => break,
                Err(owner) if owner == thread => return Ok(Claim { slot, nested: true }),
                // Another thread looks the recording up by address (a submission, a packet the
                // title writes by itself): it has it for a moment and records nothing.
                Err(owner) if owner & LOOKUP != 0 && owner != thread | LOOKUP => std::hint::spin_loop(),
                Err(_) => {
                    return Err(Error::invalid(format!(
                        "Agc: command buffer 0x{key:x} is recorded on two threads at once"
                    )))
                }
            }
        }
        let claim = Claim { slot, nested: false };
        let current = slot.recording.load(Ordering::Acquire);
        let (bottom, last) = slot.stands();
        // SAFETY: holding the slot gives this thread the only access to its recording.
        match unsafe { current.as_mut() } {
            Some(_) if bottom == cursor.bottom && last <= cursor.up => {}
            Some(recording) => {
                *recording = Recording { recorder: start() };
                slot.stand(cursor.bottom, cursor.up);
            }
            None => {
                slot.recording.store(Box::into_raw(Box::new(Recording { recorder: start() })), Ordering::Release);
                slot.stand(cursor.bottom, cursor.up);
            }
        }
        Ok(claim)
    }
}

impl Drop for Recordings {
    fn drop(&mut self) {
        for slot in self.slots.iter() {
            let recording = slot.recording.swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !recording.is_null() {
                // SAFETY: each slot owns its `Box::into_raw` recording, freed once here.
                drop(unsafe { Box::from_raw(recording) });
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/agc/record.rs"]
mod tests;
