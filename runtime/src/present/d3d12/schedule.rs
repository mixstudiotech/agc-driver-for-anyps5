//! The presenter's frame and fence schedule (FINAL-REPORT §6, Milestone 1b). Frame N = 1, 2, 3...
//! uses interchange slot N mod S. The native queue writes the slot only after F_copy reaches
//! N - S, then signals F_ready = N; D3D12 waits for F_ready = N, copies the slot and signals
//! F_copy = N. Both fences advance by exactly one per frame and F_copy never passes F_ready.
//! CPU-written slots follow the same rule with F_copy alone ([`CopyLedger`]).

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameSchedule {
    slots: u64,
}

impl FrameSchedule {
    pub fn new(slots: usize) -> Result<Self> {
        if slots == 0 {
            return Err(Error::invalid("the presenter needs at least one slot"));
        }
        Ok(Self {
            slots: slots as u64,
        })
    }

    pub fn slots(&self) -> usize {
        self.slots as usize
    }

    /// The slot frame `frame` (from 1) uses.
    pub fn slot(&self, frame: u64) -> usize {
        (frame % self.slots) as usize
    }

    /// The F_copy value that frees this frame's slot, when an earlier frame used it.
    pub fn reuse_wait(&self, frame: u64) -> Option<u64> {
        (frame > self.slots).then(|| frame - self.slots)
    }
}

/// Checks the fence invariants against what a run observed, frame by frame.
#[derive(Clone, Debug)]
pub struct FenceLedger {
    schedule: FrameSchedule,
    ready: u64,
    copied: u64,
}

impl FenceLedger {
    pub fn new(schedule: FrameSchedule) -> Self {
        Self {
            schedule,
            ready: 0,
            copied: 0,
        }
    }

    pub fn ready(&self) -> u64 {
        self.ready
    }

    pub fn copied(&self) -> u64 {
        self.copied
    }

    /// The native queue is about to write `frame`'s slot, having made it wait for the F_copy
    /// value [`FrameSchedule::reuse_wait`] names; `copied_floor` is the least F_copy value that
    /// wait guarantees.
    pub fn native_write(&mut self, frame: u64, copied_floor: u64) -> Result<()> {
        if frame != self.ready + 1 {
            return Err(Error::invalid(format!(
                "native write of frame {frame} after F_ready {}",
                self.ready
            )));
        }
        if let Some(wait) = self.schedule.reuse_wait(frame) {
            if copied_floor < wait {
                return Err(Error::invalid(format!(
                    "frame {frame} would overwrite slot {} while F_copy may be below {wait}",
                    self.schedule.slot(frame)
                )));
            }
        }
        Ok(())
    }

    pub fn signal_ready(&mut self, frame: u64) -> Result<()> {
        if frame != self.ready + 1 {
            return Err(Error::invalid(format!(
                "F_ready {frame} after {}: it must advance by one",
                self.ready
            )));
        }
        self.ready = frame;
        Ok(())
    }

    pub fn signal_copied(&mut self, frame: u64) -> Result<()> {
        if frame != self.copied + 1 || frame > self.ready {
            return Err(Error::invalid(format!(
                "F_copy {frame} after {} with F_ready {}",
                self.copied, self.ready
            )));
        }
        self.copied = frame;
        Ok(())
    }
}

/// Checks the F_copy-only schedule of CPU-written slots: the CPU writes frame N's slot after
/// F_copy reached N - S and before submitting the copy, so there is no F_ready; a resize first
/// drains every submitted copy. Skipped presents take no frame number.
#[derive(Clone, Debug)]
pub struct CopyLedger {
    schedule: FrameSchedule,
    written: u64,
    copied: u64,
}

impl CopyLedger {
    pub fn new(schedule: FrameSchedule) -> Self {
        Self {
            schedule,
            written: 0,
            copied: 0,
        }
    }

    pub fn written(&self) -> u64 {
        self.written
    }

    pub fn copied(&self) -> u64 {
        self.copied
    }

    /// The CPU is about to write `frame`'s slot, having waited until F_copy reached
    /// `copied_floor`.
    pub fn cpu_write(&mut self, frame: u64, copied_floor: u64) -> Result<()> {
        if frame != self.written + 1 {
            return Err(Error::invalid(format!(
                "CPU write of frame {frame} after frame {}",
                self.written
            )));
        }
        if let Some(wait) = self.schedule.reuse_wait(frame) {
            if copied_floor < wait {
                return Err(Error::invalid(format!(
                    "frame {frame} would overwrite slot {} while F_copy may be below {wait}",
                    self.schedule.slot(frame)
                )));
            }
        }
        self.written = frame;
        Ok(())
    }

    pub fn signal_copied(&mut self, frame: u64) -> Result<()> {
        if frame != self.copied + 1 || frame > self.written {
            return Err(Error::invalid(format!(
                "F_copy {frame} after {} with frame {} written",
                self.copied, self.written
            )));
        }
        self.copied = frame;
        Ok(())
    }

    /// A resize may reallocate the slots and back buffers once F_copy reached every written
    /// frame.
    pub fn drain(&self, copied: u64) -> Result<()> {
        if copied < self.written {
            return Err(Error::invalid(format!(
                "resize with F_copy {copied} below the last written frame {}",
                self.written
            )));
        }
        Ok(())
    }
}
