//! The timestamps of each segment, and what the device thread makes of them and of the markers.

use super::{StageEntry, StageLog, STAGE_SLOTS, VARIABLE};
use crate::backend::{host_pm4, MemoryKind};
use crate::gpu::packets::PacketStream;
use crate::{Buffer, Device, Result};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// `[gpu-time]` lines between two `[gpu-stage]` tables: one is printed about every second.
const STAGE_REPORT_INTERVAL: u64 = 10;
/// The GPU clock's rate; the measured `clock` of the `[gpu-time]` lines agrees with it.
const CLOCK_HZ: f64 = 100e6;

/// A segment's timestamps, in the order the GPU reaches them.
const STAMPS: u64 = 4;
const BEFORE_COPIES: u64 = 0;
const AFTER_COPIES: u64 = 1;
const AFTER_TITLE: u64 = 2;
const AFTER_WRITE_BACK: u64 = 3;
/// Far more than are ever on the GPU at once.
const SLOTS: u64 = 4096;

/// GPU clock ticks spent since the last report.
#[derive(Default)]
pub struct GpuTimes {
    pub segments: u64,
    /// Surface copy passes planned, and their dispatches.
    pub copy_passes: u64,
    pub copy_dispatches: u64,
    /// Target clears planned, and the segments with surface work ahead of the title's commands.
    pub clears: u64,
    pub work_segments: u64,
    pub copies: u64,
    pub title: u64,
    pub write_back: u64,
    /// The first and last timestamps seen, which give the clock's rate.
    pub first: u64,
    pub last: u64,
}

/// Which timestamps of a segment one of the stamps after its commands is.
#[derive(Clone, Copy)]
pub enum Stamp {
    AfterTitle,
    AfterWriteBack,
}

pub struct GpuTiming {
    memory: Buffer,
    next: u64,
    /// The slot of each segment on the GPU, by sequence number, and for a segment of a listed
    /// frame (`APS5_AGC_PASSES`) what its `[pass]` lines call it: frame, queue and start.
    running: HashMap<u64, (u64, Option<(u64, String, u64)>)>,
    times: GpuTimes,
    /// The marker timestamps the GPU writes.
    stage_memory: Buffer,
    stages: Arc<StageLog>,
    stage_read: u64,
    stage_reports: u64,
    /// Flips since the last table, per video-out handle: a title can flip several planes a frame.
    stage_flips: HashMap<u32, u64>,
}

impl GpuTiming {
    /// `None` unless `APS5_AGC_GPU_TIMING` is set.
    pub fn from_environment(device: &mut Device) -> Result<Option<Self>> {
        if std::env::var_os(VARIABLE).is_none() {
            return Ok(None);
        }
        let mut memory = device.allocate_kind((SLOTS * STAMPS * 8) as usize, MemoryKind::Commands)?;
        memory.fill(0)?;
        let mut stage_memory = device.allocate_kind((STAGE_SLOTS * 8) as usize, MemoryKind::Commands)?;
        stage_memory.fill(0)?;
        let entries = (0..STAGE_SLOTS).map(|_| StageEntry::default()).collect();
        let stages = Arc::new(StageLog { base: stage_memory.address(), entries, next: AtomicU64::new(0) });
        Ok(Some(Self {
            memory,
            next: 0,
            running: HashMap::new(),
            times: GpuTimes::default(),
            stage_memory,
            stages,
            stage_read: 0,
            stage_reports: 0,
            stage_flips: HashMap::new(),
        }))
    }

    /// The log recording threads note the title's markers in.
    pub fn stage_log(&self) -> Arc<StageLog> {
        self.stages.clone()
    }

    /// The marker blocks finished since the last call, by nesting path: calls and GPU ticks.
    fn stage_times(&mut self) -> BTreeMap<String, (u64, u64)> {
        let mut times = BTreeMap::new();
        let Some((memory, _)) = self.stage_memory.mapping() else { return times };
        let end = self.stages.next.load(Ordering::Acquire);
        let start = self.stage_read.max(end.saturating_sub(STAGE_SLOTS));
        self.stage_read = end;
        let mut stacks: HashMap<u64, Vec<(String, u64)>> = HashMap::new();
        for slot in start..end {
            let index = (slot % STAGE_SLOTS) as usize;
            let entry = &self.stages.entries[index];
            let header = entry.header.swap(0, Ordering::AcqRel);
            // SAFETY: the buffer's persistent mapping, inside it by `STAGE_SLOTS`; the GPU may
            // still be writing the newest entries, hence volatile.
            let time = unsafe { memory.cast::<u64>().add(index).read_volatile() };
            unsafe { memory.cast::<u64>().add(index).write_volatile(0) };
            if header == 0 || time == 0 {
                continue;
            }
            let stack = stacks.entry(header >> 1).or_default();
            if header & 1 != 0 {
                let bytes: Vec<u8> = entry.text.iter().flat_map(|word| word.load(Ordering::Relaxed).to_le_bytes()).take_while(|&byte| byte != 0).collect();
                stack.push((String::from_utf8_lossy(&bytes).into_owned(), time));
            } else if let Some((name, started)) = stack.pop() {
                let mut path: Vec<&str> = stack.iter().map(|(parent, _)| parent.as_str()).collect();
                path.push(&name);
                let total = times.entry(path.join("/")).or_insert((0, 0));
                total.0 += 1;
                total.1 += time.saturating_sub(started);
            }
        }
        times
    }

    /// A flip of the video-out `handle` was queued.
    pub fn note_flip(&mut self, handle: u32) {
        *self.stage_flips.entry(handle).or_default() += 1;
    }

    /// Every [`STAGE_REPORT_INTERVAL`] calls prints the GPU time of the title's marker blocks per
    /// frame. A block's time runs from when everything before it finished to when everything in
    /// it did, so it includes waits inside the block and blocks on other queues can overlap it.
    pub fn report_stages(&mut self) {
        self.stage_reports += 1;
        if self.stage_reports % STAGE_REPORT_INTERVAL != 0 {
            return;
        }
        let frames = self.stage_flips.values().copied().max().unwrap_or(0).max(1);
        self.stage_flips.clear();
        eprintln!("[gpu-stage] over {frames} frames:");
        for (path, (calls, ticks)) in self.stage_times() {
            let ms = ticks as f64 / CLOCK_HZ * 1e3 / frames as f64;
            if ms >= 0.05 {
                eprintln!("[gpu-stage] {ms:8.2} ms/frame {:7.1} calls/frame  {path}", calls as f64 / frames as f64);
            }
        }
    }

    /// The slot of the next segment.
    pub fn open(&mut self) -> u64 {
        let slot = self.next % SLOTS;
        self.next += 1;
        slot
    }

    fn address(&self, slot: u64, stamp: u64) -> u64 {
        self.memory.address() + (slot * STAMPS + stamp) * 8
    }

    /// `copies` with a timestamp on each side. The copy passes end with the GPU idle, so the
    /// command processor's clock read after them is when they finished. Without commands only the
    /// second is written: a pair around nothing measures the write confirmation of the first.
    pub fn around_copies(&self, slot: u64, copies: Vec<u32>, compute: bool) -> Result<Vec<u32>> {
        let stream = |stamp| -> Result<Vec<u32>> {
            let mut stream = if compute { PacketStream::new_compute() } else { PacketStream::new() };
            stream.copy_timestamp(self.address(slot, stamp))?;
            Ok(stream.into_words())
        };
        if copies.is_empty() {
            return stream(AFTER_COPIES);
        }
        let mut words = stream(BEFORE_COPIES)?;
        words.extend(copies);
        words.extend(stream(AFTER_COPIES)?);
        Ok(words)
    }

    /// An IB recording when the work before it has finished.
    pub fn stamp_image(&self, slot: u64, stamp: Stamp, compute: bool) -> Result<Vec<u32>> {
        let index = match stamp {
            Stamp::AfterTitle => AFTER_TITLE,
            Stamp::AfterWriteBack => AFTER_WRITE_BACK,
        };
        let mut stream = if compute { PacketStream::new_compute() } else { PacketStream::new() };
        stream.release_timestamp(self.address(slot, index))?;
        host_pm4::pad(stream.words())
    }

    /// A recording's surface copies were planned.
    pub fn copies_planned(&mut self, passes: usize, dispatches: usize) {
        self.times.copy_passes += passes as u64;
        self.times.copy_dispatches += dispatches as u64;
    }

    /// A target clear was planned.
    pub fn clear_planned(&mut self) {
        self.times.clears += 1;
    }

    /// The segment using `slot` is queued as `sequence`; `listed` names it when its frame is
    /// listed, and its timestamps are then printed as well.
    pub fn queued(&mut self, sequence: u64, slot: u64, listed: Option<(u64, String, u64)>) {
        self.running.insert(sequence, (slot, listed));
    }

    /// Segment `sequence` has finished: adds up its timestamps.
    pub fn finished(&mut self, sequence: u64) {
        let Some((slot, listed)) = self.running.remove(&sequence) else { return };
        let Some((memory, _)) = self.memory.mapping() else { return };
        // SAFETY: the buffer's persistent mapping, inside it by `SLOTS`, of a finished segment.
        let read = |stamp: u64| unsafe { memory.cast::<u64>().add((slot * STAMPS + stamp) as usize).read_volatile() };
        let (before, copied, titled, written) = (read(BEFORE_COPIES), read(AFTER_COPIES), read(AFTER_TITLE), read(AFTER_WRITE_BACK));
        // SAFETY: as above; the slot's next user must not see this segment's stamps.
        unsafe { std::ptr::write_bytes(memory.cast::<u64>().add((slot * STAMPS) as usize), 0, STAMPS as usize) };
        let had_work = before != 0;
        let before = if had_work { before } else { copied };
        // The raw clock values (10 ns): on one queue the time between two segments' last
        // stamps is what the later one took, waits included.
        if let Some((frame, queue, start)) = listed {
            eprintln!("[gpu-seg] frame {frame} {queue} 0x{start:x}: before {before} copied {copied} titled {titled}");
        }
        if copied == 0 || copied < before || titled < copied {
            return;
        }
        let times = &mut self.times;
        times.segments += 1;
        times.work_segments += u64::from(had_work);
        times.copies += copied - before;
        times.title += titled - copied;
        if written >= titled {
            times.write_back += written - titled;
        }
        times.first = if times.first == 0 { before } else { times.first.min(before) };
        times.last = times.last.max(titled).max(written);
    }

    /// What was added up since the last call.
    pub fn take(&mut self) -> GpuTimes {
        std::mem::take(&mut self.times)
    }
}
