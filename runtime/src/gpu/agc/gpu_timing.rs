//! GPU time of the parts of each segment (`APS5_AGC_GPU_TIMING`): timestamps before and after its
//! surface copies, after the title's commands and after a flip's write-back, written the way radv
//! writes timestamp queries. The `agc-joint-profile` skill reads the per-second `[gpu-time]` line.

use crate::amd::defs::cp_gfx11 as cp;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

#[cfg(all(feature = "wddm", target_os = "windows"))]
mod segments;
#[cfg(all(feature = "wddm", target_os = "windows"))]
pub use segments::{GpuTiming, Stamp};

/// The environment variable that turns the timing on.
const VARIABLE: &str = "APS5_AGC_GPU_TIMING";
/// Marker timestamps kept: well over what ten seconds of a title write.
const STAGE_SLOTS: u64 = 1 << 18;
/// Bytes of a marker's text kept.
const STAGE_TEXT_WORDS: usize = 5;

/// Dwords a debug marker takes to hold its timestamp, 0 when none is written.
pub fn marker_dwords() -> u32 {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var_os(VARIABLE).is_some());
    if *ENABLED {
        cp::meg::RELEASE_MEM::WORDS
    } else {
        0
    }
}

#[derive(Default)]
struct StageEntry {
    /// The recording's identity, shifted left, with bit 0 set on a push; 0 while unwritten.
    header: AtomicU64,
    text: [AtomicU64; STAGE_TEXT_WORDS],
}

/// The title's debug markers, each with a bottom-of-pipe timestamp at its address: recording
/// threads claim entries, the device thread adds up the blocks they delimit.
pub struct StageLog {
    base: u64,
    entries: Box<[StageEntry]>,
    next: AtomicU64,
}

impl StageLog {
    /// Claims the next entry for a marker of the recording `identity`; `text` is `None` on a pop.
    pub fn note(&self, identity: u64, text: Option<&[u8]>) -> u64 {
        let slot = self.next.fetch_add(1, Ordering::Relaxed);
        let entry = &self.entries[(slot % STAGE_SLOTS) as usize];
        let mut words = [0u64; STAGE_TEXT_WORDS];
        for (word, chunk) in words.iter_mut().zip(text.unwrap_or(&[]).chunks(8)) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            *word = u64::from_le_bytes(bytes);
        }
        for (cell, word) in entry.text.iter().zip(words) {
            cell.store(word, Ordering::Relaxed);
        }
        entry.header.store((identity << 1) | u64::from(text.is_some()), Ordering::Release);
        slot
    }

    /// Where the GPU writes the timestamp of the marker in entry `slot`.
    pub fn address(&self, slot: u64) -> u64 {
        self.base + (slot % STAGE_SLOTS) * 8
    }
}
