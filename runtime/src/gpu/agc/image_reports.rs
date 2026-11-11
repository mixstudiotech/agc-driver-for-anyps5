//! The runtime's side of [`shader::image_reports`]: the table translated programs report the
//! images they find at run time through, read once a frame.
//!
//! The table is memory the GPU and the CPU both read and write while work runs. Nothing here
//! waits for the GPU: a slot read while a program stores into it does not check and is reported
//! again, and an image the table wrongly holds as known is reported again once the runtime
//! takes the mark back.

use crate::backend::wddm::{Buffer, Device, MemoryKind};
use crate::{Error, Result};
use shader::image_reports::{fold, hash, known_offset, CHECK, REPORTS_OFFSET, REPORT_BYTES, REPORT_CHECK_OFFSET, REPORT_SLOTS, TABLE_BYTES};

/// Where the two hashes of a bucket lie in it.
const WAYS: [u32; 2] = [0, 4];

pub struct ImageReports {
    /// Readback memory: the programs' stores reach it, and the CPU's writes the programs,
    /// without a cache writeback.
    table: Buffer,
    words: *mut u32,
    /// Which key of a full bucket the last mark took the place of: the next takes the other.
    evicted: bool,
}

impl ImageReports {
    pub fn new(device: &mut Device) -> Result<Self> {
        let mut table = device.allocate_kind(TABLE_BYTES as usize, MemoryKind::Readback)?;
        table.fill(0)?;
        let (memory, bytes) = table.mapping().ok_or_else(|| Error::unsupported("the image report table has no CPU mapping"))?;
        if bytes < TABLE_BYTES as usize {
            return Err(Error::invalid("the image report table is mapped short"));
        }
        Ok(Self { words: memory.cast(), table, evicted: false })
    }

    /// The table's GPU address, which translations are built against.
    pub fn address(&self) -> u64 {
        self.table.address()
    }

    fn read(&self, offset: u32) -> u32 {
        // SAFETY: `offset` is inside the table's mapping, which lives as long as `self.table`.
        unsafe { self.words.byte_add(offset as usize).read_volatile() }
    }

    fn write(&mut self, offset: u32, value: u32) {
        // SAFETY: as `read`.
        unsafe { self.words.byte_add(offset as usize).write_volatile(value) }
    }

    /// The T#s programs reported since the last call, each slot emptied as it is read.
    pub fn take(&mut self) -> Vec<[u32; 8]> {
        let mut reported = Vec::new();
        for slot in 0..REPORT_SLOTS {
            let at = REPORTS_OFFSET + slot * REPORT_BYTES;
            let check = self.read(at + REPORT_CHECK_OFFSET);
            if check == 0 {
                continue;
            }
            let tsharp: [u32; 8] = std::array::from_fn(|k| self.read(at + k as u32 * 4));
            // Emptied either way: two programs that stored different images at once left
            // neither whole, and both report again.
            self.write(at + REPORT_CHECK_OFFSET, 0);
            if fold(&tsharp) ^ CHECK == check {
                reported.push(tsharp);
            }
        }
        reported
    }

    /// Programs stop reporting the image: the runtime has its host copy current, or the image
    /// needs none. Its bucket holds two images; a third takes the place of one of them, which
    /// is then reported again when next used.
    pub fn mark(&mut self, tsharp: &[u32; 8]) {
        let hash = hash(tsharp);
        let bucket = known_offset(hash);
        if WAYS.iter().any(|&way| self.read(bucket + way) == hash) {
            return;
        }
        let way = WAYS.into_iter().find(|&way| self.read(bucket + way) == 0).unwrap_or_else(|| {
            self.evicted = !self.evicted;
            WAYS[usize::from(self.evicted)]
        });
        self.write(bucket + way, hash);
    }

    /// Programs report the image again when they next use it.
    pub fn unmark(&mut self, tsharp: &[u32; 8]) {
        let hash = hash(tsharp);
        let bucket = known_offset(hash);
        for way in WAYS {
            if hash != 0 && self.read(bucket + way) == hash {
                self.write(bucket + way, 0);
            }
        }
    }
}
