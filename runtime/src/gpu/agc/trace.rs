//! Opt-in command snapshots for GPU fault diagnosis. The ordinary submission path never
//! enters this module. Private IB copies keep the original call/tail-chain structure while
//! progress markers identify packets inside guest chunks and cached native state IBs.
//! Cloned execution assumes the original command bytes are not modified after the snapshot.

use super::arena::Bump;
use super::packet_dwords;
use crate::agc::header;
use crate::amd::defs::pkt3;
use crate::backend::host_pm4::{self, IB_ALIGNMENT_DWORDS, MAX_IB_DWORDS};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

const MAX_NODES: usize = 256;
const MAX_SOURCE_DWORDS: usize = 1 << 20;
const CHAIN: u32 = 1 << 20;

#[derive(Clone, Debug)]
pub struct SourceRange {
    pub first: usize,
    pub address: u64,
    pub dwords: usize,
}

pub struct Snapshot {
    pub image: Vec<u32>,
    pub words: Vec<u32>,
    pub ranges: Vec<SourceRange>,
    pub followed_chain: bool,
}

/// Cheap preflight over an already frozen root. No child reads or arena allocations occur
/// until there is a root CHAIN to investigate.
pub fn has_root_chain(words: &[u32]) -> bool {
    let mut at = 0;
    while at < words.len() {
        let Some(size) = packet_dwords(words[at]).filter(|&size| at + size <= words.len()) else { return false };
        if size == 4 && words[at] >> 8 & 0xff == pkt3::INDIRECT_BUFFER && words[at + 3] & CHAIN != 0 {
            return true;
        }
        at += size;
    }
    false
}

#[derive(Clone, Copy)]
struct Copy {
    address: u64,
    dwords: u32,
}

struct Builder<'a, R> {
    storage: &'a Bump,
    read: R,
    trace: u64,
    sequence: u32,
    words: Vec<u32>,
    ranges: Vec<SourceRange>,
    copies: BTreeMap<(u64, u32), Copy>,
    visiting: BTreeSet<(u64, u32)>,
    followed_chain: bool,
}

impl<R: FnMut(u64, usize) -> Result<Vec<u32>>> Builder<'_, R> {
    fn child(&mut self, address: u64, dwords: u32) -> Result<Copy> {
        let key = (address, dwords);
        if let Some(&copy) = self.copies.get(&key) {
            return Ok(copy);
        }
        if !self.visiting.insert(key) {
            return Err(Error::unsupported("children trace cannot snapshot a cyclic IB graph"));
        }
        let words = (self.read)(address, dwords as usize)?;
        if words.len() != dwords as usize {
            return Err(Error::invalid("children trace received a partial IB"));
        }
        let image = self.encode(address, &words)?;
        let copy = Copy { address: self.storage.put(&image, 256)?, dwords: image.len() as u32 };
        self.copies.insert(key, copy);
        self.visiting.remove(&key);
        Ok(copy)
    }

    fn encode(&mut self, address: u64, words: &[u32]) -> Result<Vec<u32>> {
        if self.ranges.len() >= MAX_NODES || words.is_empty() || self.words.len().saturating_add(words.len()) > MAX_SOURCE_DWORDS {
            return Err(Error::unsupported("children trace snapshot exceeds its diagnostic limit"));
        }
        // Validate packet boundaries before considering any counted control-flow range.
        let mut offsets = vec![0];
        while *offsets.last().expect("starts at zero") < words.len() {
            let at = *offsets.last().expect("starts at zero");
            let size = packet_dwords(words[at]).filter(|&size| at + size <= words.len())
                .ok_or_else(|| Error::invalid(format!("children trace: invalid packet at 0x{:x}", address + at as u64 * 4)))?;
            offsets.push(at + size);
        }
        let first = self.words.len();
        self.words.extend_from_slice(words);
        self.ranges.push(SourceRange { first, address, dwords: words.len() });
        let mut image = Vec::new();
        let mut guarded_to = 0;
        for pair in offsets.windows(2) {
            let (at, end) = (pair[0], pair[1]);
            let opcode = words[at] >> 8 & 0xff;
            if opcode == pkt3::PRED_EXEC {
                return Err(Error::unsupported("children trace does not rewrite PRED_EXEC"));
            }
            if opcode == pkt3::COND_EXEC {
                if end - at != 5 {
                    return Err(Error::invalid("children trace: malformed COND_EXEC"));
                }
                let stop = end + (words[at + 4] & 0x3fff) as usize;
                if offsets.binary_search(&stop).is_err() {
                    return Err(Error::unsupported("children trace: COND_EXEC must end at a packet boundary inside its IB"));
                }
                // Nested/overlapping guards protect their union. Never insert words inside it.
                guarded_to = guarded_to.max(stop);
            }
            if at >= guarded_to && opcode != pkt3::NOP {
                image.extend([header(pkt3::WRITE_DATA, 6, 0)?, (5 << 8) | (1 << 20), self.trace as u32,
                    (self.trace >> 32) as u32, (first + at) as u32, self.sequence]);
            }
            let output = image.len();
            image.extend_from_slice(&words[at..end]);
            if opcode == pkt3::INDIRECT_BUFFER {
                if end - at != 4 {
                    return Err(Error::unsupported("children trace does not rewrite COND_INDIRECT_BUFFER"));
                }
                let target = u64::from(words[at + 1] & !3) | (u64::from(words[at + 2] & 0xffff) << 32);
                let count = words[at + 3] & 0xf_ffff;
                if target == 0 || count == 0 {
                    return Err(Error::invalid("children trace: null or empty child IB"));
                }
                let child = self.child(target, count)?;
                image[output + 1] = child.address as u32 | (words[at + 1] & 3);
                image[output + 2] = (words[at + 2] & 0xffff_0000) | ((child.address >> 32) as u32 & 0xffff);
                image[output + 3] = (words[at + 3] & !0xf_ffff) | child.dwords;
                self.followed_chain |= words[at + 3] & CHAIN != 0;
            }
        }
        let padding = (IB_ALIGNMENT_DWORDS - image.len() % IB_ALIGNMENT_DWORDS) % IB_ALIGNMENT_DWORDS;
        image.extend(host_pm4::nop_fill(padding));
        if image.len() > MAX_IB_DWORDS.min(0xf_ffff) {
            return Err(Error::unsupported("children trace: instrumented IB is too large"));
        }
        Ok(image)
    }
}

pub fn snapshot(
    words: &[u32], address: u64, trace: u64, sequence: u64, storage: &Bump,
    read: impl FnMut(u64, usize) -> Result<Vec<u32>>,
) -> Result<Snapshot> {
    let sequence = u32::try_from(sequence).map_err(|_| Error::unsupported("children trace sequence exceeds its marker"))?;
    let mut builder = Builder { storage, read, trace, sequence, words: Vec::new(), ranges: Vec::new(), copies: BTreeMap::new(),
        visiting: BTreeSet::new(), followed_chain: false };
    let image = builder.encode(address, words)?;
    Ok(Snapshot { image, words: builder.words, ranges: builder.ranges, followed_chain: builder.followed_chain })
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/agc/trace.rs"]
mod tests;
