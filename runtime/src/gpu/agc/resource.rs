//! Proven GPU fills, in execution order. Descriptor candidates never become clear evidence.

use super::surface::Fill;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct ProvenFill {
    pub address: u64,
    pub bytes: u64,
    pub fill: Fill,
    pub flip: u64,
    pub conditional: bool,
}

#[derive(Default)]
pub struct MetadataWrites {
    fills: BTreeMap<u64, ProvenFill>,
}

impl MetadataWrites {
    pub fn is_empty(&self) -> bool { self.fills.is_empty() }
    pub fn invalidate(&mut self, address: u64, bytes: u64) {
        let end = address.saturating_add(bytes);
        if address == end { return; }
        let prior = self.fills.range(..=address).next_back().map(|(&at, _)| at);
        let mut keys: Vec<u64> = self.fills.range(address..end).map(|(&at, _)| at).collect();
        if let Some(at) = prior.filter(|at| *at < address) { keys.push(at); }
        for at in keys {
            let fill = self.fills[&at];
            let old_end = fill.address.saturating_add(fill.bytes);
            if old_end <= address { continue; }
            self.fills.remove(&at);
            let left = (address - at) & !3;
            let right = at.saturating_add((end - at).div_ceil(4) * 4);
            if left != 0 { self.fills.insert(at, ProvenFill { bytes: left, ..fill }); }
            if old_end > right { self.fills.insert(right, ProvenFill { address: right, bytes: old_end - right, ..fill }); }
        }
    }

    pub fn record(&mut self, fill: ProvenFill) {
        self.invalidate(fill.address, fill.bytes);
        self.fills.insert(fill.address, fill);
    }

    pub fn covering(&self, address: u64, bytes: u64) -> Option<ProvenFill> {
        let (_, fill) = self.fills.range(..=address).next_back()?;
        (bytes != 0 && fill.address.saturating_add(fill.bytes) >= address.checked_add(bytes)?).then_some(*fill)
    }

    pub fn expire(&mut self, flip: u64) {
        self.fills.retain(|_, fill| flip <= fill.flip.saturating_add(1));
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/agc/resource.rs"]
mod tests;
