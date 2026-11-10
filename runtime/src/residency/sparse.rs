//! 64-KiB sparse image geometry and virtual-tile addressing. Mapping is independent of texel
//! content: a zero-filled resident tile remains resident and must not be mistaken for a hole.
//! The geometry follows RADV sparse image binding (`radv_queue.c`) and addrlib's PRT blocks.

use super::{GuestRange, GuestResidency, Origin, ResidencyDevice, ResidencyWindow};
use crate::{amd::surface::tiling::SurfaceLayout, Error, Result};
use std::collections::{BTreeMap, BTreeSet};

pub const TILE_BYTES: u64 = 64 * 1024;

/// Logical PRT registration is separate from physical imports. A hole remains owned and
/// addressable, while its PTE does not acquire a resident allocation merely because it reads 0.
#[derive(Default, Debug)]
pub struct SparseRegistry {
    ranges: BTreeMap<u64, u64>,
}

impl SparseRegistry {
    pub fn ranges(&self) -> Vec<GuestRange> {
        self.ranges
            .iter()
            .map(|(&address, &end)| GuestRange::new(address, end - address))
            .collect()
    }
    pub fn contains(&self, range: GuestRange) -> bool {
        range.bytes != 0
            && self
                .ranges
                .range(..=range.address)
                .next_back()
                .is_some_and(|(_, &end)| {
                    range
                        .address
                        .checked_add(range.bytes)
                        .is_some_and(|last| last <= end)
                })
    }
    pub fn overlaps(&self, range: GuestRange) -> Vec<GuestRange> {
        self.ranges()
            .into_iter()
            .filter_map(|registered| registered.intersection(range))
            .collect()
    }
    pub fn unregistered(&self, range: GuestRange) -> Vec<GuestRange> {
        let mut result = Vec::new();
        let mut at = range.address;
        for held in self.overlaps(range) {
            if at < held.address {
                result.push(GuestRange::new(at, held.address - at));
            }
            at = held.end();
        }
        if at < range.end() {
            result.push(GuestRange::new(at, range.end() - at));
        }
        result
    }
    pub fn register(&mut self, range: GuestRange) {
        let mut begin = range.address;
        let mut end = range.end();
        let merged: Vec<_> = self
            .ranges
            .iter()
            .filter(|(&start, &last)| start <= end && last >= begin)
            .map(|(&start, &last)| (start, last))
            .collect();
        for (start, last) in merged {
            self.ranges.remove(&start);
            begin = begin.min(start);
            end = end.max(last);
        }
        self.ranges.insert(begin, end);
    }
    pub fn unregister(&mut self, range: GuestRange) {
        let old = self.ranges();
        self.ranges.clear();
        for held in old {
            if held.address >= range.end() || held.end() <= range.address {
                self.ranges.insert(held.address, held.end());
                continue;
            }
            if held.address < range.address {
                self.ranges.insert(held.address, range.address);
            }
            if held.end() > range.end() {
                self.ranges.insert(range.end(), held.end());
            }
        }
    }
}

impl<W: ResidencyWindow> GuestResidency<W> {
    /// Checks allocation uses without waiting, so the device thread can keep accepting a producer
    /// whose GPU signal would retire an older sparse mapping before applying a remap.
    pub fn range_idle(&self, device: &mut W::Device, range: GuestRange) -> Result<bool> {
        self.check_inside(range)?;
        let mut completed = BTreeMap::new();
        for start in self.overlapping(range) {
            for (&queue, &serial) in &self.imports[&start].last {
                let done = *completed
                    .entry(queue)
                    .or_insert_with(|| super::ResidencyDevice::completed(device, queue));
                if serial > done {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    pub fn require_range_idle(&self, device: &mut W::Device, range: GuestRange) -> Result<()> {
        if !self.range_idle(device, range)? {
            return Err(Error::new(
                crate::ErrorKind::Busy,
                format!("guest mapping {range} is busy on an unfinished GPU queue"),
            ));
        }
        Ok(())
    }
    fn check_sparse_range(&self, range: GuestRange) -> Result<()> {
        self.check_inside(range)?;
        if range.bytes == 0
            || range.address % super::GUEST_PAGE_BYTES != 0
            || range.bytes % super::GUEST_PAGE_BYTES != 0
        {
            return Err(Error::invalid(
                "PRT mapping needs nonempty whole 16 KiB guest pages",
            ));
        }
        Ok(())
    }
    pub fn sparse_reserve(
        &mut self,
        device: &mut W::Device,
        range: GuestRange,
        registered: bool,
    ) -> Result<()> {
        self.check_sparse_range(range)?;
        if !registered && !self.sparse.contains(range) {
            return Err(Error::invalid(
                "PRT unregister is outside a registered sparse range",
            ));
        }
        if registered && self.sparse.contains(range) {
            return Ok(());
        }
        self.require_range_idle(device, range)?;
        for start in self.overlapping(range) {
            self.release_import(device, start)?;
        }
        self.window.sparse_zero(device, range)?;
        if registered {
            self.sparse.register(range);
        } else {
            self.sparse.unregister(range);
        }
        Ok(())
    }
    pub fn sparse_bind(&mut self, device: &mut W::Device, range: GuestRange) -> Result<()> {
        self.check_sparse_range(range)?;
        if !self.sparse.contains(range) {
            return Err(Error::invalid(
                "PRT bind is outside a registered sparse range",
            ));
        }
        for address in (range.address..range.end()).step_by(super::GUEST_PAGE_BYTES as usize) {
            if self.imports.contains_key(&address) {
                continue;
            }
            let page = GuestRange::new(address, super::GUEST_PAGE_BYTES);
            // No resident sparse page is evicted to satisfy a subsequent binding. Physical
            // residency changes only through bind/unbind or the guest mapping release service.
            self.make_import(device, page, Origin::Named(range), &BTreeSet::new())?;
        }
        device.settle_paging()
    }
    pub fn sparse_unbind(&mut self, device: &mut W::Device, range: GuestRange) -> Result<()> {
        self.check_sparse_range(range)?;
        if !self.sparse.contains(range) {
            return Err(Error::invalid(
                "PRT unbind is outside a registered sparse range",
            ));
        }
        self.require_range_idle(device, range)?;
        for start in self.overlapping(range) {
            self.release_import(device, start)?;
        }
        self.window.sparse_zero(device, range)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseRange {
    pub base: u64,
    pub bytes: u64,
}

impl SparseRange {
    pub fn new(base: u64, bytes: u64) -> Result<Self> {
        if base == 0
            || base % TILE_BYTES != 0
            || bytes == 0
            || bytes % TILE_BYTES != 0
            || base.checked_add(bytes).is_none()
        {
            return Err(Error::invalid(
                "sparse VA ranges need a nonempty, nonoverflowing 64-KiB-aligned extent",
            ));
        }
        Ok(Self { base, bytes })
    }
    pub fn tile_count(self) -> u64 {
        self.bytes / TILE_BYTES
    }
    pub fn tile_address(self, tile: u64) -> Result<u64> {
        if tile >= self.tile_count() {
            return Err(Error::invalid("sparse tile is outside the reserved range"));
        }
        Ok(self.base + tile * TILE_BYTES)
    }
}

/// Coordinate-based binding of one thin 2D mip level (or an array of that level). Mip tails
/// must use opaque VA tile bindings; reinterpreting a tail as independently bindable mips is
/// incorrect because several mip levels share physical pages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseImageTiles {
    pub tile_width: u32,
    pub tile_height: u32,
    pub tiles_per_row: u32,
    pub tiles_per_column: u32,
    pub tiles_per_layer: u64,
    pub layers: u32,
}

impl SparseImageTiles {
    pub fn new(layout: &SurfaceLayout) -> Result<Self> {
        let desc = layout.desc();
        if layout.is_linear()
            || layout.is_thick()
            || desc.volume
            || desc.depth > 1
            || layout.block_bytes() != TILE_BYTES
            || desc.mip_levels.max(1) != 1
        {
            return Err(Error::unsupported("coordinate PRT bindings need one thin 2D 64-KiB mip; use opaque VA tiles for mip tails"));
        }
        let [tile_width, tile_height, _] = layout.block_extent();
        let tiles_per_row = desc.width.div_ceil(tile_width);
        let tiles_per_column = desc.height.div_ceil(tile_height);
        let tiles_per_layer = layout.slice_stride() / TILE_BYTES;
        if tiles_per_layer != u64::from(tiles_per_row) * u64::from(tiles_per_column) {
            return Err(Error::unsupported(
                "the sparse surface has a nonrectangular tile grid",
            ));
        }
        Ok(Self {
            tile_width,
            tile_height,
            tiles_per_row,
            tiles_per_column,
            tiles_per_layer,
            layers: desc.layers.max(1),
        })
    }
    pub fn tile(self, x: u32, y: u32, layer: u32) -> Result<u64> {
        if x >= self.tiles_per_row || y >= self.tiles_per_column || layer >= self.layers {
            return Err(Error::invalid(
                "sparse image tile coordinates are outside the image",
            ));
        }
        Ok(u64::from(layer) * self.tiles_per_layer
            + u64::from(y) * u64::from(self.tiles_per_row)
            + u64::from(x))
    }
}
