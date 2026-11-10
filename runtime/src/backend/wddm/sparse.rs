//! Sparse virtual tiles through WDDM's Zero reservation and explicit map-to-Zero operations.
//! Zero PTEs guarantee reads of zero and discarded writes per d3dukmdt.h. That guarantee alone
//! does not establish AMD image TFE/LWE feedback; the sparse GPU probe checks that separately.

use super::*;
use crate::residency::sparse::{SparseRange, TILE_BYTES};
use std::collections::BTreeMap;

impl GuestWindow {
    /// Restores explicitly registered guest PRT pages to nonresident Zero PTEs inside the
    /// already reserved guest window. This does not reserve a second, overlapping VA range.
    pub fn sparse_zero(&self, device: &mut Device, address: u64, bytes: u64) -> Result<()> {
        if bytes == 0
            || address % GUEST_PAGE_BYTES != 0
            || bytes % GUEST_PAGE_BYTES != 0
            || address < self.base()
            || address
                .checked_add(bytes)
                .is_none_or(|end| end > self.end())
        {
            return Err(Error::invalid(
                "sparse Zero mapping is outside whole guest pages",
            ));
        }
        if !Rc::ptr_eq(&self.sparse_reservation().inner.session, &device.session) {
            return Err(Error::invalid(
                "sparse guest window belongs to another device",
            ));
        }
        device.session.healthy()?;
        if !self.overlapping(address, bytes).is_empty() {
            return Err(Error::invalid(
                "sparse Zero mapping overlaps a live guest import",
            ));
        }
        let mut map = D3DDDI_MAPGPUVIRTUALADDRESS {
            hPagingQueue: device.session.paging,
            BaseAddress: address,
            SizeInPages: bytes / GPU_PAGE_BYTES as u64,
            ..Default::default()
        };
        map.Protection.Anonymous.Value = 1 << 2;
        Kmt::status(
            unsafe { D3DKMTMapGpuVirtualAddress(&mut map) },
            "MapGpuVirtualAddress(guest sparse Zero)",
        )?;
        if map.VirtualAddress != address {
            device
                .session
                .poison("guest sparse Zero mapping changed virtual address");
            return Err(Error::new(
                ErrorKind::DeviceLost,
                "guest sparse Zero mapping changed virtual address",
            ));
        }
        device
            .session
            .wait_paging(map.PagingFenceValue)
    }
}

/// A reserved virtual-tile range with physical backing only for its resident tiles. Each bind
/// owns a separate allocation (or a pinned host alias), so unbinding can never free a neighbor.
/// Bind changes wait for uses of this range; ordinary submissions remain asynchronous.
pub struct SparseTiles {
    range: SparseRange,
    page_bytes: u64,
    reservation: VaReservation,
    resident: BTreeMap<u64, Buffer>,
    uses: BTreeMap<usize, (MonitoredFence, u64)>,
    generation: u64,
}

impl SparseTiles {
    pub fn new(device: &mut Device, base: u64, bytes: u64) -> Result<Self> {
        Self::new_paged(device, base, bytes, TILE_BYTES)
    }
    /// Uses 16-KiB guest pages or 64-KiB image tiles in one 64-KiB-aligned reservation.
    pub fn new_paged(device: &mut Device, base: u64, bytes: u64, page_bytes: u64) -> Result<Self> {
        if !matches!(page_bytes, GUEST_PAGE_BYTES | TILE_BYTES) {
            return Err(Error::invalid(
                "sparse bindings support 16-KiB guest pages or 64-KiB image tiles",
            ));
        }
        let range = SparseRange::new(base, bytes)?;
        let reservation = device.reserve_va_zero(base, bytes)?;
        Ok(Self {
            range,
            page_bytes,
            reservation,
            resident: BTreeMap::new(),
            uses: BTreeMap::new(),
            generation: 1,
        })
    }
    pub fn range(&self) -> SparseRange {
        self.range
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn is_resident(&self, tile: u64) -> Result<bool> {
        self.page_address(tile)?;
        Ok(self.resident.contains_key(&tile))
    }
    fn page_address(&self, page: u64) -> Result<u64> {
        if page >= self.range.bytes / self.page_bytes {
            return Err(Error::invalid("sparse page is outside the reserved range"));
        }
        Ok(self.range.base + page * self.page_bytes)
    }
    pub fn resident_tiles(&self) -> impl Iterator<Item = (u64, &Buffer)> {
        self.resident.iter().map(|(&tile, buffer)| (tile, buffer))
    }
    fn check_device(&self, device: &Device) -> Result<()> {
        if !Rc::ptr_eq(&self.reservation.inner.session, &device.session) {
            return Err(Error::invalid("sparse VA range belongs to another device"));
        }
        device.session.healthy()
    }
    fn wait_uses(&mut self) -> Result<()> {
        for (fence, serial) in self.uses.values() {
            self.reservation.inner.session.wait(fence, *serial)?;
        }
        self.uses.clear();
        Ok(())
    }
    fn changed(&mut self) -> Result<()> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::invalid("sparse binding generation exhausted"))?;
        Ok(())
    }

    /// Installs one fully populated tile. The upload's content has no bearing on residency:
    /// even a tile of all zero bytes is represented by a real allocation and valid PTEs.
    pub fn bind_bytes(&mut self, device: &mut Device, tile: u64, bytes: &[u8]) -> Result<()> {
        if self.page_bytes != TILE_BYTES {
            return Err(Error::unsupported("16-KiB PRT backing must import actual guest host pages; device upload allocations have 64-KiB alignment"));
        }
        self.check_device(device)?;
        let address = self.page_address(tile)?;
        if bytes.len() != self.page_bytes as usize {
            return Err(Error::invalid(
                "a sparse page upload must fill the configured binding granule",
            ));
        }
        self.wait_uses()?;
        self.unbind(device, tile)?;
        let mut buffer = device.allocate_at(
            self.page_bytes as usize,
            MemoryKind::Data,
            address,
            &self.reservation,
        )?;
        buffer.write(0, bytes)?;
        self.resident.insert(tile, buffer);
        self.changed()
    }

    /// Binds the guest's actual committed pages without a copy.
    ///
    /// # Safety
    /// The whole 64-KiB host range must satisfy [`Device::import_host_memory`]'s lifetime and
    /// protection contract until this tile has been unbound or this object and its uses retire.
    pub unsafe fn bind_host_memory(
        &mut self,
        device: &mut Device,
        tile: u64,
        memory: *mut u8,
    ) -> Result<()> {
        self.check_device(device)?;
        let address = self.page_address(tile)?;
        self.wait_uses()?;
        self.unbind(device, tile)?;
        // SAFETY: caller supplies the pinned host-memory lifetime contract above.
        let buffer = unsafe {
            device.import_host_memory(
                memory,
                self.page_bytes as usize,
                address,
                &self.reservation,
            )?
        };
        self.resident.insert(tile, buffer);
        self.changed()
    }

    /// Returns a virtual tile to the Zero PTE state before dropping its physical allocation.
    /// The returned paging fence is retired before any subsequent submission can see it.
    pub fn unbind(&mut self, device: &mut Device, tile: u64) -> Result<()> {
        self.check_device(device)?;
        let address = self.page_address(tile)?;
        self.wait_uses()?;
        if !self.resident.contains_key(&tile) {
            return Ok(());
        }
        // The allocation also records submissions made through Device directly. Retire those
        // uses before changing its virtual mapping, not just submissions through submit_on.
        self.resident[&tile].allocation.await_gpu()?;
        let mut map = D3DDDI_MAPGPUVIRTUALADDRESS {
            hPagingQueue: device.session.paging,
            BaseAddress: address,
            SizeInPages: self.page_bytes / GPU_PAGE_BYTES as u64,
            ..Default::default()
        };
        const GPU_VA_ZERO: u64 = 1 << 2;
        map.Protection.Anonymous.Value = GPU_VA_ZERO;
        Kmt::status(
            unsafe { D3DKMTMapGpuVirtualAddress(&mut map) },
            "MapGpuVirtualAddress(sparse Zero)",
        )?;
        if map.VirtualAddress != address {
            device
                .session
                .poison("a sparse Zero mapping was placed at another virtual address");
            return Err(Error::new(
                ErrorKind::DeviceLost,
                "sparse Zero mapping changed virtual address",
            ));
        }
        device
            .session
            .wait_paging(map.PagingFenceValue)?;
        self.resident.remove(&tile);
        self.changed()
    }

    /// Queues work using this sparse range and additional owned buffers, without waiting.
    /// Tracks the queue's fence even for an all-hole resource, whose reservation otherwise
    /// has no resident allocation keeping its virtual addresses alive.
    pub fn submit_on(
        &mut self,
        device: &mut Device,
        queue: QueueKind,
        words: &[u32],
        other: &[&Buffer],
    ) -> Result<u64> {
        self.check_device(device)?;
        let mut targets: Vec<&Buffer> = self.resident.values().collect();
        targets.extend_from_slice(other);
        let serial = device.queue_host_on(queue, words, &targets, SubmitLayout::Single)?;
        let fence = device.hw_queue(queue)?.progress_fence;
        self.uses.insert(queue.index(), (fence, serial));
        Ok(serial)
    }
}

impl Drop for SparseTiles {
    fn drop(&mut self) {
        if self.wait_uses().is_err() {
            self.reservation
                .inner
                .session
                .poison("sparse VA range completion was not confirmed at close");
        }
    }
}
