//! Same-VA guest memory (FINAL-REPORT §2-C): a GPU VA window reserved at the guest's own
//! addresses, and guest pages imported into it at those addresses.
//!
//! An import is made from a read-write alias view of the pagefile sections that back the guest
//! pages (AnyPS5's `GuestArenaMapAliasForImport`), never from the guest's own view: the guest view
//! is write-watched and re-protected while the KMD keeps the imported pages pinned. An import never
//! crosses an 8 MiB boundary of guest address space, so a guest unmap or remap only touches the
//! imports of the windows it covers.

use super::{Buffer, Device, VaReservation, GPU_PAGE_BYTES, Kmt, ErrorKind, D3DDDI_MAPGPUVIRTUALADDRESS, D3DKMTMapGpuVirtualAddress};
pub use crate::backend::va::{GUEST_PAGE_BYTES, IMPORT_WINDOW_BYTES};
use crate::{backend::va, Error, Result};
use std::{collections::BTreeMap, ffi::c_void, mem::size_of, rc::Rc};
use windows_sys::Win32::System::Memory::{
    VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_MAPPED, PAGE_EXECUTE_READWRITE,
    PAGE_READWRITE,
};

/// The reserved guest window and the guest ranges imported into it, by guest address.
pub struct GuestWindow {
    reservation: VaReservation,
    imports: BTreeMap<u64, Buffer>,
    backings: BTreeMap<u64, Buffer>,
    bindings: BTreeMap<u64, BackingBinding>,
}

#[derive(Clone, Copy)]
struct BackingBinding {
    bytes: u64,
    backing: u64,
    offset: u64,
}

impl Device {
    /// Reserves `[base, end)` at the guest's own GPU addresses. Call it right after
    /// [`Device::open`], before anything else maps GPU VA: dxgkrnl and the KMD place later
    /// mappings (fences, queues) in free VA, which could otherwise fall inside the window.
    /// `base` and `end` are 8 MiB aligned, at or above [`va::OS_LOW_END`] and at most
    /// [`va::GUEST_END`]; [`va::GUEST_WINDOW_BASE`] is the base to use.
    pub fn reserve_guest_window(&mut self, base: u64, end: u64) -> Result<GuestWindow> {
        check_window(base, end)?;
        Ok(GuestWindow {
            reservation: self.reserve_va(base, end - base)?,
            imports: BTreeMap::new(),
            backings: BTreeMap::new(),
            bindings: BTreeMap::new(),
        })
    }
}

/// `[base, end)` is a whole number of import windows between the OS's low range and the end of
/// the guest address space.
pub(super) fn check_window(base: u64, end: u64) -> Result<()> {
    if base < va::OS_LOW_END
        || end > va::GUEST_END
        || base >= end
        || base % IMPORT_WINDOW_BYTES != 0
        || end % IMPORT_WINDOW_BYTES != 0
    {
        return Err(Error::invalid(format!(
            "guest window 0x{base:x}..0x{end:x} is not whole 8 MiB windows inside 0x{:x}..0x{:x}",
            va::OS_LOW_END,
            va::GUEST_END
        )));
    }
    Ok(())
}

/// The import source is committed read-write section memory (mapped views) throughout, not
/// private memory and not a page the guest's write watching may protect.
pub(super) fn check_source(source: *const u8, bytes: u64) -> Result<()> {
    let start = source as usize;
    let end = usize::try_from(bytes)
        .ok()
        .and_then(|bytes| start.checked_add(bytes))
        .ok_or_else(|| Error::invalid("import source range overflows"))?;
    let mut at = start;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let written = unsafe {
            VirtualQuery(
                at as *const c_void,
                &mut info,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if written == 0 {
            return Err(Error::invalid(format!(
                "import source {source:p}: VirtualQuery failed at 0x{at:x}"
            )));
        }
        if info.State != MEM_COMMIT
            || info.Type != MEM_MAPPED
            || !matches!(info.Protect, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
        {
            return Err(Error::invalid(format!(
                "import source {source:p}+0x{bytes:x}: memory at {:p} is not committed read-write section memory (state 0x{:x}, type 0x{:x}, protection 0x{:x})",
                info.BaseAddress, info.State, info.Type, info.Protect
            )));
        }
        at = info.BaseAddress as usize + info.RegionSize;
    }
    Ok(())
}

impl GuestWindow {
    /// Imports stable physical pages once; subsequent guest bindings reuse this allocation.
    ///
    /// # Safety
    /// `source` remains committed read-write until `release_backing` succeeds, including every
    /// guest unmap/remap that happens while this backing is cached.
    pub unsafe fn import_backing(&mut self, device: &mut Device, id: u64, guest: u64, bytes: u64, source: *mut u8) -> Result<()> {
        if self.backings.contains_key(&id) {
            return Err(Error::invalid("physical backing identity is already imported"));
        }
        self.check_import(guest, bytes)?;
        check_source(source, bytes)?;
        // SAFETY: the caller retains the stable CPU alias for the backing's full lifetime.
        let buffer = unsafe { device.import_host_memory(source, bytes as usize, guest, &self.reservation) }?;
        self.backings.insert(id, buffer);
        self.bindings.insert(guest, BackingBinding { bytes, backing: id, offset: 0 });
        Ok(())
    }

    /// Maps a subrange of an already resident physical backing, without creating an allocation.
    /// Paging is batched until `Device::settle_paging` before the next GPU submission.
    pub fn bind_backing(&mut self, device: &mut Device, guest: u64, bytes: u64, id: u64, offset: u64) -> Result<()> {
        let buffer = self.backings.get(&id).ok_or_else(|| Error::invalid("unknown physical backing"))?;
        if offset % GUEST_PAGE_BYTES != 0 || offset.checked_add(bytes).is_none_or(|end| end > buffer.size() as u64) {
            return Err(Error::invalid("guest binding exceeds its physical backing"));
        }
        if self.bindings.get(&guest).is_some_and(|binding| binding.bytes == bytes && binding.backing == id && binding.offset == offset) {
            return Ok(());
        }
        self.check_import(guest, bytes)?;
        let handle = buffer.allocation.handle;
        self.map_pages(device, guest, bytes, handle, offset, super::GPU_VA_ALLOW_WRITE)?;
        self.bindings.insert(guest, BackingBinding { bytes, backing: id, offset });
        Ok(())
    }

    fn map_pages(&self, device: &mut Device, guest: u64, bytes: u64, handle: u32, offset: u64, protection: u64) -> Result<()> {
        if !Rc::ptr_eq(&self.reservation.inner.session, &device.session) {
            return Err(Error::invalid("guest binding belongs to another device"));
        }
        device.session.healthy()?;
        let mut map = D3DDDI_MAPGPUVIRTUALADDRESS {
            hPagingQueue: device.session.paging,
            BaseAddress: guest,
            hAllocation: handle,
            OffsetInPages: offset / GPU_PAGE_BYTES as u64,
            SizeInPages: bytes / GPU_PAGE_BYTES as u64,
            ..Default::default()
        };
        map.Protection.Anonymous.Value = protection;
        Kmt::status(unsafe { D3DKMTMapGpuVirtualAddress(&mut map) }, "MapGpuVirtualAddress(guest binding)")?;
        if map.VirtualAddress != guest {
            device.session.poison("guest binding changed virtual address");
            return Err(Error::new(ErrorKind::DeviceLost, "guest binding changed virtual address"));
        }
        device.pending_paging = device.pending_paging.max(map.PagingFenceValue);
        Ok(())
    }

    /// Retires guest mappings while retaining their physical allocations and CPU aliases.
    pub fn release_bindings(&mut self, device: &mut Device, guest: u64, bytes: u64) -> Result<()> {
        let end = guest.checked_add(bytes).ok_or_else(|| Error::invalid("guest binding release overflows"))?;
        if bytes == 0 || guest % GUEST_PAGE_BYTES != 0 || bytes % GUEST_PAGE_BYTES != 0 || guest < self.base() || end > self.end() {
            return Err(Error::invalid("guest binding release is outside whole guest pages"));
        }
        if !Rc::ptr_eq(&self.reservation.inner.session, &device.session) {
            return Err(Error::invalid("guest binding belongs to another device"));
        }
        if self.imports.range(..end).any(|(&start, buffer)| start + buffer.size() as u64 > guest) {
            return Err(Error::invalid("guest binding release overlaps a legacy import"));
        }
        let bindings: Vec<_> = self.bindings.range(..end).filter(|(start, binding)| **start + binding.bytes > guest).map(|(&start, binding)| (start, *binding)).collect();
        for (start, binding) in &bindings {
            if *start < guest || *start + binding.bytes > end {
                return Err(Error::invalid("guest binding release cuts a live mapping"));
            }
            self.backings[&binding.backing].allocation.await_gpu()?;
        }
        device.poll();
        if !bindings.is_empty() {
            const NO_ACCESS: u64 = 1 << 3;
            self.map_pages(device, guest, bytes, 0, 0, NO_ACCESS)?;
            device.settle_paging()?;
            for (start, _) in bindings { self.bindings.remove(&start); }
        }
        Ok(())
    }

    /// A submission lifetime target. Its address is the initial mapping, not a current binding.
    pub fn backing(&self, id: u64) -> Option<&Buffer> { self.backings.get(&id) }

    pub fn release_backing(&mut self, device: &mut Device, id: u64) -> Result<()> {
        if !Rc::ptr_eq(&self.reservation.inner.session, &device.session) {
            return Err(Error::invalid("physical backing belongs to another device"));
        }
        device.session.healthy()?;
        if self.bindings.values().any(|binding| binding.backing == id) {
            return Err(Error::invalid("physical backing still has guest bindings"));
        }
        let buffer = self.backings.get(&id).ok_or_else(|| Error::invalid("unknown physical backing"))?;
        buffer.allocation.await_gpu()?;
        device.poll();
        if Rc::strong_count(&buffer.allocation) != 1 {
            return Err(Error::invalid("physical backing is still held after its uses retired"));
        }
        self.backings.remove(&id);
        device.session.healthy()
    }

    pub(super) fn sparse_reservation(&self) -> &VaReservation {
        &self.reservation
    }
    pub fn base(&self) -> u64 {
        self.reservation.base()
    }

    pub fn end(&self) -> u64 {
        self.reservation.base() + self.reservation.size()
    }

    /// Imports the guest pages `[guest, guest + bytes)` at their guest addresses from `source`,
    /// the read-write alias view of those pages, and makes them resident. The range is whole
    /// guest pages inside one 8 MiB window and overlaps no other import. Imported memory counts
    /// against the device's local budget ([`Device::video_memory`]).
    ///
    /// # Safety
    ///
    /// `source` must map exactly the pages of `[guest, guest + bytes)` and stay mapped read-write
    /// until the import is released or the window dropped, and every submission that used the
    /// import has retired.
    pub unsafe fn import(
        &mut self,
        device: &mut Device,
        guest: u64,
        bytes: u64,
        source: *mut u8,
    ) -> Result<&Buffer> {
        self.check_import(guest, bytes)?;
        check_source(source, bytes)?;
        let buffer =
            unsafe { device.import_host_memory(source, bytes as usize, guest, &self.reservation) }?;
        Ok(self.imports.entry(guest).or_insert(buffer))
    }

    /// Whole guest pages inside one import window of this guest window, overlapping no import.
    pub(super) fn check_import(&self, guest: u64, bytes: u64) -> Result<()> {
        let window = guest & !(IMPORT_WINDOW_BYTES - 1);
        let end = guest.saturating_add(bytes);
        if bytes == 0
            || guest % GUEST_PAGE_BYTES != 0
            || bytes % GUEST_PAGE_BYTES != 0
            || end > window + IMPORT_WINDOW_BYTES
            || guest < self.base()
            || end > self.end()
        {
            return Err(Error::invalid(format!(
                "guest import 0x{guest:x}+0x{bytes:x} is not whole 16 KiB pages inside one 8 MiB window of 0x{:x}..0x{:x}",
                self.base(),
                self.end()
            )));
        }
        if let Some((&start, import)) = self.imports.range(..end).next_back() {
            if start + import.size() as u64 > guest {
                return Err(Error::invalid(format!(
                    "guest import 0x{guest:x}+0x{bytes:x} overlaps the import at 0x{start:x}+0x{:x}",
                    import.size()
                )));
            }
        }
        if self.bindings.range(..end).next_back().is_some_and(|(&start, binding)| start + binding.bytes > guest) {
            return Err(Error::invalid("guest import overlaps an existing physical binding"));
        }
        Ok(())
    }

    /// Releases the import that starts at `guest` once no submission uses it. Its GPU range
    /// goes back to the window's reservation, so those guest pages can be imported again.
    pub fn release(&mut self, device: &mut Device, guest: u64) -> Result<()> {
        let Some(import) = self.imports.get(&guest) else {
            return Err(Error::invalid(format!(
                "no guest import starts at 0x{guest:x}"
            )));
        };
        import.allocation.await_gpu()?;
        // Retired submissions drop their hold on the import, so removing it destroys it.
        device.poll();
        if Rc::strong_count(&import.allocation) != 1 {
            return Err(Error::invalid(format!(
                "the guest import at 0x{guest:x} is still held after its submissions retired"
            )));
        }
        self.imports.remove(&guest);
        Ok(())
    }

    /// Releases every import that overlaps `[guest, guest + bytes)`: the guest unmapped or
    /// remapped memory there. Returns how many were released.
    pub fn release_range(&mut self, device: &mut Device, guest: u64, bytes: u64) -> Result<usize> {
        let end = guest.saturating_add(bytes);
        let starts: Vec<_> = self.imports.range(..end).filter(|(start, import)| **start + import.size() as u64 > guest).map(|(&start, _)| start).collect();
        for &start in &starts {
            self.release(device, start)?;
        }
        let bindings: Vec<_> = self.bindings.range(..end).filter(|(start, binding)| **start + binding.bytes > guest).map(|(&start, binding)| (start, binding.bytes)).collect();
        for &(start, size) in &bindings { self.release_bindings(device, start, size)?; }
        Ok(starts.len() + bindings.len())
    }

    /// The guest addresses of the imports that overlap `[guest, guest + bytes)`.
    pub(super) fn overlapping(&self, guest: u64, bytes: u64) -> Vec<u64> {
        let end = guest.saturating_add(bytes);
        self.imports
            .range(..end)
            .filter(|(&start, import)| start + import.size() as u64 > guest)
            .map(|(&start, _)| start)
            .chain(self.bindings.range(..end).filter(|(start, binding)| **start + binding.bytes > guest).map(|(&start, _)| start))
            .collect()
    }

    /// The import that starts at `guest`.
    pub fn import_at(&self, guest: u64) -> Option<&Buffer> {
        self.imports.get(&guest)
    }

    /// The import that covers `[address, address + bytes)`, to pass as a submission target.
    pub fn covering(&self, address: u64, bytes: u64) -> Option<&Buffer> {
        let (&start, import) = self.imports.range(..=address).next_back()?;
        let end = (address - start).checked_add(bytes)?;
        (end <= import.size() as u64).then_some(import)
    }

    pub fn imports(&self) -> impl Iterator<Item = &Buffer> {
        self.imports.values()
    }

    pub fn imported_bytes(&self) -> u64 {
        self.imports
            .values()
            .map(|import| import.size() as u64)
            .sum()
    }
}

impl Drop for GuestWindow {
    /// Waits until no submission uses an import, frees the whole window, which unmaps every
    /// import in it, then destroys the imports. The alias views can be unmapped once this
    /// returns. If the GPU cannot be confirmed idle the device is quarantined and keeps them.
    fn drop(&mut self) {
        if self
            .imports
            .values()
            .chain(self.backings.values())
            .any(|import| import.allocation.await_gpu().is_err())
        {
            return;
        }
        let _ = self.reservation.inner.free_now();
    }
}
