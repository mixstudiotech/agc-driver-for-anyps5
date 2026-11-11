#![cfg(all(windows, feature = "wddm"))]

use agcdrv as runtime;
use runtime::amd::defs::cp_gfx11::pfp::COPY_DATA as Copy;
use runtime::backend::{host_pm4, va, wddm::{Buffer, Device, GuestWindow, MemoryKind, QueueKind}};
use runtime::gpu::packets::{gcr, PacketStream};
use runtime::residency::{GuestMemory, GuestRange, GuestResidency, ImportRefusal, Reach, ResidencyDevice, ResidencyOptions, Submission, WddmWindow, COPY_GRANULE_BYTES};
use std::{collections::BTreeMap, ffi::c_void, ptr::{self, NonNull}, time::{Duration, Instant}};
use windows_sys::Win32::{Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE}, System::Memory::{CreateFileMappingW, MapViewOfFile, UnmapViewOfFile, FILE_MAP_ALL_ACCESS, PAGE_READWRITE, MEMORY_MAPPED_VIEW_ADDRESS}};

const PAGE: u64 = va::GUEST_PAGE_BYTES;
static GPU_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct SectionView { section: HANDLE, view: MEMORY_MAPPED_VIEW_ADDRESS }

impl SectionView {
    fn new() -> Self {
        unsafe {
            let section = CreateFileMappingW(INVALID_HANDLE_VALUE, ptr::null(), PAGE_READWRITE, 0, (PAGE * 4) as u32, ptr::null());
            assert!(!section.is_null(), "CreateFileMappingW: {}", std::io::Error::last_os_error());
            let view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, (PAGE * 4) as usize);
            assert!(!view.Value.is_null(), "MapViewOfFile: {}", std::io::Error::last_os_error());
            for page in 0..4 { (view.Value.cast::<u8>().add(page * PAGE as usize) as *mut u32).write(0x1200_0000 + page as u32); }
            Self { section, view }
        }
    }
}

impl Drop for SectionView {
    fn drop(&mut self) { unsafe { UnmapViewOfFile(self.view); CloseHandle(self.section); } }
}

fn read_pages(device: &mut Device, window: &GuestWindow, guest: u64, pages: usize) -> runtime::Result<Vec<u32>> {
    let addresses: Vec<_> = (0..pages).map(|index| guest + index as u64 * PAGE).collect();
    read_words(device, &[window.backing(1).unwrap()], &addresses)
}

fn read_words(device: &mut Device, targets: &[&Buffer], addresses: &[u64]) -> runtime::Result<Vec<u32>> {
    let output = device.allocate_kind(65536, MemoryKind::Readback)?;
    let stream = copy_words(&output, addresses)?;
    let mut targets = targets.to_vec();
    targets.push(&output);
    let serial = device.submit_host(stream.words(), &targets)?;
    device.wait(serial)?;
    words_of(&output, addresses.len())
}

fn words_of(output: &Buffer, count: usize) -> runtime::Result<Vec<u32>> {
    Ok(output.read()?[..count * 4].chunks_exact(4).map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())).collect())
}

/// Copies the dword at each of `addresses` into consecutive dwords of `output`.
fn copy_words(output: &Buffer, addresses: &[u64]) -> runtime::Result<PacketStream> {
    let pairs: Vec<_> = addresses.iter().enumerate().map(|(index, &source)| (source, output.address() + index as u64 * 4)).collect();
    copy_dwords(&pairs)
}

/// Copies the dword at the first address of each pair to the second.
fn copy_dwords(pairs: &[(u64, u64)]) -> runtime::Result<PacketStream> {
    let mut stream = PacketStream::new();
    for &(source, destination) in pairs {
        let mut packet = [0; Copy::WORDS as usize];
        Copy::SRC_SEL.set(&mut packet, Copy::enums::src_sel::TC_L2);
        Copy::DST_SEL.set(&mut packet, Copy::enums::dst_sel::TC_L2);
        Copy::WR_CONFIRM.set(&mut packet, 1);
        packet[2] = source as u32;
        packet[3] = (source >> 32) as u32;
        packet[4] = destination as u32;
        packet[5] = (destination >> 32) as u32;
        stream.raw_packet(Copy::OPCODE, 0, &packet[1..])?;
    }
    stream.acquire_mem_global(gcr::GL2_WB | gcr::GL2_INV)?;
    Ok(stream)
}

struct Alias {
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    region: usize,
    offset: u64,
}

struct SectionMemory {
    sections: [SectionView; 3],
    aliases: BTreeMap<usize, Alias>,
}

impl SectionMemory {
    fn new() -> Self { Self { sections: std::array::from_fn(|_| SectionView::new()), aliases: BTreeMap::new() } }

    fn map(&mut self, guest: u64, bytes: u64) -> Option<NonNull<c_void>> {
        let address = guest.checked_sub(va::GUEST_WINDOW_BASE)?;
        let region = (address / va::IMPORT_WINDOW_BYTES) as usize;
        let offset = address % va::IMPORT_WINDOW_BYTES;
        if region >= self.sections.len() || bytes == 0 || offset + bytes > PAGE * 4 { return None; }
        let view = unsafe { MapViewOfFile(self.sections[region].section, FILE_MAP_ALL_ACCESS, 0, 0, (PAGE * 4) as usize) };
        let base = NonNull::new(view.Value)?;
        let alias = unsafe { NonNull::new_unchecked(base.as_ptr().cast::<u8>().add(offset as usize).cast()) };
        self.aliases.insert(alias.as_ptr() as usize, Alias { view, region, offset });
        Some(alias)
    }

    fn unmap(&mut self, alias: NonNull<c_void>) {
        let alias = self.aliases.remove(&(alias.as_ptr() as usize)).unwrap();
        assert_ne!(unsafe { UnmapViewOfFile(alias.view) }, 0);
    }
}

impl GuestMemory for SectionMemory {
    fn mapping_generation(&self) -> u64 { 1 }
    fn map_import(&mut self, guest: u64, bytes: u64) -> Result<NonNull<c_void>, ImportRefusal> { self.map(guest, bytes).ok_or(ImportRefusal::NotGpuVisible) }
    fn unmap_import(&mut self, alias: NonNull<c_void>) { self.unmap(alias); }
    fn map_backing(&mut self, guest: u64, bytes: u64) -> runtime::Result<NonNull<c_void>> { self.map(guest, bytes).ok_or_else(|| runtime::Error::invalid("section alias outside test memory")) }
    fn unmap_backing(&mut self, alias: NonNull<c_void>) { self.unmap(alias); }
    fn import_pages(&self, alias: NonNull<c_void>, bytes: u64) -> runtime::Result<Vec<u64>> {
        let mapping = &self.aliases[&(alias.as_ptr() as usize)];
        Ok((0..bytes / PAGE).map(|page| 1 + mapping.region as u64 * 4 + mapping.offset / PAGE + page).collect())
    }
    fn gpu_ranges(&mut self, range: GuestRange) -> Vec<GuestRange> {
        (0..3).filter_map(|region| {
            let start = va::GUEST_WINDOW_BASE + region * va::IMPORT_WINDOW_BYTES;
            let first = range.address.max(start);
            let end = range.end().min(start + PAGE * 4);
            (first < end).then_some(GuestRange::new(first, end.saturating_sub(first)))
        }).collect()
    }
}

#[test]
#[ignore = "requires the dedicated AMD WDDM test machine and its GPU gate"]
fn backing_survives_guest_unmap_and_maps_allocation_offsets() -> Result<(), Box<dyn std::error::Error>> {
    let _exclusive = GPU_TEST.lock().unwrap();
    let memory = SectionView::new();
    let mut device = Device::open()?;
    device.use_coherent_host_imports();
    let result = (|| -> runtime::Result<()> {
        let guest = va::GUEST_WINDOW_BASE;
        let mut window = device.reserve_guest_window(guest, va::GUEST_END)?;
        unsafe { window.import_backing(&mut device, 1, guest, PAGE * 4, memory.view.Value.cast::<u8>())?; }
        assert_eq!(read_pages(&mut device, &window, guest, 4)?, [0x1200_0000, 0x1200_0001, 0x1200_0002, 0x1200_0003]);
        let physical = window.backing(1).unwrap() as *const _;
        window.release_bindings(&mut device, guest, PAGE * 4)?;
        assert_eq!(window.backing(1).unwrap() as *const _, physical);
        let second = guest + va::IMPORT_WINDOW_BYTES;
        window.bind_backing(&mut device, second, PAGE * 2, 1, PAGE)?;
        assert_eq!(read_pages(&mut device, &window, second, 2)?, [0x1200_0001, 0x1200_0002]);
        assert_eq!(window.backing(1).unwrap() as *const _, physical);
        assert!(window.release_backing(&mut device, 1).is_err());
        window.release_range(&mut device, second, PAGE * 2)?;
        window.bind_backing(&mut device, guest, PAGE, 1, PAGE * 3)?;
        assert_eq!(read_pages(&mut device, &window, guest, 1)?, [0x1200_0003]);
        window.release_bindings(&mut device, guest, PAGE)?;
        let allocations = device.live_allocations();
        window.release_backing(&mut device, 1)?;
        assert_eq!(device.live_allocations() + 1, allocations);
        device.wait_idle()
    })();
    if device.is_lost() { std::mem::forget(memory); }
    result?;
    Ok(())
}

#[test]
#[ignore = "requires the dedicated AMD WDDM test machine and its GPU gate"]
fn copy_budget_failure_preserves_protected_bindings_after_unrelated_eviction() -> Result<(), Box<dyn std::error::Error>> {
    let _exclusive = GPU_TEST.lock().unwrap();
    let mut device = Device::open()?;
    device.use_coherent_host_imports();
    let mut residency = GuestResidency::<WddmWindow>::open(&mut device, &ResidencyOptions { import_budget: PAGE * 12, ..Default::default() })?;
    residency.set_memory(Some(Box::new(SectionMemory::new())))?;
    let a = va::GUEST_WINDOW_BASE;
    let b = a + va::IMPORT_WINDOW_BYTES;
    let c = b + va::IMPORT_WINDOW_BYTES;
    let _unrelated = residency.prepare(&mut device, Reach { ranges: &[GuestRange::new(c, PAGE * 4)], computed_addresses: false })?;
    let residents = residency.prepare(&mut device, Reach { ranges: &[GuestRange::new(a, PAGE * 4), GuestRange::new(b, PAGE * 4)], computed_addresses: false })?;
    assert_eq!(residency.import_count(), 3);
    residency.protect(&residents)?;
    assert!(residency.ensure_host_copy(&mut device, a + va::IMPORT_WINDOW_BYTES * 3, PAGE * 8).is_err());
    assert_eq!(residency.import_count(), 2, "only the unrelated physical backing may be evicted");
    assert_eq!(residency.targets(&residents)?.len(), 2);
    assert_eq!(read_words(&mut device, &residency.targets(&residents)?, &[a, b])?, [0x1200_0000, 0x1200_0000]);
    residency.close(&mut device)?;
    Ok(())
}

/// The GPU reaches memory through a shared target set, a CPU read waits for the submissions that
/// used the set, and neither the cached set nor one still in flight keeps memory from being released.
#[test]
#[ignore = "requires the dedicated AMD WDDM test machine and its GPU gate"]
fn shared_targets_reach_memory_and_never_pin_a_release() -> Result<(), Box<dyn std::error::Error>> {
    let _exclusive = GPU_TEST.lock().unwrap();
    let mut device = Device::open()?;
    device.use_coherent_host_imports();
    let mut residency = GuestResidency::<WddmWindow>::open(&mut device, &ResidencyOptions::default())?;
    residency.set_memory(Some(Box::new(SectionMemory::new())))?;
    let a = va::GUEST_WINDOW_BASE;
    let b = a + va::IMPORT_WINDOW_BYTES;
    let residents = residency.prepare(&mut device, Reach { ranges: &[GuestRange::new(a, PAGE * 4), GuestRange::new(b, PAGE * 4)], computed_addresses: false })?;
    let output = [device.allocate_kind(65536, MemoryKind::Readback)?];
    let mut image = copy_words(&output[0], &[a, b])?.into_words();
    image.extend(host_pm4::nop_fill((host_pm4::IB_ALIGNMENT_DWORDS - image.len() % host_pm4::IB_ALIGNMENT_DWORDS) % host_pm4::IB_ALIGNMENT_DWORDS));
    for _ in 0..3 {
        let targets = residency.shared_targets(&device, &residents, &output)?;
        let serial = device.queue_images(QueueKind::Graphics, std::slice::from_ref(&image), targets)?;
        residency.queued(&residents, Submission::graphics(serial));
    }
    assert_eq!(words_of(&output[0], 2)?, [0x1200_0000, 0x1200_0000]);
    let targets = residency.shared_targets(&device, &residents, &output)?;
    let serial = device.queue_images(QueueKind::Graphics, std::slice::from_ref(&image), targets)?;
    residency.queued(&residents, Submission::graphics(serial));
    residency.release_range(&mut device, GuestRange::new(a, PAGE * 4))?;
    residency.trim(&mut device, 0)?;
    assert_eq!(residency.import_count(), 0);
    residency.close(&mut device)?;
    Ok(())
}

/// A program that finds an image at run time reads the image's host copy without naming it,
/// whether or not copy memory is there. Without it the addresses read zero, and work in flight
/// that reads them keeps running while copy memory comes there and goes.
#[test]
#[ignore = "requires the dedicated AMD WDDM test machine and its GPU gate"]
fn host_copy_addresses_stay_readable_as_copy_memory_comes_and_goes() -> Result<(), Box<dyn std::error::Error>> {
    let _exclusive = GPU_TEST.lock().unwrap();
    let mut device = Device::open()?;
    device.use_coherent_host_imports();
    let mut residency = GuestResidency::<WddmWindow>::open(&mut device, &ResidencyOptions::default())?;
    let (guest, bytes) = (va::GUEST_WINDOW_BASE + (64 << 20), 4 * COPY_GRANULE_BYTES);
    let copy = guest + va::HOST_COPY_OFFSET;
    let addresses: Vec<u64> = (0..2048u64).map(|index| copy + ((index * 0x1004) % bytes & !3)).collect();
    let (probed, zero) = (&addresses[..16], [0u32; 16]);
    assert_eq!(read_words(&mut device, &[], probed)?, zero);

    // Copy memory there is what the addresses reach, and once it has gone they read zero again.
    let mut marker = device.allocate_kind(65536, MemoryKind::Readback)?;
    marker.write(0, &0x600d_c0de_u32.to_le_bytes())?;
    residency.ensure_host_copy(&mut device, guest, bytes)?;
    let mut targets = residency.host_copies(guest, bytes).expect("the copy was made");
    targets.push(&marker);
    let stream = copy_dwords(&probed.iter().map(|&address| (marker.address(), address)).collect::<Vec<_>>())?;
    let serial = device.submit_host(stream.words(), &targets)?;
    device.wait(serial)?;
    assert_eq!(read_words(&mut device, &[], probed)?, [0x600d_c0de; 16]);
    residency.release_host_copy(&mut device, guest)?;
    assert_eq!(read_words(&mut device, &[], probed)?, zero);

    // Readers that do not name the copy stay in flight while its memory comes and goes.
    let output = [device.allocate_kind(65536, MemoryKind::Readback)?];
    let mut image = copy_words(&output[0], &addresses)?.into_words();
    image.extend(host_pm4::nop_fill((host_pm4::IB_ALIGNMENT_DWORDS - image.len() % host_pm4::IB_ALIGNMENT_DWORDS) % host_pm4::IB_ALIGNMENT_DWORDS));
    let readers = device.share_targets(&[&output[0]])?;
    let (started, mut cycles, mut queued) = (Instant::now(), 0u32, 0u32);
    while started.elapsed() < Duration::from_secs(3) {
        while device.can_queue_host_on(QueueKind::Graphics) {
            device.queue_images(QueueKind::Graphics, std::slice::from_ref(&image), &readers)?;
            queued += 1;
        }
        residency.ensure_host_copy(&mut device, guest, bytes)?;
        residency.release_host_copy(&mut device, guest)?;
        cycles += 1;
    }
    device.wait_idle()?;
    println!("copy memory came and went {cycles} times under {queued} submissions of {} reads each", addresses.len());
    assert_eq!(read_words(&mut device, &[], probed)?, zero);
    drop(readers);
    residency.close(&mut device)?;
    Ok(())
}
