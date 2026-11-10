//! The WDDM device and guest window behind [`GuestResidency`].

use super::{
    import_budget, GuestResidency, LocalMemory, Queue, ResidencyDevice, ResidencyOptions,
    ResidencyWindow,
};
use crate::{
    backend::{
        va,
        wddm::{Buffer, Device, GuestWindow, MemoryKind, QueueKind, TargetSet, VaReservation},
    },
    Error, ErrorKind, Result,
};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    ptr::NonNull,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};

/// Set while a residency's guest window is reserved in this process. A D3D12 device created
/// before it would take the window's addresses (S1-r3), so v5 `presenter_create` checks it.
static GUEST_WINDOW: AtomicBool = AtomicBool::new(false);

/// Whether a guest window is reserved in this process: from [`GuestResidency::open`] until the
/// residency is closed or dropped.
pub fn guest_window_reserved() -> bool {
    GUEST_WINDOW.load(Ordering::Acquire)
}

/// The guest window and the host-copy window (§9.1), dropped in that order; the host-copy window
/// is freed once its last copy is.
pub struct WddmWindow {
    window: GuestWindow,
    host_copy: VaReservation,
    /// Host-layout copy memory in `host_copy`, by address.
    copies: BTreeMap<u64, Buffer>,
}

impl WddmWindow {
    /// The backend's guest window, with the imports this residency made in it.
    pub fn guest_window(&self) -> &GuestWindow {
        &self.window
    }

    /// `[va::HOST_COPY_BASE, va::HOST_COPY_END)`: host-layout copies of guest surfaces at guest
    /// VA + 2^40 ([`GuestResidency::ensure_host_copy`]), and later staged copies of guest memory
    /// outside the guest window (§7).
    pub fn host_copy(&self) -> &VaReservation {
        &self.host_copy
    }
}

impl Drop for WddmWindow {
    fn drop(&mut self) {
        GUEST_WINDOW.store(false, Ordering::Release);
    }
}

impl GuestResidency<WddmWindow> {
    /// The open sequence (§9.2): reserves the guest window and the host-copy window, and sets the
    /// import budget. Call it right after [`Device::open`], before the device opens further
    /// queues, rings or pools and before any D3D12 device exists in the process. One guest window
    /// per process.
    pub fn open(device: &mut Device, options: &ResidencyOptions) -> Result<Self> {
        if GUEST_WINDOW.swap(true, Ordering::AcqRel) {
            return Err(Error::invalid(
                "a guest window is already reserved in this process",
            ));
        }
        let opened = open_windows(device, options);
        if opened.is_err() {
            GUEST_WINDOW.store(false, Ordering::Release);
        }
        opened
    }
}

fn open_windows(
    device: &mut Device,
    options: &ResidencyOptions,
) -> Result<GuestResidency<WddmWindow>> {
    // Guest memory no import maps faults the device, as on PS5; the partially resident (PRT)
    // ranges the guest registers read zero through Zero PTEs ([`GuestResidency::sparse_reserve`]).
    let window = device
        .reserve_guest_window(options.window_base, options.window_end)
        .map_err(|error| match error.kind {
            ErrorKind::System => Error::new(
                ErrorKind::System,
                format!(
                    "{error}; was a D3D12 device created before the guest window was reserved?"
                ),
            ),
            _ => error,
        })?;
    // Host copies that no work retiled (an image descriptor computed at run time) read zero
    // rather than faulting the device; a driver that refuses zero reservations faults instead.
    let (base, size) = (va::HOST_COPY_BASE, va::HOST_COPY_END - va::HOST_COPY_BASE);
    let host_copy = match device.reserve_va_zero(base, size) {
        Ok(reservation) => reservation,
        Err(error) => {
            eprintln!(
                "[agc-rust] the host-copy window faults on unretiled copies: a zero reservation failed: {}",
                error.message
            );
            device.reserve_va(base, size)?
        }
    };
    let local = device.video_memory()?.local.budget;
    // APS5_IMPORT_BUDGET_MIB overrides the default (half of the local segment budget): imports
    // pin guest system memory, not the GPU's carve-out, and a budget below what a frame touches
    // releases and re-imports guest memory every frame, waiting for the GPU each time.
    let requested = std::env::var("APS5_IMPORT_BUDGET_MIB")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(options.import_budget, |mib| mib << 20);
    let budget = import_budget(requested, local);
    eprintln!(
        "[agc-rust] residency: import budget {} MiB (local segment budget {} MiB)",
        budget >> 20,
        local >> 20
    );
    Ok(GuestResidency::new(
        WddmWindow {
            window,
            host_copy,
            copies: BTreeMap::new(),
        },
        budget,
    ))
}

impl ResidencyDevice for Device {
    fn completed(&mut self, queue: Queue) -> u64 {
        self.poll();
        self.completed_on(match queue {
            Queue::Graphics => QueueKind::Graphics,
            Queue::Compute => QueueKind::Compute,
            Queue::ComputeQueue(index) => QueueKind::ComputeQueue(index),
        })
    }

    fn wait_idle(&mut self) -> Result<()> {
        Device::wait_idle(self)
    }

    fn is_lost(&self) -> bool {
        self.is_faulted()
    }

    fn settle_paging(&mut self) -> Result<()> {
        Device::settle_paging(self)
    }

    fn local_memory(&self) -> Result<LocalMemory> {
        let local = self.video_memory()?.local;
        Ok(LocalMemory {
            usage: local.usage,
            budget: local.budget,
        })
    }
}

impl ResidencyWindow for WddmWindow {
    type Device = Device;
    type Import = Buffer;
    type Targets = Rc<TargetSet>;

    fn share_targets(&self, device: &Device, targets: &[&Buffer]) -> Result<Rc<TargetSet>> {
        device.share_targets(targets)
    }

    fn sparse_zero(&mut self, device: &mut Device, range: super::GuestRange) -> Result<()> {
        self.window.sparse_zero(device, range.address, range.bytes)
    }

    fn base(&self) -> u64 {
        self.window.base()
    }

    fn end(&self) -> u64 {
        self.window.end()
    }

    unsafe fn import_backing(
        &mut self,
        device: &mut Device,
        id: u64,
        guest: u64,
        bytes: u64,
        alias: NonNull<c_void>,
    ) -> Result<()> {
        // SAFETY: the caller's contract is `GuestWindow::import`'s.
        unsafe {
            self.window
                .import_backing(device, id, guest, bytes, alias.as_ptr().cast())
        }
        .map(|_| ())
    }

    fn bind_backing(&mut self, device: &mut Device, guest: u64, bytes: u64, id: u64, offset: u64) -> Result<()> {
        self.window.bind_backing(device, guest, bytes, id, offset)
    }

    fn release_bindings(&mut self, device: &mut Device, guest: u64, bytes: u64) -> Result<()> {
        self.window.release_bindings(device, guest, bytes)
    }

    fn release_backing(&mut self, device: &mut Device, id: u64) -> Result<()> {
        self.window.release_backing(device, id)
    }

    fn backing_at(&self, id: u64) -> Option<&Buffer> {
        self.window.backing(id)
    }

    fn allocate_copy(&mut self, device: &mut Device, address: u64, bytes: u64) -> Result<()> {
        let size =
            usize::try_from(bytes).map_err(|_| Error::invalid("host copy size overflows"))?;
        let copy = device.allocate_at(size, MemoryKind::Device, address, &self.host_copy)?;
        self.copies.insert(address, copy);
        Ok(())
    }

    fn release_copy(&mut self, device: &mut Device, address: u64) -> Result<()> {
        let copy = self
            .copies
            .get(&address)
            .ok_or_else(|| Error::invalid(format!("no host copy starts at 0x{address:x}")))?;
        if !device.await_buffer(copy)? {
            return Err(Error::invalid(format!(
                "the host copy at 0x{address:x} is still held after its submissions retired"
            )));
        }
        self.copies.remove(&address);
        Ok(())
    }

    fn copy_at(&self, address: u64) -> Option<&Buffer> {
        self.copies.get(&address)
    }
}
