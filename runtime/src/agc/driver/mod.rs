//! libSceAgcDriver in Rust (PLAN P3.4 step 3, decisions D15, D21): the guest's `sceAgcDriver*`
//! functions, and the `AgcDriver*` interface AnyPS5's other modules use, in place of the C++
//! libSceAgcDriver.

pub mod abi;
pub mod anyps5;
pub mod device;
pub mod events;
pub mod flips;
pub mod interface;
pub mod memory;
pub mod present;

use anyps5::Anyps5;
use device::DeviceService;
use events::Registrations;
use flips::{Failure, Flips, Outcome};
use interface::result;
use present::HostHdr;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, OnceLock};

/// The tessellation factor ring the guest last set (`sceAgcDriverSetTFRing`).
struct TessellationRing {
    base: AtomicU64,
    bytes: AtomicU32,
}

/// What the running driver has opened.
pub struct Host {
    pub service: DeviceService,
    pub arena: Arc<Anyps5>,
    /// The device's adapter, for presenters on the same GPU: LUID low and high.
    pub adapter: (u32, i32),
}

/// The driver of this process.
pub struct Driver {
    pub flips: Arc<Flips>,
    pub registrations: Arc<Registrations>,
    pub hdr: HostHdr,
    tessellation_ring: TessellationRing,
    host: OnceLock<std::result::Result<Host, String>>,
}

impl Driver {
    fn new() -> Self {
        Self {
            flips: Arc::default(),
            registrations: Arc::default(),
            hdr: HostHdr::default(),
            tessellation_ring: TessellationRing { base: AtomicU64::new(0xf_f000_0000), bytes: AtomicU32::new(0x2_0000) },
            host: OnceLock::new(),
        }
    }

    /// The driver, created on first use.
    pub fn get() -> &'static Self {
        static DRIVER: OnceLock<Driver> = OnceLock::new();
        DRIVER.get_or_init(Self::new)
    }

    #[cfg(all(feature = "wddm", target_os = "windows"))]
    fn open(&self) -> crate::Result<Host> {
        let arena = Arc::new(Anyps5::resolve()?);
        let adapter = Arc::new(OnceLock::new());
        let opening = (self.flips.clone(), arena.clone(), adapter.clone());
        let service = DeviceService::start(
            move || {
                let (flips, arena, adapter) = opening;
                let engine = crate::agc::abi::agc().target.open(flips, arena)?;
                let _ = adapter.set(engine.adapter());
                Ok(engine)
            },
            self.flips.clone(),
            arena.clone(),
            self.registrations.clone(),
        )?;
        anyps5::GuestArena::set_import_release(&*arena, Some((Self::release_imports, std::ptr::null_mut())));
        let adapter = *adapter.get().expect("set before the device opened");
        Ok(Host { service, arena, adapter })
    }

    #[cfg(not(all(feature = "wddm", target_os = "windows")))]
    fn open(&self) -> crate::Result<Host> {
        Err(crate::Error::unsupported("libSceAgcDriver needs the wddm backend on Windows"))
    }

    /// libc's call before a guest mapping change over imports.
    #[cfg(all(feature = "wddm", target_os = "windows"))]
    unsafe extern "C" fn release_imports(_user: *mut std::ffi::c_void, address: usize, bytes: usize) -> bool {
        Self::get().started().is_some_and(|host| host.service.release(crate::residency::GuestRange::new(address as u64, bytes as u64)))
    }

    /// The opened device, opening it on first use.
    pub fn host(&self) -> Outcome<&Host> {
        self.host
            .get_or_init(|| self.open().map_err(|error| error.message))
            .as_ref()
            .map_err(|message| Failure::new(result::FAILED, format!("libSceAgcDriver could not open the device: {message}")))
    }

    /// The opened device, if it is open.
    pub fn started(&self) -> Option<&Host> {
        self.host.get().and_then(|host| host.as_ref().ok())
    }

    /// Stops device work. VideoOut unregisters its outputs after its presenter has stopped.
    pub fn shutdown(&self) {
        let Some(host) = self.started() else {
            return;
        };
        host.service.stop();
        anyps5::GuestArena::set_import_release(&*host.arena, None);
        self.flips.shutdown();
    }
}
