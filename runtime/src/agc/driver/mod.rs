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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// What the title told the driver about tessellation: the ring the geometry engine reads the
/// factors from (`sceAgcDriverSetTFRing`) and the off-chip ring's VGT_HS_OFFCHIP_PARAM
/// (`sceAgcDriverSetHsOffchipParam`). On the PS5 the kernel programs the GPU with them; a
/// title's command buffers never do.
pub struct Tessellation {
    base: AtomicU64,
    bytes: AtomicU32,
    offchip: AtomicU32,
    /// Bit 0: the title set the factor ring; bit 1: the off-chip parameter.
    set: AtomicU32,
}

/// The rings as a draw finds them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TessellationRings {
    /// Address and bytes of the factor ring, once set.
    pub factor: Option<(u64, u32)>,
    /// VGT_HS_OFFCHIP_PARAM as gfx1013 takes it, once set.
    pub offchip: Option<u32>,
}

impl Tessellation {
    const FACTOR: u32 = 1;
    const OFFCHIP: u32 = 2;

    /// What `sceAgcDriverGetTFRing` answers before the title sets a ring.
    fn new() -> Self {
        Self {
            base: AtomicU64::new(0xf_f000_0000),
            bytes: AtomicU32::new(0x2_0000),
            offchip: AtomicU32::new(0),
            set: AtomicU32::new(0),
        }
    }

    pub fn factor_ring(&self) -> (u64, u32) {
        (self.base.load(Ordering::Acquire), self.bytes.load(Ordering::Acquire))
    }

    pub fn set_factor_ring(&self, base: u64, bytes: u32) {
        self.base.store(base, Ordering::Release);
        self.bytes.store(bytes, Ordering::Release);
        if self.set.fetch_or(Self::FACTOR, Ordering::AcqRel) & Self::FACTOR == 0 {
            eprintln!("[agc] tessellation factor ring at {base:#x}, {bytes} bytes");
        }
    }

    pub fn set_offchip(&self, param: u32) {
        self.offchip.store(param, Ordering::Release);
        if self.set.fetch_or(Self::OFFCHIP, Ordering::AcqRel) & Self::OFFCHIP == 0 {
            eprintln!("[agc] off-chip tessellation parameter {param:#x}");
        }
    }

    pub fn rings(&self) -> TessellationRings {
        let set = self.set.load(Ordering::Acquire);
        TessellationRings {
            factor: (set & Self::FACTOR != 0).then(|| self.factor_ring()),
            offchip: (set & Self::OFFCHIP != 0).then(|| self.offchip.load(Ordering::Acquire)),
        }
    }
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
    pub tessellation: Tessellation,
    host: OnceLock<std::result::Result<Host, String>>,
}

impl Driver {
    fn new() -> Self {
        Self {
            flips: Arc::default(),
            registrations: Arc::default(),
            hdr: HostHdr::default(),
            tessellation: Tessellation::new(),
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
