//! What libSceAgcDriver calls in AnyPS5's libc and libkernel (decisions D21): guest memory and
//! event queues. [`Anyps5::resolve`] finds the functions among the loaded modules by export name:
//! the NID of the name for `_nid_postfix` functions, the name itself for `_nid_no_patch` ones.
//! The traits let the driver run against stand-ins in tests.

use super::interface::{GuestImportRelease, GuestRange, GuestSparseRange};
use std::ffi::c_void;
use std::ptr::NonNull;

/// libc's guest memory, as the driver uses it.
pub trait GuestArena: Send + Sync {
    /// A read-write alias of whole guest pages inside one import window, or why libc refused
    /// (`interface::import`).
    fn map_import_alias(&self, guest: u64, bytes: u64) -> Result<NonNull<c_void>, u32>;
    fn unmap_import_alias(&self, alias: NonNull<c_void>);
    fn import_alias_pages(&self, alias: NonNull<c_void>, pages: &mut [u64]) -> usize;
    fn map_backing_alias(&self, guest: u64, bytes: u64) -> crate::Result<NonNull<c_void>>;
    fn unmap_backing_alias(&self, alias: NonNull<c_void>);
    /// The registered GPU-visible pieces of a range: stores at most `pieces.len()` and returns how
    /// many there are.
    fn gpu_visible(&self, address: u64, bytes: u64, pieces: &mut [GuestRange]) -> usize;
    /// Ends the import record of exactly `[address, address + bytes)` (a lost device keeps it).
    fn forget_import(&self, address: u64, bytes: u64);
    /// Changes whenever a guest mapping changes.
    fn generation(&self) -> u64;
    /// The write-watched arena: its base and bytes, empty when nothing is write-watched.
    fn arena(&self) -> (u64, u64);
    /// The 4 KiB pages the CPU wrote in the range since the last collect that cleared them, in
    /// address order: stores at most `pages.len()` addresses and returns how many it stored;
    /// `None` when libc cannot watch the range.
    fn collect_writes(&self, address: u64, bytes: u64, pages: &mut [*mut c_void], clear: bool) -> Option<usize>;
    /// Installs the driver's side of imports (`None` uninstalls it).
    fn set_import_release(&self, release: Option<(GuestImportRelease, *mut c_void)>);
    /// Copies at most `ranges.len()` sparse ranges: their count and generation.
    fn sparse_ranges(&self, ranges: &mut [GuestSparseRange]) -> (usize, u64);
}

/// libkernel's event queues, as the driver uses them.
pub trait EventQueues: Send + Sync {
    /// Adds an event whose trigger stores the 64-bit value it is given in the event's data.
    fn add_data_event(&self, queue: i64, ident: u64, filter: i16, flags: u16, data: i64, udata: *mut c_void) -> i32;
    fn trigger(&self, queue: i64, ident: u64, filter: i16, data: &mut u64) -> i32;
    fn delete(&self, queue: i64, ident: u64, filter: i16) -> i32;
    /// The title is submitting GPU work (libkernel's sleep accounting).
    fn note_gpu_submit(&self) {}
}

/// The graphics-core event filter of `sceAgcDriverAddEqEvent`.
pub const EVFILT_GRAPHICS_CORE: i16 = -14;
pub const EV_ADD: u16 = 0x0001;
pub const EV_CLEAR: u16 = 0x0020;

type MapImportAlias = unsafe extern "C" fn(address: usize, bytes: usize, refused: *mut u32) -> *mut c_void;
type UnmapImportAlias = unsafe extern "C" fn(alias: *mut c_void);
type ImportAliasPages = unsafe extern "C" fn(alias: *const c_void, pages: *mut u64, capacity: usize) -> usize;
type MapBackingAlias = unsafe extern "C" fn(address: usize, bytes: usize, bounds_refused: *mut bool) -> *mut c_void;
type GpuVisible = unsafe extern "C" fn(address: usize, bytes: usize, pieces: *mut GuestRange, capacity: usize) -> usize;
type ForgetImport = unsafe extern "C" fn(address: usize, bytes: usize);
type Generation = unsafe extern "C" fn() -> u64;
type ArenaRange = unsafe extern "C" fn(base: *mut usize, bytes: *mut usize);
type WriteWatched = unsafe extern "C" fn() -> bool;
type CollectWrites = unsafe extern "C" fn(address: usize, bytes: usize, pages: *mut *mut c_void, count: *mut usize, clear: bool) -> bool;
type SetImportRelease = unsafe extern "C" fn(release: Option<GuestImportRelease>, user: *mut c_void);
type SparseRanges = unsafe extern "C" fn(generation: *mut u64, ranges: *mut GuestSparseRange, capacity: usize) -> usize;
type AddDataEvent = unsafe extern "sysv64" fn(queue: i64, ident: usize, filter: i16, flags: u16, data: isize, udata: *mut c_void) -> i32;
type TriggerEvent = unsafe extern "sysv64" fn(queue: i64, ident: usize, filter: i16, data: *mut c_void) -> i32;
type DeleteEvent = unsafe extern "sysv64" fn(queue: i64, ident: usize, filter: i16) -> i32;
type NoteGpuSubmit = unsafe extern "C" fn();

/// The functions, as the loaded modules export them.
pub struct Anyps5 {
    map_import_alias: MapImportAlias,
    unmap_import_alias: UnmapImportAlias,
    import_alias_pages: ImportAliasPages,
    map_backing_alias: MapBackingAlias,
    unmap_backing_alias: UnmapImportAlias,
    gpu_visible: GpuVisible,
    forget_import: ForgetImport,
    generation: Generation,
    arena_range: ArenaRange,
    write_watched: WriteWatched,
    collect_writes: CollectWrites,
    set_import_release: SetImportRelease,
    sparse_ranges: SparseRanges,
    add_data_event: AddDataEvent,
    trigger_event: TriggerEvent,
    delete_event: DeleteEvent,
    note_gpu_submit: NoteGpuSubmit,
}

impl Anyps5 {
    /// Finds every function in the loaded modules; names the ones missing.
    #[cfg(all(feature = "wddm", target_os = "windows"))]
    pub fn resolve() -> crate::Result<Self> {
        let modules = Modules::loaded()?;
        let mut missing = Vec::new();
        let mut find = |name: &'static str| -> *const c_void {
            let found = modules.find(name);
            if found.is_null() {
                missing.push(name);
            }
            found
        };
        let pointers = [
            find("GuestArenaMapImportAlias_nid_no_patch"),
            find("GuestArenaUnmapImportAlias_nid_no_patch"),
            find("GuestAllocationsGpuVisible_nid_no_patch"),
            find("GuestAllocationsForgetImport_nid_no_patch"),
            find("GuestAllocationsGeneration_nid_postfix"),
            find("GuestArenaRange_nid_postfix"),
            find("GuestArenaWriteWatched_nid_postfix"),
            find("GuestArenaCollectWrites_nid_postfix"),
            find("GuestAllocationsSetImportRelease_nid_no_patch"),
            find("GuestAllocationsSparseRanges_nid_no_patch"),
            find("EqueueAddDataEvent_nid_postfix"),
            find("EqueueTriggerEvent_nid_postfix"),
            find("EqueueDeleteEvent_nid_postfix"),
            find("KernelNoteGpuSubmit_nid_no_patch"),
            find("GuestArenaImportAliasPages_nid_no_patch"),
            find("GuestArenaMapAliasForImport_nid_no_patch"),
            find("GuestArenaUnmapAlias_nid_postfix"),
        ];
        if !missing.is_empty() {
            return Err(crate::Error::unsupported(format!("libSceAgcDriver: AnyPS5 exports none of {}", missing.join(", "))));
        }
        // SAFETY: each pointer is the function of that name, with the prototype `AgcDriver.h` or
        // libc's and libkernel's headers declare.
        unsafe {
            Ok(Self {
                map_import_alias: std::mem::transmute::<*const c_void, MapImportAlias>(pointers[0]),
                unmap_import_alias: std::mem::transmute::<*const c_void, UnmapImportAlias>(pointers[1]),
                gpu_visible: std::mem::transmute::<*const c_void, GpuVisible>(pointers[2]),
                forget_import: std::mem::transmute::<*const c_void, ForgetImport>(pointers[3]),
                generation: std::mem::transmute::<*const c_void, Generation>(pointers[4]),
                arena_range: std::mem::transmute::<*const c_void, ArenaRange>(pointers[5]),
                write_watched: std::mem::transmute::<*const c_void, WriteWatched>(pointers[6]),
                collect_writes: std::mem::transmute::<*const c_void, CollectWrites>(pointers[7]),
                set_import_release: std::mem::transmute::<*const c_void, SetImportRelease>(pointers[8]),
                sparse_ranges: std::mem::transmute::<*const c_void, SparseRanges>(pointers[9]),
                add_data_event: std::mem::transmute::<*const c_void, AddDataEvent>(pointers[10]),
                trigger_event: std::mem::transmute::<*const c_void, TriggerEvent>(pointers[11]),
                delete_event: std::mem::transmute::<*const c_void, DeleteEvent>(pointers[12]),
                note_gpu_submit: std::mem::transmute::<*const c_void, NoteGpuSubmit>(pointers[13]),
                import_alias_pages: std::mem::transmute::<*const c_void, ImportAliasPages>(pointers[14]),
                map_backing_alias: std::mem::transmute::<*const c_void, MapBackingAlias>(pointers[15]),
                unmap_backing_alias: std::mem::transmute::<*const c_void, UnmapImportAlias>(pointers[16]),
            })
        }
    }
}

impl GuestArena for Anyps5 {
    fn map_import_alias(&self, guest: u64, bytes: u64) -> Result<NonNull<c_void>, u32> {
        let mut refused = super::interface::import::MADE;
        // SAFETY: the resolved function; `refused` outlives the call.
        let alias = unsafe { (self.map_import_alias)(guest as usize, bytes as usize, &mut refused) };
        NonNull::new(alias).ok_or(refused)
    }

    fn unmap_import_alias(&self, alias: NonNull<c_void>) {
        // SAFETY: an alias `map_import_alias` returned, unmapped once.
        unsafe { (self.unmap_import_alias)(alias.as_ptr()) }
    }

    fn import_alias_pages(&self, alias: NonNull<c_void>, pages: &mut [u64]) -> usize {
        // SAFETY: a live import alias and an output buffer with the declared capacity.
        unsafe { (self.import_alias_pages)(alias.as_ptr(), pages.as_mut_ptr(), pages.len()) }
    }

    fn map_backing_alias(&self, guest: u64, bytes: u64) -> crate::Result<NonNull<c_void>> {
        let mut bounds_refused = false;
        // SAFETY: the enclosing import keeps this guest range registered and section-backed.
        let alias = unsafe { (self.map_backing_alias)(guest as usize, bytes as usize, &mut bounds_refused) };
        NonNull::new(alias).ok_or_else(|| crate::Error::invalid(format!("could not retain physical backing for 0x{guest:x}+0x{bytes:x}")))
    }

    fn unmap_backing_alias(&self, alias: NonNull<c_void>) {
        // SAFETY: a retained physical alias, released after its KMD allocation is destroyed.
        unsafe { (self.unmap_backing_alias)(alias.as_ptr()) }
    }

    fn gpu_visible(&self, address: u64, bytes: u64, pieces: &mut [GuestRange]) -> usize {
        // SAFETY: `pieces` has room for its length.
        unsafe { (self.gpu_visible)(address as usize, bytes as usize, pieces.as_mut_ptr(), pieces.len()) }
    }

    fn forget_import(&self, address: u64, bytes: u64) {
        // SAFETY: the resolved function.
        unsafe { (self.forget_import)(address as usize, bytes as usize) }
    }

    fn generation(&self) -> u64 {
        // SAFETY: the resolved function.
        unsafe { (self.generation)() }
    }

    fn arena(&self) -> (u64, u64) {
        // SAFETY: the resolved functions; the outputs outlive the call.
        unsafe {
            if !(self.write_watched)() {
                return (0, 0);
            }
            let (mut base, mut bytes) = (0usize, 0usize);
            (self.arena_range)(&mut base, &mut bytes);
            (base as u64, bytes as u64)
        }
    }

    fn collect_writes(&self, address: u64, bytes: u64, pages: &mut [*mut c_void], clear: bool) -> Option<usize> {
        let mut count = pages.len();
        // SAFETY: `pages` has room for `count` addresses.
        let walked = unsafe { (self.collect_writes)(address as usize, bytes as usize, pages.as_mut_ptr(), &mut count, clear) };
        walked.then_some(count.min(pages.len()))
    }

    fn set_import_release(&self, release: Option<(GuestImportRelease, *mut c_void)>) {
        let (function, user) = match release {
            Some((function, user)) => (Some(function), user),
            None => (None, std::ptr::null_mut()),
        };
        // SAFETY: the resolved function; the caller keeps `user` valid until it uninstalls it.
        unsafe { (self.set_import_release)(function, user) }
    }

    fn sparse_ranges(&self, ranges: &mut [GuestSparseRange]) -> (usize, u64) {
        let mut generation = 0;
        // SAFETY: `ranges` has room for its length.
        let count = unsafe { (self.sparse_ranges)(&mut generation, ranges.as_mut_ptr(), ranges.len()) };
        (count, generation)
    }
}

impl EventQueues for Anyps5 {
    fn add_data_event(&self, queue: i64, ident: u64, filter: i16, flags: u16, data: i64, udata: *mut c_void) -> i32 {
        // SAFETY: the resolved function.
        unsafe { (self.add_data_event)(queue, ident as usize, filter, flags, data as isize, udata) }
    }

    fn trigger(&self, queue: i64, ident: u64, filter: i16, data: &mut u64) -> i32 {
        // SAFETY: the resolved function; `data` outlives the call.
        unsafe { (self.trigger_event)(queue, ident as usize, filter, (data as *mut u64).cast()) }
    }

    fn delete(&self, queue: i64, ident: u64, filter: i16) -> i32 {
        // SAFETY: the resolved function.
        unsafe { (self.delete_event)(queue, ident as usize, filter) }
    }

    fn note_gpu_submit(&self) {
        // SAFETY: the resolved function.
        unsafe { (self.note_gpu_submit)() }
    }
}

// SAFETY: the functions are AnyPS5's thread-safe exports.
unsafe impl Send for Anyps5 {}
// SAFETY: as above.
unsafe impl Sync for Anyps5 {}

/// The modules loaded in this process, searched in load order.
#[cfg(all(feature = "wddm", target_os = "windows"))]
struct Modules(Vec<windows_sys::Win32::Foundation::HMODULE>);

#[cfg(all(feature = "wddm", target_os = "windows"))]
impl Modules {
    fn loaded() -> crate::Result<Self> {
        use windows_sys::Win32::System::ProcessStatus::K32EnumProcessModules;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut modules = vec![std::ptr::null_mut(); 1024];
        loop {
            let mut needed = 0u32;
            let bytes = (modules.len() * std::mem::size_of::<windows_sys::Win32::Foundation::HMODULE>()) as u32;
            // SAFETY: `modules` has room for `bytes`.
            if unsafe { K32EnumProcessModules(GetCurrentProcess(), modules.as_mut_ptr(), bytes, &mut needed) } == 0 {
                return Err(crate::Error::new(crate::ErrorKind::System, "EnumProcessModules failed"));
            }
            let count = needed as usize / std::mem::size_of::<windows_sys::Win32::Foundation::HMODULE>();
            if count <= modules.len() {
                modules.truncate(count);
                return Ok(Self(modules));
            }
            modules.resize(count, std::ptr::null_mut());
        }
    }

    /// The function `name` exports under (see the module documentation), null when no module does.
    fn find(&self, name: &str) -> *const c_void {
        use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
        let export = match name.strip_suffix("_nid_postfix") {
            Some(stripped) => crate::agc::nid::nid(stripped),
            None => name.to_owned(),
        };
        let export = std::ffi::CString::new(export).expect("export names have no NUL");
        self.0
            .iter()
            // SAFETY: loaded module handles and a NUL-terminated name.
            .find_map(|&module| unsafe { GetProcAddress(module, export.as_ptr().cast()) })
            .map_or(std::ptr::null(), |function| function as *const c_void)
    }
}
