//! Shared handles and exact generation/frame metadata passed with the device-worker request.
use super::{canvas::Canvas, d3d12::Footprint};
use std::{
    ptr,
    sync::atomic::{AtomicU64, Ordering},
};

/// An NT handle this process owns, closed on drop.
#[derive(Debug, Eq, PartialEq)]
pub struct NtHandle(usize);

impl NtHandle {
    /// A duplicate of the NT handle `handle` (a HANDLE value) that the new owner closes.
    #[cfg(all(feature = "wddm", target_os = "windows"))]
    pub fn duplicate(handle: usize) -> crate::Result<Self> {
        use windows_sys::Win32::{
            Foundation::{DuplicateHandle, DUPLICATE_SAME_ACCESS},
            System::Threading::GetCurrentProcess,
        };
        if handle == 0 {
            return Err(crate::Error::invalid("present: no NT handle to hand off"));
        }
        let mut copy = ptr::null_mut();
        let duplicated = unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                handle as *mut std::ffi::c_void,
                GetCurrentProcess(),
                &mut copy,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        };
        if duplicated == 0 || copy.is_null() {
            return Err(crate::Error::new(
                crate::ErrorKind::System,
                format!(
                    "present: DuplicateHandle failed ({})",
                    std::io::Error::last_os_error()
                ),
            ));
        }
        Ok(Self(copy as usize))
    }

    /// No handle, for targets built without a device in tests.
    pub fn none() -> Self {
        Self(0)
    }

    pub fn raw(&self) -> *mut std::ffi::c_void {
        self.0 as *mut std::ffi::c_void
    }
}

impl Drop for NtHandle {
    fn drop(&mut self) {
        #[cfg(all(feature = "wddm", target_os = "windows"))]
        if self.0 != 0 {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.raw());
            }
        }
    }
}

/// What the worker needs to convert flips for one presenter.
#[derive(Debug)]
pub struct NativeTarget {
    /// The window (HWND) the presenter shows.
    pub window: usize,
    /// Distinct for every target posted in this process; a flip converted for one generation is
    /// only presented while the presenter still shows it.
    pub generation: u64,
    pub canvas: Canvas,
    /// The slots' layout (offset 0, 256-byte row pitch).
    pub footprint: Footprint,
    /// Frame N goes to slot N mod `slots.len()`.
    pub slots: Vec<NtHandle>,
    pub ready: NtHandle,
    pub copied: NtHandle,
}

/// The exact buffers selected by VideoOut, and the presenter's exact frame/fence value.
#[derive(Clone, Copy, Debug)]
pub struct NativeFrame {
    pub generation: u64,
    pub number: u64,
    pub buffer: Option<super::DisplayBufferDesc>,
    pub plane: Option<super::DisplayBufferDesc>,
}

static GENERATION: AtomicU64 = AtomicU64::new(0);

/// A generation no earlier target used.
pub fn next_generation() -> u64 {
    GENERATION.fetch_add(1, Ordering::Relaxed) + 1
}
