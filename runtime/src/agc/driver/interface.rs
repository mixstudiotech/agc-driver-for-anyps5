//! The C interface between libSceAgcDriver and AnyPS5's other modules (decisions D21), as AnyPS5's
//! `core/libs/prx/libSceAgcDriver/include/AgcDriver.h` defines it.

use std::ffi::{c_char, c_void, CStr};

/// What the interface's functions return.
pub mod result {
    pub const OK: i32 = 0;
    /// A null, unregistered or out-of-range argument.
    pub const INVALID: i32 = -1;
    /// The driver or the output has failed; every later call fails too.
    pub const FAILED: i32 = -2;
    pub const SHUTDOWN: i32 = -3;
    /// A limited reservation found the flip queue full.
    pub const FULL: i32 = -4;
    /// The EOP flip token is already reserved, for this same flip.
    pub const ALREADY_RESERVED: i32 = -5;
}

/// Why a call failed (`AgcDriverError`): NUL-terminated, truncated to fit.
#[repr(C)]
pub struct Error {
    pub message: [c_char; 256],
}

impl Default for Error {
    fn default() -> Self {
        Self { message: [0; 256] }
    }
}

impl Error {
    /// Fills `error` (when the caller passed one) with `message`.
    ///
    /// # Safety
    /// `error` is null or points at an `AgcDriverError`.
    pub unsafe fn report(error: *mut Error, message: &str) {
        // SAFETY: the caller's contract.
        if let Some(error) = unsafe { error.as_mut() } {
            let bytes = message.as_bytes();
            let length = bytes.len().min(error.message.len() - 1);
            for (slot, &byte) in error.message.iter_mut().zip(&bytes[..length]) {
                *slot = byte as c_char;
            }
            error.message[length] = 0;
        }
    }

    /// The message the other side wrote.
    pub fn text(&self) -> String {
        let bytes: Vec<u8> = self.message.iter().take_while(|&&byte| byte != 0).map(|&byte| byte as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// The context ids an EOP interrupt carries to ready a reserved flip.
pub const EOP_FLIP_TOKEN_BASE: u32 = 0x0400_0000;
pub const EOP_FLIP_TOKEN_LIMIT: u32 = 0x07ff_ffff;

/// A flip of output `handle` to buffer `index` (-1: black) (`AgcDriverFlipInfo`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FlipInfo {
    pub handle: u32,
    pub index: i32,
    pub mode: u32,
    pub reserved: u32,
    pub argument: i64,
}

/// A display buffer, as VideoOut registered it (`AgcDriverDisplayBuffer`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DisplayBuffer {
    pub address: u64,
    pub pixel_format: u64,
    pub width: u32,
    pub height: u32,
    pub tiling_mode: u32,
    pub pitch_in_pixel: u32,
    pub dcc_address: u64,
    pub dcc_clear_color: u64,
    pub alignment: u32,
    pub reserved: u32,
}

/// A flip VideoOut reserved (`AgcDriverFlipRequest`).
#[repr(C)]
pub struct FlipRequest {
    _opaque: [u8; 0],
}

/// A captured wait for a display buffer to be off display (`AgcDriverRenderingWait`).
#[repr(C)]
pub struct RenderingWait {
    _opaque: [u8; 0],
}

pub type Reserve = unsafe extern "C" fn(this: *mut c_void, info: *const FlipInfo, limited: bool, request: *mut *mut FlipRequest, error: *mut Error) -> i32;
pub type WaitForFlipRoom = unsafe extern "C" fn(this: *mut c_void, error: *mut Error) -> i32;
pub type CaptureRenderingWait = unsafe extern "C" fn(this: *mut c_void, index: u32, wait: *mut *mut RenderingWait, error: *mut Error) -> i32;
pub type DescribeBuffer = unsafe extern "C" fn(this: *mut c_void, index: i32, buffer: *mut DisplayBuffer) -> bool;
pub type Fail = unsafe extern "C" fn(this: *mut c_void, message: *const c_char);
pub type Release = unsafe extern "C" fn(this: *mut c_void);
pub type FlipReady = unsafe extern "C" fn(request: *mut FlipRequest, frame: u64, error: *mut Error) -> i32;
pub type FlipFail = unsafe extern "C" fn(request: *mut FlipRequest, message: *const c_char);
pub type FlipRelease = unsafe extern "C" fn(request: *mut FlipRequest);
pub type RenderingWaitFor = unsafe extern "C" fn(wait: *mut RenderingWait, milliseconds: u32, error: *mut Error) -> i32;
pub type RenderingWaitRelease = unsafe extern "C" fn(wait: *mut RenderingWait);

/// A VideoOut output (`AgcDriverVideoOutput`): `this` (C's `self`) identifies it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VideoOutput {
    pub this: *mut c_void,
    pub reserve: Option<Reserve>,
    pub wait_for_flip_room: Option<WaitForFlipRoom>,
    pub capture_rendering_wait: Option<CaptureRenderingWait>,
    pub describe_buffer: Option<DescribeBuffer>,
    pub fail: Option<Fail>,
    pub release: Option<Release>,
    pub flip_ready: Option<FlipReady>,
    pub flip_fail: Option<FlipFail>,
    pub flip_release: Option<FlipRelease>,
    pub rendering_wait_for: Option<RenderingWaitFor>,
    pub rendering_wait_release: Option<RenderingWaitRelease>,
}

impl VideoOutput {
    /// Whether every function is present.
    pub fn complete(&self) -> bool {
        !self.this.is_null()
            && self.reserve.is_some()
            && self.wait_for_flip_room.is_some()
            && self.capture_rendering_wait.is_some()
            && self.describe_buffer.is_some()
            && self.fail.is_some()
            && self.release.is_some()
            && self.flip_ready.is_some()
            && self.flip_fail.is_some()
            && self.flip_release.is_some()
            && self.rendering_wait_for.is_some()
            && self.rendering_wait_release.is_some()
    }
}

/// What the driver presents with (`AGC_DRIVER_PRESENTATION_NATIVE_WINDOW`).
pub const PRESENTATION_NATIVE_WINDOW: u32 = 1;

/// The window VideoOut presents into (`AgcDriverPresentationWindow`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PresentationWindow {
    /// Identifies the window (`AgcDriverReleaseWindow`).
    pub context: *mut c_void,
    pub drawable_size: Option<unsafe extern "C" fn(context: *mut c_void, width: *mut u32, height: *mut u32)>,
    pub width: u32,
    pub height: u32,
    /// The HWND on Windows.
    pub native_window: usize,
}

/// `AgcDriverHostDisplayHdr`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HostDisplayHdr {
    pub hdr: bool,
    pub max_luminance: f32,
    pub min_luminance: f32,
    pub max_full_frame_luminance: f32,
}

/// `AgcDriverHostHdrStatus`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HostHdrStatus {
    pub available: bool,
    pub surface_hdr10: bool,
    pub display: HostDisplayHdr,
    pub generation: u64,
    pub display_generation: u64,
    pub reported_tone_map_peak: f32,
}

/// Why libc refused an import alias (`GuestArena::ImportRefusal`).
pub mod import {
    pub const MADE: u32 = 0;
    pub const NOT_GPU_VISIBLE: u32 = 1;
    pub const NOT_SECTION_MEMORY: u32 = 2;
    pub const NOT_WHOLE_PAGES: u32 = 3;
    pub const CROSSES_WINDOW: u32 = 4;
    pub const HOST_WRITE_OPEN: u32 = 5;
}

/// `GuestRange` (libc's `GuestAllocations::GuestPiece`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GuestRange {
    pub address: usize,
    pub bytes: usize,
}

/// `GuestSparseRange`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GuestSparseRange {
    pub address: usize,
    pub bytes: usize,
    pub resident: u32,
    pub reserved: u32,
}

/// libc's call into the driver before a guest mapping change over imports (`GuestImportRelease`).
pub type GuestImportRelease = unsafe extern "C" fn(user: *mut c_void, address: usize, bytes: usize) -> bool;

/// The message of a C string the other side passed, or a placeholder for null.
///
/// # Safety
/// `message` is null or a NUL-terminated string.
pub unsafe fn message(message: *const c_char) -> String {
    if message.is_null() {
        return "(no message)".to_string();
    }
    // SAFETY: the caller's contract.
    unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
}
