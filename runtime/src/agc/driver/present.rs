//! Native GPU presentation of VideoOut's window. Pixel conversion and overlay snapshots run
//! on the device worker; this thread owns only the shared D3D12 slots and swapchain.

use super::anyps5::Anyps5;
use super::flips::{Failure, Outcome};
use super::interface::{result, DisplayBuffer, HostDisplayHdr, HostHdrStatus, PresentationWindow};
use crate::present::DisplayBufferDesc;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

/// The state of the host display's HDR, updated by VideoOut and read by the title's queries.
/// Updates are rare, so each publishes a new snapshot; an old one is leaked, not freed, because a
/// reader may still be copying it.
pub struct HostHdr {
    status: AtomicPtr<HostHdrStatus>,
}

impl Default for HostHdr {
    fn default() -> Self {
        Self { status: AtomicPtr::new(Box::into_raw(Box::default())) }
    }
}

impl HostHdr {
    pub fn status(&self) -> HostHdrStatus {
        // SAFETY: snapshots are never freed.
        unsafe { *self.status.load(Ordering::Acquire) }
    }

    fn update(&self, change: impl Fn(HostHdrStatus) -> HostHdrStatus) {
        let mut current = self.status.load(Ordering::Acquire);
        loop {
            // SAFETY: snapshots are never freed.
            let before = unsafe { *current };
            let after = change(before);
            if after == before {
                return;
            }
            let next = Box::into_raw(Box::new(after));
            match self.status.compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return,
                Err(now) => {
                    // SAFETY: the snapshot just made, never published.
                    drop(unsafe { Box::from_raw(next) });
                    current = now;
                }
            }
        }
    }

    fn published(mut status: HostHdrStatus, display: HostDisplayHdr, surface_hdr10: bool) -> HostHdrStatus {
        let display_changed = status.display != display;
        let available = display.hdr && surface_hdr10 && std::env::var_os("APS5_FORCE_SDR").is_none();
        let availability_changed = available != status.available;
        status.display = display;
        status.surface_hdr10 = surface_hdr10;
        status.available = available;
        if display_changed {
            status.display_generation += 1;
        }
        if availability_changed || (available && display_changed) {
            status.generation += 1;
        }
        status
    }

    pub fn set_display(&self, display: HostDisplayHdr) {
        self.update(|status| Self::published(status, display, status.surface_hdr10));
    }

    pub fn set_surface_hdr10(&self, offered: bool) {
        self.update(|status| Self::published(status, status.display, offered));
    }

    pub fn note_tone_map_peak(&self, nits: f32) {
        self.update(|status| HostHdrStatus { reported_tone_map_peak: nits, ..status });
    }
}

impl From<&DisplayBuffer> for DisplayBufferDesc {
    fn from(buffer: &DisplayBuffer) -> Self {
        Self {
            address: buffer.address,
            pixel_format: buffer.pixel_format,
            width: buffer.width,
            height: buffer.height,
            tiling_mode: buffer.tiling_mode,
            pitch_in_pixel: buffer.pitch_in_pixel,
            dcc_address: buffer.dcc_address,
            dcc_clear_color: buffer.dcc_clear_color,
            alignment: buffer.alignment,
        }
    }
}


/// A frame selected by VideoOut while its main-buffer reuse reservation is still held.
pub struct Frame<'a> {
    pub buffer: Option<&'a DisplayBuffer>,
    pub plane: Option<&'a DisplayBuffer>,
    pub width: u32,
    pub height: u32,
}

pub struct Presentation;

impl Presentation {
    pub fn attach(window: &PresentationWindow, adapter: (u32, i32), _arena: Arc<Anyps5>) -> Outcome<()> {
        #[cfg(all(feature = "wddm", target_os = "windows"))]
        { native_window::attach(window, adapter) }
        #[cfg(not(all(feature = "wddm", target_os = "windows")))]
        { let _ = (window, adapter); Err(Failure::new(result::FAILED, "native presentation needs WDDM")) }
    }

    pub fn release(context: *mut c_void) -> bool {
        #[cfg(all(feature = "wddm", target_os = "windows"))]
        { native_window::release(context) }
        #[cfg(not(all(feature = "wddm", target_os = "windows")))]
        { let _ = context; false }
    }

    pub fn capture_overlay(buffer: Option<&DisplayBuffer>) -> Outcome<()> {
        #[cfg(all(feature = "wddm", target_os = "windows"))]
        { native_window::capture_overlay(buffer) }
        #[cfg(not(all(feature = "wddm", target_os = "windows")))]
        { let _ = buffer; Err(Failure::new(result::FAILED, "native presentation needs WDDM")) }
    }

    pub fn present(window: &PresentationWindow, frame: &Frame) -> Outcome<bool> {
        #[cfg(all(feature = "wddm", target_os = "windows"))]
        { native_window::present(window, frame) }
        #[cfg(not(all(feature = "wddm", target_os = "windows")))]
        { let _ = (window, frame); Err(Failure::new(result::FAILED, "native presentation needs WDDM")) }
    }
}

#[cfg(all(feature = "wddm", target_os = "windows"))]
mod native_window {
    use super::*;
    use crate::agc::driver::Driver;
    use crate::present::{Canvas, handoff::{self, NativeFrame, NativeTarget, NtHandle}};
    use crate::present::d3d12::{Adapter, D3d12Presenter, Output, PresenterConfig, SlotSource, SyncMode};
    use std::{cell::RefCell, time::Duration};

    struct Attached {
        context: *mut c_void,
        window: usize,
        adapter: (u32, i32),
        headless: bool,
        sync: SyncMode,
        debug: bool,
        presenter: Option<D3d12Presenter>,
        target: Option<NativeTarget>,
        generation: u64,
        canvas: Option<Canvas>,
        serial: u64,
        dump: Option<(std::path::PathBuf, Vec<u64>)>,
    }

    thread_local! { static ATTACHED: RefCell<Option<Attached>> = const { RefCell::new(None) }; }

    pub(super) fn attach(window: &PresentationWindow, adapter: (u32, i32)) -> Outcome<()> {
        if window.context.is_null() || window.native_window == 0 { return Err(Failure::new(result::INVALID, "attach needs a native window")); }
        let headless = match std::env::var("APS5_RUST_PRESENT").unwrap_or_default().as_str() {
            "" | "auto" | "d3d12" => false,
            "headless" => true,
            _ => return Err(Failure::new(result::INVALID, "APS5_RUST_PRESENT must be d3d12 or headless; CPU presentation was removed")),
        };
        let sync = match std::env::var("ANYPS5_PRESENT_MODE").unwrap_or_default().as_str() {
            "" | "fifo" | "fifo_latest_ready" => SyncMode::Fifo,
            "mailbox" => SyncMode::Mailbox,
            "immediate" => SyncMode::Immediate,
            _ => return Err(Failure::new(result::INVALID, "invalid ANYPS5_PRESENT_MODE")),
        };
        let debug = std::env::var("APS5_RUST_PRESENT_DEBUG").is_ok_and(|value| value == "1");
        let dump = std::env::var_os("APS5_AGC_DISPLAY_DUMP").map(|directory| {
            let frames = std::env::var("APS5_AGC_DISPLAY_DUMP_AT").unwrap_or_default().split(',').filter_map(|number| number.trim().parse().ok()).collect();
            (std::path::PathBuf::from(directory), frames)
        });
        eprintln!("[agc] presenting through {} (CPU pixel reads/conversion/upload=0)", if headless { "headless" } else { "native GPU + D3D12 shared slots" });
        ATTACHED.with(|slot| *slot.borrow_mut() = Some(Attached { context: window.context, window: window.native_window, adapter, headless, sync, debug, presenter: None, target: None, generation: 0, canvas: None, serial: 0, dump }));
        Ok(())
    }

    pub(super) fn release(context: *mut c_void) -> bool {
        ATTACHED.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.as_ref().is_some_and(|attached| attached.context == context) { *slot = None; true } else { false }
        })
    }

    pub(super) fn capture_overlay(buffer: Option<&DisplayBuffer>) -> Outcome<()> {
        ATTACHED.with(|slot| {
            let slot = slot.borrow();
            let attached = slot.as_ref().ok_or_else(|| Failure::new(result::INVALID, "overlay capture needs the attached presenter thread"))?;
            if attached.headless { return Ok(()) }
            Driver::get().host()?.service.capture_overlay(buffer.map(DisplayBufferDesc::from)).map_err(Failure::from)
        })
    }

    pub(super) fn present(window: &PresentationWindow, frame: &Frame) -> Outcome<bool> {
        ATTACHED.with(|slot| {
            let mut slot = slot.borrow_mut();
            let attached = slot.as_mut().filter(|attached| attached.context == window.context && attached.window == window.native_window)
                .ok_or_else(|| Failure::new(result::INVALID, "presentation came from a different window/thread"))?;
            attached.serial += 1;
            if attached.headless { return Ok(true) }
            let image = frame.buffer.or(frame.plane).map_or((frame.width, frame.height), |buffer| (buffer.width, buffer.height));
            let (mut width, mut height) = (frame.width, frame.height);
            if let Some(drawable) = window.drawable_size {
                unsafe { drawable(window.context, &mut width, &mut height); }
            }
            if width == 0 || height == 0 { return Ok(false) }
            let canvas = Canvas::fit(image.0, image.1, width, height).map_err(Failure::from)?;
            if attached.canvas != Some(canvas) {
                if let Some(previous) = attached.presenter.take() { previous.wait_copied(previous.frames()).map_err(Failure::from)?; drop(previous); }
                let mut config = PresenterConfig::new(canvas.width, canvas.height);
                config.sync = attached.sync; config.debug_layer = attached.debug; config.readback = attached.dump.is_some();
                let presenter = D3d12Presenter::create(Adapter::Luid(attached.adapter.0, attached.adapter.1), Output::Window(attached.window as *mut c_void), SlotSource::Native, config).map_err(Failure::from)?;
                attached.generation = handoff::next_generation();
                attached.target = Some(NativeTarget {
                    window: attached.window, generation: attached.generation, canvas, footprint: presenter.footprint(),
                    slots: presenter.slot_handles().into_iter().map(|handle| NtHandle::duplicate(handle as usize)).collect::<crate::Result<Vec<_>>>().map_err(Failure::from)?,
                    ready: NtHandle::duplicate(presenter.ready_fence_handle() as usize).map_err(Failure::from)?,
                    copied: NtHandle::duplicate(presenter.copy_fence_handle() as usize).map_err(Failure::from)?,
                });
                eprintln!("[agc] native display generation {}: canvas {}x{}, main {:?}, overlay {:?}, slots {}", attached.generation, canvas.width, canvas.height, frame.buffer.map(|buffer| (buffer.width, buffer.height)), frame.plane.map(|buffer| (buffer.width, buffer.height)), config.slots);
                attached.presenter = Some(presenter); attached.canvas = Some(canvas);
            }
            let presenter = attached.presenter.as_mut().expect("created above");
            if !presenter.wait_for_frame(Duration::from_millis(100)).map_err(Failure::from)? { return Ok(false) }
            let number = presenter.frames() + 1;
            Driver::get().host()?.service.present(attached.target.take(), NativeFrame { generation: attached.generation, number, buffer: frame.buffer.map(DisplayBufferDesc::from), plane: frame.plane.map(DisplayBufferDesc::from) }).map_err(Failure::from)?;
            presenter.present(number).map_err(Failure::from)?;
            if let Some((directory, frames)) = &attached.dump {
                if frames.contains(&attached.serial) {
                    let pixels = presenter.read_back(number).map_err(Failure::from)?;
                    let footprint = presenter.footprint();
                    std::fs::create_dir_all(directory).and_then(|()| std::fs::write(directory.join(format!("display-{}.bgra", attached.serial)), pixels))
                        .map_err(|error| Failure::new(result::FAILED, error.to_string()))?;
                    let metadata = format!("{{\"width\":{},\"height\":{},\"row_pitch\":{},\"cpu_converted\":false}}", footprint.width, footprint.height, footprint.row_pitch);
                    std::fs::write(directory.join(format!("display-{}.json", attached.serial)), metadata).map_err(|error| Failure::new(result::FAILED, error.to_string()))?;
                }
            }
            Ok(true)
        })
    }
}
