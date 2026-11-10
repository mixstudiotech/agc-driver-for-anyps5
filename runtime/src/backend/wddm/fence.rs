//! CPU waits on WDDM monitored fences: the paging-queue fence and the hardware-queue progress
//! fence.

use super::Kmt;
use crate::{Error, ErrorKind, Result};
use std::{
    ptr,
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};
use windows_sys::{
    Wdk::Graphics::Direct3D::*,
    Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0},
        System::Threading::{CreateEventW, WaitForSingleObject},
    },
};

const FENCE_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// A monitored fence: the kernel object CPU waits name, and the read-only CPU mapping of its
/// current value that dxgkrnl returns when the paging queue or hardware queue is created.
#[derive(Clone, Copy)]
pub(super) struct MonitoredFence {
    pub(super) object: u32,
    pub(super) value: *const u64,
}

impl MonitoredFence {
    pub(super) const NONE: Self = Self {
        object: 0,
        value: ptr::null(),
    };

    pub(super) fn current(&self) -> Option<u64> {
        if self.value.is_null() {
            return None;
        }
        // The kernel and GPU update the value concurrently; an atomic load keeps the read whole.
        Some(unsafe { AtomicU64::from_ptr(self.value.cast_mut()) }.load(Ordering::Acquire))
    }
}

pub(super) struct KmtFenceWait<'a> {
    pub(super) device: u32,
    pub(super) fence: &'a MonitoredFence,
}

impl KmtFenceWait<'_> {
    /// Waits until a fence reaches `target`. When no event can be armed the GPU work is still in
    /// flight, so the fence value is polled until the same deadline. An error means completion is
    /// unconfirmed.
    pub(super) fn reach(&self, target: u64, timeout: Duration) -> Result<()> {
        if self.current().is_some_and(|v| v >= target) {
            return Ok(());
        }
        let setup = match self.arm(target) {
            Ok(armed) => {
                return if self.wait(armed, timeout) || self.current().is_some_and(|v| v >= target) {
                    Ok(())
                } else {
                    Err(Error::new(
                        ErrorKind::DeviceLost,
                        format!(
                            "GPU/paging fence did not reach {target} in {} ms",
                            timeout.as_millis()
                        ),
                    ))
                };
            }
            Err(error) => error,
        };
        let deadline = Instant::now() + timeout;
        loop {
            match self.current() {
                Some(value) if value >= target => return Ok(()),
                Some(_) if Instant::now() < deadline => thread::sleep(FENCE_POLL_INTERVAL),
                Some(value) => {
                    return Err(Error::new(
                        ErrorKind::DeviceLost,
                        format!(
                            "{setup}; polling saw fence value {value}, not {target}, within {} ms",
                            timeout.as_millis()
                        ),
                    ))
                }
                None => {
                    return Err(Error::new(
                        ErrorKind::DeviceLost,
                        format!("{setup}; the fence has no CPU-visible value to poll"),
                    ))
                }
            }
        }
    }

    fn arm(&self, target: u64) -> Result<Event> {
        let handle = unsafe { CreateEventW(ptr::null(), 0, 0, ptr::null()) };
        if handle.is_null() {
            return Err(Error::new(ErrorKind::System, "CreateEventW failed"));
        }
        let event = Event(handle);
        let request = D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU {
            hDevice: self.device,
            ObjectCount: 1,
            ObjectHandleArray: &self.fence.object,
            FenceValueArray: &target,
            hAsyncEvent: event.0,
            ..Default::default()
        };
        Kmt::status(
            unsafe { D3DKMTWaitForSynchronizationObjectFromCpu(&request) },
            "WaitForSynchronizationObjectFromCpu",
        )?;
        Ok(event)
    }

    fn wait(&self, armed: Event, timeout: Duration) -> bool {
        // Below INFINITE (u32::MAX), which would never time out.
        let millis = timeout.as_millis().min(u128::from(u32::MAX - 1)) as u32;
        if unsafe { WaitForSingleObject(armed.0, millis) } == WAIT_OBJECT_0 {
            true
        } else {
            // The kernel may still signal the event after the deadline; keep it open.
            std::mem::forget(armed);
            false
        }
    }

    fn current(&self) -> Option<u64> {
        self.fence.current()
    }
}

pub(super) struct Event(HANDLE);

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}
