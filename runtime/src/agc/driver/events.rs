//! The guest's graphics events (`sceAgcDriverAddEqEvent`; PLAN P3.4 step 3d): event queues
//! registered for a GPU queue's end-of-pipe interrupts, triggered through libkernel. Registrations
//! change rarely and are read at every interrupt: they live in atomic slots, and a deleted one is
//! leaked rather than freed, so a reader that loaded it may still use it.

use super::anyps5::{EventQueues, EVFILT_GRAPHICS_CORE, EV_ADD, EV_CLEAR};
use super::flips::{interrupt_data, Failure, Outcome};
use super::interface::result;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Graphics events registered at once.
const SLOTS: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Registration {
    queue_handle: i64,
    queue: u32,
    ident: u64,
}

/// The event queues waiting for each GPU queue's interrupts.
pub struct Registrations {
    slots: [AtomicPtr<Registration>; SLOTS],
}

impl Default for Registrations {
    fn default() -> Self {
        Self { slots: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())) }
    }
}

impl Registrations {
    fn each(&self) -> impl Iterator<Item = (&AtomicPtr<Registration>, Registration)> {
        self.slots.iter().filter_map(|slot| {
            let pointer = slot.load(Ordering::Acquire);
            // SAFETY: registrations are never freed.
            (!pointer.is_null()).then(|| (slot, unsafe { *pointer }))
        })
    }

    /// Adds the event `ident` of event queue `queue_handle` for GPU queue `queue`'s interrupts.
    pub fn add(&self, queues: &dyn EventQueues, queue_handle: i64, queue: u32, ident: u64, udata: *mut c_void) -> Outcome<()> {
        if queue_handle == 0 {
            return Err(Failure::new(result::INVALID, "invalid event queue"));
        }
        if queues.add_data_event(queue_handle, ident, EVFILT_GRAPHICS_CORE, EV_ADD | EV_CLEAR, i64::from(queue), udata) != 0 {
            return Err(Failure::new(result::INVALID, "the event queue rejected the event"));
        }
        if self.each().any(|(_, registration)| registration.queue_handle == queue_handle && registration.ident == ident) {
            return Ok(());
        }
        let registration = Box::into_raw(Box::new(Registration { queue_handle, queue, ident }));
        for slot in &self.slots {
            if slot.compare_exchange(std::ptr::null_mut(), registration, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                return Ok(());
            }
        }
        // SAFETY: the registration just made, never published.
        drop(unsafe { Box::from_raw(registration) });
        Err(Failure::new(result::FAILED, format!("more than {SLOTS} graphics events are registered")))
    }

    pub fn delete(&self, queues: &dyn EventQueues, queue_handle: i64, ident: u64) -> Outcome<()> {
        if queue_handle == 0 {
            return Err(Failure::new(result::INVALID, "invalid event queue"));
        }
        if queues.delete(queue_handle, ident, EVFILT_GRAPHICS_CORE) != 0 {
            return Err(Failure::new(result::INVALID, "the event is not registered"));
        }
        for (slot, registration) in self.each() {
            if registration.queue_handle == queue_handle && registration.ident == ident {
                slot.store(std::ptr::null_mut(), Ordering::Release);
            }
        }
        Ok(())
    }

    /// Triggers the events registered for GPU queue `queue` with an interrupt of `context`.
    pub fn interrupt(&self, queues: &dyn EventQueues, queue: u32, context: u32) {
        for (_, registration) in self.each() {
            if registration.queue == queue {
                let mut data = interrupt_data(queue, context);
                queues.trigger(registration.queue_handle, registration.ident, EVFILT_GRAPHICS_CORE, &mut data);
            }
        }
    }
}
