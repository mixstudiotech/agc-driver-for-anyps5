//! The host queues of one device: its guest residency and the work not yet reported as finished.

use super::{SubmitEvent, SubmitOutcome};
use crate::{
    residency::{GuestResidency, Queue, ResidencyDevice, ResidencyWindow},
    Error, Result,
};

/// The device as GPU execution uses it beyond residency: its host queues.
pub trait HostQueues: ResidencyDevice {
    /// Makes what is queued on `waiter` from now on wait, in the OS, until submission `serial` of
    /// `producer` has completed.
    fn wait_for_queue(&mut self, waiter: Queue, producer: Queue, serial: u64) -> Result<()>;
    /// Waits until submission `serial` on `queue` has completed.
    fn wait(&mut self, queue: Queue, serial: u64) -> Result<()>;
    fn open_queue(&mut self, _queue: Queue) -> Result<()> { Ok(()) }
    fn available(&mut self, _queue: Queue) -> bool { true }
    fn check_health(&mut self) -> Result<()> { Ok(()) }
    /// The GPU virtual address of the page fault the device reported, if any.
    fn fault_address(&self) -> Option<u64> { None }
    /// What the device reports of its state, for a failure's message.
    fn fault_state(&self) -> String { String::new() }
}

/// GPU execution on one device.
pub struct GpuQueues<W: ResidencyWindow> {
    pub residency: GuestResidency<W>,
    pub device: W::Device,
    pending: std::collections::VecDeque<PendingSubmission>,
}

struct PendingSubmission {
    queue: Queue,
    host_serial: u64,
    serial: u64,
    events: Vec<SubmitEvent>,
}

impl<W> GpuQueues<W>
where
    W: ResidencyWindow,
    W::Device: HostQueues,
{
    pub fn new(residency: GuestResidency<W>, device: W::Device) -> Self {
        Self { residency, device, pending: std::collections::VecDeque::new() }
    }

    /// `error` with what the residency knows of the address the device faulted at, if it
    /// reported one.
    fn with_fault_context(&mut self, error: Error) -> Error {
        let Some(address) = self.device.fault_address() else {
            return error;
        };
        let described = self.residency.describe_address(address);
        let state = self.device.fault_state();
        Error::new(error.kind, format!("{}; the faulting address 0x{address:x}: {described}; device state: {state}", error.message))
    }

    /// Notes a submission on `queue`: its events are reported once `host_serial` has finished.
    pub fn queued(&mut self, queue: Queue, host_serial: u64, serial: u64, events: Vec<SubmitEvent>) {
        self.pending.push_back(PendingSubmission { queue, host_serial, serial, events });
    }

    /// Whether any queued work's completion raises a flip of display buffer `index` of output
    /// `handle`.
    pub fn flip_pending(&self, handle: u32, index: i32) -> bool {
        self.pending.iter().any(|job| {
            job.events.iter().any(|event| match event {
                SubmitEvent::Flip(flip) => flip.handle == handle && flip.index == index,
                SubmitEvent::EopFlip(_) => true,
                _ => false,
            })
        })
    }

    /// Submissions queued that have not been reported as finished.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        self.device.wait_idle()
    }

    /// The outcomes of the work that has finished since the last call, in retirement order.
    pub fn poll(&mut self) -> Result<Vec<SubmitOutcome>> {
        if let Err(error) = self.device.check_health() {
            return Err(self.with_fault_context(error));
        }
        if self.device.is_lost() {
            let error = Error::new(crate::ErrorKind::DeviceLost, "GPU transport is quarantined");
            return Err(self.with_fault_context(error));
        }
        let mut outcomes = Vec::new();
        let mut remaining = std::collections::VecDeque::new();
        while let Some(job) = self.pending.pop_front() {
            if self.device.completed(job.queue) < job.host_serial {
                remaining.push_back(job);
            } else {
                outcomes.push(outcome(job.serial, job.events));
            }
        }
        self.pending = remaining;
        Ok(outcomes)
    }
}

fn outcome(serial: u64, events: Vec<SubmitEvent>) -> SubmitOutcome {
    let flips = events
        .iter()
        .filter_map(|event| match event {
            SubmitEvent::Flip(flip) => Some(flip.clone()),
            _ => None,
        })
        .collect();
    let eop_events = events.iter().filter_map(|event| event.interrupt().copied()).collect();
    SubmitOutcome { serial, flips, eop_events, published_bytes: 0, events, submitted_flips: 0 }
}

/// The WDDM device's queues.
#[cfg(all(feature = "wddm", target_os = "windows"))]
mod wddm {
    use super::*;
    use crate::{backend::wddm::QueueKind, residency::WddmWindow, Device};

    fn kind(queue: Queue) -> QueueKind {
        match queue {
            Queue::Graphics => QueueKind::Graphics,
            Queue::Compute => QueueKind::Compute,
            Queue::ComputeQueue(index) => QueueKind::ComputeQueue(index),
        }
    }

    impl HostQueues for Device {
        fn open_queue(&mut self, queue: Queue) -> Result<()> {
            match queue {
                Queue::Graphics => Ok(()),
                Queue::Compute => self.open_compute_queue().map(|_| ()),
                Queue::ComputeQueue(index) => self.open_guest_compute_queue(index).map(|_| ()),
            }
        }

        fn available(&mut self, queue: Queue) -> bool {
            self.can_queue_host_on(kind(queue))
        }

        fn check_health(&mut self) -> Result<()> {
            self.check_async_health()
        }

        fn fault_address(&self) -> Option<u64> {
            self.query_device_state().page_fault.map(|fault| fault.virtual_address)
        }

        fn fault_state(&self) -> String {
            self.query_device_state().to_string()
        }

        fn wait(&mut self, queue: Queue, serial: u64) -> Result<()> {
            self.wait_on(kind(queue), serial)
        }

        fn wait_for_queue(&mut self, waiter: Queue, producer: Queue, serial: u64) -> Result<()> {
            Device::wait_for_queue(self, kind(waiter), kind(producer), serial)
        }
    }

    /// GPU execution on WDDM.
    pub type LiveQueues = GpuQueues<WddmWindow>;
}

#[cfg(all(feature = "wddm", target_os = "windows"))]
pub use wddm::LiveQueues;
