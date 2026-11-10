//! The thread that owns the device (PLAN P3.4 step 3b; decisions D15). The WDDM device
//! is single-threaded: guest threads queue command buffers, import releases and idle waits here
//! and go on. Finished work's events go to a second thread, because delivering them can wait for
//! flip-queue room for as long as VideoOut takes to present, and the GPU should keep being fed
//! meanwhile.
//!
//! [`Worker`] is the control flow, over the [`Engine`] the target opens (`Target::open`).
//! The device thread never waits for a guest thread: a submission whose display buffer is still on
//! screen is set aside and tried again, so import releases are always served.

use super::anyps5::EventQueues;
use super::events::Registrations;
use super::flips::{Event, Failure, Flips, Outcome};
use super::interface::result;
use crate::residency::GuestRange;
use crate::{Error, ErrorKind, Result};
use crate::present::{DisplayBufferDesc, handoff::{NativeFrame, NativeTarget}};
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::Duration;

/// How often the device thread looks at the GPU while work is outstanding.
const POLL_INTERVAL: Duration = Duration::from_micros(500);
/// Tasks queued at once: a guest that submits faster than the GPU runs blocks here.
const QUEUE_DEPTH: usize = 256;
/// How long a guest mapping change waits for its import release.
const RELEASE_TIMEOUT: Duration = Duration::from_secs(60);

/// One command buffer to run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Submission {
    pub queue: u32,
    /// The guest's command buffer: same VA as the host's.
    pub address: u64,
    pub dwords: u32,
    pub serial: u64,
}

/// What the device thread drives: the GPU, through the target's implementation.
pub trait Engine {
    /// Runs one submission, adding the events of work already finished. `Ok(false)` when it must
    /// be tried again later (its display buffer is on screen, its queue is not available).
    fn execute(&mut self, submission: &Submission, events: &mut Vec<Event>) -> Result<bool>;
    /// Adds the events of the work that has finished since the last call.
    fn poll(&mut self, events: &mut Vec<Event>) -> Result<()>;
    /// Whether the GPU is still running work whose events `poll` will report.
    fn busy(&self) -> bool;
    /// Notes what `submission` waits for as soon as it arrives, before it runs.
    fn note(&mut self, _submission: &Submission) {}
    /// Whether the last `execute` handed part of its work to the GPU, even if it set the rest
    /// aside.
    fn advanced(&self) -> bool {
        false
    }
    /// Releases the imports overlapping `range`: whether it released any.
    fn release(&mut self, range: GuestRange) -> bool;
    fn wait_idle(&mut self) -> Result<()>;
    /// Brings the device up to date with guest mapping changes, before submissions run.
    fn synchronize(&mut self) -> Result<()>;
    /// The device's adapter, for presenters on the same GPU: LUID low and high.
    fn adapter(&self) -> (u32, i32);
    /// The driver is stopping.
    fn close(&mut self) {}
    /// Converts the selected ready buffers on the device; completes source reads before return.
    fn present(&mut self, _target: Option<NativeTarget>, _frame: NativeFrame) -> Result<()> {
        Err(Error::unsupported("native presentation is not implemented by this target"))
    }
    /// Captures an overlay while VideoOut still holds its source's reuse reservation.
    fn capture_overlay(&mut self, _buffer: Option<DisplayBufferDesc>) -> Result<()> {
        Err(Error::unsupported("native overlay capture is not implemented by this target"))
    }
}

/// What other threads ask of the device thread.
#[derive(Debug)]
pub enum Task {
    Submit { queue: u32, address: u64, dwords: u32, accepted: SyncSender<()> },
    Release { range: GuestRange, reply: SyncSender<bool> },
    WaitIdle { reply: SyncSender<()> },
    Stop { reply: SyncSender<()> },
    Present { target: Option<NativeTarget>, frame: NativeFrame, reply: SyncSender<Result<()>> },
    CaptureOverlay { buffer: Option<DisplayBufferDesc>, reply: SyncSender<Result<()>> },
}

/// The device thread's control flow.
pub struct Worker {
    engine: Box<dyn Engine>,
    flips: Arc<Flips>,
    delivery: Sender<Delivered>,
    /// Submissions not yet accepted by the engine, in submission order.
    waiting: VecDeque<Submission>,
    idle: Vec<SyncSender<()>>,
    serial: u64,
    failed: bool,
    accepted: Vec<SyncSender<()>>,
}

impl Worker {
    fn new(engine: Box<dyn Engine>, flips: Arc<Flips>, delivery: Sender<Delivered>) -> Self {
        Self { engine, flips, delivery, waiting: VecDeque::new(), idle: Vec::new(), serial: 0, failed: false, accepted: Vec::new() }
    }

    fn busy(&self) -> bool {
        !self.failed && (!self.waiting.is_empty() || self.engine.busy())
    }

    /// Serves `tasks` until told to stop or every sender is gone.
    fn run(mut self, tasks: Receiver<Task>) {
        let reply = 'run: loop {
            let mut next = if self.busy() {
                match tasks.recv_timeout(POLL_INTERVAL) {
                    Ok(task) => Some(task),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => break 'run None,
                }
            } else {
                match tasks.recv() {
                    Ok(task) => Some(task),
                    Err(_) => break 'run None,
                }
            };
            // Everything queued already, then the GPU.
            while let Some(task) = next {
                if let Task::Stop { reply } = task {
                    break 'run Some(reply);
                }
                self.take(task);
                next = tasks.try_recv().ok();
            }
            self.advance();
            for accepted in self.accepted.drain(..) {
                let _ = accepted.send(());
            }
        };
        self.engine.close();
        for waiter in self.idle.drain(..) {
            let _ = waiter.send(());
        }
        // Answered by the delivery thread, once it has delivered everything before this.
        if let Some(reply) = reply {
            let _ = self.delivery.send(Delivered::Stopped(reply));
        }
    }

    fn take(&mut self, task: Task) {
        match task {
            Task::Submit { queue, address, dwords, accepted } => {
                if !self.failed {
                    self.serial += 1;
                    let submission = Submission { queue, address, dwords, serial: self.serial };
                    self.engine.note(&submission);
                    self.waiting.push_back(submission);
                }
                self.accepted.push(accepted);
            }
            Task::Release { range, reply } => {
                let _ = reply.send(self.engine.release(range));
            }
            Task::WaitIdle { reply } => self.idle.push(reply),
            Task::Present { target, frame, reply } => {
                let outcome = if self.failed { Err(Error::new(ErrorKind::Shutdown, "the device worker has stopped")) } else { self.engine.present(target, frame) };
                if let Err(error) = &outcome { self.fail(Error::new(error.kind, error.message.clone())); }
                let _ = reply.send(outcome);
            }
            Task::CaptureOverlay { buffer, reply } => {
                let outcome = if self.failed { Err(Error::new(ErrorKind::Shutdown, "the device worker has stopped")) } else { self.engine.capture_overlay(buffer) };
                if let Err(error) = &outcome { self.fail(Error::new(error.kind, error.message.clone())); }
                let _ = reply.send(outcome);
            }
            Task::Stop { .. } => unreachable!("`run` handles it"),
        }
    }

    fn advance(&mut self) {
        if !self.failed {
            if let Err(error) = self.run_waiting().and_then(|()| self.collect()) {
                self.fail(error);
            }
        }
        if !self.busy() && !self.idle.is_empty() {
            if !self.failed {
                if let Err(error) = self.engine.wait_idle() {
                    self.fail(error);
                }
            }
            for waiter in self.idle.drain(..) {
                let _ = waiter.send(());
            }
        }
    }

    /// Offers the waiting submissions to the engine in order; one it sets aside holds back the
    /// later ones of its queue, as a GPU queue would, and no others.
    fn run_waiting(&mut self) -> Result<()> {
        if self.waiting.is_empty() {
            return Ok(());
        }
        self.engine.synchronize()?;
        // What one queue submits can settle what another waits for: passes repeat until one
        // moves nothing, so a chain of waits does not cost a poll interval per link.
        loop {
            let mut moved = false;
            let mut blocked: Vec<u32> = Vec::new();
            for _ in 0..self.waiting.len() {
                let submission = self.waiting.pop_front().expect("counted");
                if blocked.contains(&submission.queue) {
                    self.waiting.push_back(submission);
                    continue;
                }
                let mut events = Vec::new();
                if self.engine.execute(&submission, &mut events)? {
                    self.deliver(events);
                    moved = true;
                } else {
                    moved |= self.engine.advanced();
                    blocked.push(submission.queue);
                    self.waiting.push_back(submission);
                }
            }
            if !moved {
                return Ok(());
            }
        }
    }

    fn collect(&mut self) -> Result<()> {
        let mut events = Vec::new();
        self.engine.poll(&mut events)?;
        self.deliver(events);
        Ok(())
    }

    fn deliver(&self, events: Vec<Event>) {
        if !events.is_empty() {
            let _ = self.delivery.send(Delivered::Events(events));
        }
    }

    fn fail(&mut self, error: Error) {
        self.failed = true;
        self.waiting.clear();
        // VideoOut cancels its waits when the process exits. This ends device work without
        // turning an orderly shutdown into a sticky failure of every output.
        if error.kind != ErrorKind::Shutdown {
            self.flips.fail(&error.message);
        }
    }
}

/// What the delivery thread receives, in order.
enum Delivered {
    Events(Vec<Event>),
    /// The device thread has stopped: nothing follows.
    Stopped(SyncSender<()>),
}

/// The thread that delivers finished work's events, in the order they arrive.
struct Delivery {
    flips: Arc<Flips>,
    queues: Arc<dyn EventQueues>,
    registrations: Arc<Registrations>,
}

impl Delivery {
    fn run(self, batches: Receiver<Delivered>) {
        let mut exiting = false;
        for delivered in batches {
            match delivered {
                Delivered::Stopped(reply) => {
                    let _ = reply.send(());
                    return;
                }
                // The process is exiting: nothing is left to tell.
                Delivered::Events(_) if exiting => {}
                Delivered::Events(events) => match self.flips.deliver(&events, &*self.queues, &self.registrations) {
                    Ok(()) => {}
                    Err(Failure { code: result::SHUTDOWN, .. }) => exiting = true,
                    Err(failure) => self.flips.fail(&failure.message),
                },
            }
        }
    }
}

/// The handle other threads have on the device thread.
pub struct DeviceService {
    tasks: SyncSender<Task>,
    thread: ThreadId,
}

impl DeviceService {
    /// Starts the device and delivery threads; `open` makes the engine on the device thread, and
    /// its error is returned here.
    pub fn start<F>(open: F, flips: Arc<Flips>, queues: Arc<dyn EventQueues>, registrations: Arc<Registrations>) -> Result<Self>
    where
        F: FnOnce() -> Result<Box<dyn Engine>> + Send + 'static,
    {
        let spawn_failed = |error: std::io::Error| Error::new(ErrorKind::System, format!("libSceAgcDriver: cannot start a thread: {error}"));
        let (tasks, receiver) = mpsc::sync_channel(QUEUE_DEPTH);
        let (delivery, batches) = mpsc::channel();
        let delivering = Delivery { flips: flips.clone(), queues, registrations };
        std::thread::Builder::new().name("agc-delivery".into()).spawn(move || delivering.run(batches)).map_err(spawn_failed)?;
        let (opened, outcome) = mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("agc-device".into())
            .spawn(move || match open() {
                Ok(engine) => {
                    let _ = opened.send(Ok(()));
                    Worker::new(engine, flips, delivery).run(receiver);
                }
                Err(error) => {
                    let _ = opened.send(Err(error));
                }
            })
            .map_err(spawn_failed)?;
        outcome.recv().map_err(|_| Error::new(ErrorKind::System, "libSceAgcDriver: the device thread ended while opening"))??;
        Ok(Self { tasks, thread: thread.thread().id() })
    }

    fn stopped() -> Failure {
        Failure::new(result::SHUTDOWN, "the device thread has stopped")
    }

    /// Queues a command buffer and waits until the device thread has taken it up: submitted, set
    /// aside behind a wait, or failed. It is not waiting for the GPU.
    pub fn submit(&self, queue: u32, address: u64, dwords: u32) -> Outcome<()> {
        let (accepted, answer) = mpsc::sync_channel(1);
        self.tasks.send(Task::Submit { queue, address, dwords, accepted }).map_err(|_| Self::stopped())?;
        answer.recv().map_err(|_| Self::stopped())
    }

    /// Releases the imports overlapping `range`, on the device thread: whether it released any.
    /// Refused on the device thread itself, which cannot wait for itself.
    pub fn release(&self, range: GuestRange) -> bool {
        if std::thread::current().id() == self.thread {
            return false;
        }
        let (reply, answer) = mpsc::sync_channel(1);
        if self.tasks.send(Task::Release { range, reply }).is_err() {
            return false;
        }
        answer.recv_timeout(RELEASE_TIMEOUT).unwrap_or(false)
    }

    /// Waits until the work submitted so far has finished.
    pub fn wait_idle(&self) {
        let (reply, done) = mpsc::sync_channel(1);
        if self.tasks.send(Task::WaitIdle { reply }).is_ok() {
            let _ = done.recv();
        }
    }

    /// Stops the device thread once it has closed the device.
    pub fn stop(&self) {
        let (reply, done) = mpsc::sync_channel(1);
        if self.tasks.send(Task::Stop { reply }).is_ok() {
            let _ = done.recv();
        }
    }

    pub fn present(&self, target: Option<NativeTarget>, frame: NativeFrame) -> Result<()> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.tasks.send(Task::Present { target, frame, reply }).map_err(|_| Error::new(ErrorKind::Shutdown, "the device thread has stopped"))?;
        answer.recv().map_err(|_| Error::new(ErrorKind::Shutdown, "the device thread has stopped"))?
    }

    pub fn capture_overlay(&self, buffer: Option<DisplayBufferDesc>) -> Result<()> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.tasks.send(Task::CaptureOverlay { buffer, reply }).map_err(|_| Error::new(ErrorKind::Shutdown, "the device thread has stopped"))?;
        answer.recv().map_err(|_| Error::new(ErrorKind::Shutdown, "the device thread has stopped"))?
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/agc/driver/device.rs"]
mod tests;
