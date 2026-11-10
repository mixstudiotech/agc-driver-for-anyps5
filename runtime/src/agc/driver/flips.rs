//! VideoOut's outputs and flips (PLAN P3.4 step 3d, decisions D21): registered outputs, flips
//! reserved under an EOP token, the delivery of a submission's flips and end-of-pipe interrupts in
//! the order the submission raised them, and rendering waits:
//! - a delivery first waits for flip-queue room for each of its flips, reserves them all, then
//!   readies flips and EOP flips in event order (the order VideoOut presents them); the room wait
//!   is skipped when the submission also readies an EOP-reserved flip, whose reservation holds a
//!   slot that only this same delivery frees;
//! - an EOP reservation runs on the calling thread: VideoOut holds its driver lock around the
//!   call, and its present loop takes that lock, so the reservation must never wait behind a
//!   delivery waiting for flip-queue room.
//!
//! Nothing here takes a lock. Outputs live in per-handle slots; an unregistered output is retired,
//! not freed, until the registry is dropped, so a reader holding its slot's pointer can still
//! clone it. VideoOut unregisters its outputs after stopping the GPU, so shutdown keeps identities.
//! EOP reservations live in token-indexed slots whose state moves by compare-and-swap.

use super::anyps5::EventQueues;
use super::interface::{self, result, DisplayBuffer, Error, FlipInfo, FlipRequest, RenderingWait, VideoOutput};
use super::events::Registrations;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// Why a flip call failed: the interface's result code and message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Failure {
    pub code: i32,
    pub message: String,
}

impl Failure {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(result::INVALID, message)
    }

    /// Reports the failure through the interface's error out-parameter; its code.
    ///
    /// # Safety
    /// `error` is null or points at an `AgcDriverError`.
    pub unsafe fn report(&self, error: *mut Error) -> i32 {
        // SAFETY: the caller's contract.
        unsafe { Error::report(error, &self.message) };
        self.code
    }
}

pub type Outcome<T> = std::result::Result<T, Failure>;

impl From<Failure> for crate::Error {
    fn from(failure: Failure) -> Self {
        let kind = if failure.code == result::SHUTDOWN { crate::ErrorKind::Shutdown } else { crate::ErrorKind::System };
        Self::new(kind, failure.message)
    }
}

impl From<crate::Error> for Failure {
    fn from(error: crate::Error) -> Self {
        Self::new(if error.kind == crate::ErrorKind::Shutdown { result::SHUTDOWN } else { result::FAILED }, error.message)
    }
}

fn checked(code: i32, error: &Error, what: &str) -> Outcome<()> {
    match code {
        result::OK => Ok(()),
        code => Err(Failure::new(code, format!("{what}: {}", error.text()))),
    }
}

/// A registered VideoOut output: its table, released when the last reference goes.
#[derive(Debug)]
pub struct Output(VideoOutput);

// SAFETY: VideoOut's outputs may be called on any thread (`AgcDriver.h`).
unsafe impl Send for Output {}
// SAFETY: as above.
unsafe impl Sync for Output {}

impl Drop for Output {
    fn drop(&mut self) {
        // SAFETY: a complete table (checked at registration), released once.
        unsafe { (self.0.release.expect("complete table"))(self.0.this) }
    }
}

impl Output {
    /// Reserves a flip; with `limited`, `None` when the flip queue is full.
    pub fn reserve(self: &Arc<Self>, info: &FlipInfo, limited: bool) -> Outcome<Option<Request>> {
        let mut request = std::ptr::null_mut();
        let mut error = Error::default();
        // SAFETY: a complete table; the out-parameters outlive the call.
        let code = unsafe { (self.0.reserve.expect("complete table"))(self.0.this, info, limited, &mut request, &mut error) };
        match code {
            result::FULL if limited => Ok(None),
            result::OK if !request.is_null() => Ok(Some(Request { output: self.clone(), handle: request })),
            result::OK => Err(Failure::new(result::FAILED, "VideoOut returned a null flip reservation")),
            code => Err(Failure::new(code, format!("VideoOut refused the flip: {}", error.text()))),
        }
    }

    pub fn wait_for_flip_room(&self) -> Outcome<()> {
        let mut error = Error::default();
        // SAFETY: a complete table; `error` outlives the call.
        let code = unsafe { (self.0.wait_for_flip_room.expect("complete table"))(self.0.this, &mut error) };
        checked(code, &error, "VideoOut's flip queue")
    }

    pub fn capture_rendering_wait(self: &Arc<Self>, index: u32) -> Outcome<Wait> {
        let mut wait = std::ptr::null_mut();
        let mut error = Error::default();
        // SAFETY: a complete table; the out-parameters outlive the call.
        let code = unsafe { (self.0.capture_rendering_wait.expect("complete table"))(self.0.this, index, &mut wait, &mut error) };
        checked(code, &error, "VideoOut's rendering wait")?;
        if wait.is_null() {
            return Err(Failure::new(result::FAILED, "VideoOut returned a null rendering wait"));
        }
        Ok(Wait { output: self.clone(), handle: wait })
    }

    pub fn describe_buffer(&self, index: i32) -> Option<DisplayBuffer> {
        let mut buffer = DisplayBuffer::default();
        // SAFETY: a complete table; `buffer` outlives the call.
        unsafe { (self.0.describe_buffer.expect("complete table"))(self.0.this, index, &mut buffer) }.then_some(buffer)
    }

    fn fail(&self, message: &CString) {
        // SAFETY: a complete table and a NUL-terminated message.
        unsafe { (self.0.fail.expect("complete table"))(self.0.this, message.as_ptr()) }
    }
}

/// A reserved flip: dropped before [`Request::ready`], its reservation is released.
pub struct Request {
    output: Arc<Output>,
    handle: *mut FlipRequest,
}

// SAFETY: VideoOut's requests may be used on any thread.
unsafe impl Send for Request {}

impl Request {
    /// The work before the flip is done: VideoOut queues it.
    pub fn ready(self, frame: u64) -> Outcome<()> {
        let mut error = Error::default();
        // SAFETY: a live request of this output's table.
        let code = unsafe { (self.output.0.flip_ready.expect("complete table"))(self.handle, frame, &mut error) };
        checked(code, &error, "VideoOut's flip")
    }

    fn fail(self, message: &CString) {
        // SAFETY: as `ready`.
        unsafe { (self.output.0.flip_fail.expect("complete table"))(self.handle, message.as_ptr()) }
    }

    /// The handle alone, for a slot that keeps it: the output reference goes, the reservation stays.
    fn into_handle(self) -> *mut FlipRequest {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is never used or dropped again; its output reference is dropped once here.
        drop(unsafe { std::ptr::read(&this.output) });
        this.handle
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        // SAFETY: a live request, released once.
        unsafe { (self.output.0.flip_release.expect("complete table"))(self.handle) }
    }
}

/// A captured wait for a display buffer to be off display.
pub struct Wait {
    output: Arc<Output>,
    handle: *mut RenderingWait,
}

// SAFETY: VideoOut's waits may be used on any thread.
unsafe impl Send for Wait {}

impl Wait {
    /// Waits at most `milliseconds`: whether the buffer is off display.
    pub fn wait_for(&self, milliseconds: u32) -> Outcome<bool> {
        let mut error = Error::default();
        // SAFETY: a live wait of this output's table.
        match unsafe { (self.output.0.rendering_wait_for.expect("complete table"))(self.handle, milliseconds, &mut error) } {
            1 => Ok(true),
            0 => Ok(false),
            code => Err(Failure::new(code, format!("VideoOut's rendering wait: {}", error.text()))),
        }
    }
}

impl Drop for Wait {
    fn drop(&mut self) {
        // SAFETY: a live wait, released once.
        unsafe { (self.output.0.rendering_wait_release.expect("complete table"))(self.handle) }
    }
}

/// Output handles VideoOut uses (`VIDEO_OUT_NUM_MAX` is smaller).
const OUTPUT_SLOTS: usize = 64;
/// EOP reservations outstanding at once; VideoOut's flip queue holds 16.
const EOP_SLOTS: usize = 256;

const EMPTY: u32 = 0;
/// A reservation is being made.
const CLAIMED: u32 = 1;
const READY: u32 = 2;
/// A delivery or an unregistration is taking it.
const TAKING: u32 = 3;

/// One EOP reservation, its fields valid while the state is `READY`.
#[derive(Default)]
struct EopSlot {
    state: AtomicU32,
    token: AtomicU32,
    handle: AtomicU32,
    index: AtomicI32,
    mode: AtomicU32,
    argument: AtomicI64,
    request: AtomicPtr<FlipRequest>,
}

impl EopSlot {
    fn info(&self) -> FlipInfo {
        FlipInfo {
            handle: self.handle.load(Ordering::Relaxed),
            index: self.index.load(Ordering::Relaxed),
            mode: self.mode.load(Ordering::Relaxed),
            reserved: 0,
            argument: self.argument.load(Ordering::Relaxed),
        }
    }

    /// Moves a `READY` slot that `matches` to `TAKING`: its token and request, the slot `EMPTY`.
    fn take(&self, matches: impl Fn(&Self) -> bool) -> Option<(FlipInfo, *mut FlipRequest)> {
        if self.state.load(Ordering::Acquire) != READY || !matches(self) {
            return None;
        }
        self.state.compare_exchange(READY, TAKING, Ordering::Acquire, Ordering::Relaxed).ok()?;
        if !matches(self) {
            self.state.store(READY, Ordering::Release);
            return None;
        }
        let taken = (self.info(), self.request.swap(std::ptr::null_mut(), Ordering::Relaxed));
        self.state.store(EMPTY, Ordering::Release);
        Some(taken)
    }
}

/// One event a submission raised, to deliver in order once its work is done.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    Flip(FlipInfo),
    /// An end-of-pipe interrupt: an EOP flip when `context` is a reserved token.
    Eop { queue: u32, context: u32 },
}

/// A node of the retired outputs.
struct Retired {
    output: *const Output,
    next: *mut Retired,
}

/// VideoOut's outputs and the flips reserved on them.
pub struct Flips {
    outputs: [AtomicPtr<Output>; OUTPUT_SLOTS],
    retired: AtomicPtr<Retired>,
    eop: Box<[EopSlot]>,
    next_token: AtomicU32,
    frames: AtomicU64,
    failure: OnceLock<String>,
    stopped: AtomicBool,
}

impl Default for Flips {
    fn default() -> Self {
        Self {
            outputs: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())),
            retired: AtomicPtr::new(std::ptr::null_mut()),
            eop: (0..EOP_SLOTS).map(|_| EopSlot::default()).collect(),
            next_token: AtomicU32::new(interface::EOP_FLIP_TOKEN_BASE),
            frames: AtomicU64::new(0),
            failure: OnceLock::new(),
            stopped: AtomicBool::new(false),
        }
    }
}

impl Flips {
    fn check_failure(&self) -> Outcome<()> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(Failure::new(result::SHUTDOWN, "the video output driver has stopped"));
        }
        match self.failure.get() {
            Some(message) => Err(Failure::new(result::FAILED, message.clone())),
            None => Ok(()),
        }
    }

    fn slot(&self, handle: u32) -> Outcome<&AtomicPtr<Output>> {
        self.outputs
            .get(handle as usize)
            .ok_or_else(|| Failure::invalid(format!("video output handle {handle} is out of range")))
    }

    /// The output registered as `handle`.
    pub fn output(&self, handle: u32) -> Outcome<Arc<Output>> {
        self.check_failure()?;
        let pointer = self.slot(handle)?.load(Ordering::Acquire);
        if pointer.is_null() {
            return Err(Failure::invalid(format!("video output {handle} is not registered")));
        }
        // SAFETY: a pointer `Arc::into_raw` made, whose count a slot or the retired list holds
        // until this registry is dropped, after every user is gone.
        unsafe {
            Arc::increment_strong_count(pointer);
            Ok(Arc::from_raw(pointer))
        }
    }

    pub fn register(&self, handle: u32, table: &VideoOutput) -> Outcome<()> {
        self.check_failure()?;
        if !table.complete() {
            return Err(Failure::invalid("the video output's table is incomplete"));
        }
        let output = Arc::into_raw(Arc::new(Output(*table))).cast_mut();
        if self.slot(handle)?.compare_exchange(std::ptr::null_mut(), output, Ordering::AcqRel, Ordering::Acquire).is_err() {
            // SAFETY: the reference just made, never shared; dropping it releases the output.
            drop(unsafe { Arc::from_raw(output) });
            return Err(Failure::invalid(format!("video output {handle} is already registered")));
        }
        Ok(())
    }

    /// Ends the registration of the output `this` registered as `handle`, and drops its EOP flips.
    pub fn unregister(&self, handle: u32, this: *mut std::ffi::c_void) -> Outcome<()> {
        let slot = self.slot(handle)?;
        let current = slot.load(Ordering::Acquire);
        // SAFETY: as in `output`: the slot's count keeps it alive.
        if current.is_null() || unsafe { (*current).0.this } != this {
            return Err(Failure::invalid(format!("video output {handle} is registered for another output")));
        }
        if slot.compare_exchange(current, std::ptr::null_mut(), Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(Failure::invalid(format!("video output {handle} changed during its unregistration")));
        }
        self.cancel_eop(handle, current);
        self.retire(current);
        Ok(())
    }

    fn cancel_eop(&self, handle: u32, output: *const Output) {
        for eop in self.eop.iter() {
            if let Some((_, request)) = eop.take(|slot| slot.handle.load(Ordering::Relaxed) == handle) {
                // SAFETY: the slot or retired list holds this output until the registry is dropped.
                let output = unsafe {
                    Arc::increment_strong_count(output);
                    Arc::from_raw(output)
                };
                drop(Request { output, handle: request });
            }
        }
    }

    fn retire(&self, output: *const Output) {
        let node = Box::into_raw(Box::new(Retired { output, next: std::ptr::null_mut() }));
        let mut head = self.retired.load(Ordering::Acquire);
        loop {
            // SAFETY: `node` is ours until it is published.
            unsafe { (*node).next = head };
            match self.retired.compare_exchange(head, node, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return,
                Err(current) => head = current,
            }
        }
    }

    fn next_frame(&self) -> u64 {
        self.frames.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// A CPU flip: waits for flip-queue room, then reserves and readies it.
    pub fn submit(&self, info: &FlipInfo) -> Outcome<()> {
        let output = self.output(info.handle)?;
        output.wait_for_flip_room()?;
        let request = output.reserve(info, false)?.ok_or_else(|| Failure::new(result::FAILED, "VideoOut refused an unlimited reservation"))?;
        request.ready(self.next_frame())
    }

    /// Reserves a flip that the EOP interrupt carrying the returned token readies: under the
    /// next token, or `at`. `Ok(None)` when the flip queue is full.
    pub fn reserve_eop(&self, info: &FlipInfo, at: Option<u32>) -> Outcome<Option<u32>> {
        self.check_failure()?;
        let token = match at {
            Some(token) => token,
            None => self.next_token.fetch_add(1, Ordering::Relaxed),
        };
        if !(interface::EOP_FLIP_TOKEN_BASE..=interface::EOP_FLIP_TOKEN_LIMIT).contains(&token) {
            return Err(Failure::invalid(format!("EOP flip token 0x{token:x} is out of range")));
        }
        let slot = &self.eop[(token - interface::EOP_FLIP_TOKEN_BASE) as usize % EOP_SLOTS];
        if slot.state.compare_exchange(EMPTY, CLAIMED, Ordering::Acquire, Ordering::Relaxed).is_err() {
            if slot.state.load(Ordering::Acquire) == READY && slot.token.load(Ordering::Relaxed) == token {
                let same = slot.info() == FlipInfo { reserved: 0, ..*info };
                return match same {
                    true => Err(Failure::new(result::ALREADY_RESERVED, "the EOP flip token is already reserved for this flip")),
                    false => Err(Failure::invalid("the EOP flip token is reserved for a different flip")),
                };
            }
            return Err(Failure::new(result::FAILED, format!("more than {EOP_SLOTS} EOP flips are outstanding")));
        }
        let release = |state| slot.state.store(state, Ordering::Release);
        let output = match self.output(info.handle) {
            Ok(output) => output,
            Err(failure) => {
                release(EMPTY);
                return Err(failure);
            }
        };
        match output.reserve(info, true) {
            Ok(Some(request)) => {
                slot.token.store(token, Ordering::Relaxed);
                slot.handle.store(info.handle, Ordering::Relaxed);
                slot.index.store(info.index, Ordering::Relaxed);
                slot.mode.store(info.mode, Ordering::Relaxed);
                slot.argument.store(info.argument, Ordering::Relaxed);
                let request = request.into_handle();
                slot.request.store(request, Ordering::Relaxed);
                release(READY);
                // Paired with shutdown's fence: its scan and this check cannot both miss the
                // other thread's store (READY versus stopped).
                std::sync::atomic::fence(Ordering::SeqCst);
                // Shutdown may have scanned this slot while the callback still held CLAIMED.
                // Keep the original output alive and cancel only this publication's token.
                if self.stopped.load(Ordering::Acquire) {
                    if let Some((_, request)) = slot.take(|slot| slot.token.load(Ordering::Relaxed) == token && slot.request.load(Ordering::Relaxed) == request) {
                        drop(Request { output, handle: request });
                    }
                    return Err(Failure::new(result::SHUTDOWN, "the video output driver has stopped"));
                }
                if at.is_some() {
                    self.next_token.fetch_max(token + 1, Ordering::Relaxed);
                }
                Ok(Some(token))
            }
            Ok(None) => {
                release(EMPTY);
                Ok(None)
            }
            Err(failure) => {
                release(EMPTY);
                Err(failure)
            }
        }
    }

    fn reserved(&self, token: u32) -> bool {
        if !(interface::EOP_FLIP_TOKEN_BASE..=interface::EOP_FLIP_TOKEN_LIMIT).contains(&token) {
            return false;
        }
        let slot = &self.eop[(token - interface::EOP_FLIP_TOKEN_BASE) as usize % EOP_SLOTS];
        slot.state.load(Ordering::Acquire) == READY && slot.token.load(Ordering::Relaxed) == token
    }

    fn take_eop(&self, token: u32) -> Option<Request> {
        if !self.reserved(token) {
            return None;
        }
        let slot = &self.eop[(token - interface::EOP_FLIP_TOKEN_BASE) as usize % EOP_SLOTS];
        let (info, request) = slot.take(|slot| slot.token.load(Ordering::Relaxed) == token)?;
        let output = self.output(info.handle).ok()?;
        Some(Request { output, handle: request })
    }

    /// Delivers the events of a submission whose work has completed, in order.
    pub fn deliver(&self, events: &[Event], queues: &dyn EventQueues, registrations: &Registrations) -> Outcome<()> {
        let readies_reserved = events.iter().any(|event| matches!(*event, Event::Eop { context, .. } if self.reserved(context)));
        if !readies_reserved {
            for event in events {
                if let Event::Flip(info) = event {
                    self.output(info.handle)?.wait_for_flip_room()?;
                }
            }
        }
        let mut flips = Vec::new();
        for event in events {
            if let Event::Flip(info) = event {
                let request = self.output(info.handle)?.reserve(info, false)?;
                flips.push(request.ok_or_else(|| Failure::new(result::FAILED, "VideoOut refused an unlimited reservation"))?);
            }
        }
        let mut flips = flips.into_iter();
        for event in events {
            match *event {
                Event::Flip(_) => flips.next().expect("one reservation per flip").ready(self.next_frame())?,
                Event::Eop { queue, context } => match self.take_eop(context) {
                    Some(_) if queue != 0 => {
                        return Err(Failure::new(result::FAILED, "EOP flips on async compute queues are not implemented"));
                    }
                    Some(request) => request.ready(self.next_frame())?,
                    None => registrations.interrupt(queues, queue, context),
                },
            }
        }
        Ok(())
    }

    /// The display buffer registered as `index` of output `handle`.
    pub fn describe_buffer(&self, handle: u32, index: i32) -> Outcome<Option<DisplayBuffer>> {
        Ok(self.output(handle)?.describe_buffer(index))
    }

    /// The wait until buffer `index` of output `handle` is off display.
    pub fn rendering_wait(&self, handle: u32, index: u32) -> Outcome<Wait> {
        self.output(handle)?.capture_rendering_wait(index)
    }

    /// Records the first failure (sticky) and fails every output and pending EOP flip with it.
    pub fn fail(&self, message: &str) {
        let first = self.failure.set(message.to_owned()).is_ok();
        let message = CString::new(message.replace('\0', " ")).expect("NULs replaced");
        for slot in &self.outputs {
            let pointer = slot.load(Ordering::Acquire);
            if !pointer.is_null() {
                // SAFETY: as in `output`.
                unsafe { (*pointer).fail(&message) };
            }
        }
        for eop in self.eop.iter() {
            if let Some((info, request)) = eop.take(|_| true) {
                let pointer = self.slot(info.handle).map_or(std::ptr::null_mut(), |slot| slot.load(Ordering::Acquire));
                if !pointer.is_null() {
                    // SAFETY: as in `output`.
                    let output = unsafe {
                        Arc::increment_strong_count(pointer);
                        Arc::from_raw(pointer)
                    };
                    Request { output, handle: request }.fail(&message);
                }
            }
        }
        if first {
            eprintln!("[agc] driver failure: {}", message.to_string_lossy());
        }
    }

    pub fn failure(&self) -> Option<&str> {
        self.failure.get().map(String::as_str)
    }

    /// Stops operations and cancels EOP reservations. VideoOut still owns registrations and
    /// unregisters them after its presenter has stopped; preserve their exact identities.
    pub fn shutdown(&self) {
        self.stopped.store(true, Ordering::Release);
        std::sync::atomic::fence(Ordering::SeqCst);
        for (handle, slot) in self.outputs.iter().enumerate() {
            let output = slot.load(Ordering::Acquire);
            if !output.is_null() {
                self.cancel_eop(handle as u32, output);
            }
        }
    }
}

impl Drop for Flips {
    fn drop(&mut self) {
        // Device stop alone does not join guest callers. Only the final owner's drop makes the
        // raw-load -> Arc acquisition gap quiescent; the process-owned driver lives until exit.
        self.shutdown();
        for slot in &self.outputs {
            let pointer = slot.swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !pointer.is_null() {
                self.retire(pointer);
            }
        }
        let mut node = self.retired.swap(std::ptr::null_mut(), Ordering::AcqRel);
        while !node.is_null() {
            // SAFETY: nodes `retire` published, each taken once here.
            let retired = unsafe { Box::from_raw(node) };
            // SAFETY: the count the slot held.
            drop(unsafe { Arc::from_raw(retired.output) });
            node = retired.next;
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/agc/driver/flips.rs"]
mod tests;

/// The interrupt `sceAgcDriverAddEqEvent` registrations receive: the queue in the low byte and
/// the context id from bit 16.
pub fn interrupt_data(queue: u32, context: u32) -> u64 {
    u64::from(queue as u8) | u64::from(context) << 16
}
