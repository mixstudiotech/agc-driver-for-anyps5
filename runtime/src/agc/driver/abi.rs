//! The libSceAgcDriver exports (PLAN P3.4 step 3f, decisions D15, D21): the guest's
//! `sceAgcDriver*` functions in the guest's convention, and the `AgcDriver*` interface of
//! AnyPS5's other modules in the host's (`AgcDriver.h`). Debug and tooling services AnyPS5 has no
//! implementation of answer as the C++ module did; a failure the C++ module would have thrown is
//! reported and returned as an error code.

use super::flips::{Failure, Outcome};
use super::interface::{self, result, DisplayBuffer, Error, FlipInfo, HostDisplayHdr, HostHdrStatus, PresentationWindow, VideoOutput};
use super::present::{Frame, Presentation};
use super::Driver;
use crate::agc::abi::report;
use crate::agc::workloads;
use crate::agc_export;
use crate::agc::target::{Kind, Notice, Op};
use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};

const FEATURE_UNAVAILABLE: i32 = 0x8a6d_0001_u32 as i32;
const NULL_OUTPUT: i32 = 0x8a6d_0003_u32 as i32;
const MODULE_REGISTRATION_FAILED: i32 = 0x8a6d_0003_u32 as i32;
const DEBUG_UNAVAILABLE: i32 = 0x8a6c_1000_u32 as i32;
const RESOURCES_UNAVAILABLE: i32 = 0x8a6c_9018_u32 as i32;
const WORKLOAD_STREAM_INVALID_ID: i32 = 0x8a6c_0033_u32 as i32;
const WORKLOAD_STREAM_NULL: i32 = 0x8a6c_0035_u32 as i32;
const WORKLOAD_STREAM_NOT_REGISTERED: i32 = 0x8a6c_003a_u32 as i32;
/// What a guest call the console library would have thrown from returns.
const INVALID: i32 = 0x8a6c_0001_u32 as i32;
const EINVAL: i32 = 0x8002_0065_u32 as i32;
const EINVAL_16: i32 = 22;

/// Graphics queue submissions; compute queues are `FIRST_COMPUTE_QUEUE..COMPUTE_QUEUE_END`.
const GRAPHICS_QUEUE: u32 = 0;
const FIRST_COMPUTE_QUEUE: u32 = 0x20;
const COMPUTE_QUEUE_END: u32 = 0x59;
const MULTI_COMPUTE_QUEUE_END: u32 = 0x58;
/// Dwords of command-buffer pointers a multi-submit may pass.
const MAX_MULTI_SUBMIT: u32 = u32::MAX / 8;

/// The guest's description of a command buffer.
#[repr(C)]
pub struct Packet {
    address: *const u32,
    dwords: u32,
    flags: u8,
    reserved: [u8; 3],
}

/// `KernelEvent`.
#[repr(C)]
pub struct KernelEvent {
    ident: usize,
    filter: i16,
    flags: u16,
    fflags: u32,
    data: isize,
    udata: *mut c_void,
}

/// One guest call, by name for reports.
struct Call(&'static str);

impl Call {
    fn fail(&self, failure: Failure) -> i32 {
        report(self.0, &failure.message);
        INVALID
    }

    /// Queues `dwords` command words at `address` for `queue`.
    ///
    /// # Safety
    /// The guest passes a command buffer of that many dwords.
    unsafe fn submit(&self, queue: u32, address: *const u32, dwords: u32) -> i32 {
        if dwords != 0 && (address.is_null() || address as usize % 4 != 0) {
            return self.fail(Failure::new(result::INVALID, "null or misaligned command buffer"));
        }
        let driver = Driver::get();
        let host = match driver.host() {
            Ok(host) => host,
            Err(failure) => return self.fail(failure),
        };
        host.arena_note_submit();
        if let Some(message) = driver.flips.failure() {
            return self.fail(Failure::new(result::FAILED, message));
        }
        if dwords == 0 {
            return 0;
        }
        let family = if queue == GRAPHICS_QUEUE { crate::agc::target::Queue::Graphics } else { crate::agc::target::Queue::Compute };
        crate::agc::abi::agc().recordings.with_recording_at(address as u64, |recording| recording.recorder.submitted(family));
        match host.service.submit(queue, address as u64, dwords) {
            Ok(()) => 0,
            Err(failure) => self.fail(failure),
        }
    }

    /// # Safety
    /// `packet` is null or the guest's packet.
    unsafe fn submit_packet(&self, queue: u32, packet: *const Packet) -> i32 {
        // SAFETY: the caller's contract.
        let Some(packet) = (unsafe { packet.as_ref() }) else {
            return self.fail(Failure::new(result::INVALID, "null packet"));
        };
        if packet.flags != 0 {
            return self.fail(Failure::new(result::INVALID, "nonzero submission flags are not implemented"));
        }
        // SAFETY: the guest's packet describes its command buffer.
        unsafe { self.submit(queue, packet.address, packet.dwords) }
    }

    /// # Safety
    /// The arrays hold `count` entries.
    unsafe fn submit_many(&self, queue: u32, addresses: *const *const u32, sizes: *const u32, count: u32) -> i32 {
        if count == 0 {
            return 0;
        }
        if count > MAX_MULTI_SUBMIT || addresses.is_null() || sizes.is_null() {
            return self.fail(Failure::new(result::INVALID, "invalid multi-submit arrays"));
        }
        // SAFETY: the caller's contract.
        let (addresses, sizes) = unsafe { (std::slice::from_raw_parts(addresses, count as usize), std::slice::from_raw_parts(sizes, count as usize)) };
        for (&address, &dwords) in addresses.iter().zip(sizes) {
            if dwords != 0 && address.is_null() {
                return self.fail(Failure::new(result::INVALID, "null multi-submit command buffer"));
            }
            // SAFETY: each entry describes a command buffer.
            let submitted = unsafe { self.submit(queue, address, dwords) };
            if submitted != 0 {
                return submitted;
            }
        }
        0
    }

    fn compute_queue(&self, queue: u32, end: u32) -> Result<(), i32> {
        if (FIRST_COMPUTE_QUEUE..end).contains(&queue) {
            Ok(())
        } else {
            Err(self.fail(Failure::new(result::INVALID, format!("unsupported compute queue 0x{queue:x}"))))
        }
    }

    /// The queue a guest queue context names: graphics or a compute queue.
    ///
    /// # Safety
    /// `context` points at a queue context.
    unsafe fn context_queue(&self, context: *const c_void) -> Result<u32, i32> {
        // SAFETY: the queue id is the context's second dword.
        let queue = unsafe { context.cast::<u32>().add(1).read_unaligned() };
        if queue == GRAPHICS_QUEUE {
            return Ok(queue);
        }
        self.compute_queue(queue, MULTI_COMPUTE_QUEUE_END).map(|()| queue)
    }

    fn event_queues(&self, body: impl FnOnce(&super::Host) -> Outcome<()>) -> i32 {
        let outcome = Driver::get().host().and_then(body);
        match outcome {
            Ok(()) => 0,
            Err(failure) => self.fail(failure),
        }
    }

    fn event(&self, event: *const KernelEvent) -> Result<&KernelEvent, i32> {
        if event.is_null() || event as usize % std::mem::align_of::<KernelEvent>() != 0 {
            return Err(self.fail(Failure::new(result::INVALID, "null or misaligned event")));
        }
        // SAFETY: the guest passes one of its events.
        Ok(unsafe { &*event })
    }
}

impl super::Host {
    /// Tells libkernel the title is submitting GPU work.
    fn arena_note_submit(&self) {
        super::anyps5::EventQueues::note_gpu_submit(&*self.arena);
    }
}

agc_export! { fn sceAgcDriverSubmitDcb(packet: *const Packet) -> i32 {
    unsafe { Call("sceAgcDriverSubmitDcb").submit_packet(GRAPHICS_QUEUE, packet) }
} }
agc_export! { fn sceAgcDriverAgrSubmitDcb(packet: *const Packet) -> i32 {
    unsafe { Call("sceAgcDriverAgrSubmitDcb").submit_packet(GRAPHICS_QUEUE, packet) }
} }
agc_export! { fn sceAgcDriverSubmitMultiDcbs(addresses: *const *const u32, sizes: *const u32, count: u32) -> i32 {
    unsafe { Call("sceAgcDriverSubmitMultiDcbs").submit_many(GRAPHICS_QUEUE, addresses, sizes, count) }
} }
agc_export! { fn sceAgcDriverAgrSubmitMultiDcbs(addresses: *const *const u32, sizes: *const u32, count: u32) -> i32 {
    unsafe { Call("sceAgcDriverAgrSubmitMultiDcbs").submit_many(GRAPHICS_QUEUE, addresses, sizes, count) }
} }
agc_export! { fn sceAgcDriverSubmitAcb(queue: u32, packet: *const Packet) -> i32 {
    let call = Call("sceAgcDriverSubmitAcb");
    match call.compute_queue(queue, COMPUTE_QUEUE_END) {
        Ok(()) => unsafe { call.submit_packet(queue, packet) },
        Err(code) => code,
    }
} }
agc_export! { fn sceAgcDriverSubmitMultiAcbs(queue: u32, addresses: *const *const u32, sizes: *const u32, count: u32) -> i32 {
    let call = Call("sceAgcDriverSubmitMultiAcbs");
    if count == 0 {
        return 0;
    }
    match call.compute_queue(queue, MULTI_COMPUTE_QUEUE_END) {
        Ok(()) => unsafe { call.submit_many(queue, addresses, sizes, count) },
        Err(code) => code,
    }
} }
agc_export! { fn sceAgcDriverSubmitCommandBuffer(context: *const c_void, packet: *const Packet) -> i32 {
    let call = Call("sceAgcDriverSubmitCommandBuffer");
    if context.is_null() || packet.is_null() {
        return EINVAL_16;
    }
    match unsafe { call.context_queue(context) } {
        Ok(queue) => unsafe { call.submit_packet(queue, packet) },
        Err(code) => code,
    }
} }
agc_export! { fn sceAgcDriverSubmitMultiCommandBuffers(context: *const c_void, addresses: *const *const u32, sizes: *const u32, count: u32) -> i32 {
    let call = Call("sceAgcDriverSubmitMultiCommandBuffers");
    if count == 0 {
        return 0;
    }
    if context.is_null() || addresses.is_null() || sizes.is_null() {
        return EINVAL_16;
    }
    match unsafe { call.context_queue(context) } {
        Ok(queue) => unsafe { call.submit_many(queue, addresses, sizes, count) },
        Err(code) => code,
    }
} }

/// The submission exports.
pub const SUBMISSIONS: &[&str] = &[
    "sceAgcDriverSubmitDcb", "sceAgcDriverAgrSubmitDcb", "sceAgcDriverSubmitMultiDcbs", "sceAgcDriverAgrSubmitMultiDcbs",
    "sceAgcDriverSubmitAcb", "sceAgcDriverSubmitMultiAcbs", "sceAgcDriverSubmitCommandBuffer", "sceAgcDriverSubmitMultiCommandBuffers",
];

agc_export! { fn sceAgcDriverAddEqEvent(queue: i64, id: i32, udata: *mut c_void) -> i32 {
    let call = Call("sceAgcDriverAddEqEvent");
    if id < 0 {
        return call.fail(Failure::new(result::INVALID, "negative event id"));
    }
    call.event_queues(|host| Driver::get().registrations.add(&*host.arena, queue, id as u32, id as u64, udata))
} }
agc_export! { fn sceAgcDriverDeleteEqEvent(queue: i64, id: i32) -> i32 {
    let call = Call("sceAgcDriverDeleteEqEvent");
    if id < 0 {
        return call.fail(Failure::new(result::INVALID, "negative event id"));
    }
    call.event_queues(|host| Driver::get().registrations.delete(&*host.arena, queue, id as u64))
} }
agc_export! { fn sceAgcDriverGetEqContextId(event: *const KernelEvent) -> u32 {
    let call = Call("sceAgcDriverGetEqContextId");
    match call.event(event) {
        Ok(event) if event.filter == -14 => (event.data as u64 >> 16) as u32,
        Ok(_) => {
            call.fail(Failure::new(result::INVALID, "not a graphics event"));
            0
        }
        Err(_) => 0,
    }
} }
agc_export! { fn sceAgcDriverGetEqEventType(event: *const KernelEvent) -> i32 {
    match Call("sceAgcDriverGetEqEventType").event(event) {
        Ok(event) => event.data as i32,
        Err(code) => code,
    }
} }

/// The event queue exports.
pub const EVENTS: &[&str] = &["sceAgcDriverAddEqEvent", "sceAgcDriverDeleteEqEvent", "sceAgcDriverGetEqContextId", "sceAgcDriverGetEqEventType"];

agc_export! { fn sceAgcDriverRegisterWorkloadStream(id: u32, stream: *const c_void) -> i32 {
    if !(1..32).contains(&id) || workloads::registered(id) {
        return WORKLOAD_STREAM_INVALID_ID;
    }
    if stream.is_null() {
        return WORKLOAD_STREAM_NULL;
    }
    if workloads::claim(id) { 0 } else { WORKLOAD_STREAM_INVALID_ID }
} }
agc_export! { fn sceAgcDriverUnregisterWorkloadStream(id: u32) -> i32 {
    if !(1..32).contains(&id) {
        return WORKLOAD_STREAM_INVALID_ID;
    }
    if workloads::unregister(id) { 0 } else { WORKLOAD_STREAM_NOT_REGISTERED }
} }
agc_export! { fn sceAgcDriverSetTFRing(base: *const c_void, bytes: u32) -> i32 {
    if base.is_null() || bytes == 0 {
        return Call("sceAgcDriverSetTFRing").fail(Failure::new(result::INVALID, "empty ring"));
    }
    let ring = &Driver::get().tessellation_ring;
    ring.base.store(base as u64, Ordering::Release);
    ring.bytes.store(bytes, Ordering::Release);
    0
} }
agc_export! { fn sceAgcDriverGetTFRing(base: *mut u64, bytes: *mut u32) -> i32 {
    if base.is_null() || bytes.is_null() {
        return Call("sceAgcDriverGetTFRing").fail(Failure::new(result::INVALID, "null output"));
    }
    let ring = &Driver::get().tessellation_ring;
    unsafe {
        *base = ring.base.load(Ordering::Acquire);
        *bytes = ring.bytes.load(Ordering::Acquire);
    }
    0
} }
agc_export! { fn sceAgcDriverGetResourceRegistrationMaxNameLength(length: *mut u32) -> i32 {
    if length.is_null() {
        return Call("sceAgcDriverGetResourceRegistrationMaxNameLength").fail(Failure::new(result::INVALID, "null output"));
    }
    unsafe { *length = 0xfc };
    0
} }
agc_export! { fn sceAgcDriverQueryResourceRegistrationUserMemoryRequirements(bytes: *mut u64) -> i32 {
    if bytes.is_null() {
        return EINVAL;
    }
    unsafe { *bytes = 0x4000 };
    0
} }
agc_export! { fn sceAgcDriverGetCoredumpAddress() -> *mut c_void {
    std::ptr::null_mut()
} }
agc_export! { fn sceAgcDriverGetCoredumpMode(mode: *mut u32) -> i32 {
    if mode.is_null() {
        return NULL_OUTPUT;
    }
    unsafe { *mode = 0 };
    0
} }
agc_export! { fn sceAgcDriverAllocateToolMemoryForGpuReset(_first: u64, _second: u64, _third: u64, range: *mut u8) -> i32 {
    if !range.is_null() {
        unsafe { range.write_bytes(0, TOOL_MEMORY_RANGE_BYTES) };
    }
    FEATURE_UNAVAILABLE
} }
agc_export! { fn sceAgcDriverGetAllocatedToolMemoryForGpuReset(_first: u64, range: *mut u8) -> i32 {
    if !range.is_null() {
        unsafe { range.write_bytes(0, TOOL_MEMORY_RANGE_BYTES) };
    }
    FEATURE_UNAVAILABLE
} }

const TOOL_MEMORY_RANGE_BYTES: usize = 12;

/// The exports with arguments that need looking at.
pub const STATE: &[&str] = &[
    "sceAgcDriverRegisterWorkloadStream", "sceAgcDriverUnregisterWorkloadStream", "sceAgcDriverSetTFRing", "sceAgcDriverGetTFRing",
    "sceAgcDriverGetResourceRegistrationMaxNameLength", "sceAgcDriverQueryResourceRegistrationUserMemoryRequirements",
    "sceAgcDriverGetCoredumpAddress", "sceAgcDriverGetCoredumpMode", "sceAgcDriverAllocateToolMemoryForGpuReset",
    "sceAgcDriverGetAllocatedToolMemoryForGpuReset",
];

/// Packet helpers: sizes of what the Agc functions write, and packets the title places itself.
fn user_data_dwords(bytes: u32) -> u32 {
    match bytes {
        0 => 3,
        _ => match (bytes + 3) >> 2 {
            1 => 4,
            payload => payload + 7,
        },
    }
}

agc_export! { fn sceAgcDriverGetSetFlipPacketSizeInDwords() -> u32 { footprint(Kind::Flip, "sceAgcDriverGetSetFlipPacketSizeInDwords") } }
agc_export! { fn sceAgcDriverGetWaitRenderingPacketSizeInDwords() -> u32 { footprint(Kind::WaitFlipDone, "sceAgcDriverGetWaitRenderingPacketSizeInDwords") } }
agc_export! { fn sceAgcDriverGetSetWorkloadsActivePacketSize() -> u32 { user_data_dwords(8) + 9 } }
agc_export! { fn sceAgcDriverGetSetWorkloadCompletePacketSize() -> u32 { user_data_dwords(0) + 9 } }
agc_export! { fn sceAgcDriverGetSetWorkloadStreamInactivePacketSize() -> u32 { user_data_dwords(8) } }
agc_export! { fn sceAgcDriverNopDataGetPacketSize(bytes: u32) -> u32 { ((u64::from(bytes) + 3) >> 2) as u32 + 2 } }
agc_export! { fn sceAgcDriverUserDataGetPacketSize(bytes: u32) -> u32 { user_data_dwords(bytes) } }

/// Dwords a command of `kind` takes; 0, reported, when the target has no form for it.
fn footprint(kind: Kind, function: &str) -> u32 {
    crate::agc::abi::agc().target.footprint(kind).unwrap_or_else(|error| {
        report(function, &error.message);
        0
    })
}

/// Thread-trace user data the host has no use for: the footprint is NOPs.
unsafe fn user_data_nops(commands: *mut u32, data: *const c_void, bytes: u32) -> u32 {
    let dwords = if data.is_null() || bytes == 0 { 3 } else { user_data_dwords(bytes) };
    // SAFETY: the title reserved the size `sceAgcDriverUserDataGetPacketSize` reported.
    crate::agc::abi::agc().target.fill(unsafe { std::slice::from_raw_parts_mut(commands, dwords as usize) });
    dwords
}

agc_export! { fn sceAgcDriverUserDataWritePacket(commands: *mut u32, _async_compute: u32, _value: u32, data: *const c_void, bytes: u32) -> u32 {
    unsafe { user_data_nops(commands, data, bytes) }
} }
agc_export! { fn sceAgcDriverUserDataWritePopMarker(commands: *mut u32, _async_compute: u32) -> u32 {
    unsafe { user_data_nops(commands, std::ptr::null(), 0) }
} }
agc_export! { fn sceAgcDriverUserDataWritePushMarker(commands: *mut u32, _async_compute: u32, name: *const c_char, bytes: u32, _color: u32) -> u32 {
    unsafe { user_data_nops(commands, name.cast(), bytes) }
} }
agc_export! { fn sceAgcDriverUserDataWriteSetMarker(commands: *mut u32, _async_compute: u32, name: *const c_char, bytes: u32, _color: u32) -> u32 {
    unsafe { user_data_nops(commands, name.cast(), bytes) }
} }
agc_export! { fn sceAgcDriverNopDataWritePacket(destination: *mut u32, address: u32, data: *const u8, bytes: u32) -> i32 {
    let total = (bytes as usize).div_ceil(4) + 2;
    // SAFETY: the title reserved the size `sceAgcDriverNopDataGetPacketSize` reported.
    let packet = unsafe { std::slice::from_raw_parts_mut(destination, total) };
    let mut body = vec![0u32; total - 1];
    body[0] = address;
    if !data.is_null() && bytes != 0 {
        // SAFETY: `data` holds `bytes` bytes, and `body` has room for them after the address.
        unsafe { std::ptr::copy_nonoverlapping(data, body[1..].as_mut_ptr().cast::<u8>(), bytes as usize) };
    }
    let nop = Op::Nop { dwords: total as u32, payload: Some(&body) };
    if let Err(error) = crate::agc::abi::agc().write_packet(&nop, packet) {
        report("sceAgcDriverNopDataWritePacket", &error.message);
        return 0;
    }
    if data.is_null() || bytes == 0 { 2 } else { total as i32 }
} }
agc_export! { fn sceAgcDriverWaitUntilSafeForRendering(command: *mut *mut u32, capacity: u32, mode: u32, handle: u32, index: i32) -> u32 {
    let call = Call("sceAgcDriverWaitUntilSafeForRendering");
    let dwords = footprint(Kind::WaitFlipDone, call.0);
    if command.is_null() || dwords == 0 || capacity < dwords || mode != 0 || index < 0 {
        call.fail(Failure::new(result::INVALID, "invalid rendering wait arguments"));
        return 0;
    }
    // SAFETY: the title passes its cursor, with `capacity` dwords behind it.
    let out = unsafe { std::slice::from_raw_parts_mut(*command, dwords as usize) };
    let wait = Op::Notice(Notice::WaitFlipDone { video_out: handle, buffer: index });
    if let Err(error) = crate::agc::abi::agc().write_packet(&wait, out) {
        report(call.0, &error.message);
        return 0;
    }
    unsafe { *command = (*command).add(dwords as usize) };
    0
} }

/// The packet helper exports.
pub const PACKETS: &[&str] = &[
    "sceAgcDriverGetSetFlipPacketSizeInDwords", "sceAgcDriverGetWaitRenderingPacketSizeInDwords",
    "sceAgcDriverGetSetWorkloadsActivePacketSize", "sceAgcDriverGetSetWorkloadCompletePacketSize",
    "sceAgcDriverGetSetWorkloadStreamInactivePacketSize", "sceAgcDriverNopDataGetPacketSize", "sceAgcDriverUserDataGetPacketSize",
    "sceAgcDriverUserDataWritePacket", "sceAgcDriverUserDataWritePopMarker", "sceAgcDriverUserDataWritePushMarker",
    "sceAgcDriverUserDataWriteSetMarker", "sceAgcDriverNopDataWritePacket", "sceAgcDriverWaitUntilSafeForRendering",
];

/// Exports that answer one constant; their arguments, if any, are ignored.
macro_rules! constants {
    ($list:ident = $value:expr => $($name:ident),* $(,)?) => {
        $(agc_export! { fn $name() -> i32 { $value } })*
        /// Exports of one constant answer.
        pub const $list: &[&str] = &[$(stringify!($name)),*];
    };
}

constants!(RETURN_ZERO = 0 =>
    sceAgcDriverDbgReadRegister, sceAgcDriverGetExceptionStatus, sceAgcDriverGetTraceInitiator, sceAgcDriverIsPaDebug,
    sceAgcDriverRegisterDefaultOwner, sceAgcDriverRegisterGpuResetCallbacks, sceAgcDriverSysSubmitFlipHandleProxy,
    sceAgcDriverUnregisterGpuResetCallbacks, sceAgcDriverUserDataImmediateWrite, sceAgcDriverUserDataImmediateWritePacket,
    sceAgcDriverIsCaptureInProgress, sceAgcDriverIsSubmitValidationEnabled, sceAgcDriverIsTraceInProgress,
    sceAgcDriverSetHsOffchipParam, sceAgcDriverInitResourceRegistration, sceAgcDriverSetResourceRegistrationInfo,
);
constants!(RETURN_ONE = 1 => sceAgcDriverGetShaderDebuggingStatus);
constants!(MODULE_FAILED = MODULE_REGISTRATION_FAILED => sceAgcDriverModuleRegistration);
constants!(DEBUG_SERVICES = DEBUG_UNAVAILABLE =>
    sceAgcDriverGetCaptureConfig, sceAgcDriverGetSubmitValidationConfig, sceAgcDriverGetSubmitValidationMode,
    sceAgcDriverSetCaptureConfig, sceAgcDriverSetSubmitValidationConfig, sceAgcDriverSetSubmitValidationMode,
    sceAgcDriverSetValidationDisabled, sceAgcDriverSetValidationErrorOutputFrequency,
    sceAgcDriverRequestCaptureStart, sceAgcDriverRequestCaptureStop, sceAgcDriverTriggerCapture,
);
constants!(FEATURES = FEATURE_UNAVAILABLE =>
    sceAgcDriverAcquireRazorACQ, sceAgcDriverDbgWriteRegister, sceAgcDriverGetGpuPrintfWorkArea, sceAgcDriverGetGpuRefClks,
    sceAgcDriverGetPaDebugInterfaceVersion, sceAgcDriverIsSuspendPointInFlight, sceAgcDriverSetGpuPrintfEnabled,
    sceAgcDriverSetHsOffchipParamDirect, sceAgcDriverSetTargetRingForDiag, sceAgcDriverSubmitMultiCommandBuffersDirect,
    sceAgcDriverSubmitToRazorACQ, sceAgcDriverSuspendPointSubmitDirect,
);
constants!(RESOURCES = RESOURCES_UNAVAILABLE =>
    sceAgcDriverFindResourcesPublic, sceAgcDriverGetDefaultOwner, sceAgcDriverGetOwnerName,
    sceAgcDriverGetResourceBaseAddressAndSizeInBytes, sceAgcDriverGetResourceName, sceAgcDriverGetResourceShaderGuid,
    sceAgcDriverGetResourceType, sceAgcDriverGetResourceUserData, sceAgcDriverInitResourceRegistrationInterface,
    sceAgcDriverRegisterGdsResource, sceAgcDriverRegisterMultipleResources, sceAgcDriverRegisterOwner,
    sceAgcDriverRegisterOwnerForSystem, sceAgcDriverRegisterResource, sceAgcDriverSetResourceUserData,
    sceAgcDriverUnregisterAllResourcesForOwner, sceAgcDriverUnregisterOwnerAndResources, sceAgcDriverUnregisterResource,
);

/// Exports AnyPS5 never implemented: each call is reported.
macro_rules! not_implemented {
    ($($name:ident),* $(,)?) => {
        $(agc_export! { fn $name() -> i32 {
            report(stringify!($name), "not implemented");
            FEATURE_UNAVAILABLE
        } })*
        /// Exports AnyPS5 never implemented.
        pub const NOT_IMPLEMENTED: &[&str] = &[$(stringify!($name)),*];
    };
}

not_implemented!(
    sceAgcDriverAcquireHDRScopesACQ, sceAgcDriverCreateQueue, sceAgcDriverCwsrResumeAcq, sceAgcDriverCwsrSuspendAcq,
    sceAgcDriverDebugHardwareStatus, sceAgcDriverDeregisterCaptureInterface, sceAgcDriverDestroyQueue, sceAgcDriverGetHsOffchipParam,
    sceAgcDriverGetRegShadowInfo, sceAgcDriverGetRegShadowInfoAgr, sceAgcDriverGetReservedDmemForAgc, sceAgcDriverGetWorkloadStreamInfo,
    sceAgcDriverHp3dLockEnd, sceAgcDriverHp3dLockStart, sceAgcDriverIDHSSubmit, sceAgcDriverIsCoredumpValid, sceAgcDriverIsEsvmProcess,
    sceAgcDriverIsSuspendPointInFlightDirect, sceAgcDriverNotifyDefaultStates, sceAgcDriverPassInfoDownward, sceAgcDriverPatchClearState,
    sceAgcDriverRegisterCaptureInterface, sceAgcDriverReleaseHDRScopesACQ, sceAgcDriverReleaseRazorACQ, sceAgcDriverSdmaCopyLinearBlocking,
    sceAgcDriverSetFlip, sceAgcDriverSetTFRingDirect, sceAgcDriverSetValueToSharedGds, sceAgcDriverSetWorkloadComplete,
    sceAgcDriverSetWorkloadStreamInactive, sceAgcDriverSetWorkloadsActive, sceAgcDriverSetupAsyncGraphics, sceAgcDriverSetupRegisterShadow,
    sceAgcDriverSubmitReprojectionAcb, sceAgcDriverSubmitToHDRScopesACQ, sceAgcDriverSuspendPointSubmit, sceAgcDriverSuspendPointSubmitCdbg,
    sceAgcDriverSysEnableSubmitDone45Exception, sceAgcDriverSysGetClientNumber, sceAgcDriverSysGetGfxAppGpuResetStatus,
    sceAgcDriverSysIsGameClosed, sceAgcDriverTmpInitIdhs,
);

/// Runs one of the interface's calls: its code, the failure described in `error`.
fn interface_call(error: *mut Error, body: impl FnOnce() -> Outcome<()>) -> i32 {
    let failure = match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => return result::OK,
        Ok(Err(failure)) => failure,
        Err(_) => Failure::new(result::FAILED, "the driver panicked"),
    };
    // SAFETY: the interface's callers pass null or an `AgcDriverError`.
    unsafe { failure.report(error) }
}

/// An interface argument that must be there.
fn required<'a, T>(pointer: *const T, what: &str) -> Outcome<&'a T> {
    // SAFETY: the interface's callers pass null or a valid pointer.
    unsafe { pointer.as_ref() }.ok_or_else(|| Failure::new(result::INVALID, format!("null {what}")))
}

/// Exports of AnyPS5's interface, in the host's calling convention.
macro_rules! interface_export {
    ($(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block)*) => {
        $(
            /// See `AgcDriver.h`.
            ///
            /// # Safety
            /// As `AgcDriver.h` specifies.
            #[no_mangle]
            pub unsafe extern "C" fn $name($($arg: $ty),*) $(-> $ret)? $body
        )*
        /// The exports of the C interface (D21).
        pub const INTERFACE: &[&str] = &[$(stringify!($name)),*];
    };
}

/// `AspectFitSize`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AspectFitSize {
    pub width: u32,
    pub height: u32,
}

/// `AspectFitRect`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AspectFitRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Fitting a source extent inside bounds, keeping its aspect ratio. A zero extent is a caller's
/// bug the C++ module threw for: it is reported and fits as 1.
struct AspectFit {
    source_width: u32,
    source_height: u32,
}

impl AspectFit {
    fn valid(&self) -> bool {
        if self.source_width == 0 || self.source_height == 0 {
            report("AspectFit", "source extent must be non-zero");
            return false;
        }
        true
    }

    fn scale(&self, bounds_width: u32, bounds_height: u32, allow_upscale: bool) -> f64 {
        if !self.valid() || bounds_width == 0 || bounds_height == 0 {
            return 1.0;
        }
        let scale = (f64::from(bounds_width) / f64::from(self.source_width)).min(f64::from(bounds_height) / f64::from(self.source_height));
        if allow_upscale { scale } else { scale.min(1.0) }
    }

    fn size(&self, bounds_width: u32, bounds_height: u32, allow_upscale: bool) -> AspectFitSize {
        let scale = self.scale(bounds_width, bounds_height, allow_upscale);
        AspectFitSize {
            width: ((f64::from(self.source_width) * scale).round() as u32).max(1),
            height: ((f64::from(self.source_height) * scale).round() as u32).max(1),
        }
    }

    fn extent(&self, length: f64, numerator: u32, denominator: u32) -> u32 {
        if !self.valid() {
            return 1;
        }
        ((length * f64::from(numerator) / f64::from(denominator)).round() as u32).max(1)
    }
}

static FRAMES_PRESENTED: AtomicU64 = AtomicU64::new(0);

interface_export! {
    fn AgcDriverRegisterVideoOutput(handle: u32, output: *const VideoOutput, error: *mut Error) -> i32 {
        interface_call(error, || Driver::get().flips.register(handle, required(output, "video output")?))
    }
    fn AgcDriverUnregisterVideoOutput(handle: u32, this: *mut c_void, error: *mut Error) -> i32 {
        interface_call(error, || Driver::get().flips.unregister(handle, this))
    }
    fn AgcDriverSubmitFlip(info: *const FlipInfo, error: *mut Error) -> i32 {
        interface_call(error, || Driver::get().flips.submit(required(info, "flip")?))
    }
    fn AgcDriverReserveEopFlip(info: *const FlipInfo, token: *mut u32, error: *mut Error) -> i32 {
        interface_call(error, || match Driver::get().flips.reserve_eop(required(info, "flip")?, None)? {
            Some(reserved) => {
                // SAFETY: the caller's out-parameter.
                unsafe { *token = reserved };
                Ok(())
            }
            None => Err(Failure::new(result::FULL, "the flip queue is full")),
        })
    }
    fn AgcDriverReserveEopFlipAt(info: *const FlipInfo, token: u32, error: *mut Error) -> i32 {
        interface_call(error, || match Driver::get().flips.reserve_eop(required(info, "flip")?, Some(token))? {
            Some(_) => Ok(()),
            None => Err(Failure::new(result::FULL, "the flip queue is full")),
        })
    }
    fn AgcDriverPresentationApi() -> u32 {
        interface::PRESENTATION_NATIVE_WINDOW
    }
    fn AgcDriverAttachWindow(window: *const PresentationWindow, error: *mut Error) -> i32 {
        interface_call(error, || {
            let driver = Driver::get();
            // The guest window has to exist before a presenter can create a D3D12 device.
            let host = driver.host()?;
            Presentation::attach(required(window, "window")?, host.adapter, host.arena.clone())?;
            // The native presenter currently exposes an SDR surface.
            driver.hdr.set_surface_hdr10(false);
            Ok(())
        })
    }
    fn AgcDriverReleaseWindow(context: *mut c_void) {
        if !Presentation::release(context) {
            report("AgcDriverReleaseWindow", "the window belongs to another thread, or is not attached");
        }
    }
    fn AgcDriverCaptureOverlay(buffer: *const DisplayBuffer, error: *mut Error) -> i32 {
        interface_call(error, || Presentation::capture_overlay(unsafe { buffer.as_ref() }))
    }
    fn AgcDriverPresentBuffer(
        window: *const PresentationWindow,
        buffer: *const DisplayBuffer,
        plane: *const DisplayBuffer,
        gpu_ready: Option<unsafe extern "C" fn(*mut c_void)>,
        context: *mut c_void,
        error: *mut Error,
    ) -> i32 {
        interface_call(error, || {
            let window = required(window, "window")?;
            let frame = Frame { buffer: Some(required(buffer, "buffer")?), plane: unsafe { plane.as_ref() }, width: window.width, height: window.height };
            present(window, &frame, gpu_ready, context)
        })
    }
    fn AgcDriverPresentClear(
        window: *const PresentationWindow,
        _opaque: bool,
        plane: *const DisplayBuffer,
        gpu_ready: Option<unsafe extern "C" fn(*mut c_void)>,
        context: *mut c_void,
        error: *mut Error,
    ) -> i32 {
        interface_call(error, || {
            let window = required(window, "window")?;
            let frame = Frame { buffer: None, plane: unsafe { plane.as_ref() }, width: window.width, height: window.height };
            present(window, &frame, gpu_ready, context)
        })
    }
    fn AgcDriverDisplayBufferSize(buffer: *const DisplayBuffer) -> usize {
        match unsafe { buffer.as_ref() } {
            Some(buffer) => crate::present::display_buffer_size(&buffer.into()).unwrap_or(0),
            None => 0,
        }
    }
    fn AgcDriverSetHostDisplayHdr(display: *const HostDisplayHdr) {
        if let Some(&display) = unsafe { display.as_ref() } {
            Driver::get().hdr.set_display(display);
        }
    }
    fn AgcDriverSetSurfaceHdr10(offered: bool) {
        Driver::get().hdr.set_surface_hdr10(offered);
    }
    fn AgcDriverNoteReportedToneMapPeak(nits: f32) {
        Driver::get().hdr.note_tone_map_peak(nits);
    }
    fn AgcDriverHostHdr(status: *mut HostHdrStatus) {
        if !status.is_null() {
            unsafe { *status = Driver::get().hdr.status() };
        }
    }
    fn AgcDriverPresentingHdr10() -> bool {
        false
    }
    fn AgcDriverSetSystemOverlay(_view: *const c_void) {}
    fn AgcDriverSetSystemDialog(_view: *const c_void) {}
    fn AgcDriverPadStatusPoll(_provider: *const c_void) {}
    fn AgcDriverPadStatusShow() {}
    fn AgcDriverTogglePerformanceOverlay() {}
    fn AgcDriverRequestScreenshot() {}
    fn AgcDriverCheckGuestMemory(pointer: *const c_void, bytes: usize, alignment: usize, writable: bool, error: *mut Error) -> i32 {
        interface_call(error, || {
            let address = pointer as u64;
            if address == 0 || alignment == 0 || !address.is_multiple_of(alignment as u64) {
                return Err(Failure::new(result::INVALID, format!("null or misaligned guest address 0x{address:x}")));
            }
            let host = Driver::get().host()?;
            if super::memory::GuestAccess::new(host.arena.clone()).accessible(address, bytes as u64, writable) {
                Ok(())
            } else {
                Err(Failure::new(result::INVALID, format!("guest memory 0x{address:x}+0x{bytes:x} is not mapped with the access asked for")))
            }
        })
    }
    fn AgcDriverStoreGuestLabel(label: *mut u64, value: u64) {
        if label.is_null() || label as usize % 8 != 0 {
            report("AgcDriverStoreGuestLabel", "null or misaligned label");
            return;
        }
        // SAFETY: an aligned label of guest memory, written as the GPU writes labels.
        unsafe { AtomicU64::from_ptr(label) }.store(value, Ordering::Release);
    }
    fn AgcDriverWaitIdle() {
        if let Some(host) = Driver::get().started() {
            host.service.wait_idle();
        }
    }
    fn AgcDriverRaiseWorkerThreadPriority(_role: *const c_char) {}
    fn AgcDriverLockVulkanLoader() {}
    fn AgcDriverUnlockVulkanLoader() {}
    fn AgcDriverTraceRecording() -> bool {
        false
    }
    fn AgcDriverTraceThread() -> *mut c_void {
        std::ptr::null_mut()
    }
    fn AgcDriverTraceCount(_scope: *const c_char) {}
    fn ComputeContainScale(source_width: u32, source_height: u32, bounds_width: u32, bounds_height: u32, allow_upscale: bool) -> f64 {
        AspectFit { source_width, source_height }.scale(bounds_width, bounds_height, allow_upscale)
    }
    fn ComputeContainSize(source_width: u32, source_height: u32, bounds_width: u32, bounds_height: u32, allow_upscale: bool) -> AspectFitSize {
        AspectFit { source_width, source_height }.size(bounds_width, bounds_height, allow_upscale)
    }
    fn ComputeContainRect(source_width: u32, source_height: u32, destination_width: u32, destination_height: u32) -> AspectFitRect {
        let size = AspectFit { source_width, source_height }.size(destination_width, destination_height, true);
        AspectFitRect {
            x: ((i64::from(destination_width) - i64::from(size.width)) / 2) as i32,
            y: ((i64::from(destination_height) - i64::from(size.height)) / 2) as i32,
            width: size.width,
            height: size.height,
        }
    }
    fn ComputeWidthForHeight(source_width: u32, source_height: u32, height: u32) -> u32 {
        AspectFit { source_width, source_height }.extent(f64::from(height), source_width, source_height)
    }
    fn ComputeHeightForWidth(source_width: u32, source_height: u32, width: u32) -> u32 {
        AspectFit { source_width, source_height }.extent(f64::from(width), source_height, source_width)
    }
    fn AgcDriverShutdown() {
        Driver::get().shutdown();
    }
}

fn present(window: &PresentationWindow, frame: &Frame, gpu_ready: Option<unsafe extern "C" fn(*mut c_void)>, context: *mut c_void) -> Outcome<()> {
    let Some(gpu_ready) = gpu_ready.filter(|_| !context.is_null()) else {
        return Err(Failure::new(result::INVALID, "missing GPU completion callback"));
    };
    let driver = Driver::get();
    if let Some(message) = driver.flips.failure() {
        return Err(Failure::new(result::FAILED, message));
    }
    if Presentation::present(window, frame)? {
        FRAMES_PRESENTED.fetch_add(1, Ordering::Relaxed);
    }
    // SAFETY: VideoOut's callback; the display buffers are released.
    unsafe { gpu_ready(context) };
    Ok(())
}

/// Frames the display has shown.
pub fn frames_presented() -> u64 {
    FRAMES_PRESENTED.load(Ordering::Relaxed)
}

/// The module's exports, by group (`agc::exports`).
pub const GROUPS: &[&[&str]] = &[
    SUBMISSIONS, EVENTS, STATE, PACKETS, RETURN_ZERO, RETURN_ONE, MODULE_FAILED, DEBUG_SERVICES, FEATURES, RESOURCES, NOT_IMPLEMENTED, INTERFACE,
];
