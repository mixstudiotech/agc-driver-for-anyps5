//! The libSceAgc sync, memory, control, marker and draw-parameter commands (PLAN P3.4 step 2c):
//! signatures as the SDK's wrappers pass them (D19), validation as the console library's.

use super::{emit, guard, report, size_of};
use crate::agc::target::{DmaEnd, Kind, Notice, Op, Queue, ReleaseData};
use crate::agc::{check_address, check_bits, workloads, CommandBuffer};
use crate::agc_export;
use crate::{Error, Result};
use std::ffi::{c_char, CStr};

fn require(condition: bool, function: &str, reason: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::invalid(format!("{function}: {reason}")))
    }
}

/// Records the op `build` makes after validating, or reports why not and returns null.
fn record(buffer: *mut CommandBuffer, queue: Queue, function: &str, build: impl FnOnce() -> Result<Op<'static>>) -> *mut u32 {
    match guard(function, None, || build().map(Some)) {
        Some(op) => emit(buffer, queue, &op, function),
        None => std::ptr::null_mut(),
    }
}

// Markers (`Agc::Marker`): the colour is not recorded.

/// The text of a NUL-terminated marker.
unsafe fn marker_text<'a>(text: *const c_char, function: &str) -> Result<&'a [u8]> {
    if text.is_null() {
        return Ok(&[]);
    }
    // SAFETY: the title passes a NUL-terminated string.
    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    require(bytes.len() < 0x1_0000, function, "marker text too long")?;
    Ok(bytes)
}

/// `length` bytes of marker text.
unsafe fn marker_span<'a>(text: *const c_char, length: u32, function: &str) -> Result<&'a [u8]> {
    require(!text.is_null() || length == 0, function, "null marker text")?;
    require((length as usize + 4) / 4 <= 0x4000, function, "marker text too long")?;
    if length == 0 {
        return Ok(&[]);
    }
    // SAFETY: the title passes `length` bytes.
    Ok(unsafe { std::slice::from_raw_parts(text as *const u8, length as usize) })
}

fn push(buffer: *mut CommandBuffer, queue: Queue, text: Result<&[u8]>, pop: bool, function: &str) -> *mut u32 {
    let text = match text {
        Ok(text) => text,
        Err(error) => {
            report(function, &error.message);
            return std::ptr::null_mut();
        }
    };
    let pushed = emit(buffer, queue, &Op::Marker { text: Some(text) }, function);
    if pop && !pushed.is_null() {
        emit(buffer, queue, &Op::Marker { text: None }, function);
    }
    pushed
}

macro_rules! markers {
    ($queue:expr, $push:ident, $span:ident, $set:ident, $set_span:ident, $pop:ident) => {
        agc_export! { fn $push(buffer: *mut CommandBuffer, text: *const c_char, _color: u32) -> *mut u32 {
            push(buffer, $queue, unsafe { marker_text(text, stringify!($push)) }, false, stringify!($push))
        } }
        agc_export! { fn $span(buffer: *mut CommandBuffer, text: *const c_char, length: u32, _color: u32) -> *mut u32 {
            push(buffer, $queue, unsafe { marker_span(text, length, stringify!($span)) }, false, stringify!($span))
        } }
        agc_export! { fn $set(buffer: *mut CommandBuffer, text: *const c_char, _color: u32) -> *mut u32 {
            push(buffer, $queue, unsafe { marker_text(text, stringify!($set)) }, true, stringify!($set))
        } }
        agc_export! { fn $set_span(buffer: *mut CommandBuffer, text: *const c_char, length: u32, _color: u32) -> *mut u32 {
            push(buffer, $queue, unsafe { marker_span(text, length, stringify!($set_span)) }, true, stringify!($set_span))
        } }
        agc_export! { fn $pop(buffer: *mut CommandBuffer) -> *mut u32 {
            emit(buffer, $queue, &Op::Marker { text: None }, stringify!($pop))
        } }
    };
}

markers!(Queue::Compute, sceAgcAcbPushMarker, sceAgcAcbPushMarkerSpan, sceAgcAcbSetMarker, sceAgcAcbSetMarkerSpan, sceAgcAcbPopMarker);
markers!(Queue::Graphics, sceAgcDcbPushMarker, sceAgcDcbPushMarkerSpan, sceAgcDcbSetMarker, sceAgcDcbSetMarkerSpan, sceAgcDcbPopMarker);

// Workloads: zero-filled NOPs with the console's payloads; bad input returns null, unreported.

fn workload(buffer: *mut CommandBuffer, queue: Queue, dwords: u32, payload: &[u32], function: &str) -> *mut u32 {
    emit(buffer, queue, &Op::Nop { dwords, payload: Some(payload) }, function)
}

unsafe fn workloads_active(buffer: *mut CommandBuffer, queue: Queue, stream: u32, ids: *const u32, count: u32, function: &str) -> *mut u32 {
    if buffer.is_null() || ids.is_null() || count == 0 || count > 63 || !workloads::registered(stream) {
        return std::ptr::null_mut();
    }
    let mut mask = 0u64;
    // SAFETY: the title passes `count` ids.
    for &id in unsafe { std::slice::from_raw_parts(ids, count as usize) } {
        if id > 63 || mask & 1 << id != 0 {
            return std::ptr::null_mut();
        }
        mask |= 1 << id;
    }
    workload(buffer, queue, 18, &[stream, mask as u32, (mask >> 32) as u32], function)
}

fn workload_complete(buffer: *mut CommandBuffer, queue: Queue, stream: u32, id: u32, function: &str) -> *mut u32 {
    if buffer.is_null() || id > 63 || !workloads::registered(stream) {
        return std::ptr::null_mut();
    }
    let rest = !(1u64 << id);
    workload(buffer, queue, 12, &[stream, id, rest as u32, (rest >> 32) as u32], function)
}

fn stream_inactive(buffer: *mut CommandBuffer, queue: Queue, stream: u32, function: &str) -> *mut u32 {
    if buffer.is_null() || !workloads::registered(stream) {
        return std::ptr::null_mut();
    }
    workload(buffer, queue, 9, &[stream], function)
}

agc_export! { fn sceAgcAcbSetWorkloadsActive(buffer: *mut CommandBuffer, stream: u32, ids: *const u32, count: u32) -> *mut u32 {
    unsafe { workloads_active(buffer, Queue::Compute, stream, ids, count, "sceAgcAcbSetWorkloadsActive") }
} }
agc_export! { fn sceAgcDcbSetWorkloadsActive(buffer: *mut CommandBuffer, stream: u32, ids: *const u32, count: u32) -> *mut u32 {
    unsafe { workloads_active(buffer, Queue::Graphics, stream, ids, count, "sceAgcDcbSetWorkloadsActive") }
} }
agc_export! { fn sceAgcAcbSetWorkloadComplete(buffer: *mut CommandBuffer, stream: u32, id: u32) -> *mut u32 {
    workload_complete(buffer, Queue::Compute, stream, id, "sceAgcAcbSetWorkloadComplete")
} }
agc_export! { fn sceAgcDcbSetWorkloadComplete(buffer: *mut CommandBuffer, stream: u32, id: u32) -> *mut u32 {
    workload_complete(buffer, Queue::Graphics, stream, id, "sceAgcDcbSetWorkloadComplete")
} }
agc_export! { fn sceAgcAcbSetWorkloadStreamInactive(buffer: *mut CommandBuffer, stream: u32) -> *mut u32 {
    stream_inactive(buffer, Queue::Compute, stream, "sceAgcAcbSetWorkloadStreamInactive")
} }
agc_export! { fn sceAgcDcbSetWorkloadStreamInactive(buffer: *mut CommandBuffer, stream: u32) -> *mut u32 {
    stream_inactive(buffer, Queue::Graphics, stream, "sceAgcDcbSetWorkloadStreamInactive")
} }

// Presentation and queue control: what submission does.

agc_export! { fn sceAgcAcbSetFlip(buffer: *mut CommandBuffer, video_out: u32, index: i32, mode: u32, argument: i64) -> *mut u32 {
    emit(buffer, Queue::Compute, &Op::Notice(Notice::Flip { video_out, buffer: index, mode, argument }), "sceAgcAcbSetFlip")
} }
agc_export! { fn sceAgcDcbSetFlip(buffer: *mut CommandBuffer, video_out: u32, index: i32, mode: u32, argument: i64) -> *mut u32 {
    emit(buffer, Queue::Graphics, &Op::Notice(Notice::Flip { video_out, buffer: index, mode, argument }), "sceAgcDcbSetFlip")
} }
agc_export! { fn sceAgcAcbWaitUntilSafeForRendering(buffer: *mut CommandBuffer, video_out: u32, index: i32) -> *mut u32 {
    emit(buffer, Queue::Compute, &Op::Notice(Notice::WaitFlipDone { video_out, buffer: index }), "sceAgcAcbWaitUntilSafeForRendering")
} }
agc_export! { fn sceAgcDcbWaitUntilSafeForRendering(buffer: *mut CommandBuffer, video_out: u32, index: i32) -> *mut u32 {
    emit(buffer, Queue::Graphics, &Op::Notice(Notice::WaitFlipDone { video_out, buffer: index }), "sceAgcDcbWaitUntilSafeForRendering")
} }
agc_export! { fn sceAgcAcbResetQueue(buffer: *mut CommandBuffer, operation: u32, _value: u32) -> *mut u32 {
    emit(buffer, Queue::Compute, &Op::Notice(Notice::ResetQueue { operation, state: 0 }), "sceAgcAcbResetQueue")
} }
agc_export! { fn sceAgcDcbResetQueue(buffer: *mut CommandBuffer, operation: u32, state: u32) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbResetQueue", || {
        check_bits(operation.into(), 0xfff, "sceAgcDcbResetQueue")?;
        check_bits(state.into(), 0xf, "sceAgcDcbResetQueue")?;
        Ok(Op::Notice(Notice::ResetQueue { operation, state }))
    })
} }
agc_export! { fn sceAgcAcbRewind(buffer: *mut CommandBuffer, state: u32, _offload: u32) -> *mut u32 {
    emit(buffer, Queue::Compute, &Op::Notice(Notice::Rewind { valid: state & 1 != 0 }), "sceAgcAcbRewind")
} }
agc_export! { fn sceAgcDcbRewind(buffer: *mut CommandBuffer, state: u32) -> *mut u32 {
    emit(buffer, Queue::Graphics, &Op::Notice(Notice::Rewind { valid: state & 1 != 0 }), "sceAgcDcbRewind")
} }
agc_export! { fn sceAgcAcbRewindGetSize() -> u32 { size_of(Kind::Rewind, "sceAgcAcbRewindGetSize") } }
agc_export! { fn sceAgcDcbRewindGetSize() -> u32 { size_of(Kind::Rewind, "sceAgcDcbRewindGetSize") } }

agc_export! { fn sceAgcDcbContextStateOp_0100(buffer: *mut CommandBuffer, operation: u32) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbContextStateOp_0100", || {
        require(operation <= 3, "sceAgcDcbContextStateOp_0100", "invalid context state operation")?;
        Ok(Op::Notice(Notice::ResetQueue { operation: 0, state: operation }))
    })
} }
agc_export! { fn sceAgcDcbContextStateOpGetSize(operation: u32) -> u64 {
    if operation > 3 { 0 } else { size_of(Kind::ResetQueue, "sceAgcDcbContextStateOpGetSize").into() }
} }
agc_export! { fn sceAgcDcbGetLodStats(
    buffer: *mut CommandBuffer, cache_policy: u8, stats: u64, bytes: u32, reset_count: u32, force_reset: u8, report_and_reset: u8, interval: u32
) -> *mut u32 {
    const F: &str = "sceAgcDcbGetLodStats";
    record(buffer, Queue::Graphics, F, || {
        check_bits(cache_policy.into(), 3, F)?;
        check_bits(reset_count.into(), 0xff, F)?;
        check_bits(force_reset.into(), 1, F)?;
        check_bits(report_and_reset.into(), 1, F)?;
        check_bits(interval.into(), 0xff, F)?;
        if stats != 0 {
            check_address(stats, 64, F)?;
        }
        Ok(Op::LodStats { address: stats, bytes })
    })
} }
agc_export! { fn sceAgcDcbGetLodStatsGetSize() -> u32 { size_of(Kind::LodStats, "sceAgcDcbGetLodStatsGetSize") } }

/// `sceAgcDcbEventWrite` of the occlusion dump the SDK's queries record.
fn occlusion_dump_bytes(function: &str) -> u32 {
    size_of(Kind::Event { event: crate::agc::target::EVENT_PIXEL_PIPE_STAT_DUMP }, function)
}

agc_export! { fn sceAgcDcbBeginOcclusionQueryGetSize(query_type: i32) -> u32 {
    const F: &str = "sceAgcDcbBeginOcclusionQueryGetSize";
    match query_type {
        // Clearing the results first: `sceAgcDcbWriteData` of 64 dwords.
        1 => size_of(Kind::WriteData { dwords: 64 }, F) + occlusion_dump_bytes(F),
        0 => occlusion_dump_bytes(F),
        _ => 0,
    }
} }
agc_export! { fn sceAgcDcbEndOcclusionQueryGetSize() -> u32 { occlusion_dump_bytes("sceAgcDcbEndOcclusionQueryGetSize") } }

// Commands the host has no form for yet: their sizes are the console's, and recording one is
// reported. No code path of TBDR records them (PLAN P3.4 step 2c).
macro_rules! unsupported_commands {
    ($($name:ident: $reason:literal),* $(,)?) => {
        $(agc_export! { fn $name() -> *mut u32 {
            report(stringify!($name), $reason);
            std::ptr::null_mut()
        } })*
        /// Commands reported as unsupported.
        pub const UNSUPPORTED_COMMANDS: &[&str] = &[$(stringify!($name)),*];
    };
}

unsupported_commands!(
    sceAgcCbBranch: "COND_INDIRECT_BUFFER needs its targets placed as IB2s",
    sceAgcAcbAtomicGds_0900: "ATOMIC_GDS needs the host's GDS stand-in",
    sceAgcDcbAtomicGds_0900: "ATOMIC_GDS needs the host's GDS stand-in",
    sceAgcDcbSetIndexIndirectArgs: "INDEX_ATTRIBUTES_INDIRECT has no gfx11 packet",
);

agc_export! { fn sceAgcCbBranchGetSize() -> u32 { 56 } }
agc_export! { fn sceAgcAcbAtomicGdsGetSize() -> u32 { 44 } }
agc_export! { fn sceAgcDcbAtomicGdsGetSize() -> u32 { 44 } }
agc_export! { fn sceAgcDcbSetIndexIndirectArgsGetSize() -> u32 { 16 } }

// Caches and synchronization.

/// A guest ACQUIRE_MEM (`sceAgcDcbAcquireMem`'s checks and range rounding).
fn acquire(engine: u32, cb_db: u32, gcr: u32, base: u64, size: u64, poll: u32, function: &str) -> Result<Op<'static>> {
    check_bits(engine.into(), 1, function)?;
    check_bits(cb_db.into(), 0x7fff_ffff, function)?;
    check_bits(gcr.into(), 0x7_ffff, function)?;
    require(poll / 40 <= 0xffff, function, "acquire poll interval overflow")?;
    let aligned = base & !0xff;
    let whole = size == u64::MAX || aligned >> 40 != 0 || size > 1 << 40;
    // Engine 0 is the prefetch parser; compute queues have no other.
    Ok(Op::Acquire { pfp: engine == 0, cb_db, gcr, ranged: !whole && (aligned != 0 || size < 1 << 40) })
}

agc_export! { fn sceAgcDcbAcquireMem(buffer: *mut CommandBuffer, engine: u8, cb_db: u32, gcr: u32, base: u64, size: u64, poll: u32) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbAcquireMem", || acquire(engine.into(), cb_db, gcr, base, size, poll, "sceAgcDcbAcquireMem"))
} }
agc_export! { fn sceAgcAcbAcquireMem(buffer: *mut CommandBuffer, gcr: u32, base: u64, size: u64, poll: u32) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbAcquireMem", || acquire(1, 0, gcr, base, size, poll, "sceAgcDcbAcquireMem"))
} }
agc_export! { fn sceAgcDcbAcquireMemGetSize() -> u32 { size_of(Kind::Acquire, "sceAgcDcbAcquireMemGetSize") } }
agc_export! { fn sceAgcAcbAcquireMemGetSize() -> u32 { size_of(Kind::Acquire, "sceAgcAcbAcquireMemGetSize") } }

agc_export! { fn sceAgcCbReleaseMem(
    buffer: *mut CommandBuffer, action: u8, gcr: u16, dst: u8, cache_policy: u8, address: u64, select: u8, data: u64,
    gds_offset: u16, gds_size: u16, interrupt: u8, context: u32
) -> *mut u32 {
    const F: &str = "sceAgcCbReleaseMem";
    record(buffer, Queue::Any, F, || {
        check_bits(action.into(), 0x3f, F)?;
        check_bits(gcr.into(), 0xfff, F)?;
        check_bits(dst.into(), 1, F)?;
        check_bits(cache_policy.into(), 3, F)?;
        require(select <= 3 || select == 5, F, "invalid release data selector")?;
        require(interrupt <= 4, F, "invalid release interrupt selector")?;
        check_bits(context.into(), 0x7ff_ffff, F)?;
        if interrupt == 4 {
            require(address == 0 && data == 0, F, "interrupt-only release cannot write data")?;
        } else if select == 5 {
            require(data == 0, F, "GDS release cannot use immediate data")?;
        } else {
            require(gds_offset == 0 && gds_size <= 1, F, "GDS parameters supplied for a non-GDS release")?;
        }
        if select != 0 && interrupt != 4 && address != 0 {
            check_address(address, if matches!(select, 2 | 3) { 8 } else { 4 }, F)?;
        }
        if select == 1 {
            check_bits(data, 0xffff_ffff, F)?;
        }
        // An L2 invalidate writes back first: the host L2 is shared (AnyPS5 does the same).
        let mut gcr = u32::from(gcr);
        if gcr & 0x300 == 0x100 {
            gcr |= 0x200;
        }
        let data = match (select, interrupt) {
            // A null address is patched later (`sceAgcQueueEndOfPipeActionPatchAddress`): the
            // target writes nothing until then.
            (_, 4) | (0, _) => ReleaseData::None,
            (1, _) => ReleaseData::Value32(data as u32),
            (2, _) => ReleaseData::Value64(data),
            (3, _) => ReleaseData::Timestamp,
            _ => ReleaseData::Gds { offset: gds_offset, dwords: gds_size },
        };
        Ok(Op::Release { event: action.into(), gcr, address, data, interrupt: interrupt.into(), context })
    })
} }
agc_export! { fn sceAgcCbQueueEndOfPipeActionGetSize() -> u32 { size_of(Kind::Release, "sceAgcCbQueueEndOfPipeActionGetSize") } }
agc_export! { fn sceAgcAcbQueueEndOfShaderActionGetSize() -> u32 { size_of(Kind::Release, "sceAgcAcbQueueEndOfShaderActionGetSize") } }
agc_export! { fn sceAgcDcbQueueEndOfShaderActionGetSize() -> u32 { size_of(Kind::Release, "sceAgcDcbQueueEndOfShaderActionGetSize") } }

agc_export! { fn sceAgcAcbEventWrite(buffer: *mut CommandBuffer, event: u8, address: u64) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbEventWrite", || {
        check_bits(event.into(), 0x3f, "sceAgcAcbEventWrite")?;
        require(address == 0, "sceAgcAcbEventWrite", "ACB event address is not supported")?;
        Ok(Op::Event { event: event.into(), address: 0 })
    })
} }
agc_export! { fn sceAgcDcbEventWrite(buffer: *mut CommandBuffer, event: u8, address: u64) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbEventWrite", || {
        check_bits(event.into(), 0x3f, "sceAgcDcbEventWrite")?;
        let dump = event & 0xfe == 0x38;
        if dump {
            check_address(address, 8, "sceAgcDcbEventWrite")?;
        }
        Ok(Op::Event { event: event.into(), address: if dump { address } else { 0 } })
    })
} }
agc_export! { fn sceAgcAcbEventWriteGetSize(event: u8) -> u64 {
    size_of(Kind::Event { event: event.into() }, "sceAgcAcbEventWriteGetSize").into()
} }
agc_export! { fn sceAgcDcbEventWriteGetSize(event: u8) -> u64 {
    size_of(Kind::Event { event: event.into() }, "sceAgcDcbEventWriteGetSize").into()
} }

/// A guest WAIT_REG_MEM (`WriteWait`'s checks). `operation` packs the engine (Dcb) and the
/// wait operation as the console's wrapper does.
#[allow(clippy::too_many_arguments)]
fn wait(compute: bool, size: u8, function: u8, operation: u8, address: u64, reference: u64, mask: u64, poll: u32, name: &str) -> Result<Op<'static>> {
    check_bits(size.into(), 1, name)?;
    require(function <= 6, name, "invalid compare function")?;
    check_bits(operation.into(), if size == 0 { 0xf } else { 7 }, name)?;
    require(poll >> 4 <= 0xffff, name, "poll interval overflow")?;
    if address != 0 {
        check_address(address, if size == 0 { 4 } else { 8 }, name)?;
    }
    check_bits(address, 0xffff_ffff_ffff, name)?;
    let (engine, wait_operation) = if size == 0 { (operation & 3, operation >> 2) } else { (operation & 1, operation >> 1) };
    require(wait_operation == 0, name, "only memory waits have a host form")?;
    let bits64 = size == 1;
    let (reference, mask) = if bits64 { (reference, mask) } else { (reference & 0xffff_ffff, mask & 0xffff_ffff) };
    Ok(Op::Wait { bits64, function: function.into(), pfp: !compute && engine == 1, address, reference, mask, poll: poll >> 4 })
}

agc_export! { fn sceAgcAcbWaitRegMem(buffer: *mut CommandBuffer, size: u8, function: u8, _cache_policy: u8, address: u64, reference: u64, mask: u64, poll: u32) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbWaitRegMem", || wait(true, size, function, 0, address, reference, mask, poll, "sceAgcAcbWaitRegMem"))
} }
agc_export! { fn sceAgcDcbWaitRegMem(buffer: *mut CommandBuffer, size: u8, function: u8, operation: u8, _cache_policy: u8, address: u64, reference: u64, mask: u64, poll: u32) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbWaitRegMem", || wait(false, size, function, operation, address, reference, mask, poll, "sceAgcDcbWaitRegMem"))
} }
agc_export! { fn sceAgcAcbWaitOnAddressGetSize(size: u32) -> u64 {
    if size > 1 { 0 } else { size_of(Kind::Wait { bits64: size == 1 }, "sceAgcAcbWaitOnAddressGetSize").into() }
} }
agc_export! { fn sceAgcDcbWaitOnAddressGetSize(size: u32) -> u32 {
    if size > 1 { 0 } else { size_of(Kind::Wait { bits64: size == 1 }, "sceAgcDcbWaitOnAddressGetSize") }
} }

agc_export! { fn sceAgcCbCondWrite(buffer: *mut CommandBuffer, function: u32, space: u32, address: u64, value: u32, poll: u64, reference: u32, mask: u32) -> *mut u32 {
    const F: &str = "sceAgcCbCondWrite";
    record(buffer, Queue::Any, F, || {
        require(function <= 6, F, "invalid compare function")?;
        require(space == 1, F, "only memory writes are supported")?;
        check_address(address, 4, F)?;
        check_address(poll, 4, F)?;
        require(address >> 48 == 0 && poll >> 48 == 0, F, "address beyond 48 bits")?;
        Ok(Op::CondWrite { function, poll, reference, mask, address, value })
    })
} }
agc_export! { fn sceAgcCbCondWriteGetSize() -> u32 { size_of(Kind::CondWrite, "sceAgcCbCondWriteGetSize") } }

fn semaphore(address: u64, _mailbox: u8, signal_type: u8, operation: u8, name: &str) -> Result<Op<'static>> {
    check_bits(signal_type.into(), 1, name)?;
    require(operation == 6 || operation == 7, name, "invalid semaphore operation")?;
    check_address(address, 8, name)?;
    // Operation 6 signals (signal type 1 writes 1, 0 increments), 7 waits.
    Ok(Op::Semaphore { address, signal: operation == 6, set_one: signal_type == 1 })
}

agc_export! { fn sceAgcAcbMemSemaphore(buffer: *mut CommandBuffer, address: u64, mailbox: u8, signal_type: u8, operation: u8) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbMemSemaphore", || semaphore(address, mailbox, signal_type, operation, "sceAgcAcbMemSemaphore"))
} }
agc_export! { fn sceAgcDcbMemSemaphore(buffer: *mut CommandBuffer, address: u64, mailbox: u8, signal_type: u8, operation: u8) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbMemSemaphore", || semaphore(address, mailbox, signal_type, operation, "sceAgcDcbMemSemaphore"))
} }

agc_export! { fn sceAgcDcbStallCommandBufferParser(buffer: *mut CommandBuffer) -> *mut u32 {
    emit(buffer, Queue::Graphics, &Op::SyncParser, "sceAgcDcbStallCommandBufferParser")
} }
agc_export! { fn sceAgcDcbStallCommandBufferParserGetSize() -> u32 { size_of(Kind::SyncParser, "sceAgcDcbStallCommandBufferParserGetSize") } }

// PRIME_UTCL2 (`(buf, permission, mode, address, pages)`, D19): a TLB prefetch the host does
// not need, recorded as a NOP of the same size.
agc_export! { fn sceAgcAcbPrimeUtcl2(buffer: *mut CommandBuffer, _permission: u32, _mode: u32, _address: u64, _pages: u32) -> *mut u32 {
    emit(buffer, Queue::Compute, &Op::Nop { dwords: 5, payload: None }, "sceAgcAcbPrimeUtcl2")
} }
agc_export! { fn sceAgcDcbPrimeUtcl2(buffer: *mut CommandBuffer, _permission: u32, _mode: u32, _address: u64, _pages: u32) -> *mut u32 {
    emit(buffer, Queue::Graphics, &Op::Nop { dwords: 5, payload: None }, "sceAgcDcbPrimeUtcl2")
} }
agc_export! { fn sceAgcAcbPrimeUtcl2GetSize() -> u32 { size_of(Kind::Nop { dwords: 5 }, "sceAgcAcbPrimeUtcl2GetSize") } }
agc_export! { fn sceAgcDcbPrimeUtcl2GetSize() -> u32 { size_of(Kind::Nop { dwords: 5 }, "sceAgcDcbPrimeUtcl2GetSize") } }

// Memory.

/// A guest WRITE_DATA (`WriteData`'s checks): on graphics bit 0 of `dst` picks the engine.
#[allow(clippy::too_many_arguments)]
unsafe fn write_data(compute: bool, dst: u8, cache_policy: u8, address: u64, data: *const u32, count: u32, increment: u8, confirm: u8, name: &str) -> Result<Op<'static>> {
    require(!data.is_null() && (1..=0x3ffd).contains(&count), name, "invalid write data payload")?;
    check_bits(dst.into(), if compute { 0xf } else { 0x1f }, name)?;
    check_bits(cache_policy.into(), 3, name)?;
    check_bits(increment.into(), 1, name)?;
    check_bits(confirm.into(), 1, name)?;
    if !compute {
        require(address & 3 == 0, name, "misaligned write address")?;
    }
    require(dst != 0 || confirm == 0, name, "register writes do not support write confirmation")?;
    let (select, pfp) = if compute { (u32::from(dst), false) } else { (u32::from(dst) >> 1, dst & 1 != 0) };
    // SAFETY: the title passes `count` dwords; the op outlives only this call.
    let values: &'static [u32] = unsafe { std::slice::from_raw_parts(data, count as usize) };
    // The console's bit 16 is ADDR_INCR: 1 keeps writing the same address.
    Ok(Op::WriteData { dst: select, pfp, address, values, increment: increment == 0, confirm: confirm != 0 })
}

agc_export! { fn sceAgcAcbWriteData(buffer: *mut CommandBuffer, dst: u8, cache_policy: u8, address: u64, data: *const u32, count: u32, increment: u8, confirm: u8) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbWriteData", || unsafe { write_data(true, dst, cache_policy, address, data, count, increment, confirm, "sceAgcAcbWriteData") })
} }
agc_export! { fn sceAgcDcbWriteData(buffer: *mut CommandBuffer, dst: u8, cache_policy: u8, address: u64, data: *const u32, count: u32, increment: u8, confirm: u8) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbWriteData", || unsafe { write_data(false, dst, cache_policy, address, data, count, increment, confirm, "sceAgcDcbWriteData") })
} }
agc_export! { fn sceAgcDcbWriteDataGetSize(count: u32) -> u32 {
    size_of(Kind::WriteData { dwords: count }, "sceAgcDcbWriteDataGetSize")
} }

/// A guest COPY_DATA (`WriteCopyData`'s checks): on graphics bit 0 of `src` is the engine and
/// bits 4:1 the selects.
#[allow(clippy::too_many_arguments)]
fn copy_data(compute: bool, dst: u8, dst_policy: u8, destination: u64, src: u8, src_policy: u8, source: u64, item: u8, confirm: u8, name: &str) -> Result<Op<'static>> {
    check_bits(dst_policy.into(), 3, name)?;
    check_bits(src_policy.into(), 3, name)?;
    check_bits(item.into(), 1, name)?;
    check_bits(confirm.into(), 1, name)?;
    let (src_select, dst_select, pfp) = if compute {
        (u32::from(src & 0xf), u32::from(dst & 0xf), false)
    } else {
        (u32::from((src >> 1) & 0xf), u32::from((dst >> 1) & 0xf), src & 1 != 0)
    };
    require(src_select != 5 || item != 0 || source >> 32 == 0, name, "immediate exceeds the 32-bit item size")?;
    Ok(Op::CopyData { src: src_select, dst: dst_select, pfp, source, destination, item64: item != 0, confirm: confirm != 0 })
}

agc_export! { fn sceAgcAcbCopyData(buffer: *mut CommandBuffer, dst: u8, dst_policy: u8, destination: u64, src: u8, src_policy: u8, source: u64, item: u8, confirm: u8) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbCopyData", || copy_data(true, dst, dst_policy, destination, src, src_policy, source, item, confirm, "sceAgcAcbCopyData"))
} }
agc_export! { fn sceAgcDcbCopyData(buffer: *mut CommandBuffer, dst: u8, dst_policy: u8, destination: u64, src: u8, src_policy: u8, source: u64, item: u8, confirm: u8) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbCopyData", || copy_data(false, dst, dst_policy, destination, src, src_policy, source, item, confirm, "sceAgcDcbCopyData"))
} }
agc_export! { fn sceAgcAcbCopyDataGetSize() -> u64 { size_of(Kind::CopyData, "sceAgcAcbCopyDataGetSize").into() } }
agc_export! { fn sceAgcDcbCopyDataGetSize() -> u64 { size_of(Kind::CopyData, "sceAgcDcbCopyDataGetSize").into() } }

/// A guest DMA end (`WriteDma`'s selects): 0 memory, 1 GDS, 2 data (source) or nowhere
/// (destination), 3 memory through L2; 0x14, 0x24, 0x25 and 0x64 name the console's own
/// memory.
fn dma_end(select: u8, address: u64, source: bool, name: &str) -> Result<DmaEnd> {
    if matches!(select, 0x14 | 0x24 | 0x25 | 0x64) {
        return Err(Error::unsupported(format!("{name}: DMA select 0x{select:x} reads console driver memory")));
    }
    check_bits(select.into(), 0xf, name)?;
    Ok(match select & 3 {
        0 | 3 => DmaEnd::Memory(address),
        1 => DmaEnd::Gds(address as u32),
        _ if source => DmaEnd::Data(address as u32),
        _ => DmaEnd::Nowhere,
    })
}

#[allow(clippy::too_many_arguments)]
fn dma_data(engine: u8, dst: u8, dst_policy: u8, destination: u64, src: u8, src_policy: u8, source: u64, bytes: u32, wait: u8, confirm: u8, block: u8, name: &str) -> Result<Op<'static>> {
    check_bits(engine.into(), 1, name)?;
    check_bits(dst.into(), 0xf, name)?;
    check_bits(dst_policy.into(), 3, name)?;
    check_bits(src_policy.into(), 3, name)?;
    check_bits(wait.into(), 1, name)?;
    check_bits(confirm.into(), 1, name)?;
    check_bits(block.into(), 1, name)?;
    check_bits(bytes.into(), 0x3ff_ffff, name)?;
    Ok(Op::DmaData {
        pfp: engine == 1,
        source: dma_end(src, source, true, name)?,
        destination: dma_end(dst, destination, false, name)?,
        bytes,
        wait_previous: wait != 0,
        confirm: confirm != 0,
    })
}

agc_export! { fn sceAgcAcbDmaData(buffer: *mut CommandBuffer, dst: u8, dst_policy: u8, destination: u64, src: u8, src_policy: u8, source: u64, bytes: u32, wait: u8, confirm: u8) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbDmaData", || dma_data(0, dst, dst_policy, destination, src, src_policy, source, bytes, wait, confirm, 0, "sceAgcAcbDmaData"))
} }
agc_export! { fn sceAgcDcbDmaData(buffer: *mut CommandBuffer, engine: u8, dst: u8, dst_policy: u8, destination: u64, src: u8, src_policy: u8, source: u64, bytes: u32, wait: u8, confirm: u8, block: u8) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbDmaData", || dma_data(engine, dst, dst_policy, destination, src, src_policy, source, bytes, wait, confirm, block, "sceAgcDcbDmaData"))
} }
agc_export! { fn sceAgcAcbDmaDataGetSize() -> u32 { size_of(Kind::DmaData, "sceAgcAcbDmaDataGetSize") } }
agc_export! { fn sceAgcDcbDmaDataGetSize() -> u32 { size_of(Kind::DmaData, "sceAgcDcbDmaDataGetSize") } }

/// A guest ATOMIC_MEM (`WriteAtomicMem`'s checks).
fn atomic_mem(op: u8, command: u8, cache_policy: u8, address: u64, source: u64, compare: u64, interval: u32, name: &str) -> Result<Op<'static>> {
    check_bits(op.into(), 0x7f, name)?;
    check_bits(command.into(), 0xf, name)?;
    check_bits(cache_policy.into(), 3, name)?;
    check_bits(interval.into(), 0x1fff, name)?;
    check_address(address, if op & 0x60 == 0x60 { 8 } else { 4 }, name)?;
    Ok(Op::AtomicMem { op: op.into(), command: command.into(), address, source, compare, interval })
}

// `AsyncCommandBuffer::atomicMem` passes the address before the cache policy (D19).
agc_export! { fn sceAgcAcbAtomicMem(buffer: *mut CommandBuffer, op: u8, command: u8, address: u64, cache_policy: u8, source: u64, compare: u64, interval: u32) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbAtomicMem", || atomic_mem(op, command, cache_policy, address, source, compare, interval, "sceAgcAcbAtomicMem"))
} }
// `DrawCommandBuffer::atomicMem` passes the CP engine first (D19).
agc_export! { fn sceAgcDcbAtomicMem(buffer: *mut CommandBuffer, _engine: u8, op: u8, command: u8, cache_policy: u8, address: u64, source: u64, compare: u64, interval: u32) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbAtomicMem", || atomic_mem(op, command, cache_policy, address, source, compare, interval, "sceAgcDcbAtomicMem"))
} }
agc_export! { fn sceAgcAcbAtomicMemGetSize() -> u32 { size_of(Kind::AtomicMem, "sceAgcAcbAtomicMemGetSize") } }
agc_export! { fn sceAgcDcbAtomicMemGetSize() -> u32 { size_of(Kind::AtomicMem, "sceAgcDcbAtomicMemGetSize") } }

// Control flow.

fn jump(chain: u8, cache_policy: u8, target: u64, dwords: u32, name: &str) -> Result<Op<'static>> {
    check_bits(chain.into(), 1, name)?;
    check_bits(cache_policy.into(), 3, name)?;
    check_bits(dwords.into(), 0xf_ffff, name)?;
    require(target & 3 == 0, name, "misaligned jump target")?;
    Ok(Op::Jump { target, dwords, chain: chain == 1 })
}

agc_export! { fn sceAgcAcbJump(buffer: *mut CommandBuffer, cache_policy: u8, target: u64, dwords: u32) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbJump", || jump(1, cache_policy, target, dwords, "sceAgcAcbJump"))
} }
agc_export! { fn sceAgcDcbJump(buffer: *mut CommandBuffer, mode: u8, cache_policy: u8, target: u64, dwords: u32) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbJump", || jump(mode, cache_policy, target, dwords, "sceAgcDcbJump"))
} }
agc_export! { fn sceAgcAcbJumpGetSize() -> u32 { size_of(Kind::Jump, "sceAgcAcbJumpGetSize") } }
agc_export! { fn sceAgcDcbJumpGetSize() -> u32 { size_of(Kind::Jump, "sceAgcDcbJumpGetSize") } }

fn cond_exec(address: u64, dwords: u32, name: &str) -> Result<Op<'static>> {
    check_address(address, 4, name)?;
    require(dwords <= 0x3fff, name, "conditional execution range is too large")?;
    Ok(Op::CondExec { address, dwords })
}

agc_export! { fn sceAgcAcbCondExec(buffer: *mut CommandBuffer, address: u64, dwords: u32) -> *mut u32 {
    record(buffer, Queue::Compute, "sceAgcAcbCondExec", || cond_exec(address, dwords, "sceAgcAcbCondExec"))
} }
agc_export! { fn sceAgcDcbCondExec(buffer: *mut CommandBuffer, address: u64, dwords: u32) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbCondExec", || cond_exec(address, dwords, "sceAgcDcbCondExec"))
} }
agc_export! { fn sceAgcAcbCondExecGetSize() -> u32 { size_of(Kind::CondExec, "sceAgcAcbCondExecGetSize") } }
agc_export! { fn sceAgcDcbCondExecGetSize() -> u32 { size_of(Kind::CondExec, "sceAgcDcbCondExecGetSize") } }

agc_export! { fn sceAgcCbNop(buffer: *mut CommandBuffer, dwords: u32) -> *mut u32 {
    record(buffer, Queue::Any, "sceAgcCbNop", || {
        require((1..=0x4001).contains(&dwords), "Header", "invalid packet header")?;
        Ok(Op::Nop { dwords, payload: None })
    })
} }
agc_export! { fn sceAgcCbNopGetSize(dwords: u32) -> u32 { size_of(Kind::Nop { dwords }, "sceAgcCbNopGetSize") } }

// Index, instance and indirect-argument state.

agc_export! { fn sceAgcDcbSetIndexBuffer(buffer: *mut CommandBuffer, address: u64) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbSetIndexBuffer", || {
        if address != 0 {
            check_address(address, 2, "sceAgcDcbSetIndexBuffer")?;
        }
        Ok(Op::IndexBase(address))
    })
} }
agc_export! { fn sceAgcDcbSetIndexBufferGetSize() -> u32 { size_of(Kind::IndexBase, "sceAgcDcbSetIndexBufferGetSize") } }
agc_export! { fn sceAgcDcbSetIndexCount(buffer: *mut CommandBuffer, count: u32) -> *mut u32 {
    emit(buffer, Queue::Graphics, &Op::IndexCount(count), "sceAgcDcbSetIndexCount")
} }
agc_export! { fn sceAgcDcbSetIndexCountGetSize() -> u32 { size_of(Kind::IndexCount, "sceAgcDcbSetIndexCountGetSize") } }

fn index_size(size: u8, cache_policy: u8, object_id: u8) -> Result<Op<'static>> {
    check_bits(object_id.into(), 1, "sceAgcDcbSetIndexSize_0600")?;
    require(size <= 2, "sceAgcDcbSetIndexSize", "invalid index element size")?;
    check_bits(cache_policy.into(), 3, "sceAgcDcbSetIndexSize")?;
    Ok(Op::IndexType(size.into()))
}

agc_export! { fn sceAgcDcbSetIndexSize(buffer: *mut CommandBuffer, size: u8, cache_policy: u8) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbSetIndexSize", || index_size(size, cache_policy, 0))
} }
agc_export! { fn sceAgcDcbSetIndexSize_0600(buffer: *mut CommandBuffer, size: u8, cache_policy: u8, object_id: u8) -> *mut u32 {
    record(buffer, Queue::Graphics, "sceAgcDcbSetIndexSize_0600", || index_size(size, cache_policy, object_id))
} }
agc_export! { fn sceAgcDcbSetIndexSizeGetSize() -> u32 { size_of(Kind::IndexType, "sceAgcDcbSetIndexSizeGetSize") } }
agc_export! { fn sceAgcDcbSetNumInstances(buffer: *mut CommandBuffer, count: u32) -> *mut u32 {
    emit(buffer, Queue::Graphics, &Op::Instances(count), "sceAgcDcbSetNumInstances")
} }
agc_export! { fn sceAgcDcbSetNumInstancesGetSize() -> u32 { size_of(Kind::Instances, "sceAgcDcbSetNumInstancesGetSize") } }
agc_export! { fn sceAgcDcbSetBaseIndirectArgs(buffer: *mut CommandBuffer, shader_type: u32, address: u64) -> *mut u32 {
    const F: &str = "sceAgcDcbSetBaseIndirectArgs";
    record(buffer, Queue::Graphics, F, || {
        check_bits(shader_type.into(), 1, F)?;
        if address != 0 {
            check_address(address, 8, F)?;
        }
        require(address >> 48 == 0, F, "address beyond 48 bits")?;
        Ok(Op::IndirectBase { dispatch: shader_type == 1, address })
    })
} }
agc_export! { fn sceAgcDcbSetBaseDispatchIndirectArgsGetSize() -> u32 { size_of(Kind::IndirectBase, "sceAgcDcbSetBaseDispatchIndirectArgsGetSize") } }
agc_export! { fn sceAgcDcbSetBaseDrawIndirectArgsGetSize() -> u32 { size_of(Kind::IndirectBase, "sceAgcDcbSetBaseDrawIndirectArgsGetSize") } }

// Predication (the SDK wrappers: z-pass `(c, 1, w, r)`, bool `(c, 3, 1, a)`, disable
// `(1, 0, 1, null)`; the count is never set and not used).
agc_export! { fn sceAgcDcbSetPredication(buffer: *mut CommandBuffer, condition: u8, operation: u8, wait: u8, address: u64, _count: u32) -> *mut u32 {
    const F: &str = "sceAgcDcbSetPredication";
    record(buffer, Queue::Graphics, F, || {
        check_bits(condition.into(), 1, F)?;
        check_bits(operation.into(), 7, F)?;
        check_bits(wait.into(), 1, F)?;
        if operation != 0 {
            check_address(address, 8, F)?;
        }
        Ok(Op::Predication { operation: operation.into(), condition: condition == 1, wait: wait == 1, address })
    })
} }
agc_export! { fn sceAgcDcbSetZPassPredicationEnableGetSize() -> u32 { size_of(Kind::Predication, "sceAgcDcbSetZPassPredicationEnableGetSize") } }
agc_export! { fn sceAgcDcbSetPredicationDisableGetSize() -> u32 { size_of(Kind::Predication, "sceAgcDcbSetPredicationDisableGetSize") } }
agc_export! { fn sceAgcDcbSetBoolPredicationEnableGetSize() -> u32 { size_of(Kind::Predication, "sceAgcDcbSetBoolPredicationEnableGetSize") } }

/// The libSceAgc functions this module exports, for `agc::exports`.
pub const COMMANDS: &[&str] = &[
    "sceAgcAcbPushMarker", "sceAgcAcbPushMarkerSpan", "sceAgcAcbSetMarker", "sceAgcAcbSetMarkerSpan", "sceAgcAcbPopMarker",
    "sceAgcDcbPushMarker", "sceAgcDcbPushMarkerSpan", "sceAgcDcbSetMarker", "sceAgcDcbSetMarkerSpan", "sceAgcDcbPopMarker",
    "sceAgcAcbSetWorkloadsActive", "sceAgcDcbSetWorkloadsActive", "sceAgcAcbSetWorkloadComplete", "sceAgcDcbSetWorkloadComplete",
    "sceAgcAcbSetWorkloadStreamInactive", "sceAgcDcbSetWorkloadStreamInactive",
    "sceAgcAcbSetFlip", "sceAgcDcbSetFlip", "sceAgcAcbWaitUntilSafeForRendering", "sceAgcDcbWaitUntilSafeForRendering",
    "sceAgcAcbResetQueue", "sceAgcDcbResetQueue", "sceAgcAcbRewind", "sceAgcDcbRewind", "sceAgcAcbRewindGetSize", "sceAgcDcbRewindGetSize",
    "sceAgcDcbAcquireMem", "sceAgcAcbAcquireMem", "sceAgcDcbAcquireMemGetSize", "sceAgcAcbAcquireMemGetSize",
    "sceAgcDcbContextStateOp_0100", "sceAgcDcbContextStateOpGetSize", "sceAgcDcbGetLodStats", "sceAgcDcbGetLodStatsGetSize",
    "sceAgcDcbBeginOcclusionQueryGetSize", "sceAgcDcbEndOcclusionQueryGetSize", "sceAgcCbBranchGetSize",
    "sceAgcAcbAtomicGdsGetSize", "sceAgcDcbAtomicGdsGetSize", "sceAgcDcbSetIndexIndirectArgsGetSize",
    "sceAgcCbReleaseMem", "sceAgcCbQueueEndOfPipeActionGetSize", "sceAgcAcbQueueEndOfShaderActionGetSize", "sceAgcDcbQueueEndOfShaderActionGetSize",
    "sceAgcAcbEventWrite", "sceAgcDcbEventWrite", "sceAgcAcbEventWriteGetSize", "sceAgcDcbEventWriteGetSize",
    "sceAgcAcbWaitRegMem", "sceAgcDcbWaitRegMem", "sceAgcAcbWaitOnAddressGetSize", "sceAgcDcbWaitOnAddressGetSize",
    "sceAgcCbCondWrite", "sceAgcCbCondWriteGetSize", "sceAgcAcbMemSemaphore", "sceAgcDcbMemSemaphore",
    "sceAgcDcbStallCommandBufferParser", "sceAgcDcbStallCommandBufferParserGetSize",
    "sceAgcAcbPrimeUtcl2", "sceAgcDcbPrimeUtcl2", "sceAgcAcbPrimeUtcl2GetSize", "sceAgcDcbPrimeUtcl2GetSize",
    "sceAgcAcbWriteData", "sceAgcDcbWriteData", "sceAgcDcbWriteDataGetSize",
    "sceAgcAcbCopyData", "sceAgcDcbCopyData", "sceAgcAcbCopyDataGetSize", "sceAgcDcbCopyDataGetSize",
    "sceAgcAcbDmaData", "sceAgcDcbDmaData", "sceAgcAcbDmaDataGetSize", "sceAgcDcbDmaDataGetSize",
    "sceAgcAcbAtomicMem", "sceAgcDcbAtomicMem", "sceAgcAcbAtomicMemGetSize", "sceAgcDcbAtomicMemGetSize",
    "sceAgcAcbJump", "sceAgcDcbJump", "sceAgcAcbJumpGetSize", "sceAgcDcbJumpGetSize",
    "sceAgcAcbCondExec", "sceAgcDcbCondExec", "sceAgcAcbCondExecGetSize", "sceAgcDcbCondExecGetSize",
    "sceAgcCbNop", "sceAgcCbNopGetSize",
    "sceAgcDcbSetIndexBuffer", "sceAgcDcbSetIndexBufferGetSize", "sceAgcDcbSetIndexCount", "sceAgcDcbSetIndexCountGetSize",
    "sceAgcDcbSetIndexSize", "sceAgcDcbSetIndexSize_0600", "sceAgcDcbSetIndexSizeGetSize",
    "sceAgcDcbSetNumInstances", "sceAgcDcbSetNumInstancesGetSize",
    "sceAgcDcbSetBaseIndirectArgs", "sceAgcDcbSetBaseDispatchIndirectArgsGetSize", "sceAgcDcbSetBaseDrawIndirectArgsGetSize",
    "sceAgcDcbSetPredication", "sceAgcDcbSetZPassPredicationEnableGetSize", "sceAgcDcbSetPredicationDisableGetSize",
    "sceAgcDcbSetBoolPredicationEnableGetSize",
];
