use super::*;
use std::ffi::{c_char, c_void};

// Only ownership is exercised here: no device or rendering implementation is substituted.
unsafe extern "C" fn reserve(_: *mut c_void, _: *const FlipInfo, _: bool, _: *mut *mut FlipRequest, _: *mut Error) -> i32 { result::SHUTDOWN }
unsafe extern "C" fn room(_: *mut c_void, _: *mut Error) -> i32 { result::SHUTDOWN }
unsafe extern "C" fn capture(_: *mut c_void, _: u32, _: *mut *mut RenderingWait, _: *mut Error) -> i32 { result::SHUTDOWN }
unsafe extern "C" fn describe(_: *mut c_void, _: i32, _: *mut DisplayBuffer) -> bool { false }
unsafe extern "C" fn fail(_: *mut c_void, _: *const c_char) {}
unsafe extern "C" fn ready(_: *mut FlipRequest, _: u64, _: *mut Error) -> i32 { result::SHUTDOWN }
unsafe extern "C" fn flip_fail(_: *mut FlipRequest, _: *const c_char) {}
unsafe extern "C" fn flip_release(_: *mut FlipRequest) {}
unsafe extern "C" fn wait(_: *mut RenderingWait, _: u32, _: *mut Error) -> i32 { result::SHUTDOWN }
unsafe extern "C" fn wait_release(_: *mut RenderingWait) {}
unsafe extern "C" fn release(pointer: *mut c_void) {
    // SAFETY: the registered table owns this box and releases it exactly once.
    let counter = unsafe { Box::from_raw(pointer.cast::<Arc<AtomicU32>>()) };
    counter.fetch_add(1, Ordering::Relaxed);
}

fn output_table(identity: *mut c_void) -> VideoOutput {
    VideoOutput {
        this: identity, reserve: Some(reserve), wait_for_flip_room: Some(room), capture_rendering_wait: Some(capture),
        describe_buffer: Some(describe), fail: Some(fail), release: Some(release), flip_ready: Some(ready),
        flip_fail: Some(flip_fail), flip_release: Some(flip_release), rendering_wait_for: Some(wait), rendering_wait_release: Some(wait_release),
    }
}

#[test]
fn videoout_unregisters_after_gpu_stop_without_reclaiming_concurrent_readers() {
    let releases = Arc::new(AtomicU32::new(0));
    let identity = Box::into_raw(Box::new(releases.clone())).cast();
    let table = output_table(identity);
    let flips = Flips::default();
    flips.register(1, &table).unwrap();
    let reader = flips.output(1).unwrap();
    flips.shutdown();
    flips.shutdown();
    assert_eq!(flips.output(1).unwrap_err().code, result::SHUTDOWN);
    assert_eq!(flips.unregister(1, std::ptr::null_mut()).unwrap_err().code, result::INVALID);
    flips.unregister(1, identity).unwrap();
    assert_eq!(flips.unregister(1, identity).unwrap_err().code, result::INVALID);
    assert_eq!(releases.load(Ordering::Relaxed), 0, "late unregister must retain the registry's reference for concurrent acquisitions");
    drop(flips);
    assert_eq!(releases.load(Ordering::Relaxed), 0, "a captured reader still owns the output");
    drop(reader);
    assert_eq!(releases.load(Ordering::Relaxed), 1);
}

#[test]
fn shutdown_cancellation_stays_distinct_from_system_failure() {
    let cancelled = crate::Error::from(Failure::new(result::SHUTDOWN, "process exit"));
    assert_eq!(cancelled.kind, crate::ErrorKind::Shutdown);
    let failed = crate::Error::from(Failure::new(result::FAILED, "real failure"));
    assert_eq!(failed.kind, crate::ErrorKind::System);
}

struct Reservation {
    outputs: AtomicU32,
    requests: AtomicU32,
    entered: std::sync::Barrier,
    resume: std::sync::Barrier,
    pause: bool,
}

unsafe extern "C" fn reserve_request(pointer: *mut c_void, _: *const FlipInfo, _: bool, request: *mut *mut FlipRequest, _: *mut Error) -> i32 {
    // SAFETY: the output's registration owns the counter; the request gets its own reference.
    let state = unsafe { &*pointer.cast::<Arc<Reservation>>() };
    if state.pause { state.entered.wait(); state.resume.wait(); }
    unsafe { *request = Box::into_raw(Box::new(state.clone())).cast(); }
    result::OK
}

unsafe extern "C" fn release_request(pointer: *mut FlipRequest) {
    let state = unsafe { Box::from_raw(pointer.cast::<Arc<Reservation>>()) };
    state.requests.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn release_output(pointer: *mut c_void) {
    let state = unsafe { Box::from_raw(pointer.cast::<Arc<Reservation>>()) };
    state.outputs.fetch_add(1, Ordering::Relaxed);
}

#[test]
fn eop_reservations_release_once_when_shutdown_races_unregister_or_publication() {
    for pause in [false, true] {
        let counts = Arc::new(Reservation { outputs: AtomicU32::new(0), requests: AtomicU32::new(0), entered: std::sync::Barrier::new(2), resume: std::sync::Barrier::new(2), pause });
        let identity = Box::into_raw(Box::new(counts.clone())).cast();
        let mut table = output_table(identity);
        table.reserve = Some(reserve_request);
        table.flip_release = Some(release_request);
        table.release = Some(release_output);
        let flips = Flips::default();
        flips.register(1, &table).unwrap();
        let info = FlipInfo { handle: 1, index: 0, mode: 0, reserved: 0, argument: 0 };
        std::thread::scope(|scope| {
            if pause {
                let reserving = scope.spawn(|| flips.reserve_eop(&info, None));
                counts.entered.wait();
                flips.shutdown();
                flips.unregister(1, identity).unwrap();
                counts.resume.wait();
                assert_eq!(reserving.join().unwrap().unwrap_err().code, result::SHUTDOWN);
            } else {
                assert!(flips.reserve_eop(&info, None).unwrap().is_some());
                scope.spawn(|| flips.shutdown());
                flips.unregister(1, identity).unwrap();
            }
        });
        assert_eq!(counts.requests.load(Ordering::Relaxed), 1);
        assert_eq!(counts.outputs.load(Ordering::Relaxed), 0);
        drop(flips);
        assert_eq!(counts.outputs.load(Ordering::Relaxed), 1);
    }
}
