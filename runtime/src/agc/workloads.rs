//! Workload streams (`sceAgcDriverRegisterWorkloadStream`): ids 1 to 31, one bit each, claimed
//! with CAS (D13: no locks).

use std::sync::atomic::{AtomicU32, Ordering};

static STREAMS: AtomicU32 = AtomicU32::new(0);

/// A free stream id, now registered.
pub fn register() -> Option<u32> {
    let mut current = STREAMS.load(Ordering::Acquire);
    loop {
        let free = !current & !1;
        if free == 0 {
            return None;
        }
        let id = free.trailing_zeros();
        match STREAMS.compare_exchange_weak(current, current | 1 << id, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Some(id),
            Err(now) => current = now,
        }
    }
}

/// Registers stream `id`, which the guest chose; false when it is out of range or taken.
pub fn claim(id: u32) -> bool {
    (1..32).contains(&id) && STREAMS.fetch_or(1 << id, Ordering::AcqRel) & 1 << id == 0
}

/// Releases `id`; false when it was not registered.
pub fn unregister(id: u32) -> bool {
    (1..32).contains(&id) && STREAMS.fetch_and(!(1 << id), Ordering::AcqRel) & 1 << id != 0
}

pub fn registered(id: u32) -> bool {
    (1..32).contains(&id) && STREAMS.load(Ordering::Acquire) & 1 << id != 0
}
