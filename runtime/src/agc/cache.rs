//! An insert-only map from 64-bit content keys to shared values, for state that recordings on
//! any thread reuse (translated draw states, state IBs): fixed capacity, linear probing, keys
//! claimed with CAS, values published with a release store (D13: no locks).
//!
//! Key 0 is reserved for empty slots; callers fold it into another key. The last quarter of the
//! table stays empty so a miss remains bounded when new states stop being cached.

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

struct Slot<V> {
    key: AtomicU64,
    value: AtomicPtr<V>,
}

pub struct Table<V> {
    slots: Box<[Slot<V>]>,
    mask: usize,
    len: AtomicUsize,
}

// SAFETY: values are only shared as `Arc<V>`, which needs `V: Send + Sync` to cross threads.
unsafe impl<V: Send + Sync> Send for Table<V> {}
unsafe impl<V: Send + Sync> Sync for Table<V> {}

impl<V> Table<V> {
    /// A table of `capacity` slots, rounded up to a power of two.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(2).next_power_of_two();
        let slots = (0..capacity)
            .map(|_| Slot { key: AtomicU64::new(0), value: AtomicPtr::new(std::ptr::null_mut()) })
            .collect();
        Self { slots, mask: capacity - 1, len: AtomicUsize::new(0) }
    }

    fn start(&self, key: u64) -> usize {
        // Fibonacci hashing: content keys are already well mixed, this only spreads clusters.
        (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize & self.mask
    }

    /// The value stored for `key`.
    pub fn get(&self, key: u64) -> Option<Arc<V>> {
        debug_assert_ne!(key, 0);
        let mut index = self.start(key);
        for _ in 0..self.slots.len() {
            let slot = &self.slots[index];
            match slot.key.load(Ordering::Acquire) {
                0 => return None,
                found if found == key => {
                    let value = slot.value.load(Ordering::Acquire);
                    if value.is_null() {
                        // Claimed, not yet published: as good as absent.
                        return None;
                    }
                    // SAFETY: published values are `Arc::into_raw` pointers that are never freed
                    // while the table lives; this adds a reference for the caller.
                    return Some(unsafe {
                        Arc::increment_strong_count(value);
                        Arc::from_raw(value)
                    });
                }
                _ => index = (index + 1) & self.mask,
            }
        }
        None
    }

    /// Whether a value is stored for `key` (without taking a reference to it).
    pub fn contains(&self, key: u64) -> bool {
        let mut index = self.start(key);
        for _ in 0..self.slots.len() {
            let slot = &self.slots[index];
            match slot.key.load(Ordering::Acquire) {
                0 => return false,
                found if found == key => return !slot.value.load(Ordering::Acquire).is_null(),
                _ => index = (index + 1) & self.mask,
            }
        }
        false
    }

    /// Stores `value` for `key` unless another thread stored one first, and returns the value
    /// the table holds for `key` (or `value` itself when the table is full).
    pub fn insert(&self, key: u64, value: Arc<V>) -> Arc<V> {
        debug_assert_ne!(key, 0);
        if self.len.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |len| (len < self.slots.len() * 3 / 4).then_some(len + 1)).is_err() {
            return self.get(key).unwrap_or(value);
        }
        let mut index = self.start(key);
        for _ in 0..self.slots.len() {
            let slot = &self.slots[index];
            match slot.key.compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    slot.value.store(Arc::into_raw(value.clone()) as *mut V, Ordering::Release);
                    return value;
                }
                Err(found) if found == key => {
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    return self.get(key).unwrap_or(value);
                }
                Err(_) => index = (index + 1) & self.mask,
            }
        }
        self.len.fetch_sub(1, Ordering::Relaxed);
        value
    }

    /// Keys stored so far.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<V> Drop for Table<V> {
    fn drop(&mut self) {
        for slot in self.slots.iter_mut() {
            let value = *slot.value.get_mut();
            if !value.is_null() {
                // SAFETY: each published pointer came from `Arc::into_raw` once.
                drop(unsafe { Arc::from_raw(value) });
            }
        }
    }
}

/// A 64-bit content hash for keys (FNV-1a over words, then a final mix), never 0.
pub fn key(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for word in words {
        hash = (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mixed = mix(hash);
    if mixed == 0 {
        1
    } else {
        mixed
    }
}

/// SplitMix64's finalizer.
pub const fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
