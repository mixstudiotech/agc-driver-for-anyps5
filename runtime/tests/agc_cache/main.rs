use agcdrv::agc::cache::Table;
use std::sync::{Arc, Barrier};

#[test]
fn saturated_cache_preserves_hits_and_leaves_room_for_misses() {
    let cache = Table::with_capacity(64);
    for key in 1..=1000 { cache.insert(key, Arc::new(key)); }
    assert_eq!(cache.len(), 48);
    for key in 1..=48 { assert_eq!(*cache.get(key).unwrap(), key); }
    assert!(cache.get(1000).is_none());
    assert_eq!(*cache.insert(1, Arc::new(99)), 1);
}

#[test]
fn simultaneous_inserts_cannot_consume_the_reserved_empty_slots() {
    let cache = Arc::new(Table::with_capacity(8));
    let barrier = Arc::new(Barrier::new(17));
    let threads: Vec<_> = (1..=16).map(|key| {
        let (cache, barrier) = (cache.clone(), barrier.clone());
        std::thread::spawn(move || { barrier.wait(); cache.insert(key, Arc::new(key)); })
    }).collect();
    barrier.wait();
    for thread in threads { thread.join().unwrap(); }
    assert_eq!(cache.len(), 6);
    assert!(cache.get(17).is_none());
}
