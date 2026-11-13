use super::*;

#[test]
fn live_recordings_share_packet_storage_and_fences_outlive_the_last_reference() {
    use crate::gpu::agc::arena::Range;
    let mut memory = vec![0u8; 8192];
    let arena = Arc::new(Arena::new(Range { cpu: memory.as_mut_ptr(), gpu: 0x80000, bytes: memory.len() }, 4096).unwrap());
    let packet = CommandBlock::new(&vec![0xc000_1000; 1024]);
    let mut tables = Vec::new();
    let mut address = None;
    for _ in 0..1000 {
        let table = SideTable::default();
        let range = table.place_commands(&arena, &packet).unwrap();
        assert_eq!(*address.get_or_insert(range.address), range.address);
        table.commands_submitted(9);
        tables.push(table);
    }
    assert_eq!(arena.usage().leased, 1);
    drop(tables);
    CommandCursor::retire_on_thread(&arena);
    let unused = arena.take().unwrap();
    assert!(arena.take().is_err(), "the shared packet must remain protected by its GPU fence");
    arena.advance(9);
    assert!(arena.take().is_ok());
    arena.give(unused);
}

#[test]
fn recording_and_submission_writers_allocate_without_sharing_mutable_cursors() {
    let mut memory = vec![0u8; 4 * 4096];
    let arena = Arc::new(Arena::new(super::super::arena::Range { cpu: memory.as_mut_ptr(), gpu: 0x10000, bytes: memory.len() }, 4096).unwrap());
    let table = Arc::new(SideTable::default());
    let barrier = Arc::new(std::sync::Barrier::new(5));
    let workers: Vec<_> = (0..4).map(|thread| {
        let (arena, table, barrier) = (arena.clone(), table.clone(), barrier.clone());
        std::thread::spawn(move || {
            barrier.wait();
            (0..100).map(|value| table.place_commands(&arena, &CommandBlock::new(&[0xc000_1000, thread * 100 + value])).unwrap().address).collect::<Vec<_>>()
        })
    }).collect();
    barrier.wait();
    for _ in 0..100 { table.commands_submitted(1); }
    let addresses: std::collections::BTreeSet<_> = workers.into_iter().flat_map(|worker| worker.join().unwrap()).collect();
    assert_eq!(addresses.len(), 400);
    table.commands_submitted(2);
    drop(table);
    assert!(arena.take().is_err());
    arena.advance(2);
    assert!(arena.take().is_ok());
}

#[test]
fn replay_signature_tracks_driver_rewrites_and_restoring_original_state() {
    let parameters = DrawParameters::default();
    let encoding = WorkEncoding::new(0xc002_3f00, Some(11), 0, parameters);
    assert!(encoding.matches(Some(11), 0, parameters));
    encoding.update(0xc002_1000, Some(22), 22, parameters);
    assert!(!encoding.matches(Some(11), 0, parameters));
    assert!(encoding.matches(Some(22), 22, parameters));
    assert_eq!(encoding.first.load(Ordering::Acquire), 0xc002_1000);
    encoding.update(0xc002_3f00, Some(11), 0, parameters);
    assert!(encoding.matches(Some(11), 0, parameters));
}

#[test]
fn command_copies_follow_recording_and_gpu_lifetimes() {
    let mut memory = vec![0u8; 1024];
    let arena = Arc::new(Arena::new(super::super::arena::Range { cpu: memory.as_mut_ptr(), gpu: 0x10000, bytes: memory.len() }, 1024).unwrap());
    let block = CommandBlock::new(&[0xc000_1000, 42]);
    let table = Arc::new(SideTable::default());
    let range = table.place_commands(&arena, &block).unwrap();
    assert_eq!(range.dwords, 8);
    assert_eq!(table.place_commands(&arena, &block).unwrap(), range);
    assert_eq!(u32::from_le_bytes(memory[4..8].try_into().unwrap()), 42);
    let pending = table.clone();
    drop(table);
    assert!(arena.take().is_err(), "a replayable or pending recording still owns its commands");
    pending.commands_submitted(7);
    drop(pending);
    CommandCursor::retire_on_thread(&arena);
    assert!(arena.take().is_err(), "GPU execution still reaches the retired commands");
    arena.advance(7);
    assert!(arena.take().is_ok());
}

#[test]
fn recording_churn_reuses_command_memory() {
    let mut memory = vec![0u8; 1024];
    let arena = Arc::new(Arena::new(super::super::arena::Range { cpu: memory.as_mut_ptr(), gpu: 0x10000, bytes: memory.len() }, 1024).unwrap());
    for value in 0..10_000 {
        let table = SideTable::default();
        table.place_commands(&arena, &CommandBlock::new(&[0xc000_1000, value])).unwrap();
    }
    CommandCursor::retire_on_thread(&arena);
    assert!(arena.take().is_ok());
}

#[test]
fn many_live_small_recordings_share_one_thread_block() {
    let mut memory = vec![0u8; 64 << 10];
    let arena = Arc::new(Arena::new(super::super::arena::Range { cpu: memory.as_mut_ptr(), gpu: 0x10000, bytes: memory.len() }, 64 << 10).unwrap());
    let mut tables = Vec::new();
    let mut addresses = HashSet::new();
    for value in 0..1000 {
        let table = SideTable::default();
        addresses.insert(table.place_commands(&arena, &CommandBlock::new(&[0xc000_1000, value])).unwrap().address);
        tables.push(table);
    }
    assert_eq!(addresses.len(), 1000);
    assert!(addresses.iter().all(|&address| address < 0x10000 + 32_000));
    CommandCursor::retire_on_thread(&arena);
    assert!(arena.take().is_err(), "all recorded ranges retain their shared backing block");
    drop(tables);
    assert!(arena.take().is_ok());
}

#[test]
fn several_new_chunks_jointly_release_an_overwritten_recording() {
    let (mut index, sender) = ChunkIndex::new();
    let (old, table) = chunk(0x1000, 0x2000, 0x2000);
    let old_weak = Arc::downgrade(&table);
    sender.send(old).unwrap();
    drop(table);
    let (left, _) = chunk(0x0800, 0x1600, 0x1600);
    sender.send(left).unwrap();
    index.owners(0x1000, 0x2000);
    assert!(old_weak.upgrade().is_some(), "the old tail still contains commands");
    let (right, _) = chunk(0x1600, 0x2200, 0x2200);
    sender.send(right).unwrap();
    index.owners(0x1000, 0x2000);
    index.prune_covered();
    assert!(old_weak.upgrade().is_none());
    assert_eq!(index.chunks().count(), 2);
}

#[test]
fn pruning_an_old_anchor_preserves_a_live_superseder_tail() {
    let (mut index, sender) = ChunkIndex::new();
    let (old, _) = chunk(0x2000, 0x3000, 0x3000);
    sender.send(old).unwrap();
    let (cover, cover_table) = chunk(0x1000, 0x4000, 0x2500);
    sender.send(cover).unwrap();
    let (prefix, _) = chunk(0x1000, 0x1900, 0x1800);
    sender.send(prefix).unwrap();
    let (suffix, _) = chunk(0x2500, 0x3500, 0x3500);
    sender.send(suffix).unwrap();
    index.owners(0x1000, 0x3500);
    index.prune_covered();
    assert_eq!(parts(&index.owners(0x2000, 0x2400)), vec![(0x2000, 0x2400, Arc::as_ptr(&cover_table))]);
    assert_eq!(parts(&index.owners(0x1800, 0x1900)), vec![(0x1800, 0x1900, Arc::as_ptr(&cover_table))]);
}

#[test]
fn ownership_queries_find_an_earlier_chunk_past_a_short_interior_overwrite() {
    let (mut index, sender) = ChunkIndex::new();
    let (outer, outer_table) = chunk(0x1000, 0x3000, 0x3000);
    sender.send(outer).unwrap();
    let (inner, _) = chunk(0x2000, 0x2100, 0x2100);
    sender.send(inner).unwrap();
    assert_eq!(parts(&index.owners(0x2200, 0x2300)), vec![(0x2200, 0x2300, Arc::as_ptr(&outer_table))]);
}

#[test]
fn appending_after_pruning_requests_a_new_index_publication() {
    let (chunk, _) = chunk(0x1000, 0x2000, 0x1100);
    assert!(chunk.prune_at(0x1100));
    assert!(chunk.extend(0x1200));
    assert_eq!(chunk.end(), 0x1200);
    assert!(!chunk.prune_at(0x1100), "a stale pruning extent must not discard appended commands");
    assert!(!chunk.extend(0x1300));
    assert!(chunk.prune_at(0x1300));
    assert!(chunk.extend(0x1100), "an in-place write must also request republication");
    assert_eq!(chunk.end(), 0x1300);
}

#[test]
fn overwritten_chunks_release_recordings_after_publication_and_pending_use() {
    let (mut index, sender) = ChunkIndex::new();
    let (old, old_table) = chunk(0x1800, 0x2000, 0x1900);
    let old_weak = Arc::downgrade(&old_table);
    sender.send(old).unwrap();
    let pending = index.owners(0x1800, 0x1900);
    drop(old_table);
    let (new, _) = chunk(0x1000, 0x2000, 0x1700);
    sender.send(new.clone()).unwrap();
    index.owners(0x1800, 0x1900);
    assert!(old_weak.upgrade().is_some(), "the newer recording has not overwritten the old commands yet");
    new.extend(0x1a00);
    index.owners(0x1800, 0x1900);
    assert!(old_weak.upgrade().is_some(), "pending work still holds the overwritten recording");
    drop(pending);
    assert!(old_weak.upgrade().is_none());
    assert_eq!(index.chunks().count(), 1);
}

#[test]
fn log_chunks_double_and_keep_every_entry() {
    assert_eq!(locate(0), (0, 0));
    assert_eq!(locate(FIRST_CHUNK - 1), (0, FIRST_CHUNK - 1));
    assert_eq!(locate(FIRST_CHUNK), (1, 0));
    assert_eq!(locate(3 * FIRST_CHUNK - 1), (1, 2 * FIRST_CHUNK - 1));
    assert_eq!(locate(3 * FIRST_CHUNK), (2, 0));
    let log = Log::default();
    for value in 0..10_000u64 {
        log.push(value);
    }
    assert_eq!(log.len(), 10_000);
    assert!(log.iter().copied().eq(0..10_000));
    assert_eq!(log.get(9_999), Some(&9_999));
    assert_eq!(log.get(10_000), None);
}

#[test]
fn log_drops_what_it_holds() {
    let counted = Arc::new(());
    {
        let log = Log::default();
        for _ in 0..1000 {
            log.push(counted.clone());
        }
        assert_eq!(Arc::strong_count(&counted), 1001);
    }
    assert_eq!(Arc::strong_count(&counted), 1);
}

#[test]
fn what_a_recording_noted_and_never_read_is_left_for_submission() {
    // A register list's footprint: a NOP that names the list in dwords 2 to 4.
    let footprint = [0xc003_1000u32, 0, 0x2000_0100, 0x12, 3];
    let list = footprint.as_ptr() as u64;
    assert_eq!(unsafe { named_list(list) }, Some((0x12_2000_0100, 3)));
    // The packet that loads a shader list's user data, or other commands: no list to read.
    let loaded = [0xc003_6300u32, 0x2000_0100, 0x12, 1 << 31, 3];
    assert_eq!(unsafe { named_list(loaded.as_ptr() as u64) }, None);
    let table = SideTable::default();
    let bank = RegisterBank::Context;
    // A list and a register behind it, read at the recording's draw; then a list after the draw.
    table.waiting.push(StateChange { at: 0x1000, kind: ChangeKind::ListAt { bank } });
    table.waiting.push(StateChange { at: 0x1014, kind: ChangeKind::Register { bank, offset: 0x202, value: 0x00cc_0000 } });
    table.settled.store(2, Ordering::Release);
    table.waiting.push(StateChange { at: list, kind: ChangeKind::ListAt { bank } });
    let unread: Vec<u64> = table.unsettled().map(|change| change.at).collect();
    assert_eq!(unread, vec![list]);
}

fn chunk(start: u64, limit: u64, end: u64) -> (Arc<Chunk>, Arc<SideTable>) {
    let table = Arc::new(SideTable::default());
    let chunk = Arc::new(Chunk::new(start, limit, table.clone()));
    chunk.extend(end);
    (chunk, table)
}

fn parts(owners: &[Owner]) -> Vec<(u64, u64, *const SideTable)> {
    owners.iter().map(|owner| (owner.from, owner.to, Arc::as_ptr(&owner.table))).collect()
}

#[test]
fn a_range_splits_between_the_recordings_that_wrote_it() {
    let (mut index, sender) = ChunkIndex::new();
    // A tail jump written by a separate recording before the body's: neither covers the other.
    let (jump, jump_table) = chunk(0x1090, 0x10a0, 0x10a0);
    let (body, body_table) = chunk(0x1000, 0x10a0, 0x1090);
    sender.send(jump).unwrap();
    sender.send(body).unwrap();
    let owners = index.owners(0x1000, 0x10a0);
    assert_eq!(parts(&owners), vec![(0x1000, 0x1090, Arc::as_ptr(&body_table)), (0x1090, 0x10a0, Arc::as_ptr(&jump_table))]);
    // A range inside the body.
    assert_eq!(parts(&index.owners(0x1010, 0x1020)), vec![(0x1010, 0x1020, Arc::as_ptr(&body_table))]);
    // Commands no recording wrote.
    assert!(index.owners(0x2000, 0x2010).is_empty());
}

#[test]
fn newer_recordings_take_over_reused_memory() {
    let (mut index, sender) = ChunkIndex::new();
    let (old, _) = chunk(0x1800, 0x1c00, 0x1900);
    sender.send(old).unwrap();
    let (new, new_table) = chunk(0x1000, 0x2000, 0x1c00);
    sender.send(new).unwrap();
    assert_eq!(parts(&index.owners(0x1000, 0x1c00)), vec![(0x1000, 0x1c00, Arc::as_ptr(&new_table))]);
    assert_eq!(parts(&index.owners(0x1880, 0x1890)), vec![(0x1880, 0x1890, Arc::as_ptr(&new_table))]);
    // A recording that starts where another started replaces it.
    let (again, again_table) = chunk(0x1000, 0x2000, 0x1100);
    sender.send(again).unwrap();
    assert_eq!(parts(&index.owners(0x1000, 0x1100)), vec![(0x1000, 0x1100, Arc::as_ptr(&again_table))]);
}
