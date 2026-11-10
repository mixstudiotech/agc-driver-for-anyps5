use super::*;

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
