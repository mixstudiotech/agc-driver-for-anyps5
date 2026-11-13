use shader::ir::{AddressForm, DescriptorTable, TableLink};

#[test]
fn idle_command_buffer_objects_do_not_keep_overwritten_gpu_storage() {
    use super::{arena::{Arena, Range}, side_table::{Chunk, ChunkIndex}, state::CommandBlock, AmdRecorder, AmdShared, GFX115};
    use std::sync::Arc;
    let mut memory = vec![0u8; 2048];
    let arena = Arc::new(Arena::new(Range { cpu: memory.as_mut_ptr(), gpu: 0x10000, bytes: memory.len() }, 1024).unwrap());
    let shared = Arc::new(AmdShared::default());
    let (mut index, sender) = ChunkIndex::new();
    let mut idle = Vec::new();
    for value in 0..1000 {
        let mut recorder = AmdRecorder::with_generation(GFX115, Arc::new(crate::agc::guest::ProcessMemory), shared.clone());
        recorder.table.place_commands(&arena, &CommandBlock::new(&[0xc000_1000, value])).unwrap();
        let chunk = Arc::new(Chunk::new(0x2000, 0x3000, recorder.table.clone()));
        chunk.extend(0x2020);
        sender.send(chunk.clone()).unwrap();
        recorder.chunk = Some(chunk);
        recorder.release_recording();
        idle.push(recorder);
        index.owners(0x2000, 0x2020);
    }
    assert_eq!(index.chunks().count(), 1);
    drop(index);
    assert!(idle.last().unwrap().saved_table.upgrade().is_none());
    idle[0].resume_recording();
    assert!(idle[0].chunk.is_none());
    assert_eq!(idle[0].waiting, 0);
    super::side_table::SideTable::retire_thread_commands(&arena);
    assert!(arena.take().is_ok(), "idle recorder registrations must not own GPU memory");
}

fn table() -> DescriptorTable {
    DescriptorTable { root: 0, root_form: AddressForm::Pointer, links: Vec::new(), offset: 0x2b8 }
}

#[test]
fn optional_tables_stop_at_unbound_roots_and_links() {
    let mut source = table();
    assert_eq!(source.bound_address(|_| 0, |_| panic!("unbound table must not be read")), None);
    source.root_form = AddressForm::Buffer;
    assert_eq!(source.bound_address(|sgpr| if sgpr == 1 { 0x0010_0000 } else { 0 }, |_| panic!("flags are not an address")), None);
    source.root_form = AddressForm::Pointer;
    source.links.push(TableLink { offset: 8, form: AddressForm::Pointer });
    let mut reads = Vec::new();
    assert_eq!(source.bound_address(|sgpr| if sgpr == 0 { 0x1000 } else { 0 }, |at| { reads.push(at); Some([0, 0]) }), None);
    assert_eq!(reads, [0x1008]);
}

#[test]
fn optional_tables_preserve_bound_addresses_and_scalar_alignment() {
    let mut source = table();
    source.links.push(TableLink { offset: 11, form: AddressForm::Buffer });
    source.offset = -5;
    let user = |sgpr| if sgpr == 0 { 0x1000 } else { 0 };
    let read = |at| { assert_eq!(at, 0x1008); Some([0x2000, 0x0010_0000]) };
    assert_eq!(source.bound_address(user, read), Some(0x1ff8));
    assert_eq!(source.address(user, read), Some(0x1ff8));
    source = table();
    assert_eq!(source.address(|_| 0, |_| None), Some(0x2b8));
}
