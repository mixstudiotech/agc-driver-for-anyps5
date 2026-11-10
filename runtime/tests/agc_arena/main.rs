use agcdrv::gpu::agc::arena::{Arena, Range};
use std::sync::Arc;

#[test]
fn pending_recording_and_gpu_fence_both_keep_transient_block_alive() {
    let mut memory = vec![0u8; 1024];
    let arena = Arc::new(Arena::new(Range { cpu: memory.as_mut_ptr(), gpu: 0x10000, bytes: memory.len() }, 1024).unwrap());
    let block = arena.take().unwrap();
    let lease = arena.lease(block.index()).unwrap();
    arena.retire(block);
    arena.advance(0);
    assert!(arena.take().is_err(), "a pending recording must keep its block");
    arena.submitted(lease.index(), 1);
    drop(lease);
    assert!(arena.take().is_err(), "the GPU fence must still keep its block");
    arena.advance(1);
    assert!(arena.take().is_ok());
}
