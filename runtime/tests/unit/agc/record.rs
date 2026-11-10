use super::*;
use crate::agc::target::{Op, Queue};
use std::sync::atomic::AtomicBool;

/// A recorder that counts the times its recording was looked up at a submission.
struct Counted(u32);

impl Recorder for Counted {
    fn encode(&mut self, _queue: Queue, _op: &Op<'_>, _out: &mut [u32], _address: u64) -> Result<()> {
        Ok(())
    }

    fn submitted(&mut self, _queue: Queue) {
        self.0 += 1;
    }
}

#[test]
fn a_lookup_by_address_is_not_in_the_way_of_other_recordings() {
    let recordings = Recordings::with_slots(64);
    // Another buffer, which the lookup passes on its way to the submitted one.
    recordings.enter(0x20, Cursor { bottom: 0x8000, up: 0x8000 }, || Box::new(Counted(0))).unwrap().stand(0x8000, 0x8000);
    // The submitted buffer: commands at 0x1000..0x1100.
    recordings.enter(0x10, Cursor { bottom: 0x1000, up: 0x1000 }, || Box::new(Counted(0))).unwrap().stand(0x1000, 0x1100);
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        // The other buffer is recorded command after command while the first is looked up.
        let recorder = scope.spawn(|| {
            let mut cursor = 0x8000;
            let mut commands = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let mut claim = recordings.enter(0x20, Cursor { bottom: 0x8000, up: cursor }, || Box::new(Counted(0))).expect("no other thread records this buffer");
                cursor += 8;
                claim.stand(0x8000, cursor);
                commands += 1;
            }
            commands
        });
        let mut found = 0;
        for _ in 0..200_000 {
            recordings.with_recording_at(0x1080, |recording| {
                recording.recorder.submitted(Queue::Graphics);
                found += 1;
            });
        }
        stop.store(true, Ordering::Relaxed);
        assert!(recorder.join().unwrap() > 0);
        assert_eq!(found, 200_000);
    });
    // A packet the title writes where a recording's commands end reaches that recording.
    let mut reached = false;
    recordings.with_recording_writing_at(0x1100, |_| reached = true);
    assert!(reached);
}

#[test]
fn a_thread_waits_out_a_lookup_of_the_buffer_it_records() {
    let recordings = Recordings::with_slots(64);
    recordings.enter(0x10, Cursor { bottom: 0x1000, up: 0x1000 }, || Box::new(Counted(0))).unwrap().stand(0x1000, 0x1100);
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        // The buffer's own thread goes on recording while its commands are looked up.
        let recorder = scope.spawn(|| {
            let mut cursor = 0x1100;
            while !stop.load(Ordering::Relaxed) {
                let mut claim = recordings.enter(0x10, Cursor { bottom: 0x1000, up: cursor }, || Box::new(Counted(0))).expect("a lookup is waited out");
                cursor += 8;
                claim.stand(0x1000, cursor);
            }
        });
        for _ in 0..200_000 {
            recordings.with_recording_at(0x1080, |recording| recording.recorder.submitted(Queue::Graphics));
        }
        stop.store(true, Ordering::Relaxed);
        recorder.join().unwrap();
    });
}
