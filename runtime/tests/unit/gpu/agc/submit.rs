use super::*;
use crate::gpu::agc::side_table::{Cut, Notice};

fn stretch(start: u64, end: u64, user_data: u32) -> Resolved {
    Resolved {
        start,
        end,
        user_data: [user_data; USER_DATA_REGISTERS],
        cuts: Vec::new(),
        writes: Vec::new(),
        notices: Vec::new(),
        roots: Vec::new(),
        dma_writes: Vec::new(),
        guards: Vec::new(),
        marks: Vec::new(),
        rewritten: 0,
        rejected: 0,
        stale: Vec::new(),
        unread: 0,
    }
}

fn cut(at: u64, reason: CutReason, user_data: u32) -> (Cut, UserData) {
    (Cut { at, reason, changes: 0 }, [user_data; USER_DATA_REGISTERS])
}

#[test]
fn stretches_are_planned_in_order_with_their_cuts_events_and_user_data() {
    let wait = WaitCondition { address: 0x9000, function: 3, reference: 1, mask: u64::MAX, bytes: 4 };
    let mut entry = stretch(0x1000, 0x1090, 1);
    // A fast clear right after the stub's last command: the child's first segment carries it.
    entry.cuts.push(cut(0x1090, CutReason::TargetClear { metadata: 0x7000 }, 2));
    let mut child = stretch(0x4000, 0x4400, 3);
    child.cuts.push(cut(0x4100, CutReason::Wait(wait), 4));
    child.notices.push(Notice { at: 0x4200, event: Event::Flip { video_out: 1, buffer: 2, mode: 1, argument: 1 } });
    child.cuts.push(cut(0x4208, CutReason::AfterFlip, 5));
    let segments = plan(&[entry, child], 0, &AwaitedLabels::default());
    let ranges: Vec<(u64, u64)> = segments.iter().map(|segment| (segment.start, segment.end)).collect();
    assert_eq!(ranges, vec![(0x1000, 0x1090), (0x4000, 0x4100), (0x4100, 0x4208), (0x4208, 0x4400)]);
    let first_values: Vec<u32> = segments.iter().map(|segment| segment.user_data.unwrap()[0]).collect();
    assert_eq!(first_values, vec![1, 3, 4, 5]);
    assert_eq!(segments[1].clears, vec![0x7000]);
    assert_eq!(segments[2].wait, Some(wait));
    assert!(matches!(segments[2].events.as_slice(), [SubmitEvent::Flip(FlipRequest { handle: 1, index: 2, .. })]));
    assert!(segments.iter().enumerate().all(|(index, segment)| index == 2 || segment.events.is_empty()));
}

#[test]
fn an_empty_stretch_passes_what_it_waits_for_on() {
    let wait = WaitCondition { address: 0x9000, function: 3, reference: 1, mask: u64::MAX, bytes: 4 };
    let mut empty = stretch(0x1000, 0x1000, 1);
    empty.cuts.push(cut(0x1000, CutReason::Wait(wait), 1));
    let segments = plan(&[empty, stretch(0x2000, 0x2100, 2)], 0, &AwaitedLabels::default());
    assert_eq!(segments.len(), 1);
    assert_eq!((segments[0].start, segments[0].end, segments[0].wait), (0x2000, 0x2100, Some(wait)));
    assert_eq!(segments[0].user_data.unwrap()[0], 2);
}

#[test]
fn nothing_is_cut_inside_a_cond_exec_range() {
    let wait = WaitCondition { address: 0x9000, function: 3, reference: 1, mask: u64::MAX, bytes: 4 };
    let mut guarded = stretch(0x1000, 0x1400, 1);
    // A COND_EXEC at 0x1100 that may skip up to 0x1200: a wait in it stays on the GPU, a flip in
    // it ends its segment where the range ends, with the user data there.
    guarded.guards.push((0x1100, 0x1200));
    guarded.marks.push((0x1200, [7; USER_DATA_REGISTERS]));
    guarded.cuts.push(cut(0x1180, CutReason::Wait(wait), 2));
    guarded.notices.push(Notice { at: 0x1188, event: Event::Flip { video_out: 1, buffer: 0, mode: 1, argument: 0 } });
    guarded.cuts.push(cut(0x1190, CutReason::AfterFlip, 3));
    // A cut right before the COND_EXEC is fine.
    guarded.cuts.push(cut(0x1100, CutReason::Wait(wait), 4));
    let segments = plan(&[guarded], 0, &AwaitedLabels::default());
    let ranges: Vec<(u64, u64)> = segments.iter().map(|segment| (segment.start, segment.end)).collect();
    assert_eq!(ranges, vec![(0x1000, 0x1100), (0x1100, 0x1200), (0x1200, 0x1400)]);
    assert_eq!(segments[1].wait, Some(wait));
    assert!(matches!(segments[1].events.as_slice(), [SubmitEvent::Flip(_)]));
    assert_eq!((segments[2].wait, segments[2].user_data.unwrap()[0]), (None, 7));
}
