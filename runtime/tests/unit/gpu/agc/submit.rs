use super::*;
use crate::gpu::agc::side_table::{Cut, Notice};

fn stretch(start: u64, end: u64, user_data: u32) -> Resolved {
    Resolved {
        start,
        end,
        commands: vec![0; ((end - start) / 4) as usize].into(),
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
fn parked_submissions_keep_their_own_target_binding_on_every_submit_path() {
    let packet = header(pkt3::SET_CONTEXT_REG, 3, 0).unwrap();
    let register = (gfx10::CB_COLOR0_BASE::OFFSET - 0x28000) / 4;
    let mut live = vec![packet, register, 0xaaaa, packet, register, 0xaaaa];
    let start = live.as_ptr() as u64;
    let end = start + live.len() as u64 * 4;
    let mut first = stretch(start, end, 0);
    first.commands = live.clone().into();
    first.cuts.push(cut(start + 12, CutReason::Wait(WaitCondition {
        address: 0x9000, function: 3, reference: 1, mask: u64::MAX, bytes: 4,
    }), 0));
    let parked = plan(&[first], 0, &AwaitedLabels::default());
    // The same recording executed again with another inherited target. Each occurrence needs
    // its own translation even though the source address is identical.
    live[2] = 0xbbbb;
    live[5] = 0xbbbb;
    let mut second = stretch(start, end, 0);
    second.commands = live.clone().into();
    let later = plan(&[second], 0, &AwaitedLabels::default());
    live.fill(0xcccc_cccc);
    let before = host_ib(&[], &[], None, false, &parked[1]).unwrap();
    assert_eq!(&before[..3], &[packet, register, 0xaaaa], "the no-resource path must not CHAIN back into shared source memory");
    let ordered = ResourceCommands::default().image(&[], &[], None, false, &parked[1], None).unwrap();
    assert_eq!(ordered.source, [packet, register, 0xaaaa]);
    let traced = traced_ib(&[], &[], None, false, &parked[1], 0x8000, 1, &[]).unwrap();
    assert_eq!(traced.source, ordered.source);
    assert_eq!(later[0].commands(), [packet, register, 0xbbbb, packet, register, 0xbbbb]);
    assert_eq!(live, [0xcccc_cccc; 6], "submission assembly must not modify the recording");
}

#[test]
fn gpu_patches_to_a_later_segment_are_rejected_before_any_segment_runs() {
    let mut commands = stretch(0x1000, 0x1100, 0);
    commands.cuts.push(cut(0x1080, CutReason::UserData, 0));
    commands.dma_writes.push(DmaWrite { at: 0x1000, address: 0x1090, bytes: 4, source: None });
    let mut segments = plan(&[commands], 0, &AwaitedLabels::default());
    assert!(validate_executable_writes(&segments).is_err());
    segments[0].dma_writes[0].address = 0x1100;
    assert!(validate_executable_writes(&segments).is_ok(), "an adjacent data write does not modify executable bytes");
}

#[test]
fn target_roots_preserve_rebinding_and_dispatch_coverage_in_command_order() {
    use crate::gpu::agc::state::StateRecord;
    let root = |key, groups| Root {
        point: Default::default(),
        record: Arc::new(StateRecord {
            key,
            block: None,
            registers: Box::new([]),
            compute_packets: None,
            primitive_type: 0,
            stages: Vec::new(),
            watched: Box::new([]),
            targets: None,
        }),
        values: vec![7].into_boxed_slice(),
        groups,
    };
    let a = root(1, None);
    let mut commands = stretch(0x1000, 0x1100, 0);
    commands.roots = vec![(0x1000, a.clone()), (0x1020, root(2, None)), (0x1040, a), (0x1060, root(3, Some([1, 1, 1]))), (0x1080, root(3, Some([64, 1, 1])))];
    let segments = plan(&[commands], 0, &AwaitedLabels::default());
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].roots.iter().map(|root| root.record.key).collect::<Vec<_>>(), [1, 2, 1, 3, 3]);
    assert_eq!(segments[0].roots[4].groups, Some([64, 1, 1]));
}

#[test]
fn ordered_resource_commands_preserve_guard_entry_exit_and_nested_counts() {
    let nop = header(pkt3::NOP, 2, 0).unwrap();
    let guard = header(pkt3::COND_EXEC, 5, 0).unwrap();
    let source = [guard, 0x1000, 0, 0, 7, guard, 0x2000, 0, 0, 2, nop, 1, nop, 2];
    let mut resources = ResourceCommands::default();
    resources.before(0x8000, vec![nop, 10]);
    resources.before(0x8000 + 10 * 4, vec![nop, 20]);
    resources.after(0x8000 + 12 * 4, vec![nop, 30]);
    resources.before(0x8000 + 12 * 4, vec![nop, 40]);
    let image = resources.weave(&source, 0x8000, None).unwrap();
    assert_eq!(&image[..2], &[nop, 10]);
    assert_eq!(image[2 + 4], 11, "outer guard includes inner prelude and postlude");
    assert_eq!(image[2 + 5 + 4], 6, "inner guard excludes the next work's prelude");
    assert_eq!(&image[18..], &[nop, 40, nop, 2]);
}

#[test]
fn ordered_resource_commands_restore_compute_values_and_predicate_only_the_work() {
    let nop = header(pkt3::NOP, 2, 0).unwrap();
    let mut words = vec![nop, 0, header(pkt3::DISPATCH_DIRECT, 5, 2).unwrap(), 1, 1, 1, 1];
    predicate_dispatches(&mut words).unwrap();
    assert_eq!(words[0] & 1, 0);
    assert_eq!(words[2] & 1, 1);
    restore_compute_user(&[7; 16], &mut words).unwrap();
    assert_eq!(&words[words.len() - 16..], &[7; 16]);
}

#[test]
fn nested_guards_with_one_endpoint_skip_only_their_own_postludes() {
    let nop = header(pkt3::NOP, 2, 0).unwrap();
    let guard = header(pkt3::COND_EXEC, 5, 0).unwrap();
    let source = [guard, 0x1000, 0, 0, 7, guard, 0x2000, 0, 0, 2, nop, 1, nop, 2];
    let mut resources = ResourceCommands::default();
    resources.end_guard(0x8000 + 5 * 4, 0x8000 + 12 * 4, vec![nop, 10]);
    resources.end_guard(0x8000, 0x8000 + 12 * 4, vec![nop, 20]);
    let image = resources.weave(&source, 0x8000, None).unwrap();
    let inner_end = 10 + image[9] as usize;
    let outer_end = 5 + image[4] as usize;
    assert_eq!(&image[inner_end..inner_end + 2], &[nop, 20]);
    assert_eq!(&image[outer_end..], &[nop, 2]);
}

#[test]
fn stretches_are_planned_in_order_with_their_cuts_events_and_user_data() {
    let wait = WaitCondition { address: 0x9000, function: 3, reference: 1, mask: u64::MAX, bytes: 4 };
    let mut entry = stretch(0x1000, 0x1090, 1);
    // A boundary right after the stub carries to the child.
    entry.cuts.push(cut(0x1090, CutReason::FlipDone { video_out: 9, buffer: 1 }, 2));
    let mut child = stretch(0x4000, 0x4400, 3);
    child.cuts.push(cut(0x4100, CutReason::Wait(wait), 4));
    child.notices.push(Notice { at: 0x4200, event: Event::Flip { video_out: 1, buffer: 2, mode: 1, argument: 1 } });
    child.cuts.push(cut(0x4208, CutReason::AfterFlip, 5));
    let segments = plan(&[entry, child], 0, &AwaitedLabels::default());
    let ranges: Vec<(u64, u64)> = segments.iter().map(|segment| (segment.start, segment.end)).collect();
    assert_eq!(ranges, vec![(0x1000, 0x1090), (0x4000, 0x4100), (0x4100, 0x4208), (0x4208, 0x4400)]);
    let first_values: Vec<u32> = segments.iter().map(|segment| segment.user_data.unwrap()[0]).collect();
    assert_eq!(first_values, vec![1, 3, 4, 5]);
    assert_eq!(segments[1].flip_waits, vec![(9, 1)]);
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
