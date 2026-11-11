use super::*;
use crate::amd::surface::tiling::{SW_4KB_S, SW_64KB_R_X};

const COPIES: u64 = 1 << 40;
/// Memory a title hands to one transient target after another.
const POOL: u64 = 0x50_5000_0000;

fn target(width: u32, height: u32, bytes_per_element: u32) -> SurfaceDesc {
    SurfaceDesc { swizzle_mode: SW_64KB_R_X, width, height, depth: 1, layers: 1, mip_levels: 1, bytes_per_element, ..SurfaceDesc::default() }
}

fn directions(passes: &[CopyPass]) -> Vec<CopyDirection> {
    passes.iter().map(|pass| pass.direction).collect()
}

#[test]
fn ranges_merge_and_split() {
    let mut ranges = Ranges::default();
    assert!(ranges.insert(0x1000, 0x2000));
    assert!(ranges.insert(0x3000, 0x4000));
    assert!(!ranges.covers(0x1000, 0x4000));
    // With the gap filled the three are one.
    assert!(ranges.insert(0x2000, 0x3000));
    assert!(ranges.covers(0x1000, 0x4000) && !ranges.insert(0x1800, 0x3800));
    assert!(ranges.remove(0x1800, 0x2800));
    assert!(ranges.covers(0x1000, 0x1800) && ranges.covers(0x2800, 0x4000));
    assert!(!ranges.overlaps(0x1800, 0x2800) && ranges.overlaps(0x2700, 0x2900));
    assert!(!ranges.remove(0x1800, 0x2800));
}

#[test]
fn reused_target_memory_is_bound_as_the_gpu_left_it() {
    let mut tracker = SurfaceTracker::new(COPIES);
    let (full, half, narrow) = (target(3328, 1872, 4), target(1664, 936, 8), target(3328, 1872, 2));
    // The first use of the memory: nothing is known of it, so the view uploads it.
    let first = tracker.bind_passes(POOL, &full, SurfaceUse::Thin2d).unwrap();
    assert_eq!(directions(&first), [CopyDirection::ToHost]);
    tracker.host_written(POOL);
    // The title hands the memory on: another target at the same address, one at an address
    // inside it, the first again. Each finds what the GPU left there, and nothing is copied.
    for (address, desc) in [(POOL, &half), (POOL + 0x10_0000, &narrow), (POOL, &full), (POOL, &half), (POOL + 0x10_0000, &narrow)] {
        assert!(tracker.bind_passes(address, desc, SurfaceUse::Thin2d).unwrap().is_empty(), "0x{address:x} {desc:?}");
        tracker.host_written(address);
    }
    let counts = tracker.take_counts();
    assert_eq!((counts.uploads, counts.write_backs, counts.rendered_views), (1, 0, 4));
}

#[test]
fn what_is_written_in_guest_layout_is_uploaded() {
    let mut tracker = SurfaceTracker::new(COPIES);
    let (full, half, small) = (target(3328, 1872, 4), target(1664, 936, 8), target(256, 256, 4));
    tracker.bind_passes(POOL, &full, SurfaceUse::Thin2d).unwrap();
    tracker.host_written(POOL);
    // The CPU writes into rendered memory: the view tracked there uploads it, and so does a
    // view new to it.
    tracker.guest_written(POOL + 0x2000, 0x1000);
    assert_eq!(tracker.taken_as_rendered(POOL, &half, SurfaceUse::Thin2d), None);
    assert_eq!(directions(&tracker.bind_passes(POOL, &full, SurfaceUse::Thin2d).unwrap()), [CopyDirection::ToHost]);
    assert_eq!(directions(&tracker.bind_passes(POOL, &half, SurfaceUse::Thin2d).unwrap()), [CopyDirection::ToHost]);
    tracker.host_written(POOL);
    // A buffer store over the whole of a target clears it in guest memory, and the target
    // uploads that.
    assert!(tracker.guest_stored(POOL, SurfaceLayout::new(&half).unwrap().size_bytes()));
    assert_eq!(directions(&tracker.bind_passes(POOL, &half, SurfaceUse::Thin2d).unwrap()), [CopyDirection::ToHost]);
    tracker.host_written(POOL);
    // A store over part of a target is another use of the memory: the target stays as it is,
    // and a view new to the stored part uploads it.
    assert!(tracker.guest_stored(POOL + 0x10_0000, 0x4_0000));
    assert!(tracker.bind_passes(POOL, &half, SurfaceUse::Thin2d).unwrap().is_empty());
    assert_eq!(directions(&tracker.bind_passes(POOL + 0x10_0000, &small, SurfaceUse::Thin2d).unwrap()), [CopyDirection::ToHost]);
    assert!(tracker.bind_passes(POOL, &half, SurfaceUse::Thin2d).unwrap().is_empty());
}

#[test]
fn a_level_written_through_its_own_view_reaches_the_chain() {
    let mut tracker = SurfaceTracker::new(COPIES);
    let chain = SurfaceDesc { swizzle_mode: SW_4KB_S, width: 256, height: 256, depth: 1, layers: 1, mip_levels: 9, bytes_per_element: 8, ..SurfaceDesc::default() };
    let level = SurfaceDesc { width: 128, height: 128, mip_levels: 1, ..chain };
    let at = POOL + SurfaceLayout::new(&chain).unwrap().mip_level(1).unwrap().offset;
    tracker.bind_passes(POOL, &chain, SurfaceUse::Thin2d).unwrap();
    assert_eq!(directions(&tracker.bind_passes(at, &level, SurfaceUse::Thin2d).unwrap()), [CopyDirection::ToHost]);
    tracker.host_written(at);
    // The chain has the level's elements: they are written back, and the chain is retiled.
    assert_eq!(directions(&tracker.bind_passes(POOL, &chain, SurfaceUse::Thin2d).unwrap()), [CopyDirection::ToGuest, CopyDirection::ToHost]);
}

#[test]
fn a_filled_surface_has_its_copy_filled() {
    let mut tracker = SurfaceTracker::new(COPIES);
    let (full, half) = (target(3328, 1872, 4), target(1664, 936, 8));
    let bytes = SurfaceLayout::new(&full).unwrap().size_bytes();
    tracker.bind_passes(POOL, &full, SurfaceUse::Thin2d).unwrap();
    tracker.host_written(POOL);
    // A fill with one value all through: the target tracked there needs no upload to hold it,
    // and neither does a view new to the filled memory.
    let zero = Fill::new([0; 4], 16).unwrap();
    assert!(tracker.guest_filled(POOL, bytes, zero));
    assert_eq!(directions(&tracker.bind_passes(POOL, &full, SurfaceUse::Thin2d).unwrap()), [CopyDirection::Fill]);
    tracker.host_written(POOL);
    assert!(tracker.guest_filled(POOL, bytes, zero));
    assert_eq!(directions(&tracker.bind_passes(POOL, &half, SurfaceUse::Thin2d).unwrap()), [CopyDirection::Fill]);
    // The segment's events applied again in order: the fill, then the copy filled ahead of it.
    assert!(tracker.guest_filled(POOL, bytes, zero));
    tracker.fill_applied(POOL);
    assert!(tracker.bind_passes(POOL, &half, SurfaceUse::Thin2d).unwrap().is_empty());
    // A record of four different dwords gives 8-byte elements that depend on where they lie:
    // the surface is read.
    assert!(tracker.guest_filled(POOL, bytes, Fill::new([1, 2, 3, 4], 16).unwrap()));
    assert_eq!(directions(&tracker.bind_passes(POOL, &half, SurfaceUse::Thin2d).unwrap()), [CopyDirection::ToHost]);
    // Written by the CPU since, a filled surface is read as well.
    assert!(tracker.guest_filled(POOL, bytes, zero));
    tracker.guest_written(POOL + 0x1000, 0x1000);
    assert_eq!(directions(&tracker.bind_passes(POOL, &half, SurfaceUse::Thin2d).unwrap()), [CopyDirection::ToHost]);
}

#[test]
fn a_view_whose_memory_is_gone_is_left_as_it_is() {
    let mut tracker = SurfaceTracker::new(COPIES);
    let texture = target(256, 256, 4);
    let bytes = SurfaceLayout::new(&texture).unwrap().size_bytes();
    // The retile would read the view's memory: refused that, the view is not tracked.
    let refused = tracker.bind_readable_texture(POOL, &texture, SurfaceUse::Thin2d, 0, |ranges| {
        assert_eq!(ranges, [(POOL, bytes)]);
        false
    });
    assert!(refused.unwrap().is_none() && !tracker.view_current(POOL, &texture, SurfaceUse::Thin2d, 0));
    let bound = tracker.bind_readable_texture(POOL, &texture, SurfaceUse::Thin2d, 0, |_| true).unwrap().unwrap();
    assert_eq!(directions(&bound), [CopyDirection::ToHost]);
    // Current: nothing is read, so nothing is asked, until the CPU writes there.
    assert!(tracker.view_current(POOL, &texture, SurfaceUse::Thin2d, 0));
    assert!(tracker.bind_readable_texture(POOL, &texture, SurfaceUse::Thin2d, 0, |_| unreachable!()).unwrap().unwrap().is_empty());
    tracker.guest_written(POOL, 0x1000);
    assert!(!tracker.view_current(POOL, &texture, SurfaceUse::Thin2d, 0));
}
