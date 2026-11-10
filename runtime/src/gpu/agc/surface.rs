//! Guest surfaces on the host: their host-layout copies at guest VA + 2^40, and which side of
//! each is current (decisions D12, D27). Render targets and the storage images of compute work
//! live in the copies; textures retile into theirs, each level when a view first reaches it. Runs
//! on the device thread at submission.

use crate::agc::state::GuestRegisters;
use crate::amd::defs::gfx10;
use crate::amd::surface::{
    formats::TexelFormat,
    retile::{clear_dispatches, copy_dispatches, CopyDispatch, COPY_GROUP_THREADS, COPY_PARAM_DWORDS},
    tiling::{host_swizzle_mode, split_pipe_bank_xor, SurfaceDesc, SurfaceLayout, SurfaceUse, SW_LINEAR},
};
use crate::gpu::kernels::element_copy;
use crate::gpu::packets::{event, gcr, PacketStream};
use crate::gpu::regs as host;
use crate::ir::RegisterBank::Context;
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

/// Bytes each dispatch's parameter block takes in the parameter memory of a pass.
pub const PARAM_STRIDE: u64 = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CopyDirection {
    /// Guest layout into the host-layout copy (retile).
    ToHost,
    /// Host-layout copy back into guest memory (detile).
    ToGuest,
}

/// How many CPU write checks a surface the CPU wrote stays in every check.
const HOT_WRITE_CHECKS: u64 = 16;
/// The checks a surface the CPU has not written recently waits between: each check asks about one
/// slice of them, so a first write to such a surface is seen within this many flips.
const COLD_WRITE_SLICES: u64 = 16;

/// The dispatches that bring one side of a surface up to date.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CopyPass {
    pub direction: CopyDirection,
    pub guest_base: u64,
    /// The levels `dispatches` copy, one dispatch each.
    pub levels: Range<u32>,
    pub dispatches: Vec<CopyDispatch>,
}

/// The transitions of one view binding ([`SurfaceTracker::plan_view`]).
struct ViewPlan {
    guest_base: u64,
    /// The view's guest and host layouts, when it is new or replaces another view of its base.
    replace: Option<(Rc<SurfaceLayout>, Rc<SurfaceLayout>)>,
    /// Dirty aliases whose host copies `to_guest` detiles first.
    materialized: Vec<u64>,
    to_guest: Vec<CopyPass>,
    /// The retile of the levels the view reaches that its host copy lacks.
    to_host: Option<CopyPass>,
}

#[derive(Clone, Debug)]
struct Tracked {
    guest: Rc<SurfaceLayout>,
    host: Rc<SurfaceLayout>,
    host_current: bool,
    guest_current: bool,
    /// The finest level the host copy holds, while it is current. A title streaming a mip chain
    /// in keeps the finer levels unmapped and out of its descriptors' reach (MIN_LOD): a level
    /// enters the copy when a view first reaches it, after the title has filled it.
    first_held_level: u32,
    /// The last CPU write check that found CPU writes to the surface.
    written_check: u64,
}

impl Tracked {
    fn level_count(&self) -> u32 {
        self.guest.desc().mip_levels.max(1)
    }

    /// Layers, slabs or slices of the guest layout, each with the whole mip chain.
    fn slices(&self) -> u64 {
        match self.guest.slice_stride() {
            0 => 1,
            stride => self.guest.size_bytes().div_ceil(stride),
        }
    }

    /// Whether a write to `[address, address + bytes)` of the guest memory of the surface at
    /// `base` makes its host copy stale: not if it only touches levels the copy does not hold.
    fn holds(&self, base: u64, address: u64, bytes: u64) -> bool {
        if !self.host_current {
            return true;
        }
        let (end, stride) = (address.saturating_add(bytes), self.guest.slice_stride());
        (0..self.slices()).any(|slice| {
            (self.first_held_level..self.level_count())
                .filter_map(|level| self.guest.mip_level(level))
                .any(|mip| {
                    let start = base + slice * stride + mip.offset;
                    start < end && address < start + mip.size
                })
        })
    }

    /// The detile of the levels the host copy holds.
    fn detile(&self, base: u64, copy_offset: u64) -> Result<CopyPass> {
        let levels = self.first_held_level..self.level_count();
        let dispatches = copy_dispatches(&self.host, base + copy_offset, &self.guest, base, levels.clone())?;
        Ok(CopyPass { direction: CopyDirection::ToGuest, guest_base: base, levels, dispatches })
    }
}

/// Which side of each guest surface with a host-layout copy is current. A surface is tracked from
/// its first use on the host. Guest-memory writes and reads are reported by whoever sees them;
/// until the runtime tracks CPU and DMA access, [`Self::guest_written`] and
/// [`Self::guest_read`] are the stubs those reports go to.
#[derive(Clone, Debug)]
pub struct SurfaceTracker {
    copy_offset: u64,
    surfaces: BTreeMap<u64, Tracked>,
    metadata: BTreeMap<(u64, u64), Rc<[u8]>>,
    /// No tracked surface's guest or host layout is larger: a surface overlapping a range
    /// starts less than this before it ([`Self::near`]).
    largest: u64,
    /// Changes whenever the tracked surfaces, their layouts, which side is current or the
    /// observed metadata change ([`Self::epoch`]).
    epoch: u64,
    /// Changes whenever [`Self::copies`] may list something else.
    copies_generation: u64,
    /// CPU write checks so far ([`Self::write_check_ranges`]).
    write_check: u64,
}

impl SurfaceTracker {
    /// Host-layout copies sit `copy_offset` above their guest surfaces.
    pub fn new(copy_offset: u64) -> Self {
        Self {
            copy_offset,
            surfaces: BTreeMap::new(),
            metadata: BTreeMap::new(),
            largest: 0,
            epoch: 0,
            copies_generation: 0,
            write_check: 0,
        }
    }

    /// A counter that changes with every change of the tracker's state: equal epochs mean a
    /// binding that changed nothing before changes nothing again.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// A counter that changes whenever [`Self::copies`] may list something else.
    pub fn copies_generation(&self) -> u64 {
        self.copies_generation
    }

    /// The keys of the tracked surfaces that may overlap `[address, address + bytes)` in guest
    /// or host layout: only those starting less than [`Self::largest`] before it, found by a range
    /// of the map instead of a walk over every surface (bindings did that for each surface and
    /// each buffer of each draw).
    fn near(&self, address: u64, bytes: u64) -> std::ops::Range<u64> {
        address.saturating_sub(self.largest)..address.saturating_add(bytes)
    }

    /// The host-layout copy of the surface at `guest_base`, if it is tracked.
    pub fn host_layout(&self, guest_base: u64) -> Option<&SurfaceLayout> {
        self.surfaces.get(&guest_base).map(|tracked| &*tracked.host)
    }

    /// A current, uncompressed render surface; its GPU address and actual host layout.
    pub fn display_surface(&self, guest_base: u64) -> Option<(u64, &SurfaceLayout)> {
        self.surfaces.get(&guest_base).filter(|tracked| tracked.host_current && tracked.first_held_level == 0)
            .map(|tracked| (guest_base + self.copy_offset, &*tracked.host))
    }

    /// The guest ranges this flip's CPU write check asks about: the surfaces the CPU wrote in
    /// the last [`HOT_WRITE_CHECKS`] checks, and one [`COLD_WRITE_SLICES`]th of the others, in
    /// turn. Asking costs time for every page asked about, and streaming textures are the ones
    /// the CPU keeps writing.
    pub fn write_check_ranges(&mut self) -> Vec<(u64, u64)> {
        self.write_check += 1;
        let check = self.write_check;
        self.surfaces
            .iter()
            .filter(|(&base, tracked)| tracked.written_check + HOT_WRITE_CHECKS >= check || (base >> 16) % COLD_WRITE_SLICES == check % COLD_WRITE_SLICES)
            .map(|(&base, tracked)| (base, tracked.guest.size_bytes()))
            .collect()
    }

    /// The guest memory of `levels` of the surface at `guest_base`, in merged `(address, bytes)`
    /// ranges.
    pub fn guest_level_ranges(&self, guest_base: u64, levels: Range<u32>) -> Vec<(u64, u64)> {
        let Some(tracked) = self.surfaces.get(&guest_base) else {
            return Vec::new();
        };
        let stride = tracked.guest.slice_stride();
        let mut ranges: Vec<(u64, u64)> = (0..tracked.slices())
            .flat_map(|slice| {
                levels.clone().filter_map(move |level| {
                    let mip = tracked.guest.mip_level(level)?;
                    (mip.size != 0).then_some((guest_base + slice * stride + mip.offset, mip.size))
                })
            })
            .collect();
        ranges.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
        for (start, size) in ranges {
            match merged.last_mut() {
                Some((address, bytes)) if start <= *address + *bytes => {
                    *bytes = (*bytes).max(start + size - *address);
                }
                _ => merged.push((start, size)),
            }
        }
        merged
    }

    /// The tracked surface a view is a window on ([`SurfaceLayout::is_window`]), in guest and
    /// host layout both: the view then uses that surface's host copy.
    fn window_owner(&self, guest_base: u64, guest: &SurfaceLayout, host: &SurfaceLayout) -> Option<u64> {
        let (&base, tracked) = self.surfaces.range(..=guest_base).next_back()?;
        if guest_base.checked_add(guest.size_bytes())? > base + tracked.guest.size_bytes() {
            return None;
        }
        let offset = guest_base - base;
        let stride = tracked.guest.slice_stride();
        // A view's host address is its guest address moved by the copy offset, so a layer other
        // than the first lies at the same offset in both layouts only if they space layers alike.
        if stride == 0 || offset % stride != 0 || (offset != 0 && stride != tracked.host.slice_stride()) {
            return None;
        }
        let first_layer = u32::try_from(offset / stride).ok()?;
        (tracked.guest.is_window(guest, first_layer) && tracked.host.is_window(host, first_layer)).then_some(base)
    }

    /// Every tracked surface: its guest base, and the address and size of its host-layout copy.
    pub fn copies(&self) -> Vec<(u64, u64, u64)> {
        self.surfaces
            .iter()
            .map(|(&base, tracked)| (base, base + self.copy_offset, tracked.host.size_bytes()))
            .collect()
    }

    /// Ranges whose newest contents live only in their host copy. Before CPU access or a
    /// mapping change the caller must execute [`Self::guest_read`]'s passes and wait for them.
    pub fn dirty_ranges(&self) -> Vec<(u64, u64)> {
        self.surfaces
            .iter()
            .filter(|(_, tracked)| tracked.host_current && !tracked.guest_current)
            .map(|(&base, tracked)| (base, tracked.guest.size_bytes()))
            .collect()
    }

    /// Forget the views entirely inside a released guest range, after its GPU users have retired.
    /// A streamed texture has pages of a mip unmapped while the rest stays bound: a view only part
    /// of which goes keeps its host copy, and its descriptors keep working.
    pub fn remove_range(&mut self, address: u64, bytes: u64) {
        self.epoch += 1;
        self.copies_generation += 1;
        let end = address.saturating_add(bytes);
        self.surfaces
            .retain(|&base, tracked| !(address <= base && base.saturating_add(tracked.guest.size_bytes()) <= end));
        self.metadata.retain(|&(base, meta), keys| {
            self.surfaces.contains_key(&base)
                && !(bytes != 0
                    && address < meta.saturating_add(keys.len() as u64)
                    && meta < address.saturating_add(bytes))
        });
    }

    /// Records a coherent metadata extent. Exact bytes are compared (not a collision-prone
    /// hash) so a newly issued metadata-only clear invalidates an existing host image.
    pub fn observe_metadata(&mut self, guest_base: u64, address: u64, keys: &[u8]) -> bool {
        let key = (guest_base, address);
        if self
            .metadata
            .get(&key)
            .is_some_and(|old| old.as_ref() == keys)
        {
            return false;
        }
        self.metadata.insert(key, Rc::from(keys));
        self.epoch += 1;
        true
    }

    /// Ordered transitions before using a view: materialize dirty aliases in guest layout,
    /// then populate this view's host layout. Copies share their fixed guest-VA + offset
    /// storage, so changing a view must invalidate other views of the overwritten storage.
    /// This follows radv's separation of image view state and allocation/metadata ownership:
    /// matching base addresses alone never prove that two image layouts are interchangeable.
    pub fn bind_passes(
        &mut self,
        guest_base: u64,
        desc: &SurfaceDesc,
        usage: SurfaceUse,
    ) -> Result<Vec<CopyPass>> {
        self.bind_texture_passes(guest_base, desc, usage, 0)
    }

    /// What binding a view takes, worked out without changing the tracker: every step that
    /// can fail happens here, so [`Self::apply_view`] cannot fail half way and no binding needs
    /// a copy of the tracker to roll back to (copying it for every bind of every draw cost most
    /// of a frame's translation time).
    fn plan_view(
        &self,
        guest_base: u64,
        desc: &SurfaceDesc,
        usage: SurfaceUse,
        first_level: u32,
    ) -> Result<ViewPlan> {
        let host_desc = SurfaceDesc {
            swizzle_mode: host_swizzle_mode(desc.swizzle_mode, usage),
            pipe_bank_xor: 0,
            pitch_elements: 0,
            ..*desc
        };
        let existing = self
            .surfaces
            .get(&guest_base)
            .filter(|tracked| tracked.guest.desc() == desc && tracked.host.desc() == &host_desc);
        // A tracked view with the same layouts keeps them, a view that is a window on a tracked
        // surface uses that surface's copy, and any other view of the base replaces it.
        let (guest_base, existing, replace) = match existing {
            Some(tracked) => (guest_base, Some(tracked), None),
            None => {
                let (guest, host) = (Rc::new(SurfaceLayout::new(desc)?), Rc::new(SurfaceLayout::host(&host_desc)?));
                match self.window_owner(guest_base, &guest, &host) {
                    Some(owner) => (owner, self.surfaces.get(&owner), None),
                    None => (guest_base, None, Some((guest, host))),
                }
            }
        };
        let (guest, host) = match (&replace, existing) {
            (Some((guest, host)), _) => (guest, host),
            (None, Some(tracked)) => (&tracked.guest, &tracked.host),
            (None, None) => unreachable!("a view is either tracked or replaced"),
        };
        let (guest_bytes, host_bytes) = (guest.size_bytes(), host.size_bytes());
        let mut passes = Vec::new();
        let mut materialized = Vec::new();
        for (&base, tracked) in self
            .surfaces
            .range(self.near(guest_base, guest_bytes.max(host_bytes)))
        {
            if (base == guest_base && existing.is_some())
                || !(overlaps(base, &tracked.guest, guest_base, guest_bytes)
                    || overlaps(base, &tracked.host, guest_base, host_bytes))
            {
                continue;
            }
            if tracked.host_current && !tracked.guest_current {
                passes.push(tracked.detile(base, self.copy_offset)?);
                materialized.push(base);
            }
        }
        let levels = match existing.filter(|tracked| tracked.host_current) {
            Some(tracked) => first_level..tracked.first_held_level,
            None => first_level..guest.desc().mip_levels.max(1),
        };
        let to_host = if levels.is_empty() {
            None
        } else {
            let dispatches = copy_dispatches(guest, guest_base, host, guest_base + self.copy_offset, levels.clone())?;
            Some(CopyPass { direction: CopyDirection::ToHost, guest_base, levels, dispatches })
        };
        Ok(ViewPlan {
            guest_base,
            replace,
            materialized,
            to_guest: passes,
            to_host,
        })
    }

    /// Carries out a [`ViewPlan`]: dirty aliases are current in guest memory, the view is
    /// tracked with its layouts, and a retiled host copy makes every other view of its storage
    /// stale.
    fn apply_view(&mut self, plan: ViewPlan) -> Vec<CopyPass> {
        let ViewPlan {
            guest_base,
            replace,
            materialized,
            to_guest: mut passes,
            to_host,
        } = plan;
        if !materialized.is_empty() || replace.is_some() || to_host.is_some() {
            self.epoch += 1;
        }
        for base in materialized {
            if let Some(alias) = self.surfaces.get_mut(&base) {
                alias.guest_current = true;
            }
        }
        if let Some((guest, host)) = replace {
            self.copies_generation += 1;
            // Views that are windows on the new surface use its copy from now on. Their dirty
            // contents are in guest memory by now, through `to_guest`.
            self.remove_windows(guest_base, &guest, &host);
            self.largest = self.largest.max(guest.size_bytes()).max(host.size_bytes());
            self.surfaces.insert(
                guest_base,
                Tracked {
                    guest,
                    host,
                    host_current: false,
                    guest_current: true,
                    first_held_level: 0,
                    written_check: self.write_check,
                },
            );
        }
        let Some(to_host) = to_host else {
            return passes;
        };
        let host_bytes = self.surfaces[&guest_base].host.size_bytes();
        let near = self.near(guest_base, host_bytes);
        for (&base, other) in self.surfaces.range_mut(near) {
            if base != guest_base && overlaps(base, &other.host, guest_base, host_bytes) {
                other.host_current = false;
            }
        }
        let tracked = self.surfaces.get_mut(&guest_base).unwrap();
        // Levels added to a current copy come from guest memory, which has them; the levels it
        // holds already stay as they are, on the side that is newest.
        if !tracked.host_current {
            tracked.guest_current = true;
        }
        tracked.host_current = true;
        tracked.first_held_level = to_host.levels.start;
        passes.push(to_host);
        passes
    }

    /// Forgets the surfaces that are windows on the surface `guest` and `host` describe at
    /// `guest_base`, other than one tracked at `guest_base` (which the caller replaces).
    fn remove_windows(&mut self, guest_base: u64, guest: &SurfaceLayout, host: &SurfaceLayout) {
        let stride = guest.slice_stride();
        if stride == 0 {
            return;
        }
        let end = guest_base + guest.size_bytes();
        self.surfaces.retain(|&base, tracked| {
            if base <= guest_base || base >= end || (base - guest_base) % stride != 0 {
                return true;
            }
            let first_layer = ((base - guest_base) / stride) as u32;
            !(guest.is_window(&tracked.guest, first_layer) && host.is_window(&tracked.host, first_layer))
        });
    }

    /// Texture view transitions, for a view that reaches `first_level` and coarser levels. View
    /// extent, element size and layout are part of the identity; a texture at an already tracked
    /// base must not silently reuse an incompatible target.
    pub fn bind_texture_passes(
        &mut self,
        guest_base: u64,
        desc: &SurfaceDesc,
        usage: SurfaceUse,
        first_level: u32,
    ) -> Result<Vec<CopyPass>> {
        let plan = self.plan_view(guest_base, desc, usage, first_level)?;
        Ok(self.apply_view(plan))
    }

    /// Whether `address` lies in the guest memory of a tracked surface.
    pub fn holds_address(&self, address: u64) -> bool {
        self.surfaces.range(self.near(address, 1)).any(|(&base, tracked)| overlaps(base, &tracked.guest, address, 1))
    }

    /// Whether the surface tracked at `guest_base` is the one `desc` describes.
    pub fn tracks(&self, guest_base: u64, desc: &SurfaceDesc) -> bool {
        self.surfaces.get(&guest_base).is_some_and(|tracked| tracked.guest.desc() == desc)
    }

    /// The pass that fills the host copy of the surface at `guest_base` with `texel`, when the
    /// surface is tracked.
    pub fn clear_pass(&self, guest_base: u64, texel: &[u8]) -> Result<Option<CopyPass>> {
        let Some(tracked) = self.surfaces.get(&guest_base) else { return Ok(None) };
        let dispatches = clear_dispatches(&tracked.host, guest_base + self.copy_offset, texel)?;
        let levels = 0..tracked.level_count();
        Ok(Some(CopyPass { direction: CopyDirection::ToHost, guest_base, levels, dispatches }))
    }

    /// Host work wrote the host copy of the surface at `guest_base`: guest memory is stale until
    /// a detile.
    pub fn host_written(&mut self, guest_base: u64) {
        // A window writes its owner's copy.
        let guest_base = if self.surfaces.contains_key(&guest_base) {
            guest_base
        } else {
            match self.surfaces.range(..guest_base).next_back() {
                Some((&base, tracked)) if guest_base < base + tracked.guest.size_bytes() => base,
                _ => return,
            }
        };
        let Some(written) = self.surfaces.get(&guest_base) else {
            return;
        };
        let (guest_bytes, host_bytes) = (written.guest.size_bytes(), written.host.size_bytes());
        let near = self.near(guest_base, guest_bytes.max(host_bytes));
        let mut changed = false;
        for (&base, tracked) in self.surfaces.range_mut(near) {
            let before = (tracked.host_current, tracked.guest_current);
            if base == guest_base {
                tracked.host_current = true;
                tracked.guest_current = false;
            } else if overlaps(base, &tracked.guest, guest_base, guest_bytes)
                || overlaps(base, &tracked.host, guest_base, host_bytes)
            {
                // The newly written view owns the latest bytes. Its bind transition has
                // materialized any previously dirty alias before the GPU overwrote it.
                tracked.host_current = false;
                tracked.guest_current = true;
            }
            changed |= before != (tracked.host_current, tracked.guest_current);
        }
        if changed {
            self.epoch += 1;
        }
    }

    /// Guest memory in `[address, address + bytes)` changed outside the host copies (CPU, DMA or
    /// guest-layout stores): host copies of overlapping surfaces are stale.
    pub fn guest_written(&mut self, address: u64, bytes: u64) {
        let mut changed = false;
        let check = self.write_check;
        for tracked in self.overlapping(address, bytes) {
            changed |= tracked.host_current || !tracked.guest_current;
            tracked.host_current = false;
            tracked.guest_current = true;
            tracked.written_check = check;
        }
        if changed {
            self.epoch += 1;
        }
    }

    /// Guest memory in `[address, address + bytes)` is about to be read outside the host copies:
    /// the detiles of overlapping surfaces whose guest memory is stale.
    pub fn guest_read(&mut self, address: u64, bytes: u64) -> Result<Vec<CopyPass>> {
        let offset = self.copy_offset;
        let mut passes = Vec::new();
        let near = self.near(address, bytes);
        for (&guest_base, tracked) in self.surfaces.range_mut(near) {
            if !tracked.host_current
                || tracked.guest_current
                || !overlaps(guest_base, &tracked.guest, address, bytes)
            {
                continue;
            }
            passes.push(tracked.detile(guest_base, offset)?);
            tracked.guest_current = true;
        }
        if !passes.is_empty() {
            self.epoch += 1;
        }
        Ok(passes)
    }

    fn overlapping(&mut self, address: u64, bytes: u64) -> impl Iterator<Item = &mut Tracked> {
        let near = self.near(address, bytes);
        self.surfaces
            .range_mut(near)
            .filter(move |(&base, tracked)| overlaps(base, &tracked.guest, address, bytes) && tracked.holds(base, address, bytes))
            .map(|(_, tracked)| tracked)
    }
}


fn overlaps(base: u64, layout: &SurfaceLayout, address: u64, bytes: u64) -> bool {
    bytes != 0
        && address < base.saturating_add(layout.size_bytes())
        && base < address.saturating_add(bytes)
}


/// Reads a bit field `[lo, lo + width)` of `value`.
const fn field(value: u32, lo: u32, width: u32) -> u32 {
    (value >> lo) & ((1 << width) - 1)
}

const fn cx(offset: u32) -> u32 {
    (offset - 0x2_8000) / 4
}

fn base_address(low: u32, high: u32) -> u64 {
    (u64::from(low) << 8) | (u64::from(high & 0xff) << 40)
}

/// DCC metadata of a colour target: one key byte per 256 bytes of surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DccMetadata {
    pub base: u64,
    pub keys: usize,
    /// CB_COLORn_CLEAR_WORD0/1, the texel of the clear-register key.
    pub clear_words: [u32; 2],
}

/// A colour target a draw renders to.
#[derive(Clone, Debug)]
pub struct ColorBinding {
    pub format: TexelFormat,
    pub layout: SurfaceLayout,
    pub base: u64,
    pub write_mask: u32,
    pub dcc: Option<DccMetadata>,
}

/// The depth plane a draw uses.
#[derive(Clone, Debug)]
pub struct DepthTarget {
    pub layout: SurfaceLayout,
    pub write_base: u64,
    /// HTILE metadata address when the plane has it.
    pub htile: Option<u64>,
}

/// The colour targets and depth buffer of a draw.
#[derive(Clone, Debug, Default)]
pub struct RenderTargets {
    pub colors: Vec<ColorBinding>,
    pub depth: Option<DepthTarget>,
    pub depth_write: bool,
}

impl RenderTargets {
    /// The guest registers [`Self::of`] decides by, as the title left them: for a draw that turns
    /// out to render to nothing.
    pub fn registers(guest: &GuestRegisters) -> String {
        use std::fmt::Write as _;
        let get = |dword: u32| guest.get(Context, dword);
        let mut text = format!(
            "CB_COLOR_CONTROL {:x?} CB_TARGET_MASK {:x?} CB_SHADER_MASK {:x?} DB_DEPTH_CONTROL {:x?} DB_RENDER_CONTROL {:x?} DB_Z_INFO {:x?}",
            get(cx(gfx10::CB_COLOR_CONTROL::OFFSET)),
            get(cx(gfx10::CB_TARGET_MASK::OFFSET)),
            get(cx(gfx10::CB_SHADER_MASK::OFFSET)),
            get(cx(gfx10::DB_DEPTH_CONTROL::OFFSET)),
            get(cx(gfx10::DB_RENDER_CONTROL::OFFSET)),
            get(cx(gfx10::DB_Z_INFO::OFFSET)),
        );
        for slot in 0..8u32 {
            let block = slot * 15;
            let info = get(cx(gfx10::CB_COLOR0_INFO::OFFSET) + block);
            let base = get(cx(gfx10::CB_COLOR0_BASE::OFFSET) + block);
            if info.is_some() || base.is_some() {
                let _ = write!(
                    text,
                    "; CB_COLOR{slot} INFO {info:x?} BASE {base:x?} BASE_EXT {:x?} ATTRIB {:x?} ATTRIB2 {:x?} ATTRIB3 {:x?}",
                    get(cx(gfx10::CB_COLOR0_BASE_EXT::OFFSET) + slot),
                    get(cx(gfx10::CB_COLOR0_ATTRIB::OFFSET) + block),
                    get(cx(gfx10::CB_COLOR0_ATTRIB2::OFFSET) + slot),
                    get(cx(gfx10::CB_COLOR0_ATTRIB3::OFFSET) + slot),
                );
            }
        }
        text
    }

    /// The targets `guest` names, for what the guest's state can bind on the host.
    pub fn of(guest: &GuestRegisters) -> Result<Self> {
        let context = |dword: u32| guest.get_or_zero(Context, dword);
        let mut targets = Self::default();
        let control = context(cx(gfx10::CB_COLOR_CONTROL::OFFSET));
        let target_mask = context(cx(gfx10::CB_TARGET_MASK::OFFSET));
        let shader_mask = context(cx(gfx10::CB_SHADER_MASK::OFFSET));
        if field(control, 4, 3) == 1 {
            for slot in 0..8u32 {
                let block = slot * 15;
                let info = context(cx(gfx10::CB_COLOR0_INFO::OFFSET) + block);
                let write_mask = (target_mask >> (4 * slot)) & (shader_mask >> (4 * slot)) & 0xf;
                if write_mask == 0 || field(info, 2, 5) == 0 {
                    continue;
                }
                let attrib = context(cx(gfx10::CB_COLOR0_ATTRIB::OFFSET) + block);
                if field(attrib, 12, 3) != 0 {
                    return Err(Error::unsupported("multisampled color targets are not implemented"));
                }
                let attrib3 = context(cx(gfx10::CB_COLOR0_ATTRIB3::OFFSET) + slot);
                let swizzle_mode = field(attrib3, 14, 5);
                let guest_base = base_address(
                    context(cx(gfx10::CB_COLOR0_BASE::OFFSET) + block),
                    context(cx(gfx10::CB_COLOR0_BASE_EXT::OFFSET) + slot),
                );
                let (base, pipe_bank_xor) = split_pipe_bank_xor(swizzle_mode, guest_base);
                if base == 0 {
                    continue;
                }
                let format = TexelFormat::from_color_info(info)?;
                let attrib2 = context(cx(gfx10::CB_COLOR0_ATTRIB2::OFFSET) + slot);
                let volume = field(attrib3, 24, 2) == 2;
                let layers = field(attrib3, 0, 13) + 1;
                let (block_w, block_h) = format.block_extent();
                let layout = SurfaceLayout::new(&SurfaceDesc {
                    swizzle_mode,
                    width: (field(attrib2, 14, 14) + 1).div_ceil(block_w),
                    height: (field(attrib2, 0, 14) + 1).div_ceil(block_h),
                    depth: if volume { layers } else { 1 },
                    layers: if volume { 1 } else { layers },
                    mip_levels: field(attrib2, 28, 4) + 1,
                    bytes_per_element: format.bytes_per_element(),
                    pitch_elements: 0,
                    volume,
                    pipe_bank_xor,
                })?;
                let dcc_base = base_address(
                    context(cx(gfx10::CB_COLOR0_DCC_BASE::OFFSET) + block),
                    context(cx(gfx10::CB_COLOR0_DCC_BASE_EXT::OFFSET) + slot),
                );
                let dcc = (info & (1 << 28) != 0 && dcc_base != 0).then(|| DccMetadata {
                    base: dcc_base,
                    keys: (layout.size_bytes() / 256) as usize,
                    clear_words: [
                        context(cx(gfx10::CB_COLOR0_CLEAR_WORD0::OFFSET) + block),
                        context(cx(gfx10::CB_COLOR0_CLEAR_WORD0::OFFSET) + 1 + block),
                    ],
                });
                targets.colors.push(ColorBinding { format, layout, base, write_mask, dcc });
            }
        }
        let depth_control = context(cx(gfx10::DB_DEPTH_CONTROL::OFFSET));
        let render = context(cx(gfx10::DB_RENDER_CONTROL::OFFSET));
        let z_info = context(cx(gfx10::DB_Z_INFO::OFFSET));
        let uses_depth = depth_control & 0xe != 0 || render & 1 != 0;
        let format = field(z_info, 0, 2);
        if uses_depth && format != 0 {
            if field(z_info, 2, 2) != 0 {
                return Err(Error::unsupported("multisampled depth buffers are not implemented"));
            }
            let size = context(cx(gfx10::DB_DEPTH_SIZE_XY::OFFSET));
            let view = context(cx(gfx10::DB_DEPTH_VIEW::OFFSET));
            let swizzle_mode = field(z_info, 4, 5);
            let bytes_per_element = if format == 1 { 2 } else { 4 };
            let guest_write = base_address(context(cx(gfx10::DB_Z_WRITE_BASE::OFFSET)), context(cx(gfx10::DB_Z_WRITE_BASE_HI::OFFSET)));
            let (write_base, pipe_bank_xor) = split_pipe_bank_xor(swizzle_mode, guest_write);
            let layout = SurfaceLayout::new(&SurfaceDesc {
                swizzle_mode,
                width: field(size, 0, 14) + 1,
                height: field(size, 16, 14) + 1,
                depth: 1,
                layers: (field(view, 13, 11) | (field(view, 30, 2) << 11)) + 1,
                mip_levels: field(z_info, 16, 4) + 1,
                bytes_per_element,
                pitch_elements: 0,
                volume: false,
                pipe_bank_xor,
            })?;
            let htile = base_address(context(cx(gfx10::DB_HTILE_DATA_BASE::OFFSET)), context(cx(gfx10::DB_HTILE_DATA_BASE_HI::OFFSET)));
            targets.depth = Some(DepthTarget { layout, write_base, htile: (z_info & (1 << 29) != 0 && htile != 0).then_some(htile) });
            targets.depth_write = depth_control & 4 != 0;
        }
        Ok(targets)
    }
}

/// The T# fields the surface path reads.
#[derive(Clone, Copy, Debug)]
pub struct ImageView {
    pub base: u64,
    pub swizzle_mode: u32,
    pub pipe_bank_xor: u32,
    pub ty: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub last_array: u32,
    pub base_level: u32,
    pub last_level: u32,
    pub max_mip: u32,
    pub min_lod: u32,
    pub pitch: u32,
    pub meta: Option<u64>,
}

impl ImageView {
    /// A descriptor with a null base or TYPE, or no view the surface path can name.
    pub fn is_null(words: &[u32; 8]) -> bool {
        (words[0] == 0 && words[1] & 0xff == 0) || words[3] >> 28 <= 1
    }

    pub fn decode(words: &[u32; 8]) -> Result<Self> {
        let ty = field(words[3], 28, 4);
        let swizzle_mode = field(words[3], 20, 5);
        let address = (u64::from(words[0]) | (u64::from(field(words[1], 0, 8)) << 32)) << 8;
        let (base, pipe_bank_xor) = split_pipe_bank_xor(swizzle_mode, address);
        if base == 0 {
            return Err(Error::invalid("image descriptor has a null base address"));
        }
        let format = field(words[1], 20, 9);
        let depth_field = field(words[4], 0, 13);
        let base_array = field(words[4], 16, 13);
        let max_mip = field(words[5], 4, 4);
        let base_level = field(words[3], 12, 4);
        let last_level = field(words[3], 16, 4).min(max_mip.max(field(words[3], 12, 4)));
        let (mut depth, mut last_array, mut pitch) = (1, 0, 0);
        match ty {
            10 => depth = depth_field + 1,
            11 | 12 | 13 | 15 => last_array = if ty == 11 { depth_field.max(base_array + 5) } else { depth_field },
            8 | 9 | 14 if depth_field != 0 && swizzle_mode == SW_LINEAR => {
                let element = TexelFormat::from_gfx10(format)?.bytes_per_element();
                let unit = 256 / element;
                let width = (field(words[1], 30, 2) | (field(words[2], 0, 12) << 2)) + 1;
                let wanted = (depth_field + 1).max(width).next_multiple_of(unit);
                pitch = if wanted == width.next_multiple_of(unit) { 0 } else { wanted };
            }
            _ => {}
        }
        Ok(Self {
            base,
            swizzle_mode,
            pipe_bank_xor,
            ty,
            format,
            width: (field(words[1], 30, 2) | (field(words[2], 0, 12) << 2)) + 1,
            height: field(words[2], 14, 14) + 1,
            depth,
            last_array,
            base_level,
            last_level,
            max_mip,
            min_lod: field(words[1], 8, 12),
            pitch,
            meta: (field(words[6], 21, 1) != 0).then(|| (u64::from(field(words[6], 24, 8)) | (u64::from(words[7]) << 8)) << 8),
        })
    }

    pub fn surface_use(&self) -> SurfaceUse {
        match self.ty {
            8 | 12 => SurfaceUse::OneD,
            10 => SurfaceUse::Volume,
            14 | 15 => SurfaceUse::Msaa,
            _ => SurfaceUse::Thin2d,
        }
    }

    pub fn surface_desc(&self) -> Result<SurfaceDesc> {
        let format = TexelFormat::from_gfx10(self.format)?;
        let (block_width, block_height) = format.block_extent();
        Ok(SurfaceDesc {
            swizzle_mode: self.swizzle_mode,
            width: self.width.div_ceil(block_width),
            height: self.height.div_ceil(block_height),
            depth: self.depth,
            layers: if matches!(self.ty, 11 | 12 | 13 | 15) { self.last_array + 1 } else { 1 },
            mip_levels: self.max_mip.max(self.last_level) + 1,
            bytes_per_element: format.bytes_per_element(),
            pitch_elements: self.pitch,
            volume: self.ty == 10,
            pipe_bank_xor: self.pipe_bank_xor,
        })
    }

    /// The first level this view can reach: BASE_LEVEL, or the level of the MIN_LOD clamp when
    /// that is higher. Titles that stream mip chains in clamp MIN_LOD to the resident levels.
    pub fn first_reachable_level(&self) -> u32 {
        self.base_level.max(self.min_lod >> 8).min(self.last_level)
    }
}

/// The surface side of one queue's submissions: the tracker, and the PM4 that keeps copies current.
pub struct SurfaceTable {
    pub tracker: SurfaceTracker,
    kernel: Option<Arc<super::program::Program>>,
    /// Parameter memory of the copy passes: address and bytes. A ring: a pass's blocks are written
    /// on its queue's timeline and read by its dispatches right after, and the queues' passes run
    /// at the same time, so they must not share blocks. The ring holds far more dispatches than
    /// are ever on the GPU at once.
    params: (u64, u64),
    next: u64,
}

impl SurfaceTable {
    pub fn new(params: (u64, u64)) -> Self {
        Self { tracker: SurfaceTracker::new(crate::backend::va::HOST_COPY_OFFSET), kernel: None, params, next: 0 }
    }

    fn kernel(&mut self, runtime: &super::runtime::TargetRuntime) -> Result<Arc<super::program::Program>> {
        if self.kernel.is_none() {
            let kernel = runtime.programs.resolve_kernel(&element_copy::WORDS, element_copy::PGM_RSRC1, element_copy::PGM_RSRC2, 64).map_err(Error::unsupported)?;
            if kernel.rsrc2 & element_copy::REQUIRED_RSRC2 != element_copy::PGM_RSRC2 & element_copy::REQUIRED_RSRC2 {
                return Err(Error::invalid("the element-copy kernel translated without its user SGPRs and group ids"));
            }
            self.kernel = Some(kernel);
        }
        Ok(self.kernel.clone().expect("set above"))
    }

    /// Appends `pass` to `stream` for a queue of the given kind, after a full idle wait.
    pub fn emit_pass(&mut self, runtime: &super::runtime::TargetRuntime, stream: &mut PacketStream, compute: bool, pass: &CopyPass) -> Result<()> {
        const PARAMS_WAVE: u8 = 64;
        let bytes = PARAM_STRIDE * pass.dispatches.len() as u64;
        if bytes > self.params.1 {
            return Err(Error::unsupported(format!("the copy of surface 0x{:x} does not fit the parameter memory", pass.guest_base)));
        }
        let kernel = self.kernel(runtime)?;
        if self.next + bytes > self.params.1 {
            self.next = 0;
        }
        let params = self.params.0 + self.next;
        self.next += bytes;
        emit_copy_pass(stream, compute, pass, &kernel, PARAMS_WAVE, params)
    }

    /// Host copies a submission needs for the guest surface at `base` to be resident.
    pub fn views(&self) -> Vec<(u64, u64, u64)> {
        self.tracker.copies()
    }
}

/// Waits until the work before is idle on the queue, with what it rendered visible to the copy
/// kernel: a PWS CACHE_FLUSH_AND_INV_TS release and wait on graphics, CS_PARTIAL_FLUSH on compute.
pub(super) fn wait_idle(stream: &mut PacketStream, compute: bool) -> Result<()> {
    if compute {
        return stream.event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH);
    }
    stream.release_mem_pws(event::CACHE_FLUSH_AND_INV_TS_EVENT, gcr::RELEASE_GLM_WB | gcr::RELEASE_GLM_INV | gcr::RELEASE_SEQ_FORWARD)?;
    stream.acquire_mem_pws_wait()
}

/// PM4 of a copy pass with the element-copy kernel: after earlier work is idle, each dispatch's
/// parameter block goes to `params` by WRITE_DATA, the shader caches are invalidated and the
/// kernel runs; work after the pass waits for it with the same caches invalidated.
fn emit_copy_pass(stream: &mut PacketStream, compute: bool, pass: &CopyPass, kernel: &super::program::Program, wave_size: u8, params: u64) -> Result<()> {
    use crate::gpu::packets::SHADER_TYPE_COMPUTE;
    const WRITE_DATA_MEMORY_CONFIRMED: u32 = (5 << 8) | (1 << 20);
    const COMPUTE_SHADER_EN: u32 = 1;
    const CS_W32_EN: u32 = 1 << 15;
    const INVALIDATE: u32 = gcr::GLK_INV | gcr::GLV_INV | gcr::GL1_INV;
    if params % 256 != 0 {
        return Err(Error::invalid(format!("copy-pass parameters at {params:#x} are not 256-byte aligned")));
    }
    wait_idle(stream, compute)?;
    for (index, dispatch) in pass.dispatches.iter().enumerate() {
        let block = params + PARAM_STRIDE * index as u64;
        let mut body = Vec::with_capacity(3 + COPY_PARAM_DWORDS);
        body.extend([WRITE_DATA_MEMORY_CONFIRMED, block as u32, (block >> 32) as u32]);
        body.extend(dispatch.params);
        stream.raw_packet(crate::amd::defs::pkt3::WRITE_DATA, 0, &body)?;
    }
    stream.acquire_mem_global(INVALIDATE)?;
    stream.set(host::COMPUTE_PGM_LO, (kernel.host_address >> 8) as u32)?;
    stream.set(host::COMPUTE_PGM_HI, (kernel.host_address >> 40) as u32 & 0xff)?;
    stream.set(host::COMPUTE_PGM_RSRC1, kernel.rsrc1)?;
    stream.set(host::COMPUTE_PGM_RSRC2, kernel.rsrc2)?;
    stream.set(host::COMPUTE_PGM_RSRC3, 0)?;
    stream.set(host::COMPUTE_TMPRING_SIZE, 0)?;
    stream.set(host::COMPUTE_RESOURCE_LIMITS, 0)?;
    stream.set_run(host::COMPUTE_START, &[0, 0, 0])?;
    stream.set_run(host::COMPUTE_NUM_THREAD, &[COPY_GROUP_THREADS, 1, 1])?;
    let initiator = COMPUTE_SHADER_EN | if wave_size == 32 { CS_W32_EN } else { 0 };
    for (index, dispatch) in pass.dispatches.iter().enumerate() {
        let block = params + PARAM_STRIDE * index as u64;
        stream.set_run(host::COMPUTE_USER_DATA, &[block as u32, (block >> 32) as u32])?;
        let [x, y, z] = dispatch.groups;
        stream.raw_packet(crate::amd::defs::pkt3::DISPATCH_DIRECT, SHADER_TYPE_COMPUTE, &[x, y, z, initiator])?;
    }
    wait_idle(stream, compute)?;
    stream.acquire_mem_global(INVALIDATE)
}


/// Fills guest memory at `address` with the 32-bit `pattern`, through L2 at both ends as radv's
/// CP DMA does.
pub fn emit_memory_fill(stream: &mut PacketStream, compute: bool, address: u64, bytes: u64, pattern: u32) -> Result<()> {
    use crate::amd::defs::cp_gfx11::meg::DMA_DATA as D;
    /// What one DMA_DATA moves, as radv's `cp_dma_max_byte_count`, word aligned.
    const MAX_BYTES: u64 = 32764;
    let end = address + (bytes & !3);
    let mut cursor = address;
    while cursor < end {
        let count = (end - cursor).min(MAX_BYTES);
        let mut packet = [0u32; D::WORDS as usize];
        D::SRC_SEL.set(&mut packet, D::enums::src_sel::DATA);
        D::DST_SEL.set(&mut packet, D::enums::dst_sel::DST_ADDR_USING_L2);
        if !compute {
            D::CP_SYNC.set(&mut packet, 1);
        }
        packet[2] = pattern;
        packet[4] = cursor as u32;
        packet[5] = (cursor >> 32) as u32;
        D::BYTE_COUNT.set(&mut packet, count as u32);
        stream.raw_packet(D::OPCODE, 0, &packet[1..])?;
        cursor += count;
    }
    Ok(())
}

/// Whether the T# conversion samples this descriptor in place, and its host swizzle mode.
pub fn host_view(view: &ImageView) -> (bool, u32) {
    let in_place = shader::descriptors::samples_in_place_for_format(view.ty as u8, view.swizzle_mode as u8, view.format as u16);
    (in_place, if in_place { view.swizzle_mode } else { host_swizzle_mode(view.swizzle_mode, view.surface_use()) })
}
