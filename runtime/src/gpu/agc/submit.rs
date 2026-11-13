//! Running what recordings wrote (decisions D25, D27, D38): a submission is cut at the guest's
//! waits that nothing in it can end, after writes something else waits for, and where a tail jump
//! continues in other commands; each piece is entered from a host IB by CHAIN, so the GPU executes
//! the title's commands where they are.

use super::chain::Resolved;
use super::side_table::{CutReason, DmaWrite, Event, LabelWrite, Root, SideTable, UserData, WaitCondition, USER_DATA_REGISTERS};
use crate::agc::header;
use crate::amd::defs::{gfx10, pkt3};
use crate::backend::host_pm4::{self, IB_ALIGNMENT_DWORDS};
use crate::ir::FlipRequest;
use crate::residency::Queue;
use crate::scheduler::{EopEvent, SubmitEvent, EOP_FLIP_TOKENS};
use crate::{Error, Result};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// A piece of a submission: the commands at `[start, end)` run after `wait` holds.
#[derive(Clone, Debug)]
pub struct Segment {
    pub start: u64,
    pub end: u64,
    source_start: u64,
    source: Arc<[u32]>,
    /// What the segment waits for before it may run.
    pub wait: Option<WaitCondition>,
    /// Display buffers that must leave the screen first.
    pub flip_waits: Vec<(u32, i32)>,
    /// The user data the segment starts with.
    pub user_data: Option<UserData>,
    /// What its completion reports.
    pub events: Vec<SubmitEvent>,
    /// The label writes it makes.
    pub writes: Vec<LabelWrite>,
    /// The descriptor roots of its draws and dispatches: the surfaces they use are brought up to
    /// date when it is queued, after what it waits for.
    pub roots: Vec<Root>,
    /// Guest memory its DMA writes: (address, bytes).
    pub dma_writes: Vec<DmaWrite>,
    pub guards: Vec<(u64, u64)>,
}

impl Segment {
    /// Submission-owned executable bytes, shared by the cuts of one stretch. In particular this
    /// never rereads a live recording when a parked segment finally becomes ready.
    pub fn commands(&self) -> &[u32] {
        let first = ((self.start - self.source_start) / 4) as usize;
        &self.source[first..first + ((self.end - self.start) / 4) as usize]
    }
}

/// Frozen commands cannot observe GPU patches to their original executable addresses. Check
/// the whole submission before queueing any segment, including writes across a wait boundary.
pub fn validate_executable_writes(segments: &[Segment]) -> Result<()> {
    let mut ranges: Vec<(u64, u64)> = segments.iter().map(|segment| (segment.start, segment.end)).collect();
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (from, to) in ranges {
        if let Some(last) = merged.last_mut().filter(|last| last.1 >= from) { last.1 = last.1.max(to); }
        else { merged.push((from, to)); }
    }
    let overlaps = |address: u64, bytes: u64| {
        let end = address.saturating_add(bytes);
        let index = merged.partition_point(|&(from, _)| from < end);
        bytes != 0 && index != 0 && merged[index - 1].1 > address
    };
    for segment in segments {
        let writes = segment.dma_writes.iter().map(|write| (write.address, write.bytes))
            .chain(segment.writes.iter().map(|write| (write.address, u64::from(write.bytes))));
        if let Some((address, bytes)) = writes.into_iter().find(|&(address, bytes)| overlaps(address, bytes)) {
            return Err(Error::unsupported(format!("GPU write 0x{address:x}+0x{bytes:x} into frozen executable commands needs execution-time relocation")));
        }
    }
    Ok(())
}

/// One label a queue was seen to wait for.
#[derive(Default)]
pub struct AwaitedLabels {
    conditions: BTreeMap<u64, Vec<WaitCondition>>,
}

impl AwaitedLabels {
    /// Remembers the waits of `table`, as soon as its submission arrives: the commands that end
    /// them may be planned before the waiting recording is (D25).
    pub fn learn(&mut self, table: &SideTable) {
        for cut in table.cuts.iter() {
            if let CutReason::Wait(condition) = cut.reason {
                let known = self.conditions.entry(condition.address).or_default();
                if !known.contains(&condition) {
                    known.push(condition);
                }
            }
        }
    }

    fn ended_by(&self, write: &LabelWrite) -> bool {
        self.conditions.get(&write.address).is_some_and(|conditions| conditions.iter().any(|condition| condition.holds(write.value)))
    }
}

/// Where a segment starts, and what it needs first.
struct Boundary {
    at: u64,
    wait: Option<WaitCondition>,
    flip_waits: Vec<(u32, i32)>,
    user_data: UserData,
}

impl Boundary {
    fn merge(&mut self, other: Boundary) {
        self.wait = self.wait.or(other.wait);
        self.flip_waits.extend(other.flip_waits);
    }
}

/// Where a segment may start at or after `at`: not inside a range a COND_EXEC guards, whose skip
/// would run on past the end of the IB (D25); with the user data there.
fn outside_guards(stretch: &Resolved, mut at: u64, mut user_data: UserData) -> (u64, UserData) {
    while let Some(&(_, end)) = stretch.guards.iter().find(|&&(from, end)| from < at && at < end) {
        at = end;
        user_data = stretch.marks.iter().find(|&&(mark, _)| mark == end).map_or(user_data, |&(_, values)| values);
    }
    (at, user_data)
}

/// The pieces of a submission's commands, from its stretches with the state worked out (D38): a
/// stretch's commands end where its tail jump was, and the next stretch starts a segment of its
/// own. A cut right at a stretch's end (after its last command) carries over to the next stretch.
/// Inside a COND_EXEC range nothing is cut (D25): a wait there stays on the GPU, and what must
/// come after a command there (a write others wait for, a flip, a fast clear, a flip wait) comes
/// after the range.
pub fn plan(stretches: &[Resolved], queue: u32, awaited: &AwaitedLabels) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut carried: Option<Boundary> = None;
    for stretch in stretches {
        let (start, end) = (stretch.start, stretch.end);
        let mut boundaries: Vec<Boundary> = Vec::new();
        for (cut, user_data) in stretch.cuts.iter().filter(|(cut, _)| cut.at >= start && cut.at <= end) {
            let (wait, flip) = match cut.reason {
                CutReason::Wait(condition) => (Some(condition), None),
                CutReason::FlipDone { video_out, buffer } => (None, Some((video_out, buffer))),
                CutReason::AfterFlip | CutReason::UserData => (None, None),
            };
            let (at, user_data) = outside_guards(stretch, cut.at, *user_data);
            if at != cut.at && wait.is_some() {
                super::state::report_once(&format!("a wait inside a COND_EXEC range (0x{:x}) stays on the GPU: a segment cannot start there", cut.at), 6);
                continue;
            }
            boundaries.push(Boundary { at, wait, flip_waits: flip.into_iter().collect(), user_data });
        }
        for (write, user_data) in stretch.writes.iter().filter(|(write, _)| write.end < end && awaited.ended_by(write)) {
            let (at, user_data) = outside_guards(stretch, write.end, *user_data);
            boundaries.push(Boundary { at, wait: None, flip_waits: Vec::new(), user_data });
        }
        boundaries.sort_by_key(|boundary| boundary.at);
        let mut starts = vec![Boundary { at: start, wait: None, flip_waits: Vec::new(), user_data: stretch.user_data }];
        if let Some(carried) = carried.take() {
            starts[0].merge(carried);
        }
        for boundary in boundaries {
            let last = starts.last_mut().expect("a stretch has its start");
            if boundary.at == last.at {
                last.merge(boundary);
            } else if boundary.at < end {
                starts.push(boundary);
            } else {
                carried.get_or_insert(Boundary { at: end, wait: None, flip_waits: Vec::new(), user_data: boundary.user_data }).merge(boundary);
            }
        }
        if start == end {
            // Nothing runs here: what it waits for waits on before the next stretch.
            let empty = starts.swap_remove(0);
            carried.get_or_insert(Boundary { at: end, wait: None, flip_waits: Vec::new(), user_data: empty.user_data }).merge(empty);
            continue;
        }
        for (index, boundary) in starts.iter().enumerate() {
            let stop = starts.get(index + 1).map_or(end, |next| next.at);
            let inside = |at: u64| at >= boundary.at && at < stop;
            let writes: Vec<LabelWrite> = stretch.writes.iter().map(|(write, _)| *write).filter(|write| inside(write.at)).collect();
            let events = stretch
                .notices
                .iter()
                .filter(|notice| inside(notice.at))
                .map(|notice| match notice.event {
                    Event::Flip { video_out, buffer, mode, argument } => SubmitEvent::Flip(FlipRequest { handle: video_out, index: buffer, mode, argument }),
                    Event::Interrupt { context, address, value } => {
                        let eop = EopEvent { queue, context, address, value };
                        if EOP_FLIP_TOKENS.contains(&context) {
                            SubmitEvent::EopFlip(eop)
                        } else {
                            SubmitEvent::Eop(eop)
                        }
                    }
                })
                .collect();
            let roots = stretch
                .roots
                .iter()
                .filter(|(at, _)| inside(*at))
                .map(|(_, root)| root.clone())
                .collect();
            let dma_writes = stretch.dma_writes.iter().filter(|write| inside(write.at)).copied().collect();
            segments.push(Segment {
                start: boundary.at,
                end: stop,
                source_start: stretch.start,
                source: stretch.commands.clone(),
                wait: boundary.wait,
                flip_waits: boundary.flip_waits.clone(),
                user_data: Some(boundary.user_data),
                events,
                writes,
                roots,
                dma_writes,
                guards: stretch.guards.iter().copied().filter(|&(from, to)| from >= boundary.at && to <= stop).collect(),
            });
        }
    }
    segments
}

/// A label write of a submitted segment.
pub struct LabelWriter {
    pub address: u64,
    pub value: u64,
    pub queue: Queue,
    pub serial: u64,
}

/// Which label writes are on the GPU, to find what a wait needs.
#[derive(Default)]
pub struct WaitLedger {
    writers: VecDeque<LabelWriter>,
}

/// What a segment's wait needs.
pub enum Settled {
    /// It holds in memory, or nothing else needs waiting for.
    Ready,
    /// The writer is on another queue: run the segment after `serial` of `queue`.
    After(Queue, u64),
    /// Nothing submitted yet can end it.
    Parked,
}

impl WaitLedger {
    /// The segment's waits resolved against memory (`read` gives the bytes of a label) and the
    /// writes on the GPU; `completed` tells how far each queue has run.
    pub fn settle(&mut self, wait: &WaitCondition, queue: Queue, mut read: impl FnMut(u64, usize) -> Option<u64>, mut completed: impl FnMut(Queue) -> u64) -> Settled {
        self.writers.retain(|writer| completed(writer.queue) < writer.serial);
        if read(wait.address, usize::from(wait.bytes)).is_some_and(|value| wait.holds(value)) {
            return Settled::Ready;
        }
        match self.writers.iter().find(|writer| writer.address == wait.address && wait.holds(writer.value)) {
            Some(writer) if writer.queue == queue => Settled::Ready,
            Some(writer) => Settled::After(writer.queue, writer.serial),
            None => Settled::Parked,
        }
    }

    pub fn note(&mut self, segment: &Segment, queue: Queue, serial: u64) {
        for write in &segment.writes {
            self.writers.push_back(LabelWriter { address: write.address, value: write.value, queue, serial });
        }
    }
}

/// `restore` as SET_SH_REG packets (graphics stages and compute, or compute alone).
fn user_data_packets(values: &UserData, compute: bool, out: &mut Vec<u32>) -> Result<()> {
    const STAGES: [(u32, usize, usize); 4] = [
        (gfx10::SPI_SHADER_USER_DATA_PS_0::OFFSET, 0, 32),
        (gfx10::SPI_SHADER_USER_DATA_GS_0::OFFSET, 32, 32),
        (gfx10::SPI_SHADER_USER_DATA_HS_0::OFFSET, 64, 32),
        (gfx10::COMPUTE_USER_DATA_0::OFFSET, 96, 16),
    ];
    debug_assert_eq!(STAGES[3].1 + STAGES[3].2, USER_DATA_REGISTERS);
    for (offset, first, count) in STAGES {
        if compute && offset != gfx10::COMPUTE_USER_DATA_0::OFFSET {
            continue;
        }
        out.push(header(pkt3::SET_SH_REG, count as u32 + 2, 0)?);
        out.push((offset - 0xb000) / 4);
        out.extend_from_slice(&values[first..first + count]);
    }
    Ok(())
}

/// A diagnostic IB and the exact source packets copied into it.
pub struct TracedIb {
    pub image: Vec<u32>,
    pub source: Vec<u32>,
    pub ranges: Vec<super::trace::SourceRange>,
    pub followed_chain: bool,
}

#[derive(Default)]
struct ResourceBoundary {
    after: Vec<u32>,
    guards: BTreeMap<u64, Vec<u32>>,
    before: Vec<u32>,
}

/// GPU resource operations at their original command positions. Ownership sets may be shared;
/// these operations remain ordered and are copied into fence-owned private IB1 storage.
#[derive(Default)]
pub struct ResourceCommands {
    boundaries: BTreeMap<u64, ResourceBoundary>,
}

impl ResourceCommands {
    pub fn before(&mut self, at: u64, words: Vec<u32>) {
        if !words.is_empty() { self.boundaries.entry(at).or_default().before.extend(words); }
    }

    pub fn after(&mut self, at: u64, words: Vec<u32>) {
        if !words.is_empty() { self.boundaries.entry(at).or_default().after.extend(words); }
    }

    pub fn is_empty(&self) -> bool { self.boundaries.is_empty() }

    pub fn end_guard(&mut self, from: u64, to: u64, words: Vec<u32>) {
        self.boundaries.entry(to).or_default().guards.insert(from, words);
    }

    fn weave(&self, source: &[u32], start: u64, trace: Option<(u64, u64)>) -> Result<Vec<u32>> {
        let extra = self.boundaries.values().map(|boundary| boundary.after.len() + boundary.before.len() + boundary.guards.values().map(Vec::len).sum::<usize>()).sum::<usize>();
        let mut image = Vec::with_capacity(source.len() + extra);
        let mut ends = vec![None; source.len() + 1];
        let mut guards = Vec::new();
        let mut guard_ends = BTreeMap::new();
        let mut visited = 0;
        let mut at = 0;
        loop {
            if let Some(boundary) = self.boundaries.get(&(start + at as u64 * 4)) {
                image.extend_from_slice(&boundary.after);
                for (&guard, words) in boundary.guards.iter().rev() {
                    image.extend_from_slice(words);
                    guard_ends.insert(guard, image.len());
                }
                // An old guard includes its last work's postlude, but not the next prelude.
                ends[at] = Some(image.len());
                image.extend_from_slice(&boundary.before);
                visited += 1;
            } else {
                ends[at] = Some(image.len());
            }
            if at == source.len() { break; }
            let packet = source[at];
            let dwords = super::packet_dwords(packet).filter(|&dwords| at + dwords <= source.len())
                .ok_or_else(|| Error::invalid(format!("ordered commands have no complete packet at 0x{:x}", start + at as u64 * 4)))?;
            let opcode = packet >> 8 & 0xff;
            if opcode == pkt3::PRED_EXEC || (opcode == pkt3::INDIRECT_BUFFER && (dwords != 4 || source[at + 3] & (1 << 20) != 0)) {
                return Err(Error::unsupported(format!("ordered resources need resolved control flow at 0x{:x} (opcode 0x{opcode:x})", start + at as u64 * 4)));
            }
            if let Some((address, sequence)) = trace {
                if opcode != pkt3::NOP {
                    image.extend([header(pkt3::WRITE_DATA, 6, 0)?, (5 << 8) | (1 << 20), address as u32, (address >> 32) as u32, at as u32, sequence as u32]);
                }
            }
            let packet_at = image.len();
            image.extend_from_slice(&source[at..at + dwords]);
            if opcode == pkt3::COND_EXEC {
                if dwords != 5 { return Err(Error::invalid("a COND_EXEC has an invalid footprint")); }
                let end = at + dwords + (source[at + 4] & 0x3fff) as usize;
                if end > source.len() { return Err(Error::invalid("a COND_EXEC leaves its submitted segment")); }
                guards.push((start + at as u64 * 4, packet_at + 4, image.len(), end));
            }
            at += dwords;
        }
        if visited != self.boundaries.len() { return Err(Error::invalid("a resource operation is not at a submitted packet boundary")); }
        for (original, count_at, from, end) in guards {
            let end = guard_ends.get(&original).copied().or(ends[end]).ok_or_else(|| Error::invalid("a COND_EXEC ends inside a packet"))?;
            let count = end - from;
            if count > 0x3fff { return Err(Error::unsupported(format!("ordered COND_EXEC needs {count} dwords; branch lowering is required"))); }
            image[count_at] = (image[count_at] & !0x3fff) | count as u32;
        }
        Ok(image)
    }

    pub fn image(&self, prefix: &[u32], preamble: &[u32], restore: Option<&UserData>, compute: bool, segment: &Segment, trace: Option<(u64, u64)>) -> Result<TracedIb> {
        let overlaps = |address: u64, bytes: u64| address < segment.end && address.saturating_add(bytes) > segment.start;
        if segment.dma_writes.iter().any(|write| overlaps(write.address, write.bytes)) || segment.writes.iter().any(|write| overlaps(write.address, u64::from(write.bytes))) {
            return Err(Error::unsupported("GPU writes into private executable commands need execution-time relocation"));
        }
        let source = segment.commands().to_vec();
        let mut image = Vec::new();
        image.extend_from_slice(preamble);
        image.extend_from_slice(prefix);
        if let Some(values) = restore { user_data_packets(values, compute, &mut image)?; }
        image.extend(self.weave(&source, segment.start, trace)?);
        let gap = (IB_ALIGNMENT_DWORDS - image.len() % IB_ALIGNMENT_DWORDS) % IB_ALIGNMENT_DWORDS;
        image.extend(host_pm4::nop_fill(gap));
        let ranges = vec![super::trace::SourceRange { first: 0, address: segment.start, dwords: source.len() }];
        Ok(TracedIb { image, source, ranges, followed_chain: false })
    }
}

pub fn restore_compute_user(values: &[u32; 16], words: &mut Vec<u32>) -> Result<()> {
    words.push(header(pkt3::SET_SH_REG, 18, 0)?);
    words.push((gfx10::COMPUTE_USER_DATA_0::OFFSET - 0xb000) / 4);
    words.extend_from_slice(values);
    Ok(())
}

pub fn predicate_dispatches(words: &mut [u32]) -> Result<()> {
    let mut at = 0;
    while at < words.len() {
        let count = super::packet_dwords(words[at]).filter(|&count| at + count <= words.len()).ok_or_else(|| Error::invalid("invalid resource packet"))?;
        if words[at] >> 8 & 0xff == pkt3::DISPATCH_DIRECT { words[at] |= 1; }
        at += count;
    }
    Ok(())
}

/// As [`host_ib`], but the segment's commands are copied into the IB, each behind a write of its
/// position in `[trace, trace + 8)`: radv's `RADV_DEBUG=hang` trace (`APS5_AGC_TRACE`). A COND_EXEC
/// range keeps its length, so nothing is added inside one.
pub fn traced_ib(copies: &[u32], preamble: &[u32], restore: Option<&UserData>, compute: bool, segment: &Segment, trace: u64, sequence: u64, tables: &[Arc<SideTable>]) -> Result<TracedIb> {
    const WRITE_DATA_MEMORY_CONFIRMED: u32 = (5 << 8) | (1 << 20);
    let words = segment.commands().to_vec();
    let mut image = Vec::with_capacity(copies.len() + preamble.len() + 160 + words.len() * 3);
    image.extend_from_slice(preamble);
    image.extend_from_slice(copies);
    if let Some(values) = restore {
        user_data_packets(values, compute, &mut image)?;
    }
    let (mut at, mut guarded_to) = (0, 0);
    while at < words.len() {
        let packet = words[at];
        let dwords = super::packet_dwords(packet).filter(|&dwords| at + dwords <= words.len()).ok_or_else(|| {
            let from = at.saturating_sub(12);
            Error::invalid(format!(
                "no packet at dword {at} of submitted segment 0x{:x}..0x{:x} ({} dwords, header 0x{packet:08x}); words from dword {from}: {:08x?}",
                segment.start, segment.end, words.len(), &words[from..]
            ))
        })?;
        let opcode = packet >> 8 & 0xff;
        if opcode == pkt3::COND_EXEC && dwords == 5 {
            guarded_to = guarded_to.max(at + dwords + (words[at + 4] & 0x3fff) as usize);
        }
        if opcode == pkt3::WAIT_REG_MEM && at != 0 {
            let address = segment.start + at as u64 * 4;
            let cuts = || tables.iter().flat_map(|table| table.cuts.iter());
            let cut = cuts().find(|cut| cut.at == address);
            eprintln!(
                "[trace] a wait inside segment {sequence} (0x{:x}..0x{:x}) at dword {at}, 0x{address:x}: words {:08x?}; the recordings have {} cuts, {} of them inside the segment; cut here: {:?}",
                segment.start, segment.end, &words[at..at + dwords], cuts().count(),
                cuts().filter(|cut| cut.at >= segment.start && cut.at < segment.end).count(),
                cut.map(|cut| cut.reason)
            );
        }
        if at >= guarded_to && opcode != pkt3::NOP {
            image.extend([header(pkt3::WRITE_DATA, 6, 0)?, WRITE_DATA_MEMORY_CONFIRMED, trace as u32, (trace >> 32) as u32, at as u32, sequence as u32]);
        }
        image.extend_from_slice(&words[at..at + dwords]);
        at += dwords;
    }
    let gap = (IB_ALIGNMENT_DWORDS - image.len() % IB_ALIGNMENT_DWORDS) % IB_ALIGNMENT_DWORDS;
    image.extend(host_pm4::nop_fill(gap));
    let ranges = vec![super::trace::SourceRange { first: 0, address: segment.start, dwords: words.len() }];
    Ok(TracedIb { image, source: words, ranges, followed_chain: false })
}

/// Diagnostic-only private snapshots of a segment and its reachable IBs. Counted control flow
/// retains its original word counts; child address/size fields target the instrumented copies.
#[allow(clippy::too_many_arguments)]
pub fn traced_tree_ib(
    copies: &[u32], preamble: &[u32], restore: Option<&UserData>, compute: bool, segment: &Segment,
    trace: u64, sequence: u64, storage: &super::arena::Bump,
    read: impl FnMut(u64, usize) -> Result<Vec<u32>>,
) -> Result<TracedIb> {
    let words = segment.commands().to_vec();
    let snapshot = super::trace::snapshot(&words, segment.start, trace, sequence, storage, read)?;
    let mut image = Vec::new();
    image.extend_from_slice(preamble);
    image.extend_from_slice(copies);
    if let Some(values) = restore { user_data_packets(values, compute, &mut image)?; }
    image.extend(snapshot.image);
    let padding = (IB_ALIGNMENT_DWORDS - image.len() % IB_ALIGNMENT_DWORDS) % IB_ALIGNMENT_DWORDS;
    image.extend(host_pm4::nop_fill(padding));
    Ok(TracedIb { image, source: snapshot.words, ranges: snapshot.ranges, followed_chain: snapshot.followed_chain })
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/agc/submit.rs"]
mod tests;

/// The host IB of a segment: the queue's preamble, `copies` (surface work, which needs the
/// preamble's compute defaults), the segment's user data, and its frozen executable commands.
pub fn host_ib(copies: &[u32], preamble: &[u32], restore: Option<&UserData>, compute: bool, segment: &Segment) -> Result<Vec<u32>> {
    let mut image = Vec::with_capacity(copies.len() + preamble.len() + 160 + segment.commands().len());
    image.extend_from_slice(preamble);
    image.extend_from_slice(copies);
    if let Some(values) = restore {
        user_data_packets(values, compute, &mut image)?;
    }
    image.extend_from_slice(segment.commands());
    let gap = (IB_ALIGNMENT_DWORDS - image.len() % IB_ALIGNMENT_DWORDS) % IB_ALIGNMENT_DWORDS;
    image.extend(host_pm4::nop_fill(gap));
    Ok(image)
}
