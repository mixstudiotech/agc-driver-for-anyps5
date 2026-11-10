//! Running what recordings wrote (decisions D25, D27, D38): a submission is cut at the guest's
//! waits that nothing in it can end, after writes something else waits for, and where a tail jump
//! continues in other commands; each piece is entered from a host IB by CHAIN, so the GPU executes
//! the title's commands where they are.

use super::chain::Resolved;
use super::side_table::{CutReason, Event, LabelWrite, Root, SideTable, UserData, WaitCondition, USER_DATA_REGISTERS};
use crate::agc::header;
use crate::amd::defs::{gfx10, pkt3};
use crate::backend::host_pm4::{self, IB_ALIGNMENT_DWORDS, MAX_IB_DWORDS};
use crate::ir::FlipRequest;
use crate::residency::Queue;
use crate::scheduler::{EopEvent, SubmitEvent, EOP_FLIP_TOKENS};
use crate::{Error, Result};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;

/// A piece of a submission: the commands at `[start, end)` run after `wait` holds.
#[derive(Clone, Debug)]
pub struct Segment {
    pub start: u64,
    pub end: u64,
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
    /// Fast-clear metadata the title wrote just before: the targets' host copies are filled first.
    pub clears: Vec<u64>,
    /// The descriptor roots of its draws and dispatches: the surfaces they use are brought up to
    /// date when it is queued, after what it waits for.
    pub roots: Vec<Root>,
    /// Guest memory its DMA writes: (address, bytes).
    pub dma_writes: Vec<(u64, u64)>,
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
    clears: Vec<u64>,
}

impl Boundary {
    fn merge(&mut self, other: Boundary) {
        self.wait = self.wait.or(other.wait);
        self.flip_waits.extend(other.flip_waits);
        for clear in other.clears {
            if !self.clears.contains(&clear) {
                self.clears.push(clear);
            }
        }
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
            let (wait, flip, clear) = match cut.reason {
                CutReason::Wait(condition) => (Some(condition), None, None),
                CutReason::FlipDone { video_out, buffer } => (None, Some((video_out, buffer)), None),
                CutReason::AfterFlip | CutReason::UserData => (None, None, None),
                CutReason::TargetClear { metadata } => (None, None, Some(metadata)),
            };
            let (at, user_data) = outside_guards(stretch, cut.at, *user_data);
            if at != cut.at && wait.is_some() {
                super::state::report_once(&format!("a wait inside a COND_EXEC range (0x{:x}) stays on the GPU: a segment cannot start there", cut.at), 6);
                continue;
            }
            boundaries.push(Boundary { at, wait, flip_waits: flip.into_iter().collect(), user_data, clears: clear.into_iter().collect() });
        }
        for (write, user_data) in stretch.writes.iter().filter(|(write, _)| write.end < end && awaited.ended_by(write)) {
            let (at, user_data) = outside_guards(stretch, write.end, *user_data);
            boundaries.push(Boundary { at, wait: None, flip_waits: Vec::new(), user_data, clears: Vec::new() });
        }
        boundaries.sort_by_key(|boundary| boundary.at);
        let mut starts = vec![Boundary { at: start, wait: None, flip_waits: Vec::new(), user_data: stretch.user_data, clears: Vec::new() }];
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
                carried.get_or_insert(Boundary { at: end, wait: None, flip_waits: Vec::new(), user_data: boundary.user_data, clears: Vec::new() }).merge(boundary);
            }
        }
        if start == end {
            // Nothing runs here: what it waits for waits on before the next stretch.
            let empty = starts.swap_remove(0);
            carried.get_or_insert(Boundary { at: end, wait: None, flip_waits: Vec::new(), user_data: empty.user_data, clears: Vec::new() }).merge(empty);
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
            let mut seen = HashSet::new();
            let roots = stretch
                .roots
                .iter()
                .filter(|(at, _)| inside(*at))
                .filter(|(_, root)| seen.insert(crate::agc::cache::key(std::iter::once(root.record.key).chain(root.values.iter().map(|&value| u64::from(value))))))
                .map(|(_, root)| root.clone())
                .collect();
            let dma_writes = stretch.dma_writes.iter().filter(|write| inside(write.at)).map(|write| (write.address, write.bytes)).collect();
            segments.push(Segment {
                start: boundary.at,
                end: stop,
                wait: boundary.wait,
                flip_waits: boundary.flip_waits.clone(),
                user_data: Some(boundary.user_data),
                events,
                writes,
                clears: boundary.clears.clone(),
                roots,
                dma_writes,
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

/// A CHAIN into guest commands at `next_ib`, which only need dword alignment (`chain_packet`'s
/// 256-byte rule is about host IBs: the title's command buffers are 4-byte aligned).
fn chain_to(next_ib: u64, dwords: u32) -> Result<[u32; 4]> {
    const CHAIN: u32 = 1 << 20;
    const VALID: u32 = 1 << 23;
    if next_ib % 4 != 0 || dwords == 0 || dwords as usize > MAX_IB_DWORDS {
        return Err(Error::invalid(format!("chained commands at 0x{next_ib:x}: {dwords} dwords")));
    }
    Ok([header(pkt3::INDIRECT_BUFFER, 4, 0)?, next_ib as u32 & !3, (next_ib >> 32) as u32, CHAIN | VALID | dwords])
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

/// As [`host_ib`], but the segment's commands are copied into the IB, each behind a write of its
/// position in `[trace, trace + 8)`: radv's `RADV_DEBUG=hang` trace (`APS5_AGC_TRACE`). A COND_EXEC
/// range keeps its length, so nothing is added inside one.
pub fn traced_ib(copies: &[u32], preamble: &[u32], restore: Option<&UserData>, compute: bool, segment: &Segment, trace: u64, sequence: u64, tables: &[Arc<SideTable>]) -> Result<TracedIb> {
    const WRITE_DATA_MEMORY_CONFIRMED: u32 = (5 << 8) | (1 << 20);
    // SAFETY: the commands of a submission the title made, in memory it keeps until they ran.
    let words = unsafe { std::slice::from_raw_parts(segment.start as *const u32, ((segment.end - segment.start) / 4) as usize) }.to_vec();
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
    mut read: impl FnMut(u64, usize) -> Result<Vec<u32>>,
) -> Result<TracedIb> {
    let words = read(segment.start, ((segment.end - segment.start) / 4) as usize)?;
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
/// preamble's compute defaults), the segment's user data, and the CHAIN into the title's
/// commands; padded so the CHAIN ends the IB on an 8-dword boundary.
pub fn host_ib(copies: &[u32], preamble: &[u32], restore: Option<&UserData>, compute: bool, segment: &Segment) -> Result<Vec<u32>> {
    let dwords = u32::try_from((segment.end - segment.start) / 4).map_err(|_| Error::invalid("a segment too large to chain"))?;
    let mut image = Vec::with_capacity(copies.len() + preamble.len() + 160);
    image.extend_from_slice(preamble);
    image.extend_from_slice(copies);
    if let Some(values) = restore {
        user_data_packets(values, compute, &mut image)?;
    }
    let gap = (IB_ALIGNMENT_DWORDS - (image.len() + 4) % IB_ALIGNMENT_DWORDS) % IB_ALIGNMENT_DWORDS;
    image.extend(host_pm4::nop_fill(gap));
    image.extend(chain_to(segment.start, dwords)?);
    Ok(image)
}
