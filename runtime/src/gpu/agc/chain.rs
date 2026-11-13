//! Submissions that run through several command buffers (decisions D35, D38).
//!
//! A title's command buffer often ends in a CHAIN to another one, recorded by another
//! `CommandBuffer` object (on another thread, or before the buffer that chains to it), and the
//! commands there run with the registers the commands before them leave. Submission follows the
//! tail jumps through what the recordings noted ([`follow`]): each piece is entered from a host IB
//! of its own, so the jump itself never runs, and the events, waits, label writes and resources of
//! every buffer on the way reach the plan. It then works out the guest state at every point, in
//! the order the commands run, from what the queue's registers hold before them ([`resolve`]),
//! and translates each execution into submission-owned bytes without changing the recording.

use super::commands::HostQueue;
use super::side_table::{named_list, BranchKind, ChangeKind, ChunkIndex, Cut, CutReason, DmaWrite, LabelWrite, Notice, Owner, Root, SideTable, StateChange, UserData, Work, WorkKind};
use super::state::{report_once, StateRecord};
use super::{apply_list, apply_register, footprint, nop, packet_dwords, patch, runtime::TargetRuntime, user_data, write_dispatch, write_draw, Baseline, Generation};
use crate::agc::target::Kind;
use crate::amd::defs::pkt3;
use crate::ir::RegisterBank;
use crate::{Error, Result};
use std::collections::HashSet;
use std::sync::Arc;

/// Tail jumps a submission may take before it is taken for a loop.
const MAX_STRETCHES: usize = 1 << 16;
/// INDIRECT_BUFFER dword 3: the size and the CHAIN bit.
const IB_SIZE: u32 = 0xf_ffff;
const IB_CHAIN: u32 = 1 << 20;
/// Dwords of the COND_EXEC packet; dword 4 holds how many dwords after it it guards.
const COND_EXEC_DWORDS: u64 = 5;
const COND_EXEC_COUNT: u32 = 0x3fff;

/// Commands that run one after another without a jump, `[start, end)`, and the recordings that
/// wrote them.
pub struct Stretch {
    pub start: u64,
    pub end: u64,
    pub owners: Vec<Owner>,
}

impl Stretch {
    /// Every recording the stretch's commands come from.
    pub fn tables(&self) -> impl Iterator<Item = &Arc<super::side_table::SideTable>> {
        self.owners.iter().map(|owner| &owner.table)
    }
}

/// A recorded CHAIN, as it stands in memory now (the title may have patched or cleared it).
struct TailJump {
    at: u64,
    target: u64,
    dwords: u32,
    /// Predicated, or inside a COND_EXEC range: it may not run.
    conditional: bool,
}

/// `N` dwords of the title's commands at `at`.
///
/// # Safety
/// The title's command memory, which it keeps until the GPU has run the commands.
unsafe fn words<const N: usize>(at: u64) -> [u32; N] {
    // SAFETY: the caller's contract.
    unsafe { std::ptr::read_volatile(at as *const [u32; N]) }
}

/// Whether `header` starts a type-3 packet of `opcode` and `dwords` dwords (whatever its flags).
fn is_packet(header: u32, opcode: u32, dwords: usize) -> bool {
    packet_dwords(header) == Some(dwords) && (header >> 8) & 0xff == opcode
}

/// The ranges the COND_EXECs a recording wrote in `owner`'s part guard, as the packets stand now:
/// from the COND_EXEC to the end of the dwords it may skip.
fn guards(owner: &Owner) -> Vec<(u64, u64)> {
    owner
        .table
        .branches
        .iter()
        .filter(|branch| branch.kind == BranchKind::CondExec && owner.holds(branch.at))
        .filter_map(|branch| {
            // SAFETY: a COND_EXEC footprint this recording wrote, in submitted commands.
            let packet = unsafe { words::<5>(branch.at) };
            is_packet(packet[0], pkt3::COND_EXEC, COND_EXEC_DWORDS as usize).then(|| (branch.at, branch.at + 4 * (COND_EXEC_DWORDS + u64::from(packet[4] & COND_EXEC_COUNT))))
        })
        .collect()
}

/// Whether a COND_EXEC before `at` in the stretch guards it.
fn guarded(owners: &[Owner], start: u64, at: u64) -> bool {
    owners.iter().flat_map(guards).any(|(from, end)| (start..at).contains(&from) && at < end)
}

/// The first CHAIN of `[start, end)` that still is one: the CP leaves the commands there.
fn tail_jump(owners: &[Owner], start: u64, end: u64) -> Option<TailJump> {
    owners.iter().find_map(|owner| {
        owner.table.branches.iter().filter(|branch| branch.kind == BranchKind::Chain && owner.holds(branch.at) && branch.at + 16 <= end).find_map(|branch| {
            // SAFETY: a jump footprint this recording wrote, in submitted commands.
            let packet = unsafe { words::<4>(branch.at) };
            (is_packet(packet[0], pkt3::INDIRECT_BUFFER, 4) && packet[3] & IB_CHAIN != 0).then(|| TailJump {
                at: branch.at,
                target: (u64::from(packet[1]) | u64::from(packet[2]) << 32) & !3,
                dwords: packet[3] & IB_SIZE,
                conditional: packet[0] & 1 != 0 || guarded(owners, start, branch.at),
            })
        })
    })
}

/// The stretches the commands at `address` run through, following each tail CHAIN into the
/// commands it continues in. A conditional CHAIN stays in its stretch and runs on the GPU; what
/// it may run is not followed (reported once).
pub fn follow(index: &mut ChunkIndex, address: u64, dwords: u32) -> Result<Vec<Stretch>> {
    let mut stretches = Vec::new();
    let mut seen = HashSet::new();
    let (mut start, mut dwords) = (address, dwords);
    loop {
        let end = start + u64::from(dwords) * 4;
        if !seen.insert(start) || stretches.len() == MAX_STRETCHES {
            return Err(Error::invalid(format!("the commands at 0x{address:x} chain back to 0x{start:x}: they never end")));
        }
        let owners = index.owners(start, end);
        if owners.is_empty() {
            return Err(Error::invalid(format!("no recording wrote the commands at 0x{start:x}")));
        }
        match tail_jump(&owners, start, end) {
            Some(jump) if !jump.conditional => {
                let owners = owners.into_iter().filter(|owner| owner.from < jump.at).map(|owner| Owner { to: owner.to.min(jump.at), ..owner }).collect();
                stretches.push(Stretch { start, end: jump.at, owners });
                if jump.dwords == 0 {
                    break;
                }
                (start, dwords) = (jump.target, jump.dwords);
            }
            Some(jump) => {
                report_once(&format!("a conditional CHAIN at 0x{:x} runs on the GPU: the commands it may continue in are not planned", jump.at), 5);
                stretches.push(Stretch { start, end, owners });
                break;
            }
            None => {
                stretches.push(Stretch { start, end, owners });
                break;
            }
        }
    }
    Ok(stretches)
}

/// Where the commands in `[start, end)` stop being whole type-3 packets, and the dword there:
/// a check for diagnostic runs (`APS5_AGC_TRACE`), which parse the commands anyway.
pub fn break_in(start: u64, end: u64) -> Option<(u64, u32)> {
    let mut at = start;
    while at < end {
        // SAFETY: commands the title submitted, in memory it keeps until they ran.
        let header = unsafe { std::ptr::read_volatile(at as *const u32) };
        match packet_dwords(header) {
            Some(dwords) if at + dwords as u64 * 4 <= end => at += dwords as u64 * 4,
            _ => return Some((at, header)),
        }
    }
    None
}

/// The recordings of `stretch` and what they noted near `at`, for a stretch that breaks there.
pub fn describe_break(stretch: &Stretch, at: u64) -> String {
    let near = |position: u64| position + 128 >= at && position <= at + 128;
    stretch
        .owners
        .iter()
        .map(|owner| {
            let table = &owner.table;
            let mut notes: Vec<String> = Vec::new();
            notes.extend(table.works.iter().filter(|work| near(work.at)).map(|work| format!("work 0x{:x} first {:08x} {:?}", work.at, work.first, work.kind)));
            notes.extend(table.cuts.iter().filter(|cut| near(cut.at)).map(|cut| format!("cut 0x{:x}", cut.at)));
            notes.extend(table.writes.iter().filter(|write| near(write.at)).map(|write| format!("write 0x{:x}..0x{:x} to 0x{:x}", write.at, write.end, write.address)));
            notes.extend(table.branches.iter().filter(|branch| near(branch.at)).map(|branch| format!("{:?} 0x{:x}", branch.kind, branch.at)));
            notes.extend(table.changes.iter().filter(|change| near(change.at)).take(8).map(|change| format!("change 0x{:x}", change.at)));
            format!("owner 0x{:x}..0x{:x} table {:p} ({} works): {}", owner.from, owner.to, Arc::as_ptr(table), table.works.len(), notes.join(", "))
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

/// A stretch with the guest state worked out at each point the plan needs.
pub struct Resolved {
    pub start: u64,
    pub end: u64,
    /// This occurrence's executable bytes. Other submissions may inherit different state from
    /// the same recording; none of them may overwrite this occurrence's translated work.
    pub commands: Arc<[u32]>,
    /// The user data the stretch starts with.
    pub user_data: UserData,
    pub cuts: Vec<(Cut, UserData)>,
    pub writes: Vec<(LabelWrite, UserData)>,
    pub notices: Vec<Notice>,
    /// Descriptor roots of its draws and dispatches, by where they are.
    pub roots: Vec<(u64, Root)>,
    pub dma_writes: Vec<DmaWrite>,
    /// Ranges a COND_EXEC guards, from the COND_EXEC to the end of what it may skip: a segment
    /// cannot start inside one (D25), the CP's skip would run on past the end of the IB.
    pub guards: Vec<(u64, u64)>,
    /// The user data where each guarded range ends, for a segment that starts there.
    pub marks: Vec<(u64, UserData)>,
    /// Draws and dispatches rewritten with the state they run with.
    pub rewritten: usize,
    /// Draws and dispatches no state could be found for with the state they run with: NOPs.
    pub rejected: usize,
    /// Draws and dispatches whose footprint holds other commands now, by where: left alone.
    pub stale: Vec<u64>,
    /// State changes read here that their recordings noted and never read themselves.
    pub unread: usize,
}

/// A point of a recording's commands where the state matters.
enum Point<'a> {
    Cut(&'a Cut),
    Write(&'a LabelWrite),
    Work(&'a Work),
    /// The end of a COND_EXEC range.
    Mark(u64),
}

/// The guest state along a submission's commands.
pub struct Walk<'a, R> {
    state: &'a mut Baseline,
    runtime: &'a TargetRuntime,
    generation: &'a Generation,
    read: R,
    list: Vec<u32>,
    bound_graphics: Option<Arc<StateRecord>>,
    bound_compute: Option<Arc<StateRecord>>,
    /// A register list left user data that no command has loaded on the GPU since.
    unloaded: bool,
    /// Prints the target registers of every draw that renders to nothing (`APS5_AGC_PASSES`).
    pub inspect: bool,
    pub split_work: bool,
}

impl<'a, R: FnMut(u64, &mut [u32]) -> Result<()>> Walk<'a, R> {
    /// A walk from `state`, what the queue's registers hold before the commands; `read` reads guest
    /// memory (register lists).
    pub fn new(state: &'a mut Baseline, runtime: &'a TargetRuntime, generation: &'a Generation, read: R) -> Self {
        Self { state, runtime, generation, read, list: Vec::new(), bound_graphics: None, bound_compute: None, unloaded: false, inspect: false, split_work: false }
    }

    /// Works the state out along `stretches`, leaving it as their commands leave it.
    pub fn resolve(&mut self, stretches: &[Stretch]) -> Result<Vec<Resolved>> {
        stretches.iter().map(|stretch| self.stretch(stretch)).collect()
    }

    fn stretch(&mut self, stretch: &Stretch) -> Result<Resolved> {
        let mut commands = vec![0; ((stretch.end - stretch.start) / 4) as usize];
        (self.read)(stretch.start, &mut commands)?;
        let mut resolved = Resolved {
            start: stretch.start,
            end: stretch.end,
            commands: Arc::from([]),
            user_data: user_data(&self.state.guest),
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
        };
        // The stretch starts a segment, which loads the user data the state holds.
        self.unloaded = false;
        for (piece, owner) in stretch.owners.iter().enumerate() {
            // Each piece is entered from a host IB, or written by another recording: no state is
            // bound when it starts, as none was when the recording started.
            self.bound_graphics = None;
            self.bound_compute = None;
            if std::mem::take(&mut self.unloaded) && piece != 0 {
                resolved.cuts.push((Cut { at: owner.from, reason: CutReason::UserData, changes: 0 }, user_data(&self.state.guest)));
            }
            let table = &owner.table;
            // What the recording noted behind a register list and never read (a list after its
            // last draw) comes after the changes it made: the commands that run after these run
            // with it. Asked for first, so that no change of a recording still going on is missed.
            let unsettled: Vec<StateChange> = table.unsettled().copied().collect();
            let logged = table.changes.len();
            let unread = unsettled.into_iter().enumerate().map(|(index, change)| (logged + index, change));
            let changes: Vec<(usize, StateChange)> = table.changes.iter().copied().take(logged).enumerate().chain(unread).filter(|(_, change)| owner.holds(change.at)).collect();
            resolved.unread += changes.iter().filter(|&&(index, _)| index >= logged).count();
            let mut points: Vec<(usize, u64, u8, Point<'_>)> = Vec::new();
            // A cut right after the part's last command (a flip, a fast clear) belongs to it too.
            let cut_here = |at: u64| owner.from <= at && at <= owner.to;
            points.extend(table.cuts.iter().filter(|cut| cut_here(cut.at)).map(|cut| (cut.changes, cut.at, 0, Point::Cut(cut))));
            points.extend(table.writes.iter().filter(|write| owner.holds(write.at)).map(|write| (write.changes, write.at, 0, Point::Write(write))));
            points.extend(table.works.iter().filter(|work| owner.holds(work.at)).map(|work| (work.changes, work.at, 1, Point::Work(work))));
            let guards = guards(owner);
            // The state where a guarded range ends: after the changes the commands before it make.
            let changes_before = |at: u64| changes.iter().find(|(_, change)| change.at >= at).map_or(usize::MAX, |&(index, _)| index);
            points.extend(guards.iter().filter(|&&(_, end)| end <= owner.to).map(|&(_, end)| (changes_before(end), end, 0, Point::Mark(end))));
            points.sort_by_key(|&(changes, at, rank, _)| (changes, at, rank));
            let mut next = 0;
            for (applied, _, _, point) in points {
                while next < changes.len() && changes[next].0 < applied {
                    self.apply(&changes[next].1);
                    next += 1;
                }
                match point {
                    Point::Cut(cut) => {
                        resolved.cuts.push((*cut, user_data(&self.state.guest)));
                        self.unbind();
                    }
                    Point::Write(write) => {
                        resolved.writes.push((*write, user_data(&self.state.guest)));
                        self.unbind();
                    }
                    Point::Work(work) => {
                        let first_word = ((work.at - stretch.start) / 4) as usize;
                        let count = footprint(work_kind(work))? as usize;
                        let out = commands.get_mut(first_word..first_word + count)
                            .ok_or_else(|| Error::invalid("a recorded work leaves its submitted stretch"))?;
                        let first = out[0];
                        if first & !1 != work.encoding.first.load(std::sync::atomic::Ordering::Acquire) & !1 {
                            report_once(&format!("the footprint at 0x{:x} holds other commands than the work recorded there: left alone", work.at), 8);
                            resolved.stale.push(work.at);
                            continue;
                        }
                        if self.split_work && !guards.iter().any(|&(from, to)| from < work.at && work.at < to) {
                            resolved.cuts.push((Cut { at: work.at, reason: CutReason::UserData, changes: work.changes }, user_data(&self.state.guest)));
                            self.unbind();
                        }
                        if let Some(mut root) = self.work(table, work, out, &mut resolved.rewritten, &mut resolved.rejected)? {
                            root.point.guarded = guards.iter().any(|&(from, to)| from < work.at && work.at < to);
                            resolved.roots.push((work.at, root));
                        }
                    }
                    Point::Mark(at) => {
                        resolved.marks.push((at, user_data(&self.state.guest)));
                        self.unbind();
                    }
                }
            }
            for (_, change) in &changes[next..] {
                self.apply(change);
            }
            // A range that runs on past this recording's part ends with the state the part leaves.
            for &(_, end) in guards.iter().filter(|&&(_, end)| end > owner.to) {
                resolved.marks.push((end, user_data(&self.state.guest)));
            }
            resolved.guards.extend(guards);
            resolved.notices.extend(table.notices.iter().filter(|notice| owner.holds(notice.at)).copied());
            resolved.dma_writes.extend(table.dma_writes.iter().filter(|write| owner.holds(write.at)).copied());
        }
        resolved.notices.sort_by_key(|notice| notice.at);
        resolved.commands = commands.into();
        Ok(resolved)
    }

    /// A segment may start here: what the CP had bound is gone.
    fn unbind(&mut self) {
        self.bound_graphics = None;
        self.bound_compute = None;
    }

    fn apply(&mut self, change: &StateChange) {
        let before = self.inspect.then(|| super::surface::RenderTargets::registers(&self.state.guest));
        self.apply_change(change);
        if let Some(before) = before {
            let after = super::surface::RenderTargets::registers(&self.state.guest);
            if after != before {
                eprintln!("[pass] the change at 0x{:x} ({:x?}) leaves the target registers as: {after}", change.at, change.kind);
            }
            // A list the title went on appending to after its recording read it.
            if let ChangeKind::List { bank, address, count } = change.kind {
                // SAFETY: a footprint its recording wrote, in commands the title submitted.
                let footprint = unsafe { words::<5>(change.at) };
                match unsafe { named_list(change.at) } {
                    Some(named) if named != (address, count) => eprintln!("[pass] the {bank:?} list at 0x{:x} was read as {count} pairs at 0x{address:x}; its footprint names {} pairs at 0x{:x} now", change.at, named.1, named.0),
                    None if footprint[4] != count => eprintln!("[pass] the {bank:?} list at 0x{:x} was read as {count} pairs at 0x{address:x}; its footprint is {footprint:x?} now", change.at),
                    _ => {}
                }
            }
        }
    }

    fn apply_change(&mut self, change: &StateChange) {
        let (guest, parameters) = (&mut self.state.guest, &mut self.state.parameters);
        let applied = match change.kind {
            ChangeKind::Register { bank, offset, value } => {
                apply_register(guest, parameters, bank, offset, value);
                Ok(())
            }
            ChangeKind::InPlace { bank, offset, count } => {
                // SAFETY: the footprint the recording wrote at `at`, from dword 2 on the values the
                // title has filled in by its submission.
                let values = unsafe { std::slice::from_raw_parts((change.at as *const u32).add(2), count as usize) };
                for (index, &value) in values.iter().enumerate() {
                    apply_register(guest, parameters, bank, offset + index as u32, value);
                }
                Ok(())
            }
            ChangeKind::List { bank, .. } | ChangeKind::ListAt { bank } => {
                // The list as the title left it, which the CP reads whole where the packet is: the
                // title appends to a list until the draw that needs it, across the dispatches on
                // the way, so its recording may have read less of it or none (D38).
                // SAFETY: a footprint its recording wrote, in commands the title submitted.
                let named = unsafe { named_list(change.at) };
                let (address, count) = match (named, change.kind) {
                    (Some(named), _) => named,
                    // The packet loads the list's user data on the GPU: as its recording read it.
                    (None, ChangeKind::List { address, count, .. }) => (address, count),
                    (None, _) => (0, 0),
                };
                // An unread list's footprint is a NOP on the GPU: user data in it reaches the
                // commands after it with the segment they start.
                self.unloaded |= matches!(change.kind, ChangeKind::ListAt { bank: RegisterBank::Shader }) && count != 0;
                self.list.resize(count as usize * 2, 0);
                if count == 0 { Ok(()) } else { (self.read)(address, &mut self.list).and_then(|()| apply_list(guest, parameters, bank, &self.list)) }
            }
            ChangeKind::IndexType(size) => {
                parameters.index_type = size;
                Ok(())
            }
            ChangeKind::ContextStateOp(operation) => {
                self.bound_graphics = None;
                guest.context_state_op(operation)
            }
            ChangeKind::IndexBase(base) => {
                parameters.index_base = base;
                Ok(())
            }
            ChangeKind::IndexCount(count) => {
                parameters.index_buffer_size = Some(count);
                Ok(())
            }
            ChangeKind::Instances(count) => {
                parameters.instance_count = count;
                Ok(())
            }
            ChangeKind::IndirectBase { dispatch: true, address } => {
                parameters.dispatch_indirect_base = address;
                Ok(())
            }
            ChangeKind::IndirectBase { dispatch: false, address } => {
                parameters.draw_indirect_base = address;
                Ok(())
            }
        };
        if let Err(error) = applied {
            report_once(&format!("the state change at 0x{:x} cannot be applied at submission: {}", change.at, error.message), 5);
        }
    }

    /// Checks a draw or dispatch against the state it runs with, rewrites it when the recording
    /// guessed another, and returns its descriptor root.
    fn work(&mut self, table: &SideTable, work: &Work, out: &mut [u32], rewritten: &mut usize, rejected: &mut usize) -> Result<Option<Root>> {
        // Resource kernels use the compute registers between any two guest works.
        if matches!(work.kind, WorkKind::Dispatch { .. }) { self.bound_compute = None; }
        let (planned, bound) = match work.kind {
            WorkKind::Draw { .. } => (self.runtime.states.draw(self.runtime, &self.state.guest), &mut self.bound_graphics),
            WorkKind::Dispatch { queue, modifier, .. } => (self.runtime.states.dispatch(self.runtime, &self.state.guest, modifier, queue == HostQueue::Compute), &mut self.bound_compute),
        };
        if self.inspect {
            eprintln!("[pass] 0x{:x}: {}", work.at, if matches!(work.kind, WorkKind::Draw { .. }) { "draw" } else { "dispatch" });
            if let Ok(Some(targets)) = planned.as_ref().map(|state| state.targets.as_ref()) {
                if targets.colors.is_empty() && targets.depth.is_none() {
                    eprintln!("[pass] the draw at 0x{:x} renders to nothing: {}", work.at, super::surface::RenderTargets::registers(&self.state.guest));
                }
            }
        }
        let state = match planned {
            Ok(state) => Some(state),
            Err(reason) => {
                *rejected += 1;
                report_once(&format!("at submission: {reason}"), 9);
                // Where a rejected draw would have drawn: a target nothing else draws stays empty.
                if matches!(work.kind, WorkKind::Draw { .. }) {
                    if let Ok(targets) = super::surface::RenderTargets::of(&self.state.guest) {
                        for color in &targets.colors {
                            let desc = color.layout.desc();
                            report_once(&format!("at submission a draw into 0x{:x} ({}x{}, {} bytes per element) is left out: {reason}", color.base, desc.width, desc.height, desc.bytes_per_element), 10);
                        }
                    }
                }
                None
            }
        };
        let same = work.encoding.matches(state.as_ref().map(|state| state.key), bound.as_ref().map_or(0, |bound| bound.key), self.state.parameters);
        if !same {
            rewrite(self.generation, self.runtime, table, work, state.as_ref(), bound.as_ref(), &self.state.parameters, out)?;
            *rewritten += 1;
        }
        let Some(state) = state else { return Ok(None) };
        let values = state.watched.iter().map(|&dword| self.state.guest.get_or_zero(RegisterBank::Shader, dword)).collect();
        *bound = Some(state.clone());
        let groups = match work.kind {
            WorkKind::Dispatch { groups, .. } => groups,
            WorkKind::Draw { .. } => None,
        };
        let predicated = patch::work_predicate(out);
        let point = super::side_table::WorkPoint {
            at: work.at,
            end: work.at + out.len() as u64 * 4,
            compute_user: user_data(&self.state.guest)[96..112].try_into().expect("compute user data"),
            threads: std::array::from_fn(|index| self.state.guest.get_or_zero(RegisterBank::Shader, (crate::amd::defs::gfx10::COMPUTE_NUM_THREAD_X::OFFSET - 0xb000) / 4 + index as u32)),
            start: std::array::from_fn(|index| self.state.guest.get_or_zero(RegisterBank::Shader, (crate::amd::defs::gfx10::COMPUTE_START_X::OFFSET - 0xb000) / 4 + index as u32)),
            predicated,
            guarded: false,
        };
        Ok(Some(Root { point, record: state, values, groups }))
    }
}

/// Writes `work`'s footprint again for `state` after `bound`, keeping the predicate the title put
/// on it; no state leaves it a NOP.
fn work_kind(work: &Work) -> Kind {
    match work.kind {
        WorkKind::Draw { draw, .. } => Kind::Draw(draw.kind()),
        WorkKind::Dispatch { groups: Some(_), .. } => Kind::Dispatch,
        WorkKind::Dispatch { groups: None, .. } => Kind::DispatchIndirect,
    }
}

fn rewrite(generation: &Generation, runtime: &TargetRuntime, table: &SideTable, work: &Work, state: Option<&Arc<StateRecord>>, bound: Option<&Arc<StateRecord>>, parameters: &super::draw::DrawParameters, out: &mut [u32]) -> Result<()> {
    let predicated = patch::work_predicate(out);
    match (state, work.kind) {
        (None, _) => nop(out),
        (Some(state), WorkKind::Draw { draw, modifier }) => write_draw(generation, runtime, table, bound, state, draw, &modifier, parameters, out)?,
        (Some(state), WorkKind::Dispatch { queue, groups, indirect, modifier }) => write_dispatch(bound, state, queue, groups, indirect, modifier, parameters, out)?,
    }
    if predicated {
        patch::set_work_predicate(out);
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/agc/chain.rs"]
mod tests;
