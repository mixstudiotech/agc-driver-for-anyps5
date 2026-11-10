//! Opt-in recording, planning and completion counts. These observe the normal execution path;
//! following the tail jumps and working out the state is `chain`'s.
//! Recording counts include unsubmitted buffers; planning counts include repeated submissions.
//! Retired metadata has passed a host fence, before VideoOut accepts or presents anything.

use super::chain::Resolved;
use super::side_table::{Event, SideTable};
use super::submit::Segment;
use crate::scheduler::{SubmitEvent, EOP_FLIP_TOKENS};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Default)]
struct Events {
    flips: AtomicU64,
    eop_flips: AtomicU64,
    eops: AtomicU64,
}

impl Events {
    fn counter(&self, event: &Event) -> &AtomicU64 {
        match event {
            Event::Flip { .. } => &self.flips,
            Event::Interrupt { context, .. } if EOP_FLIP_TOKENS.contains(context) => &self.eop_flips,
            Event::Interrupt { .. } => &self.eops,
        }
    }

    fn note(&self, event: &SubmitEvent) {
        let count = match event {
            SubmitEvent::Flip(_) => &self.flips,
            SubmitEvent::EopFlip(_) => &self.eop_flips,
            SubmitEvent::Eop(_) => &self.eops,
            SubmitEvent::Label(_) => return,
        };
        count.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Default)]
pub(super) struct Flow {
    recorded: Events,
    planned: Events,
    completed: Events,
    plans: AtomicU64,
    chains: AtomicU64,
    patched_chains: AtomicU64,
    draws: AtomicU64,
    skipped_draws: AtomicU64,
    missing_stages: AtomicU64,
    non_ngg_stages: AtomicU64,
    zero_stages: AtomicU64,
    zero_stage_writes: AtomicU64,
    nonzero_stage_writes: AtomicU64,
    deferred_stage_writes: AtomicU64,
    rejected_gds_source: AtomicU64,
    rejected_gds_destination: AtomicU64,
    present_calls: AtomicU64,
}

pub(super) fn diagnostics() -> Option<&'static Flow> {
    static FLOW: OnceLock<Option<Flow>> = OnceLock::new();
    FLOW.get_or_init(|| std::env::var_os("APS5_AGC_FLOW_DIAG").map(|_| Flow::default())).as_ref()
}

impl Flow {
    pub fn recorded(&self, table: &SideTable, at: u64, event: Event) {
        let previous = self.recorded.counter(&event).fetch_add(1, Ordering::Relaxed);
        if previous < 8 {
            eprintln!("[flow-record] table={table:p} at=0x{at:x} {event:?}");
        }
    }

    /// Counts draw requests, including cache hits, after settling the recorded register lists.
    /// A modeled zero may come from queue defaults or inheritance, not a title write.
    pub fn draw(&self, table: &SideTable, at: u64, stages: Option<u32>, skipped: bool) {
        self.draws.fetch_add(1, Ordering::Relaxed);
        if skipped {
            self.skipped_draws.fetch_add(1, Ordering::Relaxed);
        }
        let counter = match stages {
            None => &self.missing_stages,
            Some(0) => {
                self.non_ngg_stages.fetch_add(1, Ordering::Relaxed);
                &self.zero_stages
            }
            Some(value) if value & (1 << 13) == 0 => &self.non_ngg_stages,
            Some(_) => return,
        };
        if counter.fetch_add(1, Ordering::Relaxed) < 4 {
            eprintln!("[flow-draw] table={table:p} at=0x{at:x} VGT_SHADER_STAGES_EN={stages:x?} state_rejected={skipped}");
        }
    }

    pub fn stage_write(&self, table: &SideTable, value: u32) {
        let count = if value == 0 { &self.zero_stage_writes } else { &self.nonzero_stage_writes };
        if count.fetch_add(1, Ordering::Relaxed) < 16 {
            eprintln!("[flow-stage-write] table={table:p} value=0x{value:x}");
        }
    }

    pub fn deferred_stage_write(&self, table: &SideTable, at: u64) {
        if self.deferred_stage_writes.fetch_add(1, Ordering::Relaxed) < 4 {
            eprintln!("[flow-stage-deferred] table={table:p} at=0x{at:x} value_filled_later");
        }
    }

    pub fn rejected_gds_dma(&self, source: bool) {
        let count = if source { &self.rejected_gds_source } else { &self.rejected_gds_destination };
        count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn chain(&self, table: &SideTable, at: u64, target: u64, dwords: u32) {
        if self.chains.fetch_add(1, Ordering::Relaxed) < 128 {
            eprintln!("[flow-chain] table={table:p} at=0x{at:x} target=0x{target:x}+{dwords:x}dw");
        }
    }

    pub fn patched_chain(&self, at: u64, target: u64, dwords: u32) {
        if self.patched_chains.fetch_add(1, Ordering::Relaxed) < 128 {
            eprintln!("[flow-patch-chain] at=0x{at:x} target=0x{target:x}+{dwords:x}dw");
        }
    }

    pub fn planned(&self, stretches: &[Resolved], queue: u32, address: u64, dwords: u32, segments: &[Segment], leases: &[super::arena::Lease]) {
        for event in segments.iter().flat_map(|segment| &segment.events) {
            self.planned.note(event);
        }
        if self.plans.fetch_add(1, Ordering::Relaxed) < 40 {
            let entry_end = address + u64::from(dwords) * 4;
            let in_entry = stretches.first().map_or(0, |entry| entry.notices.iter().filter(|notice| (address..entry_end).contains(&notice.at)).count());
            eprintln!(
                "[flow-plan] q{queue:x} 0x{address:x}+{dwords:x}dw stretches={} notices_in_entry={in_entry} notices_planned={} roots={} rewritten={} segments={} retained_blocks={:?}",
                stretches.len(),
                stretches.iter().map(|stretch| stretch.notices.len()).sum::<usize>(),
                stretches.iter().map(|stretch| stretch.roots.len()).sum::<usize>(),
                stretches.iter().map(|stretch| stretch.rewritten).sum::<usize>(),
                segments.len(),
                leases.iter().map(super::arena::Lease::index).collect::<Vec<_>>()
            );
        }
    }

    pub fn completed(&self, event: &SubmitEvent) {
        self.completed.note(event);
    }

    pub fn present_requested(&self) {
        self.present_calls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report(&self) {
        let load = |count: &AtomicU64| count.load(Ordering::Relaxed);
        eprintln!(
            "[flow] cumulative: flips recorded/planned/retired_metadata={}/{}/{}; eop-flip-tokens={}/{}/{}; other-eops={}/{}/{}; draws={} state_rejected={} stages_missing={} stages_non_ngg={} stages_modeled_zero={}; stage_writes_applied_zero/nonzero={}/{} deferred_placeholders={}; rejected_gds_dma source/destination={}/{}; native_present_calls={}; chain/patch/plan_logs_seen={}/{}/{} (limits 128/128/40)",
            load(&self.recorded.flips), load(&self.planned.flips), load(&self.completed.flips),
            load(&self.recorded.eop_flips), load(&self.planned.eop_flips), load(&self.completed.eop_flips),
            load(&self.recorded.eops), load(&self.planned.eops), load(&self.completed.eops),
            load(&self.draws), load(&self.skipped_draws), load(&self.missing_stages), load(&self.non_ngg_stages), load(&self.zero_stages),
            load(&self.zero_stage_writes), load(&self.nonzero_stage_writes), load(&self.deferred_stage_writes),
            load(&self.rejected_gds_source), load(&self.rejected_gds_destination), load(&self.present_calls),
            load(&self.chains), load(&self.patched_chains), load(&self.plans)
        );
    }
}
