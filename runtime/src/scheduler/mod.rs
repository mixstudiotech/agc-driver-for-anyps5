pub mod gpu;

use crate::ir::FlipRequest;

/// Guest memory as the integration layer exposes it to the GPU executor.
pub trait GuestPort {
    fn validate(&mut self, address: u64, bytes: usize, writable: bool) -> bool;
    fn read(&mut self, address: u64, out: &mut [u8]) -> crate::Result<()>;
    fn write(&mut self, address: u64, data: &[u8]) -> crate::Result<()>;

}

/// An end-of-pipe interrupt raised by RELEASE_MEM or EVENT_WRITE_EOP.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EopEvent {
    pub queue: u32,
    pub context: u32,
    pub address: u64,
    pub value: u64,
}

/// VideoOut's EOP flip tokens (`EopFlipTokenBase..=EopFlipTokenLimit` in AnyPS5's
/// `VideoOutput.hpp`).
pub const EOP_FLIP_TOKENS: std::ops::RangeInclusive<u32> = 0x0400_0000..=0x07ff_ffff;

/// An end-of-pipe data write that raises no interrupt (a label).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LabelWrite {
    pub queue: u32,
    pub address: u64,
    pub value: u64,
    /// 4 or 8.
    pub bytes: u8,
}

/// One event of a guest submission, as [`SubmitOutcome::events`] lists them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SubmitEvent {
    /// A flip packet.
    Flip(FlipRequest),
    /// A RELEASE_MEM interrupt (INT_SEL 2) whose context lies in [`EOP_FLIP_TOKENS`], the form
    /// an EOP flip takes. The runtime does not track reservations: the integration layer readies
    /// the flip reserved under the token, or delivers a plain interrupt when there is none.
    EopFlip(EopEvent),
    /// Any other end-of-pipe interrupt.
    Eop(EopEvent),
    /// An end-of-pipe label write. Its data is published before any later event is reported.
    Label(LabelWrite),
}

impl SubmitEvent {
    /// The interrupt an `EopFlip` or `Eop` event raises.
    pub fn interrupt(&self) -> Option<&EopEvent> {
        match self {
            Self::EopFlip(event) | Self::Eop(event) => Some(event),
            Self::Flip(_) | Self::Label(_) => None,
        }
    }
}

/// What a guest submission produced once it executed completely.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SubmitOutcome {
    pub serial: u64,
    pub flips: Vec<FlipRequest>,
    pub eop_events: Vec<EopEvent>,
    pub published_bytes: usize,
    /// Flips, EOP flips, end-of-pipe interrupts and labels in the order the command stream
    /// raised them, following indirect buffers and REWIND tails. `flips` and `eop_events` hold
    /// its flip and interrupt subsequences for callers of the two-list form.
    pub events: Vec<SubmitEvent>,
    /// Flips and EOP flips the submission itself requests ([`flips_requested`]), counted when it
    /// is accepted, before any of it runs; 0 for a poll.
    pub submitted_flips: u32,
}
