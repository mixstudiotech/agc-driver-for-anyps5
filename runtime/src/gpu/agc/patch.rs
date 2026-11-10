//! Patches of recorded commands on the AMD target's footprints (`sceAgc*Patch*`, `sceAgcSetNop`
//! and the predication functions; PLAN P3.4 step 2g, decisions D5, D12, D14).
//!
//! - A command whose host packet keeps the console's layout for the patched field (COND_EXEC,
//!   DMA_DATA, INDIRECT_BUFFER, WAIT_REG_MEM, WRITE_DATA) is patched in place.
//! - Predicates go on work packets only (D12 point 4): draws, dispatches and memory writes.

use super::commands::dwords;
use super::sync::ReleasePlan;
use super::{nop, packet_dwords};
use crate::agc::header;
use crate::agc::target::Patch;
use crate::amd::defs::{cp_gfx11 as cp, pkt3};
use crate::{Error, Result};

/// A command the AMD target recorded, as a patch finds it: where its footprint starts, and the
/// packet there.
pub(super) struct Recorded<'a> {
    at: *mut u32,
    packet: &'a mut [u32],
}

impl<'a> Recorded<'a> {
    /// The command whose footprint starts at `at`.
    ///
    /// # Safety
    /// `at` is null, misaligned or starts a packet of a live buffer.
    pub(super) unsafe fn at(at: *mut u32) -> Result<Self> {
        if at.is_null() || at as usize % 4 != 0 {
            return Err(Error::invalid("null or misaligned packet"));
        }
        // SAFETY: the caller's contract.
        let header = unsafe { *at };
        let dwords = packet_dwords(header).ok_or_else(|| Error::invalid("not a type-3 packet"))?;
        // SAFETY: the packet's words, by its own header.
        Ok(Self { at, packet: unsafe { std::slice::from_raw_parts_mut(at, dwords) } })
    }

    fn opcode(&self) -> u32 {
        Commands::opcode(self.packet[0])
    }

    /// Requires the packet the command recorded: `dwords` dwords of `opcode`, predicated or not.
    fn expect(&self, opcode: u32, dwords: u32, what: &str) -> Result<()> {
        if self.packet[0] & !1 != header(opcode, dwords, 0)? {
            return Err(Error::invalid(format!("not the {what} packet the command recorded")));
        }
        Ok(())
    }

    fn set_u64(&mut self, at: usize, value: u64) {
        self.packet[at] = value as u32;
        self.packet[at + 1] = (value >> 32) as u32;
    }

    /// The address bits a memory wait keeps for itself: 3 for WAIT_REG_MEM, 7 for WAIT_REG_MEM64.
    fn wait_address_bits(&self) -> Result<u32> {
        match (self.opcode(), self.packet.len()) {
            (pkt3::WAIT_REG_MEM, 7) => Ok(3),
            (opcode, 9) if opcode == cp::pfp::WAIT_REG_MEM64::OPCODE => Ok(7),
            _ => Err(Error::invalid("not a WaitRegMem command")),
        }
    }

    fn require_write_data(&self) -> Result<()> {
        if self.opcode() != pkt3::WRITE_DATA {
            return Err(Error::invalid("not a WRITE_DATA packet"));
        }
        Ok(())
    }

    /// Applies `patch`.
    ///
    /// # Safety
    /// The footprint is one the AMD target recorded, in a live buffer.
    pub(super) unsafe fn apply(mut self, patch: Patch) -> Result<()> {
        match patch {
            Patch::CondExecAddress(address) => {
                self.expect(pkt3::COND_EXEC, dwords::COND_EXEC, "COND_EXEC")?;
                self.packet[1] = (self.packet[1] & 3) | (address as u32 & !3);
                self.packet[2] = (address >> 32) as u32;
            }
            Patch::CondExecEnd(end) => {
                self.expect(pkt3::COND_EXEC, dwords::COND_EXEC, "COND_EXEC")?;
                // Unsigned, as the console's: an end before the packet's wraps and fails.
                let count = end.wrapping_sub(self.at as u64 + 4 * u64::from(dwords::COND_EXEC)) / 4;
                if count > 0x3fff {
                    return Err(Error::invalid("the guarded range is longer than 0x3fff dwords"));
                }
                self.packet[4] = (self.packet[4] & !0x3fff) | count as u32;
            }
            Patch::DmaDestination(address) => {
                self.expect(pkt3::DMA_DATA, dwords::DMA_DATA, "DMA_DATA")?;
                self.set_u64(4, address);
            }
            Patch::DmaSource(address) => {
                self.expect(pkt3::DMA_DATA, dwords::DMA_DATA, "DMA_DATA")?;
                self.set_u64(2, address);
            }
            Patch::JumpTarget { target, dwords: size } => {
                self.expect(pkt3::INDIRECT_BUFFER, dwords::JUMP, "jump")?;
                self.set_u64(1, target);
                self.packet[3] = (self.packet[3] & !0xf_ffff) | (size & 0xf_ffff);
            }
            Patch::WaitAddress(address) => {
                let low = self.wait_address_bits()?;
                self.packet[2] = (address as u32 & !low) | (self.packet[2] & low);
                self.packet[3] = (address >> 32) as u32;
            }
            Patch::WaitReference(reference) => match self.wait_address_bits()? {
                3 => self.packet[4] = reference as u32,
                _ => self.set_u64(4, reference),
            },
            Patch::WaitMask(mask) => match self.wait_address_bits()? {
                3 => self.packet[5] = mask,
                _ => self.packet[6] = mask,
            },
            Patch::WaitFunction(function) => {
                self.wait_address_bits()?;
                cp::pfp::WAIT_REG_MEM::FUNCTION.set(self.packet, function & 7);
            }
            Patch::WriteDataAddress(address) => {
                self.require_write_data()?;
                self.set_u64(2, address & !3);
            }
            Patch::WriteDataDestination { dst, compute } => {
                use cp::pfp::WRITE_DATA as W;
                self.require_write_data()?;
                let (select, pfp) = if compute { (dst, false) } else { (dst >> 1, dst & 1 != 0) };
                // Memory, as when recorded: guest registers and GDS have no host form.
                if !matches!(select, 1 | 2 | 5) {
                    return Err(Error::unsupported(format!("WRITE_DATA destination {select} has no host form")));
                }
                W::DST_SEL.set(self.packet, W::enums::dst_sel::MEMORY);
                if !compute {
                    let engine = if pfp { W::enums::engine_sel::PREFETCH_PARSER } else { W::enums::engine_sel::MICRO_ENGINE };
                    W::ENGINE_SEL.set(self.packet, engine);
                }
            }
            Patch::WriteDataCachePolicy => self.require_write_data()?,
            Patch::IndirectAddress { bank: _, address } => {
                self.require_register_list()?;
                // A list already settled for the GPU holds LOAD_SH_REG_INDEX, whose address is
                // dwords 1 and 2.
                if self.opcode() == pkt3::LOAD_SH_REG_INDEX {
                    self.set_u64(1, address);
                } else {
                    self.set_u64(2, address);
                }
            }
            Patch::IndirectCount { bank: _, count, add } => {
                self.require_register_list()?;
                self.packet[4] = if add { self.packet[4].wrapping_add(count) } else { count };
            }
            Patch::ReleaseAddress(address) => {
                self.require_release()?;
                self.set_u64(3, address);
            }
            Patch::ReleaseData(data) => {
                self.require_release()?;
                self.set_u64(5, data);
            }
            Patch::ReleaseCaches(gcr) => {
                use crate::amd::defs::pkt3::dwords::RELEASE_MEM_OP_gfx11 as Op;
                use cp::meg::RELEASE_MEM as R;
                self.require_release()?;
                let event = R::EVENT_TYPE.get(self.packet);
                let plan = ReleasePlan::of(event, gcr << 12).map_err(Error::unsupported)?;
                let mask = Op::GLM_WB.mask() | Op::GLM_INV.mask() | Op::GLV_INV.mask() | Op::GL1_INV.mask() | Op::GL2_INV.mask() | Op::GL2_WB.mask() | Op::SEQ.mask();
                self.packet[1] = (self.packet[1] & !mask) | plan.actions;
            }
            Patch::ReleaseEvent(event) => {
                use cp::meg::RELEASE_MEM as R;
                self.require_release()?;
                let plan = ReleasePlan::of(event, 0).map_err(Error::unsupported)?;
                R::EVENT_TYPE.set(self.packet, plan.event);
                R::EVENT_INDEX.set(self.packet, plan.index);
            }
            // The command processor does not wait on a rewind on the host: the footprint is a NOP.
            Patch::Rewind(_) => {}
            // The prefetch parser's wait is part of the graphics sequence, which the footprint names
            // only through its IB2 call: the acquire keeps the engine it was recorded with.
            Patch::AcquireEngine(_) => {}
            Patch::Nop => nop(self.packet),
            Patch::Predicate(predicate) => {
                for dword in self.work_packets() {
                    // SAFETY: inside the footprint, by the packet walk.
                    unsafe {
                        let header = *self.at.add(dword);
                        *self.at.add(dword) = (header & !1) | u32::from(predicate);
                    }
                }
            }
        }
        Ok(())
    }

    fn require_register_list(&self) -> Result<()> {
        if self.packet.len() != 5 || !matches!(self.opcode(), pkt3::NOP | pkt3::LOAD_SH_REG_INDEX) {
            return Err(Error::invalid("not a register list packet"));
        }
        Ok(())
    }

    fn require_release(&self) -> Result<()> {
        if self.opcode() != pkt3::RELEASE_MEM {
            return Err(Error::invalid("not a RELEASE_MEM packet"));
        }
        Ok(())
    }

    /// The dword offsets of the work packets of a footprint: a draw's footprint starts with its
    /// state slot and packets that set up the draw, and the predicate belongs on the draw alone.
    fn work_packets(&self) -> Vec<usize> {
        const MAX_FOOTPRINT: usize = 64;
        let mut at = 0;
        let mut found = Vec::new();
        while at < MAX_FOOTPRINT {
            // SAFETY: footprints this command recorded are at least as long as the walk finds
            // packets for; the walk stops at the first work packet.
            let header = unsafe { *self.at.add(at) };
            let Some(dwords) = packet_dwords(header) else {
                break;
            };
            if dwords > 1 && Commands::predicable(Commands::opcode(header)) {
                found.push(at);
                break;
            }
            at += dwords;
        }
        found
    }
}

/// The dword of a draw or dispatch footprint's work packet: after the state slot and the packets
/// that set up the work.
fn work_packet(footprint: &[u32]) -> Option<usize> {
    let mut at = 0;
    while at < footprint.len() {
        let dwords = packet_dwords(footprint[at])?;
        if dwords > 1 && Commands::predicable(Commands::opcode(footprint[at])) {
            return Some(at);
        }
        at += dwords;
    }
    None
}

/// Whether the title put a draw or dispatch footprint's work under the predicate.
pub(super) fn work_predicate(footprint: &[u32]) -> bool {
    work_packet(footprint).is_some_and(|at| footprint[at] & 1 != 0)
}

/// Puts a draw or dispatch footprint's work under the predicate again, after it was rewritten.
pub(super) fn set_work_predicate(footprint: &mut [u32]) {
    if let Some(at) = work_packet(footprint) {
        footprint[at] |= 1;
    }
}

/// Whole recorded commands, as the predicate walks them (`sceAgcSetRangePredication`).
pub(super) struct Commands<'a>(&'a mut [u32]);

impl<'a> Commands<'a> {
    /// The commands in `[start, end)`.
    ///
    /// # Safety
    /// The range holds whole commands of a live buffer.
    pub(super) unsafe fn between(start: *mut u32, end: *const u32) -> Result<Self> {
        if start.is_null() || end.is_null() || (start as usize) > (end as usize) || start as usize % 4 != 0 {
            return Err(Error::invalid("invalid packet range"));
        }
        let dwords = (end as usize - start as usize) / 4;
        // SAFETY: the caller's contract.
        Ok(Self(unsafe { std::slice::from_raw_parts_mut(start, dwords) }))
    }

    fn opcode(header: u32) -> u32 {
        (header >> 8) & 0xff
    }

    /// Packets the predicate applies to: work, never state or synchronization (D12 point 4).
    fn predicable(opcode: u32) -> bool {
        matches!(
            opcode,
            pkt3::DRAW_INDEX_2
                | pkt3::DRAW_INDEX_AUTO
                | pkt3::DRAW_INDEX_OFFSET_2
                | pkt3::DRAW_INDIRECT
                | pkt3::DRAW_INDEX_INDIRECT
                | pkt3::DRAW_INDIRECT_MULTI
                | pkt3::DRAW_INDEX_INDIRECT_MULTI
                | pkt3::DISPATCH_DIRECT
                | pkt3::DISPATCH_INDIRECT
                | pkt3::WRITE_DATA
                | pkt3::COPY_DATA
                | pkt3::DMA_DATA
        )
    }

    /// Sets the predicate bit of the work packets, walking as the console's function does:
    /// type-2 headers and one-dword NOPs are skipped, and a packet may not end past the range.
    pub(super) fn predicate(&mut self, predicate: bool) -> Result<()> {
        let words = &mut *self.0;
        let mut at = 0;
        while at < words.len() {
            let header = words[at];
            if header >> 30 == 2 {
                at += 1;
                continue;
            }
            let dwords = packet_dwords(header).ok_or_else(|| Error::invalid("packet is neither type 2 nor type 3"))?;
            if dwords > words.len() - at {
                return Err(Error::invalid("range ends inside a packet"));
            }
            if dwords > 1 && Self::predicable(Self::opcode(header)) {
                words[at] = (header & !1) | u32::from(predicate);
            }
            at += dwords;
        }
        Ok(())
    }
}
