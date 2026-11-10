//! Sync, memory, control and draw-parameter commands of the AMD target (PLAN P3.4 step 2c).
//!
//! Packet layouts come from AMD's CP packet tables in Mesa (`amd::defs::cp_gfx11`; the gfx12
//! table has the same fields at the same bits for every packet here), and the sequences from
//! radv: cache flushes as `ac_gfx10_emit_barrier` (`ac_barrier.c`), releases and waits as
//! `ac_cmdbuf_cp.c`. Guest addresses are host addresses (same VA).

use super::sync::{AcquirePlan, EventPlan, ReleasePlan};
use super::{nop, Generation};
use crate::agc::header;
use crate::agc::target::{DmaEnd, Op, ReleaseData};
use crate::amd::defs::{cp_gfx11 as cp, pkt3, PacketField};
use crate::gpu::packets::event;
use crate::{Error, Result};

/// The guest's occlusion results: 16 slots of 16 bytes, one per guest render backend.
pub const GUEST_OCCLUSION_SLOTS: u32 = 16;
const OCCLUSION_SLOT_BYTES: u64 = 16;
/// Render backends the host dumps counters for (`gpu::state_block::RENDER_BACKENDS`).
const HOST_RENDER_BACKENDS: u32 = crate::gpu::state_block::RENDER_BACKENDS;

/// Bytes one DMA_DATA moves on gfx11 and later (radv `cp_dma_max_byte_count`: 32767, aligned
/// down to 32).
pub const DMA_MAX_BYTES: u32 = 32767 & !31;

/// Footprint dwords of each command (D14), the largest form any queue and generation needs.
pub mod dwords {
    /// The console's 8, which the SDK reserves: `Core::RingBuffer` keeps 16 dwords at the end of
    /// a segment for an acquire and a release (D14). Graphics: an IB2 call of the cached
    /// sequence, and a NOP with the guest's parameters. Compute: an ACQUIRE_MEM.
    pub const ACQUIRE: u32 = 8;
    pub const RELEASE: u32 = 8;
    /// An IB2 call of a cached RELEASE_MEM and ACQUIRE_MEM pair, on graphics queues.
    pub const EVENT: u32 = 4;
    /// EVENT_WRITE of the host counters, then a valid zero counter per guest slot the host
    /// lacks (WRITE_DATA of 2 dwords each).
    pub const OCCLUSION: u32 = 4 + (super::GUEST_OCCLUSION_SLOTS - super::HOST_RENDER_BACKENDS) * 6;
    pub const COPY_DATA: u32 = 6;
    pub const DMA_DATA: u32 = 7;
    pub const ATOMIC_MEM: u32 = 9;
    pub const WAIT_32: u32 = 7;
    pub const WAIT_64: u32 = 9;
    pub const COND_WRITE: u32 = 9;
    /// An idle wait (RELEASE_MEM and ACQUIRE_MEM in PWS form) and ATOMIC_MEM.
    pub const SEMAPHORE: u32 = 16 + 9;
    pub const SYNC_PARSER: u32 = 2;
    pub const JUMP: u32 = 4;
    pub const COND_EXEC: u32 = 5;
    pub const INDEX_BASE: u32 = 3;
    pub const INDEX_COUNT: u32 = 2;
    pub const INDEX_TYPE: u32 = 3;
    pub const INSTANCES: u32 = 2;
    pub const INDIRECT_BASE: u32 = 4;
    pub const PREDICATION: u32 = 4;
    pub const FLIP: u32 = 7;
    pub const WAIT_FLIP_DONE: u32 = 4;
    pub const RESET_QUEUE: u32 = 3;
    pub const REWIND: u32 = 2;
    pub const LOD_STATS: u32 = 5;
}

/// The kind of host queue a recording is for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostQueue {
    Graphics,
    /// A compute queue (MEC): compute state, dispatches, and cache, fence and memory packets.
    Compute,
}

/// Packets written one after another into a footprint, which [`Packets::finish`] pads.
pub struct Packets<'a> {
    out: &'a mut [u32],
    at: usize,
}

impl<'a> Packets<'a> {
    pub fn new(out: &'a mut [u32]) -> Self {
        Self { out, at: 0 }
    }

    /// A packet of `dwords` dwords (the header included), body zeroed.
    pub fn packet(&mut self, opcode: u32, dwords: u32) -> Result<&mut [u32]> {
        let end = self.at + dwords as usize;
        if end > self.out.len() {
            return Err(Error::invalid(format!(
                "the commands need more than the {}-dword footprint",
                self.out.len()
            )));
        }
        let packet = &mut self.out[self.at..end];
        packet[0] = header(opcode, dwords, 0)?;
        packet[1..].fill(0);
        self.at = end;
        Ok(packet)
    }

    /// Dwords written so far.
    pub fn written(&self) -> usize {
        self.at
    }

    pub fn finish(self) {
        nop(&mut self.out[self.at..]);
    }

}

fn put(packet: &mut [u32], field: PacketField, value: u32) {
    field.set(packet, value);
}

fn address(packet: &mut [u32], lo: usize, address: u64) {
    packet[lo] = address as u32;
    packet[lo + 1] = (address >> 32) as u32;
}

/// GCR_CNTL actions in RELEASE_MEM_OP form (`ac_emit_cp_release_mem_pws`): GLM and GL1 do not
/// exist in gfx12's RELEASE_MEM.
fn release_op(generation: &Generation, gcr: u32) -> u32 {
    use crate::amd::defs::pkt3::dwords::{GCR_CNTL as G, RELEASE_MEM_OP_gfx11 as R};
    let mut op = 0;
    let mut copy = |from: crate::amd::defs::Field, to: crate::amd::defs::Field| op = to.set(op, from.get(gcr));
    if !generation.gfx12 {
        copy(G::GLM_WB, R::GLM_WB);
        copy(G::GLM_INV, R::GLM_INV);
        copy(G::GL1_INV, R::GL1_INV);
    }
    copy(G::GLV_INV, R::GLV_INV);
    copy(G::GL2_INV, R::GL2_INV);
    copy(G::GL2_WB, R::GL2_WB);
    copy(G::SEQ, R::SEQ);
    copy(G::GLK_WB, R::GLK_WB);
    op
}

/// GCR_CNTL as the host generation takes it: gfx12 has no GL1 and no GLM actions.
fn host_gcr(generation: &Generation, gcr: u32) -> u32 {
    use crate::amd::defs::pkt3::dwords::GCR_CNTL as G;
    if !generation.gfx12 {
        return gcr;
    }
    gcr & !(G::GL1_INV.mask() | G::GL1_RANGE.mask() | G::GLM_WB.mask() | G::GLM_INV.mask())
}

/// Whether `gcr` asks for any cache action (the range and order fields only qualify others).
fn has_actions(generation: &Generation, gcr: u32) -> bool {
    use crate::amd::defs::pkt3::dwords::GCR_CNTL as G;
    let qualifiers = G::GL2_RANGE.mask() | G::SEQ.mask() | if generation.gfx12 { 0 } else { G::GL1_RANGE.mask() };
    gcr & !qualifiers != 0
}

fn event_write(packets: &mut Packets, event_type: u32, index: u32) -> Result<()> {
    use cp::pfp::EVENT_WRITE as E;
    let packet = packets.packet(E::OPCODE, 2)?;
    put(packet, E::EVENT_TYPE, event_type);
    put(packet, E::EVENT_INDEX, index);
    Ok(())
}

/// RELEASE_MEM of a timestamp event in PWS form, carrying `gcr`'s actions but GLI_INV.
fn release_mem_pws(packets: &mut Packets, generation: &Generation, event_type: u32, gcr: u32) -> Result<()> {
    use cp::meg::RELEASE_MEM as R;
    let packet = packets.packet(R::OPCODE, R::WORDS)?;
    put(packet, R::EVENT_TYPE, event_type);
    put(packet, R::EVENT_INDEX, R::enums::event_index::END_OF_PIPE);
    packet[1] |= release_op(generation, gcr);
    put(packet, R::GLK_INV, crate::amd::defs::pkt3::dwords::GCR_CNTL::GLK_INV.get(gcr));
    put(packet, R::PWS_ENABLE, 1);
    Ok(())
}

/// ACQUIRE_MEM in PWS form: wait at `stage` for the last timestamp event, then `gcr`.
fn acquire_mem_pws(packets: &mut Packets, stage: u32, gcr: u32) -> Result<()> {
    use cp::pfp::ACQUIRE_MEM as A;
    let packet = packets.packet(A::OPCODE, A::WORDS)?;
    put(packet, A::PWS_STAGE_SEL, stage);
    put(packet, A::PWS_COUNTER_SEL, A::enums::pws_counter_sel::TS_SELECT);
    put(packet, A::PWS_ENA2, 1);
    packet[2] = 0xffff_ffff;
    packet[3] = 0x01ff_ffff;
    put(packet, A::PWS_ENA, 1);
    packet[7] = gcr;
    Ok(())
}

/// Everything before waits until idle: a timestamp released and waited for on graphics queues,
/// CS_PARTIAL_FLUSH on compute queues (they run nothing else).
fn idle(packets: &mut Packets, generation: &Generation, queue: HostQueue) -> Result<()> {
    match queue {
        HostQueue::Graphics => {
            release_mem_pws(packets, generation, event::BOTTOM_OF_PIPE_TS, 0)?;
            acquire_mem_pws(packets, cp::pfp::ACQUIRE_MEM::enums::pws_stage_sel::CP_ME, 0)
        }
        HostQueue::Compute => event_write(packets, event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH),
    }
}

/// A guest ACQUIRE_MEM as radv's barrier for gfx11 and gfx12 (`ac_gfx10_emit_barrier`).
pub(super) fn acquire(packets: &mut Packets, generation: &Generation, queue: HostQueue, pfp: bool, cb_db: u32, gcr: u32) -> Result<()> {
    use crate::amd::defs::pkt3::dwords::GCR_CNTL as G;
    use cp::pfp::ACQUIRE_MEM as A;
    let plan = AcquirePlan::of(cb_db, gcr);
    let mut gcr = host_gcr(generation, plan.gcr_cntl);
    if plan.flush_cb || plan.flush_db {
        if queue == HostQueue::Compute {
            // No render backends: what the flush waits for is the queue's own shaders. The eight
            // dwords of the footprint hold the partial flush or the cache actions, not both.
            if !has_actions(generation, gcr) {
                event_write(packets, event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
                return Ok(());
            }
        } else {
            let flush = if (plan.flush_cb && plan.flush_db) || (!generation.gfx12 && plan.flush_db) {
                // gfx11 cannot flush DB_META alone (ac_barrier.c).
                event::CACHE_FLUSH_AND_INV_TS_EVENT
            } else if plan.flush_cb {
                event::FLUSH_AND_INV_CB_DATA_TS
            } else {
                event::FLUSH_AND_INV_DB_DATA_TS
            };
            if !generation.gfx12 && flush == event::FLUSH_AND_INV_CB_DATA_TS {
                event_write(packets, event::FLUSH_AND_INV_CB_META, 0)?;
            }
            gcr = G::SEQ.set(gcr, 1);
            release_mem_pws(packets, generation, flush, gcr & !G::GLI_INV.mask())?;
            let stage = if pfp { A::enums::pws_stage_sel::CP_PFP } else { A::enums::pws_stage_sel::CP_ME };
            acquire_mem_pws(packets, stage, gcr & G::GLI_INV.mask())?;
            return Ok(());
        }
    }
    if has_actions(generation, gcr) {
        let packet = packets.packet(A::OPCODE, A::WORDS)?;
        let graphics = queue == HostQueue::Graphics;
        if graphics {
            let engine = if pfp { A::enums::engine_sel::PREFETCH_PARSER } else { A::enums::engine_sel::MICRO_ENGINE };
            put(packet, A::ENGINE_SEL, engine);
        }
        packet[2] = 0xffff_ffff;
        packet[3] = if graphics { 0x00ff_ffff } else { 0xff };
        packet[6] = 0xa;
        packet[7] = gcr;
    } else if pfp && queue == HostQueue::Graphics {
        packets.packet(pkt3::PFP_SYNC_ME, 2)?;
    }
    Ok(())
}

/// A guest release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestRelease {
    /// The guest's end-of-pipe event.
    pub event: u32,
    /// The guest's cache actions (RELEASE_MEM_OP bits 23:12).
    pub gcr: u32,
    /// The guest's data select: 0 none, 1 32 bits, 2 64 bits, 3 the GPU clock.
    pub select: u32,
    pub address: u64,
    pub data: u64,
}

impl GuestRelease {
    /// The guest's data as a select and a value.
    pub fn data(data: ReleaseData) -> Result<(u32, u64)> {
        Ok(match data {
            ReleaseData::None => (0, 0),
            ReleaseData::Value32(value) => (1, u64::from(value)),
            ReleaseData::Value64(value) => (2, value),
            ReleaseData::Timestamp => (3, 0),
            ReleaseData::Gds { .. } => return Err(Error::unsupported("a GDS release needs a host GDS buffer")),
        })
    }

    /// The RELEASE_MEM: the event and cache actions in host form (`sync::release_event`) as the
    /// queue runs them, and the data written once the actions confirm; nothing is written while
    /// the address is null.
    pub(super) fn release_mem(&self, packets: &mut Packets, generation: &Generation, queue: HostQueue) -> Result<()> {
        use cp::meg::RELEASE_MEM as R;
        let plan = ReleasePlan::of(self.event, self.gcr << 12).map_err(Error::unsupported)?;
        let (mut event_type, mut index, mut actions) = (plan.event, plan.index, plan.actions);
        if generation.gfx12 {
            use crate::amd::defs::pkt3::dwords::RELEASE_MEM_OP_gfx11 as Op;
            actions &= !(Op::GLM_WB.mask() | Op::GLM_INV.mask() | Op::GL1_INV.mask());
        }
        if queue == HostQueue::Compute {
            if index == event::INDEX_END_OF_PIPE {
                event_type = event::BOTTOM_OF_PIPE_TS;
            } else if event_type == event::PS_DONE {
                event_type = event::CS_DONE;
            }
            index = if event_type == event::CS_DONE { event::INDEX_SHADER_DONE } else { index };
        }
        let select = match self.select {
            0 => R::enums::data_sel::NONE,
            1 => R::enums::data_sel::SEND_32_BIT_LOW,
            2 => R::enums::data_sel::SEND_64_BIT_DATA,
            _ => R::enums::data_sel::SEND_GPU_CLOCK_COUNTER,
        };
        let packet = packets.packet(R::OPCODE, R::WORDS)?;
        put(packet, R::EVENT_TYPE, event_type);
        put(packet, R::EVENT_INDEX, index);
        packet[1] |= actions;
        put(packet, R::DATA_SEL, select);
        if select != R::enums::data_sel::NONE {
            put(packet, R::INT_SEL, R::enums::int_sel::SEND_DATA_AND_WRITE_CONFIRM);
        }
        address(packet, 3, self.address);
        address(packet, 5, self.data);
        Ok(())
    }

}

/// A guest EVENT_WRITE in host form (`sync::event_write`), or an occlusion dump.
pub(super) fn event(packets: &mut Packets, generation: &Generation, queue: HostQueue, event_type: u32, target: u64) -> Result<()> {
    if event_type == event::PIXEL_PIPE_STAT_DUMP && target != 0 {
        return occlusion_dump(packets, queue, target);
    }
    match EventPlan::of(event_type) {
        EventPlan::Forward { event, .. } if queue == HostQueue::Compute && event != event::CS_PARTIAL_FLUSH => Ok(()),
        EventPlan::Forward { event, index } => event_write(packets, event, index),
        EventPlan::TimestampFlush { .. } if queue == HostQueue::Compute => Ok(()),
        EventPlan::TimestampFlush { event } => {
            release_mem_pws(packets, generation, event, 0)?;
            acquire_mem_pws(packets, cp::pfp::ACQUIRE_MEM::enums::pws_stage_sel::CP_ME, 0)
        }
        EventPlan::Unsupported(reason) => Err(Error::unsupported(reason)),
    }
}

/// RELEASE_MEM writing the GPU clock to `target` once the work before it has finished: radv's
/// `radv_write_timestamp` at the bottom of the pipe.
pub(super) fn release_timestamp(packets: &mut Packets, target: u64) -> Result<()> {
    use cp::meg::RELEASE_MEM as R;
    let packet = packets.packet(R::OPCODE, R::WORDS)?;
    put(packet, R::EVENT_TYPE, event::BOTTOM_OF_PIPE_TS);
    put(packet, R::EVENT_INDEX, R::enums::event_index::END_OF_PIPE);
    put(packet, R::DATA_SEL, R::enums::data_sel::SEND_GPU_CLOCK_COUNTER);
    put(packet, R::INT_SEL, R::enums::int_sel::SEND_DATA_AND_WRITE_CONFIRM);
    address(packet, 3, target);
    Ok(())
}

/// A guest occlusion dump: one 64-bit counter per host render backend at the guest's 16-byte
/// stride (PIXEL_PIPE_STAT_DUMP, as radv), and valid zero counters in the guest slots the host
/// lacks, so the guest's sum over its slots is the host total. The queue's preamble enables
/// the counters (PIXEL_PIPE_STAT_CONTROL).
fn occlusion_dump(packets: &mut Packets, queue: HostQueue, target: u64) -> Result<()> {
    use cp::pfp::{EVENT_WRITE as E, WRITE_DATA as W};
    if queue == HostQueue::Compute {
        return Err(Error::invalid("occlusion dumps need a graphics queue"));
    }
    let packet = packets.packet(E::OPCODE, 4)?;
    put(packet, E::EVENT_TYPE, event::PIXEL_PIPE_STAT_DUMP);
    put(packet, E::EVENT_INDEX, E::enums::event_index::PIXEL_PIPE_STAT_CONTROL_OR_DUMP);
    address(packet, 2, target);
    for slot in HOST_RENDER_BACKENDS..GUEST_OCCLUSION_SLOTS {
        let packet = packets.packet(W::OPCODE, 6)?;
        put(packet, W::DST_SEL, W::enums::dst_sel::MEMORY);
        put(packet, W::WR_CONFIRM, 1);
        address(packet, 2, target + u64::from(slot) * OCCLUSION_SLOT_BYTES);
        packet[5] = 1 << 31;
    }
    Ok(())
}

/// A guest memory select (WRITE_DATA, COPY_DATA) as the host's: memory through L2.
fn memory_select(select: u32) -> Result<u32> {
    match select {
        1 | 2 | 5 => Ok(cp::pfp::COPY_DATA::enums::src_sel::TC_L2),
        0 => Err(Error::unsupported("guest register offsets are not host registers")),
        3 => Err(Error::unsupported("GDS needs a host GDS buffer")),
        other => Err(Error::unsupported(format!("memory select {other} has no host equivalent"))),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_data(packets: &mut Packets, queue: HostQueue, dst: u32, pfp: bool, target: u64, values: &[u32], repeat: bool, confirm: bool) -> Result<()> {
    use cp::pfp::WRITE_DATA as W;
    memory_select(dst)?;
    let packet = packets.packet(W::OPCODE, 4 + values.len() as u32)?;
    put(packet, W::DST_SEL, W::enums::dst_sel::MEMORY);
    put(packet, W::ADDR_INCR, u32::from(repeat));
    put(packet, W::WR_CONFIRM, u32::from(confirm));
    if queue == HostQueue::Graphics && pfp {
        put(packet, W::ENGINE_SEL, W::enums::engine_sel::PREFETCH_PARSER);
    }
    address(packet, 2, target);
    packet[4..].copy_from_slice(values);
    Ok(())
}

/// The guest's GPU clock source of COPY_DATA.
const COPY_GPU_CLOCK: u32 = 9;
const COPY_IMMEDIATE: u32 = 5;

#[allow(clippy::too_many_arguments)]
fn copy_data(packets: &mut Packets, queue: HostQueue, src: u32, dst: u32, pfp: bool, source: u64, destination: u64, item64: bool, confirm: bool) -> Result<()> {
    use cp::pfp::COPY_DATA as C;
    memory_select(dst)?;
    // The clock is the guest's source number on the host too (`COPY_DATA_TIMESTAMP` in radv). The
    // micro engine reads it: the gfx11 table has it reserved on the prefetch parser.
    let clock = src == COPY_GPU_CLOCK;
    let host_src = match src {
        COPY_GPU_CLOCK => COPY_GPU_CLOCK,
        COPY_IMMEDIATE => C::enums::src_sel::IMMEDIATE_DATA,
        _ => memory_select(src)?,
    };
    let packet = packets.packet(C::OPCODE, 6)?;
    put(packet, C::SRC_SEL, host_src);
    put(packet, C::DST_SEL, C::enums::dst_sel::TC_L2);
    put(packet, C::COUNT_SEL, u32::from(item64));
    put(packet, C::WR_CONFIRM, u32::from(confirm));
    if queue == HostQueue::Graphics && pfp && !clock {
        put(packet, C::ENGINE_SEL, C::enums::engine_sel::PREFETCH_PARSER);
    }
    address(packet, 2, if clock { 0 } else { source });
    address(packet, 4, destination);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dma_data(packets: &mut Packets, queue: HostQueue, pfp: bool, source: DmaEnd, destination: DmaEnd, bytes: u32, wait_previous: bool) -> Result<()> {
    use cp::meg::DMA_DATA as D;
    let target = match destination {
        DmaEnd::Nowhere => return Ok(()),
        DmaEnd::Memory(address) => address,
        DmaEnd::Gds(_) => {
            if let Some(flow) = super::flow::diagnostics() {
                flow.rejected_gds_dma(false);
            }
            return Err(Error::unsupported("DMA_DATA into GDS needs a host GDS buffer"));
        }
        DmaEnd::Data(_) => return Err(Error::invalid("a DMA cannot write into an immediate")),
    };
    let (select, from) = match source {
        DmaEnd::Memory(address) => (D::enums::src_sel::SRC_ADDR_USING_L2, address),
        DmaEnd::Data(pattern) => (D::enums::src_sel::DATA, u64::from(pattern)),
        DmaEnd::Gds(_) => {
            if let Some(flow) = super::flow::diagnostics() {
                flow.rejected_gds_dma(true);
            }
            return Err(Error::unsupported("DMA_DATA from GDS needs a host GDS buffer"));
        }
        DmaEnd::Nowhere => return Err(Error::invalid("a DMA needs a source")),
    };
    if bytes > DMA_MAX_BYTES {
        return Err(Error::unsupported(format!(
            "a DMA of {bytes} bytes needs more than one host DMA_DATA (at most {DMA_MAX_BYTES} bytes each); the split goes into target memory with the draw path"
        )));
    }
    let packet = packets.packet(D::OPCODE, D::WORDS)?;
    // Through L2 at both ends, as radv's CP DMA from gfx9 on: a write that bypassed L2 stays
    // invisible to a WAIT_REG_MEM on another queue polling a line L2 already holds.
    put(packet, D::SRC_SEL, select);
    put(packet, D::DST_SEL, D::enums::dst_sel::DST_ADDR_USING_L2);
    // Only PFP and ME have it; MEC always syncs (radv).
    if queue == HostQueue::Graphics {
        put(packet, D::CP_SYNC, 1);
    }
    if queue == HostQueue::Graphics && pfp {
        put(packet, D::ENGINE_SEL, 1);
    }
    address(packet, 2, from);
    address(packet, 4, target);
    put(packet, D::BYTE_COUNT, bytes);
    put(packet, D::RAW_WAIT, u32::from(wait_previous));
    Ok(())
}

fn atomic_mem(packets: &mut Packets, op: u32, command: u32, target: u64, source: u64, compare: u64, interval: u32) -> Result<()> {
    use cp::meg::ATOMIC_MEM as A;
    let packet = packets.packet(A::OPCODE, A::WORDS)?;
    put(packet, A::ATOMIC, op);
    put(packet, A::COMMAND, command);
    address(packet, 2, target);
    address(packet, 4, source);
    address(packet, 6, compare);
    put(packet, A::LOOP_INTERVAL, interval);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn wait(packets: &mut Packets, queue: HostQueue, bits64: bool, function: u32, pfp: bool, target: u64, reference: u64, mask: u64, poll: u32) -> Result<()> {
    use cp::pfp::{WAIT_REG_MEM as W, WAIT_REG_MEM64 as W64};
    // radv waits on 32 bits on every queue; a compute queue has no 64-bit precedent. A reference
    // that fits compares the low dword (the footprint keeps its size, padded).
    let bits64 = bits64 && !(queue == HostQueue::Compute && reference >> 32 == 0);
    let packet = if bits64 { packets.packet(W64::OPCODE, W64::WORDS)? } else { packets.packet(W::OPCODE, W::WORDS)? };
    put(packet, W::FUNCTION, function);
    put(packet, W::MEM_SPACE, W::enums::mem_space::MEMORY_SPACE);
    if queue == HostQueue::Graphics && pfp {
        put(packet, W::ENGINE_SEL, W::enums::engine_sel::PREFETCH_PARSER);
    }
    // A compute queue waiting in its IB must not keep the others off the MEC: radv's gang
    // followers wait the same way.
    if queue == HostQueue::Compute {
        put(packet, W::OPERATION, W::enums::operation::WAIT_MEM_PREEMPTABLE);
    }
    address(packet, 2, target);
    if bits64 {
        address(packet, 4, reference);
        address(packet, 6, mask);
        put(packet, W64::POLL_INTERVAL, poll);
    } else {
        packet[4] = reference as u32;
        packet[5] = mask as u32;
        put(packet, W::POLL_INTERVAL, poll);
    }
    Ok(())
}

fn cond_write(packets: &mut Packets, function: u32, poll: u64, reference: u32, mask: u32, target: u64, value: u32) -> Result<()> {
    use cp::meg::COND_WRITE as C;
    let packet = packets.packet(pkt3::COND_WRITE, C::WORDS)?;
    put(packet, C::FUNCTION, function);
    put(packet, C::POLL_SPACE, C::enums::poll_space::MEMORY);
    put(packet, C::WRITE_SPACE, C::enums::write_space::MEMORY);
    address(packet, 2, poll);
    packet[4] = reference;
    packet[5] = mask;
    address(packet, 6, target);
    packet[8] = value;
    Ok(())
}

/// GL2 atomics without a return value (ATOMIC_MEM `atomic`, gfx11 and gfx12).
const ATOMIC_ADD_64: u32 = 111;
const ATOMIC_SUB_64: u32 = 112;

/// A guest MEM_SEMAPHORE: gfx11 has none, so a signal writes or adds 1 once the work before it is
/// done and a wait polls for non-zero and takes 1 (not atomic against a second waiter, as the
/// PM4 path).
fn semaphore(packets: &mut Packets, generation: &Generation, queue: HostQueue, target: u64, signal: bool, set_one: bool) -> Result<()> {
    if signal && set_one {
        let set = GuestRelease { event: event::BOTTOM_OF_PIPE_TS, gcr: 0, select: 2, address: target, data: 1 };
        return set.release_mem(packets, generation, queue);
    }
    if signal {
        idle(packets, generation, queue)?;
        return atomic_mem(packets, ATOMIC_ADD_64, 0, target, 1, 0, 10);
    }
    wait(packets, queue, true, cp::pfp::WAIT_REG_MEM64::enums::function::NOT_EQUAL_REFERENCE_VALUE, false, target, 0, u64::MAX, 10)?;
    atomic_mem(packets, ATOMIC_SUB_64, 0, target, 1, 0, 10)
}

fn jump(packets: &mut Packets, queue: HostQueue, target: u64, dwords: u32, chain: bool) -> Result<()> {
    use cp::mec::INDIRECT_BUFFER as M;
    use cp::pfp::INDIRECT_BUFFER as I;
    let packet = packets.packet(I::OPCODE, 4)?;
    address(packet, 1, target);
    put(packet, I::IB_SIZE, dwords);
    put(packet, I::CHAIN, u32::from(chain));
    // radv's chains are VALID on every queue (`radv_amdgpu_cs_chain`), its IB2 calls are not
    // (`ac_emit_cp_indirect_buffer` without flags); the MEC takes only VALID IBs.
    if chain || queue == HostQueue::Compute {
        put(packet, M::VALID, 1);
    }
    Ok(())
}

/// Writes `op` (one of this module's) into `out` for a recording on `queue`.
pub fn encode(generation: &Generation, queue: HostQueue, op: &Op<'_>, out: &mut [u32]) -> Result<()> {
    let mut packets = Packets::new(out);
    match *op {
        Op::Marker { .. } => {}
        Op::Event { event: event_type, address } => event(&mut packets, generation, queue, event_type, address)?,
        Op::WriteData { dst, pfp, address, values, increment, confirm } => {
            write_data(&mut packets, queue, dst, pfp, address, values, !increment, confirm)?
        }
        Op::CopyData { src, dst, pfp, source, destination, item64, confirm } => {
            copy_data(&mut packets, queue, src, dst, pfp, source, destination, item64, confirm)?
        }
        Op::DmaData { pfp, source, destination, bytes, wait_previous, .. } => {
            dma_data(&mut packets, queue, pfp, source, destination, bytes, wait_previous)?
        }
        Op::AtomicMem { op, command, address, source, compare, interval } => {
            atomic_mem(&mut packets, op, command, address, source, compare, interval)?
        }
        Op::Wait { bits64, function, pfp, address, reference, mask, poll } => {
            wait(&mut packets, queue, bits64, function, pfp, address, reference, mask, poll)?
        }
        Op::CondWrite { function, poll, reference, mask, address, value } => {
            cond_write(&mut packets, function, poll, reference, mask, address, value)?
        }
        Op::Semaphore { address, signal, set_one } => semaphore(&mut packets, generation, queue, address, signal, set_one)?,
        Op::SyncParser => {
            if queue == HostQueue::Graphics {
                packets.packet(pkt3::PFP_SYNC_ME, 2)?;
            }
        }
        Op::Jump { target, dwords, chain } => jump(&mut packets, queue, target, dwords, chain)?,
        Op::CondExec { address: target, dwords } => {
            use cp::pfp::COND_EXEC as C;
            let packet = packets.packet(C::OPCODE, C::WORDS)?;
            address(packet, 1, target);
            put(packet, C::EXEC_COUNT, dwords);
        }
        Op::IndexBase(target) => {
            let packet = packets.packet(pkt3::INDEX_BASE, 3)?;
            address(packet, 1, target);
        }
        Op::IndexCount(count) => packets.packet(pkt3::INDEX_BUFFER_SIZE, 2)?[1] = count,
        Op::IndexType(size) => {
            use cp::pfp::SET_UCONFIG_REG_INDEX as S;
            let packet = packets.packet(S::OPCODE, 3)?;
            put(packet, S::REG_OFFSET, (crate::amd::defs::gfx115::VGT_INDEX_TYPE::OFFSET - 0x30000) / 4);
            put(packet, S::INDEX, S::enums::index::INDEX_TYPE);
            packet[2] = crate::amd::defs::gfx115::VGT_INDEX_TYPE::INDEX_TYPE.put(size);
        }
        Op::Instances(count) => packets.packet(pkt3::NUM_INSTANCES, 2)?[1] = count,
        Op::IndirectBase { dispatch, address: target } => {
            use cp::pfp::SET_BASE as S;
            let packet = packets.packet(S::OPCODE, 4)?;
            if dispatch {
                packet[0] |= 1 << 1; // SHADER_TYPE: compute
            }
            put(packet, S::BASE_INDEX, 1);
            address(packet, 2, target);
        }
        Op::Predication { operation, condition, wait: hint, address: target } => {
            use cp::pfp::SET_PREDICATION as P;
            let packet = packets.packet(P::OPCODE, 4)?;
            put(packet, P::PRED_OP, operation);
            put(packet, P::PRED_BOOL, u32::from(condition));
            // Occlusion predicates wait on their results; radv writes no hint for the others.
            put(packet, P::HINT, u32::from(hint && matches!(operation, 1 | 2)));
            if operation != 0 {
                address(packet, 2, target);
            }
        }
        _ => return Err(Error::invalid("not a command of `gpu::agc::commands`")),
    }
    packets.finish();
    Ok(())
}
