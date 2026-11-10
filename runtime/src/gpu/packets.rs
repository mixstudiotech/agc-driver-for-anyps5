//! Host PM4 the driver builds for itself, in a growing buffer: queue preambles and presentation
//! passes. The packets follow radv and come from the definitions generated from Mesa (decisions
//! D17).

use super::regs::HostRegister;
use crate::agc::header;
use crate::amd::defs::{cp_gfx11 as cp, pkt3};
use crate::ir::RegisterBank;
use crate::{Error, Result};

/// PKT3_SHADER_TYPE (header bit 1): the packet belongs to the compute pipeline.
pub const SHADER_TYPE_COMPUTE: u32 = 1 << 1;

/// VGT_EVENT_TYPE values and EVENT_INDEX selects.
pub mod event {
    pub use crate::amd::defs::gfx115::enums::VGT_EVENT_TYPE::{
        BOTTOM_OF_PIPE_TS, CACHE_FLUSH_AND_INV_TS_EVENT, CS_DONE, CS_PARTIAL_FLUSH, FLUSH_AND_INV_CB_DATA_TS,
        FLUSH_AND_INV_CB_META, FLUSH_AND_INV_DB_DATA_TS, PIPELINESTAT_START, PIXEL_PIPE_STAT_CONTROL,
        PIXEL_PIPE_STAT_DUMP, PS_DONE, PS_PARTIAL_FLUSH,
    };

    /// EVENT_WRITE of a partial flush, as radv writes it.
    pub const INDEX_PARTIAL_FLUSH: u32 = 4;
    pub const INDEX_END_OF_PIPE: u32 = 5;
    pub const INDEX_SHADER_DONE: u32 = 6;
}

/// ACQUIRE_MEM GCR_CNTL actions.
pub mod gcr {
    use crate::amd::defs::pkt3::dwords::GCR_CNTL as G;

    pub const GLM_WB: u32 = G::GLM_WB.mask();
    pub const GLK_INV: u32 = G::GLK_INV.mask();
    pub const GLV_INV: u32 = G::GLV_INV.mask();
    pub const GL1_INV: u32 = G::GL1_INV.mask();
    pub const GL2_INV: u32 = G::GL2_INV.mask();
    pub const GL2_WB: u32 = G::GL2_WB.mask();
    /// 0xC3B1: the full invalidate radv issues after CPU uploads.
    pub const INVALIDATE_ALL: u32 =
        G::GLI_INV.put(1) | GLM_WB | G::GLM_INV.mask() | GLK_INV | GLV_INV | GL1_INV | GL2_INV | GL2_WB;

    /// RELEASE_MEM_OP cache actions.
    use crate::amd::defs::pkt3::dwords::RELEASE_MEM_OP_gfx11 as R;
    pub const RELEASE_GLM_WB: u32 = R::GLM_WB.mask();
    pub const RELEASE_GLM_INV: u32 = R::GLM_INV.mask();
    pub const RELEASE_SEQ_FORWARD: u32 = R::SEQ.put(1);
}

/// A stream of host packets.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PacketStream {
    words: Vec<u32>,
    /// Runs on a compute queue (MEC), whose packets lack some of the graphics fields.
    compute: bool,
}

impl PacketStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// A stream for a compute queue.
    pub fn new_compute() -> Self {
        Self { compute: true, ..Self::default() }
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn into_words(self) -> Vec<u32> {
        self.words
    }

    /// A packet of `dwords` dwords (the header included), its body zeroed.
    fn packet(&mut self, opcode: u32, dwords: u32) -> Result<&mut [u32]> {
        let at = self.words.len();
        self.words.resize(at + dwords as usize, 0);
        self.words[at] = header(opcode, dwords, 0)?;
        Ok(&mut self.words[at..])
    }

    /// A packet the caller built: `flags` go in the low bits of its header.
    pub fn raw_packet(&mut self, opcode: u32, flags: u32, body: &[u32]) -> Result<()> {
        self.words.push(header(opcode, body.len() as u32 + 1, flags)?);
        self.words.extend_from_slice(body);
        Ok(())
    }

    /// One SET_*_REG packet writing consecutive registers from `dword`.
    pub fn set_registers(&mut self, bank: RegisterBank, dword: u32, values: &[u32]) -> Result<()> {
        if values.is_empty() || dword.checked_add(values.len() as u32).is_none_or(|end| end > bank.size()) {
            return Err(Error::invalid(format!("register run {bank:?} 0x{dword:x}+{} is out of range", values.len())));
        }
        let mut body = Vec::with_capacity(values.len() + 1);
        body.push(dword);
        body.extend_from_slice(values);
        self.raw_packet(bank.set_opcode(), 0, &body)
    }

    /// Writes element 0 of `register`.
    pub fn set(&mut self, register: HostRegister, value: u32) -> Result<()> {
        self.set_registers(register.bank, register.dword, &[value])
    }

    /// Writes consecutive registers starting at `first`.
    pub fn set_run(&mut self, first: HostRegister, values: &[u32]) -> Result<()> {
        self.set_registers(first.bank, first.dword, values)
    }

    /// CONTEXT_CONTROL with only the update-enable bits set: no register loading or shadowing.
    pub fn context_control_no_shadowing(&mut self) -> Result<()> {
        self.raw_packet(pkt3::CONTEXT_CONTROL, 0, &[0x8000_0000, 0x8000_0000])
    }

    /// EVENT_WRITE without a payload.
    pub fn event_write(&mut self, event_type: u32, index: u32) -> Result<()> {
        use cp::pfp::EVENT_WRITE as E;
        let packet = self.packet(E::OPCODE, 2)?;
        E::EVENT_TYPE.set(packet, event_type);
        E::EVENT_INDEX.set(packet, index);
        Ok(())
    }

    /// EVENT_WRITE PIXEL_PIPE_STAT_CONTROL: occlusion counters at a 128-bit stride per render
    /// backend, enabled for `render_backends` of them.
    pub fn pixel_pipe_stat_control(&mut self, render_backends: u32) -> Result<()> {
        use cp::pfp::EVENT_WRITE as E;
        const STRIDE_128_BITS: u32 = 2 << 9;
        const INSTANCE_ENABLE_SHIFT: u32 = 11;
        let instances = if render_backends >= 64 { u64::MAX } else { (1u64 << render_backends) - 1 };
        let packet = self.packet(E::OPCODE, 4)?;
        E::EVENT_TYPE.set(packet, event::PIXEL_PIPE_STAT_CONTROL);
        E::EVENT_INDEX.set(packet, E::enums::event_index::PIXEL_PIPE_STAT_CONTROL_OR_DUMP);
        packet[2] = STRIDE_128_BITS | (instances << INSTANCE_ENABLE_SHIFT) as u32;
        packet[3] = (instances >> (32 - INSTANCE_ENABLE_SHIFT)) as u32;
        Ok(())
    }

    /// RELEASE_MEM in pixel-wait-sync form: bumps the BOTTOM_OF_PIPE timestamp counter without
    /// writing memory, for [`Self::acquire_mem_pws_wait`] to wait on.
    pub fn release_mem_pws_bottom_of_pipe(&mut self) -> Result<()> {
        self.release_mem_pws(event::BOTTOM_OF_PIPE_TS, 0)
    }

    /// RELEASE_MEM of `event_type` in PWS form with cache `actions` (RELEASE_MEM_OP bits).
    pub fn release_mem_pws(&mut self, event_type: u32, actions: u32) -> Result<()> {
        use cp::meg::RELEASE_MEM as R;
        let packet = self.packet(R::OPCODE, R::WORDS)?;
        R::EVENT_TYPE.set(packet, event_type);
        R::EVENT_INDEX.set(packet, R::enums::event_index::END_OF_PIPE);
        packet[1] |= actions;
        R::PWS_ENABLE.set(packet, 1);
        Ok(())
    }

    /// RELEASE_MEM writing the GPU clock to `target` once the work before it has finished: radv's
    /// `radv_write_timestamp` at the bottom of the pipe.
    pub fn release_timestamp(&mut self, target: u64) -> Result<()> {
        use cp::meg::RELEASE_MEM as R;
        let packet = self.packet(R::OPCODE, R::WORDS)?;
        R::EVENT_TYPE.set(packet, event::BOTTOM_OF_PIPE_TS);
        R::EVENT_INDEX.set(packet, R::enums::event_index::END_OF_PIPE);
        R::DATA_SEL.set(packet, R::enums::data_sel::SEND_GPU_CLOCK_COUNTER);
        R::INT_SEL.set(packet, R::enums::int_sel::SEND_DATA_AND_WRITE_CONFIRM);
        packet[3] = target as u32;
        packet[4] = (target >> 32) as u32;
        Ok(())
    }

    /// COPY_DATA of the GPU clock to `target` as the micro engine reaches it: radv's
    /// `radv_write_timestamp` at the top of the pipe.
    pub fn copy_timestamp(&mut self, target: u64) -> Result<()> {
        use cp::pfp::COPY_DATA as C;
        /// `COPY_DATA_TIMESTAMP`.
        const SOURCE_TIMESTAMP: u32 = 9;
        let packet = self.packet(C::OPCODE, 6)?;
        C::SRC_SEL.set(packet, SOURCE_TIMESTAMP);
        C::DST_SEL.set(packet, C::enums::dst_sel::TC_L2);
        C::COUNT_SEL.set(packet, 1);
        C::WR_CONFIRM.set(packet, 1);
        packet[4] = target as u32;
        packet[5] = (target >> 32) as u32;
        Ok(())
    }

    /// ACQUIRE_MEM in pixel-wait-sync form: the micro engine waits until every timestamp event
    /// released so far has retired.
    pub fn acquire_mem_pws_wait(&mut self) -> Result<()> {
        use cp::pfp::ACQUIRE_MEM as A;
        let packet = self.packet(A::OPCODE, A::WORDS)?;
        A::PWS_STAGE_SEL.set(packet, A::enums::pws_stage_sel::CP_ME);
        A::PWS_COUNTER_SEL.set(packet, A::enums::pws_counter_sel::TS_SELECT);
        A::PWS_ENA2.set(packet, 1);
        packet[2] = 0xffff_ffff;
        packet[3] = 0x01ff_ffff;
        A::PWS_ENA.set(packet, 1);
        Ok(())
    }

    /// ACQUIRE_MEM over the whole address space with the given GCR_CNTL actions (radv's
    /// `ac_emit_cp_acquire_mem`; COHER_SIZE_HI is 8 bits on MEC and 24 on graphics).
    pub fn acquire_mem_global(&mut self, gcr_cntl: u32) -> Result<()> {
        use cp::pfp::ACQUIRE_MEM as A;
        let size_high = if self.compute { 0xff } else { 0x00ff_ffff };
        let packet = self.packet(A::OPCODE, A::WORDS)?;
        packet[2] = 0xffff_ffff;
        packet[3] = size_high;
        packet[6] = 0x0a;
        packet[7] = gcr_cntl;
        Ok(())
    }
}
