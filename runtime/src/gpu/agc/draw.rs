//! Draws and dispatches of the AMD target (PLAN P3.4 step 2e; decisions D12).
//!
//! A draw's footprint is room for its state, then the draw's own packets: the primitive and index
//! types, the index buffer, the instance count, the indirect-argument base and the draw packet.

use super::commands::{HostQueue, Packets};
use super::Generation;
use crate::agc::target::{Draw, DrawModifier};
use crate::amd::defs::{cp_gfx11 as cp, pkt3};
use crate::ir::{QueueState, RegisterBank};
use crate::{Error, Result};

/// The SET_UCONFIG_REG_INDEX index of VGT_PRIMITIVE_TYPE, as radv writes it.
const SET_UCONFIG_INDEX_PRIMITIVE_TYPE: u32 = 1;

/// What a recording knows besides the guest registers (D12): the index buffer, instances and
/// indirect-argument bases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawParameters {
    pub index_base: u64,
    pub index_buffer_size: Option<u32>,
    pub index_type: u32,
    pub instance_count: u32,
    pub draw_indirect_base: u64,
    pub dispatch_indirect_base: u64,
}

impl Default for DrawParameters {
    fn default() -> Self {
        let state = QueueState::default();
        Self {
            index_base: state.index_base,
            index_buffer_size: state.index_buffer_size,
            index_type: state.index_type,
            instance_count: state.instance_count,
            draw_indirect_base: state.draw_indirect_base,
            dispatch_indirect_base: state.dispatch_indirect_base,
        }
    }
}

/// Footprint dwords (D14): the slot, then the largest form of the draw's own packets.
pub mod dwords {
    pub const SLOT: u32 = 4;
    const PRIMITIVE_TYPE: u32 = 3;
    const INDEX_TYPE: u32 = 3;
    const INDEX_BUFFER: u32 = 3 + 2;
    const INSTANCES: u32 = 2;
    const SET_BASE: u32 = 4;
    /// A dispatch's compute state, inline: PGM_LO/HI, RSRC1/2, RSRC3, RESOURCE_LIMITS, START,
    /// NUM_THREAD, TMPRING and the scratch base (MEC has no IB2).
    pub const COMPUTE_STATE: u32 = 4 + 4 + 3 + 3 + 5 + 5 + 3 + 4;
    pub const INDEXED: u32 = SLOT + PRIMITIVE_TYPE + INDEX_TYPE + INSTANCES + 6;
    pub const INDEXED_OFFSET: u32 = SLOT + PRIMITIVE_TYPE + INDEX_TYPE + INDEX_BUFFER + INSTANCES + 5;
    pub const AUTO: u32 = SLOT + PRIMITIVE_TYPE + INSTANCES + 3;
    pub const INDIRECT: u32 = SLOT + PRIMITIVE_TYPE + INDEX_TYPE + INDEX_BUFFER + INSTANCES + SET_BASE + 5;
    pub const INDIRECT_MULTI: u32 = SLOT + PRIMITIVE_TYPE + INDEX_TYPE + INDEX_BUFFER + INSTANCES + SET_BASE + 10;
    pub const DISPATCH: u32 = COMPUTE_STATE + 5;
    /// SET_BASE and DISPATCH_INDIRECT on graphics queues; the address form on compute queues.
    pub const DISPATCH_INDIRECT: u32 = COMPUTE_STATE + SET_BASE + 3;
}

/// DRAW_INITIATOR: SOURCE_SELECT (DMA for indexed draws, auto index otherwise). NOT_EOP is
/// never set, whatever the guest asks: radv allows it only when nothing but user VGPRs changes
/// before the next draw and never on gfx12.
fn initiator(indexed: bool) -> u32 {
    const SOURCE_AUTO_INDEX: u32 = 2;
    if indexed { 0 } else { SOURCE_AUTO_INDEX }
}

/// The SGPR location of a draw parameter: the host SH dword of the guest's user-data register,
/// or 0 for none, as radv writes it.
fn location(generation: &Generation, guest: Option<u32>) -> u32 {
    guest
        .and_then(|dword| generation.host_user_data(RegisterBank::Shader, dword, 1))
        .unwrap_or(0)
}

fn set_uconfig_index(packets: &mut Packets, register: u32, index: u32, value: u32) -> Result<()> {
    use cp::pfp::SET_UCONFIG_REG_INDEX as S;
    let packet = packets.packet(S::OPCODE, 3)?;
    S::REG_OFFSET.set(packet, (register - 0x30000) / 4);
    S::INDEX.set(packet, index);
    packet[2] = value;
    Ok(())
}

fn address(packet: &mut [u32], at: usize, address: u64) {
    packet[at] = address as u32;
    packet[at + 1] = (address >> 32) as u32;
}

/// A draw's packets.
pub fn encode_draw(
    generation: &Generation,
    out: &mut [u32],
    primitive_type: u32,
    draw: Draw,
    modifier: &DrawModifier,
    parameters: &DrawParameters,
) -> Result<()> {
    use crate::amd::defs::gfx115::{VGT_INDEX_TYPE, VGT_PRIMITIVE_TYPE};
    let mut packets = Packets::new(out);
    set_uconfig_index(&mut packets, VGT_PRIMITIVE_TYPE::OFFSET, SET_UCONFIG_INDEX_PRIMITIVE_TYPE, primitive_type)?;
    let indexed = match draw {
        Draw::Indexed { .. } | Draw::IndexedOffset { .. } => true,
        Draw::Auto { .. } => false,
        Draw::Indirect { indexed, .. } | Draw::IndirectMulti { indexed, .. } => indexed,
    };
    if indexed {
        let index = cp::pfp::SET_UCONFIG_REG_INDEX::enums::index::INDEX_TYPE;
        set_uconfig_index(&mut packets, VGT_INDEX_TYPE::OFFSET, index, VGT_INDEX_TYPE::INDEX_TYPE.put(parameters.index_type))?;
    }
    // The index buffer of offset and indirect indexed draws, as SetIndexBuffer and SetIndexCount
    // left it (the PM4 path forwards both in place as well).
    let index_buffer = |packets: &mut Packets, count: u32| -> Result<()> {
        address(packets.packet(pkt3::INDEX_BASE, 3)?, 1, parameters.index_base);
        packets.packet(pkt3::INDEX_BUFFER_SIZE, 2)?[1] = parameters.index_buffer_size.unwrap_or(count);
        Ok(())
    };
    packets.packet(pkt3::NUM_INSTANCES, 2)?[1] = parameters.instance_count;
    let initiator = initiator(indexed);
    match draw {
        Draw::Indexed { count, address: indices } => {
            let packet = packets.packet(pkt3::DRAW_INDEX_2, 6)?;
            packet[1] = count;
            address(packet, 2, indices);
            packet[4] = count;
            packet[5] = initiator;
        }
        Draw::IndexedOffset { offset, count } => {
            index_buffer(&mut packets, offset.saturating_add(count))?;
            let packet = packets.packet(pkt3::DRAW_INDEX_OFFSET_2, 5)?;
            packet[1] = parameters.index_buffer_size.unwrap_or(offset.saturating_add(count)).max(1);
            packet[2] = offset;
            packet[3] = count;
            packet[4] = initiator;
        }
        Draw::Auto { count } => {
            let packet = packets.packet(pkt3::DRAW_INDEX_AUTO, 3)?;
            packet[1] = count;
            packet[2] = initiator;
        }
        Draw::Indirect { indexed, offset } | Draw::IndirectMulti { indexed, offset, .. } => {
            if indexed {
                index_buffer(&mut packets, u32::MAX)?;
            }
            let base = packets.packet(pkt3::SET_BASE, 4)?;
            base[1] = 1;
            address(base, 2, parameters.draw_indirect_base);
            let (vertex, instance) =
                (location(generation, modifier.vertex_offset), location(generation, modifier.instance_offset));
            if let Draw::IndirectMulti { count, count_address, stride, .. } = draw {
                use cp::pfp::DRAW_INDEX_INDIRECT_MULTI as M;
                let opcode = if indexed { pkt3::DRAW_INDEX_INDIRECT_MULTI } else { pkt3::DRAW_INDIRECT_MULTI };
                let packet = packets.packet(opcode, 10)?;
                packet[1] = offset;
                packet[2] = vertex;
                packet[3] = instance;
                M::DRAW_INDEX_LOC.set(packet, location(generation, modifier.draw_index));
                M::DRAW_INDEX_ENABLE.set(packet, u32::from(modifier.draw_index.is_some()));
                M::USE_VGPRS.set(packet, u32::from(modifier.user_vgprs));
                M::COUNT_INDIRECT_ENABLE.set(packet, u32::from(count_address != 0));
                packet[5] = count;
                address(packet, 6, count_address);
                packet[8] = stride;
                packet[9] = initiator;
            } else {
                let opcode = if indexed { pkt3::DRAW_INDEX_INDIRECT } else { pkt3::DRAW_INDIRECT };
                let packet = packets.packet(opcode, 5)?;
                packet[1] = offset;
                packet[2] = vertex;
                packet[3] = instance;
                packet[4] = initiator;
            }
        }
    }
    packets.finish();
    Ok(())
}

/// A dispatch's packet (direct, or indirect from `address` on compute queues and from the dispatch
/// base plus `offset` on graphics ones).
pub fn encode_dispatch(
    queue: HostQueue,
    out: &mut [u32],
    groups: Option<[u32; 3]>,
    indirect: (u64, u32),
    initiator: u32,
    parameters: &DrawParameters,
) -> Result<()> {
    let mut packets = Packets::new(out);
    match (groups, indirect) {
        (Some([x, y, z]), _) => {
            let packet = packets.packet(pkt3::DISPATCH_DIRECT, 5)?;
            packet[0] |= 1 << 1; // SHADER_TYPE: compute, as radv sets it on every queue
            packet[1..5].copy_from_slice(&[x, y, z, initiator]);
        }
        (None, (args, _)) if queue == HostQueue::Compute => {
            if args == 0 {
                return Err(Error::invalid("a compute-queue indirect dispatch needs its argument address"));
            }
            // MEC has no SET_BASE: DISPATCH_INDIRECT carries the address (radv's compute form).
            let packet = packets.packet(pkt3::DISPATCH_INDIRECT, 4)?;
            packet[0] |= 1 << 1;
            address(packet, 1, args);
            packet[3] = initiator;
        }
        (None, (args, offset)) => {
            let base = if args != 0 { args } else { parameters.dispatch_indirect_base };
            let offset = if args != 0 { 0 } else { offset };
            let set_base = packets.packet(pkt3::SET_BASE, 4)?;
            set_base[0] |= 1 << 1; // SHADER_TYPE: the compute base
            set_base[1] = 1;
            address(set_base, 2, base);
            let packet = packets.packet(pkt3::DISPATCH_INDIRECT, 3)?;
            packet[0] |= 1 << 1;
            packet[1] = offset;
            packet[2] = initiator;
        }
    }
    packets.finish();
    Ok(())
}
