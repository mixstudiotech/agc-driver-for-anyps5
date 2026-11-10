//! The layout of host-format PM4 in IBs: filler, padding, CHAIN packets and where a stream can be
//! cut.

use crate::{amd::defs::pkt3, Error, Result};

/// Header-only PKT3 NOP (count 0x3fff encodes an empty body): the one-dword filler on gfx7+.
const NOP_PAD_ONE_DWORD: u32 = 0xffff_1000;
/// INDIRECT_BUFFER IB_SIZE is a 20-bit dword count.
pub const MAX_IB_DWORDS: usize = 0xf_ffff;
pub const DWORD_BYTES: usize = std::mem::size_of::<u32>();

/// Finished IBs are a multiple of 8 dwords.
pub const IB_ALIGNMENT_DWORDS: usize = 8;
/// IB start addresses are 256-byte aligned (the KMD itself only requires 32 bytes).
pub const IB_ADDRESS_ALIGNMENT: u64 = 256;

const TYPE_SHIFT: u32 = 30;
const TYPE_3: u32 = 3;
const COUNT_SHIFT: u32 = 16;
const COUNT_MASK: u32 = 0x3fff;
const OPCODE_SHIFT: u32 = 8;
const OPCODE_MASK: u32 = 0xff;
const ADDRESS_HIGH_SHIFT: u32 = 32;
/// EXEC_COUNT limit of a gfx11 COND_EXEC.
const COND_EXEC_MAX_DWORDS: u32 = 0x3fff;

fn fail<T>(at: usize, message: impl std::fmt::Display) -> Result<T> {
    Err(Error::invalid(format!("packet at dword {at}: {message}")))
}

fn type3_header(opcode: u32, payload_dwords: usize) -> u32 {
    (TYPE_3 << TYPE_SHIFT)
        | ((opcode & OPCODE_MASK) << OPCODE_SHIFT)
        | ((((payload_dwords - 1) as u32) & COUNT_MASK) << COUNT_SHIFT)
}

/// `dwords` of filler the CP skips: a PKT3 NOP, or the one-dword NOP pad. Never type-2.
pub fn nop_fill(dwords: usize) -> Vec<u32> {
    match dwords {
        0 => Vec::new(),
        1 => vec![NOP_PAD_ONE_DWORD],
        _ => {
            let mut fill = vec![0; dwords];
            fill[0] = type3_header(pkt3::NOP, dwords - 1);
            fill
        }
    }
}

/// Pads a host stream to a multiple of 8 dwords with NOP filler and checks the result fits one
/// IB.
pub fn pad(words: &[u32]) -> Result<Vec<u32>> {
    let gap = (IB_ALIGNMENT_DWORDS - words.len() % IB_ALIGNMENT_DWORDS) % IB_ALIGNMENT_DWORDS;
    let mut padded = words.to_vec();
    padded.extend(nop_fill(gap));
    if padded.len() > MAX_IB_DWORDS {
        return Err(Error::invalid(format!(
            "padded IB of {} dwords exceeds 0x{:x}",
            padded.len(),
            MAX_IB_DWORDS
        )));
    }
    Ok(padded)
}

/// The dword offset of every packet in a stream that frames cleanly (one-dword NOP pads count as
/// packets). Splitting a stream at these offsets never cuts a packet.
pub fn packet_offsets(words: &[u32]) -> Result<Vec<usize>> {
    let mut offsets = Vec::new();
    let mut at = 0;
    while at < words.len() {
        offsets.push(at);
        let header = words[at];
        if header == NOP_PAD_ONE_DWORD {
            at += 1;
            continue;
        }
        if header >> TYPE_SHIFT != TYPE_3 {
            return fail(
                at,
                format!(
                    "type-{} header 0x{header:08x} is not allowed on gfx11",
                    header >> TYPE_SHIFT
                ),
            );
        }
        let payload = ((header >> COUNT_SHIFT) & COUNT_MASK) as usize + 1;
        if payload > words.len() - at - 1 {
            return fail(at, "truncated packet");
        }
        at += payload + 1;
    }
    Ok(offsets)
}

/// A CHAIN packet to the next IB of a submission: INDIRECT_BUFFER with CHAIN and VALID.
pub fn chain_packet(next_ib: u64, next_ib_dwords: usize) -> Result<[u32; 4]> {
    const CHAIN: u32 = 1 << 20;
    const VALID: u32 = 1 << 23;
    if next_ib % IB_ADDRESS_ALIGNMENT != 0
        || next_ib_dwords == 0
        || next_ib_dwords > MAX_IB_DWORDS
    {
        return Err(Error::invalid(format!(
            "chained IB 0x{next_ib:x} with {next_ib_dwords} dwords breaks the 256-byte / 0xfffff rules"
        )));
    }
    Ok([
        type3_header(pkt3::INDIRECT_BUFFER, 3),
        next_ib as u32,
        (next_ib >> ADDRESS_HIGH_SHIFT) as u32,
        CHAIN | VALID | next_ib_dwords as u32,
    ])
}

/// The dword offsets at which a stream may be cut into separate IBs: packet starts outside every
/// COND_EXEC range (EXEC_COUNT counts dwords of the IB the range starts in), then the end of the
/// stream.
pub fn split_points(words: &[u32]) -> Result<Vec<usize>> {
    let mut points = Vec::new();
    let mut ends: Vec<usize> = Vec::new();
    for at in packet_offsets(words)? {
        ends.retain(|&end| end > at);
        if ends.is_empty() {
            points.push(at);
        }
        let header = words[at];
        if header != NOP_PAD_ONE_DWORD
            && (header >> OPCODE_SHIFT) & OPCODE_MASK == pkt3::COND_EXEC
            && (header >> COUNT_SHIFT) & COUNT_MASK == 3
        {
            ends.push(at + 5 + (words[at + 4] & COND_EXEC_MAX_DWORDS) as usize);
        }
    }
    points.push(words.len());
    Ok(points)
}
