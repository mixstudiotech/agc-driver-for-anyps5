//! The Milestone 1b acceptance pattern: B = x, G = y, R = N mod 256, A = 255 for frame N, written
//! by the native queue with one WRITE_DATA per row into an interchange slot, and the readback
//! check against it.

use super::footprint::{Footprint, BYTES_PER_PIXEL};
use crate::{amd::defs::pkt3, present::KernelLayout, Error, Result};

/// One WRITE_DATA packet carries at most 2^14 payload dwords: control, address and data.
const MAX_ROW_PIXELS: u32 = (1 << 14) - 3;
const WRITE_DATA_DST_MEMORY: u32 = 5 << 8;
const WRITE_DATA_CONFIRM: u32 = 1 << 20;
/// ACQUIRE_MEM GCR_CNTL: write back GL2 and the metadata cache so the D3D12 queue sees the rows.
const GCR_GLM_WB: u32 = 1 << 4;
const GCR_GL2_WB: u32 = 1 << 15;

/// The pixel at (x, y) of frame `frame` as a little-endian B8G8R8A8 dword.
pub fn pixel(x: u32, y: u32, frame: u64) -> u32 {
    (x & 0xff) | ((y & 0xff) << 8) | (((frame & 0xff) as u32) << 16) | 0xff00_0000
}

fn header(opcode: u32, payload_dwords: usize) -> u32 {
    0xc000_0000 | ((((payload_dwords - 1) as u32) & 0x3fff) << 16) | ((opcode & 0xff) << 8)
}

/// The native PM4 that writes frame `frame` into the slot at `slot_address`: one WRITE_DATA per
/// row at `offset + y * row_pitch` (never touching the pitch padding), then a global GL2
/// writeback.
pub fn write_frame(slot_address: u64, footprint: &Footprint, frame: u64) -> Result<Vec<u32>> {
    if footprint.width > MAX_ROW_PIXELS {
        return Err(Error::invalid(format!(
            "rows of {} pixels do not fit one WRITE_DATA",
            footprint.width
        )));
    }
    let row_dwords = footprint.width as usize + 4;
    let mut words = Vec::with_capacity(row_dwords * footprint.height as usize + 8);
    for y in 0..footprint.height {
        let offset = footprint.row_offset(y);
        footprint.check_write(offset, footprint.row_bytes())?;
        let address = slot_address + offset;
        words.push(header(pkt3::WRITE_DATA, footprint.width as usize + 3));
        words.extend([
            WRITE_DATA_DST_MEMORY | WRITE_DATA_CONFIRM,
            address as u32,
            (address >> 32) as u32,
        ]);
        words.extend((0..footprint.width).map(|x| pixel(x, y, frame)));
    }
    words.extend([
        header(pkt3::ACQUIRE_MEM, 7),
        0,
        0xffff_ffff,
        0xff,
        0,
        0,
        0x0a,
        GCR_GLM_WB | GCR_GL2_WB,
    ]);
    Ok(words)
}

/// Draws frame `frame` into a B8G8R8A8 guest display buffer of `width` x `height` texels laid out
/// as `layout`, which is what the display-conversion kernel then reads back.
pub fn draw_guest(
    layout: &KernelLayout,
    width: u32,
    height: u32,
    frame: u64,
    bytes: &mut [u8],
) -> Result<()> {
    for y in 0..height {
        for x in 0..width {
            let at = layout.offset(x, y) as usize;
            bytes
                .get_mut(at..at + 4)
                .ok_or_else(|| Error::invalid(format!("texel ({x}, {y}) is outside the buffer")))?
                .copy_from_slice(&pixel(x, y, frame).to_le_bytes());
        }
    }
    Ok(())
}

/// Compares a readback of a barred canvas, laid out like `footprint`, with frame `frame`: the
/// pattern of the image at (`image_x`, `image_y`) of `image_width` x `image_height`, opaque
/// black elsewhere. The error names the first differing pixel.
pub fn check_canvas(
    readback: &[u8],
    footprint: &Footprint,
    image: (u32, u32, u32, u32),
    frame: u64,
) -> Result<()> {
    const OPAQUE_BLACK: u32 = 0xff00_0000;
    let (image_x, image_y, image_width, image_height) = image;
    if (readback.len() as u64) < footprint.required_bytes() {
        return Err(Error::invalid(format!(
            "readback of {} bytes is smaller than the {}-byte canvas",
            readback.len(),
            footprint.required_bytes()
        )));
    }
    for y in 0..footprint.height {
        let row = footprint.row_offset(y) as usize;
        let inside_y = (image_y..image_y + image_height).contains(&y);
        for x in 0..footprint.width {
            let at = row + (u64::from(x) * BYTES_PER_PIXEL) as usize;
            let got = u32::from_le_bytes(readback[at..at + 4].try_into().unwrap());
            let expected = if inside_y && (image_x..image_x + image_width).contains(&x) {
                pixel(x - image_x, y - image_y, frame)
            } else {
                OPAQUE_BLACK
            };
            if got != expected {
                return Err(Error::invalid(format!(
                    "frame {frame} canvas pixel ({x}, {y}) is 0x{got:08x}, expected 0x{expected:08x}"
                )));
            }
        }
    }
    Ok(())
}

/// Compares a readback laid out like `footprint` with frame `frame`; the error names the first
/// differing pixel.
pub fn check(readback: &[u8], footprint: &Footprint, frame: u64) -> Result<()> {
    if (readback.len() as u64) < footprint.required_bytes() {
        return Err(Error::invalid(format!(
            "readback of {} bytes is smaller than the {}-byte frame",
            readback.len(),
            footprint.required_bytes()
        )));
    }
    for y in 0..footprint.height {
        let row = footprint.row_offset(y) as usize;
        for x in 0..footprint.width {
            let at = row + (u64::from(x) * BYTES_PER_PIXEL) as usize;
            let got = u32::from_le_bytes(readback[at..at + 4].try_into().unwrap());
            let expected = pixel(x, y, frame);
            if got != expected {
                return Err(Error::invalid(format!(
                    "frame {frame} pixel ({x}, {y}) is 0x{got:08x}, expected 0x{expected:08x}"
                )));
            }
        }
    }
    Ok(())
}
