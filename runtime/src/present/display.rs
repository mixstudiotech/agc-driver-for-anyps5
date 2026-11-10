//! Display format and layout metadata for GPU conversion. This module never reads pixels.

use super::DisplayBufferDesc;
use crate::amd::surface::tiling::{GcnDisplay2dLayout, SurfaceDesc, SurfaceLayout, SW_64KB_R_X};
use crate::{Error, Result};

/// Bit 56 selects the 10-bit (A2B10G10R10) variant of an 8-bit format.
const PIXEL_FORMAT_TEN_BIT: u64 = 0x0100_0000_0000_0000;
const PIXEL_FORMAT_B8G8R8A8: u64 = 0x8000_0000_0000_0000;
const PIXEL_FORMAT_R8G8B8A8: u64 = 0x8000_0000_2200_0000;
const PIXEL_FORMAT_R16G16B16A16_FLOAT: u64 = 0xc001_0006_2200_0000;
const PIXEL_FORMAT_B16G16R16A16_FLOAT: u64 = 0xc001_0006_0000_0000;
const PIXEL_FORMAT_R10G10B10A2_BT2100_PQ: u64 = 0x8100_0704_2200_0000;
const PIXEL_FORMAT_B10G10R10A2_BT2100_PQ: u64 = 0x8100_0704_0000_0000;

const TILING_64KB: u32 = 0;
const TILING_LINEAR: u32 = 1;
const TILING_GCN_2D: u32 = 2;

const ALIGNMENT_64KB: u32 = 65536;
const ALIGNMENT_4KB: u32 = 4096;
const MAX_EXTENT: u32 = 16384;

/// White level PQ (BT.2100) buffers are tone-mapped to: an SDR host's reference white.

/// One DCC key per 256 surface bytes, but at least the 4096 keys of each 512x512 metadata block
/// in which the display reads them (8x8 pixels per key, x and y bits interleaved).

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Encoding {
    Unorm8,
    /// A2B10G10R10 or A2R10G10B10, shown as 8 bits per channel.
    Unorm10,
    /// 10-bit BT.2100 PQ, tone-mapped to SDR.
    Pq,
    Half,
}

#[derive(Clone, Copy, Debug)]
struct Format {
    encoding: Encoding,
    /// Red in the low bits (R8G8B8A8 order) rather than blue.
    red_low: bool,
}

impl Format {
    fn of(pixel_format: u64) -> Result<Self> {
        let base = pixel_format & !PIXEL_FORMAT_TEN_BIT;
        let (encoding, red_low) = match pixel_format {
            PIXEL_FORMAT_R16G16B16A16_FLOAT => (Encoding::Half, true),
            PIXEL_FORMAT_B16G16R16A16_FLOAT => (Encoding::Half, false),
            PIXEL_FORMAT_R10G10B10A2_BT2100_PQ => (Encoding::Pq, true),
            PIXEL_FORMAT_B10G10R10A2_BT2100_PQ => (Encoding::Pq, false),
            _ if base == PIXEL_FORMAT_R8G8B8A8 || base == PIXEL_FORMAT_B8G8R8A8 => {
                let encoding = if pixel_format & PIXEL_FORMAT_TEN_BIT != 0 {
                    Encoding::Unorm10
                } else {
                    Encoding::Unorm8
                };
                (encoding, base == PIXEL_FORMAT_R8G8B8A8)
            }
            _ => {
                return Err(Error::unsupported(format!(
                    "VideoOut: unsupported display pixel format 0x{pixel_format:016x}"
                )))
            }
        };
        Ok(Self { encoding, red_low })
    }

    fn texel_bytes(self) -> u32 {
        if self.encoding == Encoding::Half {
            8
        } else {
            4
        }
    }
}

enum Addressing {
    Tiled(SurfaceLayout),
    Linear { pitch: u32 },
    Gcn(GcnDisplay2dLayout),
}

/// A validated display buffer: its texel format, layout and guest footprint.
struct Geometry {
    format: Format,
    addressing: Addressing,
    size: usize,
}

impl Geometry {
    fn new(desc: &DisplayBufferDesc) -> Result<Self> {
        if desc.width == 0
            || desc.height == 0
            || desc.width > MAX_EXTENT
            || desc.height > MAX_EXTENT
        {
            return Err(Error::invalid(
                "VideoOut: invalid display buffer dimensions",
            ));
        }
        let format = Format::of(desc.pixel_format)?;
        if format.encoding == Encoding::Half && desc.dcc_address != 0 {
            return Err(Error::unsupported(
                "VideoOut: DCC metadata on a 16-bit float display buffer is not implemented",
            ));
        }
        let alignment = if desc.alignment == 0 {
            ALIGNMENT_64KB
        } else {
            desc.alignment
        };
        if alignment != ALIGNMENT_64KB && alignment != ALIGNMENT_4KB {
            return Err(Error::unsupported(format!(
                "VideoOut: display buffer alignment 0x{alignment:x} is not supported"
            )));
        }
        if alignment == ALIGNMENT_4KB && !matches!(desc.tiling_mode, TILING_LINEAR | TILING_GCN_2D)
        {
            return Err(Error::unsupported(
                "VideoOut: 4 KiB display buffer alignment is only supported for linear and PS4 buffers",
            ));
        }
        if desc.address == 0 || desc.address % u64::from(alignment) != 0 {
            return Err(Error::invalid(if alignment == ALIGNMENT_64KB {
                "VideoOut: display buffer requires 64 KiB alignment"
            } else {
                "VideoOut: display buffer requires 4 KiB alignment"
            }));
        }
        let texel_bytes = format.texel_bytes();
        let (addressing, size) = match desc.tiling_mode {
            TILING_GCN_2D => {
                if desc.dcc_address != 0 || format.encoding == Encoding::Half {
                    return Err(Error::invalid(
                        "VideoOut: PS4 display buffers carry no DCC and no half-float texels",
                    ));
                }
                if desc.pitch_in_pixel < desc.width || desc.pitch_in_pixel % 128 != 0 {
                    return Err(Error::invalid(
                        "VideoOut: PS4 tiled display pitch is not padded to 128 pixels",
                    ));
                }
                let layout =
                    GcnDisplay2dLayout::new(desc.pitch_in_pixel, desc.height, texel_bytes)?;
                let size = layout.size_bytes();
                (Addressing::Gcn(layout), size)
            }
            TILING_64KB | TILING_LINEAR => {
                if desc.tiling_mode == TILING_64KB && desc.pitch_in_pixel != 0 {
                    return Err(Error::unsupported(
                        "VideoOut: tiled display pitch is unsupported",
                    ));
                }
                if desc.dcc_address != 0 && desc.tiling_mode != TILING_64KB {
                    return Err(Error::invalid(
                        "VideoOut: a linear display buffer cannot carry DCC metadata",
                    ));
                }
                if desc.dcc_address == 0 && desc.dcc_clear_color != 0 {
                    return Err(Error::invalid(
                        "VideoOut: a DCC clear color needs DCC metadata",
                    ));
                }
                let pitch = if desc.pitch_in_pixel == 0 {
                    desc.width
                } else {
                    desc.pitch_in_pixel
                };
                if pitch < desc.width || pitch > MAX_EXTENT {
                    return Err(Error::invalid("VideoOut: invalid linear display pitch"));
                }
                if desc.tiling_mode == TILING_LINEAR {
                    let size = u64::from(pitch) * u64::from(desc.height) * u64::from(texel_bytes);
                    (Addressing::Linear { pitch }, size)
                } else {
                    // PS5 display tiling is SW_64KB_R_X: 64 KiB blocks 128 pixels wide.
                    let layout = SurfaceLayout::new(&SurfaceDesc {
                        swizzle_mode: SW_64KB_R_X,
                        width: desc.width,
                        height: desc.height,
                        depth: 1,
                        layers: 1,
                        mip_levels: 1,
                        bytes_per_element: texel_bytes,
                        ..SurfaceDesc::default()
                    })?;
                    let size = layout.size_bytes();
                    (Addressing::Tiled(layout), size)
                }
            }
            other => {
                return Err(Error::unsupported(format!(
                    "VideoOut: unsupported display tiling mode {other}"
                )))
            }
        };
        let size = usize::try_from(size)
            .ok()
            .filter(|&size| desc.address.checked_add(size as u64).is_some())
            .ok_or_else(|| Error::invalid("VideoOut: display buffer range overflow"))?;
        Ok(Self {
            format,
            addressing,
            size,
        })
    }

    fn texel_offset(&self, x: u32, y: u32) -> Result<u64> {
        Ok(match &self.addressing {
            Addressing::Tiled(layout) => layout.element_offset(x, y, 0, 0)?,
            Addressing::Linear { pitch } => {
                (u64::from(y) * u64::from(*pitch) + u64::from(x))
                    * u64::from(self.format.texel_bytes())
            }
            Addressing::Gcn(layout) => layout.element_offset(x, y)?,
        })
    }

    /// Bytes of one swizzle block, when the layout has power-of-two blocks.
    fn block_bytes(&self) -> Option<u64> {
        let bytes = match &self.addressing {
            Addressing::Tiled(layout) => layout.block_bytes(),
            Addressing::Gcn(layout) => {
                let [width, height] = layout.block_extent();
                u64::from(width) * u64::from(height) * u64::from(self.format.texel_bytes())
            }
            Addressing::Linear { .. } => 0,
        };
        bytes.is_power_of_two().then_some(bytes)
    }
}

/// Guest bytes covered by a registered display buffer.
pub(crate) fn display_buffer_size(desc: &DisplayBufferDesc) -> Result<usize> { Ok(Geometry::new(desc)?.size) }

/// Addresses the display buffer in guest memory, in the guest's own display layout: for a buffer
/// no host render surface holds (the CPU or a buffer store wrote it).
pub(crate) fn guest_kernel_layout(desc: &DisplayBufferDesc) -> Result<KernelLayout> {
    kernel_equation(desc, Geometry::new(desc)?)
}

/// How the display-conversion kernel addresses a display buffer and converts its texels: blocks
/// of 2^log2 texels laid out `pitch` per row (linear rows are 1x1-texel blocks), and inside a
/// block an offset whose bit k is the parity of `masks[k]` and the packed coordinate
/// `(x & 0xfff) | (y & 0xfff) << 12`, XORed with `xor`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelLayout {
    pub pitch: u32,
    pub log2_block_width: u32,
    pub log2_block_height: u32,
    pub log2_block_bytes: u32,
    pub xor: u32,
    pub masks: [u32; 16],
    /// 0 B8G8R8A8, 1 R8G8B8A8, 2 A2B10G10R10 (red low), 3 A2R10G10B10 (red high).
    pub mode: u32,
}

impl KernelLayout {
    /// The texel offset the kernel computes, in its 32-bit arithmetic.
    pub fn offset(&self, x: u32, y: u32) -> u64 {
        let coord = (x & 0xfff) | ((y & 0xfff) << 12);
        let in_block = self
            .masks
            .iter()
            .enumerate()
            .fold(0u32, |offset, (bit, mask)| {
                offset | (((mask & coord).count_ones() & 1) << bit)
            })
            ^ self.xor;
        let block = (y >> self.log2_block_height)
            .wrapping_mul(self.pitch)
            .wrapping_add(x >> self.log2_block_width)
            .wrapping_shl(self.log2_block_bytes);
        u64::from(block.wrapping_add(in_block))
    }
}

/// The kernel's view of `desc`, checked against the layout on the last row, the last column and
/// a diagonal. PQ and half-float buffers, and layouts the kernel's equation cannot express, are
/// unsupported.
/// Addresses the uncompressed host render surface, using its actual host swizzle equation.
pub(crate) fn host_kernel_layout(desc: &DisplayBufferDesc, surface: &SurfaceLayout) -> Result<KernelLayout> {
    let format = Format::of(desc.pixel_format)?;
    let shape = surface.desc();
    if shape.bytes_per_element != format.texel_bytes() || desc.width > shape.width || desc.height > shape.height {
        return Err(Error::invalid("present: the host surface does not contain the display buffer"));
    }
    kernel_equation(desc, Geometry { format, addressing: Addressing::Tiled(surface.clone()), size: surface.size_bytes() as usize })
}

fn kernel_equation(desc: &DisplayBufferDesc, geometry: Geometry) -> Result<KernelLayout> {
    let mode =
        match (geometry.format.encoding, geometry.format.red_low) {
            (Encoding::Unorm8, false) => 0,
            (Encoding::Unorm8, true) => 1,
            (Encoding::Unorm10, true) => 2,
            (Encoding::Unorm10, false) => 3,
            (Encoding::Pq | Encoding::Half, _) => return Err(Error::unsupported(
                "present: PQ and half-float display buffers are not converted on the native queue",
            )),
        };
    if geometry.size > u32::MAX as usize {
        return Err(Error::unsupported(
            "present: the conversion kernel addresses display buffers below 4 GiB",
        ));
    }
    let texel_log2 = geometry.format.texel_bytes().trailing_zeros();
    let inexpressible = || Error::unsupported("present: the display layout has no kernel equation");
    let layout = match &geometry.addressing {
        Addressing::Linear { pitch } => KernelLayout {
            pitch: *pitch,
            log2_block_width: 0,
            log2_block_height: 0,
            log2_block_bytes: texel_log2,
            xor: 0,
            masks: [0; 16],
            mode,
        },
        addressing => {
            let (block_width, block_height, pitch) = match addressing {
                Addressing::Tiled(layout) => {
                    let level = layout.mip_level(0).ok_or_else(inexpressible)?;
                    if level.offset != 0 || level.tail_origin.is_some() {
                        return Err(inexpressible());
                    }
                    let [width, height, _] = layout.block_extent();
                    (width, height, level.pitch)
                }
                Addressing::Gcn(layout) => {
                    let [width, height] = layout.block_extent();
                    (width, height, desc.pitch_in_pixel.div_ceil(width))
                }
                Addressing::Linear { .. } => unreachable!("handled above"),
            };
            let block_bytes = geometry.block_bytes().ok_or_else(inexpressible)?;
            if desc.width > 4096
                || desc.height > 4096
                || block_bytes > 1 << 16
                || !block_width.is_power_of_two()
                || !block_height.is_power_of_two()
            {
                return Err(inexpressible());
            }
            let low = block_bytes - 1;
            let origin = geometry.texel_offset(0, 0)? & low;
            let mut masks = [0u32; 16];
            let mut column = |offset: u64, input: u32| {
                let flips = (offset & low) ^ origin;
                for (bit, mask) in masks.iter_mut().enumerate() {
                    if (flips >> bit) & 1 != 0 {
                        *mask |= 1 << input;
                    }
                }
            };
            for bit in (0..12).filter(|&bit| 1u32 << bit < desc.width) {
                column(geometry.texel_offset(1 << bit, 0)?, bit);
            }
            for bit in (0..12).filter(|&bit| 1u32 << bit < desc.height) {
                column(geometry.texel_offset(0, 1 << bit)?, 12 + bit);
            }
            KernelLayout {
                pitch,
                log2_block_width: block_width.trailing_zeros(),
                log2_block_height: block_height.trailing_zeros(),
                log2_block_bytes: block_bytes.trailing_zeros(),
                xor: origin as u32,
                masks,
                mode,
            }
        }
    };
    let (width, height) = (desc.width, desc.height);
    let samples = (0..width)
        .map(|x| (x, height - 1))
        .chain((0..height).map(|y| (width - 1, y)))
        .chain((0..height).map(|y| {
            (
                (u64::from(y) * u64::from(width) / u64::from(height)) as u32,
                y,
            )
        }));
    for (x, y) in samples {
        if layout.offset(x, y) != geometry.texel_offset(x, y)? {
            return Err(inexpressible());
        }
    }
    Ok(layout)
}
