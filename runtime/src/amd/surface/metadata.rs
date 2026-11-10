//! Guest gfx10 DCC/HTILE states that can be materialized into an uncompressed host surface.
//!
//! The encodings follow Mesa `src/amd/common/ac_descriptors.h` and RADV's
//! `meta/radv_meta_clear.c`. gfx11 clear codes are deliberately not used to interpret guest
//! metadata. Real block compression, comp-to-single and nonuniform metadata need metadata
//! address equations and decompression; they are refused rather than read as ordinary texels.

use super::{
    formats::{NumberType, TexelFormat},
    retile::{clear_dispatches, CopyDispatch},
    tiling::{SurfaceLayout, SW_64KB_R_X, SW_64KB_Z_X},
};
use crate::{Error, Result};

pub const DCC_EXPANDED: u8 = 0xff;
pub const HTILE_Z_EXPANDED: u32 = 0xfffc_000f;
pub const HTILE_ZS_EXPANDED: u32 = 0xffff_f3ff;

/// Full metadata extent in the fixed guest configuration (16 pipes, 256-byte interleave,
/// without RB+), derived from addrlib `Gfx10Lib::GetMetaBlkSize` and `HwlComputeDccInfo`.
/// This is not `surface_bytes / 256`: metadata macroblock padding and mip tails change it.
pub fn dcc_key_bytes(layout: &SurfaceLayout) -> Result<usize> {
    if layout.desc().swizzle_mode != SW_64KB_R_X {
        return Err(Error::unsupported(
            "DCC metadata extents are implemented for guest 64KB_R_X surfaces",
        ));
    }
    let element = layout.desc().bytes_per_element;
    if !element.is_power_of_two() || element > 16 {
        return Err(Error::unsupported(
            "DCC metadata requires 1, 2, 4, 8 or 16-byte elements",
        ));
    }
    // A color compression block covers 256 bytes and each metadata key occupies one byte.
    // In this configuration both pipe-aligned and unaligned DCC use a 4-KiB meta block.
    let coordinate_bits = 12 + 8 - element.trailing_zeros();
    metadata_extent(
        layout,
        1 << coordinate_bits.div_ceil(2),
        1 << (coordinate_bits / 2),
        4096,
    )
}

/// Full pipe-aligned guest HTILE extent. gfx10 addrlib refuses unaligned HTILE; each
/// 32-KiB metadata macroblock covers 1024x512 pixels in the fixed 16-pipe guest model.
pub fn htile_bytes(layout: &SurfaceLayout, pipe_aligned: bool) -> Result<usize> {
    if !pipe_aligned || layout.desc().swizzle_mode != SW_64KB_Z_X {
        return Err(Error::unsupported(
            "HTILE metadata extents need pipe-aligned guest 64KB_Z_X surfaces",
        ));
    }
    if layout.desc().mip_levels.max(1) != 1 {
        return Err(Error::unsupported(
            "HTILE metadata mip tails need an explicitly verified guest depth-tail configuration",
        ));
    }
    metadata_extent(layout, 1024, 512, 32768)
}

fn metadata_extent(
    layout: &SurfaceLayout,
    block_width: u32,
    block_height: u32,
    block_bytes: u64,
) -> Result<usize> {
    let desc = layout.desc();
    if desc.volume || desc.depth > 1 {
        return Err(Error::unsupported(
            "3D metadata extents are not implemented",
        ));
    }
    let levels = desc.mip_levels.max(1);
    let first_tail = (0..levels)
        .find(|&level| {
            layout
                .mip_level(level)
                .is_some_and(|mip| mip.tail_origin.is_some())
        })
        .unwrap_or(levels);
    let level_bytes = |level: u32| -> u64 {
        let width = desc.width.div_ceil(1 << level).max(1);
        let height = desc.height.div_ceil(1 << level).max(1);
        u64::from(width.div_ceil(block_width))
            * u64::from(height.div_ceil(block_height))
            * block_bytes
    };
    let slice = if levels == 1 {
        level_bytes(0)
    } else {
        let tail_bytes = if first_tail == levels { 0 } else { block_bytes };
        tail_bytes + (0..first_tail).map(level_bytes).sum::<u64>()
    };
    usize::try_from(slice * u64::from(desc.layers.max(1)))
        .map_err(|_| Error::invalid("metadata extent exceeds the host address space"))
}

/// A fully encoded constant element, including packed and floating-point color formats.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClearTexel {
    value: [u8; 16],
    bytes: u8,
}

impl ClearTexel {
    pub fn new(value: &[u8]) -> Result<Self> {
        if !matches!(value.len(), 1 | 2 | 4 | 8 | 16) {
            return Err(Error::unsupported(
                "metadata clears need 1, 2, 4, 8 or 16-byte elements",
            ));
        }
        let mut texel = Self {
            value: [0; 16],
            bytes: value.len() as u8,
        };
        texel.value[..value.len()].copy_from_slice(value);
        Ok(texel)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.value[..usize::from(self.bytes)]
    }

    /// A GPU fill of every mip/layer in a host-layout surface. The original guest texels are
    /// never read, which matters for a metadata-only clear of stale or uninitialized memory.
    pub fn dispatches(&self, layout: &SurfaceLayout, base: u64) -> Result<Vec<CopyDispatch>> {
        clear_dispatches(layout, base, self.bytes())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DccState {
    Expanded,
    Clear(ClearTexel),
}

/// Classifies a complete, coherent metadata extent. The caller must wait for pending GPU
/// writers before reading it, and must not classify an arbitrary prefix as the whole extent.
pub fn classify_dcc(
    keys: &[u8],
    format: TexelFormat,
    clear_words: Option<[u32; 2]>,
) -> Result<DccState> {
    let &key = keys
        .first()
        .ok_or_else(|| Error::invalid("a DCC metadata extent is empty"))?;
    if let Some(index) = keys.iter().position(|&other| other != key) {
        return Err(Error::unsupported(format!(
            "mixed DCC metadata: key {index} is {:#04x}, first key is {key:#04x}; block decompression is required",
            keys[index]
        )));
    }
    if key == DCC_EXPANDED {
        return Ok(DccState::Expanded);
    }
    if format.is_block_compressed() {
        return Err(Error::unsupported(
            "DCC fast clears of block-compressed formats are not implemented",
        ));
    }
    let bytes = format.bytes_per_element() as usize;
    let mut value = [0; 16];
    match key {
        0x00 => {}
        0x20 => {
            let words = clear_words.ok_or_else(|| {
                Error::unsupported("DCC clear-register metadata needs CB_COLOR_CLEAR_WORD0/1")
            })?;
            if bytes > 8 {
                return Err(Error::unsupported(
                    "DCC clear-register texels wider than 64 bits are not implemented",
                ));
            }
            value[..4].copy_from_slice(&words[0].to_le_bytes());
            value[4..8].copy_from_slice(&words[1].to_le_bytes());
        }
        0x40 | 0x80 | 0xc0 => {
            // Split clear codes distinguish the extra (MSB) channel. Restrict them to the
            // canonical four-component layouts, whose extra-channel position is known.
            if key != 0xc0
                && !matches!(
                    format.bits,
                    [8, 8, 8, 8] | [10, 10, 10, 2] | [16, 16, 16, 16] | [32, 32, 32, 32]
                )
            {
                return Err(Error::unsupported(
                    "split DCC clears need a known four-component MSB-alpha layout",
                ));
            }
            if !matches!(
                format.number_type,
                NumberType::Unorm | NumberType::Srgb | NumberType::Float
            ) {
                return Err(Error::unsupported("nonzero DCC constant clears of integer/scaled/signed formats need format-specific clamping"));
            }
            let one = 1.0f32.to_bits();
            // Encode all channels at one first so color-target component swaps do not change
            // which bits the metadata's MSB extra-channel selector names.
            format.encode([one; 4], &mut value[..bytes])?;
            if key != 0xc0 {
                let mut packed = u128::from_le_bytes(value);
                let bits = format
                    .bits
                    .iter()
                    .copied()
                    .rfind(|&bits| bits != 0)
                    .unwrap();
                let shift =
                    format.bits.iter().map(|&bits| u32::from(bits)).sum::<u32>() - u32::from(bits);
                let extra = ((1u128 << bits) - 1) << shift;
                packed &= if key == 0x40 { extra } else { !extra };
                value = packed.to_le_bytes();
            }
        }
        0x10 => {
            return Err(Error::unsupported(
                "DCC comp-to-single metadata needs a per-block clear texel",
            ))
        }
        other => {
            return Err(Error::unsupported(format!(
                "DCC key {other:#04x} needs block decompression"
            )))
        }
    }
    Ok(DccState::Clear(ClearTexel::new(&value[..bytes])?))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaneState {
    Expanded,
    Clear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HtileState {
    pub depth: PlaneState,
    pub stencil: Option<PlaneState>,
}

/// Decodes a uniform gfx10 HTILE extent (no VRS encoding). Depth-only and interleaved Z/S
/// metadata have different field positions. Clear-register depth is checked against the exact
/// 14-bit HiZ value; the full-precision value still comes from DB_DEPTH_CLEAR.
pub fn classify_htile(
    words: &[u32],
    depth_only: bool,
    clear_depth: f32,
    _clear_stencil: u8,
) -> Result<HtileState> {
    let &word = words
        .first()
        .ok_or_else(|| Error::invalid("an HTILE metadata extent is empty"))?;
    if words.iter().any(|&other| other != word) {
        return Err(Error::unsupported(
            "nonuniform HTILE metadata needs block decompression",
        ));
    }
    let depth = if word & 0xf == 0xf {
        PlaneState::Expanded
    } else {
        if !clear_depth.is_finite() || !(0.0..=1.0).contains(&clear_depth) {
            return Err(Error::invalid("DB_DEPTH_CLEAR is outside [0, 1]"));
        }
        let z = (clear_depth * 16383.0).round() as u32;
        let clear = if depth_only {
            (z << 4) | (z << 18)
        } else {
            z << 18
        };
        let mask = if depth_only { u32::MAX } else { 0xffff_f00f };
        if word & mask != clear {
            return Err(Error::unsupported(format!(
                "HTILE word {word:#010x} does not prove DB_DEPTH_CLEAR {clear_depth}"
            )));
        }
        PlaneState::Clear
    };
    let stencil = if depth_only {
        None
    } else {
        Some(match (word >> 8) & 3 {
            0 => PlaneState::Clear,
            3 => PlaneState::Expanded,
            _ => {
                return Err(Error::unsupported(
                    "compressed HTILE stencil blocks need decompression",
                ))
            }
        })
    };
    Ok(HtileState { depth, stencil })
}
