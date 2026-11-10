use crate::{types::TargetIsa, Error, Result};

/// gfx11 V# word1 SWIZZLE_ENABLE[31:30]: 0 off, 1/2/3 for 4/8/16-byte swizzle elements.
const GFX11_SWIZZLE_ENABLE_SHIFT: u32 = 30;
const GFX11_SWIZZLE_ENABLE_MASK: u32 = 3 << GFX11_SWIZZLE_ENABLE_SHIFT;

/// The largest V# STRIDE (word1[29:16], 14 bits on gfx10 and gfx11).
pub const MAX_BUFFER_STRIDE: u32 = 0x3fff;

/// V# word1 of a buffer at `address` with `stride` bytes per record: BASE_ADDRESS[47:32] in
/// [15:0] and STRIDE in [29:16], clamped to [`MAX_BUFFER_STRIDE`] so it never reaches the
/// swizzle bits above it.
pub fn buffer_word1(address: u64, stride: u32) -> u32 {
    ((address >> 32) as u32 & 0xffff) | (stride.min(MAX_BUFFER_STRIDE) << 16)
}

/// A gfx10 buffer resource descriptor (V#).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VSharpDescriptor {
    pub base_address: u64,
    pub stride: u16,
    pub cache_swizzle: bool,
    pub swizzle_enable: bool,
    pub num_records: u32,
    /// DST_SEL_X..W, three bits each.
    pub dst_sel: u16,
    /// Unified gfx10 FORMAT ([18:12] of dword 3).
    pub format: u8,
    pub index_stride: u8,
    pub add_tid_enable: bool,
    pub resource_level: bool,
    pub oob_select: u8,
    pub raw: [u32; 4],
}

impl VSharpDescriptor {
    pub fn from_words(words: [u32; 4]) -> Self {
        Self {
            base_address: u64::from(words[0]) | (u64::from(words[1] & 0xffff) << 32),
            stride: ((words[1] >> 16) & 0x3fff) as u16,
            cache_swizzle: (words[1] >> 30) & 1 != 0,
            swizzle_enable: (words[1] >> 31) & 1 != 0,
            num_records: words[2],
            dst_sel: (words[3] & 0xfff) as u16,
            format: ((words[3] >> 12) & 0x7f) as u8,
            index_stride: ((words[3] >> 21) & 3) as u8,
            add_tid_enable: (words[3] >> 23) & 1 != 0,
            resource_level: (words[3] >> 24) & 1 != 0,
            oob_select: ((words[3] >> 28) & 3) as u8,
            raw: words,
        }
    }

    /// The descriptor rebased to `new_base` in `target`'s encoding. gfx11 renumbers formats,
    /// drops RESOURCE_LEVEL and CACHE_SWIZZLE, and widens SWIZZLE_ENABLE to word1[31:30], where
    /// a gfx10 swizzle (4-byte elements) is the value 1.
    pub fn relocate_for_target(&self, new_base: u64, target: TargetIsa) -> Result<[u32; 4]> {
        let mut out = self.raw;
        out[0] = new_base as u32;
        out[1] = (out[1] & 0xffff_0000) | ((new_base >> 32) as u32 & 0xffff);
        if target.is_gfx11_plus() {
            out[1] = (out[1] & !GFX11_SWIZZLE_ENABLE_MASK)
                | (u32::from(self.swizzle_enable) << GFX11_SWIZZLE_ENABLE_SHIFT);
            let format = gfx11_buffer_format(self.format).ok_or_else(|| {
                Error::unsupported(format!(
                    "gfx10 buffer format {} has no gfx11 equivalent",
                    self.format
                ))
            })?;
            out[3] = (u32::from(self.dst_sel) & 0xfff)
                | (u32::from(format) << 12)
                | (u32::from(self.index_stride) << 21)
                | (u32::from(self.add_tid_enable) << 23)
                | (u32::from(self.oob_select) << 28)
                | (self.raw[3] & 0xc000_0000);
        }
        Ok(out)
    }
}

/// gfx11 buffer FORMAT for a gfx10 one (Mesa GFX10_FORMAT / GFX11_FORMAT by name).
pub fn gfx11_buffer_format(gfx10: u8) -> Option<u8> {
    match gfx10 {
        0..=29 => Some(gfx10),
        36 => Some(30),
        43 => Some(31),
        44 => Some(32),
        45 => Some(33),
        48 => Some(34),
        49 => Some(35),
        50..=77 => Some(gfx10 - 14),
        _ => None,
    }
}
