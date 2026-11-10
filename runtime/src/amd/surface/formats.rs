//! Texel formats: gfx10 image/buffer formats, color target formats and depth formats.
//!
//! Values cross this module as RGBA `[u32; 4]`: f32 bits for normalized, scaled, sRGB and float
//! classes; raw two's-complement integers for UINT/SINT classes. Components a format does not
//! store decode as 0, except alpha, which decodes as 1 (1.0, or integer 1 for integer formats).

#[path = "formats_bc.rs"]
mod bc;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumberType {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Srgb,
    Float,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Block {
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
    Bc6h { signed: bool },
    Bc7,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    /// Components packed upwards from bit 0 of a little-endian element.
    Packed,
    /// 32_FLOAT_CLAMP: a float clamped to [0, 1] when read.
    FloatClamp,
    /// 5_9_9_9: three 9-bit mantissas sharing a 5-bit exponent.
    SharedExponent,
    Block(Block),
}

/// A decoded element format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TexelFormat {
    /// Bits of each stored component in memory order (0 = absent); for block-compressed formats,
    /// the precision of each decoded component.
    pub bits: [u8; 4],
    pub number_type: NumberType,
    /// Shader component (0..=3 = R..A) each memory component holds (color target COMP_SWAP).
    swap: [u8; 4],
    layout: Layout,
    bytes: u32,
    /// Float to fixed-point conversions truncate instead of rounding (CB_COLORn_INFO.ROUND_MODE).
    truncate: bool,
}

const ONE: u32 = 0x3f80_0000;

impl TexelFormat {
    fn packed(bits: &[u8], number_type: NumberType) -> Self {
        let mut all = [0; 4];
        all[..bits.len()].copy_from_slice(bits);
        let total: u32 = bits.iter().map(|&b| u32::from(b)).sum();
        Self {
            bits: all,
            number_type,
            swap: [0, 1, 2, 3],
            layout: Layout::Packed,
            bytes: total.div_ceil(8),
            truncate: false,
        }
    }

    fn block(block: Block, bits: [u8; 4], number_type: NumberType, bytes: u32) -> Self {
        Self {
            bits,
            number_type,
            swap: [0, 1, 2, 3],
            layout: Layout::Block(block),
            bytes,
            truncate: false,
        }
    }

    /// gfx10 unified FORMAT of image (T#) and buffer (V#) descriptors.
    pub fn from_gfx10(format: u32) -> Result<Self> {
        use NumberType::*;
        const N6: [NumberType; 6] = [Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
        const N7: [NumberType; 7] = [Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
        const N3: [NumberType; 3] = [Uint, Sint, Float];
        let i = format as usize;
        let pick = |odd: NumberType, even: NumberType| if format % 2 == 1 { odd } else { even };
        Ok(match format {
            1..=6 => Self::packed(&[8], N6[i - 1]),
            7..=13 => Self::packed(&[16], N7[i - 7]),
            14..=19 => Self::packed(&[8, 8], N6[i - 14]),
            20..=22 => Self::packed(&[32], N3[i - 20]),
            23..=29 => Self::packed(&[16, 16], N7[i - 23]),
            30..=36 => Self::packed(&[11, 11, 10], N7[i - 30]),
            37..=43 => Self::packed(&[10, 11, 11], N7[i - 37]),
            44..=49 => Self::packed(&[2, 10, 10, 10], N6[i - 44]),
            50..=55 => Self::packed(&[10, 10, 10, 2], N6[i - 50]),
            56..=61 => Self::packed(&[8; 4], N6[i - 56]),
            62..=64 => Self::packed(&[32, 32], N3[i - 62]),
            65..=71 => Self::packed(&[16; 4], N7[i - 65]),
            72..=74 => Self::packed(&[32; 3], N3[i - 72]),
            75..=77 => Self::packed(&[32; 4], N3[i - 75]),
            128 => Self::packed(&[8], Srgb),
            129 => Self::packed(&[8, 8], Srgb),
            130 => Self::packed(&[8; 4], Srgb),
            132 => Self {
                layout: Layout::SharedExponent,
                ..Self::packed(&[9, 9, 9], Float)
            },
            133 => Self::packed(&[5, 6, 5], Unorm),
            134 => Self::packed(&[5, 5, 5, 1], Unorm),
            135 => Self::packed(&[1, 5, 5, 5], Unorm),
            136 => Self::packed(&[4; 4], Unorm),
            137 => Self::packed(&[4, 4], Unorm),
            140 => Self {
                layout: Layout::FloatClamp,
                ..Self::packed(&[32], Float)
            },
            141 | 142 => Self::packed(&[24, 8], pick(Unorm, Uint)),
            143 | 144 => Self::packed(&[8, 24], pick(Unorm, Uint)),
            145 | 146 => Self {
                bytes: 8,
                ..Self::packed(&[32, 8], pick(Uint, Float))
            },
            169 | 170 => Self::block(Block::Bc1, [8; 4], pick(Unorm, Srgb), 8),
            171 | 172 => Self::block(Block::Bc2, [8; 4], pick(Unorm, Srgb), 16),
            173 | 174 => Self::block(Block::Bc3, [8; 4], pick(Unorm, Srgb), 16),
            175 | 176 => Self::block(Block::Bc4, [8, 0, 0, 0], pick(Unorm, Snorm), 8),
            177 | 178 => Self::block(Block::Bc5, [8, 8, 0, 0], pick(Unorm, Snorm), 16),
            179 | 180 => Self::block(
                Block::Bc6h {
                    signed: format == 180,
                },
                [16, 16, 16, 0],
                Float,
                16,
            ),
            181 | 182 => Self::block(Block::Bc7, [8; 4], pick(Unorm, Srgb), 16),
            _ => {
                let kind = match format {
                    0 => "INVALID",
                    131 => "6E4_FLOAT",
                    138 | 139 => "1-bit",
                    147..=154 => "subsampled GB_GR/BG_RG",
                    156..=168 => "FMASK",
                    265..=284 => "multimedia",
                    285 => "7E3_FLOAT",
                    286..=292 => "YCBCR",
                    _ => "reserved",
                };
                return Err(Error::unsupported(format!(
                    "gfx10 texel format {format} ({kind}) is not supported"
                )));
            }
        })
    }

    /// Color target format from CB_COLORn_INFO (FORMAT [6:2], NUMBER_TYPE [10:8], COMP_SWAP
    /// [12:11], ROUND_MODE [18]).
    pub fn from_color_info(info: u32) -> Result<Self> {
        use NumberType::*;
        let format = (info >> 2) & 0x1f;
        let number_type =
            [Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Srgb, Float][((info >> 8) & 7) as usize];
        let endian = info & 3;
        if endian != 0 {
            return Err(Error::unsupported(format!(
                "color target ENDIAN {endian} byte swapping is not supported"
            )));
        }
        let bits: &[u8] = match format {
            1 => &[8],
            2 => &[16],
            3 => &[8, 8],
            4 => &[32],
            5 => &[16, 16],
            6 => &[11, 11, 10],
            7 => &[10, 11, 11],
            8 => &[2, 10, 10, 10],
            9 => &[10, 10, 10, 2],
            10 => &[8; 4],
            11 => &[32, 32],
            12 => &[16; 4],
            14 => &[32; 4],
            16 => &[5, 6, 5],
            17 => &[5, 5, 5, 1],
            18 => &[1, 5, 5, 5],
            19 => &[4; 4],
            20 => &[24, 8],
            21 => &[8, 24],
            22 => &[32, 8],
            31 => {
                return Err(Error::unsupported(
                    "color target format 31 (2_10_10_10_6E4) is not supported",
                ))
            }
            _ => {
                return Err(Error::unsupported(format!(
                    "color target format {format} is invalid or reserved"
                )))
            }
        };
        let float_layout =
            matches!(format, 6 | 7 | 22) || bits.iter().all(|&b| matches!(b, 16 | 32));
        if number_type == Float && !float_layout {
            return Err(Error::unsupported(format!(
                "color target format {format} has no float number type"
            )));
        }
        let packed = Self::packed(bits, number_type);
        Ok(Self {
            swap: comp_swap(bits.len(), (info >> 11) & 3),
            truncate: info & (1 << 18) != 0,
            bytes: if format == 22 { 8 } else { packed.bytes },
            ..packed
        })
    }

    /// Depth buffer format from DB_Z_INFO.FORMAT.
    pub fn from_depth(z_format: u32) -> Result<Self> {
        match z_format {
            1 => Ok(Self::packed(&[16], NumberType::Unorm)),
            2 => Ok(Self {
                bytes: 4,
                ..Self::packed(&[24], NumberType::Unorm)
            }),
            3 => Ok(Self::packed(&[32], NumberType::Float)),
            other => Err(Error::unsupported(format!(
                "depth format {other} is invalid"
            ))),
        }
    }

    pub fn bytes_per_element(&self) -> u32 {
        self.bytes
    }

    /// Texels per element: (1, 1), or (4, 4) for block-compressed formats.
    pub fn block_extent(&self) -> (u32, u32) {
        if self.is_block_compressed() {
            (4, 4)
        } else {
            (1, 1)
        }
    }

    pub fn is_block_compressed(&self) -> bool {
        matches!(self.layout, Layout::Block(_))
    }

    pub fn is_integer(&self) -> bool {
        matches!(self.number_type, NumberType::Uint | NumberType::Sint)
    }

    fn element<'a>(&self, element: &'a [u8]) -> Result<&'a [u8]> {
        element.get(..self.bytes as usize).ok_or_else(|| {
            Error::invalid(format!(
                "{}-byte element is shorter than the format's {} bytes",
                element.len(),
                self.bytes
            ))
        })
    }

    fn components(&self) -> impl Iterator<Item = (usize, u32, u32)> + '_ {
        let mut shift = 0;
        self.bits
            .iter()
            .take_while(|&&bits| bits != 0)
            .enumerate()
            .map(move |(memory, &bits)| {
                let bits = u32::from(bits);
                shift += bits;
                (usize::from(self.swap[memory]), bits, shift - bits)
            })
    }

    /// Decodes texel (x, y) of `element` (x = y = 0 unless block-compressed) to RGBA.
    pub fn decode(&self, element: &[u8], x: u32, y: u32) -> Result<[u32; 4]> {
        let element = self.element(element)?;
        let one = if self.is_integer() { 1 } else { ONE };
        match self.layout {
            Layout::Packed | Layout::FloatClamp => {
                let word = little_endian(element);
                let mut rgba = [0, 0, 0, one];
                for (channel, bits, shift) in self.components() {
                    let raw = (word >> shift) as u32 & mask(bits);
                    rgba[channel] = self.decode_component(raw, bits, channel);
                }
                if self.layout == Layout::FloatClamp {
                    rgba[0] = clamp_unit(f32::from_bits(rgba[0])).to_bits();
                }
                Ok(rgba)
            }
            Layout::SharedExponent => {
                let [r, g, b] = rgb9e5_to_f32(little_endian(element) as u32);
                Ok([r.to_bits(), g.to_bits(), b.to_bits(), ONE])
            }
            Layout::Block(block) => self.decode_block(block, element, x, y),
        }
    }

    fn decode_block(&self, block: Block, data: &[u8], x: u32, y: u32) -> Result<[u32; 4]> {
        if x >= 4 || y >= 4 {
            return Err(Error::invalid(format!(
                "texel ({x}, {y}) is outside a 4x4 block"
            )));
        }
        let texel = (y * 4 + x) as usize;
        let snorm = self.number_type == NumberType::Snorm;
        let unorm8 = |rgba: [u8; 4]| rgba.map(|c| f32::from(c) / 255.0);
        let rgba = match block {
            Block::Bc1 => unorm8(bc::color(&array(data), texel, true)),
            Block::Bc2 => {
                let mut rgba = unorm8(bc::color(&array(&data[8..]), texel, false));
                rgba[3] = f32::from(bc::bc2_alpha(&array(data), texel)) / 15.0;
                rgba
            }
            Block::Bc3 => {
                let mut rgba = unorm8(bc::color(&array(&data[8..]), texel, false));
                rgba[3] = bc::bc4(&array(data), texel, false);
                rgba
            }
            Block::Bc4 => [bc::bc4(&array(data), texel, snorm), 0.0, 0.0, 1.0],
            Block::Bc5 => [
                bc::bc4(&array(data), texel, snorm),
                bc::bc4(&array(&data[8..]), texel, snorm),
                0.0,
                1.0,
            ],
            Block::Bc6h { signed } => {
                let [r, g, b] = bc::bc6h(&array(data), texel, signed);
                [
                    f16_to_f32(r.into()),
                    f16_to_f32(g.into()),
                    f16_to_f32(b.into()),
                    1.0,
                ]
            }
            Block::Bc7 => unorm8(bc::bc7(&array(data), texel)),
        };
        let srgb = self.number_type == NumberType::Srgb;
        Ok(std::array::from_fn(|channel| {
            if srgb && channel < 3 {
                srgb_to_linear(rgba[channel]).to_bits()
            } else {
                rgba[channel].to_bits()
            }
        }))
    }

    fn decode_component(&self, raw: u32, bits: u32, channel: usize) -> u32 {
        match self.number_type {
            NumberType::Unorm => unorm_to_f32(raw, bits).to_bits(),
            NumberType::Snorm => snorm_to_f32(raw, bits).to_bits(),
            NumberType::Uscaled => (raw as f32).to_bits(),
            NumberType::Sscaled => (sign_extend(raw, bits) as f32).to_bits(),
            NumberType::Uint => raw,
            NumberType::Sint => sign_extend(raw, bits) as u32,
            NumberType::Srgb => {
                let value = unorm_to_f32(raw, bits);
                if channel < 3 {
                    srgb_to_linear(value).to_bits()
                } else {
                    value.to_bits()
                }
            }
            NumberType::Float => match bits {
                32 => raw,
                16 => f16_to_f32(raw).to_bits(),
                10 | 11 => small_float_to_f32(raw, bits - 5).to_bits(),
                // The 8-bit stencil of X24_8_32_FLOAT reads as UNORM.
                _ => unorm_to_f32(raw, bits).to_bits(),
            },
        }
    }

    /// Encodes RGBA into one element with the format's clamping and rounding.
    pub fn encode(&self, rgba: [u32; 4], out: &mut [u8]) -> Result<()> {
        let bytes = self.bytes as usize;
        let out = out.get_mut(..bytes).ok_or_else(|| {
            Error::invalid(format!(
                "output is shorter than the format's {bytes}-byte element"
            ))
        })?;
        let word = match self.layout {
            Layout::Packed | Layout::FloatClamp => {
                self.components()
                    .fold(0u128, |word, (channel, bits, shift)| {
                        let raw = self.encode_component(rgba[channel], bits, channel);
                        word | (u128::from(raw) << shift)
                    })
            }
            Layout::SharedExponent => u128::from(f32_to_rgb9e5(
                [rgba[0], rgba[1], rgba[2]].map(f32::from_bits),
            )),
            Layout::Block(block) => {
                return Err(Error::unsupported(format!(
                    "encoding block-compressed {block:?} texels is not supported"
                )))
            }
        };
        out.copy_from_slice(&word.to_le_bytes()[..bytes]);
        Ok(())
    }

    fn encode_component(&self, value: u32, bits: u32, channel: usize) -> u32 {
        let float = f32::from_bits(value);
        let truncate = self.truncate;
        match self.number_type {
            NumberType::Unorm => to_unorm(float, bits, truncate),
            NumberType::Snorm => to_snorm(float, bits, truncate),
            NumberType::Uscaled => to_scaled(float, 0.0, f64::from(mask(bits)), truncate) as u32,
            NumberType::Sscaled => {
                let max = f64::from(mask(bits - 1));
                to_scaled(float, -max - 1.0, max, truncate) as i64 as u32 & mask(bits)
            }
            NumberType::Uint => value.min(mask(bits)),
            NumberType::Sint => {
                let half = 1i64 << (bits - 1);
                i64::from(value as i32).clamp(-half, half - 1) as u32 & mask(bits)
            }
            NumberType::Srgb if channel < 3 => to_unorm(linear_to_srgb(float), bits, truncate),
            NumberType::Srgb => to_unorm(float, bits, truncate),
            NumberType::Float => match bits {
                32 => value,
                16 => f32_to_f16(float).into(),
                10 | 11 => f32_to_small_float(float, bits - 5),
                _ => to_unorm(float, bits, truncate),
            },
        }
    }
}

/// Memory component to shader component maps of CB COMP_SWAP (SWAP_STD, SWAP_ALT, SWAP_STD_REV,
/// SWAP_ALT_REV) by component count.
fn comp_swap(count: usize, swap: u32) -> [u8; 4] {
    const TABLE: [[[u8; 4]; 4]; 4] = [
        [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 3, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0]],
        [[0, 1, 2, 0], [0, 1, 3, 0], [2, 1, 0, 0], [3, 1, 0, 0]],
        [[0, 1, 2, 3], [2, 1, 0, 3], [3, 2, 1, 0], [3, 0, 1, 2]],
    ];
    TABLE[count - 1][swap as usize]
}

fn array<const N: usize>(data: &[u8]) -> [u8; N] {
    let mut out = [0; N];
    out.copy_from_slice(&data[..N]);
    out
}

fn little_endian(element: &[u8]) -> u128 {
    let mut bytes = [0; 16];
    bytes[..element.len()].copy_from_slice(element);
    u128::from_le_bytes(bytes)
}

fn mask(bits: u32) -> u32 {
    u32::MAX.checked_shr(32 - bits).unwrap_or(0)
}

fn sign_extend(raw: u32, bits: u32) -> i32 {
    ((raw << (32 - bits)) as i32) >> (32 - bits)
}

/// 2^exponent for exponents in the normal f32 range.
fn pow2(exponent: i32) -> f32 {
    f32::from_bits(((127 + exponent) as u32) << 23)
}

fn clamp_unit(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn unorm_to_f32(raw: u32, bits: u32) -> f32 {
    if bits <= 24 {
        raw as f32 / mask(bits) as f32
    } else {
        (f64::from(raw) / f64::from(mask(bits))) as f32
    }
}

/// SNORM to float; both the most negative code and the one above it read as -1.0.
fn snorm_to_f32(raw: u32, bits: u32) -> f32 {
    let value = sign_extend(raw, bits);
    if value == 0 {
        0.0
    } else {
        (value as f32 / mask(bits - 1).max(1) as f32).max(-1.0)
    }
}

/// Float to fixed point rounds half away from zero (ROUND_BY_HALF) or truncates.
fn round(value: f64, truncate: bool) -> f64 {
    if truncate {
        value.trunc()
    } else {
        value.round()
    }
}

fn to_unorm(value: f32, bits: u32, truncate: bool) -> u32 {
    let unit = f64::from(clamp_unit(value));
    round(unit * f64::from(mask(bits)), truncate) as u32
}

fn to_snorm(value: f32, bits: u32, truncate: bool) -> u32 {
    let unit = if value.is_nan() {
        0.0
    } else {
        f64::from(value).clamp(-1.0, 1.0)
    };
    round(unit * f64::from(mask(bits - 1)), truncate) as i32 as u32 & mask(bits)
}

fn to_scaled(value: f32, min: f64, max: f64, truncate: bool) -> f64 {
    let value = if value.is_nan() {
        0.0
    } else {
        f64::from(value).clamp(min, max)
    };
    round(value, truncate)
}

fn srgb_to_linear(value: f32) -> f32 {
    let value = f64::from(value);
    let linear = if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    };
    linear as f32
}

fn linear_to_srgb(value: f32) -> f32 {
    let value = f64::from(value);
    let srgb = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    srgb as f32
}

/// Unsigned float with a 5-bit exponent (bias 15) and `mantissa` bits (f16 magnitudes, and the
/// 11- and 10-bit floats of packed float formats) to f32.
fn small_float_to_f32(raw: u32, mantissa: u32) -> f32 {
    let exponent = raw >> mantissa;
    let fraction = raw & mask(mantissa);
    let bits = match exponent {
        0 => return fraction as f32 * pow2(-14 - mantissa as i32),
        31 => 0x7f80_0000 | (fraction << (23 - mantissa)),
        _ => ((exponent + 112) << 23) | (fraction << (23 - mantissa)),
    };
    f32::from_bits(bits)
}

fn f16_to_f32(half: u32) -> f32 {
    let magnitude = small_float_to_f32(half & 0x7fff, 10);
    if half & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Rounds the finite magnitude `bits` of an f32 to nearest even into a float with a 5-bit exponent
/// (bias 15) and `mantissa` bits, denormals included; results from the infinity code up overflowed.
fn round_to_small_float(bits: u32, mantissa: u32) -> u32 {
    let exponent = (bits >> 23) as i32 - 112;
    let fraction = bits & 0x7f_ffff;
    let (value, shift) = if exponent > 0 {
        (((exponent as u32) << 23) | fraction, 23 - mantissa)
    } else {
        (
            fraction | 0x80_0000,
            (24 - mantissa as i32 - exponent) as u32,
        )
    };
    if shift > 25 {
        return 0;
    }
    let half = 1 << (shift - 1);
    let rest = value & ((half << 1) - 1);
    let quotient = value >> shift;
    quotient + u32::from(rest > half || (rest == half && quotient & 1 == 1))
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let half = match magnitude {
        0x7f80_0001.. => 0x7e00 | ((magnitude >> 13) & 0x3ff),
        0x7f80_0000 => 0x7c00,
        _ => round_to_small_float(magnitude, 10).min(0x7c00),
    };
    (((bits >> 16) & 0x8000) | half) as u16
}

/// f32 to an unsigned 11- or 10-bit float: negative values and -inf give 0, NaN stays NaN, and
/// finite values past the largest code clamp to it (EXT_packed_float).
fn f32_to_small_float(value: f32, mantissa: u32) -> u32 {
    let infinity = 31 << mantissa;
    if value.is_nan() {
        infinity | (1 << (mantissa - 1))
    } else if value <= 0.0 {
        0
    } else if value.is_infinite() {
        infinity
    } else {
        round_to_small_float(value.to_bits(), mantissa).min(infinity - 1)
    }
}

fn rgb9e5_to_f32(word: u32) -> [f32; 3] {
    let scale = pow2((word >> 27) as i32 - 24);
    [0, 9, 18].map(|shift| ((word >> shift) & 0x1ff) as f32 * scale)
}

/// Shared-exponent encoding of EXT_texture_shared_exponent (N = 9, B = 15).
fn f32_to_rgb9e5(rgb: [f32; 3]) -> u32 {
    const MAX: f32 = 65408.0;
    let rgb = rgb.map(|c| if c.is_nan() { 0.0 } else { c.clamp(0.0, MAX) });
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let log2 = if max >= f32::MIN_POSITIVE {
        (max.to_bits() >> 23) as i32 - 127
    } else {
        -127
    };
    let mut exponent = log2.max(-16) + 16;
    let quantize =
        |c: f32, exponent: i32| (f64::from(c) / f64::from(pow2(exponent - 24))).round() as u32;
    if quantize(max, exponent) == 512 {
        exponent += 1;
    }
    let [r, g, b] = rgb.map(|c| quantize(c, exponent));
    r | (g << 9) | (b << 18) | ((exponent as u32) << 27)
}
