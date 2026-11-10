//! BC1-BC7 block decoders. Each decodes texel `texel` (= y * 4 + x) of one 4x4 block.

/// Two-subset partitions: bit `t` is the subset of texel `t`. BC6H uses the first 32.
const PARTITIONS2: [u16; 64] = [
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800,
    0xffe8, 0xff00, 0xfff0, 0xf000, 0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c, 0xaaaa, 0xf0f0, 0x5a5a, 0x33cc,
    0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718,
    0xccf0, 0x0fcc, 0x7744, 0xee22,
];

/// Three-subset partitions: bits [2t+1:2t] are the subset of texel `t`.
const PARTITIONS3: [u32; 64] = [
    0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
    0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
    0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
    0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
    0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
    0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
    0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
    0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
];

/// Anchor texel of subset 1 of two-subset partitions.
const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];

/// Anchor texels of subsets 1 and 2 of three-subset partitions.
const ANCHOR3: [[u8; 64]; 2] = [
    [
        3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6,
        8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8,
        5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
    ],
    [
        15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3,
        15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
    ],
];

fn weight(bits: u32, index: u32) -> u32 {
    const W2: [u32; 4] = [0, 21, 43, 64];
    const W3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
    const W4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
    match bits {
        2 => W2[index as usize],
        3 => W3[index as usize],
        _ => W4[index as usize],
    }
}

fn subset(subsets: u32, partition: usize, texel: usize) -> usize {
    match subsets {
        2 => usize::from((PARTITIONS2[partition] >> texel) & 1),
        3 => ((PARTITIONS3[partition] >> (2 * texel)) & 3) as usize,
        _ => 0,
    }
}

fn is_anchor(subsets: u32, partition: usize, texel: usize) -> bool {
    let texel = texel as u8;
    texel == 0
        || match subsets {
            2 => texel == ANCHOR2[partition],
            3 => texel == ANCHOR3[0][partition] || texel == ANCHOR3[1][partition],
            _ => false,
        }
}

fn field(data: u128, position: u32, bits: u32) -> u32 {
    ((data >> position) as u32) & ((1u64 << bits) - 1) as u32
}

/// Index of `texel` in an index block at `start` with `bits` per texel, one fewer for anchors.
fn index(data: u128, start: u32, bits: u32, texel: usize, anchor: impl Fn(usize) -> bool) -> u32 {
    let width = |t: usize| bits - u32::from(anchor(t));
    let position = start + (0..texel).map(width).sum::<u32>();
    field(data, position, width(texel))
}

/// BC1-BC3 color block as RGBA8. BC1 blocks with color0 <= color1 hold three colors and
/// transparent black; interpolation follows AMD hardware (basisu's AMD BC1 model).
pub(super) fn color(block: &[u8; 8], texel: usize, punchthrough: bool) -> [u8; 4] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let expand = |c: u16| {
        let [r, g, b] = [c >> 11, (c >> 5) & 63, c & 31].map(u32::from);
        [
            (r << 3) | (r >> 2),
            (g << 2) | (g >> 4),
            (b << 3) | (b >> 2),
        ]
    };
    let (e0, e1) = (expand(c0), expand(c1));
    let selectors = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let selector = (selectors >> (2 * texel)) & 3;
    let rgb: [u32; 3] = match selector {
        0 => e0,
        1 => e1,
        _ if !punchthrough || c0 > c1 => {
            let (a, b) = if selector == 2 { (e0, e1) } else { (e1, e0) };
            std::array::from_fn(|i| (a[i] * 43 + b[i] * 21 + 32) >> 6)
        }
        2 => std::array::from_fn(|i| (e0[i] + e1[i] + 1) >> 1),
        _ => return [0; 4],
    };
    [rgb[0] as u8, rgb[1] as u8, rgb[2] as u8, 255]
}

/// BC2 explicit 4-bit alpha.
pub(super) fn bc2_alpha(block: &[u8; 8], texel: usize) -> u8 {
    (u64::from_le_bytes(*block) >> (4 * texel)) as u8 & 15
}

/// BC4 channel (also the BC3 alpha and BC5 channels), interpolated exactly in float.
pub(super) fn bc4(block: &[u8; 8], texel: usize, signed: bool) -> f32 {
    let (e0, e1, scale) = if signed {
        let e = |b: u8| i32::from(b as i8).max(-127);
        (e(block[0]), e(block[1]), 127)
    } else {
        (i32::from(block[0]), i32::from(block[1]), 255)
    };
    let selector = ((u64::from_le_bytes(*block) >> (16 + 3 * texel)) & 7) as i32;
    let (numerator, denominator) = match selector {
        0 => (e0, 1),
        1 => (e1, 1),
        _ if e0 > e1 => ((8 - selector) * e0 + (selector - 1) * e1, 7),
        6 => (if signed { -scale } else { 0 }, 1),
        7 => (scale, 1),
        _ => ((6 - selector) * e0 + (selector - 1) * e1, 5),
    };
    numerator as f32 / (denominator * scale) as f32
}

/// BC7 modes: subsets, partition bits, rotation bits, index selection bits, color bits, alpha
/// bits, p-bits (0 none, 1 per endpoint, 2 per subset), index bits, secondary index bits.
const BC7_MODES: [[u32; 9]; 8] = [
    [3, 4, 0, 0, 4, 0, 1, 3, 0],
    [2, 6, 0, 0, 6, 0, 2, 3, 0],
    [3, 6, 0, 0, 5, 0, 0, 2, 0],
    [2, 6, 0, 0, 7, 0, 1, 2, 0],
    [1, 0, 2, 1, 5, 6, 0, 2, 3],
    [1, 0, 2, 0, 7, 8, 0, 2, 2],
    [1, 0, 0, 0, 7, 7, 1, 4, 0],
    [2, 6, 0, 0, 5, 5, 1, 2, 0],
];

/// BC7 as RGBA8; blocks with the reserved mode 8 decode to transparent black.
pub(super) fn bc7(block: &[u8; 16], texel: usize) -> [u8; 4] {
    let data = u128::from_le_bytes(*block);
    let mode = block[0].trailing_zeros();
    let Some(&info) = BC7_MODES.get(mode as usize) else {
        return [0; 4];
    };
    let [subsets, partition_bits, rotation_bits, selector_bits, ..] = info;
    let [.., color_bits, alpha_bits, p_bits, index_bits, index_bits2] = info;
    let mut position = mode + 1;
    let mut take = |bits: u32| {
        let value = field(data, position, bits);
        position += bits;
        value
    };
    let partition = take(partition_bits) as usize;
    let rotation = take(rotation_bits);
    let selector = take(selector_bits);
    let count = 2 * subsets as usize;
    let channel_bits = |channel: usize| if channel < 3 { color_bits } else { alpha_bits };
    let mut endpoints = [[0u32; 4]; 6];
    for channel in 0..4 {
        for endpoint in endpoints.iter_mut().take(count) {
            endpoint[channel] = take(channel_bits(channel));
        }
    }
    let mut p = [0u32; 6];
    for pair in p.chunks_mut(2).take(subsets as usize) {
        match p_bits {
            1 => pair.iter_mut().for_each(|bit| *bit = take(1)),
            2 => pair.fill(take(1)),
            _ => {}
        }
    }
    let index_start = position;
    for (endpoint, &p) in endpoints.iter_mut().zip(&p).take(count) {
        for (channel, value) in endpoint.iter_mut().enumerate() {
            let bits = channel_bits(channel);
            if bits == 0 {
                *value = 255;
                continue;
            }
            let (v, bits) = if p_bits == 0 {
                (*value, bits)
            } else {
                ((*value << 1) | p, bits + 1)
            };
            *value = (v << (8 - bits)) | (v >> (2 * bits - 8));
        }
    }
    let primary = index(data, index_start, index_bits, texel, |t| {
        is_anchor(subsets, partition, t)
    });
    let (color_bits, color_index, alpha_bits, alpha_index) = if index_bits2 == 0 {
        (index_bits, primary, index_bits, primary)
    } else {
        let start = index_start + 16 * index_bits - 1;
        let secondary = index(data, start, index_bits2, texel, |t| t == 0);
        if selector == 1 {
            (index_bits2, secondary, index_bits, primary)
        } else {
            (index_bits, primary, index_bits2, secondary)
        }
    };
    let s = subset(subsets, partition, texel);
    let (e0, e1) = (endpoints[2 * s], endpoints[2 * s + 1]);
    let interpolate = |channel: usize, bits: u32, index: u32| {
        let w = weight(bits, index);
        ((64 - w) * e0[channel] + w * e1[channel] + 32) >> 6
    };
    let mut rgba = [
        interpolate(0, color_bits, color_index),
        interpolate(1, color_bits, color_index),
        interpolate(2, color_bits, color_index),
        interpolate(3, alpha_bits, alpha_index),
    ];
    if rotation != 0 {
        rgba.swap(rotation as usize - 1, 3);
    }
    rgba.map(|c| c as u8)
}

/// BC6H endpoint fields: R0..B3 are endpoint * 3 + channel.
const R0: u8 = 0;
const G0: u8 = 1;
const B0: u8 = 2;
const R1: u8 = 3;
const G1: u8 = 4;
const B1: u8 = 5;
const R2: u8 = 6;
const G2: u8 = 7;
const B2: u8 = 8;
const R3: u8 = 9;
const G3: u8 = 10;
const B3: u8 = 11;

struct Bc6Mode {
    transformed: bool,
    endpoint_bits: u32,
    delta_bits: [u32; 3],
    /// Endpoint bits in stream order after the mode bits as (field, first bit, last bit)
    /// triples, stepping down for the reversed fields of the 12- and 16-bit modes.
    fields: &'static [u8],
}

const BC6_MODES: [Bc6Mode; 14] = [
    Bc6Mode {
        transformed: true,
        endpoint_bits: 10,
        delta_bits: [5, 5, 5],
        fields: &[
            G2, 4, 4, B2, 4, 4, B3, 4, 4, R0, 0, 9, G0, 0, 9, B0, 0, 9, R1, 0, 4, G3, 4, 4, G2, 0,
            3, G1, 0, 4, B3, 0, 0, G3, 0, 3, B1, 0, 4, B3, 1, 1, B2, 0, 3, R2, 0, 4, B3, 2, 2, R3,
            0, 4, B3, 3, 3,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 7,
        delta_bits: [6, 6, 6],
        fields: &[
            G2, 5, 5, G3, 4, 4, G3, 5, 5, R0, 0, 6, B3, 0, 0, B3, 1, 1, B2, 4, 4, G0, 0, 6, B2, 5,
            5, B3, 2, 2, G2, 4, 4, B0, 0, 6, B3, 3, 3, B3, 5, 5, B3, 4, 4, R1, 0, 5, G2, 0, 3, G1,
            0, 5, G3, 0, 3, B1, 0, 5, B2, 0, 3, R2, 0, 5, R3, 0, 5,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 11,
        delta_bits: [5, 4, 4],
        fields: &[
            R0, 0, 9, G0, 0, 9, B0, 0, 9, R1, 0, 4, R0, 10, 10, G2, 0, 3, G1, 0, 3, G0, 10, 10, B3,
            0, 0, G3, 0, 3, B1, 0, 3, B0, 10, 10, B3, 1, 1, B2, 0, 3, R2, 0, 4, B3, 2, 2, R3, 0, 4,
            B3, 3, 3,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 11,
        delta_bits: [4, 5, 4],
        fields: &[
            R0, 0, 9, G0, 0, 9, B0, 0, 9, R1, 0, 3, R0, 10, 10, G3, 4, 4, G2, 0, 3, G1, 0, 4, G0,
            10, 10, G3, 0, 3, B1, 0, 3, B0, 10, 10, B3, 1, 1, B2, 0, 3, R2, 0, 3, B3, 0, 0, B3, 2,
            2, R3, 0, 3, G2, 4, 4, B3, 3, 3,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 11,
        delta_bits: [4, 4, 5],
        fields: &[
            R0, 0, 9, G0, 0, 9, B0, 0, 9, R1, 0, 3, R0, 10, 10, B2, 4, 4, G2, 0, 3, G1, 0, 3, G0,
            10, 10, B3, 0, 0, G3, 0, 3, B1, 0, 4, B0, 10, 10, B2, 0, 3, R2, 0, 3, B3, 1, 1, B3, 2,
            2, R3, 0, 3, B3, 4, 4, B3, 3, 3,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 9,
        delta_bits: [5, 5, 5],
        fields: &[
            R0, 0, 8, B2, 4, 4, G0, 0, 8, G2, 4, 4, B0, 0, 8, B3, 4, 4, R1, 0, 4, G3, 4, 4, G2, 0,
            3, G1, 0, 4, B3, 0, 0, G3, 0, 3, B1, 0, 4, B3, 1, 1, B2, 0, 3, R2, 0, 4, B3, 2, 2, R3,
            0, 4, B3, 3, 3,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 8,
        delta_bits: [6, 5, 5],
        fields: &[
            R0, 0, 7, G3, 4, 4, B2, 4, 4, G0, 0, 7, B3, 2, 2, G2, 4, 4, B0, 0, 7, B3, 3, 3, B3, 4,
            4, R1, 0, 5, G2, 0, 3, G1, 0, 4, B3, 0, 0, G3, 0, 3, B1, 0, 4, B3, 1, 1, B2, 0, 3, R2,
            0, 5, R3, 0, 5,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 8,
        delta_bits: [5, 6, 5],
        fields: &[
            R0, 0, 7, B3, 0, 0, B2, 4, 4, G0, 0, 7, G2, 5, 5, G2, 4, 4, B0, 0, 7, G3, 5, 5, B3, 4,
            4, R1, 0, 4, G3, 4, 4, G2, 0, 3, G1, 0, 5, G3, 0, 3, B1, 0, 4, B3, 1, 1, B2, 0, 3, R2,
            0, 4, B3, 2, 2, R3, 0, 4, B3, 3, 3,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 8,
        delta_bits: [5, 5, 6],
        fields: &[
            R0, 0, 7, B3, 1, 1, B2, 4, 4, G0, 0, 7, B2, 5, 5, G2, 4, 4, B0, 0, 7, B3, 5, 5, B3, 4,
            4, R1, 0, 4, G3, 4, 4, G2, 0, 3, G1, 0, 4, B3, 0, 0, G3, 0, 3, B1, 0, 5, B2, 0, 3, R2,
            0, 4, B3, 2, 2, R3, 0, 4, B3, 3, 3,
        ],
    },
    Bc6Mode {
        transformed: false,
        endpoint_bits: 6,
        delta_bits: [6, 6, 6],
        fields: &[
            R0, 0, 5, G3, 4, 4, B3, 0, 0, B3, 1, 1, B2, 4, 4, G0, 0, 5, G2, 5, 5, B2, 5, 5, B3, 2,
            2, G2, 4, 4, B0, 0, 5, G3, 5, 5, B3, 3, 3, B3, 5, 5, B3, 4, 4, R1, 0, 5, G2, 0, 3, G1,
            0, 5, G3, 0, 3, B1, 0, 5, B2, 0, 3, R2, 0, 5, R3, 0, 5,
        ],
    },
    Bc6Mode {
        transformed: false,
        endpoint_bits: 10,
        delta_bits: [10, 10, 10],
        fields: &[R0, 0, 9, G0, 0, 9, B0, 0, 9, R1, 0, 9, G1, 0, 9, B1, 0, 9],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 11,
        delta_bits: [9, 9, 9],
        fields: &[
            R0, 0, 9, G0, 0, 9, B0, 0, 9, R1, 0, 8, R0, 10, 10, G1, 0, 8, G0, 10, 10, B1, 0, 8, B0,
            10, 10,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 12,
        delta_bits: [8, 8, 8],
        fields: &[
            R0, 0, 9, G0, 0, 9, B0, 0, 9, R1, 0, 7, R0, 11, 10, G1, 0, 7, G0, 11, 10, B1, 0, 7, B0,
            11, 10,
        ],
    },
    Bc6Mode {
        transformed: true,
        endpoint_bits: 16,
        delta_bits: [4, 4, 4],
        fields: &[
            R0, 0, 9, G0, 0, 9, B0, 0, 9, R1, 0, 3, R0, 15, 10, G1, 0, 3, G0, 15, 10, B1, 0, 3, B0,
            15, 10,
        ],
    },
];

fn sign_extend(value: i32, bits: u32) -> i32 {
    (value << (32 - bits)) >> (32 - bits)
}

fn unquantize(value: i32, bits: u32, signed: bool) -> i32 {
    if !signed {
        if bits >= 15 || value == 0 {
            value
        } else if value == (1 << bits) - 1 {
            0xffff
        } else {
            ((value << 16) + 0x8000) >> bits
        }
    } else if bits >= 16 {
        value
    } else {
        let magnitude = value.abs();
        let unquantized = if magnitude == 0 {
            0
        } else if magnitude >= (1 << (bits - 1)) - 1 {
            0x7fff
        } else {
            ((magnitude << 15) + 0x4000) >> (bits - 1)
        };
        if value < 0 {
            -unquantized
        } else {
            unquantized
        }
    }
}

/// Scales an interpolated value to the f16 bit pattern it denotes.
fn finish_unquantize(value: i32, signed: bool) -> u16 {
    if !signed {
        return ((value * 31) >> 6) as u16;
    }
    let magnitude = ((value.abs() * 31) >> 5) as u16;
    if value < 0 && magnitude != 0 {
        0x8000 | magnitude
    } else {
        magnitude
    }
}

/// BC6H as f16 bit patterns; reserved modes decode to zero.
pub(super) fn bc6h(block: &[u8; 16], texel: usize, signed: bool) -> [u16; 3] {
    let data = u128::from_le_bytes(*block);
    let mode_index = match data & 3 {
        0 => 0,
        1 => 1,
        _ => match data & 31 {
            2 => 2,
            6 => 3,
            10 => 4,
            14 => 5,
            18 => 6,
            22 => 7,
            26 => 8,
            30 => 9,
            3 => 10,
            7 => 11,
            11 => 12,
            15 => 13,
            _ => return [0; 3],
        },
    };
    let mode = &BC6_MODES[mode_index];
    let mut position = if mode_index < 2 { 2 } else { 5 };
    let mut raw = [[0i32; 3]; 4];
    for triple in mode.fields.chunks_exact(3) {
        let &[target, first, last] = triple else {
            continue;
        };
        let value = &mut raw[usize::from(target / 3)][usize::from(target % 3)];
        for step in 0..=first.abs_diff(last) {
            let bit = if first <= last {
                first + step
            } else {
                first - step
            };
            *value |= (field(data, position, 1) as i32) << bit;
            position += 1;
        }
    }
    let regions = if mode_index < 10 { 2 } else { 1 };
    let partition = if regions == 2 {
        let partition = field(data, position, 5) as usize;
        position += 5;
        partition
    } else {
        0
    };
    let bits = mode.endpoint_bits;
    let mut endpoints = [[0i32; 3]; 4];
    for (e, endpoint) in endpoints.iter_mut().enumerate().take(2 * regions) {
        for (c, value) in endpoint.iter_mut().enumerate() {
            let v = if e > 0 && mode.transformed {
                (raw[0][c] + sign_extend(raw[e][c], mode.delta_bits[c])) & ((1 << bits) - 1)
            } else {
                raw[e][c]
            };
            *value = if signed { sign_extend(v, bits) } else { v };
        }
    }
    let index_bits = if regions == 2 { 3 } else { 4 };
    let index = index(data, position, index_bits, texel, |t| {
        t == 0 || (regions == 2 && t == usize::from(ANCHOR2[partition]))
    });
    let w = weight(index_bits, index) as i32;
    let s = subset(regions as u32, partition, texel);
    std::array::from_fn(|c| {
        let a = unquantize(endpoints[2 * s][c], bits, signed);
        let b = unquantize(endpoints[2 * s + 1][c], bits, signed);
        finish_unquantize((a * (64 - w) + b * w + 32) >> 6, signed)
    })
}
