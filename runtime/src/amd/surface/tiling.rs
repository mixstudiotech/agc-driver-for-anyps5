//! gfx10 surface layouts (PS5 swizzle modes) for color targets, depth buffers and textures, and the
//! PS4 (GCN) 2D display tiling of PS4 display buffers.

use crate::{Error, Result};

#[path = "tiling_equations.rs"]
mod equations;

#[path = "tiling_host_equations.rs"]
mod host_equations;

pub const SW_LINEAR: u32 = 0;
pub const SW_256B_S: u32 = 1;
pub const SW_256B_D: u32 = 2;
pub const SW_4KB_S: u32 = 5;
pub const SW_4KB_D: u32 = 6;
pub const SW_64KB_S: u32 = 9;
pub const SW_64KB_D: u32 = 10;
pub const SW_64KB_S_T: u32 = 17;
pub const SW_64KB_D_T: u32 = 18;
pub const SW_4KB_S_X: u32 = 21;
pub const SW_4KB_D_X: u32 = 22;
pub const SW_64KB_Z_X: u32 = 24;
pub const SW_64KB_S_X: u32 = 25;
pub const SW_64KB_D_X: u32 = 26;
pub const SW_64KB_R_X: u32 = 27;

const MAX_EXTENT: u32 = 16384;
const MAX_SLICES: u32 = 8192;
const MAX_MIP_LEVELS: u32 = 16;

const MODE_NAMES: [&str; 32] = [
    "SW_LINEAR",
    "SW_256B_S",
    "SW_256B_D",
    "SW_256B_R",
    "SW_4KB_Z",
    "SW_4KB_S",
    "SW_4KB_D",
    "SW_4KB_R",
    "SW_64KB_Z",
    "SW_64KB_S",
    "SW_64KB_D",
    "SW_64KB_R",
    "SW_VAR_Z",
    "SW_VAR_S",
    "SW_VAR_D",
    "SW_VAR_R",
    "SW_64KB_Z_T",
    "SW_64KB_S_T",
    "SW_64KB_D_T",
    "SW_64KB_R_T",
    "SW_4KB_Z_X",
    "SW_4KB_S_X",
    "SW_4KB_D_X",
    "SW_4KB_R_X",
    "SW_64KB_Z_X",
    "SW_64KB_S_X",
    "SW_64KB_D_X",
    "SW_64KB_R_X",
    "SW_VAR_Z_X",
    "SW_VAR_S_X",
    "SW_VAR_D_X",
    "SW_VAR_R_X",
];

/// log2 of the {width, height, depth} in elements of thick 4 KiB and 64 KiB blocks, by element size.
const THICK_4KB_LOG2: [[u32; 3]; 5] = [[4, 4, 4], [3, 4, 4], [3, 4, 3], [3, 3, 3], [2, 3, 3]];
const THICK_64KB_LOG2: [[u32; 3]; 5] = [[6, 5, 5], [5, 5, 5], [5, 5, 4], [5, 4, 4], [4, 4, 4]];
/// {width, height} of the 256-byte micro block of thick swizzles, which places their mip tail slots.
const THICK_MICRO_BLOCK: [[u32; 2]; 5] = [[8, 4], [4, 4], [4, 4], [4, 2], [2, 2]];

/// Describes one guest surface allocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SurfaceDesc {
    /// gfx10 SW_MODE (CB_COLORn_ATTRIB3.COLOR_SW_MODE, DB_Z_INFO.SW_MODE, T# SW_MODE).
    pub swizzle_mode: u32,
    /// Level-0 extent in elements (texels, or blocks for block-compressed formats).
    pub width: u32,
    pub height: u32,
    /// Depth slices of a volume surface; 1 (or 0) otherwise.
    pub depth: u32,
    /// Array layers of a thin surface; 1 (or 0) for volumes.
    pub layers: u32,
    /// Mip levels of the allocation (0 counts as 1).
    pub mip_levels: u32,
    /// Bytes per element (texel, or compressed block).
    pub bytes_per_element: u32,
    /// Row pitch in elements of single-level linear surfaces (0: derived from the width).
    pub pitch_elements: u32,
    /// True for volume (3D) surfaces.
    pub volume: bool,
    /// Pipe/bank XOR in 256-byte units, as the descriptor ORs it into its 256-byte aligned base
    /// address (0 when absent). Only XOR swizzle modes (17, 18, 21, 22, 24..=27) take one.
    pub pipe_bank_xor: u32,
}

/// Where one mip level lies inside each array layer (thin surfaces), slab of block-depth slices
/// (thick volumes) or depth slice (linear and thin volumes).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MipLevel {
    /// Byte offset from the start of the layer, slab or slice.
    pub offset: u64,
    /// Bytes the level takes there; the levels in the mip tail share one block.
    pub size: u64,
    /// Row pitch: elements of a linear surface, blocks of a tiled one.
    pub pitch: u32,
    /// Element origin of a level packed into the mip tail block.
    pub tail_origin: Option<[u32; 2]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Tiling {
    Linear,
    /// One block per layer or depth slice; the swizzle mixes in the slice index.
    Thin,
    /// Blocks span several depth slices of a volume.
    Thick,
}

/// Address computation for a [`SurfaceDesc`].
#[derive(Clone, Debug)]
pub struct SurfaceLayout {
    desc: SurfaceDesc,
    tiling: Tiling,
    block_bytes: u64,
    /// Block extent in elements: width, height, depth.
    block: [u32; 3],
    swizzle: Swizzle,
    /// Pipe/bank XOR in bytes.
    xor: u32,
    mips: Vec<MipLevel>,
    slice_stride: u64,
    size: u64,
}

/// The GPU whose address equations a [`SurfaceLayout`] follows.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Addressing {
    /// The PS5 GPU (gfx10, 16 pipes, no RB+): guest surfaces.
    #[default]
    Guest,
    /// The host GPU (gfx11, GB_ADDR_CONFIG 0x343): host-layout copies in a mode that
    /// [`host_swizzle_mode`] picks.
    Host,
}

/// What a guest surface is bound as, which decides its host-layout swizzle mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceUse {
    /// 2D, cube or 2D-array texture or colour target.
    Thin2d,
    OneD,
    Msaa,
    Volume,
    Depth,
}

/// Swizzle mode of the host-layout copy of a guest surface (`FINAL-REPORT.md` §2-D): the mode the
/// shader translation's T# conversion selects ([`shader::descriptors::host_swizzle_mode`]), and
/// 64KB_Z_X for depth, the only one gfx11's depth unit accepts.
pub fn host_swizzle_mode(guest: u32, usage: SurfaceUse) -> u32 {
    const TYPE_1D: u8 = 8;
    const TYPE_2D: u8 = 9;
    const TYPE_3D: u8 = 10;
    const TYPE_2D_MSAA: u8 = 14;
    let ty = match usage {
        SurfaceUse::Depth => return SW_64KB_Z_X,
        SurfaceUse::Thin2d => TYPE_2D,
        SurfaceUse::OneD => TYPE_1D,
        SurfaceUse::Msaa => TYPE_2D_MSAA,
        SurfaceUse::Volume => TYPE_3D,
    };
    u32::from(shader::descriptors::host_swizzle_mode(
        ty,
        (guest & 0x1f) as u8,
    ))
}

impl SurfaceLayout {
    /// Proves identical byte addressing for every coordinate in this allocation by comparing
    /// the complete linear XOR maps and all block/mip/slice placement parameters. Equal
    /// sampled texels or equal SW_MODE values alone are insufficient.
    pub fn matches_host_addressing(&self) -> Result<bool> {
        let host = Self::host(&self.desc)?;
        Ok(self.tiling == host.tiling
            && self.block_bytes == host.block_bytes
            && self.block == host.block
            && self.swizzle == host.swizzle
            && self.xor == host.xor
            && self.mips == host.mips
            && self.slice_stride == host.slice_stride
            && self.size == host.size)
    }
    pub fn new(desc: &SurfaceDesc) -> Result<Self> {
        Self::with_addressing(desc, Addressing::Guest)
    }

    /// Whether `view` is a window on this layout: whole layers of it starting at `first_layer`,
    /// every element of the view at the byte position it has here. A 2D view of each face of a
    /// cube is one, so the views read and write one host copy.
    pub fn is_window(&self, view: &SurfaceLayout, first_layer: u32) -> bool {
        let (whole, part) = (&self.desc, &view.desc);
        let shared = part.mip_levels as usize;
        self.tiling == view.tiling
            && self.block_bytes == view.block_bytes
            && self.block == view.block
            && self.swizzle == view.swizzle
            && self.xor == view.xor
            && whole.bytes_per_element == part.bytes_per_element
            && (whole.width, whole.height, whole.depth) == (part.width, part.height, part.depth)
            && !whole.volume
            && !part.volume
            && first_layer + part.layers <= whole.layers
            && shared <= self.mips.len()
            && shared <= view.mips.len()
            && self.mips[..shared] == view.mips[..shared]
            && (part.layers == 1 || self.slice_stride == view.slice_stride)
    }

    /// The host-layout copy of a surface, `desc` already in its host swizzle mode. A pipe/bank
    /// XOR is masked to the host's pipe and bank bits ([`host_pipe_bank_xor`]), as addrlib and
    /// the hardware apply it.
    pub fn host(desc: &SurfaceDesc) -> Result<Self> {
        Self::with_addressing(desc, Addressing::Host)
    }

    pub fn with_addressing(desc: &SurfaceDesc, addressing: Addressing) -> Result<Self> {
        let mode = desc.swizzle_mode;
        let block_log2 = block_log2(mode).ok_or_else(|| {
            Error::unsupported(format!(
                "surface swizzle mode {mode} ({}) is not a gfx10 swizzle mode the PS5 GPU addresses",
                mode_name(mode)
            ))
        })?;
        // The host must use its own address equations. Same-named 3D S modes are not identical
        // across the modeled gfx10/16-pipe and gfx11/8-pipe configurations (addrlib oracle).
        if addressing == Addressing::Host {
            if mode != SW_LINEAR && host_equations::equation(mode, 2).is_none() {
                return Err(Error::unsupported(format!(
                    "swizzle mode {mode} ({}) is not a host-layout mode: host copies use linear, \
                     D, Z_X or R_X modes",
                    mode_name(mode)
                )));
            }
            if desc.volume && mode != SW_LINEAR {
                return Err(Error::unsupported(
                    "tiled host-layout volumes need the thick gfx11 equations, which are not \
                     generated",
                ));
            }
        }
        let (depth, layers, levels) = (
            desc.depth.max(1),
            desc.layers.max(1),
            desc.mip_levels.max(1),
        );
        if desc.width == 0
            || desc.height == 0
            || desc.width > MAX_EXTENT
            || desc.height > MAX_EXTENT
        {
            return Err(Error::invalid(format!(
                "surface extent {}x{} is outside 1..={MAX_EXTENT}",
                desc.width, desc.height
            )));
        }
        if depth > MAX_SLICES
            || layers > MAX_SLICES
            || (desc.volume && layers > 1)
            || (!desc.volume && depth > 1)
        {
            return Err(Error::invalid(format!(
                "surface with depth {depth} and {layers} array layers: volumes have one layer, \
                 thin surfaces one depth slice, each at most {MAX_SLICES}"
            )));
        }
        if levels > MAX_MIP_LEVELS {
            return Err(Error::invalid(format!(
                "surface mip count {levels} is out of range"
            )));
        }
        let bpe = desc.bytes_per_element;
        if bpe == 0 {
            return Err(Error::invalid("surface element size is zero"));
        }
        let element_log2 = (bpe.is_power_of_two() && bpe <= 16).then_some(bpe.trailing_zeros());
        if element_log2.is_none() && (mode != SW_LINEAR || bpe != 12) {
            return Err(Error::unsupported(format!(
                "{bpe}-byte elements are not supported in swizzle mode {mode} ({})",
                mode_name(mode)
            )));
        }
        let element_log2 = element_log2.unwrap_or(0);
        let tiling = if mode == SW_LINEAR {
            Tiling::Linear
        } else if !desc.volume || matches!(mode, SW_64KB_Z_X | SW_64KB_R_X) {
            Tiling::Thin
        } else if matches!(
            mode,
            SW_4KB_S | SW_4KB_S_X | SW_64KB_S | SW_64KB_S_X | SW_64KB_D_X
        ) {
            Tiling::Thick
        } else {
            return Err(Error::unsupported(format!(
                "3D surfaces in swizzle mode {mode} ({}) are not implemented: volumes are tiled thick \
                 in SW_4KB_S, SW_4KB_S_X, SW_64KB_S, SW_64KB_S_X and SW_64KB_D_X, thin in \
                 SW_64KB_Z_X and SW_64KB_R_X, or linear",
                mode_name(mode)
            )));
        };
        if desc.pitch_elements != 0
            && (tiling != Tiling::Linear || levels != 1 || desc.pitch_elements < desc.width)
        {
            return Err(Error::invalid(format!(
                "a row pitch of {} elements needs a single-level linear surface no wider than the pitch",
                desc.pitch_elements
            )));
        }
        let block_bytes = if tiling == Tiling::Linear {
            0
        } else {
            1u64 << block_log2
        };
        let xor = desc.pipe_bank_xor;
        if xor != 0 {
            if !is_xor(mode) {
                return Err(Error::invalid(format!(
                    "swizzle mode {mode} ({}) has no pipe/bank XOR, got {xor:#x}",
                    mode_name(mode)
                )));
            }
            if u64::from(xor) << 8 >= block_bytes {
                return Err(Error::invalid(format!(
                    "pipe/bank XOR {xor:#x} does not fit a {block_bytes}-byte block"
                )));
            }
            if addressing == Addressing::Guest && mode == SW_64KB_D_X && xor & 0x30 != 0 {
                return Err(Error::unsupported(format!(
                    "pipe/bank XOR {xor:#x} sets the column bits of an SW_64KB_D_X block, which is \
                     not implemented"
                )));
            }
        }
        let swizzle = match tiling {
            Tiling::Linear => Swizzle::default(),
            Tiling::Thin if matches!(mode, SW_256B_S | SW_4KB_S | SW_64KB_S) => {
                let mask = (block_bytes - 1) as u32;
                Swizzle::from_fn(|x, y| {
                    let offset = equations::standard_offset(x, y, bpe);
                    let extra = if block_log2 == 16 {
                        equations::standard_64kb_extra(x, y, bpe)
                    } else {
                        0
                    };
                    (offset ^ extra) & mask
                })
            }
            Tiling::Thin | Tiling::Thick => {
                let key = if tiling == Tiling::Thick {
                    0x100 | mode
                } else {
                    mode
                };
                let equation = match addressing {
                    Addressing::Host => host_equations::equation(mode, element_log2),
                    Addressing::Guest => equations::equation(key, element_log2),
                };
                let equation = equation.ok_or_else(|| {
                    Error::unsupported(format!(
                        "no swizzle equation for mode {mode} ({}) with {bpe}-byte elements",
                        mode_name(mode)
                    ))
                })?;
                Swizzle::from_equation(equation)
            }
        };
        let block = match tiling {
            Tiling::Linear => [1, 1, 1],
            Tiling::Thin => {
                let log2 = block_log2 - element_log2;
                [1 << log2.div_ceil(2), 1 << (log2 / 2), 1]
            }
            Tiling::Thick => {
                let table = if block_log2 == 12 {
                    &THICK_4KB_LOG2
                } else {
                    &THICK_64KB_LOG2
                };
                table[element_log2 as usize].map(|log2| 1 << log2)
            }
        };
        let mut mips = vec![MipLevel::default(); levels as usize];
        let mut offset = 0;
        if tiling == Tiling::Linear {
            for level in (0..levels).rev() {
                let pitch = if desc.pitch_elements != 0 {
                    desc.pitch_elements
                } else {
                    shift_ceil(desc.width, level).next_multiple_of(256 / bpe)
                };
                let size =
                    u64::from(pitch) * u64::from(shift_ceil(desc.height, level)) * u64::from(bpe);
                mips[level as usize] = MipLevel {
                    offset,
                    size,
                    pitch,
                    tail_origin: None,
                };
                offset += size;
            }
        } else {
            let tail = (block_log2 > 8 && levels > 1)
                .then(|| MipTail::new(mode, tiling, block_log2, element_log2, block))
                .and_then(|tail| {
                    let first = tail.first_level(desc.width, desc.height, levels)?;
                    Some((first, tail))
                });
            let mut untailed = levels;
            if let Some((first, tail)) = tail {
                for level in first..levels {
                    mips[level as usize] = MipLevel {
                        offset: 0,
                        size: block_bytes,
                        pitch: 1,
                        tail_origin: Some(tail.origin(level - first)),
                    };
                }
                untailed = first;
                offset = block_bytes;
            }
            for level in (0..untailed).rev() {
                let pitch = shift_ceil(desc.width, level).div_ceil(block[0]);
                let rows = shift_ceil(desc.height, level).div_ceil(block[1]);
                let size = u64::from(pitch) * u64::from(rows) * block_bytes;
                mips[level as usize] = MipLevel {
                    offset,
                    size,
                    pitch,
                    tail_origin: None,
                };
                offset += size;
            }
        }
        let slices = match (desc.volume, tiling) {
            (true, Tiling::Thick) => depth.div_ceil(block[2]),
            (true, _) => depth,
            (false, _) => layers,
        };
        let size = offset
            .checked_mul(u64::from(slices))
            .ok_or_else(|| Error::invalid("surface size overflows"))?;
        Ok(Self {
            desc: *desc,
            tiling,
            block_bytes,
            block,
            swizzle,
            xor: match addressing {
                Addressing::Guest => xor << 8,
                Addressing::Host => host_pipe_bank_xor(block_log2, xor) << 8,
            },
            mips,
            slice_stride: offset,
            size,
        })
    }

    pub fn desc(&self) -> &SurfaceDesc {
        &self.desc
    }

    /// Bytes of the whole allocation (every mip, layer and slice).
    pub fn size_bytes(&self) -> u64 {
        self.size
    }

    /// Bytes between consecutive array layers (thin surfaces), slabs of block-depth slices (thick
    /// volumes) or depth slices (linear and thin volumes); each holds the whole mip chain.
    pub fn slice_stride(&self) -> u64 {
        self.slice_stride
    }

    /// Placement of `mip` inside each layer, slab or slice.
    pub fn mip_level(&self, mip: u32) -> Option<&MipLevel> {
        self.mips.get(mip as usize)
    }

    pub fn is_linear(&self) -> bool {
        self.tiling == Tiling::Linear
    }

    /// True when blocks span several depth slices of a volume.
    pub fn is_thick(&self) -> bool {
        self.tiling == Tiling::Thick
    }

    /// Bytes of one block (0 for linear surfaces).
    pub fn block_bytes(&self) -> u64 {
        self.block_bytes
    }

    /// Block extent in elements: width, height, depth.
    pub fn block_extent(&self) -> [u32; 3] {
        self.block
    }

    /// Pipe/bank XOR in bytes, applied to every in-block offset.
    pub fn pipe_bank_xor_bytes(&self) -> u32 {
        self.xor
    }

    /// The in-block swizzle as per-offset-bit masks of the coordinate bits XORed into it: x bits
    /// in bits 0-11, y bits in 12-23 and slice bits in 24-31 (all zero for linear surfaces).
    pub fn equation(&self) -> [u32; 16] {
        let mut equation = [0u32; 16];
        let columns = self
            .swizzle
            .x
            .iter()
            .enumerate()
            .map(|(bit, &column)| (bit, column))
            .chain(
                self.swizzle
                    .y
                    .iter()
                    .enumerate()
                    .map(|(bit, &c)| (12 + bit, c)),
            )
            .chain(
                self.swizzle
                    .slice
                    .iter()
                    .enumerate()
                    .map(|(bit, &c)| (24 + bit, c)),
            );
        for (input, column) in columns {
            for (bit, mask) in equation.iter_mut().enumerate() {
                if (column >> bit) & 1 != 0 {
                    *mask |= 1 << input;
                }
            }
        }
        equation
    }

    /// Extent in elements of `mip`: [width, height, depth].
    pub fn mip_extent(&self, mip: u32) -> [u32; 3] {
        let level = |extent: u32| extent.checked_shr(mip).unwrap_or(0).max(1);
        [
            level(self.desc.width),
            level(self.desc.height),
            if self.desc.volume {
                level(self.desc.depth)
            } else {
                1
            },
        ]
    }

    /// Byte offset from the surface base of element (x, y) in `slice` (array layer, or depth
    /// slice of a volume) of `mip`. Coordinates may reach the level-0 extent shifted right and
    /// rounded up, which counts the 4x4 blocks of block-compressed levels.
    pub fn element_offset(&self, x: u32, y: u32, slice: u32, mip: u32) -> Result<u64> {
        let level = self.mips.get(mip as usize).ok_or_else(|| {
            Error::invalid(format!(
                "mip {mip} is outside the {}-level surface",
                self.mips.len()
            ))
        })?;
        let (width, height) = (
            shift_ceil(self.desc.width, mip),
            shift_ceil(self.desc.height, mip),
        );
        let slices = if self.desc.volume {
            (self.desc.depth.max(1) >> mip).max(1)
        } else {
            self.desc.layers.max(1)
        };
        if x >= width || y >= height || slice >= slices {
            return Err(Error::invalid(format!(
                "element ({x}, {y}) of slice {slice} lies outside mip {mip} ({width}x{height}, \
                 {slices} slices)"
            )));
        }
        if self.tiling == Tiling::Linear {
            let row = u64::from(y) * u64::from(level.pitch) + u64::from(x);
            return Ok(u64::from(slice) * self.slice_stride
                + level.offset
                + row * u64::from(self.desc.bytes_per_element));
        }
        let stride_slice = if self.tiling == Tiling::Thick {
            slice / self.block[2]
        } else {
            slice
        };
        let base = u64::from(stride_slice) * self.slice_stride + level.offset;
        let (block, in_block) = match level.tail_origin {
            Some([tail_x, tail_y]) => (0, self.swizzle.offset(x + tail_x, y + tail_y, slice)),
            None => (
                u64::from(y / self.block[1]) * u64::from(level.pitch)
                    + u64::from(x / self.block[0]),
                self.swizzle.offset(x, y, slice),
            ),
        };
        Ok(base + block * self.block_bytes + u64::from(in_block ^ self.xor))
    }
}

/// PS4 (GCN) 2D display tiling of one surface slice: 2D thin macro tiles of display-order micro
/// tiles on 8 pipes (P8_32x32_16x16).
#[derive(Clone, Debug)]
pub struct GcnDisplay2dLayout {
    width: u32,
    height: u32,
    block: [u32; 2],
    block_bytes: u64,
    blocks_per_row: u32,
    swizzle: Swizzle,
    size: u64,
}

impl GcnDisplay2dLayout {
    /// `width` is the row pitch in elements; rows and columns are padded to whole macro tiles.
    pub fn new(width: u32, height: u32, bytes_per_element: u32) -> Result<Self> {
        if width == 0 || height == 0 || width > MAX_EXTENT || height > MAX_EXTENT {
            return Err(Error::invalid(format!(
                "PS4 display surface extent {width}x{height} is outside 1..={MAX_EXTENT}"
            )));
        }
        if !bytes_per_element.is_power_of_two() || bytes_per_element > 16 {
            return Err(Error::unsupported(format!(
                "{bytes_per_element}-byte elements are not supported by the PS4 2D display tiling"
            )));
        }
        let log2 = bytes_per_element.trailing_zeros();
        let x = |bit: u32| 1u32 << bit;
        let y = |bit: u32| 1u32 << (12 + bit);
        let order = match log2 {
            0 => [x(0), x(1), x(2), y(1), y(0), y(2)],
            1 => [x(0), x(1), x(2), y(0), y(1), y(2)],
            2 => [x(0), x(1), y(0), x(2), y(1), y(2)],
            3 => [x(0), y(0), x(1), x(2), y(1), y(2)],
            _ => [x(0), y(0), x(1), y(1), x(2), y(2)],
        };
        // Bank height, bank count and macro tile aspect of the mode's macro tile at this element
        // size; a micro tile never splits since the tile split exceeds it.
        let (bank_height, banks, aspect): (u32, u32, u32) =
            [(4, 16, 4), (2, 16, 2), (1, 16, 2), (1, 16, 2), (1, 8, 1)][log2 as usize];
        let bank_height_log2 = bank_height.trailing_zeros();
        let block = [64 * aspect, 8 * bank_height * banks / aspect];
        let mut inner = [0u32; 16];
        inner[log2 as usize..log2 as usize + 6].copy_from_slice(&order);
        let mut inner_bits = 6 + log2 as usize;
        for bit in 0..bank_height_log2 {
            inner[inner_bits] = y(3 + bit);
            inner_bits += 1;
        }
        let tx = |bit: u32| x(6 + bit);
        let ty = |bit: u32| y(3 + bank_height_log2 + bit);
        let bank = if banks == 16 {
            [
                tx(0) | ty(3),
                tx(1) | ty(2) | ty(3),
                tx(2) | ty(1),
                tx(3) | ty(0),
            ]
        } else {
            [tx(0) | ty(2), tx(1) | ty(1) | ty(2), tx(2) | ty(0), 0]
        };
        let bank_bits = banks.trailing_zeros() as usize;
        let mut equation = [0u32; 16];
        equation[..8].copy_from_slice(&inner[..8]);
        equation[8..11].copy_from_slice(&[x(3) | y(3) | x(4), x(4) | y(4), x(5) | y(5)]);
        equation[11..11 + bank_bits].copy_from_slice(&bank[..bank_bits]);
        equation[11 + bank_bits..3 + bank_bits + inner_bits].copy_from_slice(&inner[8..inner_bits]);
        let block_bytes = u64::from(block[0]) * u64::from(block[1]) * u64::from(bytes_per_element);
        let blocks_per_row = width.div_ceil(block[0]);
        Ok(Self {
            width,
            height,
            block,
            block_bytes,
            blocks_per_row,
            swizzle: Swizzle::from_equation(&equation),
            size: u64::from(blocks_per_row) * u64::from(height.div_ceil(block[1])) * block_bytes,
        })
    }

    /// Macro tile extent in elements: width, height.
    pub fn block_extent(&self) -> [u32; 2] {
        self.block
    }

    /// Bytes of the padded surface.
    pub fn size_bytes(&self) -> u64 {
        self.size
    }

    pub fn element_offset(&self, x: u32, y: u32) -> Result<u64> {
        if x >= self.width || y >= self.height {
            return Err(Error::invalid(format!(
                "element ({x}, {y}) lies outside the {}x{} PS4 display surface",
                self.width, self.height
            )));
        }
        let block = u64::from(y / self.block[1]) * u64::from(self.blocks_per_row)
            + u64::from(x / self.block[0]);
        Ok(block * self.block_bytes + u64::from(self.swizzle.offset(x, y, 0)))
    }
}

/// An in-block byte offset as a linear map over GF(2): the offset bits each coordinate bit flips.
/// Coordinates enter whole, so XOR swizzles see the block position bits their pipes hash.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Swizzle {
    x: [u32; 12],
    y: [u32; 12],
    slice: [u32; 8],
}

impl Swizzle {
    fn from_equation(equation: &[u32; 16]) -> Self {
        let mut swizzle = Self::default();
        for (bit, &mask) in equation.iter().enumerate() {
            let flip = 1 << bit;
            for (i, column) in swizzle.x.iter_mut().enumerate() {
                if (mask >> i) & 1 != 0 {
                    *column |= flip;
                }
            }
            for (i, column) in swizzle.y.iter_mut().enumerate() {
                if (mask >> (12 + i)) & 1 != 0 {
                    *column |= flip;
                }
            }
            for (i, column) in swizzle.slice.iter_mut().enumerate() {
                if (mask >> (24 + i)) & 1 != 0 {
                    *column |= flip;
                }
            }
        }
        swizzle
    }

    fn from_fn(offset: impl Fn(u32, u32) -> u32) -> Self {
        Self {
            x: std::array::from_fn(|bit| offset(1 << bit, 0)),
            y: std::array::from_fn(|bit| offset(0, 1 << bit)),
            slice: [0; 8],
        }
    }

    fn offset(&self, x: u32, y: u32, slice: u32) -> u32 {
        flips(&self.x, x) ^ flips(&self.y, y) ^ flips(&self.slice, slice)
    }
}

fn flips(columns: &[u32], value: u32) -> u32 {
    columns
        .iter()
        .enumerate()
        .filter(|&(bit, _)| (value >> bit) & 1 != 0)
        .fold(0, |offset, (_, column)| offset ^ column)
}

/// The mip tail of a tiled mode: levels that fit `width` x `height` elements and are among the
/// last `slots` levels share one block, smaller levels in lower slots.
#[derive(Clone, Copy, Debug)]
struct MipTail {
    width: u32,
    height: u32,
    slots: u32,
    /// Micro block (256 bytes) extent: the unit of slot origins.
    micro: [u32; 2],
}

impl MipTail {
    fn new(mode: u32, tiling: Tiling, block_log2: u32, element_log2: u32, block: [u32; 3]) -> Self {
        let thick = tiling == Tiling::Thick;
        let (mut width, mut height) = (block[0], block[1]);
        match (thick, block_log2 % 3) {
            (false, _) | (true, 1) => width /= 2,
            (true, 0) => height /= 2,
            _ => {}
        }
        // The GPU's depth mipmap HTILE fix shrinks Z-order tails of 8- and 16-bit elements.
        if mode == SW_64KB_Z_X && element_log2 <= 1 {
            width /= 2;
            if element_log2 == 0 {
                height /= 2;
            }
        }
        let effective = if thick {
            block_log2 - (block_log2 - 8) / 3
        } else {
            block_log2
        };
        let slots = if effective <= 11 {
            1 + (1 << (effective - 9))
        } else {
            effective - 4
        };
        let micro = if thick {
            THICK_MICRO_BLOCK[element_log2 as usize]
        } else {
            let log2 = 8 - element_log2;
            [1 << log2.div_ceil(2), 1 << (log2 / 2)]
        };
        Self {
            width,
            height,
            slots,
            micro,
        }
    }

    /// The first of `levels` levels of a `width` x `height` surface that the tail holds.
    fn first_level(&self, width: u32, height: u32, levels: u32) -> Option<u32> {
        (0..levels).find(|&level| {
            shift_ceil(width, level) <= self.width
                && shift_ceil(height, level) <= self.height
                && levels - level <= self.slots
        })
    }

    /// Origin of the `index`-th level of the tail (0: the largest).
    fn origin(&self, index: u32) -> [u32; 2] {
        let slot = self.slots - 1 - index;
        let offset = if slot > 6 { 16 << slot } else { slot << 8 };
        let (mut x, mut y) = (0, 0);
        for bit in 0..6 {
            x |= ((offset >> (9 + 2 * bit)) & 1) << bit;
            y |= ((offset >> (8 + 2 * bit)) & 1) << bit;
        }
        [x * self.micro[0], y * self.micro[1]]
    }
}

fn block_log2(mode: u32) -> Option<u32> {
    Some(match mode {
        SW_LINEAR => 0,
        SW_256B_S | SW_256B_D => 8,
        SW_4KB_S | SW_4KB_D | SW_4KB_S_X | SW_4KB_D_X => 12,
        SW_64KB_S | SW_64KB_D | SW_64KB_S_T | SW_64KB_D_T | SW_64KB_Z_X | SW_64KB_S_X
        | SW_64KB_D_X | SW_64KB_R_X => 16,
        _ => return None,
    })
}

fn is_xor(mode: u32) -> bool {
    matches!(
        mode,
        SW_64KB_S_T
            | SW_64KB_D_T
            | SW_4KB_S_X
            | SW_4KB_D_X
            | SW_64KB_Z_X
            | SW_64KB_S_X
            | SW_64KB_D_X
            | SW_64KB_R_X
    )
}

/// The bits of a pipe/bank XOR (256-byte units) that the host GPU applies to blocks of
/// `1 << block_log2` bytes: its 3 pipe bits, then up to 4 bank bits above 2 column bits, the
/// mask addrlib's `Gfx11Lib` takes for GB_ADDR_CONFIG 0x343.
pub fn host_pipe_bank_xor(block_log2: u32, xor: u32) -> u32 {
    const INTERLEAVE_LOG2: u32 = 8;
    const PIPES_LOG2: u32 = 3;
    const COLUMN_BITS: u32 = 2;
    const BANK_BITS: u32 = 4;
    let bank_bits = block_log2
        .saturating_sub(INTERLEAVE_LOG2 + PIPES_LOG2 + COLUMN_BITS)
        .min(BANK_BITS);
    let pipes = (1 << PIPES_LOG2) - 1;
    let banks = ((1 << bank_bits) - 1) << (PIPES_LOG2 + COLUMN_BITS);
    xor & (pipes | banks)
}

/// Splits a surface base address into its block-aligned part and the pipe/bank XOR, in 256-byte
/// units, that XOR swizzle modes carry in the address bits below their block size.
pub fn split_pipe_bank_xor(mode: u32, address: u64) -> (u64, u32) {
    let mask = match block_log2(mode) {
        Some(log2) if is_xor(mode) => (1u64 << log2) - 1,
        _ => 0,
    };
    (address & !mask, ((address & mask) >> 8) as u32)
}

fn mode_name(mode: u32) -> &'static str {
    MODE_NAMES.get(mode as usize).copied().unwrap_or("invalid")
}

fn shift_ceil(value: u32, shift: u32) -> u32 {
    ((u64::from(value) + (1u64 << shift) - 1) >> shift) as u32
}
