//! Copies between the guest layout of a surface and its host-layout copy: the per-level parameters
//! the retile and detile utility kernels take.

use super::tiling::SurfaceLayout;
use crate::{Error, Result};
use std::ops::Range;

/// Dwords of the element-copy kernel's parameter block.
pub const COPY_PARAM_DWORDS: usize = 68;
/// Threads per workgroup of the element-copy kernel, all along x.
pub const COPY_GROUP_THREADS: u32 = 64;

/// One dispatch of the element-copy kernel: a level of a surface copied between two layouts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CopyDispatch {
    pub params: [u32; COPY_PARAM_DWORDS],
    pub groups: [u32; 3],
}

/// The dispatches that copy `levels` of a surface at `from_base` in layout `from` to `to_base`
/// in layout `to`, one per level, in the order of `levels`.
pub fn copy_dispatches(
    from: &SurfaceLayout,
    from_base: u64,
    to: &SurfaceLayout,
    to_base: u64,
    levels: Range<u32>,
) -> Result<Vec<CopyDispatch>> {
    check_pair(from, to)?;
    let element_bytes = from.desc().bytes_per_element;
    if !element_bytes.is_power_of_two() || element_bytes > 16 {
        return Err(Error::unsupported(format!(
            "the element-copy kernel moves 1, 2, 4, 8 or 16-byte elements, not {element_bytes}"
        )));
    }
    levels
        .map(|level| {
            let [width, height, slices] = level_extent(from, level);
            let mut params = [0u32; COPY_PARAM_DWORDS];
            params[..32].copy_from_slice(&layout_params(from, from_base, level)?);
            params[32..64].copy_from_slice(&layout_params(to, to_base, level)?);
            params[64] = width;
            params[65] = height;
            params[66] = element_bytes.trailing_zeros();
            Ok(CopyDispatch {
                params,
                groups: [width.div_ceil(COPY_GROUP_THREADS), height, slices],
            })
        })
        .collect()
}

/// Materializes one constant texel in every addressed element of every mip and layer. The
/// element-copy kernel's fill branch never reads source memory; `texel` is stored in otherwise
/// unused source-layout parameter words. Padding is left alone, like a normal image clear.
pub fn clear_dispatches(
    to: &SurfaceLayout,
    to_base: u64,
    texel: &[u8],
) -> Result<Vec<CopyDispatch>> {
    if texel.len() != to.desc().bytes_per_element as usize {
        return Err(Error::invalid("a surface clear needs one complete texel"));
    }
    let mut value = [0u8; 16];
    value
        .get_mut(..texel.len())
        .ok_or_else(|| {
            Error::unsupported("the surface-clear kernel supports texels of at most 16 bytes")
        })?
        .copy_from_slice(texel);
    let mut dispatches = copy_dispatches(to, to_base, to, to_base, 0..to.desc().mip_levels.max(1))?;
    for dispatch in &mut dispatches {
        for (word, bytes) in dispatch.params[11..15]
            .iter_mut()
            .zip(value.chunks_exact(4))
        {
            *word = u32::from_le_bytes(bytes.try_into().unwrap());
        }
        dispatch.params[67] = 1;
    }
    Ok(dispatches)
}

/// The 32 parameter dwords of one layout (see `gpu/kernels/element_copy.s`).
fn layout_params(layout: &SurfaceLayout, base: u64, level: u32) -> Result<[u32; 32]> {
    if layout.is_thick() {
        return Err(Error::unsupported(
            "the element-copy kernel does not address thick (3D-block) layouts",
        ));
    }
    if layout.slice_stride() > u64::from(u32::MAX) {
        return Err(Error::unsupported(format!(
            "the element-copy kernel offsets slices of at most 4 GiB, not {} bytes",
            layout.slice_stride()
        )));
    }
    let mip = layout
        .mip_level(level)
        .ok_or_else(|| Error::invalid(format!("mip {level} is outside the surface")))?;
    let start = base + mip.offset;
    let element_log2 = layout.desc().bytes_per_element.trailing_zeros();
    let (block_log2, block_bytes_log2) = if layout.is_linear() {
        ([0, 0], element_log2)
    } else {
        let [width, height, _] = layout.block_extent();
        (
            [width.trailing_zeros(), height.trailing_zeros()],
            layout.block_bytes().trailing_zeros(),
        )
    };
    let [tail_x, tail_y] = mip.tail_origin.unwrap_or([0, 0]);
    let mut words = [0u32; 32];
    words[..11].copy_from_slice(&[
        start as u32,
        (start >> 32) as u32,
        layout.slice_stride() as u32,
        (layout.slice_stride() >> 32) as u32,
        mip.pitch,
        block_log2[0],
        block_log2[1],
        block_bytes_log2,
        tail_x,
        tail_y,
        layout.pipe_bank_xor_bytes(),
    ]);
    words[16..].copy_from_slice(&layout.equation());
    Ok(words)
}

fn check_pair(from: &SurfaceLayout, to: &SurfaceLayout) -> Result<()> {
    let (a, b) = (from.desc(), to.desc());
    let same = a.width == b.width
        && a.height == b.height
        && a.depth.max(1) == b.depth.max(1)
        && a.layers.max(1) == b.layers.max(1)
        && a.mip_levels.max(1) == b.mip_levels.max(1)
        && a.bytes_per_element == b.bytes_per_element
        && a.volume == b.volume;
    if same {
        Ok(())
    } else {
        Err(Error::invalid(
            "a surface copy needs two layouts of the same extent, levels and element size",
        ))
    }
}

/// Elements a level addresses (block-compressed levels round up) and its slices.
fn level_extent(layout: &SurfaceLayout, level: u32) -> [u32; 3] {
    let desc = layout.desc();
    let shift = |extent: u32| ((u64::from(extent) + (1u64 << level) - 1) >> level) as u32;
    let slices = if desc.volume {
        (desc.depth.max(1) >> level).max(1)
    } else {
        desc.layers.max(1)
    };
    [shift(desc.width), shift(desc.height), slices]
}
