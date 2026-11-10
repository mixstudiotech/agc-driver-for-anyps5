//! The display-conversion kernel: the native queue converts a guest display buffer (PS5 64 KiB
//! tiling, linear or PS4 2D; 8- or 10-bit texels; a uniform DCC clear) into the presentation
//! canvas in a D3D12 interchange buffer, so D3D12 only copies and presents (FINAL-REPORT §2-E,
//! §6 Milestone 1b). A second pass draws the overlay plane over it as `present::overlay` does on
//! the CPU. The gfx1013 words (`display_convert_words.rs`) are assembled from
//! `display_convert.s` and translated for the host like guest shaders.

pub use super::display_convert_words::WORDS;

use crate::{
    present::{
        canvas::{Canvas, PlaneRect},
        display::KernelLayout,
        overlay::Sampling,
    },
    Error, Result,
};
use shader::{CompileOptions, Compiler, ShaderArtifact, ShaderInterface, ShaderStage, TargetIsa};

/// COMPUTE_PGM_RSRC1: 20 VGPRs (granules of 4), 88 SGPRs, FLOAT_MODE 0xc0.
pub const PGM_RSRC1: u32 = 0x000c_0000 | (10 << 6) | 4;
/// COMPUTE_PGM_RSRC2: two user SGPRs (the parameter block address) and the x and y group ids.
pub const PGM_RSRC2: u32 = (2 << 1) | (1 << 7) | (1 << 8);
/// Threads per group, along the canvas x axis.
pub const GROUP_THREADS: u32 = 64;
pub const PARAM_DWORDS: usize = 48;
/// The pixel of the bars around the image.
pub const BAR_PIXEL: u32 = 0xff00_0000;
/// The kernel mode that writes the fill pixel over the whole image.
const MODE_FILL: u32 = 4;
/// Parameter flag: draw over the destination with the texel's premultiplied alpha.
const FLAG_BLEND: u32 = 1;

/// Where a conversion writes: `canvas`, whose top-left pixel is at `destination`, rows
/// `row_pitch` bytes apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanvasTarget {
    pub destination: u64,
    pub row_pitch: u64,
    pub canvas: Canvas,
}

/// What a conversion draws: the `width` x `height` display buffer point-sampled into `rect` of
/// the canvas, over the destination with its premultiplied alpha when `blend` (the overlay
/// plane), or replacing it, with bars outside `rect`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlaneDraw {
    pub rect: PlaneRect,
    pub width: u32,
    pub height: u32,
    pub blend: bool,
}

/// One conversion's parameter block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConvertParams(pub [u32; PARAM_DWORDS]);

impl ConvertParams {
    /// Converts the display buffer of `layout` at `source` into the image of the canvas whose
    /// top-left pixel is at `destination`, rows `row_pitch` bytes apart, with bars around it. With
    /// `fill`, the image is that pixel (a uniform DCC clear) and the source is not read.
    pub fn new(
        layout: &KernelLayout,
        source: u64,
        destination: u64,
        row_pitch: u64,
        canvas: &Canvas,
        fill: Option<u32>,
    ) -> Result<Self> {
        let target = CanvasTarget {
            destination,
            row_pitch,
            canvas: *canvas,
        };
        let image = PlaneDraw {
            rect: PlaneRect {
                x: canvas.image_x,
                y: canvas.image_y,
                width: canvas.image_width,
                height: canvas.image_height,
            },
            width: canvas.image_width,
            height: canvas.image_height,
            blend: false,
        };
        Self::plane(layout, source, &target, &image, fill)
    }

    /// Draws the display buffer of `layout` at `source` into `target` as `draw` says.
    pub fn plane(
        layout: &KernelLayout,
        source: u64,
        target: &CanvasTarget,
        draw: &PlaneDraw,
        fill: Option<u32>,
    ) -> Result<Self> {
        let canvas = &target.canvas;
        let row_pitch = u32::try_from(target.row_pitch)
            .ok()
            .filter(|pitch| *pitch >= canvas.width * 4)
            .ok_or_else(|| Error::invalid("present: the canvas rows do not fit their pitch"))?;
        let rect = draw.rect;
        if rect.x + rect.width > canvas.width || rect.y + rect.height > canvas.height {
            return Err(Error::invalid(format!(
                "present: plane rectangle {rect:?} leaves the {}x{} canvas",
                canvas.width, canvas.height
            )));
        }
        let across = Sampling::new(draw.width, rect.width)?;
        let down = Sampling::new(draw.height, rect.height)?;
        let mut words = [0u32; PARAM_DWORDS];
        words[0] = source as u32;
        words[1] = (source >> 32) as u32;
        words[2] = layout.pitch;
        words[3] = layout.log2_block_width;
        words[4] = layout.log2_block_height;
        words[5] = layout.log2_block_bytes;
        words[6] = layout.xor;
        words[7] = if fill.is_some() {
            MODE_FILL
        } else {
            layout.mode
        };
        words[8..24].copy_from_slice(&layout.masks);
        words[24] = target.destination as u32;
        words[25] = (target.destination >> 32) as u32;
        words[26] = row_pitch;
        words[27] = canvas.width;
        words[28] = canvas.height;
        words[29] = rect.x;
        words[30] = rect.y;
        words[31] = rect.width;
        words[32] = rect.height;
        words[33] = fill.unwrap_or(0);
        words[34] = BAR_PIXEL;
        words[35] = across.step;
        words[36] = down.step;
        words[37] = draw.width;
        words[38] = draw.height;
        words[39] = if draw.blend { FLAG_BLEND } else { 0 };
        Ok(Self(words))
    }

    /// Dispatch groups: the canvas in 64-pixel row segments.
    pub fn groups(&self) -> [u32; 3] {
        [self.0[27].div_ceil(GROUP_THREADS), self.0[28], 1]
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.0.iter().flat_map(|word| word.to_le_bytes()).collect()
    }
}

/// The kernel translated for `target` with the guest register values above.
pub fn host_artifact(target: TargetIsa, wave_size: u8) -> Result<ShaderArtifact> {
    let interface = ShaderInterface {
        guest_pgm_rsrc1: Some(PGM_RSRC1),
        guest_pgm_rsrc2: Some(PGM_RSRC2),
        ..ShaderInterface::default()
    };
    Compiler::new(target)
        .compile_with_interface(
            &WORDS,
            CompileOptions::new(target, ShaderStage::Compute, wave_size),
            &interface,
        )
        .map_err(|error| {
            Error::invalid(format!(
                "present: the conversion kernel does not compile for {}: {error}",
                target.name()
            ))
        })
}
