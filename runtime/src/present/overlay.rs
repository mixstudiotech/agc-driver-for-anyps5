//! Sampling parameters consumed by the GPU display-conversion kernel.
use crate::{Error, Result};

/// Point sampling of `texels` texels across `pixels` pixels: pixel p reads texel
/// `(p × step + step / 2) >> 16`, clamped to the last texel, with `step = texels × 2^16 / pixels`.
/// Equal counts map each pixel to its own texel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sampling {
    pub step: u32,
    pub start: u32,
    pub last: u32,
}

impl Sampling {
    pub fn new(texels: u32, pixels: u32) -> Result<Self> {
        if texels == 0 || pixels == 0 || texels > 1 << 14 || pixels > 1 << 14 {
            return Err(Error::invalid(format!(
                "present: cannot sample {texels} texels across {pixels} pixels"
            )));
        }
        let step = ((u64::from(texels) << 16) / u64::from(pixels)) as u32;
        Ok(Self {
            step,
            start: step / 2,
            last: texels - 1,
        })
    }

    pub fn texel(&self, pixel: u32) -> u32 {
        ((pixel * self.step + self.start) >> 16).min(self.last)
    }
}
