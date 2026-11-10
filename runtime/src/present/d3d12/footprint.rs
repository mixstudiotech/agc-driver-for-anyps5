//! The placed-footprint rules a native pass follows when it writes a frame that D3D12 copies into
//! a back buffer with `CopyTextureRegion` (FINAL-REPORT §6, Milestone 1b).

use crate::{Error, Result};

/// D3D12_TEXTURE_DATA_PITCH_ALIGNMENT.
pub const PITCH_ALIGNMENT: u64 = 256;
/// D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT.
pub const PLACEMENT_ALIGNMENT: u64 = 512;
/// B8G8R8A8_UNORM, the back-buffer format.
pub const BYTES_PER_PIXEL: u64 = 4;
/// D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION.
pub const MAX_DIMENSION: u32 = 16_384;

/// Where a B8G8R8A8 frame lives in an interchange buffer: rows of `width` pixels at
/// `offset + y * row_pitch`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Footprint {
    pub offset: u64,
    pub width: u32,
    pub height: u32,
    pub row_pitch: u64,
}

impl Footprint {
    /// Offset 0 and the row pitch aligned to 256 bytes.
    pub fn bgra8(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(Error::invalid(format!(
                "frame {width}x{height} is outside 1..={MAX_DIMENSION}"
            )));
        }
        Ok(Self {
            offset: 0,
            width,
            height,
            row_pitch: (u64::from(width) * BYTES_PER_PIXEL).next_multiple_of(PITCH_ALIGNMENT),
        })
    }

    /// The bytes of one row that hold pixels; the rest of the pitch is padding.
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.width) * BYTES_PER_PIXEL
    }

    pub fn row_offset(&self, y: u32) -> u64 {
        self.offset + u64::from(y) * self.row_pitch
    }

    /// The buffer bytes a copy reads: Offset + RowPitch x (Height - 1) + Width x 4.
    pub fn required_bytes(&self) -> u64 {
        self.row_offset(self.height - 1) + self.row_bytes()
    }

    /// Checks a native write of `bytes` at `offset`: it must lie within one row's pixels, never
    /// in the pitch padding or past the frame.
    pub fn check_write(&self, offset: u64, bytes: u64) -> Result<()> {
        let relative = offset.checked_sub(self.offset);
        let fits = relative.is_some_and(|relative| {
            let y = relative / self.row_pitch;
            let x = relative % self.row_pitch;
            y < u64::from(self.height) && bytes <= self.row_bytes() && x <= self.row_bytes() - bytes
        });
        if fits {
            Ok(())
        } else {
            Err(Error::invalid(format!(
                "a {bytes}-byte write at 0x{offset:x} leaves the rows of a {}x{} frame (pitch {})",
                self.width, self.height, self.row_pitch
            )))
        }
    }
}
