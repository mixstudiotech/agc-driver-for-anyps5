//! Guest surface memory: gfx10 swizzle layouts and texel formats.

pub mod formats;
pub mod metadata;
pub mod retile;
pub mod tiling;

use crate::Result;

/// Byte access to guest memory used for surface reads and writes.
pub trait SurfaceMemory {
    fn read(&mut self, address: u64, out: &mut [u8]) -> Result<()>;
    fn write(&mut self, address: u64, data: &[u8]) -> Result<()>;
}
