//! VideoOut presentation.
//!
//! The device worker converts render surfaces and captures overlays on the GPU. D3D12 copies
//! shared interchange slots and presents; there is no CPU pixel conversion or GDI fallback.


pub mod audit;
pub mod canvas;
pub mod d3d12;
mod display;
pub(crate) use display::display_buffer_size;
pub(crate) use display::guest_kernel_layout;
pub(crate) use display::host_kernel_layout;
pub(crate) use display::diagnostic_host_layout;
pub mod handoff;
pub mod kernels;
pub mod native;
pub mod overlay;

pub use canvas::Canvas;
pub use display::KernelLayout;

/// A VideoOut display buffer as AnyPS5 registers it (Execution/include/DisplayBuffer.hpp).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DisplayBufferDesc {
    pub address: u64,
    pub pixel_format: u64,
    pub width: u32,
    pub height: u32,
    /// 0: PS5 64 KiB display tiling, 1: linear, 2: PS4 (GCN) 2D display tiling.
    pub tiling_mode: u32,
    pub pitch_in_pixel: u32,
    pub dcc_address: u64,
    pub dcc_clear_color: u64,
    pub alignment: u32,
}

impl Default for DisplayBufferDesc {
    fn default() -> Self {
        Self {
            address: 0,
            pixel_format: 0,
            width: 0,
            height: 0,
            tiling_mode: 0,
            pitch_in_pixel: 0,
            dcc_address: 0,
            dcc_clear_color: 0,
            alignment: 65536,
        }
    }
}

