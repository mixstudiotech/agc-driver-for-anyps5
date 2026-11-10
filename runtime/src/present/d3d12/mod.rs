//! Presentation through DXGI/D3D12 (FINAL-REPORT §2-E, §6 Milestone 1b).
//!
//! D3D12 owns a flip-discard swapchain and S = 2 linear interchange buffers, and only copies and
//! presents. Frame N uses slot N mod S once F_copy has reached N - S. With CPU-written upload
//! slots (the present thread decodes guest frames into them) that is the only fence; with slots
//! shared with the native device, the native queue also signals its shared monitored fence
//! F_ready = N and D3D12 waits for it before copying. The footprint, schedule and pattern parts
//! are plain data and build everywhere; the presenter needs the `wddm` feature on Windows.

pub mod footprint;
pub mod pattern;
pub mod schedule;

#[cfg(all(feature = "wddm", target_os = "windows"))]
mod presenter;

pub use footprint::Footprint;
pub use schedule::{CopyLedger, FenceLedger, FrameSchedule};

#[cfg(all(feature = "wddm", target_os = "windows"))]
pub use presenter::{
    committed_buffer_addresses, Adapter, D3d12Presenter, Output, PresentStatistics,
    PresenterConfig, SlotSource, SyncMode,
};
