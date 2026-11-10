#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(all(feature = "wddm", feature = "libdrm-amdgpu"))]
compile_error!("wddm and libdrm-amdgpu are mutually exclusive");

#[cfg(all(feature = "wddm", not(target_os = "windows")))]
compile_error!("wddm requires Windows");

#[cfg(all(feature = "libdrm-amdgpu", not(target_os = "linux")))]
compile_error!("libdrm-amdgpu requires Linux");

pub mod agc;
pub mod amd;
pub mod backend;
pub mod error;
pub mod gpu;
pub mod ir;
pub mod present;
pub mod residency;
pub mod scheduler;
pub mod types;
pub use error::{Error, ErrorKind, Result};
pub use shader::{ShaderStage, TargetIsa};

#[cfg(all(feature = "wddm", target_os = "windows"))]
pub use backend::wddm::{AdapterIdentity, Buffer, Device};

#[cfg(all(feature = "libdrm-amdgpu", target_os = "linux"))]
pub use backend::libdrm_amdgpu::{AdapterIdentity, Buffer, Device};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    None,
    Wddm,
    LibdrmAmdgpu,
}

impl BackendKind {
    pub fn selected() -> Self {
        if cfg!(all(feature = "wddm", target_os = "windows")) {
            Self::Wddm
        } else if cfg!(all(feature = "libdrm-amdgpu", target_os = "linux")) {
            Self::LibdrmAmdgpu
        } else {
            Self::None
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Wddm => "wddm",
            Self::LibdrmAmdgpu => "libdrm-amdgpu (runtime-unverified)",
        }
    }
}
