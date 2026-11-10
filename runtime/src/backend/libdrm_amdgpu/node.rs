//! Render-node discovery: `/dev/dri/renderD*` nodes whose kernel driver is amdgpu.

use super::sys;
use crate::{Error, ErrorKind, Result};
use std::{
    ffi::CStr,
    fs::{self, File, OpenOptions},
    io,
    os::fd::AsRawFd,
    vec,
};

const DRI_DIRECTORY: &str = "/dev/dri";
const RENDER_NODE_PREFIX: &str = "renderD";
const AMDGPU_DRIVER: &str = "amdgpu";

/// The DRM interface version a kernel driver reports.
#[derive(Clone, Copy, Debug)]
pub struct DriverVersion {
    pub major: u16,
    pub minor: u16,
    pub patchlevel: u16,
}

/// A render node opened read/write whose kernel driver is amdgpu.
pub struct RenderNode {
    pub path: String,
    pub file: File,
    pub version: DriverVersion,
}

/// Yields the amdgpu render nodes in node order. Each node is opened only when the iteration
/// reaches it; nodes that are passed over are recorded so a failed open can say why.
pub struct RenderNodes {
    pending: vec::IntoIter<String>,
    skipped: Vec<String>,
}

impl RenderNodes {
    /// A missing `/dev/dri` means there are no DRM devices, which is not an error.
    pub fn scan() -> Result<Self> {
        let mut paths = Vec::new();
        match fs::read_dir(DRI_DIRECTORY) {
            Ok(entries) => {
                for entry in entries {
                    let entry = entry.map_err(|error| io_error(DRI_DIRECTORY, &error))?;
                    if let Ok(name) = entry.file_name().into_string() {
                        if name.starts_with(RENDER_NODE_PREFIX) {
                            paths.push(format!("{DRI_DIRECTORY}/{name}"));
                        }
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(io_error(DRI_DIRECTORY, &error)),
        }
        // Numeric node order: renderD128 < renderD129 < renderD1000.
        paths.sort_by(|a, b| (a.len(), a).cmp(&(b.len(), b)));
        Ok(Self {
            pending: paths.into_iter(),
            skipped: Vec::new(),
        })
    }

    /// The first amdgpu render node, or `Unsupported` listing why every render node was skipped.
    pub fn first_amdgpu(mut self) -> Result<RenderNode> {
        self.next().ok_or_else(|| {
            let reasons = if self.skipped.is_empty() {
                String::new()
            } else {
                format!(" ({})", self.skipped.join("; "))
            };
            Error::unsupported(format!(
                "no amdgpu render node under {DRI_DIRECTORY}{reasons}"
            ))
        })
    }
}

impl Iterator for RenderNodes {
    type Item = RenderNode;

    fn next(&mut self) -> Option<RenderNode> {
        for path in self.pending.by_ref() {
            let file = match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(file) => file,
                Err(error) => {
                    self.skipped.push(format!("{path}: {error}"));
                    continue;
                }
            };
            match drm_version(&file) {
                Ok((driver, version)) if driver == AMDGPU_DRIVER => {
                    return Some(RenderNode {
                        path,
                        file,
                        version,
                    });
                }
                Ok((driver, _)) => self.skipped.push(format!("{path}: driver {driver}")),
                Err(error) => self.skipped.push(format!("{path}: {error}")),
            }
        }
        None
    }
}

/// The kernel driver name and DRM interface version behind `file`.
fn drm_version(file: &File) -> Result<(String, DriverVersion)> {
    let raw = unsafe { sys::drmGetVersion(file.as_raw_fd()) };
    if raw.is_null() {
        return Err(Error::new(ErrorKind::System, "drmGetVersion failed"));
    }
    // SAFETY: a non-null result stays valid until drmFreeVersion, and libdrm keeps the driver
    // name as a NUL-terminated copy.
    let (driver, numbers) = unsafe {
        let version = &*raw;
        let driver = if version.name.is_null() {
            String::new()
        } else {
            CStr::from_ptr(version.name).to_string_lossy().into_owned()
        };
        let numbers = [
            version.version_major,
            version.version_minor,
            version.version_patchlevel,
        ];
        (driver, numbers)
    };
    unsafe { sys::drmFreeVersion(raw) };
    match numbers.map(|number| u16::try_from(number).ok()) {
        [Some(major), Some(minor), Some(patchlevel)] => Ok((
            driver,
            DriverVersion {
                major,
                minor,
                patchlevel,
            },
        )),
        _ => Err(Error::new(
            ErrorKind::System,
            format!("{driver} DRM version {numbers:?} is out of range"),
        )),
    }
}

fn io_error(path: &str, error: &io::Error) -> Error {
    Error::new(ErrorKind::System, format!("{path}: {error}"))
}
