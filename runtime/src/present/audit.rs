//! Whether Vulkan is in the process, and which DLLs a build output imports.
//!
//! A module is classified by the entry points it exports, never by its file name alone, so AMD's
//! D3D12 driver (`amdxx64.dll`, no Vulkan export) is not mistaken for its Vulkan driver
//! (`amdvlk64.dll`, which exports the ICD entry points). [`audit_process`] walks the modules loaded
//! in this process; [`PeFile`] reads the import tables of an executable on disk.

use crate::{Error, Result};

/// What a module is to the Vulkan loader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VulkanRole {
    Loader,
    Icd,
    Layer,
}

impl VulkanRole {
    pub fn name(self) -> &'static str {
        match self {
            Self::Loader => "loader",
            Self::Icd => "icd",
            Self::Layer => "layer",
        }
    }
}

const LOADER_NAME: &str = "vulkan-1.dll";
const ICD_EXPORTS: [&str; 2] = [
    "vk_icdGetInstanceProcAddr",
    "vk_icdNegotiateLoaderICDInterfaceVersion",
];
const LAYER_EXPORT: &str = "vkNegotiateLoaderLayerInterfaceVersion";
const LOADER_EXPORTS: [&str; 2] = ["vkCreateInstance", "vkGetInstanceProcAddr"];

/// The Vulkan role of a module named `name` (any case) that exports `exports`, if it has one.
pub fn classify<'a>(name: &str, exports: impl IntoIterator<Item = &'a str>) -> Option<VulkanRole> {
    let (mut icd, mut layer, mut loader) = (false, false, [false; 2]);
    for export in exports {
        icd |= ICD_EXPORTS.contains(&export);
        layer |= export == LAYER_EXPORT;
        for (found, wanted) in loader.iter_mut().zip(LOADER_EXPORTS) {
            *found |= export == wanted;
        }
    }
    if icd {
        Some(VulkanRole::Icd)
    } else if layer {
        Some(VulkanRole::Layer)
    } else if name.eq_ignore_ascii_case(LOADER_NAME) || loader == [true; 2] {
        Some(VulkanRole::Loader)
    } else {
        None
    }
}

/// Whether an imported DLL name is a Vulkan component a build output must not import.
pub fn is_vulkan_import(dll: &str) -> bool {
    let dll = dll.to_ascii_lowercase();
    dll == LOADER_NAME || dll.starts_with("vulkan") || dll.starts_with("amdvlk")
}

/// Whether a module name belongs to the D3D12/DXGI stack or a GPU vendor driver, for reports.
pub fn is_graphics_module(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    [
        "d3d", "dxgi", "dxcore", "amd", "atiu", "nvwgf", "igd", "vulkan",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

/// The modules a process audit inspected and what it found.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessAudit {
    pub modules: usize,
    /// Vulkan loaders, ICDs and layers, by module name.
    pub vulkan: Vec<(String, VulkanRole)>,
    /// D3D12, DXGI and vendor driver modules, by name.
    pub graphics: Vec<String>,
}

impl ProcessAudit {
    /// `name:role` entries, then (with `graphics`) the graphics modules, comma separated.
    pub fn report(&self, graphics: bool) -> String {
        let mut entries: Vec<String> = self
            .vulkan
            .iter()
            .map(|(name, role)| format!("{name}:{}", role.name()))
            .collect();
        if graphics {
            entries.extend(self.graphics.iter().cloned());
        }
        entries.join(",")
    }
}

/// Bytes of a PE image as a reader sees them: a file on disk, or an image the loader mapped.
trait ImageBytes {
    /// `len` bytes at relative virtual address `rva`, if the image holds them.
    fn at(&self, rva: u32, len: u32) -> Option<&[u8]>;
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

const DOS_MAGIC: u16 = 0x5a4d;
const NT_SIGNATURE: u32 = 0x0000_4550;
const PE32_MAGIC: u16 = 0x10b;
const PE32_PLUS_MAGIC: u16 = 0x20b;
const DIRECTORY_EXPORT: usize = 0;
const DIRECTORY_IMPORT: usize = 1;
const DIRECTORY_DELAY_IMPORT: usize = 13;
const MAX_NAME: u32 = 512;
const MAX_ENTRIES: usize = 1 << 16;

/// The parts of the PE headers the readers need.
#[derive(Clone, Debug)]
struct Headers {
    directories: Vec<(u32, u32)>,
    /// (virtual address, virtual size, raw offset, raw size) per section.
    sections: Vec<(u32, u32, u32, u32)>,
    headers_size: u32,
}

impl Headers {
    fn parse(header: &[u8]) -> Result<Self> {
        let malformed = |what: &str| Error::invalid(format!("not a PE image: {what}"));
        if u16_at(header, 0) != Some(DOS_MAGIC) {
            return Err(malformed("no MZ header"));
        }
        let nt = u32_at(header, 0x3c).ok_or_else(|| malformed("short DOS header"))? as usize;
        if u32_at(header, nt) != Some(NT_SIGNATURE) {
            return Err(malformed("no PE signature"));
        }
        let file = nt + 4;
        let sections = u16_at(header, file + 2).ok_or_else(|| malformed("short file header"))?;
        let optional_size =
            u16_at(header, file + 16).ok_or_else(|| malformed("short file header"))?;
        let optional = file + 20;
        let (count_at, directories_at) = match u16_at(header, optional) {
            Some(PE32_PLUS_MAGIC) => (optional + 108, optional + 112),
            Some(PE32_MAGIC) => (optional + 92, optional + 96),
            _ => return Err(malformed("unknown optional header")),
        };
        let headers_size =
            u32_at(header, optional + 60).ok_or_else(|| malformed("short optional header"))?;
        let count = u32_at(header, count_at).ok_or_else(|| malformed("short optional header"))?;
        let directories = (0..count.min(16) as usize)
            .map(|index| {
                let at = directories_at + index * 8;
                Some((u32_at(header, at)?, u32_at(header, at + 4)?))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| malformed("short data directories"))?;
        let table = optional + usize::from(optional_size);
        let sections = (0..usize::from(sections))
            .map(|index| {
                let at = table + index * 40;
                Some((
                    u32_at(header, at + 12)?,
                    u32_at(header, at + 8)?,
                    u32_at(header, at + 20)?,
                    u32_at(header, at + 16)?,
                ))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| malformed("short section table"))?;
        Ok(Self {
            directories,
            sections,
            headers_size,
        })
    }

    fn directory(&self, index: usize) -> Option<(u32, u32)> {
        self.directories
            .get(index)
            .copied()
            .filter(|&(rva, size)| rva != 0 && size != 0)
    }
}

/// A NUL-terminated ASCII name at `rva`.
fn name_at(image: &dyn ImageBytes, rva: u32) -> Option<String> {
    let mut name = Vec::new();
    for offset in 0..MAX_NAME {
        let byte = *image.at(rva.checked_add(offset)?, 1)?.first()?;
        if byte == 0 {
            return String::from_utf8(name).ok();
        }
        name.push(byte);
    }
    None
}

/// Names of the DLLs the import and delay-import tables reference.
fn imported_dlls(image: &dyn ImageBytes, headers: &Headers) -> Result<Vec<String>> {
    let malformed = |what: &str| Error::invalid(format!("malformed PE import table: {what}"));
    let mut dlls = Vec::new();
    // (directory, descriptor bytes, offset of the name RVA in a descriptor)
    for (directory, stride, name_offset) in [
        (DIRECTORY_IMPORT, 20u32, 12u32),
        (DIRECTORY_DELAY_IMPORT, 32, 4),
    ] {
        let Some((start, _)) = headers.directory(directory) else {
            continue;
        };
        for index in 0..MAX_ENTRIES as u32 {
            let at = start
                .checked_add(index * stride)
                .ok_or_else(|| malformed("descriptor address overflows"))?;
            let descriptor = image
                .at(at, stride)
                .ok_or_else(|| malformed("descriptor outside the image"))?;
            if descriptor.iter().all(|&byte| byte == 0) {
                break;
            }
            let name_rva = u32_at(descriptor, name_offset as usize).unwrap_or(0);
            let name =
                name_at(image, name_rva).ok_or_else(|| malformed("DLL name outside the image"))?;
            if !dlls
                .iter()
                .any(|known: &String| known.eq_ignore_ascii_case(&name))
            {
                dlls.push(name);
            }
        }
    }
    Ok(dlls)
}

/// Names the export directory lists.
fn exported_names(image: &dyn ImageBytes, headers: &Headers) -> Result<Vec<String>> {
    let malformed = |what: &str| Error::invalid(format!("malformed PE export table: {what}"));
    let Some((start, _)) = headers.directory(DIRECTORY_EXPORT) else {
        return Ok(Vec::new());
    };
    let directory = image
        .at(start, 40)
        .ok_or_else(|| malformed("directory outside the image"))?;
    let count = u32_at(directory, 24).unwrap_or(0) as usize;
    let names = u32_at(directory, 32).unwrap_or(0);
    if count > MAX_ENTRIES {
        return Err(malformed("too many names"));
    }
    (0..count as u32)
        .map(|index| {
            let entry = image
                .at(names.checked_add(index * 4)?, 4)
                .and_then(|bytes| u32_at(bytes, 0))?;
            name_at(image, entry)
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| malformed("name outside the image"))
}

/// An executable or DLL read from disk.
pub struct PeFile {
    bytes: Vec<u8>,
    headers: Headers,
}

impl ImageBytes for PeFile {
    fn at(&self, rva: u32, len: u32) -> Option<&[u8]> {
        let end = rva.checked_add(len)?;
        let offset = if end <= self.headers.headers_size {
            rva
        } else {
            let &(address, _, raw, raw_size) =
                self.headers
                    .sections
                    .iter()
                    .find(|&&(address, size, _, raw_size)| {
                        rva >= address && end <= address.saturating_add(size.max(raw_size))
                    })?;
            // Bytes past a section's raw data are zero fill that the file does not hold.
            if end - address > raw_size {
                return None;
            }
            raw.checked_add(rva - address)?
        };
        let start = offset as usize;
        self.bytes.get(start..start.checked_add(len as usize)?)
    }
}

impl PeFile {
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let headers = Headers::parse(&bytes)?;
        Ok(Self { bytes, headers })
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            Error::new(
                crate::ErrorKind::System,
                format!("cannot read {}: {error}", path.display()),
            )
        })?;
        Self::parse(bytes)
    }

    /// The DLLs this image imports, statically or delay-loaded, in table order.
    pub fn imports(&self) -> Result<Vec<String>> {
        imported_dlls(self, &self.headers)
    }

    pub fn exports(&self) -> Result<Vec<String>> {
        exported_names(self, &self.headers)
    }
}

#[cfg(all(feature = "wddm", target_os = "windows"))]
pub use loaded::audit_process;

#[cfg(all(feature = "wddm", target_os = "windows"))]
mod loaded {
    use super::*;
    use std::ptr;
    use windows_sys::Win32::{
        Foundation::{FreeLibrary, HMODULE},
        System::{
            LibraryLoader::{GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS},
            ProcessStatus::{
                K32EnumProcessModules, K32GetModuleBaseNameW, K32GetModuleInformation, MODULEINFO,
            },
            Threading::GetCurrentProcess,
        },
    };

    /// An image the loader mapped: relative virtual addresses are offsets from its base.
    struct Mapped {
        base: *const u8,
        size: u32,
    }

    impl ImageBytes for Mapped {
        fn at(&self, rva: u32, len: u32) -> Option<&[u8]> {
            let end = rva.checked_add(len)?;
            (end <= self.size).then(|| {
                // In bounds of the image, which stays loaded while the audit holds a reference.
                unsafe { std::slice::from_raw_parts(self.base.add(rva as usize), len as usize) }
            })
        }
    }

    /// A reference on a loaded module, so it cannot unload while the audit reads it.
    struct Pinned(HMODULE);

    impl Pinned {
        fn new(module: HMODULE) -> Option<Self> {
            let mut pinned: HMODULE = ptr::null_mut();
            let found = unsafe {
                GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                    module as *const u16,
                    &mut pinned,
                )
            };
            (found != 0 && pinned == module).then_some(Self(pinned))
        }
    }

    impl Drop for Pinned {
        fn drop(&mut self) {
            unsafe {
                FreeLibrary(self.0);
            }
        }
    }

    fn loaded_modules() -> Result<Vec<HMODULE>> {
        let process = unsafe { GetCurrentProcess() };
        let mut modules: Vec<HMODULE> = vec![ptr::null_mut(); 1024];
        loop {
            let mut needed = 0u32;
            let bytes = (modules.len() * std::mem::size_of::<HMODULE>()) as u32;
            if unsafe { K32EnumProcessModules(process, modules.as_mut_ptr(), bytes, &mut needed) }
                == 0
            {
                return Err(Error::new(
                    crate::ErrorKind::System,
                    "EnumProcessModules failed",
                ));
            }
            let count = needed as usize / std::mem::size_of::<HMODULE>();
            if needed <= bytes {
                modules.truncate(count);
                return Ok(modules);
            }
            modules.resize(count + 64, ptr::null_mut());
        }
    }

    fn module_name(module: HMODULE) -> Option<String> {
        let mut name = [0u16; 260];
        let length = unsafe {
            K32GetModuleBaseNameW(
                GetCurrentProcess(),
                module,
                name.as_mut_ptr(),
                name.len() as u32,
            )
        };
        (length > 0)
            .then(|| String::from_utf16_lossy(&name[..length as usize]).to_ascii_lowercase())
    }

    fn module_exports(module: HMODULE) -> Option<Vec<String>> {
        let mut info: MODULEINFO = unsafe { std::mem::zeroed() };
        let described = unsafe {
            K32GetModuleInformation(
                GetCurrentProcess(),
                module,
                &mut info,
                std::mem::size_of::<MODULEINFO>() as u32,
            )
        };
        if described == 0 {
            return None;
        }
        let image = Mapped {
            base: module as *const u8,
            size: info.SizeOfImage,
        };
        // The loader maps the headers in the image's first page.
        let headers = Headers::parse(image.at(0, info.SizeOfImage.min(0x1000))?).ok()?;
        exported_names(&image, &headers).ok()
    }

    /// Classifies every module loaded in this process by its exports. Modules that unload
    /// during the walk are skipped.
    pub fn audit_process() -> Result<ProcessAudit> {
        let mut audit = ProcessAudit::default();
        for module in loaded_modules()? {
            let Some(_pinned) = Pinned::new(module) else {
                continue;
            };
            let Some(name) = module_name(module) else {
                continue;
            };
            audit.modules += 1;
            let exports = module_exports(module).unwrap_or_default();
            if let Some(role) = classify(&name, exports.iter().map(String::as_str)) {
                audit.vulkan.push((name.clone(), role));
            }
            if is_graphics_module(&name) {
                audit.graphics.push(name);
            }
        }
        Ok(audit)
    }
}
