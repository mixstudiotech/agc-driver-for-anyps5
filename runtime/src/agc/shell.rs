//! Export shells: the `.prx` files the title's loader opens (PLAN P3.4, decisions D7).
//!
//! AnyPS5's entry stub loads every module with `LoadLibraryExW(path, 0, 0x1100)` and resolves
//! each import with `GetProcAddress(module, nid)`. A shell is a PE32+ DLL without code: its export
//! table maps each NID to a forwarder string naming a function of the runtime DLL
//! (`runtime.sceAgcDcbDrawIndex`), so `GetProcAddress` returns the runtime's function itself and
//! the title calls it directly. The shell also imports one symbol of the runtime DLL, so
//! `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` (0x100) loads the runtime from the shell's own directory
//! before any forwarder is resolved; the forwarders then bind to that loaded module.

use crate::{Error, Result};

/// One export: `name` (the NID the title imports) forwards to `target` (`module.symbol`).
#[derive(Clone, Copy, Debug)]
pub struct Forward<'a> {
    pub name: &'a str,
    pub target: &'a str,
}

/// A symbol the shell imports, to make the loader load `dll` beside it.
#[derive(Clone, Copy, Debug)]
pub struct Dependency<'a> {
    pub dll: &'a str,
    pub symbol: &'a str,
}

const SECTION_RVA: u32 = 0x1000;
const SECTION_ALIGNMENT: u32 = 0x1000;
const FILE_ALIGNMENT: u32 = 0x200;
const HEADERS_SIZE: u32 = 0x200;
const PE_OFFSET: usize = 0x40;
const OPTIONAL_HEADER: usize = PE_OFFSET + 4 + 20;
const OPTIONAL_HEADER_SIZE: usize = 0xf0;
const SECTION_TABLE: usize = OPTIONAL_HEADER + OPTIONAL_HEADER_SIZE;
const IMAGE_BASE: u64 = 0x1_8000_0000;
const DIRECTORY_EXPORT: usize = 0;
const DIRECTORY_IMPORT: usize = 1;
const DIRECTORY_IAT: usize = 12;

fn align(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn put16(bytes: &mut [u8], at: usize, value: u16) {
    bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

fn put32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn check_text(text: &str, what: &str) -> Result<()> {
    if text.is_empty() || !text.is_ascii() || text.contains('\0') {
        return Err(Error::invalid(format!("export shell: invalid {what} {text:?}")));
    }
    Ok(())
}

/// The section's bytes, laid out from [`SECTION_RVA`].
struct Section {
    bytes: Vec<u8>,
}

impl Section {
    fn rva(&self) -> u32 {
        SECTION_RVA + self.bytes.len() as u32
    }

    fn reserve(&mut self, size: usize, alignment: usize) -> usize {
        while self.bytes.len() % alignment != 0 {
            self.bytes.push(0);
        }
        let at = self.bytes.len();
        self.bytes.resize(at + size, 0);
        at
    }

    fn string(&mut self, text: &str) -> u32 {
        let rva = self.rva();
        self.bytes.extend_from_slice(text.as_bytes());
        self.bytes.push(0);
        rva
    }
}

/// The shell DLL `dll_name` exporting `forwards`, importing `dependency` when given.
pub fn build(dll_name: &str, forwards: &[Forward], dependency: Option<Dependency>) -> Result<Vec<u8>> {
    check_text(dll_name, "DLL name")?;
    let mut sorted: Vec<&Forward> = forwards.iter().collect();
    sorted.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    for (index, forward) in sorted.iter().enumerate() {
        check_text(forward.name, "export name")?;
        check_text(forward.target, "forwarder")?;
        let dot = forward.target.find('.');
        if dot.is_none_or(|dot| dot == 0 || dot + 1 == forward.target.len()) {
            return Err(Error::invalid(format!(
                "export shell: forwarder {:?} is not module.symbol",
                forward.target
            )));
        }
        if index > 0 && sorted[index - 1].name == forward.name {
            return Err(Error::invalid(format!("export shell: {:?} is exported twice", forward.name)));
        }
    }
    let count = sorted.len();

    // Export directory, address table (forwarder RVAs), name pointers (sorted), ordinals, strings.
    let mut section = Section { bytes: Vec::new() };
    let directory = section.reserve(40, 4);
    let functions = section.reserve(4 * count, 4);
    let names = section.reserve(4 * count, 4);
    let ordinals = section.reserve(2 * count, 2);
    let dll_name_rva = section.string(dll_name);
    for (index, forward) in sorted.iter().enumerate() {
        let name = section.string(forward.name);
        let target = section.string(forward.target);
        put32(&mut section.bytes, names + 4 * index, name);
        put32(&mut section.bytes, functions + 4 * index, target);
        put16(&mut section.bytes, ordinals + 2 * index, index as u16);
    }
    let exports_size = (section.bytes.len() - directory) as u32;
    {
        let bytes = &mut section.bytes;
        put32(bytes, directory + 12, dll_name_rva);
        put32(bytes, directory + 16, 1);
        put32(bytes, directory + 20, count as u32);
        put32(bytes, directory + 24, count as u32);
        put32(bytes, directory + 28, SECTION_RVA + functions as u32);
        put32(bytes, directory + 32, SECTION_RVA + names as u32);
        put32(bytes, directory + 36, SECTION_RVA + ordinals as u32);
    }

    // Import of the runtime: one descriptor (and the terminator), lookup and address tables.
    let mut imports = None;
    if let Some(dependency) = dependency {
        check_text(dependency.dll, "dependency")?;
        check_text(dependency.symbol, "dependency symbol")?;
        let descriptors = section.reserve(40, 8);
        let lookup = section.reserve(16, 8);
        let addresses = section.reserve(16, 8);
        let hint = section.reserve(2, 2);
        let hint_rva = SECTION_RVA + hint as u32;
        section.string(dependency.symbol);
        let dll_rva = section.string(dependency.dll);
        let bytes = &mut section.bytes;
        put32(bytes, descriptors, SECTION_RVA + lookup as u32);
        put32(bytes, descriptors + 12, dll_rva);
        put32(bytes, descriptors + 16, SECTION_RVA + addresses as u32);
        put64(bytes, lookup, u64::from(hint_rva));
        put64(bytes, addresses, u64::from(hint_rva));
        imports = Some((SECTION_RVA + descriptors as u32, SECTION_RVA + addresses as u32));
    }

    let virtual_size = section.bytes.len() as u32;
    let raw_size = align(virtual_size, FILE_ALIGNMENT);
    let mut image = vec![0u8; (HEADERS_SIZE + raw_size) as usize];
    image[..2].copy_from_slice(b"MZ");
    put32(&mut image, 0x3c, PE_OFFSET as u32);
    image[PE_OFFSET..PE_OFFSET + 4].copy_from_slice(b"PE\0\0");
    let coff = PE_OFFSET + 4;
    put16(&mut image, coff, 0x8664); // AMD64
    put16(&mut image, coff + 2, 1); // one section
    put16(&mut image, coff + 16, OPTIONAL_HEADER_SIZE as u16);
    put16(&mut image, coff + 18, 0x2022); // executable image, large address aware, DLL
    let optional = OPTIONAL_HEADER;
    put16(&mut image, optional, 0x20b); // PE32+
    image[optional + 2] = 14;
    put32(&mut image, optional + 8, raw_size);
    put64(&mut image, optional + 0x18, IMAGE_BASE);
    put32(&mut image, optional + 0x20, SECTION_ALIGNMENT);
    put32(&mut image, optional + 0x24, FILE_ALIGNMENT);
    put16(&mut image, optional + 0x28, 6);
    put16(&mut image, optional + 0x30, 6);
    put32(&mut image, optional + 0x38, SECTION_RVA + align(virtual_size.max(1), SECTION_ALIGNMENT));
    put32(&mut image, optional + 0x3c, HEADERS_SIZE);
    put16(&mut image, optional + 0x44, 2); // Windows GUI subsystem
    put16(&mut image, optional + 0x46, 0x0160); // high-entropy VA, dynamic base, NX
    put64(&mut image, optional + 0x48, 0x10_0000);
    put64(&mut image, optional + 0x50, 0x1000);
    put64(&mut image, optional + 0x58, 0x10_0000);
    put64(&mut image, optional + 0x60, 0x1000);
    put32(&mut image, optional + 0x6c, 16);
    let data_directory = |index: usize| optional + 0x70 + 8 * index;
    put32(&mut image, data_directory(DIRECTORY_EXPORT), SECTION_RVA + directory as u32);
    put32(&mut image, data_directory(DIRECTORY_EXPORT) + 4, exports_size);
    if let Some((descriptors, addresses)) = imports {
        put32(&mut image, data_directory(DIRECTORY_IMPORT), descriptors);
        put32(&mut image, data_directory(DIRECTORY_IMPORT) + 4, 40);
        put32(&mut image, data_directory(DIRECTORY_IAT), addresses);
        put32(&mut image, data_directory(DIRECTORY_IAT) + 4, 16);
    }
    let header = SECTION_TABLE;
    image[header..header + 6].copy_from_slice(b".rdata");
    put32(&mut image, header + 8, virtual_size);
    put32(&mut image, header + 12, SECTION_RVA);
    put32(&mut image, header + 16, raw_size);
    put32(&mut image, header + 20, HEADERS_SIZE);
    // Initialized data, readable; writable when the loader fills an import address table.
    let writable = if imports.is_some() { 0x8000_0000 } else { 0 };
    put32(&mut image, header + 36, 0x4000_0040 | writable);
    image[HEADERS_SIZE as usize..HEADERS_SIZE as usize + section.bytes.len()]
        .copy_from_slice(&section.bytes);
    Ok(image)
}

/// The exports of a shell `build` wrote: (name, forwarder) in name order.
pub fn read_exports(image: &[u8]) -> Result<Vec<(String, String)>> {
    let bad = |what: &str| Error::invalid(format!("export shell: {what}"));
    let u16_at = |at: usize| -> Result<u16> {
        image.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]])).ok_or_else(|| bad("truncated"))
    };
    let u32_at = |at: usize| -> Result<u32> {
        image
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| bad("truncated"))
    };
    if image.get(..2) != Some(b"MZ") || image.get(PE_OFFSET..PE_OFFSET + 4) != Some(b"PE\0\0") {
        return Err(bad("not a PE image"));
    }
    let section_rva = u32_at(SECTION_TABLE + 12)?;
    let section_raw = u32_at(SECTION_TABLE + 20)?;
    let offset = |rva: u32| -> Result<usize> {
        rva.checked_sub(section_rva).map(|delta| (section_raw + delta) as usize).ok_or_else(|| bad("RVA outside the section"))
    };
    let string_at = |rva: u32| -> Result<String> {
        let start = offset(rva)?;
        let end = image[start..].iter().position(|&b| b == 0).ok_or_else(|| bad("unterminated string"))?;
        Ok(String::from_utf8_lossy(&image[start..start + end]).into_owned())
    };
    let directory_rva = u32_at(OPTIONAL_HEADER + 0x70)?;
    let directory_size = u32_at(OPTIONAL_HEADER + 0x74)?;
    let directory = offset(directory_rva)?;
    let count = u32_at(directory + 24)? as usize;
    let functions = offset(u32_at(directory + 28)?)?;
    let names = offset(u32_at(directory + 32)?)?;
    let ordinals = offset(u32_at(directory + 36)?)?;
    let mut exports = Vec::with_capacity(count);
    for index in 0..count {
        let name = string_at(u32_at(names + 4 * index)?)?;
        let ordinal = u16_at(ordinals + 2 * index)? as usize;
        let function = u32_at(functions + 4 * ordinal)?;
        if !(directory_rva..directory_rva + directory_size).contains(&function) {
            return Err(bad("export is not a forwarder"));
        }
        exports.push((name, string_at(function)?));
    }
    Ok(exports)
}

/// The Windows loader calls AnyPS5's entry stub makes for every module.
#[cfg(windows)]
pub mod loader {
    use std::ffi::c_void;
    use std::path::Path;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }

    /// `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`, as the entry stub.
    pub const ENTRY_STUB_FLAGS: u32 = 0x1100;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// A loaded module, freed on drop.
    pub struct Module(*mut c_void);

    impl Module {
        /// Loads `path` as the entry stub does.
        pub fn load(path: &Path) -> Result<Self, String> {
            let name = wide(&path.to_string_lossy());
            // SAFETY: a NUL-terminated path; the flags need an absolute one, which the caller gives.
            let module = unsafe { LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), ENTRY_STUB_FLAGS) };
            if module.is_null() {
                // SAFETY: no arguments.
                return Err(format!("LoadLibraryExW({}) failed: error {}", path.display(), unsafe { GetLastError() }));
            }
            Ok(Self(module))
        }

        /// The address `GetProcAddress` gives for `name`, following forwarders.
        pub fn symbol(&self, name: &str) -> Option<usize> {
            let name: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
            // SAFETY: a live module handle and a NUL-terminated name.
            let address = unsafe { GetProcAddress(self.0, name.as_ptr()) };
            (!address.is_null()).then_some(address as usize)
        }
    }

    impl Drop for Module {
        fn drop(&mut self) {
            // SAFETY: the handle LoadLibraryExW returned.
            unsafe { FreeLibrary(self.0) };
        }
    }

    /// `GetProcAddress(GetModuleHandleW(module), name)` for a module already loaded.
    pub fn loaded_symbol(module: &str, name: &str) -> Option<usize> {
        let module = wide(module);
        let name: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
        // SAFETY: NUL-terminated strings; a null handle is checked before use.
        unsafe {
            let handle = GetModuleHandleW(module.as_ptr());
            if handle.is_null() {
                return None;
            }
            let address = GetProcAddress(handle, name.as_ptr());
            (!address.is_null()).then_some(address as usize)
        }
    }
}

