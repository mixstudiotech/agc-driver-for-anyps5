use super::*;
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};

#[test]
fn optional_descriptor_reads_reject_unmapped_and_inaccessible_memory() {
    let mut words = [0; 4];
    assert!(ProcessMemory.probe_into(0x2b8, &mut words).is_err());
    // SAFETY: reserve two pages, commit only the first, and release the reservation below.
    let pages = unsafe { VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE, PAGE_NOACCESS) };
    assert!(!pages.is_null());
    assert!(!unsafe { VirtualAlloc(pages, 4096, MEM_COMMIT, PAGE_READWRITE) }.is_null());
    let expected = [0x11223344u32, 2, 3, 4];
    unsafe { std::ptr::copy_nonoverlapping(expected.as_ptr(), pages.cast(), expected.len()) };
    assert!(ProcessMemory.probe_into(pages as u64, &mut words).is_ok());
    assert_eq!(words, expected);
    assert!(ProcessMemory.probe_into(pages as u64 + 4096, &mut words).is_err());
    assert!(ProcessMemory.probe_into(pages as u64 + 4096 - 8, &mut words).is_err());
    assert_ne!(unsafe { VirtualFree(pages, 0, MEM_RELEASE) }, 0);
    assert!(ProcessMemory.probe_into(pages as u64, &mut words).is_err());
}
