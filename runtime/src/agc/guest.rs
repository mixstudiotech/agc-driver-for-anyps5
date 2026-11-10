//! Guest memory as the Agc modules see it (decisions D15): AnyPS5 maps guest memory at the same
//! virtual addresses in this process, so the front end and its targets read what the title
//! passes them in place, as the console library does.

use crate::{Error, Result};

/// Guest memory read in place.
pub trait GuestMemoryReader {
    fn read_words(&self, address: u64, count: usize) -> Result<Vec<u32>>;

    /// Fills `out` with the words at `address`, without allocating where the reader can.
    fn read_into(&self, address: u64, out: &mut [u32]) -> Result<()> {
        out.copy_from_slice(&self.read_words(address, out.len())?);
        Ok(())
    }

    /// Optional host inspection of a shader descriptor, whose branch may never execute.
    /// Unlike ordinary guest reads, an unbound or inaccessible address must not fault.
    fn probe_into(&self, _address: u64, _out: &mut [u32]) -> Result<()> {
        Err(Error::unsupported("optional guest memory inspection is unavailable"))
    }
}

/// Reads guest memory directly in this process.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessMemory;

impl ProcessMemory {
    fn check(address: u64, count: usize) -> Result<()> {
        if address == 0 || address % 4 != 0 {
            return Err(Error::invalid(format!(
                "guest read of {count} dwords at null or misaligned 0x{address:x}"
            )));
        }
        Ok(())
    }
}

impl GuestMemoryReader for ProcessMemory {
    fn read_words(&self, address: u64, count: usize) -> Result<Vec<u32>> {
        Self::check(address, count)?;
        // SAFETY: the title passed this address to an Agc function, which reads it in place on
        // the console too; AnyPS5 maps guest memory at the same address in this process.
        Ok(unsafe { std::slice::from_raw_parts(address as *const u32, count) }.to_vec())
    }

    fn read_into(&self, address: u64, out: &mut [u32]) -> Result<()> {
        Self::check(address, out.len())?;
        // SAFETY: as in `read_words`; `out` is the caller's own memory.
        unsafe { std::ptr::copy_nonoverlapping(address as *const u32, out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    #[cfg(all(feature = "wddm", target_os = "windows"))]
    fn probe_into(&self, address: u64, out: &mut [u32]) -> Result<()> {
        use windows_sys::Win32::System::{Diagnostics::Debug::ReadProcessMemory, Threading::GetCurrentProcess};
        Self::check(address, out.len())?;
        let bytes = std::mem::size_of_val(out);
        let mut copied = 0;
        // SAFETY: the OS validates the source; `out` has room for `bytes` and remains live.
        // A failed or partial copy is never consumed as a descriptor.
        let read = unsafe { ReadProcessMemory(GetCurrentProcess(), address as *const _, out.as_mut_ptr().cast(), bytes, &mut copied) };
        if read == 0 || copied != bytes {
            return Err(Error::invalid(format!("optional guest read of {bytes} bytes at 0x{address:x} is not readable")));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "wddm", target_os = "windows"))]
#[path = "../../tests/unit/agc/guest.rs"]
mod tests;
