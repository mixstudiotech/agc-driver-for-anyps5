//! The Agc modules (`libSceAgc`, `libSceAgcDriver` and the rest) implemented in the runtime
//! (PLAN P3.4, decisions D3, D4, D7).
//!
//! The recording functions write gfx11 PM4 straight into the command buffers the title owns: the
//! title sizes them with the `*GetSize` functions (which report these host packets) and changes
//! packets afterwards only through the Patch functions. The title links against [`shell`]
//! modules whose exports forward here; [`nid`] names those exports.

use crate::{Error, Result};
use std::ffi::c_void;

pub mod abi;
pub mod cache;
pub mod defaults_data;
pub mod driver;
pub mod exports;
pub mod guest;
pub mod nid;
pub mod record;
pub mod shell;
pub mod state;
pub mod target;
pub mod workloads;

/// The title's allocation callback: grows `buffer` to hold at least `dwords` more dwords.
pub type Grow = unsafe extern "sysv64" fn(buffer: *mut CommandBuffer, dwords: u32, user: *mut c_void) -> bool;

/// `CommandBuffer` (AnyPS5 `SceTypes.hpp`): packets go up from `cursor_up`; `cursor_down` (or
/// `top` when null) bounds them; `reserved_dw` dwords stay free for the library.
#[repr(C)]
#[derive(Debug)]
pub struct CommandBuffer {
    pub bottom: *mut u32,
    pub top: *mut u32,
    pub cursor_up: *mut u32,
    pub cursor_down: *mut u32,
    pub callback: Option<Grow>,
    pub user_data: *mut c_void,
    pub reserved_dw: u32,
}

/// `ShaderRegister`: a register offset and its value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShaderRegister {
    pub offset: u32,
    pub value: u32,
}

fn require(condition: bool, function: &str, reason: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::invalid(format!("{function}: {reason}")))
    }
}

/// A type-3 packet header of `count` dwords (the header included).
pub fn header(opcode: u32, count: u32, flags: u32) -> Result<u32> {
    require(
        (2..=0x4001).contains(&count) && opcode <= 0xff && flags <= 0xff,
        "Header",
        "invalid packet header",
    )?;
    Ok(0xc000_0000 | ((count - 2) << 16) | (opcode << 8) | flags)
}

/// Rejects `address` when null or not aligned to `alignment`.
pub fn check_address(address: u64, alignment: u64, function: &str) -> Result<()> {
    if address != 0 && address & (alignment - 1) == 0 {
        return Ok(());
    }
    Err(Error::invalid(format!(
        "{function}: null or misaligned address 0x{address:x}"
    )))
}

/// Rejects `value` with bits outside `mask`.
pub fn check_bits(value: u64, mask: u64, function: &str) -> Result<()> {
    if value & !mask == 0 {
        return Ok(());
    }
    Err(Error::invalid(format!(
        "{function}: reserved bits are set (value 0x{value:x}, allowed 0x{mask:x})"
    )))
}

/// Writes packets into title command buffers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Writer;

impl Writer {
    /// Dwords free between the cursors, less the reserved ones.
    fn available(buffer: &CommandBuffer, function: &str) -> Result<u32> {
        let bottom = buffer.bottom as usize;
        let top = buffer.top as usize;
        let up = buffer.cursor_up as usize;
        let down = if buffer.cursor_down.is_null() {
            top
        } else {
            buffer.cursor_down as usize
        };
        require((bottom | top | up | down) & 3 == 0, function, "misaligned command buffer")?;
        require(
            bottom <= up && up <= down && down <= top,
            function,
            "invalid command buffer cursors",
        )?;
        require(bottom != 0 || top == 0, function, "null command buffer storage")?;
        let count = ((down - up) / 4) as u64;
        let reserved = u64::from(buffer.reserved_dw);
        if count <= reserved {
            return Ok(0);
        }
        u32::try_from(count - reserved)
            .map_err(|_| Error::invalid(format!("{function}: command buffer capacity overflow")))
    }

    /// Makes room for `count` dwords, through the title's callback when the buffer is short.
    ///
    /// # Safety
    /// `buffer` points to a live command buffer whose cursors bound memory the caller may write;
    /// its callback, when called, keeps that true.
    pub unsafe fn reserve(&self, buffer: *mut CommandBuffer, count: u32, function: &str) -> Result<()> {
        require(!buffer.is_null() && count != 0, function, "null command buffer or empty allocation")?;
        // SAFETY: the caller's contract.
        let current = unsafe { &*buffer };
        if Self::available(current, function)? >= count {
            return Ok(());
        }
        let callback = current.callback;
        require(callback.is_some(), function, "command buffer exhausted")?;
        let wanted = count.checked_add(current.reserved_dw);
        require(wanted.is_some(), function, "command buffer allocation overflow")?;
        let user = current.user_data;
        // SAFETY: the title's callback with its own buffer and user data, as the console
        // library calls it.
        let grown = unsafe { callback.unwrap()(buffer, wanted.unwrap(), user) };
        if !grown {
            // SAFETY: the callback left a live command buffer behind.
            let after = unsafe { &*buffer };
            return Err(Error::invalid(format!(
                "{function}: command buffer allocation callback failed: {count} dwords wanted, buffer {:p}..{:p}, up cursor {:p}, down cursor {:p}, {} reserved",
                after.bottom, after.top, after.cursor_up, after.cursor_down, after.reserved_dw
            )));
        }
        // SAFETY: the callback left a live command buffer behind.
        let after = unsafe { &*buffer };
        require(
            Self::available(after, function)? >= count,
            function,
            "command buffer allocation callback returned insufficient space",
        )
    }

    /// Takes `count` dwords at the up cursor and returns where they start.
    ///
    /// # Safety
    /// As [`Self::reserve`].
    pub unsafe fn allocate(&self, buffer: *mut CommandBuffer, count: u32, function: &str) -> Result<*mut u32> {
        // SAFETY: the caller's contract.
        unsafe { self.reserve(buffer, count, function)? };
        // SAFETY: reserve proved `count` dwords free at the up cursor.
        let buffer = unsafe { &mut *buffer };
        let start = buffer.cursor_up;
        // SAFETY: within the reserved range.
        buffer.cursor_up = unsafe { start.add(count as usize) };
        Ok(start)
    }
}

/// The front end: every recording, and the target GPU they are written for (D13).
pub struct Agc {
    pub recordings: record::Recordings,
    pub target: Box<dyn target::Target>,
    pub writer: Writer,
}

impl Agc {
    pub fn new(target: Box<dyn target::Target>) -> Self {
        Self {
            recordings: record::Recordings::default(),
            target,
            writer: Writer,
        }
    }

    /// Writes `op` at `out` for a function that writes one command by itself
    /// (`sceAgcDriver*WritePacket`): into the recording that owns those commands when one does,
    /// so what it notes for submission (a cut, an event) is not lost.
    pub fn write_packet(&self, op: &target::Op<'_>, out: &mut [u32]) -> Result<()> {
        let address = out.as_mut_ptr() as u64;
        let mut result = None;
        self.recordings.with_recording_writing_at(address, |recording| result = Some(recording.recorder.encode_in_place(op, out, address)));
        match result {
            Some(result) => result,
            None => self.target.write(op, out),
        }
    }

    /// Records `op` at the up cursor of `buffer` for a function of the `queue` family and
    /// returns where its footprint starts (what the Agc function returns). A footprint the
    /// target fails to encode is left as filler and the error returned.
    ///
    /// # Safety
    /// As [`Writer::reserve`].
    pub unsafe fn emit(
        &self,
        buffer: *mut CommandBuffer,
        queue: target::Queue,
        op: &target::Op<'_>,
        function: &str,
    ) -> Result<*mut u32> {
        require(!buffer.is_null(), function, "null command buffer")?;
        let footprint = self.target.footprint(op.kind())?;
        // SAFETY: the caller's contract.
        let cursor = unsafe {
            record::Cursor {
                bottom: (*buffer).bottom as u64,
                up: (*buffer).cursor_up as u64,
            }
        };
        let mut claim = self.recordings.enter(buffer as usize, cursor, || self.target.record())?;
        if footprint == 0 {
            // SAFETY: the caller's contract.
            return Ok(unsafe { (*buffer).cursor_up });
        }
        // The title's allocation callback runs here, and may record commands of its own on this
        // buffer: they come first in the stream.
        // SAFETY: the caller's contract.
        let start = unsafe { self.writer.allocate(buffer, footprint, function)? };
        // SAFETY: `allocate` reserved `footprint` dwords at `start` for this command.
        let out = unsafe { std::slice::from_raw_parts_mut(start, footprint as usize) };
        let recording = claim.recording();
        // SAFETY: the caller's contract.
        unsafe { recording.recorder.set_bounds((*buffer).bottom as u64, (*buffer).top as u64) };
        if let Err(error) = recording.recorder.encode(queue, op, out, start as u64) {
            self.target.fill(out);
            return Err(error);
        }
        // SAFETY: the caller's contract; the callback may have moved the buffer to a new chunk,
        // which the recording continues in.
        claim.stand(unsafe { (*buffer).bottom as u64 }, start as u64 + u64::from(footprint) * 4);
        Ok(start)
    }
}
