//! The libSceAgc exports (PLAN P3.4, decisions D7, D14, D19): SysV functions with the signatures
//! the SDK's wrappers call, validating as the console library does and recording through the
//! front end. A failed call logs and returns null (no unwinding crosses the ABI); every
//! `*GetSize` returns exactly what its builder writes.

pub mod commands;
pub mod draw;
pub mod fusion;
pub mod misc;
pub mod patch;
pub mod shader;

use super::target::{Kind, Op, Queue};
use super::{Agc, CommandBuffer, ShaderRegister};
use crate::ir::RegisterBank;
use crate::{Error, Result};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

static AGC: OnceLock<Agc> = OnceLock::new();

/// The front end, created with the host's target on first use (D13: the target is chosen here).
pub fn agc() -> &'static Agc {
    AGC.get_or_init(|| Agc::new(crate::gpu::agc::default_target()))
}

/// Errors reported so far; later ones are counted, not printed.
static REPORTED: AtomicU32 = AtomicU32::new(0);
const REPORT_LIMIT: u32 = 64;

pub(crate) fn report(function: &str, message: &str) {
    if REPORTED.fetch_add(1, Ordering::Relaxed) < REPORT_LIMIT {
        eprintln!("[agc] {function}: {message}");
    }
}

/// Runs `body`, turning an error or a panic into a report and `fallback`.
pub(crate) fn guard<T>(function: &str, fallback: T, body: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            report(function, &error.message);
            fallback
        }
        Err(_) => {
            report(function, "panicked");
            fallback
        }
    }
}

/// Records `op` into `buffer` for `function`.
pub(crate) fn emit(buffer: *mut CommandBuffer, queue: Queue, op: &Op<'_>, function: &str) -> *mut u32 {
    // SAFETY: the title passes its own command buffer, as to the console library.
    guard(function, std::ptr::null_mut(), || unsafe { agc().emit(buffer, queue, op, function) })
}

/// Bytes an op of `kind` takes.
pub(crate) fn size_of(kind: Kind, function: &str) -> u32 {
    guard(function, 0, || {
        let dwords = agc().target.footprint(kind)?;
        dwords
            .checked_mul(4)
            .ok_or_else(|| Error::invalid(format!("{function}: size overflow")))
    })
}

/// A single register write (`Set*RegisterDirect`): the console's 16-bit offset limit.
fn register_direct(buffer: *mut CommandBuffer, bank: RegisterBank, register: ShaderRegister, function: &str) -> *mut u32 {
    if register.offset > 0xffff {
        report(function, "invalid register range");
        return std::ptr::null_mut();
    }
    let values = [register.value];
    let op = Op::SetRegisters { bank, offset: register.offset, count: 1, values: Some(&values) };
    emit(buffer, Queue::Graphics, &op, function)
}

/// A register list the GPU reads at execution (`Set*RegistersIndirect`).
fn registers_indirect(
    buffer: *mut CommandBuffer,
    bank: RegisterBank,
    registers: *const ShaderRegister,
    count: u32,
    function: &str,
) -> *mut u32 {
    let address = registers as u64;
    let valid = (address == 0 && count == 0) || (address != 0 && address % 4 == 0);
    if !valid || count > 0x3fff {
        report(function, "null or misaligned register list, or more than 0x3fff registers");
        return std::ptr::null_mut();
    }
    emit(buffer, Queue::Graphics, &Op::SetRegistersIndirect { bank, address, count }, function)
}

/// A run of consecutive registers (`Cb*RegisterRangeDirect`); without `values` the title fills
/// them later.
fn register_range(
    buffer: *mut CommandBuffer,
    bank: RegisterBank,
    offset: u32,
    values: *const u32,
    count: u32,
    function: &str,
) -> *mut u32 {
    if count == 0 || count > 0x3fff || offset > 0xffff || count > 0x1_0000 - offset {
        report(function, "invalid register range");
        return std::ptr::null_mut();
    }
    // SAFETY: the title passes `count` values (or null to fill them later).
    let values = (!values.is_null()).then(|| unsafe { std::slice::from_raw_parts(values, count as usize) });
    emit(buffer, Queue::Any, &Op::SetRegisters { bank, offset, count, values }, function)
}

/// (offset, value) pairs copied into the buffer (`CbSet*RegistersDirect`).
fn register_pairs(
    buffer: *mut CommandBuffer,
    bank: RegisterBank,
    registers: *const ShaderRegister,
    count: u32,
    function: &str,
) -> *mut u32 {
    if buffer.is_null() {
        report(function, "null command buffer");
        return std::ptr::null_mut();
    }
    if count == 0 {
        // SAFETY: checked non-null; nothing is written.
        return unsafe { (*buffer).cursor_up };
    }
    if registers.is_null() {
        report(function, "null register list");
        return std::ptr::null_mut();
    }
    // SAFETY: the title passes `count` pairs, read now as the console library does; the op does
    // not outlive this call.
    let pairs = unsafe { std::slice::from_raw_parts(registers, count as usize) };
    emit(buffer, Queue::Any, &Op::SetRegisterPairs { bank, pairs }, function)
}

fn pairs_size(count: u32, function: &str) -> u32 {
    if count > u32::MAX / 12 {
        report(function, "register list size overflow");
        return 0;
    }
    size_of(Kind::SetRegisterPairs { bank: RegisterBank::Shader, count }, function)
}

#[macro_export]
#[doc(hidden)]
macro_rules! agc_export {
    ($(#[$doc:meta])* fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty $body:block) => {
        $(#[$doc])*
        #[no_mangle]
        pub unsafe extern "sysv64" fn $name($($arg: $ty),*) -> $ret $body
    };
}

agc_export! { fn sceAgcDcbSetShRegisterDirect(buffer: *mut CommandBuffer, register: ShaderRegister) -> *mut u32 {
    register_direct(buffer, RegisterBank::Shader, register, "sceAgcDcbSetShRegisterDirect")
} }
agc_export! { fn sceAgcDcbSetCxRegisterDirect(buffer: *mut CommandBuffer, register: ShaderRegister) -> *mut u32 {
    register_direct(buffer, RegisterBank::Context, register, "sceAgcDcbSetCxRegisterDirect")
} }
agc_export! { fn sceAgcDcbSetUcRegisterDirect(buffer: *mut CommandBuffer, register: ShaderRegister) -> *mut u32 {
    register_direct(buffer, RegisterBank::UserConfig, register, "sceAgcDcbSetUcRegisterDirect")
} }
agc_export! { fn sceAgcDcbSetShRegisterDirectGetSize() -> u32 {
    size_of(Kind::SetRegisters { bank: RegisterBank::Shader, count: 1 }, "sceAgcDcbSetShRegisterDirectGetSize")
} }
agc_export! { fn sceAgcDcbSetCxRegisterDirectGetSize() -> u32 {
    size_of(Kind::SetRegisters { bank: RegisterBank::Context, count: 1 }, "sceAgcDcbSetCxRegisterDirectGetSize")
} }
agc_export! { fn sceAgcDcbSetUcRegisterDirectGetSize() -> u32 {
    size_of(Kind::SetRegisters { bank: RegisterBank::UserConfig, count: 1 }, "sceAgcDcbSetUcRegisterDirectGetSize")
} }

agc_export! { fn sceAgcDcbSetShRegistersIndirect(buffer: *mut CommandBuffer, registers: *const ShaderRegister, count: u32) -> *mut u32 {
    registers_indirect(buffer, RegisterBank::Shader, registers, count, "sceAgcDcbSetShRegistersIndirect")
} }
agc_export! { fn sceAgcDcbSetCxRegistersIndirect(buffer: *mut CommandBuffer, registers: *const ShaderRegister, count: u32) -> *mut u32 {
    registers_indirect(buffer, RegisterBank::Context, registers, count, "sceAgcDcbSetCxRegistersIndirect")
} }
agc_export! { fn sceAgcDcbSetUcRegistersIndirect(buffer: *mut CommandBuffer, registers: *const ShaderRegister, count: u32) -> *mut u32 {
    registers_indirect(buffer, RegisterBank::UserConfig, registers, count, "sceAgcDcbSetUcRegistersIndirect")
} }
agc_export! { fn sceAgcDcbSetShRegistersIndirectGetSize(_count: u32) -> u32 {
    size_of(Kind::SetRegistersIndirect { bank: RegisterBank::Shader }, "sceAgcDcbSetShRegistersIndirectGetSize")
} }
agc_export! { fn sceAgcDcbSetCxRegistersIndirectGetSize(_count: u32) -> u32 {
    size_of(Kind::SetRegistersIndirect { bank: RegisterBank::Context }, "sceAgcDcbSetCxRegistersIndirectGetSize")
} }
agc_export! { fn sceAgcDcbSetUcRegistersIndirectGetSize(_count: u32) -> u32 {
    size_of(Kind::SetRegistersIndirect { bank: RegisterBank::UserConfig }, "sceAgcDcbSetUcRegistersIndirectGetSize")
} }

agc_export! { fn sceAgcCbSetShRegisterRangeDirect(buffer: *mut CommandBuffer, offset: u32, values: *const u32, count: u32) -> *mut u32 {
    register_range(buffer, RegisterBank::Shader, offset, values, count, "sceAgcCbSetShRegisterRangeDirect")
} }
agc_export! { fn sceAgcCbSetUcRegisterRangeDirect(buffer: *mut CommandBuffer, offset: u32, values: *const u32, count: u32) -> *mut u32 {
    register_range(buffer, RegisterBank::UserConfig, offset, values, count, "sceAgcCbSetUcRegisterRangeDirect")
} }
agc_export! { fn sceAgcCbSetShRegisterRangeDirectGetSize(count: u32) -> u32 {
    size_of(Kind::SetRegisters { bank: RegisterBank::Shader, count: count }, "sceAgcCbSetShRegisterRangeDirectGetSize")
} }
agc_export! { fn sceAgcCbSetUcRegisterRangeDirectGetSize(count: u32) -> u32 {
    size_of(Kind::SetRegisters { bank: RegisterBank::UserConfig, count: count }, "sceAgcCbSetUcRegisterRangeDirectGetSize")
} }

agc_export! { fn sceAgcCbSetShRegistersDirect(buffer: *mut CommandBuffer, registers: *const ShaderRegister, count: u32) -> *mut u32 {
    register_pairs(buffer, RegisterBank::Shader, registers, count, "sceAgcCbSetShRegistersDirect")
} }
agc_export! { fn sceAgcCbSetUcRegistersDirect(buffer: *mut CommandBuffer, registers: *const ShaderRegister, count: u32) -> *mut u32 {
    register_pairs(buffer, RegisterBank::UserConfig, registers, count, "sceAgcCbSetUcRegistersDirect")
} }
agc_export! { fn sceAgcCbSetShRegistersDirectGetSize(count: u32) -> u32 {
    pairs_size(count, "sceAgcCbSetShRegistersDirectGetSize")
} }
agc_export! { fn sceAgcCbSetUcRegistersDirectGetSize(count: u32) -> u32 {
    pairs_size(count, "sceAgcCbSetUcRegistersDirectGetSize")
} }

/// The libSceAgc functions this module exports, for `agc::exports`.
pub const REGISTERS: &[&str] = &[
    "sceAgcDcbSetShRegisterDirect",
    "sceAgcDcbSetCxRegisterDirect",
    "sceAgcDcbSetUcRegisterDirect",
    "sceAgcDcbSetShRegisterDirectGetSize",
    "sceAgcDcbSetCxRegisterDirectGetSize",
    "sceAgcDcbSetUcRegisterDirectGetSize",
    "sceAgcDcbSetShRegistersIndirect",
    "sceAgcDcbSetCxRegistersIndirect",
    "sceAgcDcbSetUcRegistersIndirect",
    "sceAgcDcbSetShRegistersIndirectGetSize",
    "sceAgcDcbSetCxRegistersIndirectGetSize",
    "sceAgcDcbSetUcRegistersIndirectGetSize",
    "sceAgcCbSetShRegisterRangeDirect",
    "sceAgcCbSetUcRegisterRangeDirect",
    "sceAgcCbSetShRegisterRangeDirectGetSize",
    "sceAgcCbSetUcRegisterRangeDirectGetSize",
    "sceAgcCbSetShRegistersDirect",
    "sceAgcCbSetUcRegistersDirect",
    "sceAgcCbSetShRegistersDirectGetSize",
    "sceAgcCbSetUcRegistersDirectGetSize",
];
