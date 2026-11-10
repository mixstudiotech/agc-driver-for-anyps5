//! libSceAgc's register defaults, packet queries, semaphore memory and the other functions
//! that record nothing (PLAN P3.4 step 2): as the console library does them, with the SDK's
//! signatures (D19).

use super::report;
use crate::agc::defaults_data::{self, Table, INTERNAL, PUBLIC};
use crate::agc::ShaderRegister;
use crate::agc_export;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

/// `RegisterDefaults` as the SDK's `assignDefaults` reads it.
#[repr(C)]
struct RegisterDefaults {
    /// Per bank: pointers to its registers in pointer-table order (null without pointers).
    tables: [*mut *mut ShaderRegister; 4],
    /// Per bank: how many registers it has.
    counts: [u32; 4],
    /// The type entries, three dwords each (the "compact" layout).
    types: *const u32,
    count: u32,
}

const _: () = assert!(std::mem::size_of::<RegisterDefaults>() == 0x40);

/// One mutable copy of each source table for the whole process: the console library hands out
/// pointers into them, and every version sharing a table shares its copy.
static COPIES: [AtomicPtr<ShaderRegister>; defaults_data::TABLES] = [const { AtomicPtr::new(std::ptr::null_mut()) }; defaults_data::TABLES];

fn copy(table: usize, registers: &'static [(u32, u32)]) -> *mut ShaderRegister {
    let current = COPIES[table].load(Ordering::Acquire);
    if !current.is_null() {
        return current;
    }
    let made: Vec<ShaderRegister> = registers.iter().map(|&(offset, value)| ShaderRegister { offset, value }).collect();
    let made = Box::leak(made.into_boxed_slice()).as_mut_ptr();
    match COPIES[table].compare_exchange(std::ptr::null_mut(), made, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => made,
        // Another thread made it first; this copy leaks (once per table at most).
        Err(theirs) => theirs,
    }
}

fn build(table: &'static Table) -> RegisterDefaults {
    let mut defaults = RegisterDefaults { tables: [std::ptr::null_mut(); 4], counts: [0; 4], types: table.types.as_ptr(), count: (table.types.len() / 3) as u32 };
    for (index, bank) in table.banks.iter().enumerate() {
        let Some(bank) = bank else { continue };
        defaults.counts[index] = bank.registers.len() as u32;
        if bank.pointers.is_empty() {
            continue;
        }
        let registers = copy(bank.table, bank.registers);
        // SAFETY: the pointer indices lie inside the table (checked when it was converted).
        let pointers: Vec<*mut ShaderRegister> = bank.pointers.iter().map(|&at| unsafe { registers.add(usize::from(at)) }).collect();
        defaults.tables[index] = Box::leak(pointers.into_boxed_slice()).as_mut_ptr();
    }
    defaults
}

struct Built(*const RegisterDefaults);
// SAFETY: built once, never freed and never written by the runtime afterwards.
unsafe impl Send for Built {}
unsafe impl Sync for Built {}

static BUILT: [[OnceLock<Built>; 14]; 2] = [const { [const { OnceLock::new() }; 14] }; 2];

/// The defaults of `version` (internal or public), built once; null for versions past 13.
fn defaults(version: u32, internal: bool, function: &str) -> *const RegisterDefaults {
    if version >= 14 {
        report(function, "unsupported register defaults version");
        return std::ptr::null();
    }
    let table = if internal { INTERNAL[version as usize] } else { PUBLIC[version as usize] };
    BUILT[usize::from(internal)][version as usize].get_or_init(|| Built(Box::leak(Box::new(build(table))))).0
}

agc_export! { fn sceAgcGetRegisterDefaults2(version: u32) -> *const std::ffi::c_void {
    defaults(version, false, "sceAgcGetRegisterDefaults2") as *const std::ffi::c_void
} }
agc_export! { fn sceAgcGetRegisterDefaults2Internal(version: u32) -> *const std::ffi::c_void {
    defaults(version, true, "sceAgcGetRegisterDefaults2Internal") as *const std::ffi::c_void
} }

/// The default context state, 1024 registers: public V8's context bank by offset.
fn default_context() -> &'static [u32; 0x400] {
    static STATE: OnceLock<Box<[u32; 0x400]>> = OnceLock::new();
    STATE.get_or_init(|| {
        let mut state = Box::new([0u32; 0x400]);
        if let Some(bank) = &PUBLIC[8].banks[0] {
            for &(offset, value) in bank.registers {
                if let Some(slot) = state.get_mut(offset as usize) {
                    *slot = value;
                }
            }
        }
        state
    })
}

// `getDefaultCxState(dst, first, end)` checks `first < end <= 0x400` and passes the SDK's
// default-state word last (D19).
agc_export! { fn sceAgcGetDefaultCxStateFlat(destination: *mut u32, first: u32, end: u32, _state: u32) -> i32 {
    if destination.is_null() {
        return 0x8089_0001_u32 as i32;
    }
    if first >= end || end > 0x400 {
        return 0x8a6c_0044_u32 as i32;
    }
    let state = default_context();
    // SAFETY: the title passes room for registers `[first, end)`.
    unsafe { std::ptr::copy_nonoverlapping(state[first as usize..].as_ptr(), destination, (end - first) as usize) };
    0
} }

agc_export! { fn sceAgcInit_0090(state: *mut u32, version: u32) -> i32 {
    if state.is_null() || state as usize % 4 != 0 {
        report("sceAgcInit_0090", "null or misaligned state");
        return 0x8a6c_000a_u32 as i32;
    }
    if version < 14 {
        defaults(version, false, "sceAgcInit_0090");
        defaults(version, true, "sceAgcInit_0090");
    }
    // Recordings plan their draws against the device: it opens before the first one.
    if let Err(failure) = crate::agc::driver::Driver::get().host() {
        report("sceAgcInit_0090", &failure.message);
    }
    0
} }

agc_export! { fn sceAgcGetIsTrinityMode(trinity: *mut bool) -> i32 {
    if trinity.is_null() {
        report("sceAgcGetIsTrinityMode", "null output");
        return 0x8a6c_000a_u32 as i32;
    }
    // SAFETY: checked non-null.
    unsafe { *trinity = false };
    0
} }

static INSTRUMENTATION: AtomicU32 = AtomicU32::new(0);

agc_export! { fn sceAgcSetShaderInstrumentation(flags: u32) -> i32 {
    INSTRUMENTATION.store(flags, Ordering::Relaxed);
    0
} }
agc_export! { fn sceAgcGetShaderInstrumentation() -> u32 { INSTRUMENTATION.load(Ordering::Relaxed) } }

agc_export! { fn sceAgcDebugRaiseException(id: u32, _detail: u32) -> i32 {
    report("sceAgcDebugRaiseException", &format!("exception 0x{id:08x}"));
    0
} }

agc_export! { fn sceAgcSuspendPoint() -> i32 { 0 } }

agc_export! { fn sceAgcGetPacketSize(packet: *const u32) -> u32 {
    if packet.is_null() || packet as usize % 4 != 0 {
        report("sceAgcGetPacketSize", "null or misaligned packet");
        return 0;
    }
    // SAFETY: the title passes a packet it recorded.
    match unsafe { super::agc().target.command_dwords(packet) } {
        Some(dwords) => dwords,
        None => {
            report("sceAgcGetPacketSize", "source pointer is not a PM4 packet");
            0
        }
    }
} }

// Our footprints keep NOP payloads from dword 1 and register values from dword 2, where the
// console's packets keep them.
agc_export! { fn sceAgcGetDataPacketPayloadAddress_0090(address: *mut *mut u32, packet: *mut u32, kind: i32) -> i32 {
    if address.is_null() || packet.is_null() || packet as usize % 4 != 0 {
        report("sceAgcGetDataPacketPayloadAddress", "null or misaligned argument");
        return 0x8a6c_000a_u32 as i32;
    }
    // SAFETY: checked non-null; the title passes a packet it recorded.
    unsafe {
        *address = if kind != 0 {
            packet.add(2)
        } else if !*packet & 0x3fff_0000 != 0 {
            packet.add(1)
        } else {
            std::ptr::null_mut()
        };
    }
    0
} }

/// `SceAgcMemoryRange`.
#[repr(C)]
pub struct MemoryRange {
    pub base: *mut u32,
    pub size: u64,
}

agc_export! { fn sceAgcGetDataPacketPayloadRange(range: *mut MemoryRange, packet: *mut u32, kind: i32) -> i32 {
    if range.is_null() || packet.is_null() || packet as usize % 4 != 0 {
        report("sceAgcGetDataPacketPayloadRange", "null or misaligned argument");
        return 0x8a6c_000a_u32 as i32;
    }
    // SAFETY: checked non-null; the title passes a packet it recorded.
    unsafe {
        let header = *packet;
        let bytes = u64::from((header >> 14) & 0xfffc);
        *range = if kind != 0 {
            MemoryRange { base: packet.add(2), size: bytes }
        } else if (header >> 16) & 0x3fff == 0x3fff {
            MemoryRange { base: std::ptr::null_mut(), size: 0 }
        } else {
            MemoryRange { base: packet.add(1), size: bytes + 4 }
        };
    }
    0
} }

/// The semaphore labels (`sceAgcSetAmmSemaphoreMemory`): 32 bytes apart. Claimed once with CAS
/// on the size, then published with the address (D13: no locks).
static SEMAPHORE_BYTES: AtomicU64 = AtomicU64::new(0);
static SEMAPHORE_MEMORY: AtomicU64 = AtomicU64::new(0);

fn register_semaphores(memory: *mut u8, bytes: u64, limited: bool) -> i32 {
    if SEMAPHORE_BYTES.load(Ordering::Acquire) != 0 {
        return 0x8a6c_0048_u32 as i32;
    }
    if bytes % 0x4000 != 0 || memory as usize % 0x4000 != 0 {
        return 0x8a6c_0002_u32 as i32;
    }
    if limited && bytes > 0x4000_0000 {
        return 0x8a6c_000b_u32 as i32;
    }
    if bytes == 0 {
        return 0;
    }
    if memory.is_null() || (memory as u64).checked_add(bytes).is_none() {
        report("sceAgcSetAmmSemaphoreMemory", "null or wrapping semaphore memory");
        return 0x8a6c_000a_u32 as i32;
    }
    if SEMAPHORE_BYTES.compare_exchange(0, bytes, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return 0x8a6c_0048_u32 as i32;
    }
    // SAFETY: the title hands over `bytes` bytes at `memory`.
    unsafe { std::ptr::write_bytes(memory, 0, bytes as usize) };
    SEMAPHORE_MEMORY.store(memory as u64, Ordering::Release);
    0
}

agc_export! { fn sceAgcSetAmmSemaphoreMemory(memory: *mut u8, bytes: u64) -> i32 { register_semaphores(memory, bytes, false) } }
agc_export! { fn sceAgcSetSemaphoreMemory(memory: *mut u8, bytes: u64) -> i32 { register_semaphores(memory, bytes, true) } }
agc_export! { fn sceAgcGetSemaphoreLabel(index: u32, label: *mut *mut u8) -> i32 {
    let (bytes, memory) = (SEMAPHORE_BYTES.load(Ordering::Acquire), SEMAPHORE_MEMORY.load(Ordering::Acquire));
    if bytes == 0 || memory == 0 {
        return 0x8a6c_0049_u32 as i32;
    }
    if u64::from(index) >= bytes / 32 {
        return 0x8a6c_000b_u32 as i32;
    }
    if label.is_null() {
        report("sceAgcGetSemaphoreLabel", "null label output");
        return 0x8a6c_000a_u32 as i32;
    }
    // SAFETY: checked non-null.
    unsafe { *label = (memory + u64::from(index) * 32) as *mut u8 };
    0
} }

agc_export! { fn sceAgcCbMemsetExclusive(_buffer: *mut crate::agc::CommandBuffer, _address: u64, _value: u32, _bytes: u64) -> *mut u32 {
    report("sceAgcCbMemsetExclusive", "not implemented (nor in the console library the runtime replaces)");
    std::ptr::null_mut()
} }

/// The libSceAgc functions this module exports, for `agc::exports`.
pub const MISC: &[&str] = &[
    "sceAgcGetRegisterDefaults2", "sceAgcGetRegisterDefaults2Internal", "sceAgcGetDefaultCxStateFlat",
    "sceAgcInit_0090", "sceAgcGetIsTrinityMode", "sceAgcSetShaderInstrumentation", "sceAgcGetShaderInstrumentation",
    "sceAgcDebugRaiseException", "sceAgcSuspendPoint", "sceAgcGetPacketSize", "sceAgcGetDataPacketPayloadAddress_0090",
    "sceAgcGetDataPacketPayloadRange", "sceAgcSetAmmSemaphoreMemory", "sceAgcSetSemaphoreMemory", "sceAgcGetSemaphoreLabel",
    "sceAgcCbMemsetExclusive",
];
