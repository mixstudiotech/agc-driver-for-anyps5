//! The libSceAgc draws and dispatches (PLAN P3.4 step 2e): validation as the console library's,
//! the guest state as the recording has it (D12). The SDK's wrappers call the draws only for
//! modifiers with bits 29 to 31 set and the dispatches only with modifier bit 0.

use super::{emit, guard, size_of};
use crate::agc::target::{Draw, DrawKind, DrawModifier, Kind, Op, Queue};
use crate::agc::{check_address, check_bits, CommandBuffer};
use crate::agc_export;
use crate::{Error, Result};

fn require(condition: bool, function: &str, reason: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::invalid(format!("{function}: {reason}")))
    }
}

fn modifier(modifier: u64, function: &str) -> Result<DrawModifier> {
    DrawModifier::decode(modifier).ok_or_else(|| {
        Error::invalid(format!("{function}: reserved bits are set (value 0x{modifier:x}, allowed 0x1ffffffff)"))
    })
}

fn draw(buffer: *mut CommandBuffer, function: &str, build: impl FnOnce() -> Result<(Draw, DrawModifier)>) -> *mut u32 {
    match guard(function, None, || build().map(Some)) {
        Some((draw, modifier)) => emit(buffer, Queue::Graphics, &Op::Draw { draw, modifier }, function),
        None => std::ptr::null_mut(),
    }
}

agc_export! { fn sceAgcDcbDrawIndex(buffer: *mut CommandBuffer, count: u32, address: u64, bits: u64) -> *mut u32 {
    const F: &str = "sceAgcDcbDrawIndex";
    draw(buffer, F, || {
        check_address(address, 1, F)?;
        Ok((Draw::Indexed { count, address }, modifier(bits, F)?))
    })
} }
agc_export! { fn sceAgcDcbDrawIndexAuto(buffer: *mut CommandBuffer, count: u32, bits: u64) -> *mut u32 {
    draw(buffer, "sceAgcDcbDrawIndexAuto", || Ok((Draw::Auto { count }, modifier(bits, "sceAgcDcbDrawIndexAuto")?)))
} }
agc_export! { fn sceAgcDcbDrawIndexOffset(buffer: *mut CommandBuffer, offset: u32, count: u32, bits: u64) -> *mut u32 {
    draw(buffer, "sceAgcDcbDrawIndexOffset", || {
        Ok((Draw::IndexedOffset { offset, count }, modifier(bits, "sceAgcDcbDrawIndexOffset")?))
    })
} }

fn indirect(indexed: bool, offset: u32, bits: u64, function: &str) -> Result<(Draw, DrawModifier)> {
    require(offset & 3 == 0, function, "misaligned indirect argument offset")?;
    Ok((Draw::Indirect { indexed, offset }, modifier(bits, function)?))
}

agc_export! { fn sceAgcDcbDrawIndexIndirect(buffer: *mut CommandBuffer, offset: u32, bits: u64) -> *mut u32 {
    draw(buffer, "sceAgcDcbDrawIndexIndirect", || indirect(true, offset, bits, "sceAgcDcbDrawIndexIndirect"))
} }
agc_export! { fn sceAgcDcbDrawIndirect(buffer: *mut CommandBuffer, offset: u32, bits: u64) -> *mut u32 {
    draw(buffer, "sceAgcDcbDrawIndirect", || indirect(false, offset, bits, "sceAgcDcbDrawIndirect"))
} }

#[allow(clippy::too_many_arguments)]
fn indirect_multi(indexed: bool, offset: u32, count_indirect: u32, count: u32, count_address: u64, stride: u32, bits: u64, function: &str) -> Result<(Draw, DrawModifier)> {
    check_bits(count_indirect.into(), 1, function)?;
    let minimum = if indexed { 20 } else { 16 };
    require(offset % 4 == 0 && stride % 4 == 0 && stride >= minimum, function, "invalid indirect draw offset or stride")?;
    if count_indirect != 0 {
        check_address(count_address, 4, function)?;
    } else {
        require(count_address == 0, function, "a direct count takes no count address")?;
    }
    let count_address = if count_indirect != 0 { count_address } else { 0 };
    Ok((Draw::IndirectMulti { indexed, offset, count, count_address, stride }, modifier(bits, function)?))
}

agc_export! { fn sceAgcDcbDrawIndexIndirectMulti(buffer: *mut CommandBuffer, offset: u32, count_indirect: u32, count: u32, count_address: u64, stride: u32, bits: u64) -> *mut u32 {
    const F: &str = "sceAgcDcbDrawIndexIndirectMulti";
    draw(buffer, F, || indirect_multi(true, offset, count_indirect, count, count_address, stride, bits, F))
} }
agc_export! { fn sceAgcDcbDrawIndirectMulti(buffer: *mut CommandBuffer, offset: u32, count_indirect: u32, count: u32, count_address: u64, stride: u32, bits: u64) -> *mut u32 {
    const F: &str = "sceAgcDcbDrawIndirectMulti";
    draw(buffer, F, || indirect_multi(false, offset, count_indirect, count, count_address, stride, bits, F))
} }
// DRAW_INDEX_MULTI_INST has no host equivalent the runtime knows the layout of.
agc_export! { fn sceAgcDcbDrawIndexMultiInstanced(_buffer: *mut CommandBuffer, _count: u32, _indices: u64, _objects: u64, _instances: u32, _bits: u64) -> *mut u32 {
    super::report("sceAgcDcbDrawIndexMultiInstanced", "DRAW_INDEX_MULTI_INST has no host form");
    std::ptr::null_mut()
} }

agc_export! { fn sceAgcDcbDrawIndexGetSize() -> u32 { size_of(Kind::Draw(DrawKind::Indexed), "sceAgcDcbDrawIndexGetSize") } }
agc_export! { fn sceAgcDcbDrawIndexAutoGetSize() -> u32 { size_of(Kind::Draw(DrawKind::Auto), "sceAgcDcbDrawIndexAutoGetSize") } }
agc_export! { fn sceAgcDcbDrawIndexOffsetGetSize() -> u32 { size_of(Kind::Draw(DrawKind::IndexedOffset), "sceAgcDcbDrawIndexOffsetGetSize") } }
agc_export! { fn sceAgcDcbDrawIndexIndirectGetSize() -> u32 { size_of(Kind::Draw(DrawKind::Indirect), "sceAgcDcbDrawIndexIndirectGetSize") } }
agc_export! { fn sceAgcDcbDrawIndirectGetSize() -> u32 { size_of(Kind::Draw(DrawKind::Indirect), "sceAgcDcbDrawIndirectGetSize") } }
agc_export! { fn sceAgcDcbDrawIndexIndirectMultiGetSize() -> u32 { size_of(Kind::Draw(DrawKind::IndirectMulti), "sceAgcDcbDrawIndexIndirectMultiGetSize") } }
agc_export! { fn sceAgcDcbDrawIndirectMultiGetSize() -> u32 { size_of(Kind::Draw(DrawKind::IndirectMulti), "sceAgcDcbDrawIndirectMultiGetSize") } }
agc_export! { fn sceAgcDcbDrawIndexMultiInstancedGetSize() -> u32 { 0 } }

/// The guest DISPATCH_INITIATOR bits a dispatch may carry (COMPUTE_SHADER_EN, ORDERED_APPEND,
/// USE_THREAD_DIMENSIONS, ORDER_MODE, TUNNEL_ENABLE, CS_W32_EN).
const DISPATCH_MODIFIER: u64 = 0xa079;

agc_export! { fn sceAgcCbDispatch(buffer: *mut CommandBuffer, x: u32, y: u32, z: u32, modifier: u32) -> *mut u32 {
    const F: &str = "sceAgcCbDispatch";
    match guard(F, None, || check_bits(modifier.into(), DISPATCH_MODIFIER, F).map(Some)) {
        Some(()) => emit(buffer, Queue::Any, &Op::Dispatch { groups: [x, y, z], modifier: modifier | 0x41 }, F),
        None => std::ptr::null_mut(),
    }
} }
agc_export! { fn sceAgcAcbDispatchIndirect(buffer: *mut CommandBuffer, address: u64, modifier: u32) -> *mut u32 {
    const F: &str = "sceAgcAcbDispatchIndirect";
    let valid = guard(F, None, || {
        check_address(address, 4, F)?;
        check_bits(modifier.into(), DISPATCH_MODIFIER, F).map(Some)
    });
    match valid {
        Some(()) => emit(buffer, Queue::Compute, &Op::DispatchIndirect { address, offset: 0, modifier: modifier | 0x41 }, F),
        None => std::ptr::null_mut(),
    }
} }
agc_export! { fn sceAgcDcbDispatchIndirect(buffer: *mut CommandBuffer, offset: u32, modifier: u32) -> *mut u32 {
    const F: &str = "sceAgcDcbDispatchIndirect";
    match guard(F, None, || check_bits(modifier.into(), DISPATCH_MODIFIER, F).map(Some)) {
        Some(()) => emit(buffer, Queue::Graphics, &Op::DispatchIndirect { address: 0, offset, modifier: modifier | 0x41 }, F),
        None => std::ptr::null_mut(),
    }
} }
agc_export! { fn sceAgcCbDispatchGetSize() -> u32 { size_of(Kind::Dispatch, "sceAgcCbDispatchGetSize") } }
agc_export! { fn sceAgcAcbDispatchIndirectGetSize() -> u32 { size_of(Kind::DispatchIndirect, "sceAgcAcbDispatchIndirectGetSize") } }
agc_export! { fn sceAgcDcbDispatchIndirectGetSize() -> u32 { size_of(Kind::DispatchIndirect, "sceAgcDcbDispatchIndirectGetSize") } }

/// The libSceAgc functions this module exports, for `agc::exports`.
pub const DRAWS: &[&str] = &[
    "sceAgcDcbDrawIndex", "sceAgcDcbDrawIndexAuto", "sceAgcDcbDrawIndexOffset", "sceAgcDcbDrawIndexIndirect",
    "sceAgcDcbDrawIndirect", "sceAgcDcbDrawIndexIndirectMulti", "sceAgcDcbDrawIndirectMulti",
    "sceAgcDcbDrawIndexMultiInstanced",
    "sceAgcDcbDrawIndexGetSize", "sceAgcDcbDrawIndexAutoGetSize", "sceAgcDcbDrawIndexOffsetGetSize",
    "sceAgcDcbDrawIndexIndirectGetSize", "sceAgcDcbDrawIndirectGetSize", "sceAgcDcbDrawIndexIndirectMultiGetSize",
    "sceAgcDcbDrawIndirectMultiGetSize", "sceAgcDcbDrawIndexMultiInstancedGetSize",
    "sceAgcCbDispatch", "sceAgcAcbDispatchIndirect", "sceAgcDcbDispatchIndirect",
    "sceAgcCbDispatchGetSize", "sceAgcAcbDispatchIndirectGetSize", "sceAgcDcbDispatchIndirectGetSize",
];
