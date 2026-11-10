//! The libSceAgc patch and predication functions (PLAN P3.4 step 2g, decisions D5, D14, D19):
//! they change commands a title recorded earlier, through the target that recorded them. The
//! arguments are checked as the console library checks them; where the console library throws,
//! these report and return [`INVALID`].

use super::{agc, guard, report};
use crate::agc::target::Patch;
use crate::agc::{check_address, check_bits, ShaderRegister};
use crate::agc_export;
use crate::ir::RegisterBank;
use crate::Result;
use std::ffi::c_void;

/// What a patch returns when it cannot apply (the console library throws instead).
pub const INVALID: i32 = 0x8a6c_0001_u32 as i32;
/// What `sceAgcAcquireMemSetEngine` returns for a command that is not an acquire.
pub const NOT_AN_ACQUIRE: i32 = 0x8a6c_000c_u32 as i32;

/// Applies the patch `build` makes after checking its arguments; 0 on success.
fn apply(function: &str, command: *mut u32, build: impl FnOnce() -> Result<Patch>) -> i32 {
    guard(function, INVALID, || {
        let patch = build()?;
        // SAFETY: the title passes a command an Agc function returned (D5).
        unsafe { agc().target.patch(command, patch) }?;
        Ok(0)
    })
}

macro_rules! indirect_patches {
    ($bank:expr, $add:ident, $address:ident, $count:ident) => {
        agc_export! { fn $add(command: *mut u32, count: u32) -> i32 {
            apply(stringify!($add), command, || Ok(Patch::IndirectCount { bank: $bank, count, add: true }))
        } }
        agc_export! { fn $address(command: *mut u32, registers: *const ShaderRegister) -> i32 {
            apply(stringify!($address), command, || {
                check_address(registers as u64, 4, stringify!($address))?;
                Ok(Patch::IndirectAddress { bank: $bank, address: registers as u64 })
            })
        } }
        agc_export! { fn $count(command: *mut u32, count: u32) -> i32 {
            apply(stringify!($count), command, || {
                check_bits(count.into(), 0x3fff, stringify!($count))?;
                Ok(Patch::IndirectCount { bank: $bank, count, add: false })
            })
        } }
    };
}

indirect_patches!(
    RegisterBank::Context,
    sceAgcSetCxRegIndirectPatchAddRegisters,
    sceAgcSetCxRegIndirectPatchSetAddress,
    sceAgcSetCxRegIndirectPatchSetNumRegisters
);
indirect_patches!(
    RegisterBank::Shader,
    sceAgcSetShRegIndirectPatchAddRegisters,
    sceAgcSetShRegIndirectPatchSetAddress,
    sceAgcSetShRegIndirectPatchSetNumRegisters
);
indirect_patches!(
    RegisterBank::UserConfig,
    sceAgcSetUcRegIndirectPatchAddRegisters,
    sceAgcSetUcRegIndirectPatchSetAddress,
    sceAgcSetUcRegIndirectPatchSetNumRegisters
);

fn cond_exec_address(command: *mut u32, address: *const u32, function: &str) -> i32 {
    apply(function, command, || {
        check_address(address as u64, 4, function)?;
        Ok(Patch::CondExecAddress(address as u64))
    })
}

fn cond_exec_end(command: *mut u32, end: *const u32, function: &str) -> i32 {
    apply(function, command, || {
        check_address(end as u64, 4, function)?;
        Ok(Patch::CondExecEnd(end as u64))
    })
}

agc_export! { fn sceAgcCondExecPatchSetCommandAddress(command: *mut u32, address: *const u32) -> i32 {
    cond_exec_address(command, address, "sceAgcCondExecPatchSetCommandAddress")
} }
agc_export! { fn sceAgcCondExecPatchSetEnd(command: *mut u32, end: *const u32) -> i32 {
    cond_exec_end(command, end, "sceAgcCondExecPatchSetEnd")
} }
agc_export! { fn sceAgcAsyncCondExecPatchSetCommandAddress(command: *mut u32, address: *const u32) -> i32 {
    cond_exec_address(command, address, "sceAgcAsyncCondExecPatchSetCommandAddress")
} }
agc_export! { fn sceAgcAsyncCondExecPatchSetEnd(command: *mut u32, end: *const u32) -> i32 {
    cond_exec_end(command, end, "sceAgcAsyncCondExecPatchSetEnd")
} }

agc_export! { fn sceAgcDmaDataPatchSetDstAddressOrOffset(command: *mut u32, address: u64) -> i32 {
    apply("sceAgcDmaDataPatchSetDstAddressOrOffset", command, || Ok(Patch::DmaDestination(address)))
} }
agc_export! { fn sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate(command: *mut u32, address: u64) -> i32 {
    apply("sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate", command, || Ok(Patch::DmaSource(address)))
} }

/// A jump's new target; the cache policy has no host form and is only checked.
fn jump(command: *mut u32, cache_policy: u64, target: *const u32, dwords: u32, function: &str) -> i32 {
    apply(function, command, || {
        check_bits(cache_policy, 3, function)?;
        check_address(target as u64, 4, function)?;
        check_bits(dwords.into(), 0xf_ffff, function)?;
        Ok(Patch::JumpTarget { target: target as u64, dwords })
    })
}

agc_export! { fn sceAgcJumpPatchSetTarget(command: *mut u32, target: *const u32, dwords: u32) -> i32 {
    jump(command, 0, target, dwords, "sceAgcJumpPatchSetTarget")
} }
agc_export! { fn sceAgcJumpPatchSetTarget_0300(command: *mut u32, cache_policy: u64, target: *const u32, dwords: u32) -> i32 {
    jump(command, cache_policy, target, dwords, "sceAgcJumpPatchSetTarget_0300")
} }

agc_export! { fn sceAgcSetNop(command: *mut u32) -> i32 {
    apply("sceAgcSetNop", command, || Ok(Patch::Nop))
} }

agc_export! { fn sceAgcQueueEndOfPipeActionPatchAddress(command: *mut u32, label: *const u64) -> i32 {
    apply("sceAgcQueueEndOfPipeActionPatchAddress", command, || Ok(Patch::ReleaseAddress(label as u64)))
} }
agc_export! { fn sceAgcQueueEndOfPipeActionPatchData(command: *mut u32, data: u64) -> i32 {
    apply("sceAgcQueueEndOfPipeActionPatchData", command, || Ok(Patch::ReleaseData(data)))
} }
agc_export! { fn sceAgcQueueEndOfPipeActionPatchGcrCntl(command: *mut u32, gcr: u16) -> i32 {
    const F: &str = "sceAgcQueueEndOfPipeActionPatchGcrCntl";
    apply(F, command, || {
        check_bits(gcr.into(), 0xfff, F)?;
        // An L2 invalidate writes back first, as `sceAgcCbReleaseMem`.
        let mut gcr = u32::from(gcr);
        if gcr & 0x300 == 0x100 {
            gcr |= 0x200;
        }
        Ok(Patch::ReleaseCaches(gcr))
    })
} }
agc_export! { fn sceAgcQueueEndOfPipeActionPatchType(command: *mut u32, action: u8) -> i32 {
    const F: &str = "sceAgcQueueEndOfPipeActionPatchType";
    apply(F, command, || {
        check_bits(action.into(), 0x3f, F)?;
        Ok(Patch::ReleaseEvent(action.into()))
    })
} }

fn rewind(command: *mut u32, state: u8, function: &str) -> i32 {
    apply(function, command, || {
        check_bits(state.into(), 1, function)?;
        Ok(Patch::Rewind(state != 0))
    })
}

agc_export! { fn sceAgcRewindPatchSetRewindState(command: *mut u32, state: u8) -> i32 {
    rewind(command, state, "sceAgcRewindPatchSetRewindState")
} }
agc_export! { fn sceAgcAsyncRewindPatchSetRewindState(command: *mut u32, state: u8) -> i32 {
    rewind(command, state, "sceAgcAsyncRewindPatchSetRewindState")
} }

agc_export! { fn sceAgcWaitRegMemPatchAddress(command: *mut u32, address: *const c_void) -> i32 {
    apply("sceAgcWaitRegMemPatchAddress", command, || Ok(Patch::WaitAddress(address as u64)))
} }
agc_export! { /// The 64-bit reference (D19); the SDK's `setMask` calls this too.
fn sceAgcWaitRegMemPatchReference(command: *mut u32, reference: u64) -> i32 {
    apply("sceAgcWaitRegMemPatchReference", command, || Ok(Patch::WaitReference(reference)))
} }
agc_export! { fn sceAgcWaitRegMemPatchMask(command: *mut u32, mask: u32) -> i32 {
    apply("sceAgcWaitRegMemPatchMask", command, || Ok(Patch::WaitMask(mask)))
} }
agc_export! { fn sceAgcWaitRegMemPatchCompareFunction(command: *mut u32, function: u8) -> i32 {
    apply("sceAgcWaitRegMemPatchCompareFunction", command, || Ok(Patch::WaitFunction(function.into())))
} }

agc_export! { fn sceAgcWriteDataPatchSetAddressOrOffset(command: *mut u32, address: u64) -> i32 {
    apply("sceAgcWriteDataPatchSetAddressOrOffset", command, || Ok(Patch::WriteDataAddress(address)))
} }
agc_export! { fn sceAgcAsyncWriteDataPatchSetAddressOrOffset(command: *mut u32, address: u64) -> i32 {
    apply("sceAgcAsyncWriteDataPatchSetAddressOrOffset", command, || Ok(Patch::WriteDataAddress(address)))
} }
agc_export! { fn sceAgcWriteDataPatchSetDst(command: *mut u32, dst: u8) -> i32 {
    const F: &str = "sceAgcWriteDataPatchSetDst";
    apply(F, command, || {
        check_bits(dst.into(), 0x1f, F)?;
        Ok(Patch::WriteDataDestination { dst: dst.into(), compute: false })
    })
} }
agc_export! { fn sceAgcAsyncWriteDataPatchSetDst(command: *mut u32, dst: u8) -> i32 {
    const F: &str = "sceAgcAsyncWriteDataPatchSetDst";
    apply(F, command, || {
        check_bits(dst.into(), 0xf, F)?;
        Ok(Patch::WriteDataDestination { dst: dst.into(), compute: true })
    })
} }
agc_export! { fn sceAgcWriteDataPatchSetCachePolicy(command: *mut u32, policy: u8) -> i32 {
    const F: &str = "sceAgcWriteDataPatchSetCachePolicy";
    apply(F, command, || {
        check_bits(policy.into(), 3, F)?;
        Ok(Patch::WriteDataCachePolicy)
    })
} }
agc_export! { fn sceAgcAsyncWriteDataPatchSetCachePolicy(command: *mut u32, policy: u8) -> i32 {
    const F: &str = "sceAgcAsyncWriteDataPatchSetCachePolicy";
    apply(F, command, || {
        check_bits(policy.into(), 3, F)?;
        Ok(Patch::WriteDataCachePolicy)
    })
} }

agc_export! { /// The only patch that returns an error code on the console: [`NOT_AN_ACQUIRE`].
fn sceAgcAcquireMemSetEngine(command: *mut u32, engine: u8) -> i32 {
    guard("sceAgcAcquireMemSetEngine", NOT_AN_ACQUIRE, || {
        // SAFETY: as `apply`.
        unsafe { agc().target.patch(command, Patch::AcquireEngine(engine.into())) }?;
        Ok(0)
    })
} }

agc_export! { fn sceAgcSetPacketPredication(command: *mut u32, predication: u32) -> i32 {
    const F: &str = "sceAgcSetPacketPredication";
    apply(F, command, || {
        check_bits(predication.into(), 1, F)?;
        Ok(Patch::Predicate(predication != 0))
    })
} }
agc_export! { fn sceAgcSetRangePredication(start: *mut u32, end: *const u32, predication: u32) -> i32 {
    const F: &str = "sceAgcSetRangePredication";
    guard(F, INVALID, || {
        check_bits(predication.into(), 1, F)?;
        // SAFETY: the title passes a range of commands it recorded.
        unsafe { agc().target.predicate_range(start, end, predication != 0) }?;
        Ok(0)
    })
} }

// Patches of `sceAgcCbBranch`, which records nothing yet: there is no branch to find.
macro_rules! branch_patches {
    ($($name:ident),* $(,)?) => {
        $(agc_export! { fn $name() -> i32 {
            report(stringify!($name), "not a branch packet");
            INVALID
        } })*
    };
}

branch_patches!(
    sceAgcBranchPatchSetCompareAddress,
    sceAgcBranchPatchSetThenTarget,
    sceAgcBranchPatchSetElseTarget,
    sceAgcBranchPatchSetThenTarget_0300,
    sceAgcBranchPatchSetElseTarget_0300,
);

/// Exports for `agc::exports` (40 patches, the two predication functions).
pub const PATCHES: &[&str] = &[
    "sceAgcBranchPatchSetCompareAddress", "sceAgcBranchPatchSetThenTarget", "sceAgcBranchPatchSetElseTarget",
    "sceAgcBranchPatchSetThenTarget_0300", "sceAgcBranchPatchSetElseTarget_0300",
    "sceAgcSetCxRegIndirectPatchAddRegisters", "sceAgcSetCxRegIndirectPatchSetAddress", "sceAgcSetCxRegIndirectPatchSetNumRegisters",
    "sceAgcSetShRegIndirectPatchAddRegisters", "sceAgcSetShRegIndirectPatchSetAddress", "sceAgcSetShRegIndirectPatchSetNumRegisters",
    "sceAgcSetUcRegIndirectPatchAddRegisters", "sceAgcSetUcRegIndirectPatchSetAddress", "sceAgcSetUcRegIndirectPatchSetNumRegisters",
    "sceAgcCondExecPatchSetCommandAddress", "sceAgcCondExecPatchSetEnd", "sceAgcAsyncCondExecPatchSetCommandAddress",
    "sceAgcAsyncCondExecPatchSetEnd", "sceAgcDmaDataPatchSetDstAddressOrOffset", "sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate",
    "sceAgcJumpPatchSetTarget", "sceAgcJumpPatchSetTarget_0300", "sceAgcSetNop",
    "sceAgcQueueEndOfPipeActionPatchAddress", "sceAgcQueueEndOfPipeActionPatchData", "sceAgcQueueEndOfPipeActionPatchGcrCntl",
    "sceAgcQueueEndOfPipeActionPatchType", "sceAgcRewindPatchSetRewindState", "sceAgcAsyncRewindPatchSetRewindState",
    "sceAgcWaitRegMemPatchAddress", "sceAgcWaitRegMemPatchReference", "sceAgcWaitRegMemPatchMask",
    "sceAgcWaitRegMemPatchCompareFunction", "sceAgcWriteDataPatchSetAddressOrOffset", "sceAgcAsyncWriteDataPatchSetAddressOrOffset",
    "sceAgcWriteDataPatchSetDst", "sceAgcAsyncWriteDataPatchSetDst", "sceAgcWriteDataPatchSetCachePolicy",
    "sceAgcAsyncWriteDataPatchSetCachePolicy", "sceAgcAcquireMemSetEngine", "sceAgcSetPacketPredication",
    "sceAgcSetRangePredication",
];
