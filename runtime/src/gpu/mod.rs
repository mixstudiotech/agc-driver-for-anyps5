//! The native gfx11.5 (gfx1151) graphics path: the per-submission state block and host PM4
//! emission with a register log.

pub mod agc;
pub mod kernels;
pub mod packets;
pub mod regs;
pub mod state_block;
pub mod target;
