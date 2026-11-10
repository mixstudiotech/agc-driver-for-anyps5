//! The global data share (GDS) as the translation emulates it.
//!
//! gfx11 hosts do not give programs GDS, so with [`crate::ShaderInterface::gds_buffer`] every
//! guest GDS access becomes a global memory access to a buffer the runtime owns: guest GDS byte
//! `i` is buffer byte `i`. GDS addresses wrap at [`GDS_BYTES`].
//!
//! - Loads, stores and 32-bit integer atomics address `(vaddr + offset) mod GDS_BYTES`. Loads
//!   bypass the vector caches and atomics run at the L2, so every wave of the device sees the
//!   same data, as with GDS.
//! - `ds_append` and `ds_consume` add or subtract the number of active lanes at
//!   `(M0[31:16] + offset) mod GDS_BYTES` with one atomic, and every active lane gets the
//!   counter's previous value. M0's size (`M0[15:0]`) is not checked.
//! - `ds_ordered_count` (one dword) adds the first active lane's value (its ADDR VGPR) to the
//!   counter at `offset0`, or stores it in place with bit 4 of `offset1`, with one atomic, and
//!   every active lane gets the counter's previous value. M0 (the ordered wave ID) is ignored.
//!   Waves are not ordered: totals are exact, but waves may receive their ranges in another
//!   order than the hardware's launch order.
//!
//! Global wave sync (GWS), float and 64-bit atomics, `ds_rsub*`, `ds_mskor*`, `ds_wrap_rtn_b32`,
//! the two-address exchanges, `ds_condxchg32_rtn_b64` and the lane-indexed `*_addtid_b32` forms
//! are rejected on GDS.
//!
//! The runtime keeps one buffer for the guest GPU, initially zero, and redirects PM4 commands
//! that write or read GDS (`DMA_DATA` with a GDS source or destination, end-of-shader event
//! writes that copy GDS to memory) to it.

/// The guest GDS the translation addresses.
pub const GDS_BYTES: u32 = 0x1_0000;
/// The buffer the runtime allocates: the GDS and room for a 16-byte access at its last address.
pub const BUFFER_BYTES: u32 = GDS_BYTES + 16;
