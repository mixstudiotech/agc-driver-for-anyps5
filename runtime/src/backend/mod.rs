pub mod host_pm4;

#[cfg(all(feature = "wddm", target_os = "windows"))]
pub mod wddm;

#[cfg(all(feature = "libdrm-amdgpu", target_os = "linux"))]
pub mod libdrm_amdgpu;

/// What a buffer is for. The transport derives the buffer's GPU cache policy and protection from
/// it (FINAL-REPORT §2-A P1: per-buffer cache policy instead of GL2-uncached everywhere).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryKind {
    /// Shader code: GPU-executable and GL2-cached.
    Code,
    /// CPU-written data the GPU reads, such as vertices, descriptors and constants: GL2-cached.
    Data,
    /// GPU working memory the CPU does not read back, such as rings and render targets:
    /// GL2-cached.
    Device,
    /// Memory the CPU reads back or polls, such as readback targets, fences and labels: GL2
    /// bypass, so GPU writes reach memory without a cache writeback.
    Readback,
    /// Command buffers: GL2 bypass, so CPU writes reach the command processor without a flush.
    Commands,
}

impl MemoryKind {
    pub fn is_executable(self) -> bool {
        self == MemoryKind::Code
    }

    pub fn bypasses_gl2(self) -> bool {
        matches!(self, MemoryKind::Readback | MemoryKind::Commands)
    }
}

/// GPU virtual address layout of the native graphics path (FINAL-REPORT §1, "GPU VA layout").
pub mod va {
    /// Guest addresses: the same VA range as AnyPS5's guest arena.
    pub const GUEST_BASE: u64 = 0x0000_0001_0000;
    pub const GUEST_END: u64 = 0x00fc_0000_0000;
    /// GPU VA below this is already taken in our process when the device opens: dxgkrnl and the
    /// KMD map the device's fences (the graphics queue's progress fence at 0x11040) and their
    /// own allocations there. On mix (driver 32.0.32015.7029) `[base, GUEST_END)` cannot be
    /// reserved for any base up to 0x35ff_0000 and can from 0x3600_0000.
    pub const OS_LOW_END: u64 = 0x0000_3600_0000;
    /// The reserved guest window starts here: well above the OS range, and where AnyPS5 places
    /// guest memory mapped without an address (GuestArena's DefaultAllocationStart). Guest
    /// memory below it is never same-VA on the GPU.
    pub const GUEST_WINDOW_BASE: u64 = 0x0002_0000_0000;
    /// Same-VA imports never cross a boundary of this size in guest address space, so a guest
    /// unmap or remap only touches the imports of the windows it covers.
    pub const IMPORT_WINDOW_BYTES: u64 = 8 << 20;
    /// The guest page size (PS5_PAGE_SIZE): imports start and end on guest pages.
    pub const GUEST_PAGE_BYTES: u64 = 16 << 10;
    /// Host-layout copies of guest surfaces live at guest VA + 2^40.
    pub const HOST_COPY_OFFSET: u64 = 1 << 40;
    pub const HOST_COPY_BASE: u64 = 0x0100_0000_0000;
    pub const HOST_COPY_END: u64 = 0x0200_0000_0000;
    /// Runtime-owned memory: IBs, translated code, rings, readback and interchange buffers.
    pub const RUNTIME_BASE: u64 = 0x0200_0000_0000;
    /// The last GPU VA the KMD hands out on the verified profile (the driver-private va_end).
    pub const VA_END: u64 = 0x7fff_ffff_ffff;
    /// The attribute ring: 2 shader engines x 1408 KiB, on a 2 MiB boundary (FINAL-REPORT §2-A).
    pub const ATTRIBUTE_RING_BYTES: u64 = 2 * 1408 * 1024;
    pub const ATTRIBUTE_RING_ALIGNMENT: u64 = 2 << 20;

    // The windows are ordered, the host copy of every guest address lands in its window, and
    // everything fits under the KMD's va_end.
    const _: () = assert!(GUEST_BASE < GUEST_END && GUEST_END <= HOST_COPY_BASE);
    const _: () = assert!(GUEST_BASE < OS_LOW_END && OS_LOW_END < GUEST_WINDOW_BASE);
    const _: () = assert!(GUEST_WINDOW_BASE < GUEST_END);
    const _: () = assert!(GUEST_WINDOW_BASE % IMPORT_WINDOW_BYTES == 0);
    const _: () = assert!(GUEST_END % IMPORT_WINDOW_BYTES == 0);
    const _: () = assert!(IMPORT_WINDOW_BYTES % GUEST_PAGE_BYTES == 0);
    const _: () = assert!(HOST_COPY_BASE <= GUEST_BASE + HOST_COPY_OFFSET);
    const _: () = assert!(GUEST_END + HOST_COPY_OFFSET <= HOST_COPY_END);
    const _: () = assert!(HOST_COPY_END <= RUNTIME_BASE && RUNTIME_BASE < VA_END);

    /// The last usable GPU VA for a KMD reporting `virtual_address_bits` (0 when unknown), never
    /// above the profile's va_end.
    pub fn va_end(virtual_address_bits: u32) -> u64 {
        match virtual_address_bits {
            1..=63 => ((1u64 << virtual_address_bits) - 1).min(VA_END),
            _ => VA_END,
        }
    }

}
