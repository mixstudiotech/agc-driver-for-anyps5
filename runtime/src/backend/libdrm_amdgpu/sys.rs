//! Raw libdrm and libdrm_amdgpu declarations. Struct layouts are the x86_64 Linux ABI of
//! `xf86drm.h` and `libdrm/amdgpu.h`; libdrm_amdgpu handles are opaque pointers.

use std::ffi::{c_char, c_int, c_uint, c_void};

pub type AmdgpuDeviceHandle = *mut c_void;
pub type AmdgpuContextHandle = *mut c_void;
pub type AmdgpuBoHandle = *mut c_void;
pub type AmdgpuBoListHandle = *mut c_void;
pub type AmdgpuVaHandle = *mut c_void;

pub const AMDGPU_INFO_DEV_INFO: c_uint = 0x16;
pub const AMDGPU_HW_IP_GFX: c_uint = 0;
pub const AMDGPU_GEM_DOMAIN_GTT: u32 = 0x2;
pub const AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1 << 0;
/// `amdgpu_gpu_va_range_general`; the C enum is passed as a 32-bit integer.
pub const AMDGPU_GPU_VA_RANGE_GENERAL: u32 = 0;
pub const AMDGPU_VM_PAGE_READABLE: u64 = 1 << 1;
pub const AMDGPU_VM_PAGE_WRITEABLE: u64 = 1 << 2;
pub const AMDGPU_VM_PAGE_EXECUTABLE: u64 = 1 << 3;
pub const AMDGPU_VA_OP_MAP: u32 = 1;
pub const AMDGPU_VA_OP_UNMAP: u32 = 2;

// Field offsets in `struct drm_amdgpu_info_device` and `struct drm_amdgpu_info_hw_ip`. Both
// structs grow across kernel and libdrm releases, so they are read into oversized buffers and only
// these fixed prefixes are parsed.
pub const DEV_INFO_DEVICE_ID: usize = 0;
pub const DEV_INFO_CHIP_REV: usize = 4;
pub const DEV_INFO_EXTERNAL_REV: usize = 8;
pub const DEV_INFO_PCI_REV: usize = 12;
pub const DEV_INFO_FAMILY: usize = 16;
pub const HW_IP_VERSION_MAJOR: usize = 0;
pub const HW_IP_VERSION_MINOR: usize = 4;
pub const HW_IP_IB_START_ALIGNMENT: usize = 16;
pub const HW_IP_IB_SIZE_ALIGNMENT: usize = 20;
pub const HW_IP_AVAILABLE_RINGS: usize = 24;

#[repr(C)]
pub struct DrmVersion {
    pub version_major: c_int,
    pub version_minor: c_int,
    pub version_patchlevel: c_int,
    pub name_len: c_int,
    pub name: *mut c_char,
    pub date_len: c_int,
    pub date: *mut c_char,
    pub desc_len: c_int,
    pub desc: *mut c_char,
}

#[repr(C)]
pub struct AmdgpuBoAllocRequest {
    pub alloc_size: u64,
    pub phys_alignment: u64,
    pub preferred_heap: u32,
    pub flags: u64,
}

#[repr(C)]
pub struct AmdgpuCsIbInfo {
    pub flags: u64,
    pub ib_mc_address: u64,
    /// IB length in dwords.
    pub size: u32,
}

#[repr(C)]
pub struct AmdgpuCsFenceInfo {
    pub handle: AmdgpuBoHandle,
    pub offset: u64,
}

#[repr(C)]
pub struct AmdgpuCsRequest {
    pub flags: u64,
    pub ip_type: c_uint,
    pub ip_instance: c_uint,
    pub ring: u32,
    pub resources: AmdgpuBoListHandle,
    pub number_of_dependencies: u32,
    pub dependencies: *mut AmdgpuCsFence,
    pub number_of_ibs: u32,
    pub ibs: *mut AmdgpuCsIbInfo,
    /// Written by `amdgpu_cs_submit`: the fence sequence number of the submission.
    pub seq_no: u64,
    pub fence_info: AmdgpuCsFenceInfo,
}

#[repr(C)]
pub struct AmdgpuCsFence {
    pub context: AmdgpuContextHandle,
    pub ip_type: u32,
    pub ip_instance: u32,
    pub ring: u32,
    pub fence: u64,
}

#[link(name = "drm")]
unsafe extern "C" {
    pub fn drmGetVersion(fd: c_int) -> *mut DrmVersion;
    pub fn drmFreeVersion(version: *mut DrmVersion);
}

#[link(name = "drm_amdgpu")]
unsafe extern "C" {
    pub fn amdgpu_device_initialize(
        fd: c_int,
        major_version: *mut u32,
        minor_version: *mut u32,
        device_handle: *mut AmdgpuDeviceHandle,
    ) -> c_int;
    pub fn amdgpu_device_deinitialize(device_handle: AmdgpuDeviceHandle) -> c_int;
    pub fn amdgpu_query_info(
        dev: AmdgpuDeviceHandle,
        info_id: c_uint,
        size: c_uint,
        value: *mut c_void,
    ) -> c_int;
    /// `info` points to a `struct drm_amdgpu_info_hw_ip`.
    pub fn amdgpu_query_hw_ip_info(
        dev: AmdgpuDeviceHandle,
        ip_type: c_uint,
        ip_instance: c_uint,
        info: *mut c_void,
    ) -> c_int;
    pub fn amdgpu_bo_alloc(
        dev: AmdgpuDeviceHandle,
        alloc_buffer: *mut AmdgpuBoAllocRequest,
        buf_handle: *mut AmdgpuBoHandle,
    ) -> c_int;
    pub fn amdgpu_bo_free(buf_handle: AmdgpuBoHandle) -> c_int;
    pub fn amdgpu_bo_cpu_map(buf_handle: AmdgpuBoHandle, cpu: *mut *mut c_void) -> c_int;
    pub fn amdgpu_bo_cpu_unmap(buf_handle: AmdgpuBoHandle) -> c_int;
    pub fn amdgpu_va_range_alloc(
        dev: AmdgpuDeviceHandle,
        va_range_type: u32,
        size: u64,
        va_base_alignment: u64,
        va_base_required: u64,
        va_base_allocated: *mut u64,
        va_range_handle: *mut AmdgpuVaHandle,
        flags: u64,
    ) -> c_int;
    pub fn amdgpu_va_range_free(va_range_handle: AmdgpuVaHandle) -> c_int;
    pub fn amdgpu_bo_va_op_raw(
        dev: AmdgpuDeviceHandle,
        bo: AmdgpuBoHandle,
        offset: u64,
        size: u64,
        addr: u64,
        flags: u64,
        ops: u32,
    ) -> c_int;
    pub fn amdgpu_cs_ctx_create(
        dev: AmdgpuDeviceHandle,
        context: *mut AmdgpuContextHandle,
    ) -> c_int;
    pub fn amdgpu_cs_ctx_free(context: AmdgpuContextHandle) -> c_int;
    pub fn amdgpu_bo_list_create(
        dev: AmdgpuDeviceHandle,
        number_of_resources: u32,
        resources: *mut AmdgpuBoHandle,
        resource_prios: *mut u8,
        result: *mut AmdgpuBoListHandle,
    ) -> c_int;
    pub fn amdgpu_bo_list_destroy(handle: AmdgpuBoListHandle) -> c_int;
    pub fn amdgpu_cs_submit(
        context: AmdgpuContextHandle,
        flags: u64,
        ibs_request: *mut AmdgpuCsRequest,
        number_of_requests: u32,
    ) -> c_int;
    pub fn amdgpu_cs_query_fence_status(
        fence: *mut AmdgpuCsFence,
        timeout_ns: u64,
        flags: u64,
        expired: *mut u32,
    ) -> c_int;
}
