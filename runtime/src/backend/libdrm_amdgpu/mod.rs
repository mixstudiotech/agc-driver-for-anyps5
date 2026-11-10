//! Linux transport: host-format PM4 on the amdgpu GFX ring through libdrm_amdgpu.
//!
//! Mirrors the WDDM backend's API and failure model. Every buffer is a cacheable, CPU-snooped GTT
//! buffer object, mapped for the CPU and at a GPU virtual address of its own (read/write, plus
//! execute for shader code). A submission pads the stream into an indirect buffer (IB), submits it
//! on GFX ring 0 with a buffer list holding the IB and its targets, and waits up to 5 s for the
//! fence. A failed submission or an unsignalled fence quarantines the session: later operations
//! fail with `DeviceLost`, and nothing the GPU may still reference is unmapped or freed before
//! process exit.
//!
//! Runtime status: unverified. No Linux machine with an AMD GPU was available, so none of this has
//! executed against the amdgpu kernel driver or hardware.

mod node;
mod sys;

use crate::{
    backend::host_pm4,
    Error, ErrorKind, Result,
};
use node::{DriverVersion, RenderNode, RenderNodes};
use std::{
    cell::Cell,
    ffi::{c_int, c_uint, c_void},
    fs::File,
    io,
    mem::{self, size_of, ManuallyDrop},
    os::fd::AsRawFd,
    ptr,
    rc::Rc,
    sync::atomic::{fence, Ordering},
};

// The amdgpu kernel driver binds only AMD (PCI vendor 0x1002) devices.
const AMD_PCI_VENDOR_ID: u32 = 0x1002;
const BUFFER_ALIGNMENT: usize = 64 * 1024;
const GFX_IP_INSTANCE: c_uint = 0;
const GFX_RING: u32 = 0;
const FENCE_TIMEOUT_MS: u64 = 5000;
const NANOS_PER_MILLI: u64 = 1_000_000;
// PACKET3_INDIRECT_BUFFER encodes the IB length in a 20-bit dword field.
const MAX_IB_DWORDS: u32 = (1 << 20) - 1;
const DEV_INFO_BYTES: usize = 1024;
const HW_IP_INFO_BYTES: usize = 256;
const VERSION_COMPONENT_BITS: u32 = u16::BITS;
const VERSION_COMPONENT_MASK: u64 = u16::MAX as u64;
const VERSION_MAJOR_SHIFT: u32 = VERSION_COMPONENT_BITS * 3;
const VERSION_MINOR_SHIFT: u32 = VERSION_COMPONENT_BITS * 2;
const VERSION_PATCH_SHIFT: u32 = VERSION_COMPONENT_BITS;

#[derive(Clone, Debug)]
pub struct AdapterIdentity {
    pub vendor_id: u32,
    pub device_id: u32,
    /// PCI revision ID.
    pub revision: u32,
    /// amdgpu DRM interface version: 16-bit major, minor and patch level, from bit 63 down.
    pub driver_version: u64,
    pub render_node: String,
    /// `AMDGPU_FAMILY_*`.
    pub family: u32,
    pub chip_revision: u32,
    pub external_revision: u32,
    /// GFX IP block version (major, minor).
    pub gfx_ip: (u32, u32),
    /// GFX rings user space may submit to; bit 0 is ring 0.
    pub gfx_rings: u32,
}

impl AdapterIdentity {
    pub fn driver_string(&self) -> String {
        let v = self.driver_version;
        format!(
            "{}.{}.{}",
            v >> VERSION_MAJOR_SHIFT,
            (v >> VERSION_MINOR_SHIFT) & VERSION_COMPONENT_MASK,
            (v >> VERSION_PATCH_SHIFT) & VERSION_COMPONENT_MASK
        )
    }
}

fn pack_driver_version(version: DriverVersion) -> u64 {
    (u64::from(version.major) << VERSION_MAJOR_SHIFT)
        | (u64::from(version.minor) << VERSION_MINOR_SHIFT)
        | (u64::from(version.patchlevel) << VERSION_PATCH_SHIFT)
}

fn status(value: c_int, operation: &str) -> Result<()> {
    if value == 0 {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::System,
            format!("{operation}: {}", os_error(value)),
        ))
    }
}

/// libdrm reports failures as negative errno values.
fn os_error(value: c_int) -> io::Error {
    io::Error::from_raw_os_error(value.saturating_neg())
}

/// Zeroed, 8-byte-aligned storage for a kernel info struct that may be larger than the prefix
/// this backend parses.
#[repr(C, align(8))]
struct InfoBuffer<const BYTES: usize>([u8; BYTES]);

impl<const BYTES: usize> InfoBuffer<BYTES> {
    fn zeroed() -> Self {
        Self([0; BYTES])
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.0.as_mut_ptr().cast()
    }

    fn u32_at(&self, offset: usize) -> u32 {
        const FIELD_BYTES: usize = size_of::<u32>();
        let mut field = [0; FIELD_BYTES];
        field.copy_from_slice(&self.0[offset..offset + FIELD_BYTES]);
        u32::from_ne_bytes(field)
    }
}

struct Session {
    // Closed only after the device handle is deinitialized.
    node: ManuallyDrop<File>,
    identity: AdapterIdentity,
    profile: String,
    device: sys::AmdgpuDeviceHandle,
    context: sys::AmdgpuContextHandle,
    ib_start_alignment: u32,
    ib_size_alignment: u32,
    poisoned: Cell<bool>,
    live_allocations: Cell<usize>,
}

impl Drop for Session {
    fn drop(&mut self) {
        if self.poisoned.get() {
            eprintln!(
                "LIBDRM_FAULT_QUARANTINE: {}: context, buffers and device handle retained until process exit",
                self.identity.render_node
            );
            return;
        }
        unsafe {
            if !self.context.is_null()
                && !self.release("amdgpu_cs_ctx_free", sys::amdgpu_cs_ctx_free(self.context))
            {
                return;
            }
            if !self.device.is_null()
                && !self.release(
                    "amdgpu_device_deinitialize",
                    sys::amdgpu_device_deinitialize(self.device),
                )
            {
                return;
            }
            // SAFETY: libdrm no longer uses the descriptor, and this is the only place it closes.
            ManuallyDrop::drop(&mut self.node);
        }
    }
}

impl Session {
    /// Initializes libdrm_amdgpu on `node` and reads the adapter identity and GFX IB limits. The
    /// session has no submission context yet.
    fn initialize(node: RenderNode) -> Result<Self> {
        let RenderNode {
            path,
            file,
            version,
        } = node;
        let mut s = Session {
            node: ManuallyDrop::new(file),
            identity: AdapterIdentity {
                vendor_id: AMD_PCI_VENDOR_ID,
                device_id: 0,
                revision: 0,
                driver_version: pack_driver_version(version),
                render_node: path,
                family: 0,
                chip_revision: 0,
                external_revision: 0,
                gfx_ip: (0, 0),
                gfx_rings: 0,
            },
            profile: String::new(),
            device: ptr::null_mut(),
            context: ptr::null_mut(),
            ib_start_alignment: 0,
            ib_size_alignment: 0,
            poisoned: Cell::new(false),
            live_allocations: Cell::new(0),
        };
        let (mut drm_major, mut drm_minor) = (0, 0);
        let mut device = ptr::null_mut();
        status(
            unsafe {
                sys::amdgpu_device_initialize(
                    s.node.as_raw_fd(),
                    &mut drm_major,
                    &mut drm_minor,
                    &mut device,
                )
            },
            "amdgpu_device_initialize",
        )?;
        s.device = device;

        let mut info = InfoBuffer::<DEV_INFO_BYTES>::zeroed();
        status(
            unsafe {
                sys::amdgpu_query_info(
                    s.device,
                    sys::AMDGPU_INFO_DEV_INFO,
                    DEV_INFO_BYTES as c_uint,
                    info.as_mut_ptr(),
                )
            },
            "amdgpu_query_info(DEV_INFO)",
        )?;
        s.identity.device_id = info.u32_at(sys::DEV_INFO_DEVICE_ID);
        s.identity.chip_revision = info.u32_at(sys::DEV_INFO_CHIP_REV);
        s.identity.external_revision = info.u32_at(sys::DEV_INFO_EXTERNAL_REV);
        s.identity.revision = info.u32_at(sys::DEV_INFO_PCI_REV);
        s.identity.family = info.u32_at(sys::DEV_INFO_FAMILY);

        let mut gfx = InfoBuffer::<HW_IP_INFO_BYTES>::zeroed();
        status(
            unsafe {
                sys::amdgpu_query_hw_ip_info(
                    s.device,
                    sys::AMDGPU_HW_IP_GFX,
                    GFX_IP_INSTANCE,
                    gfx.as_mut_ptr(),
                )
            },
            "amdgpu_query_hw_ip_info(GFX)",
        )?;
        s.identity.gfx_ip = (
            gfx.u32_at(sys::HW_IP_VERSION_MAJOR),
            gfx.u32_at(sys::HW_IP_VERSION_MINOR),
        );
        s.identity.gfx_rings = gfx.u32_at(sys::HW_IP_AVAILABLE_RINGS);
        s.ib_start_alignment = gfx.u32_at(sys::HW_IP_IB_START_ALIGNMENT);
        s.ib_size_alignment = gfx.u32_at(sys::HW_IP_IB_SIZE_ALIGNMENT);
        s.profile = format!(
            "amdgpu-{:04x}-drm{drm_major}.{drm_minor}-gfx{}.{}",
            s.identity.device_id, s.identity.gfx_ip.0, s.identity.gfx_ip.1
        );
        Ok(s)
    }

    fn healthy(&self) -> Result<()> {
        if self.poisoned.get() {
            Err(Error::new(
                ErrorKind::DeviceLost,
                "device faulted; no further GPU/CPU buffer access",
            ))
        } else {
            Ok(())
        }
    }

    /// Reports one teardown step. A failed step may leave GPU-visible state behind, so it
    /// quarantines the session; the failure is printed because Drop cannot return it.
    fn release(&self, operation: &str, result: c_int) -> bool {
        if result == 0 {
            return true;
        }
        self.poisoned.set(true);
        eprintln!("LIBDRM_CLEANUP_FAILED {operation}: {}", os_error(result));
        false
    }

    fn check_ib_alignment(&self, address: u64, bytes: usize) -> Result<()> {
        let start = u64::from(self.ib_start_alignment.max(1));
        let size = self.ib_size_alignment.max(1) as usize;
        if address % start != 0 || bytes % size != 0 {
            return Err(Error::unsupported(format!(
                "IB 0x{address:x}+{bytes} breaks the kernel's GFX IB alignment (start {start}, size {size})"
            )));
        }
        Ok(())
    }

    fn wait(&self, sequence: u64) -> Result<()> {
        let mut pending = sys::AmdgpuCsFence {
            context: self.context,
            ip_type: sys::AMDGPU_HW_IP_GFX,
            ip_instance: GFX_IP_INSTANCE,
            ring: GFX_RING,
            fence: sequence,
        };
        let mut expired = 0;
        let result = unsafe {
            sys::amdgpu_cs_query_fence_status(
                &mut pending,
                FENCE_TIMEOUT_MS * NANOS_PER_MILLI,
                0,
                &mut expired,
            )
        };
        if result != 0 || expired == 0 {
            self.poisoned.set(true);
            let reason = if result != 0 {
                format!("amdgpu_cs_query_fence_status: {}", os_error(result))
            } else {
                format!("not signalled within {FENCE_TIMEOUT_MS} ms")
            };
            return Err(Error::new(
                ErrorKind::DeviceLost,
                format!(
                    "fence did not complete: GFX ring {GFX_RING} sequence {sequence}: {reason}"
                ),
            ));
        }
        fence(Ordering::SeqCst);
        Ok(())
    }
}

struct Allocation {
    session: Rc<Session>,
    bo: sys::AmdgpuBoHandle,
    cpu: *mut u8,
    va_range: sys::AmdgpuVaHandle,
    address: u64,
    mapped: bool,
    size: usize,
    executable: bool,
}

impl Allocation {
    fn vm_flags(&self) -> u64 {
        let access = sys::AMDGPU_VM_PAGE_READABLE | sys::AMDGPU_VM_PAGE_WRITEABLE;
        if self.executable {
            access | sys::AMDGPU_VM_PAGE_EXECUTABLE
        } else {
            access
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        let session = &self.session;
        if session.poisoned.get() {
            return;
        }
        unsafe {
            if self.mapped
                && !session.release(
                    "amdgpu_bo_va_op_raw(UNMAP)",
                    sys::amdgpu_bo_va_op_raw(
                        session.device,
                        self.bo,
                        0,
                        self.size as u64,
                        self.address,
                        self.vm_flags(),
                        sys::AMDGPU_VA_OP_UNMAP,
                    ),
                )
            {
                return;
            }
            if !self.va_range.is_null()
                && !session.release(
                    "amdgpu_va_range_free",
                    sys::amdgpu_va_range_free(self.va_range),
                )
            {
                return;
            }
            if !self.cpu.is_null()
                && !session.release("amdgpu_bo_cpu_unmap", sys::amdgpu_bo_cpu_unmap(self.bo))
            {
                return;
            }
            if !self.bo.is_null() && session.release("amdgpu_bo_free", sys::amdgpu_bo_free(self.bo))
            {
                session
                    .live_allocations
                    .set(session.live_allocations.get() - 1);
            }
        }
    }
}

pub struct Buffer {
    allocation: Allocation,
}

impl Buffer {
    pub fn address(&self) -> u64 {
        self.allocation.address
    }
    pub fn size(&self) -> usize {
        self.allocation.size
    }
    pub fn is_executable(&self) -> bool {
        self.allocation.executable
    }
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.allocation.session.healthy()?;
        if offset > self.size() || bytes.len() > self.size() - offset {
            return Err(Error::invalid("CPU write outside buffer"));
        }
        for (i, byte) in bytes.iter().enumerate() {
            unsafe {
                self.allocation.cpu.add(offset + i).write_volatile(*byte);
            }
        }
        fence(Ordering::SeqCst);
        Ok(())
    }
    pub fn read(&self) -> Result<Vec<u8>> {
        self.allocation.session.healthy()?;
        fence(Ordering::SeqCst);
        Ok((0..self.size())
            .map(|i| unsafe { self.allocation.cpu.add(i).read_volatile() })
            .collect())
    }
    pub fn fill(&mut self, byte: u8) -> Result<()> {
        self.write(0, &vec![byte; self.size()])
    }
    fn contains(&self, address: u64, bytes: u64) -> bool {
        let size = self.size() as u64;
        address >= self.address() && bytes <= size && address - self.address() <= size - bytes
    }
}

/// The kernel buffer list of one submission: the IB and every buffer the stream may touch, which
/// the kernel keeps resident and fences for the job.
struct BoList<'a> {
    session: &'a Session,
    handle: sys::AmdgpuBoListHandle,
}

impl<'a> BoList<'a> {
    fn create(session: &'a Session, buffers: &mut [sys::AmdgpuBoHandle]) -> Result<Self> {
        let count = u32::try_from(buffers.len())
            .map_err(|_| Error::invalid("too many buffers in one submission"))?;
        let mut handle = ptr::null_mut();
        status(
            unsafe {
                sys::amdgpu_bo_list_create(
                    session.device,
                    count,
                    buffers.as_mut_ptr(),
                    ptr::null_mut(),
                    &mut handle,
                )
            },
            "amdgpu_bo_list_create",
        )?;
        Ok(Self { session, handle })
    }

    fn destroy(mut self) -> Result<()> {
        let handle = mem::replace(&mut self.handle, ptr::null_mut());
        let result = unsafe { sys::amdgpu_bo_list_destroy(handle) };
        self.session.release("amdgpu_bo_list_destroy", result);
        status(result, "amdgpu_bo_list_destroy")
    }
}

impl Drop for BoList<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() && !self.session.poisoned.get() {
            self.session.release("amdgpu_bo_list_destroy", unsafe {
                sys::amdgpu_bo_list_destroy(self.handle)
            });
        }
    }
}

pub struct Device {
    session: Rc<Session>,
    submitted: u64,
    completed: u64,
}

impl Device {
    pub fn enumerate_adapters() -> Result<Vec<AdapterIdentity>> {
        RenderNodes::scan()?
            .map(|node| Session::initialize(node).map(|session| session.identity.clone()))
            .collect()
    }

    /// Opens the first amdgpu render node and creates a submission context on it.
    pub fn open() -> Result<Self> {
        let mut session = Session::initialize(RenderNodes::scan()?.first_amdgpu()?)?;
        if session.identity.gfx_rings & (1 << GFX_RING) == 0 {
            return Err(Error::unsupported(format!(
                "{}: amdgpu exposes no GFX ring {GFX_RING} to user space",
                session.identity.render_node
            )));
        }
        let mut context = ptr::null_mut();
        status(
            unsafe { sys::amdgpu_cs_ctx_create(session.device, &mut context) },
            "amdgpu_cs_ctx_create",
        )?;
        session.context = context;
        Ok(Self {
            session: Rc::new(session),
            submitted: 0,
            completed: 0,
        })
    }

    pub fn identity(&self) -> &AdapterIdentity {
        &self.session.identity
    }
    pub fn profile(&self) -> &str {
        &self.session.profile
    }
    pub fn submitted(&self) -> u64 {
        self.submitted
    }
    pub fn completed(&self) -> u64 {
        self.completed
    }
    pub fn live_allocations(&self) -> usize {
        self.session.live_allocations.get()
    }
    pub fn is_faulted(&self) -> bool {
        self.session.poisoned.get()
    }

    pub fn allocate(&mut self, requested: usize) -> Result<Buffer> {
        self.allocate_mapped(requested, false)
    }

    /// A buffer whose GPU mapping permits instruction fetch, for shader code.
    pub fn allocate_executable(&mut self, requested: usize) -> Result<Buffer> {
        self.allocate_mapped(requested, true)
    }

    fn allocate_mapped(&mut self, requested: usize, executable: bool) -> Result<Buffer> {
        self.session.healthy()?;
        let size = match requested.checked_next_multiple_of(BUFFER_ALIGNMENT) {
            Some(size) if requested != 0 => size,
            _ => return Err(Error::invalid("buffer size is zero or too large")),
        };
        let session = &self.session;
        let mut a = Allocation {
            session: session.clone(),
            bo: ptr::null_mut(),
            cpu: ptr::null_mut(),
            va_range: ptr::null_mut(),
            address: 0,
            mapped: false,
            size,
            executable,
        };
        let mut request = sys::AmdgpuBoAllocRequest {
            alloc_size: size as u64,
            phys_alignment: BUFFER_ALIGNMENT as u64,
            preferred_heap: sys::AMDGPU_GEM_DOMAIN_GTT,
            flags: sys::AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
        };
        let mut bo = ptr::null_mut();
        status(
            unsafe { sys::amdgpu_bo_alloc(session.device, &mut request, &mut bo) },
            "amdgpu_bo_alloc(GTT)",
        )?;
        a.bo = bo;
        session
            .live_allocations
            .set(session.live_allocations.get() + 1);
        let mut cpu = ptr::null_mut();
        status(
            unsafe { sys::amdgpu_bo_cpu_map(a.bo, &mut cpu) },
            "amdgpu_bo_cpu_map",
        )?;
        a.cpu = cpu.cast();
        if a.cpu.is_null() {
            return Err(Error::new(
                ErrorKind::System,
                "amdgpu_bo_cpu_map returned null",
            ));
        }
        let (mut address, mut va_range) = (0, ptr::null_mut());
        status(
            unsafe {
                sys::amdgpu_va_range_alloc(
                    session.device,
                    sys::AMDGPU_GPU_VA_RANGE_GENERAL,
                    size as u64,
                    BUFFER_ALIGNMENT as u64,
                    0,
                    &mut address,
                    &mut va_range,
                    0,
                )
            },
            "amdgpu_va_range_alloc",
        )?;
        a.address = address;
        a.va_range = va_range;
        status(
            unsafe {
                sys::amdgpu_bo_va_op_raw(
                    session.device,
                    a.bo,
                    0,
                    size as u64,
                    a.address,
                    a.vm_flags(),
                    sys::AMDGPU_VA_OP_MAP,
                )
            },
            "amdgpu_bo_va_op_raw(MAP)",
        )?;
        a.mapped = true;
        Ok(Buffer { allocation: a })
    }

    /// Submit host-format PM4. Failure modes and fencing match [`Self::submit`]: a failed
    /// submission or an incomplete fence quarantines the device.
    pub fn submit_host(&mut self, words: &[u32], targets: &[&Buffer]) -> Result<()> {
        self.session.healthy()?;
        for target in targets {
            if !Rc::ptr_eq(&self.session, &target.allocation.session) {
                return Err(Error::invalid(
                    "host submission target belongs to another device",
                ));
            }
        }
        self.execute_ib(words, targets)
    }

    /// Runs `words` as one IB on GFX ring 0 and waits for it. `targets` must be buffers of this
    /// device that hold every address the stream touches.
    fn execute_ib(&mut self, words: &[u32], targets: &[&Buffer]) -> Result<()> {
        let padded = host_pm4::pad(words)?;
        let dwords = u32::try_from(padded.len())
            .ok()
            .filter(|&dwords| dwords <= MAX_IB_DWORDS)
            .ok_or_else(|| Error::invalid("host IB exceeds the INDIRECT_BUFFER size limit"))?;
        let mut commands = self.allocate(padded.len() * host_pm4::DWORD_BYTES)?;
        let bytes: Vec<u8> = padded.iter().flat_map(|word| word.to_le_bytes()).collect();
        commands.write(0, &bytes)?;
        self.session
            .check_ib_alignment(commands.address(), bytes.len())?;
        let serial = self
            .submitted
            .checked_add(1)
            .ok_or_else(|| Error::invalid("submission serial overflow"))?;
        let mut resources: Vec<sys::AmdgpuBoHandle> = targets
            .iter()
            .map(|target| target.allocation.bo)
            .chain([commands.allocation.bo])
            .collect();
        resources.sort_unstable();
        resources.dedup();
        let list = BoList::create(&self.session, &mut resources)?;
        let mut ib = sys::AmdgpuCsIbInfo {
            flags: 0,
            ib_mc_address: commands.address(),
            size: dwords,
        };
        let mut request = sys::AmdgpuCsRequest {
            flags: 0,
            ip_type: sys::AMDGPU_HW_IP_GFX,
            ip_instance: GFX_IP_INSTANCE,
            ring: GFX_RING,
            resources: list.handle,
            number_of_dependencies: 0,
            dependencies: ptr::null_mut(),
            number_of_ibs: 1,
            ibs: &mut ib,
            seq_no: 0,
            fence_info: sys::AmdgpuCsFenceInfo {
                handle: ptr::null_mut(),
                offset: 0,
            },
        };
        if let Err(error) = status(
            unsafe { sys::amdgpu_cs_submit(self.session.context, 0, &mut request, 1) },
            "amdgpu_cs_submit(GFX ring 0)",
        ) {
            self.session.poisoned.set(true);
            return Err(error);
        }
        self.submitted = serial;
        self.session.wait(request.seq_no)?;
        list.destroy()?;
        self.completed = serial;
        Ok(())
    }
}
