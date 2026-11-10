//! Objects shared with D3D12 for presentation (FINAL-REPORT §2-E, §6 Milestone 1b): a D3D12
//! shared buffer opened by NT handle, a native monitored fence D3D12 opens (F_ready), a D3D12
//! fence the native queue waits on (F_copy), and hardware-queue waits and signals on both.

use super::{
    fence::MonitoredFence,
    map_and_make_resident,
    private::{AmdPrivateAbi, SharedRecords},
    Allocation, Buffer, Device, Kmt, MemoryKind, Session, GPU_PAGE_BYTES,
};
use crate::{Error, Result};
use std::{ffi::c_void, mem::size_of, ops::Range, ptr, rc::Rc};
use windows_sys::{
    Wdk::{Foundation::OBJECT_ATTRIBUTES, Graphics::Direct3D::*},
    Win32::Foundation::{CloseHandle, HANDLE},
};

/// D3DDDI_SYNCHRONIZATIONOBJECT_FLAGS bits.
const SYNC_SHARED: u32 = 1 << 0;
const SYNC_NT_SECURITY_SHARING: u32 = 1 << 1;
const SYNC_NO_GPU_ACCESS: u32 = 1 << 7;
/// D3DDDI_SYNC_OBJECT_ALL_ACCESS: STANDARD_RIGHTS_REQUIRED | WAIT | MODIFY_STATE.
const SYNC_OBJECT_ALL_ACCESS: u32 = 0x001f_0003;

/// A fence a hardware queue can wait on or signal.
pub trait QueueFence {
    #[doc(hidden)]
    fn sync_object(&self) -> u32;
    /// The fence's current value through its CPU mapping, when it has one.
    fn current(&self) -> Option<u64>;
}

/// A monitored fence created on this device and shared through an NT handle, which D3D12 opens
/// with `ID3D12Device::OpenSharedHandle`.
pub struct SharedFence {
    session: Rc<Session>,
    fence: MonitoredFence,
    nt_handle: HANDLE,
}

impl SharedFence {
    /// The NT handle to pass to the other device. It stays owned by this fence.
    pub fn nt_handle(&self) -> *mut c_void {
        self.nt_handle
    }
}

impl QueueFence for SharedFence {
    fn sync_object(&self) -> u32 {
        self.fence.object
    }
    fn current(&self) -> Option<u64> {
        self.fence.current()
    }
}

impl Drop for SharedFence {
    fn drop(&mut self) {
        if !self.nt_handle.is_null() {
            unsafe {
                CloseHandle(self.nt_handle);
            }
        }
        if self.session.poisoned.get() || self.session.device == 0 {
            return;
        }
        unsafe {
            D3DKMTDestroySynchronizationObject(&D3DKMT_DESTROYSYNCHRONIZATIONOBJECT {
                hSyncObject: self.fence.object,
            });
        }
    }
}

/// The private data a shared resource carries, as OpenResourceFromNtHandle returns it: the
/// creating runtime's, and the creating driver's for the resource and its allocation. A failed
/// import quotes its sizes; a probe dumps it to learn the creator's layout.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SharedResourceData {
    pub allocations: u32,
    pub runtime: Vec<u8>,
    pub resource: Vec<u8>,
    pub total: Vec<u8>,
    /// Where the allocation's own private data lies in `total`.
    pub allocation: Range<usize>,
    /// The GPU VA OpenResourceFromNtHandle reported for the allocation.
    pub gpu_virtual_address: u64,
}

impl SharedResourceData {
    /// The allocation's private driver data.
    pub fn allocation_data(&self) -> &[u8] {
        &self.total[self.allocation.clone()]
    }

    /// The AMD records in the allocation's private data.
    pub fn records(&self) -> std::result::Result<SharedRecords, String> {
        AmdPrivateAbi::shared_records(self.allocation_data())
    }

    /// The private-data sizes, for an error.
    pub fn sizes(&self) -> String {
        format!(
            "private data: runtime 0x{:x}, resource 0x{:x} and total 0x{:x} bytes, the allocation's 0x{:x} bytes at +0x{:x}",
            self.runtime.len(),
            self.resource.len(),
            self.total.len(),
            self.allocation.len(),
            self.allocation.start
        )
    }
}

/// Where an allocation's private data, as OpenResourceFromNtHandle points at it, lies in `total`;
/// all of `total` when the pointer does not lie inside it.
fn allocation_range(total: &[u8], data: *const c_void, bytes: u32) -> Range<usize> {
    let start = (data as usize).wrapping_sub(total.as_ptr() as usize);
    match start.checked_add(bytes as usize) {
        Some(end) if !data.is_null() && bytes != 0 && end <= total.len() => start..end,
        _ => 0..total.len(),
    }
}

/// The bytes of a shared allocation to map: `bytes` from its creator rounded up to whole GPU
/// pages, or without it the physical size in its AMD records. A creator's size the records show
/// is larger than the allocation is refused.
fn shared_size(data: &SharedResourceData, bytes: Option<u64>) -> Result<u64> {
    const PAGE: u64 = GPU_PAGE_BYTES as u64;
    let size = match (bytes, data.records()) {
        (Some(bytes), Ok(records)) if bytes > records.physical_size => {
            return Err(Error::invalid(format!(
                "the shared allocation's AMD records give 0x{:x} bytes, less than the 0x{bytes:x} its creator gave",
                records.physical_size
            )))
        }
        (Some(bytes), _) => bytes.checked_next_multiple_of(PAGE).unwrap_or(u64::MAX),
        (None, Ok(records)) => records.physical_size,
        (None, Err(why)) => {
            return Err(Error::unsupported(format!(
                "the shared resource's size is unknown: {why}; {}. Pass the size its creator allocated (open_shared_buffer_sized)",
                data.sizes()
            )))
        }
    };
    if size == 0 || size % PAGE != 0 || size > u64::from(u32::MAX) {
        return Err(Error::unsupported(format!(
            "shared allocation size 0x{size:x} is not whole 4 KiB pages below 4 GiB"
        )));
    }
    Ok(size)
}

/// A fence another device created, opened on this device by NT handle.
pub struct OpenedFence {
    session: Rc<Session>,
    fence: MonitoredFence,
}

impl QueueFence for OpenedFence {
    fn sync_object(&self) -> u32 {
        self.fence.object
    }
    fn current(&self) -> Option<u64> {
        self.fence.current()
    }
}

impl Drop for OpenedFence {
    fn drop(&mut self) {
        if self.session.poisoned.get() || self.session.device == 0 {
            return;
        }
        unsafe {
            D3DKMTDestroySynchronizationObject(&D3DKMT_DESTROYSYNCHRONIZATIONOBJECT {
                hSyncObject: self.fence.object,
            });
        }
    }
}

impl Device {
    /// A shared monitored fence starting at `initial`, exported with D3DKMTShareObjects.
    pub fn create_shared_fence(&mut self, initial: u64) -> Result<SharedFence> {
        self.session.healthy()?;
        let mut create = D3DKMT_CREATESYNCHRONIZATIONOBJECT2 {
            hDevice: self.session.device,
            ..Default::default()
        };
        create.Info.Type = D3DDDI_MONITORED_FENCE;
        create.Info.Flags.Anonymous.Value = SYNC_SHARED | SYNC_NT_SECURITY_SHARING;
        create.Info.Anonymous.MonitoredFence.InitialFenceValue = initial;
        create.Info.Anonymous.MonitoredFence.EngineAffinity = 1;
        Kmt::status(
            unsafe { D3DKMTCreateSynchronizationObject2(&mut create) },
            "CreateSynchronizationObject2(shared monitored fence)",
        )?;
        let mut fence = SharedFence {
            session: self.session.clone(),
            fence: MonitoredFence {
                object: create.hSyncObject,
                value: unsafe {
                    create
                        .Info
                        .Anonymous
                        .MonitoredFence
                        .FenceValueCPUVirtualAddress
                }
                .cast_const()
                .cast(),
            },
            nt_handle: ptr::null_mut(),
        };
        let attributes = OBJECT_ATTRIBUTES {
            Length: size_of::<OBJECT_ATTRIBUTES>() as u32,
            RootDirectory: ptr::null_mut(),
            ObjectName: ptr::null(),
            Attributes: 0,
            SecurityDescriptor: ptr::null(),
            SecurityQualityOfService: ptr::null(),
        };
        Kmt::status(
            unsafe {
                D3DKMTShareObjects(
                    1,
                    &fence.fence.object,
                    &attributes,
                    SYNC_OBJECT_ALL_ACCESS,
                    &mut fence.nt_handle,
                )
            },
            "ShareObjects(monitored fence)",
        )?;
        Ok(fence)
    }

    /// Opens a monitored fence another device shared through `nt_handle`. Like lfrb's import, the
    /// fence is opened without GPU access: queue waits and signals go through the OS scheduler.
    pub fn open_shared_fence(&mut self, nt_handle: *mut c_void) -> Result<OpenedFence> {
        self.session.healthy()?;
        if nt_handle.is_null() {
            return Err(Error::invalid("null fence NT handle"));
        }
        let mut open = D3DKMT_OPENSYNCOBJECTFROMNTHANDLE2 {
            hNtHandle: nt_handle,
            hDevice: self.session.device,
            ..Default::default()
        };
        open.Flags.Anonymous.Value = SYNC_SHARED | SYNC_NT_SECURITY_SHARING | SYNC_NO_GPU_ACCESS;
        Kmt::status(
            unsafe { D3DKMTOpenSyncObjectFromNtHandle2(&mut open) },
            "OpenSyncObjectFromNtHandle2",
        )?;
        Ok(OpenedFence {
            session: self.session.clone(),
            fence: MonitoredFence {
                object: open.hSyncObject,
                value: unsafe { open.Anonymous.MonitoredFence.FenceValueCPUVirtualAddress }
                    .cast_const()
                    .cast(),
            },
        })
    }

    /// Opens a single-allocation resource another device shared through `nt_handle`, maps it in
    /// the runtime VA window and makes it resident, sized by the AMD records in its private data:
    /// the whole allocation, which for a D3D12 committed buffer is its width rounded up to 256 KiB.
    /// It has no CPU mapping; the GPU writes it through PM4 like any owned buffer. A
    /// resource without readable records needs [`Self::open_shared_buffer_sized`].
    pub fn open_shared_buffer(
        &mut self,
        nt_handle: *mut c_void,
        kind: MemoryKind,
    ) -> Result<Buffer> {
        self.open_shared(nt_handle, kind, None)
    }

    /// [`Self::open_shared_buffer`] for a resource whose size its creator knows: `bytes` rounded
    /// up to whole 4 KiB pages is mapped, which for a D3D12 committed buffer is its width. Where
    /// the AMD records are readable (v8 from the D3D12 driver, v9 from Vulkan) the allocation they
    /// describe must hold at least `bytes`.
    pub fn open_shared_buffer_sized(
        &mut self,
        nt_handle: *mut c_void,
        kind: MemoryKind,
        bytes: u64,
    ) -> Result<Buffer> {
        self.open_shared(nt_handle, kind, Some(bytes))
    }

    /// Opens the resource behind `nt_handle` only to read its private data, then closes it:
    /// nothing is mapped or made resident.
    pub fn inspect_shared_resource(
        &mut self,
        nt_handle: *mut c_void,
    ) -> Result<SharedResourceData> {
        let (_opened, data) = self.open_resource(nt_handle, MemoryKind::Device)?;
        Ok(data)
    }

    fn open_shared(
        &mut self,
        nt_handle: *mut c_void,
        kind: MemoryKind,
        bytes: Option<u64>,
    ) -> Result<Buffer> {
        if kind.is_executable() {
            return Err(Error::invalid("shared buffers are never mapped executable"));
        }
        let (mut a, data) = self.open_resource(nt_handle, kind)?;
        a.size = shared_size(&data, bytes)? as usize;
        map_and_make_resident(&self.session, &mut a, 0)?;
        Ok(Buffer {
            allocation: Rc::new(a),
        })
    }

    /// Opens the single-allocation resource behind `nt_handle`: the opened allocation, not yet
    /// mapped, which closes the resource when dropped, and its private data.
    fn open_resource(
        &mut self,
        nt_handle: *mut c_void,
        kind: MemoryKind,
    ) -> Result<(Allocation, SharedResourceData)> {
        self.session.healthy()?;
        if nt_handle.is_null() {
            return Err(Error::invalid("null resource NT handle"));
        }
        let mut query = D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE {
            hDevice: self.session.device,
            hNtHandle: nt_handle,
            ..Default::default()
        };
        Kmt::status(
            unsafe { D3DKMTQueryResourceInfoFromNtHandle(&mut query) },
            "QueryResourceInfoFromNtHandle",
        )?;
        if query.NumAllocations != 1 {
            return Err(Error::unsupported(format!(
                "shared resource has {} allocations; a buffer has one",
                query.NumAllocations
            )));
        }
        let mut total = vec![0u8; query.TotalPrivateDriverDataSize as usize];
        let mut resource = vec![0u8; query.ResourcePrivateDriverDataSize as usize];
        let mut runtime = vec![0u8; query.PrivateRuntimeDataSize as usize];
        let mut info = D3DDDI_OPENALLOCATIONINFO2::default();
        let mut open = D3DKMT_OPENRESOURCEFROMNTHANDLE {
            hDevice: self.session.device,
            hNtHandle: nt_handle,
            NumAllocations: 1,
            pOpenAllocationInfo2: &mut info,
            PrivateRuntimeDataSize: runtime.len() as u32,
            pPrivateRuntimeData: runtime.as_mut_ptr().cast(),
            ResourcePrivateDriverDataSize: resource.len() as u32,
            pResourcePrivateDriverData: resource.as_mut_ptr().cast(),
            TotalPrivateDriverDataBufferSize: total.len() as u32,
            pTotalPrivateDriverDataBuffer: total.as_mut_ptr().cast(),
            ..Default::default()
        };
        Kmt::status(
            unsafe { D3DKMTOpenResourceFromNtHandle(&mut open) },
            "OpenResourceFromNtHandle",
        )?;
        let mut a = Allocation::new(&self.session, kind);
        a.handle = info.hAllocation;
        a.resource = open.hResource;
        self.session
            .live_allocations
            .set(self.session.live_allocations.get() + 1);
        let allocation =
            allocation_range(&total, info.pPrivateDriverData, info.PrivateDriverDataSize);
        Ok((
            a,
            SharedResourceData {
                allocations: query.NumAllocations,
                runtime,
                resource,
                total,
                allocation,
                gpu_virtual_address: info.GpuVirtualAddress,
            },
        ))
    }

    /// Makes later submissions on this queue wait until `fence` reaches `value`.
    pub fn queue_wait(&mut self, fence: &impl QueueFence, value: u64) -> Result<()> {
        self.session.healthy()?;
        let object = fence.sync_object();
        let wait = D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE {
            hHwQueue: self.graphics.handle,
            ObjectCount: 1,
            ObjectHandleArray: &object,
            FenceValueArray: &value,
        };
        if let Err(error) = Kmt::status(
            unsafe { D3DKMTSubmitWaitForSyncObjectsToHwQueue(&wait) },
            "SubmitWaitForSyncObjectsToHwQueue",
        ) {
            self.session.poison(&error.message);
            return Err(error);
        }
        Ok(())
    }

    /// Makes the work queued on `waiter` from now on wait, on the GPU, until submission `serial`
    /// of `producer` has completed: the waiter's hardware queue waits on the producer's progress
    /// fence, as a D3D12 queue waits on a fence. The waiting queue holds no packet that polls
    /// meanwhile, so a wait for work that takes longer than the TDR timeout cannot fault. A
    /// refused wait queues nothing and leaves the device usable.
    pub fn wait_for_queue(
        &mut self,
        waiter: super::QueueKind,
        producer: super::QueueKind,
        serial: u64,
    ) -> Result<()> {
        self.session.healthy()?;
        let (object, submitted) = {
            let queue = self.hw_queue(producer)?;
            (queue.progress_fence.object, queue.submitted)
        };
        if serial > submitted {
            return Err(Error::invalid(format!(
                "{waiter:?} cannot wait for submission {serial} of {producer:?}, which was never queued"
            )));
        }
        let wait = D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE {
            hHwQueue: self.hw_queue(waiter)?.handle,
            ObjectCount: 1,
            ObjectHandleArray: &object,
            FenceValueArray: &serial,
        };
        Kmt::status(
            unsafe { D3DKMTSubmitWaitForSyncObjectsToHwQueue(&wait) },
            "SubmitWaitForSyncObjectsToHwQueue(progress fence)",
        )
    }

    /// Signals `fence` to `value` once all earlier work on this queue has completed.
    pub fn queue_signal(&mut self, fence: &impl QueueFence, value: u64) -> Result<()> {
        self.session.healthy()?;
        let object = fence.sync_object();
        let signal = D3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE {
            BroadcastHwQueueCount: 1,
            BroadcastHwQueueArray: &self.graphics.handle,
            ObjectCount: 1,
            ObjectHandleArray: &object,
            FenceValueArray: &value,
            ..Default::default()
        };
        if let Err(error) = Kmt::status(
            unsafe { D3DKMTSubmitSignalSyncObjectsToHwQueue(&signal) },
            "SubmitSignalSyncObjectsToHwQueue",
        ) {
            self.session.poison(&error.message);
            return Err(error);
        }
        Ok(())
    }
}
