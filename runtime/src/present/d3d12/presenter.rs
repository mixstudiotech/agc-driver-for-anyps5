//! D3D12 copies GPU-converted shared slots into the swapchain. F_ready orders the native
//! writer before each copy; F_copy protects slot reuse. No CPU pixel mapping or upload path.

use super::{footprint::Footprint, schedule::FrameSchedule};
use crate::{Error, ErrorKind, Result};
use std::{ffi::c_void, mem::ManuallyDrop, ptr, time::Duration};
use windows::{
    core::{Interface, BOOL, HRESULT, PCWSTR},
    Win32::{
        Foundation::{GENERIC_ALL, HANDLE, HWND, LUID},
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_0,
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
    },
};
use windows_sys::Win32::{
    Foundation::{CloseHandle, WAIT_OBJECT_0, WAIT_TIMEOUT},
    System::Threading::{CreateEventW, WaitForSingleObject},
};

const BUFFER_COUNT: u32 = 3;
/// D3D12 allocation granularity for the interchange buffers.
const BUFFER_GRANULARITY: u64 = 64 * 1024;

fn system_error(error: &windows::core::Error, what: &str) -> Error {
    Error::new(
        ErrorKind::System,
        format!(
            "{what}: HRESULT=0x{:08x} {}",
            error.code().0 as u32,
            error.message()
        ),
    )
}

fn hr<T>(result: windows::core::Result<T>, what: &str) -> Result<T> {
    result.map_err(|error| system_error(&error, what))
}

/// The adapter the D3D12 device is created on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Adapter {
    /// The adapter with this LUID (low, high), the native device's.
    Luid(u32, i32),
    /// The first hardware adapter DXGI enumerates.
    Default,
    /// WARP, the software rasterizer, for tests.
    Warp,
}

/// Where presented frames go.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Output {
    /// A flip-discard swapchain on this HWND.
    Window(*mut c_void),
    /// A texture nothing displays, for tests; frames are only read back.
    Offscreen,
}

/// Who writes the interchange slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotSource {
    /// DEFAULT buffers shared with the native device, which signals F_ready (an NT handle).
    Shared { ready_fence: *mut c_void },
    /// Shared buffers and both fences are created here and imported by the native device.
    Native,
}

/// How Present waits for the display.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyncMode {
    /// One frame per vertical blank, never torn.
    Fifo,
    /// The latest frame at the next vertical blank, never blocking on the display.
    Mailbox,
    /// At once, torn if the system allows tearing; otherwise as mailbox.
    Immediate,
}

/// Presenter settings. `readback` copies every presented back buffer to a READBACK buffer for
/// validation runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresenterConfig {
    pub width: u32,
    pub height: u32,
    pub slots: usize,
    pub readback: bool,
    /// Bound on every CPU wait for F_copy.
    pub timeout: Duration,
    pub sync: SyncMode,
    pub max_frame_latency: u32,
    pub debug_layer: bool,
}

impl PresenterConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            slots: 2,
            readback: false,
            timeout: Duration::from_secs(2),
            sync: SyncMode::Fifo,
            max_frame_latency: 1,
            debug_layer: false,
        }
    }
}

/// DXGI's present counters after a run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PresentStatistics {
    pub last_present_count: u32,
    /// `None` while DXGI reports the statistics as disjoint or unavailable.
    pub present_count: Option<u32>,
    pub present_refresh_count: Option<u32>,
}

struct Slot {
    resource: ID3D12Resource,
    /// NT handle for the native device (shared slots).
    shared: Option<HANDLE>,
    bytes: u64,
    readback: Option<ID3D12Resource>,
}

enum Target {
    Swapchain {
        swapchain: IDXGISwapChain4,
        waitable: HANDLE,
        tearing: bool,
    },
    Offscreen {
        texture: ID3D12Resource,
    },
}

/// One flip-model swapchain (or offscreen texture) fed from S interchange buffers.
pub struct D3d12Presenter {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    target: Target,
    allocators: Vec<ID3D12CommandAllocator>,
    list: ID3D12GraphicsCommandList,
    slots: Vec<Slot>,
    copy_fence: ID3D12Fence,
    copy_shared: Option<HANDLE>,
    ready_fence: Option<ID3D12Fence>,
    ready_shared: Option<HANDLE>,
    event: *mut c_void,
    footprint: Footprint,
    schedule: FrameSchedule,
    config: PresenterConfig,
    frames: u64,
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // A non-owning copy: no AddRef, and the barrier is never dropped.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn subresource(resource: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    }
}

fn placed(resource: &ID3D12Resource, footprint: &Footprint) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: footprint.offset,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    Width: footprint.width,
                    Height: footprint.height,
                    Depth: 1,
                    RowPitch: footprint.row_pitch as u32,
                },
            },
        },
    }
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

fn texture_desc(width: u32, height: u32) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: u64::from(width),
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

fn heap(kind: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: kind,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    }
}

fn committed(
    device: &ID3D12Device,
    kind: D3D12_HEAP_TYPE,
    flags: D3D12_HEAP_FLAGS,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
    what: &str,
) -> Result<ID3D12Resource> {
    let mut resource: Option<ID3D12Resource> = None;
    hr(
        unsafe {
            device.CreateCommittedResource(&heap(kind), flags, desc, state, None, &mut resource)
        },
        what,
    )?;
    resource.ok_or_else(|| Error::new(ErrorKind::System, format!("{what} returned none")))
}

/// Waits for a Win32 handle; `Ok(false)` when `timeout` passed first.
fn wait_handle(handle: *mut c_void, timeout: Duration) -> Result<bool> {
    let millis = timeout.as_millis().min(u128::from(u32::MAX - 1)) as u32;
    match unsafe { WaitForSingleObject(handle, millis) } {
        WAIT_OBJECT_0 => Ok(true),
        WAIT_TIMEOUT => Ok(false),
        other => Err(Error::new(
            ErrorKind::System,
            format!("WaitForSingleObject returned 0x{other:x}"),
        )),
    }
}

fn adapter(factory: &IDXGIFactory6, adapter: Adapter) -> Result<IDXGIAdapter1> {
    unsafe {
        match adapter {
            Adapter::Luid(low, high) => hr(
                factory.EnumAdapterByLuid(LUID {
                    LowPart: low,
                    HighPart: high,
                }),
                "EnumAdapterByLuid",
            ),
            Adapter::Warp => hr(factory.EnumWarpAdapter(), "EnumWarpAdapter"),
            Adapter::Default => {
                let mut index = 0;
                loop {
                    let candidate = hr(
                        factory.EnumAdapters1(index),
                        "EnumAdapters1 (looking for a hardware adapter)",
                    )?;
                    let desc = hr(candidate.GetDesc1(), "IDXGIAdapter1::GetDesc1")?;
                    if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                        return Ok(candidate);
                    }
                    index += 1;
                }
            }
        }
    }
}

/// Whether DXGI lets flip-model swapchains tear (variable refresh, immediate presents).
fn tearing_allowed(factory: &IDXGIFactory6) -> bool {
    let mut allowed = BOOL(0);
    let checked = unsafe {
        factory.CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING,
            (&mut allowed as *mut BOOL).cast(),
            std::mem::size_of::<BOOL>() as u32,
        )
    };
    checked.is_ok() && allowed.as_bool()
}

fn device_lost(code: HRESULT) -> bool {
    code == DXGI_ERROR_DEVICE_REMOVED
        || code == DXGI_ERROR_DEVICE_RESET
        || code == DXGI_ERROR_DEVICE_HUNG
}

impl D3d12Presenter {
    /// The Milestone 1b presenter: the swapchain on `hwnd`, shared slots and F_copy for the
    /// native device, and F_ready opened from its NT handle.
    pub fn new(
        luid: (u32, i32),
        hwnd: *mut c_void,
        ready_fence: *mut c_void,
        config: PresenterConfig,
    ) -> Result<Self> {
        if hwnd.is_null() || ready_fence.is_null() {
            return Err(Error::invalid(
                "the presenter needs a window and the F_ready handle",
            ));
        }
        Self::create(
            Adapter::Luid(luid.0, luid.1),
            Output::Window(hwnd),
            SlotSource::Shared { ready_fence },
            config,
        )
    }

    /// Creates the D3D12 device on `adapter`, the target, the slots and F_copy.
    pub fn create(
        adapter_choice: Adapter,
        output: Output,
        source: SlotSource,
        config: PresenterConfig,
    ) -> Result<Self> {
        let footprint = Footprint::bgra8(config.width, config.height)?;
        let schedule = FrameSchedule::new(config.slots)?;
        if !(1..=3).contains(&config.max_frame_latency) {
            return Err(Error::invalid("the frame latency must be 1 to 3"));
        }
        unsafe {
            if config.debug_layer {
                let mut debug: Option<ID3D12Debug> = None;
                hr(
                    D3D12GetDebugInterface(&mut debug),
                    "D3D12GetDebugInterface (is the Graphics Tools debug layer installed?)",
                )?;
                if let Some(debug) = debug {
                    debug.EnableDebugLayer();
                }
            }
            let factory_flags = if config.debug_layer {
                DXGI_CREATE_FACTORY_DEBUG
            } else {
                DXGI_CREATE_FACTORY_FLAGS(0)
            };
            let factory: IDXGIFactory6 =
                hr(CreateDXGIFactory2(factory_flags), "CreateDXGIFactory2")?;
            let adapter = adapter(&factory, adapter_choice)?;
            let mut device: Option<ID3D12Device> = None;
            hr(
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device),
                "D3D12CreateDevice",
            )?;
            let device = device
                .ok_or_else(|| Error::new(ErrorKind::System, "D3D12CreateDevice returned none"))?;
            let queue: ID3D12CommandQueue = hr(
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                }),
                "CreateCommandQueue",
            )?;
            let target = match output {
                Output::Window(hwnd) => Self::create_swapchain(&factory, &queue, hwnd, &config)?,
                Output::Offscreen => Target::Offscreen {
                    texture: committed(
                        &device,
                        D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_HEAP_FLAG_NONE,
                        &texture_desc(config.width, config.height),
                        D3D12_RESOURCE_STATE_COMMON,
                        "CreateCommittedResource(offscreen target)",
                    )?,
                },
            };
            let copy_fence: ID3D12Fence = hr(
                device.CreateFence(0, D3D12_FENCE_FLAG_SHARED),
                "CreateFence(F_copy)",
            )?;
            let (copy_shared, ready_fence, ready_shared) = match source {
                SlotSource::Shared { ready_fence } => {
                    let shared = hr(
                        device.CreateSharedHandle(&copy_fence, None, GENERIC_ALL.0, PCWSTR::null()),
                        "CreateSharedHandle(F_copy)",
                    )?;
                    let mut ready: Option<ID3D12Fence> = None;
                    hr(
                        device.OpenSharedHandle(HANDLE(ready_fence), &mut ready),
                        "OpenSharedHandle(F_ready)",
                    )?;
                    let ready = ready.ok_or_else(|| {
                        Error::new(ErrorKind::System, "OpenSharedHandle returned none")
                    })?;
                    (Some(shared), Some(ready), None)
                }
                SlotSource::Native => {
                    let ready: ID3D12Fence = hr(device.CreateFence(0, D3D12_FENCE_FLAG_SHARED), "CreateFence(F_ready)")?;
                    let ready_shared = hr(device.CreateSharedHandle(&ready, None, GENERIC_ALL.0, PCWSTR::null()), "CreateSharedHandle(F_ready)")?;
                    let copied = match device.CreateSharedHandle(&copy_fence, None, GENERIC_ALL.0, PCWSTR::null()) {
                        Ok(handle) => handle,
                        Err(error) => { CloseHandle(ready_shared.0); return Err(system_error(&error, "CreateSharedHandle(F_copy)")); }
                    };
                    (Some(copied), Some(ready), Some(ready_shared))
                }
            };
            let mut allocators = Vec::with_capacity(config.slots);
            for _ in 0..config.slots {
                allocators.push(hr(
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT),
                    "CreateCommandAllocator",
                )?);
            }
            let list: ID3D12GraphicsCommandList = hr(
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocators[0], None),
                "CreateCommandList",
            )?;
            hr(list.Close(), "CommandList::Close")?;
            let event = CreateEventW(ptr::null(), 0, 0, ptr::null());
            if event.is_null() {
                return Err(Error::new(ErrorKind::System, "CreateEventW failed"));
            }
            let mut presenter = Self {
                device,
                queue,
                target,
                allocators,
                list,
                slots: Vec::new(),
                copy_fence,
                copy_shared,
                ready_fence,
                ready_shared,
                event,
                footprint,
                schedule,
                config,
                frames: 0,
            };
            presenter.check_footprint(&footprint)?;
            presenter.slots = presenter.create_slots(footprint.required_bytes())?;
            Ok(presenter)
        }
    }

    unsafe fn create_swapchain(
        factory: &IDXGIFactory6,
        queue: &ID3D12CommandQueue,
        hwnd: *mut c_void,
        config: &PresenterConfig,
    ) -> Result<Target> {
        if hwnd.is_null() {
            return Err(Error::invalid("the swapchain needs a window"));
        }
        let tearing = config.sync == SyncMode::Immediate && tearing_allowed(factory);
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: config.width,
            Height: config.height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: BUFFER_COUNT,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            Flags: Self::swapchain_flags(tearing).0 as u32,
        };
        let window = HWND(hwnd);
        let swapchain: IDXGISwapChain1 = hr(
            unsafe { factory.CreateSwapChainForHwnd(queue, window, &desc, None, None) },
            "CreateSwapChainForHwnd",
        )?;
        // The window belongs to VideoOut: DXGI must neither watch its messages nor toggle
        // fullscreen on Alt+Enter.
        hr(
            unsafe {
                factory.MakeWindowAssociation(
                    window,
                    DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER,
                )
            },
            "MakeWindowAssociation",
        )?;
        let swapchain: IDXGISwapChain4 = hr(swapchain.cast(), "IDXGISwapChain4")?;
        hr(
            unsafe { swapchain.SetMaximumFrameLatency(config.max_frame_latency) },
            "SetMaximumFrameLatency",
        )?;
        let waitable = unsafe { swapchain.GetFrameLatencyWaitableObject() };
        Ok(Target::Swapchain {
            swapchain,
            waitable,
            tearing,
        })
    }

    fn swapchain_flags(tearing: bool) -> DXGI_SWAP_CHAIN_FLAG {
        let waitable = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        if tearing {
            waitable | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            waitable
        }
    }

    /// The resource frames are copied into: the current back buffer or the offscreen texture.
    fn copy_target(&self) -> Result<ID3D12Resource> {
        match &self.target {
            Target::Swapchain { swapchain, .. } => unsafe {
                let index = swapchain.GetCurrentBackBufferIndex();
                hr(swapchain.GetBuffer(index), "GetBuffer")
            },
            Target::Offscreen { texture } => Ok(texture.clone()),
        }
    }

    /// Checks `footprint` against `GetCopyableFootprints` for the target's description.
    fn check_footprint(&self, footprint: &Footprint) -> Result<()> {
        let desc = unsafe { self.copy_target()?.GetDesc() };
        let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        unsafe {
            self.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut layout),
                None,
                None,
                Some(&mut total),
            );
        }
        if u64::from(layout.Footprint.RowPitch) != footprint.row_pitch
            || total != footprint.required_bytes()
            || layout.Footprint.Width != footprint.width
            || layout.Footprint.Height != footprint.height
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!(
                    "GetCopyableFootprints gives {}x{} pitch {} / {} bytes, the frame layout {}x{} {} / {}",
                    layout.Footprint.Width,
                    layout.Footprint.Height,
                    layout.Footprint.RowPitch,
                    total,
                    footprint.width,
                    footprint.height,
                    footprint.row_pitch,
                    footprint.required_bytes()
                ),
            ));
        }
        Ok(())
    }

    fn create_slots(&self, required: u64) -> Result<Vec<Slot>> {
        let bytes = required.next_multiple_of(BUFFER_GRANULARITY);
        let mut slots = Vec::with_capacity(self.config.slots);
        for _ in 0..self.config.slots {
            let resource = committed(
                &self.device,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_HEAP_FLAG_SHARED,
                &buffer_desc(bytes),
                D3D12_RESOURCE_STATE_COMMON,
                "CreateCommittedResource(shared interchange buffer)",
            )?;
            let shared = hr(
                unsafe {
                    self.device.CreateSharedHandle(
                        &resource,
                        None,
                        GENERIC_ALL.0,
                        PCWSTR::null(),
                    )
                },
                "CreateSharedHandle(interchange buffer)",
            )?;
            let shared = Some(shared);
            let readback = if self.config.readback {
                Some(committed(
                    &self.device,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(bytes),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    "CreateCommittedResource(readback)",
                )?)
            } else {
                None
            };
            slots.push(Slot {
                resource,
                shared,
                bytes,
                readback,
            });
        }
        Ok(slots)
    }

    pub fn footprint(&self) -> Footprint {
        self.footprint
    }

    pub fn schedule(&self) -> FrameSchedule {
        self.schedule
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Whether immediate presents tear.
    pub fn tearing(&self) -> bool {
        matches!(self.target, Target::Swapchain { tearing: true, .. })
    }

    /// The NT handles of shared interchange buffers, in slot order, for the native device.
    pub fn slot_handles(&self) -> Vec<*mut c_void> {
        self.slots
            .iter()
            .filter_map(|slot| slot.shared.map(|handle| handle.0))
            .collect()
    }

    /// The NT handle of F_copy for the native device; null with upload slots.
    pub fn copy_fence_handle(&self) -> *mut c_void {
        self.copy_shared.map_or(ptr::null_mut(), |handle| handle.0)
    }

    pub fn ready_fence_handle(&self) -> *mut c_void {
        self.ready_shared.map_or(ptr::null_mut(), |handle| handle.0)
    }

    pub fn copied(&self) -> u64 {
        unsafe { self.copy_fence.GetCompletedValue() }
    }

    /// F_ready's completed value; 0 with upload slots.
    pub fn ready(&self) -> u64 {
        self.ready_fence
            .as_ref()
            .map_or(0, |fence| unsafe { fence.GetCompletedValue() })
    }

    /// Resizes the target to `width` x `height` once every presented frame has been copied,
    /// growing the slots when the frame no longer fits them.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if (width, height) == (self.footprint.width, self.footprint.height) {
            return Ok(());
        }
        let footprint = Footprint::bgra8(width, height)?;
        self.wait_copied(self.frames)?;
        match &mut self.target {
            Target::Swapchain {
                swapchain, tearing, ..
            } => {
                let result = unsafe {
                    swapchain.ResizeBuffers(
                        BUFFER_COUNT,
                        width,
                        height,
                        DXGI_FORMAT_B8G8R8A8_UNORM,
                        Self::swapchain_flags(*tearing),
                    )
                };
                if let Err(error) = result {
                    return Err(self.device_error(error, "ResizeBuffers"));
                }
            }
            Target::Offscreen { texture } => {
                *texture = committed(
                    &self.device,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_HEAP_FLAG_NONE,
                    &texture_desc(width, height),
                    D3D12_RESOURCE_STATE_COMMON,
                    "CreateCommittedResource(offscreen target)",
                )?;
            }
        }
        self.check_footprint(&footprint)?;
        if self
            .slots
            .iter()
            .any(|slot| slot.bytes < footprint.required_bytes())
        {
            if self.ready_fence.is_some() {
                return Err(Error::unsupported(
                    "shared interchange buffers cannot grow without a new handle exchange",
                ));
            }
            self.slots = self.create_slots(footprint.required_bytes())?;
        }
        self.config.width = width;
        self.config.height = height;
        self.footprint = footprint;
        Ok(())
    }

    /// Waits on the frame-latency waitable object; `Ok(false)` when `timeout` passed first, which
    /// means the display is not taking frames (an occluded or cloaked window).
    pub fn wait_for_frame(&self, timeout: Duration) -> Result<bool> {
        match &self.target {
            Target::Swapchain { waitable, .. } => wait_handle(waitable.0, timeout),
            Target::Offscreen { .. } => Ok(true),
        }
    }

    /// Presents frame `frame` (the previous frame + 1): with shared slots the queue first waits
    /// for F_ready = frame; it copies the frame's slot into the back buffer, signals
    /// F_copy = frame and presents with the configured sync mode. The caller waits on
    /// [`Self::wait_for_frame`] beforehand.
    pub fn present(&mut self, frame: u64) -> Result<()> {
        if frame != self.frames + 1 {
            return Err(Error::invalid(format!(
                "frame {frame} presented after frame {}",
                self.frames
            )));
        }
        if let Some(previous) = self.schedule.reuse_wait(frame) {
            // The allocator and readback buffer of this slot were last used by that frame.
            self.wait_copied(previous)?;
        }
        let slot = self.schedule.slot(frame);
        let back = self.copy_target()?;
        unsafe {
            let allocator = &self.allocators[slot];
            hr(allocator.Reset(), "CommandAllocator::Reset")?;
            hr(self.list.Reset(allocator, None), "CommandList::Reset")?;
            self.list.ResourceBarrier(&[transition(
                &back,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            self.list.CopyTextureRegion(
                &subresource(&back),
                0,
                0,
                0,
                &placed(&self.slots[slot].resource, &self.footprint),
                None,
            );
            match &self.slots[slot].readback {
                Some(readback) => {
                    self.list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    self.list.CopyTextureRegion(
                        &placed(readback, &self.footprint),
                        0,
                        0,
                        0,
                        &subresource(&back),
                        None,
                    );
                    self.list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                }
                None => self.list.ResourceBarrier(&[transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                )]),
            }
            hr(self.list.Close(), "CommandList::Close")?;
            if let Some(ready) = &self.ready_fence {
                hr(self.queue.Wait(ready, frame), "Queue::Wait(F_ready)")?;
            }
            let list: ID3D12CommandList = hr(self.list.cast(), "ID3D12CommandList")?;
            self.queue.ExecuteCommandLists(&[Some(list)]);
            if let Err(error) = self.queue.Signal(&self.copy_fence, frame) {
                return Err(self.device_error(error, "Queue::Signal(F_copy)"));
            }
            if let Target::Swapchain {
                swapchain, tearing, ..
            } = &self.target
            {
                let (interval, flags) = match self.config.sync {
                    SyncMode::Fifo => (1, DXGI_PRESENT(0)),
                    SyncMode::Immediate if *tearing => (0, DXGI_PRESENT_ALLOW_TEARING),
                    SyncMode::Mailbox | SyncMode::Immediate => (0, DXGI_PRESENT(0)),
                };
                // DXGI_STATUS_OCCLUDED is a success code: the frame counts as presented.
                let presented =
                    swapchain.Present1(interval, flags, &DXGI_PRESENT_PARAMETERS::default());
                if let Err(error) = presented.ok() {
                    return Err(self.device_error(error, "Present1"));
                }
            }
        }
        self.frames = frame;
        Ok(())
    }

    /// Maps a failed D3D12/DXGI call to device loss when the device is gone.
    fn device_error(&self, error: windows::core::Error, what: &str) -> Error {
        let removed = self.removed_reason();
        if device_lost(error.code()) || removed.is_some() {
            Error::new(
                ErrorKind::DeviceLost,
                format!(
                    "{what}: the D3D12 device was lost (HRESULT=0x{:08x}; removal reason {})",
                    error.code().0 as u32,
                    removed.as_deref().unwrap_or("none")
                ),
            )
        } else {
            system_error(&error, what)
        }
    }

    /// Blocks until F_copy reaches `frame` or the configured timeout passes.
    pub fn wait_copied(&self, frame: u64) -> Result<()> {
        if self.copied() >= frame {
            return Ok(());
        }
        hr(
            unsafe {
                self.copy_fence
                    .SetEventOnCompletion(frame, HANDLE(self.event))
            },
            "Fence::SetEventOnCompletion(F_copy)",
        )?;
        if wait_handle(self.event, self.config.timeout)? {
            return Ok(());
        }
        Err(Error::new(
            ErrorKind::DeviceLost,
            format!(
                "F_copy {frame} did not complete within {} ms (removal reason {})",
                self.config.timeout.as_millis(),
                self.removed_reason().as_deref().unwrap_or("none")
            ),
        ))
    }

    /// The back buffer frame `frame` presented, as copied to the readback buffer, laid out like
    /// [`Self::footprint`]. Waits for the frame's F_copy.
    pub fn read_back(&self, frame: u64) -> Result<Vec<u8>> {
        if frame == 0 || frame > self.frames {
            return Err(Error::invalid(format!(
                "frame {frame} has not been presented (last {})",
                self.frames
            )));
        }
        self.wait_copied(frame)?;
        let readback = self.slots[self.schedule.slot(frame)]
            .readback
            .as_ref()
            .ok_or_else(|| Error::invalid("the presenter was created without readback"))?;
        let bytes = self.footprint.required_bytes() as usize;
        let mut data = ptr::null_mut();
        unsafe {
            hr(
                readback.Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: bytes,
                    }),
                    Some(&mut data),
                ),
                "Resource::Map(readback)",
            )?;
            let copy = std::slice::from_raw_parts(data.cast::<u8>(), bytes).to_vec();
            readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            Ok(copy)
        }
    }

    /// DXGI's present counters; all zero for an offscreen target.
    pub fn statistics(&self) -> Result<PresentStatistics> {
        let Target::Swapchain { swapchain, .. } = &self.target else {
            return Ok(PresentStatistics::default());
        };
        unsafe {
            let last_present_count = hr(swapchain.GetLastPresentCount(), "GetLastPresentCount")?;
            let mut frame = DXGI_FRAME_STATISTICS::default();
            let available = swapchain.GetFrameStatistics(&mut frame).is_ok();
            Ok(PresentStatistics {
                last_present_count,
                present_count: available.then_some(frame.PresentCount),
                present_refresh_count: available.then_some(frame.PresentRefreshCount),
            })
        }
    }

    /// The device-removed reason, when the D3D12 device was removed.
    pub fn removed_reason(&self) -> Option<String> {
        unsafe { self.device.GetDeviceRemovedReason() }
            .err()
            .map(|error| {
                format!(
                    "HRESULT=0x{:08x} {}",
                    error.code().0 as u32,
                    error.message()
                )
            })
    }
}

/// Experiment S1b: creates a D3D12 device on the adapter with `luid`, places committed buffers
/// of `sizes` bytes in DEFAULT memory, and returns their GPU virtual addresses, so a caller can
/// check they land outside its fixed-VA reservations. Everything is released before returning.
pub fn committed_buffer_addresses(luid: (u32, i32), sizes: &[u64]) -> Result<Vec<u64>> {
    unsafe {
        let factory: IDXGIFactory6 = hr(
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)),
            "CreateDXGIFactory2",
        )?;
        let adapter = adapter(&factory, Adapter::Luid(luid.0, luid.1))?;
        let mut device: Option<ID3D12Device> = None;
        hr(
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device),
            "D3D12CreateDevice",
        )?;
        let device = device
            .ok_or_else(|| Error::new(ErrorKind::System, "D3D12CreateDevice returned none"))?;
        let mut resources = Vec::with_capacity(sizes.len());
        for &bytes in sizes {
            resources.push(committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(bytes),
                D3D12_RESOURCE_STATE_COMMON,
                "CreateCommittedResource",
            )?);
        }
        Ok(resources.iter().map(|r| r.GetGPUVirtualAddress()).collect())
    }
}

impl Drop for D3d12Presenter {
    fn drop(&mut self) {
        // Let the queue finish with the slots before they and the fences go away.
        let _ = self.wait_copied(self.frames);
        unsafe {
            for slot in &self.slots {
                if let Some(shared) = slot.shared {
                    CloseHandle(shared.0);
                }
            }
            if let Some(shared) = self.copy_shared {
                CloseHandle(shared.0);
            }
            if let Some(shared) = self.ready_shared {
                CloseHandle(shared.0);
            }
            if let Target::Swapchain { waitable, .. } = &self.target {
                CloseHandle(waitable.0);
            }
            CloseHandle(self.event);
        }
    }
}
