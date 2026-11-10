use super::*;
use crate::present::d3d12::{Adapter, D3d12Presenter, Output, PresenterConfig, SlotSource};
use crate::present::handoff::{self, NtHandle};
use windows_sys::Win32::{Foundation::{HWND, LPARAM, LRESULT, WPARAM}, System::LibraryLoader::GetModuleHandleW, UI::WindowsAndMessaging::*};

unsafe extern "system" fn window_proc(window: HWND, message: u32, w: WPARAM, l: LPARAM) -> LRESULT { unsafe { DefWindowProcW(window, message, w, l) } }
struct TestWindow(HWND);
impl TestWindow {
    fn new() -> Self {
        let name: Vec<u16> = "AnyPS5 native GPU shared presentation\0".encode_utf16().collect();
        let instance = unsafe { GetModuleHandleW(std::ptr::null()) };
        let class = WNDCLASSW { lpfnWndProc: Some(window_proc), hInstance: instance, lpszClassName: name.as_ptr(), ..unsafe { std::mem::zeroed() } };
        assert_ne!(unsafe { RegisterClassW(&class) }, 0);
        let window = unsafe { CreateWindowExW(0, name.as_ptr(), name.as_ptr(), WS_OVERLAPPEDWINDOW | WS_VISIBLE, 120, 120, 640, 360, std::ptr::null_mut(), std::ptr::null_mut(), instance, std::ptr::null()) };
        assert!(!window.is_null()); Self(window)
    }
    fn pump(&self) {
        let mut message = unsafe { std::mem::zeroed() };
        while unsafe { PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) } != 0 { unsafe { TranslateMessage(&message); DispatchMessageW(&message); } }
    }
}
impl Drop for TestWindow { fn drop(&mut self) { unsafe { DestroyWindow(self.0); } } }

#[test]
#[ignore = "requires an active Windows desktop and real native AMD/D3D12 GPU queues"]
fn native_shared_slots_capture_overlay_reuse_and_resize() {
    let window = TestWindow::new();
    let mut device = Device::open().unwrap();
    let isa = crate::gpu::target::host_target(device.identity());
    let luid = (device.identity().luid_low, device.identity().luid_high);
    let mut display = NativeDisplay::new(&mut device, isa).unwrap();
    for (width, height) in [(64u32, 32u32), (96, 40)] {
        let canvas = Canvas::exact(width, height).unwrap();
        let mut config = PresenterConfig::new(width, height); config.readback = true;
        let mut presenter = D3d12Presenter::create(Adapter::Luid(luid.0, luid.1), Output::Window(window.0.cast()), SlotSource::Native, config).unwrap();
        let generation = handoff::next_generation();
        let mut target = Some(NativeTarget { window: window.0 as usize, generation, canvas, footprint: presenter.footprint(),
            slots: presenter.slot_handles().into_iter().map(|handle| NtHandle::duplicate(handle as usize).unwrap()).collect(),
            ready: NtHandle::duplicate(presenter.ready_fence_handle() as usize).unwrap(), copied: NtHandle::duplicate(presenter.copy_fence_handle() as usize).unwrap() });
        let layout = KernelLayout { pitch: width, log2_block_width: 0, log2_block_height: 0, log2_block_bytes: 2, xor: 0, masks: [0; 16], mode: 0 };
        let mut main = device.allocate_kind((width * height * 4) as usize, MemoryKind::Device).unwrap();
        let mut overlay = device.allocate_kind((width * height * 4) as usize, MemoryKind::Device).unwrap();
        let descriptor = |address| DisplayBufferDesc { address, width, height, pixel_format: 0x8000_0000_0000_0000, tiling_mode: 1, pitch_in_pixel: width, dcc_address: 0, dcc_clear_color: 0, alignment: 4096 };
        let (main_desc, plane_desc) = (descriptor(main.address()), descriptor(overlay.address()));
        for number in 1..=3 {
            window.pump();
            assert!(presenter.wait_for_frame(std::time::Duration::from_secs(2)).unwrap());
            // CPU writes are test input fixtures, never a presentation conversion.
            let mut pixels = Vec::new();
            for y in 0..height { for x in 0..width { pixels.extend([number as u8 * 20 + (x % 16) as u8, 30 + (y % 16) as u8, 80, 255]); } }
            main.write(0, &pixels).unwrap();
            overlay.write(0, &[20u8, 10, 5, 128].repeat((width * height) as usize)).unwrap();
            display.capture_overlay(&mut device, Some(DisplaySource { desc: plane_desc, address: overlay.address(), layout, bytes: overlay.size() as u64, in_guest: false }), &[&overlay]).unwrap();
            // Simulate immediate guest reuse after VideoOut's overlay acknowledgement.
            overlay.fill(0).unwrap();
            display.present(&mut device, target.take(), NativeFrame { generation, number, buffer: Some(main_desc), plane: Some(plane_desc) },
                Some(DisplaySource { desc: main_desc, address: main.address(), layout, bytes: main.size() as u64, in_guest: false }), &[&main]).unwrap();
            presenter.present(number).unwrap();
            let output = presenter.read_back(number).unwrap();
            let footprint = presenter.footprint();
            for y in 0..height { for x in 0..width {
                let at = footprint.row_offset(y) as usize + x as usize * 4;
                let input = &pixels[(y as usize * width as usize + x as usize) * 4..][..4];
                let expected = [((u32::from(input[0]) * 127 + 127) / 255 + 20) as u8, ((u32::from(input[1]) * 127 + 127) / 255 + 10) as u8, ((80 * 127 + 127) / 255 + 5) as u8, 255];
                assert_eq!(&output[at..at + 4], &expected, "frame {number} pixel ({x},{y})");
            } }
        }
        presenter.wait_copied(presenter.frames()).unwrap();
    }
    let end = std::time::Instant::now() + std::time::Duration::from_secs(4);
    while std::time::Instant::now() < end { window.pump(); std::thread::sleep(std::time::Duration::from_millis(20)); }
}
