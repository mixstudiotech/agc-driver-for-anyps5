//! Native GPU display conversion. No guest pixels are mapped or read by the CPU.

use super::surface;
use crate::backend::wddm::{OpenedFence, QueueFence, QueueKind};
use crate::gpu::packets::PacketStream;
use crate::present::{Canvas, DisplayBufferDesc, KernelLayout};
use crate::present::d3d12::{Footprint, FrameSchedule};
use crate::present::handoff::{NativeFrame, NativeTarget};
use crate::present::kernels::display_convert::{self, CanvasTarget, ConvertParams, PlaneDraw};
use crate::{Buffer, Device, Error, Result};
use crate::backend::MemoryKind;
use shader::{ShaderArtifact, TargetIsa};
use std::rc::Rc;

pub(super) struct DisplaySource {
    pub desc: DisplayBufferDesc,
    pub address: u64,
    pub layout: KernelLayout,
    /// Bytes the source spans from `address`.
    pub bytes: u64,
    /// The source is the guest memory itself, not a host render surface.
    pub in_guest: bool,
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/agc/present.rs"]
mod tests;
#[cfg(test)]
#[path = "../../../tests/unit/gpu/agc/shader_isa.rs"]
mod shader_isa_tests;

struct Interchange {
    generation: u64,
    canvas: Canvas,
    footprint: Footprint,
    slots: Vec<Rc<Buffer>>,
    ready: OpenedFence,
    copied: OpenedFence,
    frame: u64,
}

struct Overlay {
    desc: DisplayBufferDesc,
    footprint: Footprint,
    pixels: Rc<Buffer>,
}

pub(super) struct NativeDisplay {
    artifact: ShaderArtifact,
    code: Buffer,
    params: Buffer,
    target: Option<Interchange>,
    overlay: Option<Overlay>,
}

impl NativeDisplay {
    pub fn new(device: &mut Device, isa: TargetIsa) -> Result<Self> {
        let artifact = display_convert::host_artifact(isa, 32)?;
        let mut code = device.allocate_executable(artifact.code.len() * 4)?;
        let bytes: Vec<u8> = artifact.code.iter().flat_map(|word| word.to_le_bytes()).collect();
        code.write(0, &bytes)?;
        let params = device.allocate_kind(512, MemoryKind::Commands)?;
        Ok(Self { artifact, code, params, target: None, overlay: None })
    }

    fn configure(&mut self, device: &mut Device, target: NativeTarget) -> Result<()> {
        if let Some(previous) = &self.target {
            if previous.copied.current().unwrap_or(0) < previous.frame {
                return Err(Error::invalid("present: the previous generation still owns an interchange slot"));
            }
        }
        FrameSchedule::new(target.slots.len())?;
        let slots = target.slots.iter().map(|handle| device.open_shared_buffer_sized(handle.raw(), MemoryKind::Device, target.footprint.required_bytes()).map(Rc::new)).collect::<Result<Vec<_>>>()?;
        let ready = device.open_shared_fence(target.ready.raw())?;
        let copied = device.open_shared_fence(target.copied.raw())?;
        self.target = Some(Interchange { generation: target.generation, canvas: target.canvas, footprint: target.footprint, slots, ready, copied, frame: 0 });
        Ok(())
    }

    fn dispatch(&mut self, device: &mut Device, parameters: &[ConvertParams], targets: &[&Buffer]) -> Result<u64> {
        for (index, block) in parameters.iter().enumerate() {
            self.params.write(index * 256, &block.bytes())?;
        }
        let passes: Vec<_> = parameters.iter().enumerate().map(|(index, params)| (self.params.address() + index as u64 * 256, params.groups())).collect();
        // This producer barrier includes CB/DB render-target writeback, not just read-cache invalidation.
        let mut before = PacketStream::new();
        surface::wait_idle(&mut before, false)?;
        let mut words = before.into_words();
        words.extend(crate::present::native::conversion_passes(&self.artifact, self.code.address(), &passes)?);
        let mut resident = targets.to_vec();
        resident.extend([&self.code, &self.params]);
        // The queue holds a bounded number of submissions, and the title's segments may have
        // filled it (the engine holds its own back then). This one is waited for by its caller
        // and runs after everything queued before it, so waiting for the oldest of those first
        // costs nothing; what is in flight depends only on work already queued.
        while !device.can_queue_host_on(QueueKind::Graphics) {
            let oldest = device.completed() + 1;
            device.wait(oldest)?;
        }
        device.submit_host(&words, &resident)
    }

    /// VideoOut calls this before completing the overlay's reuse ticket. The copy is immutable
    /// until the same presenter thread captures the next overlay, after the previous main read.
    pub fn capture_overlay(&mut self, device: &mut Device, source: Option<DisplaySource>, targets: &[&Buffer]) -> Result<()> {
        let Some(source) = source else { self.overlay = None; return Ok(()) };
        let canvas = Canvas::exact(source.desc.width, source.desc.height)?;
        let footprint = Footprint::bgra8(canvas.width, canvas.height)?;
        let pixels = match self.overlay.take().filter(|previous| previous.footprint == footprint) {
            Some(previous) => previous.pixels,
            None => Rc::new(device.allocate_kind(footprint.required_bytes() as usize, MemoryKind::Device)?),
        };
        let params = ConvertParams::new(&source.layout, source.address, pixels.address(), footprint.row_pitch, &canvas, None)?;
        let mut resident = targets.to_vec();
        resident.push(&pixels);
        let serial = self.dispatch(device, &[params], &resident)?;
        device.wait(serial)?;
        self.overlay = Some(Overlay { desc: source.desc, footprint, pixels });
        Ok(())
    }

    pub fn present(&mut self, device: &mut Device, target: Option<NativeTarget>, frame: NativeFrame, main: Option<DisplaySource>, sources: &[&Buffer]) -> Result<()> {
        if let Some(target) = target { self.configure(device, target)?; }
        let current = self.target.as_ref().ok_or_else(|| Error::invalid("present: no shared target"))?;
        if current.generation != frame.generation || frame.number != current.frame + 1 {
            return Err(Error::invalid("present: mismatched generation or frame number"));
        }
        let schedule = FrameSchedule::new(current.slots.len())?;
        if let Some(copied) = schedule.reuse_wait(frame.number) { device.queue_wait(&current.copied, copied)?; }
        let output = current.slots[schedule.slot(frame.number)].clone();
        let (mut canvas, footprint) = (current.canvas, current.footprint);
        if let Some(source) = &main {
            canvas = crate::present::Canvas::fit(source.desc.width, source.desc.height, canvas.width, canvas.height)?;
        }
        let linear = KernelLayout { pitch: canvas.width, log2_block_width: 0, log2_block_height: 0, log2_block_bytes: 2, xor: 0, masks: [0; 16], mode: 0 };
        let mut parameters = vec![match &main {
            Some(source) => ConvertParams::new(&source.layout, source.address, output.address(), footprint.row_pitch, &canvas, None)?,
            None => ConvertParams::new(&linear, 0, output.address(), footprint.row_pitch, &canvas, Some(display_convert::BAR_PIXEL))?,
        }];
        let mut resident = sources.to_vec();
        resident.push(&output);
        let overlay_pixels;
        if let Some(plane) = frame.plane {
            let overlay = self.overlay.as_ref().filter(|overlay| overlay.desc == plane).ok_or_else(|| Error::invalid("present: the selected overlay has no captured GPU image"))?;
            let layout = KernelLayout { pitch: overlay.footprint.row_pitch as u32 / 4, ..linear };
            let target = CanvasTarget { destination: output.address(), row_pitch: footprint.row_pitch, canvas };
            let draw = PlaneDraw { rect: canvas.plane_rect(plane.width, plane.height)?, width: plane.width, height: plane.height, blend: true };
            parameters.push(ConvertParams::plane(&layout, overlay.pixels.address(), &target, &draw, None)?);
            overlay_pixels = overlay.pixels.clone();
            resident.push(&overlay_pixels);
        }
        let serial = self.dispatch(device, &parameters, &resident)?;
        let current = self.target.as_mut().expect("configured above");
        device.queue_signal(&current.ready, frame.number)?;
        // Only this submission: queued guest work still advances normally between requests.
        // The source reuse callback must not run before all native source reads have completed.
        device.wait(serial)?;
        current.frame = frame.number;
        Ok(())
    }
}
