use super::*;
use agcdrv::backend::wddm::{Buffer, Device, MemoryKind};
use agcdrv::gpu::{packets::{event, gcr, PacketStream, SHADER_TYPE_COMPUTE}, regs as host};

#[path = "kernels.rs"]
mod kernels;

const ACTIVE: u64 = 0xa5a5_5aa5_ffff_ffff;

fn buffer(address: u64, stride: u32, count: u32) -> [u32; 4] {
    [address as u32, (address >> 32) as u32 | (stride << 16), count, 0xfac | (22 << 12)]
}

fn kernel(max: bool, returning: bool, oracle: bool) -> &'static [u32] {
    match (max, returning, oracle) {
        (false, false, false) => kernels::IMAGE_FMIN_0,
        (false, true, false) => kernels::IMAGE_FMIN_1,
        (true, false, false) => kernels::IMAGE_FMAX_0,
        (true, true, false) => kernels::IMAGE_FMAX_1,
        (false, false, true) => kernels::BUFFER_FMIN_0,
        (false, true, true) => kernels::BUFFER_FMIN_1,
        (true, false, true) => kernels::BUFFER_FMAX_0,
        (true, true, true) => kernels::BUFFER_FMAX_1,
    }
}

fn pairs() -> Vec<(u32, u32)> {
    vec![
        (1.0f32.to_bits(), 2.0f32.to_bits()), (2.0f32.to_bits(), 1.0f32.to_bits()),
        ((-2.0f32).to_bits(), (-1.0f32).to_bits()), ((-1.0f32).to_bits(), (-2.0f32).to_bits()),
        (0, 0x8000_0000), (0x8000_0000, 0), (0x7f80_0000, 0xff80_0000), (0xff80_0000, 0x7f80_0000),
        (0x7fc0_1234, 1.0f32.to_bits()), (1.0f32.to_bits(), 0x7fc0_1234),
        (0x7f80_1234, 1.0f32.to_bits()), (1.0f32.to_bits(), 0x7f80_1234),
        (0x7f80_1234, 0xff80_5678), (0xff80_5678, 0x7f80_1234),
        (0x7fc0_1234, 0xffc0_5678), (0xffc0_5678, 0x7fc0_1234),
        (1, 2), (2, 1), (0x8000_0001, 0x8000_0002), (0x8000_0002, 0x8000_0001),
        (1, 0x8000_0001), (0x8000_0001, 1), (1, 0), (0, 1),
        (0x8000_0001, 0x8000_0000), (0x8000_0000, 0x8000_0001),
        (0x7fc0_1234, 0x7f80_5678), (0x7f80_5678, 0x7fc0_1234),
        (0x0080_0000, 0x007f_ffff), (0x807f_ffff, 0x8080_0000), (0, 0), (0x8000_0000, 0x8000_0000),
    ]
}

fn bytes(words: &[u32]) -> Vec<u8> { words.iter().flat_map(|word| word.to_le_bytes()).collect() }

struct Dispatch {
    code: Buffer,
    memory: Buffer,
    input: Buffer,
    output: Buffer,
}

impl Dispatch {
    fn new(device: &mut Device) -> agcdrv::Result<Self> {
        Ok(Self { code: device.allocate_executable(65536)?, memory: device.allocate_kind(65536, MemoryKind::Device)?, input: device.allocate_kind(65536, MemoryKind::Data)?, output: device.allocate_kind(65536, MemoryKind::Readback)? })
    }

    fn run(&mut self, device: &mut Device, words: &[u32], options: CompileOptions, float_mode: u8, ieee: bool, initial: &[u32], input: &[u32], oracle: bool) -> Result<(Vec<u32>, Vec<u32>), Box<dyn std::error::Error>> {
        let interface = ShaderInterface { guest_pgm_rsrc1: Some((u32::from(float_mode) << 12) | if ieee { 1 << 23 } else { 0 }), guest_pgm_rsrc2: Some(16 << 1), ..ShaderInterface::default() };
        let compiled = Compiler::new(options.target).compile_with_interface(words, options, &interface)?;
        self.code.write(0, &compiled.code_bytes())?;
        self.memory.write(0, &bytes(initial))?;
        self.input.write(0, &bytes(input))?;
        self.output.fill(0xcd)?;
        let address = self.memory.address();
        let mut user = [0u32; 16];
        if oracle { user[..4].copy_from_slice(&buffer(address, 4, input.len() as u32)); }
        else {
            let width = input.len() as u32 - 1;
            user[..8].copy_from_slice(&[(address >> 8) as u32, (address >> 40) as u32 | (22 << 20) | ((width & 3) << 30), width >> 2, 0xfac | (8 << 28), 0, 0, 0, 0]);
        }
        user[8..12].copy_from_slice(&buffer(self.input.address(), 4, input.len() as u32));
        user[12..16].copy_from_slice(&buffer(self.output.address(), 16, input.len() as u32));
        let mut stream = PacketStream::new_compute();
        stream.acquire_mem_global(gcr::GLK_INV | gcr::GLV_INV | gcr::GL1_INV | gcr::GL2_INV)?;
        stream.set(host::COMPUTE_PGM_LO, (self.code.address() >> 8) as u32)?;
        stream.set(host::COMPUTE_PGM_HI, (self.code.address() >> 40) as u32)?;
        stream.set(host::COMPUTE_PGM_RSRC1, compiled.pgm_rsrc1)?;
        stream.set(host::COMPUTE_PGM_RSRC2, compiled.pgm_rsrc2)?;
        stream.set(host::COMPUTE_PGM_RSRC3, 0)?;
        stream.set(host::COMPUTE_TMPRING_SIZE, 0)?;
        stream.set(host::COMPUTE_RESOURCE_LIMITS, 0)?;
        stream.set_run(host::COMPUTE_START, &[0, 0, 0])?;
        stream.set_run(host::COMPUTE_NUM_THREAD, &[u32::from(options.wave_size), 1, 1])?;
        stream.set_run(host::COMPUTE_USER_DATA, &user)?;
        stream.raw_packet(agcdrv::amd::defs::pkt3::DISPATCH_DIRECT, SHADER_TYPE_COMPUTE, &[1, 1, 1, 1 | if options.wave_size == 32 { 1 << 15 } else { 0 }])?;
        stream.event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
        stream.acquire_mem_global(gcr::GL2_WB | gcr::GL2_INV)?;
        let serial = device.submit_host(stream.words(), &[&self.code, &self.memory, &self.input, &self.output])?;
        device.wait(serial)?;
        let read = |buffer: &Buffer, count: usize| -> agcdrv::Result<Vec<u32>> { Ok(buffer.read()?[..count * 4].chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect()) };
        Ok((read(&self.memory, input.len())?, read(&self.output, input.len() * 4)?))
    }
}

#[test]
#[ignore = "requires mix AMD WDDM and a clear GPU gate"]
fn image_float_cas_matches_native_buffer_atomics() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = Device::open()?;
    let target = agcdrv::gpu::target::host_target(device.identity());
    assert!(target.is_gfx11());
    let mut dispatch = Dispatch::new(&mut device)?;
    let pairs = pairs();
    let mut checks = 0;
    for wave in [32, 64] {
        let initial: Vec<_> = (0..wave as usize).map(|i| pairs[i % pairs.len()].0).collect();
        let input: Vec<_> = (0..wave as usize).map(|i| pairs[i % pairs.len()].1).collect();
        for max in [false, true] {
            for returning in [false, true] {
                for denorm in 0..4 {
                    for ieee in [false, true] {
                        let options = CompileOptions::new(target, ShaderStage::Compute, wave);
                        let actual = dispatch.run(&mut device, kernel(max, returning, false), options, 0xc0 | denorm << 4, ieee, &initial, &input, false)?;
                        let reference = dispatch.run(&mut device, kernel(max, returning, true), options, 0xc0 | denorm << 4, ieee, &initial, &input, true)?;
                        if actual != reference {
                            eprintln!("actual memory {:08x?}; native memory {:08x?}", actual.0, reference.0);
                            eprintln!("actual returns {:08x?}; native returns {:08x?}", actual.1, reference.1);
                        }
                        for lane in 0..wave as usize {
                            let active = ACTIVE & (1u64 << lane) != 0;
                            assert_eq!(actual.0[lane], reference.0[lane], "memory lane={lane} max={max} return={returning} denorm={denorm} ieee={ieee} wave={wave} initial={:08x} input={:08x}", initial[lane], input[lane]);
                            assert_eq!(&actual.1[lane*4..lane*4+4], &reference.1[lane*4..lane*4+4], "return/flags lane={lane} max={max} return={returning} denorm={denorm} ieee={ieee} wave={wave}");
                            if active { checks += 5; }
                        }
                    }
                }
            }
        }
    }
    println!("image_float_cas checks={checks}");
    Ok(())
}
