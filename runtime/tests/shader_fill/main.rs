use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa, ir::BufferFill};

/// A title's memset on the GPU: every thread stores s[4:7] through the V# in s[0:3], at
/// `(s8 << 6) + v0`.
const FILL: [u32; 9] = [0xd746_0004, 0x0401_0c08, 0x7e00_0204, 0x7e02_0205, 0x7e04_0206, 0x7e06_0207, 0xe01c_2000, 0x8000_0004, 0xbf81_0000];
/// The same title's masked update: it loads each record before it stores it.
const MASKED: [u32; 16] = [
    0xd746_0004, 0x0401_0c0c, 0xe00c_2000, 0x8000_0004, 0xbf8c_3f70, 0xd771_0000, 0x0012_0008, 0xd771_0001, 0x0016_0209, 0xd771_0002, 0x001a_040a,
    0xd771_0003, 0x001e_060b, 0xe01c_2000, 0x8000_0004, 0xbf81_0000,
];

fn fill(words: &[u32], rsrc2: u32) -> Option<BufferFill> {
    let interface = ShaderInterface { guest_pgm_rsrc2: Some(rsrc2), ..ShaderInterface::default() };
    Compiler::new(TargetIsa::Gfx115).compile_with_interface(words, CompileOptions::new(TargetIsa::Gfx115, ShaderStage::Compute, 64), &interface).unwrap().buffer_fill
}

#[test]
fn a_program_that_only_fills_is_recognised() {
    // Eight user SGPRs with the group's index along x behind them (COMPUTE_PGM_RSRC2 0x90).
    assert_eq!(fill(&FILL, 0x90), Some(BufferFill { buffer: 0, value: 4 }));
    // Without the group's index the record's place is not the thread's index over the dispatch.
    assert_eq!(fill(&FILL, 0x10), None);
    // Twelve user SGPRs; what it stores depends on what the buffer held.
    assert_eq!(fill(&MASKED, 0x98), None);
}

#[test]
#[cfg(all(target_os = "windows", feature = "wddm"))]
#[ignore = "requires Mix AMD WDDM and an idle GPU"]
fn comp_to_single_expands_each_guest_block_on_the_gpu() -> Result<(), Box<dyn std::error::Error>> {
    use agcdrv::{backend::wddm::{Device, MemoryKind}, gpu::{kernels::element_copy, packets::{PacketStream, event, gcr, SHADER_TYPE_COMPUTE}, regs as host}};
    use agcdrv::amd::surface::{retile::copy_dispatches, tiling::{host_swizzle_mode, SurfaceDesc, SurfaceLayout, SurfaceUse, SW_64KB_R_X}};
    let mut device = Device::open()?;
    let artifact = element_copy::compile(agcdrv::gpu::target::host_target(device.identity()))?;
    let mut code = device.allocate_executable(65536)?;
    code.write(0, &artifact.code_bytes())?;
    for element in [1, 2, 4, 8, 16] {
        let desc = SurfaceDesc { swizzle_mode: SW_64KB_R_X, width: 64, height: 64, depth: 1, layers: 1, mip_levels: 1, bytes_per_element: element, ..Default::default() };
        let guest = SurfaceLayout::new(&desc)?;
        let native = SurfaceLayout::host(&SurfaceDesc { swizzle_mode: host_swizzle_mode(desc.swizzle_mode, SurfaceUse::Thin2d), ..desc })?;
        let mut source = device.allocate_kind(guest.size_bytes() as usize, MemoryKind::Data)?;
        let mut destination = device.allocate_kind(native.size_bytes() as usize, MemoryKind::Readback)?;
        let mut data = vec![0xabu8; guest.size_bytes() as usize];
        for (index, block) in data.chunks_exact_mut(256).enumerate() {
            for (byte, value) in block[..16].iter_mut().enumerate() { *value = (index.wrapping_mul(37) + byte) as u8; }
        }
        source.write(0, &data)?;
        destination.fill(0)?;
        let mut params = device.allocate_kind(512, MemoryKind::Data)?;
        let mut dispatch = copy_dispatches(&guest, source.address(), &native, destination.address(), 0..1)?.remove(0);
        dispatch.params[67] = 2;
        let bytes: Vec<u8> = dispatch.params.iter().flat_map(|word| word.to_le_bytes()).collect();
        params.write(0, &bytes)?;
        let mut stream = PacketStream::new_compute();
        stream.acquire_mem_global(gcr::GLK_INV | gcr::GLV_INV | gcr::GL1_INV | gcr::GL2_INV)?;
        stream.set(host::COMPUTE_PGM_LO, (code.address() >> 8) as u32)?;
        stream.set(host::COMPUTE_PGM_HI, (code.address() >> 40) as u32)?;
        stream.set(host::COMPUTE_PGM_RSRC1, artifact.pgm_rsrc1)?;
        stream.set(host::COMPUTE_PGM_RSRC2, artifact.pgm_rsrc2)?;
        stream.set(host::COMPUTE_PGM_RSRC3, 0)?;
        stream.set(host::COMPUTE_TMPRING_SIZE, 0)?;
        stream.set(host::COMPUTE_RESOURCE_LIMITS, 0)?;
        stream.set_run(host::COMPUTE_START, &[0, 0, 0])?;
        stream.set_run(host::COMPUTE_NUM_THREAD, &[64, 1, 1])?;
        stream.set_run(host::COMPUTE_USER_DATA, &[params.address() as u32, (params.address() >> 32) as u32])?;
        let [x, y, z] = dispatch.groups;
        stream.raw_packet(agcdrv::amd::defs::pkt3::DISPATCH_DIRECT, SHADER_TYPE_COMPUTE, &[x, y, z, 1])?;
        stream.event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
        stream.acquire_mem_global(gcr::GL2_WB | gcr::GL2_INV)?;
        let serial = device.submit_host(stream.words(), &[&code, &source, &destination, &params])?;
        device.wait(serial)?;
        let output = destination.read()?;
        for y in 0..64 { for x in 0..64 {
            let from = (guest.element_offset(x, y, 0, 0)? & !255) as usize;
            let to = native.element_offset(x, y, 0, 0)? as usize;
            assert_eq!(&output[to..to + element as usize], &data[from..from + element as usize], "{element}B pixel ({x},{y})");
        }}
    }
    Ok(())
}
