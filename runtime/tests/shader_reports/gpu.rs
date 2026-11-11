use super::*;
use agcdrv::backend::wddm::{Buffer, Device, MemoryKind};
use agcdrv::gpu::{
    packets::{event, gcr, PacketStream, SHADER_TYPE_COMPUTE},
    regs as host,
};
use shader::image_reports::{fold, hash, known_offset, report_offset, CHECK, REPORT_CHECK_OFFSET, TABLE_BYTES};

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn words(buffer: &Buffer, offset: u32, count: usize) -> agcdrv::Result<Vec<u32>> {
    let offset = offset as usize;
    Ok(buffer.read()?[offset..offset + count * 4].chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect())
}

/// One wave of the program at `code`, with `user` as its user data.
fn dispatch(device: &mut Device, code: &Buffer, artifact: &ShaderArtifact, wave: u8, user: &[u32], targets: &[&Buffer]) -> agcdrv::Result<()> {
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
    stream.set_run(host::COMPUTE_NUM_THREAD, &[u32::from(wave), 1, 1])?;
    stream.set_run(host::COMPUTE_USER_DATA, user)?;
    stream.raw_packet(agcdrv::amd::defs::pkt3::DISPATCH_DIRECT, SHADER_TYPE_COMPUTE, &[1, 1, 1, 1 | if wave == 32 { 1 << 15 } else { 0 }])?;
    stream.event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
    stream.acquire_mem_global(gcr::GL2_WB | gcr::GL2_INV)?;
    let mut all = vec![code];
    all.extend_from_slice(targets);
    let serial = device.submit_host(stream.words(), &all)?;
    device.wait(serial)
}

#[test]
#[ignore = "requires Mix AMD WDDM and an idle GPU"]
fn a_found_image_is_reported_until_the_runtime_knows_it() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = Device::open()?;
    let target = agcdrv::gpu::target::host_target(device.identity());
    let mut code = device.allocate_executable(65536)?;
    let mut lut = device.allocate_kind(65536, MemoryKind::Data)?;
    lut.write(0, &shader::descriptors::lut())?;
    let mut table = device.allocate_kind(TABLE_BYTES as usize, MemoryKind::Readback)?;
    let mut found = device.allocate_kind(65536, MemoryKind::Data)?;
    let mut texels = device.allocate_kind(65536, MemoryKind::Data)?;
    texels.fill(0)?;
    // Two linear images of 32-bit elements over the same memory, four wide, four and two high:
    // they differ in one dword that is not their address.
    let tsharp = |height: u32| {
        let address = texels.address();
        [(address >> 8) as u32, (address >> 40) as u32 | (22 << 20) | (3 << 30), (height - 1) << 14, 0xfac | (9 << 28), 0, 0, 0, 0]
    };
    let (first, second) = (tsharp(4), tsharp(2));
    found.write(0, &bytes(&first))?;
    found.write(64, &bytes(&second))?;
    let slot = report_offset(first[0]);
    for wave in [32, 64] {
        let artifact = compile(&FOUND, target, wave, lut.address(), Some(table.address()));
        code.write(0, &artifact.code_bytes())?;
        let run = |device: &mut Device, table: &Buffer, offset: u32| {
            let user = [found.address() as u32, (found.address() >> 32) as u32, offset];
            dispatch(device, &code, &artifact, wave, &user, &[&lut, table, &found, &texels])
        };
        // Unknown to the runtime: its slot gets the T# and the check word, and nothing else is stored.
        table.fill(0)?;
        run(&mut device, &table, 0)?;
        assert_eq!(words(&table, slot, 8)?, first, "wave{wave}");
        assert_eq!(words(&table, slot + REPORT_CHECK_OFFSET, 1)?, [fold(&first) ^ CHECK], "wave{wave}");
        assert_eq!(table.read()?.iter().filter(|&&byte| byte != 0).count(), bytes(&first).iter().chain(&(fold(&first) ^ CHECK).to_le_bytes()).filter(|&&byte| byte != 0).count(), "wave{wave}");
        // Known: the program stores nothing.
        table.fill(0)?;
        // In the second place of its bucket: a program looks at both.
        table.write(known_offset(hash(&first)) as usize + 4, &hash(&first).to_le_bytes())?;
        run(&mut device, &table, 0)?;
        assert_eq!(words(&table, slot, 9)?, [0; 9], "wave{wave}");
        // Another image at the known one's address is not the known one.
        run(&mut device, &table, 64)?;
        assert_eq!(words(&table, slot, 8)?, second, "wave{wave}");
        assert_eq!(words(&table, slot + REPORT_CHECK_OFFSET, 1)?, [fold(&second) ^ CHECK], "wave{wave}");
    }
    Ok(())
}
