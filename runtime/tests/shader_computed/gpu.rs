use super::kernels;
use agcdrv::backend::wddm::{Buffer, Device, MemoryKind};
use agcdrv::gpu::{
    packets::{PacketStream, SHADER_TYPE_COMPUTE, event, gcr},
    regs as host,
};
use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage};

fn dispatch(
    device: &mut Device,
    code: &Buffer,
    output: &mut Buffer,
    wave: u8,
    target_address: u64,
    selector: u32,
    rsrc1: u32,
    rsrc2: u32,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    output.fill(0xcd)?;
    let user = [
        target_address as u32,
        (target_address >> 32) as u32,
        selector,
        0,
        output.address() as u32,
        (output.address() >> 32) as u32,
    ];
    let mut stream = PacketStream::new_compute();
    stream.acquire_mem_global(gcr::GLK_INV | gcr::GLV_INV | gcr::GL1_INV | gcr::GL2_INV)?;
    stream.set(host::COMPUTE_PGM_LO, (code.address() >> 8) as u32)?;
    stream.set(host::COMPUTE_PGM_HI, (code.address() >> 40) as u32)?;
    stream.set(host::COMPUTE_PGM_RSRC1, rsrc1)?;
    stream.set(host::COMPUTE_PGM_RSRC2, rsrc2)?;
    stream.set(host::COMPUTE_PGM_RSRC3, 0)?;
    stream.set(host::COMPUTE_TMPRING_SIZE, 0)?;
    stream.set(host::COMPUTE_RESOURCE_LIMITS, 0)?;
    stream.set_run(host::COMPUTE_START, &[0, 0, 0])?;
    stream.set_run(host::COMPUTE_NUM_THREAD, &[u32::from(wave), 1, 1])?;
    stream.set_run(host::COMPUTE_USER_DATA, &user)?;
    stream.raw_packet(
        agcdrv::amd::defs::pkt3::DISPATCH_DIRECT,
        SHADER_TYPE_COMPUTE,
        &[1, 1, 1, 1 | if wave == 32 { 1 << 15 } else { 0 }],
    )?;
    stream.event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
    stream.acquire_mem_global(gcr::GL2_WB | gcr::GL2_INV)?;
    let serial = device.submit_host(stream.words(), &[code, output])?;
    device.wait(serial)?;
    Ok(output.read()?[..usize::from(wave) * 4]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect())
}

#[test]
#[ignore = "requires Mix AMD WDDM and an idle GPU"]
fn computed_targets_keep_scc_and_reject_high_bit_aliases() -> Result<(), Box<dyn std::error::Error>>
{
    let guest = kernels::DISPATCH;
    let mut device = Device::open()?;
    println!("adapter {:?}", device.identity());
    let target = agcdrv::gpu::target::host_target(device.identity());
    let mut code = device.allocate_executable(65536)?;
    let mut output = device.allocate_kind(65536, MemoryKind::Readback)?;
    let mut checks = 0;
    let mut failures = 0;
    for wave in [32, 64] {
        for base in [0x1_0000_0000u64, 0xffff_ffe0] {
            let interface = ShaderInterface {
                guest_code_address: Some(base),
                guest_pgm_rsrc2: Some(6 << 1),
                ..ShaderInterface::default()
            };
            let artifact = Compiler::new(target).compile_with_interface(
                guest,
                CompileOptions::new(target, ShaderStage::Compute, wave),
                &interface,
            )?;
            code.write(0, &artifact.code_bytes())?;
            for (kind, address, valid) in [
                ("valid", base + 0x1c, true),
                ("bit63", (base + 0x1c) | (1 << 63), false),
                ("unaligned", base + 0x1d, false),
                ("before", base - 4, false),
                ("after", base + guest.len() as u64 * 4, false),
            ] {
                for selector in [0, 1] {
                    let actual = dispatch(
                        &mut device,
                        &code,
                        &mut output,
                        wave,
                        address.wrapping_sub(base + 8),
                        selector,
                        artifact.pgm_rsrc1,
                        artifact.pgm_rsrc2,
                    )?;
                    let expected = if valid {
                        if selector == 0 { 17 } else { 34 }
                    } else {
                        0xcdcd_cdcd
                    };
                    let bad = actual.iter().filter(|&&value| value != expected).count();
                    checks += actual.len();
                    failures += bad;
                    println!(
                        "wave={wave} base={base:#x} kind={kind} selector={selector} expected={expected:08x} first={:08x} failures={bad}",
                        actual[0]
                    );
                }
            }
        }
    }
    println!("dispatch_gpu checks={checks} failures={failures}");
    assert_eq!(failures, 0);
    Ok(())
}
