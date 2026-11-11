use super::{case_names, kernels};
use agcdrv::backend::wddm::{Device, MemoryKind};
use agcdrv::gpu::{
    packets::{PacketStream, SHADER_TYPE_COMPUTE, event, gcr},
    regs as host,
};
use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage};

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}
fn words(value: &[u8]) -> Vec<u32> {
    value
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}
fn buffer(address: u64, stride: u32, count: u32) -> [u32; 4] {
    [
        address as u32,
        (address >> 32) as u32 | (stride << 16),
        count,
        0xfac | (22 << 12),
    ]
}
#[test]
#[ignore = "requires Mix AMD WDDM and an idle GPU"]
fn bvh8_matches_sdk_reference_vectors() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = [
        &include_bytes!("fixtures/oracle-inputs.bin")[..],
        &include_bytes!("fixtures/manual-inputs.bin")[..],
    ]
    .concat();
    let outputs = [
        &include_bytes!("fixtures/oracle-outputs.bin")[..],
        &include_bytes!("fixtures/manual-outputs.bin")[..],
    ]
    .concat();
    let mut device = Device::open()?;
    println!("adapter {:?}", device.identity());
    let target = agcdrv::gpu::target::host_target(device.identity());
    let mut code = device.allocate_executable(1 << 20)?;
    let mut nodes = device.allocate_kind(65536, MemoryKind::Data)?;
    let mut input = device.allocate_kind(65536, MemoryKind::Data)?;
    let mut output = device.allocate_kind(65536, MemoryKind::Readback)?;
    let (mut checks, mut failures) = (0, 0);
    for (variant, guest) in [
        (0, kernels::BVH8),
        (1, kernels::BVH8_ALIAS),
        (2, kernels::BVH8_BASE_ALIAS),
        (3, kernels::BVH8_STATE),
    ] {
        for wave in [32u8, 64] {
            let output_words = if variant == 3 { 22 } else { 16 };
            let stride = output_words * 4;
            let interface = ShaderInterface {
                guest_pgm_rsrc2: Some(12 << 1),
                ..ShaderInterface::default()
            };
            let artifact = Compiler::new(target).compile_with_interface(
                guest,
                CompileOptions::new(target, ShaderStage::Compute, wave),
                &interface,
            )?;
            code.write(0, &artifact.code_bytes())?;
            println!(
                "wave={wave} code_words={} sgprs={} vgprs={}",
                artifact.program_words, artifact.num_sgprs, artifact.num_vgprs
            );
            for (case, (record, output_record)) in inputs
                .chunks_exact(384)
                .zip(outputs.chunks_exact(64))
                .enumerate()
            {
                let vector = words(record);
                // The SDK CPU model ignores these controls. Cases 100+ supply geometric expectations.
                if case < 100 && vector[6] & 0x7ff0_0000 != 0 {
                    continue;
                }
                if variant == 1 && vector[0] & 7 == 6 {
                    continue;
                }
                nodes.fill(0)?;
                nodes.write(0, &bytes(&vector[16..96]))?;
                let encoded = (nodes.address() >> 3) | ((u64::from(vector[4])) << 54);
                let mut values = vec![encoded as u32, (encoded >> 32) as u32, vector[1], vector[2]];
                values.extend(vector[9..12].iter().copied());
                values.extend(vector[12..15].iter().copied());
                values.extend([vector[0], 0]);
                let inputs: Vec<u32> = (0..wave)
                    .flat_map(|lane| {
                        let mut lane_values = values.clone();
                        if lane & 1 != 0 {
                            lane_values[1] |= 0xc0 << 22;
                        }
                        lane_values
                    })
                    .collect();
                input.write(0, &bytes(&inputs))?;
                output.fill(0xcd)?;
                let mut user = [0u32; 12];
                user[..4].copy_from_slice(&buffer(input.address(), 48, u32::from(wave)));
                user[4..8].copy_from_slice(&vector[5..9]);
                user[8..].copy_from_slice(&buffer(
                    output.address(),
                    stride as u32,
                    u32::from(wave),
                ));
                let mut stream = PacketStream::new_compute();
                stream.acquire_mem_global(
                    gcr::GLK_INV | gcr::GLV_INV | gcr::GL1_INV | gcr::GL2_INV,
                )?;
                for (reg, value) in [
                    (host::COMPUTE_PGM_LO, (code.address() >> 8) as u32),
                    (host::COMPUTE_PGM_HI, (code.address() >> 40) as u32),
                    (host::COMPUTE_PGM_RSRC1, artifact.pgm_rsrc1),
                    (host::COMPUTE_PGM_RSRC2, artifact.pgm_rsrc2),
                    (host::COMPUTE_PGM_RSRC3, 0),
                    (host::COMPUTE_TMPRING_SIZE, 0),
                    (host::COMPUTE_RESOURCE_LIMITS, 0),
                ] {
                    stream.set(reg, value)?;
                }
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
                let serial =
                    device.submit_host(stream.words(), &[&code, &nodes, &input, &output])?;
                device.wait(serial)?;
                let actual: Vec<u32> = output.read()?[..usize::from(wave) * stride]
                    .chunks_exact(4)
                    .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                    .collect();
                let expected = words(output_record);
                let mut bad = 0;
                for (i, a) in actual.iter().enumerate() {
                    checks += 1;
                    let word = i % output_words;
                    let wanted = if word >= 16 {
                        [1, 0x33, 0xa5a5a5a5, 0x5a5a5a5a, 0x12345678, 0x6fedcba9][word - 16]
                    } else if (i / output_words) & 1 != 0 {
                        if word < 10 {
                            if variant == 3 {
                                0xdead0000 + word as u32
                            } else {
                                u32::MAX
                            }
                        } else {
                            values[word - 6]
                        }
                    } else {
                        expected[word]
                    };
                    if *a != wanted {
                        bad += 1;
                    }
                }
                failures += bad;
                if bad != 0 {
                    println!(
                        "variant={variant} wave={wave} case={case} ({}) failures={bad} actual={:08x?} expected={:08x?}",
                        case_names::NAMES
                            .get(case)
                            .copied()
                            .unwrap_or("descriptor geometry"),
                        &actual[..16],
                        expected
                    );
                }
            }
        }
    }
    println!(
        "bvh8_oracle_gpu vectors={} checks={checks} failures={failures}",
        inputs.len() / 384
    );
    assert_eq!(failures, 0);
    Ok(())
}
