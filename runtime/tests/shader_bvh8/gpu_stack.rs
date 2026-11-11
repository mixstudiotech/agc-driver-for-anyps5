use super::kernels;
use agcdrv::backend::wddm::{Device, MemoryKind};
use agcdrv::gpu::{
    packets::{PacketStream, SHADER_TYPE_COMPUTE, event, gcr},
    regs as host,
};
use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa};
use shader::{
    decode::Decoder,
    encode::encode_instruction,
    ir::{Fields, MemKind},
};

fn field(ctrl: u32, shift: u32, width: u32) -> u32 {
    (ctrl >> shift) & ((1 << width) - 1)
}
fn put(ctrl: &mut u32, shift: u32, width: u32, value: u32) {
    let mask = ((1 << width) - 1) << shift;
    *ctrl = (*ctrl & !mask) | ((value << shift) & mask);
}
fn reference(mut ctrl: u32, last: u32, nodes: &[u32], ring: &mut [u32], size: u32) -> (u32, u32) {
    let walkback = last >> 3 != 0x1fff_ffff;
    let mut valid = !walkback || nodes.contains(&last);
    let (mut skip_lo, mut skip_hi) = (false, false);
    for i in (0..8).rev() {
        let node = nodes[i];
        let sentinel = node >> 3 == 0x1fff_ffff;
        if i >= 4 {
            if sentinel && node & 2 == 0 && valid && !skip_hi {
                skip_lo = true;
            }
            if sentinel && node & 4 == 0 && valid && !skip_hi {
                skip_hi = true;
            }
        } else if sentinel && node & 2 == 0 && valid && !skip_lo {
            skip_lo = true;
        }
        if walkback && node == last {
            valid = false;
        }
        if !valid || sentinel || if i >= 4 { skip_hi } else { skip_lo } {
            continue;
        }
        let count = field(ctrl, 0, 5);
        let tlas = field(ctrl, 5, 5);
        let at = field(ctrl, 10, 5);
        if count == size {
            ctrl |= 1 << 30;
        }
        ring[at as usize] = node;
        put(&mut ctrl, 10, 5, if at == size - 1 { 0 } else { at + 1 });
        if tlas == size {
            ctrl &= !(1 << 29);
        }
        put(&mut ctrl, 5, 5, (tlas + 1).min(size));
        put(&mut ctrl, 0, 5, (count + 1).min(size));
    }
    let count = field(ctrl, 0, 5);
    let tlas = field(ctrl, 5, 5);
    let back = tlas == 0 && ctrl & (1 << 29) != 0;
    put(&mut ctrl, 31, 1, u32::from(back));
    if back {
        ctrl &= !(1 << 29);
    }
    if count == 0 {
        ctrl &= !((1 << 31) | (1 << 29));
        return (
            ctrl,
            if ctrl & (1 << 30) != 0 {
                u32::MAX
            } else {
                u32::MAX - 1
            },
        );
    }
    put(&mut ctrl, 0, 5, count - 1);
    put(&mut ctrl, 5, 5, tlas.wrapping_sub(1));
    let at = field(ctrl, 10, 5);
    let at = if at == 0 { size - 1 } else { at - 1 };
    put(&mut ctrl, 10, 5, at);
    let result = ring[at as usize];
    if result & 7 == 6 {
        if ctrl & (1 << 29) != 0 {
            let tlas = field(ctrl, 5, 5);
            put(&mut ctrl, 0, 5, tlas);
            ctrl |= 1 << 30;
        }
        ctrl |= 1 << 29;
        put(&mut ctrl, 5, 5, 0);
    }
    (ctrl, result)
}
fn buffer(address: u64, stride: u32, count: u32) -> [u32; 4] {
    [
        address as u32,
        (address >> 32) as u32 | (stride << 16),
        count,
        0xfac | (22 << 12),
    ]
}
fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
fn rng(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}
#[test]
#[ignore = "requires Mix AMD WDDM and an idle GPU"]
fn stack8_matches_reference_and_preserves_wave_state() -> Result<(), Box<dyn std::error::Error>> {
    let control = false;
    let kernels: &[(u32, &[u32])] = &[
        (8, kernels::STACK8),
        (13, kernels::STACK13),
        (14, kernels::STACK14),
        (15, kernels::STACK15),
        (16, kernels::STACK16),
        (18, kernels::STACK18),
        (20, kernels::STACK20),
        (30, kernels::STACK30),
        (31, kernels::STACK31),
    ];
    let mut device = Device::open()?;
    println!("adapter {:?}", device.identity());
    let target = agcdrv::gpu::target::host_target(device.identity());
    let mut code = device.allocate_executable(1 << 20)?;
    let mut input = device.allocate_kind(65536, MemoryKind::Data)?;
    let mut output = device.allocate_kind(65536, MemoryKind::Readback)?;
    let (mut checks, mut failures) = (0, 0);
    let mut seed = 0x29473a15;
    for wave in [32u8, 64] {
        for &(size, source) in kernels {
            for alias in [false, true] {
                let mut guest = source.to_vec();
                if alias {
                    let mut ir = Decoder::new().decode(&guest)?;
                    for instruction in &mut ir.instructions {
                        match &mut instruction.fields {
                            Fields::Ds(d) if instruction.opcode == 0xe1 => d.vdst = 12,
                            Fields::Mubuf(b) if b.kind == MemKind::Store && b.offset == 4 => {
                                b.vdata = 12
                            }
                            _ => {}
                        }
                    }
                    guest = ir
                        .instructions
                        .iter()
                        .map(|i| encode_instruction(i, TargetIsa::Gfx1013).unwrap())
                        .flatten()
                        .collect();
                }
                if control {
                    let at = guest
                        .iter()
                        .position(|&word| word == (0xdb84_0000 | size))
                        .unwrap();
                    guest[at..at + 2].fill(0xbf80_0000);
                }
                let interface = ShaderInterface {
                    guest_pgm_rsrc2: Some((12 << 1) | (16 << 15)),
                    ..ShaderInterface::default()
                };
                let artifact = Compiler::new(target).compile_with_interface(
                    &guest,
                    CompileOptions::new(target, ShaderStage::Compute, wave),
                    &interface,
                )?;
                code.write(0, &artifact.code_bytes())?;
                for iteration in 0..20 {
                    let partial = iteration & 1 != 0;
                    let zero = iteration >= 16;
                    let selector = (iteration >> 1) & 1;
                    let mut inputs = vec![0u32; usize::from(wave) * 64];
                    let mut expected = vec![0xcdcdcdcdu32; usize::from(wave) * 64];
                    for lane in 0..usize::from(wave) {
                        let at = lane * 64;
                        let mut ctrl = 0;
                        put(
                            &mut ctrl,
                            15,
                            14,
                            ((lane / 32) * 32 * size as usize + lane % 32) as u32,
                        );
                        if iteration >= 4 {
                            put(&mut ctrl, 0, 5, rng(&mut seed) % (size + 1));
                            put(&mut ctrl, 5, 5, rng(&mut seed) % (size + 1));
                            put(&mut ctrl, 10, 5, rng(&mut seed) % size);
                            ctrl |= rng(&mut seed) & 0xe000_0000;
                        }
                        let nodes: [u32; 8] = std::array::from_fn(|i| {
                            match (iteration + lane as u32 + i as u32) % 13 {
                                0..=7 => u32::MAX - ((iteration + lane as u32 + i as u32) % 8),
                                _ => (rng(&mut seed) % 1024) * 8 + (rng(&mut seed) % 7),
                            }
                        });
                        let last = match (iteration + lane as u32) % 3 {
                            0 => u32::MAX,
                            1 => nodes[lane % 8],
                            _ => 0x7fff_fff0,
                        };
                        let mut ring: [u32; 31] = std::array::from_fn(|_| rng(&mut seed) % 8192);
                        inputs[at] = ctrl;
                        inputs[at + 1] = last;
                        inputs[at + 2..at + 10].copy_from_slice(&nodes);
                        inputs[at + 10..at + 41].copy_from_slice(&ring);
                        let active = !control && !zero && (!partial || lane % 2 == 0);
                        let (updated, result) = if active {
                            reference(ctrl, last, &nodes, &mut ring, size)
                        } else {
                            (ctrl, if alias { nodes[0] } else { 0xdeadbeef })
                        };
                        expected[at..at + 5]
                            .copy_from_slice(&[updated, result, selector, 0xa5a5a5a5, 0x5a5a5a5a]);
                        expected[at + 5..at + 5 + size as usize]
                            .copy_from_slice(&ring[..size as usize]);
                        expected[at + 5 + size as usize] = 0x6fedcba9;
                    }
                    input.write(0, &bytes(&inputs))?;
                    output.fill(0xcd)?;
                    let mut user = [0u32; 12];
                    user[..4].copy_from_slice(&buffer(input.address(), 256, u32::from(wave)));
                    let exec = if zero {
                        0
                    } else if partial {
                        0x55555555
                    } else {
                        u32::MAX
                    };
                    user[4..8].copy_from_slice(&[exec, exec, selector, 0]);
                    user[8..].copy_from_slice(&buffer(output.address(), 256, u32::from(wave)));
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
                    let serial = device.submit_host(stream.words(), &[&code, &input, &output])?;
                    device.wait(serial)?;
                    let actual: Vec<u32> = output.read()?[..usize::from(wave) * 256]
                        .chunks_exact(4)
                        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                        .collect();
                    let mut bad = 0;
                    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                        checks += 1;
                        if a != e {
                            if failures + bad < 16 {
                                println!(
                                    "wave={wave} size={size} iter={iteration} lane={} word={} actual={a:08x} expected={e:08x}",
                                    i / 64,
                                    i % 64
                                );
                            }
                            bad += 1;
                        }
                    }
                    failures += bad;
                }
                println!(
                    "wave={wave} size={size} alias={alias} words={} running_failures={failures}",
                    artifact.program_words
                );
            }
        }
    }
    println!("stack_gpu checks={checks} failures={failures}");
    assert_eq!(failures, 0);
    Ok(())
}
