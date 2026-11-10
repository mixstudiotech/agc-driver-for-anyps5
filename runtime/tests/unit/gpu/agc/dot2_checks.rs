use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa};
#[path = "../../../shader_dot2/kernels.rs"]
mod kernels;

pub(super) fn check(window: &super::TestWindow, device: &mut crate::Device, target: TargetIsa) {
    let mut code = device.allocate_executable(65536).unwrap();
    let mut output = device
        .allocate_kind(4096, crate::backend::MemoryKind::Readback)
        .unwrap();
    let cases = [
        (0, 0, 0),
        (0xffff_ffff, 0xffff_ffff, 0xffff_ffff),
        (0x8000_8000, 0x7fff_8000, 0x7000_0000),
        (0x8000_8000, 0x8000_8000, 0x7fff_ffff),
        (0x8000_8000, 0x7fff_7fff, 0x8000_0000),
        (0x1234_abcd, 0x6789_ffff, 0x5678_9123),
    ];
    let mut checks = 0;
    for wave in [32, 64] {
        for kernel in kernels::KERNELS {
            for lo in 0..4 {
                for hi in 0..4 {
                    let words = kernels::selectors(kernel.words, lo, hi);
                    let interface = ShaderInterface {
                        guest_pgm_rsrc2: Some(12 << 1),
                        ..ShaderInterface::default()
                    };
                    let artifact = Compiler::new(target)
                        .compile_with_interface(
                            &words,
                            CompileOptions::new(target, ShaderStage::Compute, wave),
                            &interface,
                        )
                        .unwrap();
                    for (a, b, accumulator) in cases {
                        for flag in [0, 1] {
                            window.pump();
                            let mut user = [0u32; 12];
                            user[..6].copy_from_slice(&[
                                a,
                                b,
                                accumulator,
                                flag,
                                if flag == 0 { 0 } else { 0xa5a5_a5a5 },
                                if flag == 0 || wave == 32 {
                                    0
                                } else {
                                    0x5a5a_5a5a
                                },
                            ]);
                            let read = super::isa_dispatch::run(
                                device,
                                &mut code,
                                &mut output,
                                &artifact,
                                &user,
                            );
                            let half = |value: u32, select: u8| -> i64 {
                                let bits = (value >> (16 * u32::from(select & 1))) as u16;
                                if kernel.signed {
                                    i64::from(bits as i16)
                                } else {
                                    i64::from(bits)
                                }
                            };
                            let sum = half(a, lo) * half(b, lo >> 1)
                                + half(a, hi) * half(b, hi >> 1)
                                + if kernel.signed {
                                    i64::from(accumulator as i32)
                                } else {
                                    i64::from(accumulator)
                                };
                            let expected = if !kernel.clamp {
                                sum as u32
                            } else if kernel.signed {
                                sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as u32
                            } else {
                                sum.clamp(0, i64::from(u32::MAX)) as u32
                            };
                            assert_eq!(
                                read[0], expected,
                                "signed={} clamp={} alias={} lo={lo} hi={hi} a={a:x} b={b:x} acc={accumulator:x} wave={wave}",
                                kernel.signed, kernel.clamp, kernel.alias
                            );
                            assert_eq!(read[1], flag, "DOT2 changed SCC");
                            assert_eq!(read[2], user[4], "DOT2 changed VCC low");
                            assert_eq!(read[3], user[5], "DOT2 changed VCC high");
                            assert!(
                                read[4..].iter().all(|&v| v == 0xcdcd_cdcd),
                                "inactive lanes must not store"
                            );
                            checks += 4;
                        }
                    }
                }
            }
        }
    }
    println!("integer_dot2 checks={checks}");
}
