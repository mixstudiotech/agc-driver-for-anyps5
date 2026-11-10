use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa};

#[path = "../../../shader_scalar/kernels.rs"]
mod kernels;

pub(super) fn check(window: &super::TestWindow, device: &mut crate::Device, target: TargetIsa) {
    let mut code = device.allocate_executable(65536).unwrap();
    let mut output = device
        .allocate_kind(4096, crate::backend::MemoryKind::Readback)
        .unwrap();
    let values: [u64; 10] = [
        0,
        1,
        2,
        3,
        0xffff_ffff,
        0xffff_ffff_0000_0000,
        u64::MAX,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0xffff_ffff_ffff_ffef,
    ];
    let mut checks = 0;
    for wave in [32, 64] {
        for (wide, words) in [
            (false, kernels::FF0_32),
            (false, kernels::FF0_32_HIGH),
            (true, kernels::FF0_64),
            (true, kernels::FF0_64_HIGH),
        ] {
            let interface = ShaderInterface {
                guest_pgm_rsrc2: Some(12 << 1),
                ..ShaderInterface::default()
            };
            let artifact = Compiler::new(target)
                .compile_with_interface(
                    words,
                    CompileOptions::new(target, ShaderStage::Compute, wave),
                    &interface,
                )
                .unwrap();
            for value in values {
                for flag in [0, 1] {
                    window.pump();
                    let mut user = [0u32; 12];
                    user[0] = value as u32;
                    user[1] = (value >> 32) as u32;
                    user[2] = flag;
                    let read =
                        super::isa_dispatch::run(device, &mut code, &mut output, &artifact, &user);
                    let result = read[0];
                    let scc = read[1];
                    let zeros = if wide {
                        value.trailing_ones()
                    } else {
                        (value as u32).trailing_ones()
                    };
                    let expected = if zeros == if wide { 64 } else { 32 } {
                        u32::MAX
                    } else {
                        zeros
                    };
                    assert_eq!(result, expected, "value={value:x} wide={wide} wave={wave}");
                    assert_eq!(scc, flag, "SCC must survive FF0");
                    assert!(
                        read[2..4].iter().all(|&v| v == 0xcdcd_cdcd),
                        "inactive lanes must not store"
                    );
                    checks += 2;
                }
            }
        }
    }
    println!("scalar_first_zero checks={checks}");
}
