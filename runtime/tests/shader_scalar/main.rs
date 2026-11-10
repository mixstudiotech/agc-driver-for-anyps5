use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa};
mod kernels;

#[test]
fn first_zero_compiles_for_every_target_and_register_pressure() {
    for target in TargetIsa::ALL {
        for wave in [32, 64] {
            for words in [
                kernels::FF0_32,
                kernels::FF0_32_HIGH,
                kernels::FF0_64,
                kernels::FF0_64_HIGH,
            ] {
                let interface = ShaderInterface {
                    guest_pgm_rsrc2: Some(12 << 1),
                    ..ShaderInterface::default()
                };
                Compiler::new(target)
                    .compile_with_interface(
                        words,
                        CompileOptions::new(target, ShaderStage::Compute, wave),
                        &interface,
                    )
                    .unwrap();
            }
        }
    }
}
