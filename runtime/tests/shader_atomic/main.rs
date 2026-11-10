use shader::{CompileOptions, Compiler, PrecisionMode, ShaderInterface, ShaderStage, TargetIsa};
use shader::ir::{opcodes::flag, Fields, InstructionFamily as F};
use shader::lowering::Lowerer;
use shader::regalloc::RegisterAllocator;

#[cfg(all(windows, feature = "wddm"))]
mod gpu;

fn image_program(max: bool, returning: bool) -> [u32; 4] {
    [0xf000_0100 | ((if max { 0x1f } else { 0x1e }) << 18) | if returning { 0x2000 } else { 0 }, 0x0002_0402, 0xbf8c_3f70, 0xbf81_0000]
}

#[test]
fn float_image_atomics_compile_with_cas_on_gfx11_and_keep_native_paths() {
    for target in TargetIsa::ALL {
        for wave in [32, 64] {
            for max in [false, true] {
                for returning in [false, true] {
                    let words = image_program(max, returning);
                    let options = CompileOptions::new(target, ShaderStage::Compute, wave);
                    let compiler = Compiler::new(target);
                    let interface = ShaderInterface::default();
                    let ir = compiler.analyze(&words, options, &interface).unwrap();
                    let mut registers = RegisterAllocator::new(&ir.metadata, options, &interface);
                    let lowered = Lowerer::new(options, PrecisionMode::Exact).lower(&ir, &mut registers).unwrap();
                    let images: Vec<_> = lowered.instructions().filter(|inst| inst.family == F::Mimg).collect();
                    assert_eq!(images.len(), 1);
                    if target.is_gfx11() {
                        let Fields::Mimg(image) = images[0].fields else { unreachable!() };
                        assert_eq!(images[0].opcode, 0x0b);
                        assert!(images[0].has_flag(flag::CMPSWAP));
                        assert_eq!(image.dmask, 3);
                        assert_eq!(image.data_dwords, 2);
                        assert!(image.glc);
                        let branch = lowered.instructions().find(|inst| inst.branch_target.is_some()).unwrap();
                        let header = lowered.instructions().find(|inst| Some(inst.pc) == branch.branch_target).unwrap();
                        assert_eq!((header.family, header.opcode), (F::Sopp, 0));
                    }
                    compiler.compile(&words, options).unwrap();
                }
            }
        }
    }
}

#[test]
fn float_image_cas_rejects_status_and_non_dword_forms() {
    for (first, second) in [(0x1_0000, 0), (0x2_0000, 0), (0, 0x8000_0000), (0x200, 0)] {
        let mut words = image_program(true, true);
        words[0] |= first;
        words[1] |= second;
        let error = Compiler::new(TargetIsa::Gfx115).compile(&words, CompileOptions::new(TargetIsa::Gfx115, ShaderStage::Compute, 32)).unwrap_err();
        assert!(error.to_string().contains("without D16, TFE or LWE"), "{error}");
    }
}
