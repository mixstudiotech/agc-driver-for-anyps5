use shader::PrecisionMode;
use shader::ir::{Fields, InstructionFamily as F, Operand, RegFile};
use shader::lowering::Lowerer;
use shader::regalloc::RegisterAllocator;
use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa};
mod kernels;

use kernels::selectors;

#[test]
fn integer_dot2_compiles_all_half_selectors_clamps_and_aliases() {
    for target in TargetIsa::ALL {
        for wave in [32, 64] {
            for kernel in kernels::KERNELS {
                for lo in 0..4 {
                    for hi in 0..4 {
                        let words = selectors(kernel.words, lo, hi);
                        let options = CompileOptions::new(target, ShaderStage::Compute, wave);
                        let interface = ShaderInterface {
                            guest_pgm_rsrc2: Some(12 << 1),
                            ..ShaderInterface::default()
                        };
                        Compiler::new(target)
                            .compile_with_interface(&words, options, &interface)
                            .unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn lowered_clamp_keeps_its_explicit_scalar_mask_dependency() {
    let kernel = kernels::KERNELS
        .iter()
        .find(|k| k.signed && k.clamp && k.alias == 6)
        .unwrap();
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115, TargetIsa::Gfx12] {
        for wave in [32, 64] {
            let options = CompileOptions::new(target, ShaderStage::Compute, wave);
            let interface = ShaderInterface {
                guest_pgm_rsrc2: Some(12 << 1),
                ..ShaderInterface::default()
            };
            let ir = Compiler::new(target)
                .analyze(kernel.words, options, &interface)
                .unwrap();
            let mut regs = RegisterAllocator::new(&ir.metadata, options, &interface);
            let lowered = Lowerer::new(options, PrecisionMode::Exact)
                .lower(&ir, &mut regs)
                .unwrap();
            let cndmask = lowered.instructions().find(|i| i.family == F::Vop3 && matches!(i.fields,
                Fields::Vop3(v) if matches!(v.src[2], Operand::Sgpr(_)) && v.vdst == Operand::Vgpr(6))).unwrap();
            let Fields::Vop3(v) = cndmask.fields else {
                unreachable!()
            };
            let Operand::Sgpr(mask) = v.src[2] else {
                unreachable!()
            };
            assert!(
                cndmask
                    .uses(wave == 64)
                    .iter()
                    .any(|r| r.file == RegFile::Scalar
                        && r.first == u16::from(mask)
                        && r.count == if wave == 64 { 2 } else { 1 })
            );
        }
    }
}

#[test]
fn gfx12_reused_clamp_mask_has_the_required_valu_scalar_wait() {
    let kernel = kernels::KERNELS
        .iter()
        .find(|k| k.signed && k.clamp && k.alias == 6)
        .unwrap();
    let mut words = kernel.words.to_vec();
    let at = words
        .iter()
        .position(|&v| v & 0xff80_0000 == 0xcc00_0000)
        .unwrap();
    let second = words[at..at + 2].to_vec();
    words.splice(at + 2..at + 2, second);
    for wave in [32, 64] {
        let options = CompileOptions::new(TargetIsa::Gfx12, ShaderStage::Compute, wave);
        let interface = ShaderInterface {
            guest_pgm_rsrc2: Some(12 << 1),
            ..ShaderInterface::default()
        };
        let ir = Compiler::new(options.target)
            .analyze(&words, options, &interface)
            .unwrap();
        let mut regs = RegisterAllocator::new(&ir.metadata, options, &interface);
        let lowered = Lowerer::new(options, PrecisionMode::Exact)
            .lower(&ir, &mut regs)
            .unwrap();
        let masks: Vec<_> = lowered
            .instructions()
            .filter(|i| {
                i.family == F::Vop3
                    && matches!(i.fields,
            Fields::Vop3(v) if matches!(v.src[2], Operand::Sgpr(_)) && v.vdst == Operand::Vgpr(6))
            })
            .collect();
        assert_eq!(masks.len(), 2);
        let needle = shader::encode::encode_instruction(masks[1], options.target).unwrap();
        let encoded = shader::encode::Encoder::new(options)
            .encode(&lowered)
            .unwrap();
        let positions: Vec<_> = encoded.code[..encoded.program_words]
            .windows(needle.len())
            .enumerate()
            .filter_map(|(i, values)| (values == needle).then_some(i))
            .collect();
        assert_eq!(positions.len(), 2);
        assert_eq!(
            encoded.code[positions[1] - 1] & 0xffff_0e00,
            0xbf88_0000,
            "the second clamp must wait for its reused mask"
        );
    }
}
