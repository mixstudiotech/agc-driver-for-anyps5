use shader::ir::{Fields, InstructionFamily as F, Operand};
use shader::lowering::Lowerer;
use shader::regalloc::RegisterAllocator;
use shader::{CompileOptions, Compiler, PrecisionMode, ShaderInterface, ShaderStage, TargetIsa};

fn sample_d(dim: u8, addresses: &[u8]) -> Vec<u32> {
    let extra = (addresses.len() - 1).div_ceil(4);
    let mut words = vec![
        0xf000_0f00 | (0x22 << 18) | (u32::from(dim) << 3) | ((extra as u32) << 1),
        0x0020_2000 | u32::from(addresses[0]),
    ];
    for chunk in addresses[1..].chunks(4) {
        words.push(
            chunk
                .iter()
                .enumerate()
                .fold(0, |word, (i, &reg)| word | (u32::from(reg) << (8 * i))),
        );
    }
    words.push(0xbf81_0000);
    words
}

#[test]
fn long_nsa_samples_preserve_every_address_component() {
    for (dim, addresses) in [
        (1, vec![4, 8, 12, 16, 20, 24]),
        (2, vec![4, 8, 12, 16, 20, 24, 28, 36, 40]),
        (5, vec![4, 8, 12, 16, 20, 24, 28]),
    ] {
        let words = sample_d(dim, &addresses);
        for target in [
            TargetIsa::Gfx103,
            TargetIsa::Gfx11,
            TargetIsa::Gfx115,
            TargetIsa::Gfx12,
        ] {
            for wave in [32, 64] {
                let options = CompileOptions::new(target, ShaderStage::Compute, wave);
                let compiler = Compiler::new(target);
                let interface = ShaderInterface::default();
                let ir = compiler.analyze(&words, options, &interface).unwrap();
                let mut registers = RegisterAllocator::new(&ir.metadata, options, &interface);
                let lowered = Lowerer::new(options, PrecisionMode::Exact)
                    .lower(&ir, &mut registers)
                    .unwrap();
                let mut values: Vec<usize> = (0..256).collect();
                let mut images = 0;
                for inst in lowered.instructions() {
                    match inst.fields {
                        Fields::Vop1 {
                            vdst: Operand::Vgpr(dst),
                            src0: Operand::Vgpr(src),
                            ..
                        } if inst.family == F::Vop1 && inst.opcode == 1 => {
                            values[usize::from(dst)] = values[usize::from(src)]
                        }
                        Fields::Mimg(m) => {
                            images += 1;
                            assert_eq!(usize::from(m.addr_dwords), addresses.len());
                            for (i, &reg) in addresses.iter().enumerate() {
                                assert_eq!(
                                    values[usize::from(m.address_register(i))],
                                    usize::from(reg),
                                    "{target:?} wave{wave} component {i}"
                                );
                            }
                            if target.is_gfx11_plus() {
                                let tail = if target.is_gfx12() { 3 } else { 4 };
                                for i in tail + 1..addresses.len() {
                                    assert_eq!(m.addr[i], m.addr[tail] + (i - tail) as u8);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                assert_eq!(images, 1, "the image operation must remain in the output");
                compiler.compile(&words, options).unwrap();
            }
        }
    }
}

#[test]
fn already_legal_nsa_tail_needs_no_address_copies() {
    let words = sample_d(1, &[4, 8, 12, 16, 17, 18]);
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115, TargetIsa::Gfx12] {
        let options = CompileOptions::new(target, ShaderStage::Compute, 64);
        let compiler = Compiler::new(target);
        let interface = ShaderInterface::default();
        let ir = compiler.analyze(&words, options, &interface).unwrap();
        let mut registers = RegisterAllocator::new(&ir.metadata, options, &interface);
        let lowered = Lowerer::new(options, PrecisionMode::Exact)
            .lower(&ir, &mut registers)
            .unwrap();
        assert!(
            !lowered
                .instructions()
                .any(|inst| inst.family == F::Vop1 && inst.opcode == 1)
        );
        compiler.compile(&words, options).unwrap();
    }
}

#[test]
fn nsa_tail_does_not_wrap_the_vgpr_file() {
    let words = sample_d(1, &[4, 8, 12, 16, 255, 0]);
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115, TargetIsa::Gfx12] {
        let options = CompileOptions::new(target, ShaderStage::Compute, 64);
        assert!(matches!(
            Compiler::new(target).compile(&words, options),
            Err(shader::CompileError::OutOfRegisters { .. })
        ));
    }
}
