use shader::{CompileOptions, Compiler, PrecisionMode, ShaderInterface, ShaderStage, TargetIsa};
use shader::ir::{Fields, InstructionFamily as F, Operand};
use shader::lowering::{waitcnt, Lowerer, SOPP_GFX11_WAITCNT};
use shader::regalloc::RegisterAllocator;

#[test]
fn descriptor_spills_wait_for_outstanding_scalar_loads() {
    // The image reads an already-ready descriptor in s[36:43], while a later descriptor
    // is still loading into s[84:99]. Borrowing the latter registers must wait first.
    let words = [
        0xf410_0900, 0xfa00_0000, // s_load_dwordx16 s[36:51], s[0:1], 0
        0xbf8c_c07f,             // s_waitcnt lgkmcnt(0)
        0xf410_1500, 0xfa00_0040, // s_load_dwordx16 s[84:99], s[0:1], 0x40
        0xf000_0108, 0x0009_0907, // image_load v9, v[7:8], s[36:43]
        0xbf8c_c07f,
        0xf000_0108, 0x0015_0a07, // image_load v10, v[7:8], s[84:91]
        0xbf81_0000,
    ];
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115] {
        for wave in [32, 64] {
            let options = CompileOptions::new(target, ShaderStage::Compute, wave);
            let interface = ShaderInterface {
                descriptor_lut: Some(0x200_0000_0000),
                ..ShaderInterface::default()
            };
            let compiler = Compiler::new(target);
            let ir = compiler.analyze(&words, options, &interface).unwrap();
            let mut registers = RegisterAllocator::new(&ir.metadata, options, &interface);
            let lowered = Lowerer::new(options, PrecisionMode::Exact)
                .with_descriptor_conversion(interface.descriptor_lut, false)
                .lower(&ir, &mut registers).unwrap();
            let instructions: Vec<_> = lowered.instructions().collect();
            let load = instructions.iter().position(|inst| matches!(inst.fields,
                Fields::Smem(s) if s.sdata == Operand::Sgpr(84) && s.dwords == 16)).unwrap();
            let spill = instructions.iter().enumerate().skip(load + 1).find(|(_, inst)| {
                inst.mnemonic() == "v_writelane_b32" && matches!(inst.fields,
                    Fields::Vop3(v) if matches!(v.src[0], Operand::Sgpr(84..=99)))
            }).map(|(index, _)| index).expect("the loaded SGPRs must actually be spilled");
            assert!(instructions[load + 1..spill].iter().any(|inst| {
                if inst.family != F::Sopp { return false; }
                let Fields::Sopp { simm16 } = inst.fields else { return false; };
                inst.opcode == SOPP_GFX11_WAITCNT && waitcnt::decode_gfx11(simm16).2 == 0
            }), "{target:?} wave{wave}: an unfinished scalar load is saved by v_writelane");
            compiler.compile_with_interface(&words, options, &interface).unwrap();
        }
    }
}
