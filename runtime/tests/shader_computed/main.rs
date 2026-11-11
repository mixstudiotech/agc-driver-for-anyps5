use shader::{
    CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa, ir::ShaderMetadata,
    regalloc::RegisterAllocator,
};

#[cfg(all(windows, feature = "wddm"))]
mod gpu;
mod kernels;

#[test]
fn backward_helper_branch_after_endpgm_stops_before_trailing_data() {
    let words = [
        0xbf84_0001, // conditional branch to the helper at pc 8
        0xbf81_0000, // main path ends, but the helper remains reachable
        0xbe80_0380, // helper: s_mov_b32 s0, 0
        0xbf82_fffc, // unconditional branch back to pc 0
        0x7e02_0200, // trailing data that resembles a fallthrough instruction
        0x7e02_0200,
    ];
    let ir = shader::decode::Decoder::new().decode(&words).unwrap();
    assert_eq!(ir.code_words, 4);
    assert_eq!(
        ir.instructions.iter().map(|i| i.pc).collect::<Vec<_>>(),
        [0, 4, 8, 12]
    );
    assert_eq!(ir.instructions.last().unwrap().branch_target, Some(0));
}

fn allocator(guest_sgprs: u16) -> RegisterAllocator {
    RegisterAllocator::new(
        &ShaderMetadata {
            sgpr_count: guest_sgprs,
            ..ShaderMetadata::default()
        },
        CompileOptions::new(TargetIsa::Gfx115, ShaderStage::Compute, 64),
        &ShaderInterface::default(),
    )
}

#[test]
fn dispatch_tail_registers_preserve_persistent_reservations() {
    for count in 0..=6 {
        let mut registers = allocator(100);
        if count != 0 {
            assert_eq!(registers.reserve_sgprs(count), Some(100));
        }
        assert_eq!(
            registers.top_sgprs(4),
            if count <= 2 { Some(102) } else { None },
            "reservation {count}"
        );
    }
    let mut registers = allocator(100);
    assert_eq!(registers.reserve_sgprs(2), Some(100));
    assert_eq!(registers.reserve_sgprs(2), Some(104));
    assert_eq!(registers.top_sgprs(4), None);
}

#[test]
fn empty_and_failed_reservations_keep_the_free_tail() {
    for guest in [101, 102] {
        let mut registers = allocator(guest);
        assert_eq!(registers.reserve_sgprs(0), Some(104));
        assert_eq!(registers.top_sgprs(4), Some(102));
    }
    let mut registers = allocator(100);
    assert_eq!(registers.reserve_sgprs(7), None);
    for count in [0, 107, 255] {
        assert_eq!(registers.top_sgprs(count), None);
    }
    assert_eq!(registers.top_sgprs(4), Some(102));
    let mut registers = RegisterAllocator::new(
        &ShaderMetadata {
            uses_relative_sgprs: true,
            ..ShaderMetadata::default()
        },
        CompileOptions::new(TargetIsa::Gfx115, ShaderStage::Compute, 64),
        &ShaderInterface::default(),
    );
    assert_eq!(registers.top_sgprs(4), None);
}

#[test]
fn high_pressure_computed_dispatch_translates_for_every_target() {
    for target in TargetIsa::ALL {
        for wave in [32, 64] {
            let interface = ShaderInterface {
                guest_code_address: Some(0x1_0000_0000),
                guest_pgm_rsrc2: Some(6 << 1),
                ..ShaderInterface::default()
            };
            let artifact = Compiler::new(target)
                .compile_with_interface(
                    kernels::DISPATCH,
                    CompileOptions::new(target, ShaderStage::Compute, wave),
                    &interface,
                )
                .unwrap();
            assert!(artifact.num_sgprs <= 106, "{target:?} wave{wave}");
        }
    }
}
