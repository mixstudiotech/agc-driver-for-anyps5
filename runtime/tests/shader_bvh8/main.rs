use shader::{
    CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa,
    decode::Decoder,
    encode::encode_instruction,
    ir::{Fields, RegRange},
};

#[cfg(all(windows, feature = "wddm"))]
mod case_names;
#[cfg(all(windows, feature = "wddm"))]
mod gpu_bvh8;
#[cfg(all(windows, feature = "wddm"))]
mod gpu_stack;
#[cfg(all(windows, feature = "wddm"))]
mod kernels;

// SDK 10.000 Pro oracle: result v[32:41], inputs v[16:26], descriptor s[4:7].
const BVH8: [u32; 4] = [0xf194_9f05, 0x0001_2010, 0x1413_1211, 0x0000_1a17];

#[test]
fn pro_grouped_addresses_preserve_raw_encoding_and_all_registers() {
    let instruction = Decoder::new().decode_at(&BVH8, 0).unwrap();
    let Fields::Mimg(image) = instruction.fields else {
        panic!("MIMG")
    };
    assert_eq!(instruction.mnemonic(), "image_bvh8_intersect_ray");
    assert_eq!(instruction.word_count(), 4);
    assert_eq!(
        (image.addr_dwords, image.data_dwords, image.nsa_words),
        (11, 10, 2)
    );
    assert_eq!(
        (0..11)
            .map(|i| image.address_register(i))
            .collect::<Vec<_>>(),
        (16..=26).collect::<Vec<_>>()
    );
    for wave64 in [false, true] {
        let definitions = instruction.defs(wave64);
        for range in [
            RegRange::vector(32, 10),
            RegRange::vector(20, 3),
            RegRange::vector(23, 3),
        ] {
            assert!(definitions.contains(&range), "missing definition {range:?}");
        }
        let uses = instruction.uses(wave64);
        for register in 16..=26 {
            assert!(uses.contains(&RegRange::vector(register, 1)));
        }
    }
    assert_eq!(
        encode_instruction(&instruction, TargetIsa::Gfx1013).unwrap(),
        BVH8
    );
    let mut words = BVH8.to_vec();
    words.push(0xbf81_0000);
    let ir = Compiler::new(TargetIsa::Gfx115)
        .analyze(
            &words,
            CompileOptions::new(TargetIsa::Gfx115, ShaderStage::Compute, 64),
            &ShaderInterface::default(),
        )
        .unwrap();
    assert_eq!(ir.metadata.vgpr_count, 42);
}

#[test]
fn malformed_or_wrapping_bvh8_groups_are_refused() {
    for (word, value) in [
        (0, 0xf194_9f01),
        (0, 0xf194_9f03),
        (0, 0xf194_9f07),
        (0, 0xf194_bf05),
        (0, 0xf194_9f85),
        (0, 0xf394_9f05),
        (1, 0x8001_2010),
        (1, 0x4001_2010),
        (1, 0x0001_fa10),
        (1, 0x0021_2010),
        (2, 0x1413_1212),
        (2, 0x1414_1211),
        (2, 0xfe13_1211),
        (3, 0x0000_1afe),
    ] {
        let mut words = BVH8.to_vec();
        words.push(0);
        words[word] = value;
        assert!(Decoder::new().decode_at(&words, 0).is_err(), "{words:08x?}");
    }
}

#[test]
fn sdk_read_only_allocation_can_alias_direction_with_results() {
    let words = [0xf194_9f05, 0x0005_000d, 0x0a04_030e, 0x0000_0500];
    let instruction = Decoder::new().decode_at(&words, 0).unwrap();
    let Fields::Mimg(image) = instruction.fields else {
        panic!("MIMG")
    };
    assert_eq!(image.vdata, 0);
    assert_eq!(
        (7..10)
            .map(|i| image.address_register(i))
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(
        encode_instruction(&instruction, TargetIsa::Gfx1013).unwrap(),
        words
    );
}

#[test]
fn stack8_tracks_control_writeback_and_all_eight_nodes() {
    let words = [0xdb84_0010, 0x0000_201b];
    let instruction = Decoder::new().decode_at(&words, 0).unwrap();
    assert_eq!(instruction.mnemonic(), "ds_bvh_stack_push8_pop1_rtn_b32");
    assert_eq!(
        instruction.defs(true),
        [RegRange::vector(0, 1), RegRange::vector(27, 1)]
    );
    for range in [
        RegRange::vector(27, 1),
        RegRange::vector(32, 1),
        RegRange::vector(0, 8),
    ] {
        assert!(instruction.uses(true).contains(&range));
    }
    assert_eq!(
        encode_instruction(&instruction, TargetIsa::Gfx1013).unwrap(),
        words
    );
    for size in [0, 7, 32, 256] {
        let bad = [0xdb84_0000 | size, words[1]];
        assert!(Decoder::new().decode_at(&bad, 0).is_err());
    }
    for bad in [
        [0xdb86_0010, words[1]],
        [words[0], 0x00f9_201b],
        [words[0], 0x1b00_201b],
    ] {
        assert!(Decoder::new().decode_at(&bad, 0).is_err());
    }
}

#[test]
fn stack8_expands_where_the_host_lacks_it() {
    let program = [0xdb84_001f, 0x0000_201b, 0xbf81_0000];
    for target in [
        TargetIsa::Gfx1013,
        TargetIsa::Gfx103,
        TargetIsa::Gfx11,
        TargetIsa::Gfx115,
        TargetIsa::Gfx12,
    ] {
        for wave in [32, 64] {
            let artifact = Compiler::new(target)
                .compile(
                    &program,
                    CompileOptions::new(target, ShaderStage::Compute, wave),
                )
                .unwrap();
            // gfx12 has the instruction, with the encoding the Pro gives it.
            let native = artifact.program().windows(2).any(|w| w == &program[..2]);
            assert_eq!(native, target == TargetIsa::Gfx12, "{target:?}");
            assert!(native || artifact.program_words > program.len());
            assert!(artifact.num_vgprs <= 256);
        }
    }
}

#[test]
fn odd_pair_starts_and_high_in_out_groups_round_trip() {
    let words = [0xf194_9f05, 0x0001_2017, 0xe61a_1918, 0x0000_33f0];
    let instruction = Decoder::new().decode_at(&words, 0).unwrap();
    assert_eq!(
        encode_instruction(&instruction, TargetIsa::Gfx1013).unwrap(),
        words
    );
    let Fields::Mimg(image) = instruction.fields else {
        panic!("MIMG")
    };
    assert_eq!(
        (0..11)
            .map(|i| image.address_register(i))
            .collect::<Vec<_>>(),
        [23, 24, 25, 26, 230, 231, 232, 240, 241, 242, 51]
    );
    let mut program = words.to_vec();
    program.push(0xbf81_0000);
    let ir = Compiler::new(TargetIsa::Gfx115)
        .analyze(
            &program,
            CompileOptions::new(TargetIsa::Gfx115, ShaderStage::Compute, 32),
            &ShaderInterface::default(),
        )
        .unwrap();
    assert_eq!(ir.metadata.vgpr_count, 243);
}

#[test]
fn pro_instructions_expand_on_each_host_isa() {
    let mut program = BVH8.to_vec();
    program.extend([0xdb84_0010, 0x0030_2a2b, 0xbf81_0000]);
    for target in [
        TargetIsa::Gfx1013,
        TargetIsa::Gfx103,
        TargetIsa::Gfx11,
        TargetIsa::Gfx115,
        TargetIsa::Gfx12,
    ] {
        for wave in [32, 64] {
            let artifact = Compiler::new(target)
                .compile(
                    &program,
                    CompileOptions::new(target, ShaderStage::Compute, wave),
                )
                .unwrap();
            assert!(artifact.program_words > program.len());
            assert!(artifact.num_vgprs <= 256);
        }
    }
}

#[test]
fn pro_expansion_fits_the_corpus_register_pressure() {
    // v167 is the observed corpus maximum; s99 forces transient scalar spilling.
    let mut program = vec![0x7f4e_0280, 0xbee3_0381];
    program.extend(BVH8);
    program.extend([0xdb84_0010, 0x0030_2a2b, 0xbf81_0000]);
    for target in [TargetIsa::Gfx115, TargetIsa::Gfx12] {
        for wave in [32, 64] {
            let artifact = Compiler::new(target)
                .compile(
                    &program,
                    CompileOptions::new(target, ShaderStage::Compute, wave),
                )
                .unwrap();
            assert!(artifact.num_vgprs <= 256);
        }
    }
}
