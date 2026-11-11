use shader::{
    ArtifactCache, CompileError, CompileOptions, Compiler, PrecisionMode, ResourceKind,
    ShaderInterface, ShaderStage, TargetIsa,
};

const COMPUTE: &[u32] = &[
    0xf408_0100,
    0xfa00_0000,
    0x3402_0082,
    0xbf8c_c07f,
    0xe030_1000,
    0x8001_0201,
    0xbf8c_3f70,
    0x1004_04f4,
    0x3e04_0502,
    0xe070_1000,
    0x8001_0201,
    0xbf81_0000,
];

fn options(target: TargetIsa) -> CompileOptions {
    CompileOptions::new(target, ShaderStage::Compute, 32)
}

#[test]
fn compute_shader_translates_for_every_target() {
    for target in TargetIsa::ALL {
        let artifact = Compiler::new(target)
            .compile(COMPUTE, options(target))
            .expect("compile");
        assert_eq!(artifact.options.target, target);
        assert_eq!(artifact.guest_words, COMPUTE.len());
        assert_eq!(artifact.user_sgpr_count, 2);
        assert!(artifact.num_vgprs >= 3);
        assert_eq!(
            artifact.code.len() % (target.instruction_cache_line_bytes() / 4),
            0
        );
        let kinds: Vec<_> = artifact
            .bindings
            .iter()
            .map(|b| (b.kind, b.base_sgpr, b.writable))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (ResourceKind::ScalarMemoryPair, 0, false),
                (ResourceKind::BufferDescriptor, 4, true)
            ]
        );
    }
}

#[test]
fn target_specific_translations_are_visible_in_the_code() {
    let gfx1013 = Compiler::new(TargetIsa::Gfx1013)
        .compile(COMPUTE, options(TargetIsa::Gfx1013))
        .unwrap();
    assert_eq!(gfx1013.program(), COMPUTE);

    let gfx103 = Compiler::new(TargetIsa::Gfx103)
        .with_precision(PrecisionMode::Fused)
        .compile(COMPUTE, options(TargetIsa::Gfx103))
        .unwrap();
    assert!(gfx103.program().contains(&0x5604_0502), "v_fmac_f32");
    assert!(!gfx103.program().contains(&0x3e04_0502));
    let exact = Compiler::new(TargetIsa::Gfx103)
        .compile(COMPUTE, options(TargetIsa::Gfx103))
        .unwrap();
    assert!(
        !exact.program().contains(&0x5604_0502),
        "v_mac_f32 stays unfused"
    );

    let gfx11 = Compiler::new(TargetIsa::Gfx11)
        .compile(COMPUTE, options(TargetIsa::Gfx11))
        .unwrap();
    assert!(gfx11.program().contains(&0xbf89_fc07));
    assert_eq!(gfx11.program().last(), Some(&0xbfb0_0000));

    let gfx12 = Compiler::new(TargetIsa::Gfx12)
        .compile(COMPUTE, options(TargetIsa::Gfx12))
        .unwrap();
    let program = gfx12.program();
    assert!(program.contains(&0xbfc7_0000), "kmcnt wait");
    assert!(program.contains(&0xbfc0_0000), "loadcnt wait");
    assert!(
        !program.iter().any(|&w| w >> 16 == 0xbf8c),
        "no gfx10 s_waitcnt"
    );
}

#[test]
fn analysis_exposes_cfg_and_metadata() {
    let ir = Compiler::new(TargetIsa::Gfx11)
        .analyze(
            COMPUTE,
            options(TargetIsa::Gfx11),
            &ShaderInterface::default(),
        )
        .unwrap();
    assert_eq!(ir.blocks.len(), 1);
    assert_eq!(ir.metadata.max_sgpr, Some(7));
    assert_eq!(ir.metadata.max_vgpr, Some(2));
    assert!(!ir.metadata.uses_lds && !ir.metadata.uses_scratch);
}

#[test]
fn compile_errors_are_descriptive() {
    let gfx11 = Compiler::new(TargetIsa::Gfx11);
    assert!(matches!(
        gfx11.compile(COMPUTE, options(TargetIsa::Gfx12)),
        Err(CompileError::TargetMismatch { .. })
    ));
    let bad_wave = CompileOptions::new(TargetIsa::Gfx11, ShaderStage::Compute, 16);
    assert!(matches!(
        gfx11.compile(COMPUTE, bad_wave),
        Err(CompileError::InvalidOptions(_))
    ));
    assert_eq!(
        gfx11.compile(&[], options(TargetIsa::Gfx11)),
        Err(CompileError::EmptyProgram)
    );

    // v_interp_p1ll_f16 v0, v1, attr0.x translates; with an output modifier (omod 1, VOP3
    // bits 60:59) it has no gfx11 form.
    let interp_f16 = [0xd742_0000, 0x0002_0200, 0xbf81_0000];
    assert!(gfx11
        .compile(&interp_f16, options(TargetIsa::Gfx11))
        .is_ok());
    let with_omod = [0xd742_0000, 0x0802_0200, 0xbf81_0000];
    let err = gfx11
        .compile(&with_omod, options(TargetIsa::Gfx11))
        .unwrap_err();
    assert!(err.is_unsupported());
    assert_eq!(err.pc(), Some(0));
    assert!(err.to_string().contains("v_interp_p1ll_f16"), "{err}");
    assert!(CompileError::Unsupported.is_unsupported());
}

#[test]
fn gfx_ip_versions_select_targets() {
    assert_eq!(TargetIsa::from_gfx_ip(10, 1, 3), Some(TargetIsa::Gfx1013));
    assert_eq!(TargetIsa::from_gfx_ip(10, 3, 0), Some(TargetIsa::Gfx103));
    assert_eq!(TargetIsa::from_gfx_ip(11, 0, 0), Some(TargetIsa::Gfx11));
    assert_eq!(TargetIsa::from_gfx_ip(11, 5, 1), Some(TargetIsa::Gfx115));
    assert_eq!(TargetIsa::from_gfx_ip(12, 0, 1), Some(TargetIsa::Gfx12));
    assert_eq!(TargetIsa::from_gfx_ip(9, 0, 0), None);
}

#[test]
fn cache_is_keyed_by_words_and_options() {
    let compiler = Compiler::new(TargetIsa::Gfx11);
    let mut cache = ArtifactCache::new();
    let first = cache
        .get_or_compile(&compiler, COMPUTE, options(TargetIsa::Gfx11))
        .unwrap()
        .clone();
    let again = cache
        .get_or_compile(&compiler, COMPUTE, options(TargetIsa::Gfx11))
        .unwrap()
        .clone();
    assert_eq!(first, again);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(first.code_bytes().len(), first.code.len() * 4);
}

/// A pixel program that orders its section as gfx10 does, with two user SGPRs: `prim_mask` is
/// s2, the collision word s3 and the intrawave collision mask s4.
///   s_mov_b32 s6, s4; s_cmp_lt_i32 s3, 0; s_cbranch_scc0 over the wait;
///   s_setreg_b32 hwreg(HW_REG_POPS_PACKER, 0, 3), s3; s_add_i32 s5, s3, src_pops_exiting_wave_id;
///   s_sendmsg MSG_ORDERED_PS_DONE; exp null done vm; s_endpgm
const ORDERED_PIXEL: &[u32] = &[
    0xbe86_0304,
    0xbf04_8003,
    0xbf84_0002,
    0xb983_1019,
    0x8105_ef03,
    0xbf90_0007,
    0xf800_1890,
    0x0000_0000,
    0xbf81_0000,
];
/// SPI_SHADER_PGM_RSRC2_PS: two user SGPRs, LOAD_COLLISION_WAVEID and LOAD_INTRAWAVE_COLLISION.
const ORDERED_RSRC2: u32 = 2 << 1 | 1 << 25 | 1 << 26;

fn ordered_pixel(words: &[u32], target: TargetIsa) -> Result<shader::ShaderArtifact, CompileError> {
    let interface = ShaderInterface {
        guest_pgm_rsrc2: Some(ORDERED_RSRC2),
        ..ShaderInterface::default()
    };
    let options = CompileOptions::new(target, ShaderStage::Fragment, 64);
    Compiler::new(target).compile_with_interface(words, options, &interface)
}

#[test]
fn ordered_pixel_shading_waits_the_way_the_target_does() {
    let has = |code: &[u32], run: &[u32]| code.windows(run.len()).any(|w| w == run);
    for (target, wait) in [(TargetIsa::Gfx115, 0xbf8b_0000), (TargetIsa::Gfx12, 0xbf8b_0002)] {
        let artifact = ordered_pixel(ORDERED_PIXEL, target).unwrap();
        let code = artifact.program();
        // The prologue provides the inputs the target does not load: this wave overlaps an
        // earlier one, and none of its quads overlap each other.
        assert!(has(code, &[0xbe83_00ff, 0x8000_0001]), "s_mov_b32 s3, 0x80000001: {code:08x?}");
        assert!(has(code, &[0xbe84_0080]), "s_mov_b32 s4, 0: {code:08x?}");
        // The packer write is the wait, the exiting wave is this one, and the done export ends
        // the section: no register write, no message.
        assert_eq!(code.iter().filter(|&&w| w == wait).count(), 1, "s_wait_event: {code:08x?}");
        assert!(has(code, &[0x8105_8103]), "s_add_i32 s5, s3, 1: {code:08x?}");
        assert!(!code.iter().any(|&w| w >> 23 == 0xb900_0000 >> 23), "s_setreg_b32: {code:08x?}");
        assert!(!code.iter().any(|&w| w >> 16 == 0xbfb6), "s_sendmsg: {code:08x?}");
        assert_eq!(artifact.pgm_rsrc2 & (3 << 25), 0, "the target loads no collision SGPRs");
    }
    // gfx10.3 orders as the guest does.
    let gfx103 = ordered_pixel(ORDERED_PIXEL, TargetIsa::Gfx103).unwrap();
    assert_eq!(gfx103.program(), ORDERED_PIXEL);
    assert_eq!(gfx103.pgm_rsrc2 & (3 << 25), 3 << 25);
}

#[test]
fn a_done_export_before_the_end_of_the_ordered_section_is_refused() {
    // The export and the message swapped: on gfx11 the export would end the section first.
    let mut words = ORDERED_PIXEL.to_vec();
    words.copy_within(6..8, 5);
    words[7] = 0xbf90_0007;
    let error = ordered_pixel(&words, TargetIsa::Gfx115).unwrap_err();
    assert!(error.is_unsupported() && error.to_string().contains("ordered section"), "{error}");
    assert!(ordered_pixel(&words, TargetIsa::Gfx103).is_ok());
}
