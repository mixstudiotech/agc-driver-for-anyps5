use super::*;
use crate::{Compiler, ShaderArtifact};

const S_ENDPGM: u32 = 0xbf81_0000;
const SDWA_ADD: [u32; 2] = [0x0600_04f9, 0x0204_1501];

fn build(
    words: &[u32],
    target: TargetIsa,
    stage: ShaderStage,
    wave_size: u8,
    interface: ShaderInterface,
) -> Result<ShaderArtifact, CompileError> {
    Compiler::new(target).compile_with_interface(
        words,
        CompileOptions::new(target, stage, wave_size),
        &interface,
    )
}

fn compute(words: &[u32], target: TargetIsa) -> ShaderArtifact {
    build(
        words,
        target,
        ShaderStage::Compute,
        32,
        ShaderInterface::default(),
    )
    .expect("compile")
}

#[test]
fn vgpr_blocks_follow_wave_granularity() {
    let cases = [
        (1, false, 0),
        (8, false, 0),
        (9, false, 1),
        (256, false, 31),
        (4, true, 0),
        (5, true, 1),
        (256, true, 63),
    ];
    for (count, wave64, blocks) in cases {
        assert_eq!(
            encode_vgpr_blocks(count, wave64),
            blocks,
            "{count} {wave64}"
        );
    }
    assert_eq!(decode_vgpr_blocks(1, false), 16);
    assert_eq!(decode_vgpr_blocks(1, true), 8);
}

#[test]
fn rsrc1_reports_vgprs_and_default_modes() {
    let words = [0x7e12_02f2, S_ENDPGM];
    let a = compute(&words, TargetIsa::Gfx11);
    assert_eq!(a.num_vgprs, 10);
    assert_eq!(a.pgm_rsrc1 & RSRC1_VGPRS_MASK, 1);
    assert_eq!(
        (a.pgm_rsrc1 >> RSRC1_FLOAT_MODE_SHIFT) & 0xff,
        FLOAT_MODE_DEFAULT
    );
    assert_ne!(a.pgm_rsrc1 & RSRC1_WGP_MODE, 0);
    assert_ne!(a.pgm_rsrc1 & RSRC1_MEM_ORDERED, 0);
    assert_ne!(a.pgm_rsrc1 & RSRC1_DX10_CLAMP, 0);
    assert_eq!(a.pgm_rsrc1 & RSRC1_SGPRS_MASK, 0);

    let wave64 = build(
        &words,
        TargetIsa::Gfx11,
        ShaderStage::Compute,
        64,
        ShaderInterface::default(),
    )
    .unwrap();
    assert_eq!(wave64.pgm_rsrc1 & RSRC1_VGPRS_MASK, 2);
    let gfx12 = compute(&words, TargetIsa::Gfx12);
    assert_eq!(gfx12.pgm_rsrc1 & (RSRC1_DX10_CLAMP | RSRC1_IEEE_MODE), 0);
}

#[test]
fn user_sgprs_are_inferred_from_live_ins() {
    let words = [0xf408_0100, 0xfa00_0010, 0xbe89_0302, S_ENDPGM];
    let a = compute(&words, TargetIsa::Gfx103);
    assert_eq!(a.user_sgpr_count, 3);
    assert_eq!(
        (a.pgm_rsrc2 & RSRC2_USER_SGPR_MASK) >> RSRC2_USER_SGPR_SHIFT,
        3
    );
    assert_eq!(a.bindings[0].kind, ResourceKind::ScalarMemoryPair);

    let explicit = ShaderInterface {
        user_sgpr_count: Some(5),
        ..ShaderInterface::default()
    };
    let b = build(
        &words,
        TargetIsa::Gfx103,
        ShaderStage::Compute,
        32,
        explicit,
    )
    .unwrap();
    assert_eq!(b.user_sgpr_count, 5);
}

#[test]
fn guest_register_values_are_carried_over() {
    let guest_rsrc2 = RSRC2_TGID_X_EN
        | (RSRC2_TGID_X_EN << 1)
        | (2 << RSRC2_LDS_SHIFT)
        | (4 << RSRC2_USER_SGPR_SHIFT);
    let guest_rsrc1 = (0xf0 << RSRC1_FLOAT_MODE_SHIFT) | RSRC1_DX10_CLAMP | 5;
    let interface = ShaderInterface {
        guest_pgm_rsrc1: Some(guest_rsrc1),
        guest_pgm_rsrc2: Some(guest_rsrc2),
        ..ShaderInterface::default()
    };
    let a = build(
        &[0x7e12_02f2, S_ENDPGM],
        TargetIsa::Gfx11,
        ShaderStage::Compute,
        32,
        interface,
    )
    .unwrap();
    assert_eq!(a.user_sgpr_count, 4);
    assert_eq!(a.pgm_rsrc2, guest_rsrc2);
    assert_eq!(a.lds_bytes, 1024);
    assert_eq!(a.pgm_rsrc1, (guest_rsrc1 & !RSRC1_VGPRS_MASK) | 1);
}

#[test]
fn lds_size_is_granular() {
    let interface = ShaderInterface {
        lds_bytes: 1000,
        ..ShaderInterface::default()
    };
    let a = build(
        &[0xd8dc_0201, 0x0000_0002, S_ENDPGM],
        TargetIsa::Gfx11,
        ShaderStage::Compute,
        32,
        interface,
    )
    .unwrap();
    assert_eq!((a.pgm_rsrc2 & RSRC2_LDS_MASK) >> RSRC2_LDS_SHIFT, 2);
    assert_eq!(a.lds_bytes, 1000);
}

#[test]
fn graphics_rsrc1_defaults_use_the_graphics_layout() {
    for stage in [ShaderStage::Fragment, ShaderStage::Vertex] {
        let a = build(
            &[0x7e12_02f2, S_ENDPGM],
            TargetIsa::Gfx115,
            stage,
            32,
            ShaderInterface::default(),
        )
        .unwrap();
        assert_ne!(a.pgm_rsrc1 & RSRC1_GRAPHICS_MEM_ORDERED, 0, "{stage:?}");
        assert_eq!(
            a.pgm_rsrc1 & (RSRC1_MEM_ORDERED | RSRC1_WGP_MODE),
            0,
            "{stage:?}: GS_VGPR_COMP_CNT / FP16_OVFL bits left alone"
        );
        assert_eq!(
            (a.pgm_rsrc1 >> RSRC1_FLOAT_MODE_SHIFT) & 0xff,
            FLOAT_MODE_DEFAULT
        );
    }
}

#[test]
fn lds_sizes_cover_whole_kibibytes_on_gfx11() {
    let field = |a: &ShaderArtifact, mask: u32, shift: u32| (a.pgm_rsrc2 & mask) >> shift;
    let lds = ShaderInterface {
        lds_bytes: 1500,
        ..ShaderInterface::default()
    };
    let compute = |target| {
        build(
            &[0xd8dc_0201, 0x0000_0002, S_ENDPGM],
            target,
            ShaderStage::Compute,
            32,
            lds,
        )
        .unwrap()
    };
    assert_eq!(
        field(&compute(TargetIsa::Gfx103), RSRC2_LDS_MASK, RSRC2_LDS_SHIFT),
        3
    );
    assert_eq!(
        field(&compute(TargetIsa::Gfx115), RSRC2_LDS_MASK, RSRC2_LDS_SHIFT),
        4
    );
    let graphics = |target, stage, rsrc2| {
        let guest = ShaderInterface {
            guest_pgm_rsrc2: Some(rsrc2),
            ..ShaderInterface::default()
        };
        build(&[S_ENDPGM], target, stage, 32, guest).unwrap()
    };
    // EXTRA_LDS_SIZE: three 512-byte units become two 1 KiB units.
    let ps = |target| graphics(target, ShaderStage::Fragment, 3 << RSRC2_PS_EXTRA_LDS_SHIFT);
    let (ps_mask, ps_shift) = (RSRC2_PS_EXTRA_LDS_MASK, RSRC2_PS_EXTRA_LDS_SHIFT);
    assert_eq!(field(&ps(TargetIsa::Gfx103), ps_mask, ps_shift), 3);
    assert_eq!(field(&ps(TargetIsa::Gfx115), ps_mask, ps_shift), 2);
    let gs = graphics(
        TargetIsa::Gfx115,
        ShaderStage::Vertex,
        5 << RSRC2_GS_LDS_SHIFT,
    );
    assert_eq!(field(&gs, RSRC2_GS_LDS_MASK, RSRC2_GS_LDS_SHIFT), 6);
}

#[test]
fn vgpr_allocation_covers_the_hardware_loaded_inputs() {
    // PERSP_CENTER in v0-v1; with POS_X..POS_W laid out, POS_W lands in v5.
    assert_eq!(ps_input_vgprs(0x802, 0xf02), 6);
    assert_eq!(ps_input_vgprs(0x802, 0x802), 3);
    assert_eq!(ps_input_vgprs(0x2, 0xf02), 2);
    let program = |target, stage, interface| {
        build(&[S_ENDPGM], target, stage, 32, interface)
            .unwrap()
            .num_vgprs
    };
    let ps = ShaderInterface {
        ps_input_ena: Some(0x802),
        ps_input_addr: Some(0xf02),
        ..ShaderInterface::default()
    };
    assert_eq!(program(TargetIsa::Gfx115, ShaderStage::Fragment, ps), 6);
    // ES_VGPR_COMP_CNT 3: vertex ID in v5 through instance ID in v8.
    let ngg = ShaderInterface {
        guest_pgm_rsrc2: Some(3 << 16),
        ..ShaderInterface::default()
    };
    assert_eq!(program(TargetIsa::Gfx115, ShaderStage::Vertex, ngg), 9);
    // gfx10 loads each thread ID component into its own VGPR, gfx11 packs them into v0.
    let tids = ShaderInterface {
        guest_pgm_rsrc2: Some(2 << RSRC2_TIDIG_SHIFT),
        ..ShaderInterface::default()
    };
    assert_eq!(program(TargetIsa::Gfx103, ShaderStage::Compute, tids), 3);
    assert_eq!(program(TargetIsa::Gfx115, ShaderStage::Compute, tids), 1);
}

#[test]
fn scratch_vgprs_extend_the_allocation() {
    let words = [SDWA_ADD[0], SDWA_ADD[1], S_ENDPGM];
    let frag = |target| {
        build(
            &words,
            target,
            ShaderStage::Fragment,
            32,
            ShaderInterface::default(),
        )
        .unwrap()
    };
    assert_eq!(frag(TargetIsa::Gfx103).num_vgprs, 3);
    assert_eq!(frag(TargetIsa::Gfx11).num_vgprs, 6);
}

#[test]
fn register_pressure_is_reported() {
    let words = [0x07fe_04f9, SDWA_ADD[1], S_ENDPGM];
    let err = build(
        &words,
        TargetIsa::Gfx11,
        ShaderStage::Fragment,
        32,
        ShaderInterface::default(),
    )
    .unwrap_err();
    assert_eq!(err, CompileError::OutOfRegisters { vgprs: 259 });
}

#[test]
fn bindings_and_exports_are_collected() {
    let words = [
        0xf420_0202,
        0xfa00_0000,
        0xe070_1000,
        0x8002_0100,
        0xf080_0f08,
        0x0082_0004,
        0xf800_0817,
        0x0002_0100,
        0xf800_08cf,
        0x0706_0504,
        S_ENDPGM,
    ];
    let a = build(
        &words,
        TargetIsa::Gfx1013,
        ShaderStage::Vertex,
        32,
        ShaderInterface::default(),
    )
    .unwrap();
    let summary: Vec<_> = a
        .bindings
        .iter()
        .map(|b| (b.kind, b.base_sgpr, b.dword_count, b.writable))
        .collect();
    assert_eq!(
        summary,
        vec![
            (ResourceKind::BufferDescriptor, 4, 4, false),
            (ResourceKind::BufferDescriptor, 8, 4, true),
            (ResourceKind::ImageDescriptor, 8, 8, false),
            (ResourceKind::SamplerDescriptor, 16, 4, false),
        ]
    );
    assert_eq!(a.exports.color_mask, 0x70);
    assert_eq!(a.exports.position_mask, 1);
    assert_eq!(a.exports.param_mask, 0);
}

#[test]
fn scratch_and_thread_ids_set_rsrc2_fields() {
    let scratch = compute(&[0xdc30_4000, 0x017d_0000, S_ENDPGM], TargetIsa::Gfx103);
    assert_ne!(scratch.pgm_rsrc2 & RSRC2_SCRATCH_EN, 0);

    let tid = compute(&[0x4a06_0501, S_ENDPGM], TargetIsa::Gfx11);
    assert_eq!((tid.pgm_rsrc2 & RSRC2_TIDIG_MASK) >> RSRC2_TIDIG_SHIFT, 2);
    assert_eq!(tid.pgm_rsrc2 & RSRC2_SCRATCH_EN, 0);
}
