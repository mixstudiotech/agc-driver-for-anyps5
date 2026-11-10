use super::hazards::{DEPCTR_SA_SDST_0, DEPCTR_VA_SDST_0, DEPCTR_VA_VDST_0, DEPCTR_VM_VSRC_0};
use super::*;
use crate::decode::Decoder;
use crate::ir::{LdsDir, MemKind};
use crate::lowering::{self, derive, LoweredBlock};
use crate::{Compiler, PrecisionMode, ShaderArtifact, ShaderStage};

const S_ENDPGM: u32 = 0xbf81_0000;
const GFX11_ENDPGM: u32 = 0xbfb0_0000;
const SDWA_ADD: [u32; 2] = [0x0600_04f9, 0x0204_1501];

fn depctr(simm16: u16) -> u32 {
    0xbf88_0000 | u32::from(simm16)
}

fn compile(words: &[u32], target: TargetIsa, stage: ShaderStage) -> ShaderArtifact {
    compile_wave(words, target, stage, 32)
}

fn compile_wave(words: &[u32], target: TargetIsa, stage: ShaderStage, wave: u8) -> ShaderArtifact {
    Compiler::new(target)
        .compile(words, CompileOptions::new(target, stage, wave))
        .expect("compile")
}

fn translate(words: &[u32], target: TargetIsa) -> Vec<u32> {
    translate_wave(words, target, 32)
}

fn translate_wave(words: &[u32], target: TargetIsa, wave: u8) -> Vec<u32> {
    let mut program = words.to_vec();
    program.push(S_ENDPGM);
    let artifact = compile_wave(&program, target, ShaderStage::Fragment, wave);
    let code = artifact.program();
    let endpgm = if target.is_gfx11_plus() {
        GFX11_ENDPGM
    } else {
        S_ENDPGM
    };
    assert_eq!(code.last(), Some(&endpgm));
    code[..code.len() - 1].to_vec()
}

const CORPUS: &[u32] = &[
    0xbe80_0380,
    0x8002_ff00,
    0x1234_5678,
    0xbefc_03c1,
    0x7e02_02f2,
    0x0600_04f9,
    0x0204_1501,
    0x7e00_02fa,
    0xff00_b101,
    0xd54b_8200,
    0x240e_0501,
    0xd70f_6a00,
    0x0002_0501,
    0xd503_0000,
    0x0002_02ff,
    0x4049_0fdb,
    0x7d84_0280,
    0xf408_0100,
    0xfa00_0010,
    0xf420_0202,
    0xfa00_0000,
    0xe030_1000,
    0x8002_0100,
    0xea6b_2000,
    0x8002_0400,
    0xdc30_8000,
    0x017d_0002,
    0xdc70_8ff8,
    0x0004_0302,
    0xd8dc_0201,
    0x0000_0002,
    0xf080_0f0a,
    0x0082_0004,
    0x0000_0007,
    0xc808_0500,
    0xbf8c_0070,
    0xf800_180f,
    0x0302_0100,
    S_ENDPGM,
];

#[test]
fn gfx1013_reencoding_round_trips_every_family() {
    let artifact = compile(CORPUS, TargetIsa::Gfx1013, ShaderStage::Compute);
    assert_eq!(artifact.program(), CORPUS);
    assert_eq!(artifact.guest_words, CORPUS.len());
}

#[test]
fn gfx1013_reencodes_bvh_intersections_unchanged() {
    let program = [
        0xf198_9f01,
        0x0001_0004,
        0xf19c_9f01,
        0x4001_0004,
        0xf198_9f07,
        0x0001_0004,
        0x0c0b_0a09,
        0x100f_0e0d,
        0x0000_1411,
        0xf19c_9f05,
        0x4001_0004,
        0x0b0a_0905,
        0x0f0e_0d0c,
        S_ENDPGM,
    ];
    let artifact = compile(&program, TargetIsa::Gfx1013, ShaderStage::Compute);
    assert_eq!(artifact.program(), program);
}

#[test]
fn code_is_padded_with_code_end_for_prefetch() {
    for (target, line) in [(TargetIsa::Gfx1013, 16), (TargetIsa::Gfx11, 32)] {
        let artifact = compile(CORPUS, target, ShaderStage::Fragment);
        let program = artifact.program_words;
        assert_eq!(
            artifact.code.len(),
            program.next_multiple_of(line) + 3 * line
        );
        assert!(artifact.code[program..].iter().all(|&w| w == S_CODE_END));
    }
}

#[test]
fn gfx11_scalar_and_vector_encodings() {
    assert_eq!(
        translate(&[0xbefc_03c1], TargetIsa::Gfx11),
        vec![0xbefd_00c1]
    );
    assert_eq!(
        translate(&[0xbf8c_0070], TargetIsa::Gfx11),
        vec![0xbf89_0007]
    );
    assert_eq!(
        translate(&[0x3400_0284], TargetIsa::Gfx11),
        vec![0x3000_0284]
    );
    assert_eq!(
        translate(&[0x7d84_0280], TargetIsa::Gfx11),
        vec![0x7c94_0280]
    );
}

#[test]
fn gfx11_memory_and_export_encodings() {
    let cases: [(&[u32], &[u32]); 6] = [
        (&[0xf408_0100, 0xfa00_0010], &[0xf408_0100, 0xf800_0010]),
        (&[0xe030_1000, 0x8002_0100], &[0xe050_0000, 0x8042_0100]),
        (&[0xdc30_8000, 0x017d_0002], &[0xdc52_0000, 0x017c_0002]),
        (&[0xd8dc_0201, 0x0000_0002], &[0xd8dc_0201, 0x0000_0002]),
        (&[0xf080_0f08, 0x0082_0004], &[0xf06c_0f04, 0x1002_0004]),
        (&[0xf800_180f, 0x0302_0100], &[0xf800_080f, 0x0302_0100]),
    ];
    for (input, expected) in cases {
        assert_eq!(translate(input, TargetIsa::Gfx11), expected, "{input:08x?}");
    }
}

#[test]
fn gfx12_memory_encodings_and_split_waits() {
    assert_eq!(
        translate(&[0xe030_1000, 0x8002_0100], TargetIsa::Gfx12),
        vec![0xc405_007c, 0x4000_1001, 0x0000_0000]
    );
    assert_eq!(
        translate(&[0xf420_0202, 0xfa00_0000], TargetIsa::Gfx12),
        vec![0xf402_0202, 0xf800_0000]
    );
    assert!(translate(&[0xbf8c_0070], TargetIsa::Gfx12).is_empty());
}

#[test]
fn forward_branches_are_relocated_after_expansion() {
    let words = [0xbf06_8400, 0xbf85_0002, SDWA_ADD[0], SDWA_ADD[1], S_ENDPGM];
    let gfx11 = compile(&words, TargetIsa::Gfx11, ShaderStage::Fragment);
    let code = gfx11.program();
    assert_eq!(code.len(), 13);
    assert_eq!(code[1], 0xbfa2_000a);
    assert_eq!(code[12], GFX11_ENDPGM);
    let gfx103 = compile(&words, TargetIsa::Gfx103, ShaderStage::Fragment);
    assert_eq!(gfx103.program(), words);
}

#[test]
fn backward_branches_are_relocated_after_expansion() {
    let words = [
        0xbe80_0380,
        SDWA_ADD[0],
        SDWA_ADD[1],
        0x8000_8100,
        0xbf06_8400,
        0xbf84_fffb,
        S_ENDPGM,
    ];
    let gfx11 = compile(&words, TargetIsa::Gfx11, ShaderStage::Fragment);
    assert_eq!(gfx11.program()[13], 0xbfa1_fff3);
    let gfx1013 = compile(&words, TargetIsa::Gfx1013, ShaderStage::Fragment);
    assert_eq!(gfx1013.program(), words);
}

#[test]
fn gfx12_sample_unorm_is_bit_13() {
    let unorm_sample = [0xf080_1f08, 0x0082_0402];
    let code = translate(&unorm_sample, TargetIsa::Gfx12);
    let word0 = code
        .iter()
        .copied()
        .find(|w| w >> 26 == 0x39)
        .expect("VSAMPLE");
    assert_eq!(word0 & (1 << 13), 1 << 13, "{word0:#010x}");
    assert_eq!(word0 & (1 << 7), 0, "{word0:#010x}");
}

#[test]
fn vgprs_a_pending_load_writes_are_not_overwritten_early_on_gfx11() {
    let load_v1 = [0xe030_1000, 0x8001_0100]; // buffer_load_dword v1, v0, s[4:7], 0 offen
    let load_v2 = [0xe030_1000, 0x8001_0200];
    let mov_v1 = 0x7e02_0280; // v_mov_b32 v1, 0
    let wait = |vm: u32| 0xbf89_0000 | (vm << 10) | 0x3f7;
    let waits = |code: &[u32]| {
        code.iter()
            .filter(|&&w| w & 0xffff_0000 == 0xbf89_0000)
            .copied()
            .collect::<Vec<u32>>()
    };
    let program = [load_v1[0], load_v1[1], mov_v1];
    assert_eq!(
        waits(&translate(&program, TargetIsa::Gfx115)),
        vec![wait(0)]
    );
    assert!(waits(&translate(&program, TargetIsa::Gfx103)).is_empty());
    let younger = [load_v1[0], load_v1[1], load_v2[0], load_v2[1], mov_v1];
    assert_eq!(waits(&translate(&younger, TargetIsa::Gfx11)), vec![wait(1)]);
    // The guest's own wait (s_waitcnt vmcnt(0)) leaves nothing to add.
    let waited = [load_v1[0], load_v1[1], 0xbf8c_3f70, mov_v1];
    assert_eq!(waits(&translate(&waited, TargetIsa::Gfx115)).len(), 1);
    // Across blocks: s_cbranch_scc0 over a write to v2 into a write to v1.
    let branchy = [load_v1[0], load_v1[1], 0xbf84_0001, 0x7e04_0280, mov_v1];
    let code = translate(&branchy, TargetIsa::Gfx115);
    let at = code.iter().position(|&w| w == mov_v1).unwrap();
    assert_eq!(code[at - 1], wait(0), "{code:08x?}");
}

#[test]
fn image_operations_without_a_sampler_encode_no_sampler_register() {
    // image_load v[4:7], v[1:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D with a stray SSAMP of 3.
    let load = [0xf000_0f08, 0x0062_0401];
    let gfx1030 = translate(&load, TargetIsa::Gfx103);
    assert_eq!(gfx1030[1], 0x0002_0401, "{gfx1030:08x?}");
    let gfx1151 = translate(&load, TargetIsa::Gfx115);
    let at = gfx1151.iter().position(|w| w >> 26 == 0x3c).expect("MIMG");
    assert_eq!(gfx1151[at + 1] >> 26, 0, "{gfx1151:08x?}");
}

#[test]
fn gfx12_msaa_loads_are_vsample_and_wait_on_the_sample_counter() {
    // image_msaa_load v[4:7], v[2:4], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA;
    // s_waitcnt vmcnt(0); v_mov_b32 v8, v4.
    let program = [
        0xf000_0131,
        0x0002_0402,
        0xbf8c_3f70,
        0x7e10_0304,
        0xbf81_0000,
    ];
    let code = translate(&program, TargetIsa::Gfx12);
    // llvm-mc -mcpu=gfx1200: image_msaa_load v[4:7], [v2, v3, v4], s[8:15] dmask:0x1
    // dim:SQ_RSRC_IMG_2D_MSAA.
    let at = code
        .iter()
        .position(|&w| w == 0xe446_0006)
        .expect("VSAMPLE");
    assert_eq!(code[at + 1..at + 3], [0x0000_1004, 0x0004_0302]);
    let s_wait_samplecnt_0 = 0xbfc2_0000;
    let s_wait_loadcnt_0 = 0xbfc0_0000;
    assert!(code.contains(&s_wait_samplecnt_0), "{code:08x?}");
    assert!(!code.contains(&s_wait_loadcnt_0), "{code:08x?}");
}

#[test]
fn gfx12_bvh_addresses_are_operand_groups() {
    let decode = |words: &[u32]| Decoder::new().decode_at(words, 0).unwrap();
    // image_bvh_intersect_ray v[4:7], v[8:18], s[20:23] (llvm-mc -mcpu=gfx1200 encodes the
    // address as [v8, v9, v[10:12], v[13:15], v[16:18]]).
    let bvh = lowering::retarget(&decode(&[0xf198_9f01, 0x0005_0408]), 0x19);
    assert_eq!(
        encode_instruction(&bvh, TargetIsa::Gfx12).unwrap(),
        vec![0xd3c6_4010, 0x1000_2804, 0x0d0a_0908]
    );
    // image_bvh64_intersect_ray v[4:7], v[8:19]: [v[8:9], v10, v[11:13], v[14:16], v[17:19]].
    let bvh64 = lowering::retarget(&decode(&[0xf19c_9f01, 0x0005_0408]), 0x1a);
    let words = encode_instruction(&bvh64, TargetIsa::Gfx12).unwrap();
    assert_eq!((words[1] >> 24, words[2]), (17, 0x0e0b_0a08));
    let mut split = bvh.clone();
    if let Fields::Mimg(m) = &mut split.fields {
        m.nsa = true;
        m.addr = [8, 9, 10, 11, 30, 13, 14, 15, 16, 17, 18, 0, 0];
    }
    assert!(
        encode_instruction(&split, TargetIsa::Gfx12).is_err(),
        "origin is not consecutive"
    );
}

const V_RCP_F32_V1_V0: u32 = 0x7e02_5500;
const V_MUL_F32_V2_V1_V3: u32 = 0x1004_0701;
const V_MOV_B32_V4_1F: u32 = 0x7e08_02f2;

#[test]
fn transcendental_use_waits_only_on_gfx11_0() {
    let words = [V_RCP_F32_V1_V0, V_MUL_F32_V2_V1_V3];
    let waited = vec![
        V_RCP_F32_V1_V0,
        depctr(DEPCTR_VA_VDST_0),
        V_MUL_F32_V2_V1_V3,
    ];
    assert_eq!(translate(&words, TargetIsa::Gfx11), waited);
    for target in [TargetIsa::Gfx115, TargetIsa::Gfx12] {
        assert!(
            !translate(&words, target).contains(&depctr(DEPCTR_VA_VDST_0)),
            "{target:?}"
        );
    }
    assert_eq!(translate(&words, TargetIsa::Gfx103), words);
    for (between, waits) in [(5, true), (6, false)] {
        let mut spaced = vec![V_RCP_F32_V1_V0];
        spaced.extend(std::iter::repeat_n(V_MOV_B32_V4_1F, between));
        spaced.push(V_MUL_F32_V2_V1_V3);
        let code = translate(&spaced, TargetIsa::Gfx11);
        assert_eq!(
            code.contains(&depctr(DEPCTR_VA_VDST_0)),
            waits,
            "{between} VALUs"
        );
    }
    let two_trans = [
        V_RCP_F32_V1_V0,
        0x7e08_5500,
        0x7e0a_5500,
        V_MUL_F32_V2_V1_V3,
    ];
    assert!(!translate(&two_trans, TargetIsa::Gfx11).contains(&depctr(DEPCTR_VA_VDST_0)));
}

#[test]
fn exec_writes_need_no_wait_in_wave32() {
    let words = [0xbefe_0302, V_MOV_B32_V4_1F, 0x0a0c_0b04];
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115, TargetIsa::Gfx12] {
        let code = translate(&words, target);
        assert!(
            !code.iter().any(|w| w >> 16 == 0xbf88),
            "{target:?}: {code:08x?}"
        );
    }
}

#[test]
fn partial_forwarding_waits_in_wave64_on_gfx11() {
    let words = [0x7e08_0302, 0xbefe_0402, 0x7e0a_0303, 0x060c_0b04];
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115] {
        let code = translate_wave(&words, target, 64);
        let add = code.iter().position(|&w| w == 0x060c_0b04).unwrap();
        assert_eq!(
            code[add - 1],
            depctr(DEPCTR_VA_VDST_0),
            "{target:?}: {code:08x?}"
        );
        assert!(!translate_wave(&words, target, 32).contains(&depctr(DEPCTR_VA_VDST_0)));
    }
    assert!(!translate_wave(&words, TargetIsa::Gfx12, 64).contains(&depctr(DEPCTR_VA_VDST_0)));
}

#[test]
fn lane_mask_overwrites_wait_after_the_write_in_wave64() {
    let words = [0x0208_0702, 0xbeea_0404, 0x7e08_02f2];
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115] {
        let code = translate_wave(&words, target, 64);
        let write = code.iter().position(|&w| w == 0xbeea_0104).unwrap();
        assert_eq!(
            code[write + 1],
            depctr(DEPCTR_SA_SDST_0),
            "{target:?}: {code:08x?}"
        );
        assert!(!translate_wave(&words, target, 32).contains(&depctr(DEPCTR_SA_SDST_0)));
    }
    let mitigated = [0x0208_0702, 0x060a_0a00, 0xbeea_0404];
    assert!(!translate_wave(&mitigated, TargetIsa::Gfx11, 64).contains(&depctr(DEPCTR_SA_SDST_0)));
}

#[test]
fn cmpx_before_permlane_gets_a_vector_move() {
    let words = [0x7da4_0300, 0xd777_0002, 0x0010_0703];
    let mov_v3 = 0x7e06_0303;
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115, TargetIsa::Gfx12] {
        let code = translate(&words, target);
        let permlane = code.iter().position(|&w| w == 0xd65b_0002).unwrap();
        assert_eq!(code[permlane - 1], mov_v3, "{target:?}: {code:08x?}");
    }
    assert_eq!(translate(&words, TargetIsa::Gfx103), words);
    let separated = [0x7da4_0300, V_MOV_B32_V4_1F, 0xd777_0002, 0x0010_0703];
    assert!(!translate(&separated, TargetIsa::Gfx11).contains(&mov_v3));
}

#[test]
fn lds_parameter_loads_wait_for_memory_reading_their_vgpr_on_gfx11() {
    let origin = crate::decode::Decoder::new()
        .decode_at(&[0xe030_1000, 0x8002_0500], 0)
        .unwrap();
    let load = origin.clone();
    let Fields::Mubuf(b) = load.fields else {
        panic!("expected MUBUF");
    };
    assert_eq!((b.kind, b.vdata), (MemKind::Load, 5));
    let lds = |vdst| {
        let fields = Fields::Ldsdir(LdsDir {
            vdst,
            attr: 0,
            chan: 0,
            wait_vdst: 0,
        });
        derive(&origin, F::Ldsdir, 0, 0, fields)
    };
    for (target, waits) in [
        (TargetIsa::Gfx11, true),
        (TargetIsa::Gfx115, true),
        (TargetIsa::Gfx12, false),
    ] {
        for (vdst, hazard) in [(5, true), (6, false)] {
            let mut program = LoweredProgram {
                target,
                stage: ShaderStage::Fragment,
                wave64: false,
                blocks: vec![LoweredBlock {
                    guest_pc: 0,
                    instructions: vec![load.clone(), lds(vdst)],
                    dynamic_entry: false,
                }],
                dispatch: None,
                preconverted: Vec::new(),
                attribute_ring: None,
            };
            hazards::insert(&mut program);
            let waited = program.blocks[0].instructions.iter().any(|i| {
                i.fields
                    == Fields::Sopp {
                        simm16: DEPCTR_VM_VSRC_0,
                    }
                    && i.family == F::Sopp
            });
            assert_eq!(waited, waits && hazard, "{target:?} v{vdst}");
        }
    }
}

#[test]
fn gfx11_5_graphics_lower_priority_after_exports() {
    let setprio = |p: u32| 0xbfb5_0000 | p;
    let nop = 0xbf80_0000;
    let exp = [0xf800_080f, 0x0302_0100];
    let at_end = translate(&exp, TargetIsa::Gfx115);
    assert_eq!(
        at_end,
        vec![setprio(2), 0xf800_080f, 0x0302_0100, setprio(0), nop, nop]
    );
    let mut mid = exp.to_vec();
    mid.push(V_MOV_B32_V4_1F);
    let code = translate(&mid, TargetIsa::Gfx115);
    assert_eq!(
        code,
        vec![
            setprio(2),
            0xf800_080f,
            0x0302_0100,
            setprio(0),
            0xbd7c_0000,
            nop,
            nop,
            setprio(2),
            V_MOV_B32_V4_1F
        ]
    );
    assert_eq!(translate(&exp, TargetIsa::Gfx11), exp);
    let guest_low_priority = [0xbf8f_0001, exp[0], exp[1]];
    assert_eq!(
        translate(&guest_low_priority, TargetIsa::Gfx115),
        vec![setprio(3), 0xf800_080f, 0x0302_0100, setprio(0), nop, nop]
    );
    let compute = compile(
        &[exp[0], exp[1], S_ENDPGM],
        TargetIsa::Gfx115,
        ShaderStage::Compute,
    );
    assert!(!compute.program().contains(&setprio(2)));
}

#[test]
fn gfx12_waits_for_sgpr_writes_once_a_valu_has_read_them() {
    let read_s4 = 0x0600_0204;
    let salu_write_s4 = 0xbe84_0381;
    let code = translate(&[read_s4, salu_write_s4, read_s4], TargetIsa::Gfx12);
    assert_eq!(
        code,
        vec![read_s4, 0xbe84_0081, depctr(DEPCTR_SA_SDST_0), read_s4]
    );
    let untracked = translate(&[salu_write_s4, read_s4], TargetIsa::Gfx12);
    assert!(
        !untracked.iter().any(|w| w >> 16 == 0xbf88),
        "{untracked:08x?}"
    );
    let valu_write_s4 = [0xd401_0004, 0x0002_0300];
    let cndmask_s4 = [0xd501_0002, 0x0012_0300];
    let mut words = vec![read_s4];
    words.extend(valu_write_s4);
    words.extend(cndmask_s4);
    let code = translate(&words, TargetIsa::Gfx12);
    let cndmask = code.iter().position(|&w| w == 0xd501_0002).unwrap();
    assert_eq!(code[cndmask - 1], depctr(DEPCTR_VA_SDST_0), "{code:08x?}");
}

#[test]
fn exact_precision_is_the_default_and_changes_the_artifact() {
    let words = [0xd541_0000, 0x040e_0501, S_ENDPGM];
    let options = CompileOptions::new(TargetIsa::Gfx11, ShaderStage::Fragment, 32);
    let fused = Compiler::new(TargetIsa::Gfx11)
        .with_precision(PrecisionMode::Fused)
        .compile(&words, options)
        .unwrap();
    let exact = Compiler::new(TargetIsa::Gfx11)
        .compile(&words, options)
        .unwrap();
    assert_eq!(exact.precision, PrecisionMode::Exact);
    assert_eq!(fused.program().len(), 3);
    assert_eq!(exact.program().len(), 5);
    assert_eq!(exact.num_vgprs, fused.num_vgprs + 1);
}
