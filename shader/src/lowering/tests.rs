use super::waitcnt::{GFX12_WAIT_KMCNT, GFX12_WAIT_LOADCNT};
use super::*;
use crate::ir::{LdsDir, VInterp, VopExt};
use crate::{Compiler, ShaderInterface};

const S_ENDPGM: u32 = 0xbf81_0000;
const V_MAC_F32: u32 = 0x3e00_0501;
const V_MAD_F32: [u32; 2] = [0xd541_0000, 0x040e_0501];
const V_ADD_F32_SDWA: [u32; 2] = [0x0600_04f9, 0x0204_1501];

fn lower_with(
    words: &[u32],
    target: TargetIsa,
    precision: PrecisionMode,
    wave_size: u8,
) -> Result<LoweredProgram, CompileError> {
    lower_stage(words, target, precision, wave_size, ShaderStage::Fragment)
}

fn lower_stage(
    words: &[u32],
    target: TargetIsa,
    precision: PrecisionMode,
    wave_size: u8,
    stage: ShaderStage,
) -> Result<LoweredProgram, CompileError> {
    let options = CompileOptions::new(target, stage, wave_size);
    let interface = ShaderInterface::default();
    let ir = Compiler::new(target).analyze(words, options, &interface)?;
    let mut regs = RegisterAllocator::new(&ir.metadata, options, &interface);
    Lowerer::new(options, precision).lower(&ir, &mut regs)
}

fn lower_compute(
    words: &[u32],
    target: TargetIsa,
    interface: ShaderInterface,
) -> Result<LoweredProgram, CompileError> {
    let options = CompileOptions::new(target, ShaderStage::Compute, 32);
    let ir = Compiler::new(target).analyze(words, options, &interface)?;
    let mut regs = RegisterAllocator::new(&ir.metadata, options, &interface);
    Lowerer::new(options, PrecisionMode::Exact).lower(&ir, &mut regs)
}

const LUT_ADDRESS: u64 = 0x7_0000_0000;

fn lower_with_descriptors(
    words: &[u32],
    target: TargetIsa,
    stage: ShaderStage,
    preconvert: bool,
) -> Result<LoweredProgram, CompileError> {
    let options = CompileOptions::new(target, stage, 32);
    let interface = ShaderInterface {
        descriptor_lut: Some(LUT_ADDRESS),
        preconvert_user_descriptors: preconvert,
        ..ShaderInterface::default()
    };
    let ir = Compiler::new(target).analyze(words, options, &interface)?;
    let mut regs = RegisterAllocator::new(&ir.metadata, options, &interface);
    Lowerer::new(options, PrecisionMode::Exact)
        .with_descriptor_conversion(interface.descriptor_lut, preconvert)
        .lower(&ir, &mut regs)
}

/// Lowers a pixel shader with `interface` the way [`Compiler::compile_with_interface`] does.
fn lower_pixel(
    words: &[u32],
    target: TargetIsa,
    wave_size: u8,
    interface: ShaderInterface,
) -> Result<LoweredProgram, CompileError> {
    let options = CompileOptions::new(target, ShaderStage::Fragment, wave_size);
    let ir = Compiler::new(target).analyze(words, options, &interface)?;
    let mut regs = RegisterAllocator::new(&ir.metadata, options, &interface);
    let known = interface.user_sgpr_count.is_some() || interface.guest_pgm_rsrc2.is_some();
    Lowerer::new(options, PrecisionMode::Exact)
        .with_pixel_interface(
            known.then_some(ir.metadata.user_sgpr_count),
            interface.ps_col_format,
            interface.ps_z_format,
            interface.dual_source_blend,
        )
        .lower(&ir, &mut regs)
}

fn lower(words: &[u32], target: TargetIsa) -> Vec<Instruction> {
    lower_with(words, target, PrecisionMode::Fused, 32)
        .expect("lowering")
        .instructions()
        .cloned()
        .collect()
}

fn summary(insts: &[Instruction]) -> Vec<(F, u16)> {
    insts.iter().map(|i| (i.family, i.opcode)).collect()
}

fn vop3(inst: &Instruction) -> Vop3 {
    match inst.fields {
        Fields::Vop3(v) => v,
        ref other => panic!("expected VOP3, got {other:?}"),
    }
}

#[test]
fn gfx1013_lowering_is_identity() {
    let insts = lower(&[V_MAC_F32, S_ENDPGM], TargetIsa::Gfx1013);
    assert_eq!(summary(&insts), vec![(F::Vop2, 0x1f), (F::Sopp, 0x01)]);
}

#[test]
fn mac_becomes_fmac_on_every_newer_target() {
    for (target, endpgm) in [
        (TargetIsa::Gfx103, 0x01),
        (TargetIsa::Gfx11, 0x30),
        (TargetIsa::Gfx12, 0x30),
    ] {
        let insts = lower(&[V_MAC_F32, S_ENDPGM], target);
        assert_eq!(
            summary(&insts),
            vec![(F::Vop2, 0x2b), (F::Sopp, endpgm)],
            "{target:?}"
        );
        assert!(matches!(
            insts[0].fields,
            Fields::Vop2 {
                vdst: Operand::Vgpr(0),
                ..
            }
        ));
    }
}

#[test]
fn mad_becomes_fma_with_modifiers_preserved() {
    let words = [0xd541_8200, 0x240e_0501, S_ENDPGM];
    for (target, fma) in [
        (TargetIsa::Gfx103, 0x14b),
        (TargetIsa::Gfx11, 0x213),
        (TargetIsa::Gfx12, 0x213),
    ] {
        let insts = lower(&words, target);
        assert_eq!((insts[0].family, insts[0].opcode), (F::Vop3, fma));
        let v = vop3(&insts[0]);
        assert_eq!((v.abs, v.neg, v.clamp), (0b010, 0b001, true));
        assert_eq!(
            v.src,
            [Operand::Vgpr(1), Operand::Vgpr(2), Operand::Vgpr(3)]
        );
    }
}

#[test]
fn legacy_mad_is_unfused_on_every_target() {
    let mad_legacy = [0xd540_0200, 0x840e_0501, S_ENDPGM];
    let mac_legacy = [0x0c00_0501, S_ENDPGM];
    for target in [
        TargetIsa::Gfx103,
        TargetIsa::Gfx11,
        TargetIsa::Gfx115,
        TargetIsa::Gfx12,
    ] {
        let insts = lower(&mad_legacy, target);
        assert_eq!(
            summary(&insts[..2]),
            vec![(F::Vop3, 0x107), (F::Vop3, 0x103)],
            "{target:?}"
        );
        let (mul, add) = (vop3(&insts[0]), vop3(&insts[1]));
        assert_eq!(&mul.src[..2], &[Operand::Vgpr(1), Operand::Vgpr(2)]);
        assert_eq!((mul.abs, mul.neg, mul.clamp), (0b010, 0b000, false));
        assert_eq!((add.vdst, add.src[1]), (Operand::Vgpr(0), Operand::Vgpr(3)));
        assert_eq!((add.abs, add.neg, add.clamp), (0, 0b010, false));

        let insts = lower(&mac_legacy, target);
        assert_eq!(
            summary(&insts[..2]),
            vec![(F::Vop3, 0x107), (F::Vop3, 0x103)]
        );
        assert_eq!(vop3(&insts[1]).src[1], Operand::Vgpr(0));
    }
    assert_eq!(lower(&mad_legacy, TargetIsa::Gfx1013)[0].opcode, 0x140);
}

#[test]
fn legacy_mad_flushes_f32_denormals_when_the_mode_keeps_them() {
    let words = [0xd540_0000, 0x040e_0501, S_ENDPGM];
    let options = CompileOptions::new(TargetIsa::Gfx11, ShaderStage::Fragment, 32);
    let interface = ShaderInterface {
        guest_pgm_rsrc1: Some(0xf0 << 12),
        ..ShaderInterface::default()
    };
    let ir = Compiler::new(TargetIsa::Gfx11)
        .analyze(&words, options, &interface)
        .unwrap();
    let mut regs = RegisterAllocator::new(&ir.metadata, options, &interface);
    let program = Lowerer::new(options, PrecisionMode::Fused)
        .lower(&ir, &mut regs)
        .unwrap();
    let insts: Vec<_> = program.instructions().cloned().collect();
    assert_eq!(
        summary(&insts[..4]),
        vec![
            (F::Sopp, 0x12),
            (F::Vop3, 0x107),
            (F::Vop3, 0x103),
            (F::Sopp, 0x12)
        ]
    );
    assert_eq!(insts[0].fields, Fields::Sopp { simm16: 0xc });
    assert_eq!(insts[3].fields, Fields::Sopp { simm16: 0xf });
}

#[test]
fn exact_precision_splits_mad_into_mul_and_add() {
    let words = [V_MAD_F32[0], V_MAD_F32[1], S_ENDPGM];
    let program = lower_with(&words, TargetIsa::Gfx11, PrecisionMode::Exact, 32).unwrap();
    let insts: Vec<_> = program.instructions().cloned().collect();
    assert_eq!(
        summary(&insts[..2]),
        vec![(F::Vop3, 0x108), (F::Vop3, 0x103)]
    );
    let (mul, add) = (vop3(&insts[0]), vop3(&insts[1]));
    assert_eq!(mul.vdst, Operand::Vgpr(4));
    assert_eq!(&mul.src[..2], &[Operand::Vgpr(1), Operand::Vgpr(2)]);
    assert_eq!(add.vdst, Operand::Vgpr(0));
    assert_eq!(&add.src[..2], &[Operand::Vgpr(4), Operand::Vgpr(3)]);
}

#[test]
fn sdwa_expands_to_bitfield_sequences_on_gfx11() {
    let insts = lower(
        &[V_ADD_F32_SDWA[0], V_ADD_F32_SDWA[1], S_ENDPGM],
        TargetIsa::Gfx11,
    );
    assert_eq!(
        summary(&insts),
        vec![
            (F::Vop3, 0x210),
            (F::Vop3, 0x210),
            (F::Vop3, 0x103),
            (F::Vop2, 0x18),
            (F::Vop3, 0x212),
            (F::Sopp, 0x30),
        ]
    );
    let lo = vop3(&insts[0]);
    assert_eq!(lo.vdst, Operand::Vgpr(3));
    assert_eq!(
        lo.src,
        [
            Operand::Vgpr(1),
            Operand::InlineInt(0),
            Operand::InlineInt(16)
        ]
    );
    let byte = vop3(&insts[1]);
    assert_eq!(
        byte.src,
        [
            Operand::Vgpr(2),
            Operand::InlineInt(16),
            Operand::InlineInt(8)
        ]
    );
    let add = vop3(&insts[2]);
    assert_eq!(
        (add.vdst, add.src[0], add.src[1]),
        (Operand::Vgpr(5), Operand::Vgpr(3), Operand::Vgpr(4))
    );
    let insert = vop3(&insts[4]);
    assert_eq!(
        insert.src,
        [
            Operand::Literal(0xffff_0000),
            Operand::Vgpr(5),
            Operand::Vgpr(0)
        ]
    );
    assert_eq!(insert.vdst, Operand::Vgpr(0));
}

#[test]
fn sdwa_is_kept_on_rdna2() {
    let insts = lower(
        &[V_ADD_F32_SDWA[0], V_ADD_F32_SDWA[1], S_ENDPGM],
        TargetIsa::Gfx103,
    );
    assert!(matches!(
        insts[0].fields,
        Fields::Vop2 {
            ext: VopExt::Sdwa(_),
            ..
        }
    ));
}

#[test]
fn waitcnt_is_repacked_for_gfx11() {
    let insts = lower(&[0xbf8c_0070, S_ENDPGM], TargetIsa::Gfx11);
    assert_eq!(insts[0].opcode, SOPP_GFX11_WAITCNT);
    assert_eq!(insts[0].fields, Fields::Sopp { simm16: 0x0007 });
}

#[test]
fn waitcnt_splits_into_gfx12_counters() {
    let words = [
        0xf400_0100,
        0xfa00_0000,
        0xe030_1000,
        0x8002_0100,
        0xbf8c_0070,
        S_ENDPGM,
    ];
    let insts = lower(&words, TargetIsa::Gfx12);
    let waits: Vec<_> = insts
        .iter()
        .filter(|i| i.family == F::Sopp && i.opcode != 0x30)
        .map(|i| (i.opcode, i.fields))
        .collect();
    assert_eq!(
        waits,
        vec![
            (GFX12_WAIT_LOADCNT, Fields::Sopp { simm16: 0 }),
            (GFX12_WAIT_KMCNT, Fields::Sopp { simm16: 0 }),
        ]
    );
}

#[test]
fn compares_are_renumbered_and_keep_exec_semantics() {
    let insts = lower(&[0x7c24_0280, S_ENDPGM], TargetIsa::Gfx11);
    assert_eq!(insts[0].opcode, 0x92);
    assert!(insts[0].has_flag(flag::CMPX | flag::WRITES_EXEC));
    let vop3_cmpx = lower(&[0xd412_007e, 0x0002_0280, S_ENDPGM], TargetIsa::Gfx11);
    assert_eq!((vop3_cmpx[0].family, vop3_cmpx[0].opcode), (F::Vop3, 0x92));
    assert_eq!(
        lower(&[0x7d84_0280, S_ENDPGM], TargetIsa::Gfx11)[0].opcode,
        0x4a
    );
    assert_eq!(
        lower(&[0x7d84_0280, S_ENDPGM], TargetIsa::Gfx103)[0].opcode,
        0xc2
    );
}

#[test]
fn gfx12_replaces_constant_compares() {
    let never = lower(&[0x7c00_0300, S_ENDPGM], TargetIsa::Gfx12);
    assert_eq!(
        (never[0].family, never[0].fields),
        (
            F::Sop1,
            Fields::Sop1 {
                sdst: Operand::VccLo,
                src0: Operand::InlineInt(0)
            }
        )
    );
    let always = lower_with(
        &[0x7d0e_0300, S_ENDPGM],
        TargetIsa::Gfx12,
        PrecisionMode::Fused,
        64,
    )
    .unwrap();
    let first = always.instructions().next().unwrap();
    assert_eq!(first.opcode, 0x01);
    assert_eq!(
        first.fields,
        Fields::Sop1 {
            sdst: Operand::VccLo,
            src0: Operand::ExecLo
        }
    );
    let noop = lower(&[0x7d2e_0300, S_ENDPGM], TargetIsa::Gfx12);
    assert_eq!(summary(&noop), vec![(F::Sopp, 0x30)]);
}

/// `s_mov_b32 m0, s2; v_interp_p1_f32 v2, v0, attr1.y; v_interp_p2_f32 v2, v1, attr1.y`.
const INTERP_P1_P2: [u32; 4] = [0xbefc_0302, 0xc808_0500, 0xc809_0501, S_ENDPGM];

fn sop1(inst: &Instruction) -> (Operand, Operand) {
    match inst.fields {
        Fields::Sop1 { sdst, src0 } => (sdst, src0),
        ref other => panic!("expected SOP1, got {other:?}"),
    }
}

#[test]
fn interpolation_loads_parameters_with_every_lane_enabled_on_gfx11() {
    // Without a known user SGPR count the prim_mask SGPR is a guess, so nothing is hoisted:
    // each load enables every lane around itself and VINTERP waits for it.
    let insts = lower(&INTERP_P1_P2, TargetIsa::Gfx11);
    assert_eq!(
        summary(&insts),
        vec![
            (F::Sop1, 0x00),
            (F::Sop1, 0x00),
            (F::Sop1, 0x00),
            (F::Ldsdir, 0),
            (F::Sop1, 0x00),
            (F::Vinterp, 0),
            (F::Sop1, 0x00),
            (F::Sop1, 0x00),
            (F::Ldsdir, 0),
            (F::Sop1, 0x00),
            (F::Vinterp, 1),
            (F::Sopp, 0x30),
        ]
    );
    let (saved, exec) = sop1(&insts[1]);
    assert_eq!(exec, Operand::ExecLo);
    assert_eq!(sop1(&insts[2]), (Operand::ExecLo, Operand::InlineInt(-1)));
    assert_eq!(sop1(&insts[4]), (Operand::ExecLo, saved));
    assert_eq!(
        insts[3].fields,
        Fields::Ldsdir(LdsDir {
            vdst: 3,
            attr: 1,
            chan: 1,
            wait_vdst: 0
        })
    );
    let Fields::Vinterp(VInterp {
        vdst,
        src,
        wait_exp,
        ..
    }) = insts[5].fields
    else {
        panic!("expected VINTERP");
    };
    assert_eq!(
        (vdst, src, wait_exp),
        (2, [Operand::Vgpr(3), Operand::Vgpr(0), Operand::Vgpr(3)], 0)
    );
    let Fields::Vinterp(VInterp { src, .. }) = insts[10].fields else {
        panic!("expected VINTERP");
    };
    assert_eq!(src, [Operand::Vgpr(3), Operand::Vgpr(1), Operand::Vgpr(2)]);
}

#[test]
fn a_known_prim_mask_hoists_each_parameter_load_to_the_prologue() {
    let interface = ShaderInterface {
        user_sgpr_count: Some(2),
        ..ShaderInterface::default()
    };
    let program = lower_pixel(&INTERP_P1_P2, TargetIsa::Gfx115, 32, interface).unwrap();
    let prologue = &program.blocks[0];
    assert_eq!(prologue.guest_pc, PROLOGUE_LABEL);
    assert_eq!(
        summary(&prologue.instructions),
        vec![
            (F::Sop1, 0x00),
            (F::Sop1, 0x00),
            (F::Sop1, 0x00),
            (F::Ldsdir, 0),
            (F::Sop1, 0x00),
            (F::Sopp, SOPP_GFX11_WAITCNT),
        ]
    );
    assert_eq!(
        sop1(&prologue.instructions[0]),
        (Operand::M0, Operand::Sgpr(2))
    );
    let Fields::Ldsdir(load) = prologue.instructions[3].fields else {
        panic!("expected LDSDIR");
    };
    assert_eq!((load.attr, load.chan), (1, 1), "one load for both uses");
    let body: Vec<&Instruction> = program.blocks[1..]
        .iter()
        .flat_map(|b| &b.instructions)
        .collect();
    assert!(body.iter().all(|i| i.family != F::Ldsdir));
    let reads: Vec<(Operand, u8)> = body
        .iter()
        .filter_map(|i| match i.fields {
            Fields::Vinterp(v) => Some((v.src[0], v.wait_exp)),
            _ => None,
        })
        .collect();
    assert_eq!(reads, vec![(Operand::Vgpr(load.vdst), 7); 2]);

    // M0 from another SGPR is not provably prim_mask: that interpolation loads in place.
    let other = [0xbefc_0303, 0xc808_0500, S_ENDPGM];
    let program = lower_pixel(&other, TargetIsa::Gfx115, 32, interface).unwrap();
    assert!(program.blocks.iter().all(|b| b.guest_pc != PROLOGUE_LABEL));
    assert_eq!(
        program
            .instructions()
            .filter(|i| i.family == F::Ldsdir)
            .count(),
        1
    );
}

#[test]
fn prim_mask_copies_and_rewrites_are_followed_through_control_flow() {
    let interface = ShaderInterface {
        user_sgpr_count: Some(2),
        ..ShaderInterface::default()
    };
    let in_place = |words: &[u32]| {
        lower_pixel(words, TargetIsa::Gfx115, 32, interface)
            .unwrap()
            .blocks
            .iter()
            .filter(|b| b.guest_pc != PROLOGUE_LABEL)
            .flat_map(|b| &b.instructions)
            .filter(|i| i.family == F::Ldsdir)
            .count()
    };
    // s_mov_b32 s5, s2; s_mov_b32 m0, s5: a copy of a copy.
    assert_eq!(
        in_place(&[0xbe85_0302, 0xbefc_0305, 0xc808_0500, S_ENDPGM]),
        0
    );
    // s_mov_b32 m0, s2; s_add_u32 s2, s2, 0; s_mov_b32 m0, s2: prim_mask was overwritten.
    assert_eq!(
        in_place(&[0xbefc_0302, 0x8002_8002, 0xbefc_0302, 0xc808_0500, S_ENDPGM]),
        1
    );
    // s_mov_b32 m0, s2; s_cbranch_scc0 +1; s_mov_b32 m0, s3; interp: one path changes M0.
    assert_eq!(
        in_place(&[0xbefc_0302, 0xbf84_0001, 0xbefc_0303, 0xc808_0500, S_ENDPGM]),
        1
    );
}

#[test]
fn removed_features_are_rejected_explicitly() {
    let param_export = [0xf800_0a0f, 0x0302_0100, S_ENDPGM];
    let err = lower_with(&param_export, TargetIsa::Gfx11, PrecisionMode::Fused, 32).unwrap_err();
    assert!(err.is_unsupported(), "{err}");
    assert!(err.to_string().contains("attribute ring"));

    let src2 = [0xda00_0000, 0x0000_0000, S_ENDPGM];
    for target in [TargetIsa::Gfx103, TargetIsa::Gfx11, TargetIsa::Gfx12] {
        let err = lower_with(&src2, target, PrecisionMode::Fused, 32).unwrap_err();
        assert!(err.to_string().contains("src2"), "{err}");
    }
    assert!(lower_with(&src2, TargetIsa::Gfx1013, PrecisionMode::Fused, 32).is_ok());
}

#[test]
fn typed_buffer_formats_are_remapped_for_gfx11() {
    let insts = lower(&[0xea6b_2000, 0x8002_0400, S_ENDPGM], TargetIsa::Gfx11);
    let Fields::Mtbuf(b) = insts[0].fields else {
        panic!("expected MTBUF");
    };
    assert_eq!(b.format, 63);
    assert_eq!(memops::remap_buffer_format(36, TargetIsa::Gfx11), Some(30));
    assert_eq!(memops::remap_buffer_format(31, TargetIsa::Gfx11), None);
    assert_eq!(memops::remap_buffer_format(31, TargetIsa::Gfx103), Some(31));
}

#[test]
fn compressed_exports_become_packed_dword_exports() {
    let insts = lower(&[0xf800_1c0f, 0x0000_0100, S_ENDPGM], TargetIsa::Gfx11);
    let Fields::Exp(e) = insts[0].fields else {
        panic!("expected EXP");
    };
    assert_eq!(
        (e.en, e.compr, e.vm, e.vsrc),
        (0b11, false, false, [0, 1, 0, 0])
    );
}

#[test]
fn packed_thread_ids_are_unpacked_for_compute_on_gfx11() {
    let compute = |target| -> Vec<Instruction> {
        lower_stage(
            &[0x4a04_0300, S_ENDPGM],
            target,
            PrecisionMode::Fused,
            32,
            ShaderStage::Compute,
        )
        .unwrap()
        .instructions()
        .cloned()
        .collect()
    };
    let insts = compute(TargetIsa::Gfx11);
    assert_eq!(
        summary(&insts[..2]),
        vec![(F::Vop3, 0x210), (F::Vop2, 0x1b)]
    );
    let bfe = vop3(&insts[0]);
    assert_eq!(bfe.vdst, Operand::Vgpr(1));
    assert_eq!(
        bfe.src,
        [
            Operand::Vgpr(0),
            Operand::InlineInt(10),
            Operand::InlineInt(10)
        ]
    );
    assert_eq!(
        summary(&compute(TargetIsa::Gfx103)),
        vec![(F::Vop2, 0x25), (F::Sopp, 0x01)]
    );
    let fragment = lower(&[0x4a04_0300, S_ENDPGM], TargetIsa::Gfx11);
    assert_eq!(summary(&fragment), vec![(F::Vop2, 0x25), (F::Sopp, 0x30)]);
}

#[test]
fn sixteen_bit_operands_above_v127_use_vop3_on_gfx11() {
    let insts = lower(&[0x6590_0501, S_ENDPGM], TargetIsa::Gfx11);
    assert_eq!((insts[0].family, insts[0].opcode), (F::Vop3, 0x132));
    assert_eq!(vop3(&insts[0]).vdst, Operand::Vgpr(200));
    let low = lower(&[0x6400_0501, S_ENDPGM], TargetIsa::Gfx11);
    assert_eq!((low[0].family, low[0].opcode), (F::Vop2, 0x32));
}

#[test]
fn scalar_compare_immediates_lower_to_sopc_on_gfx12() {
    let insts = lower(&[0xb181_ffff, S_ENDPGM], TargetIsa::Gfx12);
    assert_eq!(
        (insts[0].family, insts[0].opcode, insts[0].fields),
        (
            F::Sopc,
            0x00,
            Fields::Sopc {
                src0: Operand::Sgpr(1),
                src1: Operand::InlineInt(-1)
            }
        )
    );
    assert_eq!(
        lower(&[0xb181_ffff, S_ENDPGM], TargetIsa::Gfx11)[0].opcode,
        0x03
    );
}

fn ds_data(insts: &[Instruction]) -> (u8, u8) {
    match insts[0].fields {
        Fields::Ds(d) => (d.data0, d.data1),
        ref other => panic!("expected DS, got {other:?}"),
    }
}

#[test]
fn compare_stores_swap_their_data_operands_on_gfx11_and_later() {
    let cmpst_rtn_b32 = [0xd8c0_0000, 0x0406_0502, S_ENDPGM];
    let cmpst_rtn_b64 = [0xd9c0_0000, 0x0406_0802, S_ENDPGM];
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115, TargetIsa::Gfx12] {
        let insts = lower(&cmpst_rtn_b32, target);
        assert_eq!(
            (insts[0].opcode, ds_data(&insts)),
            (0x30, (6, 5)),
            "{target:?}"
        );
        assert_eq!(
            ds_data(&lower(&cmpst_rtn_b64, target)),
            (6, 8),
            "{target:?}"
        );
    }
    for target in [TargetIsa::Gfx1013, TargetIsa::Gfx103] {
        assert_eq!(
            ds_data(&lower(&cmpst_rtn_b32, target)),
            (5, 6),
            "{target:?}"
        );
    }
    let add_rtn = [0xd880_0000, 0x0400_0502, S_ENDPGM];
    assert_eq!(ds_data(&lower(&add_rtn, TargetIsa::Gfx11)), (5, 0));
}

#[test]
fn gfx12_rejects_the_lds_operations_it_removed() {
    let cmpst_rtn_f32 = [0xd8c4_0000, 0x0406_0502, S_ENDPGM];
    let wrap_rtn = [0xd8d0_0000, 0x0406_0502, S_ENDPGM];
    for words in [&cmpst_rtn_f32[..], &wrap_rtn] {
        let err = lower_with(words, TargetIsa::Gfx12, PrecisionMode::Fused, 32).unwrap_err();
        assert!(err.to_string().contains("gfx12 removed"), "{err}");
        let gfx11 = lower(words, TargetIsa::Gfx11);
        assert_eq!(gfx11[0].family, F::Ds);
    }
    assert_eq!(ds_data(&lower(&cmpst_rtn_f32, TargetIsa::Gfx11)), (6, 5));
}

#[test]
fn table_corrections_for_older_and_newer_targets() {
    let mul_lo_i32 = [0xd56b_0000, 0x0002_0501, S_ENDPGM];
    assert_eq!(lower(&mul_lo_i32, TargetIsa::Gfx103)[0].opcode, 0x169);
    assert_eq!(lower(&mul_lo_i32, TargetIsa::Gfx11)[0].opcode, 0x32c);

    let ordered_ps_done = [0xbf9e_0000];
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115] {
        assert_eq!(
            lower(&ordered_ps_done, target)[0].opcode,
            0x32,
            "{target:?}"
        );
    }
    let err = lower_with(&ordered_ps_done, TargetIsa::Gfx12, PrecisionMode::Fused, 32);
    assert!(err.unwrap_err().is_unsupported());

    let cdbgsys = [0xbf97_0000, S_ENDPGM];
    assert_eq!(
        summary(&lower(&cdbgsys, TargetIsa::Gfx12)),
        vec![(F::Sopp, 0x30)]
    );
    assert_eq!(lower(&cdbgsys, TargetIsa::Gfx11)[0].opcode, 0x27);

    let pipeflush = [0x7e00_3600, S_ENDPGM];
    assert_eq!(lower(&pipeflush, TargetIsa::Gfx12)[0].opcode, 0x1b);
}

#[test]
fn image_opcodes_that_exist_on_newer_targets_are_used() {
    let gather4h = [0xf184_0108, 0x0082_0402, S_ENDPGM];
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115, TargetIsa::Gfx12] {
        assert_eq!(lower(&gather4h, target)[0].opcode, 0x90, "{target:?}");
    }
    assert_eq!(lower(&gather4h, TargetIsa::Gfx103)[0].opcode, 0x61);
    for (op, gfx12) in [(0x1e, 0x84), (0x1f, 0x85)] {
        let atomic = [0xf000_0100 | (op << 18), 0x0002_0402, S_ENDPGM];
        assert_eq!(lower(&atomic, TargetIsa::Gfx12)[0].opcode, gfx12);
        for target in [TargetIsa::Gfx11, TargetIsa::Gfx115] {
            let images: Vec<_> = lower(&atomic, target).into_iter().filter(|inst| inst.family == F::Mimg).collect();
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].opcode, 0x0b);
        }
    }
}

#[test]
fn scalar_memory_operations_missing_on_gfx10_3_are_dropped_or_rejected() {
    let smem = |op: u32| [0xf400_0000 | (op << 18), 0xfa00_0000, S_ENDPGM];
    for op in [0x21, 0x28, 0x29] {
        assert_eq!(
            summary(&lower(&smem(op), TargetIsa::Gfx103)),
            vec![(F::Sopp, 0x01)]
        );
    }
    for op in [0x05, 0x06, 0x07, 0x2a] {
        let err = lower_with(&smem(op), TargetIsa::Gfx103, PrecisionMode::Fused, 32);
        assert!(err.unwrap_err().is_unsupported(), "{op:#x}");
    }
    assert_eq!(lower(&smem(0x26), TargetIsa::Gfx12)[0].opcode, 0x22);
    assert_eq!(lower(&smem(0x27), TargetIsa::Gfx12)[0].opcode, 0x23);
}

#[test]
fn zero_flag_operands_are_computed_on_gfx11_and_later() {
    let vccz = [0xbe84_03fb, S_ENDPGM];
    let execz_valu = [0x7e02_02fc, S_ENDPGM];
    for target in [TargetIsa::Gfx11, TargetIsa::Gfx115, TargetIsa::Gfx12] {
        let insts = lower(&vccz, target);
        assert_eq!(
            summary(&insts[..5]),
            vec![
                (F::Sop2, 0x30),
                (F::Sopc, 0x06),
                (F::Sop2, 0x30),
                (F::Sopc, 0x07),
                (F::Sop1, 0x00),
            ],
            "{target:?}"
        );
        assert_eq!(
            insts[1].fields,
            Fields::Sopc {
                src0: Operand::VccLo,
                src1: Operand::InlineInt(0)
            }
        );
        let Fields::Sop1 { src0, .. } = insts[4].fields else {
            panic!("expected s_mov_b32");
        };
        assert!(matches!(src0, Operand::Sgpr(_)), "{src0:?}");
        let valu = lower(&execz_valu, target);
        assert!(valu
            .iter()
            .all(|i| !format!("{:?}", i.fields).contains("ExecZ")));
    }
    assert_eq!(
        lower(&vccz, TargetIsa::Gfx103)[0].fields,
        one_sop1(Operand::VccZ)
    );
    let wave64 = lower_with(&vccz, TargetIsa::Gfx11, PrecisionMode::Fused, 64).unwrap();
    let compare = wave64.instructions().nth(1).unwrap();
    assert_eq!(
        (compare.family, compare.opcode),
        (F::Sopc, 0x10),
        "s_cmp_eq_u64"
    );
}

/// Loads from the descriptor conversion table: one-dword scalar loads at a LUT offset, indexed
/// by an SGPR (where the table slot lies depends on which slots the program reserves).
fn table_loads(program: &LoweredProgram) -> usize {
    use crate::descriptors::{LUT_BUFFER_FORMATS, LUT_IMAGE_FORMATS, LUT_SWIZZLE_MODES};
    let offsets = [LUT_BUFFER_FORMATS, LUT_IMAGE_FORMATS, LUT_SWIZZLE_MODES];
    program
        .instructions()
        .filter(|i| {
            matches!(i.fields, Fields::Smem(s)
                if !s.buffer
                    && s.dwords == 1
                    && matches!(s.soffset, Operand::Sgpr(_))
                    && offsets.contains(&(s.offset as u32)))
        })
        .count()
}

#[test]
fn inline_constants_cover_the_32_bit_set() {
    for value in -16..=64 {
        let expected = Some(Operand::InlineInt(value as i8));
        assert_eq!(Operand::inline_32(value as u32), expected, "{value}");
    }
    for value in [-17, 65, 0x3f80_0001, 0x3e22_f984, 0x3c00, 0x3fe0_0000] {
        assert_eq!(Operand::inline_32(value as u32), None, "{value:#x}");
        assert_eq!(
            Operand::constant(value as u32),
            Operand::Literal(value as u32)
        );
    }
    let floats = [0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0];
    for (code, value) in (240..).zip(floats) {
        let expected = Some(Operand::InlineFloat(code));
        assert_eq!(Operand::inline_32(value.to_bits()), expected, "{value}");
    }
    assert_eq!(
        Operand::inline_32(0x3e22_f983),
        Some(Operand::InlineFloat(248)),
        "1/(2π)"
    );
}

#[test]
fn guest_scalar_literals_that_fit_become_inline_constants() {
    let words = [
        0xbe80_03ff,
        0x0000_0000, // s_mov_b32 s0, lit(0)
        0x8000_ff00,
        0x3f80_0000, // s_add_u32 s0, s0, lit(1.0)
        0xbf06_ff00,
        0xffff_fff0, // s_cmp_eq_u32 s0, lit(-16)
        0x8700_ff00,
        0x0000_0041, // s_and_b32 s0, s0, lit(65): no inline constant
        0xbe82_04ff,
        0x0000_0001, // s_mov_b64 s[2:3], lit(1): a 64-bit source
        S_ENDPGM,
    ];
    let sources: Vec<Operand> = lower(&words, TargetIsa::Gfx115)
        .iter()
        .flat_map(|i| match i.fields {
            Fields::Sop1 { src0, .. } => vec![src0],
            Fields::Sop2 { src0, src1, .. } | Fields::Sopc { src0, src1 } => vec![src0, src1],
            _ => vec![],
        })
        .filter(|op| !matches!(op, Operand::Sgpr(_)))
        .collect();
    assert_eq!(
        sources,
        [
            Operand::InlineInt(0),
            Operand::InlineFloat(242),
            Operand::InlineInt(-16),
            Operand::Literal(65),
            Operand::Literal(1),
        ]
    );
}

#[test]
fn descriptors_convert_where_used_and_only_for_gfx11() {
    // buffer_load_dword v1, v0, s[4:7], 0 offen, twice.
    let twice = [0xe030_1000, 0x8001_0100, 0xe030_1000, 0x8001_0200, S_ENDPGM];
    let program = |target| lower_with_descriptors(&twice, target, ShaderStage::Compute, false);
    let gfx115 = program(TargetIsa::Gfx115).unwrap();
    assert_eq!(table_loads(&gfx115), 1, "the second load reuses the slot");
    let prologue = &gfx115.blocks[0];
    assert_eq!(prologue.guest_pc, PROLOGUE_LABEL);
    // The table at 0x7_0000_0000: both halves fit inline constants.
    let address: Vec<Operand> = prologue
        .instructions
        .iter()
        .filter_map(|i| match i.fields {
            Fields::Sop1 { src0, .. } => Some(src0),
            _ => None,
        })
        .collect();
    assert_eq!(address, vec![Operand::InlineInt(0), Operand::InlineInt(7)]);
    let srsrcs: Vec<u8> = gfx115
        .instructions()
        .filter_map(|i| match i.fields {
            Fields::Mubuf(b) => Some(b.srsrc),
            _ => None,
        })
        .collect();
    assert!(
        srsrcs.iter().all(|&s| s != 4 && s == srsrcs[0]),
        "{srsrcs:?}"
    );
    let first = gfx115.instructions().find(|i| i.family == F::Sop2).unwrap();
    assert_eq!(first.opcode, 0x30, "SCC is saved first (s_cselect_b32)");
    // gfx10.3 reads PS5 descriptors as they are; gfx12 has no conversion yet.
    assert_eq!(table_loads(&program(TargetIsa::Gfx103).unwrap()), 0);
    assert!(program(TargetIsa::Gfx12).is_err());
    // A guest write to the V# between the loads (s_mov_b32 s5, 0) converts it again.
    let rewritten = [
        0xe030_1000,
        0x8001_0100,
        0xbe85_0380,
        0xe030_1000,
        0x8001_0200,
        S_ENDPGM,
    ];
    let program =
        lower_with_descriptors(&rewritten, TargetIsa::Gfx115, ShaderStage::Compute, false);
    assert_eq!(table_loads(&program.unwrap()), 2);
}

#[test]
fn bvh_and_scalar_buffer_loads_read_guest_descriptors() {
    // s_buffer_load_dword s8, s[4:7], 0x0; image_bvh_intersect_ray v[4:7], v[8:18], s[20:23].
    let words = [0xf420_0202, 0xfa00_0000, 0xf198_9f01, 0x0005_0408, S_ENDPGM];
    let program =
        lower_with_descriptors(&words, TargetIsa::Gfx115, ShaderStage::Compute, false).unwrap();
    let smem: Vec<u8> = program
        .instructions()
        .filter_map(|i| match i.fields {
            Fields::Smem(s) => Some(s.sbase),
            _ => None,
        })
        .collect();
    assert_eq!(
        smem,
        vec![4],
        "only the guest's s_buffer_load, on its own V#"
    );
}

#[test]
fn r128_images_get_an_eight_dword_descriptor() {
    // image_sample v[4:7], v[1:2], s[8:11], s[16:19] dmask:0xf r128
    let words = [0xf080_8f08, 0x0082_0401, S_ENDPGM];
    let program =
        lower_with_descriptors(&words, TargetIsa::Gfx115, ShaderStage::Fragment, false).unwrap();
    let sample = program
        .instructions()
        .find_map(|i| match i.fields {
            Fields::Mimg(m) => Some(m),
            _ => None,
        })
        .unwrap();
    assert!(!sample.r128);
    assert_eq!(sample.srsrc % 4, 0);
    assert!(sample.srsrc >= 20 && sample.ssamp >= 20);
}

#[test]
fn user_descriptors_used_only_as_descriptors_are_left_to_the_runtime() {
    // buffer_load_dword v1, v0, s[4:7], 0 offen; image_sample v[4:7], v[1:2], s[8:15], s[16:19]
    // with the sampler's dword 3 (s19) written first: only the V# and T# qualify.
    let words = [
        0xbe93_0380, // s_mov_b32 s19, 0
        0xe030_1000,
        0x8001_0100,
        0xf080_0f08,
        0x0082_0401,
        S_ENDPGM,
    ];
    let program =
        lower_with_descriptors(&words, TargetIsa::Gfx115, ShaderStage::Fragment, true).unwrap();
    let kinds: Vec<(ResourceKind, u8)> = program
        .preconverted
        .iter()
        .map(|b| (b.kind, b.base_sgpr))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (ResourceKind::BufferDescriptor, 4),
            (ResourceKind::ImageDescriptor, 8)
        ]
    );
    let sample = program
        .instructions()
        .find_map(|i| match i.fields {
            Fields::Mimg(m) => Some(m),
            _ => None,
        })
        .unwrap();
    assert_eq!(sample.srsrc, 8, "the T# is used as the runtime left it");
    assert_ne!(sample.ssamp, 16, "the written S# is converted");
}

#[test]
fn the_prologue_is_its_own_block_so_branches_to_pc_0_skip_it() {
    // v_add_nc_u32 v3, v0, v1; s_cmp_eq_u32 s0, 0; s_cbranch_scc0 -> pc 0; s_endpgm.
    let words = [0x4a06_0300, 0xbf06_8000, 0xbf84_fffd, S_ENDPGM];
    let program = lower_compute(&words, TargetIsa::Gfx115, ShaderInterface::default()).unwrap();
    assert_eq!(program.blocks[0].guest_pc, PROLOGUE_LABEL);
    assert_eq!(
        summary(&program.blocks[0].instructions),
        vec![(F::Vop3, 0x210), (F::Vop2, 0x1b)],
        "v_bfe_u32 v1, v0, 10, 10; v_and_b32 v0, 0x3ff, v0"
    );
    assert_eq!(program.blocks[1].guest_pc, 0);
    let branch = program
        .instructions()
        .find(|i| i.branch_target.is_some())
        .unwrap();
    assert_eq!(branch.branch_target, Some(0));
    // gfx10 targets load the thread IDs unpacked: no prologue.
    let gfx103 = lower_compute(&words, TargetIsa::Gfx103, ShaderInterface::default()).unwrap();
    assert_eq!(gfx103.blocks[0].guest_pc, 0);
}

#[test]
fn packed_thread_ids_are_masked_whenever_y_or_z_is_enabled() {
    // v_mov_b32 v4, v0 reads only the X ID.
    let words = [0x7e08_0300, S_ENDPGM];
    let tidig = |count: u32| ShaderInterface {
        guest_pgm_rsrc2: Some(count << crate::regalloc::RSRC2_TIDIG_SHIFT),
        ..ShaderInterface::default()
    };
    let prologue = |interface| {
        let program = lower_compute(&words, TargetIsa::Gfx115, interface).unwrap();
        let first = &program.blocks[0];
        if first.guest_pc == PROLOGUE_LABEL {
            summary(&first.instructions)
        } else {
            Vec::new()
        }
    };
    assert_eq!(prologue(tidig(2)), vec![(F::Vop2, 0x1b)]);
    assert_eq!(prologue(tidig(0)), Vec::new());
}

#[test]
fn shader_clock_reads_scale_the_realtime_counter() {
    // s_memtime s[10:11]
    let words = [0xf490_0280, 0x0000_0000, S_ENDPGM];
    let program = lower_compute(&words, TargetIsa::Gfx115, ShaderInterface::default()).unwrap();
    let insts: Vec<&Instruction> = program.instructions().collect();
    assert_eq!(
        (insts[0].family, insts[0].opcode),
        (F::Sop1, SOP1_GFX11_SENDMSG_RTN_B64)
    );
    let factor = |i: &Instruction| {
        matches!(
            i.fields,
            Fields::Sop2 {
                src1: Operand::Literal(715),
                ..
            }
        )
    };
    assert_eq!(
        insts.iter().filter(|i| factor(i)).count(),
        3,
        "mul_hi, mul lo, mul hi"
    );
    let shift = insts
        .iter()
        .find(|i| {
            matches!(
                i.fields,
                Fields::Sop2 {
                    src1: Operand::InlineInt(5),
                    ..
                }
            )
        })
        .expect("s_lshr_b64 by 5");
    assert_eq!(shift.opcode, 0x09 + 2, "gfx11 s_lshr_b64");
    // s_memrealtime is the counter itself.
    let realtime = [0xf494_0280, 0x0000_0000, S_ENDPGM];
    let program = lower_compute(&realtime, TargetIsa::Gfx115, ShaderInterface::default()).unwrap();
    assert_eq!(program.instructions().count(), 2);
}

#[test]
fn flat_scratch_setup_is_left_to_gfx11_hardware() {
    // s_add_u32 s9, s8, 4; s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s9
    let words = [0x8009_8408, 0xb989_f814, S_ENDPGM];
    let setregs = |target| {
        lower_compute(&words, target, ShaderInterface::default())
            .unwrap()
            .instructions()
            .filter(|i| i.family == F::Sopk)
            .count()
    };
    assert_eq!(setregs(TargetIsa::Gfx103), 1);
    assert_eq!(setregs(TargetIsa::Gfx115), 0);
}

#[test]
fn other_reads_of_the_scratch_wave_offset_are_rejected_on_gfx11() {
    // s_add_u32 s9, s8, 4 with SCRATCH_EN and eight user SGPRs: s8 is the wave offset.
    let words = [0x8009_8408, 0x7e02_0209, S_ENDPGM];
    let scratch = ShaderInterface {
        guest_pgm_rsrc2: Some(1 | (8 << 1)),
        ..ShaderInterface::default()
    };
    assert!(lower_compute(&words, TargetIsa::Gfx103, scratch).is_ok());
    let err = lower_compute(&words, TargetIsa::Gfx115, scratch).unwrap_err();
    assert!(err.to_string().contains("scratch wave offset"), "{err}");
}

#[test]
fn gpr_alloc_reads_are_rejected_where_its_layout_changed() {
    // s_getreg_b32 s4, hwreg(HW_REG_GPR_ALLOC)
    let words = [0xb904_f805, S_ENDPGM];
    assert!(lower_compute(&words, TargetIsa::Gfx103, ShaderInterface::default()).is_ok());
    let err = lower_compute(&words, TargetIsa::Gfx115, ShaderInterface::default()).unwrap_err();
    assert!(
        err.to_string().contains("hardware register layout"),
        "{err}"
    );
}

#[test]
fn adjusted_samples_translate_to_the_plain_form_with_a_cleaned_sampler() {
    // image_sample_a v[4:7], v[1:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D.
    let words = [0xf080_0f09, 0x0040_0401, S_ENDPGM];
    assert_eq!(
        summary(&lower(&words, TargetIsa::Gfx1013)[..1]),
        vec![(F::Mimg, 0xa0)]
    );
    for (target, plain) in [
        (TargetIsa::Gfx103, 0x20),
        (TargetIsa::Gfx115, 0x1b),
        (TargetIsa::Gfx12, 0x1b),
    ] {
        let insts = lower(&words, target);
        let at = insts.iter().position(|i| i.family == F::Mimg).unwrap();
        let Fields::Mimg(m) = insts[at].fields else {
            panic!("expected MIMG");
        };
        assert_eq!(insts[at].opcode, plain, "{target:?}");
        assert!(
            m.ssamp >= 12 && m.ssamp % 4 == 0,
            "{target:?}: copy at s{}",
            m.ssamp
        );
        let mask = insts[..at]
            .iter()
            .find_map(|i| match i.fields {
                Fields::Sop2 {
                    sdst,
                    src0,
                    src1: Operand::Literal(0xc000_0fff),
                } => Some((sdst, src0)),
                _ => None,
            })
            .expect("dword 3 mask");
        assert_eq!(mask, (Operand::Sgpr(m.ssamp + 3), Operand::Sgpr(11)));
        assert!(
            matches!(insts[at - 1].fields, Fields::Sopc { src0: Operand::Sgpr(r), .. } if r == m.ssamp + 4),
            "{target:?}: SCC is restored last"
        );
    }
    // image_gather4_c_a / image_sample_c_o_a follow their plain forms the same way.
    for (word0, plain) in [(0xf120_0109, 0x33), (0xf0e0_0109, 0x2a)] {
        let insts = lower(&[word0, 0x0080_1e33, S_ENDPGM], TargetIsa::Gfx115);
        assert!(insts
            .iter()
            .any(|i| i.family == F::Mimg && i.opcode == plain));
    }
}

#[test]
fn gfx10_clauses_are_kept_only_while_their_members_survive_lowering() {
    let s_clause_1 = 0xbfa1_0001;
    let clause = |insts: &[Instruction]| {
        insts
            .iter()
            .any(|i| i.family == F::Sopp && i.opcode == 0x21)
    };
    // buffer_load_dword v1, v0, s[4:7], 0 offen; buffer_load_dword v2, ... offset:4.
    let loads = [
        s_clause_1,
        0xe030_1000,
        0x8001_0100,
        0xe030_1004,
        0x8001_0200,
        S_ENDPGM,
    ];
    assert!(clause(&lower(&loads, TargetIsa::Gfx103)));
    // Two image_bvh_intersect_ray v[4:7], v[8:18], s[20:23], native only on the PS5.
    let bvh = [
        s_clause_1,
        0xf198_9f01,
        0x0005_0408,
        0xf198_9f01,
        0x0005_0408,
        S_ENDPGM,
    ];
    assert!(clause(&lower(&bvh, TargetIsa::Gfx1013)));
    assert!(!clause(&lower(&bvh, TargetIsa::Gfx103)));
    // s_load_dword s8, s[4:5], 0x0; s_dcache_wb (dropped on gfx10.3).
    let smem = [
        s_clause_1,
        0xf400_0202,
        0xfa00_0000,
        0xf484_0000,
        0,
        S_ENDPGM,
    ];
    assert!(clause(&lower(&smem, TargetIsa::Gfx1013)));
    assert!(!clause(&lower(&smem, TargetIsa::Gfx103)));
}

#[test]
fn gfx11_0_64_bit_mads_never_overlap_their_sources() {
    // v_mad_u64_u32 v[4:5], s4, v2, v3, v[4:5] accumulates in place.
    let in_place = [0xd576_0404, 0x0412_0702, S_ENDPGM];
    let insts = lower(&in_place, TargetIsa::Gfx11);
    assert_eq!(
        summary(&insts[..3]),
        vec![(F::Vop3, 0x2fe), (F::Vop1, 0x01), (F::Vop1, 0x01)]
    );
    let Operand::Vgpr(tmp) = vop3(&insts[0]).vdst else {
        panic!("expected a VGPR destination");
    };
    assert!(tmp >= 6, "scratch above the guest's VGPRs");
    for target in [TargetIsa::Gfx115, TargetIsa::Gfx12] {
        let direct = lower(&in_place, target);
        assert_eq!(vop3(&direct[0]).vdst, Operand::Vgpr(4), "{target:?}");
    }
    let separate = [0xd576_0404, 0x041a_0702, S_ENDPGM];
    assert_eq!(
        vop3(&lower(&separate, TargetIsa::Gfx11)[0]).vdst,
        Operand::Vgpr(4)
    );
}

fn one_sop1(src0: Operand) -> Fields {
    Fields::Sop1 {
        sdst: Operand::Sgpr(4),
        src0,
    }
}

#[test]
fn barriers_split_into_signal_and_wait_on_gfx12() {
    let insts = lower(&[0xbf8a_0000, S_ENDPGM], TargetIsa::Gfx12);
    assert_eq!(
        summary(&insts),
        vec![
            (F::Sop1, SOP1_GFX12_BARRIER_SIGNAL),
            (F::Sopp, SOPP_GFX12_BARRIER_WAIT),
            (F::Sopp, 0x30)
        ]
    );
    assert_eq!(
        lower(&[0xbf8a_0000, S_ENDPGM], TargetIsa::Gfx11)[0].opcode,
        0x3d
    );
}

