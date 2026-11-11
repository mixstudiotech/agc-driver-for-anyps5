use super::*;
use crate::ir::{MemKind, SDWA_SEL_WORD_1, SDWA_UNUSED_PRESERVE};

const S_ENDPGM: u32 = 0xbf81_0000;

fn one(words: &[u32]) -> Instruction {
    Decoder::new().decode_at(words, 0).expect("decode")
}

fn fields(words: &[u32]) -> Fields {
    one(words).fields
}

#[test]
fn family_is_selected_from_the_encoding_prefix() {
    let cases = [
        (0xbe80_0380, F::Sop1),
        (0x8002_ff00, F::Sop2),
        (0xb000_0000, F::Sopk),
        (0xbf06_8400, F::Sopc),
        (S_ENDPGM, F::Sopp),
        (0x7e02_02f2, F::Vop1),
        (0x0600_0501, F::Vop2),
        (0x7d84_0280, F::Vopc),
        (0xd54b_0000, F::Vop3),
        (0xcc0e_0000, F::Vop3p),
        (0xc808_0500, F::Vintrp),
        (0xf408_0100, F::Smem),
        (0xe030_1000, F::Mubuf),
        (0xea6b_2000, F::Mtbuf),
        (0xdc30_8000, F::Flat),
        (0xd8dc_0201, F::Ds),
        (0xf080_0f08, F::Mimg),
        (0xf800_180f, F::Exp),
    ];
    for (word, family) in cases {
        assert_eq!(family_of(word), Some(family), "{word:#010x}");
    }
    assert_eq!(family_of(0xffff_ffff), None);
}

#[test]
fn sop2_reads_a_trailing_literal() {
    let inst = one(&[0x8002_ff00, 0x1234_5678]);
    assert_eq!(
        (inst.family, inst.opcode, inst.word_count()),
        (F::Sop2, 0, 2)
    );
    assert_eq!(
        inst.fields,
        Fields::Sop2 {
            sdst: Operand::Sgpr(2),
            src0: Operand::Sgpr(0),
            src1: Operand::Literal(0x1234_5678),
        }
    );
    assert_eq!(inst.mnemonic(), "s_add_u32");
}

#[test]
fn scalar_operands_cover_special_registers() {
    assert_eq!(
        fields(&[0xbefc_03c1]),
        Fields::Sop1 {
            sdst: Operand::M0,
            src0: Operand::InlineInt(-1),
        }
    );
    let saveexec = one(&[0xbe80_3c6a]);
    assert!(saveexec.has_flag(flag::WRITES_EXEC));
    assert_eq!(
        saveexec.fields,
        Fields::Sop1 {
            sdst: Operand::Sgpr(0),
            src0: Operand::VccLo,
        }
    );
}

#[test]
fn sopp_branches_resolve_word_relative_targets() {
    let forward = Decoder::new().decode_at(&[0, 0, 0xbf84_0002], 2).unwrap();
    assert_eq!(forward.branch_target, Some(20));
    let backward = Decoder::new()
        .decode_at(&[0, 0, 0, 0xbf84_fffd], 3)
        .unwrap();
    assert_eq!(backward.branch_target, Some(4));
    assert_eq!(backward.mnemonic(), "s_cbranch_scc0");
}

#[test]
fn vop1_decodes_inline_float_constants() {
    let inst = one(&[0x7e02_02f2]);
    assert_eq!(
        inst.fields,
        Fields::Vop1 {
            vdst: Operand::Vgpr(1),
            src0: Operand::InlineFloat(242),
            ext: VopExt::None,
        }
    );
    assert_eq!(Operand::inline_float_bits(242), Some(1.0f32.to_bits()));
}

#[test]
fn vop2_sdwa_selectors_are_decoded() {
    let Fields::Vop2 {
        vdst,
        src0,
        vsrc1,
        ext: VopExt::Sdwa(s),
        ..
    } = fields(&[0x0600_04f9, 0x0204_1501])
    else {
        panic!("expected SDWA");
    };
    assert_eq!(
        (vdst, src0, vsrc1),
        (Operand::Vgpr(0), Operand::Vgpr(1), Operand::Vgpr(2))
    );
    assert_eq!(s.dst_sel, SDWA_SEL_WORD_1);
    assert_eq!(s.dst_unused, SDWA_UNUSED_PRESERVE);
    assert_eq!((s.src0_sel, s.src1_sel), (4, 2));
}

#[test]
fn vop1_dpp16_control_word_is_decoded() {
    let Fields::Vop1 {
        src0,
        ext: VopExt::Dpp16(dpp),
        ..
    } = fields(&[0x7e00_02fa, 0xff00_b101])
    else {
        panic!("expected DPP");
    };
    assert_eq!(src0, Operand::Vgpr(1));
    assert_eq!((dpp.ctrl, dpp.row_mask, dpp.bank_mask), (0xb1, 0xf, 0xf));
}

#[test]
fn vop3_decodes_modifiers_and_carry_out() {
    let Fields::Vop3(fma) = fields(&[0xd54b_8200, 0x240e_0501]) else {
        panic!("expected VOP3");
    };
    assert_eq!(
        fma.src,
        [Operand::Vgpr(1), Operand::Vgpr(2), Operand::Vgpr(3)]
    );
    assert_eq!((fma.abs, fma.neg, fma.clamp), (0b010, 0b001, true));

    let Fields::Vop3(add) = fields(&[0xd70f_6a00, 0x0002_0501]) else {
        panic!("expected VOP3B");
    };
    assert_eq!(add.sdst, Some(Operand::VccLo));
    assert_eq!(&add.src[..2], &[Operand::Vgpr(1), Operand::Vgpr(2)]);

    let literal = one(&[0xd503_0000, 0x0002_02ff, 0x4049_0fdb]);
    assert_eq!(literal.word_count(), 3);
    let Fields::Vop3(v) = literal.fields else {
        panic!("expected VOP3");
    };
    assert_eq!(v.src[0], Operand::Literal(0x4049_0fdb));
}

#[test]
fn vopc_destinations_follow_the_compare_kind() {
    let cmp = one(&[0x7d84_0280]);
    assert_eq!(cmp.mnemonic(), "v_cmp_eq_u32");
    assert_eq!(
        cmp.fields,
        Fields::Vopc {
            sdst: Operand::VccLo,
            src0: Operand::InlineInt(0),
            vsrc1: Operand::Vgpr(1),
            ext: VopExt::None,
        }
    );
    let cmpx = one(&[0x7c28_0280]);
    assert_eq!(cmpx.mnemonic(), "v_cmpx_gt_f32");
    assert!(cmpx.has_flag(flag::CMPX | flag::WRITES_EXEC));
    assert!(matches!(
        cmpx.fields,
        Fields::Vopc {
            sdst: Operand::ExecLo,
            ..
        }
    ));
}

#[test]
fn vintrp_fields_are_decoded() {
    let p1 = one(&[0xc808_0500]);
    assert_eq!(p1.opcode, opcodes::VINTRP_P1);
    assert_eq!(
        p1.fields,
        Fields::Vintrp(Interp {
            vdst: 2,
            vsrc: 0,
            attr: 1,
            chan: 1,
        })
    );
    assert!(one(&[0xc809_0501]).has_flag(flag::READS_DST));
}

#[test]
fn scalar_memory_decodes_base_offset_and_buffer_kind() {
    let Fields::Smem(load) = fields(&[0xf408_0100, 0xfa00_0010]) else {
        panic!("expected SMEM");
    };
    assert_eq!(
        (load.sdata, load.sbase, load.offset, load.dwords),
        (Operand::Sgpr(4), 0, 0x10, 4)
    );
    assert_eq!(load.soffset, Operand::Null);
    assert!(!load.buffer);
    let Fields::Smem(buffer) = fields(&[0xf420_0202, 0xfa00_0000]) else {
        panic!("expected SMEM");
    };
    assert_eq!((buffer.sbase, buffer.buffer), (4, true));
}

#[test]
fn buffer_instructions_decode_addressing_and_format() {
    let Fields::Mubuf(load) = fields(&[0xe030_1000, 0x8002_0100]) else {
        panic!("expected MUBUF");
    };
    assert_eq!((load.vdata, load.vaddr, load.srsrc), (1, 0, 8));
    assert_eq!(load.soffset, Operand::InlineInt(0));
    assert!(load.offen && !load.idxen);
    assert_eq!((load.kind, load.data_dwords), (MemKind::Load, 1));

    let typed = one(&[0xea6b_2000, 0x8002_0400]);
    assert_eq!(typed.mnemonic(), "tbuffer_load_format_xyzw");
    let Fields::Mtbuf(t) = typed.fields else {
        panic!("expected MTBUF");
    };
    assert_eq!((t.format, t.data_dwords, t.vdata), (77, 4, 4));
    assert!(t.idxen);
}

#[test]
fn flat_segments_and_scalar_bases_are_decoded() {
    let Fields::Flat(load) = fields(&[0xdc30_8000, 0x017d_0002]) else {
        panic!("expected FLAT");
    };
    assert_eq!(load.segment, FlatSegment::Global);
    assert_eq!(
        (load.vdst, load.vaddr, load.saddr),
        (1, 2, Some(Operand::Null))
    );
    assert!(load.uses_vaddr);

    let Fields::Flat(store) = fields(&[0xdc70_8ff8, 0x0004_0302]) else {
        panic!("expected FLAT");
    };
    assert_eq!(
        (store.saddr, store.offset, store.vdata),
        (Some(Operand::Sgpr(4)), -8, 3)
    );
    assert_eq!(store.kind, MemKind::Store);
}

#[test]
fn ds_and_export_fields_are_decoded() {
    let Fields::Ds(read2) = fields(&[0xd8dc_0201, 0x0000_0002]) else {
        panic!("expected DS");
    };
    assert_eq!(
        (
            read2.vdst,
            read2.addr,
            read2.offset0,
            read2.offset1,
            read2.dst_dwords
        ),
        (0, 2, 1, 2, 2)
    );

    assert_eq!(
        fields(&[0xf800_180f, 0x0302_0100]),
        Fields::Exp(Export {
            target: 0,
            en: 0xf,
            compr: false,
            done: true,
            vm: true,
            row_en: false,
            vsrc: [0, 1, 2, 3],
        })
    );
}

#[test]
fn mimg_decodes_contiguous_and_nsa_addresses() {
    let Fields::Mimg(sample) = fields(&[0xf080_0f08, 0x0082_0004]) else {
        panic!("expected MIMG");
    };
    assert_eq!(
        (
            sample.dim,
            sample.dmask,
            sample.vdata,
            sample.srsrc,
            sample.ssamp
        ),
        (1, 0xf, 0, 8, 16)
    );
    assert_eq!(
        (sample.addr[0], sample.addr_dwords, sample.data_dwords),
        (4, 2, 4)
    );
    assert!(sample.uses_sampler && !sample.nsa);

    let nsa = one(&[0xf080_0f0a, 0x0082_0004, 0x0000_0007]);
    assert_eq!(nsa.word_count(), 3);
    let Fields::Mimg(m) = nsa.fields else {
        panic!("expected MIMG");
    };
    assert!(m.nsa);
    assert_eq!((m.address_register(0), m.address_register(1)), (4, 7));
}

fn mimg(words: &[u32]) -> (Instruction, Image) {
    let inst = one(words);
    let Fields::Mimg(m) = inst.fields else {
        panic!("expected MIMG");
    };
    (inst, m)
}

fn addresses(m: &Image) -> Vec<u8> {
    (0..usize::from(m.addr_dwords))
        .map(|i| m.address_register(i))
        .collect()
}

/// AnyPS5's `_a` vectors (`ImageSampleGather.cpp`): the 2D forms take exactly the plain forms'
/// address operands (compare, x, y; offset, compare, x, y).
#[test]
fn adjusted_samples_decode_with_the_plain_address_operands() {
    let cases = [
        (0xf120_0109, 0xc8, "image_gather4_c_a", 3),
        (0xf160_0109, 0xd8, "image_gather4_c_o_a", 4),
        (0xf0e0_0109, 0xb8, "image_sample_c_o_a", 4),
    ];
    for (word0, opcode, name, addresses) in cases {
        let (inst, m) = mimg(&[word0, 0x0080_1e33]);
        assert_eq!((inst.opcode, inst.mnemonic().as_str()), (opcode, name));
        assert_eq!((m.addr_dwords, m.addr[0], m.vdata), (addresses, 51, 30));
        assert_eq!((m.dim, m.srsrc, m.ssamp), (1, 0, 16));
        assert!(m.uses_sampler);
        let plain = one(&[word0 & !1, 0x0080_1e33]);
        let Fields::Mimg(p) = plain.fields else {
            panic!("expected MIMG");
        };
        assert_eq!(plain.opcode, opcode - 0x80);
        assert_eq!(
            (p.addr_dwords, p.data_dwords),
            (m.addr_dwords, m.data_dwords)
        );
    }
}

/// Encodings from `llvm-mc -mcpu=gfx1013`; gfx1030 produces the same bytes for these forms.
#[test]
fn bvh_intersections_decode_their_address_layouts() {
    let cases: [(&[u32], u16, bool, &[u8]); 6] = [
        (
            &[0xf198_9f01, 0x0001_0004],
            0xe6,
            false,
            &[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ),
        (
            &[0xf198_9f01, 0x4001_0004],
            0xe6,
            true,
            &[4, 5, 6, 7, 8, 9, 10, 11],
        ),
        (
            &[0xf19c_9f01, 0x0001_0004],
            0xe7,
            false,
            &[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        (
            &[0xf19c_9f01, 0x4001_0004],
            0xe7,
            true,
            &[4, 5, 6, 7, 8, 9, 10, 11, 12],
        ),
        (
            &[
                0xf198_9f07,
                0x0001_0004,
                0x0c0b_0a09,
                0x100f_0e0d,
                0x0000_1411,
            ],
            0xe6,
            false,
            &[4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20],
        ),
        (
            &[0xf19c_9f05, 0x4001_0004, 0x0b0a_0905, 0x0f0e_0d0c],
            0xe7,
            true,
            &[4, 5, 9, 10, 11, 12, 13, 14, 15],
        ),
    ];
    for (words, opcode, a16, regs) in cases {
        let (inst, m) = mimg(words);
        assert_eq!((inst.opcode, inst.word_count()), (opcode, words.len()));
        assert_eq!(addresses(&m), regs, "{words:x?}");
        assert_eq!((m.a16, m.vdata, m.srsrc, m.data_dwords), (a16, 0, 4, 4));
        assert!(m.r128 && m.unorm && m.dmask == 0xf);
        let defs = inst.defs(false);
        assert_eq!(defs, vec![crate::ir::RegRange::vector(0, 4)]);
    }
    // The data width follows the instruction, not DMASK.
    assert_eq!(mimg(&[0xf198_9101, 0x0001_0004]).1.data_dwords, 4);
}

/// MIMG 0xe4 is not a supported PS5 or Pro instruction.
#[test]
fn mimg_opcode_0xe4_is_unknown() {
    for (word0, opcode) in [(0xf190_9f05, 0xe4)] {
        let words = [word0, 0x000b_0010, 0x0d15_1411, 0x0000_200a];
        assert!(matches!(
            Decoder::new().decode_at(&words, 0),
            Err(CompileError::UnknownOpcode {
                family: InstructionFamily::Mimg,
                opcode: op,
                ..
            }) if op == opcode
        ));
    }
}

#[test]
fn program_decoding_stops_after_the_last_forward_target() {
    let words = [0xbf84_0001, S_ENDPGM, 0xbe80_0380, S_ENDPGM, 0xdead_beef];
    let ir = Decoder::new().decode(&words).unwrap();
    assert_eq!(ir.code_words, 4);
    assert_eq!(ir.instructions.len(), 4);
}

#[test]
fn program_decoding_stops_at_a_tail_jump() {
    let ir = Decoder::new()
        .decode(&[0xbe80_0380, 0xbe80_2006, 0xffff_ffff])
        .unwrap();
    assert_eq!(ir.code_words, 2);
    assert!(ir.instructions[1].has_flag(flag::JUMP));
}

#[test]
fn decoding_errors_are_reported() {
    let d = Decoder::new();
    assert_eq!(d.decode(&[]), Err(CompileError::EmptyProgram));
    assert_eq!(
        d.decode(&[0xbe80_0380]),
        Err(CompileError::MissingEndProgram)
    );
    assert_eq!(
        d.decode(&[0xbe80_7f00, S_ENDPGM]),
        Err(CompileError::UnknownOpcode {
            pc: 0,
            family: F::Sop1,
            opcode: 0x7f
        })
    );
    assert_eq!(
        d.decode(&[0xd54b_0000]),
        Err(CompileError::Truncated { pc: 0 })
    );
    assert_eq!(
        d.decode(&[0xbf82_0010, S_ENDPGM]),
        Err(CompileError::InvalidBranch { pc: 0, target: 68 })
    );
    assert_eq!(
        d.decode(&[0xbf82_0001, S_ENDPGM, 0xffff_ffff]),
        Err(CompileError::InvalidEncoding {
            pc: 8,
            word: 0xffff_ffff
        })
    );
}

fn unknown(words: &[u32]) -> bool {
    matches!(
        Decoder::new().decode_at(words, 0),
        Err(CompileError::UnknownOpcode { .. })
    )
}

#[test]
fn encodings_no_oracle_defines_are_rejected() {
    let flat = |op: u32, segment: u32| [0xdc00_0000 | (op << 18) | (segment << 14), 0x007f_0102];
    let (flat_seg, scratch, global) = (0, 1, 2);
    for op in [0x30, 0x31, 0x40, 0x50, 0x60] {
        assert!(unknown(&flat(op, scratch)), "scratch atomic {op:#x}");
        assert!(!unknown(&flat(op, global)), "global atomic {op:#x}");
    }
    for op in [0x16, 0x17, 0x34] {
        assert!(unknown(&flat(op, flat_seg)), "flat {op:#x}");
        assert!(unknown(&flat(op, scratch)), "scratch {op:#x}");
        assert!(!unknown(&flat(op, global)), "global {op:#x}");
    }
    let vop3_only_forms = [
        0x102, 0x10d, 0x120, 0x121, 0x12c, 0x12d, 0x137, 0x138, 0x13c, 0x182, 0x1e5, 0x1e8,
    ];
    for op in vop3_only_forms {
        assert!(
            unknown(&[0xd400_0000 | (op << 16), 0x0002_0501]),
            "VOP3 {op:#x}"
        );
    }
    assert!(
        !unknown(&[0xd506_0000, 0x0002_0501]),
        "v_mac_legacy_f32_e64 exists"
    );
}

#[test]
fn g16_coarse_derivative_samples_decode() {
    let names = [
        "image_sample_cd_g16",
        "image_sample_cd_cl_g16",
        "image_sample_c_cd_g16",
        "image_sample_c_cd_cl_g16",
        "image_sample_cd_o_g16",
        "image_sample_cd_cl_o_g16",
        "image_sample_c_cd_o_g16",
        "image_sample_c_cd_cl_o_g16",
    ];
    for (op, name) in (0xe8u32..).zip(names) {
        let word0 = 0xf000_0f09 | ((op & 0x7f) << 18);
        let inst = one(&[word0, 0x0082_0402]);
        assert_eq!((inst.opcode, inst.mnemonic().as_str()), (op as u16, name));
        let Fields::Mimg(m) = inst.fields else {
            panic!("expected MIMG");
        };
        assert!(m.uses_sampler);
    }
    let Fields::Mimg(cd) = fields(&[0xf1a0_0f09, 0x0082_0402]) else {
        panic!("expected MIMG");
    };
    assert_eq!(
        cd.addr_dwords,
        2 + 2,
        "packed 16-bit 2D gradients plus coordinates"
    );
}

#[test]
fn decoding_continues_to_computed_jump_targets() {
    let words = [0xbe80_2004, 0xdead_0000, S_ENDPGM, 0xbe80_0380, S_ENDPGM];
    let scope = DecodeScope {
        entries: [12].into_iter().collect(),
        through_jumps: false,
    };
    let ir = Decoder::new().decode_scoped(&words, &scope).unwrap();
    assert_eq!(ir.code_words, 5);
    assert!(ir.instruction_at(12).is_some());
    let resync = [0xbe80_2004, 0x7e02_02ff, S_ENDPGM, S_ENDPGM];
    let scope = DecodeScope {
        entries: [8].into_iter().collect(),
        through_jumps: false,
    };
    let ir = Decoder::new().decode_scoped(&resync, &scope).unwrap();
    assert!(
        ir.instruction_at(8).is_some(),
        "data must not swallow a target"
    );
}

#[test]
fn open_jumps_decode_every_run_that_reads_as_code() {
    const V_MOV_V1_S0: u32 = 0x7e02_0200;
    let open = DecodeScope {
        entries: BTreeSet::new(),
        through_jumps: true,
    };
    let words = [
        0xbe80_2004, // s_setpc_b64 s[4:5]
        S_ENDPGM,    // pc 4
        V_MOV_V1_S0, // pc 8: a run ending in s_endpgm
        S_ENDPGM,    // pc 12
        V_MOV_V1_S0, // pc 16: a run cut by an undecodable word
        0xffff_ffff, // pc 20
        0x7e02_02ff, // pc 24: a run ending in a branch to its own start
        0x0000_0111, // pc 28
        0xbf82_fffd, // pc 32: s_branch to pc 24
        V_MOV_V1_S0, // pc 36: a run whose branch misses
        0xbf82_fffc, // pc 40: s_branch to pc 28, inside the literal
        V_MOV_V1_S0, // pc 44: a run running off the end
    ];
    let ir = Decoder::new().decode_scoped(&words, &open).unwrap();
    let pcs: Vec<u32> = ir.instructions.iter().map(|i| i.pc).collect();
    assert_eq!(pcs, [0, 4, 8, 12, 24, 32], "{pcs:?}");
    assert_eq!(ir.code_words, 9);
    assert_eq!(ir.instructions[5].branch_target, Some(24));

    // Data that decodes, then data that doesn't, between a branch and its target: the old
    // decoder took the first word for code and failed on the second.
    let gap = [0xbf82_0002, V_MOV_V1_S0, 0xffff_ffff, S_ENDPGM];
    let ir = Decoder::new().decode(&gap).unwrap();
    let pcs: Vec<u32> = ir.instructions.iter().map(|i| i.pc).collect();
    assert_eq!(pcs, [0, 12]);
}
