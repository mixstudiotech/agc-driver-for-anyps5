//! Checks the hand-written decode and translation tables against the opcode tables generated
//! from LLVM and the AMD ISA XML in `isa_gen`.

use super::isa_gen::{self, Encoding, Encoding::*, Isa, Opcode, OPERAND_SWAPS};
use super::memops::{self, MemOp};
use super::opcodes::{self, flag, OpInfo, VopcInfo, Xlat};
use crate::TargetIsa;
use std::collections::BTreeSet;

/// PS5 image forms only AnyPS5's decoder (`RdnaImageOpDecoder.cpp`) defines: neither LLVM nor
/// the AMD XML lists them. They differ from the plain forms only in sampler dword 3 bits 12-29,
/// which no target reads the same way (`img::ADJUST`).
const ANYPS5_ONLY: [(Encoding, u16); 8] = [
    (Mimg, 0xa0),
    (Mimg, 0xa1),
    (Mimg, 0xa5),
    (Mimg, 0xa8),
    (Mimg, 0xb0),
    (Mimg, 0xb8),
    (Mimg, 0xc8),
    (Mimg, 0xd8),
];

/// Translations to a differently named instruction that computes the same result for every
/// input the PS5 instruction accepts.
const SUBSTITUTIONS: &[(&str, &str, &str)] = &[
    // The low 32 bits of a product do not depend on the signedness of its operands.
    ("v_mul_lo_i32", "v_mul_lo_u32", ALL),
    // gfx12 *_num_* minimum and maximum (IEEE 754-2019 minimumNumber/maximumNumber) differ
    // from gfx10 only for signaling NaN inputs; see `opcodes::VOP2`.
    ("v_min_f32", "v_min_num_f32", "gfx1200"),
    ("v_max_f32", "v_max_num_f32", "gfx1200"),
    ("v_min_f16", "v_min_num_f16", "gfx1200"),
    ("v_max_f16", "v_max_num_f16", "gfx1200"),
    ("v_min_f64", "v_min_num_f64", "gfx1200"),
    ("v_max_f64", "v_max_num_f64", "gfx1200"),
    ("v_min3_f32", "v_min3_num_f32", "gfx1200"),
    ("v_max3_f32", "v_max3_num_f32", "gfx1200"),
    ("v_min3_f16", "v_min3_num_f16", "gfx1200"),
    ("v_max3_f16", "v_max3_num_f16", "gfx1200"),
    ("v_med3_f32", "v_med3_num_f32", "gfx1200"),
    ("v_med3_f16", "v_med3_num_f16", "gfx1200"),
    ("v_pk_min_f16", "v_pk_min_num_f16", "gfx1200"),
    ("v_pk_max_f16", "v_pk_max_num_f16", "gfx1200"),
    // PS5 `_a` forms run as their plain forms once the sampler's lane bits are cleared.
    ("image_sample_a", "image_sample", ALL),
    ("image_sample_cl_a", "image_sample_cl", ALL),
    ("image_sample_b_a", "image_sample_b", ALL),
    ("image_sample_c_a", "image_sample_c", ALL),
    ("image_sample_o_a", "image_sample_o", ALL),
    ("image_sample_c_o_a", "image_sample_c_o", ALL),
    ("image_gather4_c_a", "image_gather4_c", ALL),
    ("image_gather4_c_o_a", "image_gather4_c_o", ALL),
];

const ALL: &str = "gfx1030 gfx1100 gfx1151 gfx1200";

/// What the crate emits for a PS5 instruction on one target.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Emit {
    Op(Encoding, u16),
    /// A lowering sequence.
    Rule,
    Drop,
    No,
}

/// One encoding and opcode the crate decodes, with its translation for each of [`targets`].
#[derive(Clone, Debug)]
struct Entry {
    encoding: Encoding,
    opcode: u16,
    name: String,
    emits: [Emit; 4],
}

fn targets() -> [(TargetIsa, &'static Isa); 4] {
    [
        (TargetIsa::Gfx103, &isa_gen::gfx1030::ISA),
        (TargetIsa::Gfx11, &isa_gen::gfx1100::ISA),
        (TargetIsa::Gfx115, &isa_gen::gfx1151::ISA),
        (TargetIsa::Gfx12, &isa_gen::gfx1200::ISA),
    ]
}

fn find(table: &'static [Opcode], encoding: Encoding, opcode: u16) -> Option<&'static Opcode> {
    table
        .iter()
        .find(|o| o.encoding == encoding && o.opcode == opcode)
}

fn names(op: &Opcode) -> impl Iterator<Item = &'static str> {
    op.mnemonics.iter().chain(op.records).chain(op.xml).copied()
}

fn named(op: &Opcode, name: &str) -> bool {
    names(op).any(|n| n == name)
}

fn emit(x: Xlat, encoding: Encoding) -> Emit {
    match x {
        Xlat::Same => unreachable!("translate() resolves Same"),
        Xlat::Op(op) => Emit::Op(encoding, op),
        Xlat::Rule(_) => Emit::Rule,
        Xlat::Drop => Emit::Drop,
        Xlat::No => Emit::No,
    }
}

fn entry(encoding: Encoding, opcode: u16, name: &str, f: impl Fn(TargetIsa) -> Emit) -> Entry {
    Entry {
        encoding,
        opcode,
        name: name.to_string(),
        emits: targets().map(|(target, _)| f(target)),
    }
}

fn op_entries(out: &mut Vec<Entry>) {
    let tables: [(&[OpInfo], Encoding); 9] = [
        (opcodes::SOP1, Sop1),
        (opcodes::SOP2, Sop2),
        (opcodes::SOPK, Sopk),
        (opcodes::SOPC, Sopc),
        (opcodes::SOPP, Sopp),
        (opcodes::VOP1, Vop1),
        (opcodes::VOP2, Vop2),
        (opcodes::VOP3, Vop3),
        (opcodes::VOP3P, Vop3p),
    ];
    for (table, encoding) in tables {
        for info in table {
            out.push(entry(encoding, info.op, info.name, |t| {
                emit(info.translate(t), encoding)
            }));
            let promoted = match encoding {
                Vop1 => opcodes::VOP3_VOP1_BASE,
                Vop2 => opcodes::VOP3_VOP2_BASE,
                _ => continue,
            };
            if info.flags & flag::NO_VOP3 == 0 {
                out.push(entry(Vop3, promoted + info.op, info.name, |t| {
                    match info.translate(t) {
                        Xlat::Op(op) => Emit::Op(Vop3, promoted + op),
                        x => emit(x, Vop3),
                    }
                }));
            }
        }
    }
}

fn vopc_emit(info: VopcInfo, op: u16, encoding: Encoding, target: TargetIsa) -> Emit {
    match target {
        TargetIsa::Gfx12 if info.is_constant_result().is_some() => Emit::Rule,
        _ if target.is_gfx11_plus() => Emit::Op(encoding, info.gfx11_opcode()),
        _ => Emit::Op(encoding, op),
    }
}

fn vopc_entries(out: &mut Vec<Entry>) {
    for op in 0..=0xff {
        if let Some(info) = VopcInfo::decode(op) {
            for encoding in [Vopc, Vop3] {
                out.push(entry(encoding, op, &info.name(), |t| {
                    vopc_emit(info, op, encoding, t)
                }));
            }
        }
    }
}

fn interp_and_export_entries(out: &mut Vec<Entry>) {
    for (op, name) in [
        (opcodes::VINTRP_P1, "v_interp_p1_f32"),
        (opcodes::VINTRP_P2, "v_interp_p2_f32"),
        (opcodes::VINTRP_MOV, "v_interp_mov_f32"),
    ] {
        out.push(entry(Vintrp, op, name, |t| match t {
            TargetIsa::Gfx103 => Emit::Op(Vintrp, op),
            _ => Emit::Rule,
        }));
    }
    out.push(entry(Exp, 0, "exp", |_| Emit::Op(Exp, 0)));
}

fn memory_entries(out: &mut Vec<Entry>) {
    for op in memops::SMEM {
        out.push(entry(Smem, op.op, op.name, |t| emit(op.translate(t), Smem)));
    }
    let buffer = |t: TargetIsa| if t.is_gfx12() { Vbuffer } else { Mubuf };
    for op in memops::VMEM_COMMON.iter().chain(memops::MUBUF_ONLY) {
        let name = format!("buffer_{}", op.name);
        out.push(entry(Mubuf, op.op, &name, |t| {
            emit(op.translate(t), buffer(t))
        }));
    }
    for op in memops::MTBUF {
        out.push(entry(Mtbuf, op.op, op.name, |t| match op.translate(t) {
            Xlat::Op(n) if t.is_gfx12() => Emit::Op(Vbuffer, 0x80 | n),
            x => emit(x, Mtbuf),
        }));
    }
    for op in memops::VMEM_COMMON.iter().chain(memops::FLAT_ONLY) {
        for (segment, prefix) in [(Flat, "flat"), (Global, "global"), (Scratch, "scratch")] {
            if flat_segment_decodes(op, segment) {
                let name = format!("{prefix}_{}", op.name);
                out.push(entry(segment, op.op, &name, |t| {
                    emit(op.translate(t), segment)
                }));
            }
        }
    }
    for op in memops::DS {
        out.push(entry(Ds, op.op, op.name, |t| emit(op.translate(t), Ds)));
    }
    for op in memops::MIMG {
        out.push(entry(Mimg, op.op, op.name, |t| match op.translate(t) {
            Xlat::Op(n) if t.is_gfx12() && memops::gfx12_vsample(n, op.uses_sampler()) => {
                Emit::Op(Vsample, n)
            }
            x if t.is_gfx12() => emit(x, Vimage),
            x => emit(x, Mimg),
        }));
    }
}

/// The FLAT segment rules of the decoder.
fn flat_segment_decodes(op: &MemOp, segment: Encoding) -> bool {
    let global_only =
        memops::FLAT_ONLY.iter().any(|o| o.op == op.op) || op.op == memops::FLAT_ATOMIC_CSUB;
    let atomic = op.kind == super::MemKind::Atomic;
    !(global_only && segment != Global || atomic && segment == Scratch)
}

fn entries() -> Vec<Entry> {
    let mut out = Vec::new();
    op_entries(&mut out);
    vopc_entries(&mut out);
    interp_and_export_entries(&mut out);
    memory_entries(&mut out);
    out
}

fn report(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn generated_tables_record_their_sources() {
    assert_eq!(
        isa_gen::LLVM_COMMIT,
        "3c1efa15830c5c60dbc45a3d3d47a1cacf5aba57"
    );
    assert_eq!(
        isa_gen::XML_SHA256,
        "82404F1126761B7877595B622AFA7E1F311F2F41E89A3ABE9AAF8AD045C082E2"
    );
    let cpus: Vec<&str> = targets().iter().map(|(_, isa)| isa.cpu).collect();
    assert_eq!(isa_gen::ps5::ISA.cpu, "gfx1013");
    assert_eq!(cpus, ["gfx1030", "gfx1100", "gfx1151", "gfx1200"]);
}

#[test]
fn every_decoded_ps5_opcode_is_defined_by_an_oracle() {
    let ps5 = &isa_gen::ps5::ISA;
    let mut failures = Vec::new();
    for e in entries() {
        let defined = find(ps5.opcodes, e.encoding, e.opcode)
            .or_else(|| find(isa_gen::ps5::XML_ONLY, e.encoding, e.opcode));
        match defined {
            Some(op) if named(op, &e.name) => {}
            Some(op) => failures.push(format!(
                "{:?} {:#x}: decoded as {}, the oracles name it {:?}",
                e.encoding,
                e.opcode,
                e.name,
                names(op).collect::<BTreeSet<_>>()
            )),
            None if ANYPS5_ONLY.contains(&(e.encoding, e.opcode)) => {}
            None => failures.push(format!(
                "{:?} {:#x}: {} is decoded but neither oracle defines it",
                e.encoding, e.opcode, e.name
            )),
        }
    }
    report(&failures);
}

#[test]
fn every_ps5_opcode_both_oracles_define_is_decoded() {
    let decoded: BTreeSet<(Encoding, u16)> =
        entries().iter().map(|e| (e.encoding, e.opcode)).collect();
    let failures: Vec<String> = isa_gen::ps5::ISA
        .opcodes
        .iter()
        .filter(|op| !op.xml.is_empty() && !decoded.contains(&(op.encoding, op.opcode)))
        .map(|op| {
            format!(
                "{:?} {:#x} {:?} is not decoded",
                op.encoding, op.opcode, op.xml
            )
        })
        .collect();
    report(&failures);
}

#[test]
fn every_translation_exists_on_its_target() {
    let mut failures = Vec::new();
    for e in entries() {
        for (k, (target, isa)) in targets().into_iter().enumerate() {
            let Emit::Op(encoding, opcode) = e.emits[k] else {
                continue;
            };
            let xml_only = || {
                XML_ONLY_TRANSLATIONS.contains(&(isa.cpu, e.name.as_str()))
                    && find(xml_only_table(target), encoding, opcode)
                        .is_some_and(|op| named(op, &e.name))
            };
            let Some(op) = find(isa.opcodes, encoding, opcode) else {
                if xml_only() {
                    continue;
                }
                failures.push(format!(
                    "{}: {} becomes {encoding:?} {opcode:#x}, which {target:?} lacks",
                    isa.cpu, e.name
                ));
                continue;
            };
            let substitute = SUBSTITUTIONS.iter().any(|(from, to, cpus)| {
                *from == e.name && named(op, to) && cpus.split(' ').any(|c| c == isa.cpu)
            });
            if !named(op, &e.name) && !substitute {
                failures.push(format!(
                    "{}: {} becomes {encoding:?} {opcode:#x} {:?}",
                    isa.cpu,
                    e.name,
                    names(op).collect::<BTreeSet<_>>()
                ));
            }
        }
    }
    report(&failures);
}

/// Translations to opcodes only the target's AMD XML defines. RDNA2 lists the packed
/// half-precision gathers, which LLVM has never implemented, at the same MIMG opcodes as the
/// PS5.
const XML_ONLY_TRANSLATIONS: &[(&str, &str)] = &[
    ("gfx1030", "image_gather4h_pck"),
    ("gfx1030", "image_gather8h_pck"),
];

fn xml_only_table(target: TargetIsa) -> &'static [Opcode] {
    match target {
        TargetIsa::Gfx103 => isa_gen::gfx1030::XML_ONLY,
        TargetIsa::Gfx11 => isa_gen::gfx1100::XML_ONLY,
        TargetIsa::Gfx115 => isa_gen::gfx1151::XML_ONLY,
        _ => isa_gen::gfx1200::XML_ONLY,
    }
}

/// Instructions a target still has under the PS5 name that the crate deliberately rejects.
const REJECTED_ALTHOUGH_PRESENT: &[(&str, &str)] = &[];

/// Hints whose omission changes no result, dropped even where the target has them. gfx11 split
/// gfx10's single VMEM hard-clause type into load, store, atomic and sample clauses, so a
/// clause valid on the PS5 can be invalid there; the prefetch distance is a hint tuned for the
/// PS5's instruction stream.
const DROPPED_ALTHOUGH_PRESENT: &[(&str, &str)] = &[
    ("gfx1100", "s_clause"),
    ("gfx1151", "s_clause"),
    ("gfx1200", "s_clause"),
    ("gfx1100", "s_inst_prefetch"),
    ("gfx1151", "s_inst_prefetch"),
];

#[test]
fn removed_and_dropped_marks_agree_with_the_targets() {
    let mut failures = Vec::new();
    for e in entries() {
        for (k, (_, isa)) in targets().into_iter().enumerate() {
            let allowed = match e.emits[k] {
                Emit::No => REJECTED_ALTHOUGH_PRESENT,
                Emit::Drop => DROPPED_ALTHOUGH_PRESENT,
                _ => continue,
            };
            let present: Vec<String> = isa
                .opcodes
                .iter()
                .filter(|op| named(op, &e.name))
                .map(|op| format!("{:?} {:#x}", op.encoding, op.opcode))
                .collect();
            if !present.is_empty() && !allowed.contains(&(isa.cpu, e.name.as_str())) {
                failures.push(format!(
                    "{}: {} is marked {:?} but the target has it at {}",
                    isa.cpu,
                    e.name,
                    e.emits[k],
                    present.join(", ")
                ));
            }
        }
    }
    report(&failures);
}

#[test]
fn vopc_decoding_and_renumbering_are_exhaustive() {
    let ps5 = &isa_gen::ps5::ISA;
    let mut failures = Vec::new();
    for op in 0..=0xffu16 {
        let info = VopcInfo::decode(op);
        for encoding in [Vopc, Vop3] {
            let oracle = find(ps5.opcodes, encoding, op);
            match (info, oracle) {
                (None, None) => {}
                (Some(info), Some(o)) if named(o, &info.name()) => {}
                (info, oracle) => failures.push(format!(
                    "gfx1013 {encoding:?} {op:#x}: decoded as {:?}, LLVM has {:?}",
                    info.map(VopcInfo::name),
                    oracle.map(|o| o.mnemonics)
                )),
            }
        }
        let Some(info) = info else {
            continue;
        };
        for (target, isa) in targets() {
            for encoding in [Vopc, Vop3] {
                match vopc_emit(info, op, encoding, target) {
                    Emit::Op(_, n) => match find(isa.opcodes, encoding, n) {
                        Some(o) if named(o, &info.name()) => {}
                        o => failures.push(format!(
                            "{}: {} becomes {encoding:?} {n:#x} {:?}",
                            isa.cpu,
                            info.name(),
                            o.map(|o| o.mnemonics)
                        )),
                    },
                    _ => {
                        if let Some(o) = isa.opcodes.iter().find(|o| named(o, &info.name())) {
                            failures.push(format!(
                                "{}: {} is computed although the target has it at {:?} {:#x}",
                                isa.cpu,
                                info.name(),
                                o.encoding,
                                o.opcode
                            ));
                        }
                    }
                }
            }
        }
    }
    report(&failures);
}

#[test]
fn operand_swaps_match_the_generated_list() {
    for (target, isa) in targets() {
        let swapped: BTreeSet<(u16, Emit)> = memops::DS
            .iter()
            .filter(|op| op.swaps_data(target))
            .map(|op| (op.op, emit(op.translate(target), Ds)))
            .filter(|(_, emits)| matches!(emits, Emit::Op(..)))
            .collect();
        let expected: BTreeSet<(u16, Emit)> = OPERAND_SWAPS
            .iter()
            .filter(|s| s.cpu == isa.cpu)
            .inspect(|s| assert_eq!(s.encoding, Ds))
            .map(|s| (s.ps5_opcode, Emit::Op(Ds, s.target_opcode)))
            .collect();
        assert_eq!(swapped, expected, "{}", isa.cpu);
    }
}
