//! The host ISA definitions `isa-proc` generates from AMD's XML (decisions D18) against the
//! opcode tables `ir::isa_gen` derived from LLVM and the same XML bundle: every opcode the tables
//! name from the XML must be the generated instruction of that name at that opcode.

use shader::ir::isa_gen::{self, Encoding, Opcode};
use shader::isa::{self, Isa};

/// The tables' encoding of each XML encoding (`driver/tools/isa-tables/amd_isa_xml.py`).
fn table_encoding(xml: &str) -> Option<Encoding> {
    use Encoding::*;
    Some(match xml {
        "ENC_SOP1" | "SOP1_INST_LITERAL" => Sop1,
        "ENC_SOP2" | "SOP2_INST_LITERAL" => Sop2,
        "ENC_SOPK" | "SOPK_INST_LITERAL" => Sopk,
        "ENC_SOPC" | "SOPC_INST_LITERAL" => Sopc,
        "ENC_SOPP" => Sopp,
        "ENC_VOP1" | "VOP1_INST_LITERAL" | "VOP1_VOP_DPP16" | "VOP1_VOP_DPP8" | "VOP1_VOP_SDWA" => Vop1,
        "ENC_VOP2" | "VOP2_INST_LITERAL" | "VOP2_VOP_DPP16" | "VOP2_VOP_DPP8" | "VOP2_VOP_SDWA"
        | "VOP2_VOP_SDWA_SDST_ENC" => Vop2,
        "ENC_VOPC" | "VOPC_INST_LITERAL" | "VOPC_VOP_DPP16" | "VOPC_VOP_DPP8" | "VOPC_VOP_SDWA_SDST_ENC" => Vopc,
        "ENC_VOP3" | "VOP3_INST_LITERAL" | "VOP3_VOP_DPP16" | "VOP3_VOP_DPP8" | "VOP3_SDST_ENC"
        | "VOP3_SDST_ENC_INST_LITERAL" | "VOP3_SDST_ENC_VOP_DPP16" | "VOP3_SDST_ENC_VOP_DPP8" => Vop3,
        "ENC_VOP3P" | "VOP3P_INST_LITERAL" | "VOP3P_VOP_DPP16" | "VOP3P_VOP_DPP8" => Vop3p,
        "ENC_VINTRP" => Vintrp,
        "ENC_VINTERP" => Vinterp,
        "ENC_LDSDIR" | "ENC_VDSDIR" => Ldsdir,
        "ENC_SMEM" => Smem,
        "ENC_MUBUF" => Mubuf,
        "ENC_MTBUF" => Mtbuf,
        "ENC_VBUFFER" => Vbuffer,
        "ENC_FLAT" | "ENC_VFLAT" => Flat,
        "ENC_FLAT_GLBL" | "ENC_FLAT_GLOBAL" | "ENC_VGLOBAL" => Global,
        "ENC_FLAT_SCRATCH" | "ENC_VSCRATCH" => Scratch,
        "ENC_DS" | "ENC_VDS" => Ds,
        "ENC_MIMG" | "MIMG_NSA1" | "MIMG_NSA2" | "MIMG_NSA3" => Mimg,
        "ENC_VIMAGE" => Vimage,
        "ENC_VSAMPLE" => Vsample,
        "ENC_EXP" | "ENC_VEXPORT" => Exp,
        _ => return None,
    })
}

/// Rows of `table` whose XML names none of the generated `xml` ISAs gives that opcode.
fn mismatches(xml: &[&Isa], rows: &[Opcode]) -> Vec<String> {
    rows.iter()
        .filter(|row| !row.xml.is_empty())
        .filter(|row| {
            !xml.iter().any(|isa| {
                isa.instructions.iter().any(|instruction| {
                    let named = std::iter::once(instruction.name)
                        .chain(instruction.aliases.iter().copied())
                        .any(|name| row.xml.contains(&name.to_ascii_lowercase().as_str()));
                    named
                        && instruction.encodings.iter().any(|form| {
                            table_encoding(isa.encoding_of(form).name) == Some(row.encoding) && form.opcode == row.opcode
                        })
                })
            })
        })
        .map(|row| format!("{:?} {:#x} {:?}", row.encoding, row.opcode, row.xml))
        .collect()
}

#[test]
fn the_xml_definitions_agree_with_the_opcode_tables() {
    let pairs: [(&[&Isa], &isa_gen::Isa, &[Opcode]); 2] = [
        (&[&isa::rdna3_5::ISA], &isa_gen::gfx1151::ISA, isa_gen::gfx1151::XML_ONLY),
        (&[&isa::rdna4::ISA], &isa_gen::gfx1200::ISA, isa_gen::gfx1200::XML_ONLY),
    ];
    for (xml, table, xml_only) in pairs {
        let mut wrong = mismatches(xml, table.opcodes);
        wrong.extend(mismatches(xml, xml_only));
        assert!(wrong.is_empty(), "{} vs {}: {} rows differ: {:?}", xml[0].architecture, table.cpu, wrong.len(), &wrong[..wrong.len().min(10)]);
    }
    assert_eq!(isa::rdna3_5::ISA.architecture, "AMD RDNA 3.5");
    let sop1 = isa::rdna3_5::ISA.encoding("ENC_SOP1").unwrap();
    assert_eq!((sop1.bits, sop1.mask), (32, &[0xff80_ff00][..]));
    let rdna3_5 = &isa::rdna3_5::ISA;
    let mov = rdna3_5.instruction("S_MOV_B32").unwrap();
    assert_eq!((rdna3_5.encoding_of(&mov.encodings[0]).name, mov.encodings[0].opcode), ("ENC_SOP1", 0));
    let fields: Vec<&str> = rdna3_5.operands_of(&mov.encodings[0]).map(|operand| operand.field).collect();
    assert_eq!(fields, ["SDST", "SSRC0"]);
}
