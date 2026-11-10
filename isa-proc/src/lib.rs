//! AMD shader ISA definitions generated at compile time from AMD's machine-readable GPU ISA
//! specification (decisions D18): the XML bundle in `xml/`, read with `zip` and parsed with
//! `roxmltree`.
//!
//! `isa!(rdna3_5 in crate::isa)` makes a module `rdna3_5` holding `ISA`, the architecture as the
//! XML defines it:
//! - `encodings`: name, bit count, the mask and values that identify the encoding (dwords, low
//!   dword first) and the microcode fields with their bit ranges (low bits first);
//! - `instructions`: name, aliases, flags, functional group and subgroups, and per encoding the
//!   condition, the opcode and the operands;
//! - `operands`, `conditions`, `groups`: the distinct operands, encoding conditions and
//!   functional (sub)group names, which instructions name by index — the XML repeats a few
//!   hundred distinct operands some 17 000 times per architecture;
//! - `data_formats`, and `operand_types` with their predefined values.
//!
//! The types are the caller's — `Isa`, `Encoding`, `Field`, `BitRange`, `Instruction`,
//! `InstructionEncoding`, `Operand`, `DataFormat`, `OperandType` — in the module named after
//! `in`. Instruction flags are bits: 0 branch, 1 conditional branch, 2 indirect branch, 3 program
//! terminator, 4 immediately executed. Descriptions are left out.

use proc_macro::{TokenStream, TokenTree};
use roxmltree::Node;
use std::collections::HashMap;
use std::fmt::Write;
use std::io::Read;

const BUNDLE: &[u8] = include_bytes!("../xml/AMD_GPU_MR_ISA_XML_2026_08_06.zip");
/// The host architectures (decisions D16); the PS5 guest ISA is decoded by hand.
const ARCHITECTURES: &[&str] = &["rdna3", "rdna3_5", "rdna4"];
/// `InstructionFlags` children, in the order of their flag bits.
const FLAGS: &[&str] = &[
    "IsBranch",
    "IsConditionalBranch",
    "IsIndirectBranch",
    "IsProgramTerminator",
    "IsImmediatelyExecuted",
];

fn fail(message: String) -> TokenStream {
    format!("compile_error!({message:?});").parse().unwrap()
}

/// `architecture in path`.
fn arguments(input: TokenStream) -> Result<(String, String), String> {
    let mut architecture = None;
    let mut path = String::new();
    let mut seen_in = false;
    for tree in input {
        let text = tree.to_string();
        match tree {
            TokenTree::Ident(_) if !seen_in && text == "in" => seen_in = true,
            TokenTree::Ident(_) if !seen_in => architecture = Some(text),
            _ if seen_in => path.push_str(&text),
            _ => return Err(format!("unexpected `{text}`; expected `architecture in path::to::types`")),
        }
    }
    match (architecture, seen_in && !path.is_empty()) {
        (Some(architecture), true) => Ok((architecture, path)),
        _ => Err("expected `architecture in path::to::types`".into()),
    }
}

fn read_xml(architecture: &str) -> Result<String, String> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(BUNDLE)).map_err(|error| error.to_string())?;
    let mut file = archive
        .by_name(&format!("amdgpu_isa_{architecture}.xml"))
        .map_err(|error| format!("amdgpu_isa_{architecture}.xml: {error}"))?;
    let mut text = String::new();
    file.read_to_string(&mut text).map_err(|error| error.to_string())?;
    Ok(text)
}

fn child<'a, 'i>(node: Node<'a, 'i>, tag: &str) -> Option<Node<'a, 'i>> {
    node.children().find(|candidate| candidate.has_tag_name(tag))
}

fn need<'a, 'i>(node: Node<'a, 'i>, tag: &str) -> Result<Node<'a, 'i>, String> {
    child(node, tag).ok_or_else(|| format!("<{}> has no <{tag}>", node.tag_name().name()))
}

fn children<'a, 'i>(node: Node<'a, 'i>, tag: &'static str) -> impl Iterator<Item = Node<'a, 'i>> {
    node.children().filter(move |candidate| candidate.has_tag_name(tag))
}

fn text<'a>(node: Node<'a, '_>, tag: &str) -> &'a str {
    child(node, tag).and_then(|found| found.text()).map(str::trim).unwrap_or("")
}

/// Texts of the `item` children of the `list` child of `node`.
fn texts<'a>(node: Node<'a, '_>, list: &str, item: &'static str) -> Vec<&'a str> {
    child(node, list)
        .map(|found| children(found, item).filter_map(|entry| entry.text()).map(str::trim).collect())
        .unwrap_or_default()
}

/// An element's number in its `Radix` (10 when absent).
fn number(node: Node) -> Result<i64, String> {
    let digits = node.text().map(str::trim).unwrap_or("");
    let radix = node.attribute("Radix").map_or(Ok(10), str::parse).map_err(|error| format!("Radix: {error}"))?;
    let (negative, digits) = digits.strip_prefix('-').map_or((false, digits), |rest| (true, rest));
    let digits = if radix == 16 { digits.trim_start_matches("0x") } else { digits };
    let value = i64::from_str_radix(digits, radix).map_err(|error| format!("{digits:?} in radix {radix}: {error}"))?;
    Ok(if negative { -value } else { value })
}

/// A binary string, most significant bit first, as dwords low first for `bits` bits.
fn dwords(binary: &str, bits: usize) -> Result<Vec<u32>, String> {
    let mut out = vec![0u32; bits.div_ceil(32)];
    for (index, digit) in binary.trim().bytes().rev().enumerate() {
        match digit {
            b'0' => {}
            b'1' if index < bits => out[index / 32] |= 1 << (index % 32),
            _ => return Err(format!("bad identifier bit {:?} at {index} of a {bits}-bit encoding", digit as char)),
        }
    }
    Ok(out)
}

fn list(items: impl IntoIterator<Item = String>) -> String {
    let mut out = String::from("&[");
    for item in items {
        out.push_str(&item);
        out.push(',');
    }
    out.push(']');
    out
}

fn string(value: &str) -> String {
    format!("{value:?}")
}

fn hex_dwords(values: &[u32]) -> String {
    list(values.iter().map(|value| format!("{value:#x}")))
}

/// Distinct values in first-seen order, by index.
#[derive(Default)]
struct Interner {
    index: HashMap<String, usize>,
    values: Vec<String>,
}

impl Interner {
    fn intern(&mut self, value: String) -> usize {
        if let Some(&index) = self.index.get(&value) {
            return index;
        }
        self.values.push(value.clone());
        self.index.insert(value, self.values.len() - 1);
        self.values.len() - 1
    }
}

fn encodings(path: &str, isa: Node) -> Result<(String, HashMap<String, usize>), String> {
    let mut entries = Vec::new();
    let mut by_name = HashMap::new();
    for encoding in children(need(isa, "Encodings")?, "Encoding") {
        let name = text(encoding, "EncodingName");
        let bits: usize = text(encoding, "BitCount").parse().map_err(|error| format!("{name} BitCount: {error}"))?;
        let mask = dwords(text(encoding, "EncodingIdentifierMask"), bits)?;
        let mut identifiers = Vec::new();
        if let Some(node) = child(encoding, "EncodingIdentifiers") {
            for identifier in children(node, "EncodingIdentifier") {
                identifiers.push(hex_dwords(&dwords(identifier.text().unwrap_or(""), bits)?));
            }
        }
        let mut fields = Vec::new();
        if let Some(format) = child(encoding, "MicrocodeFormat") {
            for field in format.descendants().filter(|node| node.has_tag_name("Field")) {
                let mut ranges = Vec::new();
                if let Some(layout) = child(field, "BitLayout") {
                    for range in children(layout, "Range") {
                        let order: i64 = range.attribute("Order").unwrap_or("0").parse().unwrap_or(0);
                        let offset: u16 = text(range, "BitOffset").parse().map_err(|error| format!("{name} BitOffset: {error}"))?;
                        let count: u16 = text(range, "BitCount").parse().map_err(|error| format!("{name} BitCount: {error}"))?;
                        ranges.push((order, offset, count));
                    }
                }
                ranges.sort_by_key(|&(order, _, _)| order);
                fields.push(format!(
                    "{path}::Field {{ name: {}, conditional: {}, ranges: {} }}",
                    string(text(field, "FieldName")),
                    field.attribute("IsConditional") == Some("true"),
                    list(ranges.iter().map(|&(_, offset, bits)| format!("{path}::BitRange {{ offset: {offset}, bits: {bits} }}"))),
                ));
            }
        }
        by_name.insert(name.to_string(), entries.len());
        entries.push(format!(
            "{path}::Encoding {{ name: {}, bits: {bits}, mask: {}, identifiers: {}, fields: {} }}",
            string(name),
            hex_dwords(&mask),
            list(identifiers),
            list(fields),
        ));
    }
    Ok((list(entries), by_name))
}

fn generate(input: TokenStream) -> Result<String, String> {
    let (architecture, path) = arguments(input)?;
    if !ARCHITECTURES.contains(&architecture.as_str()) {
        return Err(format!("isa!: no `{architecture}` (have {ARCHITECTURES:?})"));
    }
    let xml = read_xml(&architecture)?;
    let document = roxmltree::Document::parse(&xml).map_err(|error| format!("{architecture}: {error}"))?;
    let isa = need(document.root_element(), "ISA")?;
    let (encodings, encoding_index) = encodings(&path, isa)?;

    let (mut operands, mut conditions, mut groups) = (Interner::default(), Interner::default(), Interner::default());
    let mut instructions = Vec::new();
    for instruction in children(need(isa, "Instructions")?, "Instruction") {
        let name = text(instruction, "InstructionName");
        let mut flags = 0u8;
        if let Some(node) = child(instruction, "InstructionFlags") {
            for (bit, tag) in FLAGS.iter().enumerate() {
                if text(node, tag).eq_ignore_ascii_case("true") {
                    flags |= 1 << bit;
                }
            }
        }
        let (group, subgroups) = match child(instruction, "FunctionalGroup") {
            Some(node) => (text(node, "Name"), texts(node, "FunctionalSubgroups", "Subgroup")),
            None => ("", Vec::new()),
        };
        let group = groups.intern(group.to_string());
        let subgroups: Vec<String> = subgroups.into_iter().map(|subgroup| groups.intern(subgroup.to_string()).to_string()).collect();
        let mut forms = Vec::new();
        if let Some(node) = child(instruction, "InstructionEncodings") {
            for form in children(node, "InstructionEncoding") {
                let encoding_name = text(form, "EncodingName");
                let encoding = *encoding_index
                    .get(encoding_name)
                    .ok_or_else(|| format!("{name}: unknown encoding {encoding_name}"))?;
                let opcode = match child(form, "Opcode") {
                    Some(node) => number(node).map_err(|error| format!("{name} opcode: {error}"))?,
                    None => 0,
                };
                let opcode = u16::try_from(opcode).map_err(|_| format!("{name}: opcode {opcode} does not fit 16 bits"))?;
                let mut ordered = Vec::new();
                if let Some(nodes) = child(form, "Operands") {
                    for operand in children(nodes, "Operand") {
                        let order: i64 = operand.attribute("Order").unwrap_or("0").parse().unwrap_or(0);
                        let size: u16 = match text(operand, "OperandSize") {
                            "" => 0,
                            size => size.parse().map_err(|error| format!("{name} OperandSize {size:?}: {error}"))?,
                        };
                        let truth = |attribute: &str| operand.attribute(attribute) == Some("true");
                        let literal = format!(
                            "{path}::Operand {{ field: {}, data_format: {}, kind: {}, size: {size}, input: {}, output: {}, implicit: {}, in_microcode: {} }}",
                            string(text(operand, "FieldName")),
                            string(text(operand, "DataFormatName")),
                            string(text(operand, "OperandType")),
                            truth("Input"),
                            truth("Output"),
                            truth("IsImplicit"),
                            truth("IsBinaryMicrocodeRequired"),
                        );
                        ordered.push((order, operands.intern(literal)));
                    }
                }
                ordered.sort_by_key(|&(order, _)| order);
                let condition = conditions.intern(text(form, "EncodingCondition").to_string());
                forms.push(format!(
                    "{path}::InstructionEncoding {{ encoding: {encoding}, condition: {condition}, opcode: {opcode}, operands: {} }}",
                    list(ordered.into_iter().map(|(_, index)| index.to_string())),
                ));
            }
        }
        let aliases = texts(instruction, "AliasedInstructionNames", "InstructionName");
        instructions.push(format!(
            "{path}::Instruction {{ name: {}, aliases: {}, flags: {flags}, group: {group}, subgroups: {}, encodings: {} }}",
            string(name),
            list(aliases.into_iter().map(string)),
            list(subgroups),
            list(forms),
        ));
    }
    for (count, what) in [(operands.values.len(), "operands"), (conditions.values.len(), "conditions"), (groups.values.len(), "groups")] {
        if count > usize::from(if what == "operands" { u16::MAX } else { u16::from(u8::MAX) }) {
            return Err(format!("{architecture}: {count} distinct {what} do not fit their index type"));
        }
    }
    if encoding_index.len() > usize::from(u8::MAX) {
        return Err(format!("{architecture}: {} encodings do not fit u8", encoding_index.len()));
    }

    let formats = children(need(isa, "DataFormats")?, "DataFormat").map(|format| {
        format!(
            "{path}::DataFormat {{ name: {}, data_type: {}, bits: {} }}",
            string(text(format, "DataFormatName")),
            string(text(format, "DataType")),
            text(format, "BitCount").parse::<u16>().unwrap_or(0),
        )
    });
    let mut types = Vec::new();
    for ty in children(need(isa, "OperandTypes")?, "OperandType") {
        let mut values = Vec::new();
        if let Some(predefined) = child(ty, "OperandPredefinedValues") {
            for value in children(predefined, "PredefinedValue") {
                if let Some(Ok(number)) = child(value, "Value").map(number) {
                    values.push(format!("({}, {number})", string(text(value, "Name"))));
                }
            }
        }
        types.push(format!(
            "{path}::OperandType {{ name: {}, partitioned: {}, values: {} }}",
            string(text(ty, "OperandTypeName")),
            ty.attribute("IsPartitioned") == Some("true"),
            list(values),
        ));
    }

    let name = text(need(isa, "Architecture")?, "ArchitectureName");
    let mut out = String::with_capacity(2 << 20);
    let _ = write!(
        out,
        "pub mod {architecture} {{ pub static ISA: {path}::Isa = {path}::Isa {{ architecture: {}, encodings: {encodings}, instructions: {}, operands: {}, conditions: {}, groups: {}, data_formats: {}, operand_types: {} }}; }}",
        string(name),
        list(instructions),
        list(operands.values),
        list(conditions.values.iter().map(|condition| string(condition))),
        list(groups.values.iter().map(|group| string(group))),
        list(formats),
        list(types),
    );
    Ok(out)
}

/// The ISA of one architecture.
#[proc_macro]
pub fn isa(input: TokenStream) -> TokenStream {
    match generate(input) {
        Ok(code) => code.parse().unwrap_or_else(|error| fail(format!("isa!: generated code does not parse: {error}"))),
        Err(message) => fail(format!("isa!: {message}")),
    }
}
