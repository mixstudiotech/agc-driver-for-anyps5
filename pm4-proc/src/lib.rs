//! AMD register, resource descriptor and PM4 packet definitions, generated at compile time from
//! the copy of Mesa AMD's in `mesa/` (decisions D16, D17).
//!
//! - `registers!(gfx115 in crate::amd::defs)` makes a module `gfx115` with one module per
//!   register (`OFFSET`, its byte address as Mesa's `mm` names it, and its type's fields), plus
//!   `types` (the fields of every register type) and `enums`. Chips: `gfx10` (the PS5 guest),
//!   `gfx115` (RDNA3.5), `gfx12` (RDNA4), and their descriptors `gfx10_rsrc`, `gfx115_rsrc`,
//!   `gfx12_rsrc`.
//! - `packets!(in crate::amd::defs)` makes `pkt3` (the type-3 opcodes of `sid.h`),
//!   `pkt3::dwords` (the packet dwords `pkt3.json` defines, with `OPCODE` and `DWORD`) and `sid`
//!   (every other integer macro of `sid.h` that evaluates: constants as `u32`, one-parameter
//!   field macros as fields).
//! - `cp_packets!(gfx11 in crate::amd::defs)` makes `cp_gfx11` (or `cp_gfx12`) from AMD's CP
//!   packet tables: a module per CP engine (`pfp`, `meg`, `mec`) and in it one per packet, with
//!   `OPCODE`, `WORDS` (the dwords the table describes, the header included), its fields as the
//!   caller's `PacketField` (`PacketField::new(dword, shift, width)`, dword 0 the header) and
//!   `enums`.
//!
//! Fields are the caller's `Field` type, built with `Field::new(shift, width)`.

mod cexpr;

use proc_macro::{TokenStream, TokenTree};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

const SOURCES: &[(&str, &str)] = &[
    ("gfx10", include_str!("../mesa/gfx10.json")),
    ("gfx115", include_str!("../mesa/gfx115.json")),
    ("gfx12", include_str!("../mesa/gfx12.json")),
    ("gfx10_rsrc", include_str!("../mesa/gfx10-rsrc.json")),
    ("gfx115_rsrc", include_str!("../mesa/gfx11-rsrc.json")),
    ("gfx12_rsrc", include_str!("../mesa/gfx12-rsrc.json")),
];
const PKT3_JSON: &str = include_str!("../mesa/pkt3.json");
const SID_H: &str = include_str!("../mesa/sid.h");

const ALLOW: &str = "#[allow(non_snake_case, non_upper_case_globals, non_camel_case_types, dead_code, clippy::all)]";

/// `chip in path` or `in path`.
fn arguments(input: TokenStream) -> Result<(Option<String>, String), String> {
    let mut chip = None;
    let mut path = String::new();
    let mut seen_in = false;
    for tree in input {
        let text = tree.to_string();
        match tree {
            TokenTree::Ident(_) if !seen_in && text == "in" => seen_in = true,
            TokenTree::Ident(_) if !seen_in => chip = Some(text),
            _ if seen_in => path.push_str(&text),
            _ => return Err(format!("unexpected `{text}`; expected `chip in path::to::types`")),
        }
    }
    if !seen_in || path.is_empty() {
        return Err("expected `in path::to::types` naming the module that defines `Field`".into());
    }
    Ok((chip, path))
}

const KEYWORDS: &[&str] = &[
    "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "do", "dyn",
    "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl", "in", "let", "loop",
    "macro", "match", "mod", "move", "mut", "override", "priv", "pub", "ref", "return", "static",
    "struct", "trait", "true", "try", "type", "typeof", "unsafe", "unsized", "use", "virtual",
    "where", "while", "yield",
];

/// `name` as a Rust identifier.
fn ident(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    if matches!(out.as_str(), "self" | "super" | "crate" | "Self" | "_") {
        out.push('_');
    } else if KEYWORDS.contains(&out.as_str()) {
        out.insert_str(0, "r#");
    }
    out
}

fn fail(message: String) -> TokenStream {
    format!("compile_error!({message:?});").parse().unwrap()
}

fn field(path: &str, shift: u64, width: u64) -> String {
    format!("{path}::Field::new({shift}, {width})")
}

fn array(value: Option<&Value>) -> &[Value] {
    value.and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

/// Field constants of a regdb register type.
fn type_fields(out: &mut String, path: &str, ty: &Value) {
    let mut seen = HashSet::new();
    for entry in array(ty.get("fields")) {
        let (Some(name), bits) = (entry.get("name").and_then(Value::as_str), array(entry.get("bits"))) else {
            continue;
        };
        let (Some(low), Some(high)) = (bits.first().and_then(Value::as_u64), bits.get(1).and_then(Value::as_u64)) else {
            continue;
        };
        if high < low || !seen.insert(name.to_string()) {
            continue;
        }
        let _ = writeln!(out, "pub const {}: {path}::Field = {};", ident(name), field(path, low, high - low + 1));
    }
}

/// Enum value constants of a regdb database.
fn enums(out: &mut String, database: &Value) {
    out.push_str("pub mod enums {\n");
    let empty = Map::new();
    for (name, value) in database.get("enums").and_then(Value::as_object).unwrap_or(&empty) {
        let _ = writeln!(out, "pub mod {} {{", ident(name));
        let mut seen = HashSet::new();
        for entry in array(value.get("entries")) {
            let (Some(entry_name), Some(number)) = (
                entry.get("name").and_then(Value::as_str),
                entry.get("value").and_then(Value::as_u64),
            ) else {
                continue;
            };
            if seen.insert(entry_name.to_string()) {
                let _ = writeln!(out, "pub const {}: u32 = {number:#x};", ident(entry_name));
            }
        }
        out.push_str("}\n");
    }
    out.push_str("}\n");
}

/// A mapping's name and `map.at`.
fn mapping(mapping: &Value) -> Option<(&str, u64)> {
    Some((
        mapping.get("name")?.as_str()?,
        mapping.get("map")?.get("at")?.as_u64()?,
    ))
}

/// The registers of one chip.
#[proc_macro]
pub fn registers(input: TokenStream) -> TokenStream {
    let (chip, path) = match arguments(input) {
        Ok((Some(chip), path)) => (chip, path),
        Ok((None, _)) => return fail("registers!: name a chip".into()),
        Err(message) => return fail(format!("registers!: {message}")),
    };
    let Some(&(_, text)) = SOURCES.iter().find(|(name, _)| *name == chip) else {
        let known: Vec<&str> = SOURCES.iter().map(|(name, _)| *name).collect();
        return fail(format!("registers!: no Mesa register database for `{chip}` (have {known:?})"));
    };
    let database: Value = match serde_json::from_str(text) {
        Ok(database) => database,
        Err(error) => return fail(format!("registers!({chip}): {error}")),
    };
    let empty = Map::new();
    let types = database.get("register_types").and_then(Value::as_object).unwrap_or(&empty);
    let mut out = String::new();
    let _ = writeln!(out, "{ALLOW}\npub mod {chip} {{");
    out.push_str("pub mod types {\n");
    for (name, ty) in types {
        let _ = writeln!(out, "pub mod {} {{", ident(name));
        type_fields(&mut out, &path, ty);
        out.push_str("}\n");
    }
    out.push_str("}\n");
    enums(&mut out, &database);
    let mut seen = HashSet::new();
    for entry in array(database.get("register_mappings")) {
        let Some((name, at)) = mapping(entry) else { continue };
        if !seen.insert(name.to_string()) {
            continue;
        }
        let _ = writeln!(out, "pub mod {} {{\npub const OFFSET: u32 = {at:#x};", ident(name));
        if let Some(ty) = entry.get("type_ref").and_then(Value::as_str).filter(|ty| types.contains_key(*ty)) {
            let _ = writeln!(out, "pub use super::types::{}::*;", ident(ty));
        }
        out.push_str("}\n");
    }
    out.push_str("}\n");
    out.parse()
        .unwrap_or_else(|error| fail(format!("registers!({chip}): generated code does not parse: {error}")))
}

/// The PM4 type-3 opcodes, packet dwords and `sid.h` constants.
#[proc_macro]
pub fn packets(input: TokenStream) -> TokenStream {
    let path = match arguments(input) {
        Ok((None, path)) => path,
        Ok((Some(chip), _)) => return fail(format!("packets!: packets are not per chip (got `{chip}`)")),
        Err(message) => return fail(format!("packets!: {message}")),
    };
    let defines = cexpr::defines(SID_H);
    // The first definition of a name wins, as the generated constants keep the first.
    let macros: HashMap<String, cexpr::Macro> = defines
        .iter()
        .rev()
        .map(|(name, definition, _)| (name.clone(), definition.clone()))
        .collect();
    let evaluator = cexpr::Evaluator { macros: &macros };
    let mut out = String::new();
    let _ = writeln!(out, "{ALLOW}\npub mod pkt3 {{");
    let mut seen = HashSet::new();
    for (name, definition, _) in &defines {
        let (Some(opcode), cexpr::Macro::Object(body)) = (name.strip_prefix("PKT3_"), definition) else {
            continue;
        };
        if let Some(value) = evaluator.eval(body, &[]).filter(|&value| value <= 0xff) {
            if seen.insert(opcode.to_string()) {
                let _ = writeln!(out, "pub const {}: u32 = {value:#x};", ident(opcode));
            }
        }
    }
    let pkt3: Value = match serde_json::from_str(PKT3_JSON) {
        Ok(pkt3) => pkt3,
        Err(error) => return fail(format!("packets!: pkt3.json: {error}")),
    };
    let empty = Map::new();
    let types = pkt3.get("register_types").and_then(Value::as_object).unwrap_or(&empty);
    out.push_str("pub mod dwords {\n");
    let mut seen = HashSet::new();
    for entry in array(pkt3.get("register_mappings")) {
        let Some((name, at)) = mapping(entry) else { continue };
        let ty = entry.get("type_ref").and_then(Value::as_str).unwrap_or(name);
        // A dword with layouts for several chips (`RELEASE_MEM_OP` and `RELEASE_MEM_OP_gfx11`):
        // the first under the dword's name, the others under their type's.
        let module = if seen.insert(name.to_string()) {
            name
        } else if ty != name && seen.insert(ty.to_string()) {
            ty
        } else {
            continue;
        };
        let _ = writeln!(
            out,
            "pub mod {} {{\npub const OPCODE: u32 = {:#x};\npub const DWORD: u32 = {};",
            ident(module),
            at >> 4,
            at & 0xf
        );
        if let Some(ty) = types.get(ty) {
            type_fields(&mut out, &path, ty);
        }
        out.push_str("}\n");
    }
    out.push_str("}\n");
    enums(&mut out, &pkt3);
    out.push_str("}\n");
    let _ = writeln!(out, "{ALLOW}\npub mod sid {{");
    let mut seen = HashSet::new();
    for (name, definition, _) in &defines {
        if name.starts_with("PKT3_") || !seen.insert(name.clone()) {
            continue;
        }
        match definition {
            cexpr::Macro::Object(body) => {
                if let Some(value) = evaluator.eval(body, &[]).filter(|&value| value <= u64::from(u32::MAX)) {
                    let _ = writeln!(out, "pub const {}: u32 = {value:#x};", ident(name));
                }
            }
            cexpr::Macro::Function { parameters, body } if parameters.len() == 1 => {
                // A field setter: x lands at `shift` (1 gives 1 << shift, 0 gives 0) and the set
                // bits for an all-ones x are contiguous.
                let parameter = parameters[0].clone();
                let at = |value: u64| {
                    evaluator
                        .eval(body, &[(parameter.clone(), value)])
                        .map(|result| result & 0xffff_ffff)
                };
                let (Some(bits), Some(zero), Some(one)) = (at(0xffff_ffff), at(0), at(1)) else {
                    continue;
                };
                let shift = u64::from(bits.trailing_zeros());
                let run = bits >> shift;
                if bits != 0 && zero == 0 && run & run.wrapping_add(1) == 0 && one == 1 << shift {
                    let _ = writeln!(
                        out,
                        "pub const {}: {path}::Field = {};",
                        ident(name),
                        field(&path, shift, u64::from(bits.count_ones()))
                    );
                }
            }
            cexpr::Macro::Function { .. } => {}
        }
    }
    out.push_str("}\n");
    out.parse()
        .unwrap_or_else(|error| fail(format!("packets!: generated code does not parse: {error}")))
}

/// AMD's CP packet tables (`src/amd/packets` in Mesa), per host generation: the table and the
/// opcode header.
const CP_TABLES: &[(&str, &str, &str)] = &[
    (
        "gfx11",
        include_str!("../mesa/packets/cp_pm4_table_data_gfx11.json"),
        include_str!("../mesa/packets/pm4_it_opcodes_gfx11.h"),
    ),
    (
        "gfx12",
        include_str!("../mesa/packets/cp_pm4_table_data_gfx12.json"),
        include_str!("../mesa/packets/pm4_it_opcodes_gfx12.h"),
    ),
];

/// `hi:lo` or `bit` as (shift, width).
fn bit_range(bits: &str) -> Option<(u64, u64)> {
    let (high, low) = match bits.split_once(':') {
        Some((high, low)) => (high.trim().parse::<u64>().ok()?, low.trim().parse::<u64>().ok()?),
        None => {
            let bit = bits.trim().parse::<u64>().ok()?;
            (bit, bit)
        }
    };
    (high >= low && high < 32).then(|| (low, high - low + 1))
}

/// `IT_NAME = 0x..` lines of an opcode header.
fn opcodes(header: &str) -> HashMap<String, u64> {
    header
        .lines()
        .filter_map(|line| {
            let (name, value) = line.trim().trim_end_matches(',').split_once('=')?;
            let name = name.trim().strip_prefix("IT_")?;
            let value = u64::from_str_radix(value.trim().strip_prefix("0x")?, 16).ok()?;
            Some((name.to_string(), value))
        })
        .collect()
}

/// One packet of a CP table: its opcode, the words it describes and their fields as
/// `PacketField`s (dword 0 is the header), and its enums.
fn cp_packet(out: &mut String, path: &str, name: &str, packet: &Value, opcode: Option<u64>) {
    let _ = writeln!(out, "pub mod {} {{", ident(name));
    if let Some(opcode) = opcode {
        let _ = writeln!(out, "pub const OPCODE: u32 = {opcode:#x};");
    }
    let empty = Map::new();
    let words = packet.get("word").and_then(Value::as_object).unwrap_or(&empty);
    let mut fields: Vec<(String, u64, u64, u64)> = Vec::new();
    let mut count = 0;
    for (word, layouts) in words {
        let Ok(word) = word.parse::<u64>() else { continue };
        count = count.max(word);
        for layout in layouts.as_object().unwrap_or(&empty).values() {
            for (field, bits) in layout.as_object().unwrap_or(&empty) {
                let Some((shift, width)) = bits.get("bits").and_then(Value::as_str).and_then(bit_range) else {
                    continue;
                };
                if !field.starts_with("reserved") && word > 1 {
                    fields.push((field.to_ascii_uppercase(), word - 1, shift, width));
                }
            }
        }
    }
    let _ = writeln!(out, "pub const WORDS: u32 = {count};");
    // A name the layouts of one packet use for different bits gets the dword, then the bits.
    let mut emitted = HashSet::new();
    for (field, dword, shift, width) in &fields {
        let same = |other: &&(String, u64, u64, u64)| &other.0 == field;
        let distinct: HashSet<(u64, u64, u64)> = fields.iter().filter(same).map(|f| (f.1, f.2, f.3)).collect();
        let in_dword = fields.iter().filter(same).filter(|f| f.1 == *dword).count();
        let constant = match (distinct.len(), in_dword) {
            (1, _) => field.clone(),
            (_, 1) => format!("{field}_DW{dword}"),
            _ => format!("{field}_DW{dword}_{shift}"),
        };
        if emitted.insert(constant.clone()) {
            let _ = writeln!(
                out,
                "pub const {}: {path}::PacketField = {path}::PacketField::new({dword}, {shift}, {width});",
                ident(&constant)
            );
        }
    }
    out.push_str("pub mod enums {\n");
    for (field, values) in packet.get("enum").and_then(Value::as_object).unwrap_or(&empty) {
        let _ = writeln!(out, "pub mod {} {{", ident(field));
        let mut seen = HashSet::new();
        for (value_name, value) in values.as_object().unwrap_or(&empty) {
            let Some(number) = value.get("value").and_then(Value::as_u64) else { continue };
            let constant = value_name.to_ascii_uppercase();
            if !value_name.starts_with("reserved") && seen.insert(constant.clone()) {
                let _ = writeln!(out, "pub const {}: u32 = {number:#x};", ident(&constant));
            }
        }
        out.push_str("}\n");
    }
    out.push_str("}\n}\n");
}

/// The CP packets of one host generation: `cp_gfx11` or `cp_gfx12`, with a module per CP
/// engine (`pfp` and `meg` of the graphics queue, `mec` of compute queues) and one per packet in
/// it (`OPCODE`, `WORDS`, its fields and `enums`).
#[proc_macro]
pub fn cp_packets(input: TokenStream) -> TokenStream {
    let (generation, path) = match arguments(input) {
        Ok((Some(generation), path)) => (generation, path),
        Ok((None, _)) => return fail("cp_packets!: name a generation".into()),
        Err(message) => return fail(format!("cp_packets!: {message}")),
    };
    let Some(&(_, table, header)) = CP_TABLES.iter().find(|(name, _, _)| *name == generation) else {
        let known: Vec<&str> = CP_TABLES.iter().map(|(name, _, _)| *name).collect();
        return fail(format!("cp_packets!: no CP packet table for `{generation}` (have {known:?})"));
    };
    let table: Value = match serde_json::from_str(table) {
        Ok(table) => table,
        Err(error) => return fail(format!("cp_packets!({generation}): {error}")),
    };
    let opcodes = opcodes(header);
    let empty = Map::new();
    let mut out = String::new();
    let _ = writeln!(out, "{ALLOW}\npub mod cp_{generation} {{");
    for (engine, packets) in table.get("pm4_packets").and_then(Value::as_object).unwrap_or(&empty) {
        let _ = writeln!(out, "pub mod {} {{", ident(engine));
        for (name, packet) in packets.as_object().unwrap_or(&empty) {
            cp_packet(&mut out, &path, name, packet, opcodes.get(name).copied());
        }
        out.push_str("}\n");
    }
    out.push_str("}\n");
    out.parse()
        .unwrap_or_else(|error| fail(format!("cp_packets!({generation}): generated code does not parse: {error}")))
}
