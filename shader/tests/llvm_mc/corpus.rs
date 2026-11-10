//! The llvm-mc corpus: `cases.txt`, the interface each case compiles with, and the golden
//! translations. Shared by the llvm-mc test and the throughput bench, which use different
//! parts of it.
#![allow(dead_code)]

use shader::{
    AttributeRingAbi, CodeRegion, CompileOptions, Compiler, Linkage, ShaderInterface, ShaderStage,
    TargetIsa, VertexStage,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const GUEST: u64 = 0x1_0000_0000;
pub const TARGETS: [(TargetIsa, &str); 4] = [
    (TargetIsa::Gfx103, "gfx1030"),
    (TargetIsa::Gfx11, "gfx1100"),
    (TargetIsa::Gfx115, "gfx1151"),
    (TargetIsa::Gfx12, "gfx1200"),
];

pub struct Case {
    pub name: String,
    pub stage: ShaderStage,
    pub wave: u8,
    pub words: Vec<u32>,
    /// Linked code (an optional fifth field, `@address:words` entries and `half=address`
    /// separated by `;`), compiled with [`Compiler::compile_linked`].
    pub regions: Vec<(u64, Vec<u32>)>,
    pub second_half: Option<u64>,
}

/// A golden entry: the translation (or its compile error) and llvm-mc's disassembly.
#[derive(Clone, Debug, PartialEq)]
pub struct Golden {
    pub words: Result<Vec<u32>, String>,
    pub text: Vec<String>,
}

impl Default for Golden {
    fn default() -> Self {
        Self {
            words: Ok(Vec::new()),
            text: Vec::new(),
        }
    }
}

pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/llvm_mc")
}

pub fn cases() -> Vec<Case> {
    let text = std::fs::read_to_string(dir().join("cases.txt")).expect("cases.txt");
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let parts: Vec<&str> = line.split('|').collect();
            let mut case = Case {
                name: parts[0].to_string(),
                stage: match parts[1] {
                    "cs" => ShaderStage::Compute,
                    "vs" => ShaderStage::Vertex,
                    _ => ShaderStage::Fragment,
                },
                wave: parts[2].parse().expect("wave size"),
                words: parse_words(parts[3]),
                regions: Vec::new(),
                second_half: None,
            };
            for item in parts.get(4).into_iter().flat_map(|link| link.split(';')) {
                let hex = |text: &str| u64::from_str_radix(text, 16).expect("hex address");
                if let Some(address) = item.strip_prefix("half=") {
                    case.second_half = Some(hex(address));
                } else if let Some((address, words)) =
                    item.strip_prefix('@').and_then(|r| r.split_once(':'))
                {
                    case.regions.push((hex(address), parse_words(words)));
                }
            }
            case
        })
        .collect()
}

pub fn parse_words(text: &str) -> Vec<u32> {
    text.split([',', ' '])
        .filter(|w| !w.is_empty())
        .map(|w| u32::from_str_radix(w, 16).expect("hex word"))
        .collect()
}

pub fn golden_path(cpu: &str) -> PathBuf {
    dir().join("golden").join(format!("{cpu}.txt"))
}

pub fn read_golden(cpu: &str) -> BTreeMap<String, Golden> {
    let text = std::fs::read_to_string(golden_path(cpu)).unwrap_or_default();
    let mut entries = BTreeMap::new();
    let mut current: Option<(String, Golden)> = None;
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("## ") {
            entries.extend(current.take());
            current = Some((name.to_string(), Golden::default()));
        } else if let Some((_, entry)) = current.as_mut() {
            if let Some(words) = line.strip_prefix("words: ") {
                entry.words = Ok(parse_words(words));
            } else if let Some(error) = line.strip_prefix("error: ") {
                entry.words = Err(error.to_string());
            } else if let Some(asm) = line.strip_prefix("    ") {
                entry.text.push(asm.to_string());
            }
        }
    }
    entries.extend(current);
    entries
}

/// Descriptor table address for the `lut_*` cases (descriptor conversion on).
pub const LUT: u64 = 0x7_0000_0000;
/// The GDS buffer of `gds_` cases.
pub const GDS: u64 = 0x6_0000_0000;
/// The attribute ring of the `ring_*` cases.
pub const RING: AttributeRingAbi = AttributeRingAbi {
    address: 0x5_0000_0000,
    bytes: 0x2c_0000,
};

/// The interface a case compiles with, chosen by its name: `lut_*` convert descriptors,
/// `ps_known_*` and `dual_*` know the user SGPR count (so pixel shaders hoist parameter loads),
/// `dual_*` blend with two sources, `zonly_*` have depth but no colour export memory, and
/// `ring_*` store parameters to an attribute ring.
pub fn interface(name: &str) -> ShaderInterface {
    let zonly = name.starts_with("zonly_");
    let ring = name.starts_with("ring_");
    // Merged LS-HS programs: 16 user SGPRs, LS_VGPR_COMP_CNT 3, 2 KiB of LDS.
    if name.starts_with("hull_") {
        return ShaderInterface {
            guest_code_address: Some(GUEST),
            user_sgpr_count: Some(16),
            guest_pgm_rsrc1: Some(3 << 28),
            guest_pgm_rsrc2: Some((16 << 1) | (4 << 18)),
            vertex_stage: VertexStage::LsHs,
            ..ShaderInterface::default()
        };
    }
    // Domain programs (NGG ES with ES_EN = DS): 14 user SGPRs, ES_VGPR_COMP_CNT 3.
    if name.starts_with("domain_") {
        return ShaderInterface {
            guest_code_address: Some(GUEST),
            user_sgpr_count: Some(14),
            guest_pgm_rsrc2: Some((14 << 1) | (3 << 16)),
            ..ShaderInterface::default()
        };
    }
    // Private-segment programs: 8 user SGPRs, then the scratch wave offset (SCRATCH_EN).
    if name.starts_with("priv_") {
        return ShaderInterface {
            guest_code_address: Some(GUEST),
            guest_pgm_rsrc2: Some(1 | (8 << 1)),
            ..ShaderInterface::default()
        };
    }
    ShaderInterface {
        guest_code_address: Some(GUEST),
        descriptor_lut: name.starts_with("lut_").then_some(LUT),
        user_sgpr_count: (name.starts_with("ps_known_") || name.starts_with("dual_")).then_some(2),
        dual_source_blend: name.starts_with("dual_"),
        ps_col_format: zonly.then_some(0),
        ps_z_format: zonly.then_some(1),
        attribute_ring: ring.then_some(RING),
        guest_pgm_rsrc2: ring.then_some(3 << 16),
        gds_buffer: name.starts_with("gds_").then_some(GDS),
        ..ShaderInterface::default()
    }
}

pub fn translate(case: &Case, target: TargetIsa) -> Result<Vec<u32>, String> {
    let interface = interface(&case.name);
    let regions: Vec<CodeRegion<'_>> = case
        .regions
        .iter()
        .map(|(address, words)| CodeRegion {
            address: *address,
            words,
        })
        .collect();
    let linkage = Linkage {
        regions: &regions,
        second_half: case.second_half,
    };
    Compiler::new(target)
        .compile_linked(
            &case.words,
            CompileOptions::new(target, case.stage, case.wave),
            &interface,
            &linkage,
        )
        .map(|artifact| artifact.program().to_vec())
        .map_err(|e| e.to_string())
}
