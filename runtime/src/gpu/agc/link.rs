//! Guest code a translation reaches outside its own words, linked into it ([`shader::Linkage`]):
//! the back half a merged-stage front half jumps to through `s[6:7]`, and subroutines (fetch
//! shaders) it calls through addresses in user data.

use shader::{
    cache::hash_words,
    ir::{memops, ExternalTarget, Fields, InstructionFamily, MemKind},
    CodeRegion, CompileOptions, Compiler, Linkage, ShaderInterface, ShaderIr,
};
use std::collections::{BTreeMap, BTreeSet};

/// The SGPR pair through which a merged-stage front half jumps to its back half.
const SECOND_HALF_SGPR: u8 = 6;
/// Regions linked into one translation at most.
pub const MAX_REGIONS: usize = 16;
const FOUR_GIB: u64 = 1 << 32;

/// Guest state a translation links code from.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LinkInputs {
    /// Guest SPI_SHADER_PGM_LO/HI_GS as an address (`LO << 8 | HI << 40`): where a merged-stage
    /// front half continues. 0 for other stages.
    pub second_half: u64,
    /// The stage's user-data registers.
    pub user_data: [u32; 32],
    /// The first SGPR user data loads: 8 for NGG vertex programs, 0 for compute and pixel ones.
    pub user_data_sgpr: u8,
    /// User-data registers the guest RSRC2 loads (USER_SGPR).
    pub user_sgprs: u8,
    /// User-data words loaded on the GPU whose CPU model is stale. They cannot prove a
    /// native call/fetch target during translation.
    pub unknown_user_data: u32,
}

impl LinkInputs {
    /// The address a wave starts with in `s[sgpr:sgpr + 1]`, when user data loads both.
    pub fn sgpr_pair(&self, sgpr: u8) -> Option<u64> {
        let first = usize::from(sgpr.checked_sub(self.user_data_sgpr)?);
        let loaded = usize::from(self.user_sgprs).min(self.user_data.len());
        if first + 1 >= loaded {
            return None;
        }
        if self.unknown_user_data & (3u32 << first) != 0 {
            return None;
        }
        Some(u64::from(self.user_data[first]) | (u64::from(self.user_data[first + 1]) << 32))
    }
}

/// Guest code linked into one translation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LinkedCode {
    /// Each region's guest address and words, in the order they were found.
    pub regions: Vec<(u64, Vec<u32>)>,
    pub second_half: Option<u64>,
}

impl LinkedCode {
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty() && self.second_half.is_none()
    }

    /// Whether a linked region holds guest address `address`.
    pub fn covers(&self, address: u64) -> bool {
        self.regions
            .iter()
            .any(|(start, words)| address >= *start && address - start < 4 * words.len() as u64)
    }

    /// Each region's address and word hash, and the second-half address: what a translation
    /// cache key needs.
    pub fn key(&self) -> (Vec<(u64, u64)>, Option<u64>) {
        let regions = self
            .regions
            .iter()
            .map(|(address, words)| (*address, hash_words(words)))
            .collect();
        (regions, self.second_half)
    }

    pub fn code_regions(&self) -> Vec<CodeRegion<'_>> {
        self.regions
            .iter()
            .map(|(address, words)| CodeRegion {
                address: *address,
                words,
            })
            .collect()
    }
}

/// What linking reads from a program's analysis.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Analysis {
    pub tail_sgpr: Option<u8>,
    pub external: Vec<ExternalTarget>,
    /// The decoded code may store to memory ([`may_store`]).
    pub stores: bool,
    pub unbounded_stores: bool,
    pub unbounded_reads: bool,
    /// The decoded code sends GS_ALLOC_REQ ([`sends_gs_alloc_req`]).
    pub gs_alloc_req: bool,
}

/// gfx10's GS_ALLOC_REQ message ID, and the `s_sendmsg` and `s_sendmsghalt` opcodes.
const MSG_GS_ALLOC_REQ: u16 = 9;
const SOPP_SENDMSG: [u16; 2] = [0x10, 0x11];

/// Whether `ir` sends GS_ALLOC_REQ, which an NGG subgroup must send before its exports unless
/// the hardware allocates for it.
pub fn sends_gs_alloc_req(ir: &ShaderIr) -> bool {
    ir.instructions.iter().any(|inst| {
        matches!(inst.family, InstructionFamily::Sopp)
            && SOPP_SENDMSG.contains(&inst.opcode)
            && matches!(inst.fields, Fields::Sopp { simm16 } if simm16 & 0xf == MSG_GS_ALLOC_REQ)
    })
}

/// Whether `ir` may store to guest memory: a buffer, image, flat or scalar store or atomic, or
/// a memory instruction of unknown kind. GDS accesses go to the runtime's GDS buffer instead.
pub fn may_store(ir: &ShaderIr) -> bool {
    ir.instructions.iter().any(|inst| {
        let kind = match inst.family {
            InstructionFamily::Smem => memops::smem(inst.opcode).map(|op| op.kind),
            InstructionFamily::Mubuf => memops::mubuf(inst.opcode).map(|op| op.kind),
            InstructionFamily::Mtbuf => memops::mtbuf(inst.opcode).map(|op| op.kind),
            InstructionFamily::Flat => memops::flat(inst.opcode).map(|op| op.kind),
            InstructionFamily::Mimg => memops::image(inst.opcode).map(|op| op.kind()),
            _ => return false,
        };
        !matches!(kind, Some(MemKind::Load | MemKind::Control))
    })
}

/// Buffer/image stores carry descriptors the integration can bound at the work item. Flat
/// and pointer-based scalar stores lack that resource bound. Scratch is runtime-owned.
pub fn has_unbounded_stores(ir: &ShaderIr) -> bool {
    ir.instructions.iter().any(|inst| match &inst.fields {
        Fields::Flat(flat) => {
            flat.segment != shader::ir::FlatSegment::Scratch
                && !matches!(flat.kind, MemKind::Load | MemKind::Control)
        }
        _ => false,
    })
}

pub fn has_unbounded_reads(ir: &ShaderIr) -> bool {
    ir.instructions.iter().any(|inst| {
        matches!(&inst.fields,
        Fields::Flat(flat) if flat.segment != shader::ir::FlatSegment::Scratch
            && matches!(flat.kind, MemKind::Load | MemKind::Atomic))
    })
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AnalysisKey {
    address: u64,
    hash: u64,
    words: usize,
    stage: u8,
    wave_size: u8,
    rsrc1: Option<u32>,
    rsrc2: Option<u32>,
    regions: Vec<(u64, u64)>,
    second_half: Option<u64>,
}

/// Program analyses, each run once per program words and linked code.
#[derive(Debug, Default)]
pub struct Analyses {
    cache: BTreeMap<AnalysisKey, Analysis>,
    runs: usize,
}

impl Analyses {
    /// Analyses run (cache misses).
    pub fn runs(&self) -> usize {
        self.runs
    }

    fn get(
        &mut self,
        compiler: &Compiler,
        program: &Program<'_>,
        code: &LinkedCode,
    ) -> Result<Analysis, String> {
        let (regions, second_half) = code.key();
        let key = AnalysisKey {
            address: program.address,
            hash: program.hash.unwrap_or_else(|| hash_words(program.words)),
            words: program.words.len(),
            stage: program.options.stage as u8,
            wave_size: program.options.wave_size,
            rsrc1: program.interface.guest_pgm_rsrc1,
            rsrc2: program.interface.guest_pgm_rsrc2,
            regions,
            second_half,
        };
        if let Some(analysis) = self.cache.get(&key) {
            return Ok(analysis.clone());
        }
        let regions = code.code_regions();
        let linkage = Linkage {
            regions: &regions,
            second_half: code.second_half,
        };
        let ir = compiler
            .analyze_linked(program.words, program.options, program.interface, &linkage)
            .map_err(|e| e.to_string())?;
        self.runs += 1;
        let analysis = Analysis {
            tail_sgpr: ir.metadata.user_data_tail_sgpr,
            stores: may_store(&ir),
            unbounded_stores: has_unbounded_stores(&ir),
            unbounded_reads: has_unbounded_reads(&ir),
            gs_alloc_req: sends_gs_alloc_req(&ir),
            external: ir.flow.external,
        };
        self.cache.insert(key, analysis.clone());
        Ok(analysis)
    }
}

/// A guest program as the translation sees it.
#[derive(Clone, Copy, Debug)]
pub struct Program<'a> {
    pub address: u64,
    pub words: &'a [u32],
    /// `hash_words(words)` when already known (a registered program), else computed here.
    pub hash: Option<u64>,
    pub options: CompileOptions,
    pub interface: &'a ShaderInterface,
}

/// The code linked into one translation, and what stayed unlinked and why.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Linked {
    pub code: LinkedCode,
    /// The input SGPRs (each the low half of a pair) whose values linking read as code
    /// addresses: what else of the user data the linked code depends on.
    pub pointer_sgprs: BTreeSet<u8>,
    pub notes: Vec<String>,
    /// Whether the translation may store to guest memory; `None` without an analysis. Code an
    /// unlinked jump or call reaches is unknown, so it counts as storing.
    pub stores: Option<bool>,
    pub unbounded_stores: Option<bool>,
    pub unbounded_reads: Option<bool>,
    /// Whether the translation sends GS_ALLOC_REQ; `None` without an analysis, or when it
    /// does not but unlinked code it reaches might.
    pub gs_alloc_req: Option<bool>,
}

/// Links the code `program` reaches outside itself: its back half when it is a merged-stage
/// front half (it ends with a jump through `s[6:7]`), and every jump or call target it holds
/// in user data, re-analyzing with each new region until no new target appears. `load` reads
/// the guest code at an address.
pub fn link(
    compiler: &Compiler,
    program: &Program<'_>,
    inputs: &LinkInputs,
    analyses: &mut Analyses,
    load: &mut dyn FnMut(u64) -> Result<Vec<u32>, String>,
) -> Linked {
    let mut linked = Linked::default();
    let own = (program.address, 4 * program.words.len() as u64);
    let mut tried = BTreeSet::new();
    for pass in 0.. {
        let analysis = match analyses.get(compiler, program, &linked.code) {
            Ok(analysis) => analysis,
            Err(error) => {
                linked.notes.push(format!("analysis failed: {error}"));
                linked.stores = None;
                linked.unbounded_stores = None;
                linked.unbounded_reads = None;
                linked.gs_alloc_req = None;
                break;
            }
        };
        linked.pointer_sgprs.extend(analysis.external.iter().filter_map(|target| target.sgpr));
        let unlinked = analysis.external.iter().any(|target| {
            !target
                .sgpr
                .and_then(|sgpr| inputs.sgpr_pair(sgpr))
                .is_some_and(|address| linked.code.covers(address))
        });
        linked.stores = Some(analysis.stores || unlinked);
        linked.unbounded_stores = Some(analysis.unbounded_stores || unlinked);
        linked.unbounded_reads = Some(analysis.unbounded_reads || unlinked);
        linked.gs_alloc_req = match (analysis.gs_alloc_req, unlinked) {
            (true, _) => Some(true),
            (false, true) => None,
            (false, false) => Some(false),
        };
        let mut wanted: Vec<(u64, bool)> = Vec::new();
        if pass == 0 && analysis.tail_sgpr == Some(SECOND_HALF_SGPR) {
            if inputs.second_half == 0 {
                linked.notes.push(
                    "merged-stage front half without SPI_SHADER_PGM_LO/HI_GS: back half unlinked"
                        .into(),
                );
            } else {
                wanted.push((inputs.second_half, true));
            }
        }
        for target in &analysis.external {
            let Some(address) = target.sgpr.and_then(|sgpr| inputs.sgpr_pair(sgpr)) else {
                continue;
            };
            if address != 0 && !linked.code.covers(address) {
                wanted.push((address, false));
            }
        }
        let mut added = false;
        for (address, second_half) in wanted {
            if !tried.insert(address) {
                continue;
            }
            if linked.code.regions.len() == MAX_REGIONS {
                linked
                    .notes
                    .push(format!("more than {MAX_REGIONS} linked regions"));
                break;
            }
            let words = match load(address) {
                Ok(words) => words,
                Err(error) => {
                    linked.notes.push(format!("code at 0x{address:x}: {error}"));
                    continue;
                }
            };
            let bytes = 4 * words.len() as u64;
            if let Some(problem) = placement_problem(own, &linked.code, address, bytes) {
                linked
                    .notes
                    .push(format!("code at 0x{address:x} {problem}"));
                continue;
            }
            linked.code.regions.push((address, words));
            if second_half {
                linked.code.second_half = Some(address);
            }
            added = true;
        }
        if !added {
            break;
        }
    }
    linked
}

/// Why `bytes` of code at `address` cannot be linked beside `own` (the program) and `code`.
fn placement_problem(
    own: (u64, u64),
    code: &LinkedCode,
    address: u64,
    bytes: u64,
) -> Option<&'static str> {
    let overlaps = |(start, len): (u64, u64)| address < start + len && start < address + bytes;
    if bytes == 0 || address / FOUR_GIB != (address + bytes - 1) / FOUR_GIB {
        Some("crosses a 4 GiB boundary")
    } else if overlaps(own) {
        Some("overlaps the program")
    } else if code
        .regions
        .iter()
        .any(|(start, words)| overlaps((*start, 4 * words.len() as u64)))
    {
        Some("overlaps another linked region")
    } else {
        None
    }
}

/// How many of a translation's external `targets` reach linked code.
pub fn coverage(targets: &[ExternalTarget], inputs: &LinkInputs, code: &LinkedCode) -> usize {
    targets
        .iter()
        .filter(|t| {
            t.sgpr
                .and_then(|sgpr| inputs.sgpr_pair(sgpr))
                .is_some_and(|address| code.covers(address))
        })
        .count()
}

