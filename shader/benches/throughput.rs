//! Translation throughput: compile times for large synthetic programs (straight-line code, a
//! deep CFG, many computed jumps, linked regions) and for the llvm-mc corpus.
//!
//! Every workload prints a hash of its translations (code and tables), so a change that alters
//! the output shows, and the corpus is checked against the golden files. Arguments select
//! workloads by name prefix; `--loop NAME SECONDS` compiles one workload over and over, for a
//! sampling profiler to attach to.
//!
//! ```text
//! cargo bench -p shader --offline --bench throughput
//! cargo bench -p shader --offline --bench throughput -- cfg jumps
//! cargo bench -p shader --offline --bench throughput -- --loop jumps/gfx1151 30
//! ```

#[path = "../tests/llvm_mc/corpus.rs"]
mod corpus;

use shader::{
    CodeRegion, CompileOptions, Compiler, Linkage, ShaderInterface, ShaderStage, TargetIsa,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// One compile.
struct Job {
    name: String,
    stage: ShaderStage,
    wave: u8,
    words: Vec<u32>,
    interface: ShaderInterface,
    regions: Vec<(u64, Vec<u32>)>,
    second_half: Option<u64>,
    /// The golden translation, for corpus jobs.
    golden: Option<Result<Vec<u32>, String>>,
}

impl Job {
    fn synthetic(name: &str, words: Vec<u32>, interface: ShaderInterface) -> Self {
        Self {
            name: name.to_string(),
            stage: ShaderStage::Compute,
            wave: 32,
            words,
            interface,
            regions: Vec::new(),
            second_half: None,
            golden: None,
        }
    }

    fn compile(&self, target: TargetIsa) -> Result<shader::ShaderArtifact, String> {
        let regions = self.code_regions();
        let linkage = Linkage {
            regions: &regions,
            second_half: self.second_half,
        };
        Compiler::new(target)
            .compile_linked(&self.words, self.options(target), &self.interface, &linkage)
            .map_err(|e| e.to_string())
    }

    /// The guest instructions the compile decodes, or 0 when it fails.
    fn instructions(&self, target: TargetIsa) -> usize {
        let regions = self.code_regions();
        let linkage = Linkage {
            regions: &regions,
            second_half: self.second_half,
        };
        Compiler::new(target)
            .analyze_linked(&self.words, self.options(target), &self.interface, &linkage)
            .map_or(0, |ir| ir.instructions.len())
    }

    fn analyze(&self, target: TargetIsa) {
        let regions = self.code_regions();
        let linkage = Linkage {
            regions: &regions,
            second_half: self.second_half,
        };
        let ir = Compiler::new(target).analyze_linked(
            &self.words,
            self.options(target),
            &self.interface,
            &linkage,
        );
        black_box(ir.is_ok());
    }

    fn options(&self, target: TargetIsa) -> CompileOptions {
        CompileOptions::new(target, self.stage, self.wave)
    }

    fn code_regions(&self) -> Vec<CodeRegion<'_>> {
        self.regions
            .iter()
            .map(|(address, words)| CodeRegion {
                address: *address,
                words,
            })
            .collect()
    }
}

struct Workload {
    name: String,
    target: TargetIsa,
    jobs: Vec<Job>,
}

impl Workload {
    fn run_once(&self) {
        for job in &self.jobs {
            black_box(job.compile(self.target).is_ok());
        }
    }
}

const GUEST: u64 = corpus::GUEST;
const LUT: u64 = corpus::LUT;

fn workloads(selected: &[String]) -> Vec<Workload> {
    let wanted = |name: &str| selected.is_empty() || selected.iter().any(|s| name.starts_with(s));
    let base = ShaderInterface {
        guest_code_address: Some(GUEST),
        lds_bytes: 0x1000,
        ..ShaderInterface::default()
    };
    let lut = ShaderInterface {
        descriptor_lut: Some(LUT),
        ..base
    };
    let corpus = corpus::cases();
    let mut out = Vec::new();
    for (target, cpu) in corpus::TARGETS {
        let gfx11 = matches!(target, TargetIsa::Gfx11 | TargetIsa::Gfx115);
        let mut add = |name: &str, jobs: Vec<Job>| {
            let name = format!("{name}/{cpu}");
            if wanted(&name) {
                out.push(Workload { name, target, jobs });
            }
        };
        let golden = corpus::read_golden(cpu);
        let jobs = corpus
            .iter()
            .map(|case| Job {
                name: case.name.clone(),
                stage: case.stage,
                wave: case.wave,
                words: case.words.clone(),
                interface: corpus::interface(&case.name),
                regions: case.regions.clone(),
                second_half: case.second_half,
                golden: golden.get(&case.name).map(|g| g.words.clone()),
            })
            .collect();
        add("corpus", jobs);
        let straight = straight_line(40_000, 1);
        add(
            "straight",
            vec![Job::synthetic("straight", straight.clone(), base)],
        );
        if gfx11 {
            add(
                "straight_lut",
                vec![Job::synthetic("straight_lut", straight, lut)],
            );
        }
        add(
            "cfg",
            vec![Job::synthetic("cfg", deep_cfg(6_000, 32, 2, 32), base)],
        );
        add(
            "straight_w64",
            vec![Job {
                wave: 64,
                ..Job::synthetic("straight_w64", straight_line(40_000, 1), base)
            }],
        );
        add(
            "cfg_w64",
            vec![Job {
                wave: 64,
                ..Job::synthetic("cfg_w64", deep_cfg(6_000, 16, 2, 64), base)
            }],
        );
        add(
            "jumps",
            vec![Job::synthetic("jumps", computed_jumps(600, 3), base)],
        );
        add("linked", vec![linked(64, 16, 1_000, 4, &base)]);
    }
    out
}

fn fnv(hash: &mut u64, word: u32) {
    *hash = (*hash ^ u64::from(word)).wrapping_mul(0x0100_0000_01b3);
}

/// What a translation reports besides its code (descriptor uses, bindings, registers), less
/// the offsets that move with the code.
fn metadata(artifact: &shader::ShaderArtifact) -> String {
    let a = artifact;
    format!(
        "{:?}",
        (
            (a.pgm_rsrc1, a.pgm_rsrc2, a.num_sgprs, a.num_vgprs),
            (a.lds_bytes, a.scratch_bytes_per_lane, a.user_sgpr_count),
            (a.guest_words, a.guest_hash, a.user_data_tail_sgpr),
            (&a.bindings, &a.exports, a.uses_pc_relative),
            (&a.preconverted_descriptors, a.spi_shader_col_format),
            a.ngg.as_ref().map(|n| (n.param_count, n.spi_vs_out_config)),
            (&a.external_targets, &a.descriptor_uses),
        )
    )
}

/// FNV-1a over every translation's words and, separately, its [`metadata`], with failures
/// folded into both by message.
fn hash_outputs(workload: &Workload) -> (u64, u64, usize, Vec<String>) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut meta = hash;
    let mut words = 0;
    let mut mismatches = Vec::new();
    for job in &workload.jobs {
        let result = job.compile(workload.target);
        match &result {
            Ok(artifact) => {
                words += artifact.code.len();
                artifact.code.iter().for_each(|&w| fnv(&mut hash, w));
                metadata(artifact)
                    .bytes()
                    .for_each(|b| fnv(&mut meta, u32::from(b)));
            }
            Err(e) => e.bytes().for_each(|b| {
                fnv(&mut hash, u32::from(b) | 0x8000_0000);
                fnv(&mut meta, u32::from(b) | 0x8000_0000);
            }),
        }
        // Corpus jobs must match their golden translation, synthetic ones must compile.
        match (&job.golden, result) {
            (Some(golden), result) => {
                if &result.map(|a| a.program().to_vec()) != golden {
                    mismatches.push(job.name.clone());
                }
            }
            (None, Err(e)) => mismatches.push(format!("{}: {e}", job.name)),
            (None, Ok(_)) => {}
        }
    }
    (hash, meta, words, mismatches)
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times[times.len() / 2]
}

fn measure(workload: &Workload) {
    let (hash, meta, out_words, mismatches) = hash_outputs(workload);
    let guest_words: usize = workload
        .jobs
        .iter()
        .map(|j| j.words.len() + j.regions.iter().map(|r| r.1.len()).sum::<usize>())
        .sum();
    let instructions: usize = workload
        .jobs
        .iter()
        .map(|j| j.instructions(workload.target))
        .sum();
    let budget = Duration::from_millis(1500);
    let mut compile = Vec::new();
    let start = Instant::now();
    while compile.len() < 3 || (start.elapsed() < budget && compile.len() < 200) {
        let t = Instant::now();
        workload.run_once();
        compile.push(t.elapsed());
    }
    let mut analyze = Vec::new();
    let start = Instant::now();
    while analyze.len() < 3 || (start.elapsed() < budget / 2 && analyze.len() < 200) {
        let t = Instant::now();
        for job in &workload.jobs {
            job.analyze(workload.target);
        }
        analyze.push(t.elapsed());
    }
    let runs = compile.len();
    let best = *compile.iter().min().unwrap();
    let typical = median(&mut compile);
    let analyzed = median(&mut analyze);
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!(
        "{:<20} {:>4} {:>8} {:>8} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>9.0} {:>4}  {:016x} {:016x} {}",
        workload.name,
        workload.jobs.len(),
        guest_words,
        instructions,
        out_words,
        ms(typical),
        ms(best),
        ms(analyzed),
        instructions as f64 / typical.as_secs_f64() / 1e3,
        runs,
        hash,
        meta,
        if mismatches.is_empty() {
            "ok".to_string()
        } else {
            format!("MISMATCH {}", mismatches.join(", "))
        }
    );
}

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    if let Some(at) = args.iter().position(|a| a == "--loop") {
        let name = args.get(at + 1).expect("--loop NAME SECONDS");
        let seconds: f64 = args
            .get(at + 2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(20.0);
        let all = workloads(std::slice::from_ref(name));
        let workload = all
            .iter()
            .find(|w| &w.name == name)
            .expect("no such workload");
        let start = Instant::now();
        let mut runs = 0;
        while start.elapsed().as_secs_f64() < seconds {
            workload.run_once();
            runs += 1;
        }
        println!(
            "{name}: {runs} runs in {:.1} s",
            start.elapsed().as_secs_f64()
        );
        return;
    }
    println!(
        "{:<20} {:>4} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>9} {:>4}  {:<16} {:<16} check",
        "workload",
        "jobs",
        "words",
        "insts",
        "out",
        "median ms",
        "min ms",
        "analyze ms",
        "kinst/s",
        "runs",
        "code hash",
        "metadata hash"
    );
    for workload in workloads(&args) {
        measure(&workload);
    }
}

// ---------------------------------------------------------------------------------------------
// A small gfx1013 assembler for the synthetic programs.

const VCC_LO: u32 = 106;
const SGPR_NULL: u32 = 125;
const EXEC_LO: u32 = 126;
const LITERAL: u32 = 255;

const fn int(value: u32) -> u32 {
    128 + value
}

const fn vgpr(n: u32) -> u32 {
    256 + n
}

const SOPP_ENDPGM: u32 = 0x01;
const SOPP_BRANCH: u32 = 0x02;
const SOPP_CBRANCH_SCC0: u32 = 0x04;
const SOPP_CBRANCH_SCC1: u32 = 0x05;
const SOPP_CBRANCH_EXECZ: u32 = 0x08;
const SOPP_WAITCNT: u32 = 0x0c;
const WAIT_LGKM: u16 = 0xc07f;
const WAIT_VM: u16 = 0x3f70;
const WAIT_ALL: u16 = 0x0070;

const SOP1_MOV_B32: u32 = 0x03;
const SOP1_GETPC_B64: u32 = 0x1f;
const SOP1_SETPC_B64: u32 = 0x20;
const SOP1_SWAPPC_B64: u32 = 0x21;
const SOP1_AND_SAVEEXEC_B64: u32 = 0x24;
const SOP1_AND_SAVEEXEC_B32: u32 = 0x3c;

const SOP2_ADD_U32: u32 = 0x00;
const SOP2_SUB_U32: u32 = 0x01;
const SOP2_ADDC_U32: u32 = 0x04;
const SOP2_MAX_U32: u32 = 0x09;
const SOP2_AND_B32: u32 = 0x0e;
const SOP2_OR_B32: u32 = 0x10;
const SOP2_OR_B64: u32 = 0x11;
const SOP2_XOR_B32: u32 = 0x12;
const SOP2_LSHL_B32: u32 = 0x1e;
const SOP2_MUL_I32: u32 = 0x26;
const SALU_OPS: [u32; 7] = [
    SOP2_ADD_U32,
    SOP2_AND_B32,
    SOP2_OR_B32,
    SOP2_XOR_B32,
    SOP2_LSHL_B32,
    SOP2_MUL_I32,
    SOP2_MAX_U32,
];

const SOPC_CMP_LG_U32: u32 = 0x07;

const VOP1_CVT_F32_U32: u32 = 0x06;
const VOP1_OPS: [u32; 5] = [0x06, 0x20, 0x25, 0x2a, 0x33];

const VOP2_CNDMASK: u32 = 0x01;
const VOP2_LSHLREV: u32 = 0x1a;
const VOP2_OPS: [u32; 10] = [0x03, 0x04, 0x08, 0x0f, 0x10, 0x1b, 0x1c, 0x1d, 0x25, 0x2b];

const VOPC_CMP_GT_F32: u32 = 0x04;
const VOPC_CMP_GT_U32: u32 = 0xc4;

const VOP3_OPS: [u32; 6] = [0x14b, 0x157, 0x148, 0x143, 0x36d, 0x346];

const SMEM_LOAD_DWORD: u32 = 0;
const SMEM_LOAD_X4: u32 = 2;
const SMEM_LOAD_X8: u32 = 3;
const MUBUF_LOAD_DWORD: u32 = 0x0c;
const DS_WRITE_B32: u32 = 0x0d;
const DS_READ_B32: u32 = 0x36;
const GLOBAL_STORE_DWORD: u32 = 0x1c;
const MIMG_SAMPLE: u32 = 0x20;

/// Data VGPRs; v63 holds the lane's byte address and v1..v2 the sample coordinates.
const DATA_VGPRS: std::ops::Range<u32> = 3..63;
const ADDRESS_VGPR: u32 = 63;
/// SGPRs the generated SALU code writes, clear of the descriptors and the CFG's exec saves
/// (s20 up), and low enough to leave the translation its own SGPRs.
const SALU_SGPRS: std::ops::Range<u32> = 56..72;

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 32) % u64::from(n)) as u32
    }

    fn in_range(&mut self, range: std::ops::Range<u32>) -> u32 {
        range.start + self.below(range.end - range.start)
    }

    fn pick(&mut self, from: &[u32]) -> u32 {
        from[self.below(from.len() as u32) as usize]
    }
}

#[derive(Default)]
struct Asm {
    words: Vec<u32>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, usize)>,
}

impl Asm {
    fn here(&self) -> usize {
        self.words.len()
    }

    fn label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }

    fn bind(&mut self, label: usize) {
        self.labels[label] = Some(self.here());
    }

    fn word(&mut self, word: u32) {
        self.words.push(word);
    }

    fn sopp(&mut self, op: u32, imm: u16) {
        self.word(0xbf80_0000 | op << 16 | u32::from(imm));
    }

    fn branch(&mut self, op: u32, label: usize) {
        self.fixups.push((self.here(), label));
        self.sopp(op, 0);
    }

    fn sop1(&mut self, op: u32, sdst: u32, src0: u32) {
        self.word(0xbe80_0000 | sdst << 16 | op << 8 | src0);
    }

    fn sop2(&mut self, op: u32, sdst: u32, src0: u32, src1: u32) {
        self.word(0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0);
    }

    fn sopc(&mut self, op: u32, src0: u32, src1: u32) {
        self.word(0xbf00_0000 | op << 16 | src1 << 8 | src0);
    }

    fn vop1(&mut self, op: u32, vdst: u32, src0: u32) {
        self.word(0x7e00_0000 | vdst << 17 | op << 9 | src0);
    }

    fn vop2(&mut self, op: u32, vdst: u32, src0: u32, vsrc1: u32) {
        self.word(op << 25 | vdst << 17 | vsrc1 << 9 | src0);
    }

    fn vopc(&mut self, op: u32, src0: u32, vsrc1: u32) {
        self.word(0x7c00_0000 | op << 17 | vsrc1 << 9 | src0);
    }

    fn vop3(&mut self, op: u32, vdst: u32, sources: [u32; 3]) {
        self.word(0xd400_0000 | op << 16 | vdst);
        self.word(sources[0] | sources[1] << 9 | sources[2] << 18);
    }

    fn smem(&mut self, op: u32, sdata: u32, sbase: u32, soffset: u32, offset: u32) {
        self.word(0xf400_0000 | op << 18 | sdata << 6 | sbase >> 1);
        self.word(soffset << 25 | offset);
    }

    fn mubuf_offen(&mut self, op: u32, vdata: u32, vaddr: u32, srsrc: u32, offset: u32) {
        self.word(0xe000_0000 | op << 18 | 1 << 12 | offset);
        self.word(vaddr | vdata << 8 | (srsrc >> 2) << 16 | int(0) << 24);
    }

    fn ds(&mut self, op: u32, vdst: u32, addr: u32, data0: u32, offset: u32) {
        self.word(0xd800_0000 | op << 18 | offset);
        self.word(addr | data0 << 8 | vdst << 24);
    }

    fn global_store(&mut self, addr: u32, data: u32, saddr: u32) {
        self.word(0xdc00_0000 | GLOBAL_STORE_DWORD << 18 | 2 << 14);
        self.word(addr | data << 8 | saddr << 16);
    }

    /// image_sample v[vdata:vdata+3], v[1:2], s[srsrc:+7], s[ssamp:+3] dmask:0xf dim:2D.
    fn sample(&mut self, vdata: u32, srsrc: u32, ssamp: u32) {
        self.word(0xf000_0000 | MIMG_SAMPLE << 18 | 0xf << 8 | 1 << 3);
        self.word(1 | vdata << 8 | (srsrc >> 2) << 16 | (ssamp >> 2) << 21);
    }

    fn finish(mut self) -> Vec<u32> {
        for (at, label) in self.fixups {
            let target = self.labels[label].expect("bound label");
            let offset = target as i64 - at as i64 - 1;
            let offset = i16::try_from(offset).expect("branch in range");
            self.words[at] |= u32::from(offset as u16);
        }
        self.words
    }
}

/// Loads the buffer, image and sampler descriptors from the table at s[0:1] and sets up the
/// lane address and sample coordinates.
fn prologue(a: &mut Asm) {
    a.smem(SMEM_LOAD_X4, 8, 0, SGPR_NULL, 0x0);
    a.smem(SMEM_LOAD_X8, 16, 0, SGPR_NULL, 0x10);
    a.smem(SMEM_LOAD_X4, 24, 0, SGPR_NULL, 0x30);
    a.vop2(VOP2_LSHLREV, ADDRESS_VGPR, int(2), 0);
    a.vop1(VOP1_CVT_F32_U32, 1, vgpr(0));
    a.vop1(VOP1_CVT_F32_U32, 2, vgpr(ADDRESS_VGPR));
    a.sopp(SOPP_WAITCNT, WAIT_LGKM);
}

fn epilogue(a: &mut Asm) {
    a.global_store(ADDRESS_VGPR, 3, 0);
    a.sopp(SOPP_WAITCNT, WAIT_ALL);
    a.sopp(SOPP_ENDPGM, 0);
}

/// `count` instructions of mixed VALU, SALU, memory and sampling.
fn mixed(a: &mut Asm, rng: &mut Rng, count: usize) {
    for _ in 0..count {
        let d = rng.in_range(DATA_VGPRS);
        let s0 = vgpr(rng.in_range(1..63));
        let s1 = rng.in_range(1..63);
        match rng.below(100) {
            0..=49 => a.vop2(rng.pick(&VOP2_OPS), d, s0, s1),
            50..=59 => a.vop3(
                rng.pick(&VOP3_OPS),
                d,
                [s0, vgpr(s1), vgpr(rng.in_range(1..63))],
            ),
            60..=67 => {
                let src1 = if rng.below(2) == 0 {
                    int(rng.below(65))
                } else {
                    rng.in_range(SALU_SGPRS)
                };
                let (sdst, src0) = (rng.in_range(SALU_SGPRS), rng.in_range(SALU_SGPRS));
                a.sop2(rng.pick(&SALU_OPS), sdst, src0, src1);
            }
            68..=71 => a.vop1(rng.pick(&VOP1_OPS), d, s0),
            72..=75 => {
                a.vopc(VOPC_CMP_GT_F32, s0, s1);
                a.vop2(VOP2_CNDMASK, d, vgpr(rng.in_range(1..63)), s1);
            }
            76..=80 => a.mubuf_offen(MUBUF_LOAD_DWORD, d, ADDRESS_VGPR, 8, 4 * rng.below(1024)),
            81..=83 => a.sopp(SOPP_WAITCNT, WAIT_VM),
            84..=85 => a.ds(DS_WRITE_B32, 0, ADDRESS_VGPR, d, 4 * rng.below(64)),
            86..=87 => a.ds(DS_READ_B32, d, ADDRESS_VGPR, 0, 4 * rng.below(64)),
            88..=89 => a.sopp(SOPP_WAITCNT, WAIT_LGKM),
            90..=92 => a.global_store(ADDRESS_VGPR, d, 0),
            93..=95 => a.sample(rng.in_range(3..60), 16, 24),
            _ => a.vop1(VOP1_CVT_F32_U32, d, vgpr(s1)),
        }
    }
}

fn straight_line(count: usize, seed: u64) -> Vec<u32> {
    let mut a = Asm::default();
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    prologue(&mut a);
    mixed(&mut a, &mut rng, count);
    epilogue(&mut a);
    a.finish()
}

/// Nested divergent ifs, uniform branches and loops, `constructs` in all, up to `depth` deep,
/// for waves of `wave` lanes.
fn deep_cfg(constructs: usize, depth: u32, seed: u64, wave: u8) -> Vec<u32> {
    let mut a = Asm::default();
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    prologue(&mut a);
    let mut left = constructs;
    while left > 0 {
        let mut chunk = left.min(150);
        left -= chunk;
        while chunk > 0 {
            nest(&mut a, &mut rng, 0, depth, &mut chunk, wave == 64);
        }
    }
    epilogue(&mut a);
    a.finish()
}

fn nest(a: &mut Asm, rng: &mut Rng, level: u32, depth: u32, budget: &mut usize, wave64: bool) {
    for _ in 0..1 + rng.below(3) {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        // One exec save (an SGPR pair in wave64) or loop counter per level.
        let save = 20 + if wave64 { 2 * level } else { level };
        match if level >= depth { 3 } else { rng.below(4) } {
            0 => {
                a.vopc(VOPC_CMP_GT_U32, int(rng.below(64)), 0);
                if wave64 {
                    a.sop1(SOP1_AND_SAVEEXEC_B64, save, VCC_LO);
                } else {
                    a.sop1(SOP1_AND_SAVEEXEC_B32, save, VCC_LO);
                }
                let skip = a.label();
                a.branch(SOPP_CBRANCH_EXECZ, skip);
                mixed(a, rng, 2);
                nest(a, rng, level + 1, depth, budget, wave64);
                a.bind(skip);
                if wave64 {
                    a.sop2(SOP2_OR_B64, EXEC_LO, EXEC_LO, save);
                } else {
                    a.sop2(SOP2_OR_B32, EXEC_LO, EXEC_LO, save);
                }
            }
            1 => {
                a.sopc(
                    SOPC_CMP_LG_U32,
                    rng.in_range(SALU_SGPRS),
                    int(rng.below(64)),
                );
                let skip = a.label();
                a.branch(SOPP_CBRANCH_SCC0, skip);
                mixed(a, rng, 2);
                nest(a, rng, level + 1, depth, budget, wave64);
                a.bind(skip);
            }
            2 => {
                a.sop1(SOP1_MOV_B32, save, int(3));
                let top = a.label();
                a.bind(top);
                mixed(a, rng, 2);
                nest(a, rng, level + 1, depth, budget, wave64);
                a.sop2(SOP2_SUB_U32, save, save, int(1));
                a.sopc(SOPC_CMP_LG_U32, save, int(0));
                a.branch(SOPP_CBRANCH_SCC1, top);
            }
            _ => {
                let count = 3 + rng.below(6) as usize;
                mixed(a, rng, count);
            }
        }
    }
}

/// `sites` computed jumps: jump tables with four cases (read from the code after the program,
/// so every instruction may be a target) alternating with calls of subroutines placed after
/// `s_endpgm` that return through `s_setpc_b64`.
fn computed_jumps(sites: usize, seed: u64) -> Vec<u32> {
    let mut a = Asm::default();
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    prologue(&mut a);
    let mut tables = Vec::new();
    let mut calls = Vec::new();
    for site in 0..sites {
        a.sop1(SOP1_GETPC_B64, 4, 0);
        let anchor = a.here();
        if site % 2 == 0 {
            a.sop2(SOP2_AND_B32, 6, 2, int(3));
            a.sop2(SOP2_LSHL_B32, 6, 6, int(2));
            a.sop2(SOP2_ADD_U32, 8, 4, LITERAL);
            let literal = a.here();
            a.word(0);
            a.sop2(SOP2_ADDC_U32, 9, 5, int(0));
            a.smem(SMEM_LOAD_DWORD, 7, 8, 6, 0);
            a.sopp(SOPP_WAITCNT, WAIT_LGKM);
            a.sop2(SOP2_ADD_U32, 4, 4, 7);
            a.sop2(SOP2_ADDC_U32, 5, 5, int(0));
            a.sop1(SOP1_SETPC_B64, 0, 4);
            let join = a.label();
            let mut cases = Vec::new();
            for _ in 0..4 {
                cases.push(a.here());
                mixed(&mut a, &mut rng, 3);
                a.branch(SOPP_BRANCH, join);
            }
            a.bind(join);
            tables.push((anchor, literal, cases));
        } else {
            a.sop2(SOP2_ADD_U32, 4, 4, LITERAL);
            let literal = a.here();
            a.word(0);
            a.sop2(SOP2_ADDC_U32, 5, 5, int(0));
            a.sop1(SOP1_SWAPPC_B64, 30, 4);
            calls.push((anchor, literal));
        }
        mixed(&mut a, &mut rng, 4);
    }
    epilogue(&mut a);
    for (anchor, literal) in calls {
        a.words[literal] = 4 * (a.here() - anchor) as u32;
        mixed(&mut a, &mut rng, 3);
        a.sop1(SOP1_SETPC_B64, 0, 30);
    }
    for (anchor, literal, cases) in tables {
        a.words[literal] = 4 * (a.here() - anchor) as u32;
        for case in cases {
            a.word(4 * (case - anchor) as u32);
        }
    }
    a.finish()
}

/// A program making `calls` calls through user data (s[2:3]) into `regions` linked
/// subroutines of `size` instructions each, with `between` instructions around each call.
fn linked(calls: usize, regions: u64, size: usize, between: usize, base: &ShaderInterface) -> Job {
    let mut rng = Rng(0x5eed);
    let mut a = Asm::default();
    prologue(&mut a);
    for _ in 0..calls {
        mixed(&mut a, &mut rng, between);
        a.sop1(SOP1_SWAPPC_B64, 30, 2);
    }
    epilogue(&mut a);
    let regions = (0..regions)
        .map(|k| {
            let mut r = Asm::default();
            mixed(&mut r, &mut rng, size);
            r.sop1(SOP1_SETPC_B64, 0, 30);
            (GUEST + 0x40_0000 + k * 0x1_0000, r.finish())
        })
        .collect();
    Job {
        regions,
        ..Job::synthetic("linked", a.finish(), *base)
    }
}
