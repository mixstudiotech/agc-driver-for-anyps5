//! Computed control flow: where `s_setpc_b64` and `s_swappc_b64` jump.
//!
//! A forward dataflow over the scalar registers tracks which hold code addresses.
//! `s_getpc_b64` yields `code base + pc + 4`, and 64-bit add/subtract pairs with constants
//! (`s_add_u32` + `s_addc_u32`, `s_sub_u32` + `s_subb_u32`) keep the offset known, so a jump
//! through such a pair has a [`JumpTarget::Static`] target. Calls write their return address,
//! and a jump through return addresses is a [`JumpTarget::Return`]. A code address plus data
//! loaded from the code (a jump table) may reach any instruction: [`JumpTarget::Open`].
//! Values from outside the program (user data, memory, constants) are
//! [`JumpTarget::External`]. Anything the dataflow cannot follow is treated as open.
//!
//! Calls flow into their callee and returns flow back to every return site, so a callee's
//! effects reach the code after its calls.

use std::collections::{BTreeMap, BTreeSet};

use crate::decode::DecodeScope;
use crate::ir::opcodes::flag;
use crate::ir::{
    CodeLayout, ComputedFlow, ExternalTarget, Fields, Instruction, InstructionFamily as F,
    JumpTarget, Operand, RegFile, ShaderIr,
};
use crate::CompileError;

const SCALARS: usize = 128;
/// s0-s105: the SGPRs a wave starts with values in.
const ENTRY_SGPRS: usize = 106;
const SECOND_HALF_SGPR: usize = 6;
const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP1_CMOV_B32: u16 = 0x05;
const SOP1_CMOV_B64: u16 = 0x06;
pub const SOP1_GETPC_B64: u16 = 0x1f;
pub const SOP1_SETPC_B64: u16 = 0x20;
pub const SOP1_SWAPPC_B64: u16 = 0x21;
pub const SOPK_CALL_B64: u16 = 0x16;
const SOP2_ADD_U32: u16 = 0x00;
const SOP2_SUB_U32: u16 = 0x01;
const SOP2_ADDC_U32: u16 = 0x04;
const SOP2_SUBB_U32: u16 = 0x05;
const SOP2_CSELECT_B32: u16 = 0x0a;
const SOP2_CSELECT_B64: u16 = 0x0b;

/// What a scalar register holds, as far as code addresses are concerned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Word {
    /// No path reaches this point yet.
    Unset,
    /// From outside the program: user data, system values or memory.
    External,
    /// The value SGPR n held when the wave started (also from outside the program).
    Entry(u8),
    Const(u32),
    /// The low or high dword of `code base + offset`, where `offset` counts guest bytes from
    /// the program's base, linked regions included (see [`CodeLayout`]).
    Code {
        offset: i64,
        high: bool,
    },
    /// The low or high dword of some call's return address.
    Return {
        high: bool,
    },
    /// The low or high dword of a code address plus an unknown amount.
    CodeRelative {
        high: bool,
    },
    /// A dword loaded from the code region.
    CodeData,
    Unknown,
}

impl Word {
    fn code_derived(self) -> bool {
        matches!(
            self,
            Self::Code { .. } | Self::Return { .. } | Self::CodeRelative { .. } | Self::CodeData
        )
    }

    fn outside(self) -> bool {
        matches!(self, Self::External | Self::Entry(_) | Self::Const(_))
    }
}

/// SCC as the carry or borrow of a 64-bit address computation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Carry {
    Unset,
    /// Out of adding or subtracting `amount` to or from the low dword of `code base + offset`.
    Code {
        offset: i64,
        amount: u32,
        subtract: bool,
    },
    CodeRelative,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct State {
    words: [Word; SCALARS],
    scc: Carry,
}

impl State {
    fn unset() -> Self {
        Self {
            words: [Word::Unset; SCALARS],
            scc: Carry::Unset,
        }
    }

    fn entry() -> Self {
        let mut words = [Word::External; SCALARS];
        for (n, word) in words.iter_mut().enumerate().take(ENTRY_SGPRS) {
            *word = Word::Entry(n as u8);
        }
        Self {
            words,
            scc: Carry::Unknown,
        }
    }

    fn reached(&self) -> bool {
        self.words[0] != Word::Unset
    }

    fn read(&self, op: Operand) -> Word {
        match op {
            Operand::InlineInt(v) => Word::Const(v as i32 as u32),
            Operand::InlineFloat(c) => Word::Const(Operand::inline_float_bits(c).unwrap_or(0)),
            Operand::Literal(v) => Word::Const(v),
            Operand::Null => Word::Const(0),
            _ => match op.scalar_index() {
                Some(i) if usize::from(i) < SCALARS => self.words[usize::from(i)],
                _ => Word::External,
            },
        }
    }

    fn read_pair(&self, op: Operand) -> (Word, Word) {
        match op.scalar_index() {
            Some(i) if usize::from(i) + 1 < SCALARS => {
                (self.words[usize::from(i)], self.words[usize::from(i) + 1])
            }
            _ => (self.read(op), Word::Const(0)),
        }
    }

    fn write(&mut self, op: Operand, word: Word) {
        if let Some(i) = op.scalar_index().filter(|&i| usize::from(i) < SCALARS) {
            self.words[usize::from(i)] = word;
        }
    }

    fn write_pair(&mut self, op: Operand, (lo, hi): (Word, Word)) {
        if let Some(i) = op.scalar_index().filter(|&i| usize::from(i) + 1 < SCALARS) {
            self.words[usize::from(i)] = lo;
            self.words[usize::from(i) + 1] = hi;
        }
    }

    fn merge(&mut self, other: &Self, returns: &BTreeSet<i64>) -> bool {
        let mut changed = false;
        for (word, &incoming) in self.words.iter_mut().zip(&other.words) {
            let merged = merge_word(*word, incoming, returns);
            changed |= merged != *word;
            *word = merged;
        }
        let scc = match (self.scc, other.scc) {
            (Carry::Unset, c) | (c, Carry::Unset) => c,
            (a, b) if a == b => a,
            (
                Carry::Code { .. } | Carry::CodeRelative,
                Carry::Code { .. } | Carry::CodeRelative,
            ) => Carry::CodeRelative,
            _ => Carry::Unknown,
        };
        changed |= scc != self.scc;
        self.scc = scc;
        changed
    }
}

fn merge_word(a: Word, b: Word, returns: &BTreeSet<i64>) -> Word {
    use Word::*;
    let is_return = |w: Word, high: bool| match w {
        Return { high: h } => h == high,
        Code { offset, high: h } => h == high && returns.contains(&offset),
        _ => false,
    };
    match (a, b) {
        (Unset, w) | (w, Unset) => w,
        (x, y) if x == y => x,
        (Code { high, .. } | Return { high }, _) if is_return(a, high) && is_return(b, high) => {
            Return { high }
        }
        (
            Code { high, .. } | Return { high } | CodeRelative { high },
            Code { high: other, .. } | Return { high: other } | CodeRelative { high: other },
        ) if high == other => CodeRelative { high },
        (x, y) if x.outside() && y.outside() => External,
        _ => Unknown,
    }
}

/// The low dword of a 64-bit add or subtract.
fn low_dword(a: Word, b: Word, subtract: bool) -> (Word, Carry) {
    use Word::*;
    match (a, b) {
        (
            Code {
                offset,
                high: false,
            },
            Const(amount),
        )
        | (
            Const(amount),
            Code {
                offset,
                high: false,
            },
        ) if !subtract || matches!(a, Code { .. }) => {
            let amount64 = i64::from(amount);
            let moved = if subtract {
                offset.wrapping_sub(amount64)
            } else {
                offset.wrapping_add(amount64)
            };
            let carry = Carry::Code {
                offset,
                amount,
                subtract,
            };
            (
                Code {
                    offset: moved,
                    high: false,
                },
                carry,
            )
        }
        (Code { high: false, .. } | CodeRelative { high: false }, _)
        | (_, Code { high: false, .. } | CodeRelative { high: false }) => {
            (CodeRelative { high: false }, Carry::CodeRelative)
        }
        (x, y) if x.outside() && y.outside() => (External, Carry::Unknown),
        _ => (Unknown, Carry::Unknown),
    }
}

/// The high dword of a 64-bit add or subtract, given the low dword's carry in SCC.
fn high_dword(a: Word, b: Word, carry: Carry, subtract: bool) -> Word {
    use Word::*;
    let code_high = match (a, b) {
        (Code { offset, high: true }, Const(h)) => Some((offset, h)),
        (Const(h), Code { offset, high: true }) if !subtract => Some((offset, h)),
        _ => None,
    };
    match (code_high, carry) {
        (
            Some((offset, h)),
            Carry::Code {
                offset: low_offset,
                amount,
                subtract: low_subtract,
            },
        ) if low_offset == offset && low_subtract == subtract => {
            let total = ((u64::from(h) << 32) | u64::from(amount)) as i64;
            let moved = if subtract {
                offset.wrapping_sub(total)
            } else {
                offset.wrapping_add(total)
            };
            Code {
                offset: moved,
                high: true,
            }
        }
        _ if a.code_derived() || b.code_derived() => CodeRelative { high: true },
        _ if a.outside()
            && b.outside()
            && !matches!(carry, Carry::Code { .. } | Carry::CodeRelative) =>
        {
            External
        }
        _ => Unknown,
    }
}

/// The analysis' view of one program: return sites and how jumps are classified.
struct Analysis<'a> {
    ir: &'a ShaderIr,
    layout: CodeLayout,
    /// Return sites as the guest offsets calls write, and as pcs.
    returns: BTreeSet<i64>,
    return_pcs: Vec<u32>,
    code_bytes: i64,
}

impl Analysis<'_> {
    /// The guest address of `pc` as an offset from the program's base.
    fn offset(&self, pc: i64) -> i64 {
        let base = self.layout.base.unwrap_or(0);
        self.layout.guest_address(pc as u32).wrapping_sub(base) as i64
    }

    fn classify(&self, state: &State, inst: &Instruction) -> Option<JumpTarget> {
        let Fields::Sop1 { src0, .. } = inst.fields else {
            return None;
        };
        let (lo, hi) = state.read_pair(src0);
        let is_return = |w: Word, high: bool| match w {
            Word::Return { high: h } => h == high,
            Word::Code { offset, high: h } => h == high && self.returns.contains(&offset),
            _ => false,
        };
        let base = self.layout.base;
        Some(match (lo, hi) {
            (Word::Unset, _) | (_, Word::Unset) => return None,
            (
                Word::Code {
                    offset: low,
                    high: false,
                },
                Word::Code { offset, high: true },
            ) if low as u32 == offset as u32 => {
                self.code_target(base.unwrap_or(0).wrapping_add(offset as u64))
            }
            _ if is_return(lo, false) && is_return(hi, true) => JumpTarget::Return,
            (Word::Const(low), Word::Const(high)) if base.is_some() => {
                self.code_target((u64::from(high) << 32) | u64::from(low))
            }
            (x, y) if x.outside() && y.outside() => JumpTarget::External,
            _ => JumpTarget::Open,
        })
    }

    /// The SGPR pair a jump reads, when it still holds the wave's input.
    fn entry_pair(state: &State, inst: &Instruction) -> Option<u8> {
        let Fields::Sop1 { src0, .. } = inst.fields else {
            return None;
        };
        match state.read_pair(src0) {
            (Word::Entry(lo), Word::Entry(hi)) if hi == lo + 1 => Some(lo),
            _ => None,
        }
    }

    fn code_target(&self, address: u64) -> JumpTarget {
        match self.layout.pc_of(address) {
            Some(pc) if i64::from(pc) < self.code_bytes && pc % 4 == 0 => JumpTarget::Static(pc),
            _ => JumpTarget::External,
        }
    }

    fn block_at(&self, pc: u32) -> Option<usize> {
        let index = self.ir.instruction_at(pc)?;
        self.ir.blocks.iter().position(|b| b.first == index)
    }

    fn return_blocks(&self) -> Vec<usize> {
        self.return_pcs
            .iter()
            .filter_map(|&pc| self.block_at(pc))
            .collect()
    }

    fn region_blocks(&self) -> Vec<usize> {
        self.layout
            .regions
            .iter()
            .filter_map(|r| self.block_at(r.start))
            .collect()
    }
}

/// Applies one instruction to the register state.
fn step(state: &mut State, inst: &Instruction, analysis: &Analysis) {
    let next_pc = i64::from(inst.pc) + 4 * inst.word_count() as i64;
    let address = |high| Word::Code {
        offset: analysis.offset(next_pc),
        high,
    };
    let preserves_scc = !inst.family.is_scalar_alu()
        || inst.family == F::Sopp
        || matches!(
            (inst.family, inst.opcode),
            (
                F::Sop1,
                SOP1_MOV_B32
                    | SOP1_MOV_B64
                    | SOP1_CMOV_B32
                    | SOP1_CMOV_B64
                    | SOP1_GETPC_B64
                    | SOP1_SETPC_B64
                    | SOP1_SWAPPC_B64
            ) | (F::Sopk, SOPK_CALL_B64)
        );
    match (&inst.fields, inst.family, inst.opcode) {
        (Fields::Sop1 { sdst, .. }, F::Sop1, SOP1_GETPC_B64) => {
            state.write_pair(*sdst, (address(false), address(true)));
        }
        (Fields::Sop1 { sdst, .. }, F::Sop1, SOP1_SWAPPC_B64) if !inst.has_flag(flag::NO_DST) => {
            state.write_pair(*sdst, (address(false), address(true)));
        }
        (Fields::Sopk { sdst, .. }, F::Sopk, SOPK_CALL_B64) => {
            state.write_pair(*sdst, (address(false), address(true)));
        }
        (Fields::Sop1 { sdst, src0 }, F::Sop1, SOP1_MOV_B32) => {
            let word = state.read(*src0);
            state.write(*sdst, word);
        }
        (Fields::Sop1 { sdst, src0 }, F::Sop1, SOP1_MOV_B64) => {
            let pair = state.read_pair(*src0);
            state.write_pair(*sdst, pair);
        }
        (Fields::Sop1 { sdst, src0 }, F::Sop1, SOP1_CMOV_B32 | SOP1_CMOV_B64) => {
            let wide = inst.opcode == SOP1_CMOV_B64;
            for k in 0..=u8::from(wide) {
                let (old, new) = (state.read(offset(*sdst, k)), state.read(offset(*src0, k)));
                state.write(offset(*sdst, k), merge_word(old, new, &BTreeSet::new()));
            }
        }
        (Fields::Sop2 { sdst, src0, src1 }, F::Sop2, SOP2_ADD_U32 | SOP2_SUB_U32) => {
            let subtract = inst.opcode == SOP2_SUB_U32;
            let (word, carry) = low_dword(state.read(*src0), state.read(*src1), subtract);
            state.write(*sdst, word);
            state.scc = carry;
            return;
        }
        (Fields::Sop2 { sdst, src0, src1 }, F::Sop2, SOP2_ADDC_U32 | SOP2_SUBB_U32) => {
            let subtract = inst.opcode == SOP2_SUBB_U32;
            let word = high_dword(state.read(*src0), state.read(*src1), state.scc, subtract);
            state.write(*sdst, word);
            state.scc = Carry::Unknown;
            return;
        }
        (Fields::Sop2 { sdst, src0, src1 }, F::Sop2, SOP2_CSELECT_B32 | SOP2_CSELECT_B64) => {
            let wide = inst.opcode == SOP2_CSELECT_B64;
            for k in 0..=u8::from(wide) {
                let (a, b) = (state.read(offset(*src0, k)), state.read(offset(*src1, k)));
                state.write(offset(*sdst, k), merge_word(a, b, &BTreeSet::new()));
            }
            return;
        }
        (Fields::Smem(s), F::Smem, _) if s.dwords > 0 => {
            let base = Operand::Sgpr(s.sbase);
            let (lo, hi) = state.read_pair(base);
            let loaded = if !s.buffer && (lo.code_derived() || hi.code_derived()) {
                Word::CodeData
            } else {
                Word::External
            };
            for k in 0..s.dwords {
                state.write(offset(s.sdata, k), loaded);
            }
        }
        _ => {
            if inst.has_flag(flag::RELATIVE) && inst.family.is_scalar_alu() {
                state.words = [Word::Unknown; SCALARS];
            }
            let scalars = |r: crate::ir::RegRange| {
                let all = if r.file == RegFile::Scalar {
                    r.first..r.first + r.count
                } else {
                    0..0
                };
                all.map(usize::from).filter(|&i| i < SCALARS)
            };
            let mut code_input = false;
            inst.each_use(true, |r| {
                code_input |= scalars(r)
                    .any(|i| state.words[i].code_derived() || state.words[i] == Word::Unknown);
            });
            let result = if code_input || inst.family.is_vector_alu() {
                Word::Unknown
            } else {
                Word::External
            };
            inst.each_def(true, |r| {
                for i in scalars(r) {
                    state.words[i] = result;
                }
            });
        }
    }
    if !preserves_scc {
        state.scc = Carry::Unknown;
    }
}

fn offset(op: Operand, k: u8) -> Operand {
    match op {
        Operand::Sgpr(n) => Operand::Sgpr(n.wrapping_add(k)),
        Operand::Ttmp(n) => Operand::Ttmp(n.wrapping_add(k)),
        Operand::VccLo if k == 1 => Operand::VccHi,
        Operand::ExecLo if k == 1 => Operand::ExecHi,
        other if k == 0 => other,
        _ => Operand::Null,
    }
}

fn is_computed_jump(inst: &Instruction) -> bool {
    inst.family == F::Sop1 && matches!(inst.opcode, SOP1_SETPC_B64 | SOP1_SWAPPC_B64)
}

fn is_call(inst: &Instruction) -> bool {
    inst.has_flag(flag::CALL)
}

/// Classifies every reachable computed jump of `ir`, whose blocks must be built. `base` is the
/// guest address of the first code word, when known (it defaults to `ir.layout`'s), and
/// `code_words` the size of the code buffer, which may extend past what has been decoded so
/// far. A jump or call to code outside the program may enter every linked region, which is
/// how a call through user data reaches a linked fetch subroutine.
pub fn analyze(ir: &ShaderIr, base: Option<u64>, code_words: usize) -> ComputedFlow {
    if !ir
        .instructions
        .iter()
        .any(|i| is_computed_jump(i) || is_call(i))
    {
        return ComputedFlow {
            entries: ir.layout.regions.iter().map(|r| r.start).collect(),
            ..ComputedFlow::default()
        };
    }
    let mut layout = ir.layout.clone();
    layout.base = base.or(layout.base);
    if layout.own_bytes == 0 {
        layout.own_bytes = 4 * code_words as u32;
    }
    let mut analysis = Analysis {
        ir,
        layout,
        returns: BTreeSet::new(),
        return_pcs: Vec::new(),
        code_bytes: 4 * code_words as i64,
    };
    analysis.return_pcs = ir
        .instructions
        .iter()
        .filter(|i| is_call(i))
        .map(|i| i.pc + 4 * i.word_count() as u32)
        .collect();
    analysis.returns = analysis
        .return_pcs
        .iter()
        .map(|&pc| analysis.offset(i64::from(pc)))
        .collect();
    let region_blocks = analysis.region_blocks();
    let count = ir.blocks.len();
    let mut entry = vec![State::unset(); count];
    entry[0] = State::entry();
    if let Some(address) = analysis.layout.second_half {
        entry[0].words[SECOND_HALF_SGPR] = Word::Const(address as u32);
        entry[0].words[SECOND_HALF_SGPR + 1] = Word::Const((address >> 32) as u32);
    }
    let mut jumps: BTreeMap<u32, JumpTarget> = BTreeMap::new();
    let mut external: BTreeMap<u32, ExternalTarget> = BTreeMap::new();
    let mut work: BTreeSet<usize> = (0..count).collect();
    while let Some(id) = work.pop_first() {
        if !entry[id].reached() {
            continue;
        }
        let block = &ir.blocks[id];
        let mut state = entry[id].clone();
        let instructions = &ir.instructions[block.instructions()];
        let (last, body) = instructions.split_last().expect("blocks are not empty");
        for inst in body {
            step(&mut state, inst, &analysis);
        }
        let mut edges: Vec<(usize, State)> = Vec::new();
        if is_computed_jump(last) || is_call(last) {
            let target = if last.family == F::Sopk {
                last.branch_target.map(JumpTarget::Static)
            } else {
                analysis.classify(&state, last)
            };
            if let (Some(target), true) = (target, is_computed_jump(last)) {
                jumps.insert(last.pc, target);
            }
            let returns = is_call(last) && !last.has_flag(flag::NO_DST);
            if target == Some(JumpTarget::External) {
                let target = ExternalTarget {
                    pc: last.pc,
                    call: returns,
                    sgpr: Analysis::entry_pair(&state, last),
                };
                external.insert(last.pc, target);
            }
            step(&mut state, last, &analysis);
            match target {
                Some(JumpTarget::Static(pc)) => {
                    edges.extend(analysis.block_at(pc).map(|b| (b, state.clone())));
                }
                Some(JumpTarget::Return) if !returns => {
                    for b in analysis.return_blocks() {
                        edges.push((b, state.clone()));
                    }
                }
                Some(JumpTarget::External) => {
                    edges.extend(region_blocks.iter().map(|&b| (b, state.clone())));
                }
                _ => {}
            }
            // A call to unknown code comes back with unknown registers. A call out of the
            // program with linked regions enters a region, whose return jump carries its state
            // back, or misses and ends the wave.
            let external = target == Some(JumpTarget::External);
            let linked = external && !region_blocks.is_empty();
            if returns && !linked && !matches!(target, Some(JumpTarget::Static(_))) {
                let mut clobbered = State::entry();
                let left = if external {
                    Word::External
                } else {
                    Word::Unknown
                };
                clobbered.words = [left; SCALARS];
                if let Fields::Sop1 { sdst, .. } | Fields::Sopk { sdst, .. } = last.fields {
                    clobbered.write_pair(sdst, state.read_pair(sdst));
                }
                edges.extend(block.successors.iter().map(|&s| (s, clobbered.clone())));
            }
        } else {
            step(&mut state, last, &analysis);
            edges.extend(block.successors.iter().map(|&s| (s, state.clone())));
        }
        for (successor, outgoing) in edges {
            if entry[successor].merge(&outgoing, &analysis.returns) {
                work.insert(successor);
            }
        }
    }
    let mut flow = ComputedFlow {
        jumps,
        entries: BTreeSet::new(),
        external: external.into_values().collect(),
    };
    if flow.has_dynamic_jumps() {
        flow.entries.extend(analysis.return_pcs.iter().copied());
    }
    flow.entries
        .extend(analysis.layout.regions.iter().map(|r| r.start));
    if flow.is_open() {
        flow.entries.extend(ir.instructions.iter().map(|i| i.pc));
    }
    flow
}

/// Records `flow` in `ir` before its blocks are built: every static computed jump gets its
/// target as `branch_target`, so blocks, edges and reachability follow it.
pub fn apply(ir: &mut ShaderIr, flow: &ComputedFlow) -> Result<(), CompileError> {
    for (&pc, &target) in &flow.jumps {
        let JumpTarget::Static(to) = target else {
            continue;
        };
        if ir.instruction_at(to).is_none() {
            return Err(CompileError::InvalidBranch { pc, target: to });
        }
        if let Some(index) = ir.instruction_at(pc) {
            ir.instructions[index].branch_target = Some(to);
        }
    }
    ir.flow = flow.clone();
    Ok(())
}

/// The decoding a flow needs: its entries and static targets, and the whole buffer when some
/// jump is open.
pub fn decode_scope(flow: &ComputedFlow) -> DecodeScope {
    let mut entries = flow.entries.clone();
    entries.extend(flow.jumps.values().filter_map(|t| match t {
        JumpTarget::Static(pc) => Some(*pc),
        _ => None,
    }));
    DecodeScope {
        entries,
        through_jumps: flow.is_open(),
    }
}
