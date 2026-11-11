//! Where descriptors come from: for every instruction that uses a V#, T# or S#, whether the
//! descriptor is still the wave's user data, was read from memory by scalar loads at constant
//! offsets starting from user data ([`DescriptorTable`]), or cannot be followed.
//!
//! A forward dataflow over the scalar registers follows copies (`s_mov_b32`, `s_mov_b64`),
//! constants, 64-bit pointer arithmetic with constants (`s_add_u32` + `s_addc_u32`,
//! `s_sub_u32` + `s_subb_u32`) and scalar loads, through a pointer (`s_load_dword*`) or a V#
//! (`s_buffer_load_dword*`). Paths join by agreement: a register whose incoming values differ
//! is unknown. Computed jumps add the edges `cfg::computed` found: returns reach the code after
//! every call, and jumps out of the program reach every linked region, so a linked fetch
//! subroutine sees its caller's registers.

use std::collections::BTreeSet;
use std::ops::Range;

use crate::ir::memops::{self, img};
use crate::ir::opcodes::flag;
use crate::ir::{
    AddressForm, BufferAccess, DescriptorAccess, DescriptorSource, DescriptorTable, DescriptorUse,
    Fields, Instruction, InstructionFamily as F, JumpTarget, MemKind, Operand, RegFile,
    ResourceKind, ScalarOffset, ShaderIr, Smem, TableLink,
};

/// s0-s105.
const SGPRS: usize = 106;
/// Longest chain of addresses followed to a descriptor.
const MAX_LINKS: usize = 16;
const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP2_ADD_U32: u16 = 0x00;
const SOP2_SUB_U32: u16 = 0x01;
const SOP2_ADDC_U32: u16 = 0x04;
const SOP2_SUBB_U32: u16 = 0x05;
const SOP2_AND_B32: u16 = 0x0e;
const SOP2_LSHL_B32: u16 = 0x1e;
const SOP2_LSHR_B32: u16 = 0x20;
const SOPK_MOVK_I32: u16 = 0x00;
/// `s_load_dword` to `s_load_dwordx16`.
const SMEM_LOAD_LAST: u16 = 0x04;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Val {
    /// No path reaches this point yet.
    Unset,
    Unknown,
    Const(u32),
    /// A subset of the bits in `mask` can be set; all other bits are zero. This is an
    /// over-approximation of scalar table indices, never a guessed descriptor address.
    Masked(u32),
    /// Dword `dword` of `origin`, with `add` added to the 64-bit pointer it is part of.
    Word {
        origin: Origin,
        dword: u8,
        add: i64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Origin {
    /// The wave's input SGPRs; `dword` is the SGPR.
    Input,
    /// What a scalar load read, an index into [`Analysis::tables`].
    Load(usize),
}

/// SCC as the carry or borrow out of a pointer's low dword.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Carry {
    Unset,
    Unknown,
    Pointer {
        origin: Origin,
        dword: u8,
        add: i64,
        amount: u32,
        subtract: bool,
    },
}

/// The SGPRs the dataflow tracks: those a descriptor use reads, and everything a tracked
/// register is copied, computed or loaded from (a flow-insensitive closure). No other register
/// can reach a result, so the others read as [`Val::Unknown`].
#[derive(Debug)]
struct Slots {
    /// Each SGPR's index in [`State::regs`], or [`UNTRACKED`].
    of: [u8; SGPRS],
    count: usize,
}

const UNTRACKED: u8 = u8::MAX;

impl Slots {
    fn new(ir: &ShaderIr) -> Self {
        let mut tracked = [false; SGPRS];
        let mark = |tracked: &mut [bool; SGPRS], first: u16, count: u16| {
            let mut changed = false;
            for n in first..first + count {
                if let Some(t) = tracked.get_mut(usize::from(n)) {
                    changed |= !*t;
                    *t = true;
                }
            }
            changed
        };
        let mark_op = |tracked: &mut [bool; SGPRS], op: Operand, count: u16| match op {
            Operand::Sgpr(n) => mark(tracked, u16::from(n), count),
            _ => false,
        };
        let any = |tracked: &[bool; SGPRS], first: u16, count: u16| {
            (first..first + count).any(|n| tracked.get(usize::from(n)) == Some(&true))
        };
        for inst in &ir.instructions {
            for (first, count) in descriptor_operands(inst) {
                mark(&mut tracked, u16::from(first), u16::from(count));
            }
            if let Fields::Smem(s) = &inst.fields {
                if !s.buffer
                    && s.dwords > 0
                    && memops::smem(inst.opcode).is_some_and(|op| op.kind != MemKind::Control)
                {
                    mark(&mut tracked, u16::from(s.sbase), 2);
                    mark_op(&mut tracked, s.soffset, 1);
                }
            }
        }
        // Whether some add-with-carry builds a tracked register, so every carry may matter.
        let mut carry = false;
        loop {
            let mut changed = false;
            for inst in &ir.instructions {
                match (&inst.fields, inst.family, inst.opcode) {
                    (&Fields::Sop1 { sdst, src0 }, F::Sop1, SOP1_MOV_B32 | SOP1_MOV_B64) => {
                        let wide = u16::from(inst.opcode == SOP1_MOV_B64);
                        if let Operand::Sgpr(d) = sdst {
                            if any(&tracked, u16::from(d), 1 + wide) {
                                changed |= mark_op(&mut tracked, src0, 1 + wide);
                            }
                        }
                    }
                    (&Fields::Sop2 { sdst, src0, src1 }, F::Sop2, op)
                        if matches!(
                            op,
                            SOP2_ADD_U32
                                | SOP2_SUB_U32
                                | SOP2_ADDC_U32
                                | SOP2_SUBB_U32
                                | SOP2_AND_B32
                                | SOP2_LSHL_B32
                                | SOP2_LSHR_B32
                        ) =>
                    {
                        let with_carry = matches!(op, SOP2_ADDC_U32 | SOP2_SUBB_U32);
                        let builds =
                            matches!(sdst, Operand::Sgpr(d) if any(&tracked, u16::from(d), 1));
                        if builds || (carry && !with_carry) {
                            changed |= mark_op(&mut tracked, src0, 1);
                            changed |= mark_op(&mut tracked, src1, 1);
                        }
                        if builds && with_carry && !carry {
                            carry = true;
                            changed = true;
                        }
                    }
                    (Fields::Smem(s), F::Smem, op) if is_table_load(inst, s, op) => {
                        let loads_tracked = matches!(s.sdata, Operand::Sgpr(first)
                            if any(&tracked, u16::from(first), u16::from(s.dwords)));
                        if loads_tracked {
                            let base = if s.buffer { 4 } else { 2 };
                            changed |= mark(&mut tracked, u16::from(s.sbase), base);
                            changed |= mark_op(&mut tracked, s.soffset, 1);
                        }
                    }
                    (Fields::Mubuf(b) | Fields::Mtbuf(b), _, _) if b.kind != MemKind::Control => {
                        changed |= mark_op(&mut tracked, b.soffset, 1);
                    }
                    _ => {}
                }
            }
            if !changed {
                break;
            }
        }
        let mut of = [UNTRACKED; SGPRS];
        let mut count = 0;
        for (slot, _) in of.iter_mut().zip(tracked).filter(|(_, t)| *t) {
            *slot = count as u8;
            count += 1;
        }
        Self { of, count }
    }
}

/// The SGPR ranges `inst` reads as descriptors (see [`Analysis::record`]).
fn descriptor_operands(inst: &Instruction) -> impl Iterator<Item = (u8, u8)> {
    let ranges: [Option<(u8, u8)>; 2] = match &inst.fields {
        Fields::Smem(s) if s.buffer && s.dwords > 0 && !inst.has_flag(flag::NO_SRC) => {
            [Some((s.sbase, 4)), None]
        }
        Fields::Mubuf(b) | Fields::Mtbuf(b) if b.kind != MemKind::Control => {
            [Some((b.srsrc, 4)), None]
        }
        Fields::Mimg(m) => [
            Some((m.srsrc, if m.r128 { 4 } else { 8 })),
            m.uses_sampler.then_some((m.ssamp, 4)),
        ],
        _ => [None, None],
    };
    ranges.into_iter().flatten()
}

/// A scalar load whose address the dataflow follows (see [`Analysis::table_load`]).
fn is_table_load(inst: &Instruction, s: &Smem, op: u16) -> bool {
    s.dwords > 0
        && !inst.has_flag(flag::NO_SRC)
        && (s.buffer || op <= SMEM_LOAD_LAST)
        && memops::smem(op).is_some_and(|op| op.kind == MemKind::Load)
}

#[derive(Clone, Debug)]
struct State<'s> {
    slots: &'s Slots,
    /// The tracked SGPRs, by slot.
    regs: Vec<Val>,
    scc: Carry,
    reached: bool,
}

fn join_val(a: Val, b: Val) -> Val {
    match (a, b) {
        (Val::Unset, v) | (v, Val::Unset) => v,
        (x, y) if x == y => x,
        _ => Val::Unknown,
    }
}

impl<'s> State<'s> {
    fn unset(slots: &'s Slots) -> Self {
        Self {
            slots,
            regs: vec![Val::Unset; slots.count],
            scc: Carry::Unset,
            reached: false,
        }
    }

    fn entry(slots: &'s Slots) -> Self {
        let mut state = Self::unknown(slots);
        for (n, &slot) in slots.of.iter().enumerate() {
            if slot != UNTRACKED {
                state.regs[usize::from(slot)] = Val::Word {
                    origin: Origin::Input,
                    dword: n as u8,
                    add: 0,
                };
            }
        }
        state
    }

    fn unknown(slots: &'s Slots) -> Self {
        Self {
            slots,
            regs: vec![Val::Unknown; slots.count],
            scc: Carry::Unknown,
            reached: true,
        }
    }

    fn reached(&self) -> bool {
        self.reached
    }

    fn tracks(&self, n: u16) -> bool {
        self.slots
            .of
            .get(usize::from(n))
            .is_some_and(|&slot| slot != UNTRACKED)
    }

    /// `self = other.clone()`, keeping the allocation.
    fn copy_from(&mut self, other: &Self) {
        self.regs.copy_from_slice(&other.regs);
        self.scc = other.scc;
        self.reached = other.reached;
    }

    fn reg(&self, n: u16) -> Val {
        match self.slots.of.get(usize::from(n)) {
            Some(&slot) if slot != UNTRACKED => self.regs[usize::from(slot)],
            _ => Val::Unknown,
        }
    }

    fn read(&self, op: Operand) -> Val {
        match op {
            Operand::Sgpr(n) => self.reg(u16::from(n)),
            Operand::InlineInt(v) => Val::Const(v as i32 as u32),
            Operand::InlineFloat(c) => {
                Operand::inline_float_bits(c).map_or(Val::Unknown, Val::Const)
            }
            Operand::Literal(v) => Val::Const(v),
            Operand::Null => Val::Const(0),
            _ => Val::Unknown,
        }
    }

    /// The high dword of a 64-bit source: the next SGPR, or a constant's sign extension.
    fn read_high(&self, op: Operand) -> Val {
        match op {
            Operand::Sgpr(n) => self.reg(u16::from(n) + 1),
            Operand::InlineInt(v) => Val::Const(if v < 0 { u32::MAX } else { 0 }),
            Operand::Null => Val::Const(0),
            _ => Val::Unknown,
        }
    }

    fn write(&mut self, op: Operand, value: Val) {
        if let Operand::Sgpr(n) = op {
            self.write_reg(u16::from(n), value);
        }
    }

    fn write_reg(&mut self, n: u16, value: Val) {
        if let Some(&slot) = self.slots.of.get(usize::from(n)) {
            if slot != UNTRACKED {
                self.regs[usize::from(slot)] = value;
            }
        }
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut changed = other.reached && !self.reached;
        self.reached |= other.reached;
        for (reg, &incoming) in self.regs.iter_mut().zip(&other.regs) {
            let joined = join_val(*reg, incoming);
            changed |= joined != *reg;
            *reg = joined;
        }
        let scc = match (self.scc, other.scc) {
            (Carry::Unset, c) | (c, Carry::Unset) => c,
            (a, b) if a == b => a,
            _ => Carry::Unknown,
        };
        changed |= scc != self.scc;
        self.scc = scc;
        changed
    }
}

/// The low dword of a 64-bit add or subtract.
fn low_dword(a: Val, b: Val, subtract: bool) -> (Val, Carry) {
    match (a, b) {
        (Val::Unset, _) | (_, Val::Unset) => (Val::Unset, Carry::Unset),
        (Val::Word { origin, dword, add }, Val::Const(amount))
        | (Val::Const(amount), Val::Word { origin, dword, add })
            if !subtract || matches!(a, Val::Word { .. }) =>
        {
            let delta = i64::from(amount);
            let moved = if subtract {
                add.wrapping_sub(delta)
            } else {
                add.wrapping_add(delta)
            };
            let carry = Carry::Pointer {
                origin,
                dword,
                add,
                amount,
                subtract,
            };
            (
                Val::Word {
                    origin,
                    dword,
                    add: moved,
                },
                carry,
            )
        }
        (Val::Const(x), Val::Const(y)) => {
            let value = if subtract {
                x.wrapping_sub(y)
            } else {
                x.wrapping_add(y)
            };
            (Val::Const(value), Carry::Unknown)
        }
        _ => (Val::Unknown, Carry::Unknown),
    }
}

/// The high dword of a 64-bit add or subtract, given the low dword's carry in SCC.
fn high_dword(a: Val, b: Val, carry: Carry, subtract: bool) -> Val {
    let word_and_high = match (a, b) {
        (Val::Word { origin, dword, add }, Val::Const(h)) => Some((origin, dword, add, h)),
        (Val::Const(h), Val::Word { origin, dword, add }) if !subtract => {
            Some((origin, dword, add, h))
        }
        _ => None,
    };
    match (word_and_high, carry) {
        (_, Carry::Unset) => Val::Unset,
        (
            Some((origin, dword, add, h)),
            Carry::Pointer {
                origin: low_origin,
                dword: low_dword,
                add: low_add,
                amount,
                subtract: low_subtract,
            },
        ) if origin == low_origin
            && dword == low_dword.wrapping_add(1)
            && add == low_add
            && subtract == low_subtract =>
        {
            let total = ((u64::from(h) << 32) | u64::from(amount)) as i64;
            let moved = if subtract {
                add.wrapping_sub(total)
            } else {
                add.wrapping_add(total)
            };
            Val::Word {
                origin,
                dword,
                add: moved,
            }
        }
        _ => Val::Unknown,
    }
}

/// A table load as the dataflow knows it: [`DescriptorTable`] with `offset` the load's.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Table {
    root: u8,
    root_form: AddressForm,
    links: Vec<TableLink>,
    offset: i64,
    offset_mask: u32,
}

struct Analysis {
    /// Every distinct load the dataflow has seen; equal loads share an index, so their values
    /// compare equal across paths.
    tables: Vec<Table>,
    /// The input SGPRs that hold user data.
    user: Range<u16>,
}

impl Analysis {
    fn intern(&mut self, table: Table) -> usize {
        match self.tables.iter().position(|t| *t == table) {
            Some(index) => index,
            None => {
                self.tables.push(table);
                self.tables.len() - 1
            }
        }
    }

    fn is_user(&self, first: u16, count: u16) -> bool {
        self.user.start <= first && first + count <= self.user.end
    }

    /// The table a scalar load reads, when its address is followed.
    fn table_load(&mut self, state: &State, s: &Smem) -> Option<usize> {
        let (extra, offset_mask) = match state.read(s.soffset) {
            Val::Const(c) => (i64::from(c), 0),
            Val::Masked(mask) if mask.count_ones() <= 8 => (0, mask),
            _ => return None,
        };
        let base = u16::from(s.sbase);
        let (origin, dword, add, form) = if s.buffer {
            match state.reg(base) {
                Val::Word {
                    origin,
                    dword,
                    add: 0,
                } if (1..4u8).all(|k| {
                    state.reg(base + u16::from(k))
                        == Val::Word {
                            origin,
                            dword: dword.wrapping_add(k),
                            add: 0,
                        }
                }) =>
                {
                    (origin, dword, 0, AddressForm::Buffer)
                }
                _ => return None,
            }
        } else {
            match (state.reg(base), state.reg(base + 1)) {
                (
                    Val::Word { origin, dword, add },
                    Val::Word {
                        origin: high_origin,
                        dword: high_dword,
                        add: high_add,
                    },
                ) if high_origin == origin
                    && high_dword == dword.wrapping_add(1)
                    && high_add == add =>
                {
                    (origin, dword, add, AddressForm::Pointer)
                }
                _ => return None,
            }
        };
        let offset = add.wrapping_add(i64::from(s.offset)).wrapping_add(extra);
        let table = match origin {
            Origin::Input => {
                let held = if form == AddressForm::Buffer { 4 } else { 2 };
                if !self.is_user(u16::from(dword), held) {
                    return None;
                }
                Table {
                    root: dword,
                    root_form: form,
                    links: Vec::new(),
                    offset,
                    offset_mask,
                }
            }
            Origin::Load(parent) => {
                let parent = &self.tables[parent];
                if parent.links.len() >= MAX_LINKS || parent.offset_mask != 0 {
                    return None;
                }
                let mut links = parent.links.clone();
                links.push(TableLink {
                    offset: parent.offset.wrapping_add(4 * i64::from(dword)),
                    form,
                });
                Table {
                    root: parent.root,
                    root_form: parent.root_form,
                    links,
                    offset,
                    offset_mask,
                }
            }
        };
        Some(self.intern(table))
    }

    /// Applies one instruction to the register state.
    fn step(&mut self, state: &mut State, inst: &Instruction, wave64: bool) {
        let preserves_scc = !inst.family.is_scalar_alu()
            || inst.family == F::Sopp
            || matches!(
                (inst.family, inst.opcode),
                (F::Sop1, SOP1_MOV_B32 | SOP1_MOV_B64) | (F::Sopk, SOPK_MOVK_I32)
            );
        match (&inst.fields, inst.family, inst.opcode) {
            (Fields::Sop1 { sdst, src0 }, F::Sop1, SOP1_MOV_B32) => {
                let value = state.read(*src0);
                state.write(*sdst, value);
            }
            (Fields::Sop1 { sdst, src0 }, F::Sop1, SOP1_MOV_B64) => {
                let (lo, hi) = (state.read(*src0), state.read_high(*src0));
                if let Operand::Sgpr(n) = *sdst {
                    state.write_reg(u16::from(n), lo);
                    state.write_reg(u16::from(n) + 1, hi);
                }
            }
            (Fields::Sopk { sdst, simm16, .. }, F::Sopk, SOPK_MOVK_I32) => {
                state.write(*sdst, Val::Const(*simm16 as i16 as i32 as u32));
            }
            (Fields::Sop2 { sdst, src0, src1 }, F::Sop2, SOP2_ADD_U32 | SOP2_SUB_U32) => {
                let subtract = inst.opcode == SOP2_SUB_U32;
                let (value, carry) = low_dword(state.read(*src0), state.read(*src1), subtract);
                state.write(*sdst, value);
                state.scc = carry;
                return;
            }
            (Fields::Sop2 { sdst, src0, src1 }, F::Sop2, SOP2_ADDC_U32 | SOP2_SUBB_U32) => {
                let subtract = inst.opcode == SOP2_SUBB_U32;
                let value = high_dword(state.read(*src0), state.read(*src1), state.scc, subtract);
                state.write(*sdst, value);
                state.scc = Carry::Unknown;
                return;
            }
            (Fields::Sop2 { sdst, src0, src1 }, F::Sop2, SOP2_AND_B32) => {
                let a = state.read(*src0);
                let b = state.read(*src1);
                let value = match (a, b) {
                    (Val::Const(a), Val::Const(b)) => Val::Const(a & b),
                    (Val::Masked(a), Val::Const(b)) | (Val::Const(b), Val::Masked(a)) => {
                        Val::Masked(a & b)
                    }
                    (_, Val::Const(mask)) | (Val::Const(mask), _) => Val::Masked(mask),
                    _ => Val::Unknown,
                };
                state.write(*sdst, value);
            }
            (Fields::Sop2 { sdst, src0, src1 }, F::Sop2, op @ (SOP2_LSHL_B32 | SOP2_LSHR_B32)) => {
                let shift = |value: u32, amount: u32| {
                    if op == SOP2_LSHL_B32 {
                        value.wrapping_shl(amount & 31)
                    } else {
                        value.wrapping_shr(amount & 31)
                    }
                };
                let value = match (state.read(*src0), state.read(*src1)) {
                    (Val::Const(v), Val::Const(n)) => Val::Const(shift(v, n)),
                    (Val::Masked(mask), Val::Const(n)) => Val::Masked(shift(mask, n)),
                    _ => Val::Unknown,
                };
                state.write(*sdst, value);
            }
            (Fields::Smem(s), F::Smem, op) if is_table_load(inst, s, op) => {
                if let Operand::Sgpr(first) = s.sdata {
                    let first_reg = u16::from(first);
                    let table = (0..u16::from(s.dwords))
                        .any(|k| state.tracks(first_reg + k))
                        .then(|| self.table_load(state, s))
                        .flatten();
                    for k in 0..s.dwords {
                        let value = table.map_or(Val::Unknown, |index| Val::Word {
                            origin: Origin::Load(index),
                            dword: k,
                            add: 0,
                        });
                        state.write_reg(u16::from(first) + u16::from(k), value);
                    }
                }
            }
            _ => {
                if inst.has_flag(flag::RELATIVE) && inst.family.is_scalar_alu() {
                    state.regs.fill(Val::Unknown);
                }
                inst.each_def(wave64, |range| {
                    if range.file == RegFile::Scalar {
                        for n in range.first..range.first + range.count {
                            state.write_reg(n, Val::Unknown);
                        }
                    }
                });
            }
        }
        if !preserves_scc {
            state.scc = Carry::Unknown;
        }
    }

    /// Where the descriptor in `s[first..first + count]` came from.
    fn source(&self, state: &State, first: u8, count: u8) -> DescriptorSource {
        let first = u16::from(first);
        let Val::Word {
            origin,
            dword,
            add: 0,
        } = state.reg(first)
        else {
            return DescriptorSource::Dynamic;
        };
        let whole = (1..count).all(|k| {
            state.reg(first + u16::from(k))
                == Val::Word {
                    origin,
                    dword: dword.wrapping_add(k),
                    add: 0,
                }
        });
        match origin {
            _ if !whole => DescriptorSource::Dynamic,
            Origin::Input if self.is_user(u16::from(dword), u16::from(count)) => {
                DescriptorSource::UserSgprs { first: dword }
            }
            Origin::Input => DescriptorSource::Dynamic,
            Origin::Load(index) => {
                let table = &self.tables[index];
                let fixed = DescriptorTable {
                    root: table.root,
                    root_form: table.root_form,
                    links: table.links.clone(),
                    offset: table.offset.wrapping_add(4 * i64::from(dword)),
                };
                if table.offset_mask == 0 {
                    DescriptorSource::Table(fixed)
                } else {
                    let mut offsets = vec![0u32];
                    for bit in 0..32 {
                        if table.offset_mask & (1 << bit) != 0 {
                            let added: Vec<u32> =
                                offsets.iter().map(|offset| offset | (1 << bit)).collect();
                            offsets.extend(added);
                        }
                    }
                    offsets.sort_unstable();
                    DescriptorSource::TableChoices(
                        offsets
                            .into_iter()
                            .map(|offset| DescriptorTable {
                                offset: fixed.offset.wrapping_add(i64::from(offset)),
                                ..fixed.clone()
                            })
                            .collect(),
                    )
                }
            }
        }
    }

    /// The descriptor uses of `inst`, its state before it executes.
    fn record(&self, state: &State, inst: &Instruction, uses: &mut Vec<DescriptorUse>) {
        let offset = |operand| match state.read(operand) {
            Val::Const(value) | Val::Masked(value) => ScalarOffset::Bounded(value),
            Val::Word {
                origin: Origin::Input,
                dword,
                add: 0,
            } if self.is_user(u16::from(dword), 1) => ScalarOffset::UserSgpr(dword),
            _ => ScalarOffset::Unknown,
        };
        let buffer_access = match &inst.fields {
            Fields::Mubuf(b) | Fields::Mtbuf(b) => Some(BufferAccess {
                scalar: false,
                indexed: b.idxen,
                vector_offset: b.offen,
                immediate_offset: u32::from(b.offset),
                scalar_offset: offset(b.soffset),
            }),
            Fields::Smem(s) if s.buffer => Some(BufferAccess {
                scalar: true,
                indexed: false,
                vector_offset: false,
                immediate_offset: s.offset as u32,
                scalar_offset: offset(s.soffset),
            }),
            _ => None,
        };
        let mut add = |kind, base_sgpr: u8, dword_count: u8, access| {
            uses.push(DescriptorUse {
                pc: inst.pc,
                kind,
                base_sgpr,
                dword_count,
                access,
                source: self.source(state, base_sgpr, dword_count),
                buffer_access: (kind == ResourceKind::BufferDescriptor)
                    .then_some(buffer_access)
                    .flatten(),
            });
        };
        let written = |kind: MemKind| match kind {
            MemKind::Store | MemKind::Atomic => DescriptorAccess::Written,
            MemKind::Load | MemKind::Control => DescriptorAccess::Loaded,
        };
        match &inst.fields {
            Fields::Smem(s) if s.buffer && s.dwords > 0 && !inst.has_flag(flag::NO_SRC) => {
                add(
                    ResourceKind::BufferDescriptor,
                    s.sbase,
                    4,
                    written(memops::smem(inst.opcode).map_or(MemKind::Store, |op| op.kind)),
                );
            }
            Fields::Mubuf(b) | Fields::Mtbuf(b) if b.kind != MemKind::Control => {
                add(ResourceKind::BufferDescriptor, b.srsrc, 4, written(b.kind));
            }
            Fields::Mimg(m) => {
                let flags = memops::image(inst.opcode).map_or(0, |op| op.flags);
                let access = if flags & img::RESINFO != 0 {
                    DescriptorAccess::Queried
                } else if m.uses_sampler {
                    DescriptorAccess::Sampled
                } else {
                    written(m.kind)
                };
                let dwords = if m.r128 { 4 } else { 8 };
                add(ResourceKind::ImageDescriptor, m.srsrc, dwords, access);
                if m.uses_sampler {
                    add(
                        ResourceKind::SamplerDescriptor,
                        m.ssamp,
                        4,
                        DescriptorAccess::Sampled,
                    );
                }
            }
            _ => {}
        }
    }

    fn record_pointer(&self, state: &State, inst: &Instruction, uses: &mut Vec<DescriptorUse>) {
        let Fields::Smem(s) = &inst.fields else {
            return;
        };
        let Some(op) = memops::smem(inst.opcode) else {
            return;
        };
        if s.buffer || s.dwords == 0 || op.kind == MemKind::Control {
            return;
        }
        let scalar_offset = match state.read(s.soffset) {
            Val::Const(value) | Val::Masked(value) => ScalarOffset::Bounded(value),
            Val::Word {
                origin: Origin::Input,
                dword,
                add: 0,
            } if self.is_user(u16::from(dword), 1) => ScalarOffset::UserSgpr(dword),
            _ => ScalarOffset::Unknown,
        };
        let mut base_state = state.clone();
        let mut immediate_offset = s.offset;
        if let Val::Word { origin, dword, add } = state.reg(u16::from(s.sbase)) {
            if state.reg(u16::from(s.sbase) + 1)
                == (Val::Word {
                    origin,
                    dword: dword.wrapping_add(1),
                    add,
                })
            {
                if let Some(offset) = i32::try_from(add)
                    .ok()
                    .and_then(|add| s.offset.checked_add(add))
                {
                    immediate_offset = offset;
                    base_state.write_reg(
                        u16::from(s.sbase),
                        Val::Word {
                            origin,
                            dword,
                            add: 0,
                        },
                    );
                    base_state.write_reg(
                        u16::from(s.sbase) + 1,
                        Val::Word {
                            origin,
                            dword: dword.wrapping_add(1),
                            add: 0,
                        },
                    );
                }
            }
        }
        uses.push(DescriptorUse {
            pc: inst.pc,
            kind: ResourceKind::ScalarMemoryPair,
            base_sgpr: s.sbase,
            dword_count: 2,
            access: if op.kind == MemKind::Load {
                DescriptorAccess::Loaded
            } else {
                DescriptorAccess::Written
            },
            source: self.source(&base_state, s.sbase, 2),
            buffer_access: Some(BufferAccess {
                scalar: true,
                indexed: false,
                vector_offset: false,
                immediate_offset: immediate_offset as u32,
                scalar_offset,
            }),
        });
    }
}

/// Each block's successors, with the edges of computed jumps, and whether it ends with an open
/// jump, which may reach every block.
fn successors(ir: &ShaderIr) -> (Vec<Vec<usize>>, Vec<bool>) {
    let block_at = |pc: u32| {
        let index = ir.instruction_at(pc)?;
        ir.blocks.binary_search_by_key(&index, |b| b.first).ok()
    };
    let returns: Vec<usize> = ir
        .instructions
        .iter()
        .filter(|i| i.has_flag(flag::CALL))
        .filter_map(|i| block_at(i.pc + 4 * i.word_count() as u32))
        .collect();
    let regions: Vec<usize> = ir
        .layout
        .regions
        .iter()
        .filter_map(|r| block_at(r.start))
        .collect();
    ir.blocks
        .iter()
        .map(|block| {
            let mut next = block.successors.clone();
            let last = &ir.instructions[block.end - 1];
            let jump = ir.flow.jumps.get(&last.pc);
            match jump {
                Some(JumpTarget::Return) => next.extend(&returns),
                Some(JumpTarget::External) => next.extend(&regions),
                _ => {}
            }
            (next, jump == Some(&JumpTarget::Open))
        })
        .unzip()
}

/// Every reachable descriptor use of `ir` (blocks built), in pc order. `user` holds the input
/// SGPRs that carry user data.
#[derive(Default)]
pub struct Uses {
    pub descriptors: Vec<DescriptorUse>,
    pub pointers: Vec<DescriptorUse>,
}

pub fn descriptor_uses(ir: &ShaderIr, wave64: bool, user: Range<u16>) -> Uses {
    let used = |i: &Instruction| {
        descriptor_operands(i).next().is_some()
            || matches!(&i.fields, Fields::Smem(s) if !s.buffer && s.dwords > 0)
    };
    if ir.blocks.is_empty() || !ir.instructions.iter().any(used) {
        return Uses::default();
    }
    let slots = Slots::new(ir);
    let (successors, open) = successors(ir);
    let mut analysis = Analysis {
        tables: Vec::new(),
        user,
    };
    let mut entry = vec![State::unset(&slots); ir.blocks.len()];
    entry[0] = State::entry(&slots);
    // Open jumps may reach every block: all blocks join `opened`, what the open jumps leave.
    let mut opened = State::unset(&slots);
    let mut state = State::unset(&slots);
    let mut work = BTreeSet::from([0usize]);
    while let Some(id) = work.pop_first() {
        state.copy_from(&entry[id]);
        for inst in &ir.instructions[ir.blocks[id].instructions()] {
            analysis.step(&mut state, inst, wave64);
        }
        for &next in &successors[id] {
            if entry[next].join(&state) {
                work.insert(next);
            }
        }
        if open[id] && opened.join(&state) {
            for (next, incoming) in entry.iter_mut().enumerate() {
                if incoming.join(&opened) {
                    work.insert(next);
                }
            }
        }
    }
    let mut uses = Vec::new();
    let mut pointers = Vec::new();
    for (id, block) in ir.blocks.iter().enumerate().filter(|(_, b)| b.reachable) {
        // Blocks only computed control flow the dataflow cannot follow reaches.
        let mut state = if entry[id].reached() {
            entry[id].clone()
        } else {
            State::unknown(&slots)
        };
        for inst in &ir.instructions[block.instructions()] {
            analysis.record(&state, inst, &mut uses);
            analysis.record_pointer(&state, inst, &mut pointers);
            analysis.step(&mut state, inst, wave64);
        }
    }
    uses.sort_by_key(|u| u.pc);
    pointers.sort_by_key(|u| u.pc);
    Uses {
        descriptors: uses,
        pointers,
    }
}
