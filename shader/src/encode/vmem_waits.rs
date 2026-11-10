//! gfx11 VGPR write-after-write waits: an instruction that writes a VGPR a vector memory load
//! has yet to write waits for that load first. LLVM's `SIInsertWaitcnts` waits there on every
//! target (except between loads of one type, which return in order); gfx10 code that relies
//! on the lanes being disjoint does not, so this pass adds `s_waitcnt vmcnt(n)` where the guest
//! has no wait. A forward dataflow over the lowered blocks tracks the pending loads in issue
//! order; where paths with different queues meet, the order is unknown and waits use 0.

use crate::ir::{Fields, Instruction, InstructionFamily as F, Operand, RegFile, RegRange};
use crate::lowering::{derive, is_vmem_load, waitcnt, LoweredProgram};
use crate::TargetIsa;

const SOPP_GFX11_WAITCNT: u16 = 0x09;
const SOPK_GFX11_WAITCNT_VMCNT: u16 = 0x19;

/// Loads return in order only among one type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoadType {
    Plain,
    Sample,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Pending {
    /// Destinations of the loads still in flight, oldest first.
    loads: Vec<(RegRange, LoadType)>,
    /// Paths with different queues merged: positions in `loads` are unknown.
    unordered: bool,
}

impl Pending {
    fn merge(&mut self, other: &Pending) -> bool {
        if *self == *other || other.loads.is_empty() && !other.unordered {
            return false;
        }
        if self.loads.is_empty() && !self.unordered {
            *self = other.clone();
            return true;
        }
        let before = self.clone();
        for load in &other.loads {
            if !self.loads.contains(load) {
                self.loads.push(*load);
            }
        }
        self.unordered = true;
        *self != before
    }

    /// Keeps the loads `vmcnt(count)` leaves in flight.
    fn wait(&mut self, count: usize) {
        if count == 0 {
            self.loads.clear();
            self.unordered = false;
        } else if !self.unordered && self.loads.len() > count {
            self.loads.drain(..self.loads.len() - count);
        }
    }

    /// The `vmcnt` that retires every load writing one of `defs`, if any is in flight.
    fn needed(&self, defs: &[RegRange], ty: Option<LoadType>) -> Option<usize> {
        let overlaps = |r: &RegRange| defs.iter().any(|d| overlap(*d, *r));
        let youngest = self
            .loads
            .iter()
            .rposition(|(r, t)| overlaps(r) && Some(*t) != ty)?;
        let mixed = self.loads.iter().any(|(_, t)| *t != self.loads[youngest].1);
        Some(if self.unordered || mixed {
            0
        } else {
            self.loads.len() - 1 - youngest
        })
    }
}

fn overlap(a: RegRange, b: RegRange) -> bool {
    a.file == b.file && a.first < b.first + b.count && b.first < a.first + a.count
}

fn load_type(inst: &Instruction) -> LoadType {
    match inst.fields {
        Fields::Mimg(m) if m.uses_sampler => LoadType::Sample,
        _ => LoadType::Plain,
    }
}

/// The `vmcnt` a guest wait instruction leaves.
fn guest_wait(inst: &Instruction) -> Option<usize> {
    match inst.fields {
        Fields::Sopp { simm16 } if inst.family == F::Sopp && inst.opcode == SOPP_GFX11_WAITCNT => {
            Some(usize::from(waitcnt::decode_gfx11(simm16).0))
        }
        Fields::Sopk { sdst, simm16, .. }
            if inst.opcode == SOPK_GFX11_WAITCNT_VMCNT && sdst == Operand::Null =>
        {
            Some(usize::from(simm16))
        }
        _ => None,
    }
}

/// Runs a block from `state`, calling `wait(index, vmcnt)` for each wait it needs before the
/// instruction at `index`.
fn run(
    block: &[Instruction],
    state: &mut Pending,
    wave64: bool,
    mut wait: impl FnMut(usize, usize),
) {
    let mut defs = Vec::new();
    for (at, inst) in block.iter().enumerate() {
        if let Some(count) = guest_wait(inst) {
            state.wait(count);
            continue;
        }
        defs.clear();
        inst.each_def(wave64, |r| {
            if r.file == RegFile::Vector {
                defs.push(r);
            }
        });
        let vmem = is_vmem_load(inst);
        let ty = vmem.then(|| load_type(inst));
        if let Some(count) = state.needed(&defs, ty) {
            state.wait(count);
            wait(at, count);
        }
        if vmem {
            let ty = load_type(inst);
            state.loads.extend(defs.iter().map(|&r| (r, ty)));
            state.wait(usize::from(waitcnt::GFX10_VMCNT_MAX));
        }
    }
}

/// `block` with an `s_waitcnt vmcnt(n)` before each instruction `waits` names, in order.
fn splice(block: &mut Vec<Instruction>, waits: &[(usize, usize)]) {
    if waits.is_empty() {
        return;
    }
    let old = std::mem::take(block);
    block.reserve(old.len() + waits.len());
    let mut waits = waits.iter().peekable();
    for (at, inst) in old.into_iter().enumerate() {
        if let Some(&(_, count)) = waits.next_if(|&&(index, _)| index == at) {
            let fields = Fields::Sopp {
                simm16: waitcnt::encode_gfx11(
                    count as u8,
                    waitcnt::GFX10_EXPCNT_MAX,
                    waitcnt::GFX10_LGKMCNT_MAX,
                ),
            };
            block.push(derive(&inst, F::Sopp, SOPP_GFX11_WAITCNT, 0, fields));
        }
        block.push(inst);
    }
}

pub fn insert(program: &mut LoweredProgram) {
    if !matches!(program.target, TargetIsa::Gfx11 | TargetIsa::Gfx115) || program.blocks.is_empty()
    {
        return;
    }
    let succ = program.successors();
    let mut entry = vec![Pending::default(); program.blocks.len()];
    // Passes in block order over the blocks whose entry changed since they last ran; merging
    // a state again changes nothing, so the others need no run.
    let mut dirty = vec![true; program.blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for id in 0..program.blocks.len() {
            if !std::mem::take(&mut dirty[id]) {
                continue;
            }
            let mut state = entry[id].clone();
            run(
                &program.blocks[id].instructions,
                &mut state,
                program.wave64,
                |_, _| {},
            );
            for &next in &succ[id] {
                if entry[next].merge(&state) {
                    dirty[next] = true;
                    changed = true;
                }
            }
        }
    }
    let mut waits = Vec::new();
    for (block, mut state) in program.blocks.iter_mut().zip(entry) {
        waits.clear();
        run(
            &block.instructions,
            &mut state,
            program.wave64,
            |at, count| {
                waits.push((at, count));
            },
        );
        splice(&mut block.instructions, &waits);
    }
}
