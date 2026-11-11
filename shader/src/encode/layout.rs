//! Where the encoded instructions lie, and branches whose target is out of reach.
//!
//! A branch holds a signed 16-bit count of words. Translations grow (descriptor conversion and
//! ray intersection emulation turn a guest program of 12,000 words into 90,000), so a guest
//! branch can end up farther from its target than that. Such a branch goes through islands:
//! unconditional branches collected in pools between instructions, each within reach of the one
//! before, the last within reach of the target. A pool that execution would fall into starts
//! with a branch over it.
//!
//! ACO and LLVM jump through a register instead (`s_getpc_b64`, add, `s_setpc_b64`: ACO's
//! `emit_long_jump`), which needs an SGPR pair that is free at the branch. A translation cannot
//! promise one (the guest may use every SGPR), and islands need no register, leave SCC alone
//! and do not depend on where the code is placed.
//!
//! Pools go only where a branch between two instructions changes nothing: before a block, or
//! on gfx11 and later between the translations of two guest instructions, exports aside (the
//! gfx11.5 export priority workaround counts on its instructions following the export). gfx10
//! targets keep hard clauses, whose members must stay adjacent, so they get pools before blocks
//! only; their translations barely grow.

use std::collections::BTreeMap;

use crate::CompileError;

/// How far a branch reaches: its target is this many words after the word that follows it.
const REACH: std::ops::RangeInclusive<i64> = -(1 << 15)..=(1 << 15) - 1;
/// Words between pools, where the instructions allow: a branch then finds a pool well within
/// reach in either direction and an island can skip pools.
const POOL_SPACING: usize = 1 << 13;

/// One encoded instruction, or a word the encoder adds.
pub(super) struct Unit {
    pub words: Vec<u32>,
    /// The label the unit branches to; the low 16 bits of its first word hold the distance.
    pub target: Option<u32>,
    /// The guest pc the unit came from, for errors.
    pub pc: u32,
    pub falls_through: bool,
    /// A pool may lie before the unit.
    pub site: bool,
}

/// Where a branch or island goes: to its label, or to the island for that label in a pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Hop {
    Label,
    Island { pool: usize, slot: usize },
}

/// A place for islands, before a unit.
struct Pool {
    unit: usize,
    /// The label each island leads to, and where it goes next.
    islands: Vec<(u32, Hop)>,
}

pub(super) struct Layout {
    units: Vec<Unit>,
    /// Each label and the unit it stands before (`units.len()`: the end of the code).
    labels: BTreeMap<u32, usize>,
    /// The target's `s_branch`.
    branch: u16,
}

/// The code and where everything in it landed.
pub(super) struct Placed {
    pub code: Vec<u32>,
    /// The word each unit starts at.
    pub units: Vec<usize>,
    /// The word each label stands for.
    pub labels: BTreeMap<u32, usize>,
}

impl Layout {
    pub fn new(branch: u16) -> Self {
        Self {
            units: Vec::new(),
            labels: BTreeMap::new(),
            branch,
        }
    }

    /// Binds `label` to the next unit.
    pub fn label(&mut self, label: u32) {
        self.labels.insert(label, self.units.len());
    }

    /// Appends a unit and returns its index.
    pub fn push(&mut self, unit: Unit) -> usize {
        self.units.push(unit);
        self.units.len() - 1
    }

    pub fn place(self) -> Result<Placed, CompileError> {
        let mut pools: Vec<Pool> = Vec::new();
        let mut hops = vec![Hop::Label; self.units.len()];
        if !self.reaches_directly()? {
            pools = self.pools();
            self.route(&mut pools, &mut hops)?;
        }
        let (units, pool_starts) = self.offsets(&pools);
        let end = self.end(&units);
        // A label stands for its unit, after any pool before it: a branch to it lands past the
        // pool, and falling into the pool takes the branch over it.
        let labels: BTreeMap<u32, usize> = self.labels.iter().map(|(&label, &unit)| (label, units.get(unit).copied().unwrap_or(end))).collect();
        let skips: Vec<bool> = pools.iter().map(|pool| self.skips(pool)).collect();
        let branch = self.branch;
        let landing = |label: u32, hop: Hop| match hop {
            Hop::Label => labels[&label],
            Hop::Island { pool, slot } => pool_starts[pool] + usize::from(skips[pool]) + slot,
        };
        // The distance field of a branch at word `at` to word `to`; routing left it within reach.
        let distance = |at: usize, to: usize| {
            let delta = to as i64 - (at as i64 + 1);
            debug_assert!(REACH.contains(&delta), "a branch at word {at} does not reach word {to}");
            delta as i16 as u16
        };
        let mut code = Vec::with_capacity(end);
        let mut next_pool = 0;
        for (index, unit) in self.units.into_iter().enumerate() {
            while next_pool < pools.len() && pools[next_pool].unit == index {
                let pool = &pools[next_pool];
                if !pool.islands.is_empty() {
                    debug_assert_eq!(code.len(), pool_starts[next_pool]);
                    if skips[next_pool] {
                        code.push(super::sopp(branch, pool.islands.len() as u16));
                    }
                    for &(label, hop) in &pool.islands {
                        code.push(super::sopp(branch, distance(code.len(), landing(label, hop))));
                    }
                }
                next_pool += 1;
            }
            debug_assert_eq!(code.len(), units[index]);
            let at = code.len();
            code.extend(unit.words);
            if let Some(label) = unit.target {
                code[at] = (code[at] & 0xffff_0000) | u32::from(distance(at, landing(label, hops[index])));
            }
        }
        Ok(Placed { code, units, labels })
    }

    /// Whether the pool is entered from the unit before it, so it starts with a branch over
    /// its islands.
    fn skips(&self, pool: &Pool) -> bool {
        pool.unit > 0 && self.units[pool.unit - 1].falls_through
    }

    /// The word after the last unit, given where the units start.
    fn end(&self, units: &[usize]) -> usize {
        units.last().map_or(0, |&last| last + self.units.last().map_or(0, |unit| unit.words.len()))
    }

    /// The word each unit and each pool starts at.
    fn offsets(&self, pools: &[Pool]) -> (Vec<usize>, Vec<usize>) {
        let mut units = Vec::with_capacity(self.units.len());
        let mut starts = Vec::with_capacity(pools.len());
        let (mut at, mut next_pool) = (0, 0);
        for (index, unit) in self.units.iter().enumerate() {
            while next_pool < pools.len() && pools[next_pool].unit == index {
                starts.push(at);
                let pool = &pools[next_pool];
                if !pool.islands.is_empty() {
                    at += usize::from(self.skips(pool)) + pool.islands.len();
                }
                next_pool += 1;
            }
            units.push(at);
            at += unit.words.len();
        }
        starts.resize(pools.len(), at);
        (units, starts)
    }

    fn unknown_label(&self, unit: &Unit, label: u32) -> CompileError {
        CompileError::InvalidBranch {
            pc: unit.pc,
            target: label,
        }
    }

    /// Whether every branch reaches its label without islands.
    fn reaches_directly(&self) -> Result<bool, CompileError> {
        let (units, _) = self.offsets(&[]);
        let end = self.end(&units);
        let mut all = true;
        for (index, unit) in self.units.iter().enumerate() {
            let Some(label) = unit.target else { continue };
            let to = *self.labels.get(&label).ok_or_else(|| self.unknown_label(unit, label))?;
            let to = units.get(to).copied().unwrap_or(end);
            all &= REACH.contains(&(to as i64 - (units[index] as i64 + 1)));
        }
        Ok(all)
    }

    /// Pools at most [`POOL_SPACING`] words apart, where the instructions allow them.
    fn pools(&self) -> Vec<Pool> {
        let mut pools = Vec::new();
        // Words since the last pool, and the last site since then with the words before it.
        let mut since = 0;
        let mut site: Option<(usize, usize)> = None;
        for (index, unit) in self.units.iter().enumerate() {
            if unit.site && index > 0 {
                site = Some((index, since));
            }
            since += unit.words.len();
            if since > POOL_SPACING {
                if let Some((at, before)) = site.take() {
                    pools.push(Pool {
                        unit: at,
                        islands: Vec::new(),
                    });
                    since -= before;
                }
            }
        }
        pools
    }

    /// Gives every branch and island a hop within reach, adding islands until none is missing.
    fn route(&self, pools: &mut [Pool], hops: &mut [Hop]) -> Result<(), CompileError> {
        loop {
            let (units, pool_starts) = self.offsets(pools);
            let end = self.end(&units);
            let mut added = false;
            for (index, unit) in self.units.iter().enumerate() {
                let Some(label) = unit.target else { continue };
                let label_unit = *self.labels.get(&label).ok_or_else(|| self.unknown_label(unit, label))?;
                let to = units.get(label_unit).copied().unwrap_or(end);
                // The pools between the branch and its target, nearest to the target first.
                let between = pools.partition_point(|p| p.unit <= index)..pools.partition_point(|p| p.unit < label_unit);
                let backward = pools.partition_point(|p| p.unit <= label_unit)..pools.partition_point(|p| p.unit <= index);
                let hop = self.hop(pools, &pool_starts, units[index], to, label, if label_unit > index { between } else { backward }, label_unit > index, &mut added);
                hops[index] = hop.ok_or(CompileError::BranchOutOfRange { pc: unit.pc })?;
            }
            // Islands, including the ones this pass added.
            let mut pool = 0;
            while pool < pools.len() {
                let mut slot = 0;
                while slot < pools[pool].islands.len() {
                    let label = pools[pool].islands[slot].0;
                    let label_unit = self.labels[&label];
                    let to = units.get(label_unit).copied().unwrap_or(end);
                    let from = pool_starts[pool] + usize::from(self.skips(&pools[pool])) + slot;
                    let forward = label_unit >= pools[pool].unit;
                    let onward = if forward { pool + 1..pools.partition_point(|p| p.unit < label_unit) } else { pools.partition_point(|p| p.unit <= label_unit)..pool };
                    let hop = self.hop(pools, &pool_starts, from, to, label, onward, forward, &mut added);
                    pools[pool].islands[slot].1 = hop.ok_or(CompileError::BranchOutOfRange { pc: self.units[pools[pool].unit].pc })?;
                    slot += 1;
                }
                pool += 1;
            }
            if !added {
                return Ok(());
            }
        }
    }

    /// The hop for a branch at word `from` to `label` at word `to`: the label if it is within
    /// reach, else the island for it in the pool of `candidates` nearest to the label that is.
    /// `candidates` are the pools on the way, in address order.
    #[allow(clippy::too_many_arguments)]
    fn hop(&self, pools: &mut [Pool], pool_starts: &[usize], from: usize, to: usize, label: u32, candidates: std::ops::Range<usize>, forward: bool, added: &mut bool) -> Option<Hop> {
        let reaches = |word: usize| REACH.contains(&(word as i64 - (from as i64 + 1)));
        if reaches(to) {
            return Some(Hop::Label);
        }
        let mut order: Vec<usize> = candidates.collect();
        if forward {
            order.reverse();
        }
        for pool in order {
            let existing = pools[pool].islands.iter().position(|&(l, _)| l == label);
            let slot = existing.unwrap_or(pools[pool].islands.len());
            // An island added to an empty pool brings the branch over the pool with it.
            let skip = usize::from(self.skips(&pools[pool]));
            if !reaches(pool_starts[pool] + skip + slot) {
                continue;
            }
            if existing.is_none() {
                pools[pool].islands.push((label, Hop::Label));
                *added = true;
            }
            return Some(Hop::Island { pool, slot });
        }
        None
    }
}
