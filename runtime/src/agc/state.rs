//! The guest registers as a recording leaves them (D12 point 1), in flat files so that a
//! register write costs a few nanoseconds, with content fingerprints that identify what a draw
//! or dispatch translates: two states with the same fingerprint translate the same.
//!
//! A fingerprint is the XOR of a mixed `(bank, offset, value)` per register it covers, kept up
//! to date on each write; it does not depend on the order of the writes. The graphics
//! fingerprint covers every register but the graphics and compute user data and the compute
//! registers; the compute fingerprint covers the compute registers but their user data. User
//! data passes to the host in place, so no translation reads it.

use super::cache::mix;
use crate::ir::{QueueState, RegisterBank};

/// Shader-bank dwords (`0xb000..0xc000`).
const SHADER_DWORDS: usize = 0x400;
/// Context-bank dwords (`0x28000..0x30000`).
const CONTEXT_DWORDS: usize = 0x2000;

/// Guest shader-bank dwords of the compute registers (COMPUTE_DISPATCH_INITIATOR through
/// COMPUTE_USER_DATA_15) and of their user data.
const COMPUTE: std::ops::Range<u32> = 0x200..0x250;
const COMPUTE_USER_DATA: std::ops::Range<u32> = 0x240..0x250;

/// Which fingerprint a register counts in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Class {
    Graphics,
    Compute,
    /// User data: in no fingerprint.
    Neither,
}

fn class(bank: RegisterBank, offset: u32) -> Class {
    match bank {
        RegisterBank::Shader if COMPUTE_USER_DATA.contains(&offset) => Class::Neither,
        RegisterBank::Shader if COMPUTE.contains(&offset) => Class::Compute,
        RegisterBank::Shader if crate::ir::graphics_user_data(offset) => Class::Neither,
        _ => Class::Graphics,
    }
}

fn bank_bits(bank: RegisterBank) -> u64 {
    match bank {
        RegisterBank::Context => 0,
        RegisterBank::Shader => 1,
        RegisterBank::UserConfig => 2,
    }
}

/// What one register contributes to its fingerprint.
fn contribution(bank: RegisterBank, offset: u32, value: u32) -> u64 {
    mix(bank_bits(bank) << 48 | u64::from(offset & 0xffff) << 32 | u64::from(value))
}

/// A flat bank: values and the set of written dwords.
#[derive(Clone)]
struct Bank<const N: usize> {
    values: Box<[u32]>,
    set: Box<[u64]>,
}

impl<const N: usize> Bank<N> {
    fn new() -> Self {
        Self { values: vec![0; N].into_boxed_slice(), set: vec![0; N / 64].into_boxed_slice() }
    }

    fn get(&self, offset: usize) -> Option<u32> {
        (self.set[offset / 64] >> (offset % 64) & 1 != 0).then(|| self.values[offset])
    }

    /// Stores `value`; returns the value it replaces.
    fn put(&mut self, offset: usize, value: u32) -> Option<u32> {
        let old = self.get(offset);
        self.values[offset] = value;
        self.set[offset / 64] |= 1 << (offset % 64);
        old
    }

    fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.set.iter().enumerate().flat_map(move |(word, &bits)| {
            let mut bits = bits;
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let offset = word * 64 + bit;
                Some((offset as u32, self.values[offset]))
            })
        })
    }
}

/// The guest registers of one queue as a recording sees them.
#[derive(Clone)]
pub struct GuestRegisters {
    shader: Bank<SHADER_DWORDS>,
    context: Bank<CONTEXT_DWORDS>,
    /// Sorted by offset; titles write few user-config registers.
    user_config: Vec<(u32, u32)>,
    /// The graphics fingerprint's context part, kept apart so that a pushed context restores
    /// it, and the rest.
    context_part: u64,
    graphics: u64,
    compute: u64,
    /// The context a push saved (`sceAgcDcbContextStateOp`, CLEAR_STATE).
    saved_context: Option<Box<(Bank<CONTEXT_DWORDS>, u64)>>,
}

impl Default for GuestRegisters {
    /// The registers as a queue starts (`QueueState::default`).
    fn default() -> Self {
        static DEFAULT: std::sync::OnceLock<GuestRegisters> = std::sync::OnceLock::new();
        DEFAULT.get_or_init(|| Self::from_queue_state(&QueueState::default())).clone()
    }
}

impl GuestRegisters {
    pub fn empty() -> Self {
        Self {
            shader: Bank::new(),
            context: Bank::new(),
            user_config: Vec::new(),
            context_part: 0,
            graphics: 0,
            compute: 0,
            saved_context: None,
        }
    }

    pub fn from_queue_state(state: &QueueState) -> Self {
        let mut registers = Self::empty();
        for bank in [RegisterBank::Context, RegisterBank::Shader, RegisterBank::UserConfig] {
            for (offset, value) in state.bank(bank).iter() {
                registers.write(bank, offset, value);
            }
        }
        registers
    }

    /// The registers as a [`QueueState`].
    pub fn to_queue_state(&self) -> QueueState {
        let mut state = QueueState::default();
        state.context = Default::default();
        state.shader = Default::default();
        state.user_config = Default::default();
        for (offset, value) in self.context.iter() {
            state.context.insert(offset, value);
        }
        for (offset, value) in self.shader.iter() {
            state.shader.insert(offset, value);
        }
        for &(offset, value) in &self.user_config {
            state.write_register(RegisterBank::UserConfig, offset, value);
        }
        state
    }

    /// The register at `offset` of `bank`, if written.
    pub fn get(&self, bank: RegisterBank, offset: u32) -> Option<u32> {
        match bank {
            RegisterBank::Shader => ((offset as usize) < SHADER_DWORDS).then(|| self.shader.get(offset as usize))?,
            RegisterBank::Context => ((offset as usize) < CONTEXT_DWORDS).then(|| self.context.get(offset as usize))?,
            RegisterBank::UserConfig => self
                .user_config
                .binary_search_by_key(&offset, |&(at, _)| at)
                .ok()
                .map(|index| self.user_config[index].1),
        }
    }

    pub fn get_or_zero(&self, bank: RegisterBank, offset: u32) -> u32 {
        self.get(bank, offset).unwrap_or(0)
    }

    /// Writes one register; offsets outside the bank are ignored (the console's placeholder
    /// offsets above 0xffff never reach here, see `encode_pairs`).
    pub fn write(&mut self, bank: RegisterBank, offset: u32, value: u32) {
        let old = match bank {
            RegisterBank::Shader if (offset as usize) < SHADER_DWORDS => self.shader.put(offset as usize, value),
            RegisterBank::Context if (offset as usize) < CONTEXT_DWORDS => self.context.put(offset as usize, value),
            RegisterBank::UserConfig => match self.user_config.binary_search_by_key(&offset, |&(at, _)| at) {
                Ok(index) => Some(std::mem::replace(&mut self.user_config[index].1, value)),
                Err(index) => {
                    self.user_config.insert(index, (offset, value));
                    None
                }
            },
            _ => return,
        };
        if old == Some(value) {
            return;
        }
        let delta = old.map_or(0, |old| contribution(bank, offset, old)) ^ contribution(bank, offset, value);
        match class(bank, offset) {
            Class::Graphics if bank == RegisterBank::Context => self.context_part ^= delta,
            Class::Graphics => self.graphics ^= delta,
            Class::Compute => self.compute ^= delta,
            Class::Neither => {}
        }
    }

    /// Applies a guest list of flattened `(offset, value)` pairs as SET_*_REG_INDIRECT does:
    /// every offset is checked before any register changes (`ir::register_offset`).
    pub fn write_list(&mut self, bank: RegisterBank, words: &[u32]) -> crate::Result<()> {
        for pair in words.chunks_exact(2) {
            crate::ir::register_offset(pair[0])?;
        }
        for pair in words.chunks_exact(2) {
            self.write(bank, crate::ir::register_offset(pair[0])?, pair[1]);
        }
        Ok(())
    }

    /// The context registers as CLEAR_STATE leaves them (`QueueState::clear_context`).
    pub fn clear_context(&mut self) {
        self.context = Bank::new();
        self.context_part = 0;
        let mut cleared = QueueState::initial_context_registers();
        cleared.insert(0x1b6, 2);
        for (offset, value) in cleared.iter() {
            self.write(RegisterBank::Context, offset, value);
        }
    }

    /// A context-state operation: 0 clears the context, 1 pushes it, 2 pops it, 3 pushes then
    /// clears (`QueueState::execute_context_state_op`).
    pub fn context_state_op(&mut self, operation: u32) -> crate::Result<()> {
        let push = |registers: &mut Self| {
            if registers.saved_context.is_some() {
                return Err(crate::Error::invalid("context state is already pushed"));
            }
            registers.saved_context = Some(Box::new((registers.context.clone(), registers.context_part)));
            Ok(())
        };
        match operation {
            0 => self.clear_context(),
            1 => push(self)?,
            2 => {
                let saved = self
                    .saved_context
                    .take()
                    .ok_or_else(|| crate::Error::invalid("context state has not been pushed"))?;
                (self.context, self.context_part) = *saved;
            }
            3 => {
                push(self)?;
                self.clear_context();
            }
            other => return Err(crate::Error::invalid(format!("unknown context-state operation {other}"))),
        }
        Ok(())
    }

    /// What draws translate: equal fingerprints, equal translations (never 0, a cache key).
    pub fn graphics_fingerprint(&self) -> u64 {
        (self.context_part ^ self.graphics).max(1)
    }

    /// What dispatches translate (never 0).
    pub fn compute_fingerprint(&self) -> u64 {
        self.compute.max(1)
    }
}
