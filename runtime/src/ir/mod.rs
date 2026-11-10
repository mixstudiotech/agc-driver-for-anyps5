use crate::{amd::defs::gfx10, Error, Result};
use std::collections::BTreeMap;

/// The user-config dword of a guest register at byte `offset`.
const fn user_config_dword(offset: u32) -> u32 {
    (offset - 0x30000) / 4
}

/// The SH dword of a guest register at byte `offset`.
const fn shader_dword(offset: u32) -> u32 {
    (offset - 0xb000) / 4
}

/// VGT_INDEX_TYPE among the guest's user-config registers.
pub const GUEST_VGT_INDEX_TYPE: u32 = user_config_dword(gfx10::VGT_INDEX_TYPE::OFFSET);

pub const SHADER_REGISTER_BASE: u32 = 0x2c00;
pub const CONTEXT_REGISTER_BASE: u32 = 0xa000;
pub const UCONFIG_REGISTER_BASE: u32 = 0xc000;
/// Bytes of the global data share DMA_DATA can address.
pub const GDS_BYTES: usize = 0x10000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum RegisterBank {
    Shader,
    Context,
    UserConfig,
}

impl RegisterBank {
    pub const fn base(self) -> u32 {
        match self {
            Self::Shader => SHADER_REGISTER_BASE,
            Self::Context => CONTEXT_REGISTER_BASE,
            Self::UserConfig => UCONFIG_REGISTER_BASE,
        }
    }

    pub const fn set_opcode(self) -> u32 {
        match self {
            Self::Shader => 0x76,
            Self::Context => 0x69,
            Self::UserConfig => 0x79,
        }
    }

    pub const fn size(self) -> u32 {
        match self {
            Self::Shader | Self::Context => 0x400,
            Self::UserConfig => 0x4000,
        }
    }

}

#[derive(Clone, Debug, Default)]
pub struct RegisterMap {
    values: BTreeMap<u32, u32>,
    /// Changes whenever a value does ([`Self::version`]); not part of the map's identity.
    version: u64,
    /// Changes whenever a value outside the `volatile` registers does ([`Self::stable_version`]).
    stable_version: u64,
    volatile: Option<fn(u32) -> bool>,
}

/// Whether shader-bank register `dword` is a graphics stage's user data
/// (`SPI_SHADER_USER_DATA_{PS,GS,HS}_0` on, 32 each): the registers almost every draw changes.
pub fn graphics_user_data(dword: u32) -> bool {
    [
        shader_dword(gfx10::SPI_SHADER_USER_DATA_PS_0::OFFSET),
        shader_dword(gfx10::SPI_SHADER_USER_DATA_GS_0::OFFSET),
        shader_dword(gfx10::SPI_SHADER_USER_DATA_HS_0::OFFSET),
    ]
    .iter()
    .any(|&first| (first..first + 32).contains(&dword))
}

impl PartialEq for RegisterMap {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl Eq for RegisterMap {}

impl RegisterMap {
    pub fn get(&self, offset: u32) -> Option<u32> {
        self.values.get(&offset).copied()
    }

    pub fn get_or_zero(&self, offset: u32) -> u32 {
        self.get(offset).unwrap_or(0)
    }

    /// A map whose [`Self::stable_version`] ignores changes of the registers `volatile` names.
    pub fn with_volatile(volatile: fn(u32) -> bool) -> Self {
        Self {
            volatile: Some(volatile),
            ..Self::default()
        }
    }

    pub fn insert(&mut self, offset: u32, value: u32) {
        if self.values.insert(offset, value) != Some(value) {
            self.version += 1;
            if !self.volatile.is_some_and(|volatile| volatile(offset)) {
                self.stable_version += 1;
            }
        }
    }

    pub fn remove(&mut self, offset: u32) {
        if self.values.remove(&offset).is_some() {
            self.version += 1;
            self.stable_version += 1;
        }
    }

    pub fn clear(&mut self) {
        if !self.values.is_empty() {
            self.values.clear();
            self.version += 1;
            self.stable_version += 1;
        }
    }

    /// A counter that changes whenever a register outside the volatile ones changes
    /// ([`Self::with_volatile`]): equal stable versions mean those registers are unchanged.
    pub fn stable_version(&self) -> u64 {
        self.stable_version
    }

    /// A counter that changes whenever a register's value does: equal versions of the same map
    /// mean equal contents.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The registers in `range` with their values, in offset order.
    pub fn range(&self, range: std::ops::Range<u32>) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.values.range(range).map(|(&k, &v)| (k, v))
    }

    pub fn contains(&self, offset: u32) -> bool {
        self.values.contains_key(&offset)
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.values.iter().map(|(&k, &v)| (k, v))
    }

    pub fn fingerprint(&self) -> u64 {
        let mut result = 0u64;
        for (&offset, &value) in &self.values {
            let mut hash =
                ((u64::from(offset) << 32) | u64::from(value)).wrapping_add(0x9e37_79b9_7f4a_7c15);
            hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            result ^= hash ^ (hash >> 31);
        }
        result
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PredicationState {
    pub address: u64,
    pub operation: u32,
    pub execute_when_set: bool,
}

impl PredicationState {
    /// Bytes the predicate reads (operation 3 compares a 64-bit value).
    pub const fn bytes(&self) -> usize {
        if self.operation == 3 {
            8
        } else {
            4
        }
    }

    pub const fn passes(&self, value: u64) -> bool {
        self.operation == 0 || (value != 0) == self.execute_when_set
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueueState {
    pub shader: RegisterMap,
    pub context: RegisterMap,
    pub user_config: RegisterMap,
    pub saved_context: Option<RegisterMap>,
    pub constant_ram: Vec<u32>,
    pub index_base: u64,
    pub draw_indirect_base: u64,
    pub dispatch_indirect_base: u64,
    pub index_buffer_size: Option<u32>,
    pub index_type: u32,
    pub instance_count: u32,
    pub markers: Vec<String>,
    pub predication: PredicationState,
}

impl Default for QueueState {
    fn default() -> Self {
        let mut user_config = RegisterMap::default();
        user_config.insert(user_config_dword(gfx10::GE_INDX_OFFSET::OFFSET), 0);
        user_config.insert(user_config_dword(gfx10::GE_MULTI_PRIM_IB_RESET_EN::OFFSET), 0);
        Self {
            shader: RegisterMap::with_volatile(graphics_user_data),
            context: Self::initial_context_registers(),
            user_config,
            saved_context: None,
            constant_ram: Vec::new(),
            index_base: 0,
            draw_indirect_base: 0,
            dispatch_indirect_base: 0,
            index_buffer_size: None,
            index_type: 0,
            instance_count: 1,
            markers: Vec::new(),
            predication: PredicationState::default(),
        }
    }
}

impl QueueState {
    pub fn initial_context_registers() -> RegisterMap {
        let mut reg = RegisterMap::default();
        for &(offset, value) in &[
            (0x200, 0),
            (0x201, 0),
            (0x202, 0x00cc_0010),
            (0x203, 0),
            (0x204, 0),
            (0x205, 0),
            (0x206, 1087),
            (0x207, 0),
            (0x000, 0),
            (0x002, 0),
            (0x003, 0),
            (0x004, 0),
            (0x008, 0),
            (0x009, 0x3f80_0000),
            (0x00a, 0),
            (0x00b, 0),
            (0x010, 0),
            (0x011, 0),
            (0x080, 0),
            (0x083, 0xffff),
            (0x08c, 0xaa99_aaaa),
            (0x08d, 0),
            (0x08e, 0),
            (0x08f, 0),
            (0x00c, 0),
            (0x00d, 0x4000_4000),
            (0x081, 0x8000_0000),
            (0x082, 0x4000_4000),
            (0x090, 0x8000_0000),
            (0x091, 0x4000_4000),
            (0x105, 0),
            (0x106, 0),
            (0x107, 0),
            (0x108, 0),
            (0x10b, 0),
            (0x10c, 0),
            (0x10d, 0),
            (0x1b1, 0),
            (0x1b3, 0),
            (0x1b4, 0),
            (0x1b6, 0),
            (0x1c3, 0),
            (0x1c4, 0),
            (0x1c5, 0),
            (0x1ff, 0),
            (0x280, 0x0008_0008),
            (0x281, 0xffff_0000),
            (0x292, 2),
            (0x293, 0),
            (0x29b, 0),
            (0x2ce, 0),
            (0x2d3, 0),
            (0x2d5, 0),
            (0x2d6, 0),
            (0x2db, 0),
            (0x2dc, 0xaa00),
            (0x2df, 0),
            (0x2e4, 0),
            (0x2f8, 0),
            (0x2f9, 0x2d),
            (0x30e, 0xffff_ffff),
            (0x30f, 0xffff_ffff),
            (0x313, 0x6000),
            (0x318, 0),
            (0x31b, 0),
            (0x31c, 0),
            (0x31d, 0),
            (0x390, 0),
            (0x3b0, 0),
            (0x3b8, 0),
        ] {
            reg.insert(offset, value);
        }
        for i in 0..32 {
            reg.insert(0x191 + i, 0);
        }
        for i in 0..8 {
            reg.insert(0x1e0 + i, 0x2001_0001);
            reg.insert(0x31c + 0x0f * i, 0);
        }
        for i in 0..16 {
            reg.insert(0x2fe + i, 0);
            reg.insert(0x094 + 2 * i, 0x8000_0000);
            reg.insert(0x095 + 2 * i, 0x4000_4000);
            reg.insert(0x0b4 + 2 * i, 0);
            reg.insert(0x0b5 + 2 * i, 0);
            for j in 0..6 {
                reg.insert(0x10f + 6 * i + j, if j % 2 == 0 { 0x3f80_0000 } else { 0 });
            }
        }
        reg
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn clear_context(&mut self) {
        let mut cleared = Self::initial_context_registers();
        cleared.insert(0x1b6, 2);
        self.context = cleared;
    }

    pub fn push_context(&mut self) -> Result<()> {
        if self.saved_context.is_some() {
            return Err(Error::invalid("context state is already pushed"));
        }
        self.saved_context = Some(self.context.clone());
        Ok(())
    }

    pub fn pop_context(&mut self) -> Result<()> {
        let saved = self
            .saved_context
            .take()
            .ok_or_else(|| Error::invalid("context state has not been pushed"))?;
        self.context = saved;
        Ok(())
    }

    pub fn execute_context_state_op(&mut self, op: u32) -> Result<()> {
        match op {
            0 => {
                self.clear_context();
                Ok(())
            }
            1 => self.push_context(),
            2 => self.pop_context(),
            3 => {
                self.push_context()?;
                self.clear_context();
                Ok(())
            }
            other => Err(Error::invalid(format!(
                "unknown context-state operation {other}"
            ))),
        }
    }

    pub fn bank_mut(&mut self, bank: RegisterBank) -> &mut RegisterMap {
        match bank {
            RegisterBank::Shader => &mut self.shader,
            RegisterBank::Context => &mut self.context,
            RegisterBank::UserConfig => &mut self.user_config,
        }
    }

    pub fn bank(&self, bank: RegisterBank) -> &RegisterMap {
        match bank {
            RegisterBank::Shader => &self.shader,
            RegisterBank::Context => &self.context,
            RegisterBank::UserConfig => &self.user_config,
        }
    }

    pub fn write_register(&mut self, bank: RegisterBank, offset: u32, value: u32) {
        self.bank_mut(bank).insert(offset, value);
        if bank == RegisterBank::UserConfig && offset == GUEST_VGT_INDEX_TYPE {
            self.index_type = value & 3;
        }
    }

    pub fn index_size_bytes(&self) -> Result<u32> {
        index_size_bytes(self.index_type)
    }

}

pub fn index_size_bytes(index_type: u32) -> Result<u32> {
    match index_type {
        0 => Ok(2),
        1 => Ok(4),
        2 => Ok(1),
        other => Err(Error::unsupported(format!(
            "unsupported index type {other}"
        ))),
    }
}

/// Register offset dword of SET_*_REG and indirect register lists: bits 28..30 select a
/// register index and are ignored; offsets above 0xffff are extended semantics.
pub fn register_offset(raw: u32) -> Result<u32> {
    if raw == u32::MAX {
        return Err(Error::unsupported(
            "indirect register sentinel semantics are not implemented",
        ));
    }
    let offset = raw & !0x7000_0000;
    if offset > 0xffff {
        return Err(Error::unsupported(format!(
            "extended register semantics are not implemented (offset dword 0x{raw:08x})"
        )));
    }
    Ok(offset)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlipRequest {
    pub handle: u32,
    pub index: i32,
    pub mode: u32,
    pub argument: i64,
}

