pub mod isa_gen;
pub mod memops;
pub mod opcodes;

#[cfg(test)]
mod isa_check;

use crate::{ShaderStage, TargetIsa};
use opcodes::flag;

pub const SGPR_LIMIT: u16 = 106;
pub const VGPR_LIMIT: u16 = 256;
pub const SCALAR_VCC_LO: u16 = 106;
pub const SCALAR_VCC_HI: u16 = 107;
pub const SCALAR_TTMP_BASE: u16 = 108;
pub const SCALAR_M0: u16 = 124;
pub const SCALAR_EXEC_LO: u16 = 126;
pub const SCALAR_EXEC_HI: u16 = 127;
pub const MAX_NSA_ADDRESSES: usize = 13;

pub const SDWA_SEL_WORD_1: u8 = 5;
pub const SDWA_SEL_DWORD: u8 = 6;
pub const SDWA_UNUSED_PAD: u8 = 0;
pub const SDWA_UNUSED_SEXT: u8 = 1;
pub const SDWA_UNUSED_PRESERVE: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum InstructionFamily {
    Sop1,
    Sop2,
    Sopk,
    Sopc,
    Sopp,
    Vop1,
    Vop2,
    Vop3,
    Vop3p,
    Vopc,
    Vintrp,
    Smem,
    Mubuf,
    Mtbuf,
    Flat,
    Ds,
    Mimg,
    Exp,
    Ldsdir,
    Vinterp,
}

impl InstructionFamily {
    pub fn name(self) -> &'static str {
        match self {
            Self::Sop1 => "SOP1",
            Self::Sop2 => "SOP2",
            Self::Sopk => "SOPK",
            Self::Sopc => "SOPC",
            Self::Sopp => "SOPP",
            Self::Vop1 => "VOP1",
            Self::Vop2 => "VOP2",
            Self::Vop3 => "VOP3",
            Self::Vop3p => "VOP3P",
            Self::Vopc => "VOPC",
            Self::Vintrp => "VINTRP",
            Self::Smem => "SMEM",
            Self::Mubuf => "MUBUF",
            Self::Mtbuf => "MTBUF",
            Self::Flat => "FLAT",
            Self::Ds => "DS",
            Self::Mimg => "MIMG",
            Self::Exp => "EXP",
            Self::Ldsdir => "LDSDIR",
            Self::Vinterp => "VINTERP",
        }
    }

    pub fn is_scalar_alu(self) -> bool {
        matches!(
            self,
            Self::Sop1 | Self::Sop2 | Self::Sopk | Self::Sopc | Self::Sopp
        )
    }

    pub fn is_vector_alu(self) -> bool {
        matches!(
            self,
            Self::Vop1
                | Self::Vop2
                | Self::Vop3
                | Self::Vop3p
                | Self::Vopc
                | Self::Vintrp
                | Self::Vinterp
        )
    }

    pub fn is_vector_memory(self) -> bool {
        matches!(self, Self::Mubuf | Self::Mtbuf | Self::Flat | Self::Mimg)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Operand {
    Sgpr(u8),
    Vgpr(u8),
    VccLo,
    VccHi,
    Ttmp(u8),
    M0,
    Null,
    ExecLo,
    ExecHi,
    InlineInt(i8),
    InlineFloat(u8),
    Literal(u32),
    SharedBase,
    SharedLimit,
    PrivateBase,
    PrivateLimit,
    PopsExitingWaveId,
    VccZ,
    ExecZ,
    Scc,
    LdsDirect,
    Raw(u16),
}

const INLINE_INT_ZERO: u16 = 128;
const INLINE_INT_MAX: u16 = 192;
const INLINE_INT_NEG_MAX: u16 = 208;
const INLINE_FLOAT_FIRST: u16 = 240;
const INLINE_FLOAT_LAST: u16 = 248;
const LITERAL_CODE: u16 = 255;
const VGPR_BASE: u16 = 256;
const GFX10_M0: u16 = 124;
const GFX10_NULL: u16 = 125;
const GFX11_NULL: u16 = 124;
const GFX11_M0: u16 = 125;

impl Operand {
    pub fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            0..=105 => Self::Sgpr(code as u8),
            106 => Self::VccLo,
            107 => Self::VccHi,
            108..=123 => Self::Ttmp((code - SCALAR_TTMP_BASE) as u8),
            GFX10_M0 => Self::M0,
            GFX10_NULL => Self::Null,
            126 => Self::ExecLo,
            127 => Self::ExecHi,
            INLINE_INT_ZERO..=INLINE_INT_MAX => Self::InlineInt((code - INLINE_INT_ZERO) as i8),
            193..=INLINE_INT_NEG_MAX => Self::InlineInt(-((code - INLINE_INT_MAX) as i8)),
            235 => Self::SharedBase,
            236 => Self::SharedLimit,
            237 => Self::PrivateBase,
            238 => Self::PrivateLimit,
            239 => Self::PopsExitingWaveId,
            INLINE_FLOAT_FIRST..=INLINE_FLOAT_LAST => Self::InlineFloat(code as u8),
            251 => Self::VccZ,
            252 => Self::ExecZ,
            253 => Self::Scc,
            254 => Self::LdsDirect,
            LITERAL_CODE => Self::Literal(0),
            VGPR_BASE..=511 => Self::Vgpr((code - VGPR_BASE) as u8),
            _ => return None,
        })
    }

    pub fn scalar_destination(code: u16) -> Option<Self> {
        match Self::from_code(code)? {
            op @ (Self::Sgpr(_)
            | Self::VccLo
            | Self::VccHi
            | Self::Ttmp(_)
            | Self::M0
            | Self::Null
            | Self::ExecLo
            | Self::ExecHi) => Some(op),
            _ => None,
        }
    }

    pub fn code(self, target: TargetIsa) -> Option<u16> {
        let gfx11 = target.is_gfx11_plus();
        Some(match self {
            Self::Sgpr(n) if u16::from(n) < SGPR_LIMIT => u16::from(n),
            Self::Sgpr(_) => return None,
            Self::Vgpr(n) => VGPR_BASE + u16::from(n),
            Self::VccLo => SCALAR_VCC_LO,
            Self::VccHi => SCALAR_VCC_HI,
            Self::Ttmp(n) if n < 16 => SCALAR_TTMP_BASE + u16::from(n),
            Self::Ttmp(_) => return None,
            Self::M0 if gfx11 => GFX11_M0,
            Self::M0 => GFX10_M0,
            Self::Null if gfx11 => GFX11_NULL,
            Self::Null => GFX10_NULL,
            Self::ExecLo => SCALAR_EXEC_LO,
            Self::ExecHi => SCALAR_EXEC_HI,
            Self::InlineInt(v @ 0..=64) => INLINE_INT_ZERO + v as u16,
            Self::InlineInt(v @ -16..=-1) => INLINE_INT_MAX + u16::from(v.unsigned_abs()),
            Self::InlineInt(_) => return None,
            Self::InlineFloat(c)
                if (INLINE_FLOAT_FIRST..=INLINE_FLOAT_LAST).contains(&u16::from(c)) =>
            {
                u16::from(c)
            }
            Self::InlineFloat(_) => return None,
            Self::Literal(_) => LITERAL_CODE,
            Self::SharedBase => 235,
            Self::SharedLimit => 236,
            Self::PrivateBase => 237,
            Self::PrivateLimit => 238,
            Self::PopsExitingWaveId if gfx11 => return None,
            Self::PopsExitingWaveId => 239,
            Self::VccZ => 251,
            Self::ExecZ => 252,
            Self::Scc => 253,
            Self::LdsDirect if gfx11 => return None,
            Self::LdsDirect => 254,
            Self::Raw(v) => v,
        })
    }

    pub fn from_code_or_raw(code: u16) -> Self {
        Self::from_code(code).unwrap_or(Self::Raw(code))
    }

    pub fn vgpr(self) -> Option<u8> {
        match self {
            Self::Vgpr(n) => Some(n),
            _ => None,
        }
    }

    pub fn is_vgpr(self) -> bool {
        matches!(self, Self::Vgpr(_))
    }

    pub fn literal(self) -> Option<u32> {
        match self {
            Self::Literal(v) => Some(v),
            _ => None,
        }
    }

    pub fn scalar_index(self) -> Option<u16> {
        Some(match self {
            Self::Sgpr(n) => u16::from(n),
            Self::VccLo => SCALAR_VCC_LO,
            Self::VccHi => SCALAR_VCC_HI,
            Self::Ttmp(n) => SCALAR_TTMP_BASE + u16::from(n),
            Self::M0 => SCALAR_M0,
            Self::ExecLo => SCALAR_EXEC_LO,
            Self::ExecHi => SCALAR_EXEC_HI,
            _ => return None,
        })
    }

    pub fn inline_float_bits(code: u8) -> Option<u32> {
        Some(match code {
            240 => 0.5f32.to_bits(),
            241 => (-0.5f32).to_bits(),
            242 => 1.0f32.to_bits(),
            243 => (-1.0f32).to_bits(),
            244 => 2.0f32.to_bits(),
            245 => (-2.0f32).to_bits(),
            246 => 4.0f32.to_bits(),
            247 => (-4.0f32).to_bits(),
            248 => 0x3e22_f983,
            _ => return None,
        })
    }

    /// `value` for a 32-bit operand: the inline constant that reads as it when there is one
    /// ([`Self::inline_32`]), else a literal.
    pub fn constant(value: u32) -> Self {
        Self::inline_32(value).unwrap_or(Self::Literal(value))
    }

    /// The inline constant that a 32-bit operand reads as `value`: an integer in -16..=64, or
    /// the bits of 0.5, 1.0, 2.0, 4.0 (either sign) or 1/(2π) (LLVM's `isInlinableLiteral32`).
    /// 16- and 64-bit operands read inline constants differently.
    pub fn inline_32(value: u32) -> Option<Self> {
        let signed = value as i32;
        if (-16..=64).contains(&signed) {
            return Some(Self::InlineInt(signed as i8));
        }
        (240..=248)
            .find(|&code| Self::inline_float_bits(code) == Some(value))
            .map(Self::InlineFloat)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Sdwa {
    pub dst_sel: u8,
    pub dst_unused: u8,
    pub clamp: bool,
    pub omod: u8,
    pub src0_sel: u8,
    pub src0_sext: bool,
    pub src0_neg: bool,
    pub src0_abs: bool,
    pub src1_sel: u8,
    pub src1_sext: bool,
    pub src1_neg: bool,
    pub src1_abs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Dpp16 {
    pub ctrl: u16,
    pub fetch_inactive: bool,
    pub bound_ctrl: bool,
    pub src0_neg: bool,
    pub src0_abs: bool,
    pub src1_neg: bool,
    pub src1_abs: bool,
    pub bank_mask: u8,
    pub row_mask: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VopExt {
    None,
    Sdwa(Sdwa),
    Dpp16(Dpp16),
    Dpp8 { selects: u32, fetch_inactive: bool },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Vop3 {
    pub vdst: Operand,
    pub sdst: Option<Operand>,
    pub src: [Operand; 3],
    pub abs: u8,
    pub neg: u8,
    pub opsel: u8,
    pub clamp: bool,
    pub omod: u8,
}

impl Vop3 {
    pub fn new(vdst: Operand, src: [Operand; 3]) -> Self {
        Self {
            vdst,
            sdst: None,
            src,
            abs: 0,
            neg: 0,
            opsel: 0,
            clamp: false,
            omod: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Vop3p {
    pub vdst: Operand,
    pub src: [Operand; 3],
    pub neg: u8,
    pub neg_hi: u8,
    pub opsel: u8,
    pub opsel_hi: u8,
    pub clamp: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MemKind {
    Load,
    Store,
    Atomic,
    Control,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Smem {
    pub sdata: Operand,
    pub sbase: u8,
    pub offset: i32,
    pub soffset: Operand,
    pub glc: bool,
    pub dlc: bool,
    pub dwords: u8,
    pub buffer: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Buffer {
    pub vdata: u8,
    pub vaddr: u8,
    pub srsrc: u8,
    pub soffset: Operand,
    pub offset: u16,
    pub offen: bool,
    pub idxen: bool,
    pub glc: bool,
    pub dlc: bool,
    pub slc: bool,
    pub tfe: bool,
    pub lds: bool,
    pub format: u8,
    pub kind: MemKind,
    pub data_dwords: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FlatSegment {
    Flat,
    Scratch,
    Global,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Flat {
    pub segment: FlatSegment,
    pub vdst: u8,
    pub vdata: u8,
    pub vaddr: u8,
    pub saddr: Option<Operand>,
    pub uses_vaddr: bool,
    pub offset: i32,
    pub glc: bool,
    pub dlc: bool,
    pub slc: bool,
    pub lds: bool,
    pub kind: MemKind,
    pub data_dwords: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Ds {
    pub vdst: u8,
    pub addr: u8,
    pub data0: u8,
    pub data1: u8,
    pub offset0: u8,
    pub offset1: u8,
    pub gds: bool,
    pub dst_dwords: u8,
    pub data0_dwords: u8,
    pub data1_dwords: u8,
    pub uses_addr: bool,
    pub writes_addr: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Image {
    pub dim: u8,
    pub dmask: u8,
    pub unorm: bool,
    pub glc: bool,
    pub slc: bool,
    pub dlc: bool,
    pub r128: bool,
    pub tfe: bool,
    pub lwe: bool,
    pub a16: bool,
    pub d16: bool,
    pub vdata: u8,
    pub addr: [u8; MAX_NSA_ADDRESSES],
    pub nsa: bool,
    pub nsa_words: u8,
    pub addr_dwords: u8,
    /// Pro BVH8 stores seven NSA address bytes for eleven logical input dwords.
    pub bvh8: bool,
    pub srsrc: u8,
    pub ssamp: u8,
    pub uses_sampler: bool,
    pub kind: MemKind,
    pub data_dwords: u8,
}

impl Image {
    pub fn address_register(&self, index: usize) -> u8 {
        if self.bvh8 {
            let (group, component) = match index {
                0..=1 => (0, index),
                2..=3 => (2, index - 2),
                4..=6 => (4, index - 4),
                7..=9 => (5, index - 7),
                10 => (6, 0),
                _ => unreachable!("BVH8 has eleven input dwords"),
            };
            self.addr[group] + component as u8
        } else if self.nsa {
            self.addr[index]
        } else {
            self.addr[0].wrapping_add(index as u8)
        }
    }
}

/// The gfx10 null export target (gfx11 has none).
pub const EXP_TARGET_NULL: u8 = 9;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Export {
    pub target: u8,
    pub en: u8,
    pub compr: bool,
    pub done: bool,
    pub vm: bool,
    pub row_en: bool,
    pub vsrc: [u8; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Interp {
    pub vdst: u8,
    pub vsrc: u8,
    pub attr: u8,
    pub chan: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LdsDir {
    pub vdst: u8,
    pub attr: u8,
    pub chan: u8,
    pub wait_vdst: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VInterp {
    pub vdst: u8,
    pub src: [Operand; 3],
    pub neg: u8,
    pub opsel: u8,
    pub clamp: bool,
    pub wait_exp: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Fields {
    Sop1 {
        sdst: Operand,
        src0: Operand,
    },
    Sop2 {
        sdst: Operand,
        src0: Operand,
        src1: Operand,
    },
    Sopk {
        sdst: Operand,
        simm16: u16,
        imm32: Option<u32>,
    },
    Sopc {
        src0: Operand,
        src1: Operand,
    },
    Sopp {
        simm16: u16,
    },
    Vop1 {
        vdst: Operand,
        src0: Operand,
        ext: VopExt,
    },
    Vop2 {
        vdst: Operand,
        src0: Operand,
        vsrc1: Operand,
        constant: Option<u32>,
        ext: VopExt,
    },
    Vopc {
        sdst: Operand,
        src0: Operand,
        vsrc1: Operand,
        ext: VopExt,
    },
    Vop3(Vop3),
    Vop3p(Vop3p),
    Vintrp(Interp),
    Smem(Smem),
    Mubuf(Buffer),
    Mtbuf(Buffer),
    Flat(Flat),
    Ds(Ds),
    Mimg(Image),
    Exp(Export),
    Ldsdir(LdsDir),
    Vinterp(VInterp),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Instruction {
    pub pc: u32,
    pub family: InstructionFamily,
    pub opcode: u16,
    pub flags: u64,
    pub fields: Fields,
    pub branch_target: Option<u32>,
    pub words: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegFile {
    Scalar,
    Vector,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegRange {
    pub file: RegFile,
    pub first: u16,
    pub count: u16,
}

impl RegRange {
    pub fn vector(first: u8, count: u8) -> Self {
        Self {
            file: RegFile::Vector,
            first: u16::from(first),
            count: u16::from(count),
        }
    }

    pub fn last(self) -> u16 {
        self.first + self.count.max(1) - 1
    }

    pub fn contains(self, file: RegFile, index: u16) -> bool {
        self.file == file && index >= self.first && index < self.first + self.count
    }
}

fn scalar(op: Operand, dwords: u8) -> Option<RegRange> {
    op.scalar_index().map(|first| RegRange {
        file: RegFile::Scalar,
        first,
        count: u16::from(dwords.max(1)),
    })
}

fn register(op: Operand, dwords: u8) -> Option<RegRange> {
    match op {
        Operand::Vgpr(n) => Some(RegRange::vector(n, dwords.max(1))),
        _ => scalar(op, dwords),
    }
}

fn scalar_code_range(first: u16, count: u16) -> RegRange {
    RegRange {
        file: RegFile::Scalar,
        first,
        count,
    }
}

impl Instruction {
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    pub fn has_flag(&self, mask: u64) -> bool {
        self.flags & mask != 0
    }

    pub fn is_branch(&self) -> bool {
        self.has_flag(flag::BRANCH | flag::COND_BRANCH)
    }

    pub fn ends_block(&self) -> bool {
        self.has_flag(flag::BRANCH | flag::COND_BRANCH | flag::END | flag::JUMP)
    }

    pub fn falls_through(&self) -> bool {
        !self.has_flag(flag::BRANCH | flag::END | flag::JUMP)
    }

    pub fn mnemonic(&self) -> String {
        opcodes::mnemonic(self.family, self.opcode)
    }

    fn dst_dwords(&self) -> u8 {
        if self.has_flag(flag::D2) {
            2
        } else {
            1
        }
    }

    fn src_dwords(&self, index: usize) -> u8 {
        let wide = [flag::S0_2, flag::S1_2, flag::S2_2];
        if self.has_flag(wide[index.min(2)]) {
            2
        } else {
            1
        }
    }

    pub fn defs(&self, wave64: bool) -> Vec<RegRange> {
        let mut out = Vec::new();
        self.each_def(wave64, |range| out.push(range));
        out
    }

    /// Calls `f` with each of [`Self::defs`], in order, without collecting them.
    pub fn each_def(&self, wave64: bool, mut f: impl FnMut(RegRange)) {
        let mask = if wave64 { 2 } else { 1 };
        let mut push = |range: Option<RegRange>| {
            if let Some(range) = range {
                f(range);
            }
        };
        match &self.fields {
            Fields::Sop1 { sdst, .. } | Fields::Sop2 { sdst, .. } => {
                if !self.has_flag(flag::NO_DST) {
                    push(scalar(*sdst, self.dst_dwords()));
                }
                if self.has_flag(flag::WRITES_EXEC) {
                    push(Some(scalar_code_range(SCALAR_EXEC_LO, u16::from(mask))));
                }
            }
            Fields::Sopk { sdst, .. } => {
                if !self.has_flag(flag::NO_DST) {
                    push(scalar(*sdst, self.dst_dwords()));
                }
            }
            Fields::Vop1 { vdst, src0, .. } => {
                if !self.has_flag(flag::NO_OPERANDS) {
                    push(register(*vdst, self.dst_dwords()));
                }
                if self.has_flag(flag::SWAP) {
                    push(register(*src0, 1));
                }
            }
            Fields::Vop2 { vdst, .. } => {
                push(register(*vdst, self.dst_dwords()));
                if self.has_flag(flag::CARRY_OUT) {
                    push(Some(scalar_code_range(SCALAR_VCC_LO, u16::from(mask))));
                }
            }
            Fields::Vopc { sdst, .. } => push(scalar(*sdst, mask)),
            Fields::Vop3(v) => {
                let dwords = if v.vdst.is_vgpr() {
                    self.dst_dwords()
                } else if self.has_flag(flag::SDST) {
                    1
                } else {
                    mask
                };
                push(register(v.vdst, dwords));
                if let Some(sdst) = v.sdst {
                    push(scalar(sdst, mask));
                }
            }
            Fields::Vop3p(v) => push(register(v.vdst, self.dst_dwords())),
            Fields::Vintrp(i) => push(Some(RegRange::vector(i.vdst, 1))),
            Fields::Smem(s) if s.dwords > 0 => push(scalar(s.sdata, s.dwords)),
            Fields::Mubuf(b) | Fields::Mtbuf(b) => {
                if b.kind == MemKind::Load || (b.kind == MemKind::Atomic && b.glc) {
                    let ret = if b.kind == MemKind::Atomic && self.has_flag(flag::CMPSWAP) {
                        b.data_dwords / 2
                    } else {
                        b.data_dwords
                    };
                    push(Some(RegRange::vector(b.vdata, ret + u8::from(b.tfe))));
                }
            }
            Fields::Flat(f) => {
                if f.kind == MemKind::Load || (f.kind == MemKind::Atomic && f.glc) {
                    let ret = if f.kind == MemKind::Atomic && self.has_flag(flag::CMPSWAP) {
                        f.data_dwords / 2
                    } else {
                        f.data_dwords
                    };
                    push(Some(RegRange::vector(f.vdst, ret)));
                }
            }
            Fields::Ds(d) => {
                if d.dst_dwords > 0 { push(Some(RegRange::vector(d.vdst, d.dst_dwords))); }
                if d.writes_addr { push(Some(RegRange::vector(d.addr, 1))); }
            }
            Fields::Mimg(m) if m.kind != MemKind::Store && m.data_dwords > 0 => {
                if m.bvh8 {
                    push(Some(RegRange::vector(m.address_register(4), 3)));
                    push(Some(RegRange::vector(m.address_register(7), 3)));
                }
                let ret = if m.kind == MemKind::Atomic && self.has_flag(flag::CMPSWAP) {
                    m.data_dwords / 2
                } else {
                    m.data_dwords
                };
                if m.kind != MemKind::Atomic || m.glc {
                    push(Some(RegRange::vector(
                        m.vdata,
                        ret + u8::from(m.tfe || m.lwe),
                    )));
                }
            }
            Fields::Ldsdir(l) => push(Some(RegRange::vector(l.vdst, 1))),
            Fields::Vinterp(v) => push(Some(RegRange::vector(v.vdst, 1))),
            _ => {}
        }
        if self.has_flag(flag::WRITES_VCC) {
            push(Some(scalar_code_range(SCALAR_VCC_LO, u16::from(mask))));
        }
    }

    pub fn uses(&self, wave64: bool) -> Vec<RegRange> {
        let mut out = Vec::new();
        self.each_use(wave64, |range| out.push(range));
        out
    }

    /// Calls `f` with each of [`Self::uses`], in order, without collecting them.
    pub fn each_use(&self, wave64: bool, mut f: impl FnMut(RegRange)) {
        let mask = if wave64 { 2 } else { 1 };
        let mut push = |range: Option<RegRange>| {
            if let Some(range) = range {
                f(range);
            }
        };
        match &self.fields {
            Fields::Sop1 { src0, sdst } => {
                if !self.has_flag(flag::NO_SRC) {
                    push(scalar(*src0, self.src_dwords(0)));
                }
                if self.has_flag(flag::READS_DST) {
                    push(scalar(*sdst, self.dst_dwords()));
                }
                if self.has_flag(flag::WRITES_EXEC) {
                    push(Some(scalar_code_range(SCALAR_EXEC_LO, u16::from(mask))));
                }
            }
            Fields::Sop2 { src0, src1, .. } => {
                push(scalar(*src0, self.src_dwords(0)));
                push(scalar(*src1, self.src_dwords(1)));
            }
            Fields::Sopk { sdst, .. } => {
                if self.has_flag(flag::READS_DST) {
                    push(scalar(*sdst, 1));
                }
            }
            Fields::Sopc { src0, src1 } => {
                push(scalar(*src0, self.src_dwords(0)));
                push(scalar(*src1, self.src_dwords(1)));
            }
            Fields::Vop1 { src0, vdst, .. } => {
                if !self.has_flag(flag::NO_OPERANDS) {
                    push(register(*src0, self.src_dwords(0)));
                }
                if self.has_flag(flag::SWAP) {
                    push(register(*vdst, 1));
                }
            }
            Fields::Vop2 {
                vdst, src0, vsrc1, ..
            } => {
                push(register(*src0, self.src_dwords(0)));
                push(register(*vsrc1, self.src_dwords(1)));
                if self.has_flag(flag::READS_DST) {
                    push(register(*vdst, self.dst_dwords()));
                }
                if self.has_flag(flag::CARRY_IN) {
                    push(Some(scalar_code_range(SCALAR_VCC_LO, u16::from(mask))));
                }
            }
            Fields::Vopc { src0, vsrc1, .. } => {
                push(register(*src0, self.src_dwords(0)));
                push(register(*vsrc1, self.src_dwords(1)));
            }
            Fields::Vop3(v) => {
                let count = opcodes::source_count(self);
                for (i, src) in v.src.iter().enumerate().take(count) {
                    let dwords = if !src.is_vgpr()
                        && i == 2
                        && self.has_flag(flag::CARRY_IN | flag::MASK_SRC2)
                    {
                        mask
                    } else {
                        self.src_dwords(i)
                    };
                    push(register(*src, dwords));
                }
                if self.has_flag(flag::READS_DST) {
                    push(register(v.vdst, self.dst_dwords()));
                }
            }
            Fields::Vop3p(v) => {
                for (i, src) in v.src.iter().enumerate().take(opcodes::source_count(self)) {
                    push(register(*src, self.src_dwords(i)));
                }
                if self.has_flag(flag::READS_DST) {
                    push(register(v.vdst, 1));
                }
            }
            Fields::Vintrp(i) => {
                if self.opcode != opcodes::VINTRP_MOV {
                    push(Some(RegRange::vector(i.vsrc, 1)));
                }
                if self.opcode == opcodes::VINTRP_P2 {
                    push(Some(RegRange::vector(i.vdst, 1)));
                }
                push(Some(scalar_code_range(SCALAR_M0, 1)));
            }
            Fields::Smem(s) => {
                let base = if s.buffer { 4 } else { 2 };
                if !self.has_flag(flag::NO_SRC) && (s.dwords > 0 || self.has_flag(flag::SMEM_ADDR))
                {
                    push(Some(scalar_code_range(u16::from(s.sbase), base)));
                }
                push(scalar(s.soffset, 1));
            }
            Fields::Mubuf(b) | Fields::Mtbuf(b) => {
                if b.kind != MemKind::Control {
                    if b.offen || b.idxen {
                        push(Some(RegRange::vector(
                            b.vaddr,
                            u8::from(b.offen) + u8::from(b.idxen),
                        )));
                    }
                    push(Some(scalar_code_range(u16::from(b.srsrc), 4)));
                    push(scalar(b.soffset, 1));
                    if b.kind != MemKind::Load {
                        push(Some(RegRange::vector(b.vdata, b.data_dwords)));
                    }
                }
            }
            Fields::Flat(f) => {
                if f.uses_vaddr {
                    let wide = f.segment != FlatSegment::Scratch
                        && matches!(f.saddr, None | Some(Operand::Null));
                    push(Some(RegRange::vector(f.vaddr, if wide { 2 } else { 1 })));
                }
                if let Some(saddr) = f.saddr {
                    let width = if f.segment == FlatSegment::Scratch {
                        1
                    } else {
                        2
                    };
                    push(scalar(saddr, width));
                }
                if f.kind != MemKind::Load {
                    push(Some(RegRange::vector(f.vdata, f.data_dwords)));
                }
            }
            Fields::Ds(d) => {
                if d.uses_addr {
                    push(Some(RegRange::vector(d.addr, 1)));
                }
                if d.data0_dwords > 0 {
                    push(Some(RegRange::vector(d.data0, d.data0_dwords)));
                }
                if d.data1_dwords > 0 {
                    push(Some(RegRange::vector(d.data1, d.data1_dwords)));
                }
                if d.gds || self.has_flag(flag::READS_M0) {
                    push(Some(scalar_code_range(SCALAR_M0, 1)));
                }
            }
            Fields::Mimg(m) => {
                if m.nsa {
                    for index in 0..usize::from(m.addr_dwords) {
                        push(Some(RegRange::vector(m.address_register(index), 1)));
                    }
                } else {
                    push(Some(RegRange::vector(m.addr[0], m.addr_dwords)));
                }
                push(Some(scalar_code_range(
                    u16::from(m.srsrc),
                    if m.r128 { 4 } else { 8 },
                )));
                if m.uses_sampler {
                    push(Some(scalar_code_range(u16::from(m.ssamp), 4)));
                }
                if m.kind != MemKind::Load && m.data_dwords > 0 {
                    push(Some(RegRange::vector(m.vdata, m.data_dwords)));
                }
            }
            Fields::Exp(e) => {
                for (i, &reg) in e.vsrc.iter().enumerate() {
                    let live = if e.compr {
                        i < 2 && e.en & (0b11 << (i * 2)) != 0
                    } else {
                        e.en & (1 << i) != 0
                    };
                    if live {
                        push(Some(RegRange::vector(reg, 1)));
                    }
                }
            }
            Fields::Ldsdir(_) => push(Some(scalar_code_range(SCALAR_M0, 1))),
            Fields::Vinterp(v) => {
                for src in v.src {
                    push(register(src, 1));
                }
            }
            Fields::Sopp { .. } => {}
        }
        if self.family.is_vector_alu() || self.has_flag(flag::READS_EXEC) {
            push(Some(scalar_code_range(SCALAR_EXEC_LO, u16::from(mask))));
        }
        if self.has_flag(flag::READS_VCC) {
            push(Some(scalar_code_range(SCALAR_VCC_LO, u16::from(mask))));
        }
        if self.has_flag(flag::READS_M0) && !matches!(self.fields, Fields::Ds(_)) {
            push(Some(scalar_code_range(SCALAR_M0, 1)));
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BasicBlock {
    pub id: usize,
    pub start_pc: u32,
    pub end_pc: u32,
    pub first: usize,
    pub end: usize,
    pub successors: Vec<usize>,
    pub predecessors: Vec<usize>,
    pub reachable: bool,
}

impl BasicBlock {
    pub fn instructions(&self) -> std::ops::Range<usize> {
        self.first..self.end
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum ResourceKind {
    BufferDescriptor,
    ImageDescriptor,
    SamplerDescriptor,
    ScalarMemoryPair,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ResourceBinding {
    pub kind: ResourceKind,
    pub base_sgpr: u8,
    pub dword_count: u8,
    pub writable: bool,
}

/// One instruction's use of a V#, T# or S#: the SGPRs it reads, how it uses the resource and
/// where the descriptor came from.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DescriptorUse {
    /// The instruction.
    pub pc: u32,
    /// [`ResourceKind::BufferDescriptor`], [`ResourceKind::ImageDescriptor`] or
    /// [`ResourceKind::SamplerDescriptor`].
    pub kind: ResourceKind,
    pub base_sgpr: u8,
    pub dword_count: u8,
    pub access: DescriptorAccess,
    pub source: DescriptorSource,
    /// Buffer instruction addressing, needed to bound writes before asynchronous submission.
    pub buffer_access: Option<BufferAccess>,
}

/// A compute program whose only effect is to fill a buffer: every thread stores the same four
/// user data dwords through the V# in user data, at its own index along x in groups of 64
/// threads (a title's memset on the GPU; see `cfg::sources`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferFill {
    /// The user SGPR the V# starts at.
    pub buffer: u8,
    /// The first of the four consecutive user SGPRs that hold the record stored.
    pub value: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScalarOffset {
    /// The value is at most this unsigned bound (a constant or explicit bit mask).
    Bounded(u32),
    /// An unmodified input user SGPR, whose value the runtime reads at the work item.
    UserSgpr(u8),
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferAccess {
    pub scalar: bool,
    pub indexed: bool,
    pub vector_offset: bool,
    pub immediate_offset: u32,
    pub scalar_offset: ScalarOffset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DescriptorAccess {
    /// Read through a sampler: samples, gathers and `image_get_lod`, and the S# they use.
    Sampled,
    /// Read without one: image, buffer and scalar buffer loads, BVH intersections.
    Loaded,
    /// Written by stores, or read and written by atomics.
    Written,
    /// Only the descriptor is read (`image_get_resinfo`).
    Queried,
}

/// Where a descriptor came from.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum DescriptorSource {
    /// Still the wave's input: user SGPRs `s[first..first + dword_count]` (user data starts at
    /// s8 in merged vertex programs).
    UserSgprs { first: u8 },
    /// Read from memory by scalar loads at constant offsets, starting from user data.
    Table(DescriptorTable),
    /// One of a finite set of table entries. The scalar offset is bounded by an explicit
    /// mask/shift in the guest program; each table has the same pointer chain and one possible
    /// byte offset. The runtime prepares all candidates before executing the shader.
    TableChoices(Vec<DescriptorTable>),
    /// Anything else: offsets or addresses computed at run time, values from vector registers,
    /// paths that disagree, descriptors modified after they were read.
    Dynamic,
}

/// A descriptor in memory, reached by following addresses from user data: the first is held
/// in the user SGPRs from `root` (as `root_form`), each link reads the next address at a
/// constant offset from the current one, and the descriptor is `offset` bytes past the last.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DescriptorTable {
    pub root: u8,
    pub root_form: AddressForm,
    pub links: Vec<TableLink>,
    pub offset: i64,
}

/// One address read on the way to a descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TableLink {
    /// Bytes past the current address where the next address is held.
    pub offset: i64,
    pub form: AddressForm,
}

/// How an address is held in registers or memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AddressForm {
    /// A 64-bit pointer, low dword first, as `s_load_dword*` takes it.
    Pointer,
    /// A V#, whose base address is the address, as `s_buffer_load_dword*` takes it.
    Buffer,
}

impl AddressForm {
    /// The address held in a pointer's two dwords or a V#'s first two.
    pub fn address(self, lo: u32, hi: u32) -> u64 {
        let hi = match self {
            Self::Pointer => hi,
            Self::Buffer => hi & 0xffff,
        };
        (u64::from(hi) << 32) | u64::from(lo)
    }
}

impl DescriptorTable {
    /// The descriptor's guest address: `sgpr(n)` gives the wave's input SGPR n (user data) and
    /// `read(address)` two dwords of guest memory. Scalar loads ignore the low two address
    /// bits, and so does this.
    pub fn address(
        &self,
        sgpr: impl Fn(u8) -> u32,
        read: impl FnMut(u64) -> Option<[u32; 2]>,
    ) -> Option<u64> {
        self.resolve_address(sgpr, read, false)
    }

    /// For optional host inspection: an unbound root or link is absent, even when adding
    /// its offset would produce a nonzero address. This does not change shader addressing.
    pub fn bound_address(
        &self,
        sgpr: impl Fn(u8) -> u32,
        read: impl FnMut(u64) -> Option<[u32; 2]>,
    ) -> Option<u64> {
        self.resolve_address(sgpr, read, true)
    }

    fn resolve_address(
        &self,
        sgpr: impl Fn(u8) -> u32,
        mut read: impl FnMut(u64) -> Option<[u32; 2]>,
        bound: bool,
    ) -> Option<u64> {
        let at = |address: u64, offset: i64| address.wrapping_add(offset as u64) & !3;
        let mut address = self
            .root_form
            .address(sgpr(self.root), sgpr(self.root.wrapping_add(1)));
        for link in &self.links {
            if bound && address == 0 {
                return None;
            }
            let [lo, hi] = read(at(address, link.offset))?;
            address = link.form.address(lo, hi);
        }
        if bound && address == 0 {
            return None;
        }
        Some(at(address, self.offset))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ExportInfo {
    pub color_mask: u32,
    pub depth_mask: u8,
    pub position_mask: u8,
    pub param_mask: u32,
    pub primitive: bool,
    pub compressed: bool,
    pub valid_mask: bool,
}

impl ExportInfo {
    pub fn exports_color(&self) -> bool {
        self.color_mask != 0
    }

    pub fn param_count(&self) -> u32 {
        u32::BITS - self.param_mask.leading_zeros()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShaderMetadata {
    pub max_sgpr: Option<u8>,
    pub max_vgpr: Option<u8>,
    pub sgpr_count: u16,
    pub vgpr_count: u16,
    pub live_in_scalar: u128,
    pub live_in_vector: u128,
    pub user_sgpr_count: u8,
    pub uses_vcc: bool,
    pub uses_lds: bool,
    pub uses_gds: bool,
    pub uses_scratch: bool,
    pub uses_pc_relative: bool,
    /// Some instruction indexes registers by M0 (`v_movrel*`, `s_movrel*`).
    pub uses_relative_indexing: bool,
    /// Some instruction indexes SGPRs by M0 (`s_movrel*`): any SGPR may then be the guest's.
    /// The vector forms reach VGPRs only.
    pub uses_relative_sgprs: bool,
    pub uses_interpolation: bool,
    pub bindings: Vec<ResourceBinding>,
    /// Every reachable descriptor use, in pc order (see `cfg::sources`).
    pub descriptor_uses: Vec<DescriptorUse>,
    /// Scalar pointer loads/stores, with the provenance of their base address and bounded
    /// scalar offset. Kept separately so resource-descriptor uses retain their API meaning.
    pub pointer_uses: Vec<DescriptorUse>,
    /// The program does nothing but fill a buffer with user data.
    pub buffer_fill: Option<BufferFill>,
    pub exports: ExportInfo,
    pub user_data_tail_sgpr: Option<u8>,
    /// The stage the program was analysed for.
    pub stage: Option<ShaderStage>,
    /// The program reads its wave's index (`s_get_waveid_in_workgroup`).
    pub uses_wave_id: bool,
}

/// Where an `s_setpc_b64` or `s_swappc_b64` goes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JumpTarget {
    /// The target is provably this guest pc (`s_getpc_b64` plus constants).
    Static(u32),
    /// A call's return address.
    Return,
    /// A value from outside the program (user data or memory): another program's code.
    External,
    /// A code address computed from data, e.g. a jump table: any instruction.
    Open,
}

/// Computed control flow (see `cfg::computed`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComputedFlow {
    /// The computed jump at each guest pc.
    pub jumps: std::collections::BTreeMap<u32, JumpTarget>,
    /// Guest pcs a computed jump may reach at run time: call return addresses, static
    /// targets and, when some jump is open, every instruction.
    pub entries: std::collections::BTreeSet<u32>,
    /// Jumps and calls to code outside the program ([`JumpTarget::External`]), in pc order.
    pub external: Vec<ExternalTarget>,
}

/// A jump or call whose target comes from outside the program: another program's code, such
/// as a fetch subroutine whose address is in user data. Linking that code
/// ([`crate::Linkage`]) lets the translation follow it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ExternalTarget {
    /// The `s_setpc_b64`/`s_swappc_b64`.
    pub pc: u32,
    /// `s_swappc_b64` with a return address (control comes back after it).
    pub call: bool,
    /// The SGPR pair holding the target, when it still holds the value the wave started with:
    /// the target is `s[sgpr]` | `s[sgpr + 1]` << 32 of the wave's input.
    pub sgpr: Option<u8>,
}

/// Where a program's words lie in guest memory: the shader's own words, then each linked region
/// ([`crate::Linkage`]), end to end. Instruction pcs are byte offsets into that sequence.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CodeLayout {
    /// Guest address of the shader's first word, when known.
    pub base: Option<u64>,
    /// Bytes of the shader's own words; linked regions follow.
    pub own_bytes: u32,
    pub regions: Vec<LinkedRegion>,
    /// The guest address PS5 hardware puts in `s[6:7]` for a merged-stage front half: the back
    /// half's (see [`crate::Linkage::second_half`]).
    pub second_half: Option<u64>,
}

/// One linked region in a [`CodeLayout`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkedRegion {
    pub address: u64,
    /// pc of its first word.
    pub start: u32,
    pub bytes: u32,
}

impl CodeLayout {
    /// The guest address of `pc`: within the shader's own words `base + pc` (counted from zero
    /// when the base is unknown), within a linked region the region's address plus the offset.
    pub fn guest_address(&self, pc: u32) -> u64 {
        self.regions
            .iter()
            .find(|r| (r.start..r.start + r.bytes).contains(&pc))
            .map_or_else(
                || self.base.unwrap_or(0).wrapping_add(u64::from(pc)),
                |r| r.address + u64::from(pc - r.start),
            )
    }

    /// The pc of a guest address in the shader's own words or a linked region.
    pub fn pc_of(&self, address: u64) -> Option<u32> {
        let own = address.checked_sub(self.base.unwrap_or(0));
        if let Some(own) = own.filter(|&o| o < u64::from(self.own_bytes)) {
            return Some(own as u32);
        }
        self.regions.iter().find_map(|r| {
            let offset = address.checked_sub(r.address)?;
            (offset < u64::from(r.bytes)).then_some(r.start + offset as u32)
        })
    }
}

impl ComputedFlow {
    /// Whether some jump needs the run-time guest-to-host lookup.
    pub fn has_dynamic_jumps(&self) -> bool {
        self.jumps
            .values()
            .any(|t| !matches!(t, JumpTarget::Static(_)))
    }

    pub fn is_open(&self) -> bool {
        self.jumps.values().any(|t| *t == JumpTarget::Open)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShaderIr {
    pub instructions: Vec<Instruction>,
    pub blocks: Vec<BasicBlock>,
    pub code_words: usize,
    pub metadata: ShaderMetadata,
    pub flow: ComputedFlow,
    /// Where the decoded words lie in guest memory.
    pub layout: CodeLayout,
}

impl ShaderIr {
    pub fn instruction_at(&self, pc: u32) -> Option<usize> {
        self.instructions.binary_search_by_key(&pc, |i| i.pc).ok()
    }

    pub fn block_containing(&self, index: usize) -> Option<usize> {
        self.blocks
            .iter()
            .position(|b| b.instructions().contains(&index))
    }
}
