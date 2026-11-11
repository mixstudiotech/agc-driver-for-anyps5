use crate::ir::memops::{self, img, DS_GDS_ONLY, DS_GWS, DS_NO_ADDR};
use crate::ir::opcodes::{self, flag, VopcInfo};
use crate::ir::{
    Buffer, Dpp16, Ds, Export, Fields, Flat, FlatSegment, Image, Instruction, InstructionFamily,
    Interp, LdsDir, MemKind, Operand, Sdwa, ShaderIr, Smem, VInterp, Vop3, Vop3p, VopExt,
    MAX_NSA_ADDRESSES,
};
use crate::lowering::waitcnt;
use crate::CompileError;
use host::HostTables;
use std::collections::BTreeSet;
use std::ops::Range;

pub mod host;
#[cfg(test)]
mod tests;

use InstructionFamily as F;

pub const SDWA_MARKER: u16 = 249;
pub const DPP16_MARKER: u16 = 250;
pub const DPP8_MARKER: u16 = 233;
pub const DPP8_FI_MARKER: u16 = 234;
const SCALAR_NULL: u16 = 125;
const SCALAR_M0: u16 = 124;
const FLAT_SADDR_NULL: u16 = 0x7d;
const FLAT_SADDR_OFF: u16 = 0x7f;
const SOP1_SWAPPC_B64: u16 = 0x21;
pub const SOPK_SETREG_IMM32: u16 = 0x15;
const VOP3_INTERP_OPS: [u16; 6] = [0x200, 0x201, 0x202, 0x342, 0x343, 0x35a];
const EXP_RESERVED_BITS: u32 = 0x03ff_e000;

pub fn family_of(word: u32) -> Option<InstructionFamily> {
    if word & 0x8000_0000 == 0 {
        return Some(match (word >> 25) & 0x3f {
            0x3e => F::Vopc,
            0x3f => F::Vop1,
            _ => F::Vop2,
        });
    }
    if word & 0xc000_0000 == 0x8000_0000 {
        return Some(match (word >> 23) & 0x7f {
            0x7d => F::Sop1,
            0x7e => F::Sopc,
            0x7f => F::Sopp,
            op if op >= 0x60 => F::Sopk,
            _ => F::Sop2,
        });
    }
    Some(match word >> 26 {
        0x32 => F::Vintrp,
        0x33 => F::Vop3p,
        0x35 => F::Vop3,
        0x36 => F::Ds,
        0x37 => F::Flat,
        0x38 => F::Mubuf,
        0x3a => F::Mtbuf,
        0x3c => F::Mimg,
        0x3d => F::Smem,
        0x3e => F::Exp,
        _ => return None,
    })
}

fn bit(word: u32, index: u32) -> bool {
    (word >> index) & 1 != 0
}

fn field(word: u32, shift: u32, mask: u32) -> u16 {
    ((word >> shift) & mask) as u16
}

fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

struct Decoded {
    family: InstructionFamily,
    opcode: u16,
    flags: u64,
    fields: Fields,
    words: usize,
    branch: Option<u16>,
}

struct Ctx<'a> {
    words: &'a [u32],
    index: usize,
    pc: u32,
    /// Decoding translated code: the target's field layouts, with each opcode mapped back to
    /// the gfx1013 instruction it implements.
    host: Option<&'a HostTables>,
}

impl Ctx<'_> {
    fn gfx11(&self) -> bool {
        self.host.is_some_and(|h| h.target().is_gfx11_plus())
    }

    /// The gfx1013 opcode an opcode field stands for.
    fn opcode(&self, family: InstructionFamily, raw: u16) -> Result<u16, CompileError> {
        match self.host {
            None => Ok(raw),
            Some(host) => host
                .guest_opcode(family, raw)
                .ok_or_else(|| self.unknown(family, raw)),
        }
    }

    /// The gfx10 encoding of an operand code (gfx11 swapped M0 and NULL).
    fn code(&self, code: u16) -> u16 {
        match code {
            host::GFX11_NULL if self.gfx11() => SCALAR_NULL,
            host::GFX11_M0 if self.gfx11() => SCALAR_M0,
            other => other,
        }
    }

    fn raw_operand(&self, code: u16) -> Operand {
        Operand::from_code_or_raw(self.code(code))
    }

    fn word(&self, offset: usize) -> Result<u32, CompileError> {
        self.words
            .get(self.index + offset)
            .copied()
            .ok_or(CompileError::Truncated { pc: self.pc })
    }

    fn unknown(&self, family: InstructionFamily, opcode: u16) -> CompileError {
        CompileError::UnknownOpcode {
            pc: self.pc,
            family,
            opcode,
        }
    }

    fn src(&self, code: u16) -> Result<Operand, CompileError> {
        Operand::from_code(self.code(code)).ok_or(CompileError::InvalidOperand {
            pc: self.pc,
            reason: "reserved source operand encoding",
        })
    }

    fn sdst(&self, code: u16) -> Result<Operand, CompileError> {
        Operand::scalar_destination(self.code(code)).ok_or(CompileError::InvalidOperand {
            pc: self.pc,
            reason: "invalid scalar destination",
        })
    }

    fn literal(&self, offset: usize, ops: &mut [&mut Operand]) -> Result<usize, CompileError> {
        if !ops.iter().any(|op| matches!(op, Operand::Literal(_))) {
            return Ok(0);
        }
        let value = self.word(offset)?;
        for op in ops.iter_mut() {
            if let Operand::Literal(v) = op {
                *v = value;
            }
        }
        Ok(1)
    }

    fn decode(&self) -> Result<Decoded, CompileError> {
        let word = self.word(0)?;
        let invalid = CompileError::InvalidEncoding { pc: self.pc, word };
        let mut family = family_of(word).ok_or_else(|| invalid.clone())?;
        if self.gfx11() {
            family = match word >> 24 {
                0xcd => F::Vinterp,
                0xce => F::Ldsdir,
                _ if family == F::Vintrp => return Err(invalid),
                _ => family,
            };
        }
        match family {
            F::Sop1 => self.sop1(word),
            F::Sop2 => self.sop2(word),
            F::Sopk => self.sopk(word),
            F::Sopc => self.sopc(word),
            F::Sopp => self.sopp(word),
            F::Vop1 => self.vop1(word),
            F::Vop2 => self.vop2(word),
            F::Vopc => self.vopc(word),
            F::Vop3 => self.vop3(word),
            F::Vop3p => self.vop3p(word),
            F::Vintrp => self.vintrp(word),
            F::Smem => self.smem(word),
            F::Mubuf | F::Mtbuf => self.buffer(word, family),
            F::Flat => self.flat(word),
            F::Ds => self.ds(word),
            F::Mimg => self.mimg(word),
            F::Exp => self.exp(word),
            F::Ldsdir => self.ldsdir(word),
            F::Vinterp => self.vinterp(word),
        }
    }

    /// gfx11 `lds_param_load` / `lds_direct_load` (translated interpolation), kept with their
    /// gfx11 opcodes: gfx1013 has no counterpart.
    fn ldsdir(&self, w: u32) -> Result<Decoded, CompileError> {
        let fields = Fields::Ldsdir(LdsDir {
            vdst: w as u8,
            attr: field(w, 10, 0x3f) as u8,
            chan: field(w, 8, 3) as u8,
            wait_vdst: field(w, 16, 0xf) as u8,
        });
        Ok(self.simple(F::Ldsdir, field(w, 20, 3), flag::READS_M0, fields, 1))
    }

    /// gfx11 VINTERP (`v_interp_p10_f32` and relatives), kept with its gfx11 opcode.
    fn vinterp(&self, w0: u32) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let src = [
            self.src(field(w1, 0, 0x1ff))?,
            self.src(field(w1, 9, 0x1ff))?,
            self.src(field(w1, 18, 0x1ff))?,
        ];
        let fields = Fields::Vinterp(VInterp {
            vdst: w0 as u8,
            src,
            neg: field(w1, 29, 7) as u8,
            opsel: field(w0, 11, 0xf) as u8,
            clamp: bit(w0, 15),
            wait_exp: field(w0, 8, 7) as u8,
        });
        Ok(self.simple(F::Vinterp, field(w0, 16, 0x7f), 0, fields, 2))
    }

    fn simple(
        &self,
        family: InstructionFamily,
        opcode: u16,
        flags: u64,
        fields: Fields,
        words: usize,
    ) -> Decoded {
        Decoded {
            family,
            opcode,
            flags,
            fields,
            words,
            branch: None,
        }
    }

    fn sop1(&self, w: u32) -> Result<Decoded, CompileError> {
        let raw = field(w, 8, 0xff);
        if self.gfx11() && raw == host::GFX11_SENDMSG_RTN_B64 {
            return self.sendmsg_rtn(w);
        }
        let op = self.opcode(F::Sop1, raw)?;
        let info = opcodes::lookup(F::Sop1, op).ok_or_else(|| self.unknown(F::Sop1, op))?;
        let mut flags = info.flags;
        let sdst_code = field(w, 16, 0x7f);
        let sdst = if flags & flag::NO_DST != 0 {
            self.raw_operand(sdst_code)
        } else {
            self.sdst(sdst_code)?
        };
        let mut src0 = if flags & flag::NO_SRC != 0 {
            self.raw_operand(field(w, 0, 0xff))
        } else {
            self.src(field(w, 0, 0xff))?
        };
        if op == SOP1_SWAPPC_B64 && sdst == Operand::Null {
            flags =
                (flags & !(flag::CALL | flag::D2 | flag::PC_RELATIVE)) | flag::JUMP | flag::NO_DST;
        }
        let extra = if flags & flag::NO_SRC == 0 {
            self.literal(1, &mut [&mut src0])?
        } else {
            0
        };
        Ok(self.simple(F::Sop1, op, flags, Fields::Sop1 { sdst, src0 }, 1 + extra))
    }

    /// gfx11 `s_sendmsg_rtn_b64 sdst, sendmsg(MSG_RTN_GET_REALTIME)`, which the translation emits
    /// for `s_memrealtime`.
    fn sendmsg_rtn(&self, w: u32) -> Result<Decoded, CompileError> {
        if field(w, 0, 0xff) != host::MSG_RTN_GET_REALTIME {
            return Err(self.unknown(F::Sop1, host::GFX11_SENDMSG_RTN_B64));
        }
        let info = memops::smem(host::SMEM_MEMREALTIME)
            .ok_or_else(|| self.unknown(F::Smem, host::SMEM_MEMREALTIME))?;
        let fields = Fields::Smem(Smem {
            sdata: self.sdst(field(w, 16, 0x7f))?,
            sbase: 0,
            offset: 0,
            soffset: Operand::Null,
            glc: false,
            dlc: false,
            dwords: info.dwords,
            buffer: false,
        });
        Ok(self.simple(F::Smem, info.op, info.flags, fields, 1))
    }

    fn sop2(&self, w: u32) -> Result<Decoded, CompileError> {
        let op = self.opcode(F::Sop2, field(w, 23, 0x7f))?;
        let info = opcodes::lookup(F::Sop2, op).ok_or_else(|| self.unknown(F::Sop2, op))?;
        let sdst = self.sdst(field(w, 16, 0x7f))?;
        let mut src0 = self.src(field(w, 0, 0xff))?;
        let mut src1 = self.src(field(w, 8, 0xff))?;
        let extra = self.literal(1, &mut [&mut src0, &mut src1])?;
        Ok(self.simple(
            F::Sop2,
            op,
            info.flags,
            Fields::Sop2 { sdst, src0, src1 },
            1 + extra,
        ))
    }

    fn sopk(&self, w: u32) -> Result<Decoded, CompileError> {
        let op = self.opcode(F::Sopk, field(w, 23, 0x1f))?;
        let info = opcodes::lookup(F::Sopk, op).ok_or_else(|| self.unknown(F::Sopk, op))?;
        let sdst = self.raw_operand(field(w, 16, 0x7f));
        let simm16 = field(w, 0, 0xffff);
        let imm32 = if op == SOPK_SETREG_IMM32 {
            Some(self.word(1)?)
        } else {
            None
        };
        let branch = (info.flags & (flag::CALL | flag::COND_BRANCH) != 0).then_some(simm16);
        Ok(Decoded {
            family: F::Sopk,
            opcode: op,
            flags: info.flags,
            fields: Fields::Sopk {
                sdst,
                simm16,
                imm32,
            },
            words: 1 + usize::from(imm32.is_some()),
            branch,
        })
    }

    fn sopc(&self, w: u32) -> Result<Decoded, CompileError> {
        let op = self.opcode(F::Sopc, field(w, 16, 0x7f))?;
        let info = opcodes::lookup(F::Sopc, op).ok_or_else(|| self.unknown(F::Sopc, op))?;
        let mut src0 = self.src(field(w, 0, 0xff))?;
        let mut src1 = self.src(field(w, 8, 0xff))?;
        let extra = self.literal(1, &mut [&mut src0, &mut src1])?;
        Ok(self.simple(
            F::Sopc,
            op,
            info.flags,
            Fields::Sopc { src0, src1 },
            1 + extra,
        ))
    }

    fn sopp(&self, w: u32) -> Result<Decoded, CompileError> {
        let op = self.opcode(F::Sopp, field(w, 16, 0x7f))?;
        let info = opcodes::lookup(F::Sopp, op).ok_or_else(|| self.unknown(F::Sopp, op))?;
        let mut simm16 = field(w, 0, 0xffff);
        if self.gfx11() && op == opcodes::SOPP_WAITCNT {
            let (vm, exp, lgkm) = waitcnt::decode_gfx11(simm16);
            simm16 = waitcnt::encode_gfx10(vm, exp, lgkm);
        }
        let branch = (info.flags & (flag::BRANCH | flag::COND_BRANCH) != 0).then_some(simm16);
        Ok(Decoded {
            family: F::Sopp,
            opcode: op,
            flags: info.flags,
            fields: Fields::Sopp { simm16 },
            words: 1,
            branch,
        })
    }

    fn sdwa_word(w1: u32, vopc: bool) -> Sdwa {
        Sdwa {
            dst_sel: if vopc { 0 } else { field(w1, 8, 7) as u8 },
            dst_unused: if vopc { 0 } else { field(w1, 11, 3) as u8 },
            clamp: !vopc && bit(w1, 13),
            omod: if vopc { 0 } else { field(w1, 14, 3) as u8 },
            src0_sel: field(w1, 16, 7) as u8,
            src0_sext: bit(w1, 19),
            src0_neg: bit(w1, 20),
            src0_abs: bit(w1, 21),
            src1_sel: field(w1, 24, 7) as u8,
            src1_sext: bit(w1, 27),
            src1_neg: bit(w1, 28),
            src1_abs: bit(w1, 29),
        }
    }

    fn dpp16_word(w1: u32) -> Dpp16 {
        Dpp16 {
            ctrl: field(w1, 8, 0x1ff),
            fetch_inactive: bit(w1, 18),
            bound_ctrl: bit(w1, 19),
            src0_neg: bit(w1, 20),
            src0_abs: bit(w1, 21),
            src1_neg: bit(w1, 22),
            src1_abs: bit(w1, 23),
            bank_mask: field(w1, 24, 0xf) as u8,
            row_mask: field(w1, 28, 0xf) as u8,
        }
    }

    fn sdwa_src(
        &self,
        w1: u32,
        shift: u32,
        select_bit: u32,
        vgpr: u16,
    ) -> Result<Operand, CompileError> {
        let code = if shift == 0 { field(w1, 0, 0xff) } else { vgpr };
        self.src(if bit(w1, select_bit) {
            code
        } else {
            code + 256
        })
    }

    fn vop_ext(
        &self,
        src0_code: u16,
    ) -> Result<(Operand, VopExt, usize, Option<u32>), CompileError> {
        match src0_code {
            SDWA_MARKER => {
                let w1 = self.word(1)?;
                Ok((
                    self.sdwa_src(w1, 0, 23, 0)?,
                    VopExt::Sdwa(Self::sdwa_word(w1, false)),
                    2,
                    Some(w1),
                ))
            }
            DPP16_MARKER => {
                let w1 = self.word(1)?;
                Ok((
                    Operand::Vgpr(w1 as u8),
                    VopExt::Dpp16(Self::dpp16_word(w1)),
                    2,
                    Some(w1),
                ))
            }
            DPP8_MARKER | DPP8_FI_MARKER => {
                let w1 = self.word(1)?;
                let ext = VopExt::Dpp8 {
                    selects: w1 >> 8,
                    fetch_inactive: src0_code == DPP8_FI_MARKER,
                };
                Ok((Operand::Vgpr(w1 as u8), ext, 2, Some(w1)))
            }
            code => Ok((self.src(code)?, VopExt::None, 1, None)),
        }
    }

    fn vop1(&self, w: u32) -> Result<Decoded, CompileError> {
        let op = self.opcode(F::Vop1, field(w, 9, 0xff))?;
        let info = opcodes::lookup(F::Vop1, op).ok_or_else(|| self.unknown(F::Vop1, op))?;
        let vdst_code = field(w, 17, 0xff);
        let vdst = if info.flags & flag::SDST != 0 {
            self.sdst(vdst_code)?
        } else {
            Operand::Vgpr(vdst_code as u8)
        };
        let src0_code = field(w, 0, 0x1ff);
        if info.flags & flag::NO_OPERANDS != 0 {
            let src0 = self.raw_operand(src0_code);
            let fields = Fields::Vop1 {
                vdst,
                src0,
                ext: VopExt::None,
            };
            return Ok(self.simple(F::Vop1, op, info.flags, fields, 1));
        }
        let (mut src0, ext, words, _) = self.vop_ext(src0_code)?;
        let extra = self.literal(words, &mut [&mut src0])?;
        Ok(self.simple(
            F::Vop1,
            op,
            info.flags,
            Fields::Vop1 { vdst, src0, ext },
            words + extra,
        ))
    }

    fn vop2(&self, w: u32) -> Result<Decoded, CompileError> {
        let op = self.opcode(F::Vop2, field(w, 25, 0x3f))?;
        let info = opcodes::lookup(F::Vop2, op).ok_or_else(|| self.unknown(F::Vop2, op))?;
        let vdst = Operand::Vgpr(field(w, 17, 0xff) as u8);
        let vsrc1_field = field(w, 9, 0xff);
        let (mut src0, ext, words, ext_word) = self.vop_ext(field(w, 0, 0x1ff))?;
        let vsrc1 = match (ext, ext_word) {
            (VopExt::Sdwa(_), Some(w1)) => self.sdwa_src(w1, 24, 31, vsrc1_field)?,
            _ => Operand::Vgpr(vsrc1_field as u8),
        };
        let mut extra = self.literal(words, &mut [&mut src0])?;
        let constant = if info.flags & flag::LITK != 0 {
            if extra != 0 || ext != VopExt::None {
                return Err(CompileError::InvalidOperand {
                    pc: self.pc,
                    reason: "inline-constant MAD forms take exactly one literal",
                });
            }
            extra = 1;
            Some(self.word(words)?)
        } else {
            None
        };
        let fields = Fields::Vop2 {
            vdst,
            src0,
            vsrc1,
            constant,
            ext,
        };
        Ok(self.simple(F::Vop2, op, info.flags, fields, words + extra))
    }

    fn vopc(&self, w: u32) -> Result<Decoded, CompileError> {
        let op = self.opcode(F::Vopc, field(w, 17, 0xff))?;
        let info = VopcInfo::decode(op).ok_or_else(|| self.unknown(F::Vopc, op))?;
        let flags = info.flags();
        let vsrc1_field = field(w, 9, 0xff);
        let src0_code = field(w, 0, 0x1ff);
        let mut sdst = if info.exec {
            Operand::ExecLo
        } else {
            Operand::VccLo
        };
        let (mut src0, mut ext, words, ext_word) = self.vop_ext(src0_code)?;
        let vsrc1 = match ext_word {
            Some(w1) if src0_code == SDWA_MARKER => {
                ext = VopExt::Sdwa(Self::sdwa_word(w1, true));
                if bit(w1, 15) && !info.exec {
                    sdst = self.sdst(field(w1, 8, 0x7f))?;
                }
                self.sdwa_src(w1, 24, 31, vsrc1_field)?
            }
            _ => Operand::Vgpr(vsrc1_field as u8),
        };
        let extra = self.literal(words, &mut [&mut src0])?;
        let fields = Fields::Vopc {
            sdst,
            src0,
            vsrc1,
            ext,
        };
        Ok(self.simple(F::Vopc, op, flags, fields, words + extra))
    }

    fn vop3(&self, w0: u32) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let op = self.opcode(F::Vop3, field(w0, 16, 0x3ff))?;
        let flags = opcodes::vop3_flags(op).ok_or_else(|| self.unknown(F::Vop3, op))?;
        let base = opcodes::vop3_base(op).map(|(family, _)| family);
        let vop3b =
            flags & flag::VOP3B != 0 || (base == Some(F::Vop2) && flags & flag::CARRY_OUT != 0);
        let vdst_code = field(w0, 0, 0xff);
        let vdst = if flags & flag::CMPX != 0 {
            self.raw_operand(vdst_code)
        } else if base == Some(F::Vopc) || flags & flag::SDST != 0 {
            self.sdst(vdst_code)?
        } else {
            Operand::Vgpr(vdst_code as u8)
        };
        let codes = [
            field(w1, 0, 0x1ff),
            field(w1, 9, 0x1ff),
            field(w1, 18, 0x1ff),
        ];
        let mut src = codes.map(|code| self.raw_operand(code));
        let interp = VOP3_INTERP_OPS.contains(&op);
        if interp {
            src[0] = Operand::Raw(codes[0]);
            if op == opcodes::VOP3_INTERP_MOV_F32 {
                src[1] = Operand::Raw(codes[1]);
            }
        } else {
            for (slot, &code) in src
                .iter_mut()
                .zip(&codes)
                .take(opcodes::vop3_source_count(op, flags))
            {
                *slot = self.src(code)?;
            }
        }
        let (abs, opsel, sdst) = if vop3b {
            (0, 0, Some(self.sdst(field(w0, 8, 0x7f))?))
        } else {
            (field(w0, 8, 7) as u8, field(w0, 11, 0xf) as u8, None)
        };
        let count = if interp {
            0
        } else {
            opcodes::vop3_source_count(op, flags)
        };
        let [s0, s1, s2] = &mut src;
        let mut used: Vec<&mut Operand> = vec![s0, s1, s2];
        used.truncate(count);
        let extra = self.literal(2, &mut used)?;
        let fields = Fields::Vop3(Vop3 {
            vdst,
            sdst,
            src,
            abs,
            neg: field(w1, 29, 7) as u8,
            opsel,
            clamp: bit(w0, 15),
            omod: field(w1, 27, 3) as u8,
        });
        Ok(self.simple(F::Vop3, op, flags, fields, 2 + extra))
    }

    fn vop3p(&self, w0: u32) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let raw = field(w0, 16, 0x7f);
        let op = self.opcode(F::Vop3p, raw)?;
        let signed_dot = self.host.is_some_and(|h| h.is_signed_dot(raw));
        if signed_dot && field(w1, 29, 7) as u8 != host::SIGNED_DOT_SOURCES {
            return Err(self.unknown(F::Vop3p, raw));
        }
        let info = opcodes::lookup(F::Vop3p, op).ok_or_else(|| self.unknown(F::Vop3p, op))?;
        let codes = [
            field(w1, 0, 0x1ff),
            field(w1, 9, 0x1ff),
            field(w1, 18, 0x1ff),
        ];
        let count = opcodes::flag_source_count(info.flags).unwrap_or(3);
        let mut src = codes.map(|code| self.raw_operand(code));
        for (slot, &code) in src.iter_mut().zip(&codes).take(count) {
            *slot = self.src(code)?;
        }
        let [s0, s1, s2] = &mut src;
        let mut used: Vec<&mut Operand> = vec![s0, s1, s2];
        used.truncate(count);
        let extra = self.literal(2, &mut used)?;
        let fields = Fields::Vop3p(Vop3p {
            vdst: Operand::Vgpr(w0 as u8),
            src,
            neg: if signed_dot {
                0
            } else {
                field(w1, 29, 7) as u8
            },
            neg_hi: field(w0, 8, 7) as u8,
            opsel: field(w0, 11, 7) as u8,
            opsel_hi: (field(w1, 27, 3) | (field(w0, 14, 1) << 2)) as u8,
            clamp: bit(w0, 15),
        });
        Ok(self.simple(F::Vop3p, op, info.flags, fields, 2 + extra))
    }

    fn vintrp(&self, w: u32) -> Result<Decoded, CompileError> {
        let op = field(w, 16, 3);
        if op > opcodes::VINTRP_MOV {
            return Err(self.unknown(F::Vintrp, op));
        }
        let mut flags = flag::READS_M0;
        if op == opcodes::VINTRP_P2 {
            flags |= flag::READS_DST;
        }
        let fields = Fields::Vintrp(Interp {
            vdst: field(w, 18, 0xff) as u8,
            vsrc: w as u8,
            attr: field(w, 10, 0x3f) as u8,
            chan: field(w, 8, 3) as u8,
        });
        Ok(self.simple(F::Vintrp, op, flags, fields, 1))
    }

    fn smem(&self, w0: u32) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let op = self.opcode(F::Smem, field(w0, 18, 0xff))?;
        let info = memops::smem(op).ok_or_else(|| self.unknown(F::Smem, op))?;
        let sdata_code = field(w0, 6, 0x7f);
        let sdata = if info.dwords > 0 {
            self.sdst(sdata_code)?
        } else {
            self.raw_operand(sdata_code)
        };
        let soffset_code = self.code(field(w1, 25, 0x7f));
        let soffset = if soffset_code == SCALAR_NULL {
            Operand::Null
        } else {
            self.src(field(w1, 25, 0x7f))?
        };
        let (glc, dlc) = if self.gfx11() { (14, 13) } else { (16, 14) };
        let fields = Fields::Smem(Smem {
            sdata,
            sbase: (field(w0, 0, 0x3f) * 2) as u8,
            offset: sign_extend(w1 & 0x1f_ffff, 21),
            soffset,
            glc: bit(w0, glc),
            dlc: bit(w0, dlc),
            dwords: info.dwords,
            buffer: memops::is_buffer_smem(info),
        });
        Ok(self.simple(F::Smem, op, info.flags, fields, 2))
    }

    fn buffer(&self, w0: u32, family: InstructionFamily) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let gfx11 = self.gfx11();
        let (raw, format) = match (family, gfx11) {
            (F::Mubuf, false) => (field(w0, 18, 0x7f) | (field(w0, 25, 1) << 7), 0),
            (F::Mubuf, true) => (field(w0, 18, 0xff), 0),
            (_, false) => (
                field(w0, 16, 7) | (field(w1, 21, 1) << 3),
                field(w0, 19, 0x7f) as u8,
            ),
            (_, true) => (field(w0, 15, 0xf), field(w0, 19, 0x7f) as u8),
        };
        let op = self.opcode(family, raw)?;
        let info = if family == F::Mubuf {
            memops::mubuf(op)
        } else {
            memops::mtbuf(op)
        }
        .ok_or_else(|| self.unknown(family, op))?;
        let format = match self.host {
            Some(host) if family == F::Mtbuf => host
                .guest_format(format)
                .ok_or_else(|| self.unknown(family, op))?,
            _ => format,
        };
        // Bit positions: offen, idxen, glc, dlc, slc, tfe (word 0 below 32, word 1 above).
        let [offen, idxen, glc, dlc, slc, tfe] = if gfx11 {
            [54, 55, 14, 13, 12, 53]
        } else {
            [12, 13, 14, 15, 54, 55]
        };
        let at = |index: u32| {
            if index < 32 {
                bit(w0, index)
            } else {
                bit(w1, index - 32)
            }
        };
        let buffer = Buffer {
            vdata: field(w1, 8, 0xff) as u8,
            vaddr: w1 as u8,
            srsrc: (field(w1, 16, 0x1f) * 4) as u8,
            soffset: self.src(field(w1, 24, 0xff))?,
            offset: field(w0, 0, 0xfff),
            offen: at(offen),
            idxen: at(idxen),
            glc: at(glc),
            dlc: at(dlc),
            slc: at(slc),
            tfe: at(tfe),
            lds: family == F::Mubuf && !gfx11 && bit(w0, 16),
            format,
            kind: info.kind,
            data_dwords: info.dwords,
        };
        let fields = if family == F::Mubuf {
            Fields::Mubuf(buffer)
        } else {
            Fields::Mtbuf(buffer)
        };
        Ok(self.simple(family, op, info.flags, fields, 2))
    }

    fn flat(&self, w0: u32) -> Result<Decoded, CompileError> {
        if self.gfx11() {
            return self.flat_gfx11(w0);
        }
        let w1 = self.word(1)?;
        let op = self.opcode(F::Flat, field(w0, 18, 0x7f))?;
        let info = memops::flat(op).ok_or_else(|| self.unknown(F::Flat, op))?;
        let segment = match field(w0, 14, 3) {
            0 => FlatSegment::Flat,
            1 => FlatSegment::Scratch,
            2 => FlatSegment::Global,
            _ => {
                return Err(CompileError::InvalidEncoding {
                    pc: self.pc,
                    word: w0,
                })
            }
        };
        let addtid = memops::FLAT_ONLY.iter().any(|e| e.op == op);
        let global_only = addtid || op == memops::FLAT_ATOMIC_CSUB;
        let no_scratch_form = segment == FlatSegment::Scratch && info.kind == MemKind::Atomic;
        if (global_only && segment != FlatSegment::Global) || no_scratch_form {
            return Err(self.unknown(F::Flat, op));
        }
        let saddr = match field(w1, 16, 0x7f) {
            FLAT_SADDR_OFF => None,
            FLAT_SADDR_NULL => Some(Operand::Null),
            code => Some(self.src(code)?),
        };
        let uses_vaddr = !addtid
            && match segment {
                FlatSegment::Flat | FlatSegment::Global => true,
                FlatSegment::Scratch => saddr == Some(Operand::Null),
            };
        let raw_offset = w0 & 0xfff;
        let fields = Fields::Flat(Flat {
            segment,
            vdst: field(w1, 24, 0xff) as u8,
            vdata: field(w1, 8, 0xff) as u8,
            vaddr: w1 as u8,
            saddr,
            uses_vaddr,
            offset: if segment == FlatSegment::Flat {
                raw_offset as i32
            } else {
                sign_extend(raw_offset, 12)
            },
            glc: bit(w0, 16),
            dlc: bit(w0, 12),
            slc: bit(w0, 17),
            lds: bit(w0, 13),
            kind: info.kind,
            data_dwords: info.dwords,
        });
        Ok(self.simple(F::Flat, op, info.flags, fields, 2))
    }

    /// gfx11 FLAT/GLOBAL/SCRATCH: segment in bits 17:16, a 13-bit offset, SADDR 0x7c for none,
    /// and SVE (bit 55) marking a scratch address in VADDR.
    fn flat_gfx11(&self, w0: u32) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let op = self.opcode(F::Flat, field(w0, 18, 0x7f))?;
        let info = memops::flat(op).ok_or_else(|| self.unknown(F::Flat, op))?;
        let segment = match field(w0, 16, 3) {
            0 => FlatSegment::Flat,
            1 => FlatSegment::Scratch,
            2 => FlatSegment::Global,
            _ => {
                return Err(CompileError::InvalidEncoding {
                    pc: self.pc,
                    word: w0,
                })
            }
        };
        let sve = bit(w1, 23);
        let register = match field(w1, 16, 0x7f) {
            host::GFX11_NULL => None,
            code => Some(self.src(code)?),
        };
        let (saddr, uses_vaddr) = match segment {
            FlatSegment::Scratch if sve => (Some(register.unwrap_or(Operand::Null)), true),
            FlatSegment::Scratch => (register, false),
            _ => (register, !memops::FLAT_ONLY.iter().any(|e| e.op == op)),
        };
        let raw_offset = w0 & 0x1fff;
        let fields = Fields::Flat(Flat {
            segment,
            vdst: field(w1, 24, 0xff) as u8,
            vdata: field(w1, 8, 0xff) as u8,
            vaddr: w1 as u8,
            saddr,
            uses_vaddr,
            offset: if segment == FlatSegment::Flat {
                (raw_offset & 0xfff) as i32
            } else {
                sign_extend(raw_offset, 13)
            },
            glc: bit(w0, 14),
            dlc: bit(w0, 13),
            slc: bit(w0, 15),
            lds: false,
            kind: info.kind,
            data_dwords: info.dwords,
        });
        Ok(self.simple(F::Flat, op, info.flags, fields, 2))
    }

    fn ds(&self, w0: u32) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let op = self.opcode(F::Ds, field(w0, 18, 0xff))?;
        let info = memops::ds_op(op).ok_or_else(|| self.unknown(F::Ds, op))?;
        let gds = bit(w0, 17);
        let writes_addr = info.flags & memops::DS_STACK8 != 0;
        if writes_addr && (gds || bit(w0, 16) || !(8..=31).contains(&(w0 & 0xffff))
            || field(w1, 16, 0xff) > 248 || (w1 as u8) == field(w1, 24, 0xff) as u8)
        {
            return Err(CompileError::InvalidOperand {
                pc: self.pc,
                reason: "Pro push8/pop1 requires LDS, capacity 8..31, a complete eight-register tuple and distinct outputs",
            });
        }
        let mut flags = 0;
        if gds || info.flags & (DS_GWS | DS_GDS_ONLY) != 0 {
            flags |= flag::READS_M0;
        }
        let fields = Fields::Ds(Ds {
            vdst: field(w1, 24, 0xff) as u8,
            addr: w1 as u8,
            data0: field(w1, 8, 0xff) as u8,
            data1: field(w1, 16, 0xff) as u8,
            offset0: w0 as u8,
            offset1: field(w0, 8, 0xff) as u8,
            gds,
            dst_dwords: info.dst,
            data0_dwords: info.data0,
            data1_dwords: info.data1,
            uses_addr: info.flags & DS_NO_ADDR == 0,
            writes_addr,
        });
        Ok(self.simple(F::Ds, op, flags, fields, 2))
    }

    fn mimg(&self, w0: u32) -> Result<Decoded, CompileError> {
        if self.gfx11() {
            return self.mimg_gfx11(w0);
        }
        let w1 = self.word(1)?;
        let op = field(w0, 18, 0x7f) | ((w0 as u16 & 1) << 7);
        let info = memops::image(op).ok_or_else(|| self.unknown(F::Mimg, op))?;
        let nsa_words = field(w0, 1, 3) as usize;
        let dim = field(w0, 3, 7) as u8;
        let a16 = bit(w1, 30);
        let d16 = bit(w1, 31);
        let dmask = field(w0, 8, 0xf) as u8;
        let mut addr = [0u8; MAX_NSA_ADDRESSES];
        addr[0] = w1 as u8;
        for i in 0..nsa_words * 4 {
            addr[1 + i] = (self.word(2 + i / 4)? >> ((i % 4) * 8)) as u8;
        }
        let addr_dwords = memops::image_address_dwords(info.flags, dim, a16);
        let bvh8 = info.flags & img::BVH8 != 0;
        if bvh8 {
            if nsa_words != 2
                || dim != 0
                || a16 || d16
                || !bit(w0, 12) || !bit(w0, 15)
                || dmask != 0xf || bit(w0, 16) || bit(w0, 17)
                || bit(w0, 6) || bit(w0, 7) || bit(w0, 13) || bit(w0, 14) || bit(w0, 25)
                || w1 & 0xffe0_0000 != 0
                || addr[0].checked_add(1) != Some(addr[1])
                || addr[2].checked_add(1) != Some(addr[3])
                || addr[4] > 253 || addr[5] > 253
                || field(w1, 8, 0xff) > 246
            {
                return Err(CompileError::InvalidOperand {
                    pc: self.pc,
                    reason: "BVH8 requires canonical NSA2, complete register groups, DMASK=0xF, R128, UNORM and no cache, coordinate or status modifiers",
                });
            }
        }
        let flags = if info.flags & img::CMPSWAP != 0 {
            flag::CMPSWAP
        } else {
            0
        };
        if !bvh8 && nsa_words > 0 && usize::from(addr_dwords) > 1 + nsa_words * 4 {
            return Err(CompileError::InvalidOperand {
                pc: self.pc,
                reason: "MIMG NSA payload is shorter than the address",
            });
        }
        let fields = Fields::Mimg(Image {
            dim,
            dmask,
            unorm: bit(w0, 12),
            glc: bit(w0, 13),
            slc: bit(w0, 25),
            dlc: bit(w0, 7),
            r128: bit(w0, 15),
            tfe: bit(w0, 16),
            lwe: bit(w0, 17),
            a16,
            d16,
            vdata: field(w1, 8, 0xff) as u8,
            addr,
            nsa: nsa_words > 0,
            nsa_words: nsa_words as u8,
            addr_dwords,
            bvh8,
            srsrc: (field(w1, 16, 0x1f) * 4) as u8,
            ssamp: (field(w1, 21, 0x1f) * 4) as u8,
            uses_sampler: info.uses_sampler(),
            kind: info.kind(),
            data_dwords: memops::image_data_dwords(info.flags, dmask, d16),
        });
        Ok(self.simple(F::Mimg, op, flags, fields, 2 + nsa_words))
    }

    /// gfx11 MIMG: the 8-bit opcode in bits 25:18, dim in 4:2, NSA in bit 0 with one address
    /// word (addresses 1-4), and the sampler in word 1 bits 31:26.
    fn mimg_gfx11(&self, w0: u32) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let op = self.opcode(F::Mimg, field(w0, 18, 0xff))?;
        let info = memops::image(op).ok_or_else(|| self.unknown(F::Mimg, op))?;
        let nsa_words = usize::from(bit(w0, 0));
        let dim = field(w0, 2, 7) as u8;
        let a16 = bit(w0, 16);
        let d16 = bit(w0, 17);
        let dmask = field(w0, 8, 0xf) as u8;
        let mut addr = [0u8; MAX_NSA_ADDRESSES];
        addr[0] = w1 as u8;
        if nsa_words > 0 {
            addr[1..5].copy_from_slice(&self.word(2)?.to_le_bytes());
        }
        if nsa_words > 0 && info.flags & img::BVH != 0 {
            // A BVH intersection names five register groups (four with A16): the node pointer,
            // the ray extent, and the origin, direction and inverse-direction triples.
            let node = if info.flags & img::BVH64 != 0 { 2 } else { 1 };
            let widths = [node, 1, 3, 3, if a16 { 0 } else { 3 }];
            let starts = addr;
            let mut dword = 0;
            for (start, width) in starts.into_iter().zip(widths) {
                for k in 0..width {
                    addr[dword] = start.wrapping_add(k);
                    dword += 1;
                }
            }
        }
        let flags = if info.flags & img::CMPSWAP != 0 {
            flag::CMPSWAP
        } else {
            0
        };
        let fields = Fields::Mimg(Image {
            dim,
            dmask,
            unorm: bit(w0, 7),
            glc: bit(w0, 14),
            slc: bit(w0, 12),
            dlc: bit(w0, 13),
            r128: bit(w0, 15),
            tfe: bit(w1, 21),
            lwe: bit(w1, 22),
            a16,
            d16,
            vdata: field(w1, 8, 0xff) as u8,
            addr,
            nsa: nsa_words > 0,
            nsa_words: nsa_words as u8,
            addr_dwords: memops::image_address_dwords(info.flags, dim, a16),
            bvh8: false,
            srsrc: (field(w1, 16, 0x1f) * 4) as u8,
            ssamp: (field(w1, 26, 0x3f) * 4) as u8,
            uses_sampler: info.uses_sampler(),
            kind: info.kind(),
            data_dwords: memops::image_data_dwords(info.flags, dmask, d16),
        });
        Ok(self.simple(F::Mimg, op, flags, fields, 2 + nsa_words))
    }

    fn exp(&self, w0: u32) -> Result<Decoded, CompileError> {
        let w1 = self.word(1)?;
        let gfx11 = self.gfx11();
        let reserved = if gfx11 {
            EXP_RESERVED_BITS & !(1 << 13)
        } else {
            EXP_RESERVED_BITS
        };
        if w0 & reserved != 0 {
            return Err(CompileError::InvalidEncoding {
                pc: self.pc,
                word: w0,
            });
        }
        let fields = Fields::Exp(Export {
            target: field(w0, 4, 0x3f) as u8,
            en: field(w0, 0, 0xf) as u8,
            compr: !gfx11 && bit(w0, 10),
            done: bit(w0, 11),
            vm: !gfx11 && bit(w0, 12),
            row_en: gfx11 && bit(w0, 13),
            vsrc: w1.to_le_bytes(),
        });
        Ok(self.simple(F::Exp, 0, 0, fields, 2))
    }
}

fn decode_instruction(
    words: &[u32],
    index: usize,
    host: Option<&HostTables>,
) -> Result<Instruction, CompileError> {
    let pc = u32::try_from(index * 4).map_err(|_| CompileError::Truncated { pc: u32::MAX })?;
    let decoded = Ctx {
        words,
        index,
        pc,
        host,
    }
    .decode()?;
    let branch_target = match decoded.branch {
        Some(simm16) => {
            let target = i64::from(pc) + 4 + i64::from(simm16 as i16) * 4;
            let target = u32::try_from(target).map_err(|_| CompileError::InvalidBranch {
                pc,
                target: target as u32,
            })?;
            Some(target)
        }
        None => None,
    };
    Ok(Instruction {
        pc,
        family: decoded.family,
        opcode: decoded.opcode,
        flags: decoded.flags,
        fields: decoded.fields,
        branch_target,
        words: words[index..index + decoded.words].to_vec(),
    })
}

/// Code to decode beyond what static control flow reaches.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DecodeScope {
    /// Byte offsets that computed jumps reach.
    pub entries: BTreeSet<u32>,
    /// An open computed jump may land on any word: decode through every jump and end of
    /// program to the end of the buffer.
    pub through_jumps: bool,
}

/// What [`Decoder::decode_scoped`] keeps, and the branch targets it has seen.
#[derive(Default)]
struct Kept {
    instructions: Vec<Instruction>,
    /// Byte ranges of runs kept although no static control flow reaches them.
    runs: Vec<Range<u32>>,
    targets: BTreeSet<usize>,
    furthest_target: usize,
}

impl Kept {
    fn push(&mut self, inst: Instruction) {
        if let Some(target) = inst.branch_target {
            let index = (target / 4) as usize;
            self.furthest_target = self.furthest_target.max(index);
            self.targets.insert(index);
        }
        self.instructions.push(inst);
    }

    fn keep_run(&mut self, run: Vec<Instruction>) {
        let (Some(first), Some(last)) = (run.first(), run.last()) else {
            return;
        };
        self.runs
            .push(first.pc..last.pc + 4 * last.word_count() as u32);
        for inst in run {
            self.push(inst);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Decoder;

impl Decoder {
    pub fn new() -> Self {
        Self
    }

    pub fn decode_at(&self, words: &[u32], index: usize) -> Result<Instruction, CompileError> {
        decode_instruction(words, index, None)
    }

    pub fn decode(&self, words: &[u32]) -> Result<ShaderIr, CompileError> {
        self.decode_scoped(words, &DecodeScope::default())
    }

    /// Decodes from the first word and past the statically reachable code as `scope` asks.
    /// Words after a transfer that doesn't fall through may be data until a known target, so
    /// a run of them is kept only if it reads as code: it ends in such a transfer or falls into
    /// a target. A run is dropped at an undecodable word, at a branch that leaves the buffer
    /// or misses every instruction, at an instruction that would swallow a known target, and
    /// at the end of the buffer.
    pub fn decode_scoped(
        &self,
        words: &[u32],
        scope: &DecodeScope,
    ) -> Result<ShaderIr, CompileError> {
        if words.is_empty() {
            return Err(CompileError::EmptyProgram);
        }
        let targets: BTreeSet<usize> = scope
            .entries
            .iter()
            .filter(|&&pc| pc % 4 == 0 && ((pc / 4) as usize) < words.len())
            .map(|&pc| (pc / 4) as usize)
            .collect();
        let mut kept = Kept {
            furthest_target: targets.last().copied().unwrap_or(0),
            targets,
            ..Kept::default()
        };
        let mut skipped: Vec<(u32, CompileError)> = Vec::new();
        // The instructions since the last transfer while they may be data; `None` while
        // static control flow reaches them.
        let mut run: Option<Vec<Instruction>> = None;
        let mut index = 0usize;
        let mut end = None;
        while index < words.len() {
            let pc = (index * 4) as u32;
            if kept.targets.contains(&index) {
                if let Some(pending) = run.take() {
                    kept.keep_run(pending);
                }
            }
            let decoded = self
                .decode_at(words, index)
                .and_then(|inst| match inst.branch_target {
                    Some(target) if target % 4 != 0 || (target / 4) as usize >= words.len() => {
                        Err(CompileError::InvalidBranch { pc, target })
                    }
                    _ => Ok(inst),
                });
            let inst = match (decoded, &mut run) {
                (Ok(inst), _) => inst,
                (Err(err), Some(pending)) => {
                    pending.clear();
                    skipped.push((pc, err));
                    index += 1;
                    continue;
                }
                (Err(err), None) => return Err(err),
            };
            if let Some(pending) = &mut run {
                let swallows = index + 1..index + inst.word_count();
                if let Some(&target) = kept.targets.range(swallows).next() {
                    pending.clear();
                    index = target;
                    continue;
                }
            }
            index += inst.word_count();
            let transfers = !inst.falls_through();
            let stops = !scope.through_jumps && transfers;
            match &mut run {
                Some(pending) => {
                    pending.push(inst);
                    if !transfers {
                        continue;
                    }
                    let pending = std::mem::take(pending);
                    kept.keep_run(pending);
                }
                None => {
                    kept.push(inst);
                    if transfers {
                        run = Some(Vec::new());
                    }
                }
            }
            if stops && kept.furthest_target < index {
                end = Some(index);
                break;
            }
        }
        loop {
            let missing = kept.instructions.iter().find_map(|inst| {
                let target = inst.branch_target?;
                let found = kept
                    .instructions
                    .binary_search_by_key(&target, |i| i.pc)
                    .is_ok();
                (!found).then_some((inst.pc, target))
            });
            let Some((pc, target)) = missing else {
                break;
            };
            let Some(run) = kept.runs.iter().position(|r| r.contains(&pc)) else {
                if let Some((_, err)) = skipped.iter().find(|(at, _)| *at == target) {
                    return Err(err.clone());
                }
                return Err(CompileError::InvalidBranch { pc, target });
            };
            let run = kept.runs.swap_remove(run);
            kept.instructions.retain(|i| !run.contains(&i.pc));
        }
        if scope.through_jumps {
            end = kept
                .instructions
                .last()
                .filter(|last| !last.falls_through())
                .map(|last| (last.pc / 4) as usize + last.word_count());
        }
        let code_words = end.ok_or(CompileError::MissingEndProgram)?;
        Ok(ShaderIr {
            instructions: kept.instructions,
            code_words,
            ..ShaderIr::default()
        })
    }
}
