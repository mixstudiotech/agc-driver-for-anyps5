//! Decoding translated code back into the gfx1013 instructions it implements.
//!
//! Each target opcode maps back through the translation tables (`Xlat::Op` and the rules that
//! keep their instruction); the field layouts and operand codes are the target's. Instructions
//! the translation emits without a gfx1013 counterpart map to their gfx1013 equivalent
//! (`s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME` is `s_memrealtime`). gfx10.3 and gfx11/gfx11.5
//! code decodes; gfx12 does not yet.

use super::decode_instruction;
use crate::ir::memops::{self, img, MemOp};
use crate::ir::opcodes::{self, flag, OpInfo, Rule, VopcInfo, Xlat};
use crate::ir::{InstructionFamily as F, ShaderIr, ShaderMetadata};
use crate::{CompileError, ShaderStage, TargetIsa};
use std::collections::HashMap;

pub(super) const GFX11_NULL: u16 = 0x7c;
pub(super) const GFX11_M0: u16 = 0x7d;
pub(super) const GFX11_SENDMSG_RTN_B64: u16 = 0x4d;
pub(super) const MSG_RTN_GET_REALTIME: u16 = 0x83;
pub(super) const SMEM_MEMREALTIME: u16 = 0x25;
const SOPP_GFX11_WAITCNT: u16 = 0x09;

/// Both sources of a gfx11 mixed-signedness dot (`v_dot4_i32_iu8`, `v_dot8_i32_iu4`) signed:
/// the signed gfx10 dot product.
pub(super) const SIGNED_DOT_SOURCES: u8 = 0b011;

/// The gfx1013 opcode behind each target opcode.
#[derive(Clone, Debug)]
pub struct HostTables {
    target: TargetIsa,
    opcodes: HashMap<(F, u16), u16>,
    vopc: HashMap<u16, u16>,
    formats: HashMap<u8, u8>,
    signed_dots: Vec<u16>,
}

impl HostTables {
    pub fn new(target: TargetIsa) -> Self {
        let mut tables = Self {
            target,
            opcodes: HashMap::new(),
            vopc: HashMap::new(),
            formats: HashMap::new(),
            signed_dots: Vec::new(),
        };
        tables.add_op_tables();
        tables.add_memory_tables();
        if target.is_gfx11_plus() {
            tables.insert(F::Sopp, SOPP_GFX11_WAITCNT, opcodes::SOPP_WAITCNT);
        }
        for op in 0..=0xff {
            if let Some(info) = VopcInfo::decode(op) {
                let host = if target.is_gfx11_plus() {
                    info.gfx11_opcode()
                } else {
                    op
                };
                tables.vopc.entry(host).or_insert(op);
            }
        }
        for format in 0..=0x7f {
            if let Some(host) = memops::remap_buffer_format(format, target) {
                tables.formats.entry(host).or_insert(format);
            }
        }
        tables
    }

    pub fn target(&self) -> TargetIsa {
        self.target
    }

    /// The gfx1013 opcode of a target instruction of `family` with opcode field `host`.
    pub fn guest_opcode(&self, family: F, host: u16) -> Option<u16> {
        match family {
            F::Vopc => self.vopc.get(&host).copied(),
            F::Vop3 if host <= opcodes::VOP3_VOPC_LAST => self.vopc.get(&host).copied(),
            _ => self.opcodes.get(&(family, host)).copied(),
        }
    }

    /// The gfx10 MTBUF format of a target format.
    pub fn guest_format(&self, host: u8) -> Option<u8> {
        self.formats.get(&host).copied()
    }

    /// Whether VOP3P `host` is a mixed-signedness dot the translation emits for a signed one.
    pub fn is_signed_dot(&self, host: u16) -> bool {
        self.signed_dots.contains(&host)
    }

    /// Keeps the first gfx1013 instruction for a target opcode, unless a later one is the
    /// identity (gfx10.3 keeps most opcodes).
    fn insert(&mut self, family: F, host: u16, guest: u16) {
        let entry = self.opcodes.entry((family, host)).or_insert(guest);
        if guest == host {
            *entry = guest;
        }
    }

    fn add_op_tables(&mut self) {
        let tables: [(F, &[OpInfo]); 9] = [
            (F::Sop1, opcodes::SOP1),
            (F::Sop2, opcodes::SOP2),
            (F::Sopk, opcodes::SOPK),
            (F::Sopc, opcodes::SOPC),
            (F::Sopp, opcodes::SOPP),
            (F::Vop1, opcodes::VOP1),
            (F::Vop2, opcodes::VOP2),
            (F::Vop3, opcodes::VOP3),
            (F::Vop3p, opcodes::VOP3P),
        ];
        for (family, table) in tables {
            for info in table {
                let host = match info.translate(self.target) {
                    Xlat::Op(op) | Xlat::Rule(Rule::Hwreg(op) | Rule::Sendmsg(op)) => op,
                    Xlat::Rule(Rule::DotSigned(op)) => {
                        self.signed_dots.push(op);
                        op
                    }
                    _ => continue,
                };
                self.insert(family, host, info.op);
                let promoted = match family {
                    F::Vop1 => opcodes::VOP3_VOP1_BASE,
                    F::Vop2 => opcodes::VOP3_VOP2_BASE,
                    _ => continue,
                };
                if info.flags & flag::NO_VOP3 == 0 {
                    self.insert(F::Vop3, promoted + host, promoted + info.op);
                }
            }
        }
    }

    fn add_memory_tables(&mut self) {
        let target = self.target;
        let add = |tables: &mut Self, family: F, ops: &mut dyn Iterator<Item = &MemOp>| {
            for op in ops {
                if let Xlat::Op(host) = op.translate(target) {
                    tables.insert(family, host, op.op);
                }
            }
        };
        add(self, F::Smem, &mut memops::SMEM.iter());
        add(
            self,
            F::Mubuf,
            &mut memops::VMEM_COMMON.iter().chain(memops::MUBUF_ONLY),
        );
        add(self, F::Mtbuf, &mut memops::MTBUF.iter());
        add(
            self,
            F::Flat,
            &mut memops::VMEM_COMMON.iter().chain(memops::FLAT_ONLY),
        );
        for op in memops::DS {
            if let Xlat::Op(host) = op.translate(target) {
                self.insert(F::Ds, host, op.op);
            }
        }
        for op in memops::MIMG.iter().filter(|op| op.flags & img::ADJUST == 0) {
            if let Xlat::Op(host) = op.translate(target) {
                self.insert(F::Mimg, host, op.op);
            }
        }
    }
}

/// Decodes the program words of a translation (`ShaderArtifact::program`).
#[derive(Clone, Debug)]
pub struct HostDecoder {
    tables: HostTables,
}

impl HostDecoder {
    pub fn new(target: TargetIsa) -> Result<Self, CompileError> {
        if matches!(target, TargetIsa::Gfx1013 | TargetIsa::Gfx12) {
            return Err(CompileError::InvalidOptions(
                "host decoding covers gfx10.3, gfx11 and gfx11.5",
            ));
        }
        Ok(Self {
            tables: HostTables::new(target),
        })
    }

    /// Decodes every instruction of `words`, a translated program of `stage`.
    pub fn decode(&self, words: &[u32], stage: ShaderStage) -> Result<ShaderIr, CompileError> {
        let mut instructions = Vec::new();
        let mut index = 0;
        while index < words.len() {
            let inst = decode_instruction(words, index, Some(&self.tables))?;
            index += inst.word_count();
            instructions.push(inst);
        }
        for inst in &instructions {
            if let Some(target) = inst.branch_target {
                if instructions
                    .binary_search_by_key(&target, |i| i.pc)
                    .is_err()
                {
                    return Err(CompileError::InvalidBranch {
                        pc: inst.pc,
                        target,
                    });
                }
            }
        }
        Ok(ShaderIr {
            instructions,
            code_words: words.len(),
            metadata: ShaderMetadata {
                stage: Some(stage),
                ..ShaderMetadata::default()
            },
            ..ShaderIr::default()
        })
    }
}
