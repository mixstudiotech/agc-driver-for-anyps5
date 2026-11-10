//! GDS accesses as global memory accesses to the runtime's GDS buffer (the ABI is
//! [`crate::gds`]).
//!
//! Every access computes its byte address in a VGPR, wrapped to the GDS size, and goes through
//! the buffer's address in an SGPR pair (`global_* vdst, vaddr, s[buffer]`). `ds_append`,
//! `ds_consume` and `ds_ordered_count` narrow EXEC to the first active lane for their one atomic,
//! then hand its result to every active lane with `v_readfirstlane_b32`. Each access waits for
//! its completion, so the guest's LGKM waits, which do not count global accesses, need not
//! cover it. SCC is kept.

use super::waitcnt::{encode_gfx10, GFX10_EXPCNT_MAX, GFX10_LGKMCNT_MAX};
use super::{unsupported, Ctx};
use crate::gds::GDS_BYTES;
use crate::ir::{
    memops, opcodes, Ds, Fields, Flat, FlatSegment, Instruction, InstructionFamily as F, Operand,
    VopExt,
};
use crate::CompileError;

const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP1_BCNT1_I32_B32: u16 = 0x0f;
const SOP1_BCNT1_I32_B64: u16 = 0x10;
const SOP2_ADD_U32: u16 = 0x00;
const SOP2_SUB_U32: u16 = 0x01;
const SOP2_SUBB_U32: u16 = 0x05;
const SOP2_AND_B32: u16 = 0x0e;
const SOP2_AND_B64: u16 = 0x0f;
const SOP2_LSHR_B32: u16 = 0x20;
const SOPC_CMP_LG_U32: u16 = 0x07;
const SOPP_WAITCNT: u16 = 0x0c;
const VOP1_READFIRSTLANE_B32: u16 = 0x02;
const VOP2_AND_B32: u16 = 0x1b;
const VOP2_ADD_NC_U32: u16 = 0x25;

const GLOBAL_LOAD_DWORD: u16 = 0x0c;
const GLOBAL_STORE_DWORD: u16 = 0x1c;
const GLOBAL_ATOMIC_SWAP: u16 = 0x30;
const GLOBAL_ATOMIC_CMPSWAP: u16 = 0x31;
const GLOBAL_ATOMIC_ADD: u16 = 0x32;
const GLOBAL_ATOMIC_SUB: u16 = 0x33;

/// Why every `ds_gws_*` instruction is rejected on a host target, with or without a GDS buffer.
pub(super) const GWS_NOT_EMULATED: &str =
    "GWS is not emulated: ds_gws_* barriers and semaphores across workgroups are rejected";

const DS_CONSUME: u16 = 0x3d;
const DS_APPEND: u16 = 0x3e;
const DS_ORDERED_COUNT: u16 = 0x3f;
/// `ds_ordered_count` `offset1`: swap instead of add, and the dword count less one.
const ORDERED_SWAP: u8 = 0x10;
const ORDERED_DWORDS_SHIFT: u8 = 6;

/// What a GDS instruction becomes.
#[derive(Clone, Copy)]
enum Access {
    Load(u16),
    Store(u16),
    Atomic {
        op: u16,
        rtn: bool,
    },
    CmpSwap {
        rtn: bool,
    },
    /// `ds_read2*_b32`: two dword loads, offsets scaled by `stride`.
    Read2 {
        stride: u32,
    },
    Write2 {
        stride: u32,
    },
    Append {
        subtract: bool,
    },
    Ordered,
    Nop,
}

/// The global atomic of each 32-bit integer DS atomic, `ds_add_u32` to `ds_xor_b32`.
const ATOMICS: [Option<u16>; 12] = [
    Some(GLOBAL_ATOMIC_ADD),
    Some(GLOBAL_ATOMIC_SUB),
    None,
    Some(0x3c),
    Some(0x3d),
    Some(0x35),
    Some(0x37),
    Some(0x36),
    Some(0x38),
    Some(0x39),
    Some(0x3a),
    Some(0x3b),
];

fn access(op: u16) -> Result<Access, &'static str> {
    use Access::*;
    let atomic = |index: u16, rtn| {
        ATOMICS[usize::from(index)]
            .map(|op| Atomic { op, rtn })
            .ok_or("GDS reverse subtracts are not emulated")
    };
    Ok(match op {
        0x00..=0x0b => atomic(op, false)?,
        0x20..=0x2b => atomic(op - 0x20, true)?,
        0x2d => Atomic {
            op: GLOBAL_ATOMIC_SWAP,
            rtn: true,
        },
        0x10 => CmpSwap { rtn: false },
        0x30 => CmpSwap { rtn: true },
        0x0d => Store(GLOBAL_STORE_DWORD),
        0x1e => Store(0x18),
        0x1f => Store(0x1a),
        0x4d => Store(0x1d),
        0xa0 => Store(0x19),
        0xa1 => Store(0x1b),
        0xde => Store(0x1f),
        0xdf => Store(0x1e),
        0x36 => Load(GLOBAL_LOAD_DWORD),
        0x39 => Load(0x09),
        0x3a => Load(0x08),
        0x3b => Load(0x0b),
        0x3c => Load(0x0a),
        0x76 => Load(0x0d),
        0xa2 => Load(0x20),
        0xa3 => Load(0x21),
        0xa4 => Load(0x22),
        0xa5 => Load(0x23),
        0xa6 => Load(0x24),
        0xa7 => Load(0x25),
        0xfe => Load(0x0f),
        0xff => Load(0x0e),
        0x0e => Write2 { stride: 4 },
        0x0f => Write2 { stride: 256 },
        0x37 => Read2 { stride: 4 },
        0x38 => Read2 { stride: 256 },
        DS_CONSUME => Append { subtract: true },
        DS_APPEND => Append { subtract: false },
        DS_ORDERED_COUNT => Ordered,
        0x14 => Nop,
        0x18..=0x1d => return Err(GWS_NOT_EMULATED),
        0x11..=0x13 | 0x15 | 0x31..=0x33 | 0x55 => {
            return Err("float GDS atomics are not emulated")
        }
        0x40..=0x4c | 0x50..=0x53 | 0x60..=0x73 => {
            return Err("64-bit GDS atomics are not emulated")
        }
        _ => return Err("this GDS operation is not emulated"),
    })
}

/// Lowers a GDS instruction to accesses of the GDS buffer at `buffer`.
pub(super) fn lower(
    ctx: &mut Ctx<'_>,
    inst: &Instruction,
    buffer: u64,
) -> Result<(), CompileError> {
    let Fields::Ds(d) = inst.fields else {
        return Err(unsupported(inst, ctx.target, "malformed DS"));
    };
    let access = access(inst.opcode).map_err(|reason| unsupported(inst, ctx.target, reason))?;
    if !d.gds && !matches!(access, Access::Nop) {
        return Err(unsupported(
            inst,
            ctx.target,
            "LDS ordered counts are not emulated",
        ));
    }
    let offset = u32::from(d.offset0) | (u32::from(d.offset1) << 8);
    let mut e = Emit {
        ctx,
        origin: inst.clone(),
        buffer,
    };
    match access {
        Access::Nop => Ok(()),
        Access::Load(op) => {
            let v = e.vgprs(1)?;
            e.address(&d, offset, v)?;
            e.global(op, d.vdst, v, 0, false)
        }
        Access::Store(op) => {
            let v = e.vgprs(1)?;
            e.address(&d, offset, v)?;
            e.global(op, 0, v, d.data0, false)
        }
        Access::Atomic { op, rtn } => {
            let v = e.vgprs(1)?;
            e.address(&d, offset, v)?;
            e.global(op, d.vdst, v, d.data0, rtn)
        }
        Access::CmpSwap { rtn } => {
            // DS compares with DATA0 and stores DATA1; the global form takes {new, compare}.
            let v = e.vgprs(3)?;
            e.address(&d, offset, v)?;
            e.vmov(v + 1, Operand::Vgpr(d.data1))?;
            e.vmov(v + 2, Operand::Vgpr(d.data0))?;
            e.global(GLOBAL_ATOMIC_CMPSWAP, d.vdst, v, v + 1, rtn)
        }
        Access::Read2 { stride } => {
            let v = e.vgprs(1)?;
            for (k, part) in [d.offset0, d.offset1].into_iter().enumerate() {
                e.address(&d, u32::from(part) * stride, v)?;
                e.global(GLOBAL_LOAD_DWORD, d.vdst + k as u8, v, 0, false)?;
            }
            Ok(())
        }
        Access::Write2 { stride } => {
            let v = e.vgprs(1)?;
            for (part, data) in [(d.offset0, d.data0), (d.offset1, d.data1)] {
                e.address(&d, u32::from(part) * stride, v)?;
                e.global(GLOBAL_STORE_DWORD, 0, v, data, false)?;
            }
            Ok(())
        }
        Access::Append { subtract } => e.first_lane(&d, Some((offset, subtract))),
        Access::Ordered => {
            if d.offset1 >> ORDERED_DWORDS_SHIFT != 0 {
                return Err(unsupported(
                    inst,
                    e.ctx.target,
                    "GDS ordered counts of more than one dword are not emulated",
                ));
            }
            e.first_lane(&d, None)
        }
    }
}

struct Emit<'c, 'r> {
    ctx: &'c mut Ctx<'r>,
    origin: Instruction,
    buffer: u64,
}

impl Emit<'_, '_> {
    fn op(&mut self, family: F, opcode: u16, fields: Fields) -> Result<(), CompileError> {
        self.ctx.lower_gfx1013(&self.origin, family, opcode, fields)
    }

    fn vgprs(&mut self, count: u8) -> Result<u8, CompileError> {
        self.ctx.regs.scratch_vgprs(count)
    }

    fn sop1(&mut self, opcode: u16, sdst: Operand, src0: Operand) -> Result<(), CompileError> {
        self.op(F::Sop1, opcode, Fields::Sop1 { sdst, src0 })
    }

    fn sop2(
        &mut self,
        opcode: u16,
        sdst: Operand,
        src0: Operand,
        src1: Operand,
    ) -> Result<(), CompileError> {
        self.op(F::Sop2, opcode, Fields::Sop2 { sdst, src0, src1 })
    }

    fn vop1(&mut self, opcode: u16, vdst: Operand, src0: Operand) -> Result<(), CompileError> {
        let fields = Fields::Vop1 {
            vdst,
            src0,
            ext: VopExt::None,
        };
        self.op(F::Vop1, opcode, fields)
    }

    fn vmov(&mut self, vdst: u8, src0: Operand) -> Result<(), CompileError> {
        self.vop1(opcodes::VOP1_MOV_B32, Operand::Vgpr(vdst), src0)
    }

    fn vop2(
        &mut self,
        opcode: u16,
        vdst: u8,
        src0: Operand,
        vsrc1: u8,
    ) -> Result<(), CompileError> {
        let fields = Fields::Vop2 {
            vdst: Operand::Vgpr(vdst),
            src0,
            vsrc1: Operand::Vgpr(vsrc1),
            constant: None,
            ext: VopExt::None,
        };
        self.op(F::Vop2, opcode, fields)
    }

    /// `v` = the GDS byte address of `d` plus `offset`, wrapped to the GDS size.
    fn address(&mut self, d: &Ds, offset: u32, v: u8) -> Result<(), CompileError> {
        match (d.uses_addr, offset) {
            (true, 0) => self.vmov(v, Operand::Vgpr(d.addr))?,
            (true, _) => self.vop2(VOP2_ADD_NC_U32, v, Operand::constant(offset), d.addr)?,
            (false, _) => self.vmov(v, Operand::constant(offset))?,
        }
        self.vop2(VOP2_AND_B32, v, Operand::constant(GDS_BYTES - 1), v)
    }

    /// The buffer's address in a transient SGPR pair, the first of `count` transient SGPRs.
    fn buffer_sgprs(&mut self, count: u8) -> Result<u8, CompileError> {
        let s = self.ctx.transient_sgprs(&self.origin, count)?;
        let address = self.buffer;
        self.sop1(
            SOP1_MOV_B32,
            Operand::Sgpr(s),
            Operand::constant(address as u32),
        )?;
        self.sop1(
            SOP1_MOV_B32,
            Operand::Sgpr(s + 1),
            Operand::constant((address >> 32) as u32),
        )?;
        Ok(s)
    }

    /// One global access at the byte address in `vaddr`, then a wait for its completion.
    /// Atomics return the previous value when `rtn` is set; loads bypass the vector caches.
    fn global(
        &mut self,
        opcode: u16,
        vdst: u8,
        vaddr: u8,
        vdata: u8,
        rtn: bool,
    ) -> Result<(), CompileError> {
        let s = self.buffer_sgprs(2)?;
        self.global_at(s, opcode, vdst, vaddr, vdata, rtn)
    }

    fn global_at(
        &mut self,
        buffer: u8,
        opcode: u16,
        vdst: u8,
        vaddr: u8,
        vdata: u8,
        rtn: bool,
    ) -> Result<(), CompileError> {
        let info = memops::flat(opcode)
            .ok_or_else(|| unsupported(&self.origin, self.ctx.target, "no global form"))?;
        let load = info.kind == crate::ir::MemKind::Load;
        let fields = Fields::Flat(Flat {
            segment: FlatSegment::Global,
            vdst,
            vdata,
            vaddr,
            saddr: Some(Operand::Sgpr(buffer)),
            uses_vaddr: true,
            offset: 0,
            glc: load || rtn,
            dlc: load,
            slc: false,
            lds: false,
            kind: info.kind,
            data_dwords: info.dwords,
        });
        self.op(F::Flat, opcode, fields)?;
        if load || rtn {
            let simm16 = encode_gfx10(0, GFX10_EXPCNT_MAX, GFX10_LGKMCNT_MAX);
            self.op(F::Sopp, SOPP_WAITCNT, Fields::Sopp { simm16 })
        } else {
            self.ctx.wait_for_stores(&self.origin)
        }
    }

    /// `ds_append`/`ds_consume` (`counter`: offset, subtract) or `ds_ordered_count`: one atomic
    /// from the first active lane, whose result every active lane gets.
    fn first_lane(&mut self, d: &Ds, counter: Option<(u32, bool)>) -> Result<(), CompileError> {
        let wave64 = self.ctx.wave64;
        // s: buffer (2), saved EXEC (2), first-lane mask (2), lane count, address, SCC, result.
        let s = self.buffer_sgprs(10)?;
        let (saved, mask, count, address, scc, result) = (s + 2, s + 4, s + 6, s + 7, s + 8, s + 9);
        let v = self.vgprs(3)?;
        let (vaddr, vdata, vresult) = (v, v + 1, v + 2);
        self.ctx.scalar_select(&self.origin, Operand::Sgpr(scc))?;
        let (opcode, data) = match counter {
            Some((offset, subtract)) => {
                let bcnt = if wave64 {
                    SOP1_BCNT1_I32_B64
                } else {
                    SOP1_BCNT1_I32_B32
                };
                self.sop1(bcnt, Operand::Sgpr(count), Operand::ExecLo)?;
                self.sop2(
                    SOP2_LSHR_B32,
                    Operand::Sgpr(address),
                    Operand::M0,
                    Operand::InlineInt(16),
                )?;
                self.sop2(
                    SOP2_ADD_U32,
                    Operand::Sgpr(address),
                    Operand::Sgpr(address),
                    Operand::constant(offset),
                )?;
                self.sop2(
                    SOP2_AND_B32,
                    Operand::Sgpr(address),
                    Operand::Sgpr(address),
                    Operand::constant(GDS_BYTES - 1),
                )?;
                self.vmov(vaddr, Operand::Sgpr(address))?;
                self.vmov(vdata, Operand::Sgpr(count))?;
                let opcode = if subtract {
                    GLOBAL_ATOMIC_SUB
                } else {
                    GLOBAL_ATOMIC_ADD
                };
                (opcode, vdata)
            }
            None => {
                self.vmov(vaddr, Operand::constant(u32::from(d.offset0)))?;
                let opcode = if d.offset1 & ORDERED_SWAP != 0 {
                    GLOBAL_ATOMIC_SWAP
                } else {
                    GLOBAL_ATOMIC_ADD
                };
                (opcode, d.addr)
            }
        };
        // EXEC & -EXEC: the first active lane alone, or none.
        if wave64 {
            self.sop1(SOP1_MOV_B64, Operand::Sgpr(saved), Operand::ExecLo)?;
            self.sop2(
                SOP2_SUB_U32,
                Operand::Sgpr(mask),
                Operand::InlineInt(0),
                Operand::ExecLo,
            )?;
            self.sop2(
                SOP2_SUBB_U32,
                Operand::Sgpr(mask + 1),
                Operand::InlineInt(0),
                Operand::ExecHi,
            )?;
            self.sop2(
                SOP2_AND_B64,
                Operand::ExecLo,
                Operand::ExecLo,
                Operand::Sgpr(mask),
            )?;
        } else {
            self.sop1(SOP1_MOV_B32, Operand::Sgpr(saved), Operand::ExecLo)?;
            self.sop2(
                SOP2_SUB_U32,
                Operand::Sgpr(mask),
                Operand::InlineInt(0),
                Operand::ExecLo,
            )?;
            self.sop2(
                SOP2_AND_B32,
                Operand::ExecLo,
                Operand::ExecLo,
                Operand::Sgpr(mask),
            )?;
        }
        self.global_at(s, opcode, vresult, vaddr, data, true)?;
        let restore = if wave64 { SOP1_MOV_B64 } else { SOP1_MOV_B32 };
        self.sop1(restore, Operand::ExecLo, Operand::Sgpr(saved))?;
        self.vop1(
            VOP1_READFIRSTLANE_B32,
            Operand::Sgpr(result),
            Operand::Vgpr(vresult),
        )?;
        self.vmov(d.vdst, Operand::Sgpr(result))?;
        let compare = Fields::Sopc {
            src0: Operand::Sgpr(scc),
            src1: Operand::InlineInt(0),
        };
        self.op(F::Sopc, SOPC_CMP_LG_U32, compare)
    }
}
