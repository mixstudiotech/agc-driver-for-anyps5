//! PC-relative code and computed control flow.
//!
//! The translated program hands the guest only guest PCs. Guest memory stays mapped at the same
//! addresses on the GPU, so data that a program reads relative to its own PC (constants after
//! `s_endpgm`, jump tables) is read from the original guest bytes:
//!
//! - `s_getpc_b64` becomes two `s_mov_b32` of the next instruction's guest address.
//! - `s_call_b64` and `s_swappc_b64` write the guest return address.
//! - A computed jump (`s_setpc_b64`, `s_swappc_b64`) whose target `cfg::computed` proves
//!   constant becomes `s_branch` to the translated target.
//! - Any other computed jump copies its target to the dispatch SGPRs and branches to the
//!   program's dispatch stub, which maps the guest address to translated code through a table
//!   of one word per guest code word, placed after the program by the encoder. Addresses the
//!   table does not map (unlinked code of other programs, data, unmapped words) end the wave
//!   at the miss stub.
//!
//! The stub (gfx1013 opcodes; `x` is the first dispatch SGPR, T the guest target in
//! `x[0:1]`, B the guest base, N the guest code bytes) takes four SGPRs, so that a program
//! that uses nearly all of them (a bytecode interpreter dispatching through a jump table) can
//! still have it. The full 64-bit target is checked before its high dword is reused. SCC is
//! saved in x2, moved to x1 before the PC anchor overwrites x2, then carried in bit 0 of the
//! dword-aligned anchor, as ACO's long jump keeps it (`emit_long_jump`):
//!
//! ```text
//!     s_cselect_b32 x2, 1, 0         ; save SCC
//!     s_sub_u32     x0, x0, lo(B)
//!     s_subb_u32    x1, x1, hi(B)     ; x1:x0 = T - B
//!     s_cmp_eq_u32  x1, 0
//!     s_cselect_b32 x0, x0, N         ; byte index, N selects the miss entry
//!     s_and_b32     x1, x0, 3
//!     s_cselect_b32 x0, N, x0         ; unaligned targets miss
//!     s_min_u32     x0, x0, N         ; so do targets past the code
//!     s_mov_b32     x1, x2            ; saved SCC
//!     s_getpc_b64   x[2:3]            ; G, the table's anchor
//!     s_load_dword  x0, x[2:3], x0 offset:table-G (patched by the encoder)
//!     s_waitcnt     lgkmcnt(0)
//!     s_or_b32      x2, x2, x1        ; SCC in bit 0 of G
//!     s_ashr_i32    x1, x0, 31
//!     s_add_u32     x2, x2, x0
//!     s_addc_u32    x3, x3, x1        ; G + entry
//!     s_bitcmp1_b32 x2, 0             ; restore SCC
//!     s_bitset0_b32 x2, 0
//!     s_setpc_b64   x[2:3]
//! ```
//!
//! Table entries are byte offsets from G, so the code needs no relocation when loaded.
//!
//! With linked regions ([`crate::Linkage`]) the program's pcs run through the shader's words
//! and then each region's, and the table covers them all. That stub takes eight SGPRs (`y`):
//! it keeps SCC in y6 and G in `y[4:5]`, and finds the byte index span by span (y2 starts at
//! N; A is a span's guest address, S its first pc and L its bytes), so no span may cross a
//! 4 GiB boundary:
//!
//! ```text
//!     s_getpc_b64   y[4:5]
//!     s_cselect_b32 y6, 1, 0
//!     s_mov_b32     y2, N
//!   per span:
//!     s_sub_u32     y3, y0, lo(A)
//!     s_add_u32     y7, y3, S
//!     s_cmp_lt_u32  y3, L
//!     s_cselect_b32 y7, y7, y2
//!     s_cmp_eq_u32  y1, hi(A)
//!     s_cselect_b32 y2, y7, y2        ; the pc of T when it lies in this span
//!   then:
//!     s_and_b32     y3, y2, 3
//!     s_cselect_b32 y2, N, y2
//!     s_load_dword  y3, y[4:5], y2 offset:table-G
//!     s_waitcnt     lgkmcnt(0)
//!     s_ashr_i32    y7, y3, 31
//!     s_add_u32     y4, y4, y3
//!     s_addc_u32    y5, y5, y7
//!     s_cmp_lg_u32  y6, 0
//!     s_setpc_b64   y[4:5]
//! ```

use super::waitcnt::encode_gfx10;
use super::{derive, unsupported, Ctx};
use crate::cfg::computed::{SOP1_GETPC_B64, SOP1_SETPC_B64, SOP1_SWAPPC_B64, SOPK_CALL_B64};
use crate::ir::opcodes::{self, flag, Xlat};
use crate::ir::{Fields, Instruction, InstructionFamily as F, JumpTarget, Operand, Smem};
use crate::CompileError;

/// Block label of the dispatch stub; guest pcs, offsets into the program, stay far below it.
pub const DISPATCH_LABEL: u32 = 0xffff_fff0;
/// Block label of the miss stub.
pub const MISS_LABEL: u32 = 0xffff_fff4;
/// SGPRs the dispatch sequence uses.
pub const DISPATCH_SGPRS: u8 = 4;
/// SGPRs the dispatch sequence of a program with linked regions uses.
pub const LINKED_DISPATCH_SGPRS: u8 = 8;

const SOP1_MOV_B32: u16 = 0x03;
const SOP1_MOV_B64: u16 = 0x04;
const SOP2_ADD_U32: u16 = 0x00;
const SOP2_SUB_U32: u16 = 0x01;
const SOP2_ADDC_U32: u16 = 0x04;
const SOP2_SUBB_U32: u16 = 0x05;
const SOP2_CSELECT_B32: u16 = 0x0a;
const SOP2_AND_B32: u16 = 0x0e;
const SOP2_ASHR_I32: u16 = 0x22;
const SOP2_MIN_U32: u16 = 0x07;
const SOP2_OR_B32: u16 = 0x10;
const SOP1_BITSET0_B32: u16 = 0x1b;
const SOPC_BITCMP1_B32: u16 = 0x0d;
const SOPC_CMP_EQ_U32: u16 = 0x06;
const SOPC_CMP_LG_U32: u16 = 0x07;
const SOPC_CMP_LT_U32: u16 = 0x0a;
const SOPP_ENDPGM: u16 = 0x01;
const SOPP_BRANCH: u16 = 0x02;
const SOPP_WAITCNT: u16 = 0x0c;
const SMEM_LOAD_DWORD: u16 = 0x00;
const SECOND_HALF_SGPR: u8 = 6;

impl Ctx<'_> {
    /// Lowers the instructions that read or write PCs; `None` for every other instruction.
    pub(super) fn lower_pc_relative(
        &mut self,
        inst: &Instruction,
    ) -> Option<Result<(), CompileError>> {
        match (inst.family, inst.opcode) {
            (F::Sop1, SOP1_GETPC_B64) => Some(self.getpc(inst)),
            (F::Sop1, SOP1_SETPC_B64 | SOP1_SWAPPC_B64) => Some(self.computed_jump(inst)),
            (F::Sopk, SOPK_CALL_B64) => Some(self.static_call(inst)),
            _ => None,
        }
    }

    fn getpc(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let Fields::Sop1 { sdst, .. } = inst.fields else {
            return Err(unsupported(inst, self.target, "malformed s_getpc_b64"));
        };
        if self.guest_base.is_none() {
            return Err(unsupported(
                inst,
                self.target,
                "the guest code address is unknown (ShaderInterface::guest_code_address)",
            ));
        }
        self.write_address(inst, sdst, self.return_address(inst))
    }

    fn static_call(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let (Fields::Sopk { sdst, .. }, Some(callee)) = (inst.fields, inst.branch_target) else {
            return Err(unsupported(inst, self.target, "malformed s_call_b64"));
        };
        self.write_address(inst, sdst, self.return_address(inst))?;
        self.branch(inst, callee)
    }

    fn computed_jump(&mut self, inst: &Instruction) -> Result<(), CompileError> {
        let Fields::Sop1 { sdst, src0 } = inst.fields else {
            return Err(unsupported(inst, self.target, "malformed computed jump"));
        };
        let destination = match self.flow.jumps.get(&inst.pc) {
            Some(JumpTarget::Static(target)) => *target,
            _ => {
                let x = self.dispatch.ok_or_else(|| {
                    unsupported(
                        inst,
                        self.target,
                        "computed jumps need SGPRs above the shader's own",
                    )
                })?;
                let copy = Fields::Sop1 {
                    sdst: Operand::Sgpr(x),
                    src0,
                };
                self.lower_gfx1013(inst, F::Sop1, SOP1_MOV_B64, copy)?;
                DISPATCH_LABEL
            }
        };
        if inst.has_flag(flag::CALL) {
            self.write_address(inst, sdst, self.return_address(inst))?;
        }
        self.branch(inst, destination)
    }

    /// PS5 hardware gives a merged-stage front half its back half's address in `s[6:7]`; gfx11
    /// leaves them undefined. The tail jump through them branches to the linked back half, and
    /// guest code that reads them otherwise gets the PS5 value.
    pub(super) fn set_second_half(
        &mut self,
        origin: &Instruction,
        live_in: u128,
    ) -> Result<(), CompileError> {
        match self.layout.second_half {
            Some(address) if live_in & (0b11 << SECOND_HALF_SGPR) != 0 => {
                self.write_address(origin, Operand::Sgpr(SECOND_HALF_SGPR), address)
            }
            _ => Ok(()),
        }
    }

    /// The guest address after a one-word instruction (a call's return address, what
    /// `s_getpc_b64` reads): in guest terms when the code address is known, and counted from
    /// zero otherwise (the dispatch subtracts the same base). Code in a linked region has the
    /// region's addresses.
    fn return_address(&self, inst: &Instruction) -> u64 {
        self.layout.guest_address(next_pc(inst) as u32)
    }

    fn write_address(
        &mut self,
        origin: &Instruction,
        sdst: Operand,
        address: u64,
    ) -> Result<(), CompileError> {
        for (k, half) in [(0, address as u32), (1, (address >> 32) as u32)] {
            let fields = Fields::Sop1 {
                sdst: high_part(sdst, k),
                src0: Operand::constant(half),
            };
            self.lower_gfx1013(origin, F::Sop1, SOP1_MOV_B32, fields)?;
        }
        Ok(())
    }

    pub(super) fn branch(&mut self, origin: &Instruction, to: u32) -> Result<(), CompileError> {
        let mut branch = derive(
            origin,
            F::Sopp,
            SOPP_BRANCH,
            flag::BRANCH,
            Fields::Sopp { simm16: 0 },
        );
        branch.branch_target = Some(to);
        self.lower(&branch)
    }

    /// The dispatch stub; see the module documentation. `code_bytes` is the size of the guest
    /// code the table maps.
    pub(super) fn dispatch_stub(
        &mut self,
        origin: &Instruction,
        code_bytes: u32,
    ) -> Result<(), CompileError> {
        let x = self.dispatch.expect("the dispatch SGPRs are allocated");
        let s = |k: u8| Operand::Sgpr(x + k);
        let base = self.guest_base.unwrap_or(0);
        let n = Operand::constant(code_bytes);
        let sop2 = |op, d, a, b| {
            (
                F::Sop2,
                op,
                Fields::Sop2 {
                    sdst: d,
                    src0: a,
                    src1: b,
                },
            )
        };
        let sopc = |op, a, b| (F::Sopc, op, Fields::Sopc { src0: a, src1: b });
        let one = Operand::InlineInt(1);
        let zero = Operand::InlineInt(0);
        if self.layout.regions.is_empty() {
            return self.narrow_dispatch_stub(origin, x, base, n);
        }
        self.host_sop1(origin, SOP1_GETPC_B64, s(4), Operand::Raw(0))?;
        // Linked code: each span of guest code (the shader's words, then every region) maps
        // T into the table when T has the span's high dword and its low dword falls in the
        // span (no span crosses a 4 GiB boundary). y2 starts at the miss entry.
        let mut before_load = vec![
            sop2(SOP2_CSELECT_B32, s(6), one, zero),
            (
                F::Sop1,
                SOP1_MOV_B32,
                Fields::Sop1 {
                    sdst: s(2),
                    src0: n,
                },
            ),
        ];
        let own = (base, 0, self.layout.own_bytes);
        let spans = self
            .layout
            .regions
            .iter()
            .map(|r| (r.address, r.start, r.bytes));
        for (address, start, bytes) in std::iter::once(own).chain(spans) {
            before_load.extend([
                sop2(SOP2_SUB_U32, s(3), s(0), Operand::constant(address as u32)),
                sop2(SOP2_ADD_U32, s(7), s(3), Operand::constant(start)),
                sopc(SOPC_CMP_LT_U32, s(3), Operand::constant(bytes)),
                sop2(SOP2_CSELECT_B32, s(7), s(7), s(2)),
                sopc(
                    SOPC_CMP_EQ_U32,
                    s(1),
                    Operand::constant((address >> 32) as u32),
                ),
                sop2(SOP2_CSELECT_B32, s(2), s(7), s(2)),
            ]);
        }
        before_load.extend([
            sop2(SOP2_AND_B32, s(3), s(2), Operand::InlineInt(3)),
            sop2(SOP2_CSELECT_B32, s(2), n, s(2)),
        ]);
        for (family, opcode, fields) in before_load {
            self.lower_gfx1013(origin, family, opcode, fields)?;
        }
        let load = Smem {
            sdata: s(3),
            sbase: x + 4,
            offset: 0,
            soffset: s(2),
            glc: false,
            dlc: false,
            dwords: 1,
            buffer: false,
        };
        self.lower_gfx1013(origin, F::Smem, SMEM_LOAD_DWORD, Fields::Smem(load))?;
        let lgkm_zero = Fields::Sopp {
            simm16: encode_gfx10(63, 7, 0),
        };
        self.lower_gfx1013(origin, F::Sopp, SOPP_WAITCNT, lgkm_zero)?;
        let after_load = [
            sop2(SOP2_ASHR_I32, s(7), s(3), Operand::InlineInt(31)),
            sop2(SOP2_ADD_U32, s(4), s(4), s(3)),
            sop2(SOP2_ADDC_U32, s(5), s(5), s(7)),
            sopc(SOPC_CMP_LG_U32, s(6), zero),
        ];
        for (family, opcode, fields) in after_load {
            self.lower_gfx1013(origin, family, opcode, fields)?;
        }
        self.host_sop1(origin, SOP1_SETPC_B64, Operand::Raw(0), s(4))
    }

    /// The stub of a program without linked regions, in four SGPRs from `x`; see the module
    /// documentation. `n` is the guest code's size in bytes, the miss entry's index.
    fn narrow_dispatch_stub(
        &mut self,
        origin: &Instruction,
        x: u8,
        base: u64,
        n: Operand,
    ) -> Result<(), CompileError> {
        let s = |k: u8| Operand::Sgpr(x + k);
        let sop2 = |op, d, a, b| {
            let fields = Fields::Sop2 {
                sdst: d,
                src0: a,
                src1: b,
            };
            (F::Sop2, op, fields)
        };
        let sopc = |op, a, b| (F::Sopc, op, Fields::Sopc { src0: a, src1: b });
        let (zero, one) = (Operand::InlineInt(0), Operand::InlineInt(1));
        let before_anchor = [
            sop2(SOP2_CSELECT_B32, s(2), one, zero),
            sop2(SOP2_SUB_U32, s(0), s(0), Operand::constant(base as u32)),
            sop2(
                SOP2_SUBB_U32,
                s(1),
                s(1),
                Operand::constant((base >> 32) as u32),
            ),
            sopc(SOPC_CMP_EQ_U32, s(1), zero),
            sop2(SOP2_CSELECT_B32, s(0), s(0), n),
            sop2(SOP2_AND_B32, s(1), s(0), Operand::InlineInt(3)),
            sop2(SOP2_CSELECT_B32, s(0), n, s(0)),
            sop2(SOP2_MIN_U32, s(0), s(0), n),
            (F::Sop1, SOP1_MOV_B32, Fields::Sop1 { sdst: s(1), src0: s(2) }),
        ];
        for (family, opcode, fields) in before_anchor {
            self.lower_gfx1013(origin, family, opcode, fields)?;
        }
        self.host_sop1(origin, SOP1_GETPC_B64, s(2), Operand::Raw(0))?;
        let load = Smem {
            sdata: s(0),
            sbase: x + 2,
            offset: 0,
            soffset: s(0),
            glc: false,
            dlc: false,
            dwords: 1,
            buffer: false,
        };
        self.lower_gfx1013(origin, F::Smem, SMEM_LOAD_DWORD, Fields::Smem(load))?;
        let lgkm_zero = Fields::Sopp {
            simm16: encode_gfx10(63, 7, 0),
        };
        self.lower_gfx1013(origin, F::Sopp, SOPP_WAITCNT, lgkm_zero)?;
        let bit0 = |sdst| Fields::Sop1 { sdst, src0: zero };
        let after_load = [
            sop2(SOP2_OR_B32, s(2), s(2), s(1)),
            sop2(SOP2_ASHR_I32, s(1), s(0), Operand::InlineInt(31)),
            sop2(SOP2_ADD_U32, s(2), s(2), s(0)),
            sop2(SOP2_ADDC_U32, s(3), s(3), s(1)),
            sopc(SOPC_BITCMP1_B32, s(2), zero),
            (F::Sop1, SOP1_BITSET0_B32, bit0(s(2))),
        ];
        for (family, opcode, fields) in after_load {
            self.lower_gfx1013(origin, family, opcode, fields)?;
        }
        self.host_sop1(origin, SOP1_SETPC_B64, Operand::Raw(0), s(2))
    }

    pub(super) fn miss_stub(&mut self, origin: &Instruction) -> Result<(), CompileError> {
        self.lower_gfx1013(origin, F::Sopp, SOPP_ENDPGM, Fields::Sopp { simm16: 0 })
    }

    /// A host `s_getpc_b64`/`s_setpc_b64`, which the guest-PC lowering must not rewrite.
    fn host_sop1(
        &mut self,
        origin: &Instruction,
        gfx1013_op: u16,
        sdst: Operand,
        src0: Operand,
    ) -> Result<(), CompileError> {
        let info = opcodes::lookup(F::Sop1, gfx1013_op).expect("SOP1 opcode is tabled");
        let Xlat::Op(op) = info.translate(self.target) else {
            return Err(unsupported(origin, self.target, "no host PC instruction"));
        };
        self.push(derive(
            origin,
            F::Sop1,
            op,
            info.flags,
            Fields::Sop1 { sdst, src0 },
        ));
        Ok(())
    }
}

/// `s_getpc_b64`, `s_call_b64` and `s_swappc_b64` are one word long.
fn next_pc(inst: &Instruction) -> u64 {
    u64::from(inst.pc) + 4
}

/// The `k`th dword of a 64-bit scalar destination.
fn high_part(op: Operand, k: u8) -> Operand {
    match (op, k) {
        (op, 0) => op,
        (Operand::Sgpr(n), _) => Operand::Sgpr(n + k),
        (Operand::Ttmp(n), _) => Operand::Ttmp(n + k),
        (Operand::VccLo, _) => Operand::VccHi,
        (Operand::ExecLo, _) => Operand::ExecHi,
        _ => Operand::Null,
    }
}
