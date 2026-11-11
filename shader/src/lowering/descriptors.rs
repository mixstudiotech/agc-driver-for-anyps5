//! Guest V#, T# and S# converted to gfx11 form where instructions use them, computing
//! [`crate::descriptors::convert_vsharp`] and its siblings with SALU instructions and the
//! runtime's table ([`crate::descriptors::lut`]). Each conversion writes a slot the translation
//! reserves for the whole program; within a block a slot is reused until the guest writes the
//! descriptor's registers. SCC is kept.

use super::image_reports::{self, Report};
use super::{unsupported, waitcnt, Ctx};
use crate::descriptors::{
    DIRECT_D, LUT_BUFFER_FORMATS, LUT_IMAGE_FORMATS, LUT_SWIZZLE_MODES, XOR_SHIFT,
};
use crate::ir::{
    memops, Fields, Instruction, InstructionFamily as F, Operand, RegFile, RegRange,
    ResourceBinding, ResourceKind, ShaderIr, Smem,
};
use crate::CompileError;

const SOP1_MOV_B32: u16 = 0x03;
const SOP1_BITSET0_B32: u16 = 0x1b;
const SOP2_CSELECT_B32: u16 = 0x0a;
const SOP2_AND_B32: u16 = 0x0e;
const SOP2_OR_B32: u16 = 0x10;
const SOP2_ANDN2_B32: u16 = 0x14;
const SOP2_LSHL_B32: u16 = 0x1e;
const SOP2_LSHR_B32: u16 = 0x20;
const SOP2_BFE_U32: u16 = 0x27;
const SOPC_BITCMP1_B32: u16 = 0x0d;
const SOPC_CMP_LG_U32: u16 = 0x07;
const SOPP_WAITCNT: u16 = 0x0c;
const SMEM_LOAD_DWORD: u16 = 0x00;

/// SGPRs reserved for converted descriptors and the table address.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DescriptorSlots {
    /// SGPR pair the prologue loads with the table address.
    pub table: u8,
    pub vsharp: u8,
    pub tsharp: u8,
    pub ssharp: u8,
    pub address: u64,
}

/// A slot the program's resource bindings never need, left unreserved.
pub(crate) const NO_SLOT: u8 = u8::MAX;

/// The guest descriptor each slot holds in the current block (its first SGPR).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DescriptorCache {
    vsharp: Option<u8>,
    tsharp: Option<(u8, bool)>,
    ssharp: Option<u8>,
}

impl DescriptorCache {
    /// Forgets slots whose guest registers `inst` writes.
    pub fn invalidate(&mut self, inst: &Instruction, wave64: bool) {
        if self.vsharp.is_none() && self.tsharp.is_none() && self.ssharp.is_none() {
            return;
        }
        let defs = inst.defs(wave64);
        let writes = |first: u8, count: u8| {
            defs.iter()
                .any(|r| (0..count).any(|k| r.contains(RegFile::Scalar, u16::from(first + k))))
        };
        if self.vsharp.is_some_and(|s| writes(s, 4)) {
            self.vsharp = None;
        }
        if self
            .tsharp
            .is_some_and(|(s, r128)| writes(s, if r128 { 4 } else { 8 }))
        {
            self.tsharp = None;
        }
        if self.ssharp.is_some_and(|s| writes(s, 4)) {
            self.ssharp = None;
        }
    }
}

/// Emits one conversion sequence.
struct Seq<'c, 'r> {
    ctx: &'c mut Ctx<'r>,
    origin: Instruction,
}

impl Seq<'_, '_> {
    fn op(&mut self, family: F, opcode: u16, fields: Fields) -> Result<(), CompileError> {
        self.ctx.lower_gfx1013(&self.origin, family, opcode, fields)
    }

    fn sop2(
        &mut self,
        opcode: u16,
        sdst: u8,
        src0: Operand,
        src1: Operand,
    ) -> Result<(), CompileError> {
        let fields = Fields::Sop2 {
            sdst: Operand::Sgpr(sdst),
            src0,
            src1,
        };
        self.op(F::Sop2, opcode, fields)
    }

    fn mov(&mut self, sdst: u8, src0: Operand) -> Result<(), CompileError> {
        let fields = Fields::Sop1 {
            sdst: Operand::Sgpr(sdst),
            src0,
        };
        self.op(F::Sop1, SOP1_MOV_B32, fields)
    }

    /// `s_load_dword sdst, table, index offset:offset`.
    fn table_load(
        &mut self,
        sdst: u8,
        table: u8,
        index: u8,
        offset: u32,
    ) -> Result<(), CompileError> {
        let fields = Fields::Smem(Smem {
            sdata: Operand::Sgpr(sdst),
            sbase: table,
            offset: offset as i32,
            soffset: Operand::Sgpr(index),
            glc: false,
            dlc: false,
            dwords: 1,
            buffer: false,
        });
        self.op(F::Smem, SMEM_LOAD_DWORD, fields)
    }

    fn wait_for_loads(&mut self) -> Result<(), CompileError> {
        let fields = Fields::Sopp {
            simm16: waitcnt::encode_gfx10(waitcnt::GFX10_VMCNT_MAX, waitcnt::GFX10_EXPCNT_MAX, 0),
        };
        self.op(F::Sopp, SOPP_WAITCNT, fields)
    }

    fn keep_scc<T>(
        &mut self,
        saved: u8,
        body: impl FnOnce(&mut Self) -> Result<T, CompileError>,
    ) -> Result<T, CompileError> {
        self.sop2(
            SOP2_CSELECT_B32,
            saved,
            Operand::InlineInt(1),
            Operand::InlineInt(0),
        )?;
        let value = body(self)?;
        let restore = Fields::Sopc {
            src0: Operand::Sgpr(saved),
            src1: Operand::InlineInt(0),
        };
        self.op(F::Sopc, SOPC_CMP_LG_U32, restore)?;
        Ok(value)
    }
}

fn lit(value: u32) -> Operand {
    Operand::constant(value)
}

impl Ctx<'_> {
    fn descriptor_slots(&self, inst: &Instruction) -> Result<DescriptorSlots, CompileError> {
        self.descriptors
            .ok_or_else(|| unsupported(inst, self.target, "descriptor conversion is not set up"))
    }

    /// `slot` of `slots`, unless the program's bindings left it unreserved.
    fn slot(
        &self,
        inst: &Instruction,
        slot: u8,
        slots: &DescriptorSlots,
    ) -> Result<u8, CompileError> {
        if slot == NO_SLOT || (slot != slots.ssharp && slots.table == NO_SLOT) {
            return Err(unsupported(
                inst,
                self.target,
                "a descriptor the program's resource bindings do not list needs conversion",
            ));
        }
        Ok(slot)
    }

    /// Whether `slot` is converted in a spilled block ([`Ctx::spill_block`]): the program kept no
    /// slot for it because its SGPRs leave no room.
    fn spilled(&self, slot: u8) -> bool {
        slot == NO_SLOT && self.spill.is_some()
    }

    /// The table address into the SGPR pair at `table`, for a conversion in a spilled block.
    fn load_table(
        &mut self,
        inst: &Instruction,
        table: u8,
        address: u64,
    ) -> Result<(), CompileError> {
        let mut seq = Seq {
            ctx: self,
            origin: inst.clone(),
        };
        seq.mov(table, Operand::constant(address as u32))?;
        seq.mov(table + 1, Operand::constant((address >> 32) as u32))
    }

    /// Transient SGPRs for a conversion into a reserved slot: above the program's SGPRs, or a
    /// spilled block when they leave no room ([`Ctx::transient_sgprs`]).
    fn conversion_temps(&mut self, inst: &Instruction, count: u8) -> Result<u8, CompileError> {
        self.transient_sgprs(inst, count)
    }

    fn check_source(&self, inst: &Instruction, first: u8, count: u8) -> Result<(), CompileError> {
        if u16::from(first) + u16::from(count) > crate::ir::SGPR_LIMIT {
            return Err(unsupported(
                inst,
                self.target,
                "descriptor outside the SGPRs",
            ));
        }
        Ok(())
    }

    /// The gfx11 form of the V# at `source`.
    pub(super) fn converted_vsharp(
        &mut self,
        inst: &Instruction,
        source: u8,
    ) -> Result<u8, CompileError> {
        let slots = self.descriptor_slots(inst)?;
        let spilled = self.spilled(slots.vsharp);
        if !spilled {
            self.slot(inst, slots.vsharp, &slots)?;
            if self.descriptor_cache.vsharp == Some(source) {
                return Ok(slots.vsharp);
            }
        }
        self.check_source(inst, source, 4)?;
        // A spilled block: the V#, 4 temporaries, the table pair.
        let (t, temps, table) = if spilled {
            let block = self.spill_block(inst, 10)?;
            self.load_table(inst, block + 8, slots.address)?;
            (block, block + 4, block + 8)
        } else {
            (slots.vsharp, self.conversion_temps(inst, 4)?, slots.table)
        };
        let (index, swizzle, format, saved) = (temps, temps + 1, temps + 2, temps + 3);
        let g = |k: u8| Operand::Sgpr(source + k);
        let mut seq = Seq {
            ctx: self,
            origin: inst.clone(),
        };
        seq.keep_scc(saved, |s| {
            s.sop2(SOP2_LSHR_B32, index, g(3), lit(10))?;
            s.sop2(SOP2_AND_B32, index, Operand::Sgpr(index), lit(0x1fc))?;
            s.table_load(format, table, index, LUT_BUFFER_FORMATS)?;
            s.sop2(SOP2_AND_B32, t + 3, g(3), lit(0xf0e0_0fff))?;
            s.sop2(SOP2_LSHR_B32, swizzle, g(1), lit(1))?;
            s.sop2(
                SOP2_AND_B32,
                swizzle,
                Operand::Sgpr(swizzle),
                lit(0x4000_0000),
            )?;
            s.sop2(SOP2_AND_B32, t + 1, g(1), lit(0x3fff_ffff))?;
            s.sop2(
                SOP2_OR_B32,
                t + 1,
                Operand::Sgpr(t + 1),
                Operand::Sgpr(swizzle),
            )?;
            s.mov(t, g(0))?;
            s.mov(t + 2, g(2))?;
            s.wait_for_loads()?;
            s.sop2(
                SOP2_OR_B32,
                t + 3,
                Operand::Sgpr(t + 3),
                Operand::Sgpr(format),
            )
        })?;
        if !spilled {
            self.descriptor_cache.vsharp = Some(source);
        }
        Ok(t)
    }

    /// The gfx11 form (always 8 dwords) of the T# at `source`, 4 dwords with `r128`. With
    /// `report`, the table of [`crate::image_reports`], the program found the T# at run time:
    /// a conversion also reports the image unless the runtime knows it.
    pub(super) fn converted_tsharp(
        &mut self,
        inst: &Instruction,
        source: u8,
        r128: bool,
        report: Option<u64>,
    ) -> Result<u8, CompileError> {
        let slots = self.descriptor_slots(inst)?;
        let spilled = self.spilled(slots.tsharp);
        if !spilled {
            self.slot(inst, slots.tsharp, &slots)?;
            if self.descriptor_cache.tsharp == Some((source, r128)) {
                return Ok(slots.tsharp);
            }
        }
        self.check_source(inst, source, if r128 { 4 } else { 8 })?;
        // A report's registers come with the conversion's own, an even number of them: asked
        // for first with them, then without where the program cannot give that many. The 32
        // lanes of a narrow wave are too few for what an instruction would then borrow.
        let wanted = report.filter(|_| self.wave64 || self.spill.is_none());
        let mut extra = if wanted.is_some() { image_reports::SGPRS } else { 0 };
        // A spilled block: the T#, 8 temporaries, the table pair.
        let (t, temps, table, spare) = if spilled {
            let block = match self.spill_block(inst, 18 + extra) {
                Err(_) if extra != 0 => {
                    extra = 0;
                    self.spill_block(inst, 18)?
                }
                block => block?,
            };
            self.load_table(inst, block + 16, slots.address)?;
            (block, block + 8, block + 16, block + 18)
        } else {
            let temps = match self.conversion_temps(inst, 8 + extra) {
                Err(_) if extra != 0 => {
                    extra = 0;
                    self.conversion_temps(inst, 8)?
                }
                temps => temps?,
            };
            (slots.tsharp, temps, slots.table, temps + 8)
        };
        let [fmt_index, mode_index, format, mode, min_lod, max_mip, tmp, saved] =
            std::array::from_fn(|k| temps + k as u8);
        let report = wanted.filter(|_| extra != 0 && spare % 2 == 0).and_then(|table| Report::new(self, table, source, r128, spare));
        let g = |k: u8| {
            if r128 && k >= 4 {
                Operand::InlineInt(0)
            } else {
                Operand::Sgpr(source + k)
            }
        };
        let s_ = Operand::Sgpr;
        let mut seq = Seq {
            ctx: self,
            origin: inst.clone(),
        };
        seq.keep_scc(saved, |s| {
            if let Some(report) = &report {
                report.begin(s.ctx, &s.origin)?;
            }
            s.sop2(SOP2_LSHR_B32, fmt_index, g(1), lit(18))?;
            s.sop2(SOP2_AND_B32, fmt_index, s_(fmt_index), lit(0x7fc))?;
            s.sop2(SOP2_LSHR_B32, mode_index, g(3), lit(18))?;
            s.sop2(SOP2_AND_B32, mode_index, s_(mode_index), lit(0x3ffc))?;
            s.table_load(format, table, fmt_index, LUT_IMAGE_FORMATS)?;
            s.table_load(mode, table, mode_index, LUT_SWIZZLE_MODES)?;
            s.sop2(SOP2_BFE_U32, min_lod, g(1), lit(8 | (12 << 16)))?;
            s.sop2(SOP2_BFE_U32, max_mip, g(5), lit(4 | (4 << 16)))?;
            s.sop2(SOP2_AND_B32, t + 1, g(1), lit(0xc000_00ff))?;
            s.sop2(SOP2_LSHL_B32, max_mip, s_(max_mip), lit(16))?;
            s.sop2(SOP2_OR_B32, t + 1, s_(t + 1), s_(max_mip))?;
            s.sop2(SOP2_AND_B32, t + 2, g(2), lit(0x0fff_cfff))?;
            s.sop2(SOP2_AND_B32, t + 5, g(5), lit(0x04ff_ff0f))?;
            s.sop2(SOP2_LSHL_B32, tmp, s_(min_lod), lit(27))?;
            s.sop2(SOP2_OR_B32, t + 5, s_(t + 5), s_(tmp))?;
            s.sop2(SOP2_LSHR_B32, t + 6, s_(min_lod), lit(5))?;
            s.mov(t + 7, Operand::InlineInt(0))?;
            s.sop2(SOP2_AND_B32, t + 3, g(3), lit(0xfe0f_ffff))?;
            s.mov(t + 4, g(4))?;
            s.wait_for_loads()?;
            s.sop2(SOP2_AND_B32, tmp, s_(mode), s_(format))?;
            s.op(
                F::Sopc,
                SOPC_BITCMP1_B32,
                Fields::Sopc {
                    src0: s_(tmp),
                    src1: lit(DIRECT_D.trailing_zeros()),
                },
            )?;
            s.sop2(SOP2_CSELECT_B32, tmp, lit(!3u32), Operand::InlineInt(-1))?;
            s.sop2(SOP2_AND_B32, mode, s_(mode), s_(tmp))?;
            s.sop2(SOP2_ANDN2_B32, format, s_(format), lit(DIRECT_D))?;
            s.sop2(SOP2_BFE_U32, tmp, s_(mode), lit(XOR_SHIFT | (8 << 16)))?;
            s.sop2(SOP2_ANDN2_B32, t, g(0), s_(tmp))?;
            s.sop2(SOP2_OR_B32, t + 1, s_(t + 1), s_(format))?;
            s.sop2(SOP2_AND_B32, tmp, s_(mode), lit(0x01f0_0000))?;
            s.sop2(SOP2_OR_B32, t + 3, s_(t + 3), s_(tmp))?;
            s.sop2(SOP2_AND_B32, tmp, s_(mode), lit(1))?;
            s.sop2(SOP2_OR_B32, t + 1, s_(t + 1), s_(tmp))?;
            let test = Fields::Sopc {
                src0: s_(mode),
                src1: lit(1),
            };
            s.op(F::Sopc, SOPC_BITCMP1_B32, test)?;
            s.sop2(
                SOP2_CSELECT_B32,
                tmp,
                lit(0xffff_c000),
                Operand::InlineInt(-1),
            )?;
            s.sop2(SOP2_AND_B32, t + 4, s_(t + 4), s_(tmp))?;
            match &report {
                Some(report) => report.finish(s.ctx, &s.origin),
                None => Ok(()),
            }
        })?;
        if !spilled {
            self.descriptor_cache.tsharp = Some((source, r128));
        }
        Ok(t)
    }

    /// The gfx11 form of the S# at `source`. Its BORDER_COLOR_PTR moves to bits 29:18, which
    /// also clears the lane bits of PS5 `_a` samples (`img::ADJUST`).
    pub(super) fn converted_ssharp(
        &mut self,
        inst: &Instruction,
        source: u8,
    ) -> Result<u8, CompileError> {
        let slots = self.descriptor_slots(inst)?;
        let spilled = self.spilled(slots.ssharp);
        if !spilled {
            self.slot(inst, slots.ssharp, &slots)?;
            if self.descriptor_cache.ssharp == Some(source) {
                return Ok(slots.ssharp);
            }
        }
        self.check_source(inst, source, 4)?;
        // A spilled block: the S# and 2 temporaries.
        let (t, temps) = if spilled {
            let block = self.spill_block(inst, 6)?;
            (block, block + 4)
        } else {
            (slots.ssharp, self.conversion_temps(inst, 2)?)
        };
        let (border, saved) = (temps, temps + 1);
        let g = |k: u8| Operand::Sgpr(source + k);
        let mut seq = Seq {
            ctx: self,
            origin: inst.clone(),
        };
        seq.keep_scc(saved, |s| {
            s.sop2(SOP2_LSHL_B32, border, g(3), lit(18))?;
            s.sop2(
                SOP2_AND_B32,
                border,
                Operand::Sgpr(border),
                lit(0x3ffc_0000),
            )?;
            s.sop2(SOP2_AND_B32, t + 3, g(3), lit(0xc000_0000))?;
            s.sop2(
                SOP2_OR_B32,
                t + 3,
                Operand::Sgpr(t + 3),
                Operand::Sgpr(border),
            )?;
            s.mov(t + 2, g(2))?;
            let clear = Fields::Sop1 {
                sdst: Operand::Sgpr(t + 2),
                src0: lit(28),
            };
            s.op(F::Sop1, SOP1_BITSET0_B32, clear)?;
            s.mov(t, g(0))?;
            s.mov(t + 1, g(1))
        })?;
        if !spilled {
            self.descriptor_cache.ssharp = Some(source);
        }
        Ok(t)
    }

    /// The prologue's part: the table address into its SGPR pair.
    pub(super) fn load_descriptor_table(
        &mut self,
        origin: &Instruction,
    ) -> Result<(), CompileError> {
        let Some(slots) = self.descriptors.filter(|slots| slots.table != NO_SLOT) else {
            return Ok(());
        };
        let mut seq = Seq {
            ctx: self,
            origin: origin.clone(),
        };
        seq.mov(slots.table, Operand::constant(slots.address as u32))?;
        seq.mov(
            slots.table + 1,
            Operand::constant((slots.address >> 32) as u32),
        )
    }
}

/// User-SGPR descriptors the program uses only as descriptors of one kind: never written,
/// never read as data or by a BVH intersection (which reads the guest T#). The runtime can
/// convert these before the dispatch or draw. A V# may also feed `s_buffer_load`, which reads
/// fields the conversion keeps.
pub(super) fn pure_user_descriptors(ir: &ShaderIr, wave64: bool) -> Vec<ResourceBinding> {
    let user = u16::from(ir.metadata.user_sgpr_count);
    ir.metadata
        .bindings
        .iter()
        .filter(|b| {
            matches!(
                b.kind,
                ResourceKind::BufferDescriptor
                    | ResourceKind::ImageDescriptor
                    | ResourceKind::SamplerDescriptor
            ) && u16::from(b.base_sgpr) + u16::from(b.dword_count) <= user
        })
        .filter(|b| only_used_as(ir, b, wave64))
        .cloned()
        .collect()
}

fn only_used_as(ir: &ShaderIr, binding: &ResourceBinding, wave64: bool) -> bool {
    let (base, dwords) = (binding.base_sgpr, binding.dword_count);
    let touches = |r: &RegRange| {
        (0..dwords).any(|k| r.contains(RegFile::Scalar, u16::from(base) + u16::from(k)))
    };
    let exact = |r: &RegRange| {
        r.file == RegFile::Scalar && r.first == u16::from(base) && r.count == u16::from(dwords)
    };
    ir.instructions.iter().all(|inst| {
        let mut written = false;
        inst.each_def(wave64, |r| written |= touches(&r));
        if written {
            return false;
        }
        let descriptor = match (&inst.fields, binding.kind) {
            (Fields::Mubuf(b) | Fields::Mtbuf(b), ResourceKind::BufferDescriptor) => {
                b.srsrc == base
            }
            (Fields::Smem(s), ResourceKind::BufferDescriptor) => s.buffer && s.sbase == base,
            (Fields::Mimg(m), ResourceKind::ImageDescriptor) => {
                let bvh = memops::image(inst.opcode)
                    .is_some_and(|info| info.flags & memops::img::BVH != 0);
                m.srsrc == base && !bvh && dwords == if m.r128 { 4 } else { 8 }
            }
            (Fields::Mimg(m), ResourceKind::SamplerDescriptor) => m.uses_sampler && m.ssamp == base,
            _ => false,
        };
        let mut as_descriptor = true;
        inst.each_use(wave64, |r| {
            if touches(&r) {
                as_descriptor &= descriptor && exact(&r);
            }
        });
        as_descriptor
    })
}
