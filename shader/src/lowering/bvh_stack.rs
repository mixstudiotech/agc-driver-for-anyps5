use super::{Ctx, retarget, unsupported};
use crate::{CompileError, TargetIsa};
use crate::ir::{Ds, Fields, Instruction, InstructionFamily as F, Operand, Vop3, VopExt};

// Packed control word layout (SDK ds_bvh_stack_ctrl_dword).
const VALID_OFF: u32 = 0;
const ETT_OFF: u32 = 5;
const RING_OFF: u32 = 10;
const VALID_MASK: u32 = 0x0000_001f;
const ETT_MASK: u32 = 0x0000_03e0;
const RING_MASK: u32 = 0x0000_7c00;
const HAS_TLAS: u32 = 1 << 29;
const OVERFLOW: u32 = 1 << 30;
const B2T: u32 = 1 << 31;

// Sentinel class: node >> 3 == 0x1fffffff  <=>  node >= 0xfffffff8.
const SENTINEL_MIN: u32 = 0xffff_fff8;
const TERMINAL_NODE: u32 = 0xffff_fffe;
const INSTANCE_KIND: u32 = 6;
const RING_STRIDE_SHIFT: u32 = 7;

// FLAGS bits.
const FL_WALKBACK: u32 = 1;
const FL_VALID: u32 = 2;
const FL_SKIP_HIGH: u32 = 4;
const FL_SKIP_LOW: u32 = 8;

// SOP opcodes.
const S_MOV_B32: u16 = 0x03;
const S_MOV_B64: u16 = 0x04;
const S_CSELECT_B32: u16 = 0x0a;
const S_AND_B32: u16 = 0x0e;
const S_AND_B64: u16 = 0x0f;
const S_CMP_LG_U32: u16 = 0x07;
const S_WAITCNT: u16 = 0x0c;
const WAIT_LGKM0: u16 = 0xc07f;

// VALU opcodes.
const V_MOV_B32: u16 = 0x01;
const V_MIN_U32: u16 = 0x13;
const V_LSHRREV_B32: u16 = 0x16;
const V_LSHLREV_B32: u16 = 0x1a;
const V_AND_B32: u16 = 0x1b;
const V_OR_B32: u16 = 0x1c;
const V_ADD_NC_U32: u16 = 0x25;
const V_CNDMASK_B32: u16 = 0x101;
const V_BFE_U32: u16 = 0x148;
const V_BFI_B32: u16 = 0x14a;
const V_CMP_EQ_U32: u16 = 0xc2;
const V_CMP_LE_U32: u16 = 0xc3;
const V_CMP_NE_U32: u16 = 0xc5;

// DS opcodes.
const DS_WRITE_B32: u16 = 0x0d;
const DS_READ_B32: u16 = 0x36;

fn k(x: u32) -> Operand {
    Operand::constant(x)
}

fn vg(r: u8) -> Operand {
    Operand::Vgpr(r)
}

struct Gen {
    ops: Vec<(F, u16, Fields)>,
    w64: bool,
    // SGPR temps.
    exec_save: u8,
    p: u8,
    q: u8,
    scc_save: u8,
    // VGPR temps.
    c: u8,
    fl: u8,
    a: u8,
    t: u8,
    r: u8,
    // Ring capacity in entries.
    n: u32,
}

impl Gen {
    fn emit(&mut self, f: F, op: u16, fields: Fields) {
        self.ops.push((f, op, fields));
    }

    // ---------------- scalar helpers ----------------

    fn s_mov_mask(&mut self, sdst: Operand, src0: Operand) {
        let op = if self.w64 { S_MOV_B64 } else { S_MOV_B32 };
        self.emit(F::Sop1, op, Fields::Sop1 { sdst, src0 });
    }

    fn s_and_mask(&mut self, sdst: Operand, src0: Operand, src1: Operand) {
        let op = if self.w64 { S_AND_B64 } else { S_AND_B32 };
        self.emit(F::Sop2, op, Fields::Sop2 { sdst, src0, src1 });
    }

    /// EXEC = original EXEC & pred.
    fn exec_restrict(&mut self, pred: u8) {
        let e = self.exec_save;
        self.s_and_mask(Operand::ExecLo, Operand::Sgpr(e), Operand::Sgpr(pred));
    }

    fn exec_restore(&mut self) {
        let e = self.exec_save;
        self.s_mov_mask(Operand::ExecLo, Operand::Sgpr(e));
    }

    fn wait_lgkm0(&mut self) {
        self.emit(F::Sopp, S_WAITCNT, Fields::Sopp { simm16: WAIT_LGKM0 });
    }

    // ---------------- vector helpers ----------------

    fn v_mov(&mut self, vdst: u8, src0: Operand) {
        self.emit(
            F::Vop1,
            V_MOV_B32,
            Fields::Vop1 {
                vdst: vg(vdst),
                src0,
                ext: VopExt::None,
            },
        );
    }

    fn v2(&mut self, op: u16, vdst: u8, src0: Operand, vsrc1: u8) {
        self.emit(
            F::Vop2,
            op,
            Fields::Vop2 {
                vdst: vg(vdst),
                src0,
                vsrc1: vg(vsrc1),
                constant: None,
                ext: VopExt::None,
            },
        );
    }

    fn v3(&mut self, op: u16, vdst: u8, srcs: [Operand; 3]) {
        self.emit(F::Vop3, op, Fields::Vop3(Vop3::new(vg(vdst), srcs)));
    }

    /// sdst mask = cmp(src0, vsrc1). Explicit SGPR destination, VCC untouched.
    fn cmp(&mut self, op: u16, sdst: u8, src0: Operand, vsrc1: u8) {
        self.emit(
            F::Vop3,
            op,
            Fields::Vop3(Vop3::new(
                Operand::Sgpr(sdst),
                [src0, vg(vsrc1), Operand::Raw(0)],
            )),
        );
    }

    /// vdst = (src >> off) & ((1 << width) - 1)
    fn bfe(&mut self, vdst: u8, src: u8, off: u32, width: u32) {
        self.v3(V_BFE_U32, vdst, [vg(src), k(off), k(width)]);
    }

    /// vdst = (mask & ins) | (~mask & base)
    fn bfi(&mut self, vdst: u8, mask: Operand, ins: Operand, base: u8) {
        self.v3(V_BFI_B32, vdst, [mask, ins, vg(base)]);
    }

    /// vdst = sel[lane] ? yes : no
    fn cnd(&mut self, vdst: u8, no: Operand, yes: Operand, sel: u8) {
        self.v3(V_CNDMASK_B32, vdst, [no, yes, Operand::Sgpr(sel)]);
    }

    /// Insert TMP (already shifted into place) into CTRL under field mask.
    fn insert_ctrl_tmp(&mut self, mask: u32) {
        let (c, t) = (self.c, self.t);
        self.bfi(c, k(mask), vg(t), c);
    }

    /// Clear CTRL bits where TMP has ones (TMP used as per-lane mask).
    fn clear_ctrl_by_tmp(&mut self) {
        let (c, t) = (self.c, self.t);
        self.bfi(c, vg(t), k(0), c);
    }

    /// CTRL |= TMP
    fn or_ctrl_tmp(&mut self) {
        let (c, t) = (self.c, self.t);
        self.v2(V_OR_B32, c, vg(t), c);
    }

    /// LDS_ADDR = 4*base + 128*ring, computed from current CTRL (CTRL preserved).
    fn compute_lds_addr(&mut self) {
        let (c, a, t) = (self.c, self.a, self.t);
        // 4 * base_dword == (ctrl >> 13) & 0xfffc
        self.v2(V_LSHRREV_B32, a, k(13), c);
        self.v2(V_AND_B32, a, k(0xfffc), a);
        self.bfe(t, c, RING_OFF, 5);
        self.v2(V_LSHLREV_B32, t, k(RING_STRIDE_SHIFT), t);
        self.v2(V_ADD_NC_U32, a, vg(a), t);
    }

    /// p = sentinel-class(node)
    fn cmp_sentinel(&mut self, sdst: u8, node: u8) {
        self.cmp(V_CMP_LE_U32, sdst, k(SENTINEL_MIN), node);
    }

    // ---------------- algorithm ----------------

    fn init(&mut self, ctrl_in: u8, last: u8, cands: &[u8; 8]) {
        let (c, fl, t, p) = (self.c, self.fl, self.t, self.p);
        self.v_mov(c, vg(ctrl_in));
        // walkback = !sentinel(last)
        self.cmp_sentinel(p, last);
        self.cnd(fl, k(FL_WALKBACK), k(0), p);
        // data_valid = !walkback || any(candidate == last)
        self.cnd(t, k(0), k(1), p);
        for &cand in cands.iter() {
            self.cmp(V_CMP_EQ_U32, p, vg(last), cand);
            self.cnd(t, vg(t), k(1), p);
        }
        self.v2(V_LSHLREV_B32, t, k(1), t);
        self.v2(V_OR_B32, fl, vg(t), fl);
    }

    fn candidate(&mut self, node: u8, high: bool, last: u8) {
        let (fl, a, t, p, q) = (self.fl, self.a, self.t, self.p, self.q);
        let skip_bit = if high { FL_SKIP_HIGH } else { FL_SKIP_LOW };

        // q = sentinel(node) && data_valid && !skip_half
        self.cmp_sentinel(p, node);
        self.v2(V_AND_B32, t, k(FL_VALID | skip_bit), fl);
        self.cnd(t, k(0), vg(t), p);
        self.cmp(V_CMP_EQ_U32, q, k(FL_VALID), t);

        // skip_0_3 |= q && !(node & 2)
        self.v2(V_AND_B32, a, k(2), node);
        self.cmp(V_CMP_EQ_U32, p, k(0), a);
        self.cnd(t, k(0), k(FL_SKIP_LOW), q);
        self.cnd(t, k(0), vg(t), p);
        self.v2(V_OR_B32, fl, vg(t), fl);

        if high {
            // skip_4_7 |= q && !(node & 4)   (q uses pre-update skip_4_7)
            self.v2(V_AND_B32, a, k(4), node);
            self.cmp(V_CMP_EQ_U32, p, k(0), a);
            self.cnd(t, k(0), k(FL_SKIP_HIGH), q);
            self.cnd(t, k(0), vg(t), p);
            self.v2(V_OR_B32, fl, vg(t), fl);
        }

        // if walkback && node == last: data_valid = false
        self.cmp(V_CMP_EQ_U32, p, vg(last), node);
        self.cnd(t, k(0), vg(fl), p);
        self.v2(V_AND_B32, t, k(FL_WALKBACK), t);
        self.v2(V_LSHLREV_B32, t, k(1), t);
        self.bfi(fl, vg(t), k(0), fl);

        // push iff data_valid && !sentinel(node) && !skip_half (post-update)
        self.v2(V_AND_B32, t, k(FL_VALID | skip_bit), fl);
        self.cmp_sentinel(p, node);
        self.cnd(t, vg(t), k(0), p);
        self.cmp(V_CMP_EQ_U32, q, k(FL_VALID), t);

        self.exec_restrict(q);
        self.push(node);
        self.exec_restore();
    }

    fn push(&mut self, node: u8) {
        let (c, a, t, p, n) = (self.c, self.a, self.t, self.p, self.n);

        // 1. valid == N  ->  overflow latched
        self.bfe(t, c, VALID_OFF, 5);
        self.cmp(V_CMP_EQ_U32, p, k(n), t);
        self.cnd(t, k(0), k(OVERFLOW), p);
        self.or_ctrl_tmp();

        // 2. store node at current ring slot
        self.compute_lds_addr();
        self.emit(
            F::Ds,
            DS_WRITE_B32,
            Fields::Ds(Ds {
                vdst: 0 as _,
                addr: a as _,
                data0: node as _,
                data1: 0 as _,
                offset0: 0 as _,
                offset1: 0 as _,
                gds: false,
                dst_dwords: 0 as _,
                data0_dwords: 1 as _,
                data1_dwords: 0 as _,
                uses_addr: true,
                writes_addr: false,
            }),
        );
        self.wait_lgkm0();

        // 3. ring = (ring == N-1) ? 0 : ring + 1
        self.bfe(t, c, RING_OFF, 5);
        self.cmp(V_CMP_EQ_U32, p, k(n - 1), t);
        self.v2(V_ADD_NC_U32, t, k(1), t);
        self.cnd(t, vg(t), k(0), p);
        self.v2(V_LSHLREV_B32, t, k(RING_OFF), t);
        self.insert_ctrl_tmp(RING_MASK);

        // 4. entries_to_tlas == N  ->  has_tlas = false
        self.bfe(t, c, ETT_OFF, 5);
        self.cmp(V_CMP_EQ_U32, p, k(n), t);
        self.cnd(t, k(0), k(HAS_TLAS), p);
        self.clear_ctrl_by_tmp();

        // 5. entries_to_tlas = min(ett + 1, N)
        self.bfe(t, c, ETT_OFF, 5);
        self.v2(V_ADD_NC_U32, t, k(1), t);
        self.v2(V_MIN_U32, t, k(n), t);
        self.v2(V_LSHLREV_B32, t, k(ETT_OFF), t);
        self.insert_ctrl_tmp(ETT_MASK);

        //    valid = min(valid + 1, N)
        self.bfe(t, c, VALID_OFF, 5);
        self.v2(V_ADD_NC_U32, t, k(1), t);
        self.v2(V_MIN_U32, t, k(n), t);
        self.insert_ctrl_tmp(VALID_MASK);
    }

    fn pop(&mut self) {
        let (c, a, t, r, p, q, n) = (self.c, self.a, self.t, self.r, self.p, self.q, self.n);

        // 1. b2t = (ett == 0 && has_tlas); if b2t: has_tlas = false
        self.bfe(t, c, ETT_OFF, 5);
        self.cmp(V_CMP_EQ_U32, p, k(0), t);
        self.cnd(t, k(0), vg(c), p);
        self.v2(V_AND_B32, t, k(HAS_TLAS), t);
        self.v2(V_LSHLREV_B32, t, k(2), t);
        self.insert_ctrl_tmp(B2T);
        self.v2(V_LSHRREV_B32, t, k(2), t);
        self.clear_ctrl_by_tmp();

        // 2. empty: clear b2t/has_tlas, result = overflow ? INVALID : TERMINAL
        self.bfe(t, c, VALID_OFF, 5);
        self.cmp(V_CMP_EQ_U32, p, k(0), t);
        self.cnd(t, k(0), k(B2T | HAS_TLAS), p);
        self.clear_ctrl_by_tmp();
        self.bfe(r, c, 30, 1);
        self.v2(V_ADD_NC_U32, r, k(TERMINAL_NODE), r);

        // 3. nonempty lanes only
        self.bfe(t, c, VALID_OFF, 5);
        self.cmp(V_CMP_NE_U32, q, k(0), t);
        self.exec_restrict(q);

        //    entries_to_tlas = ett - 1 (unsigned wrap, truncated to 5 bits: 0 -> 31)
        self.bfe(t, c, ETT_OFF, 5);
        self.v2(V_ADD_NC_U32, t, k(0xffff_ffff), t);
        self.v2(V_LSHLREV_B32, t, k(ETT_OFF), t);
        self.insert_ctrl_tmp(ETT_MASK);

        //    valid = valid - 1 (nonzero here)
        self.bfe(t, c, VALID_OFF, 5);
        self.v2(V_ADD_NC_U32, t, k(0xffff_ffff), t);
        self.insert_ctrl_tmp(VALID_MASK);

        //    ring = (ring == 0) ? N-1 : ring - 1
        self.bfe(t, c, RING_OFF, 5);
        self.cmp(V_CMP_EQ_U32, p, k(0), t);
        self.v2(V_ADD_NC_U32, t, k(0xffff_ffff), t);
        self.cnd(t, vg(t), k(n - 1), p);
        self.v2(V_LSHLREV_B32, t, k(RING_OFF), t);
        self.insert_ctrl_tmp(RING_MASK);

        //    result = LDS[4*base + 128*ring]
        self.compute_lds_addr();
        self.emit(
            F::Ds,
            DS_READ_B32,
            Fields::Ds(Ds {
                vdst: r as _,
                addr: a as _,
                data0: 0 as _,
                data1: 0 as _,
                offset0: 0 as _,
                offset1: 0 as _,
                gds: false,
                dst_dwords: 1 as _,
                data0_dwords: 0 as _,
                data1_dwords: 0 as _,
                uses_addr: true,
                writes_addr: false,
            }),
        );
        self.wait_lgkm0();

        // 4. instance node popped
        self.v2(V_AND_B32, t, k(7), r);
        self.cmp(V_CMP_EQ_U32, p, k(INSTANCE_KIND), t);
        //    q = instance && has_tlas (pre-update)
        self.bfe(t, c, 29, 1);
        self.cmp(V_CMP_NE_U32, q, k(0), t);
        self.s_and_mask(Operand::Sgpr(q), Operand::Sgpr(q), Operand::Sgpr(p));
        //    if q: valid = entries_to_tlas, overflow = true
        self.bfe(a, c, VALID_OFF, 5);
        self.bfe(t, c, ETT_OFF, 5);
        self.cnd(t, vg(a), vg(t), q);
        self.insert_ctrl_tmp(VALID_MASK);
        self.cnd(t, k(0), k(OVERFLOW), q);
        self.or_ctrl_tmp();
        //    if instance: has_tlas = true, entries_to_tlas = 0
        self.cnd(t, k(0), k(HAS_TLAS), p);
        self.or_ctrl_tmp();
        self.cnd(t, k(0), k(ETT_MASK), p);
        self.clear_ctrl_by_tmp();

        self.exec_restore();
    }
}

pub(super) fn lower(ctx: &mut Ctx<'_>, inst: &Instruction) -> Result<(), CompileError> {
    let d = match &inst.fields {
        Fields::Ds(d) => d.clone(),
        _ => {
            return Err(unsupported(
                inst,
                ctx.target,
                "ds_bvh_stack_push8_pop1: expected DS fields",
            ));
        }
    };
    let n = d.offset0 as u32;
    if !(8..=31).contains(&n) || d.offset1 as u32 != 0 || d.gds {
        return Err(unsupported(
            inst,
            ctx.target,
            "ds_bvh_stack_push8_pop1: unsupported stack size/encoding",
        ));
    }
    let ctrl_in = d.addr as u8;
    let vdst = d.vdst as u8;
    let last = d.data0 as u8;
    let data1 = d.data1 as u32;
    if data1 + 7 > 255 || ctrl_in == vdst {
        return Err(unsupported(
            inst,
            ctx.target,
            "ds_bvh_stack_push8_pop1: invalid register operands",
        ));
    }
    if ctx.target == TargetIsa::Gfx12 {
        // gfx12 has the instruction: the same opcode and operands, with the stack size in
        // OFFSET0 (ACO `emit_ds_bvh_stack_push8_pop1_rtn`), and the same control word, flags,
        // sentinels and ring stride (decision D44).
        ctx.push(retarget(inst, inst.opcode));
        return Ok(());
    }
    let mut cands = [0u8; 8];
    for (i, c) in cands.iter_mut().enumerate() {
        *c = (data1 + i as u32) as u8;
    }

    let vb = ctx.regs.scratch_vgprs(5)? as u8;
    let sb = ctx.transient_sgprs(inst, 8)? as u8;
    let w64 = ctx.wave64;

    let mut g = Gen {
        ops: Vec::with_capacity(512),
        w64,
        exec_save: sb,
        p: sb + 2,
        q: sb + 4,
        scc_save: sb + 6,
        c: vb,
        fl: vb + 1,
        a: vb + 2,
        t: vb + 3,
        r: vb + 4,
        n,
    };

    // Preserve SCC, then original EXEC.
    let scc = g.scc_save;
    g.emit(
        F::Sop2,
        S_CSELECT_B32,
        Fields::Sop2 {
            sdst: Operand::Sgpr(scc),
            src0: k(1),
            src1: k(0),
        },
    );
    let e = g.exec_save;
    g.s_mov_mask(Operand::Sgpr(e), Operand::ExecLo);

    g.init(ctrl_in, last, &cands);

    // Reverse push order: 7,6,5,4 (high half), then 3,2,1,0 (low half).
    for i in (4..8).rev() {
        g.candidate(cands[i], true, last);
    }
    for i in (0..4).rev() {
        g.candidate(cands[i], false, last);
    }

    g.pop();

    // Final commits under original EXEC (already restored by pop).
    let (c, r) = (g.c, g.r);
    g.v_mov(ctrl_in, vg(c));
    g.v_mov(vdst, vg(r));

    // Restore SCC.
    g.emit(
        F::Sopc,
        S_CMP_LG_U32,
        Fields::Sopc {
            src0: Operand::Sgpr(scc),
            src1: k(0),
        },
    );

    let ops = std::mem::take(&mut g.ops);
    for (f, op, fields) in ops {
        ctx.lower_gfx1013(inst, f, op, fields)?;
    }
    Ok(())
}
