use super::is_vmem_load;
use crate::ir::{memops, Fields, FlatSegment, Instruction};
use crate::TargetIsa;

pub const GFX10_VMCNT_MAX: u8 = 63;
pub const GFX10_EXPCNT_MAX: u8 = 7;
pub const GFX10_LGKMCNT_MAX: u8 = 63;
pub const GFX10_VSCNT_MAX: u8 = 63;
pub const GFX12_WAIT_LOADCNT: u16 = 0x40;
pub const GFX12_WAIT_STORECNT: u16 = 0x41;
pub const GFX12_WAIT_SAMPLECNT: u16 = 0x42;
pub const GFX12_WAIT_BVHCNT: u16 = 0x43;
pub const GFX12_WAIT_EXPCNT: u16 = 0x44;
pub const GFX12_WAIT_DSCNT: u16 = 0x46;
pub const GFX12_WAIT_KMCNT: u16 = 0x47;
const GFX12_BVH_OPS: [u16; 2] = [0x19, 0x1a];
const GFX11_SENDMSG_RTN: [u16; 2] = [0x4c, 0x4d];
const QUEUE_LIMIT: usize = GFX10_VMCNT_MAX as usize;

pub fn decode_gfx10(simm16: u16) -> (u8, u8, u8) {
    let vm = (simm16 & 0xf) | ((simm16 >> 10) & 0x30);
    let exp = (simm16 >> 4) & 0x7;
    let lgkm = (simm16 >> 8) & 0x3f;
    (vm as u8, exp as u8, lgkm as u8)
}

pub fn encode_gfx10(vm: u8, exp: u8, lgkm: u8) -> u16 {
    let vm = u16::from(vm & 0x3f);
    (vm & 0xf) | ((vm >> 4) << 14) | (u16::from(exp & 0x7) << 4) | (u16::from(lgkm & 0x3f) << 8)
}

pub fn decode_gfx11(simm16: u16) -> (u8, u8, u8) {
    (
        ((simm16 >> 10) & 0x3f) as u8,
        (simm16 & 0x7) as u8,
        ((simm16 >> 4) & 0x3f) as u8,
    )
}

pub fn encode_gfx11(vm: u8, exp: u8, lgkm: u8) -> u16 {
    (u16::from(vm & 0x3f) << 10) | (u16::from(lgkm & 0x3f) << 4) | u16::from(exp & 0x7)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VmType {
    Load,
    Sample,
    Bvh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LgkmType {
    Ds,
    Km,
}

const VM_TYPES: [(VmType, u16); 3] = [
    (VmType::Load, GFX12_WAIT_LOADCNT),
    (VmType::Sample, GFX12_WAIT_SAMPLECNT),
    (VmType::Bvh, GFX12_WAIT_BVHCNT),
];

#[derive(Clone, Debug)]
pub struct WaitTracker {
    vm: Vec<VmType>,
    vm_prefix: [bool; 3],
    lgkm: Vec<LgkmType>,
    lgkm_prefix: [bool; 2],
}

impl WaitTracker {
    pub fn new(program_entry: bool) -> Self {
        Self {
            vm: Vec::new(),
            vm_prefix: [!program_entry; 3],
            lgkm: Vec::new(),
            lgkm_prefix: [!program_entry; 2],
        }
    }

    pub fn record(&mut self, target: TargetIsa, inst: &Instruction) {
        if !target.is_gfx12() {
            return;
        }
        match &inst.fields {
            Fields::Mimg(m) if is_vmem_load(inst) => {
                let ty = if GFX12_BVH_OPS.contains(&inst.opcode) {
                    VmType::Bvh
                } else if memops::gfx12_vsample(inst.opcode, m.uses_sampler) {
                    VmType::Sample
                } else {
                    VmType::Load
                };
                self.push_vm(ty);
            }
            Fields::Mubuf(_) | Fields::Mtbuf(_) if is_vmem_load(inst) => self.push_vm(VmType::Load),
            Fields::Flat(f) if is_vmem_load(inst) => {
                self.push_vm(VmType::Load);
                if f.segment == FlatSegment::Flat {
                    self.push_lgkm(LgkmType::Ds);
                }
            }
            Fields::Ds(_) => self.push_lgkm(LgkmType::Ds),
            Fields::Smem(s) if s.dwords > 0 => self.push_lgkm(LgkmType::Km),
            Fields::Sop1 { .. } if GFX11_SENDMSG_RTN.contains(&inst.opcode) => {
                self.push_lgkm(LgkmType::Km)
            }
            _ => {}
        }
    }

    fn push_vm(&mut self, ty: VmType) {
        self.vm.push(ty);
        if self.vm.len() > QUEUE_LIMIT {
            self.vm.remove(0);
            self.vm_prefix = [false; 3];
        }
    }

    fn push_lgkm(&mut self, ty: LgkmType) {
        self.lgkm.push(ty);
        if self.lgkm.len() > QUEUE_LIMIT {
            self.lgkm.remove(0);
            self.lgkm_prefix = [false; 2];
        }
    }

    pub fn gfx12_waits(
        &mut self,
        vm: Option<u8>,
        exp: Option<u8>,
        lgkm: Option<u8>,
        vs: Option<u8>,
    ) -> Vec<(u16, u8)> {
        let mut out = Vec::new();
        if let Some(n) = vm.filter(|&n| n < GFX10_VMCNT_MAX) {
            self.vm_waits(n, &mut out);
        }
        if let Some(n) = exp.filter(|&n| n < GFX10_EXPCNT_MAX) {
            out.push((GFX12_WAIT_EXPCNT, n));
        }
        if let Some(n) = lgkm.filter(|&n| n < GFX10_LGKMCNT_MAX) {
            self.lgkm_waits(n, &mut out);
        }
        if let Some(n) = vs.filter(|&n| n < GFX10_VSCNT_MAX) {
            out.push((GFX12_WAIT_STORECNT, n));
        }
        out
    }

    fn vm_waits(&mut self, n: u8, out: &mut Vec<(u16, u8)>) {
        let split = self.vm.len() - usize::from(n).min(self.vm.len());
        for (index, (ty, opcode)) in VM_TYPES.into_iter().enumerate() {
            let older = self.vm[..split].contains(&ty);
            if !older && !self.vm_prefix[index] {
                continue;
            }
            let younger = self.vm[split..].iter().filter(|&&t| t == ty).count();
            out.push((opcode, younger as u8));
            self.vm_prefix[index] = false;
        }
        self.vm.drain(..split);
    }

    fn lgkm_waits(&mut self, n: u8, out: &mut Vec<(u16, u8)>) {
        let split = self.lgkm.len() - usize::from(n).min(self.lgkm.len());
        let older_ds = self.lgkm[..split].contains(&LgkmType::Ds);
        let older_km = self.lgkm[..split].contains(&LgkmType::Km);
        if older_ds || self.lgkm_prefix[0] {
            let younger = self.lgkm[split..]
                .iter()
                .filter(|&&t| t == LgkmType::Ds)
                .count();
            out.push((GFX12_WAIT_DSCNT, younger as u8));
            self.lgkm_prefix[0] = false;
        }
        self.lgkm.drain(..split);
        if older_km || self.lgkm_prefix[1] {
            out.push((GFX12_WAIT_KMCNT, 0));
            self.lgkm_prefix[1] = false;
            self.lgkm.retain(|&t| t != LgkmType::Km);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx10_fields_unpack() {
        assert_eq!(decode_gfx10(0x0070), (0, 7, 0));
        assert_eq!(decode_gfx10(0xc07f), (63, 7, 0));
        assert_eq!(decode_gfx10(0x0a25), (5, 2, 10));
        assert_eq!(decode_gfx10(0xbf78), (40, 7, 63));
    }

    #[test]
    fn gfx10_to_gfx11_repacks_every_counter() {
        for (gfx10, gfx11) in [
            (0x0070, 0x0007),
            (0xc07f, 0xfc07),
            (0x0a25, 0x14a2),
            (0xbf78, 0xa3f7),
        ] {
            let (vm, exp, lgkm) = decode_gfx10(gfx10);
            assert_eq!(encode_gfx11(vm, exp, lgkm), gfx11, "{gfx10:#06x}");
            assert_eq!(decode_gfx11(gfx11), (vm, exp, lgkm));
            assert_eq!(encode_gfx10(vm, exp, lgkm), gfx10);
        }
    }

    #[test]
    fn gfx12_waits_at_program_entry_only_cover_issued_types() {
        let mut t = WaitTracker::new(true);
        t.vm = vec![VmType::Load, VmType::Sample, VmType::Load];
        assert_eq!(
            t.gfx12_waits(Some(1), None, None, None),
            vec![(GFX12_WAIT_LOADCNT, 1), (GFX12_WAIT_SAMPLECNT, 0)]
        );
        assert_eq!(t.vm, vec![VmType::Load]);
        assert!(t.gfx12_waits(Some(1), None, None, None).is_empty());
    }

    #[test]
    fn gfx12_waits_in_unknown_blocks_are_conservative() {
        let mut t = WaitTracker::new(false);
        t.vm = vec![VmType::Sample];
        let waits = t.gfx12_waits(Some(3), Some(0), Some(0), Some(0));
        assert_eq!(
            waits,
            vec![
                (GFX12_WAIT_LOADCNT, 0),
                (GFX12_WAIT_SAMPLECNT, 1),
                (GFX12_WAIT_BVHCNT, 0),
                (GFX12_WAIT_EXPCNT, 0),
                (GFX12_WAIT_DSCNT, 0),
                (GFX12_WAIT_KMCNT, 0),
                (GFX12_WAIT_STORECNT, 0),
            ]
        );
    }

    #[test]
    fn gfx12_scalar_loads_always_drain_to_zero() {
        let mut t = WaitTracker::new(true);
        t.lgkm = vec![LgkmType::Km, LgkmType::Ds, LgkmType::Ds];
        assert_eq!(
            t.gfx12_waits(None, None, Some(2), None),
            vec![(GFX12_WAIT_KMCNT, 0)]
        );
        assert_eq!(t.lgkm, vec![LgkmType::Ds, LgkmType::Ds]);
        assert_eq!(
            t.gfx12_waits(None, None, Some(1), None),
            vec![(GFX12_WAIT_DSCNT, 1)]
        );
        assert!(t.gfx12_waits(None, Some(7), Some(63), None).is_empty());
    }
}
