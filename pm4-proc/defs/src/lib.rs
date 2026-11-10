//! AMD hardware definitions generated from Mesa (decisions D16, D17): the PS5 guest's gfx10, the
//! RDNA3.5 (gfx11.5) and RDNA4 (gfx12) hosts, their resource descriptors, and PM4 packets. They
//! live in their own crate so that the crates using them do not regenerate them on every build.
//!
//! Each register module has `OFFSET` (the byte address Mesa's `mm` names) and its fields;
//! `pkt3` holds the type-3 opcodes, `pkt3::dwords` the packet dwords `pkt3.json` describes, and
//! `sid` the other constants and field macros of `sid.h`, under Mesa's names. `cp_gfx11` and
//! `cp_gfx12` are AMD's CP packet tables for the two hosts: per CP engine and packet, its
//! opcode, fields ([`PacketField`]) and enums.

/// A register, descriptor or packet field: `width` bits from bit `shift`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Field {
    pub shift: u32,
    pub width: u32,
}

impl Field {
    pub const fn new(shift: u32, width: u32) -> Self {
        Self { shift, width }
    }

    /// The field's bits in place.
    pub const fn mask(self) -> u32 {
        (((1u64 << self.width) - 1) << self.shift) as u32
    }

    /// The field's value in `word`.
    pub const fn get(self, word: u32) -> u32 {
        (word & self.mask()) >> self.shift
    }

    /// `value` in place, bits beyond the field's width dropped.
    pub const fn put(self, value: u32) -> u32 {
        ((value as u64) << self.shift) as u32 & self.mask()
    }

    /// `word` with the field set to `value`.
    pub const fn set(self, word: u32, value: u32) -> u32 {
        (word & !self.mask()) | self.put(value)
    }
}

/// A field of a packet dword (dword 0 is the header).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PacketField {
    pub dword: u32,
    pub field: Field,
}

impl PacketField {
    pub const fn new(dword: u32, shift: u32, width: u32) -> Self {
        Self { dword, field: Field::new(shift, width) }
    }

    /// `value` in the field of `packet`, bits beyond the field's width dropped.
    pub fn set(self, packet: &mut [u32], value: u32) {
        let word = &mut packet[self.dword as usize];
        *word = self.field.set(*word, value);
    }

    /// The field's value in `packet`.
    pub fn get(self, packet: &[u32]) -> u32 {
        self.field.get(packet[self.dword as usize])
    }
}

pm4_proc::cp_packets!(gfx11 in crate);
pm4_proc::cp_packets!(gfx12 in crate);
pm4_proc::registers!(gfx10 in crate);
pm4_proc::registers!(gfx115 in crate);
pm4_proc::registers!(gfx12 in crate);
pm4_proc::registers!(gfx10_rsrc in crate);
pm4_proc::registers!(gfx115_rsrc in crate);
pm4_proc::registers!(gfx12_rsrc in crate);
pm4_proc::packets!(in crate);
