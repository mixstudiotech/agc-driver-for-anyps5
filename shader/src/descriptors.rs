//! Guest (gfx10) resource descriptors in gfx11 form.
//!
//! The translation converts a guest V#, T# or S# where an instruction uses it (12, 30 and 8
//! SALU instructions, `lowering::descriptors`), reading formats and swizzle modes from a table
//! the runtime places in GPU memory ([`lut`], [`crate::ShaderInterface::descriptor_lut`]). The
//! functions here compute the same conversions on the CPU: the runtime uses them for
//! descriptors it converts itself, and the tests check the emitted code against them.
//!
//! Field changes (Mesa `gfx10-rsrc.json` / `gfx11-rsrc.json`): V# word1 SWIZZLE_ENABLE moves to
//! bits 31:30 and CACHE_SWIZZLE goes; word3 FORMAT shrinks to 6 bits (renumbered) and
//! RESOURCE_LEVEL goes. T# word1 loses MIN_LOD (to word5 bits 31:27 and word6 bits 6:0) and
//! gains MAX_MIP from word5; FORMAT shrinks to 8 bits; word6 metadata fields go; word4 pitch
//! is cleared for redirected 1D, 2D and MSAA copies. Direct linear pitch is preserved.
//! S# BORDER_COLOR_PTR moves from bits 11:0 to 29:18 and
//! MIP_POINT_PRECLAMP goes. T# addresses of non-linear surfaces move 1 TiB up, to the host
//! layout copies the runtime keeps there. Host copies have no pipe/bank XOR, so the guest's
//! (the BASE_ADDRESS bits below the block size in XOR swizzle modes, [`xor_bits`]) is cleared:
//! the PS5 applies XOR bits gfx11 does not.

/// Bytes of [`lut`]: 512 image formats, 4096 T# swizzle entries, 128 buffer formats.
pub const LUT_BYTES: usize = 18_944;
/// Image format entries: `gfx11 FORMAT << 20` at `4 * gfx10 FORMAT`; bit2 marks a
/// 64/128-bit uncompressed element whose thin plain-D equations match both profiles.
pub const LUT_IMAGE_FORMATS: u32 = 0;
/// T# entries at `4 * word3[31:20]` (TYPE, BC_SWIZZLE, SW_MODE): host `SW_MODE << 20`, bit 0
/// to redirect the address to the host copy, bit 1 to clear word4 bits 13:0, bits 15:8 the
/// word0 bits to clear (the guest's pipe/bank XOR, [`xor_bits`]); bit2 marks thin plain-D.
pub const LUT_SWIZZLE_MODES: u32 = 2048;
/// Buffer format entries: `gfx11 FORMAT << 12` at `4 * gfx10 FORMAT`.
pub const LUT_BUFFER_FORMATS: u32 = 2048 + 16_384;

const IMAGE_ENTRIES: usize = 512;
const MODE_ENTRIES: usize = 4096;
const BUFFER_ENTRIES: usize = 128;
const SHADOW_REDIRECT: u32 = 1;
const CLEAR_PITCH: u32 = 2;
pub(crate) const DIRECT_D: u32 = 4;
pub(crate) const XOR_SHIFT: u32 = 8;

/// gfx10 FORMAT and gfx11 FORMAT of every format both generations have, from Mesa's register
/// database (MIT). The 32 others have no gfx11 equivalent and convert to FORMAT_INVALID (0).
#[rustfmt::skip]
const FORMATS: [(u16, u8); 128] = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
    (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19),
    (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28),
    (29, 29), (36, 30), (43, 31), (44, 32), (45, 33), (48, 34), (49, 35), (50, 36), (51, 37),
    (52, 38), (53, 39), (54, 40), (55, 41), (56, 42), (57, 43), (58, 44), (59, 45), (60, 46),
    (61, 47), (62, 48), (63, 49), (64, 50), (65, 51), (66, 52), (67, 53), (68, 54), (69, 55),
    (70, 56), (71, 57), (72, 58), (73, 59), (74, 60), (75, 61), (76, 62), (77, 63), (128, 64),
    (129, 65), (130, 66), (131, 227), (132, 67), (133, 68), (134, 69), (135, 70), (136, 71),
    (137, 72), (138, 73), (139, 74), (140, 75), (141, 76), (142, 77), (143, 78), (144, 79),
    (145, 80), (146, 81), (147, 82), (148, 83), (149, 84), (150, 85), (151, 86), (152, 87),
    (153, 88), (154, 89), (169, 109), (170, 110), (171, 111), (172, 112), (173, 113), (174, 114),
    (175, 115), (176, 116), (177, 117), (178, 118), (179, 119), (180, 120), (181, 121),
    (182, 122), (265, 207), (266, 208), (267, 209), (268, 210), (269, 211), (270, 212),
    (271, 213), (272, 214), (273, 215), (274, 216), (275, 217), (276, 218), (277, 219),
    (278, 220), (279, 90), (280, 91), (281, 92), (282, 93), (283, 94), (284, 95), (285, 228),
    (286, 205), (292, 206),
];

pub fn gfx11_format(gfx10: u16) -> Option<u8> {
    FORMATS
        .iter()
        .find(|&&(old, _)| old == gfx10)
        .map(|&(_, new)| new)
}

/// gfx11 3D surfaces accept these swizzle modes (`Gfx11Rsrc3dSwModeMask`).
const GFX11_3D_MODES: [u8; 12] = [0, 5, 9, 17, 21, 24, 25, 26, 27, 28, 29, 31];

/// The swizzle mode of the host copy of a guest surface with T# TYPE `ty` and SW_MODE `sw`:
/// gfx11 lacks 2D S modes (their D forms are used), 1D and MSAA surfaces take R_X unless they
/// are Z_X, and 3D surfaces keep their mode when gfx11 has it (else 64KB_S). The runtime creates
/// the host copies with this mode.
/// Whether the modeled gfx10 guest layout can be addressed in place on this gfx11 host.
/// Sharing a SW_MODE number does not prove this: Mesa addrlib gives different addresses even
/// for the accepted 3D S modes under guest GB_ADDR_CONFIG 0x44 and host 0x343. Until an exact
/// layout/type/element-size proof is available only linear images pass this coarse predicate.
/// [`samples_in_place_for_format`] additionally admits the proven thin-D element sizes.
pub fn samples_in_place(_ty: u8, sw: u8) -> bool {
    sw == 0
}

/// The complete GFX10/16-pipe and GFX11/0x343 equations are equal for thin plain
/// 64KB_D with 64/128-bit uncompressed elements. It has no pipe/bank XOR; both
/// layouts use the same block/mip/slice placement. Other element sizes differ.
pub fn samples_in_place_for_format(ty: u8, sw: u8, format: u16) -> bool {
    samples_in_place(ty, sw) || (direct_d_type(ty, sw) && direct_d_format(format))
}

fn direct_d_type(ty: u8, sw: u8) -> bool {
    matches!(ty, 9 | 11 | 13) && sw == 10
}
fn direct_d_format(format: u16) -> bool {
    matches!(format,62..=71|75..=77)
}

pub fn host_swizzle_mode(ty: u8, sw: u8) -> u8 {
    const TYPE_3D: u8 = 10;
    const Z_X_64KB: u8 = 24;
    const R_X_64KB: u8 = 27;
    const S_64KB: u8 = 9;
    match (ty, sw) {
        (_, 0) => 0,
        (TYPE_3D, sw) if GFX11_3D_MODES.contains(&sw) => sw,
        (TYPE_3D, _) => S_64KB,
        (8 | 12 | 14 | 15, Z_X_64KB) => Z_X_64KB,
        (8 | 12 | 14 | 15, _) => R_X_64KB,
        (_, 1 | 5 | 9 | 17 | 21 | 25) => sw + 1,
        (_, sw) => sw,
    }
}

/// The T# BASE_ADDRESS (word0) bits that hold the pipe/bank XOR in swizzle mode `sw`: those
/// below the block size in the XOR modes (`_T` and `_X`), 3:0 for 4 KiB blocks and 7:0 for
/// larger ones. Other modes have none; their surfaces are block aligned.
pub fn xor_bits(sw: u8) -> u32 {
    match sw {
        20..=23 => 0x0f,
        16..=19 | 24..=31 => 0xff,
        _ => 0,
    }
}

/// The conversion table the translation reads (little-endian dwords, [`LUT_BYTES`] bytes).
pub fn lut() -> Vec<u8> {
    let mut words = vec![0u32; IMAGE_ENTRIES + MODE_ENTRIES + BUFFER_ENTRIES];
    for &(old, new) in &FORMATS {
        words[usize::from(old)] = image_entry(u32::from(old));
        if usize::from(old) < BUFFER_ENTRIES {
            words[IMAGE_ENTRIES + MODE_ENTRIES + usize::from(old)] = u32::from(new) << 12;
        }
    }
    for index in 0..MODE_ENTRIES {
        words[IMAGE_ENTRIES + index] = swizzle_entry(index);
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn swizzle_entry(index: usize) -> u32 {
    let (ty, sw) = ((index >> 8) as u8, (index & 0x1f) as u8);
    let mut entry = u32::from(host_swizzle_mode(ty, sw)) << 20;
    if !samples_in_place(ty, sw) {
        entry |= SHADOW_REDIRECT | (xor_bits(sw) << XOR_SHIFT);
    }
    if !samples_in_place(ty, sw) && matches!(ty, 8 | 9 | 14) {
        entry |= CLEAR_PITCH;
    }
    if direct_d_type(ty, sw) {
        entry |= DIRECT_D;
    }
    entry
}

fn image_entry(format: u32) -> u32 {
    let format = format as u16;
    gfx11_format(format).map_or(0, |f| {
        (u32::from(f) << 20) | if direct_d_format(format) { DIRECT_D } else { 0 }
    })
}

fn buffer_entry(format: u32) -> u32 {
    let format = format as u16;
    match gfx11_format(format) {
        Some(f) if format < BUFFER_ENTRIES as u16 => u32::from(f) << 12,
        _ => 0,
    }
}

fn bfe(value: u32, offset: u32, width: u32) -> u32 {
    (value >> offset) & ((1 << width) - 1)
}

/// A gfx10 V# as gfx11 reads it.
pub fn convert_vsharp(g: [u32; 4]) -> [u32; 4] {
    let format = buffer_entry(bfe(g[3], 12, 7));
    let swizzle = (g[1] >> 1) & 0x4000_0000;
    [
        g[0],
        (g[1] & 0x3fff_ffff) | swizzle,
        g[2],
        (g[3] & 0xf0e0_0fff) | format,
    ]
}

/// A gfx10 T# (`r128`: only words 0-3) as gfx11 reads it.
pub fn convert_tsharp(g: [u32; 8]) -> [u32; 8] {
    let format = image_entry(bfe(g[1], 20, 9));
    let mut mode = swizzle_entry((g[3] >> 20) as usize);
    if format & mode & DIRECT_D != 0 {
        mode &= !(SHADOW_REDIRECT | CLEAR_PITCH);
    }
    let min_lod = bfe(g[1], 8, 12);
    let max_mip = bfe(g[5], 4, 4);
    let pitch_mask = if mode & CLEAR_PITCH != 0 {
        0xffff_c000
    } else {
        u32::MAX
    };
    [
        g[0] & !bfe(mode, XOR_SHIFT, 8),
        (g[1] & 0xc000_00ff) | (max_mip << 16) | (format & !DIRECT_D) | (mode & SHADOW_REDIRECT),
        g[2] & 0x0fff_cfff,
        (g[3] & 0xfe0f_ffff) | (mode & 0x01f0_0000),
        g[4] & pitch_mask,
        (g[5] & 0x04ff_ff0f) | (min_lod << 27),
        min_lod >> 5,
        0,
    ]
}

/// A gfx10 S# as gfx11 reads it.
pub fn convert_ssharp(g: [u32; 4]) -> [u32; 4] {
    let border_color_ptr = (g[3] << 18) & 0x3ffc_0000;
    [
        g[0],
        g[1],
        g[2] & !(1 << 28),
        (g[3] & 0xc000_0000) | border_color_ptr,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn pick<T: Copy>(&mut self, items: &[T]) -> T {
            items[self.below(items.len() as u64) as usize]
        }
    }

    fn put(value: u64, lo: u32, width: u32) -> u32 {
        ((value & ((1 << width) - 1)) as u32) << lo
    }

    /// V# fields placed the way Mesa's `ac_build_buffer_descriptor` places them.
    fn build_v(gfx11: bool, s: &[u64; 11]) -> [u32; 4] {
        let [va, stride, swz, cache_swz, size, dsel, fmt10, istride, tid, oob, ty] = *s;
        let mut w1 = put(va >> 32, 0, 16) | put(stride, 16, 14);
        let fmt = if gfx11 {
            w1 |= put(swz, 30, 2);
            u64::from(gfx11_format(fmt10 as u16).unwrap())
        } else {
            w1 |= put(swz, 31, 1) | put(cache_swz, 30, 1);
            fmt10
        };
        let mut w3 = put(dsel, 0, 12)
            | put(fmt, 12, if gfx11 { 6 } else { 7 })
            | put(istride, 21, 2)
            | put(tid, 23, 1)
            | put(oob, 28, 2)
            | put(ty, 30, 2);
        if !gfx11 {
            w3 |= put(1, 24, 1);
        }
        [va as u32, w1, size as u32, w3]
    }

    struct Image {
        va: u64,
        xor: u64,
        fmt10: u16,
        width: u64,
        height: u64,
        dsel: u64,
        base_level: u64,
        last_level: u64,
        sw: u8,
        bc: u64,
        ty: u8,
        depth: u64,
        base_array: u64,
        array_pitch: u64,
        min_lod_warn: u64,
        perf_mod: u64,
        corner: u64,
        prt_default: u64,
        min_lod: u64,
        max_mip: u64,
        lod_cnt_en: u64,
        bank_id: u64,
        comp: u64,
        wce: u64,
        meta: u64,
        it256: u64,
    }

    /// T# fields placed the way Mesa's `ac_build_gfx10_texture_descriptor` places them; the
    /// gfx11 form addresses the XOR-free host copy of non-linear surfaces.
    fn build_t(gfx11: bool, s: &Image) -> [u32; 8] {
        let host_copy = gfx11 && !samples_in_place_for_format(s.ty, s.sw, s.fmt10);
        let shadow = if host_copy { 1 << 40 } else { 0 };
        let xor = if host_copy { 0 } else { s.xor };
        let va = s.va + shadow;
        let (sw, fmt) = if gfx11 {
            (
                u64::from(host_swizzle_mode(s.ty, s.sw)),
                u64::from(gfx11_format(s.fmt10).unwrap()),
            )
        } else {
            (u64::from(s.sw), u64::from(s.fmt10))
        };
        let mut w = [0u32; 8];
        w[0] = ((va >> 8) as u32) | xor as u32;
        w[1] =
            put(va >> 40, 0, 8) | put(fmt, 20, if gfx11 { 8 } else { 9 }) | put(s.width - 1, 30, 2);
        w[2] = put((s.width - 1) >> 2, 0, 12) | put(s.height - 1, 14, 14);
        if !gfx11 {
            w[2] |= put(1, 31, 1);
        }
        w[3] = put(s.dsel, 0, 12)
            | put(s.base_level, 12, 4)
            | put(s.last_level, 16, 4)
            | put(sw, 20, 5)
            | put(s.bc, 25, 3)
            | put(u64::from(s.ty), 28, 4);
        let flat = host_copy && matches!(s.ty, 8 | 9 | 14);
        w[4] = put(if flat { 0 } else { s.depth }, 0, 13) | put(s.base_array, 16, 13);
        w[5] = put(s.array_pitch, 0, 4)
            | put(s.min_lod_warn, 8, 12)
            | put(s.perf_mod, 20, 3)
            | put(s.corner, 23, 1)
            | put(s.prt_default, 26, 1);
        if gfx11 {
            w[1] |= put(s.max_mip, 16, 4);
            w[5] |= put(s.min_lod, 27, 5);
            w[6] = put(s.min_lod >> 5, 0, 7);
        } else {
            w[1] |= put(s.min_lod, 8, 12);
            w[5] |= put(s.max_mip, 4, 4) | put(s.lod_cnt_en, 25, 1);
            w[6] = put(s.bank_id, 0, 8)
                | put(s.comp, 21, 1)
                | put(s.wce, 20, 1)
                | put(s.meta >> 8, 24, 8)
                | put(s.it256, 10, 1);
            w[7] = if s.comp != 0 {
                (s.meta >> 16) as u32
            } else {
                0
            };
        }
        w
    }

    /// S# fields placed the way Mesa's `ac_build_sampler_descriptor` places them.
    fn build_s(gfx11: bool, s: &[u64; 10]) -> [u32; 4] {
        let [w0, w1, lod_bias, lod_bias_sec, filter, aniso, blend_prt, preclamp, bc_ptr, bc_type] =
            *s;
        let mut w2 = put(lod_bias, 0, 14)
            | put(lod_bias_sec, 14, 6)
            | put(filter, 20, 8)
            | put(aniso, 29, 1)
            | put(blend_prt, 30, 1);
        let w3 = if gfx11 {
            put(bc_ptr, 18, 12) | put(bc_type, 30, 2)
        } else {
            w2 |= put(preclamp, 28, 1);
            put(bc_ptr, 0, 12) | put(bc_type, 30, 2)
        };
        [w0 as u32, w1 as u32, w2, w3]
    }

    const GUEST_ARENA: u64 = 0xfc_0000_0000;

    #[test]
    fn the_table_has_its_documented_layout() {
        let lut = lut();
        assert_eq!(lut.len(), LUT_BYTES);
        let word =
            |offset: u32| u32::from_le_bytes(lut[offset as usize..][..4].try_into().unwrap());
        assert_eq!(word(LUT_IMAGE_FORMATS + 4 * 36), 30 << 20);
        assert_eq!(word(LUT_BUFFER_FORMATS + 4 * 77), 63 << 12);
        // TYPE 2D (9), SW_MODE 64KB_S (9): the host copy is 64KB_D, redirected, pitch cleared.
        assert_eq!(word(LUT_SWIZZLE_MODES + 4 * ((9 << 8) | 9)), (10 << 20) | 3);
        // XOR modes also clear the XOR: 2D 64KB_S_X (25) and 4KB_D_X (22), 3D 64KB_S_X.
        let mode = |ty: u32, sw: u32| word(LUT_SWIZZLE_MODES + 4 * ((ty << 8) | sw));
        assert_eq!(mode(9, 25), (26 << 20) | (0xff << 8) | 3);
        assert_eq!(mode(9, 22), (22 << 20) | (0x0f << 8) | 3);
        assert_eq!(mode(10, 25), (25 << 20) | (0xff << 8) | 1);
        // A same-number 3D S mode still redirects: guest/host addrlib equations differ.
        assert_eq!(mode(10, 5), (5 << 20) | 1);
        assert_eq!(mode(10, 9), (9 << 20) | 1);
        assert!(!samples_in_place(10, 5));
        assert!(!samples_in_place(10, 9));
        assert!(samples_in_place(10, 0));
    }

    #[test]
    fn host_copies_drop_the_guest_xor_with_ps5_only_bits() {
        // The PS5 applies XOR bits 0xcf to 64 KiB blocks where gfx11 (GB_ADDR_CONFIG 0x343)
        // applies 0xe7, and 0x0f to 4 KiB blocks against 0x07: bit 3 exists only on the PS5.
        // A 2D 8_8_8_8_UNORM T# at `base` with `xor` in BASE_ADDRESS.
        let tsharp = |base: u64, sw: u32, xor: u32| {
            let mut t = [0u32; 8];
            t[0] = (base >> 8) as u32 | xor;
            t[1] = (base >> 40) as u32 | (56 << 20);
            t[3] = (sw << 20) | (9 << 28);
            t
        };
        let base_of = |t: [u32; 8]| (u64::from(t[0]) | (u64::from(t[1] & 0xff) << 32)) << 8;
        let host = 1 << 40;
        for (base, sw, xor) in [
            (0x12_3456_0000, 25, 0xcf), // 64KB_S_X, every PS5 bit
            (0x12_3456_0000, 24, 0x08), // 64KB_Z_X, only the PS5's bit
            (0x12_3456_0000, 17, 0x45), // 64KB_S_T, valid on both
            (0x12_3456_7000, 21, 0x0f), // 4KB_S_X: address bits 15:12 stay
        ] {
            let converted = convert_tsharp(tsharp(base, sw, xor));
            assert_eq!(base_of(converted), base + host, "mode {sw}, XOR {xor:#x}");
        }
        // Without an XOR mode the low bits are address bits: kept, and linear stays put.
        let linear = tsharp(0x12_3456_7800, 0, 0);
        assert_eq!(base_of(convert_tsharp(linear)), 0x12_3456_7800);
        let plain = tsharp(0x12_3456_7000, 5, 0);
        assert_eq!(base_of(convert_tsharp(plain)), 0x12_3456_7000 + host);
    }

    #[test]
    fn buffer_formats_agree_with_the_instruction_format_remap() {
        for format in 0..128u8 {
            let table = gfx11_format(u16::from(format)).filter(|&f| f < 64);
            let remap = crate::ir::memops::remap_buffer_format(format, crate::TargetIsa::Gfx11);
            assert_eq!(table, remap.filter(|_| format <= 77), "format {format}");
        }
    }

    #[test]
    fn conversions_match_descriptors_built_for_gfx11() {
        let mut rng = Rng(1013);
        let buffer_formats: Vec<u16> = FORMATS.iter().map(|f| f.0).filter(|&f| f < 128).collect();
        let image_formats: Vec<u16> = FORMATS.iter().map(|f| f.0).collect();
        for _ in 0..20_000 {
            let v = [
                rng.below(GUEST_ARENA),
                rng.below(1 << 14),
                rng.below(2),
                rng.below(2),
                rng.below(1 << 32),
                rng.below(1 << 12),
                u64::from(rng.pick(&buffer_formats)),
                rng.below(4),
                rng.below(2),
                rng.below(4),
                rng.pick(&[0, 0, 0, 2, 3]),
            ];
            assert_eq!(
                convert_vsharp(build_v(false, &v)),
                build_v(true, &v),
                "{v:x?}"
            );

            // Surfaces are aligned to their block; the XOR modes put a pipe/bank XOR below it.
            let sw = rng.pick(&[0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 24, 25, 26, 27]);
            let (block, xor_mask) = match sw {
                0..=2 => (256, 0),
                5 | 6 => (4096, 0),
                21 | 22 => (4096, 0x0f),
                9 | 10 => (65536, 0),
                _ => (65536, 0xff),
            };
            let image = Image {
                va: rng.below(GUEST_ARENA) & !(block - 1),
                xor: rng.below(256) & xor_mask,
                fmt10: rng.pick(&image_formats),
                width: rng.below(16384) + 1,
                height: rng.below(16384) + 1,
                dsel: rng.below(1 << 12),
                base_level: rng.below(16),
                last_level: rng.below(16),
                sw,
                bc: rng.below(6),
                ty: rng.pick(&[8, 9, 10, 11, 12, 13, 14, 15]),
                depth: rng.below(1 << 13),
                base_array: rng.below(1 << 13),
                array_pitch: rng.below(16),
                min_lod_warn: rng.below(1 << 12),
                perf_mod: rng.below(8),
                corner: rng.below(2),
                prt_default: rng.below(2),
                min_lod: rng.below(1 << 12),
                max_mip: rng.below(16),
                lod_cnt_en: rng.below(2),
                bank_id: rng.below(256),
                comp: rng.below(2),
                wce: rng.below(2),
                meta: rng.below(GUEST_ARENA) & !0xff,
                it256: rng.below(2),
            };
            assert_eq!(
                convert_tsharp(build_t(false, &image)),
                build_t(true, &image)
            );

            let s = [
                rng.below(1 << 32),
                rng.below(1 << 32),
                rng.below(1 << 14),
                rng.below(64),
                rng.below(256),
                rng.below(2),
                rng.below(2),
                rng.below(2),
                rng.below(4096),
                rng.below(4),
            ];
            assert_eq!(
                convert_ssharp(build_s(false, &s)),
                build_s(true, &s),
                "{s:x?}"
            );
        }
    }
}

#[cfg(test)]
mod direct_descriptor_tests {
    use super::*;
    #[test]
    fn direct_d_requires_large_elements_and_linear_keeps_pitch_and_feedback_fields() {
        for ty in [9, 11, 13] {
            for format in [56, 62, 65, 71, 72, 75, 77] {
                let expected = matches!(format,62..=71|75..=77);
                let mut descriptor = [0u32; 8];
                descriptor[0] = 0x10000000;
                descriptor[1] = u32::from(format) << 20;
                descriptor[3] = (ty << 28) | (10 << 20) | 0xfac;
                descriptor[5] = (1 << 26) | (0x123 << 8);
                let converted = convert_tsharp(descriptor);
                assert_eq!(converted[1] & 1 == 0, expected, "type{ty} format{format}");
                assert_eq!(samples_in_place_for_format(ty as u8, 10, format), expected);
                assert_eq!(converted[5] & 0x04ffff00, descriptor[5] & 0x04ffff00);
                assert_eq!(converted[1] & DIRECT_D, 0, "LUT property leaked into T#");
            }
        }
        let mut descriptor = [0u32; 8];
        descriptor[1] = 20 << 20;
        descriptor[3] = 9 << 28;
        descriptor[4] = (1 << 13) | 1023;
        descriptor[5] = (1 << 26) | (0x321 << 8);
        let converted = convert_tsharp(descriptor);
        assert_eq!(converted[4], descriptor[4]);
        assert_eq!(converted[5] & 0x04ffff00, descriptor[5] & 0x04ffff00);
    }
}
