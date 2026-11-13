use super::*;

fn fill(address: u64, bytes: u64, word: u32) -> ProvenFill {
    ProvenFill { address, bytes, fill: Fill::new([word; 4], 16).unwrap(), flip: 0, conditional: false }
}

#[test]
fn repeated_identical_fills_are_new_events_and_consumption_is_range_scoped() {
    let mut writes = MetadataWrites::default();
    writes.record(fill(0x1000, 0x3000, 0x20202020));
    writes.invalidate(0x2000, 0x1000);
    assert!(writes.covering(0x1000, 0x1000).is_some());
    assert!(writes.covering(0x2000, 1).is_none());
    assert!(writes.covering(0x3000, 0x1000).is_some());
    writes.record(fill(0x2000, 0x1000, 0x20202020));
    assert!(writes.covering(0x2000, 0x1000).is_some());
    writes.invalidate(0x2800, 4);
    assert!(writes.covering(0x2000, 0x1000).is_none());
}

#[test]
fn metadata_evidence_has_no_fixed_address_limit_and_expires_without_a_consumer() {
    let mut writes = MetadataWrites::default();
    for i in 1..=2048 { writes.record(fill(i * 0x1000, 0x800, 0)); }
    assert!(writes.covering(2048 * 0x1000, 0x800).is_some());
    writes.expire(2);
    assert!(writes.fills.is_empty());
}

#[test]
fn unaligned_overwrites_discard_partial_words_without_changing_fill_phase() {
    let mut writes = MetadataWrites::default();
    writes.record(fill(0x1000, 0x100, 0xfffc000f));
    writes.invalidate(0x1011, 2);
    assert!(writes.covering(0x1010, 4).is_none());
    let right = writes.covering(0x1014, 4).unwrap();
    assert_eq!(right.address % 4, 0);
    assert_eq!(right.fill.uniform_word(), Some(0xfffc000f));
}

#[test]
fn dcc_constant_codes_follow_component_count_swap_and_integer_clamping() {
    use crate::amd::surface::{formats::TexelFormat, metadata::{classify_dcc, DccState}};
    let value = |format, key| match classify_dcc(&[key; 4], format, None).unwrap() {
        DccState::Clear(texel) => texel.bytes().to_vec(),
        DccState::Expanded => panic!("expected a clear"),
    };
    let rg = TexelFormat::from_gfx10(14).unwrap();
    assert_eq!(value(rg, 0x40), [0, 255]);
    assert_eq!(value(rg, 0x80), [255, 0]);
    let reversed = TexelFormat::from_color_info((10 << 2) | (2 << 11)).unwrap();
    assert_eq!(value(reversed, 0x40), [255, 0, 0, 0]);
    assert_eq!(value(reversed, 0x80), [0, 255, 255, 255]);
    let rgb = TexelFormat::from_gfx10(36).unwrap();
    assert_eq!(value(rgb, 0x40), [0; 4]);
    assert_eq!(value(rgb, 0x80), 0x781e03c0u32.to_le_bytes());
    assert_eq!(value(TexelFormat::from_gfx10(27).unwrap(), 0xc0), [255; 4]);
    assert_eq!(value(TexelFormat::from_gfx10(24).unwrap(), 0xc0), [255, 127, 255, 127]);
}
