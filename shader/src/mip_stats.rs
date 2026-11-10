//! The table of texture LOD statistics translated pixel shaders update, and the report
//! `sceAgcDcbGetLodStats` makes of it (`sce::Agc::MipStatsBuffer`).

/// The statistics counters a texture can name in its T#.
pub const SLOTS: u32 = 256;
/// Bytes of the table: one `sce::Agc::MipStatsBuffer::Slot` per counter.
pub const TABLE_BYTES: u32 = SLOTS * 8;
/// Where the slots start in a `sce::Agc::MipStatsBuffer`, after the label and its padding.
pub const REPORT_SLOTS_OFFSET: u32 = 64;
/// Bytes of a `sce::Agc::MipStatsBuffer`.
pub const REPORT_BYTES: u32 = REPORT_SLOTS_OFFSET + TABLE_BYTES;
/// `m_finestMipSampled` of a counter no sample has reached.
const NO_MIP: u32 = 0xf;

/// The table with nothing counted: each slot holds its counter's id and no mip.
pub fn empty_table() -> Vec<u32> {
    (0..SLOTS).flat_map(|slot| [slot << 24, NO_MIP << 24]).collect()
}
