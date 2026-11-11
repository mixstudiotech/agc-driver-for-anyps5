//! The table through which translated programs tell the runtime which images they found at run
//! time ([`crate::ir::DescriptorSource::Dynamic`]): the T# of a light picked by a loop, of a
//! material picked by a lane's index. No work names such an image, so the runtime cannot bring
//! its host copy up to date ahead of the program; it does so for the next frame, from what the
//! program reported.
//!
//! The table has two parts. The runtime writes the images it already knows into the first, a
//! [`hash`] per image, two to a bucket, and a program looks its T# up there with one scalar
//! load: a known image costs it nothing more. An unknown one it stores into the second part, a
//! T# and a check word per slot; the slot is picked by the T#'s first dword, so storing an
//! image again changes nothing.
//!
//! Only the runtime can make an image known. A program's waves do not see each other's stores
//! through scalar loads: on gfx11.5, of 200,000 waves that each stored a mark where they found
//! none, every one found none; with GLC on the load 41 to 84 did, at the price of a load that
//! goes past the scalar cache every time.

/// Buckets of the known part: two hashes each.
pub const KNOWN_BUCKETS: u32 = 1 << 13;
/// Bytes of a bucket.
pub const KNOWN_BUCKET_BYTES: u32 = 8;
pub const KNOWN_BYTES: u32 = KNOWN_BUCKETS * KNOWN_BUCKET_BYTES;
/// Slots of the report part.
pub const REPORT_SLOTS: u32 = 1 << 12;
/// Bytes of a report slot: the T#'s eight dwords, the check word, and padding to a cache line.
pub const REPORT_BYTES: u32 = 64;
/// Where the check word lies in a report slot.
pub const REPORT_CHECK_OFFSET: u32 = 32;
/// Where the report part starts in the table.
pub const REPORTS_OFFSET: u32 = KNOWN_BYTES;
pub const TABLE_BYTES: u32 = REPORTS_OFFSET + REPORT_SLOTS * REPORT_BYTES;
/// A report slot's check word is the T#'s [`fold`] with this: a slot nothing was stored in, or
/// one read while a program is storing into it, does not check.
pub const CHECK: u32 = 0x5a3c_a5c3;
/// Spreads hashes over the buckets and first dwords over the report slots (the 32-bit golden
/// ratio).
pub const SPREAD: u32 = 0x9e37_79b9;

/// Dwords 1 to 5 of a T# in one: with dword 0, the base address, what tells the images a title
/// has at one address apart. Dwords 6 and 7 hold only metadata addresses.
pub fn fold(tsharp: &[u32; 8]) -> u32 {
    tsharp[1] ^ tsharp[2] ^ tsharp[3] ^ tsharp[4] ^ tsharp[5]
}

/// What the known part holds for an image. Two images of one bucket with one hash would pass
/// for each other; 0 is an empty place in a bucket, and the hash of the null T#.
pub fn hash(tsharp: &[u32; 8]) -> u32 {
    (tsharp[0].wrapping_mul(SPREAD) ^ fold(tsharp)).wrapping_mul(SPREAD)
}

/// Byte offset of the bucket of the image with this hash.
pub fn known_offset(hash: u32) -> u32 {
    (hash >> (32 - KNOWN_BUCKETS.trailing_zeros())) * KNOWN_BUCKET_BYTES
}

/// Byte offset in the table of the report slot of the image whose T# starts with `first`.
pub fn report_offset(first: u32) -> u32 {
    REPORTS_OFFSET + ((first.wrapping_mul(SPREAD) >> 14) & ((REPORT_SLOTS - 1) << 6))
}
