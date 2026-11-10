//! Symbol NIDs: the names PS5 modules import and export functions by.
//!
//! A NID is the first eight bytes of SHA-1(name ‖ suffix), reversed, in base64 with `+` and `-`
//! as the last two digits and no padding (11 characters), as AnyPS5's `tools/nid_names.py`
//! (`compute_nid`) and `core/libs/nid` compute them.

/// The constant appended to every name before hashing.
const SUFFIX: [u8; 16] = [
    0x51, 0x8d, 0x64, 0xa6, 0x35, 0xde, 0xd8, 0xc1, 0xe6, 0xb0, 0x39, 0xb1, 0xc3, 0xe5, 0x52, 0x30,
];

/// The NID of `name`.
pub fn nid(name: &str) -> String {
    let mut message = Vec::with_capacity(name.len() + SUFFIX.len());
    message.extend_from_slice(name.as_bytes());
    message.extend_from_slice(&SUFFIX);
    let digest = sha1(&message);
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    bytes.reverse();
    base64(&bytes)
}

/// Base64 of `bytes` with `+` and `-` as digits 62 and 63, without padding.
fn base64(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let group = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        let digits = chunk.len() + 1;
        for index in 0..digits {
            text.push(DIGITS[((group >> (18 - 6 * index)) & 63) as usize] as char);
        }
    }
    text
}

/// SHA-1 (FIPS 180-4) of `data`.
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut state: [u32; 5] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476, 0xc3d2_e1f0];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut schedule = [0u32; 80];
        for (index, word) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..80 {
            schedule[index] = (schedule[index - 3] ^ schedule[index - 8] ^ schedule[index - 14]
                ^ schedule[index - 16])
                .rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = state;
        for (index, word) in schedule.iter().enumerate() {
            let (mixed, constant) = match index {
                0..=19 => ((b & c) | (!b & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(mixed)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut digest = [0u8; 20];
    for (bytes, word) in digest.chunks_exact_mut(4).zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}
