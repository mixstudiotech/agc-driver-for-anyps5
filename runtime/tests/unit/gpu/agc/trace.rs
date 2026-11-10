use super::*;
use crate::gpu::agc::arena::Range;

fn arena(memory: &mut [u32]) -> Bump {
    Bump::new(Range { cpu: memory.as_mut_ptr().cast(), gpu: 0x100000, bytes: std::mem::size_of_val(memory) })
}

#[test]
fn clones_preserve_ib_control_bits_and_source_addresses() {
    let mut memory = vec![0u32; 4096];
    let storage = arena(&mut memory);
    let child = [header(pkt3::SET_SH_REG, 3, 0).unwrap(), 0x240, 7];
    let control = 0x8120_0000 | CHAIN | child.len() as u32;
    let root = [header(pkt3::INDIRECT_BUFFER, 4, 1).unwrap(), 0x4000, 0, control];
    let result = snapshot(&root, 0x2000, 0x8000, 9, &storage, |address, count| {
        assert_eq!((address, count), (0x4000, child.len())); Ok(child.to_vec())
    }).unwrap();
    // Marker then the original, predicated IB packet with only its address/size relocated.
    assert_eq!(result.image[6], root[0]);
    assert_eq!(result.image[9] & !0xf_ffff, control & !0xf_ffff);
    assert_ne!(result.image[7], root[1]);
    assert_eq!(result.ranges.iter().map(|range| range.address).collect::<Vec<_>>(), [0x2000, 0x4000]);
    assert_eq!(result.words, [root.to_vec(), child.to_vec()].concat());
    assert!(result.followed_chain);
    let offset = (u64::from(result.image[7]) - 0x100000) as usize / 4;
    assert_eq!(memory[offset + 4], 4); // Child source packet starts after the four root dwords.
    assert_eq!(memory[offset + 6], child[0]);
}

#[test]
fn nested_guards_keep_their_union_and_original_counts() {
    let mut memory = vec![0u32; 4096];
    let storage = arena(&mut memory);
    let cond = header(pkt3::COND_EXEC, 5, 0).unwrap();
    let set = header(pkt3::SET_SH_REG, 3, 0).unwrap();
    let words = [cond, 0x1000, 0, 0, 11, cond, 0x1000, 0, 0, 3, set, 0x240, 1, set, 0x240, 2, set, 0x240, 3];
    let result = snapshot(&words, 0x2000, 0x8000, 9, &storage, |_, _| unreachable!()).unwrap();
    assert_eq!(&result.image[..16], &words[..16]);
    assert_eq!(result.image[16], header(pkt3::WRITE_DATA, 6, 0).unwrap());
    assert_eq!(&result.image[22..25], &words[16..]);
}

#[test]
fn unsupported_or_unreadable_control_flow_is_rejected() {
    let mut memory = vec![0u32; 4096];
    let storage = arena(&mut memory);
    for words in [
        vec![header(pkt3::PRED_EXEC, 2, 0).unwrap(), 0],
        vec![header(pkt3::INDIRECT_BUFFER, 14, 0).unwrap(); 14],
        vec![header(pkt3::COND_EXEC, 5, 0).unwrap(), 0x1000, 0, 0, 1],
    ] {
        assert!(snapshot(&words, 0x2000, 0x8000, 9, &storage, |_, _| unreachable!()).is_err());
    }
    let words = [header(pkt3::INDIRECT_BUFFER, 4, 1).unwrap(), 0x4000, 0, 3];
    assert!(snapshot(&words, 0x2000, 0x8000, 9, &storage, |_, _| Err(Error::invalid("unreadable optional child"))).is_err());
}

#[test]
fn root_chain_preflight_ignores_ib2_and_nop_payloads() {
    let ib = header(pkt3::INDIRECT_BUFFER, 4, 0).unwrap();
    assert!(!has_root_chain(&[ib, 0x4000, 0, 3]));
    assert!(!has_root_chain(&[header(pkt3::NOP, 5, 0).unwrap(), ib, 0x4000, 0, CHAIN | 3]));
    assert!(has_root_chain(&[ib, 0x4000, 0, CHAIN | 3]));
}
