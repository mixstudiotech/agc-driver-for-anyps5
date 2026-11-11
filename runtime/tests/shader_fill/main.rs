use shader::{CompileOptions, Compiler, ShaderInterface, ShaderStage, TargetIsa, ir::BufferFill};

/// A title's memset on the GPU: every thread stores s[4:7] through the V# in s[0:3], at
/// `(s8 << 6) + v0`.
const FILL: [u32; 9] = [0xd746_0004, 0x0401_0c08, 0x7e00_0204, 0x7e02_0205, 0x7e04_0206, 0x7e06_0207, 0xe01c_2000, 0x8000_0004, 0xbf81_0000];
/// The same title's masked update: it loads each record before it stores it.
const MASKED: [u32; 16] = [
    0xd746_0004, 0x0401_0c0c, 0xe00c_2000, 0x8000_0004, 0xbf8c_3f70, 0xd771_0000, 0x0012_0008, 0xd771_0001, 0x0016_0209, 0xd771_0002, 0x001a_040a,
    0xd771_0003, 0x001e_060b, 0xe01c_2000, 0x8000_0004, 0xbf81_0000,
];

fn fill(words: &[u32], rsrc2: u32) -> Option<BufferFill> {
    let interface = ShaderInterface { guest_pgm_rsrc2: Some(rsrc2), ..ShaderInterface::default() };
    Compiler::new(TargetIsa::Gfx115).compile_with_interface(words, CompileOptions::new(TargetIsa::Gfx115, ShaderStage::Compute, 64), &interface).unwrap().buffer_fill
}

#[test]
fn a_program_that_only_fills_is_recognised() {
    // Eight user SGPRs with the group's index along x behind them (COMPUTE_PGM_RSRC2 0x90).
    assert_eq!(fill(&FILL, 0x90), Some(BufferFill { buffer: 0, value: 4 }));
    // Without the group's index the record's place is not the thread's index over the dispatch.
    assert_eq!(fill(&FILL, 0x10), None);
    // Twelve user SGPRs; what it stores depends on what the buffer held.
    assert_eq!(fill(&MASKED, 0x98), None);
}
