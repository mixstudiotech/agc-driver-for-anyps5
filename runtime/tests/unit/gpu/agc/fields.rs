use super::*;

#[test]
fn ordered_pixel_shading_is_programmed_as_gfx11_takes_it() {
    const PRIMITIVE_ORDERED_PIXEL_SHADER: u32 = 1 << 16;
    const EXEC_IF_OVERLAPPED: u32 = 1 << 17;
    const OVERRIDE_INTRINSIC_RATE_ENABLE: u32 = 1 << 25;
    // A value the title's ordered pixel programs carry: Z order, kill, EXEC_ON_NOOP, the mode
    // and EXEC_IF_OVERLAPPED. The mode stays and needs the intrinsic rate override enabled.
    let guest = 0x3_0470;
    assert_eq!(guest & (PRIMITIVE_ORDERED_PIXEL_SHADER | EXEC_IF_OVERLAPPED), 0x3_0000);
    assert_eq!(db_shader_control(guest), Ok(0x1_0470 | OVERRIDE_INTRINSIC_RATE_ENABLE));
    // POPS_OVERLAP_NUM_SAMPLES [22:20] is the rate, at [28:26].
    assert_eq!(db_shader_control(PRIMITIVE_ORDERED_PIXEL_SHADER | 2 << 20), Ok(PRIMITIVE_ORDERED_PIXEL_SHADER | OVERRIDE_INTRINSIC_RATE_ENABLE | 2 << 26));
    // Without the mode nothing is added, and the bits gfx11 lacks are dropped.
    assert_eq!(db_shader_control(0x50 | EXEC_IF_OVERLAPPED | 3 << 20 | 1 << 7), Ok(0x50));
    // The rasterizer is not asked for collision inputs the translated programs do not take.
    assert_eq!(pa_sc_shader_control(0b0110_1111), Ok(0b0110_0011));
}

#[test]
fn tessellation_rings_are_split_between_the_shader_engines() {
    // A 16 MB off-chip ring in blocks of 8192 dwords: 512 blocks on the PS5, where the
    // register counts all of them, and 256 for each of two shader engines on gfx11.
    assert_eq!(vgt_hs_offchip_param(511, 2), Ok(255));
    // The granularity moves up a bit: 4 MB in blocks of 4096 dwords.
    assert_eq!(vgt_hs_offchip_param(255 | 1 << 9, 2), Ok(127 | 1 << 10));
    assert_eq!(vgt_hs_offchip_param(0, 2), Ok(0));
    // A 128 KiB factor ring is 32768 dwords, half for each engine.
    assert_eq!(vgt_tf_ring_size(0x8000, 2), Ok(0x4000));
}
