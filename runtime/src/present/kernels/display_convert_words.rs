//! Machine words of `display_convert.s`, assembled by
//! `driver/tools/present/display_convert.py` with llvm-mc (LLVM 21.1.6, `-arch=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize64`).
//! Do not edit by hand.

pub const WORDS: [u32; 275] = [
    0xf410_0200, // s_load_dwordx16 s[8:23], s[0:1], 0x0
    0xfa00_0000,
    0xf410_0600, // s_load_dwordx16 s[24:39], s[0:1], 0x40
    0xfa00_0040,
    0xf40c_0a00, // s_load_dwordx8 s[40:47], s[0:1], 0x80
    0xfa00_0080,
    0xbf8c_c07f, // s_waitcnt lgkmcnt(0)
    0xbf09_2403, // s_cmp_ge_u32 s3, s36
    0xbf85_0106, // s_cbranch_scc1 done
    0xd746_0001, // v_lshl_add_u32 v1, s2, 6, v0
    0x0401_0c02,
    0x7da8_0223, // v_cmpx_gt_u32_e32 s35, v1
    0xbf88_0102, // s_cbranch_execz done
    0x9330_2203, // s_mul_i32 s48, s3, s34
    0x9ab1_2203, // s_mul_hi_u32 s49, s3, s34
    0x8030_2030, // s_add_u32 s48, s48, s32
    0x8231_2131, // s_addc_u32 s49, s49, s33
    0x3404_0282, // v_lshlrev_b32_e32 v2, 2, v1
    0xd70f_6a0a, // v_add_co_u32 v10, vcc, s48, v2
    0x0002_0430,
    0x7e16_0231, // v_mov_b32_e32 v11, s49
    0x5016_1680, // v_add_co_ci_u32_e32 v11, vcc, 0, v11, vcc
    0x7e18_022a, // v_mov_b32_e32 v12, s42
    0x8738_812f, // s_and_b32 s56, s47, 1
    0x80b2_2603, // s_sub_u32 s50, s3, s38
    0xbf09_2832, // s_cmp_ge_u32 s50, s40
    0xbf85_00f5, // s_cbranch_scc1 outside_row
    0x4e06_0225, // v_subrev_nc_u32_e32 v3, s37, v1
    0xbf06_8038, // s_cmp_eq_u32 s56, 0
    0xbf85_0002, // s_cbranch_scc1 sample
    0x7da8_0627, // v_cmpx_gt_u32_e32 s39, v3
    0xbf88_00ef, // s_cbranch_execz done
    // sample:
    0xbf06_840f, // s_cmp_eq_u32 s15, 4
    0xbf85_00b0, // s_cbranch_scc1 fill
    0x8034_c127, // s_add_u32 s52, s39, -1
    0x2608_0634, // v_min_u32_e32 v4, s52, v3
    0xd569_0004, // v_mul_lo_u32 v4, v4, s43
    0x0000_5704,
    0x9033_812b, // s_lshr_b32 s51, s43, 1
    0x4a08_0833, // v_add_nc_u32_e32 v4, s51, v4
    0x2c08_0890, // v_lshrrev_b32_e32 v4, 16, v4
    0x8035_c12d, // s_add_u32 s53, s45, -1
    0x2608_0835, // v_min_u32_e32 v4, s53, v4
    0x9336_2c32, // s_mul_i32 s54, s50, s44
    0x9037_812c, // s_lshr_b32 s55, s44, 1
    0x8036_3736, // s_add_u32 s54, s54, s55
    0x9036_9036, // s_lshr_b32 s54, s54, 16
    0x8037_c12e, // s_add_u32 s55, s46, -1
    0x83b6_3736, // s_min_u32 s54, s54, s55
    0x360a_08ff, // v_and_b32_e32 v5, 0xfff, v4
    0x0000_0fff,
    0x8737_ff36, // s_and_b32 s55, s54, 0xfff
    0x0000_0fff,
    0x8f37_8c37, // s_lshl_b32 s55, s55, 12
    0x380a_0a37, // v_or_b32_e32 v5, s55, v5
    0x7e0c_0280, // v_mov_b32_e32 v6, 0
    0x360e_0a10, // v_and_b32_e32 v7, s16, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 0, v6
    0x0419_0107,
    0x360e_0a11, // v_and_b32_e32 v7, s17, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 1, v6
    0x0419_0307,
    0x360e_0a12, // v_and_b32_e32 v7, s18, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 2, v6
    0x0419_0507,
    0x360e_0a13, // v_and_b32_e32 v7, s19, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 3, v6
    0x0419_0707,
    0x360e_0a14, // v_and_b32_e32 v7, s20, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 4, v6
    0x0419_0907,
    0x360e_0a15, // v_and_b32_e32 v7, s21, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 5, v6
    0x0419_0b07,
    0x360e_0a16, // v_and_b32_e32 v7, s22, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 6, v6
    0x0419_0d07,
    0x360e_0a17, // v_and_b32_e32 v7, s23, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 7, v6
    0x0419_0f07,
    0x360e_0a18, // v_and_b32_e32 v7, s24, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 8, v6
    0x0419_1107,
    0x360e_0a19, // v_and_b32_e32 v7, s25, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 9, v6
    0x0419_1307,
    0x360e_0a1a, // v_and_b32_e32 v7, s26, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 10, v6
    0x0419_1507,
    0x360e_0a1b, // v_and_b32_e32 v7, s27, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 11, v6
    0x0419_1707,
    0x360e_0a1c, // v_and_b32_e32 v7, s28, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 12, v6
    0x0419_1907,
    0x360e_0a1d, // v_and_b32_e32 v7, s29, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 13, v6
    0x0419_1b07,
    0x360e_0a1e, // v_and_b32_e32 v7, s30, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 14, v6
    0x0419_1d07,
    0x360e_0a1f, // v_and_b32_e32 v7, s31, v5
    0xd764_0007, // v_bcnt_u32_b32 v7, v7, 0
    0x0001_0107,
    0x360e_0e81, // v_and_b32_e32 v7, 1, v7
    0xd76f_0006, // v_lshl_or_b32 v6, v7, 15, v6
    0x0419_1f07,
    0x3a0c_0c0e, // v_xor_b32_e32 v6, s14, v6
    0x2c0e_080b, // v_lshrrev_b32_e32 v7, s11, v4
    0x9037_0c36, // s_lshr_b32 s55, s54, s12
    0x9337_0a37, // s_mul_i32 s55, s55, s10
    0x4a0e_0e37, // v_add_nc_u32_e32 v7, s55, v7
    0x340e_0e0d, // v_lshlrev_b32_e32 v7, s13, v7
    0x4a0c_0d07, // v_add_nc_u32_e32 v6, v7, v6
    0xd70f_6a08, // v_add_co_u32 v8, vcc, s8, v6
    0x0002_0c08,
    0x7e12_0209, // v_mov_b32_e32 v9, s9
    0x5012_1280, // v_add_co_ci_u32_e32 v9, vcc, 0, v9, vcc
    0xdc30_8000, // global_load_dword v13, v[8:9], off
    0x0d7d_0008,
    0xbf8c_3f70, // s_waitcnt vmcnt(0)
    0xbf06_810f, // s_cmp_eq_u32 s15, 1
    0xbf85_0006, // s_cbranch_scc1 swap
    0xbf06_820f, // s_cmp_eq_u32 s15, 2
    0xbf85_000e, // s_cbranch_scc1 ten_red_low
    0xbf06_830f, // s_cmp_eq_u32 s15, 3
    0xbf85_0019, // s_cbranch_scc1 ten_red_high
    0x7e1c_030d, // v_mov_b32_e32 v14, v13
    0xbf82_0025, // s_branch inside
    // swap:
    0x361c_1aff, // v_and_b32_e32 v14, 0xff00ff00, v13
    0xff00_ff00,
    0xd548_000f, // v_bfe_u32 v15, v13, 16, 8
    0x0221_210d,
    0x381c_1f0e, // v_or_b32_e32 v14, v14, v15
    0x361e_1aff, // v_and_b32_e32 v15, 0xff, v13
    0x0000_00ff,
    0xd76f_000e, // v_lshl_or_b32 v14, v15, 16, v14
    0x0439_210f,
    0xbf82_001b, // s_branch inside
    // ten_red_low:
    0xd548_000e, // v_bfe_u32 v14, v13, 22, 8
    0x0221_2d0d,
    0xd548_000f, // v_bfe_u32 v15, v13, 12, 8
    0x0221_190d,
    0xd76f_000e, // v_lshl_or_b32 v14, v15, 8, v14
    0x0439_110f,
    0xd548_000f, // v_bfe_u32 v15, v13, 2, 8
    0x0221_050d,
    0xd76f_000e, // v_lshl_or_b32 v14, v15, 16, v14
    0x0439_210f,
    0x381c_1cff, // v_or_b32_e32 v14, 0xff000000, v14
    0xff00_0000,
    0xbf82_000e, // s_branch inside
    // ten_red_high:
    0xd548_000e, // v_bfe_u32 v14, v13, 2, 8
    0x0221_050d,
    0xd548_000f, // v_bfe_u32 v15, v13, 12, 8
    0x0221_190d,
    0xd76f_000e, // v_lshl_or_b32 v14, v15, 8, v14
    0x0439_110f,
    0xd548_000f, // v_bfe_u32 v15, v13, 22, 8
    0x0221_2d0d,
    0xd76f_000e, // v_lshl_or_b32 v14, v15, 16, v14
    0x0439_210f,
    0x381c_1cff, // v_or_b32_e32 v14, 0xff000000, v14
    0xff00_0000,
    0xbf82_0001, // s_branch inside
    // fill:
    0x7e1c_0229, // v_mov_b32_e32 v14, s41
    // inside:
    0xbf06_8038, // s_cmp_eq_u32 s56, 0
    0xbf84_0003, // s_cbranch_scc0 blend
    0x7d88_0627, // v_cmp_gt_u32_e32 vcc, s39, v3
    0x0218_1d0c, // v_cndmask_b32_e32 v12, v12, v14, vcc
    0xbf82_0035, // s_branch store
    // blend:
    0xdc30_8000, // global_load_dword v16, v[10:11], off
    0x107d_000a,
    0xbf8c_3f70, // s_waitcnt vmcnt(0)
    0x2c22_1c98, // v_lshrrev_b32_e32 v17, 24, v14
    0x4c22_22ff, // v_sub_nc_u32_e32 v17, 0xff, v17
    0x0000_00ff,
    0x3618_20ff, // v_and_b32_e32 v12, 0xff000000, v16
    0xff00_0000,
    0xd548_0012, // v_bfe_u32 v18, v16, 0, 8
    0x0221_0110,
    0x1624_2312, // v_mul_u32_u24_e32 v18, v18, v17
    0x4a24_24ff, // v_add_nc_u32_e32 v18, 0x7f, v18
    0x0000_007f,
    0x1624_24ff, // v_mul_u32_u24_e32 v18, 0x8081, v18
    0x0000_8081,
    0x2c24_2497, // v_lshrrev_b32_e32 v18, 23, v18
    0xd548_0013, // v_bfe_u32 v19, v14, 0, 8
    0x0221_010e,
    0x4a24_2513, // v_add_nc_u32_e32 v18, v19, v18
    0x2624_24ff, // v_min_u32_e32 v18, 0xff, v18
    0x0000_00ff,
    0xd76f_000c, // v_lshl_or_b32 v12, v18, 0, v12
    0x0431_0112,
    0xd548_0012, // v_bfe_u32 v18, v16, 8, 8
    0x0221_1110,
    0x1624_2312, // v_mul_u32_u24_e32 v18, v18, v17
    0x4a24_24ff, // v_add_nc_u32_e32 v18, 0x7f, v18
    0x0000_007f,
    0x1624_24ff, // v_mul_u32_u24_e32 v18, 0x8081, v18
    0x0000_8081,
    0x2c24_2497, // v_lshrrev_b32_e32 v18, 23, v18
    0xd548_0013, // v_bfe_u32 v19, v14, 8, 8
    0x0221_110e,
    0x4a24_2513, // v_add_nc_u32_e32 v18, v19, v18
    0x2624_24ff, // v_min_u32_e32 v18, 0xff, v18
    0x0000_00ff,
    0xd76f_000c, // v_lshl_or_b32 v12, v18, 8, v12
    0x0431_1112,
    0xd548_0012, // v_bfe_u32 v18, v16, 16, 8
    0x0221_2110,
    0x1624_2312, // v_mul_u32_u24_e32 v18, v18, v17
    0x4a24_24ff, // v_add_nc_u32_e32 v18, 0x7f, v18
    0x0000_007f,
    0x1624_24ff, // v_mul_u32_u24_e32 v18, 0x8081, v18
    0x0000_8081,
    0x2c24_2497, // v_lshrrev_b32_e32 v18, 23, v18
    0xd548_0013, // v_bfe_u32 v19, v14, 16, 8
    0x0221_210e,
    0x4a24_2513, // v_add_nc_u32_e32 v18, v19, v18
    0x2624_24ff, // v_min_u32_e32 v18, 0xff, v18
    0x0000_00ff,
    0xd76f_000c, // v_lshl_or_b32 v12, v18, 16, v12
    0x0431_2112,
    // store:
    0xdc70_8000, // global_store_dword v[10:11], v12, off
    0x007d_0c0a,
    // done:
    0xbf81_0000, // s_endpgm
    // outside_row:
    0xbf06_8038, // s_cmp_eq_u32 s56, 0
    0xbf85_fffb, // s_cbranch_scc1 store
    0xbf81_0000, // s_endpgm
];
