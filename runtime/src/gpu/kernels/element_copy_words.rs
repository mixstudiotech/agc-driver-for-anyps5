//! Machine words of `element_copy.s`, assembled by `driver/tools/kernels/element_copy.py`
//! with llvm-mc (LLVM 21.1.6, `-arch=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize64`).
//! Do not edit by hand.

pub const WORDS: [u32; 338] = [
    0xf410_0200, // s_load_dwordx16 s[8:23], s[0:1], 0x0
    0xfa00_0000,
    0xf410_0600, // s_load_dwordx16 s[24:39], s[0:1], 0x40
    0xfa00_0040,
    0xf410_0a00, // s_load_dwordx16 s[40:55], s[0:1], 0x80
    0xfa00_0080,
    0xf410_0e00, // s_load_dwordx16 s[56:71], s[0:1], 0xc0
    0xfa00_00c0,
    0xf408_1200, // s_load_dwordx4 s[72:75], s[0:1], 0x100
    0xfa00_0100,
    0xbf8c_c07f, // s_waitcnt lgkmcnt(0)
    0xbf09_4903, // s_cmp_ge_u32 s3, s73
    0xbf85_0144, // s_cbranch_scc1 done
    0xd746_0001, // v_lshl_add_u32 v1, s2, 6, v0
    0x0401_0c02,
    0x7da8_0248, // v_cmpx_gt_u32_e32 s72, v1
    0xbf88_0140, // s_cbranch_execz done
    0x874e_ff04, // s_and_b32 s78, s4, 0xff
    0x0000_00ff,
    0x8f4e_984e, // s_lshl_b32 s78, s78, 24
    0x4a06_0210, // v_add_nc_u32_e32 v3, s16, v1
    0x3606_06ff, // v_and_b32_e32 v3, 0xfff, v3
    0x0000_0fff,
    0x8050_1103, // s_add_u32 s80, s3, s17
    0x8750_ff50, // s_and_b32 s80, s80, 0xfff
    0x0000_0fff,
    0x8f50_8c50, // s_lshl_b32 s80, s80, 12
    0x8850_4e50, // s_or_b32 s80, s80, s78
    0x3806_0650, // v_or_b32_e32 v3, s80, v3
    0x7e0a_0280, // v_mov_b32_e32 v5, 0
    0x360c_0618, // v_and_b32_e32 v6, s24, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 0, v5
    0x0415_0106,
    0x360c_0619, // v_and_b32_e32 v6, s25, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 1, v5
    0x0415_0306,
    0x360c_061a, // v_and_b32_e32 v6, s26, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 2, v5
    0x0415_0506,
    0x360c_061b, // v_and_b32_e32 v6, s27, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 3, v5
    0x0415_0706,
    0x360c_061c, // v_and_b32_e32 v6, s28, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 4, v5
    0x0415_0906,
    0x360c_061d, // v_and_b32_e32 v6, s29, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 5, v5
    0x0415_0b06,
    0x360c_061e, // v_and_b32_e32 v6, s30, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 6, v5
    0x0415_0d06,
    0x360c_061f, // v_and_b32_e32 v6, s31, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 7, v5
    0x0415_0f06,
    0x360c_0620, // v_and_b32_e32 v6, s32, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 8, v5
    0x0415_1106,
    0x360c_0621, // v_and_b32_e32 v6, s33, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 9, v5
    0x0415_1306,
    0x360c_0622, // v_and_b32_e32 v6, s34, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 10, v5
    0x0415_1506,
    0x360c_0623, // v_and_b32_e32 v6, s35, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 11, v5
    0x0415_1706,
    0x360c_0624, // v_and_b32_e32 v6, s36, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 12, v5
    0x0415_1906,
    0x360c_0625, // v_and_b32_e32 v6, s37, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 13, v5
    0x0415_1b06,
    0x360c_0626, // v_and_b32_e32 v6, s38, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 14, v5
    0x0415_1d06,
    0x360c_0627, // v_and_b32_e32 v6, s39, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 15, v5
    0x0415_1f06,
    0x3a0a_0a12, // v_xor_b32_e32 v5, s18, v5
    0x2c0c_020d, // v_lshrrev_b32_e32 v6, s13, v1
    0x9050_0e03, // s_lshr_b32 s80, s3, s14
    0x9350_0c50, // s_mul_i32 s80, s80, s12
    0x4a0c_0c50, // v_add_nc_u32_e32 v6, s80, v6
    0x340c_0c0f, // v_lshlrev_b32_e32 v6, s15, v6
    0x4a0a_0b06, // v_add_nc_u32_e32 v5, v6, v5
    0x9350_0a04, // s_mul_i32 s80, s4, s10
    0x9ad1_0a04, // s_mul_hi_u32 s81, s4, s10
    0x9352_0b04, // s_mul_i32 s82, s4, s11
    0x8051_5251, // s_add_u32 s81, s81, s82
    0x8050_0850, // s_add_u32 s80, s80, s8
    0x8251_0951, // s_addc_u32 s81, s81, s9
    0xd70f_6a0a, // v_add_co_u32 v10, vcc, s80, v5
    0x0002_0a50,
    0x7e16_0251, // v_mov_b32_e32 v11, s81
    0x5016_1680, // v_add_co_ci_u32_e32 v11, vcc, 0, v11, vcc
    0x4a06_0230, // v_add_nc_u32_e32 v3, s48, v1
    0x3606_06ff, // v_and_b32_e32 v3, 0xfff, v3
    0x0000_0fff,
    0x8050_3103, // s_add_u32 s80, s3, s49
    0x8750_ff50, // s_and_b32 s80, s80, 0xfff
    0x0000_0fff,
    0x8f50_8c50, // s_lshl_b32 s80, s80, 12
    0x8850_4e50, // s_or_b32 s80, s80, s78
    0x3806_0650, // v_or_b32_e32 v3, s80, v3
    0x7e0a_0280, // v_mov_b32_e32 v5, 0
    0x360c_0638, // v_and_b32_e32 v6, s56, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 0, v5
    0x0415_0106,
    0x360c_0639, // v_and_b32_e32 v6, s57, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 1, v5
    0x0415_0306,
    0x360c_063a, // v_and_b32_e32 v6, s58, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 2, v5
    0x0415_0506,
    0x360c_063b, // v_and_b32_e32 v6, s59, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 3, v5
    0x0415_0706,
    0x360c_063c, // v_and_b32_e32 v6, s60, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 4, v5
    0x0415_0906,
    0x360c_063d, // v_and_b32_e32 v6, s61, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 5, v5
    0x0415_0b06,
    0x360c_063e, // v_and_b32_e32 v6, s62, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 6, v5
    0x0415_0d06,
    0x360c_063f, // v_and_b32_e32 v6, s63, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 7, v5
    0x0415_0f06,
    0x360c_0640, // v_and_b32_e32 v6, s64, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 8, v5
    0x0415_1106,
    0x360c_0641, // v_and_b32_e32 v6, s65, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 9, v5
    0x0415_1306,
    0x360c_0642, // v_and_b32_e32 v6, s66, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 10, v5
    0x0415_1506,
    0x360c_0643, // v_and_b32_e32 v6, s67, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 11, v5
    0x0415_1706,
    0x360c_0644, // v_and_b32_e32 v6, s68, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 12, v5
    0x0415_1906,
    0x360c_0645, // v_and_b32_e32 v6, s69, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 13, v5
    0x0415_1b06,
    0x360c_0646, // v_and_b32_e32 v6, s70, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 14, v5
    0x0415_1d06,
    0x360c_0647, // v_and_b32_e32 v6, s71, v3
    0xd764_0006, // v_bcnt_u32_b32 v6, v6, 0
    0x0001_0106,
    0x360c_0c81, // v_and_b32_e32 v6, 1, v6
    0xd76f_0005, // v_lshl_or_b32 v5, v6, 15, v5
    0x0415_1f06,
    0x3a0a_0a32, // v_xor_b32_e32 v5, s50, v5
    0x2c0c_022d, // v_lshrrev_b32_e32 v6, s45, v1
    0x9050_2e03, // s_lshr_b32 s80, s3, s46
    0x9350_2c50, // s_mul_i32 s80, s80, s44
    0x4a0c_0c50, // v_add_nc_u32_e32 v6, s80, v6
    0x340c_0c2f, // v_lshlrev_b32_e32 v6, s47, v6
    0x4a0a_0b06, // v_add_nc_u32_e32 v5, v6, v5
    0x9350_2a04, // s_mul_i32 s80, s4, s42
    0x9ad1_2a04, // s_mul_hi_u32 s81, s4, s42
    0x9352_2b04, // s_mul_i32 s82, s4, s43
    0x8051_5251, // s_add_u32 s81, s81, s82
    0x8050_2850, // s_add_u32 s80, s80, s40
    0x8251_2951, // s_addc_u32 s81, s81, s41
    0xd70f_6a0e, // v_add_co_u32 v14, vcc, s80, v5
    0x0002_0a50,
    0x7e1e_0251, // v_mov_b32_e32 v15, s81
    0x501e_1e80, // v_add_co_ci_u32_e32 v15, vcc, 0, v15, vcc
    0xbf0d_804b, // s_bitcmp1_b32 s75, 0
    0xbf85_002a, // s_cbranch_scc1 fill
    0xbf0d_814b, // s_bitcmp1_b32 s75, 1
    0xbf84_0002, // s_cbranch_scc0 copy_value
    0x3614_14ff, // v_and_b32_e32 v10, 0xffffff00, v10
    0xffff_ff00,
    // copy_value:
    0xbf06_804a, // s_cmp_eq_u32 s74, 0
    0xbf85_001e, // s_cbranch_scc1 copy1
    0xbf06_814a, // s_cmp_eq_u32 s74, 1
    0xbf85_0016, // s_cbranch_scc1 copy2
    0xbf06_824a, // s_cmp_eq_u32 s74, 2
    0xbf85_000e, // s_cbranch_scc1 copy4
    0xbf06_834a, // s_cmp_eq_u32 s74, 3
    0xbf85_0006, // s_cbranch_scc1 copy8
    0xdc38_8000, // global_load_dwordx4 v[16:19], v[10:11], off
    0x107d_000a,
    0xbf8c_3f70, // s_waitcnt vmcnt(0)
    0xdc78_8000, // global_store_dwordx4 v[14:15], v[16:19], off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // copy8:
    0xdc34_8000, // global_load_dwordx2 v[16:17], v[10:11], off
    0x107d_000a,
    0xbf8c_3f70, // s_waitcnt vmcnt(0)
    0xdc74_8000, // global_store_dwordx2 v[14:15], v[16:17], off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // copy4:
    0xdc30_8000, // global_load_dword v16, v[10:11], off
    0x107d_000a,
    0xbf8c_3f70, // s_waitcnt vmcnt(0)
    0xdc70_8000, // global_store_dword v[14:15], v16, off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // copy2:
    0xdc28_8000, // global_load_ushort v16, v[10:11], off
    0x107d_000a,
    0xbf8c_3f70, // s_waitcnt vmcnt(0)
    0xdc68_8000, // global_store_short v[14:15], v16, off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // copy1:
    0xdc20_8000, // global_load_ubyte v16, v[10:11], off
    0x107d_000a,
    0xbf8c_3f70, // s_waitcnt vmcnt(0)
    0xdc60_8000, // global_store_byte v[14:15], v16, off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // fill:
    0x7e20_0213, // v_mov_b32_e32 v16, s19
    0x7e22_0214, // v_mov_b32_e32 v17, s20
    0x7e24_0215, // v_mov_b32_e32 v18, s21
    0x7e26_0216, // v_mov_b32_e32 v19, s22
    0xbf06_804a, // s_cmp_eq_u32 s74, 0
    0xbf85_0012, // s_cbranch_scc1 fill1
    0xbf06_814a, // s_cmp_eq_u32 s74, 1
    0xbf85_000d, // s_cbranch_scc1 fill2
    0xbf06_824a, // s_cmp_eq_u32 s74, 2
    0xbf85_0008, // s_cbranch_scc1 fill4
    0xbf06_834a, // s_cmp_eq_u32 s74, 3
    0xbf85_0003, // s_cbranch_scc1 fill8
    0xdc78_8000, // global_store_dwordx4 v[14:15], v[16:19], off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // fill8:
    0xdc74_8000, // global_store_dwordx2 v[14:15], v[16:17], off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // fill4:
    0xdc70_8000, // global_store_dword v[14:15], v16, off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // fill2:
    0xdc68_8000, // global_store_short v[14:15], v16, off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // fill1:
    0xdc60_8000, // global_store_byte v[14:15], v16, off
    0x007d_100e,
    0xbf81_0000, // s_endpgm
    // done:
    0xbf81_0000, // s_endpgm
];
