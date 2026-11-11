.text
v_mov_b32 v31, v0
v_mov_b32 v167, 0x6fedcba9
s_mov_b32 s99, 0x12345678
buffer_load_dwordx4 v[16:19], v31, s[0:3], 0 idxen offset:0
buffer_load_dwordx4 v[20:23], v31, s[0:3], 0 idxen offset:16
buffer_load_dwordx4 v[24:27], v31, s[0:3], 0 idxen offset:32
s_waitcnt vmcnt(0)
v_mov_b32 v32, 0xdead0000
v_mov_b32 v33, 0xdead0001
v_mov_b32 v34, 0xdead0002
v_mov_b32 v35, 0xdead0003
v_mov_b32 v36, 0xdead0004
v_mov_b32 v37, 0xdead0005
v_mov_b32 v38, 0xdead0006
v_mov_b32 v39, 0xdead0007
v_mov_b32 v40, 0xdead0008
v_mov_b32 v41, 0xdead0009
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 8), 0x33
s_mov_b64 s[20:21], exec
v_and_b32 v62, 1, v31
v_cmp_eq_u32 vcc_lo, 0, v62
s_and_b64 exec, exec, vcc
s_mov_b32 vcc_lo, 0xa5a5a5a5
s_mov_b32 vcc_hi, 0x5a5a5a5a
s_cmp_eq_u32 s99, 0x12345678
.long 0xf1949f05, 0x00012010, 0x14131211, 0x00001a17
s_cselect_b32 s16, 1, 0
s_getreg_b32 s17, hwreg(HW_REG_MODE, 0, 8)
s_mov_b32 s18, vcc_lo
s_mov_b32 s19, vcc_hi
s_mov_b64 exec, s[20:21]
v_mov_b32 v48, s16
v_mov_b32 v49, s17
v_mov_b32 v50, s18
v_mov_b32 v51, s19
v_mov_b32 v52, s99
v_mov_b32 v53, v167
s_waitcnt vmcnt(0)
v_mov_b32 v42, v20
v_mov_b32 v43, v21
v_mov_b32 v44, v22
v_mov_b32 v45, v23
v_mov_b32 v46, v24
v_mov_b32 v47, v25
buffer_store_dwordx4 v[32:35], v31, s[8:11], 0 idxen offset:0
buffer_store_dwordx4 v[36:39], v31, s[8:11], 0 idxen offset:16
buffer_store_dwordx4 v[40:43], v31, s[8:11], 0 idxen offset:32
buffer_store_dwordx4 v[44:47], v31, s[8:11], 0 idxen offset:48
buffer_store_dwordx4 v[48:51], v31, s[8:11], 0 idxen offset:64
buffer_store_dwordx2 v[52:53], v31, s[8:11], 0 idxen offset:80
s_endpgm
