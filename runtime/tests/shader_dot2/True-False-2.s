s_mov_b64 exec, 1
s_mov_b64 vcc, s[4:5]
s_cmp_eq_u32 s3, 1
v_mov_b32 v0, s0
v_mov_b32 v1, s1
v_mov_b32 v2, s2
v_dot2_i32_i16 v2, v0, v1, v2
s_cselect_b32 s6, 1, 0
v_mov_b32 v7, v2
v_mov_b32 v8, s6
v_mov_b32 v9, vcc_lo
v_mov_b32 v10, vcc_hi
buffer_store_dwordx4 v[7:10], off, s[8:11], 0
s_endpgm
