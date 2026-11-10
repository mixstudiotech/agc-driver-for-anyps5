s_mov_b64 exec, 1
s_cmp_eq_u32 s2, 1
s_ff0_i32_b64 s3, s[0:1]
s_cselect_b32 s4, 1, 0
v_mov_b32 v0, s3
v_mov_b32 v1, s4
buffer_store_dwordx2 v[0:1], off, s[8:11], 0
s_endpgm
