.text
s_mov_b32 s99, 0
s_getpc_b64 s[6:7]
s_add_u32 s0, s6, s0
s_addc_u32 s1, s7, s1
s_cmp_eq_u32 s2, 0
s_setpc_b64 s[0:1]
s_endpgm
target:
s_cbranch_scc0 .Lscc0
v_mov_b32 v1, 17
s_branch .Lstore
.Lscc0:
v_mov_b32 v1, 34
.Lstore:
v_lshlrev_b32 v2, 2, v0
global_store_dword v2, v1, s[4:5]
s_endpgm
