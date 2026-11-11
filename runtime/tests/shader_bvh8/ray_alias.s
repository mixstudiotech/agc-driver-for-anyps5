.text
v_mov_b32 v31, v0
buffer_load_dwordx4 v[16:19], v31, s[0:3], 0 idxen offset:0
buffer_load_dwordx4 v[20:23], v31, s[0:3], 0 idxen offset:16
buffer_load_dwordx4 v[24:27], v31, s[0:3], 0 idxen offset:32
s_waitcnt vmcnt(0)
v_mov_b32 v0, v23
v_mov_b32 v1, v24
v_mov_b32 v2, v25
.long 0xf1949f05, 0x00010010, 0x14131211, 0x00001a00
v_mov_b32 v32, v0
v_mov_b32 v33, v1
v_mov_b32 v34, v2
v_mov_b32 v35, v3
v_mov_b32 v36, v4
v_mov_b32 v37, v5
v_mov_b32 v38, v6
v_mov_b32 v39, v7
v_mov_b32 v40, v8
v_mov_b32 v41, v9
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
s_endpgm
