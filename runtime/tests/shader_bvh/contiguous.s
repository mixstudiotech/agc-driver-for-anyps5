.text
v_mov_b32 v60, v0
buffer_load_dwordx4 v[16:19], v60, s[0:3], 0 idxen offset:0
buffer_load_dwordx4 v[20:23], v60, s[0:3], 0 idxen offset:16
buffer_load_dwordx4 v[24:27], v60, s[0:3], 0 idxen offset:32
buffer_load_dwordx4 v[28:31], v60, s[0:3], 0 idxen offset:48
s_waitcnt vmcnt(0)
v_mov_b32 v40, v16
v_mov_b32 v41, v18
v_mov_b32 v42, v20
v_mov_b32 v43, v21
v_mov_b32 v44, v22
v_mov_b32 v45, v24
v_mov_b32 v46, v25
v_mov_b32 v47, v26
v_mov_b32 v48, v28
v_mov_b32 v49, v29
v_mov_b32 v50, v30
image_bvh_intersect_ray v[48:51], v[40:50], s[4:7]
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[48:51], v60, s[8:11], 0 idxen offset:0
s_endpgm
