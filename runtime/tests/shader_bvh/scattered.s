.text
v_mov_b32 v60, v0
buffer_load_dwordx4 v[16:19], v60, s[0:3], 0 idxen offset:0
buffer_load_dwordx4 v[20:23], v60, s[0:3], 0 idxen offset:16
buffer_load_dwordx4 v[24:27], v60, s[0:3], 0 idxen offset:32
buffer_load_dwordx4 v[28:31], v60, s[0:3], 0 idxen offset:48
s_waitcnt vmcnt(0)
v_mov_b32 v40, v20
v_mov_b32 v45, v21
v_mov_b32 v41, v22
v_mov_b32 v50, v24
v_mov_b32 v52, v25
v_mov_b32 v51, v26
image_bvh_intersect_ray v[32:35], [v16, v18, v40, v45, v41, v50, v52, v51, v28, v29, v30], s[4:7]
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[32:35], v60, s[8:11], 0 idxen offset:0
s_endpgm
