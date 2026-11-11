.text
v_mov_b32 v60, v0
buffer_load_dwordx4 v[16:19], v60, s[0:3], 0 idxen offset:0
buffer_load_dwordx4 v[20:23], v60, s[0:3], 0 idxen offset:16
buffer_load_dwordx4 v[24:27], v60, s[0:3], 0 idxen offset:32
buffer_load_dwordx4 v[28:31], v60, s[0:3], 0 idxen offset:48
s_waitcnt vmcnt(0)
image_bvh64_intersect_ray v[32:35], [v16, v17, v18, v20, v21, v22, v24, v25, v26, v28, v29, v30], s[4:7]
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[32:35], v60, s[8:11], 0 idxen offset:0
s_endpgm
