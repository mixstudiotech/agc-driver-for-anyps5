.text
v_mov_b32 v60, v0
buffer_load_dwordx4 v[16:19], v60, s[0:3], 0 idxen offset:0
buffer_load_dwordx4 v[20:23], v60, s[0:3], 0 idxen offset:16
buffer_load_dwordx4 v[24:27], v60, s[0:3], 0 idxen offset:32
buffer_load_dwordx4 v[28:31], v60, s[0:3], 0 idxen offset:48
s_waitcnt vmcnt(0)
v_cvt_f16_f32 v36, v24
v_cvt_f16_f32 v37, v25
v_cvt_f16_f32 v38, v26
v_cvt_f16_f32 v39, v28
v_cvt_f16_f32 v40, v29
v_cvt_f16_f32 v41, v30
v_and_b32 v36, 0xffff, v36
v_and_b32 v38, 0xffff, v38
v_and_b32 v40, 0xffff, v40
v_lshlrev_b32 v37, 16, v37
v_lshlrev_b32 v39, 16, v39
v_lshlrev_b32 v41, 16, v41
v_or_b32 v44, v36, v37
v_or_b32 v45, v38, v39
v_or_b32 v46, v40, v41
image_bvh_intersect_ray v[32:35], [v16, v18, v20, v21, v22, v44, v45, v46], s[4:7] a16
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[32:35], v60, s[8:11], 0 idxen offset:0
s_endpgm
