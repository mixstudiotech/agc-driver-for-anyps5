import pathlib
import struct
import subprocess
import tempfile

MC = r"D:\projects\apple-llvm-project\.build-amdgpu\bin\llvm-mc.exe"
ROOT = pathlib.Path(__file__).resolve().parent


def words(assembly):
    with tempfile.TemporaryDirectory() as directory:
        obj = pathlib.Path(directory) / "shader.o"
        subprocess.run([MC, "-arch=amdgcn", "-mcpu=gfx1013", "-mattr=+wavefrontsize64", "-filetype=obj", "-o", str(obj)], input=assembly.encode(), check=True)
        data = obj.read_bytes()
    offset = struct.unpack_from("<Q", data, 0x28)[0]
    stride, count, strings = struct.unpack_from("<HHH", data, 0x3a)
    sections = [struct.unpack_from("<IIQQQQ", data, offset + i * stride) for i in range(count)]
    names = sections[strings][4]
    for name, _, _, _, start, size in sections:
        if data[names + name:data.index(b"\0", names + name)] == b".text":
            return struct.unpack_from(f"<{size // 4}I", data, start)
    raise RuntimeError("no text section")


output = ["// Generated with generate.py using llvm-mc gfx1013."]
for kind in ["image", "buffer"]:
    for op in ["fmin", "fmax"]:
        for returning in [False, True]:
            instruction = (f"image_atomic_{op} v2, v1, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D" if kind == "image" else f"buffer_atomic_{op} v2, v1, s[0:3], 0 idxen") + (" glc" if returning else "")
            assembly = f"""
s_mov_b32 s20, 0xffffffff
s_mov_b32 s21, 0xa5a55aa5
s_and_b64 exec, exec, s[20:21]
buffer_load_dword v2, v0, s[8:11], 0 idxen
v_mov_b32 v1, v0
s_mov_b32 s16, 0x13579bdf
s_mov_b32 s17, 0x2468ace0
s_mov_b64 vcc, s[16:17]
s_cmp_eq_u32 s0, s0
s_waitcnt vmcnt(0)
{instruction}
s_waitcnt vmcnt(0)
s_mov_b64 s[16:17], vcc
s_cselect_b32 s18, 1, 0
v_mov_b32 v3, s16
v_mov_b32 v4, s17
v_mov_b32 v5, s18
buffer_store_dwordx4 v[2:5], v0, s[12:15], 0 idxen
s_waitcnt_vscnt null, 0
s_endpgm
"""
            name = f"{kind}_{op}_{int(returning)}".upper()
            binary = words(assembly)
            output.append(f"pub const {name}: &[u32] = &[")
            for i in range(0, len(binary), 6):
                output.append("    " + ", ".join(f"0x{x:08x}" for x in binary[i:i+6]) + ",")
            output.append("];\n")
ROOT.joinpath("kernels.rs").write_text("\n".join(output) + "\n", encoding="utf-8")
