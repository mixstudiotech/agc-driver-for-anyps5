"""Regenerate the SDK-independent ISA fixtures with an AMDGPU-enabled LLVM MC."""
import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

ROOT = Path(__file__).parent
SIZES = [8, 13, 14, 15, 16, 18, 20, 30, 31]


def stack_source(size):
    lines = [".text", "v_mov_b32 v60, v0", "s_mov_b32 s99, 0x6fedcba9"]
    for offset in range(0, 40, 4):
        lines += [f"buffer_load_dwordx4 v[{10+offset}:{13+offset}], v60, s[0:3], 0 idxen offset:{offset*4}"]
    lines += ["buffer_load_dword v50, v60, s[0:3], 0 idxen offset:160", "s_waitcnt vmcnt(0)",
              "v_bfe_u32 v52, v10, 15, 14", "v_lshlrev_b32 v52, 2, v52"]
    lines += [f"ds_write_b32 v52, v{20+slot} offset:{128*slot}" for slot in range(size)]
    lines += ["s_waitcnt lgkmcnt(0)", "v_mov_b32 v51, 0xdeadbeef", "s_mov_b64 s[20:21], exec",
              "s_and_b64 exec, exec, s[4:5]", "s_mov_b32 vcc_lo, 0xa5a5a5a5",
              "s_mov_b32 vcc_hi, 0x5a5a5a5a", "s_cmp_eq_u32 s6, 1",
              f".long 0xdb840000 | {size}, 0x330c0b0a", "s_cselect_b32 s16, 1, 0",
              "s_mov_b64 exec, s[20:21]", "v_mov_b32 v54, s16", "v_mov_b32 v55, vcc_lo",
              "v_mov_b32 v56, vcc_hi", "v_mov_b32 v57, s99"]
    for register, offset in [(10, 0), (51, 4), (54, 8), (55, 12), (56, 16), (57, 20+size*4)]:
        lines += [f"buffer_store_dword v{register}, v60, s[8:11], 0 idxen offset:{offset}"]
    for slot in range(size):
        lines += [f"ds_read_b32 v53, v52 offset:{128*slot}", "s_waitcnt lgkmcnt(0)",
                  f"buffer_store_dword v53, v60, s[8:11], 0 idxen offset:{20+slot*4}"]
    return "\n".join(lines+["s_endpgm", ""])


def assemble(mc, source, directory):
    asm = directory / "kernel.s"
    obj = directory / "kernel.o"
    asm.write_text(source)
    subprocess.run([mc, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1030", "-filetype=obj", str(asm), "-o", str(obj)], check=True)
    elf = obj.read_bytes()
    assert elf[:5] == b"\x7fELF\x02", "expected ELF64"
    offset = struct.unpack_from("<Q", elf, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", elf, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", elf, offset+i*size) for i in range(count)]
    names = sections[names_index]
    names = elf[names[4]:names[4]+names[5]]
    text = next(s for s in sections if names[s[0]:].split(b"\0")[0] == b".text")
    data = elf[text[4]:text[4]+text[5]]
    return list(struct.unpack("<"+"I"*(len(data)//4), data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm-mc", default=os.environ.get("ANYPS5_LLVM_MC"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.llvm_mc:
        parser.error("set ANYPS5_LLVM_MC or pass --llvm-mc")
    sources = [("BVH8", (ROOT/"ray.s").read_text())]
    sources += [(f"STACK{size}", stack_source(size)) for size in SIZES]
    sources += [(name, (ROOT/file).read_text()) for name, file in [
        ("BVH8_ALIAS", "ray_alias.s"), ("BVH8_BASE_ALIAS", "ray_base_alias.s"), ("BVH8_STATE", "ray_state.s")]]
    with tempfile.TemporaryDirectory(prefix="anyps5-bvh8-") as directory:
        generated = {name: assemble(args.llvm_mc, source, Path(directory)) for name, source in sources}
    path = ROOT/"kernels.rs"
    if args.check:
        current = {name: [int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", body)]
                   for name, body in re.findall(r"pub const (\w+): &\[u32\] = &\[(.*?)\];", path.read_text(), re.S)}
        assert current == generated, "fixture words differ; run generate.py to regenerate"
        print(f"verified {len(generated)} LLVM MC fixtures")
        return
    result = "// LLVM MC gfx1030 fixtures; Pro opcodes are embedded as raw words.\n"
    for name, values in generated.items():
        result += f"pub const {name}: &[u32] = &[\n"
        result += "".join("    "+", ".join(f"0x{x:08x}" for x in values[i:i+8])+",\n" for i in range(0, len(values), 8))
        result += "];\n"
    path.write_text(result)


if __name__ == "__main__":
    main()
