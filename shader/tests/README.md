# Shader translation tests

`translation.rs` drives the public `shader::Compiler` API end to end: a gfx1013
compute shader is translated for gfx1013, gfx10.3, gfx11 and gfx12, and the
tests check target-specific code (MAC to FMAC, wait-counter repacking and
splitting), metadata, caching and error reporting. `xml_isa.rs` compares the
definitions generated from AMD's ISA XML with the opcode tables, and `llvm_mc.rs`
compares encodings with LLVM's output.

The differential tests that ran a translation next to its guest program on a CPU
interpreter were removed with the interpreter (decision D26); they are at commit
`0e31224`.

Per-stage unit tests live next to the code in `src/` (decode, cfg, lowering,
regalloc, encode, cache). Encodings for gfx11/gfx12 are checked against the
bit layouts implemented in `src/encode`; they still need hardware-derived
vectors before being treated as validated on RDNA3/RDNA4 devices.

## Stage ABI coverage

The native gfx11/gfx11.5 path retains the guest's merged hardware stages. It does
not split a wave64 hull shader into Vulkan invocations. The reference is Mesa
`radv_shader_args.c`, `ac_shader_args.h` and ACO's merged-stage EXEC setup in the
local reference repository at `801c9763c6043f0de8408e905a5324eea06d81d7`.
From the workspace root, those files are available with
`git -C references/mesa-wddm show HEAD:<path>` even when absent from the partial
reference working tree.

| Stage | Inputs and translated interface |
|---|---|
| CS | User SGPRs, enabled TGIDs, TG_SIZE; packed gfx11 local IDs unpacked for guest reads. Wave index is captured before guest code can overwrite TG_SIZE, including when it must live in a spill VGPR lane. Private buffer accesses use native scratch. |
| PS | Input VGPR allocation follows SPI_PS_INPUT_ENA/ADDR; prim_mask follows user SGPRs. In the ordinary PS ABI, the gfx10 scratch-wave-offset follows prim_mask and is replaced by native scratch accesses. Interpolation loads cover the required quad and restore guest EXEC. |
| Merged LS/HS | s2 offchip offset, s3 merged-wave info, s4 factor offset, s5 gfx11 wave index; v0/v1 hull patch inputs, v2 vertex ID, v3 guest local-thread index reconstructed from s5 and the lane index, v5 instance ID. LDS, factor-ring and offchip-ring accesses retain guest layout. |
| DS as NGG ES | Domain coordinates and patch inputs in v5-v8; s4 offchip offset. Domain code remains the ES portion of the merged ES/GS interface. |
| NGG ES/GS | Full EXEC initialized before guest code. s3 merged-wave info and s5 attribute offset captured before overwrite. Parameter exports become attribute-ring stores; stores finish before the done position export. At SGPR exhaustion, descriptor, offset and deferred EXEC live in persistent spill lanes. |

The spilled attribute descriptor is materialized once in the prologue. The
reported two ring-address literal relocations therefore cover every parameter
store, including a program exporting more than one parameter. Transient spill
blocks do not overlap the persistent lanes.

gfx12 merged LS/HS, legacy streamout and special PS system-input variants such as
POPS/provoking-vertex inputs remain outside this ABI. The runtime must retain scratch, LDS/offchip/factor-ring backing and
attribute-ring regions until the relevant GPU fences retire; shader compilation
and its cache key cannot make shared storage safe for concurrent queues.
