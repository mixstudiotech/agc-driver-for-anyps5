# AnyPS5 GPU Driver

A Rust implementation of the PS5 Agc libraries that records native host GPU commands. The AMD runtime targets RDNA 3.5 (`gfx11.5`) and RDNA 4 (`gfx12`).

## Implementation goals

**The ultimate goal is to run the PS5 version of GTA6 smoothly through AnyPS5 on AMD GPUs under Windows and Linux.**

The design takes PS5's backward compatibility with PS4 as a conceptual model for **native compatibility across AMD GPU generations**. Alongside the other AnyPS5 runtime components, the driver preserves the game's AGC behavior and memory semantics while adapting commands, shader ISA and resource layouts for execution on the host GPU.

Each target handles its generation's PM4 encoding, shader lowering, register state and resource layouts. The shared AGC front end retains the game's programming model, while the target maps operations onto the host GPU's native capabilities. The current AMD implementation targets RDNA 3.5 (`gfx1151`) and RDNA 4 (`gfx1200`). Development toward the ultimate goal covers the following areas.

- **Game compatibility.** Cover the drawing, compute, command patching, queue synchronization and display behavior used by the game, preserving rendering results and execution order.
- **Native GPU execution.** Emit host PM4 during recording, translate and cache PS5 shaders as target ISA, and use the target GPU's graphics and compute capabilities.
- **Native BVH and ray tracing.** Use native AMD BVH layouts and hardware-accelerated ray intersection and traversal. Support GPU-side BVH construction and updates, cross-generation layout adaptation and acceleration-structure synchronization.
- **Machine-learning super-resolution.** Run super-resolution inference on the GPU, using native matrix or dot-product instructions where available. Integrate model execution, input/output surfaces and synchronization with rendering and presentation, balancing reconstruction quality and frame cost.
- **Resource and memory correctness.** Handle descriptors, texture and render-target layouts, compression metadata, sparse mappings and memory aliases while preserving CPU/GPU coherence, cross-queue visibility and resource lifetimes.
- **Windows and Linux support.** Share the AGC front end, shader translator and AMD hardware logic across WDDM and DRM/amdgpu backends, with platform-specific memory management, submission, synchronization and window presentation.
- **Sustained smooth gameplay.** Reduce submission-time CPU state reconstruction, repeated compilation, resource transfers and synchronization stalls. Validate image correctness, frame pacing and extended gameplay on real hardware, recording the GPU, resolution, quality settings, average FPS and 1% low FPS.

## LLVM toolchain

The project uses an **LLVM 21.1.6 fork** with AMDGPU and PS5 `gfx1013` support. The ISA table generator pins the revision below. The local `llvm-mc --version` output reports `LLVM version 21.1.6`.

| Item | Version or path |
|---|---|
| LLVM version | `21.1.6` |
| Source commit | `3c1efa15830c5c60dbc45a3d3d47a1cacf5aba57` |
| Local source | `D:\projects\apple-llvm-project` |
| AMDGPU build directory | `D:\projects\apple-llvm-project\.build-amdgpu` |
| LLVM MC | `D:\projects\apple-llvm-project\.build-amdgpu\bin\llvm-mc.exe` |

LLVM generates AMDGPU ISA tables, assembles built-in GPU kernels, and disassembles translated shaders for encoding checks. The Rust `shader` crate performs runtime shader translation. Regular Cargo builds consume the generated ISA tables and embedded machine code. Revision provenance is recorded in the [ISA table sources](shader/src/ir/isa_gen/mod.rs) and [generation script](tools/isa-tables/gen_isa_tables.py).

ISA table generation stays pinned to this LLVM revision. The LLVM comparison tests check exact encoded words and successful disassembly, allowing disassembly formatting to vary between LLVM releases. Recorded disassembly remains available in the golden files for inspection.

Run these PowerShell commands from `driver` to check the tool version and enable the LLVM comparison tests.

```powershell
& "D:\projects\apple-llvm-project\.build-amdgpu\bin\llvm-mc.exe" --version
$env:ANYPS5_LLVM_MC = "D:\projects\apple-llvm-project\.build-amdgpu\bin\llvm-mc.exe"
cargo test -p shader --release --test llvm_mc
```

## Architecture

| Layer | Role |
|---|---|
| `runtime/src/agc` | Exposes the guest ABI, tracks guest state, and passes operations through the `Target` and `Recorder` interfaces. |
| `runtime/src/gpu/agc` | Encodes AMD PM4 into title-owned command buffers; manages translated state, shaders, and surfaces. |
| `shader` | Translates PS5 `gfx1013` binaries and Pro extensions through control-flow analysis, IR, lowering, register allocation, and host ISA encoding. |
| `runtime/src/backend`, `residency`, `scheduler` | Manage device queues, guest-memory imports, dependencies, and completion. WDDM is the Windows transport. |
| `runtime/src/present` | Converts display surfaces on the GPU and hands shared buffers to D3D12 for copying and presentation. |

## Design

- **Translate during recording.** Agc calls emit host commands directly. Size queries and patch operations share fixed command footprints, preserving the title's buffer layout.
- **Submit from metadata.** Recordings retain links, state changes, resources, waits, and events. Submission currently resolves inherited queue state into a private executable snapshot for each occurrence. It never rewrites the shared recording; parked segments retain their own bindings (D50). Removing this remaining submission-time state reconstruction requires explicit inheritance and late-patch handling.
- **Keep binding order.** Resource preparation retains every draw and dispatch root in command order, including A/B/A target rebinding and repeated dispatches with different coverage. Resource lifetime sets may deduplicate ownership; binding events may not (D49).
- **Execute resource operations in order.** GPU layout conversions and proven metadata clears are woven around their consuming work, with guard extents preserved. Dynamic descriptor discovery, conditional register-state joins, stencil planes, and general self-modifying commands remain incomplete; UI-only game output is not graphics validation (D50).
- **Separate guest semantics from hardware.** The front end describes guest operations; targets own command encoding and hardware rules. Guest-only operations expand into GPU instructions. OS backends own memory and queue transport.
- **Preserve memory identity.** Import guest pages at matching GPU addresses where supported. Keep host-layout surface copies separate and synchronize them as views and writes require. Translated programs report the images they locate at run time, so the runtime can prepare those copies for the next frame.
- **Tie lifetime to completion.** Retain command storage, translated programs, and resident memory until GPU fences retire their use. Schedule cross-queue waits between submissions.
- **Keep command copies with their recording.** Cache state packets on the CPU, allocate their GPU copies through thread-owned cursors, and retain them through executable command ranges and pending submissions. Idle recorder registrations hold weak references. Residency uses a GPU mapping generation when libc provides one (D48).
- **Generate hardware definitions.** `pm4-proc` derives registers and packets from Mesa definitions; `isa-proc` derives host shader encodings from AMD ISA XML.
