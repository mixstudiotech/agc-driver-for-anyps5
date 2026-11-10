# AnyPS5 GPU Driver

A Rust implementation of the PS5 Agc libraries that records native host GPU commands. The AMD runtime targets RDNA 3.5 (`gfx11.5`) and RDNA 4 (`gfx12`).

## Architecture

| Layer | Role |
|---|---|
| `runtime/src/agc` | Exposes the guest ABI, tracks guest state, and passes operations through the `Target` and `Recorder` interfaces. |
| `runtime/src/gpu/agc` | Encodes AMD PM4 into title-owned command buffers; manages translated state, shaders, and surfaces. |
| `shader` | Translates PS5 `gfx1013` binaries through control-flow analysis, IR, lowering, register allocation, and host ISA encoding. |
| `runtime/src/backend`, `residency`, `scheduler` | Manage device queues, guest-memory imports, dependencies, and completion. WDDM is the Windows transport. |
| `runtime/src/present` | Converts display surfaces on the GPU and hands shared buffers to D3D12 for copying and presentation. |

## Design

- **Translate during recording.** Agc calls emit host commands directly. Size queries and patch operations share fixed command footprints, preserving the title's buffer layout.
- **Submit from metadata.** Recordings retain links, state changes, resources, waits, and events. Submission follows those records, resolves inherited queue state, and patches affected draws or dispatches without translating the entire command stream again.
- **Separate guest semantics from hardware.** The front end describes guest operations; targets own command encoding and hardware rules. OS backends own memory and queue transport.
- **Preserve memory identity.** Import guest pages at matching GPU addresses where supported. Keep host-layout surface copies separate and synchronize them as views and writes require.
- **Tie lifetime to completion.** Retain command storage, translated programs, and resident memory until GPU fences retire their use. Schedule cross-queue waits between submissions.
- **Generate hardware definitions.** `pm4-proc` derives registers and packets from Mesa definitions; `isa-proc` derives host shader encodings from AMD ISA XML.
