# Vendored Mesa AMD definitions

Copied unchanged from Mesa (the `mesa-manet` tree) at commit
`c6ffbdc5ea4685203e89d648e71cc6c7ecfc44ca` (2026-10-25). Mesa's AMD register
database and `sid.h` are MIT licensed (`SPDX-License-Identifier: MIT`,
Copyright Advanced Micro Devices, Inc.).

| File | Mesa path | Used for |
| --- | --- | --- |
| `gfx10.json` | `src/amd/registers/gfx10.json` | guest registers (PS5, gfx10.1) |
| `gfx115.json` | `src/amd/registers/gfx115.json` | host registers, RDNA3.5 |
| `gfx12.json` | `src/amd/registers/gfx12.json` | host registers, RDNA4 |
| `gfx10-rsrc.json` | `src/amd/registers/gfx10-rsrc.json` | guest buffer, image and sampler descriptors |
| `gfx11-rsrc.json` | `src/amd/registers/gfx11-rsrc.json` | host descriptors, gfx11 and gfx11.5 |
| `gfx12-rsrc.json` | `src/amd/registers/gfx12-rsrc.json` | host descriptors, gfx12 |
| `pkt3.json` | `src/amd/registers/pkt3.json` | PM4 packet dwords not in `sid.h` |
| `sid.h` | `src/amd/common/sid.h` | PM4 opcodes and packet fields |
| `packets/cp_pm4_table_data_gfx11.json` | `src/amd/packets/cp_pm4_table_data_gfx11.json` | CP packet layouts, gfx11 and gfx11.5 hosts |
| `packets/cp_pm4_table_data_gfx12.json` | `src/amd/packets/cp_pm4_table_data_gfx12.json` | CP packet layouts, gfx12 hosts |
| `packets/pm4_it_opcodes_gfx11.h` | `src/amd/packets/pm4_it_opcodes_gfx11.h` | CP packet opcodes, gfx11 |
| `packets/pm4_it_opcodes_gfx12.h` | `src/amd/packets/pm4_it_opcodes_gfx12.h` | CP packet opcodes, gfx12 |

To update, copy the same files from a newer Mesa and change the commit above.
