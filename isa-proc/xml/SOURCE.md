# AMD machine-readable GPU ISA specification

`AMD_GPU_MR_ISA_XML_2026_08_06.zip` is AMD's machine-readable ISA XML bundle, copied
unchanged. Its SHA-256 is
`82404F1126761B7877595B622AFA7E1F311F2F41E89A3ABE9AAF8AD045C082E2`, the same bundle the
shader crate's `ir::isa_gen` tables record. Each XML's `Document` element says:

- Copyright (c) 2026 Advanced Micro Devices, Inc., or its affiliates.
- License: MIT
- Release date: 2026-02-20
- Sensitivity: AMD Public Use.

`isa-proc` reads these architectures from it:

| XML | Architecture | Used for |
| --- | --- | --- |
| `amdgpu_isa_rdna3_5.xml` | AMD RDNA 3.5 | gfx11.5 hosts (Radeon 8060S) |
| `amdgpu_isa_rdna4.xml` | AMD RDNA 4 | gfx12 hosts |

To update, replace the zip with a newer bundle from AMD, rename the file after it, and change
the name in `src/lib.rs`.

The PS5 guest ISA (gfx1013) is decoded by hand, not from this bundle.
