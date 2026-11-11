# BVH8 reference data

The inputs and outputs contain 100 deterministic cases from the installed SDK 10 CPU intersector, compiled directly with host Clang at O0 and O2. Both builds produced identical bytes. No SDK implementation source is included here.

Each 384-byte input record contains 96 little-endian DWORDs: encoded node index, extent bits, instance mask, options, pointer flags, four descriptor words, origin xyz, direction xyz, one padding word, and 80 node words. Each 64-byte output record contains ten result words, origin xyz, and direction xyz. Floating-point values are binary32 bit patterns. Node bases are replaced with actual GPU buffer addresses at dispatch. The stable synthetic child address in instance data is returned but never dereferenced.

Outputs use x86 FTZ/DAZ, corresponding to the expansion's f32 flush mode. The GPU test excludes three descriptor heuristic/grow diagnostics because the SDK CPU implementation ignores those controls. Such controls need separate expected results. The remaining cases cover every node kind, shared exponents, culling, top-left triangle edges, procedural flags, instance flag merging, and mask misses.

`manual-inputs.bin` and `manual-outputs.bin` add eight geometric cases for closest entry, largest interval, midpoint before clamping, triangle priority, sorting within groups of four, and maximum-plane growth. `manual-cases.txt` lists their order; `generate_manual.py` reproduces them. Their expectations are computed from explicit ray intervals rather than the SDK model. The field definitions and sorting modes follow [AMD's RDNA4 ISA descriptor table](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content); these tests do not establish identical PS5 Pro hardware behavior for undocumented cases.

Source dependency hashes, original JSON, IEEE outputs, build and verification commands are retained in `evidence/shader-refusals-20261111/sdk-cpu-oracle` outside the driver tree. This is a CPU reference, not PS5 Pro hardware evidence.

The GPU test also runs base/result aliasing and the SDK's read-only direction/result allocation. The latter excludes instance transforms, where that compiler promise does not preserve the aliased ray outputs. Odd lanes force both opaque and non-opaque culling, exercising divergent memory predicates and unchanged ray coordinates on early return.

The state fixture sets v167 and s99 sentinels, changes MODE to `0x33`, initializes VCC and SCC, and executes the intersection only in even lanes. It checks inactive outputs, the restored mode and flags, and both sentinels. This exercises SGPR spilling at 255 host VGPRs.

From the driver root, `python runtime/tests/shader_bvh8/generate.py --llvm-mc <path> --check` verifies all 13 recorded ISA fixtures against their assembly sources and stack template. Regeneration needs LLVM with AMDGPU support; ordinary tests use the recorded words and need no SDK or assembler. GPU tests are ignored by default and require WDDM plus an idle supported AMD device.
