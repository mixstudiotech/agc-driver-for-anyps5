use super::kernels;
use agcdrv::backend::wddm::{Buffer, Device, MemoryKind};
use agcdrv::gpu::{
    packets::{PacketStream, SHADER_TYPE_COMPUTE, event, gcr},
    regs as host,
};
use shader::{CompileOptions, Compiler, RayUnit, ShaderInterface, ShaderStage};
use std::collections::BTreeMap;
use std::io::Write;

/// Bytes of one lane's input: the node pointer (two dwords), the ray extent and a dword of
/// padding, then origin, direction and inverse direction, each followed by a dword of padding.
const RECORD: usize = 64;
/// Bytes a node pointer counts in.
const SLOT: usize = 64;
/// Bytes kept for each vector's node (a box32 node takes two slots).
const NODE: usize = 128;
const VECTORS: usize = 8192;
const INVALID: u32 = u32::MAX;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }

    fn point(&mut self, extent: f32) -> [f32; 3] {
        [0; 3].map(|_| self.range(-extent, extent))
    }
}

fn half(value: f32) -> u32 {
    let bits = value.to_bits();
    let (sign, exponent, mantissa) = ((bits >> 16) & 0x8000, (bits >> 23) & 0xff, bits & 0x7f_ffff);
    if exponent == 0xff {
        return sign | 0x7c00 | u32::from(mantissa != 0) << 9;
    }
    let exponent = exponent as i32 - 127 + 15;
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        return sign;
    }
    let rounded = (exponent as u32) << 10 | mantissa >> 13;
    // Round to nearest even; a carry out of the mantissa raises the exponent as it should.
    let rest = mantissa & 0x1fff;
    sign | (rounded + u32::from(rest > 0x1000 || (rest == 0x1000 && rounded & 1 != 0)))
}

/// A node, the pointer type that selects how the unit reads it, and a ray.
struct Vector {
    class: &'static str,
    node: [u32; NODE / 4],
    ty: u32,
    /// The node's slot is past what the descriptor of the 32-bit form covers.
    out_of_range: bool,
    extent: f32,
    origin: [f32; 3],
    dir: [f32; 3],
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// A direction from `origin` to `target`, at times with components forced to zero or the whole
/// ray made unusual.
fn aim(rng: &mut Rng, origin: [f32; 3], target: [f32; 3]) -> ([f32; 3], f32) {
    let mut dir = sub(target, origin);
    let mut extent = f32::INFINITY;
    match rng.below(32) {
        0 => dir[rng.below(3) as usize] = 0.0,
        1 => dir[rng.below(3) as usize] = -0.0,
        2 => dir = rng.point(1.0),
        3 => extent = rng.range(0.0, 2.0),
        4 => extent = 1.0,
        5 => extent = -1.0,
        6 => extent = f32::NAN,
        7 => extent = 0.0,
        8 => dir = [0.0; 3],
        9 => dir[rng.below(3) as usize] = f32::NAN,
        10 => dir[rng.below(3) as usize] = f32::INFINITY,
        11 => dir = dir.map(|d| d * 1.0e-30),
        12 => dir = dir.map(|d| d * 1.0e30),
        _ => {}
    }
    (dir, extent)
}

fn boxes(rng: &mut Rng, wide: bool) -> Vector {
    let mut node = [0u32; NODE / 4];
    let mut centers = [[0f32; 3]; 4];
    for child in 0..4 {
        node[child] = match rng.below(8) {
            0 => INVALID,
            _ => (0x1000 * (child as u32 + 1)) | rng.below(0x1000),
        };
        let center = rng.point(8.0);
        let size = rng.point(1.5).map(f32::abs);
        let (mut low, mut high) = (sub(center, size), [0, 1, 2].map(|k| center[k] + size[k]));
        match rng.below(24) {
            0 => low[rng.below(3) as usize] = high[rng.below(3) as usize] + 1.0,
            1 => low = high,
            2 => high[rng.below(3) as usize] = f32::NAN,
            3 => low[rng.below(3) as usize] = f32::NEG_INFINITY,
            _ => {}
        }
        centers[child] = center;
        if wide {
            for k in 0..3 {
                node[4 + child * 6 + k] = low[k].to_bits();
                node[7 + child * 6 + k] = high[k].to_bits();
            }
        } else {
            // (x min, y min), (z min, x max), (y max, z max), the first in the low half.
            let h = [low[0], low[1], low[2], high[0], high[1], high[2]].map(half);
            for pair in 0..3 {
                node[4 + child * 3 + pair] = h[pair * 2] | h[pair * 2 + 1] << 16;
            }
        }
    }
    let origin = rng.point(12.0);
    let near = centers[rng.below(4) as usize];
    let target = [0, 1, 2].map(|k| near[k] + rng.range(-1.0, 1.0));
    let (dir, extent) = aim(rng, origin, target);
    Vector {
        class: if wide { "box32" } else { "box16" },
        node,
        ty: if wide { 5 } else { 4 },
        out_of_range: false,
        extent,
        origin,
        dir,
    }
}

fn triangles(rng: &mut Rng) -> Vector {
    const VERTICES: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]];
    let mut node = [0u32; NODE / 4];
    let vertices = [0; 5].map(|_| rng.point(4.0));
    for (v, vertex) in vertices.iter().enumerate() {
        for k in 0..3 {
            node[v * 3 + k] = vertex[k].to_bits();
        }
    }
    node[15] = rng.next() ^ rng.next() << 16;
    let ty = rng.below(4);
    let [a, b, c] = VERTICES[ty as usize].map(|v| vertices[v]);
    let origin = rng.point(10.0);
    let along = |p: [f32; 3], q: [f32; 3], s: f32| [0, 1, 2].map(|k| p[k] + (q[k] - p[k]) * s);
    let (class, target) = match rng.below(8) {
        // On an edge or a vertex of the triangle.
        0 => ("triangle edge", along(a, b, [0.0, 0.25, 0.5, 1.0][rng.below(4) as usize])),
        1 => ("triangle edge", along(b, c, [0.0, 0.25, 0.5, 1.0][rng.below(4) as usize])),
        2 => ("triangle edge", along(c, a, [0.0, 0.25, 0.5, 1.0][rng.below(4) as usize])),
        3 => ("triangle", rng.point(6.0)),
        _ => {
            let (s, t) = (rng.range(0.0, 1.0), rng.range(0.0, 1.0));
            ("triangle", along(along(a, b, s), c, t))
        }
    };
    let (dir, extent) = aim(rng, origin, target);
    Vector {
        class,
        node,
        ty,
        out_of_range: false,
        extent,
        origin,
        dir,
    }
}

fn vectors() -> Vec<Vector> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    (0..VECTORS)
        .map(|index| {
            let mut vector = match index % 8 {
                0 | 1 => boxes(&mut rng, true),
                2 | 3 => boxes(&mut rng, false),
                _ => triangles(&mut rng),
            };
            match rng.below(64) {
                // Types the unit does not intersect, and slots the descriptor does not cover.
                0 => (vector.class, vector.ty) = ("type 6 or 7", 6 + rng.below(2)),
                1 => (vector.class, vector.out_of_range) = ("out of range", true),
                _ => {}
            }
            vector
        })
        .collect()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn buffer(address: u64, stride: u32, count: u32) -> [u32; 4] {
    [
        address as u32,
        (address >> 32) as u32 | (stride << 16),
        count,
        0xfac | (22 << 12),
    ]
}

/// What the descriptor asks of the unit.
#[derive(Clone, Copy, Debug)]
struct Controls {
    sort: bool,
    barycentrics: bool,
    grow: u32,
}

impl Controls {
    /// The descriptor over `slots` node slots at `base`, or over all of memory for 64-bit node
    /// pointers.
    fn descriptor(self, base: u64, slots: u64, wide_pointers: bool) -> [u32; 4] {
        let (base, last) = match wide_pointers {
            true => (0, (1u64 << 42) - 1),
            false => (base, slots - 1),
        };
        [
            (base >> 8) as u32,
            ((base >> 40) & 0xff) as u32 | self.grow << 23 | u32::from(self.sort) << 31,
            last as u32,
            (last >> 32) as u32 | u32::from(self.barycentrics) << 24 | 0x8000_0000,
        ]
    }
}

struct Program {
    code: Buffer,
    rsrc1: u32,
    rsrc2: u32,
}

fn run(
    device: &mut Device,
    program: &Program,
    buffers: [&Buffer; 3],
    wave: u8,
    user: &[u32; 12],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let [nodes, input, output] = buffers;
    let mut stream = PacketStream::new_compute();
    stream.acquire_mem_global(gcr::GLK_INV | gcr::GLV_INV | gcr::GL1_INV | gcr::GL2_INV)?;
    for (reg, value) in [
        (host::COMPUTE_PGM_LO, (program.code.address() >> 8) as u32),
        (host::COMPUTE_PGM_HI, (program.code.address() >> 40) as u32),
        (host::COMPUTE_PGM_RSRC1, program.rsrc1),
        (host::COMPUTE_PGM_RSRC2, program.rsrc2),
        (host::COMPUTE_PGM_RSRC3, 0),
        (host::COMPUTE_TMPRING_SIZE, 0),
        (host::COMPUTE_RESOURCE_LIMITS, 0),
    ] {
        stream.set(reg, value)?;
    }
    stream.set_run(host::COMPUTE_START, &[0, 0, 0])?;
    stream.set_run(host::COMPUTE_NUM_THREAD, &[u32::from(wave), 1, 1])?;
    stream.set_run(host::COMPUTE_USER_DATA, user)?;
    stream.raw_packet(
        agcdrv::amd::defs::pkt3::DISPATCH_DIRECT,
        SHADER_TYPE_COMPUTE,
        &[1, 1, 1, 1 | if wave == 32 { 1 << 15 } else { 0 }],
    )?;
    stream.event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)?;
    stream.acquire_mem_global(gcr::GL2_WB | gcr::GL2_INV)?;
    let serial = device.submit_host(stream.words(), &[&program.code, nodes, input, output])?;
    device.wait(serial)?;
    Ok(output.read()?[..usize::from(wave) * 16]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect())
}

/// What the SDK's model of the unit says of a ray and a triangle
/// (`triangle_intersection_emulation` in its PSR sources).
struct Expected {
    words: [u32; 4],
    hit: bool,
    /// The smallest edge function over the sum of the three: how far from the triangle's edges
    /// the ray passes.
    margin: f32,
}

fn triangle_model(
    id: u32,
    ty: u32,
    barycentrics: bool,
    origin: [f32; 3],
    dir: [f32; 3],
    vertices: [[f32; 3]; 3],
) -> Expected {
    // The axis the direction is largest on becomes z.
    let mut kz = 0;
    if dir[0].abs() < dir[1].abs() {
        kz = 1;
    }
    if dir[kz].abs() < dir[2].abs() {
        kz = 2;
    }
    let turn = |p: [f32; 3]| match kz {
        2 => p,
        1 => [p[2], p[0], p[1]],
        _ => [p[1], p[2], p[0]],
    };
    let (dir, origin) = (turn(dir), turn(origin));
    let shear = |p: [f32; 3]| {
        let p = sub(turn(p), origin);
        [p[0] * dir[2] - dir[0] * p[2], p[1] * dir[2] - dir[1] * p[2], p[2]]
    };
    let [[ax, ay, az], [bx, by, bz], [cx, cy, cz]] = vertices.map(shear);
    let (u, v, w) = (cx * by - cy * bx, ax * cy - ay * cx, bx * ay - by * ax);
    let (uz, vz, wz) = (u * dir[2], v * dir[2], w * dir[2]);
    let (mut t_num, mut t_denom) = (u * az + v * bz + w * cz, uz + vz + wz);
    let mut missed = (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0);
    missed |= t_denom == 0.0;
    let sign = if t_denom > 0.0 { 1.0 } else { -1.0 };
    missed |= t_num * sign < 0.0 || t_num.is_nan();
    // A ray through an edge hits the triangle on one side of it only (the top-left rule).
    let winding = t_denom > 0.0;
    for (edge, first, second, third) in [(u, by, cy, ay), (v, cy, ay, by), (w, ay, by, cy)] {
        if edge == 0.0 {
            missed |= match first == 0.0 && second == 0.0 {
                true => third > 0.0,
                false => (first < 0.0 || (first == 0.0 && second > 0.0)) ^ winding,
            };
        }
    }
    if missed {
        (t_num, t_denom) = (f32::INFINITY, 1.0);
    }
    let control = id >> (ty * 8);
    let pick = |source: u32| match source & 3 {
        1 => vz,
        2 => wz,
        _ => uz,
    };
    let words = match barycentrics {
        true => [t_num, t_denom, pick(control), pick(control >> 2)].map(f32::to_bits),
        false => [t_num.to_bits(), t_denom.to_bits(), id.wrapping_add(ty), u32::from(!missed)],
    };
    let sum = u.abs() + v.abs() + w.abs();
    Expected {
        words,
        hit: !missed,
        margin: u.abs().min(v.abs()).min(w.abs()) / sum,
    }
}

/// What is wrong with the unit's `ours` for `vector`, if anything. `reference` is the
/// reference expansion's result, which holds for boxes only.
fn fault(
    vector: &Vector,
    control: Controls,
    covered: bool,
    ours: &[u32],
    reference: &[u32],
) -> Option<&'static str> {
    let nan = vector.extent.is_nan() || vector.dir.iter().any(|d| d.is_nan());
    if vector.ty >= 6 || !covered {
        return (ours != [INVALID; 4]).then_some("a node the unit cannot read is not refused");
    }
    if nan {
        // The unit refuses a ray with a NaN; the reference model goes on with it.
        return None;
    }
    if vector.ty >= 4 {
        return (ours != reference).then_some("box results differ from the reference");
    }
    let float = |word: u32| f32::from_bits(word);
    let infinite = ours[0] & 0x7fff_ffff == f32::INFINITY.to_bits();
    if infinite && (ours[0] != f32::INFINITY.to_bits() || ours[1] != 1.0f32.to_bits()) {
        return Some("a miss is not +inf over 1.0");
    }
    let id = vector.node[15].wrapping_add(vector.ty);
    if !control.barycentrics && (ours[2] != id || ours[3] != u32::from(!infinite)) {
        return Some("the ID or the hit flag is wrong");
    }
    const VERTICES: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]];
    let vertex = |v: usize| [0, 1, 2].map(|k| float(vector.node[v * 3 + k]));
    let model = triangle_model(
        vector.node[15],
        vector.ty,
        control.barycentrics,
        vector.origin,
        vector.dir,
        VERTICES[vector.ty as usize].map(vertex),
    );
    // Rays through an edge, and directions the model's products overflow or underflow on, are
    // counted but not judged.
    let size = vector.dir.iter().fold(0f32, |m, d| m.max(d.abs()));
    if !(model.margin > 1.0e-3) || !(1.0e-6..1.0e6).contains(&size) {
        return None;
    }
    if model.hit == infinite {
        return Some("the hit differs from the SDK model");
    }
    if !model.hit {
        return None;
    }
    let expected = model.words.map(float);
    let close = |a: f32, b: f32, tolerance: f32| (a - b).abs() <= tolerance * b.abs().max(1.0);
    let (ours_d, expected_d) = (float(ours[1]), expected[1]);
    if (ours_d < 0.0) != (expected_d < 0.0) || !close(float(ours[0]) / ours_d, expected[0] / expected_d, 1.0e-4) {
        return Some("the distance differs from the SDK model");
    }
    let coordinates = control.barycentrics
        && [2, 3].iter().any(|&k| !close(float(ours[k]) / ours_d, expected[k] / expected_d, 1.0e-3));
    coordinates.then_some("the barycentrics differ from the SDK model")
}

#[derive(Default)]
struct Tally {
    lanes: u64,
    /// Lanes the unit answered with something other than all ones.
    answered: u64,
    /// Lanes that are not the reference expansion's result bit for bit (expected of triangles).
    unlike_reference: u64,
    faults: u64,
    shown: u32,
}

#[test]
#[ignore = "requires Mix AMD WDDM and an idle GPU"]
fn the_host_ray_unit_matches_the_reference_model() -> Result<(), Box<dyn std::error::Error>> {
    let vectors = vectors();
    let mut device = Device::open()?;
    println!("adapter {:?}", device.identity());
    let target = agcdrv::gpu::target::host_target(device.identity());
    let mut nodes = device.allocate_kind(2 * VECTORS * NODE, MemoryKind::Data)?;
    let mut input = device.allocate_kind(65536, MemoryKind::Data)?;
    let mut output = device.allocate_kind(65536, MemoryKind::Readback)?;
    // The second half of the node memory holds the nodes no 32-bit descriptor covers.
    let slots = (VECTORS * NODE / SLOT) as u64;
    let slot_of = |index: usize, vector: &Vector| {
        (index * NODE / SLOT) as u64 + if vector.out_of_range { slots } else { 0 }
    };
    nodes.fill(0)?;
    for (index, vector) in vectors.iter().enumerate() {
        nodes.write(slot_of(index, vector) as usize * SLOT, &bytes(&vector.node))?;
    }
    let interface = ShaderInterface {
        guest_pgm_rsrc2: Some(12 << 1),
        ..ShaderInterface::default()
    };
    let every: Vec<Controls> = (0..8)
        .map(|bits| Controls {
            sort: bits & 1 != 0,
            barycentrics: bits & 2 != 0,
            grow: if bits & 4 != 0 { 5 } else { 0 },
        })
        .collect();
    let two = [every[3], every[4]];
    let kernels: [(&str, &[u32], bool, &[Controls]); 4] = [
        ("wide", kernels::WIDE, true, &every),
        ("scattered", kernels::SCATTERED, false, &every),
        ("contiguous", kernels::CONTIGUOUS, false, &two),
        ("halves", kernels::HALVES, false, &two),
    ];
    let mut report = std::fs::File::create("shader_bvh-faults.tsv")?;
    writeln!(report, "fault\tkernel\twave\tcontrols\tvector\tclass\ttype\textent\torigin\tdir\thost\treference")?;
    let mut tallies: BTreeMap<(&str, &str), Tally> = BTreeMap::new();
    for (name, guest, wide_pointers, controls) in kernels {
        for wave in [32u8, 64] {
            let mut programs = Vec::new();
            for unit in [RayUnit::Host, RayUnit::Reference] {
                let options =
                    CompileOptions::new(target, ShaderStage::Compute, wave).with_ray_unit(unit);
                let artifact =
                    Compiler::new(target).compile_with_interface(guest, options, &interface)?;
                let mut code = device.allocate_executable(1 << 20)?;
                code.write(0, &artifact.code_bytes())?;
                println!(
                    "{name} wave={wave} {unit:?}: code_words={} sgprs={} vgprs={}",
                    artifact.program_words, artifact.num_sgprs, artifact.num_vgprs
                );
                programs.push(Program {
                    code,
                    rsrc1: artifact.pgm_rsrc1,
                    rsrc2: artifact.pgm_rsrc2,
                });
            }
            for &control in controls {
                for (chunk, lanes) in vectors.chunks(usize::from(wave)).enumerate() {
                    let first = chunk * usize::from(wave);
                    let mut records = vec![0u32; usize::from(wave) * RECORD / 4];
                    for (lane, vector) in lanes.iter().enumerate() {
                        let slot = slot_of(first + lane, vector);
                        let pointer = match wide_pointers {
                            true => (nodes.address() + slot * SLOT as u64) >> 3,
                            false => slot << 3,
                        } | u64::from(vector.ty);
                        let record = &mut records[lane * RECORD / 4..][..RECORD / 4];
                        record[0] = pointer as u32;
                        record[1] = (pointer >> 32) as u32;
                        record[2] = vector.extent.to_bits();
                        for k in 0..3 {
                            record[4 + k] = vector.origin[k].to_bits();
                            record[8 + k] = vector.dir[k].to_bits();
                            record[12 + k] = (1.0 / vector.dir[k]).to_bits();
                        }
                    }
                    input.write(0, &bytes(&records))?;
                    let mut user = [0u32; 12];
                    user[..4].copy_from_slice(&buffer(
                        input.address(),
                        RECORD as u32,
                        u32::from(wave),
                    ));
                    user[4..8].copy_from_slice(&control.descriptor(
                        nodes.address(),
                        slots,
                        wide_pointers,
                    ));
                    user[8..].copy_from_slice(&buffer(output.address(), 16, u32::from(wave)));
                    let mut results = Vec::new();
                    for program in &programs {
                        output.fill(0xcd)?;
                        let buffers = [&nodes, &input, &output];
                        results.push(run(&mut device, program, buffers, wave, &user)?);
                    }
                    for (lane, vector) in lanes.iter().enumerate() {
                        let (ours, reference) =
                            (&results[0][lane * 4..][..4], &results[1][lane * 4..][..4]);
                        // The A16 kernel's directions are halves, which the model is not given.
                        let class = match (name, vector.ty) {
                            ("halves", 0..=3) => "triangle (halves)",
                            _ => vector.class,
                        };
                        let tally = tallies.entry((name, class)).or_default();
                        tally.lanes += 1;
                        tally.answered += u64::from(ours != [INVALID; 4]);
                        tally.unlike_reference += u64::from(ours != reference);
                        let covered = wide_pointers || !vector.out_of_range;
                        let judged = match class {
                            "triangle (halves)" if ours[0] == f32::NEG_INFINITY.to_bits() => {
                                Some("a miss is not +inf over 1.0")
                            }
                            "triangle (halves)" => None,
                            _ => fault(vector, control, covered, ours, reference),
                        };
                        let Some(what) = judged else {
                            continue;
                        };
                        tally.faults += 1;
                        writeln!(
                            report,
                            "{what}\t{name}\t{wave}\t{control:?}\t{}\t{}\t{}\t{:08x}\t{:08x?}\t{:08x?}\t{ours:08x?}\t{reference:08x?}",
                            first + lane,
                            vector.class,
                            vector.ty,
                            vector.extent.to_bits(),
                            vector.origin.map(f32::to_bits),
                            vector.dir.map(f32::to_bits),
                        )?;
                        if tally.shown < 6 {
                            tally.shown += 1;
                            println!(
                                "{what}: {name} wave={wave} {control:?} vector={} {} type={}: host {ours:08x?} reference {reference:08x?}",
                                first + lane,
                                vector.class,
                                vector.ty
                            );
                        }
                    }
                }
            }
        }
    }
    let mut faults = 0;
    for ((name, class), tally) in &tallies {
        faults += tally.faults;
        println!(
            "{name:10} {class:18} lanes={} answered={} unlike_reference={} faults={}",
            tally.lanes, tally.answered, tally.unlike_reference, tally.faults
        );
    }
    println!("ray_unit_gpu vectors={} faults={faults}", vectors.len());
    assert_eq!(faults, 0);
    Ok(())
}
