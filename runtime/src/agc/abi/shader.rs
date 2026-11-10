//! libSceAgc's shader objects (PLAN P3.4 step 2f): `sceAgcCreateShader` and the functions that
//! derive register state from shaders. The console library's driver calls here only record
//! (shader preparation scopes, ABI resolution) and have nothing to do in the runtime: the
//! programs are translated when a draw first uses them (D4).

use super::report;
use crate::agc::ShaderRegister;
use crate::agc_export;

/// `Shader` (0x60 bytes): the header a shader binary starts with.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Shader {
    pub file_header: u32,
    pub version: u32,
    pub user_data: *mut ShaderUserData,
    pub code: u64,
    pub cx_registers: *mut ShaderRegister,
    pub sh_registers: *mut ShaderRegister,
    pub specials: *mut ShaderSpecialRegs,
    pub input_semantics: *const u32,
    pub output_semantics: *const u32,
    pub header_size: u32,
    pub shader_size: u32,
    pub embedded_constant_buffer_size_dqw: u32,
    pub target: u32,
    pub num_input_semantics: u32,
    pub scratch_size_dw_per_thread: u16,
    pub num_output_semantics: u16,
    pub special_sizes_bytes: u16,
    pub kind: u8,
    pub num_cx_registers: u8,
    pub num_sh_registers: u8,
    pub reserved: [u8; 3],
}

/// `ShaderUserData` (0x38 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShaderUserData {
    pub direct_resource_offset: *mut u16,
    pub sharp_resource_offset: [*mut u16; 4],
    pub eud_size_dw: u16,
    pub srt_size_dw: u16,
    pub direct_resource_count: u16,
    pub sharp_resource_count: [u16; 4],
}

/// `ShaderSpecialRegs` (0x30 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShaderSpecialRegs {
    pub ge_cntl: ShaderRegister,
    pub vgt_shader_stages_en: ShaderRegister,
    pub dispatch_modifier: u32,
    pub user_data_range: [u16; 2],
    pub draw_modifier: [u32; 2],
    pub vgt_gs_out_prim_type: ShaderRegister,
    pub ge_user_vgpr_en: ShaderRegister,
}

const _: () = assert!(std::mem::size_of::<Shader>() == 0x60);
const _: () = assert!(std::mem::size_of::<ShaderUserData>() == 0x38);
const _: () = assert!(std::mem::size_of::<ShaderSpecialRegs>() == 0x30);

/// `ShaderBinaryType`.
pub mod kind {
    pub const CS: u8 = 0;
    pub const PS: u8 = 1;
    pub const GS: u8 = 2;
    pub const HS: u8 = 3;
    pub const GS_FRONT: u8 = 4;
    pub const HS_FRONT: u8 = 5;
    pub const GS_BACK: u8 = 6;
    pub const HS_BACK: u8 = 7;
    pub const FS: u8 = 8;
}

/// SDK error codes.
pub mod error {
    pub const INVALID_ALIGNMENT: i32 = 0x8a6c_0002_u32 as i32;
    pub const INVALID_SHADER_PROGRAM: i32 = 0x8a6c_0005_u32 as i32;
    pub const INVALID_SHADER_HALVES: i32 = 0x8a6c_0008_u32 as i32;
    pub const INVALID_ARGUMENT: i32 = 0x8a6c_000a_u32 as i32;
    pub const OUT_OF_RANGE: i32 = 0x8a6c_000b_u32 as i32;
}

const SHADER_MAGIC: u32 = 0x3433_3231;
const SHADER_VERSION: u32 = 0x18;

/// The SH register of a shader type's program address (`PGM_LO`; `PGM_HI` follows), `None` for
/// types without one; `Err` for unknown types.
pub fn program_register(kind: u8) -> Result<Option<u32>, &'static str> {
    Ok(match kind {
        kind::CS => Some(0x20c),
        kind::PS => Some(0x008),
        kind::GS => Some(0x0c8),
        kind::HS => Some(0x148),
        kind::GS_BACK => Some(0x088),
        kind::HS_BACK => Some(0x108),
        kind::GS_FRONT | kind::HS_FRONT | kind::FS => None,
        _ => return Err("invalid shader binary type"),
    })
}

/// Adds `base` to the program address the `lo`/`lo + 1` pair of `registers` holds relative to
/// the binary (`PatchProgramAddressRegister`).
pub fn rebase_program(registers: &mut [ShaderRegister], lo: u32, base: u64) -> Result<(), &'static str> {
    let at = registers.iter().position(|register| register.offset == lo).ok_or("program address lo register not found")?;
    if registers.get(at + 1).map(|register| register.offset) != Some(lo + 1) {
        return Err("program address hi register missing");
    }
    let relative = u64::from(registers[at].value) << 8 | u64::from(registers[at + 1].value & 0xff) << 40;
    if base > 0xffff_ffff_ffff || relative > 0xffff_ffff_ffff - base {
        return Err("program address out of range");
    }
    let address = base + relative;
    registers[at].value = (address >> 8) as u32;
    registers[at + 1].value = (registers[at + 1].value & 0xffff_ff00) | ((address >> 40) & 0xff) as u32;
    Ok(())
}

/// A self-relative pointer of a binary as an absolute one (null stays null).
///
/// # Safety
/// `field` points to a pointer-sized field of the binary.
unsafe fn resolve<T>(field: *mut *mut T) {
    // SAFETY: the caller's contract.
    unsafe {
        let offset = *field as usize;
        if offset != 0 {
            *field = (offset.wrapping_add(field as usize)) as *mut T;
        }
    }
}

agc_export! { fn sceAgcCreateShader(destination: *mut *mut Shader, header: *mut Shader, code: u64) -> i32 {
    const F: &str = "sceAgcCreateShader";
    if destination.is_null() || header.is_null() || code == 0 {
        return error::INVALID_ARGUMENT;
    }
    if code & 0xffff_0000_0000_00ff != 0 {
        return error::INVALID_ALIGNMENT;
    }
    // SAFETY: the title passes the binary's header, in memory it owns.
    let shader = unsafe { &mut *header };
    if shader.file_header != SHADER_MAGIC || shader.version != SHADER_VERSION {
        report(F, "invalid shader header or version");
        return error::INVALID_SHADER_PROGRAM;
    }
    if shader.shader_size == 0 || shader.shader_size % 4 != 0 {
        report(F, "invalid shader code size");
        return error::INVALID_SHADER_PROGRAM;
    }
    let lo = match program_register(shader.kind) {
        Ok(lo) => lo,
        Err(reason) => {
            report(F, reason);
            return error::INVALID_SHADER_PROGRAM;
        }
    };
    // In a binary the pointers hold offsets from themselves; the header becomes the object.
    // SAFETY: the fields lie in the header the title passed (and the user data it points to).
    unsafe {
        resolve(&mut shader.cx_registers);
        resolve(&mut shader.sh_registers);
        resolve(&mut shader.user_data);
        resolve(&mut shader.specials);
        resolve(&mut shader.input_semantics as *mut *const u32 as *mut *mut u32);
        resolve(&mut shader.output_semantics as *mut *const u32 as *mut *mut u32);
        if let Some(user_data) = shader.user_data.as_mut() {
            resolve(&mut user_data.direct_resource_offset);
            for sharp in &mut user_data.sharp_resource_offset {
                resolve(sharp);
            }
        }
    }
    shader.code = code;
    if let Some(lo) = lo {
        if shader.sh_registers.is_null() || shader.num_sh_registers == 0 {
            report(F, "the shader has no program address registers");
            return error::INVALID_SHADER_PROGRAM;
        }
        // SAFETY: `num_sh_registers` registers at the resolved pointer.
        let registers = unsafe { std::slice::from_raw_parts_mut(shader.sh_registers, usize::from(shader.num_sh_registers)) };
        if let Err(reason) = rebase_program(registers, lo, code) {
            report(F, reason);
            return error::INVALID_SHADER_PROGRAM;
        }
    }
    // SAFETY: checked non-null.
    unsafe { *destination = header };
    0
} }

/// A shader's context registers.
///
/// # Safety
/// `shader` was made by `sceAgcCreateShader` (or fused).
unsafe fn cx_registers<'a>(shader: &Shader) -> &'a [ShaderRegister] {
    if shader.cx_registers.is_null() {
        return &[];
    }
    // SAFETY: the caller's contract.
    unsafe { std::slice::from_raw_parts(shader.cx_registers, usize::from(shader.num_cx_registers)) }
}

agc_export! { fn sceAgcGetGsPrimPayload(payload: *mut u32, gs: *const Shader) -> i32 {
    // SAFETY: the title passes a shader object.
    let Some(gs) = (unsafe { gs.as_ref() }) else { return error::INVALID_ARGUMENT };
    if payload.is_null() {
        return error::INVALID_ARGUMENT;
    }
    if gs.kind != kind::GS {
        return 0x8a6c_003c_u32 as i32;
    }
    // SPI_SHADER_IDX_FORMAT 2COMP: primitives carry a payload.
    // SAFETY: a shader object's registers.
    let two_comp = unsafe { cx_registers(gs) }.iter().any(|register| register.offset == 0x1c2 && register.value & 0xf == 2);
    // SAFETY: checked non-null.
    unsafe { *payload = if two_comp { 8 } else { 0 } };
    0
} }

/// `GraphicsPrimTypeToGsOut`: the GS output primitive type of a primitive type.
fn gs_out(primitive: u32) -> Option<u32> {
    Some(match primitive {
        1 => 0,
        2 | 3 | 10 | 11 | 18 => 1,
        7 => 3,
        17 => 4,
        0 | 4 | 5 | 6 | 9 | 12 | 13 | 19 | 20 | 21 => 2,
        _ => return None,
    })
}

agc_export! { fn sceAgcCreatePrimState(cx: *mut ShaderRegister, uc: *mut ShaderRegister, hs: *const Shader, gs: *const Shader, primitive: u32) -> i32 {
    const F: &str = "sceAgcCreatePrimState";
    if cx.is_null() && uc.is_null() {
        return 0;
    }
    // SAFETY: the title passes shader objects.
    let (hs, gs) = unsafe { (hs.as_ref(), gs.as_ref()) };
    let Some(gs) = gs else { return error::INVALID_ARGUMENT };
    if gs.kind != kind::GS || gs.specials.is_null() || hs.is_some_and(|hs| hs.kind != kind::HS || hs.specials.is_null()) {
        return error::INVALID_SHADER_HALVES;
    }
    let Some(out_type) = gs_out(primitive) else {
        report(F, "invalid primitive type");
        return error::OUT_OF_RANGE;
    };
    // SAFETY: checked non-null.
    let (gs_specials, hs_specials) = unsafe { (*gs.specials, hs.map(|hs| *hs.specials)) };
    if gs_specials.vgt_shader_stages_en.offset != 0x2d5
        || hs_specials.is_some_and(|hs| hs.vgt_shader_stages_en.offset != 0x2d5)
        || gs_specials.ge_cntl.offset != 0x25b
    {
        report(F, "special register at an unexpected offset");
        return error::INVALID_SHADER_PROGRAM;
    }
    let mut stages = gs_specials.vgt_shader_stages_en;
    if let Some(hs) = hs_specials {
        stages.value |= hs.vgt_shader_stages_en.value;
    }
    let out = if stages.value & 0x20 != 0 {
        gs_specials.vgt_gs_out_prim_type
    } else if let Some(hs) = hs_specials {
        hs.vgt_gs_out_prim_type
    } else {
        ShaderRegister { offset: 0x29b, value: out_type }
    };
    if out.offset != 0x29b {
        report(F, "VGT_GS_OUT_PRIM_TYPE special register at an unexpected offset");
        return error::INVALID_SHADER_PROGRAM;
    }
    let user_vgprs = hs_specials.unwrap_or(gs_specials).ge_user_vgpr_en;
    let context = [stages, out];
    let primitive_state = [gs_specials.ge_cntl, user_vgprs, ShaderRegister { offset: 0x242, value: primitive }];
    // SAFETY: the title passes room for 2 and 3 registers.
    unsafe {
        if !cx.is_null() {
            std::ptr::copy_nonoverlapping(context.as_ptr(), cx, 2);
        }
        if !uc.is_null() {
            std::ptr::copy_nonoverlapping(primitive_state.as_ptr(), uc, 3);
        }
    }
    0
} }

agc_export! { fn sceAgcUpdatePrimState(cx: *mut ShaderRegister, uc: *mut ShaderRegister, primitive: u32) -> i32 {
    // SAFETY: the title passes the state `sceAgcCreatePrimState` filled.
    unsafe {
        if let Some(stages) = cx.as_mut() {
            if stages.value & (0x20 | 0x04) == 0 {
                let Some(out_type) = gs_out(primitive) else {
                    report("sceAgcUpdatePrimState", "invalid primitive type");
                    return error::OUT_OF_RANGE;
                };
                let out = &mut *cx.add(1);
                out.value = (out.value & !7) | out_type;
            }
        }
        if !uc.is_null() {
            let primitive_type = &mut *uc.add(2);
            primitive_type.value = (primitive_type.value & !0x1f) | primitive;
        }
    }
    0
} }

/// The interpolant control of one pixel input (`SPI_PS_INPUT_CNTL_n`) for the `_10000` mapping
/// (`SplitF16`): `p` is the pixel input semantic, `g` the vertex output it reads, if any.
fn interpolant(p: u32, g: Option<u32>) -> u32 {
    let defaulted = |mut v: u32| {
        v &= !0x1f;
        v &= !0x400;
        (v & !0x300) | ((p >> 28) & 3) << 8
    };
    let mapped = |mut v: u32, g: u32| {
        let flat = if p & 0x0040_0000 != 0 || p & 0x0100_0000 != 0 { 0x400 } else { 0 };
        v = (v & !0x1f) | ((g >> 8) & 0x1f);
        v = (v & !0x400) | flat;
        (v & !0x300) | ((p >> 28) & 3) << 8
    };
    let f16 = |g: Option<u32>| {
        let mut v = (p << 4) & 0x0300_0000;
        match g {
            None => v |= 0x0018_0020,
            Some(g) => {
                let c = p & g;
                v &= 0xfff7_ffdf;
                v |= (c >> 15) & 0x20;
                v ^= 0x0008_0020;
                v &= !0x0010_0000;
                v |= (!c >> 1) & 0x0010_0000;
            }
        }
        (v & !0x0060_0000) | ((p >> 30) & 3) << 21
    };
    if (p >> 20) & 3 == 2 {
        // The high half of a 16-bit pair (`F16Hi`).
        let mut v = ((p << 4) & 0x0300_0000) | 0x0008_0000;
        if g.is_none_or(|g| p & g & 0x0020_0000 == 0) {
            v |= 0x20;
        }
        v = (v & !0x0060_0000) | ((p >> 30) & 3) << 21;
        v |= ((p >> 28) & 3) << 8;
        let Some(g) = g else { return v };
        v |= (g >> 8) & 0x1f;
        if p & 0x0140_0000 != 0 {
            v |= 0x400;
        }
        return v;
    }
    let base = if p & 0x0030_0000 != 0 { f16(g) } else if p & 0x0100_0000 != 0 || g.is_none() { 0x20 } else { 0 };
    match g {
        Some(g) => mapped(base, g),
        None => defaulted(base),
    }
}

/// `CreateInterpolantMapping`: the `SPI_PS_INPUT_CNTL_n` registers that map `ps`'s inputs to
/// `gs`'s outputs; with `fill`, all 32 (the rest as identity).
unsafe fn interpolant_mapping(registers: *mut ShaderRegister, gs: *const Shader, ps: *const Shader, fill: bool, function: &str) -> i32 {
    if registers.is_null() {
        return error::INVALID_ARGUMENT;
    }
    // SAFETY: the title passes shader objects.
    let (gs, ps) = unsafe { (gs.as_ref(), ps.as_ref()) };
    if ps.is_none() && !fill {
        return error::INVALID_ARGUMENT;
    }
    let inputs = ps.map_or(0, |ps| ps.num_input_semantics as usize);
    if inputs > 32 {
        report(function, "more than 32 pixel inputs");
        return error::OUT_OF_RANGE;
    }
    let mut values: [ShaderRegister; 32] = std::array::from_fn(|i| ShaderRegister { offset: 0x191 + i as u32, value: i as u32 });
    if let Some(ps) = ps.filter(|_| inputs > 0) {
        let Some(gs) = gs else { return error::INVALID_ARGUMENT };
        if ps.input_semantics.is_null() || (gs.num_output_semantics != 0 && gs.output_semantics.is_null()) {
            report(function, "missing semantics");
            return error::INVALID_SHADER_PROGRAM;
        }
        // SAFETY: the semantic arrays of shader objects.
        let (inputs_of, outputs) = unsafe {
            (
                std::slice::from_raw_parts(ps.input_semantics, inputs),
                if gs.output_semantics.is_null() { &[][..] } else { std::slice::from_raw_parts(gs.output_semantics, usize::from(gs.num_output_semantics)) },
            )
        };
        for (slot, &p) in values.iter_mut().zip(inputs_of) {
            let output = outputs.iter().copied().find(|&g| g & 0xff == p & 0xff);
            slot.value = interpolant(p, output);
        }
    }
    let count = if fill { 32 } else { inputs };
    // SAFETY: the title passes room for `count` registers.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), registers, count) };
    0
}

agc_export! { fn sceAgcCreateInterpolantMapping_10000(registers: *mut ShaderRegister, gs: *const Shader, ps: *const Shader) -> i32 {
    unsafe { interpolant_mapping(registers, gs, ps, true, "sceAgcCreateInterpolantMapping") }
} }
agc_export! { fn sceAgcUpdateInterpolantMapping_10000(registers: *mut ShaderRegister, gs: *const Shader, ps: *const Shader) -> i32 {
    unsafe { interpolant_mapping(registers, gs, ps, false, "sceAgcUpdateInterpolantMapping") }
} }

/// The register of `shader`'s context list at `offset`.
fn context_register(registers: &[ShaderRegister], offset: u32) -> Option<u32> {
    registers.iter().find(|register| register.offset == offset).map(|register| register.value)
}

agc_export! { fn sceAgcGetGsOversubscription(registers: *mut ShaderRegister, gs: *const Shader, budget: i32, factor: f32) -> i32 {
    const F: &str = "sceAgcGetGsOversubscription";
    // SAFETY: the title passes a shader object.
    let Some(gs) = (unsafe { gs.as_ref() }) else { return error::INVALID_ARGUMENT };
    if registers.is_null() {
        return error::INVALID_ARGUMENT;
    }
    if budget < -1 || factor.is_nan() || factor < 0.0 {
        return error::OUT_OF_RANGE;
    }
    if gs.kind != kind::GS {
        return error::INVALID_SHADER_HALVES;
    }
    let (mut alloc, mut rsrc4) = (0u32, 0u32);
    if budget == -1 {
        (alloc, rsrc4) = (0x7ff, 0x007f_0000);
    } else if budget != 0 {
        if gs.specials.is_null() {
            report(F, "the shader has no special registers");
            return error::INVALID_SHADER_PROGRAM;
        }
        // SAFETY: shader object data.
        let (cx, stages) = unsafe { (cx_registers(gs), (*gs.specials).vgt_shader_stages_en.value) };
        let limits = |vertices: u32, exports_cap: u32| -> Option<(u32, u32)> {
            let read = |offset| context_register(cx, offset);
            let (onchip, subgroup, vs_out, cl_out, max_out) = (read(0x291)?, read(0x2d3)?, read(0x1b1)?, read(0x207)?, read(0x1ff)?);
            let out_waves = ((max_out & 0x3ff) + 31) >> 5;
            if out_waves == 0 {
                return None;
            }
            let work = ((((onchip >> 11) & 0x7ff) * (subgroup & 0x1ff)) + 31) >> 5;
            let mut waves = work.max(out_waves);
            if stages & 0x0040_0000 == 0 {
                waves >>= 1;
            }
            let waves = waves.max(1);
            let exports = 1 + (cl_out >> 21 & 1) + (cl_out >> 22 & 1) + (cl_out >> 23 & 1);
            let export_limit = (exports_cap / exports) * 4;
            let vertex_limit = if vs_out & 0x80 != 0 { 2048 } else { vertices / (((vs_out >> 2) & 0xf) + 1) };
            Some((waves * (vertex_limit / out_waves), waves * (export_limit / out_waves)))
        };
        let (Some(base), Some(extended)) = (limits(1024, 128), limits(2048, 382)) else {
            report(F, "the shader lacks a context register the limits need");
            return error::INVALID_SHADER_PROGRAM;
        };
        let base_min = base.0.min(base.1);
        let shift = if stages & 0x0040_0000 != 0 { 5 } else { 6 };
        let extended_limit = extended.0.min(extended.1).min((budget as u32) >> shift).min(1024);
        let head = extended_limit.saturating_sub(base_min);
        let target = factor.mul_add(head as f32, base_min as f32);
        if !(0.0..4_294_967_296.0).contains(&target) {
            return error::OUT_OF_RANGE;
        }
        let target = target as u32;
        if target > base_min {
            if target < base.1 {
                let range = u64::from(extended.0.saturating_sub(base.0).max(1));
                let value = ((u64::from(target.saturating_sub(base.0)) << 10) / range).clamp(1, 1024);
                (alloc, rsrc4) = (((value << 1) - 1) as u32 & 0x7ff, 0x007f_0000);
            } else {
                let range = u64::from(extended.1.saturating_sub(base.1).max(1));
                let value = (u64::from(target - base.1) * 127 / range).min(127) as u32;
                (alloc, rsrc4) = (0x7ff, value << 16);
            }
        }
    }
    // SAFETY: the title passes room for two registers.
    unsafe {
        *registers = ShaderRegister { offset: 0x260, value: alloc };
        *registers.add(1) = ShaderRegister { offset: 0x081, value: rsrc4 };
    }
    0
} }

/// The libSceAgc functions this module exports, for `agc::exports`.
pub const SHADERS: &[&str] = &[
    "sceAgcCreateShader", "sceAgcGetGsPrimPayload", "sceAgcCreatePrimState", "sceAgcUpdatePrimState",
    "sceAgcCreateInterpolantMapping_10000", "sceAgcUpdateInterpolantMapping_10000", "sceAgcGetGsOversubscription",
];
