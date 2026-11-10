//! `sceAgcFuseShaderHalves` and `sceAgcGetFusedShaderSize` (PLAN P3.4 step 2f), as AnyPS5's
//! `Misc/src/ShaderFusion.cpp` does them: hull halves are fused in place; geometry halves become
//! one program in memory the runtime owns (front code, `s_nop` padding to 256 bytes, back code,
//! the front's embedded constants), built once per content and shared.

use super::report;
use super::shader::{error, kind, rebase_program, Shader, ShaderSpecialRegs, ShaderUserData};
use crate::agc::cache::{self, Table};
use crate::agc::ShaderRegister;
use crate::agc_export;
use std::sync::{Arc, OnceLock};

const RSRC1_GS: u32 = 0x08a;
const RSRC2_GS: u32 = 0x08b;
const CHKSUM_HS: u32 = 0x100;
const RSRC1_HS: u32 = 0x10a;
const RSRC2_HS: u32 = 0x10b;
const PGM_LO_LS: u32 = 0x148;
const PGM_LO_GS: u32 = 0x088;
const S_CODE_END: u32 = 0xbf9f_0000;
const S_NOP: u32 = 0xbf80_0000;
const S_GETPC_MASK: u32 = 0xff80_ff00;
const S_GETPC: u32 = 0xbe80_1f00;
const S_SETPC_S6: u32 = 0xbe80_2006;
const S_SWAPPC_NULL_S6: u32 = 0xbefd_2106;
const CODE_ALIGNMENT: usize = 256;
const TRAILER_MAGIC: &[u8; 8] = b"barefoot";
const TRAILER_CODE_BYTES: usize = 8 + 3 * 4;
const RSRC1_VGPRS: u32 = 0x3f;
const RSRC2_OFFCHIP_LDS: u32 = 1 << 18;
const RSRC2_USER_SGPRS: u32 = (0x1f << 1) | (1 << 27);

type Fail = String;

fn align(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// # Safety
/// `pointer` holds `count` elements when not null.
unsafe fn slice<'a, T>(pointer: *const T, count: usize) -> &'a [T] {
    if pointer.is_null() || count == 0 {
        &[]
    } else {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts(pointer, count) }
    }
}

fn find(registers: &[ShaderRegister], offset: u32, occurrence: usize) -> Result<usize, Fail> {
    registers
        .iter()
        .enumerate()
        .filter(|(_, register)| register.offset == offset)
        .nth(occurrence)
        .map(|(at, _)| at)
        .ok_or_else(|| format!("shader half lacks register {offset}"))
}

fn merge_max(destination: &mut ShaderRegister, source: &ShaderRegister, shift: u32, mask: u32) {
    let field = ((destination.value >> shift) & mask).max((source.value >> shift) & mask);
    destination.value = (destination.value & !(mask << shift)) | (field << shift);
}

/// Shared VGPR blocks of a merged wave: what neither half's own allocation covers.
fn shared_vgprs(front: (u32, u32), back: (u32, u32)) -> u32 {
    let front_vgprs = ((front.0 & RSRC1_VGPRS) + 1) * 4;
    let back_vgprs = ((back.0 & RSRC1_VGPRS) + 1) * 4;
    let front_total = front_vgprs + (front.1 >> 28) * 8;
    let back_total = back_vgprs + (back.1 >> 28) * 8;
    let total = front_total.max(back_total);
    if front_vgprs.max(back_vgprs) >= total {
        0
    } else {
        (total - front_total.min(back_total) + 7) / 64
    }
}

struct GeometryLayout {
    front_bytes: usize,
    constants_offset: usize,
    constants_bytes: usize,
    constants_source: usize,
    literals: Vec<(usize, u32)>,
    back_offset: usize,
    code_bytes: usize,
    sh: Vec<ShaderRegister>,
    cx: Vec<ShaderRegister>,
    header_offset: usize,
    sh_offset: usize,
    cx_offset: usize,
    specials_offset: usize,
    inputs_offset: usize,
    outputs_offset: usize,
    user_data_offset: usize,
    direct_offset: usize,
    sharp_offsets: [usize; 4],
    total_bytes: usize,
}

fn validate_half(shader: &Shader) -> Result<(), Fail> {
    if shader.file_header != 0x3433_3231 || shader.version != 0x18 {
        return Err("invalid shader header or version".into());
    }
    if shader.code == 0 || shader.shader_size == 0 || shader.shader_size % 4 != 0 {
        return Err("invalid shader half code".into());
    }
    if (shader.num_sh_registers != 0 && shader.sh_registers.is_null()) || (shader.num_cx_registers != 0 && shader.cx_registers.is_null()) {
        return Err("shader half registers missing".into());
    }
    Ok(())
}

/// The front half's program, without its closing jump to the back half: (program bytes,
/// trailer code bytes, trailer offset).
fn front_program(front: &Shader) -> Result<(usize, usize, usize), Fail> {
    let size = front.shader_size as usize;
    // SAFETY: a validated half's code.
    let bytes = unsafe { slice(front.code as *const u8, size) };
    if size < TRAILER_CODE_BYTES + 4 {
        return Err("front half has no program trailer".into());
    }
    let trailer = (0..=size - TRAILER_CODE_BYTES - 4)
        .rev()
        .find(|&offset| &bytes[offset..offset + 8] == TRAILER_MAGIC)
        .ok_or("front half has no program trailer")?;
    let code_bytes = u32::from_le_bytes(bytes[trailer + TRAILER_CODE_BYTES..][..4].try_into().unwrap()) as usize;
    if code_bytes == 0 || code_bytes % 4 != 0 || code_bytes > trailer {
        return Err("front half trailer has an invalid code size".into());
    }
    let word = |index: usize| u32::from_le_bytes(bytes[index * 4..][..4].try_into().unwrap());
    let mut end = code_bytes / 4;
    while end > 0 && word(end - 1) == S_CODE_END {
        end -= 1;
    }
    if end == 0 || (word(end - 1) != S_SETPC_S6 && word(end - 1) != S_SWAPPC_NULL_S6) {
        return Err("front half does not end with s_setpc_b64 s[6:7] or s_swappc_b64 null, s[6:7]".into());
    }
    Ok(((end - 1) * 4, code_bytes, trailer))
}

/// The literals of the front's `s_getpc_b64` / `s_add_u32` / `s_addc_u32` sequences that reach
/// its embedded constants, moved to `moved`.
fn constant_literals(front: &Shader, program_bytes: usize, constants: usize, constants_bytes: usize, moved: usize) -> Result<Vec<(usize, u32)>, Fail> {
    let words_count = program_bytes / 4;
    // SAFETY: a validated half's code.
    let code: &[u32] = unsafe { slice(front.code as *const u32, words_count) };
    let decoder = shader::decode::Decoder::new();
    let mut literals = Vec::new();
    let mut i = 0;
    while i < words_count {
        let length = decoder
            .decode_at(code, i)
            .map(|instruction| instruction.word_count())
            .map_err(|error| format!("front half with an embedded constant buffer holds an instruction that cannot be decoded at word {i}: {error}"))?;
        let word = code[i];
        if word & S_GETPC_MASK == S_GETPC {
            let register = (word >> 16) & 0x7f;
            let add = 0x8000_0000 | register << 16 | register << 8 | 0xff;
            let carry = 0x8200_0000 | (register + 1) << 16 | (register + 1) << 8 | 0x80;
            if i + 3 >= words_count || code[i + 1] != add || code[i + 3] != carry {
                return Err("front half reads its embedded constant buffer through an s_getpc_b64 not followed by s_add_u32 with a literal and s_addc_u32 0".into());
            }
            let target = (i + 1) * 4 + code[i + 2] as usize;
            if target >= program_bytes {
                if target < constants || target >= constants + constants_bytes {
                    return Err("front half's s_getpc_b64 offset points outside its program and embedded constant buffer".into());
                }
                literals.push(((i + 2) * 4, code[i + 2].wrapping_add((moved - constants) as u32)));
            }
        }
        i += length.max(1);
    }
    Ok(literals)
}

fn geometry_resources(front: (u32, u32), back: (u32, u32)) -> (u32, u32) {
    let shared = shared_vgprs(front, back);
    let vgprs = (front.0 & RSRC1_VGPRS).max(back.0 & RSRC1_VGPRS);
    let rsrc1_field = ((front.0 >> 29) & 3).max((back.0 >> 29) & 3);
    let rsrc2_field = ((front.1 >> 16) & 3).max((back.1 >> 16) & 3);
    let rsrc1 = (back.0 & !(RSRC1_VGPRS | (3 << 29))) | vgprs | (rsrc1_field << 29);
    let rsrc2 = (back.1 & !(0xf000_0000 | (3 << 16) | RSRC2_USER_SGPRS | RSRC2_OFFCHIP_LDS))
        | ((shared & 15) << 28)
        | (rsrc2_field << 16)
        | (front.1 & (RSRC2_USER_SGPRS | RSRC2_OFFCHIP_LDS));
    (rsrc1, rsrc2)
}

fn half_register(shader: &Shader, offset: u32) -> Option<u32> {
    // SAFETY: a validated half's registers.
    unsafe { slice(shader.sh_registers, usize::from(shader.num_sh_registers)) }
        .iter()
        .find(|register| register.offset == offset)
        .map(|register| register.value)
}

fn geometry_layout(front: &Shader, back: &Shader) -> Result<GeometryLayout, Fail> {
    validate_half(back)?;
    validate_half(front)?;
    if back.embedded_constant_buffer_size_dqw != 0 {
        return Err("fusing a back half with an embedded constant buffer is not implemented".into());
    }
    let (front_bytes, program_code_bytes, trailer) = front_program(front)?;
    let back_offset = align(front_bytes, CODE_ALIGNMENT);
    let mut code_bytes = back_offset + back.shader_size as usize;
    let constants_bytes = front.embedded_constant_buffer_size_dqw as usize * 16;
    let (mut constants_source, mut constants_offset, mut literals) = (0, 0, Vec::new());
    if constants_bytes != 0 {
        constants_source = program_code_bytes;
        if constants_source + constants_bytes > trailer {
            return Err("front half's embedded constant buffer runs into its program trailer".into());
        }
        constants_offset = align(code_bytes, 16);
        literals = constant_literals(front, front_bytes, constants_source, constants_bytes, constants_offset)?;
        code_bytes = constants_offset + constants_bytes;
    }
    let (Some(front_rsrc1), Some(front_rsrc2)) = (half_register(front, RSRC1_GS), half_register(front, RSRC2_GS)) else {
        return Err("front half has no GS resource registers".into());
    };
    let (Some(back_rsrc1), Some(back_rsrc2)) = (half_register(back, RSRC1_GS), half_register(back, RSRC2_GS)) else {
        return Err("back half has no GS resource registers".into());
    };
    let (rsrc1, rsrc2) = geometry_resources((front_rsrc1, front_rsrc2), (back_rsrc1, back_rsrc2));
    // SAFETY: a validated half's registers.
    let mut sh: Vec<ShaderRegister> = unsafe { slice(back.sh_registers, usize::from(back.num_sh_registers)) }.to_vec();
    for register in &mut sh {
        if register.offset == RSRC1_GS {
            register.value = rsrc1;
        } else if register.offset == RSRC2_GS {
            register.value = rsrc2;
        }
    }
    // SAFETY: a validated half's registers.
    let cx: Vec<ShaderRegister> = unsafe { slice(back.cx_registers, usize::from(back.num_cx_registers)) }.to_vec();
    if sh.len() > 0xff || cx.len() > 0xff {
        return Err("fused register count exceeds the shader header".into());
    }
    let header_offset = align(code_bytes, 8);
    let sh_offset = align(header_offset + std::mem::size_of::<Shader>(), 4);
    let cx_offset = sh_offset + sh.len() * 8;
    let specials_offset = align(cx_offset + cx.len() * 8, 4);
    let inputs_offset = align(specials_offset + usize::from(back.special_sizes_bytes), 4);
    let outputs_offset = inputs_offset + front.num_input_semantics as usize * 4;
    let user_data_offset = align(outputs_offset + usize::from(back.num_output_semantics) * 4, 8);
    // SAFETY: a validated half's user data.
    let user_data = unsafe { back.user_data.as_ref() };
    let mut end = user_data_offset + if user_data.is_some() { std::mem::size_of::<ShaderUserData>() } else { 0 };
    let direct_offset = align(end, 2);
    if let Some(user_data) = user_data {
        end = direct_offset + usize::from(user_data.direct_resource_count) * 2;
    }
    let mut sharp_offsets = [0; 4];
    for (index, offset) in sharp_offsets.iter_mut().enumerate() {
        *offset = align(end, 2);
        if let Some(user_data) = user_data {
            end = *offset + usize::from(user_data.sharp_resource_count[index]) * 2;
        }
    }
    Ok(GeometryLayout {
        front_bytes,
        constants_offset,
        constants_bytes,
        constants_source,
        literals,
        back_offset,
        code_bytes,
        sh,
        cx,
        header_offset,
        sh_offset,
        cx_offset,
        specials_offset,
        inputs_offset,
        outputs_offset,
        user_data_offset,
        direct_offset,
        sharp_offsets,
        total_bytes: end,
    })
}

/// What a fused program depends on, byte for byte (`FusionKey`).
fn fusion_key(front: &Shader, back: &Shader, layout: &GeometryLayout) -> Vec<u8> {
    let mut key = Vec::new();
    let mut append = |pointer: *const u8, bytes: usize| {
        // SAFETY: each call names memory of the halves (validated) or of the layout.
        key.extend_from_slice(unsafe { slice(pointer, bytes) });
    };
    // The header up to num_sh_registers: 0x5d bytes.
    append(front as *const Shader as *const u8, 0x5d);
    append(back as *const Shader as *const u8, 0x5d);
    append(front.code as *const u8, layout.front_bytes);
    append(back.code as *const u8, back.shader_size as usize);
    if layout.constants_bytes != 0 {
        append((front.code as usize + layout.constants_source) as *const u8, layout.constants_bytes);
    }
    append(layout.sh.as_ptr() as *const u8, layout.sh.len() * 8);
    append(layout.cx.as_ptr() as *const u8, layout.cx.len() * 8);
    append(back.specials as *const u8, usize::from(back.special_sizes_bytes));
    append(front.input_semantics as *const u8, front.num_input_semantics as usize * 4);
    append(back.output_semantics as *const u8, usize::from(back.num_output_semantics) * 4);
    // SAFETY: a validated half's user data.
    if let Some(user_data) = unsafe { back.user_data.as_ref() } {
        append(user_data as *const ShaderUserData as *const u8, 0x36);
        append(user_data.direct_resource_offset as *const u8, usize::from(user_data.direct_resource_count) * 2);
        for index in 0..4 {
            append(user_data.sharp_resource_offset[index] as *const u8, usize::from(user_data.sharp_resource_count[index]) * 2);
        }
    }
    key
}

/// A fused geometry program: its memory (never freed) and header.
struct Fused {
    key: Vec<u8>,
    _storage: Box<[u8]>,
    header: *const Shader,
}

// SAFETY: the storage is written once, before the program is shared, and never again.
unsafe impl Send for Fused {}
unsafe impl Sync for Fused {}

fn build(front: &Shader, back: &Shader, mut layout: GeometryLayout, key: Vec<u8>) -> Result<Fused, Fail> {
    let mut storage = vec![0u8; layout.total_bytes + CODE_ALIGNMENT - 1].into_boxed_slice();
    let start = storage.as_ptr() as usize;
    let base = align(start, CODE_ALIGNMENT);
    if base as u64 & 0xffff_0000_0000_00ff != 0 {
        return Err("fused shader memory is not a valid program address".into());
    }
    let memory = &mut storage[base - start..];
    // SAFETY: validated halves' code.
    let (front_code, back_code) =
        unsafe { (slice(front.code as *const u8, front.shader_size as usize), slice(back.code as *const u8, back.shader_size as usize)) };
    memory[..layout.front_bytes].copy_from_slice(&front_code[..layout.front_bytes]);
    for offset in (layout.front_bytes..layout.back_offset).step_by(4) {
        memory[offset..offset + 4].copy_from_slice(&S_NOP.to_le_bytes());
    }
    memory[layout.back_offset..layout.back_offset + back_code.len()].copy_from_slice(back_code);
    if layout.constants_bytes != 0 {
        let source = &front_code[layout.constants_source..layout.constants_source + layout.constants_bytes];
        memory[layout.constants_offset..layout.constants_offset + layout.constants_bytes].copy_from_slice(source);
        for &(offset, literal) in &layout.literals {
            memory[offset..offset + 4].copy_from_slice(&literal.to_le_bytes());
        }
    }
    let base = base as u64;
    // The back half runs from its own place; the front's address is the program's.
    let pair = (0..layout.sh.len().saturating_sub(1))
        .find(|&at| layout.sh[at].offset == PGM_LO_GS && layout.sh[at + 1].offset == PGM_LO_GS + 1)
        .ok_or("back half has no program address register pair")?;
    let back_address = base + layout.back_offset as u64;
    layout.sh[pair].value = (back_address >> 8) as u32;
    layout.sh[pair + 1].value = (layout.sh[pair + 1].value & 0xffff_ff00) | ((back_address >> 40) & 0xff) as u32;
    rebase_program(&mut layout.sh, 0x0c8, base).map_err(|reason| format!("fused program address patch failed: {reason}"))?;
    let at = |offset: usize| memory.as_ptr() as usize + offset;
    // SAFETY: every array goes to its place in `memory`, which the layout sized.
    unsafe {
        let copy = |from: *const u8, to: usize, bytes: usize| {
            if bytes != 0 && !from.is_null() {
                std::ptr::copy_nonoverlapping(from, to as *mut u8, bytes);
            }
        };
        copy(layout.sh.as_ptr() as *const u8, at(layout.sh_offset), layout.sh.len() * 8);
        copy(layout.cx.as_ptr() as *const u8, at(layout.cx_offset), layout.cx.len() * 8);
        copy(back.specials as *const u8, at(layout.specials_offset), usize::from(back.special_sizes_bytes));
        copy(front.input_semantics as *const u8, at(layout.inputs_offset), front.num_input_semantics as usize * 4);
        copy(back.output_semantics as *const u8, at(layout.outputs_offset), usize::from(back.num_output_semantics) * 4);
        let mut fused = *back;
        if let Some(user_data) = back.user_data.as_ref() {
            let copied = at(layout.user_data_offset) as *mut ShaderUserData;
            *copied = *user_data;
            let direct = at(layout.direct_offset);
            copy(user_data.direct_resource_offset as *const u8, direct, usize::from(user_data.direct_resource_count) * 2);
            (*copied).direct_resource_offset = if user_data.direct_resource_count != 0 { direct as *mut u16 } else { std::ptr::null_mut() };
            for index in 0..4 {
                let sharps = at(layout.sharp_offsets[index]);
                let count = usize::from(user_data.sharp_resource_count[index]);
                copy(user_data.sharp_resource_offset[index] as *const u8, sharps, count * 2);
                (*copied).sharp_resource_offset[index] = if count != 0 { sharps as *mut u16 } else { std::ptr::null_mut() };
            }
            fused.user_data = copied;
        }
        fused.header_size = (layout.total_bytes - layout.header_offset) as u32;
        fused.code = base;
        fused.sh_registers = at(layout.sh_offset) as *mut ShaderRegister;
        fused.cx_registers = if layout.cx.is_empty() { std::ptr::null_mut() } else { at(layout.cx_offset) as *mut ShaderRegister };
        fused.specials = if back.special_sizes_bytes != 0 { at(layout.specials_offset) as *mut ShaderSpecialRegs } else { std::ptr::null_mut() };
        fused.input_semantics = if front.num_input_semantics != 0 { at(layout.inputs_offset) as *const u32 } else { std::ptr::null() };
        fused.output_semantics = if back.num_output_semantics != 0 { at(layout.outputs_offset) as *const u32 } else { std::ptr::null() };
        fused.shader_size = layout.code_bytes as u32;
        fused.num_input_semantics = front.num_input_semantics;
        fused.scratch_size_dw_per_thread = front.scratch_size_dw_per_thread.max(back.scratch_size_dw_per_thread);
        fused.kind = kind::GS;
        fused.num_sh_registers = layout.sh.len() as u8;
        fused.num_cx_registers = layout.cx.len() as u8;
        let header = at(layout.header_offset) as *mut Shader;
        *header = fused;
        Ok(Fused { key, _storage: storage, header })
    }
}

fn programs() -> &'static Table<Fused> {
    static PROGRAMS: OnceLock<Table<Fused>> = OnceLock::new();
    PROGRAMS.get_or_init(|| Table::with_capacity(1 << 12))
}

fn fuse_geometry(result: &mut Shader, front: &Shader, back: &Shader, scratch: *mut ShaderRegister) -> Result<(), Fail> {
    if scratch.is_null() {
        return Err("geometry halves need fused shader memory".into());
    }
    let layout = geometry_layout(front, back)?;
    let key = fusion_key(front, back, &layout);
    let hash = cache::key(key.chunks(8).map(|chunk| {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        u64::from_le_bytes(word)
    }));
    let program = match programs().get(hash) {
        Some(program) if program.key == key => program,
        // A hash collision builds the program without caching it.
        Some(_) => Arc::new(build(front, back, layout, key)?),
        None => programs().insert(hash, Arc::new(build(front, back, layout, key)?)),
    };
    // SAFETY: a built program's header; the program lives for the process.
    let mut fused = unsafe { *program.header };
    // SAFETY: the title's scratch holds `num_sh_registers` registers (`sceAgcGetFusedShaderSize`).
    unsafe { std::ptr::copy_nonoverlapping(fused.sh_registers, scratch, usize::from(fused.num_sh_registers)) };
    fused.sh_registers = scratch;
    *result = fused;
    std::mem::forget(program);
    Ok(())
}

fn fuse_hull(result: &mut Shader, front: &Shader, back: &Shader, scratch: *mut ShaderRegister) -> Result<(), Fail> {
    *result = *back;
    result.kind = kind::HS;
    if !scratch.is_null() {
        // SAFETY: the title's scratch holds the back's registers.
        unsafe { std::ptr::copy_nonoverlapping(back.sh_registers, scratch, usize::from(back.num_sh_registers)) };
        result.sh_registers = scratch;
    }
    // SAFETY: the registers of shader objects (with no scratch, the back's own, as AnyPS5).
    let (fused, front_registers) = unsafe {
        (
            std::slice::from_raw_parts_mut(result.sh_registers, usize::from(result.num_sh_registers)),
            slice(front.sh_registers, usize::from(front.num_sh_registers)),
        )
    };
    for occurrence in 0..2 {
        let to = find(fused, CHKSUM_HS, occurrence)?;
        fused[to].value = front_registers[find(front_registers, CHKSUM_HS, occurrence)?].value;
    }
    let front_rsrc1 = front_registers[find(front_registers, RSRC1_HS, 0)?];
    let front_rsrc2 = front_registers[find(front_registers, RSRC2_HS, 0)?];
    let (rsrc1, rsrc2) = (find(fused, RSRC1_HS, 0)?, find(fused, RSRC2_HS, 0)?);
    let shared = shared_vgprs((front_rsrc1.value, front_rsrc2.value), (fused[rsrc1].value, fused[rsrc2].value));
    fused[rsrc2].value = (fused[rsrc2].value & 0x0fff_ffff) | ((shared & 0xf) << 28);
    merge_max(&mut fused[rsrc1], &front_rsrc1, 0, 0x3f);
    merge_max(&mut fused[rsrc1], &front_rsrc1, 28, 0x3);
    fused[rsrc2].value = (fused[rsrc2].value & 0xf7ff_ffc1) | (front_rsrc2.value & 0x0800_003e);
    let (lo, hi) = (find(fused, PGM_LO_LS, 0)?, find(fused, PGM_LO_LS + 1, 0)?);
    fused[lo].value = (front.code >> 8) as u32;
    fused[hi].value = (fused[hi].value & 0xffff_ff00) | ((front.code >> 40) & 0xff) as u32;
    result.user_data = std::ptr::null_mut();
    Ok(())
}

fn valid_halves(front: &Shader, back: &Shader) -> bool {
    (front.kind == kind::GS_FRONT && back.kind == kind::GS_BACK) || (front.kind == kind::HS_FRONT && back.kind == kind::HS_BACK)
}

agc_export! { fn sceAgcFuseShaderHalves_0200(result: *mut Shader, front: *const Shader, back: *const Shader, scratch: *mut ShaderRegister) -> i32 {
    const F: &str = "sceAgcFuseShaderHalves";
    // SAFETY: the title passes shader objects and the result to fill.
    let (Some(result), Some(front), Some(back)) = (unsafe { result.as_mut() }, unsafe { front.as_ref() }, unsafe { back.as_ref() }) else {
        return error::INVALID_ARGUMENT;
    };
    if !valid_halves(front, back) || front.specials.is_null() || back.specials.is_null() {
        return error::INVALID_SHADER_HALVES;
    }
    let geometry = front.kind == kind::GS_FRONT;
    let wave_bit = if geometry { 1 << 22 } else { 1 << 21 };
    // SAFETY: checked non-null.
    let (front_stages, back_stages) = unsafe { ((*front.specials).vgt_shader_stages_en.value, (*back.specials).vgt_shader_stages_en.value) };
    if (front_stages ^ back_stages) & wave_bit != 0 {
        return error::INVALID_SHADER_HALVES;
    }
    let fused = if geometry { fuse_geometry(result, front, back, scratch) } else { fuse_hull(result, front, back, scratch) };
    match fused {
        Ok(()) => 0,
        Err(reason) => {
            report(F, &reason);
            error::INVALID_SHADER_PROGRAM
        }
    }
} }

/// `SizeAlign`.
#[repr(C)]
pub struct SizeAlign {
    pub size: u64,
    pub align: usize,
}

agc_export! { fn sceAgcGetFusedShaderSize_0080(destination: *mut SizeAlign, front: *const Shader, back: *const Shader) -> i32 {
    // SAFETY: the title passes shader objects and the result to fill.
    let (Some(destination), Some(front), Some(back)) = (unsafe { destination.as_mut() }, unsafe { front.as_ref() }, unsafe { back.as_ref() }) else {
        return error::INVALID_ARGUMENT;
    };
    if !valid_halves(front, back) {
        return error::INVALID_SHADER_HALVES;
    }
    if front.kind == kind::GS_FRONT {
        if let Err(reason) = geometry_layout(front, back) {
            report("sceAgcGetFusedShaderSize", &reason);
            return error::INVALID_SHADER_PROGRAM;
        }
    }
    *destination = SizeAlign { size: u64::from(back.num_sh_registers) * 8, align: 4 };
    0
} }

/// The libSceAgc functions this module exports, for `agc::exports`.
pub const FUSION: &[&str] = &["sceAgcFuseShaderHalves_0200", "sceAgcGetFusedShaderSize_0080"];
