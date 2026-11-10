//! GPU display-conversion packets: the main image and the immutable overlay snapshot.
use super::kernels::display_convert;
use crate::{amd::defs::pkt3, gpu::{packets::{PacketStream, SHADER_TYPE_COMPUTE}, regs as host}, ir::RegisterBank, Error, Result};
use shader::ShaderArtifact;

/// COMPUTE_STATIC_THREAD_MGMT_SE0/SE1 and SE2/SE3 (SET_SH_REG offsets, RDNA3 register map).
const COMPUTE_STATIC_THREAD_MGMT_SE0: u32 = 0x216;
const COMPUTE_STATIC_THREAD_MGMT_SE2: u32 = 0x219;
/// Every CU of every shader engine may run the dispatch.
const ALL_CUS: u32 = 0xffff_ffff;
/// COMPUTE_DISPATCH_INITIATOR: COMPUTE_SHADER_EN, FORCE_START_AT_000 and CS_W32_EN.
const CS_SHADER_EN: u32 = 1 << 0;
const CS_FORCE_START_AT_000: u32 = 1 << 2;
const CS_W32_EN: u32 = 1 << 15;
/// EVENT_WRITE CS_PARTIAL_FLUSH.
const EVENT_CS_PARTIAL_FLUSH: u32 = 0x07;
const EVENT_INDEX_PARTIAL_FLUSH: u32 = 4;
/// ACQUIRE_MEM GCR_CNTL: invalidate the instruction, scalar, vector, GL1 and GL2 caches before
/// the kernel reads its code, parameters and source; write back and invalidate GL2 and the
/// metadata cache after it, so D3D12's copy sees the pixels.
const GCR_GLI_INV: u32 = 1 << 0;
const GCR_GLM_WB: u32 = 1 << 4;
const GCR_GLK_INV: u32 = 1 << 7;
const GCR_GLV_INV: u32 = 1 << 8;
const GCR_GL1_INV: u32 = 1 << 9;
const GCR_GL2_INV: u32 = 1 << 14;
const GCR_GL2_WB: u32 = 1 << 15;
const INVALIDATE_READ_CACHES: u32 =
    GCR_GLI_INV | GCR_GLK_INV | GCR_GLV_INV | GCR_GL1_INV | GCR_GL2_INV;
const WRITE_BACK: u32 = GCR_GLM_WB | GCR_GL2_WB | GCR_GL2_INV | GCR_GLV_INV | GCR_GL1_INV;

/// The host PM4 of one conversion on the graphics ring: invalidate the read caches, dispatch the
/// translated kernel at `code` with the parameter block at `params`, wait for it and write its
/// pixels back. The caller signals F_ready after the submission.
pub fn conversion_stream(
    artifact: &ShaderArtifact,
    code: u64,
    params: u64,
    groups: [u32; 3],
) -> Result<Vec<u32>> {
    conversion_passes(artifact, code, &[(params, groups)])
}

/// [`conversion_stream`] with one dispatch per pass, in order (the main buffer, then the overlay
/// plane over it): each pass waits for the one before and reads its pixels through invalidated
/// vector and GL1 caches.
pub fn conversion_passes(
    artifact: &ShaderArtifact,
    code: u64,
    passes: &[(u64, [u32; 3])],
) -> Result<Vec<u32>> {
    let Some(&(first, _)) = passes.first() else {
        return Err(Error::invalid("present: a conversion needs a pass"));
    };
    let mut stream = PacketStream::new();
    stream.acquire_mem_global(INVALIDATE_READ_CACHES)?;
    for (dword, values) in [
        (host::COMPUTE_START.dword, vec![0, 0, 0]),
        (host::COMPUTE_NUM_THREAD.dword, vec![display_convert::GROUP_THREADS, 1, 1]),
        (host::COMPUTE_PGM_LO.dword, vec![(code >> 8) as u32, (code >> 40) as u32 & 0xff]),
        (host::COMPUTE_PGM_RSRC1.dword, vec![artifact.pgm_rsrc1, artifact.pgm_rsrc2]),
        (host::COMPUTE_RESOURCE_LIMITS.dword, vec![0]),
        (COMPUTE_STATIC_THREAD_MGMT_SE0, vec![ALL_CUS, ALL_CUS]),
        (COMPUTE_STATIC_THREAD_MGMT_SE2, vec![ALL_CUS, ALL_CUS]),
        (host::COMPUTE_TMPRING_SIZE.dword, vec![0]),
        (host::COMPUTE_PGM_RSRC3.dword, vec![0]),
        (host::COMPUTE_USER_DATA.dword, vec![first as u32, (first >> 32) as u32]),
    ] {
        stream.set_registers(RegisterBank::Shader, dword, &values)?;
    }
    let wave32 = if artifact.options.wave64() { 0 } else { CS_W32_EN };
    for (index, &(params, [x, y, z])) in passes.iter().enumerate() {
        if index > 0 {
            stream.event_write(EVENT_CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH)?;
            stream.acquire_mem_global(GCR_GLV_INV | GCR_GL1_INV)?;
            stream.set_run(host::COMPUTE_USER_DATA, &[params as u32, (params >> 32) as u32])?;
        }
        let initiator = CS_SHADER_EN | CS_FORCE_START_AT_000 | wave32;
        stream.raw_packet(pkt3::DISPATCH_DIRECT, SHADER_TYPE_COMPUTE, &[x, y, z, initiator])?;
    }
    stream.event_write(EVENT_CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH)?;
    stream.acquire_mem_global(WRITE_BACK)?;
    Ok(stream.into_words())
}
