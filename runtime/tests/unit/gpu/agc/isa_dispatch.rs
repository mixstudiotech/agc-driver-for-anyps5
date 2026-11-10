use crate::gpu::{
    packets::{PacketStream, SHADER_TYPE_COMPUTE, event, gcr},
    regs as host,
};
use crate::{Buffer, Device};
use shader::ShaderArtifact;

pub(super) fn run(
    device: &mut Device,
    code: &mut Buffer,
    output: &mut Buffer,
    artifact: &ShaderArtifact,
    input: &[u32; 12],
) -> Vec<u32> {
    code.write(0, &artifact.code_bytes()).unwrap();
    output.fill(0xcd).unwrap();
    let mut user = *input;
    // Mesa ac_build_raw_buffer_descriptor: byte size and OOB_SELECT_RAW=3.
    user[8..12].copy_from_slice(&[
        output.address() as u32,
        (output.address() >> 32) as u32,
        4096,
        0x3001_6fac,
    ]);
    let mut stream = PacketStream::new_compute();
    stream.acquire_mem_global(gcr::INVALIDATE_ALL).unwrap();
    stream
        .set_run(host::COMPUTE_STATIC_THREAD_MGMT_SE0, &[u32::MAX; 2])
        .unwrap();
    stream
        .set_run(host::COMPUTE_STATIC_THREAD_MGMT_SE2, &[u32::MAX; 2])
        .unwrap();
    stream
        .set(host::COMPUTE_PGM_LO, (code.address() >> 8) as u32)
        .unwrap();
    stream
        .set(host::COMPUTE_PGM_HI, (code.address() >> 40) as u32)
        .unwrap();
    stream
        .set(host::COMPUTE_PGM_RSRC1, artifact.pgm_rsrc1)
        .unwrap();
    stream
        .set(host::COMPUTE_PGM_RSRC2, artifact.pgm_rsrc2)
        .unwrap();
    stream.set(host::COMPUTE_PGM_RSRC3, 0).unwrap();
    stream.set(host::COMPUTE_TMPRING_SIZE, 0).unwrap();
    stream.set(host::COMPUTE_RESOURCE_LIMITS, 0).unwrap();
    stream.set_run(host::COMPUTE_START, &[0, 0, 0]).unwrap();
    stream
        .set_run(
            host::COMPUTE_NUM_THREAD,
            &[u32::from(artifact.options.wave_size), 1, 1],
        )
        .unwrap();
    stream.set_run(host::COMPUTE_USER_DATA, &user).unwrap();
    stream
        .raw_packet(
            crate::amd::defs::pkt3::DISPATCH_DIRECT,
            SHADER_TYPE_COMPUTE,
            &[
                1,
                1,
                1,
                5 | if artifact.options.wave_size == 32 {
                    1 << 15
                } else {
                    0
                },
            ],
        )
        .unwrap();
    stream
        .event_write(event::CS_PARTIAL_FLUSH, event::INDEX_PARTIAL_FLUSH)
        .unwrap();
    stream
        .acquire_mem_global(gcr::GL2_WB | gcr::GL2_INV)
        .unwrap();
    let serial = device
        .submit_host(stream.words(), &[&*code, &*output])
        .unwrap();
    device.wait(serial).unwrap();
    output.read().unwrap()[..32]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}
