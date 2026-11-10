//! The element-copy kernel: one surface level copied between two layouts, which retiles a guest
//! surface into its host-layout copy and detiles it back. The gfx1013 words
//! (`element_copy_words.rs`) are assembled from `element_copy.s` and translated for the host like
//! guest shaders; the parameter block comes from
//! [`crate::amd::surface::retile::copy_dispatches`].

pub use super::element_copy_words::WORDS;

use crate::Result;

/// COMPUTE_PGM_RSRC1: 20 VGPRs (granules of 4), 88 SGPRs, FLOAT_MODE 0xc0.
pub const PGM_RSRC1: u32 = 0x000c_0000 | (10 << 6) | 4;
/// COMPUTE_PGM_RSRC2: two user SGPRs (the parameter block address), the x, y and z group ids, and
/// x thread ids.
pub const PGM_RSRC2: u32 = (2 << 1) | (1 << 7) | (1 << 8) | (1 << 9);

/// COMPUTE_PGM_RSRC2 bits the kernel depends on: two user SGPRs and the x, y and z group ids.
pub const REQUIRED_RSRC2: u32 = (0x1f << 1) | (7 << 7);

/// The kernel translated for `target` (wave64). The translation must know [`PGM_RSRC2`]: the
/// group ids the kernel reads from s2-s4 are only loaded when its RSRC2 enables them.
pub fn compile(target: shader::TargetIsa) -> Result<shader::ShaderArtifact> {
    let interface = shader::ShaderInterface {
        guest_pgm_rsrc1: Some(PGM_RSRC1),
        guest_pgm_rsrc2: Some(PGM_RSRC2),
        ..shader::ShaderInterface::default()
    };
    shader::Compiler::new(target)
        .compile_with_interface(
            &WORDS,
            shader::CompileOptions::new(target, shader::ShaderStage::Compute, 64),
            &interface,
        )
        .map_err(|e| crate::Error::invalid(format!("element-copy kernel: {e}")))
}
