use shader::ir::{DescriptorSource, ResourceKind};
use shader::{CompileOptions, Compiler, ShaderArtifact, ShaderInterface, ShaderStage, TargetIsa};

#[cfg(all(windows, feature = "wddm"))]
mod gpu;

/// Reads an image whose T# it loads from the table in s[0:1] at the offset in s2: where the T#
/// lies is not known before the program runs.
const FOUND: [u32; 9] = [0xf40c_0100, 0x0400_0000, 0xbf8c_c07f, 0x7e02_0280, 0x7e04_0280, 0xf000_0108, 0x0001_0301, 0xbf8c_3f70, 0xbf81_0000];
/// The same with the T# at a fixed place in the table.
const NAMED: [u32; 9] = [0xf40c_0100, 0xfa00_0020, 0xbf8c_c07f, 0x7e02_0280, 0x7e04_0280, 0xf000_0108, 0x0001_0301, 0xbf8c_3f70, 0xbf81_0000];

/// Three user SGPRs, descriptors converted through the table at `lut`, reports to `reports`.
fn compile(words: &[u32], target: TargetIsa, wave: u8, lut: u64, reports: Option<u64>) -> ShaderArtifact {
    let interface = ShaderInterface { guest_pgm_rsrc2: Some(3 << 1), descriptor_lut: Some(lut), image_reports: reports, ..ShaderInterface::default() };
    Compiler::new(target).compile_with_interface(words, CompileOptions::new(target, ShaderStage::Compute, wave), &interface).unwrap()
}

#[test]
fn only_an_image_found_at_run_time_is_reported() {
    let (lut, reports) = (0x200_0000_0000, 0x200_0010_0000);
    let compile = |words: &[u32], reports| compile(words, TargetIsa::Gfx115, 64, lut, reports);
    let source = |artifact: &ShaderArtifact| artifact.descriptor_uses.iter().find(|u| u.kind == ResourceKind::ImageDescriptor).unwrap().source.clone();
    let (found, named) = (compile(&FOUND, Some(reports)), compile(&NAMED, Some(reports)));
    assert_eq!(source(&found), DescriptorSource::Dynamic);
    assert!(matches!(source(&named), DescriptorSource::Table(_)));
    // The report is code ahead of the image instruction. The runtime reads the place of a named
    // image itself, so a program whose images are all named is the same with and without.
    assert!(found.code_bytes().len() > compile(&FOUND, None).code_bytes().len());
    assert_eq!(named.code_bytes(), compile(&NAMED, None).code_bytes());
}
