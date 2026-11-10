use std::collections::BTreeMap;

use crate::{
    CompileError, CompileOptions, Compiler, Linkage, PrecisionMode, ShaderArtifact,
    ShaderInterface, ShaderStage, TargetIsa, COMPILER_VERSION,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const ABSENT: u64 = u64::MAX;

#[derive(Clone, Copy, Debug)]
pub struct StableHasher(u64);

impl Default for StableHasher {
    fn default() -> Self {
        Self(FNV_OFFSET)
    }
}

impl StableHasher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
    }

    pub fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub fn finish(self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

pub fn hash_words(words: &[u32]) -> u64 {
    let mut h = StableHasher::new();
    h.write_u64(words.len() as u64);
    for &w in words {
        h.write_u32(w);
    }
    h.finish()
}

fn target_id(target: TargetIsa) -> u32 {
    match target {
        TargetIsa::Gfx1013 => 0x000a_0103,
        TargetIsa::Gfx103 => 0x000a_0300,
        TargetIsa::Gfx11 => 0x000b_0000,
        TargetIsa::Gfx115 => 0x000b_0500,
        TargetIsa::Gfx12 => 0x000c_0000,
    }
}

fn stage_id(stage: ShaderStage) -> u32 {
    match stage {
        ShaderStage::Compute => 0,
        ShaderStage::Vertex => 1,
        ShaderStage::Fragment => 2,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheKey {
    pub hash: u64,
    pub words: Vec<u32>,
    pub options: CompileOptions,
    pub precision: PrecisionMode,
    pub interface: ShaderInterface,
    pub compiler_version: u32,
    /// Linked code ([`Linkage`]): each region's address and words.
    pub regions: Vec<(u64, Vec<u32>)>,
    pub second_half: Option<u64>,
}

impl CacheKey {
    pub fn new(words: &[u32], options: CompileOptions) -> Self {
        Self::with(
            words,
            options,
            PrecisionMode::default(),
            ShaderInterface::default(),
        )
    }

    pub fn with(
        words: &[u32],
        options: CompileOptions,
        precision: PrecisionMode,
        interface: ShaderInterface,
    ) -> Self {
        let optional = |v: Option<u32>| v.map_or(ABSENT, u64::from);
        let mut h = StableHasher::new();
        h.write_u32(COMPILER_VERSION);
        h.write_u32(target_id(options.target));
        h.write_u32(stage_id(options.stage));
        h.write_u32(u32::from(options.wave_size));
        h.write_u32(u32::from(precision == PrecisionMode::Exact));
        h.write_u64(optional(interface.user_sgpr_count.map(u32::from)));
        h.write_u64(optional(interface.guest_pgm_rsrc1));
        h.write_u64(optional(interface.guest_pgm_rsrc2));
        h.write_u32(interface.lds_bytes);
        h.write_u32(interface.scratch_bytes_per_lane);
        h.write_u64(interface.guest_code_address.unwrap_or(ABSENT));
        h.write_u64(optional(interface.ps_input_ena));
        h.write_u64(optional(interface.ps_input_addr));
        h.write_u64(optional(interface.ps_col_format));
        h.write_u64(optional(interface.ps_z_format));
        h.write_u32(u32::from(interface.dual_source_blend));
        h.write_u64(optional(interface.param_exports.map(u32::from)));
        h.write_u64(interface.descriptor_lut.unwrap_or(ABSENT));
        h.write_u32(u32::from(interface.preconvert_user_descriptors));
        h.write_u64(interface.attribute_ring.map_or(ABSENT, |r| r.address));
        h.write_u64(optional(interface.attribute_ring.map(|r| r.bytes)));
        h.write_u64(interface.gds_buffer.unwrap_or(ABSENT));
        h.write_u64(interface.mip_stats_table.unwrap_or(ABSENT));
        h.write_u32(u32::from(
            interface.vertex_stage == crate::VertexStage::LsHs,
        ));
        h.write_u64(hash_words(words));
        Self {
            hash: h.finish(),
            words: words.to_vec(),
            options,
            precision,
            interface,
            compiler_version: COMPILER_VERSION,
            regions: Vec::new(),
            second_half: None,
        }
    }

    /// The key of a [`Compiler::compile_linked`] translation: [`Self::with`] plus every linked
    /// region (address and words) and the second-half address.
    pub fn with_linkage(
        words: &[u32],
        options: CompileOptions,
        precision: PrecisionMode,
        interface: ShaderInterface,
        linkage: &Linkage<'_>,
    ) -> Self {
        let mut key = Self::with(words, options, precision, interface);
        if linkage.regions.is_empty() && linkage.second_half.is_none() {
            return key;
        }
        let mut h = StableHasher::new();
        h.write_u64(key.hash);
        for region in linkage.regions {
            h.write_u64(region.address);
            h.write_u64(hash_words(region.words));
        }
        h.write_u64(linkage.second_half.unwrap_or(ABSENT));
        key.hash = h.finish();
        key.regions = linkage
            .regions
            .iter()
            .map(|r| (r.address, r.words.to_vec()))
            .collect();
        key.second_half = linkage.second_half;
        key
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Default)]
pub struct ArtifactCache {
    entries: BTreeMap<u64, Vec<(CacheKey, ShaderArtifact)>>,
    hits: u64,
    misses: u64,
}

impl ArtifactCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }

    pub fn get(&self, key: &CacheKey) -> Option<&ShaderArtifact> {
        self.entries
            .get(&key.hash)?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, artifact)| artifact)
    }

    pub fn insert(&mut self, key: CacheKey, artifact: ShaderArtifact) {
        let bucket = self.entries.entry(key.hash).or_default();
        match bucket.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = artifact,
            None => bucket.push((key, artifact)),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn get_or_compile(
        &mut self,
        compiler: &Compiler,
        words: &[u32],
        options: CompileOptions,
    ) -> Result<&ShaderArtifact, CompileError> {
        self.get_or_compile_with_interface(compiler, words, options, &ShaderInterface::default())
    }

    pub fn get_or_compile_with_interface(
        &mut self,
        compiler: &Compiler,
        words: &[u32],
        options: CompileOptions,
        interface: &ShaderInterface,
    ) -> Result<&ShaderArtifact, CompileError> {
        self.get_or_compile_linked(compiler, words, options, interface, &Linkage::default())
    }

    /// [`Self::get_or_compile_with_interface`] for [`Compiler::compile_linked`].
    pub fn get_or_compile_linked(
        &mut self,
        compiler: &Compiler,
        words: &[u32],
        options: CompileOptions,
        interface: &ShaderInterface,
        linkage: &Linkage<'_>,
    ) -> Result<&ShaderArtifact, CompileError> {
        let key = CacheKey::with_linkage(words, options, compiler.precision(), *interface, linkage);
        if self.get(&key).is_some() {
            self.hits += 1;
        } else {
            let artifact = compiler.compile_linked(words, options, interface, linkage)?;
            self.misses += 1;
            self.insert(key.clone(), artifact);
        }
        self.get(&key).ok_or(CompileError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROGRAM: [u32; 2] = [0x7e02_02f2, 0xbf81_0000];

    fn options(target: TargetIsa, wave_size: u8) -> CompileOptions {
        CompileOptions::new(target, ShaderStage::Compute, wave_size)
    }

    #[test]
    fn hashing_is_stable_and_sensitive() {
        let a = CacheKey::new(&PROGRAM, options(TargetIsa::Gfx11, 32));
        let b = CacheKey::new(&PROGRAM, options(TargetIsa::Gfx11, 32));
        assert_eq!(a, b);
        assert_eq!(a.hash, b.hash);
        assert_ne!(
            a.hash,
            CacheKey::new(&PROGRAM, options(TargetIsa::Gfx11, 64)).hash
        );
        assert_ne!(
            a.hash,
            CacheKey::new(&PROGRAM, options(TargetIsa::Gfx12, 32)).hash
        );
        assert_ne!(
            a.hash,
            CacheKey::new(&PROGRAM[..1], options(TargetIsa::Gfx11, 32)).hash
        );
        let fused = CacheKey::with(
            &PROGRAM,
            options(TargetIsa::Gfx11, 32),
            PrecisionMode::Fused,
            ShaderInterface::default(),
        );
        assert_ne!(a.hash, fused.hash);
        assert_eq!(hash_words(&[]), hash_words(&[]));
        assert_ne!(hash_words(&[0]), hash_words(&[]));
        let ring = |address, bytes| {
            CacheKey::with(
                &PROGRAM,
                options(TargetIsa::Gfx11, 32),
                PrecisionMode::default(),
                ShaderInterface {
                    attribute_ring: Some(crate::AttributeRingAbi { address, bytes }),
                    ..ShaderInterface::default()
                },
            )
            .hash
        };
        assert_ne!(a.hash, ring(0x1_0000_0000, 0x2c_0000));
        assert_ne!(
            ring(0x1_0000_0000, 0x2c_0000),
            ring(0x2_0000_0000, 0x2c_0000)
        );
        assert_ne!(
            ring(0x1_0000_0000, 0x2c_0000),
            ring(0x1_0000_0000, 0x10_0000)
        );
    }

    #[test]
    fn cache_reuses_compiled_artifacts() {
        let compiler = Compiler::new(TargetIsa::Gfx11);
        let mut cache = ArtifactCache::new();
        let first = cache
            .get_or_compile(&compiler, &PROGRAM, options(TargetIsa::Gfx11, 32))
            .expect("compile")
            .clone();
        let second = cache
            .get_or_compile(&compiler, &PROGRAM, options(TargetIsa::Gfx11, 32))
            .expect("cached")
            .clone();
        assert_eq!(first, second);
        assert_eq!(
            cache.stats(),
            CacheStats {
                entries: 1,
                hits: 1,
                misses: 1
            }
        );
        cache
            .get_or_compile(&compiler, &PROGRAM, options(TargetIsa::Gfx11, 64))
            .expect("wave64");
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn failed_compiles_are_not_cached() {
        let compiler = Compiler::new(TargetIsa::Gfx11);
        let mut cache = ArtifactCache::new();
        let err = cache
            .get_or_compile(&compiler, &PROGRAM, options(TargetIsa::Gfx12, 32))
            .unwrap_err();
        assert!(matches!(err, CompileError::TargetMismatch { .. }));
        assert!(cache.is_empty());
    }
}
