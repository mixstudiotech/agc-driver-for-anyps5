use crate::{Error, Result};
use std::mem::size_of;

pub struct AmdPrivateAbi;

// Packed(4) field offsets from references/mesa-wddm/src/amd/agc/wddm_private.h.
// Observed private flags retain neutral names: undocumented bits are not assigned invented semantics.

struct ContextRecord;
impl ContextRecord {
    const SIZE: usize = 424;
    const HEADER_SIZE: usize = 0;
    const FLAGS: usize = 4;
    const VERSION: usize = 52;
    const DW13: usize = 56;
    const SECTION_SIZE: usize = 64;
}

struct QueueRecord;
impl QueueRecord {
    const SIZE: usize = 64;
    const HEADER_SIZE: usize = 0;
    const FLAGS: usize = 16;
    /// The KMD writes its identifier for the new queue back here.
    const QUEUE_ID: usize = 20;
}

struct AllocationPrivateRecord;
impl AllocationPrivateRecord {
    const SIZE: usize = 64;
    const FLAGS: usize = 8;
    const CHECKSUM: usize = 12;
    const RECORDS_SIZE: usize = 60;
}

struct AllocationHeader;
impl AllocationHeader {
    const SIZE: usize = 64;
    /// `entry_size[4]`: the size of each entry, entries following the header back to back.
    const ENTRY_SIZE: usize = 0;
    const MAX_ENTRIES: usize = 4;
    const ADAPTER_ID: usize = 16;
    const CHECKSUM: usize = 20;
    const ENTRY_COUNT: usize = 60;
}

struct AllocationEntry;
impl AllocationEntry {
    const SIZE: usize = 196;
    const ENTRY_SIZE: usize = 0;
    const ENABLED_SECTIONS: usize = 4;
    const BO_INFO: usize = 8;
    const METADATA_OFFSET: usize = 176;
    const VERSION: usize = 188;
}

struct BufferObjectRecord;
impl BufferObjectRecord {
    const SIZE: usize = 128;
    const SECTION_SIZE: usize = 0;
    const FLAGS: usize = 8;
    const BYTE_SIZE: usize = 12;
    const ALIGNMENT: usize = 16;
    const PRIORITY: usize = 20;
    const HEAP_PRIMARY: usize = 24;
    const HEAP_SECONDARY: usize = 25;
    const FLAGS2: usize = 92;
    const PHYSICAL_SIZE: usize = 96;
    const VA_ADDRESS: usize = 104;
    const VA_SIZE: usize = 112;
}

struct AllocationMetadata;
impl AllocationMetadata {
    const SIZE: usize = 276;
    const SECTION_SIZE: usize = 0;
    const VA_ADDRESS: usize = 8;
    const VA_SIZE: usize = 16;
    const MTYPE: usize = 204;
    const MALL_POLICY: usize = 212;
}

struct SubmitHeader;
impl SubmitHeader {
    const SIZE: usize = 8;
    const VERSION: usize = 0;
    const ENTRY_COUNT: usize = 4;
}

struct GfxIbRecord;
impl GfxIbRecord {
    const SIZE: usize = 64;
    const TYPE: usize = 0;
    const SECTION_SIZE: usize = 4;
    const LENGTH: usize = 8;
    const FLAGS: usize = 12;
    const IP_TYPE: usize = 20;
    const ADDRESS_LOW: usize = 24;
    const ADDRESS_HIGH: usize = 28;
}

/// GPU cache policy of an allocation: the metadata section's mtype and MALL policy. lfrb sets
/// mtype 4 (L2 uncached) with MALL "never" only for GL2-bypass buffers and leaves both 0 otherwise.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CachePolicy {
    pub mtype: u32,
    pub mall_policy: u32,
}

impl CachePolicy {
    pub const GL2_CACHED: Self = Self {
        mtype: 0,
        mall_policy: 0,
    };
    pub const GL2_BYPASS: Self = Self {
        mtype: 4,
        mall_policy: 1,
    };
}

/// The buffer-object section values that differ between allocation kinds.
struct BoParameters {
    flags: u32,
    priority: u32,
    alignment: u32,
    heaps: [u8; 2],
    policy: CachePolicy,
    va_address: u64,
}

impl BoParameters {
    fn device(policy: CachePolicy) -> Self {
        Self {
            flags: AmdPrivateAbi::OBSERVED_BO_FLAGS,
            priority: AmdPrivateAbi::OBSERVED_BO_PRIORITY,
            alignment: AmdPrivateAbi::BUFFER_ALIGNMENT as u32,
            heaps: [
                AmdPrivateAbi::OBSERVED_PRIMARY_HEAP,
                AmdPrivateAbi::OBSERVED_SECONDARY_HEAP,
            ],
            policy,
            va_address: 0,
        }
    }

    fn host(va_address: u64) -> Self {
        Self {
            flags: AmdPrivateAbi::HOST_BO_FLAGS,
            priority: AmdPrivateAbi::HOST_BO_PRIORITY,
            alignment: AmdPrivateAbi::HOST_ALIGNMENT as u32,
            heaps: [AmdPrivateAbi::OBSERVED_PRIMARY_HEAP, 0],
            policy: CachePolicy::GL2_CACHED,
            va_address,
        }
    }
}

/// One IB of a submission. `flags` is 0 for a main IB, 0xC for a preamble and 0x104 for a
/// postamble; the meaning of the bits is undocumented.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IbEntry {
    pub address: u64,
    pub bytes: usize,
    pub flags: u32,
}

impl IbEntry {
    pub const MAIN: u32 = 0;
    pub const PREAMBLE: u32 = 0xc;
}

/// The AMD allocation records found in a shared allocation's private driver data: the fields
/// lfrb's `radv_wddm2_bo_from_handle` and amdvlk's NT-handle import read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedRecords {
    /// Where the record header starts in the allocation's private data.
    pub offset: usize,
    /// The entry's version: 9 from amdvlk and lfrb, 8 from the D3D12 driver.
    pub version: u32,
    pub adapter_id: u32,
    pub entries: u32,
    /// Whether the header checksum matches; the KMD may rewrite fields after creation, so a
    /// mismatch is reported, not refused.
    pub checksum_ok: bool,
    pub physical_size: u64,
}

impl AmdPrivateAbi {
    pub const PROFILE: &str = "amd-v9-1586-32.0.32015.7029";
    pub const VENDOR_ID: u32 = 0x1002;
    pub const DEVICE_ID: u32 = 0x1586;
    pub const REVISION_ID: u32 = 0xc1;
    pub const DRIVER_VERSION: u64 = (32u64 << 48) | (32015u64 << 16) | 7029;
    pub const BUFFER_ALIGNMENT: usize = 64 * 1024;
    const ABI_VERSION: u32 = 9;
    /// Entry versions of shared allocations whose layout is read: the D3D12 driver (amdxc64)
    /// writes version 8 with the same header, entry, buffer-object, SURF and metadata layout as
    /// version 9 (a committed buffer on mix, 2026-11-07: entry size 0x2f8, sections 0xd, header
    /// checksum valid).
    const SHARED_VERSIONS: [u32; 2] = [8, 9];
    const CONTEXT_HEADER_BYTES: u32 = 0x40;
    const CONTEXT_SECTION_BYTES: u32 = 0x168;
    const OBSERVED_CONTEXT_FLAGS: u32 = 0x140000;
    /// lfrb's compute context: the graphics flags without 0x40000, and no version field.
    const OBSERVED_COMPUTE_CONTEXT_FLAGS: u32 = 0x100000;
    const OBSERVED_CONTEXT_DW13: u32 = 0x200000;
    const OBSERVED_QUEUE_FLAGS: u32 = 0x21000;
    const OBSERVED_ALLOCATION_FLAGS: u32 = 0x80;
    const OBSERVED_BO_FLAGS: u32 = 0x2000200c;
    const OBSERVED_BO_FLAGS2: u32 = 0x1000000;
    const OBSERVED_BO_PRIORITY: u32 = 5;
    const OBSERVED_PRIMARY_HEAP: u8 = 4;
    const OBSERVED_SECONDARY_HEAP: u8 = 3;
    /// lfrb's `radv_wddm2_bo_from_ptr`: NOT_VIRTUAL | CPU_VISIBLE | HOST_ALLOCATED and the
    /// GART-cacheable heap only, priority 5 | 0x800, GART-page alignment.
    const HOST_BO_FLAGS: u32 = 0x2400_2008;
    const HOST_BO_PRIORITY: u32 = 0x805;
    pub const HOST_ALIGNMENT: usize = 4096;
    const SECTION_BO_INFO: u32 = 1;
    const GFX_IB_ENTRY_TYPE: u32 = 0;
    const GFX_IP_TYPE: u32 = 1;
    const SINGLE_ENTRY: u32 = 1;
    /// More IB entries than any layout the transport builds.
    pub const MAX_IB_ENTRIES: usize = 64;
    const ADDRESS_HIGH_SHIFT: u32 = u32::BITS;
    const RECORDS_BYTES: usize =
        AllocationHeader::SIZE + AllocationEntry::SIZE + AllocationMetadata::SIZE;
    const ENTRY_BYTES: usize = AllocationEntry::SIZE + AllocationMetadata::SIZE;

    fn put32(blob: &mut [u8], at: usize, value: u32) {
        blob[at..at + size_of::<u32>()].copy_from_slice(&value.to_le_bytes());
    }

    fn put64(blob: &mut [u8], at: usize, value: u64) {
        blob[at..at + size_of::<u64>()].copy_from_slice(&value.to_le_bytes());
    }

    fn checksum(blob: &[u8]) -> u32 {
        blob.chunks_exact(size_of::<u32>())
            .enumerate()
            .fold(0u32, |sum, (i, word)| {
                sum.wrapping_add(i as u32 ^ u32::from_le_bytes(word.try_into().unwrap()))
            })
    }

    pub fn context(&self) -> [u8; ContextRecord::SIZE] {
        let mut data = self.compute_context();
        Self::put32(
            &mut data,
            ContextRecord::FLAGS,
            Self::OBSERVED_CONTEXT_FLAGS,
        );
        Self::put32(&mut data, ContextRecord::VERSION, Self::ABI_VERSION);
        data
    }

    pub fn compute_context(&self) -> [u8; ContextRecord::SIZE] {
        let mut data = [0; ContextRecord::SIZE];
        Self::put32(
            &mut data,
            ContextRecord::HEADER_SIZE,
            Self::CONTEXT_HEADER_BYTES,
        );
        Self::put32(
            &mut data,
            ContextRecord::FLAGS,
            Self::OBSERVED_COMPUTE_CONTEXT_FLAGS,
        );
        Self::put32(&mut data, ContextRecord::DW13, Self::OBSERVED_CONTEXT_DW13);
        Self::put32(
            &mut data,
            ContextRecord::SECTION_SIZE,
            Self::CONTEXT_SECTION_BYTES,
        );
        data
    }

    pub fn queue(&self) -> [u8; QueueRecord::SIZE] {
        let mut data = [0; QueueRecord::SIZE];
        Self::put32(
            &mut data,
            QueueRecord::HEADER_SIZE,
            QueueRecord::SIZE as u32,
        );
        Self::put32(&mut data, QueueRecord::FLAGS, Self::OBSERVED_QUEUE_FLAGS);
        data
    }

    /// The AMD records in `data`, one allocation's private driver data as
    /// OpenResourceFromNtHandle returns it for a shared resource: a header with 1 to 4 entries
    /// back to back, the first a v8 or v9 entry with a buffer-object section. The header is
    /// looked for at every 4-byte offset, so data the creating runtime put in front of it is
    /// skipped; it is taken where the header's entry size matches the entry's own. The error says
    /// what the data holds where the header was expected.
    pub fn shared_records(data: &[u8]) -> std::result::Result<SharedRecords, String> {
        (0..data.len())
            .step_by(size_of::<u32>())
            .find_map(|offset| Self::records_at(data, offset))
            .ok_or_else(|| {
                let at = AllocationHeader::SIZE + AllocationEntry::VERSION;
                format!(
                    "no v8 or v9 AMD allocation record in 0x{:x} bytes of allocation private data (entry count {}, entry size {} and version {} where a record at +0 keeps them; first dwords {})",
                    data.len(),
                    dword_at(data, AllocationHeader::ENTRY_COUNT),
                    dword_at(data, AllocationHeader::ENTRY_SIZE),
                    dword_at(data, at),
                    leading_dwords(data, 16)
                )
            })
    }

    fn records_at(data: &[u8], offset: usize) -> Option<SharedRecords> {
        let header = data.get(offset..offset.checked_add(AllocationHeader::SIZE)?)?;
        let read32 =
            |blob: &[u8], at: usize| u32::from_le_bytes(blob[at..at + 4].try_into().unwrap());
        let entries = read32(header, AllocationHeader::ENTRY_COUNT);
        if entries == 0 || entries as usize > AllocationHeader::MAX_ENTRIES {
            return None;
        }
        let entry_size = read32(header, AllocationHeader::ENTRY_SIZE) as usize;
        let start = offset + AllocationHeader::SIZE;
        let entry = data.get(start..start.checked_add(entry_size)?)?;
        let bo = AllocationEntry::BO_INFO;
        if entry_size < AllocationEntry::SIZE
            || read32(entry, AllocationEntry::ENTRY_SIZE) as usize != entry_size
            || !Self::SHARED_VERSIONS.contains(&read32(entry, AllocationEntry::VERSION))
            || read32(entry, bo + BufferObjectRecord::SECTION_SIZE)
                != BufferObjectRecord::SIZE as u32
        {
            return None;
        }
        let at = bo + BufferObjectRecord::PHYSICAL_SIZE;
        let mut zeroed = header.to_vec();
        zeroed[AllocationHeader::CHECKSUM..AllocationHeader::CHECKSUM + 4].fill(0);
        Some(SharedRecords {
            offset,
            version: read32(entry, AllocationEntry::VERSION),
            adapter_id: read32(header, AllocationHeader::ADAPTER_ID),
            entries,
            checksum_ok: Self::checksum(&zeroed) == read32(header, AllocationHeader::CHECKSUM),
            physical_size: u64::from_le_bytes(entry[at..at + 8].try_into().unwrap()),
        })
    }

    /// The queue identifier the KMD wrote into a queue record during CreateHwQueue.
    pub fn queue_id(record: &[u8; QueueRecord::SIZE]) -> u32 {
        u32::from_le_bytes(
            record[QueueRecord::QUEUE_ID..QueueRecord::QUEUE_ID + size_of::<u32>()]
                .try_into()
                .unwrap(),
        )
    }

    /// Allocation records for a GL2-bypass buffer, the policy every buffer used before per-kind
    /// cache policies.
    pub fn allocation(
        &self,
        size: u32,
    ) -> (
        [u8; AllocationPrivateRecord::SIZE],
        [u8; Self::RECORDS_BYTES],
    ) {
        self.allocation_with(size, CachePolicy::GL2_BYPASS)
    }

    pub fn allocation_with(
        &self,
        size: u32,
        policy: CachePolicy,
    ) -> (
        [u8; AllocationPrivateRecord::SIZE],
        [u8; Self::RECORDS_BYTES],
    ) {
        self.records(size, &BoParameters::device(policy))
    }

    /// Records for a device allocation that will be mapped at the fixed GPU VA `address`.
    pub fn allocation_at(
        &self,
        size: u32,
        policy: CachePolicy,
        address: u64,
    ) -> (
        [u8; AllocationPrivateRecord::SIZE],
        [u8; Self::RECORDS_BYTES],
    ) {
        self.records(
            size,
            &BoParameters {
                va_address: address,
                ..BoParameters::device(policy)
            },
        )
    }

    /// Records for host memory imported as a pinned allocation (`pSystemMem`) and mapped at the
    /// fixed GPU VA `address`.
    pub fn host_allocation(
        &self,
        size: u32,
        address: u64,
    ) -> (
        [u8; AllocationPrivateRecord::SIZE],
        [u8; Self::RECORDS_BYTES],
    ) {
        self.records(size, &BoParameters::host(address))
    }

    pub fn host_allocation_with_policy(&self, size: u32, address: u64, policy: CachePolicy) -> ([u8; AllocationPrivateRecord::SIZE], [u8; Self::RECORDS_BYTES]) {
        self.records(size, &BoParameters { policy, ..BoParameters::host(address) })
    }

    fn records(
        &self,
        size: u32,
        parameters: &BoParameters,
    ) -> (
        [u8; AllocationPrivateRecord::SIZE],
        [u8; Self::RECORDS_BYTES],
    ) {
        let policy = parameters.policy;
        let mut records = [0; Self::RECORDS_BYTES];
        Self::put32(
            &mut records,
            AllocationHeader::ENTRY_SIZE,
            Self::ENTRY_BYTES as u32,
        );
        Self::put32(
            &mut records,
            AllocationHeader::ENTRY_COUNT,
            Self::SINGLE_ENTRY,
        );
        let sum = Self::checksum(&records[..AllocationHeader::SIZE]);
        Self::put32(&mut records, AllocationHeader::CHECKSUM, sum);
        let entry = AllocationHeader::SIZE;
        Self::put32(
            &mut records,
            entry + AllocationEntry::ENTRY_SIZE,
            Self::ENTRY_BYTES as u32,
        );
        Self::put32(
            &mut records,
            entry + AllocationEntry::ENABLED_SECTIONS,
            Self::SECTION_BO_INFO,
        );
        let bo = entry + AllocationEntry::BO_INFO;
        Self::put32(
            &mut records,
            bo + BufferObjectRecord::SECTION_SIZE,
            BufferObjectRecord::SIZE as u32,
        );
        Self::put32(
            &mut records,
            bo + BufferObjectRecord::FLAGS,
            parameters.flags,
        );
        Self::put32(&mut records, bo + BufferObjectRecord::BYTE_SIZE, size);
        Self::put32(
            &mut records,
            bo + BufferObjectRecord::ALIGNMENT,
            parameters.alignment,
        );
        Self::put32(
            &mut records,
            bo + BufferObjectRecord::PRIORITY,
            parameters.priority,
        );
        records[bo + BufferObjectRecord::HEAP_PRIMARY] = parameters.heaps[0];
        records[bo + BufferObjectRecord::HEAP_SECONDARY] = parameters.heaps[1];
        Self::put64(
            &mut records,
            bo + BufferObjectRecord::VA_ADDRESS,
            parameters.va_address,
        );
        Self::put32(
            &mut records,
            bo + BufferObjectRecord::FLAGS2,
            Self::OBSERVED_BO_FLAGS2,
        );
        Self::put64(
            &mut records,
            bo + BufferObjectRecord::PHYSICAL_SIZE,
            u64::from(size),
        );
        Self::put64(
            &mut records,
            bo + BufferObjectRecord::VA_SIZE,
            u64::from(size),
        );
        Self::put32(
            &mut records,
            entry + AllocationEntry::METADATA_OFFSET,
            AllocationEntry::SIZE as u32,
        );
        Self::put32(
            &mut records,
            entry + AllocationEntry::VERSION,
            Self::ABI_VERSION,
        );
        let metadata = entry + AllocationEntry::SIZE;
        Self::put32(
            &mut records,
            metadata + AllocationMetadata::SECTION_SIZE,
            AllocationMetadata::SIZE as u32,
        );
        Self::put64(
            &mut records,
            metadata + AllocationMetadata::VA_ADDRESS,
            parameters.va_address,
        );
        Self::put64(
            &mut records,
            metadata + AllocationMetadata::VA_SIZE,
            u64::from(size),
        );
        Self::put32(
            &mut records,
            metadata + AllocationMetadata::MTYPE,
            policy.mtype,
        );
        Self::put32(
            &mut records,
            metadata + AllocationMetadata::MALL_POLICY,
            policy.mall_policy,
        );
        let mut private = [0; AllocationPrivateRecord::SIZE];
        Self::put32(
            &mut private,
            AllocationPrivateRecord::FLAGS,
            Self::OBSERVED_ALLOCATION_FLAGS,
        );
        Self::put32(
            &mut private,
            AllocationPrivateRecord::RECORDS_SIZE,
            Self::RECORDS_BYTES as u32,
        );
        let sum = Self::checksum(&private);
        Self::put32(&mut private, AllocationPrivateRecord::CHECKSUM, sum);
        (private, records)
    }

    /// Submit private data for one main IB.
    pub fn submission(&self, address: u64, bytes: usize) -> Result<Vec<u8>> {
        self.submission_entries(&[IbEntry {
            address,
            bytes,
            flags: IbEntry::MAIN,
        }])
    }

    /// Submit private data listing `entries` in execution order.
    pub fn submission_entries(&self, entries: &[IbEntry]) -> Result<Vec<u8>> {
        if entries.is_empty() || entries.len() > Self::MAX_IB_ENTRIES {
            return Err(Error::invalid("a submission lists 1 to 64 IBs"));
        }
        let mut data = vec![0; SubmitHeader::SIZE + entries.len() * GfxIbRecord::SIZE];
        Self::put32(&mut data, SubmitHeader::VERSION, Self::ABI_VERSION);
        Self::put32(&mut data, SubmitHeader::ENTRY_COUNT, entries.len() as u32);
        for (i, entry) in entries.iter().enumerate() {
            let bytes =
                u32::try_from(entry.bytes).map_err(|_| Error::invalid("IB byte size overflow"))?;
            let ib = SubmitHeader::SIZE + i * GfxIbRecord::SIZE;
            Self::put32(&mut data, ib + GfxIbRecord::TYPE, Self::GFX_IB_ENTRY_TYPE);
            Self::put32(
                &mut data,
                ib + GfxIbRecord::SECTION_SIZE,
                GfxIbRecord::SIZE as u32,
            );
            Self::put32(&mut data, ib + GfxIbRecord::LENGTH, bytes);
            Self::put32(&mut data, ib + GfxIbRecord::FLAGS, entry.flags);
            Self::put32(&mut data, ib + GfxIbRecord::IP_TYPE, Self::GFX_IP_TYPE);
            Self::put32(
                &mut data,
                ib + GfxIbRecord::ADDRESS_LOW,
                entry.address as u32,
            );
            Self::put32(
                &mut data,
                ib + GfxIbRecord::ADDRESS_HIGH,
                (entry.address >> Self::ADDRESS_HIGH_SHIFT) as u32,
            );
        }
        Ok(data)
    }
}

/// The dword at `at` in `data`, or "absent" past its end.
pub(super) fn dword_at(data: &[u8], at: usize) -> String {
    data.get(at..at + size_of::<u32>()).map_or_else(
        || "absent".to_string(),
        |word| format!("0x{:08x}", u32::from_le_bytes(word.try_into().unwrap())),
    )
}

/// Up to `count` leading dwords of `data`, in hex.
pub(super) fn leading_dwords(data: &[u8], count: usize) -> String {
    let words: Vec<String> = data
        .chunks_exact(size_of::<u32>())
        .take(count)
        .map(|word| format!("{:08x}", u32::from_le_bytes(word.try_into().unwrap())))
        .collect();
    format!("[{}]", words.join(" "))
}
