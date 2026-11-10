//! Guest (gfx10) cache controls of ACQUIRE_MEM, RELEASE_MEM and EVENT_WRITE in host terms.
//!
//! Field layouts are Mesa's (`pkt3.json`, `gfx10.json`), the sequences radv's
//! (`ac_gfx10_emit_barrier`, `ac_emit_cp_release_mem_pws`). Every plan performs at least the cache
//! actions the guest asked for: ranges widen to the whole address space (tiled and depth surfaces
//! live at guest VA + 2^40 on the host, so a guest range can miss their copies) and GL2_DISCARD
//! becomes write-back and invalidate (the host L2 is shared with other processes).

use crate::amd::defs::gfx10::CP_COHER_CNTL as Coher;
use crate::amd::defs::pkt3::dwords::{GCR_CNTL as Gcr, RELEASE_MEM_OP_gfx11 as ReleaseOp};
use crate::gpu::packets::event;

/// CB0_DEST_BASE_ENA..CB7_DEST_BASE_ENA and DB_DEST_BASE_ENA: the destination-base enables of
/// the guest's CP_ME_COHER_CNTL, which share the word with CP_COHER_CNTL's actions.
const CB_DEST_BASE_ENA: u32 = 0xff << 6;
const DB_DEST_BASE_ENA: u32 = 1 << 14;

/// The guest's ZPASS_DONE event, which gfx11.5's enum no longer names.
const ZPASS_DONE: u32 = 0x15;

const SEQ_MASK: u32 = Gcr::SEQ.mask();
/// GLI_INV = ALL.
const GLI_INV_ALL: u32 = Gcr::GLI_INV.put(1);

/// What a guest ACQUIRE_MEM needs on the host.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AcquirePlan {
    /// Flush and invalidate the colour caches (CB_ACTION_ENA or a CB destination base).
    pub flush_cb: bool,
    /// Flush and invalidate the depth caches (DB_ACTION_ENA or the DB destination base).
    pub flush_db: bool,
    /// GCR_CNTL over the whole address space, run after any CB/DB flush.
    pub gcr_cntl: u32,
}

impl AcquirePlan {
    /// The plan of a guest acquire with COHER_CNTL `coher` and GCR_CNTL `gcr`.
    pub fn of(coher: u32, gcr: u32) -> Self {
        let mut g = gcr
            & (Gcr::GLM_WB.mask()
                | Gcr::GLM_INV.mask()
                | Gcr::GLK_WB.mask()
                | Gcr::GLK_INV.mask()
                | Gcr::GLV_INV.mask()
                | Gcr::GL1_INV.mask()
                | Gcr::GL2_INV.mask()
                | Gcr::GL2_WB.mask());
        if gcr & Gcr::GLI_INV.mask() != 0 {
            g |= GLI_INV_ALL;
        }
        if gcr & Gcr::GL2_DISCARD.mask() != 0 {
            g |= Gcr::GL2_WB.mask() | Gcr::GL2_INV.mask();
        }
        g |= Self::legacy_coher_actions(coher) | (gcr & SEQ_MASK);
        if g & Gcr::GLM_WB.mask() != 0 {
            // radv: "GLM doesn't support WB alone. If WB is set, INV must be set too."
            g |= Gcr::GLM_INV.mask();
        }
        Self {
            flush_cb: coher & (Coher::CB_ACTION_ENA.mask() | CB_DEST_BASE_ENA) != 0,
            flush_db: coher & (Coher::DB_ACTION_ENA.mask() | DB_DEST_BASE_ENA) != 0,
            gcr_cntl: g,
        }
    }

    /// gfx10 COHER_CNTL cache actions as GCR_CNTL, as radv encodes the same requests in
    /// `si_cs_emit_cache_flush` and `gfx10_cs_emit_cache_flush`: TC_NC/TC_WC only narrow TC
    /// actions to some lines, so the actions here cover every line.
    fn legacy_coher_actions(coher: u32) -> u32 {
        let on = |field: crate::amd::defs::Field| coher & field.mask() != 0;
        let mut g = 0;
        if on(Coher::SH_ICACHE_ACTION_ENA) {
            g |= GLI_INV_ALL;
        }
        if on(Coher::SH_KCACHE_ACTION_ENA) || on(Coher::SH_KCACHE_VOL_ACTION_ENA) {
            g |= Gcr::GLK_INV.mask() | Gcr::GL1_INV.mask();
        }
        if on(Coher::SH_KCACHE_WB_ACTION_ENA) {
            g |= Gcr::GLK_WB.mask();
        }
        if on(Coher::TCL1_ACTION_ENA) || on(Coher::TCL1_VOL_ACTION_ENA) {
            g |= Gcr::GLV_INV.mask() | Gcr::GL1_INV.mask();
        }
        if on(Coher::TC_ACTION_ENA) {
            g |= Gcr::GL2_INV.mask() | Gcr::GL2_WB.mask() | Gcr::GLM_INV.mask() | Gcr::GLM_WB.mask();
        }
        if on(Coher::TC_WB_ACTION_ENA) {
            g |= Gcr::GL2_WB.mask() | Gcr::GLM_INV.mask() | Gcr::GLM_WB.mask();
        }
        if on(Coher::TC_INV_METADATA_ACTION_ENA) {
            g |= Gcr::GLM_INV.mask() | Gcr::GLM_WB.mask();
        }
        g
    }
}

/// How a guest EVENT_WRITE runs on the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventPlan {
    /// EVENT_WRITE with this type and index.
    Forward { event: u32, index: u32 },
    /// A CB/DB cache flush gfx11 performs only as a timestamp event: release `event` in PWS
    /// form, then wait for it.
    TimestampFlush { event: u32 },
    Unsupported(&'static str),
}

impl EventPlan {
    pub fn of(event_type: u32) -> Self {
        use crate::amd::defs::gfx115::enums::VGT_EVENT_TYPE as E;
        match event_type {
            E::CS_PARTIAL_FLUSH | E::PS_PARTIAL_FLUSH | E::VS_PARTIAL_FLUSH => {
                Self::Forward { event: event_type, index: event::INDEX_PARTIAL_FLUSH }
            }
            E::PIPELINESTAT_START | E::PIPELINESTAT_STOP | E::VGT_FLUSH | E::FLUSH_AND_INV_CB_META => {
                Self::Forward { event: event_type, index: 0 }
            }
            // radv: "GFX11 can't flush DB_META and should use a TS event instead."
            E::FLUSH_AND_INV_DB_META | E::CACHE_FLUSH_AND_INV_EVENT | E::DB_CACHE_FLUSH_AND_INV => {
                Self::TimestampFlush { event: E::CACHE_FLUSH_AND_INV_TS_EVENT }
            }
            E::CACHE_FLUSH => Self::TimestampFlush { event: E::CACHE_FLUSH_TS },
            E::FLUSH_AND_INV_CB_PIXEL_DATA => Self::TimestampFlush { event: E::FLUSH_AND_INV_CB_DATA_TS },
            ZPASS_DONE | E::PIXEL_PIPE_STAT_DUMP => {
                Self::Unsupported("occlusion results are 16 B per host RB and need conversion into the guest layout")
            }
            _ => Self::Unsupported("this EVENT_WRITE type has no host translation"),
        }
    }
}

/// A guest RELEASE_MEM's host event, EVENT_INDEX and cache actions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleasePlan {
    pub event: u32,
    pub index: u32,
    /// RELEASE_MEM_OP cache action bits.
    pub actions: u32,
}

impl ReleasePlan {
    /// The plan of a guest release of `event_type` with RELEASE_MEM_OP bits `guest`.
    pub fn of(event_type: u32, guest: u32) -> Result<Self, &'static str> {
        use crate::amd::defs::gfx115::enums::VGT_EVENT_TYPE as E;
        let actions = Self::actions(guest);
        match event_type {
            // radv: a DB-only flush uses CACHE_FLUSH_AND_INV_TS_EVENT on GFX11.
            E::FLUSH_AND_INV_DB_DATA_TS => {
                Ok(Self { event: E::CACHE_FLUSH_AND_INV_TS_EVENT, index: event::INDEX_END_OF_PIPE, actions })
            }
            E::CACHE_FLUSH_TS
            | E::CACHE_FLUSH_AND_INV_TS_EVENT
            | E::FLUSH_AND_INV_CB_DATA_TS
            | E::BOTTOM_OF_PIPE_TS
            | E::FLUSH_SX_TS => Ok(Self { event: event_type, index: event::INDEX_END_OF_PIPE, actions }),
            E::CS_DONE | E::PS_DONE if actions & !ReleaseOp::SEQ.mask() == 0 => {
                Ok(Self { event: event_type, index: event::INDEX_SHADER_DONE, actions: 0 })
            }
            // radv: "Only GFX12+ supports GCR ops with PS_DONE & CS_DONE in RELEASE_MEM."
            E::CS_DONE | E::PS_DONE => {
                Ok(Self { event: E::BOTTOM_OF_PIPE_TS, index: event::INDEX_END_OF_PIPE, actions })
            }
            _ => Err("RELEASE_MEM event has no host end-of-pipe equivalent"),
        }
    }

    /// gfx10 RELEASE_MEM_OP cache actions as gfx11 RELEASE_MEM_OP.
    fn actions(guest: u32) -> u32 {
        const KEPT: u32 = ReleaseOp::GLM_WB.mask()
            | ReleaseOp::GLM_INV.mask()
            | ReleaseOp::GLV_INV.mask()
            | ReleaseOp::GL1_INV.mask()
            | ReleaseOp::GL2_INV.mask()
            | ReleaseOp::GL2_WB.mask()
            | ReleaseOp::SEQ.mask();
        let mut value = guest & KEPT;
        if guest & ReleaseOp::GL2_DISCARD.mask() != 0 {
            value |= ReleaseOp::GL2_WB.mask() | ReleaseOp::GL2_INV.mask();
        }
        if value & ReleaseOp::GLM_WB.mask() != 0 {
            value |= ReleaseOp::GLM_INV.mask();
        }
        value
    }
}
