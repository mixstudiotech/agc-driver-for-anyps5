//! gfx11.5 (gfx1151) host registers the native path may write, by byte offset as the register
//! database and the research listings give them (SH from 0xB000, context from 0x28000, user
//! config from 0x30000).

use crate::ir::RegisterBank;

/// One host register, or an array of `count` registers `stride_bytes` apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostRegister {
    pub name: &'static str,
    pub bank: RegisterBank,
    /// Dword offset of element 0 within its bank (the SET_*_REG offset).
    pub dword: u32,
    pub count: u32,
    pub stride_bytes: u32,
}

impl HostRegister {
    const fn new(name: &'static str, byte: u32, count: u32, stride_bytes: u32) -> Self {
        let (bank, base) = if byte >= 0x3_0000 {
            (RegisterBank::UserConfig, 0x3_0000)
        } else if byte >= 0x2_8000 {
            (RegisterBank::Context, 0x2_8000)
        } else {
            (RegisterBank::Shader, 0xb000)
        };
        Self {
            name,
            bank,
            dword: (byte - base) / 4,
            count,
            stride_bytes,
        }
    }

    /// Dword offset of element `index`.
    pub const fn at(&self, index: u32) -> u32 {
        self.dword + index * self.stride_bytes / 4
    }

    /// Byte address in the register space, as the register database lists it.
    pub const fn byte_address(&self, index: u32) -> u32 {
        let base = match self.bank {
            RegisterBank::Shader => 0xb000,
            RegisterBank::Context => 0x2_8000,
            RegisterBank::UserConfig => 0x3_0000,
        };
        base + self.at(index) * 4
    }

    fn element_of(&self, bank: RegisterBank, dword: u32) -> Option<u32> {
        if bank != self.bank || dword < self.dword {
            return None;
        }
        let step = (self.stride_bytes / 4).max(1);
        let delta = dword - self.dword;
        (delta % step == 0 && delta / step < self.count).then_some(delta / step)
    }
}

/// A located register write target: the table entry and the array element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterName {
    pub register: &'static HostRegister,
    pub index: u32,
}

impl std::fmt::Display for RegisterName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.register.count > 1 {
            write!(f, "{}[{}]", self.register.name, self.index)
        } else {
            f.write_str(self.register.name)
        }
    }
}

macro_rules! host_registers {
    ($($name:ident = $byte:literal $([$count:literal; $stride:literal])?;)*) => {
        $(pub const $name: HostRegister =
            host_registers!(@entry $name, $byte $(, $count, $stride)?);)*
        /// Every register the host path may write.
        pub const HOST_REGISTERS: &[HostRegister] = &[$($name),*];
    };
    (@entry $name:ident, $byte:literal) => {
        HostRegister::new(stringify!($name), $byte, 1, 4)
    };
    (@entry $name:ident, $byte:literal, $count:literal, $stride:literal) => {
        HostRegister::new(stringify!($name), $byte, $count, $stride)
    };
}

host_registers! {
    // SH: pixel stage.
    SPI_SHADER_PGM_RSRC4_PS = 0xb004;
    SPI_SHADER_PGM_RSRC3_PS = 0xb01c;
    SPI_SHADER_PGM_LO_PS = 0xb020;
    SPI_SHADER_PGM_HI_PS = 0xb024;
    SPI_SHADER_PGM_RSRC1_PS = 0xb028;
    SPI_SHADER_PGM_RSRC2_PS = 0xb02c;
    SPI_SHADER_USER_DATA_PS = 0xb030 [32; 4];
    SPI_SHADER_REQ_CTRL_PS = 0xb0c0;
    SPI_SHADER_USER_ACCUM_PS = 0xb0c8 [4; 4];
    // SH: merged ES/GS (NGG) stage. On gfx11, PGM_LO/HI_GS load the wave's s[0:1] user-data
    // pointer; the program address is PGM_LO/HI_ES.
    SPI_SHADER_PGM_RSRC4_GS = 0xb204;
    SPI_SHADER_PGM_RSRC3_GS = 0xb21c;
    SPI_SHADER_PGM_LO_GS = 0xb220;
    SPI_SHADER_PGM_HI_GS = 0xb224;
    SPI_SHADER_PGM_RSRC1_GS = 0xb228;
    SPI_SHADER_PGM_RSRC2_GS = 0xb22c;
    SPI_SHADER_USER_DATA_GS = 0xb230 [32; 4];
    SPI_SHADER_USER_ACCUM_ESGS = 0xb2c8 [4; 4];
    SPI_SHADER_PGM_LO_ES = 0xb320;
    SPI_SHADER_PGM_HI_ES = 0xb324;
    // SH: merged LS/HS stage (PGM_LO/HI_HS are the HS s[0:1] pointer on gfx11).
    SPI_SHADER_PGM_RSRC4_HS = 0xb404;
    SPI_SHADER_PGM_RSRC3_HS = 0xb41c;
    SPI_SHADER_PGM_LO_HS = 0xb420;
    SPI_SHADER_PGM_HI_HS = 0xb424;
    SPI_SHADER_PGM_RSRC1_HS = 0xb428;
    SPI_SHADER_PGM_RSRC2_HS = 0xb42c;
    SPI_SHADER_USER_DATA_HS = 0xb430 [32; 4];
    SPI_SHADER_USER_ACCUM_LSHS = 0xb4c8 [4; 4];
    SPI_SHADER_PGM_LO_LS = 0xb520;
    SPI_SHADER_PGM_HI_LS = 0xb524;
    // SH: compute.
    COMPUTE_DISPATCH_INITIATOR = 0xb800;
    COMPUTE_DIM = 0xb804 [3; 4];
    COMPUTE_START = 0xb810 [3; 4];
    COMPUTE_NUM_THREAD = 0xb81c [3; 4];
    COMPUTE_PGM_LO = 0xb830;
    COMPUTE_PGM_HI = 0xb834;
    COMPUTE_DISPATCH_SCRATCH_BASE_LO = 0xb840;
    COMPUTE_DISPATCH_SCRATCH_BASE_HI = 0xb844;
    COMPUTE_PGM_RSRC1 = 0xb848;
    COMPUTE_PGM_RSRC2 = 0xb84c;
    COMPUTE_RESOURCE_LIMITS = 0xb854;
    COMPUTE_STATIC_THREAD_MGMT_SE0 = 0xb858;
    COMPUTE_STATIC_THREAD_MGMT_SE1 = 0xb85c;
    COMPUTE_TMPRING_SIZE = 0xb860;
    COMPUTE_STATIC_THREAD_MGMT_SE2 = 0xb864;
    COMPUTE_STATIC_THREAD_MGMT_SE3 = 0xb868;
    COMPUTE_USER_ACCUM = 0xb890 [4; 4];
    COMPUTE_PGM_RSRC3 = 0xb8a0;
    COMPUTE_STATIC_THREAD_MGMT_SE4 = 0xb8ac [4; 4];
    COMPUTE_DISPATCH_INTERLEAVE = 0xb8bc;
    COMPUTE_USER_DATA = 0xb900 [16; 4];
    COMPUTE_DISPATCH_TUNNEL = 0xb9f4;
    // Context: depth/stencil.
    DB_RENDER_CONTROL = 0x28000;
    DB_COUNT_CONTROL = 0x28004;
    DB_DEPTH_VIEW = 0x28008;
    DB_RENDER_OVERRIDE = 0x2800c;
    DB_RENDER_OVERRIDE2 = 0x28010;
    DB_HTILE_DATA_BASE = 0x28014;
    DB_DEPTH_SIZE_XY = 0x2801c;
    DB_DEPTH_BOUNDS_MIN = 0x28020;
    DB_DEPTH_BOUNDS_MAX = 0x28024;
    DB_STENCIL_CLEAR = 0x28028;
    DB_DEPTH_CLEAR = 0x2802c;
    PA_SC_SCREEN_SCISSOR_TL = 0x28030;
    PA_SC_SCREEN_SCISSOR_BR = 0x28034;
    DB_Z_INFO = 0x28040;
    DB_STENCIL_INFO = 0x28044;
    DB_Z_READ_BASE = 0x28048;
    DB_STENCIL_READ_BASE = 0x2804c;
    DB_Z_WRITE_BASE = 0x28050;
    DB_STENCIL_WRITE_BASE = 0x28054;
    DB_Z_READ_BASE_HI = 0x28068;
    DB_STENCIL_READ_BASE_HI = 0x2806c;
    DB_Z_WRITE_BASE_HI = 0x28070;
    DB_STENCIL_WRITE_BASE_HI = 0x28074;
    DB_HTILE_DATA_BASE_HI = 0x28078;
    DB_RMI_L2_CACHE_CONTROL = 0x2807c;
    TA_BC_BASE_ADDR = 0x28080;
    TA_BC_BASE_ADDR_HI = 0x28084;
    // Context: scan converter, scissors and viewports.
    PA_SC_WINDOW_OFFSET = 0x28200;
    PA_SC_WINDOW_SCISSOR_TL = 0x28204;
    PA_SC_WINDOW_SCISSOR_BR = 0x28208;
    PA_SC_CLIPRECT_RULE = 0x2820c;
    PA_SC_EDGERULE = 0x28230;
    PA_SU_HARDWARE_SCREEN_OFFSET = 0x28234;
    CB_TARGET_MASK = 0x28238;
    CB_SHADER_MASK = 0x2823c;
    PA_SC_GENERIC_SCISSOR_TL = 0x28240;
    PA_SC_GENERIC_SCISSOR_BR = 0x28244;
    PA_SC_VPORT_SCISSOR_TL = 0x28250 [16; 8];
    PA_SC_VPORT_SCISSOR_BR = 0x28254 [16; 8];
    PA_SC_VPORT_ZMIN = 0x282d0 [16; 8];
    PA_SC_VPORT_ZMAX = 0x282d4 [16; 8];
    PA_SC_TILE_STEERING_OVERRIDE = 0x2835c;
    PA_SC_VRS_OVERRIDE_CNTL = 0x283d0;
    PA_SC_VRS_RATE_CACHE_CNTL = 0x283e4;
    VGT_MULTI_PRIM_IB_RESET_INDX = 0x2840c;
    CB_RMI_GL2_CACHE_CONTROL = 0x28410;
    CB_BLEND_RED = 0x28414;
    CB_BLEND_GREEN = 0x28418;
    CB_BLEND_BLUE = 0x2841c;
    CB_BLEND_ALPHA = 0x28420;
    CB_COVERAGE_OUT_CONTROL = 0x28428;
    DB_STENCIL_CONTROL = 0x2842c;
    DB_STENCILREFMASK = 0x28430;
    DB_STENCILREFMASK_BF = 0x28434;
    PA_CL_VPORT_XSCALE = 0x2843c [16; 0x18];
    PA_CL_VPORT_XOFFSET = 0x28440 [16; 0x18];
    PA_CL_VPORT_YSCALE = 0x28444 [16; 0x18];
    PA_CL_VPORT_YOFFSET = 0x28448 [16; 0x18];
    PA_CL_VPORT_ZSCALE = 0x2844c [16; 0x18];
    PA_CL_VPORT_ZOFFSET = 0x28450 [16; 0x18];
    PA_RATE_CNTL = 0x28620;
    // Context: shader interface.
    SPI_PS_INPUT_CNTL = 0x28644 [32; 4];
    SPI_VS_OUT_CONFIG = 0x286c4;
    SPI_PS_INPUT_ENA = 0x286cc;
    SPI_PS_INPUT_ADDR = 0x286d0;
    SPI_INTERP_CONTROL_0 = 0x286d4;
    SPI_PS_IN_CONTROL = 0x286d8;
    SPI_BARYC_SSAA_CNTL = 0x286dc;
    SPI_BARYC_CNTL = 0x286e0;
    SPI_TMPRING_SIZE = 0x286e8;
    SPI_GFX_SCRATCH_BASE_LO = 0x286ec;
    SPI_GFX_SCRATCH_BASE_HI = 0x286f0;
    SPI_SHADER_IDX_FORMAT = 0x28708;
    SPI_SHADER_POS_FORMAT = 0x2870c;
    SPI_SHADER_Z_FORMAT = 0x28710;
    SPI_SHADER_COL_FORMAT = 0x28714;
    SX_PS_DOWNCONVERT_CONTROL = 0x28750;
    SX_PS_DOWNCONVERT = 0x28754;
    SX_BLEND_OPT_EPSILON = 0x28758;
    SX_BLEND_OPT_CONTROL = 0x2875c;
    SX_MRT_BLEND_OPT = 0x28760 [8; 4];
    CB_BLEND_CONTROL = 0x28780 [8; 4];
    PA_CL_POINT_X_RAD = 0x287d4;
    PA_CL_POINT_Y_RAD = 0x287d8;
    PA_CL_POINT_SIZE = 0x287dc;
    PA_CL_POINT_CULL_RAD = 0x287e0;
    GE_MAX_OUTPUT_PER_SUBGROUP = 0x287fc;
    DB_DEPTH_CONTROL = 0x28800;
    DB_EQAA = 0x28804;
    CB_COLOR_CONTROL = 0x28808;
    DB_SHADER_CONTROL = 0x2880c;
    PA_CL_CLIP_CNTL = 0x28810;
    PA_SU_SC_MODE_CNTL = 0x28814;
    PA_CL_VTE_CNTL = 0x28818;
    PA_CL_VS_OUT_CNTL = 0x2881c;
    PA_CL_NANINF_CNTL = 0x28820;
    PA_SU_LINE_STIPPLE_CNTL = 0x28824;
    PA_SU_LINE_STIPPLE_SCALE = 0x28828;
    PA_SU_PRIM_FILTER_CNTL = 0x2882c;
    PA_SU_SMALL_PRIM_FILTER_CNTL = 0x28830;
    PA_CL_NGG_CNTL = 0x28838;
    PA_SU_OVER_RASTERIZATION_CNTL = 0x2883c;
    PA_STEREO_CNTL = 0x28840;
    PA_CL_VRS_CNTL = 0x28848;
    PA_SU_POINT_SIZE = 0x28a00;
    PA_SU_POINT_MINMAX = 0x28a04;
    PA_SU_LINE_CNTL = 0x28a08;
    PA_SC_LINE_STIPPLE = 0x28a0c;
    VGT_HOS_MAX_TESS_LEVEL = 0x28a18;
    VGT_HOS_MIN_TESS_LEVEL = 0x28a1c;
    PA_SC_MODE_CNTL_0 = 0x28a48;
    PA_SC_MODE_CNTL_1 = 0x28a4c;
    VGT_ENHANCE = 0x28a50;
    VGT_PRIMITIVEID_EN = 0x28a84;
    VGT_PRIMITIVEID_RESET = 0x28a8c;
    VGT_DRAW_PAYLOAD_CNTL = 0x28a98;
    VGT_ESGS_RING_ITEMSIZE = 0x28aac;
    VGT_REUSE_OFF = 0x28ab4;
    DB_HTILE_SURFACE = 0x28abc;
    VGT_GS_MAX_VERT_OUT = 0x28b38;
    GE_NGG_SUBGRP_CNTL = 0x28b4c;
    VGT_TESS_DISTRIBUTION = 0x28b50;
    VGT_SHADER_STAGES_EN = 0x28b54;
    VGT_LS_HS_CONFIG = 0x28b58;
    VGT_TF_PARAM = 0x28b6c;
    DB_ALPHA_TO_MASK = 0x28b70;
    PA_SU_POLY_OFFSET_DB_FMT_CNTL = 0x28b78;
    PA_SU_POLY_OFFSET_CLAMP = 0x28b7c;
    PA_SU_POLY_OFFSET_FRONT_SCALE = 0x28b80;
    PA_SU_POLY_OFFSET_FRONT_OFFSET = 0x28b84;
    PA_SU_POLY_OFFSET_BACK_SCALE = 0x28b88;
    PA_SU_POLY_OFFSET_BACK_OFFSET = 0x28b8c;
    VGT_GS_INSTANCE_CNT = 0x28b90;
    PA_SC_CENTROID_PRIORITY_0 = 0x28bd4;
    PA_SC_CENTROID_PRIORITY_1 = 0x28bd8;
    PA_SC_LINE_CNTL = 0x28bdc;
    PA_SC_AA_CONFIG = 0x28be0;
    PA_SU_VTX_CNTL = 0x28be4;
    PA_CL_GB_VERT_CLIP_ADJ = 0x28be8;
    PA_CL_GB_VERT_DISC_ADJ = 0x28bec;
    PA_CL_GB_HORZ_CLIP_ADJ = 0x28bf0;
    PA_CL_GB_HORZ_DISC_ADJ = 0x28bf4;
    PA_SC_AA_SAMPLE_LOCS_PIXEL = 0x28bf8 [16; 4];
    PA_SC_AA_MASK_X0Y0_X1Y0 = 0x28c38;
    PA_SC_AA_MASK_X0Y1_X1Y1 = 0x28c3c;
    PA_SC_SHADER_CONTROL = 0x28c40;
    PA_SC_BINNER_CNTL_0 = 0x28c44;
    PA_SC_BINNER_CNTL_1 = 0x28c48;
    PA_SC_CONSERVATIVE_RASTERIZATION_CNTL = 0x28c4c;
    PA_SC_NGG_MODE_CNTL = 0x28c50;
    PA_SC_BINNER_CNTL_2 = 0x28c54;
    // Context: color targets (gfx11 layout: no PITCH/SLICE/CMASK/FMASK/CLEAR_WORD).
    CB_COLOR_BASE = 0x28c60 [8; 0x3c];
    CB_COLOR_VIEW = 0x28c6c [8; 0x3c];
    CB_COLOR_INFO = 0x28c70 [8; 0x3c];
    CB_COLOR_ATTRIB = 0x28c74 [8; 0x3c];
    CB_COLOR_FDCC_CONTROL = 0x28c78 [8; 0x3c];
    CB_COLOR_DCC_BASE = 0x28c94 [8; 0x3c];
    CB_COLOR_BASE_EXT = 0x28e40 [8; 4];
    CB_COLOR_DCC_BASE_EXT = 0x28ea0 [8; 4];
    CB_COLOR_ATTRIB2 = 0x28ec0 [8; 4];
    CB_COLOR_ATTRIB3 = 0x28ee0 [8; 4];
    // User config.
    VGT_PRIMITIVE_TYPE = 0x30908;
    VGT_INDEX_TYPE = 0x3090c;
    GE_MIN_VTX_INDX = 0x30924;
    GE_INDX_OFFSET = 0x30928;
    GE_MULTI_PRIM_IB_RESET_EN = 0x3092c;
    VGT_NUM_INSTANCES = 0x30934;
    VGT_TF_RING_SIZE = 0x30938;
    VGT_HS_OFFCHIP_PARAM = 0x3093c;
    VGT_TF_MEMORY_BASE = 0x30940;
    GE_MAX_VTX_INDX = 0x30964;
    VGT_INSTANCE_BASE_ID = 0x30968;
    GE_CNTL = 0x3096c;
    GE_STEREO_CNTL = 0x3097c;
    GE_PC_ALLOC = 0x30980;
    VGT_TF_MEMORY_BASE_HI = 0x30984;
    GE_USER_VGPR_EN = 0x30988;
    VGT_GS_OUT_PRIM_TYPE = 0x30998;
    PA_SU_LINE_STIPPLE_VALUE = 0x30a00;
    PA_SC_LINE_STIPPLE_STATE = 0x30a04;
    TA_CS_BC_BASE_ADDR = 0x30e00;
    TA_CS_BC_BASE_ADDR_HI = 0x30e04;
    SPI_GS_THROTTLE_CNTL1 = 0x31110;
    SPI_GS_THROTTLE_CNTL2 = 0x31114;
    SPI_ATTRIBUTE_RING_BASE = 0x31118;
    SPI_ATTRIBUTE_RING_SIZE = 0x3111c;
}

/// Host graphics program address registers (PS, merged ES/GS, merged LS/HS). Unlike the guest
/// list, 0xB220 and 0xB420 are absent: on gfx11 they are user-data pointers.
pub const HOST_GRAPHICS_PROGRAMS: [HostRegister; 3] = [
    SPI_SHADER_PGM_LO_PS,
    SPI_SHADER_PGM_LO_ES,
    SPI_SHADER_PGM_LO_LS,
];

/// Locates `dword` of `bank` in the host table.
pub fn lookup(bank: RegisterBank, dword: u32) -> Option<RegisterName> {
    HOST_REGISTERS.iter().find_map(|register| {
        register
            .element_of(bank, dword)
            .map(|index| RegisterName { register, index })
    })
}

/// A printable name for any host register write, known or not.
pub fn describe(bank: RegisterBank, dword: u32) -> String {
    match lookup(bank, dword) {
        Some(name) => name.to_string(),
        None => format!("UNKNOWN_{bank:?}_0x{dword:03x}"),
    }
}
