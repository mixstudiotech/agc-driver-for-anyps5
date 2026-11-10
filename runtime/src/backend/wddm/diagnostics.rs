//! What the KMD says about a device after a fault or an unconfirmed completion:
//! D3DKMTGetDeviceState for the execution state, then for the page-fault state.

use std::fmt;
use windows_sys::Wdk::Graphics::Direct3D::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionState {
    Active,
    Reset,
    Hung,
    Stopped,
    OutOfMemory,
    DmaFault,
    DmaPageFault,
    Other(i32),
}

impl ExecutionState {
    fn from_raw(raw: D3DKMT_DEVICEEXECUTION_STATE) -> Self {
        match raw {
            D3DKMT_DEVICEEXECUTION_ACTIVE => Self::Active,
            D3DKMT_DEVICEEXECUTION_RESET => Self::Reset,
            D3DKMT_DEVICEEXECUTION_HUNG => Self::Hung,
            D3DKMT_DEVICEEXECUTION_STOPPED => Self::Stopped,
            D3DKMT_DEVICEEXECUTION_ERROR_OUTOFMEMORY => Self::OutOfMemory,
            D3DKMT_DEVICEEXECUTION_ERROR_DMAFAULT => Self::DmaFault,
            D3DKMT_DEVICEEXECUTION_ERROR_DMAPAGEFAULT => Self::DmaPageFault,
            other => Self::Other(other),
        }
    }
}

/// D3DKMT_DEVICEPAGEFAULT_STATE.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageFault {
    pub virtual_address: u64,
    pub sequence_number: u64,
    pub pipeline_stage: i32,
    pub bind_table_entry: u32,
    pub flags: i32,
    /// DXGK_FAULT_ERROR_CODE: bit 0 selects a device-specific code in bits 31:1.
    pub error_code: u32,
}

impl PageFault {
    pub fn is_device_specific(&self) -> bool {
        self.error_code & 1 != 0
    }

    pub fn code(&self) -> u32 {
        self.error_code >> 1
    }
}

/// Both queries' results. A status is the call's NTSTATUS; `None` means no device existed to ask.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceState {
    pub execution: Option<ExecutionState>,
    pub execution_status: Option<u32>,
    pub page_fault: Option<PageFault>,
    pub page_fault_status: Option<u32>,
}

impl DeviceState {
    pub(super) fn query(device: u32) -> Self {
        let mut state = Self {
            execution: None,
            execution_status: None,
            page_fault: None,
            page_fault_status: None,
        };
        if device == 0 {
            return state;
        }
        let mut request = D3DKMT_GETDEVICESTATE {
            hDevice: device,
            StateType: D3DKMT_DEVICESTATE_EXECUTION,
            ..Default::default()
        };
        let status = unsafe { D3DKMTGetDeviceState(&mut request) };
        state.execution_status = Some(status as u32);
        if status >= 0 {
            state.execution = Some(ExecutionState::from_raw(unsafe {
                request.Anonymous.ExecutionState
            }));
        }
        let mut request = D3DKMT_GETDEVICESTATE {
            hDevice: device,
            StateType: D3DKMT_DEVICESTATE_PAGE_FAULT,
            ..Default::default()
        };
        let status = unsafe { D3DKMTGetDeviceState(&mut request) };
        state.page_fault_status = Some(status as u32);
        if status >= 0 {
            let fault = unsafe { request.Anonymous.PageFaultState };
            state.page_fault = Some(PageFault {
                virtual_address: fault.FaultedVirtualAddress,
                sequence_number: fault.FaultedPrimitiveAPISequenceNumber,
                pipeline_stage: fault.FaultedPipelineStage,
                bind_table_entry: fault.FaultedBindTableEntry,
                flags: fault.PageFaultFlags,
                error_code: unsafe { fault.FaultErrorCode.Anonymous.Anonymous1._bitfield },
            });
        }
        state
    }
}

impl fmt::Display for DeviceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.execution, self.execution_status) {
            (_, None) => {
                return f.write_str("execution=unavailable page_fault=unavailable (no device)")
            }
            (Some(state), Some(status)) => write!(f, "execution={state:?} (0x{status:08x})")?,
            (None, Some(status)) => write!(f, "execution=query-failed (0x{status:08x})")?,
        }
        match (self.page_fault, self.page_fault_status) {
            (Some(fault), Some(status)) => write!(
                f,
                " page_fault_va=0x{:x} flags=0x{:x} code=0x{:x} device_specific={} stage={} bind_table_entry={} sequence={} (0x{status:08x})",
                fault.virtual_address,
                fault.flags,
                fault.code(),
                fault.is_device_specific(),
                fault.pipeline_stage,
                fault.bind_table_entry,
                fault.sequence_number
            ),
            (_, Some(status)) => write!(f, " page_fault=query-failed (0x{status:08x})"),
            (_, None) => Ok(()),
        }
    }
}
