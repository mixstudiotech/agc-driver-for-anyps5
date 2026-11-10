pub use crate::error::{Error, ErrorKind, Result};
pub use shader::{ShaderStage, TargetIsa};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct GuestAddress(pub u64);

impl GuestAddress {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }

    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    pub fn checked_add(self, offset: u64) -> Option<Self> {
        self.0.checked_add(offset).map(Self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MappingGeneration(pub u64);

impl MappingGeneration {
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum QueueId {
    Graphics,
    AsyncCompute(u32),
}

impl QueueId {
    pub const GRAPHICS_RAW: u32 = 0;
    pub const ASYNC_COMPUTE_BEGIN: u32 = 0x20;
    pub const UNMEASURED_SUBMISSION_QUEUE_END: u32 = 0x58;
    pub const SUBMISSION_QUEUE_END: u32 = 0x59;

    pub fn from_raw(raw: u32, multi_submit: bool) -> Result<Self> {
        if raw == Self::GRAPHICS_RAW {
            return Ok(Self::Graphics);
        }
        let limit = if multi_submit {
            Self::UNMEASURED_SUBMISSION_QUEUE_END
        } else {
            Self::SUBMISSION_QUEUE_END
        };
        if (Self::ASYNC_COMPUTE_BEGIN..limit).contains(&raw) {
            Ok(Self::AsyncCompute(raw))
        } else {
            Err(Error::unsupported(format!(
                "unsupported guest queue 0x{raw:x}"
            )))
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Graphics => Self::GRAPHICS_RAW,
            Self::AsyncCompute(id) => id,
        }
    }

}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompareFunc {
    Always,
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
}

impl CompareFunc {
    pub fn from_raw(raw: u32) -> Result<Self> {
        match raw & 7 {
            0 => Ok(Self::Always),
            1 => Ok(Self::Less),
            2 => Ok(Self::LessEqual),
            3 => Ok(Self::Equal),
            4 => Ok(Self::NotEqual),
            5 => Ok(Self::GreaterEqual),
            6 => Ok(Self::Greater),
            other => Err(Error::invalid(format!("invalid compare function {other}"))),
        }
    }

    pub const fn to_raw(self) -> u32 {
        match self {
            Self::Always => 0,
            Self::Less => 1,
            Self::LessEqual => 2,
            Self::Equal => 3,
            Self::NotEqual => 4,
            Self::GreaterEqual => 5,
            Self::Greater => 6,
        }
    }

}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessKind {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccessRange {
    pub address: GuestAddress,
    pub bytes: usize,
    pub kind: AccessKind,
}

