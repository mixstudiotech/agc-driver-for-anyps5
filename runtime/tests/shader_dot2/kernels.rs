pub struct Kernel {
    pub words: &'static [u32],
    pub signed: bool,
    pub clamp: bool,
    pub alias: u8,
}
pub const KERNELS: &[Kernel] = &[
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc154000,
            0x1c0a0300, 0x85068081, 0x7e0e0300, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: false,
        clamp: false,
        alias: 0,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc154001,
            0x1c0a0300, 0x85068081, 0x7e0e0301, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: false,
        clamp: false,
        alias: 1,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc154002,
            0x1c0a0300, 0x85068081, 0x7e0e0302, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: false,
        clamp: false,
        alias: 2,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc154006,
            0x1c0a0300, 0x85068081, 0x7e0e0306, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: false,
        clamp: false,
        alias: 6,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc15c000,
            0x1c0a0300, 0x85068081, 0x7e0e0300, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: false,
        clamp: true,
        alias: 0,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc15c001,
            0x1c0a0300, 0x85068081, 0x7e0e0301, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: false,
        clamp: true,
        alias: 1,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc15c002,
            0x1c0a0300, 0x85068081, 0x7e0e0302, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: false,
        clamp: true,
        alias: 2,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc15c006,
            0x1c0a0300, 0x85068081, 0x7e0e0306, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: false,
        clamp: true,
        alias: 6,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc144000,
            0x1c0a0300, 0x85068081, 0x7e0e0300, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: true,
        clamp: false,
        alias: 0,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc144001,
            0x1c0a0300, 0x85068081, 0x7e0e0301, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: true,
        clamp: false,
        alias: 1,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc144002,
            0x1c0a0300, 0x85068081, 0x7e0e0302, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: true,
        clamp: false,
        alias: 2,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc144006,
            0x1c0a0300, 0x85068081, 0x7e0e0306, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: true,
        clamp: false,
        alias: 6,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc14c000,
            0x1c0a0300, 0x85068081, 0x7e0e0300, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: true,
        clamp: true,
        alias: 0,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc14c001,
            0x1c0a0300, 0x85068081, 0x7e0e0301, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: true,
        clamp: true,
        alias: 1,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc14c002,
            0x1c0a0300, 0x85068081, 0x7e0e0302, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: true,
        clamp: true,
        alias: 2,
    },
    Kernel {
        words: &[
            0xbefe0481, 0xbeea0404, 0xbf068103, 0x7e000200, 0x7e020201, 0x7e040202, 0xcc14c006,
            0x1c0a0300, 0x85068081, 0x7e0e0306, 0x7e100206, 0x7e12026a, 0x7e14026b, 0xe0780000,
            0x80020700, 0xbf810000,
        ],
        signed: true,
        clamp: true,
        alias: 6,
    },
];

pub fn selectors(words: &[u32], lo: u8, hi: u8) -> Vec<u32> {
    let mut words = words.to_vec();
    let at = words
        .iter()
        .position(|&v| v & 0xff80_0000 == 0xcc00_0000)
        .unwrap();
    words[at] = (words[at] & !(3 << 11)) | (u32::from(lo) << 11);
    words[at + 1] = (words[at + 1] & !(3 << 27)) | (u32::from(hi) << 27);
    words
}
