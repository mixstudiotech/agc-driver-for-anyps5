//! What each Agc module exports (PLAN P3.4, decisions D3, D7): the SDK functions the runtime DLL
//! implements, by the module titles import them from. `agc::shell` turns each list into a
//! shell whose exports are the functions' NIDs, forwarding to the runtime DLL's functions of the
//! same names.

/// The runtime DLL, as shells name it.
pub const RUNTIME_DLL: &str = "agcdrv.dll";
/// The module part of a forwarder (`agcdrv.symbol`).
pub const RUNTIME_MODULE: &str = "agcdrv";
/// The runtime export a shell imports, so the loader loads the runtime from the shell's directory.
pub const RUNTIME_ANCHOR: &str = "agc_runtime_anchor";

/// The export [`RUNTIME_ANCHOR`] names.
#[no_mangle]
pub extern "C" fn agc_runtime_anchor() {}

/// One `.prx` and the SDK functions (names as titles import them, version suffix included),
/// in groups as the runtime's modules export them.
#[derive(Clone, Copy, Debug)]
pub struct Module {
    pub file: &'static str,
    pub groups: &'static [&'static [&'static str]],
}

impl Module {
    pub fn functions(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.groups.iter().flat_map(|group| group.iter().copied())
    }
}

/// The five Agc modules (D3). The lists grow as the functions move into the runtime.
pub const MODULES: &[Module] = &[
    Module { file: "libSceAgc.prx", groups: &[super::abi::REGISTERS, super::abi::commands::COMMANDS, super::abi::commands::UNSUPPORTED_COMMANDS, super::abi::draw::DRAWS, super::abi::misc::MISC, super::abi::shader::SHADERS, super::abi::fusion::FUSION, super::abi::patch::PATCHES] },
    Module { file: "libSceAgcDriver.prx", groups: super::driver::abi::GROUPS },
    Module { file: "libSceAgcResourceRegistration.prx", groups: &[] },
    Module { file: "libSceAgcVsh.prx", groups: &[] },
    Module { file: "libSceAgcVshDebug.prx", groups: &[] },
];

/// A runtime export reached through a shell, proving the forwarding chain.
pub const PROBE: &str = "mesa_agc_shell_probe";
/// What [`mesa_agc_shell_probe`] returns.
pub const PROBE_VALUE: u32 = 0x00a9_c5e1;

/// Returns [`PROBE_VALUE`]; called through a shell forwarder.
#[no_mangle]
pub extern "sysv64" fn mesa_agc_shell_probe() -> u32 {
    PROBE_VALUE
}
