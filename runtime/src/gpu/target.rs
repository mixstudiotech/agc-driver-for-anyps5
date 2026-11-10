//! The host instruction set guest shaders are translated for.

use shader::TargetIsa;

/// gfx11.5, the Radeon 8060S (gfx1151): the target of the CPU path and of any adapter whose GFX
/// IP version is unknown.
pub const DEFAULT_TARGET: TargetIsa = TargetIsa::Gfx115;

/// GFX IP versions (major, minor, stepping) of adapters known by PCI ID. The WDDM transport
/// reports no GFX IP version, so it is looked up here.
const KNOWN_ADAPTERS: &[(u32, u32, (u32, u32, u32))] = &[
    // Radeon 8060S (Strix Halo), gfx1151: AMD 1002:1586 on mix.
    (0x1002, 0x1586, (11, 5, 1)),
];

/// The GFX IP version of a known adapter.
pub fn gfx_ip_for_device(vendor_id: u32, device_id: u32) -> Option<(u32, u32, u32)> {
    KNOWN_ADAPTERS
        .iter()
        .find(|&&(vendor, device, _)| vendor == vendor_id && device == device_id)
        .map(|&(_, _, gfx_ip)| gfx_ip)
}

/// The target for a GFX IP version, or [`DEFAULT_TARGET`] when it is unknown or has none.
pub fn target_for_gfx_ip(gfx_ip: Option<(u32, u32, u32)>) -> TargetIsa {
    gfx_ip
        .and_then(|(major, minor, stepping)| TargetIsa::from_gfx_ip(major, minor, stepping))
        .unwrap_or(DEFAULT_TARGET)
}

/// The target for an opened WDDM adapter, by PCI ID.
#[cfg(feature = "wddm")]
pub fn host_target(identity: &crate::AdapterIdentity) -> TargetIsa {
    target_for_gfx_ip(gfx_ip_for_device(identity.vendor_id, identity.device_id))
}

/// The target for an opened amdgpu device, from the GFX IP version the kernel reports (the
/// stepping is not reported and does not change the target).
#[cfg(feature = "libdrm-amdgpu")]
pub fn host_target(identity: &crate::AdapterIdentity) -> TargetIsa {
    let (major, minor) = identity.gfx_ip;
    target_for_gfx_ip((major != 0).then_some((major, minor, 0)))
}
