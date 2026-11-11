//! The host's ray unit against the gfx1013 reference model, both run on the GPU with the same
//! nodes and rays (decision D44).

#[cfg(all(windows, feature = "wddm"))]
mod gpu;
#[cfg(all(windows, feature = "wddm"))]
mod kernels;
