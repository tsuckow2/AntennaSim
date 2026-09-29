//! Yee-grid Finite-Difference Time-Domain solver for antenna S-parameters.
//!
//! Built as an independent cross-check of the H(curl) FEM solver
//! (`wasm/hcurl`). The primary validation target is the Snyder 2.44 GHz
//! probe-fed patch: broadband Zin(f)/S11(f) from a single Gaussian-pulse
//! run, via a hard voltage source and an Ampère-loop current probe.
//!
//! Modules:
//! - [`grid`]: Yee-grid field storage and indexing.
//! - [`material`]: per-cell Ca/Cb update coefficients and PEC masks.
//! - [`cpml`]: convolutional PML absorbing boundary (6 faces).
//! - [`geometry`]: Snyder patch voxelization.
//! - [`source`]: Gaussian source, V/I probes, on-the-fly DFT.
//! - [`solver`]: time-stepping loop.

pub mod cpml;
pub mod geometry;
pub mod grid;
pub mod material;
pub mod solver;
pub mod source;
pub mod thin_wire;
