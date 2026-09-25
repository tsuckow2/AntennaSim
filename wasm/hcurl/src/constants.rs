//! Physical constants (SI units).
//!
//! All geometry is in **meters**, frequencies in **Hz**, impedances in
//! **ohms**. Time-harmonic convention: fields vary as `e^{+jωt}`.

/// Vacuum permittivity, F/m.
pub const EPS0: f64 = 8.8541878128e-12;
/// Vacuum permeability, H/m.
pub const MU0: f64 = 4.0 * std::f64::consts::PI * 1e-7;
/// Speed of light in vacuum, m/s.
pub const C0: f64 = 299792458.0;
/// Intrinsic impedance of free space, Ω.
pub const ETA0: f64 = 376.73031346177066; // sqrt(MU0 / EPS0)

/// Conductivity at or above which a tet is treated as a perfect conductor
/// and excluded from the computational domain (its surface becomes PEC).
pub const CONDUCTOR_SIGMA_THRESHOLD: f64 = 1e4;
