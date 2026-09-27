//! Yee-grid field storage and indexing.
//!
//! Standard staggered layout on `nx × ny × nz` cells with spacings
//! `dx, dy, dz`:
//!
//! ```text
//! ex : (nx+1) × ny × nz      Ex(i·dx,        (j+½)·dy, (k+½)·dz)
//! ey : nx × (ny+1) × nz      Ey((i+½)·dx,    j·dy,     (k+½)·dz)
//! ez : nx × ny × (nz+1)      Ez((i+½)·dx,    (j+½)·dy, k·dz)
//! hx : nx × (ny+1) × (nz+1)  Hx((i+½)·dx,    j·dy,     k·dz)
//! hy : (nx+1) × ny × (nz+1)  Hy(i·dx,        (j+½)·dy, k·dz)
//! hz : (nx+1) × (ny+1) × nz  Hz(i·dx,        j·dy,     (k+½)·dz)
//! ```
//!
//! Fields are `f32` (4.5 M cells × 6 components ≈ 108 MB at Δ = 0.5 mm for
//! the Snyder domain). Spectral quantities (DFT) accumulate in `f64`.

/// Yee grid: dimensions, spacings, and the six field arrays.
pub struct YeeGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub ex: Vec<f32>,
    pub ey: Vec<f32>,
    pub ez: Vec<f32>,
    pub hx: Vec<f32>,
    pub hy: Vec<f32>,
    pub hz: Vec<f32>,
}

impl YeeGrid {
    /// Allocate a zeroed grid.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f32, dy: f32, dz: f32) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            ex: vec![0.0; (nx + 1) * ny * nz],
            ey: vec![0.0; nx * (ny + 1) * nz],
            ez: vec![0.0; nx * ny * (nz + 1)],
            hx: vec![0.0; nx * (ny + 1) * (nz + 1)],
            hy: vec![0.0; (nx + 1) * ny * (nz + 1)],
            hz: vec![0.0; (nx + 1) * (ny + 1) * nz],
        }
    }

    /// Courant-limited timestep: `dt = cfl / (c·√(1/dx² + 1/dy² + 1/dz²))`.
    pub fn courant_dt(&self, cfl: f32) -> f32 {
        let c = 299_792_458.0f32;
        let inv = (1.0 / (self.dx * self.dx)
            + 1.0 / (self.dy * self.dy)
            + 1.0 / (self.dz * self.dz))
            .sqrt();
        cfl / (c * inv)
    }

    // -- index helpers (row-major: x fastest) --

    #[inline]
    pub fn ex_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * (self.nx + 1) + i
    }
    #[inline]
    pub fn ey_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * (self.ny + 1) + j) * self.nx + i
    }
    #[inline]
    pub fn ez_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }
    #[inline]
    pub fn hx_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * (self.ny + 1) + j) * self.nx + i
    }
    #[inline]
    pub fn hy_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * (self.nx + 1) + i
    }
    #[inline]
    pub fn hz_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * (self.ny + 1) + j) * (self.nx + 1) + i
    }

    /// Physical position of an Ex sample.
    pub fn ex_pos(&self, i: usize, j: usize, k: usize) -> (f32, f32, f32) {
        (
            i as f32 * self.dx,
            (j as f32 + 0.5) * self.dy,
            (k as f32 + 0.5) * self.dz,
        )
    }
    /// Physical position of an Ez sample.
    pub fn ez_pos(&self, i: usize, j: usize, k: usize) -> (f32, f32, f32) {
        (
            (i as f32 + 0.5) * self.dx,
            (j as f32 + 0.5) * self.dy,
            k as f32 * self.dz,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_layout() {
        let g = YeeGrid::new(10, 12, 8, 0.5e-3, 0.5e-3, 0.5e-3);
        assert_eq!(g.ex.len(), 11 * 12 * 8);
        assert_eq!(g.hz.len(), 11 * 13 * 8);
        // Courant sanity: dt < dx/(c·√3) for cubic cells.
        let dt = g.courant_dt(0.99);
        assert!(dt < 0.5e-3 / (299_792_458.0 * 3.0f32.sqrt()));
        assert!(dt > 0.0);
    }
}
