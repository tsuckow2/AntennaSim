//! Materials as precomputed FDTD update coefficients.
//!
//! The E update with electric conductivity σ is
//! `E^{n+1} = Ca·E^n + Cb·(∇×H)`, with
//! `Ca = (1 − σ·dt/2ε)/(1 + σ·dt/2ε)`, `Cb = (dt/ε)/(1 + σ·dt/2ε)`.
//! We precompute `Ca, Cb` per Yee cell (from the cell-centered εr, σ) so
//! the inner loop is two multiplies. PEC is a separate boolean mask per
//! E-component: after the E update, masked samples are zeroed.

/// Per-cell update coefficients and PEC masks, sized to the E-field arrays.
pub struct Materials {
    /// `Ca` for ex, ey, ez.
    pub ca_x: Vec<f32>,
    pub ca_y: Vec<f32>,
    pub ca_z: Vec<f32>,
    /// `Cb` for ex, ey, ez.
    pub cb_x: Vec<f32>,
    pub cb_y: Vec<f32>,
    pub cb_z: Vec<f32>,
    /// PEC mask: 1 = perfect conductor, force E = 0 after update.
    pub pec_x: Vec<u8>,
    pub pec_y: Vec<u8>,
    pub pec_z: Vec<u8>,
}

impl Materials {
    /// Free-space everywhere, no PEC.
    pub fn vacuum(nx: usize, ny: usize, nz: usize, dt: f32) -> Self {
        let eps0 = 8.854_187_817e-12f32;
        let cb = dt / eps0;
        Self {
            ca_x: vec![1.0; (nx + 1) * ny * nz],
            ca_y: vec![1.0; nx * (ny + 1) * nz],
            ca_z: vec![1.0; nx * ny * (nz + 1)],
            cb_x: vec![cb; (nx + 1) * ny * nz],
            cb_y: vec![cb; nx * (ny + 1) * nz],
            cb_z: vec![cb; nx * ny * (nz + 1)],
            pec_x: vec![0; (nx + 1) * ny * nz],
            pec_y: vec![0; nx * (ny + 1) * nz],
            pec_z: vec![0; nx * ny * (nz + 1)],
        }
    }

    /// Set a dielectric voxel: relative permittivity `eps_r` and
    /// conductivity `sigma` (S/m) at the E-sample `(comp, i, j, k)`.
    ///
    /// `comp`: 0 = Ex, 1 = Ey, 2 = Ez.
    pub fn set_dielectric(
        &mut self,
        comp: usize,
        idx: usize,
        eps_r: f32,
        sigma: f32,
        dt: f32,
    ) {
        let eps0 = 8.854_187_817e-12f32;
        let eps = eps0 * eps_r;
        let s = sigma * dt / (2.0 * eps);
        let ca = (1.0 - s) / (1.0 + s);
        let cb = (dt / eps) / (1.0 + s);
        match comp {
            0 => {
                self.ca_x[idx] = ca;
                self.cb_x[idx] = cb;
            }
            1 => {
                self.ca_y[idx] = ca;
                self.cb_y[idx] = cb;
            }
            _ => {
                self.ca_z[idx] = ca;
                self.cb_z[idx] = cb;
            }
        }
    }

    /// Mark an E-sample as PEC.
    pub fn set_pec(&mut self, comp: usize, idx: usize) {
        match comp {
            0 => self.pec_x[idx] = 1,
            1 => self.pec_y[idx] = 1,
            _ => self.pec_z[idx] = 1,
        }
    }
}
