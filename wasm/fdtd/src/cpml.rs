//! Convolutional PML (CPML) absorbing boundary, Gedney formulation.
//!
//! For a PML normal to x, the stretched-coordinate derivative is
//! `∂/∂x → (1/κ_x)·∂/∂x + ψ_x`, where the convolution term updates as
//! `ψ^{n+1} = b_x·ψ^n + a_x·(∂F/∂x)` with
//! `b_x = exp(−(σ_x/κ_x)·Δt/ε₀)`, `a_x = (b_x − 1)/κ_x`
//! (α = 0 in this build; κ-grading still damps evanescent modes).
//!
//! Only field components whose curl contains a derivative along the PML
//! normal need a ψ term (4 per face). ψ storage is face-local:
//! `npml × face_area` per term, not full-grid.
//!
//! Grading (depth `p = 0` at the outer boundary, `p = npml−1` at the
//! PML/interior interface):
//! `σ(p) = σ_max·((npml−p)/npml)^m`, `κ(p) = 1 + (κ_max−1)·((npml−p)/npml)^m`,
//! `σ_max = (m+1)/(150·π·Δ)`.
//!
//! Indexing convention: for a `lo` face the PML occupies the lowest
//! indices and `p = 0` is the outermost layer; for a `hi` face `p = 0`
//! is the outermost layer at the high-index end. Field values outside
//! the grid (past the outer boundary) are taken as 0 (PEC-backed PML).

use crate::grid::YeeGrid;

/// One PML face: normal axis, side, depth profiles, and 4 ψ arrays.
///
/// ψ assignment by axis (a,b for E-updates; c,d for H-updates):
/// - x-normal: a=ψ(H_z,x)→Ey, b=ψ(H_y,x)→Ez, c=ψ(E_z,x)→Hy, d=ψ(E_y,x)→Hz
/// - y-normal: a=ψ(H_z,y)→Ex, b=ψ(H_x,y)→Ez, c=ψ(E_z,y)→Hx, d=ψ(E_x,y)→Hz
/// - z-normal: a=ψ(H_y,z)→Ex, b=ψ(H_x,z)→Ey, c=ψ(E_y,z)→Hx, d=ψ(E_x,z)→Hy
struct PmlFace {
    axis: u8,
    lo: bool,
    npml: usize,
    b_e: Vec<f32>,
    a_e: Vec<f32>,
    b_h: Vec<f32>,
    a_h: Vec<f32>,
    kinv: Vec<f32>,
    psi_a: Vec<f32>,
    psi_b: Vec<f32>,
    psi_c: Vec<f32>,
    psi_d: Vec<f32>,
    /// Face dimensions transverse to the normal: (n1, n2).
    n1: usize,
    n2: usize,
}

impl PmlFace {
    fn new(axis: u8, lo: bool, npml: usize, n1: usize, n2: usize, delta: f32, dt: f32) -> Self {
        let eps0 = 8.854_187_817e-12f32;
        let mu0 = 4.0e-7 * std::f32::consts::PI;
        let m = 3.0f32;
        let kappa_max = 7.0f32;
        let sigma_max = (m + 1.0) / (150.0 * std::f32::consts::PI * delta);

        let mut b_e = vec![0.0; npml];
        let mut a_e = vec![0.0; npml];
        let mut b_h = vec![0.0; npml];
        let mut a_h = vec![0.0; npml];
        let mut kinv = vec![0.0; npml];
        for p in 0..npml {
            let rho = (npml - p) as f32 / npml as f32;
            let sigma = sigma_max * rho.powf(m);
            let kappa = 1.0 + (kappa_max - 1.0) * rho.powf(m);
            let be = (-(sigma / kappa) * dt / eps0).exp();
            let bh = (-(sigma / kappa) * dt / mu0).exp();
            b_e[p] = be;
            a_e[p] = (be - 1.0) / kappa;
            b_h[p] = bh;
            a_h[p] = (bh - 1.0) / kappa;
            kinv[p] = 1.0 / kappa;
        }
        let area = npml * n1 * n2;
        Self {
            axis,
            lo,
            npml,
            b_e,
            a_e,
            b_h,
            a_h,
            kinv,
            psi_a: vec![0.0; area],
            psi_b: vec![0.0; area],
            psi_c: vec![0.0; area],
            psi_d: vec![0.0; area],
            n1,
            n2,
        }
    }

    #[inline]
    fn psi_idx(&self, p: usize, q1: usize, q2: usize) -> usize {
        (p * self.n1 + q1) * self.n2 + q2
    }
}

/// Six-face CPML.
pub struct Cpml {
    faces: Vec<PmlFace>, // xlo, xhi, ylo, yhi, zlo, zhi
}

impl Cpml {
    /// Build CPML for a grid. `delta` is the cell size (uniform grid).
    pub fn new(g: &YeeGrid, npml: usize, dt: f32) -> Self {
        let d = g.dx;
        let mut faces = Vec::with_capacity(6);
        faces.push(PmlFace::new(0, true, npml, g.ny, g.nz, d, dt));
        faces.push(PmlFace::new(0, false, npml, g.ny, g.nz, d, dt));
        faces.push(PmlFace::new(1, true, npml, g.nx, g.nz, d, dt));
        faces.push(PmlFace::new(1, false, npml, g.nx, g.nz, d, dt));
        faces.push(PmlFace::new(2, true, npml, g.nx, g.ny, d, dt));
        faces.push(PmlFace::new(2, false, npml, g.nx, g.ny, d, dt));
        Self { faces }
    }

    /// E-field PML correction after the standard E update.
    ///
    /// Per face/component: `ΔE = Cb·((1−1/κ)·∂F/∂n − ψ/κ)`,
    /// `ψ ← b·ψ + a·∂F/∂n`.
    pub fn correct_e(&mut self, g: &mut YeeGrid, cb_x: &[f32], cb_y: &[f32], cb_z: &[f32]) {
        for fi in 0..self.faces.len() {
            let (axis, lo, npml, n1, n2) = {
                let f = &self.faces[fi];
                (f.axis, f.lo, f.npml, f.n1, f.n2)
            };
            match axis {
                0 => self.correct_e_x(fi, lo, npml, n1, n2, g, cb_y, cb_z),
                1 => self.correct_e_y(fi, lo, npml, n1, n2, g, cb_x, cb_z),
                _ => self.correct_e_z(fi, lo, npml, n1, n2, g, cb_x, cb_y),
            }
        }
    }

    /// x-normal face: correct Ey (via dHz/dx) and Ez (via dHy/dx).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_x(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        ny: usize,
        nz: usize,
        g: &mut YeeGrid,
        cb_y: &[f32],
        cb_z: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let nx = g.nx;
        let dx = g.dx;
        for p in 0..npml {
            let be = f.b_e[p];
            let ae = f.a_e[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for j in 0..ny {
                for k in 0..nz {
                    // Ey has x-size nx; outermost layer is i=nx−1 (hi) or i=0 (lo).
                    let ie = if lo { p } else { nx - 1 - p };
                    // dHz/dx at Ey(ie+½): (hz[ie+1] − hz[ie])/dx.
                    let dhzdx = (g.hz[g.hz_idx(ie + 1, j, k)] - g.hz[g.hz_idx(ie, j, k)]) / dx;
                    let pi = f.psi_idx(p, j, k);
                    let psi = be * f.psi_a[pi] + ae * dhzdx;
                    f.psi_a[pi] = psi;
                    let ei = g.ey_idx(ie, j, k);
                    g.ey[ei] += cb_y[ei] * (-omk * dhzdx + psi * ki);

                    // Ez: dHy/dx at Ez(ie+½). Update has +∂Hy/∂x (s=+1):
                    // Δ = Cb·(−(1−1/κ)·d + ψ/κ).
                    let dhydx = (g.hy[g.hy_idx(ie + 1, j, k)] - g.hy[g.hy_idx(ie, j, k)]) / dx;
                    let psi2 = be * f.psi_b[pi] + ae * dhydx;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ez_idx(ie, j, k);
                    g.ez[ei2] += cb_z[ei2] * (omk * dhydx - psi2 * ki);
                }
            }
        }
    }

    /// y-normal face: correct Ex (via dHz/dy) and Ez (via dHx/dy).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_y(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        nx: usize,
        nz: usize,
        g: &mut YeeGrid,
        cb_x: &[f32],
        cb_z: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let ny = g.ny;
        let dy = g.dy;
        for p in 0..npml {
            let be = f.b_e[p];
            let ae = f.a_e[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for i in 0..nx {
                for k in 0..nz {
                    let je = if lo { p } else { ny - 1 - p };
                    // Ex: update has +∂Hz/∂y (s=+1): Δ = Cb·(−(1−1/κ)·d + ψ/κ).
                    let dhzdy = (g.hz[g.hz_idx(i, je + 1, k)] - g.hz[g.hz_idx(i, je, k)]) / dy;
                    let pi = f.psi_idx(p, i, k);
                    let psi = be * f.psi_a[pi] + ae * dhzdy;
                    f.psi_a[pi] = psi;
                    let ei = g.ex_idx(i, je, k);
                    g.ex[ei] += cb_x[ei] * (omk * dhzdy - psi * ki);

                    let dhxdy = (g.hx[g.hx_idx(i, je + 1, k)] - g.hx[g.hx_idx(i, je, k)]) / dy;
                    let psi2 = be * f.psi_b[pi] + ae * dhxdy;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ez_idx(i, je, k);
                    g.ez[ei2] += cb_z[ei2] * (-omk * dhxdy + psi2 * ki);
                }
            }
        }
    }

    /// z-normal face: correct Ex (via dHy/dz) and Ey (via dHx/dz).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_z(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        nx: usize,
        ny: usize,
        g: &mut YeeGrid,
        cb_x: &[f32],
        cb_y: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let nz = g.nz;
        let dz = g.dz;
        for p in 0..npml {
            let be = f.b_e[p];
            let ae = f.a_e[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for i in 0..nx {
                for j in 0..ny {
                    let ke = if lo { p } else { nz - 1 - p };
                    let dhydz = (g.hy[g.hy_idx(i, j, ke + 1)] - g.hy[g.hy_idx(i, j, ke)]) / dz;
                    let pi = f.psi_idx(p, i, j);
                    let psi = be * f.psi_a[pi] + ae * dhydz;
                    f.psi_a[pi] = psi;
                    let ei = g.ex_idx(i, j, ke);
                    g.ex[ei] += cb_x[ei] * (-omk * dhydz + psi * ki);

                    let dhxdz = (g.hx[g.hx_idx(i, j, ke + 1)] - g.hx[g.hx_idx(i, j, ke)]) / dz;
                    let psi2 = be * f.psi_b[pi] + ae * dhxdz;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ey_idx(i, j, ke);
                    // Ey: update has +∂Hx/∂z (s=+1): Δ = Cb·(−(1−1/κ)·d + ψ/κ).
                    g.ey[ei2] += cb_y[ei2] * (omk * dhxdz - psi2 * ki);
                }
            }
        }
    }

    /// H-field PML correction after the standard H update.
    ///
    /// Per face/component: `ΔH = −(dt/μ₀)·s·(−(1−1/κ)·∂F/∂n + ψ/κ)`,
    /// where `s = ±1` is the sign of `∂F/∂n` in the bracket
    /// (`H −= c·[s·∂F/∂n + …]`), `ψ ← b·ψ + a·∂F/∂n`.
    pub fn correct_h(&mut self, g: &mut YeeGrid, dt: f32) {
        let mu0 = 4.0e-7 * std::f32::consts::PI;
        let c = dt / mu0;
        for fi in 0..self.faces.len() {
            let (axis, lo, npml, n1, n2) = {
                let f = &self.faces[fi];
                (f.axis, f.lo, f.npml, f.n1, f.n2)
            };
            match axis {
                0 => self.correct_h_x(fi, lo, npml, n1, n2, g, c),
                1 => self.correct_h_y(fi, lo, npml, n1, n2, g, c),
                _ => self.correct_h_z(fi, lo, npml, n1, n2, g, c),
            }
        }
    }

    /// x-normal: correct Hy (via dEz/dx) and Hz (via dEy/dx).
    fn correct_h_x(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        ny: usize,
        nz: usize,
        g: &mut YeeGrid,
        c: f32,
    ) {
        let f = &mut self.faces[fi];
        let nx = g.nx;
        let dx = g.dx;
        for p in 0..npml {
            let bh = f.b_h[p];
            let ah = f.a_h[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for j in 0..ny {
                for k in 0..nz {
                    // Hy has x-size nx+1; outermost is i=nx (hi) or i=0 (lo).
                    let ih = if lo { p } else { nx - p };
                    // dEz/dx at Hy(ih): (ez[ih] − ez[ih−1])/dx, out-of-range = 0.
                    let e_hi = if ih < nx { g.ez[g.ez_idx(ih, j, k)] } else { 0.0 };
                    let e_lo = if ih > 0 { g.ez[g.ez_idx(ih - 1, j, k)] } else { 0.0 };
                    // Hy: bracket has +∂Ez/∂x (s=+1): Δ = −c·(−(1−1/κ)·d + ψ/κ).
                    let dezdx = (e_hi - e_lo) / dx;
                    let pi = f.psi_idx(p, j, k);
                    let psi = bh * f.psi_c[pi] + ah * dezdx;
                    f.psi_c[pi] = psi;
                    let hi = g.hy_idx(ih, j, k);
                    g.hy[hi] -= c * (-omk * dezdx + psi * ki);

                    // Hz: dEy/dx at Hz(ih).
                    let e2_hi = if ih < nx { g.ey[g.ey_idx(ih, j, k)] } else { 0.0 };
                    let e2_lo = if ih > 0 { g.ey[g.ey_idx(ih - 1, j, k)] } else { 0.0 };
                    let deydx = (e2_hi - e2_lo) / dx;
                    let psi2 = bh * f.psi_d[pi] + ah * deydx;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hz_idx(ih, j, k);
                    g.hz[hi2] -= c * (omk * deydx - psi2 * ki);
                }
            }
        }
    }

    /// y-normal: correct Hx (via dEz/dy) and Hz (via dEx/dy).
    fn correct_h_y(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        nx: usize,
        nz: usize,
        g: &mut YeeGrid,
        c: f32,
    ) {
        let f = &mut self.faces[fi];
        let ny = g.ny;
        let dy = g.dy;
        for p in 0..npml {
            let bh = f.b_h[p];
            let ah = f.a_h[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for i in 0..nx {
                for k in 0..nz {
                    let jh = if lo { p } else { ny - p };
                    let e_hi = if jh < ny { g.ez[g.ez_idx(i, jh, k)] } else { 0.0 };
                    let e_lo = if jh > 0 { g.ez[g.ez_idx(i, jh - 1, k)] } else { 0.0 };
                    let dezdy = (e_hi - e_lo) / dy;
                    let pi = f.psi_idx(p, i, k);
                    let psi = bh * f.psi_c[pi] + ah * dezdy;
                    f.psi_c[pi] = psi;
                    let hi = g.hx_idx(i, jh, k);
                    g.hx[hi] -= c * (omk * dezdy - psi * ki);

                    let e2_hi = if jh < ny { g.ex[g.ex_idx(i, jh, k)] } else { 0.0 };
                    let e2_lo = if jh > 0 { g.ex[g.ex_idx(i, jh - 1, k)] } else { 0.0 };
                    let dexdy = (e2_hi - e2_lo) / dy;
                    let psi2 = bh * f.psi_d[pi] + ah * dexdy;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hz_idx(i, jh, k);
                    // Hz: bracket has +∂Ex/∂y (s=+1): Δ = −c·(−(1−1/κ)·d + ψ/κ).
                    g.hz[hi2] -= c * (-omk * dexdy + psi2 * ki);
                }
            }
        }
    }

    /// z-normal: correct Hx (via dEy/dz) and Hy (via dEx/dz).
    fn correct_h_z(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        nx: usize,
        ny: usize,
        g: &mut YeeGrid,
        c: f32,
    ) {
        let f = &mut self.faces[fi];
        let nz = g.nz;
        let dz = g.dz;
        for p in 0..npml {
            let bh = f.b_h[p];
            let ah = f.a_h[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for i in 0..nx {
                for j in 0..ny {
                    let kh = if lo { p } else { nz - p };
                    let e_hi = if kh < nz { g.ey[g.ey_idx(i, j, kh)] } else { 0.0 };
                    let e_lo = if kh > 0 { g.ey[g.ey_idx(i, j, kh - 1)] } else { 0.0 };
                    let deydz = (e_hi - e_lo) / dz;
                    let pi = f.psi_idx(p, i, j);
                    let psi = bh * f.psi_c[pi] + ah * deydz;
                    f.psi_c[pi] = psi;
                    let hi = g.hx_idx(i, j, kh);
                    // Hx: bracket has +∂Ey/∂z (s=+1): Δ = −c·(−(1−1/κ)·d + ψ/κ).
                    g.hx[hi] -= c * (-omk * deydz + psi * ki);

                    let e2_hi = if kh < nz { g.ex[g.ex_idx(i, j, kh)] } else { 0.0 };
                    let e2_lo = if kh > 0 { g.ex[g.ex_idx(i, j, kh - 1)] } else { 0.0 };
                    let dexdz = (e2_hi - e2_lo) / dz;
                    let psi2 = bh * f.psi_d[pi] + ah * dexdz;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hy_idx(i, j, kh);
                    g.hy[hi2] -= c * (omk * dexdz - psi2 * ki);
                }
            }
        }
    }
}
