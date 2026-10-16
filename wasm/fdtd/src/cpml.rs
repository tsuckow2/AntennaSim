//! Convolutional PML (CPML) absorbing boundary, Gedney formulation.
//!
//! For a PML normal to x, the stretched-coordinate derivative is
//! `∂/∂x → (1/κ_x)·∂/∂x + ψ_x`, where the convolution term updates as
//! `ψ^{n+1} = b_x·ψ^n + a_x·(∂F/∂x)` with (CFS α included)
//! `b_x = exp(−(σ_x/κ_x + α_x)·Δt/ε₀)`,
//! `a_x = (b_x − 1)·σ_x/(κ_x·(σ_x + κ_x·α_x))`.
//! The H-side ψ coefficients equal the E-side ones: the matched
//! magnetic conductivity σ* = σ·μ₀/ε₀ has decay rate σ*/(κμ₀) =
//! σ/(κε₀). α is graded 0 at the outer boundary to α_max = 0.05 S/m
//! at the interface; it damps grazing/evanescent modes that make
//! α = 0 CPML late-time unstable.
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
        let m = 3.0f32;
        let kappa_max = 7.0f32;
        let sigma_max = (m + 1.0) / (150.0 * std::f32::consts::PI * delta);
        // CFS shift α: damps the grazing/evanescent late-time modes that
        // plain (α = 0) CPML is known to destabilize. Graded opposite σ:
        // 0 at the outer boundary (p = 0), α_max at the PML/interior
        // interface (p = npml−1). α_max = 0.3 S/m is empirical: at 0.05 a
        // PEC-channel grazing mode grew ×1.009/step in the slab probe
        // (examples/cpml_probe.rs, ground + slab blob); 0.3 suppresses
        // it over 6 ns, as does npml = 12 at α = 0.05.
        let alpha_max = 0.3f32;

        let mut b_e = vec![0.0; npml];
        let mut a_e = vec![0.0; npml];
        let mut b_h = vec![0.0; npml];
        let mut a_h = vec![0.0; npml];
        let mut kinv = vec![0.0; npml];
        for p in 0..npml {
            let rho = (npml - p) as f32 / npml as f32;
            let sigma = sigma_max * rho.powf(m);
            let kappa = 1.0 + (kappa_max - 1.0) * rho.powf(m);
            let alpha = alpha_max * (p as f32 / npml as f32);
            // Gedney CPML coefficients. The H-side ψ uses the magnetic
            // conductivity σ* = σ·μ₀/ε₀ (impedance-matched PML), whose
            // decay rate σ*/(κ·μ₀) = σ/(κ·ε₀) is identical to the
            // E-side rate; hence b_h = b_e and a_h = a_e exactly.
            let b = (-(sigma / kappa + alpha) * dt / eps0).exp();
            let a = if sigma + kappa * alpha > 0.0 {
                (b - 1.0) * sigma / (kappa * (sigma + kappa * alpha))
            } else {
                0.0
            };
            b_e[p] = b;
            a_e[p] = a;
            b_h[p] = b;
            a_h[p] = a;
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

/// Owned snapshot of one PML face, for external backends (e.g. the GPU
/// solver) that need the grading coefficients and ψ state.
pub struct PmlFaceData {
    pub axis: u8,
    pub lo: bool,
    pub npml: usize,
    /// Transverse face dimensions.
    pub n1: usize,
    pub n2: usize,
    pub b_e: Vec<f32>,
    pub a_e: Vec<f32>,
    pub b_h: Vec<f32>,
    pub a_h: Vec<f32>,
    pub kinv: Vec<f32>,
    pub psi_a: Vec<f32>,
    pub psi_b: Vec<f32>,
    pub psi_c: Vec<f32>,
    pub psi_d: Vec<f32>,
}

impl Cpml {
    /// Clone all six faces' coefficients and ψ arrays (in face order
    /// xlo, xhi, ylo, yhi, zlo, zhi) for upload to an external backend.
    pub fn snapshot(&self) -> Vec<PmlFaceData> {
        self.faces
            .iter()
            .map(|f| PmlFaceData {
                axis: f.axis,
                lo: f.lo,
                npml: f.npml,
                n1: f.n1,
                n2: f.n2,
                b_e: f.b_e.clone(),
                a_e: f.a_e.clone(),
                b_h: f.b_h.clone(),
                a_h: f.a_h.clone(),
                kinv: f.kinv.clone(),
                psi_a: f.psi_a.clone(),
                psi_b: f.psi_b.clone(),
                psi_c: f.psi_c.clone(),
                psi_d: f.psi_d.clone(),
            })
            .collect()
    }
}

impl Cpml {
    /// Build CPML for a grid. Each face's σ grading uses the cell size
    /// along that face's normal axis (grids may be anisotropic).
    pub fn new(g: &YeeGrid, npml: usize, dt: f32) -> Self {
        let mut faces = Vec::with_capacity(6);
        faces.push(PmlFace::new(0, true, npml, g.ny, g.nz, g.dx, dt));
        faces.push(PmlFace::new(0, false, npml, g.ny, g.nz, g.dx, dt));
        faces.push(PmlFace::new(1, true, npml, g.nx, g.nz, g.dy, dt));
        faces.push(PmlFace::new(1, false, npml, g.nx, g.nz, g.dy, dt));
        faces.push(PmlFace::new(2, true, npml, g.nx, g.ny, g.dz, dt));
        faces.push(PmlFace::new(2, false, npml, g.nx, g.ny, g.dz, dt));
        Self { faces }
    }

    /// E-field PML correction after the standard E update.
    ///
    /// Per face/component: `ΔE = Cb·(s·((1/κ−1)·∂F/∂n) + s·ψ)`,
    /// i.e. the stretched derivative is `(1/κ)·∂F/∂n + ψ` with `ψ`
    /// entering UNSCALED; `ψ ← b·ψ + a·∂F/∂n`.
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
                    // dHz/dx as update_e pairs it with Ey(ie):
                    // (hz[ie] − hz[ie−1])/dx, 0 outside.
                    let hz_lo = if ie > 0 { g.hz[g.hz_idx(ie - 1, j, k)] } else { 0.0 };
                    let dhzdx = (g.hz[g.hz_idx(ie, j, k)] - hz_lo) / dx;
                    let pi = f.psi_idx(p, j, k);
                    let psi = be * f.psi_a[pi] + ae * dhzdx;
                    f.psi_a[pi] = psi;
                    let ei = g.ey_idx(ie, j, k);
                    g.ey[ei] += cb_y[ei] * (-omk * dhzdx + psi);

                    // Ez: dHy/dx as update_e pairs it with Ez(ie):
                    // (hy[ie] − hy[ie−1])/dx, 0 outside.
                    let hy_lo = if ie > 0 { g.hy[g.hy_idx(ie - 1, j, k)] } else { 0.0 };
                    let dhydx = (g.hy[g.hy_idx(ie, j, k)] - hy_lo) / dx;
                    let psi2 = be * f.psi_b[pi] + ae * dhydx;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ez_idx(ie, j, k);
                    g.ez[ei2] += cb_z[ei2] * (omk * dhydx - psi2);
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
                    // Ex: dHz/dy as update_e pairs it with Ex(je):
                    // (hz[je] − hz[je−1])/dy, 0 outside.
                    let hz_lo = if je > 0 { g.hz[g.hz_idx(i, je - 1, k)] } else { 0.0 };
                    let dhzdy = (g.hz[g.hz_idx(i, je, k)] - hz_lo) / dy;
                    let pi = f.psi_idx(p, i, k);
                    let psi = be * f.psi_a[pi] + ae * dhzdy;
                    f.psi_a[pi] = psi;
                    let ei = g.ex_idx(i, je, k);
                    g.ex[ei] += cb_x[ei] * (-omk * dhzdy + psi);

                    let hx_lo = if je > 0 { g.hx[g.hx_idx(i, je - 1, k)] } else { 0.0 };
                    let dhxdy = (g.hx[g.hx_idx(i, je, k)] - hx_lo) / dy;
                    let psi2 = be * f.psi_b[pi] + ae * dhxdy;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ez_idx(i, je, k);
                    g.ez[ei2] += cb_z[ei2] * (-omk * dhxdy + psi2);
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
                    // dHy/dz as update_e pairs it with Ex(ke):
                    // (hy[ke] − hy[ke−1])/dz, 0 outside.
                    let hy_lo = if ke > 0 { g.hy[g.hy_idx(i, j, ke - 1)] } else { 0.0 };
                    let dhydz = (g.hy[g.hy_idx(i, j, ke)] - hy_lo) / dz;
                    let pi = f.psi_idx(p, i, j);
                    let psi = be * f.psi_a[pi] + ae * dhydz;
                    f.psi_a[pi] = psi;
                    let ei = g.ex_idx(i, j, ke);
                    g.ex[ei] += cb_x[ei] * (-omk * dhydz + psi);

                    let hx_lo = if ke > 0 { g.hx[g.hx_idx(i, j, ke - 1)] } else { 0.0 };
                    let dhxdz = (g.hx[g.hx_idx(i, j, ke)] - hx_lo) / dz;
                    let psi2 = be * f.psi_b[pi] + ae * dhxdz;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ey_idx(i, j, ke);
                    // Ey: update has +∂Hx/∂z (s=+1): Δ = Cb·(−(1−1/κ)·d + ψ/κ).
                    g.ey[ei2] += cb_y[ei2] * (omk * dhxdz - psi2);
                }
            }
        }
    }

    /// H-field PML correction after the standard H update.
    ///
    /// Per face/component: `ΔH = −(dt/μ₀)·(−s·(1−1/κ)·∂F/∂n − s·ψ)`,
    /// where `s = ±1` is the sign of `∂F/∂n` in the bracket
    /// (`H −= c·[s·∂F/∂n + …]`) and `ψ` enters UNSCALED (the stretched
    /// derivative is `(1/κ)·∂F/∂n + ψ`); `ψ ← b·ψ + a·∂F/∂n`.
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
                    // dEz/dx as update_h pairs it with Hy(ih):
                    // (ez[ih+1] − ez[ih])/dx, out-of-range = 0 (mirrors
                    // update_h's guards: at ih=nx both samples are 0).
                    let e_hi = if ih + 1 < nx { g.ez[g.ez_idx(ih + 1, j, k)] } else { 0.0 };
                    let e_lo = if ih < nx { g.ez[g.ez_idx(ih, j, k)] } else { 0.0 };
                    // Hy: bracket has +∂Ez/∂x (s=+1): Δ = −c·(−(1−1/κ)·d + ψ/κ).
                    let dezdx = (e_hi - e_lo) / dx;
                    let pi = f.psi_idx(p, j, k);
                    let psi = bh * f.psi_c[pi] + ah * dezdx;
                    f.psi_c[pi] = psi;
                    let hi = g.hy_idx(ih, j, k);
                    g.hy[hi] -= c * (-omk * dezdx + psi);

                    // Hz: dEy/dx as update_h pairs it with Hz(ih):
                    // (ey[ih+1] − ey[min(ih,nx−1)])/dx, out-of-range = 0.
                    let e2_hi = if ih + 1 < nx { g.ey[g.ey_idx(ih + 1, j, k)] } else { 0.0 };
                    let e2_lo = g.ey[g.ey_idx(ih.min(nx - 1), j, k)];
                    let deydx = (e2_hi - e2_lo) / dx;
                    let psi2 = bh * f.psi_d[pi] + ah * deydx;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hz_idx(ih, j, k);
                    g.hz[hi2] -= c * (omk * deydx - psi2);
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
                    // dEz/dy as update_h pairs it with Hx(jh):
                    // (ez[jh+1] − ez[jh])/dy, out-of-range = 0.
                    let e_hi = if jh + 1 < ny { g.ez[g.ez_idx(i, jh + 1, k)] } else { 0.0 };
                    let e_lo = if jh < ny { g.ez[g.ez_idx(i, jh, k)] } else { 0.0 };
                    let dezdy = (e_hi - e_lo) / dy;
                    let pi = f.psi_idx(p, i, k);
                    let psi = bh * f.psi_c[pi] + ah * dezdy;
                    f.psi_c[pi] = psi;
                    let hi = g.hx_idx(i, jh, k);
                    g.hx[hi] -= c * (omk * dezdy - psi);

                    // dEx/dy as update_h pairs it with Hz(jh):
                    // (ex[jh+1] − ex[min(jh,ny−1)])/dy, out-of-range = 0.
                    let e2_hi = if jh + 1 < ny { g.ex[g.ex_idx(i, jh + 1, k)] } else { 0.0 };
                    let e2_lo = g.ex[g.ex_idx(i, jh.min(ny - 1), k)];
                    let dexdy = (e2_hi - e2_lo) / dy;
                    let psi2 = bh * f.psi_d[pi] + ah * dexdy;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hz_idx(i, jh, k);
                    // Hz: bracket has +∂Ex/∂y (s=+1): Δ = −c·(−(1−1/κ)·d + ψ/κ).
                    g.hz[hi2] -= c * (-omk * dexdy + psi2);
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
                    // dEy/dz as update_h pairs it with Hx(kh):
                    // (ey[kh+1] − ey[kh])/dz, out-of-range = 0.
                    let e_hi = if kh + 1 < nz { g.ey[g.ey_idx(i, j, kh + 1)] } else { 0.0 };
                    let e_lo = if kh < nz { g.ey[g.ey_idx(i, j, kh)] } else { 0.0 };
                    let deydz = (e_hi - e_lo) / dz;
                    let pi = f.psi_idx(p, i, j);
                    let psi = bh * f.psi_c[pi] + ah * deydz;
                    f.psi_c[pi] = psi;
                    let hi = g.hx_idx(i, j, kh);
                    // Hx: bracket has +∂Ey/∂z (s=+1): Δ = −c·(−(1−1/κ)·d + ψ/κ).
                    g.hx[hi] -= c * (-omk * deydz + psi);

                    // dEx/dz as update_h pairs it with Hy(kh):
                    // (ex[kh+1] − ex[kh])/dz, out-of-range = 0.
                    let e2_hi = if kh + 1 < nz { g.ex[g.ex_idx(i, j, kh + 1)] } else { 0.0 };
                    let e2_lo = if kh < nz { g.ex[g.ex_idx(i, j, kh)] } else { 0.0 };
                    let dexdz = (e2_hi - e2_lo) / dz;
                    let psi2 = bh * f.psi_d[pi] + ah * dexdz;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hy_idx(i, j, kh);
                    g.hy[hi2] -= c * (omk * dexdz - psi2);
                }
            }
        }
    }
}
