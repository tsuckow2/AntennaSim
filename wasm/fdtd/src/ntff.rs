//! Near-to-far-field transform (NTFF) on a Huygens box.
//!
//! Tangential E and H on a closed box inside the CPML are accumulated
//! as running DFT phasors at each requested frequency — the same
//! kernel as `DftProbe` (Σ x·e^(−iωt), engineering e^(+iωt) phasor
//! convention, phasor = (2/N)·DFT for a real signal). E is sampled at
//! its own time level, H at t − dt/2 (after a full `Fdtd::step`, H
//! lags E by half a step), each with its own phase, so no time
//! interpolation is needed. Tangential components are colocated to
//! the face quadrature points by the standard Yee averaging (E:
//! 4-point, H: 2-point); component positions follow the update
//! stencils in solver.rs:
//!   Ex(i,j,k): (i, j+½, k+½)   Ey: (i+½, j, k+½)   Ez: (i+½, j+½, k)
//!   Hx(i,j,k): (i+½, j+1, k+1) Hy: (i+1, j+½, k+1) Hz: (i+1, j+1, k+½)
//! (cell units, grid origin excluded — the caller passes it).
//!
//! Far field (Taflove's L/N radiation-vector form) from the surface
//! equivalence currents Js = n̂×H, Ms = E×n̂:
//!   N(r̂) = ∮ Js e^(+ik r̂·r′) dS,   L(r̂) = ∮ Ms e^(+ik r̂·r′) dS
//!   E_θ = −(ik e^(−ikr)/4πr)(L_φ + η N_θ)
//!   E_φ = +(ik e^(−ikr)/4πr)(L_θ − η N_φ)
//! Directivity uses U = r²|E|²/(2η) integrated over the sphere; the
//! independent box-flux integral ½∮Re{E×H*}·n̂ dS must agree with
//! the sphere integral — `box_flux` exposes it as a self-check.

use crate::grid::YeeGrid;

pub const C0: f64 = 299_792_458.0;
pub const ETA0: f64 = 376.730_313_668;

#[derive(Clone, Copy, Default)]
struct C64 {
    re: f64,
    im: f64,
}
impl C64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Box planes: x faces on Ex planes i0/i1, y faces on Ey planes
/// j0/j1, z faces on Ez planes k0/k1. Quadrature covers the cells
/// [i0,i1) × [j0,j1) × [k0,k1) faces. Keep ≥1 cell inside the domain
/// and ≥2 cells from any CPML face.
#[derive(Clone, Copy)]
pub struct NtffBox {
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    pub k0: usize,
    pub k1: usize,
}

struct Sample {
    pos: [f64; 3],
    da: f64,
    n: [f64; 3],
    /// Signed Cartesian unit axes of the two tangential components.
    a: [f64; 3],
    b: [f64; 3],
    /// Sparse gathers: (field array index, weight). E comps are
    /// 4-point averages, H comps 2-point.
    e_a: [(usize, f64); 4],
    e_b: [(usize, f64); 4],
    h_a: [(usize, f64); 2],
    h_b: [(usize, f64); 2],
}

pub struct Ntff {
    freqs: Vec<f64>,
    dt: f64,
    steps: usize,
    samples: Vec<Sample>,
    /// Reused per-step gather buffer (was a fresh allocation per
    /// step — a measurable slice of the accumulation cost).
    scratch: Vec<[f64; 4]>,
    /// Face sample ranges in push order: [+z, −z, +x, −x, +y, −y].
    face_ranges: [(usize, usize); 6],
    /// acc[freq][sample] = [E_a, E_b, H_a, H_b] raw DFT sums.
    acc: Vec<Vec<[C64; 4]>>,
}

type G4 = [(usize, f64); 4];
type G2 = [(usize, f64); 2];
fn avg4(idxs: [usize; 4]) -> G4 {
    [
        (idxs[0], 0.25),
        (idxs[1], 0.25),
        (idxs[2], 0.25),
        (idxs[3], 0.25),
    ]
}
fn avg2(idxs: [usize; 2]) -> G2 {
    [(idxs[0], 0.5), (idxs[1], 0.5)]
}

impl Ntff {
    pub fn new(g: &YeeGrid, bx: NtffBox, origin: (f64, f64, f64), freqs: Vec<f64>, dt: f32) -> Self {
        assert!(bx.i0 >= 1 && bx.j0 >= 1 && bx.k0 >= 1, "box must be interior");
        assert!(bx.i1 < g.nx && bx.j1 < g.ny && bx.k1 < g.nz, "box must be interior");
        let (dx, dy, dz) = (g.dx as f64, g.dy as f64, g.dz as f64);
        let (ox, oy, oz) = origin;
        let mut samples: Vec<Sample> = Vec::new();
        let push = |samples: &mut Vec<Sample>,
                    pos: [f64; 3],
                    da: f64,
                    n: [f64; 3],
                    a: [f64; 3],
                    b: [f64; 3],
                    e_a: G4,
                    e_b: G4,
                    h_a: G2,
                    h_b: G2| {
            samples.push(Sample {
                pos,
                da,
                n,
                a,
                b,
                e_a,
                e_b,
                h_a,
                h_b,
            });
        };
        // ---- z faces (n = ±z), tangents a=+x (Ex), b=+y (Ey) ----
        for (kplane, ns) in [(bx.k1, 1.0f64), (bx.k0, -1.0f64)] {
            for j in bx.j0..bx.j1 {
                for i in bx.i0..bx.i1 {
                    let pos = [
                        ox + (i as f64 + 0.5) * dx,
                        oy + (j as f64 + 0.5) * dy,
                        oz + kplane as f64 * dz,
                    ];
                    let e_a = avg4([
                        g.ex_idx(i, j, kplane - 1),
                        g.ex_idx(i + 1, j, kplane - 1),
                        g.ex_idx(i, j, kplane),
                        g.ex_idx(i + 1, j, kplane),
                    ]);
                    let e_b = avg4([
                        g.ey_idx(i, j, kplane - 1),
                        g.ey_idx(i, j + 1, kplane - 1),
                        g.ey_idx(i, j, kplane),
                        g.ey_idx(i, j + 1, kplane),
                    ]);
                    let h_a = avg2([g.hx_idx(i, j, kplane - 1), g.hx_idx(i, j - 1, kplane - 1)]);
                    let h_b = avg2([g.hy_idx(i, j, kplane - 1), g.hy_idx(i - 1, j, kplane - 1)]);
                    push(
                        &mut samples,
                        pos,
                        dx * dy,
                        [0.0, 0.0, ns],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        e_a,
                        e_b,
                        h_a,
                        h_b,
                    );
                }
            }
        }
        // ---- x faces (n = ±x), tangents a=+y (Ey), b=+z (Ez) ----
        for (iplane, ns) in [(bx.i1, 1.0f64), (bx.i0, -1.0f64)] {
            for k in bx.k0..bx.k1 {
                for j in bx.j0..bx.j1 {
                    let pos = [
                        ox + iplane as f64 * dx,
                        oy + (j as f64 + 0.5) * dy,
                        oz + (k as f64 + 0.5) * dz,
                    ];
                    let e_a = avg4([
                        g.ey_idx(iplane - 1, j, k),
                        g.ey_idx(iplane, j, k),
                        g.ey_idx(iplane - 1, j + 1, k),
                        g.ey_idx(iplane, j + 1, k),
                    ]);
                    let e_b = avg4([
                        g.ez_idx(iplane - 1, j, k),
                        g.ez_idx(iplane, j, k),
                        g.ez_idx(iplane - 1, j, k + 1),
                        g.ez_idx(iplane, j, k + 1),
                    ]);
                    let h_a = avg2([g.hy_idx(iplane - 1, j, k), g.hy_idx(iplane - 1, j, k - 1)]);
                    let h_b = avg2([g.hz_idx(iplane - 1, j, k), g.hz_idx(iplane - 1, j - 1, k)]);
                    push(
                        &mut samples,
                        pos,
                        dy * dz,
                        [ns, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        e_a,
                        e_b,
                        h_a,
                        h_b,
                    );
                }
            }
        }
        // ---- y faces (n = ±y), tangents a=+z (Ez), b=+x (Ex) ----
        for (jplane, ns) in [(bx.j1, 1.0f64), (bx.j0, -1.0f64)] {
            for k in bx.k0..bx.k1 {
                for i in bx.i0..bx.i1 {
                    let pos = [
                        ox + (i as f64 + 0.5) * dx,
                        oy + jplane as f64 * dy,
                        oz + (k as f64 + 0.5) * dz,
                    ];
                    let e_a = avg4([
                        g.ez_idx(i, jplane - 1, k),
                        g.ez_idx(i, jplane, k),
                        g.ez_idx(i, jplane - 1, k + 1),
                        g.ez_idx(i, jplane, k + 1),
                    ]);
                    let e_b = avg4([
                        g.ex_idx(i, jplane - 1, k),
                        g.ex_idx(i + 1, jplane - 1, k),
                        g.ex_idx(i, jplane, k),
                        g.ex_idx(i + 1, jplane, k),
                    ]);
                    let h_a = avg2([g.hz_idx(i, jplane - 1, k), g.hz_idx(i - 1, jplane - 1, k)]);
                    let h_b = avg2([g.hx_idx(i, jplane - 1, k), g.hx_idx(i, jplane - 1, k - 1)]);
                    push(
                        &mut samples,
                        pos,
                        dx * dz,
                        [0.0, ns, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        e_a,
                        e_b,
                        h_a,
                        h_b,
                    );
                }
            }
        }
        let acc = (0..freqs.len())
            .map(|_| vec![[C64::default(); 4]; samples.len()])
            .collect();
        let nzf = (bx.i1 - bx.i0) * (bx.j1 - bx.j0);
        let nxf = (bx.j1 - bx.j0) * (bx.k1 - bx.k0);
        let nyf = (bx.i1 - bx.i0) * (bx.k1 - bx.k0);
        let face_ranges = [
            (0, nzf),
            (nzf, 2 * nzf),
            (2 * nzf, 2 * nzf + nxf),
            (2 * nzf + nxf, 2 * nzf + 2 * nxf),
            (2 * nzf + 2 * nxf, 2 * nzf + 2 * nxf + nyf),
            (2 * nzf + 2 * nxf + nyf, 2 * nzf + 2 * nxf + 2 * nyf),
        ];
        Self {
            freqs,
            dt: dt as f64,
            steps: 0,
            scratch: vec![[0.0; 4]; samples.len()],
            samples,
            face_ranges,
            acc,
        }
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Checkpoint export: (steps, accumulator re/im flattened in
    /// [freq][sample][comp] order). For long NTFF runs that must
    /// survive process/machine restarts.
    pub fn export_state(&self) -> (usize, Vec<f64>) {
        let mut out = Vec::with_capacity(self.acc.len() * self.samples.len() * 8);
        for f in &self.acc {
            for s in f {
                for c in s {
                    out.push(c.re);
                    out.push(c.im);
                }
            }
        }
        (self.steps, out)
    }

    /// Restore a state produced by `export_state` (same box/freqs).
    pub fn import_state(&mut self, steps: usize, data: &[f64]) {
        let mut it = data.iter();
        for f in &mut self.acc {
            for s in f.iter_mut() {
                for c in s.iter_mut() {
                    c.re = *it.next().expect("ntff state size");
                    c.im = *it.next().expect("ntff state size");
                }
            }
        }
        assert!(it.next().is_none(), "ntff state size mismatch");
        self.steps = steps;
    }

    /// Accumulate one step. Call AFTER `Fdtd::step` with `t_e` = the
    /// E-field time level just produced (steps_done · dt); H is
    /// phased at t_e − dt/2 automatically.
    pub fn accumulate(&mut self, g: &YeeGrid, t_e: f64) {
        self.steps += 1;
        // Gather the colocated tangential components once per step
        // into the reusable scratch buffer.
        for (si, s) in self.samples.iter().enumerate() {
            self.scratch[si] = [
                gather_e(g, s, 0),
                gather_e(g, s, 1),
                gather_h(g, s, 0),
                gather_h(g, s, 1),
            ];
        }
        let vals = &self.scratch;
        for (fi, &f) in self.freqs.iter().enumerate() {
            let w = 2.0 * std::f64::consts::PI * f;
            let pe = C64::new((w * t_e).cos(), -(w * t_e).sin());
            let th = t_e - 0.5 * self.dt;
            let ph = C64::new((w * th).cos(), -(w * th).sin());
            let acc = &mut self.acc[fi];
            for (si, v) in vals.iter().enumerate() {
                acc[si][0] = acc[si][0].add(pe.scale(v[0]));
                acc[si][1] = acc[si][1].add(pe.scale(v[1]));
                acc[si][2] = acc[si][2].add(ph.scale(v[2]));
                acc[si][3] = acc[si][3].add(ph.scale(v[3]));
            }
        }
    }

    fn phasors(&self, fi: usize, si: usize) -> [C64; 4] {
        let s = 2.0 / self.steps.max(1) as f64;
        let mut p = self.acc[fi][si].map(|c| c.scale(s));
        // SIGN: the solver stores h = −H_phys. Its update pair
        // (update_e with −∇×h, update_h with +∇×E) is Maxwell's
        // system in (E, −H); every other consumer (port loop,
        // energies) is sign-tuned or sign-blind to that convention.
        // NTFF is not: Js = n̂×H and S = E×H* need the physical H,
        // so negate the H phasors here. (Found 2026-10-16 via the
        // ntff_dipole unit validation: all-face negative flux at
        // −1.28× the sphere power on both the dipole and efs.)
        p[2] = p[2].scale(-1.0);
        p[3] = p[3].scale(-1.0);
        p
    }

    /// Diagnostic: normalized phasors [E_a, E_b, H_a, H_b] of one
    /// sample, as (re, im) pairs.
    pub fn sample_phasors(&self, fi: usize, si: usize) -> [(f64, f64); 4] {
        self.phasors(fi, si).map(|c| (c.re, c.im))
    }

    /// Diagnostic: the colocated tangential values of one sample
    /// from the current grid state, [E_a, E_b, H_a, H_b].
    pub fn gather_at(&self, g: &YeeGrid, si: usize) -> [f64; 4] {
        let s = &self.samples[si];
        [
            gather_e(g, s, 0),
            gather_e(g, s, 1),
            gather_h(g, s, 0),
            gather_h(g, s, 1),
        ]
    }

    /// Diagnostic: box flux split by face, in push order
    /// [+z, −z, +x, −x, +y, −y].
    pub fn box_flux_per_face(&self, fi: usize) -> [f64; 6] {
        let mut out = [0.0f64; 6];
        for (f, &(lo, hi)) in self.face_ranges.iter().enumerate() {
            for si in lo..hi {
                out[f] += self.flux_at(fi, si);
            }
        }
        out
    }

    fn flux_at(&self, fi: usize, si: usize) -> f64 {
        let s = &self.samples[si];
        let [ea, eb, ha, hb] = self.phasors(fi, si);
        let e = vec3(ea, s.a, eb, s.b);
        let h = vec3(ha, s.a, hb, s.b);
        let sx = e[1].mul(h[2].conj()).add(e[2].mul(h[1].conj()).scale(-1.0));
        let sy = e[2].mul(h[0].conj()).add(e[0].mul(h[2].conj()).scale(-1.0));
        let sz = e[0].mul(h[1].conj()).add(e[1].mul(h[0].conj()).scale(-1.0));
        0.5 * (sx.re * s.n[0] + sy.re * s.n[1] + sz.re * s.n[2]) * s.da
    }

    /// Radiated power (W) at frequency `fi` from the box Poynting
    /// flux of the DFT phasors. Must approximately equal the
    /// sphere-integrated power from `far_field` — the NTFF
    /// self-check.
    pub fn box_flux(&self, fi: usize) -> f64 {
        (0..self.samples.len()).map(|si| self.flux_at(fi, si)).sum()
    }

    /// Radiation intensity U(θ,φ) [W/sr] at frequency `fi` on a
    /// (n_theta × n_phi) grid; θ from +z, φ from +x toward +y.
    /// Returns (thetas, phis, U row-major [θ][φ]).
    pub fn far_field(&self, fi: usize, n_theta: usize, n_phi: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let f = self.freqs[fi];
        let k = 2.0 * std::f64::consts::PI * f / C0;
        // Per-sample Cartesian currents (Js, Ms) at this frequency.
        let mut js: Vec<[C64; 3]> = Vec::with_capacity(self.samples.len());
        let mut ms: Vec<[C64; 3]> = Vec::with_capacity(self.samples.len());
        for (si, s) in self.samples.iter().enumerate() {
            let [ea, eb, ha, hb] = self.phasors(fi, si);
            let e = vec3(ea, s.a, eb, s.b);
            let h = vec3(ha, s.a, hb, s.b);
            js.push(cross_fv(s.n, h));
            ms.push(cross_vf(e, s.n));
        }
        let thetas: Vec<f64> = (0..n_theta)
            .map(|i| std::f64::consts::PI * i as f64 / (n_theta - 1) as f64)
            .collect();
        let phis: Vec<f64> = (0..n_phi)
            .map(|i| 2.0 * std::f64::consts::PI * i as f64 / n_phi as f64)
            .collect();
        let mut u = vec![0.0f64; n_theta * n_phi];
        for (ti, &th) in thetas.iter().enumerate() {
            let (st, ct) = (th.sin(), th.cos());
            for (pi, &ph) in phis.iter().enumerate() {
                let (sp, cp) = (ph.sin(), ph.cos());
                let r = [st * cp, st * sp, ct];
                let th_hat = [ct * cp, ct * sp, -st];
                let ph_hat = [-sp, cp, 0.0];
                let mut nv = [C64::default(); 3];
                let mut lv = [C64::default(); 3];
                for (si, s) in self.samples.iter().enumerate() {
                    let phz = k * (r[0] * s.pos[0] + r[1] * s.pos[1] + r[2] * s.pos[2]);
                    let w = C64::new(phz.cos(), phz.sin()).scale(s.da);
                    for c in 0..3 {
                        nv[c] = nv[c].add(js[si][c].mul(w));
                        lv[c] = lv[c].add(ms[si][c].mul(w));
                    }
                }
                let n_th = dot(nv, th_hat);
                let n_ph = dot(nv, ph_hat);
                let l_th = dot(lv, th_hat);
                let l_ph = dot(lv, ph_hat);
                // E at r = 1 m; the e^(−ikr) phase drops out of |E|².
                let e_th = l_ph.add(n_th.scale(ETA0)).scale(k / (4.0 * std::f64::consts::PI));
                let e_ph = l_th.add(n_ph.scale(-ETA0)).scale(k / (4.0 * std::f64::consts::PI));
                u[ti * n_phi + pi] = (e_th.abs2() + e_ph.abs2()) / (2.0 * ETA0);
            }
        }
        (thetas, phis, u)
    }

    /// (Dmax linear, Dmax dBi, Prad W, θ at max, φ at max).
    pub fn directivity(&self, fi: usize, n_theta: usize, n_phi: usize) -> (f64, f64, f64, f64, f64) {
        let (thetas, phis, u) = self.far_field(fi, n_theta, n_phi);
        let mut prad = 0.0f64;
        let mut umax = 0.0f64;
        let mut at = (0usize, 0usize);
        for ti in 0..thetas.len() {
            // Trapezoid weights in θ (endpoints half), φ periodic.
            let wt = if ti == 0 || ti == thetas.len() - 1 { 0.5 } else { 1.0 };
            let dth = std::f64::consts::PI / (thetas.len() - 1) as f64;
            let dph = 2.0 * std::f64::consts::PI / phis.len() as f64;
            for pi in 0..phis.len() {
                let uu = u[ti * phis.len() + pi];
                prad += uu * thetas[ti].sin() * wt * dth * dph;
                if uu > umax {
                    umax = uu;
                    at = (ti, pi);
                }
            }
        }
        let d = 4.0 * std::f64::consts::PI * umax / prad;
        (d, 10.0 * d.log10(), prad, thetas[at.0], phis[at.1])
    }
}

fn vec3(x1: C64, ax1: [f64; 3], x2: C64, ax2: [f64; 3]) -> [C64; 3] {
    [
        x1.scale(ax1[0]).add(x2.scale(ax2[0])),
        x1.scale(ax1[1]).add(x2.scale(ax2[1])),
        x1.scale(ax1[2]).add(x2.scale(ax2[2])),
    ]
}
/// a × v for a real vector a and complex vector v.
fn cross_fv(a: [f64; 3], v: [C64; 3]) -> [C64; 3] {
    [
        v[2].scale(a[1]).add(v[1].scale(-a[2])),
        v[0].scale(a[2]).add(v[2].scale(-a[0])),
        v[1].scale(a[0]).add(v[0].scale(-a[1])),
    ]
}
/// v × a for complex vector v and real vector a.
fn cross_vf(v: [C64; 3], a: [f64; 3]) -> [C64; 3] {
    [
        v[1].scale(a[2]).add(v[2].scale(-a[1])),
        v[2].scale(a[0]).add(v[0].scale(-a[2])),
        v[0].scale(a[1]).add(v[1].scale(-a[0])),
    ]
}
fn dot(v: [C64; 3], a: [f64; 3]) -> C64 {
    v[0].scale(a[0]).add(v[1].scale(a[1])).add(v[2].scale(a[2]))
}

fn gather_e(g: &YeeGrid, s: &Sample, which: usize) -> f64 {
    let (gr, axis) = if which == 0 {
        (&s.e_a, s.a)
    } else {
        (&s.e_b, s.b)
    };
    // All tangent axes are positive Cartesian axes by construction.
    let arr = if axis[0] != 0.0 {
        &g.ex
    } else if axis[1] != 0.0 {
        &g.ey
    } else {
        &g.ez
    };
    gr.iter().map(|&(ix, w)| arr[ix] as f64 * w).sum::<f64>()
}
fn gather_h(g: &YeeGrid, s: &Sample, which: usize) -> f64 {
    let (gr, axis) = if which == 0 {
        (&s.h_a, s.a)
    } else {
        (&s.h_b, s.b)
    };
    let arr = if axis[0] != 0.0 {
        &g.hx
    } else if axis[1] != 0.0 {
        &g.hy
    } else {
        &g.hz
    };
    gr.iter().map(|&(ix, w)| arr[ix] as f64 * w).sum::<f64>()
}
