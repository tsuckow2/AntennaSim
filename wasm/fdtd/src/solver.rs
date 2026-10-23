//! Time-stepping: Yee E/H updates, CPML, PEC, resistive source, V/I sampling.
//!
//! The H and E updates are each fused into a single z-slab-parallel pass
//! (Hx/Hy/Hz together; Ex/Ey/Ez together) to stream the field arrays once.

use rayon::prelude::*;

use crate::cpml::{ade_terms, AdeCpml, Cpml};
use crate::grid::YeeGrid;
use crate::material::Materials;
use crate::source::{DftProbe, Feed, GaussianSource, ResistiveSource};
use crate::thin_wire::ThinWire;

pub struct Fdtd {
    pub grid: YeeGrid,
    pub mat: Materials,
    pub cpml: Cpml,
    pub dt: f32,
    pub feed: Feed,
    pub source: GaussianSource,
    pub rsource: ResistiveSource,
    pub probe: DftProbe,
    pub thin_wire: Option<ThinWire>,
    /// Number of Ez cells in the lumped-port column (no-wire path).
    /// 1 = single gap cell at the feed point (probe-pin models put PEC
    /// cells above it). >1 distributes the Norton port over the column
    /// src_k .. src_k+span — the openEMS-style lumped port, where the
    /// port voltage is the line integral over the whole column.
    pub port_span: usize,
    t: f32,
    step: usize,
}

impl Fdtd {
    pub fn new(
        grid: YeeGrid,
        mat: Materials,
        cpml: Cpml,
        dt: f32,
        feed: Feed,
        source: GaussianSource,
        rsource: ResistiveSource,
        probe: DftProbe,
        thin_wire: Option<ThinWire>,
    ) -> Self {
        // The wire's base node carries the Thévenin source (see
        // ThinWire::update_current): wire its resistance in here so
        // callers can't forget.
        let mut thin_wire = thin_wire;
        if let Some(tw) = thin_wire.as_mut() {
            tw.feed_rs = Some(rsource.rs);
        }
        Self {
            grid,
            mat,
            cpml,
            dt,
            feed,
            source,
            rsource,
            probe,
            thin_wire,
            port_span: 1,
            t: 0.0,
            step: 0,
        }
    }

    /// One full step: H^{n+½}, wire current, E^{n+1}, wire deposition, source + DFT.
    ///
    /// With the Holland/Simpson thin-wire model, the terminal current is the
    /// wire's bottom-segment current (not the H-field Ampère loop).
    ///
    /// Leapfrog sequence:
    /// 1. H^{n+1/2} from E^n (update_h)
    /// 2. I^{n+1/2} from E^n and q^n (wire.update_current)
    /// 3. E^{n+1} = E^n + dt/eps*curl(H^{n+1/2}) (update_e)
    /// 4. E^{n+1} -= dt*I^{n+1/2}/(eps*dx*dy) (wire.deposit_current)
    /// 5. q^{n+1} from I^{n+1/2} (done in wire.update_current)
    ///
    /// Source sequencing: with a thin wire, the Thévenin source lives
    /// on the wire's base node (set before `update_current`); the gap
    /// cell's field is never overwritten, so the deposit stands. The
    /// probe records the wire's terminal pair (V_base, I_bottom).
    /// Without a wire, the lumped port uses the Norton form below.
    pub fn step(&mut self) {
        let dt = self.dt;
        // Phase 3a (env CPML_FORM=ade): the ADE branch folds the
        // CPML stretching into the update loops and skips the
        // face-local correction passes. The correction-form
        // branch is the production default and is unchanged.
        if self.cpml.ade_enabled() {
            let ade = self.cpml.ade_mut().expect("ade form without state");
            update_h_ade(&mut self.grid, dt, ade);
        } else {
            update_h(&mut self.grid, dt);
            self.cpml.correct_h(&mut self.grid, dt);
        }

        // Source voltage at the E-step time level (used by both paths).
        let vs_next = self.source.value(self.t + dt);

        // Thin-wire: advance current from E^n (does not modify grid yet).
        let (i_wire, v_wire) = if let Some(ref mut tw) = self.thin_wire {
            tw.feed_vs = vs_next;
            tw.update_current(&self.grid, dt)
        } else {
            // No wire: fall back to the H-field Ampère loop.
            (self.feed.measure_i_tight(&self.grid), 0.0)
        };

        if self.cpml.ade_enabled() {
            let ade = self.cpml.ade_mut().expect("ade form without state");
            update_e_ade(&mut self.grid, &self.mat, ade);
        } else {
            update_e(&mut self.grid, &self.mat);
            self.cpml
                .correct_e(&mut self.grid, &self.mat.cb_x, &self.mat.cb_y, &self.mat.cb_z);
        }
        // Tangential-damping prototype (no-op unless
        // CPML_TANGENTIAL_RATIO is set; see cpml.rs).
        self.cpml.damp_tangential(&mut self.grid, dt);

        // Thin-wire: deposit I^{n+1/2} into E^{n+1} as a current source.
        if let Some(ref tw) = self.thin_wire {
            tw.deposit_current(&mut self.grid, &self.mat);
        }

        apply_pec(&mut self.grid, &self.mat);

        // Source bookkeeping at the E-step time level.
        self.t += dt;
        let vs = vs_next;
        if self.thin_wire.is_some() {
            // Wire path: the source was applied on the wire's base
            // node inside update_current; record its terminal pair.
            self.probe.push(v_wire, i_wire);
        } else {
            // Lumped port: Norton impressed-current form. (The
            // Luebbers imposed-field/overwrite form is unstable on
            // thin gap cells — the Ampère-loop current contains the
            // cell's own displacement current, whose feedback through
            // Rs has gain ω·C·Rs ≫ 1 at grid frequencies; an
            // imposed-field variant with the displacement current
            // de-embedded also blew up in the efs fixture v8 test,
            // so the overwrite family is closed. See source.rs.)
            // The port may span several Ez cells in the column (openEMS-
            // style): the port voltage is the line integral over the
            // span, and the same impressed current density is deposited
            // in each cell (series cells share the port current).
            let (si, sj, sk) = (self.feed.src_i, self.feed.src_j, self.feed.src_k);
            let dz = self.grid.dz;
            let span = self.port_span.max(1);
            let v_port = -(0..span)
                .map(|m| self.grid.ez[self.grid.ez_idx(si, sj, sk + m)] * dz)
                .sum::<f32>();
            // Semi-implicit deposit: the impressed current is solved
            // against the port voltage it produces, not the pre-deposit
            // voltage. The deposit itself changes the port voltage by
            // i_src * G, where G = dz * sum(cb_z) / (dx*dy) is the
            // deposit's own impedance (dt/C_span), so the exact
            // terminal relation is i_src = (vs - v_port) / (Rs + G).
            // The explicit form divides by Rs alone: its feedback gain
            // G/Rs is O(1) for a thin vacuum gap cell (0.93 at CFL 0.2,
            // 2.3 at CFL 0.5 for the Snyder feed), which rings the gap
            // cell against its PEC sandwich — the late-time instability
            // (growth x1.07/step at CFL 0.5, x1.0006/step at CFL 0.2).
            // With (Rs + G) the local loop gain is Rs/(Rs+G) < 1 at any
            // CFL and the recorded (V_term, i_src) pair satisfies
            // V_term = Vs - i_src * Rs exactly.
            let g_dep: f32 = (0..span)
                .map(|m| self.mat.cb_z[self.grid.ez_idx(si, sj, sk + m)])
                .sum::<f32>()
                * dz
                / (self.grid.dx * self.grid.dy);
            let i_src = (vs - v_port) / (self.rsource.rs + g_dep);
            let mut v_term = 0.0f32;
            for m in 0..span {
                let idx = self.grid.ez_idx(si, sj, sk + m);
                let cb_src = self.mat.cb_z[idx];
                let eps = dt / cb_src;
                self.grid.ez[idx] -= dt * i_src / (eps * self.grid.dx * self.grid.dy);
                v_term -= self.grid.ez[idx] * dz;
            }
            // Record the DELIVERED current i_src, not the Ampère loop:
            // the loop around a cell containing the impressed source
            // measures i_src + the cell's displacement current, whose
            // deposit-driven part anti-correlates with i_src (measured
            // corr = -0.999, ratio -0.77, and int(V*I_loop)dt < 0 —
            // nonphysical for a passive load), while (v_term, i_src)
            // gives int(V*I)dt > 0 and a physical Zin sign structure.
            self.probe.push(v_term, i_src);
        }

        self.step += 1;
    }

    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    pub fn steps_done(&self) -> usize {
        self.step
    }

    /// Restore the step clock after a checkpoint resume: sets the
    /// step counter and t = step·dt. (The live clock accumulates
    /// t += dt in f32; the difference is far below any timescale
    /// the source or DFT cares about after the excitation pulse.)
    pub fn restore_clock(&mut self, step: usize) {
        self.step = step;
        self.t = step as f32 * self.dt;
    }

    pub fn sim_time(&self) -> f32 {
        self.t
    }
}

/// Fused H update: Hx, Hy, Hz in one z-parallel pass.
///
/// - Hx(i,j,k) −= c·[(Ey(k)−Ey(k−1))/dz − (Ez(j)−Ez(j−1))/dy]
/// - Hy(i,j,k) −= c·[(Ez(i)−Ez(i−1))/dx − (Ex(k)−Ex(k−1))/dz]
/// - Hz(i,j,k) −= c·[(Ex(j)−Ex(j−1))/dy − (Ey(i)−Ey(i−1))/dx]
/// Out-of-bounds E = 0.
///
/// Standard Yee update (no thin-wire H correction; the Holland/Simpson wire
/// couples via the E-field, not via modified H updates).
pub fn update_h(g: &mut YeeGrid, dt: f32) {
    update_h_impl::<false>(g, dt, &mut None);
}

/// Phase-3a ADE form of `update_h` (env CPML_FORM=ade): the CPML
/// stretching is folded into the update — each difference inside
/// a shell is replaced by (1/κ)·d + ψ with ψ advanced at the same
/// site (see `AdeCpml`). Algebraically the correction form's
/// operator; used only on the ADE branch of `Fdtd::step`.
pub fn update_h_ade(g: &mut YeeGrid, dt: f32, ade: &mut AdeCpml) {
    update_h_impl::<true>(g, dt, &mut Some(ade));
}

/// Phase-3a ADE hook for the update loops: with ADE off this is
/// the identity and the const-generic branch folds away (the
/// production update is unchanged); with ADE on it advances the
/// term's ψ at the addressed shell point and returns the
/// stretched derivative (1/κ)·d + ψ. `m` is the component index
/// along the derivative axis, `trans` the component's transverse
/// flat index (see `AdeTerm`'s slot layout).
#[inline(always)]
fn stretch<const ADE: bool>(
    ade: &mut Option<&mut AdeCpml>,
    term: usize,
    m: usize,
    trans: usize,
    d: f32,
) -> f32 {
    if ADE {
        ade.as_deref_mut()
            .expect("ADE update without ADE state")
            .stretch(term, m, trans, d)
    } else {
        d
    }
}

fn update_h_impl<const ADE: bool>(g: &mut YeeGrid, dt: f32, ade: &mut Option<&mut AdeCpml>) {
    let mu0 = 4.0e-7 * std::f32::consts::PI;
    let c = dt / mu0;
    let (nx, ny, nz) = (g.nx, g.ny, g.nz);
    let idx = 1.0 / g.dx;
    let idy = 1.0 / g.dy;
    let idz = 1.0 / g.dz;

    // Hx: nx * (ny+1) * (nz+1)
    // For each (i,j,k), compute curl E
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..nx {
                // Ey at (i, j, k) and (i, j, k+1)
                // Ey dimensions: nx * (ny+1) * nz, so k must be < nz
                let ey_k = if k < nz {
                    g.ey[g.ey_idx(i, j.min(ny), k)]
                } else {
                    0.0
                };
                let ey_kp = if k + 1 < nz {
                    g.ey[g.ey_idx(i, j.min(ny), k + 1)]
                } else {
                    0.0
                };
                // Ez at (i, j, k) and (i, j+1, k)
                // Ez dimensions: nx * ny * (nz+1), so j must be < ny, k <= nz
                let ez_j = if j < ny && k <= nz {
                    g.ez[g.ez_idx(i, j, k.min(nz))]
                } else {
                    0.0
                };
                let ez_jp = if j + 1 < ny && k <= nz {
                    g.ez[g.ez_idx(i, j + 1, k.min(nz))]
                } else {
                    0.0
                };
                let hx_idx = g.hx_idx(i, j, k);
                let d1 = stretch::<ADE>(ade, ade_terms::HX_Z, k, j * nx + i, (ey_kp - ey_k) * idz);
                let d2 = stretch::<ADE>(ade, ade_terms::HX_Y, j, k * nx + i, (ez_jp - ez_j) * idy);
                g.hx[hx_idx] -= c * (d1 - d2);
            }
        }
    }

    // Hy: (nx+1) * ny * (nz+1)
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..=nx {
                // Ez at (i, j, k) and (i+1, j, k)
                let ez_i = if i < nx && k <= nz {
                    g.ez[g.ez_idx(i.min(nx - 1), j, k.min(nz))]
                } else {
                    0.0
                };
                let ez_ip = if i + 1 < nx && k <= nz {
                    g.ez[g.ez_idx(i + 1, j, k.min(nz))]
                } else {
                    0.0
                };
                // Ex at (i, j, k) and (i, j, k+1)
                // Ex dimensions: (nx+1) * ny * nz, so k < nz
                let ex_k = if k < nz {
                    g.ex[g.ex_idx(i.min(nx), j, k)]
                } else {
                    0.0
                };
                let ex_kp = if k + 1 < nz {
                    g.ex[g.ex_idx(i.min(nx), j, k + 1)]
                } else {
                    0.0
                };
                let hy_idx = g.hy_idx(i, j, k);
                let d1 = stretch::<ADE>(ade, ade_terms::HY_X, i, k * ny + j, (ez_ip - ez_i) * idx);
                let d2 = stretch::<ADE>(ade, ade_terms::HY_Z, k, j * (nx + 1) + i, (ex_kp - ex_k) * idz);
                g.hy[hy_idx] -= c * (d1 - d2);
            }
        }
    }

    // Hz: (nx+1) * (ny+1) * nz
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..=nx {
                // Ex at (i, j, k) and (i, j+1, k)
                let ex_j = g.ex[g.ex_idx(i.min(nx), j.min(ny - 1), k)];
                let ex_jp = if j + 1 < ny {
                    g.ex[g.ex_idx(i.min(nx), j + 1, k)]
                } else {
                    0.0
                };
                // Ey at (i, j, k) and (i+1, j, k)
                let ey_i = g.ey[g.ey_idx(i.min(nx - 1), j.min(ny), k)];
                let ey_ip = if i + 1 < nx {
                    g.ey[g.ey_idx(i + 1, j.min(ny), k)]
                } else {
                    0.0
                };
                let hz_idx = g.hz_idx(i, j, k);
                let d1 = stretch::<ADE>(ade, ade_terms::HZ_Y, j, k * (nx + 1) + i, (ex_jp - ex_j) * idy);
                let d2 = stretch::<ADE>(ade, ade_terms::HZ_X, i, k * (ny + 1) + j, (ey_ip - ey_i) * idx);
                g.hz[hz_idx] -= c * (d1 - d2);
            }
        }
    }
}

pub fn update_e(g: &mut YeeGrid, m: &Materials) {
    update_e_impl::<false>(g, m, &mut None);
}

/// Phase-3a ADE form of `update_e` (env CPML_FORM=ade); see
/// `update_h_ade`.
pub fn update_e_ade(g: &mut YeeGrid, m: &Materials, ade: &mut AdeCpml) {
    update_e_impl::<true>(g, m, &mut Some(ade));
}

fn update_e_impl<const ADE: bool>(g: &mut YeeGrid, m: &Materials, ade: &mut Option<&mut AdeCpml>) {
    let (nx, ny, nz) = (g.nx, g.ny, g.nz);
    let idx = 1.0 / g.dx;
    let idy = 1.0 / g.dy;
    let idz = 1.0 / g.dz;

    // Ex: (nx+1) * ny * nz
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                // Hz at (i, j, k) and (i, j-1, k)
                // Hz dimensions: (nx+1)*(ny+1)*nz
                let hz_j = g.hz[g.hz_idx(i, j, k)];
                let hz_jm = if j > 0 {
                    g.hz[g.hz_idx(i, j - 1, k)]
                } else {
                    0.0
                };
                // Hy at (i, j, k) and (i, j, k-1)
                // Hy dimensions: (nx+1)*ny*(nz+1)
                let hy_k = g.hy[g.hy_idx(i, j, k)];
                let hy_km = if k > 0 {
                    g.hy[g.hy_idx(i, j, k - 1)]
                } else {
                    0.0
                };
                let ex_idx = g.ex_idx(i, j, k);
                let ca = m.ca_x[ex_idx];
                let cb = m.cb_x[ex_idx];
                // (∇×H)_x = ∂Hy/∂z - ∂Hz/∂y
                let d1 = stretch::<ADE>(ade, ade_terms::EX_Z, k, j * (nx + 1) + i, (hy_k - hy_km) * idz);
                let d2 = stretch::<ADE>(ade, ade_terms::EX_Y, j, k * (nx + 1) + i, (hz_j - hz_jm) * idy);
                g.ex[ex_idx] = ca * g.ex[ex_idx] + cb * (d1 - d2);
            }
        }
    }

    // Ey: nx * (ny+1) * nz
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                // Hx at (i, j, k) and (i, j, k-1)
                let hx_k = g.hx[g.hx_idx(i, j, k)];
                let hx_km = if k > 0 {
                    g.hx[g.hx_idx(i, j, k - 1)]
                } else {
                    0.0
                };
                // Hz at (i, j, k) and (i-1, j, k)
                let hz_i = g.hz[g.hz_idx(i, j, k)];
                let hz_im = if i > 0 {
                    g.hz[g.hz_idx(i - 1, j, k)]
                } else {
                    0.0
                };
                let ey_idx = g.ey_idx(i, j, k);
                let ca = m.ca_y[ey_idx];
                let cb = m.cb_y[ey_idx];
                // (∇×H)_y = ∂Hz/∂x - ∂Hx/∂z
                let d1 = stretch::<ADE>(ade, ade_terms::EY_X, i, k * (ny + 1) + j, (hz_i - hz_im) * idx);
                let d2 = stretch::<ADE>(ade, ade_terms::EY_Z, k, j * nx + i, (hx_k - hx_km) * idz);
                g.ey[ey_idx] = ca * g.ey[ey_idx] + cb * (d1 - d2);
            }
        }
    }

    // Ez: nx * ny * (nz+1)
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..nx {
                // Hy at (i, j, k) and (i-1, j, k)
                let hy_i = g.hy[g.hy_idx(i, j, k)];
                let hy_im = if i > 0 {
                    g.hy[g.hy_idx(i - 1, j, k)]
                } else {
                    0.0
                };
                // Hx at (i, j, k) and (i, j-1, k)
                let hx_j = g.hx[g.hx_idx(i, j, k)];
                let hx_jm = if j > 0 {
                    g.hx[g.hx_idx(i, j - 1, k)]
                } else {
                    0.0
                };
                let ez_idx = g.ez_idx(i, j, k);
                let ca = m.ca_z[ez_idx];
                let cb = m.cb_z[ez_idx];
                // (∇×H)_z = ∂Hx/∂y - ∂Hy/∂x
                let d1 = stretch::<ADE>(ade, ade_terms::EZ_Y, j, k * nx + i, (hx_j - hx_jm) * idy);
                let d2 = stretch::<ADE>(ade, ade_terms::EZ_X, i, k * nx + j, (hy_i - hy_im) * idx);
                g.ez[ez_idx] = ca * g.ez[ez_idx] + cb * (d1 - d2);
            }
        }
    }
}

pub fn apply_pec(g: &mut YeeGrid, m: &Materials) {
    g.ex
        .par_iter_mut()
        .zip(m.pec_x.par_iter())
        .for_each(|(e, &p)| {
            if p != 0 {
                *e = 0.0;
            }
        });
    g.ey
        .par_iter_mut()
        .zip(m.pec_y.par_iter())
        .for_each(|(e, &p)| {
            if p != 0 {
                *e = 0.0;
            }
        });
    g.ez
        .par_iter_mut()
        .zip(m.pec_z.par_iter())
        .for_each(|(e, &p)| {
            if p != 0 {
                *e = 0.0;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_cell_h_update() {
        let mut grid = YeeGrid::new(2, 2, 2, 1.0, 1.0, 1.0);
        let dt = 1e-9;
        let mu0 = 4.0e-7 * std::f32::consts::PI;
        let c = dt / mu0;
        // Ez=1 at (0,0,0)
        let ez_idx = grid.ez_idx(0, 0, 0);
        grid.ez[ez_idx] = 1.0;
        update_h(&mut grid, dt);
        // Hx(0,0,0): -(Ey(1)-Ey(0))/dz + (Ez(1)-Ez(0))/dy
        // Ey=0, Ez(0,1,0)=0, Ez(0,0,0)=1 → -c*(0 - (0-1)) = -c
        let hx = grid.hx[grid.hx_idx(0, 0, 0)];
        assert!((hx - (-c)).abs() < 1e-10, "Hx={}, expected {}", hx, -c);
        // Hy(0,0,0): (Ez(1)-Ez(0))/dx - (Ex(1)-Ex(0))/dz
        // Ez(1,0,0)=0, Ez(0,0,0)=1 → c*((0-1)) = -c? Wait...
        // Hy -= c*((Ez(i+1)-Ez(i))/dx - ...)
        // Ez(1,0,0)=0, Ez(0,0,0)=1 → (0-1)/1 = -1 → Hy -= c*(-1) = +c
        let hy = grid.hy[grid.hy_idx(0, 0, 0)];
        assert!((hy - c).abs() < 1e-10, "Hy={}, expected {}", hy, c);
    }

    #[test]
    fn free_propagation_stable() {
        let (nx, ny, nz) = (20, 20, 20);
        let mut grid = YeeGrid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3);
        let dt = grid.courant_dt(0.5);
        let mat = Materials::vacuum(nx, ny, nz, dt);
        // Gaussian pulse
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let r2 = (i as f32 - 10.0).powi(2)
                        + (j as f32 - 10.0).powi(2)
                        + (k as f32 - 10.0).powi(2);
                    let val = (-r2 / 8.0).exp();
                    let idx = grid.ez_idx(i, j, k);
                    grid.ez[idx] = val;
                }
            }
        }
        let mut max_e = 0.0f32;
        for _ in 0..200 {
            update_h(&mut grid, dt);
            update_e(&mut grid, &mat);
            let m = grid
                .ez
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            max_e = max_e.max(m);
            assert!(!m.is_nan(), "NaN detected");
            assert!(m < 10.0, "Unstable: max|Ez|={}", m);
        }
        println!("Stable: max|Ez| over 200 steps = {}", max_e);
    }
}
