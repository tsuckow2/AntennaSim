//! Time-stepping: Yee E/H updates, CPML, PEC, resistive source, V/I sampling.
//!
//! The H and E updates are each fused into a single z-slab-parallel pass
//! (Hx/Hy/Hz together; Ex/Ey/Ez together) to stream the field arrays once.

use rayon::prelude::*;

use crate::cpml::Cpml;
use crate::grid::YeeGrid;
use crate::material::Materials;
use crate::source::{DftProbe, Feed, GaussianSource, ResistiveSource};

pub struct Fdtd {
    pub grid: YeeGrid,
    pub mat: Materials,
    pub cpml: Cpml,
    pub dt: f32,
    pub feed: Feed,
    pub source: GaussianSource,
    pub rsource: ResistiveSource,
    pub probe: DftProbe,
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
    ) -> Self {
        Self {
            grid,
            mat,
            cpml,
            dt,
            feed,
            source,
            rsource,
            probe,
            t: 0.0,
            step: 0,
        }
    }

    /// One full step: H^{n+½}, then E^{n+1}, then resistive source + DFT.
    ///
    /// Luebbers source sequencing: measure the tight-loop current I_s at
    /// the H-step, run the ordinary E update, enforce PEC, then overwrite
    /// the source cell with E_z = −V_s/dz + I_s·Rs/dz. Record the ACTUAL
    /// gap voltage V = −E_z·dz (never the prescribed V_s) with I_s.
    pub fn step(&mut self) {
        let dt = self.dt;
        update_h(&mut self.grid, dt);
        self.cpml.correct_h(&mut self.grid, dt);

        // Terminal current at the H-step (tight 1-cell loop, +z upward).
        let i_up = self.feed.measure_i_tight(&self.grid);

        update_e(&mut self.grid, &self.mat);
        self.cpml
            .correct_e(&mut self.grid, &self.mat.cb_x, &self.mat.cb_y, &self.mat.cb_z);
        apply_pec(&mut self.grid, &self.mat);

        // Resistive source at the gap cell (E-step time).
        self.t += dt;
        let vs = self.source.value(self.t);
        let (v_term, i_term) = self.rsource.apply(&mut self.grid, &self.feed, vs, i_up);
        self.probe.push(v_term, i_term);

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
pub fn update_h(g: &mut YeeGrid, dt: f32) {
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
                g.hx[hx_idx] -= c * ((ey_kp - ey_k) * idz - (ez_jp - ez_j) * idy);
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
                g.hy[hy_idx] -= c * ((ez_ip - ez_i) * idx - (ex_kp - ex_k) * idz);
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
                g.hz[hz_idx] -= c * ((ex_jp - ex_j) * idy - (ey_ip - ey_i) * idx);
            }
        }
    }
}

pub fn update_e(g: &mut YeeGrid, m: &Materials) {
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
                g.ex[ex_idx] = ca * g.ex[ex_idx] + cb * ((hy_k - hy_km) * idz - (hz_j - hz_jm) * idy);
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
                g.ey[ey_idx] = ca * g.ey[ey_idx] + cb * ((hz_i - hz_im) * idx - (hx_k - hx_km) * idz);
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
                g.ez[ez_idx] = ca * g.ez[ez_idx] + cb * ((hx_j - hx_jm) * idy - (hy_i - hy_im) * idx);
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
