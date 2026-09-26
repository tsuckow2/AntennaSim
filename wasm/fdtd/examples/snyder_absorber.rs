//! Full Snyder FDTD run with conductive absorber (no CPML).
//! Uses the corrected Yee solver.

use fdtd::geometry::build_snyder;
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::{update_e, update_h, apply_pec};
use fdtd::source::{DftProbe, GaussianSource};
use std::time::Instant;

fn add_absorber(mat: &mut Materials, grid: &YeeGrid, dt: f32, nabs: usize, sigma_max: f32) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let mut sigma = 0.0f32;
                // x boundaries
                let dx_lo = i.min(nabs) as f32 / nabs as f32;
                let dx_hi = (nx - 1 - i).min(nabs) as f32 / nabs as f32;
                let dx_min = dx_lo.min(dx_hi);
                if dx_min < 1.0 {
                    sigma = sigma_max * (1.0 - dx_min).powi(3);
                }
                // y boundaries
                let dy_lo = j.min(nabs) as f32 / nabs as f32;
                let dy_hi = (ny - 1 - j).min(nabs) as f32 / nabs as f32;
                let dy_min = dy_lo.min(dy_hi);
                if dy_min < 1.0 {
                    sigma = sigma.max(sigma_max * (1.0 - dy_min).powi(3));
                }
                // z top boundary (bottom is ground)
                if k >= nz - nabs {
                    let dz_hi = (nz - 1 - k) as f32 / nabs as f32;
                    sigma = sigma.max(sigma_max * (1.0 - dz_hi).powi(3));
                }
                
                if sigma > 1e-6 {
                    // Apply to Ex, Ey, Ez at this location
                    // Ex(i,j,k): i=0..nx, j=0..ny-1, k=0..nz-1
                    if i <= nx && j < ny && k < nz {
                        let idx = grid.ex_idx(i, j, k);
                        if idx < mat.ca_x.len() {
                            mat.set_dielectric(0, idx, 1.0, sigma, dt);
                        }
                    }
                    // Ey(i,j,k): i=0..nx-1, j=0..ny, k=0..nz-1
                    if i < nx && j <= ny && k < nz {
                        let idx = grid.ey_idx(i, j, k);
                        if idx < mat.ca_y.len() {
                            mat.set_dielectric(1, idx, 1.0, sigma, dt);
                        }
                    }
                    // Ez(i,j,k): i=0..nx-1, j=0..ny-1, k=0..nz
                    if i < nx && j < ny && k <= nz {
                        let idx = grid.ez_idx(i, j, k);
                        if idx < mat.ca_z.len() {
                            mat.set_dielectric(2, idx, 1.0, sigma, dt);
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    const NX: usize = 232;
    const NY: usize = 232;
    const NZ: usize = 61;
    const DX: f32 = 0.5e-3;
    const DY: f32 = 0.5e-3;
    const DZ: f32 = 0.525e-3;
    
    println!("Snyder FDTD with corrected Yee + conductive absorber");
    println!("Grid: {}x{}x{}, dx={}mm", NX, NY, NZ, DX*1e3);
    
    let mut grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let dt = grid.courant_dt(0.99);
    println!("dt = {:.3e} s", dt);
    
    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let feed = build_snyder(&grid, &mut mat, dt);
    println!("Source at ({}, {}, {})", feed.src_i, feed.src_j, feed.src_k);
    
    // Add absorber (replaces CPML)
    add_absorber(&mut mat, &grid, dt, 8, 15.0);
    println!("Conductive absorber added (8 cells, sigma_max=15 S/m)");
    
    let source = GaussianSource::new(2.5e9, 1.0);
    let freqs: Vec<f64> = (0..101).map(|i| 2.0e9 + i as f64 * 10e6).collect();
    let mut probe = DftProbe::new(freqs.clone(), dt);
    
    // Simulation parameters
    let t_max = 17e-9; // 17 ns full run
    let n_steps = (t_max / dt) as usize;
    println!("Running {} steps ({:.1} ns)...", n_steps, t_max*1e9);
    
    let start = Instant::now();
    let mut t = 0.0f32;
    let mut i_prev = 0.0f32;
    
    for step in 0..n_steps {
        update_h(&mut grid, dt);
        
        // Measure current at H-step
        let i_now = feed.measure_i(&grid);
        let i_avg = 0.5 * (i_prev + i_now);
        i_prev = i_now;
        
        update_e(&mut grid, &mat);
        apply_pec(&mut grid, &mat);
        
        // Hard source
        t += dt;
        let vs = source.value(t);
        let si = grid.ez_idx(feed.src_i, feed.src_j, feed.src_k);
        grid.ez[si] = -vs / grid.dz;
        // The Ampère loop measures downward (-z) gap current; the antenna
        // feed current (into the probe) is upward (+z), so negate.
        probe.push(vs, -i_avg);
        
        if step % 2000 == 0 {
            let max_e = grid.ez.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let elapsed = start.elapsed().as_secs_f32();
            println!("step {:5}/{}, t={:.2}ns, max|Ez|={:.2e}, elapsed={:.1}s", 
                     step, n_steps, t*1e9, max_e, elapsed);
            if !max_e.is_finite() {
                println!("UNSTABLE!");
                return;
            }
        }
    }
    
    let elapsed = start.elapsed().as_secs_f32();
    println!("Completed {} steps in {:.1}s", n_steps, elapsed);
    
    // Extract impedance
    println!("\nFrequency (GHz) | Zin (Ω) | S11 (dB)");
    println!("----------------|---------|----------");
    let (vf, cf) = probe.dft();
    for (idx, f) in freqs.iter().enumerate().step_by(10) {
        let (v_re, v_im) = vf[idx];
        let (i_re, i_im) = cf[idx];
        let denom = i_re*i_re + i_im*i_im;
        if denom > 1e-30 {
            let z_re = (v_re*i_re + v_im*i_im) / denom;
            let z_im = (v_im*i_re - v_re*i_im) / denom;
            // S11 vs 50 ohm
            let z0 = 50.0;
            let gamma_re = ((z_re - z0)*(z_re + z0) + z_im*z_im) / ((z_re + z0)*(z_re + z0) + z_im*z_im);
            let gamma_im = (2.0*z_im*z0) / ((z_re + z0)*(z_re + z0) + z_im*z_im);
            let s11_db = 10.0 * (gamma_re*gamma_re + gamma_im*gamma_im).log10();
            println!("{:14.2} | {:5.1} {:+5.1}j | {:6.2}", 
                     f/1e9, z_re, z_im, s11_db);
        } else {
            println!("{:14.2} | (no signal)", f/1e9);
        }
    }
}
