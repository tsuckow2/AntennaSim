//! CPML isolation probe: free-space anisotropic grid, Gaussian Ez blob,
//! no materials/source/PEC. If a corner mode grows here, the CPML is at
//! fault independent of the antenna fixture.
//!
//! Usage: `cargo run --release --example cpml_probe -- [t_end_ns] [cfl]`

use fdtd::cpml::Cpml;
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::{update_e, update_h};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t_end_ns: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let cfl: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let (nx, ny, nz) = (61usize, 61usize, 121usize);
    let (dx, dy, dz) = (2.0e-3f32, 2.0e-3f32, 0.381e-3f32);
    let mut g = YeeGrid::new(nx, ny, nz, dx, dy, dz);
    let dt = g.courant_dt(cfl);
    let mat = Materials::vacuum(nx, ny, nz, dt);
    let npml: usize = std::env::var("PROBE_NPML")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let mut cpml = Cpml::new(&g, npml, dt);
    // Gaussian blob at the domain center.
    let (ci, cj, ck) = (nx / 2, ny / 2, nz / 2);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let r2 = (i as f32 - ci as f32).powi(2)
                    + (j as f32 - cj as f32).powi(2)
                    + ((k as f32 - ck as f32) * dz / dx).powi(2);
                let idx = g.ez_idx(i, j, k);
                g.ez[idx] = (-r2 / 18.0).exp();
            }
        }
    }
    let n_steps = (t_end_ns * 1e-9 / dt).ceil() as usize;
    println!("dt={dt:.4e} steps={n_steps} cfl={cfl}");
    // Optional fixture-like features: PROBE_GROUND=1 adds a PEC sheet at
    // k=12 spanning i,j in 15..46 (the efs fixture's ground); PROBE_BLOB
    // = "i,j,k" relocates the blob (e.g. into the below-ground slab).
    let ground = std::env::var("PROBE_GROUND").is_ok();
    if let Ok(b) = std::env::var("PROBE_BLOB") {
        let v: Vec<i32> = b.split(',').map(|s| s.parse().unwrap()).collect();
        // Rebuild the blob at the requested center (fields were zeroed
        // above only if we re-init; simplest: shift by re-exponentiating
        // from the current field is wrong, so redo from scratch.)
        for e in g.ez.iter_mut() {
            *e = 0.0;
        }
        let (bi, bj, bk) = (v[0], v[1], v[2]);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let r2 = (i as f32 - bi as f32).powi(2)
                        + (j as f32 - bj as f32).powi(2)
                        + ((k as f32 - bk as f32) * dz / dx).powi(2);
                    let idx = g.ez_idx(i, j, k);
                    g.ez[idx] = (-r2 / 18.0).exp();
                }
            }
        }
        println!("blob at ({bi},{bj},{bk}), ground={ground}");
    }
    let chunk = (n_steps / 20).max(1);
    let mut done = 0;
    while done < n_steps {
        for _ in 0..chunk.min(n_steps - done) {
            update_h(&mut g, dt);
            cpml.correct_h(&mut g, dt);
            update_e(&mut g, &mat);
            cpml.correct_e(&mut g, &mat.cb_x, &mat.cb_y, &mat.cb_z);
            if ground {
                // PEC sheet at k=12, i in 15..=46 (Ex), j in 15..46.
                for j in 15..46usize {
                    for i in 15..=46usize {
                        let e = g.ex_idx(i, j, 12);
                        g.ex[e] = 0.0;
                    }
                }
                for j in 15..=46usize {
                    for i in 15..46usize {
                        let e = g.ey_idx(i, j, 12);
                        g.ey[e] = 0.0;
                    }
                }
            }
            done += 1;
        }
        let (mx, arg) = g
            .ez
            .iter()
            .enumerate()
            .map(|(n, x)| (x.abs(), n))
            .fold((0.0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a });
        let (ai, aj, ak) = (arg % nx, (arg / nx) % ny, arg / (nx * ny));
        let energy: f64 = [&g.ex, &g.ey, &g.ez]
            .iter()
            .flat_map(|v| v.iter())
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            + [&g.hx, &g.hy, &g.hz]
                .iter()
                .flat_map(|v| v.iter())
                .map(|x| {
                    let h = *x as f64;
                    h * h * 1.42e5 // μ0/ε0 impedance-squared scale
                })
                .sum::<f64>();
        println!(
            "  step {done}/{n_steps}  t={:.2} ns  max|Ez|={mx:.3e} at ({ai},{aj},{ak})  E2={energy:.6e}",
            done as f32 * dt * 1e9
        );
    }
}
