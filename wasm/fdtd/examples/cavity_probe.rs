//! Cavity probe: the efs-patch voxelization (substrate, ground, patch)
//! with NO port — an initial Ez blob inside the cavity, stepped with the
//! raw solver kernels. Dumps Ez at a cavity point so the cavity's
//! natural resonances can be read off a spectrum, independent of all
//! port/source machinery.
//!
//! Usage: `cargo run --release --example cavity_probe -- [t_end_ns] [cfl]`

use fdtd::cpml::Cpml;
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::{apply_pec, update_e, update_h};

const NX: usize = 61;
const NY: usize = 61;
const DX: f32 = 2.0e-3;
const DY: f32 = 2.0e-3;
const X0: f32 = -61.0e-3;
const Y0: f32 = -61.0e-3;
const SUB_HALF: f32 = 30.0e-3;
const SUB_H: f32 = 1.524e-3;
const PATCH_HX: f32 = 16.0e-3;
const PATCH_HY: f32 = 20.0e-3;
const EPS_R: f32 = 3.38;
const SIGMA: f32 = 4.61e-4;

fn eps_r() -> f32 {
    std::env::var("CAVITY_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(EPS_R)
}

fn patch_hx() -> f32 {
    std::env::var("CAVITY_PHX_MM")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|mm| mm * 1e-3)
        .unwrap_or(PATCH_HX)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t_end_ns: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8.0);
    let cfl: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.5);

    // Geometry overrides for bisection experiments (defaults = the
    // efs fixture): CAVITY_SUBCELLS (substrate layers), CAVITY_KG
    // (ground plane k), CAVITY_NZ (grid depth).
    let subcells: usize = std::env::var("CAVITY_SUBCELLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let k_ground: usize = std::env::var("CAVITY_KG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let NZ: usize = std::env::var("CAVITY_NZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(121);
    let DZ: f32 = SUB_H / subcells as f32;
    let Z0: f32 = -(k_ground as f32 + 0.5) * DZ;
    let k_patch = k_ground + subcells;
    let rec_k = k_ground + subcells / 2;

    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let dt = grid.courant_dt(cfl);

    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    // Substrate dielectric — same box tests as efs_patch.rs.
    for k in 0..NZ {
        let zc = (k as f32 + 0.5) * DZ + Z0;
        if zc <= 0.0 || zc >= SUB_H {
            continue;
        }
        for j in 0..NY {
            let y = (j as f32 + 0.5) * DY + Y0;
            if y.abs() > SUB_HALF {
                continue;
            }
            for i in 0..=NX {
                let x = i as f32 * DX + X0;
                if x.abs() <= SUB_HALF {
                    mat.set_dielectric(0, grid.ex_idx(i, j, k), eps_r(), SIGMA, dt);
                }
            }
        }
    }
    for k in 0..NZ {
        let zc = (k as f32 + 0.5) * DZ + Z0;
        if zc <= 0.0 || zc >= SUB_H {
            continue;
        }
        for j in 0..=NY {
            let y = j as f32 * DY + Y0;
            if y.abs() > SUB_HALF {
                continue;
            }
            for i in 0..NX {
                let x = (i as f32 + 0.5) * DX + X0;
                if x.abs() <= SUB_HALF {
                    mat.set_dielectric(1, grid.ey_idx(i, j, k), eps_r(), SIGMA, dt);
                }
            }
        }
    }
    for k in 0..NZ {
        let z = k as f32 * DZ + Z0;
        if z < 0.0 || z >= SUB_H {
            continue;
        }
        for j in 0..NY {
            let y = (j as f32 + 0.5) * DY + Y0;
            if y.abs() > SUB_HALF {
                continue;
            }
            for i in 0..NX {
                let x = (i as f32 + 0.5) * DX + X0;
                if x.abs() <= SUB_HALF {
                    mat.set_dielectric(2, grid.ez_idx(i, j, k), eps_r(), SIGMA, dt);
                }
            }
        }
    }
    // Ground sheet (k_ground) and patch sheet (k_patch).
    for j in 0..NY {
        let y = (j as f32 + 0.5) * DY + Y0;
        if y.abs() > SUB_HALF {
            continue;
        }
        for i in 0..=NX {
            let x = i as f32 * DX + X0;
            if x.abs() <= SUB_HALF {
                mat.set_pec(0, grid.ex_idx(i, j, k_ground));
            }
        }
    }
    for j in 0..=NY {
        let y = j as f32 * DY + Y0;
        if y.abs() > SUB_HALF {
            continue;
        }
        for i in 0..NX {
            let x = (i as f32 + 0.5) * DX + X0;
            if x.abs() <= SUB_HALF {
                mat.set_pec(1, grid.ey_idx(i, j, k_ground));
            }
        }
    }
    // CAVITY_THICK_GROUND=1: volumetric ground — PEC Ez in the two
    // layers below the sheet plane (k_ground, k_ground-1) over the
    // ground footprint, so the below-sheet Ez cannot share the sheet's
    // H samples with the cavity (tests the sheet-straddle coupling).
    if std::env::var("CAVITY_THICK_GROUND").is_ok() {
        for &kk in &[k_ground, k_ground - 1] {
            for j in 0..NY {
                let y = (j as f32 + 0.5) * DY + Y0;
                if y.abs() > SUB_HALF {
                    continue;
                }
                for i in 0..NX {
                    let x = (i as f32 + 0.5) * DX + X0;
                    if x.abs() <= SUB_HALF {
                        mat.set_pec(2, grid.ez_idx(i, j, kk));
                    }
                }
            }
        }
    }
    for j in 0..NY {
        let y = (j as f32 + 0.5) * DY + Y0;
        if y.abs() > PATCH_HY {
            continue;
        }
        for i in 0..=NX {
            let x = i as f32 * DX + X0;
            if x.abs() <= patch_hx() {
                mat.set_pec(0, grid.ex_idx(i, j, k_patch));
            }
        }
    }
    for j in 0..=NY {
        let y = j as f32 * DY + Y0;
        if y.abs() > PATCH_HY {
            continue;
        }
        for i in 0..NX {
            let x = (i as f32 + 0.5) * DX + X0;
            if x.abs() <= patch_hx() {
                mat.set_pec(1, grid.ey_idx(i, j, k_patch));
            }
        }
    }

    let mut g = grid;
    // Mask audit: actual PEC extents at the patch plane.
    {
        let xs: Vec<usize> = (0..=NX)
            .filter(|&i| mat.pec_x[g.ex_idx(i, 30, k_patch)] != 0)
            .collect();
        let ys: Vec<usize> = (0..=NY)
            .filter(|&j| mat.pec_y[g.ey_idx(35, j, k_patch)] != 0)
            .collect();
        println!(
            "patch pec_x i range: {:?}..={:?}; pec_y j range: {:?}..={:?}",
            xs.first(),
            xs.last(),
            ys.first(),
            ys.last()
        );
        let gx: Vec<usize> = (0..=NX)
            .filter(|&i| mat.pec_x[g.ex_idx(i, 30, k_ground)] != 0)
            .collect();
        println!("ground pec_x i range: {:?}..={:?}", gx.first(), gx.last());
    }
    let mut cpml = Cpml::new(&g, 12, dt);
    // Initial Ez blob at (35,30), spanning the substrate layers — or
    // below the ground sheet with CAVITY_BLOB_BELOW=1 (sheet-leak test).
    let (bi, bj) = (35usize, 30usize);
    let blob_below_vec: Vec<usize> = ((k_ground - subcells)..k_ground).collect();
    let blob_above_vec: Vec<usize> = ((k_ground + 1)..=k_patch).collect();
    let blob_layers: &[usize] = if std::env::var("CAVITY_BLOB_BELOW").is_ok() {
        &blob_below_vec
    } else {
        &blob_above_vec
    };
    for &k in blob_layers {
        for j in 0..NY {
            for i in 0..NX {
                let r2 = (i as f32 - bi as f32).powi(2) + (j as f32 - bj as f32).powi(2);
                let idx = g.ez_idx(i, j, k);
                g.ez[idx] = (-r2 / 8.0).exp();
            }
        }
    }
    // CAVITY_TM10=1: replace the blob with a pure TM10-shaped initial
    // field (z-uniform cos across the patch footprint, zero elsewhere)
    // — near-zero divergence residue, rings at the cavity's true f0.
    if std::env::var("CAVITY_TM10").is_ok() {
        for e in g.ez.iter_mut() {
            *e = 0.0;
        }
        for k in (k_ground + 1)..=k_patch {
            for j in 0..NY {
                let y = (j as f32 + 0.5) * DY + Y0;
                if y.abs() > PATCH_HY {
                    continue;
                }
                for i in 0..NX {
                    let x = (i as f32 + 0.5) * DX + X0;
                    if x.abs() > patch_hx() {
                        continue;
                    }
                    let idx = g.ez_idx(i, j, k);
                    g.ez[idx] = (std::f32::consts::PI * x / (2.0 * patch_hx())).cos();
                }
            }
        }
        println!("TM10-shaped initial field installed");
    }

    let n_steps = (t_end_ns * 1e-9 / dt).ceil() as usize;
    println!("dt={dt:.4e} steps={n_steps} cfl={cfl}");
    let mut out = String::new();
    let mut map = String::new();
    for n in 0..n_steps {
        update_h(&mut g, dt);
        cpml.correct_h(&mut g, dt);
        update_e(&mut g, &mat);
        cpml.correct_e(&mut g, &mat.cb_x, &mat.cb_y, &mat.cb_z);
        apply_pec(&mut g, &mat);
        out.push_str(&format!(
            "{},{:.6e},{:.6e}\n",
            n,
            g.ez[g.ez_idx(35, 30, rec_k)],
            g.ez[g.ez_idx(45, 30, rec_k)]
        ));
        // Per-layer Ez energy over the substrate footprint, printed
        // periodically: where does the ringdown energy live in z?
        if n % 2000 == 1999 {
            let mut line = format!("E_layer step {}:", n + 1);
            for k in (k_ground - 4)..=(k_patch + 4) {
                let mut s = 0.0f32;
                for j in 15..46usize {
                    for i in 15..46usize {
                        let e = g.ez[g.ez_idx(i, j, k)];
                        s += e * e;
                    }
                }
                line.push_str(&format!(" k{}={:.3e}", k, s));
            }
            println!("{line}");
        }
        // Full row/column snapshots for mode-shape mapping.
        for i in 10..=51 {
            map.push_str(&format!("{:.5e},", g.ez[g.ez_idx(i, 30, rec_k)]));
        }
        map.push_str(";");
        for j in 10..=51 {
            map.push_str(&format!("{:.5e},", g.ez[g.ez_idx(35, j, rec_k)]));
        }
        map.push_str(";");
        for k in (k_ground - 6)..=(k_patch + 10) {
            map.push_str(&format!("{:.5e},", g.ez[g.ez_idx(35, 30, k)]));
        }
        map.push('\n');
    }
    std::fs::write("/tmp/cavity.csv", out).expect("write cavity series");
    std::fs::write("/tmp/cavity_map.csv", map).expect("write cavity map");
    println!("wrote /tmp/cavity.csv and /tmp/cavity_map.csv");
}
