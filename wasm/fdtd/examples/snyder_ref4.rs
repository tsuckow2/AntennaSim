//! Reference for the Z-REFINED Snyder variant (dz = 0.2625 mm,
//! NZ = 123, 6 substrate layers): material/CPML checksums the JS
//! builder must reproduce for config {zref:true}, plus CPU (V,I)
//! series aggregates for parity.
//!
//! Geometry mirrors geometry.rs `build_snyder` loop-for-loop with:
//! k_ground 31->63, k_patch 34->69, k_src 32->64, post Ez 33..=34 ->
//! 65..=69, thick-ground Ez {31,30} -> {63,62}, Z0 = -63.5*dz.
//!
//! Usage: `cargo run --release --example snyder_ref4 -- [steps]`

use fdtd::cpml::Cpml;
use fdtd::geometry::{DX, DY, NX, NY};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{DftProbe, Feed, GaussianSource, ResistiveSource};

const DZ2: f32 = 6.5625e-5;
const NZ2: usize = 492;
const X0: f32 = -58.0e-3;
const Y0: f32 = -58.0e-3;
const KG: usize = 255;
const KP: usize = 279;
const KSRC: usize = 256;

fn wsum(a: &[f32]) -> (f64, f64) {
    let mut s = 0.0f64;
    let mut w = 0.0f64;
    for (i, &x) in a.iter().enumerate() {
        s += x as f64;
        w += x as f64 * (i as f64 + 1.0);
    }
    (s, w)
}

fn build_refined(g: &YeeGrid, mat: &mut Materials, dt: f32) -> Feed {
    let eps0 = 8.854_187_817e-12f32;
    let eps_r = 4.4f32;
    let sigma_sub = 2.0 * std::f32::consts::PI * 2.5e9 * eps0 * eps_r * 0.02;
    let sub_half = 101.6e-3f32 / 2.0;
    let patch_hx = 28.2084e-3f32 / 2.0;
    let patch_hy = 37.5789e-3f32 / 2.0;
    let sub_h = 1.575e-3f32;
    let probe_x = -14.1042e-3f32 + 7.5658e-3;
    let probe_half = 0.65e-3f32;
    let z0 = -(KG as f32 + 0.5) * DZ2;
    // Ex substrate.
    for k in 0..g.nz {
        let zc = (k as f32 + 0.5) * DZ2 + z0;
        if zc < 0.0 || zc > sub_h {
            continue;
        }
        for j in 0..g.ny {
            let y = (j as f32 + 0.5) * DY + Y0;
            if y.abs() > sub_half {
                continue;
            }
            for i in 0..=g.nx {
                let x = i as f32 * DX + X0;
                if x.abs() > sub_half {
                    continue;
                }
                if (x - probe_x).abs() < probe_half + DX
                    && y.abs() < probe_half + DY
                    && k >= KG
                    && k <= KP
                {
                    continue;
                }
                mat.set_dielectric(0, g.ex_idx(i, j, k), eps_r, sigma_sub, dt);
            }
        }
    }
    // Ey substrate.
    for k in 0..g.nz {
        for j in 0..=g.ny {
            let y = j as f32 * DY + Y0;
            if y.abs() > sub_half {
                continue;
            }
            let zc = (k as f32 + 0.5) * DZ2 + z0;
            if zc < 0.0 || zc > sub_h {
                continue;
            }
            for i in 0..g.nx {
                let x = (i as f32 + 0.5) * DX + X0;
                if x.abs() > sub_half {
                    continue;
                }
                if (x - probe_x).abs() < probe_half + DX && y.abs() < probe_half + DY {
                    continue;
                }
                mat.set_dielectric(1, g.ey_idx(i, j, k), eps_r, sigma_sub, dt);
            }
        }
    }
    // Ez substrate.
    for k in 0..=g.nz {
        let z = k as f32 * DZ2 + z0;
        if z < 0.0 || z > sub_h {
            continue;
        }
        for j in 0..g.ny {
            let y = (j as f32 + 0.5) * DY + Y0;
            if y.abs() > sub_half {
                continue;
            }
            for i in 0..g.nx {
                let x = (i as f32 + 0.5) * DX + X0;
                if x.abs() > sub_half {
                    continue;
                }
                if (x - probe_x).abs() < probe_half + DX && y.abs() < probe_half + DY {
                    continue;
                }
                mat.set_dielectric(2, g.ez_idx(i, j, k), eps_r, sigma_sub, dt);
            }
        }
    }
    // Ground sheet.
    for j in 0..g.ny {
        let y = (j as f32 + 0.5) * DY + Y0;
        if y.abs() > sub_half {
            continue;
        }
        for i in 0..=g.nx {
            let x = i as f32 * DX + X0;
            if x.abs() <= sub_half {
                mat.set_pec(0, g.ex_idx(i, j, KG));
            }
        }
    }
    for j in 0..=g.ny {
        let y = j as f32 * DY + Y0;
        if y.abs() > sub_half {
            continue;
        }
        for i in 0..g.nx {
            let x = (i as f32 + 0.5) * DX + X0;
            if x.abs() <= sub_half {
                mat.set_pec(1, g.ey_idx(i, j, KG));
            }
        }
    }
    // Patch sheet.
    for j in 0..g.ny {
        let y = (j as f32 + 0.5) * DY + Y0;
        if y.abs() > patch_hy {
            continue;
        }
        for i in 0..=g.nx {
            let x = i as f32 * DX + X0;
            if x.abs() <= patch_hx {
                mat.set_pec(0, g.ex_idx(i, j, KP));
            }
        }
    }
    for j in 0..=g.ny {
        let y = j as f32 * DY + Y0;
        if y.abs() > patch_hy {
            continue;
        }
        for i in 0..g.nx {
            let x = (i as f32 + 0.5) * DX + X0;
            if x.abs() <= patch_hx {
                mat.set_pec(1, g.ey_idx(i, j, KP));
            }
        }
    }
    // PEC post: plus-cross at (102,115), Ez k=65..=69.
    for (di, dj) in [(0isize, 0isize), (1, 0), (-1, 0), (0, 1), (0, -1)] {
        let (ii, jj) = ((102 + di) as usize, (115 + dj) as usize);
        for k in 257..=279 {
            mat.set_pec(2, g.ez_idx(ii, jj, k));
        }
    }
    // Thick ground: Ez k=63,62 over the footprint.
    for j in 14..218usize {
        for i in 14..218usize {
            mat.set_pec(2, g.ez_idx(i, j, 255));
            mat.set_pec(2, g.ez_idx(i, j, 254));
        }
    }
    Feed {
        src_i: 102,
        src_j: 115,
        src_k: KSRC,
        k_loop: KSRC,
        hx_j_lo: 113,
        hx_j_hi: 118,
        hx_i0: 100,
        hx_i1: 105,
        hy_i_lo: 100,
        hy_i_hi: 105,
        hy_j0: 113,
        hy_j1: 118,
    }
}

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6000);
    let grid = YeeGrid::new(NX, NY, NZ2, DX, DY, DZ2);
    let dt = grid.courant_dt(0.1);
    let mut mat = Materials::vacuum(NX, NY, NZ2, dt);
    let feed = build_refined(&grid, &mut mat, dt);
    println!("zref: dt={dt:.6e} src=({},{},{})", feed.src_i, feed.src_j, feed.src_k);
    for (nm, a) in [
        ("ca_x", &mat.ca_x),
        ("ca_y", &mat.ca_y),
        ("ca_z", &mat.ca_z),
        ("cb_x", &mat.cb_x),
        ("cb_y", &mat.cb_y),
        ("cb_z", &mat.cb_z),
    ] {
        let (s, w) = wsum(a);
        println!("{nm}: sum={s:.6e} wsum={w:.6e}");
    }
    for (nm, a) in [
        ("pec_x", &mat.pec_x),
        ("pec_y", &mat.pec_y),
        ("pec_z", &mat.pec_z),
    ] {
        let c: u64 = a.iter().map(|&x| x as u64).sum();
        println!("{nm}: count={c}");
    }
    let cpml = Cpml::new(&grid, 12, dt);
    let snap = cpml.snapshot();
    let mut coeff: Vec<f32> = Vec::new();
    let mut psi_len = 0usize;
    for f in &snap {
        coeff.extend_from_slice(&f.b_e);
        coeff.extend_from_slice(&f.a_e);
        coeff.extend_from_slice(&f.b_h);
        coeff.extend_from_slice(&f.a_h);
        coeff.extend_from_slice(&f.kinv);
        psi_len += f.psi_a.len() + f.psi_b.len() + f.psi_c.len() + f.psi_d.len();
    }
    let (s, w) = wsum(&coeff);
    println!("cpml coeff: sum={s:.6e} wsum={w:.6e} psi_len={psi_len}");
    let source = GaussianSource::new(2.5e9, 1.0);
    let rsource = ResistiveSource { rs: 50.0 };
    let freqs: Vec<f64> = (0..=100).map(|m| 2.0e9 + m as f64 * 10e6).collect();
    let probe = DftProbe::new(freqs, dt);
    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, None);
    sim.run(steps);
    let (v, i) = sim.probe.series();
    let sa: f64 = v.iter().map(|x| (*x as f64).abs()).sum();
    let sb: f64 = i.iter().map(|x| (*x as f64).abs()).sum();
    let mx = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!(
        "series[{steps}]: sumAbsV={sa:.6e} sumAbsI={sb:.6e} maxAbsV={mx:.6e} lastV={:.6e} lastI={:.6e}",
        v[steps - 1], i[steps - 1]
    );
}
