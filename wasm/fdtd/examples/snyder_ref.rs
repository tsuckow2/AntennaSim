//! Reference aggregates for the WebGPU solver page (post + thick
//! ground Snyder configuration): material/CPML checksums the JS
//! builder must reproduce, plus the CPU (V,I) series aggregates for
//! the first N steps that the GPU page run must match.
//!
//! Usage: `cargo run --release --example snyder_ref -- [steps]`

use fdtd::cpml::Cpml;
use fdtd::geometry::{build_snyder_wire, DX, DY, DZ, NX, NY, NZ};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{DftProbe, GaussianSource, ResistiveSource};

fn wsum(a: &[f32]) -> (f64, f64) {
    // (sum, index-weighted sum) in f64 over f32 data.
    let mut s = 0.0f64;
    let mut w = 0.0f64;
    for (i, &x) in a.iter().enumerate() {
        s += x as f64;
        w += x as f64 * (i as f64 + 1.0);
    }
    (s, w)
}

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let cfl: f32 = std::env::var("SNYDER_CFL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    let dt = grid.courant_dt(cfl);
    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let (feed, _tw) = build_snyder_wire(&grid, &mut mat, dt);
    let (pi, pj) = (feed.src_i, feed.src_j);
    for (di, dj) in [(0isize, 0isize), (1, 0), (-1, 0), (0, 1), (0, -1)] {
        let (ii, jj) = ((pi as isize + di) as usize, (pj as isize + dj) as usize);
        for k in 33..=34 {
            mat.set_pec(2, grid.ez_idx(ii, jj, k));
        }
    }
    for j in 14..218usize {
        for i in 14..218usize {
            mat.set_pec(2, grid.ez_idx(i, j, 31));
            mat.set_pec(2, grid.ez_idx(i, j, 30));
        }
    }
    println!("src cell: ({pi},{pj},{})", feed.src_k);
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
        println!(
            "face axis={} lo={} npml={} n1={} n2={}",
            f.axis, f.lo, f.npml, f.n1, f.n2
        );
    }
    let (s, w) = wsum(&coeff);
    println!("cpml coeff: sum={s:.6e} wsum={w:.6e} psi_len={psi_len}");

    // CPU series aggregates for `steps` steps of the post config.
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
