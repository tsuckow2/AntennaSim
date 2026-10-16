//! Undriven TM10 ringdown of the Snyder build (post + thick ground
//! configuration) with a SPECTRAL readout: initialize Ez in the slab
//! with the TM10 shape, step the raw solver, record an edge-column Ez
//! probe, and scan a DFT for the dominant peaks. Decides whether the
//! driven ~2.02 GHz resonance is the cavity's own TM10.
//!
//! Usage: `cargo run --release --example snyder_spec -- [t_ns]`

use fdtd::cpml::Cpml;
use fdtd::geometry::{build_snyder_wire, DX, DY, DZ, NX, NY, NZ};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{DftProbe, GaussianSource, ResistiveSource};

fn main() {
    let t_ns: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let dt = grid.courant_dt(0.1);
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
    let cpml = Cpml::new(&grid, 12, dt);
    let source = GaussianSource::new(2.5e9, 0.0); // zero drive
    let rsource = ResistiveSource { rs: 50.0 };
    let probe = DftProbe::new(vec![2.3e9], dt);
    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, None);
    // TM10 IC: Ez in slab layers k=32..34 over the patch footprint,
    // sin(pi * u) along x (u across the 28.2 mm resonant length),
    // uniform along y. Patch spans i in [88,144], j in [78,154] roughly
    // (edges at x = +/-14.1042 mm, y = +/-18.789 mm).
    for k in 32..=34usize {
        for j in 78..=154usize {
            for i in 88..=144usize {
                let u = (i - 88) as f32 / 56.0;
                let idx = sim.grid.ez_idx(i, j, k);
                sim.grid.ez[idx] = (std::f32::consts::PI * u).sin();
            }
        }
    }
    let steps = (t_ns * 1e-9 / sim.dt) as usize;
    let mut series: Vec<f32> = Vec::with_capacity(steps);
    let pe = sim.grid.ez_idx(88, 115, 33); // radiating-edge column
    let pc = sim.grid.ez_idx(116, 115, 33); // TM10 node (center)
    for _ in 0..steps {
        sim.step();
        series.push(sim.grid.ez[pe]);
    }
    println!(
        "spec: {} steps ({:.2} ns), center probe final Ez = {:.3e} (expect ~0 for TM10)",
        steps,
        steps as f32 * sim.dt * 1e9,
        sim.grid.ez[pc]
    );
    // DFT scan 1.4..3.6 GHz at 4 MHz.
    let n = series.len();
    let mut best: Vec<(f64, f64)> = Vec::new();
    let mut f = 1.4e9f64;
    while f < 3.6e9 {
        let wdt = 2.0 * std::f64::consts::PI * f * sim.dt as f64;
        let (cw, sw) = (wdt.cos(), wdt.sin());
        let (mut c, mut s) = (1.0f64, 0.0f64);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for m in 0..n {
            let v = series[m] as f64;
            // Hann window to tame the static/low-freq residue.
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * m as f64 / n as f64).cos();
            re += v * w * c;
            im -= v * w * s;
            let cn = c * cw - s * sw;
            s = s * cw + c * sw;
            c = cn;
        }
        best.push((f, (re * re + im * im).sqrt()));
        f += 4e6;
    }
    let mut sorted = best.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    print!("top spectral peaks (GHz, amp):");
    for (f, a) in sorted.iter().take(6) {
        print!("  {:.3}/{:.3e}", f / 1e9, a);
    }
    println!();
}
