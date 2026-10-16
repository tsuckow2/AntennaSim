//! Undriven ringdown spectra of the Snyder build (post + thick
//! ground): TM10-shaped Ez initial condition in the slab, raw
//! solver, TWO field probes recorded every step —
//!   A: (88,115,33)  x-edge antinode: sees TM10 + TM01
//!   B: (116,78,33)  y-edge, x-center: TM10 node → isolates TM01
//! Hann-windowed DFT scan 1.0..6.5 GHz with distinct-peak picking.
//!
//! Usage: `cargo run --release --example snyder_spec2 -- [t_ns]`

use fdtd::cpml::Cpml;
use fdtd::geometry::{build_snyder_wire, DX, DY, DZ, NX, NY, NZ};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{DftProbe, GaussianSource, ResistiveSource};

fn spectrum(series: &[f32], dt: f32) -> Vec<(f64, f64)> {
    let n = series.len();
    let mut out = Vec::new();
    let mut f = 1.0e9f64;
    while f < 6.5e9 {
        let wdt = 2.0 * std::f64::consts::PI * f * dt as f64;
        let (cw, sw) = (wdt.cos(), wdt.sin());
        let (mut c, mut s) = (1.0f64, 0.0f64);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for m in 0..n {
            let v = series[m] as f64;
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * m as f64 / n as f64).cos();
            re += v * w * c;
            im -= v * w * s;
            let cn = c * cw - s * sw;
            s = s * cw + c * sw;
            c = cn;
        }
        out.push((f, (re * re + im * im).sqrt()));
        f += 4e6;
    }
    out
}

fn peaks(spec: &[(f64, f64)], n: usize) -> Vec<(f64, f64)> {
    // Local maxima separated by >= 120 MHz, strongest first.
    let mut cand: Vec<(f64, f64)> = Vec::new();
    for w in 1..spec.len() - 1 {
        if spec[w].1 >= spec[w - 1].1 && spec[w].1 >= spec[w + 1].1 {
            cand.push(spec[w]);
        }
    }
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut picked: Vec<(f64, f64)> = Vec::new();
    for c in cand {
        if picked.iter().all(|p: &(f64, f64)| (p.0 - c.0).abs() > 120e6) {
            picked.push(c);
            if picked.len() >= n {
                break;
            }
        }
    }
    picked
}

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
    let source = GaussianSource::new(2.5e9, 0.0);
    let rsource = ResistiveSource { rs: 50.0 };
    let probe = DftProbe::new(vec![2.3e9], dt);
    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, None);
    for k in 32..=34usize {
        for j in 78..=154usize {
            for i in 88..=144usize {
                let u = (i - 88) as f32 / 56.0;
                let idx = sim.grid.ez_idx(i, j, k);
                // TM10 Ez shape: ANTIPHASE at the radiating edges,
                // node at patch center (i=116): sin(pi*(i-116)/56.4).
                // (v1 used a same-sign dome = cos shape, which projects
                // onto TM20 -- see spec2 log analysis.)
                sim.grid.ez[idx] = ((i as f32 - 116.0) / 56.4 * std::f32::consts::PI).sin();
                let _ = u;
            }
        }
    }
    let steps = (t_ns * 1e-9 / sim.dt) as usize;
    let pa = sim.grid.ez_idx(88, 115, 33);
    let pb = sim.grid.ez_idx(116, 78, 33);
    let mut sa: Vec<f32> = Vec::with_capacity(steps);
    let mut sb: Vec<f32> = Vec::with_capacity(steps);
    for _ in 0..steps {
        sim.step();
        sa.push(sim.grid.ez[pa]);
        sb.push(sim.grid.ez[pb]);
    }
    for (nm, s) in [("A(x-edge)", &sa), ("B(y-edge)", &sb)] {
        let spec = spectrum(s, sim.dt);
        let pk = peaks(&spec, 5);
        print!("{nm} peaks:");
        for (f, a) in &pk {
            print!("  {:.3}GHz/{:.3e}", f / 1e9, a);
        }
        println!();
    }
}
