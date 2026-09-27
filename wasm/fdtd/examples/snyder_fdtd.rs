//! Snyder 2.44 GHz patch: broadband FDTD run → Zin(f), S11(f).
//!
//! Usage: `cargo run --release --example snyder_fdtd -- [t_end_ns]`
//! Prints a table of frequency, Zin, |S11| dB at 2.0–3.0 GHz.

use fdtd::cpml::Cpml;
use fdtd::geometry::{build_snyder, DX, DY, DZ, NX, NY, NZ, X0, Y0, Z0};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{cdiv, DftProbe, GaussianSource, ResistiveSource};

fn main() {
    let t_end_ns: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(17.0);
    let t_end = t_end_ns * 1e-9;

    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    // Sanity: the origin constants must match the grid we built.
    assert!((X0 - -58.0e-3).abs() < 1e-12);
    assert!((Y0 - -58.0e-3).abs() < 1e-12);
    assert!((Z0 - -31.5 * DZ).abs() < 1e-12);

    let dt = grid.courant_dt(0.99);
    println!("grid {NX}x{NY}x{NZ}  dx={DX:.3e} dy={DY:.3e} dz={DZ:.3e}");
    println!("dt = {dt:.4e} s  (cfl 0.99)");

    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let feed = build_snyder(&grid, &mut mat, dt);
    println!(
        "source Ez cell ({},{},{})  loop k={}",
        feed.src_i, feed.src_j, feed.src_k, feed.k_loop
    );

    let cpml = Cpml::new(&grid, 8, dt);
    let source = GaussianSource::new(2.5e9, 1.0);
    let rsource = ResistiveSource { rs: 50.0 };
    let freqs: Vec<f64> = (0..=100).map(|m| 2.0e9 + m as f64 * 10e6).collect();
    let probe = DftProbe::new(freqs.clone(), dt);

    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe);
    let n_steps = (t_end / dt).ceil() as usize;
    println!("running {n_steps} steps ({t_end_ns} ns)...");
    let t_start = std::time::Instant::now();
    let chunk = (n_steps / 20).max(1);
    let mut done = 0;
    while done < n_steps {
        let n = chunk.min(n_steps - done);
        sim.run(n);
        done += n;
        let el = t_start.elapsed().as_secs_f64();
        println!(
            "  step {done}/{n_steps}  t={:.2} ns  tail={:.2e}  ({:.1}s, {:.0} steps/s)",
            sim.sim_time() * 1e9,
            sim.probe.tail_ratio(),
            el,
            done as f64 / el,
        );
    }
    println!("wall: {:.1}s", t_start.elapsed().as_secs_f64());

    let (vf, cf) = sim.probe.dft();
    println!("# MHz  Re(Zin)  Im(Zin)  |S11|_dB");
    for (m, &f) in freqs.iter().enumerate() {
        let (vr, vi) = vf[m];
        let (ir, ii) = cf[m];
        let (zr, zi) = cdiv(vr, vi, ir, ii);
        // S11 = (Z − 50)/(Z + 50)
        let (sr, si) = cdiv(zr - 50.0, zi, zr + 50.0, zi);
        let s11_db = 10.0 * (sr * sr + si * si).log10();
        if (m % 10) == 0 {
            println!(
                "{:.0}  {:.4}  {:+.4}  {:.2}",
                f / 1e6,
                zr,
                zi,
                s11_db
            );
        }
    }
    // Also print the six FEM sweep frequencies exactly.
    println!("# --- FEM sweep frequencies ---");
    for &f in &[2.3e9f64, 2.4e9, 2.5e9, 2.6e9, 2.7e9, 2.8e9] {
        // nearest precomputed bin
        let m = ((f - 2.0e9) / 10e6).round() as usize;
        let (vr, vi) = vf[m];
        let (ir, ii) = cf[m];
        let (zr, zi) = cdiv(vr, vi, ir, ii);
        let (sr, si) = cdiv(zr - 50.0, zi, zr + 50.0, zi);
        let s11_db = 10.0 * (sr * sr + si * si).log10();
        println!(
            "{:.0}  Zin={:.4}{:+.4}j  S11={:.2} dB",
            f / 1e6,
            zr,
            zi,
            s11_db
        );
    }
}
