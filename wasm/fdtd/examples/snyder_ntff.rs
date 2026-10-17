//! Snyder patch gain/directivity via the Huygens-box NTFF.
//!
//! Production geometry (`build_snyder`: PEC post + volumetric
//! ground), base grid, CFL 0.2. NTFF phasors at 2.00–2.03 GHz
//! (the driven dip is at ~2.015 GHz); gain = 4π·Umax / P_acc with
//! P_acc = ½·Re{V·I*} from the port DFT over the same steps.
//!
//! Usage: `cargo run --release --example snyder_ntff -- [t_ns]`
//! Env: SNYDER_NTFF_CFL (default 0.2).

use fdtd::cpml::Cpml;
use fdtd::geometry::{build_snyder, DX, DY, DZ, NX, NY, NZ, X0, Y0, Z0};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::ntff::{Ntff, NtffBox};
use fdtd::solver::Fdtd;
use fdtd::source::{cdiv, DftProbe, GaussianSource, ResistiveSource};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t_ns: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8.0);
    let cfl: f32 = std::env::var("SNYDER_NTFF_CFL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.2);

    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let dt = grid.courant_dt(cfl);
    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let (feed, _wire) = build_snyder(&grid, &mut mat, dt);
    let cpml = Cpml::new(&grid, 12, dt);
    let source = GaussianSource::new(2.5e9, 1.0);
    let rsource = ResistiveSource { rs: 50.0 };
    // Port DFT band: 2.0–3.0 GHz in 10 MHz steps (includes the
    // NTFF frequencies), for Zin context + P_acc.
    let band: Vec<f64> = (0..=100).map(|m| 2.0e9 + m as f64 * 10e6).collect();
    let probe = DftProbe::new(band.clone(), dt);
    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, None);

    let ntff_freqs = vec![2.00e9, 2.01e9, 2.02e9, 2.03e9];
    let mut ntff = Ntff::new(
        &sim.grid,
        NtffBox { i0: 16, i1: 216, j0: 16, j1: 216, k0: 16, k1: 45 },
        (X0 as f64, Y0 as f64, Z0 as f64),
        ntff_freqs.clone(),
        dt,
    );
    println!("ntff samples: {}", ntff.n_samples());

    let n_steps = (t_ns * 1e-9 / dt as f64).ceil() as usize;
    println!("running {n_steps} steps ({t_ns} ns, CFL {cfl})...");
    let t0 = std::time::Instant::now();
    for s in 0..n_steps {
        sim.step();
        ntff.accumulate(&sim.grid, sim.sim_time() as f64);
        if s % 5000 == 4999 {
            println!(
                "  step {}/{n_steps} tail={:.2e} ({:.0}s)",
                s + 1,
                sim.probe.tail_ratio(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    println!("wall: {:.1}s", t0.elapsed().as_secs_f64());

    let (vf, cf) = sim.probe.dft();
    let n = sim.probe.len() as f64;
    let ph = 2.0 / n; // raw DFT sum -> phasor, same rule as Ntff
    for (fi, &f) in ntff_freqs.iter().enumerate() {
        let bi = band.iter().position(|&x| (x - f).abs() < 1.0).unwrap();
        let (zre, zim) = cdiv(vf[bi].0, vf[bi].1, cf[bi].0, cf[bi].1);
        // P_acc = ½ Re{V_ph · conj(I_ph)}
        let p_acc = 0.5
            * ph
            * ph
            * (vf[bi].0 * cf[bi].0 + vf[bi].1 * cf[bi].1);
        let flux = ntff.box_flux(fi);
        let (dlin, dbi, prad, th, ph_) = ntff.directivity(fi, 91, 180);
        let umax = prad * dlin / (4.0 * std::f64::consts::PI);
        let gain = 4.0 * std::f64::consts::PI * umax / p_acc;
        println!(
            "f={:.0} MHz: Zin={:.2}+j{:.2}  P_acc={:.4e} W  flux={:.4e}  Prad={:.4e}  eff={:.3}",
            f / 1e6,
            zre,
            zim,
            p_acc,
            flux,
            prad,
            prad / p_acc
        );
        println!(
            "   Dmax={:.2} dBi (lin {:.3}) at theta={:.1} phi={:.1}   Gain={:.2} dBi",
            dbi,
            dlin,
            th.to_degrees(),
            ph_.to_degrees(),
            10.0 * gain.log10()
        );
    }
}
