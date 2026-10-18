//! Snyder patch gain/directivity via the Huygens-box NTFF.
//!
//! Production geometry (`build_snyder`: PEC post + volumetric
//! ground), base grid, CFL 0.2. NTFF phasors at 2.00–2.03 GHz
//! (the driven dip is at ~2.015 GHz); gain = 4π·Umax / P_acc with
//! P_acc = ½·Re{V·I*} from the port DFT over the same steps.
//!
//! Checkpointed: full sim state (fields, CPML ψ, probe series,
//! NTFF accumulators) is written every 2000 steps to
//! SNYDER_NTFF_CKPT (default ./ntff_snyder_ckpt.bin) and resumed
//! automatically on startup — a killed run loses at most one
//! checkpoint interval. (Written after the 2026-10-16 run died
//! uncheckpointed at 36.7% to a VM reboot.)
//!
//! Usage: `cargo run --release --example snyder_ntff -- [t_ns]`
//! Env: SNYDER_NTFF_CFL (default 0.2), SNYDER_NTFF_CKPT.

use fdtd::cpml::PmlFaceData;
use fdtd::cpml::Cpml;
use fdtd::geometry::{build_snyder, DX, DY, DZ, NX, NY, NZ, X0, Y0, Z0};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::ntff::{Ntff, NtffBox};
use fdtd::solver::Fdtd;
use fdtd::source::{cdiv, DftProbe, GaussianSource, ResistiveSource};

fn put_f32s(buf: &mut Vec<u8>, a: &[f32]) {
    buf.extend_from_slice(&(a.len() as u64).to_le_bytes());
    for x in a {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}
fn put_f64s(buf: &mut Vec<u8>, a: &[f64]) {
    buf.extend_from_slice(&(a.len() as u64).to_le_bytes());
    for x in a {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        v
    }
    fn f32s(&mut self) -> Vec<f32> {
        let n = self.u64() as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(f32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap()));
            self.p += 4;
        }
        out
    }
    fn f64s(&mut self) -> Vec<f64> {
        let n = self.u64() as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(f64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap()));
            self.p += 8;
        }
        out
    }
}

fn save_ckpt(path: &str, sim: &Fdtd, ntff: &Ntff, step_done: usize, n_steps: usize) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"SNCK1");
    buf.extend_from_slice(&(step_done as u64).to_le_bytes());
    buf.extend_from_slice(&(n_steps as u64).to_le_bytes());
    let g = &sim.grid;
    for a in [&g.ex, &g.ey, &g.ez, &g.hx, &g.hy, &g.hz] {
        put_f32s(&mut buf, a);
    }
    let snap = sim.cpml.snapshot();
    buf.extend_from_slice(&(snap.len() as u64).to_le_bytes());
    for f in &snap {
        for a in [&f.psi_a, &f.psi_b, &f.psi_c, &f.psi_d] {
            put_f32s(&mut buf, a);
        }
    }
    let (v, i) = sim.probe.export_series();
    put_f32s(&mut buf, &v);
    put_f32s(&mut buf, &i);
    let (nsteps, acc) = ntff.export_state();
    buf.extend_from_slice(&(nsteps as u64).to_le_bytes());
    put_f64s(&mut buf, &acc);
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &buf).expect("write ckpt");
    std::fs::rename(&tmp, path).expect("rename ckpt");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t_ns: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8.0);
    let cfl: f32 = std::env::var("SNYDER_NTFF_CFL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.2);
    let ckpt = std::env::var("SNYDER_NTFF_CKPT")
        .unwrap_or_else(|_| "ntff_snyder_ckpt.bin".to_string());
    let ckpt_every: usize = std::env::var("SNYDER_NTFF_CKPT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let dt = grid.courant_dt(cfl);
    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let (feed, _wire) = build_snyder(&grid, &mut mat, dt);
    let cpml = Cpml::new(&grid, 12, dt);
    let source = GaussianSource::new(2.5e9, 1.0);
    let rsource = ResistiveSource { rs: 50.0 };
    let band: Vec<f64> = (0..=100).map(|m| 2.0e9 + m as f64 * 10e6).collect();
    let probe = DftProbe::new(band.clone(), dt);
    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, None);

    let ntff_freqs = vec![2.00e9, 2.01e9, 2.02e9, 2.03e9];
    // Box must FULLY enclose the substrate/ground (they span
    // i/j 14..=218): faces at 13/219 clear the material by one
    // cell, one cell inside the CPML (starts 12/220). The first
    // run's box (16..216) cut through the substrate and ground,
    // and its box flux ran 14% below the sphere power while the
    // dipole/efs checks agreed to ~5% — v2 tests whether the
    // material-crossing surface was the cause.
    let mut ntff = Ntff::new(
        &sim.grid,
        NtffBox { i0: 13, i1: 219, j0: 13, j1: 219, k0: 13, k1: 48 },
        (X0 as f64, Y0 as f64, Z0 as f64),
        ntff_freqs.clone(),
        dt,
    );
    println!("ntff samples: {}", ntff.n_samples());

    let n_steps = (t_ns * 1e-9 / dt as f64).ceil() as usize;
    let n_steps = std::env::var("SNYDER_NTFF_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_steps);
    let mut start = 0usize;
    if let Ok(bytes) = std::fs::read(&ckpt) {
        let mut rd = Rd { b: &bytes, p: 0 };
        assert_eq!(&bytes[0..5], b"SNCK1", "ckpt magic");
        rd.p = 5;
        let step_done = rd.u64() as usize;
        let total = rd.u64() as usize;
        if total == n_steps && step_done < n_steps {
            for a in [
                &mut sim.grid.ex,
                &mut sim.grid.ey,
                &mut sim.grid.ez,
                &mut sim.grid.hx,
                &mut sim.grid.hy,
                &mut sim.grid.hz,
            ] {
                let v = rd.f32s();
                assert_eq!(v.len(), a.len(), "ckpt field size");
                a.copy_from_slice(&v);
            }
            let nf = rd.u64() as usize;
            let mut snap = Vec::with_capacity(nf);
            for _ in 0..nf {
                snap.push(PmlFaceData {
                    axis: 0,
                    lo: false,
                    npml: 0,
                    n1: 0,
                    n2: 0,
                    b_e: vec![],
                    a_e: vec![],
                    b_h: vec![],
                    a_h: vec![],
                    kinv: vec![],
                    psi_a: rd.f32s(),
                    psi_b: rd.f32s(),
                    psi_c: rd.f32s(),
                    psi_d: rd.f32s(),
                });
            }
            sim.cpml.restore_psi(&snap);
            let v = rd.f32s();
            let i = rd.f32s();
            sim.probe.import_series(v, i);
            sim.restore_clock(step_done);
            let ns = rd.u64() as usize;
            let acc = rd.f64s();
            ntff.import_state(ns, &acc);
            start = step_done;
            println!("RESUMED from checkpoint at step {start}/{n_steps}");
        } else {
            println!("ckpt config mismatch (total {total} vs {n_steps}); starting fresh");
        }
    }

    println!("running {n_steps} steps ({t_ns} ns, CFL {cfl}) from {start}...");
    let t0 = std::time::Instant::now();
    let dtf = dt as f64;
    for s in start..n_steps {
        sim.step();
        ntff.accumulate(&sim.grid, (s + 1) as f64 * dtf);
        if s % 500 == 499 {
            println!(
                "  step {}/{n_steps} tail={:.2e} ({:.0}s)",
                s + 1,
                sim.probe.tail_ratio(),
                t0.elapsed().as_secs_f64()
            );
        }
        if s % ckpt_every == ckpt_every - 1 {
            save_ckpt(&ckpt, &sim, &ntff, s + 1, n_steps);
        }
    }
    println!("wall: {:.1}s", t0.elapsed().as_secs_f64());
    if std::env::var("SNYDER_NTFF_DUMP").is_ok() {
        // Parity-gate dump: scaled phasors (H negated) at freq 0 for
        // a few samples + raw |acc| sums per frequency.
        for si in [0usize, 1, 1000, 50000, ntff.n_samples() - 1] {
            let ph = ntff.sample_phasors(0, si);
            println!(
                "probe si={si} {}",
                ph.iter()
                    .map(|(r, i)| format!("{r:.6e},{i:.6e}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        let (_, acc) = ntff.export_state();
        let per = ntff.n_samples() * 8;
        for fi in 0..ntff_freqs.len() {
            let s: f64 = acc[fi * per..(fi + 1) * per].iter().map(|x| x.abs()).sum();
            println!("accAbsSum f{fi} {s:.5e}");
        }
    }
    // NOTE: the checkpoint is deleted only AFTER the results are
    // printed below — the far-field evaluation takes minutes, and
    // deleting at the end of stepping lost a finished run's last
    // two frequencies to a restart on 2026-10-18.

    let (vf, cf) = sim.probe.dft();
    let n = sim.probe.len() as f64;
    let ph = 2.0 / n;
    for (fi, &f) in ntff_freqs.iter().enumerate() {
        let bi = band.iter().position(|&x| (x - f).abs() < 1.0).unwrap();
        let (zre, zim) = cdiv(vf[bi].0, vf[bi].1, cf[bi].0, cf[bi].1);
        let p_acc = 0.5 * ph * ph * (vf[bi].0 * cf[bi].0 + vf[bi].1 * cf[bi].1);
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
    let _ = std::fs::remove_file(&ckpt);
}
