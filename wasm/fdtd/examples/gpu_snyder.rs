//! GPU (wgpu) Snyder validation driver.
//!
//! Modes:
//!   gpu_snyder parity [steps]   — run CPU and GPU in lockstep from the
//!                                 same initial state; report field/wire/
//!                                 probe agreement after `steps` steps.
//!   gpu_snyder run [t_ns]       — GPU-only run for `t_ns` ns; print the
//!                                 Zin/S11 table like snyder_fdtd.rs.
//!
//! Both build the exact Snyder configuration of snyder_fdtd.rs
//! (0.5 mm grid, dt = 0.1×CFL, CPML 8, Gaussian peak 2.5 GHz, Rs = 50 Ω).

use fdtd::cpml::Cpml;
use fdtd::geometry::*;
use fdtd::gpu::GpuFdtd;
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{cdiv, DftProbe, GaussianSource, ResistiveSource};

fn build_cpu() -> (Fdtd, f32, Vec<f64>) {
    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let cfl: f32 = std::env::var("GPU_CFL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    let dt = grid.courant_dt(cfl);
    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let (feed, thin_wire) = build_snyder(&grid, &mut mat, dt);
    // SNYDER_POST=1: grid-faithful PEC post + Norton gap port instead
    // of the thin-wire filament (same option as snyder_diag.rs).
    let wire_arg = if std::env::var("SNYDER_POST").is_ok() {
        let (pi, pj) = (feed.src_i, feed.src_j);
        for (di, dj) in [(0isize, 0isize), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (ii, jj) = ((pi as isize + di) as usize, (pj as isize + dj) as usize);
            for k in 33..=34 {
                mat.set_pec(2, grid.ez_idx(ii, jj, k));
            }
        }
        None
    } else {
        Some(thin_wire)
    };
    // SNYDER_THICKGND=1: volumetric ground (Ez PEC at k=30,31 over the
    // ground footprint) — kills the ground-sheet straddle mirror that
    // cancelled the port voltage (see snyder_diag.rs / MEMORY.md).
    if std::env::var("SNYDER_THICKGND").is_ok() {
        for j in 14..218usize {
            for i in 14..218usize {
                mat.set_pec(2, grid.ez_idx(i, j, 31));
                mat.set_pec(2, grid.ez_idx(i, j, 30));
            }
        }
    }
    let cpml = Cpml::new(&grid, 12, dt); // npml 12: 8 let a PEC-channel grazing mode grow (see cpml.rs)
    let source = GaussianSource::new(2.5e9, 1.0);
    let rs: f32 = std::env::var("GPU_RS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);
    let rsource = ResistiveSource { rs };
    let freqs: Vec<f64> = (0..=100).map(|m| 2.0e9 + m as f64 * 10e6).collect();
    let probe = DftProbe::new(freqs.clone(), dt);
    let sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, wire_arg);
    (sim, dt, freqs)
}

fn diff_stats(name: &str, a: &[f32], b: &[f32]) {
    let mut max_a = 0.0f32;
    let mut max_d = 0.0f32;
    let mut argmax = 0usize;
    let mut sum = 0.0f64;
    for (n, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        max_a = max_a.max(x.abs());
        let d = (x - y).abs();
        if d > max_d {
            max_d = d;
            argmax = n;
        }
        sum += (d as f64) * (d as f64);
    }
    let rms = (sum / a.len() as f64).sqrt();
    println!(
        "{name:<10} max|CPU|={max_a:.4e}  max|diff|={max_d:.4e} at [{argmax}] cpu={:.4e} gpu={:.4e}  rms_diff={rms:.4e}  rel={:.3e}",
        a[argmax], b[argmax],
        if max_a > 0.0 { max_d / max_a } else { 0.0 }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("parity");

    if mode == "parity" {
        let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
        let (mut cpu, dt, _freqs) = build_cpu();
        let mut gpu = pollster::block_on(GpuFdtd::try_new(&cpu, steps))
            .expect("no GPU adapter available");
        println!("adapter: {}", gpu.adapter_name());
        println!("parity run: {steps} steps, dt={dt:.4e}");
        let t0 = std::time::Instant::now();
        gpu.run(steps);
        let gpu_s = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        cpu.run(steps);
        let cpu_s = t0.elapsed().as_secs_f64();
        println!("gpu {gpu_s:.2}s  cpu {cpu_s:.2}s");

        let fields = pollster::block_on(gpu.download_fields());
        diff_stats("ex", &cpu.grid.ex, &fields[0]);
        diff_stats("ey", &cpu.grid.ey, &fields[1]);
        diff_stats("ez", &cpu.grid.ez, &fields[2]);
        diff_stats("hx", &cpu.grid.hx, &fields[3]);
        diff_stats("hy", &cpu.grid.hy, &fields[4]);
        diff_stats("hz", &cpu.grid.hz, &fields[5]);

        if let (Some(tw), Some((cur, chg))) =
            (&cpu.thin_wire, pollster::block_on(gpu.download_wire()))
        {
            diff_stats("wire_I", &tw.currents, &cur);
            diff_stats("wire_q", &tw.charges, &chg);
        }
        let (gv, gi, _gf) = pollster::block_on(gpu.download_probe());
        println!("gpu probe: steps={}", gv.len());
        // Series-level localization: first index where GPU and CPU V/I
        // series diverge materially, with surrounding values.
        let (cv, ci) = cpu.probe.series();
        for s in 0..gv.len().min(cv.len()) {
            if (gv[s] - cv[s]).abs() > 1e-7 || (gi[s] - ci[s]).abs() > 1e-9 {
                println!(
                    "series diverge at step {s}: V cpu={:.6e} gpu={:.6e} | I cpu={:.6e} gpu={:.6e}",
                    cv[s], gv[s], ci[s], gi[s]
                );
                let lo = s.saturating_sub(2);
                let hi = (s + 3).min(gv.len());
                for m in lo..hi {
                    println!("  step {m}: V cpu={:.6e} gpu={:.6e} | I cpu={:.6e} gpu={:.6e}", cv[m], gv[m], ci[m], gi[m]);
                }
                break;
            }
        }
        // End-to-end: DFT the GPU series with the crate's probe math and
        // compare against the CPU probe's DFT (spectra + Zin).
        let mut gprobe = DftProbe::new(_freqs.clone(), dt);
        for s in 0..gv.len() {
            gprobe.push(gv[s], gi[s]);
        }
        let (gvf, gcf) = gprobe.dft();
        let (cvf, ccf) = cpu.probe.dft();
        let mut max_dv = 0.0f64;
        let mut max_di = 0.0f64;
        let mut max_z = 0.0f64;
        let mut max_zref = 0.0f64;
        for m in 0.._freqs.len() {
            let dv = (gvf[m].0 - cvf[m].0).hypot(gvf[m].1 - cvf[m].1);
            let di = (gcf[m].0 - ccf[m].0).hypot(gcf[m].1 - ccf[m].1);
            max_dv = max_dv.max(dv);
            max_di = max_di.max(di);
            let (gzr, gzi) = cdiv(gvf[m].0, gvf[m].1, gcf[m].0, gcf[m].1);
            let (czr, czi) = cdiv(cvf[m].0, cvf[m].1, ccf[m].0, ccf[m].1);
            max_z = max_z.max((gzr - czr).hypot(gzi - czi));
            max_zref = max_zref.max(czr.hypot(czi));
        }
        println!("DFT: max|dV|={max_dv:.4e}  max|dI|={max_di:.4e}  max|dZin|={max_z:.4e} (max|Zin|={max_zref:.2})");
        return;
    }

    if mode == "run" {
        let t_ns: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10.0);
        let (cpu, dt, freqs) = build_cpu();
        let n_steps = (t_ns * 1e-9 / dt as f64).ceil() as usize;
        let mut gpu = pollster::block_on(GpuFdtd::try_new(&cpu, n_steps))
            .expect("no GPU adapter available");
        println!("adapter: {}", gpu.adapter_name());
        println!("dt = {dt:.4e} s, running {n_steps} steps ({t_ns} ns)...");

        // Start-up gate: on dell's shared iGPU, a process occasionally
        // starts in a state where the source/wire dispatches silently
        // stop landing after ~2 steps (V series all zero) with no wgpu
        // error. That corruption is total, so a short CPU-vs-GPU
        // comparison catches it: run both 256 steps and require the
        // recorded (V, I) to agree relatively. (The base-node source's
        // V legitimately lags vs(t) while the node charges, so the old
        // "V tracks vs" check no longer applies.) Exit code 3 lets a
        // wrapper retry with a fresh process.
        gpu.run(256);
        let (gv, gi, _gf) = pollster::block_on(gpu.download_probe());
        let (mut gate_cpu, _, _) = build_cpu();
        gate_cpu.run(256);
        let (cv, ci) = gate_cpu.probe.series();
        let mut gate_fail = false;
        for s in 0..gv.len() {
            let dv = (gv[s] - cv[s]).abs() > 1e-2 * cv[s].abs() + 1e-12;
            let di = (gi[s] - ci[s]).abs() > 1e-2 * ci[s].abs() + 1e-12;
            if dv || di || !gv[s].is_finite() || !gi[s].is_finite() {
                gate_fail = true;
            }
        }
        if gate_fail {
            eprintln!("GPU START GATE FAILED: GPU series departs from CPU in the first 256 steps (corrupt GPU start; retry)");
            std::process::exit(3);
        }
        println!("start gate ok (GPU matches CPU for 256 steps)");

        let t0 = std::time::Instant::now();
        let chunk = (n_steps / 20).max(1);
        let mut done = gpu.steps_done();
        while done < n_steps {
            let n = chunk.min(n_steps - done);
            gpu.run(n);
            done += n;
            // Watchdog: while the pulse is active (t < 2 ns), a healthy
            // V series never contains a long run of exact zeros, and is
            // always finite. The iGPU flake kills the source outright,
            // which shows up here immediately. Exit 4 = restart the run.
            let (pv, pi, _pf) = pollster::block_on(gpu.download_probe());
            let nonfinite = pv.iter().chain(pi.iter()).any(|x| !x.is_finite());
            let t_now = done as f64 * dt as f64;
            let zero_bad = t_now < 2e-9 && {
                let start = pv.len().saturating_sub(n);
                pv[start..].iter().filter(|&&x| x == 0.0).count() > n / 2
            };
            let bad = nonfinite || zero_bad;
            if bad {
                eprintln!("GPU WATCHDOG: probe corrupt at step {done} (restart the run)");
                std::process::exit(4);
            }
            let el = t0.elapsed().as_secs_f64();
            println!(
                "  step {done}/{n_steps}  t={:.2} ns  ({:.1}s, {:.0} steps/s)",
                done as f64 * dt as f64 * 1e9,
                el,
                done as f64 / el
            );
        }
        let cells = (NX * NY * NZ) as f64;
        let el = t0.elapsed().as_secs_f64();
        println!(
            "wall: {el:.1}s  ({:.1} Mcell/s incl. CPML+wire+source)",
            cells * n_steps as f64 / el / 1e6
        );
        let (v, i, v_field) = pollster::block_on(gpu.download_probe());
        if let Ok(path) = std::env::var("GPU_DUMP_SERIES") {
            let mut out = String::new();
            for s in 0..v.len() {
                out.push_str(&format!(
                    "{}, {:.6e}, {:.6e}, {:.6e}\n",
                    s, v[s], i[s], v_field[s]
                ));
            }
            std::fs::write(&path, out).expect("write series");
            println!("wrote series to {path}");
        }
        // Zin from BOTH voltage definitions against the same current:
        // (a) the wire model's base-node voltage q0/C_node, (b) the
        // physical gap-field voltage -Ez*dz at the base cell. Whichever
        // tracks the FEM anchor discriminates the port-voltage model.
        for (label, vv) in [("Vnode", &v), ("Vfield", &v_field)] {
            let mut probe = DftProbe::new(freqs.clone(), dt);
            for s in 0..vv.len() {
                probe.push(vv[s], i[s]);
            }
            let (vf, cf) = probe.dft();
            println!("# Zin[{label}]  MHz  Re(Zin)  Im(Zin)  |S11|_dB");
            for (m, &f) in freqs.iter().enumerate() {
                let (vr, vi) = vf[m];
                let (ir, ii) = cf[m];
                let (zr, zi) = cdiv(vr, vi, ir, ii);
                let (sr, si) = cdiv(zr - 50.0, zi, zr + 50.0, zi);
                let s11_db = 10.0 * (sr * sr + si * si).log10();
                println!(
                    "{:.0}  {:.4}  {:+.4}  {:.2}",
                    f / 1e6,
                    zr,
                    zi,
                    s11_db
                );
            }
        }
        return;
    }

    eprintln!("usage: gpu_snyder parity [steps] | run [t_ns]");
    std::process::exit(2);
}
