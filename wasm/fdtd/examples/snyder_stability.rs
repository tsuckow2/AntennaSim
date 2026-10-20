//! Late-time stability probe for the production Snyder model
//! (`build_snyder`: PEC post + volumetric ground, base grid).
//!
//! The WebGPU bench page blows up at ~16 ns at every CFL (localized
//! parasitic mode, growth ~x1.0006/step, onset receding as CFL drops);
//! the CPU cross-check was never done. This example runs the CPU
//! solver long and logs, every DIAG_EVERY steps: ezAbsSum (the page's
//! blow-up metric), the argmax|Ez| cell with its region (CPML shell vs
//! interior, z-band), and the CPML/interior split of the |Ez| mass —
//! enough to localize the mode and measure its growth rate.
//!
//! Usage: `cargo run --release --example snyder_stability -- [t_ns]`
//! Env: SNYDER_STAB_CFL (default 0.2), SNYDER_STAB_EVERY (default 500),
//! SNYDER_STAB_CKPT (default ./snyder_stab_ckpt.bin),
//! SNYDER_STAB_CKPT_EVERY (default 5000). Checkpoints (SSCK1: fields +
//! CPML psi + step clock, config-guarded) make multi-hour probes
//! survive VM restarts; the file is removed on completion.

use fdtd::cpml::{Cpml, PmlFaceData};
use fdtd::geometry::{build_snyder, DX, DY, DZ, NX, NY, NZ, X0, Y0, Z0};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{DftProbe, GaussianSource, ResistiveSource};
use std::io::Write;
use std::time::Instant;

const NPML: usize = 12;
const K_GROUND: usize = 31;
const K_PATCH: usize = 34;

// --- Checkpointing (SSCK1), mirroring snyder_ntff's SNCK1: field
// arrays + CPML psi + step clock. A 16 ns probe is ~4.5 h of CPU and
// this VM restarts often enough to kill uncheckpointed attempts;
// with a dump every CKPT_EVERY steps a restart costs minutes.
fn put_f32s(buf: &mut Vec<u8>, a: &[f32]) {
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
}

fn save_ckpt(path: &str, sim: &Fdtd, step_done: usize, total_steps: usize, cfl: f32, rs: f32) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"SSCK1");
    buf.extend_from_slice(&(step_done as u64).to_le_bytes());
    buf.extend_from_slice(&(total_steps as u64).to_le_bytes());
    buf.extend_from_slice(&(cfl.to_bits() as u64).to_le_bytes());
    buf.extend_from_slice(&(rs.to_bits() as u64).to_le_bytes());
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
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &buf).expect("write ckpt");
    std::fs::rename(&tmp, path).expect("rename ckpt");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t_ns: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20.0);
    let cfl: f32 = std::env::var("SNYDER_STAB_CFL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.2);
    let every: usize = std::env::var("SNYDER_STAB_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let dt = grid.courant_dt(cfl);
    let total_steps = (t_ns * 1e-9 / dt as f64) as usize;
    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let (feed, _wire) = build_snyder(&grid, &mut mat, dt);
    let (fi, fj, fk) = (feed.src_i, feed.src_j, feed.src_k);
    let cpml = Cpml::new(&grid, NPML, dt);
    let source = GaussianSource::new(2.5e9, 1.0);
    let rs: f32 = std::env::var("SNYDER_STAB_RS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);
    let rsource = ResistiveSource { rs };
    let probe = DftProbe::new(vec![2.01e9], dt);
    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, None);

    println!(
        "snyder_stability: cfl={cfl} rs={rs} dt={dt:.4e} steps={total_steps} (t={t_ns} ns) diag every {every}; \
         feed cell ({fi},{fj},{fk})"
    );
    std::io::stdout().flush().ok();

    let ckpt = std::env::var("SNYDER_STAB_CKPT")
        .unwrap_or_else(|_| "snyder_stab_ckpt.bin".to_string());
    let ckpt_every: usize = std::env::var("SNYDER_STAB_CKPT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);
    let mut step = 0usize;
    if let Ok(bytes) = std::fs::read(&ckpt) {
        let mut rd = Rd { b: &bytes, p: 0 };
        assert_eq!(&bytes[0..5], b"SSCK1", "ckpt magic");
        rd.p = 5;
        let step_done = rd.u64() as usize;
        let total = rd.u64() as usize;
        let cfl_ok = rd.u64() == cfl.to_bits() as u64;
        let rs_ok = rd.u64() == rs.to_bits() as u64;
        if total == total_steps && cfl_ok && rs_ok && step_done < total_steps {
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
            sim.restore_clock(step_done);
            step = step_done;
            println!("RESUMED from checkpoint at step {step}/{total_steps}");
            std::io::stdout().flush().ok();
        } else {
            println!("ckpt config mismatch; starting fresh");
            std::io::stdout().flush().ok();
        }
    }
    let t0 = Instant::now();
    loop {
        let ez = &sim.grid.ez;
        let mut abs_sum = 0.0f64;
        let mut sum_pml = 0.0f64;
        let mut sum_int = 0.0f64;
        let mut am_idx = 0usize;
        let mut am_val = 0.0f32;
        for (idx, &v) in ez.iter().enumerate() {
            let a = v.abs();
            abs_sum += a as f64;
            if a > am_val {
                am_val = a;
                am_idx = idx;
            }
            let i = idx % NX;
            let jk = idx / NX;
            let j = jk % NY;
            let k = jk / NY;
            if i < NPML || i >= NX - NPML || j < NPML || j >= NY - NPML || k < NPML || k > NZ - NPML {
                sum_pml += a as f64;
            } else {
                sum_int += a as f64;
            }
        }
        // Feed-cell value and the strongest cell AWAY from the feed
        // (3x3x3 exclusion): separates a source-cell mode from a mode
        // living elsewhere in the domain.
        let feed_ez = ez[sim.grid.ez_idx(fi, fj, fk)];
        let mut ex_idx = 0usize;
        let mut ex_val = 0.0f32;
        for (idx, &v) in ez.iter().enumerate() {
            let i = idx % NX;
            let jk = idx / NX;
            let j = jk % NY;
            let k = jk / NY;
            if i.abs_diff(fi) <= 1 && j.abs_diff(fj) <= 1 && k.abs_diff(fk) <= 1 {
                continue;
            }
            if v.abs() > ex_val {
                ex_val = v.abs();
                ex_idx = idx;
            }
        }
        let (ei, ej, ek) = (ex_idx % NX, (ex_idx / NX) % NY, ex_idx / (NX * NY));
        let ai = am_idx % NX;
        let aj = (am_idx / NX) % NY;
        let ak = am_idx / (NX * NY);
        let in_pml = ai < NPML || ai >= NX - NPML || aj < NPML || aj >= NY - NPML || ak < NPML || ak > NZ - NPML;
        let zband = if ak < K_GROUND {
            "below-ground"
        } else if ak <= K_PATCH {
            "substrate"
        } else {
            "above-patch"
        };
        let elapsed = t0.elapsed().as_secs_f64();
        let mcells = (NX * NY * NZ) as f64 * step.max(1) as f64 / elapsed / 1e6;
        println!(
            "step {step} t={:.3} ns ezAbsSum={:.6e} argmax={:.6e} at ({ai},{aj},{ak}) \
             x={:.2} y={:.2} z={:.2} mm pml={in_pml} band={zband} sumPml={:.6e} sumInt={:.6e} \
             feedEz={:.6e} argmaxExFeed={:.6e} at ({ei},{ej},{ek}) {:.1} Mcell/s",
            step as f64 * dt as f64 * 1e9,
            abs_sum,
            am_val,
            (X0 + ai as f32 * DX) * 1e3,
            (Y0 + (aj as f32 + 0.5) * DY) * 1e3,
            (Z0 + ak as f32 * DZ) * 1e3,
            sum_pml,
            sum_int,
            feed_ez,
            ex_val,
            mcells,
        );
        std::io::stdout().flush().ok();
        if step >= total_steps || abs_sum > 1e16 {
            println!("done: step {step}, ezAbsSum={abs_sum:.6e}");
            std::fs::remove_file(&ckpt).ok();
            break;
        }
        let n = every.min(total_steps - step);
        for _ in 0..n {
            sim.step();
        }
        step += n;
        if step % ckpt_every == 0 {
            save_ckpt(&ckpt, &sim, step, total_steps, cfl, rs);
        }
    }
}
