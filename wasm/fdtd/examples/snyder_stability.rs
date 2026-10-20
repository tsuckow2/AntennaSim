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
//! Env: SNYDER_STAB_CFL (default 0.2), SNYDER_STAB_EVERY (default 500).

use fdtd::cpml::Cpml;
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
    let t0 = Instant::now();
    let mut step = 0usize;
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
            break;
        }
        let n = every.min(total_steps - step);
        for _ in 0..n {
            sim.step();
        }
        step += n;
    }
}
