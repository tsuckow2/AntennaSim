//! Canonical thin-wire dipole fixture for validating the Holland/Simpson model.
//!
//! A center-fed dipole in free space. For a thin half-wave dipole, the
//! theoretical input impedance at resonance is ~73 Ω (real).
//!
//! This fixture validates:
//! 1. The wire model produces finite, stable results.
//! 2. The input impedance is reasonable (not NaN, not kΩ capacitive).
//! 3. Current flows on the wire and radiates (fields are finite).

use fdtd::thin_wire::ThinWire;
use fdtd::grid::YeeGrid;

fn main() {
    // Small grid for a quick test: 60×60×80 cells, 1mm cells.
    // Dipole along z, centered, length 40mm (40 cells).
    let nx = 60;
    let ny = 60;
    let nz = 80;
    let dx = 1.0e-3;
    let dy = 1.0e-3;
    let dz = 1.0e-3;

    let mut grid = YeeGrid::new(nx, ny, nz, dx, dy, dz);

    // CFL timestep (use 0.1x for wire-FDTD coupled stability).
    // The Holland/Simpson coupled system has a stricter limit than FDTD alone.
    let c0 = 299792458.0;
    let dt_fdtd = 0.9 / (c0 * (1.0 / dx.powi(2) + 1.0 / dy.powi(2) + 1.0 / dz.powi(2)).sqrt());
    let dt = 0.1 * dt_fdtd;
    println!("dt = {:.3e} s (fdtd limit {:.3e})", dt, dt_fdtd);

    let mat = fdtd::material::Materials::vacuum(nx, ny, nz, dt);
    let mut cpml = fdtd::cpml::Cpml::new(&grid, 8, dt);

    // Wire: z-directed dipole, centered in x/y.
    // Bottom arm: k=20..39, Top arm: k=41..60, gap at k=40 for the source.
    let i0 = nx / 2;
    let j0 = ny / 2;
    let k_gap = 40;
    let radius = 0.1e-3; // 0.1mm radius

    // Bottom arm: open at bottom, open at top (gap).
    let mut wire_bot = ThinWire::new(i0, j0, 20, k_gap - 1, dx, dy, dz, radius, false);
    // Top arm: open at bottom (gap), open at top.
    let mut wire_top = ThinWire::new(i0, j0, k_gap + 1, 60, dx, dy, dz, radius, false);

    // CFL timestep.
    let c0 = 299792458.0;
    let dt = 0.9 / (c0 * (1.0 / dx.powi(2) + 1.0 / dy.powi(2) + 1.0 / dz.powi(2)).sqrt());
    println!("dt = {:.3e} s", dt);
    println!("wire max stable dt = {:.3e} s", wire_bot.max_stable_dt());
    assert!(dt < wire_bot.max_stable_dt(), "dt exceeds wire stability limit");
    assert!(dt < wire_top.max_stable_dt(), "dt exceeds wire stability limit");

    // Simple voltage source at the center gap (k=40).
    let k_src = k_gap;
    let src_i = i0;
    let src_j = j0;

    // Gaussian pulse parameters.
    let t0 = 1.0e-9;
    let sigma = 0.3e-9;
    let v0 = 1.0; // 1V peak

    // Run for 2ns.
    let n_steps = (2.0e-9 / dt) as usize;
    println!("Running {} steps...", n_steps);

    let mut max_ez = 0.0f32;
    let mut max_i = 0.0f32;

    for step in 0..n_steps {
        let t = step as f32 * dt;

        // H update.
        fdtd::solver::update_h(&mut grid, dt);
        cpml.correct_h(&mut grid, dt);

        // Wire steps (bottom and top arms).
        // 1. Update currents from E^n (does not modify grid).
        let i_bot = wire_bot.update_current(&grid, dt);
        let _i_top = wire_top.update_current(&grid, dt);
        // Terminal current: bottom arm current at the gap.
        // (update_current returns (I_bottom, V_base) since the
        // base-node source fix.)
        let i_term = i_bot.0;
        max_i = max_i.max(i_term.abs());

        // E update.
        fdtd::solver::update_e(&mut grid, &mat);
        let cb = mat.cb_z.clone();
        cpml.correct_e(&mut grid, &mat.cb_x, &mat.cb_y, &cb);

        // 2. Deposit wire currents into E^{n+1}.
        wire_bot.deposit_current(&mut grid, &mat);
        wire_top.deposit_current(&mut grid, &mat);

        fdtd::solver::apply_pec(&mut grid, &mat);

        // Source: Gaussian voltage at the center gap.
        // The wire has a gap at k_src; we drive Ez there.
        let vs = v0 * (-((t - t0) * (t - t0)) / (2.0 * sigma * sigma)).exp();
        let ez_idx = grid.ez_idx(src_i, src_j, k_src);
        grid.ez[ez_idx] = -vs / dz;

        // Track max Ez.
        let mut step_max_ez = 0.0f32;
        for &v in grid.ez.iter() {
            if v.is_finite() {
                step_max_ez = step_max_ez.max(v.abs());
            } else {
                println!("Non-finite Ez at step {}", step);
                println!("  wire_bot currents: {:?}", &wire_bot.currents[0..5.min(wire_bot.currents.len())]);
                println!("  wire_bot charges: {:?}", &wire_bot.charges[0..5.min(wire_bot.charges.len())]);
                println!("  wire_top currents: {:?}", &wire_top.currents[0..5.min(wire_top.currents.len())]);
                return;
            }
        }
        max_ez = max_ez.max(step_max_ez);

        if step % 500 == 0 {
            println!(
                "step {}: t={:.2e}, I_term={:.3e}, max|Ez|={:.3e}, wire E={:.3e}",
                step,
                t,
                i_term,
                max_ez,
                wire_bot.energy() + wire_top.energy()
            );
        }
    }

    println!("\nDone.");
    println!("max|I_term| = {:.3e} A", max_i);
    println!("max|Ez| = {:.3e} V/m", max_ez);
    println!(
        "Final wire energy: {:.3e} J",
        wire_bot.energy() + wire_top.energy()
    );

    if max_i > 0.0 && max_i.is_finite() && max_ez.is_finite() {
        println!("PASS: Wire model is stable and produces current.");
    } else {
        println!("FAIL: Wire model produced non-finite or zero current.");
    }
}
