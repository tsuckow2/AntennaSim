//! Quick Snyder feed test: 2ns run with the Luebbers resistive source.
//!
//! Validates the full feed chain: via–patch overlap, tight-loop current,
//! resistive source, and V/I extraction. Expects Re(Zin) > 0 and a
//! reactance magnitude far below the old hard-source −j1100 Ω.
use fdtd::geometry::build_snyder;
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::{apply_pec, update_e, update_h};
use fdtd::source::{cdiv, DftProbe, GaussianSource, ResistiveSource};

fn main() {
    let mut grid = YeeGrid::new(232, 232, 61, 0.5e-3, 0.5e-3, 0.525e-3);
    let dt = grid.courant_dt(0.99);
    let mut mat = Materials::vacuum(232, 232, 61, dt);
    let feed = build_snyder(&grid, &mut mat, dt);

    let source = GaussianSource::new(2.5e9, 1.0);
    let rsource = ResistiveSource { rs: 50.0 };
    let freqs: Vec<f64> = vec![2.0e9, 2.4e9, 2.5e9, 2.6e9, 3.0e9];
    let mut probe = DftProbe::new(freqs.clone(), dt);

    let n_steps = 2065; // 2ns
    let mut t = 0.0f32;
    let mut max_i = 0.0f32;

    for step in 0..n_steps {
        update_h(&mut grid, dt);
        // Tight 1-cell loop current at the H-step (+z upward).
        let i_up = feed.measure_i_tight(&grid);
        if i_up.abs() > max_i {
            max_i = i_up.abs();
        }

        update_e(&mut grid, &mat);
        apply_pec(&mut grid, &mat);

        t += dt;
        let vs = source.value(t);
        let (v_term, i_term) = rsource.apply(&mut grid, &feed, vs, i_up);
        probe.push(v_term, i_term);

        if step % 1000 == 0 {
            let max_e = grid.ez.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            println!(
                "step {}/{}, max|Ez|={:.2e}, max|I|={:.2e}, finite={}",
                step,
                n_steps,
                max_e,
                max_i,
                max_e.is_finite()
            );
            if !max_e.is_finite() {
                println!("UNSTABLE!");
                return;
            }
        }
    }

    let (vf, cf) = probe.dft();
    println!("\nFreq (GHz) | Zin (Ω)  [Luebbers Rs=50, tight loop, actual V]");
    for (idx, f) in freqs.iter().enumerate() {
        let (v_re, v_im) = vf[idx];
        let (i_re, i_im) = cf[idx];
        let (z_re, z_im) = cdiv(v_re, v_im, i_re, i_im);
        println!("{:10.2} | {:8.1} {:+9.1}j", f / 1e9, z_re, z_im);
    }
    println!("max|I| = {:.3e} A", max_i);
}
