//! NTFF unit validation: center-fed PEC dipole in free space.
//!
//! A short z-directed dipole (PEz arms + Norton gap port, the
//! production feed path) has an analytically known pattern:
//! Dmax ≈ 1.5–1.6, maximum in the equatorial plane (θ=90°), nulls
//! on the dipole axis (θ=0°, 180°), and box Poynting flux ≈
//! sphere-integrated radiated power, both positive. Any sign,
//! mirroring, or collocation defect in `ntff` shows up here in
//! seconds — the efs validation failure (Dmax 3.60 dBi vs 6.92,
//! negative flux, downward peak) is what this isolates.
//!
//! Grid: 64³ cells of 2.5 mm (λ=100 mm at 3 GHz), CPML 12.
//! Dipole: Ez PEC at (32,32), k=28..=31 and 33..=36, gap k=32 —
//! total length ≈ 22.5 mm ≈ 0.225λ. NTFF box planes at index
//! 20/44 on all axes (±30 mm from center).

use fdtd::cpml::Cpml;
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::ntff::{Ntff, NtffBox};
use fdtd::solver::Fdtd;
use fdtd::source::{cdiv, DftProbe, Feed, GaussianSource, ResistiveSource};

fn main() {
    let (nx, ny, nz) = (64usize, 64usize, 64usize);
    let d = 2.5e-3f32;
    let grid = YeeGrid::new(nx, ny, nz, d, d, d);
    let cfl: f32 = std::env::var("DIP_CFL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.9);
    let dt = grid.courant_dt(cfl);
    let mut mat = Materials::vacuum(nx, ny, nz, dt);
    let (ci, cj, ck) = (32usize, 32usize, 32usize);
    for k in (28..=31).chain(33..=36) {
        mat.set_pec(2, grid.ez_idx(ci, cj, k));
    }
    let feed = Feed {
        src_i: ci,
        src_j: cj,
        src_k: ck,
        k_loop: ck,
        hx_j_lo: cj - 2,
        hx_j_hi: cj + 3,
        hx_i0: ci - 2,
        hx_i1: ci + 3,
        hy_i_lo: ci - 2,
        hy_i_hi: ci + 3,
        hy_j0: cj - 2,
        hy_j1: cj + 3,
    };
    let cpml = Cpml::new(&grid, 12, dt);
    let source = GaussianSource::new(3.0e9, 1.0);
    let rsource = ResistiveSource { rs: 50.0 };
    let probe = DftProbe::new(vec![3.0e9], dt);
    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, None);

    let origin = (-(nx as f64) * d as f64 / 2.0, -(ny as f64) * d as f64 / 2.0, -(nz as f64) * d as f64 / 2.0);
    let mut ntff = Ntff::new(
        &sim.grid,
        NtffBox { i0: 20, i1: 44, j0: 20, j1: 44, k0: 20, k1: 44 },
        origin,
        vec![3.0e9],
        dt,
    );
    println!("ntff samples: {}", ntff.n_samples());

    let n_steps = (6.0e-9 / dt as f64).ceil() as usize;
    // Record gathered values at three box samples for an offline
    // DFT cross-check: +z center (300), +x center (1452), +y (2604).
    let watch = [300usize, 1452, 2604];
    let mut rec: Vec<Vec<[f64; 4]>> = watch.iter().map(|_| Vec::new()).collect();
    for s in 0..n_steps {
        sim.step();
        ntff.accumulate(&sim.grid, sim.sim_time() as f64);
        for (wi, &si) in watch.iter().enumerate() {
            rec[wi].push(ntff.gather_at(&sim.grid, si));
        }
        if s < 5 || s % 200 == 199 {
            let mx = sim.grid.ez.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let (vs, is) = sim.probe.series();
            let n = vs.len();
            println!(
                "step {} max|Ez|={:.3e} V={:.3e} I={:.3e}",
                s + 1,
                mx,
                vs[n - 1],
                is[n - 1]
            );
        }
    }
    // Offline DFT of the recorded series (same kernel as Ntff).
    let w = 2.0 * std::f64::consts::PI * 3.0e9;
    let dtf = dt as f64;
    for (wi, &si) in watch.iter().enumerate() {
        let n = rec[wi].len() as f64;
        let mut off = [(0.0f64, 0.0f64); 4];
        for (m, v) in rec[wi].iter().enumerate() {
            let te = (m + 1) as f64 * dtf;
            for c in 0..4 {
                let t = if c < 2 { te } else { te - 0.5 * dtf };
                let (co, si_) = ((w * t).cos(), -(w * t).sin());
                off[c].0 += v[c] * co * 2.0 / n;
                off[c].1 += v[c] * si_ * 2.0 / n;
            }
        }
        let on = ntff.sample_phasors(0, si);
        println!("watch si={si}:");
        for c in 0..4 {
            println!(
                "  comp{c}: ntff=({:.4e},{:.4e}) offline=({:.4e},{:.4e})",
                on[c].0, on[c].1, off[c].0, off[c].1
            );
        }
    }
    let pf = ntff.box_flux_per_face(0);
    println!(
        "flux per face [+z,-z,+x,-x,+y,-y]: {}",
        pf.iter().map(|x| format!("{x:.3e}")).collect::<Vec<_>>().join(" ")
    );
    println!("steps {n_steps}  tail={:.2e}", sim.probe.tail_ratio());

    let (vf, cf) = sim.probe.dft();
    let (zre, zim) = cdiv(vf[0].0, vf[0].1, cf[0].0, cf[0].1);
    println!("Zin(3 GHz) = {zre:.2} + j{zim:.2} ohm");

    let flux = ntff.box_flux(0);
    let (dlin, dbi, prad, th, ph) = ntff.directivity(0, 91, 180);
    println!(
        "NTFF dipole: box flux={flux:.4e} W  sphere Prad={prad:.4e} W  ratio={:.3}",
        flux / prad
    );
    println!(
        "Dmax = {dbi:.2} dBi (lin {dlin:.3}) at theta={:.1} phi={:.1}",
        th.to_degrees(),
        ph.to_degrees()
    );
    let (thetas, _phis, u) = ntff.far_field(0, 91, 180);
    let umax = u.iter().cloned().fold(0.0f64, f64::max);
    let cut: Vec<String> = (0..=18)
        .map(|ti| {
            format!(
                "{:.0}:{:.1}",
                thetas[ti * 5].to_degrees(),
                10.0 * (u[ti * 5 * 180] / umax).log10()
            )
        })
        .collect();
    println!("cut phi=0: {}", cut.join(" "));
}
