//! efs/openEMS patch-antenna cross-validation fixture.
//!
//! Replicates the geometry of efs `examples/patch_antenna.mjs` (itself a
//! port of the openEMS Simple_Patch_Antenna tutorial) in OUR solver, with
//! a lumped-port feed comparable to openEMS's: a 1-cell resistive
//! (Thevenin, 50 Ω) source gap at the ground end of a PEC post — the same
//! feed machinery the Snyder fixture uses, minus the Holland wire.
//!
//! efs reference (~/workspace/efs-oracle/PATCH_REFERENCE.md):
//!   f0 = 2.430 GHz, S11min = −24.90 dB, Zin(f0) = 45.45 + j2.96 Ω,
//!   −10 dB BW 2.415–2.450 GHz, Dmax = 6.92 dBi (no NF2FF here).
//!
//! Geometry (efs §1): patch 32 (x) × 40 (y) mm PEC at z = 1.524 mm;
//! substrate 60 × 60 × 1.524 mm, εr = 3.38, σ = 4.61e-4 S/m (constant);
//! ground PEC 60 × 60 at z = 0; port at (x, y) = (−6, 0) mm.
//!
//! Deliberate differences from the efs run (documented for the writeup):
//! - Uniform mesh: dx = dy = 2 mm, dz = 0.381 mm (substrate exactly
//!   4 cells, matching efs's z resolution; efs used a smoothed automesh).
//! - CPML boundaries (efs used MUR) and a smaller air box: ±61 mm in
//!   x/y, 4.6 mm below ground, ~40 mm above the patch.
//! - Feed: PEC post + 1-cell gap at the ground (openEMS applies the
//!   port across the full substrate height; expect a small feed-
//!   inductance difference in Zin, f0 should transfer).
//! - No thin wire -> full CFL is legitimate; dt uses 0.9 × Courant.
//!
//! Usage: `cargo run --release --example efs_patch -- [t_end_ns] [csv_out]`

use fdtd::cpml::Cpml;
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{cdiv, DftProbe, Feed, GaussianSource, ResistiveSource};

const DX: f32 = 2.0e-3;
const DY: f32 = 2.0e-3;
const NX: usize = 61;
const NY: usize = 61;
const X0: f32 = -61.0e-3;
const Y0: f32 = -61.0e-3;
/// Half-integer offset (Snyder convention): Ex/Ey sample planes sit at
/// (k+½)·dz + Z0, so Z0 = −(k_ground+½)·dz puts the ground plane exactly
/// at z = 0 and the patch plane exactly at z = SUB_H. An integer offset
/// would float both PEC sheets half a cell and disconnect the feed post
/// from the patch. DZ/NZ/Z0/k_ground are computed in main from
/// EFS_SUBCELLS (default 8): with only 4 substrate layers the discrete
/// cavity resonates ~17% high (2.86 vs 2.43 GHz — cavity_probe bisection,
/// 2026-10-15); 8 layers lands it at 2.45 GHz.

const SUB_HALF: f32 = 30.0e-3;
const SUB_H: f32 = 1.524e-3;
const PATCH_HX: f32 = 16.0e-3;
const PATCH_HY: f32 = 20.0e-3;
const FEED_X: f32 = -6.0e-3;
const EPS_R: f32 = 3.38;
const SIGMA: f32 = 4.61e-4;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t_end_ns: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let csv_out: Option<String> = args.get(2).cloned();
    let t_end = t_end_ns * 1e-9;

    // Mesh in z: SUB_H split into EFS_SUBCELLS layers (default 8; the
    // historical 4-layer mesh resonates ~17% high — see above).
    let subcells: usize = std::env::var("EFS_SUBCELLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let k_ground: usize = std::env::var("EFS_KG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3 * subcells);
    let NZ: usize = 30 * subcells + 1;
    let DZ: f32 = SUB_H / subcells as f32;
    let Z0: f32 = -(k_ground as f32 + 0.5) * DZ;

    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let cfl: f32 = std::env::var("EFS_CFL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.9);
    let dt = grid.courant_dt(cfl);
    println!("grid {NX}x{NY}x{NZ}  dx={DX:.3e} dy={DY:.3e} dz={DZ:.3e}");
    println!("dt = {dt:.4e} s  (cfl {cfl}, no thin wire)");

    // Snyder index arithmetic: ground Ex/Ey plane k_ground (z=0), patch
    // k_patch = k_ground + subcells (z=1.524), source gap cell Ez
    // k_ground+1, post Ez above it up to k_patch.
    let k_patch = k_ground + subcells;
    let z_ex = |k: usize| (k as f32 + 0.5) * DZ + Z0;
    assert!(z_ex(k_ground).abs() < 1e-6, "ground plane at {}", z_ex(k_ground));
    assert!(
        (z_ex(k_patch) - SUB_H).abs() < 1e-6,
        "patch plane at {}",
        z_ex(k_patch)
    );

    let mut mat = Materials::vacuum(NX, NY, NZ, dt);

    // --- Substrate dielectric (per-component sample-position box tests,
    //     same convention as build_snyder) ---
    for k in 0..NZ {
        let zc = (k as f32 + 0.5) * DZ + Z0; // Ex/Ey plane height
        if zc <= 0.0 || zc >= SUB_H {
            continue;
        }
        for j in 0..NY {
            let y = (j as f32 + 0.5) * DY + Y0;
            if y.abs() > SUB_HALF {
                continue;
            }
            for i in 0..=NX {
                let x = i as f32 * DX + X0;
                if x.abs() <= SUB_HALF {
                    mat.set_dielectric(0, grid.ex_idx(i, j, k), EPS_R, SIGMA, dt);
                }
            }
        }
    }
    for k in 0..NZ {
        let zc = (k as f32 + 0.5) * DZ + Z0;
        if zc <= 0.0 || zc >= SUB_H {
            continue;
        }
        for j in 0..=NY {
            let y = j as f32 * DY + Y0;
            if y.abs() > SUB_HALF {
                continue;
            }
            for i in 0..NX {
                let x = (i as f32 + 0.5) * DX + X0;
                if x.abs() <= SUB_HALF {
                    mat.set_dielectric(1, grid.ey_idx(i, j, k), EPS_R, SIGMA, dt);
                }
            }
        }
    }
    for k in 0..NZ {
        let z = k as f32 * DZ + Z0; // Ez cell bottom edge
        if z < 0.0 || z >= SUB_H {
            continue;
        }
        for j in 0..NY {
            let y = (j as f32 + 0.5) * DY + Y0;
            if y.abs() > SUB_HALF {
                continue;
            }
            for i in 0..NX {
                let x = (i as f32 + 0.5) * DX + X0;
                if x.abs() <= SUB_HALF {
                    mat.set_dielectric(2, grid.ez_idx(i, j, k), EPS_R, SIGMA, dt);
                }
            }
        }
    }

    // --- Ground PEC sheet (z = 0) ---
    for j in 0..NY {
        let y = (j as f32 + 0.5) * DY + Y0;
        if y.abs() > SUB_HALF {
            continue;
        }
        for i in 0..=NX {
            let x = i as f32 * DX + X0;
            if x.abs() <= SUB_HALF {
                mat.set_pec(0, grid.ex_idx(i, j, k_ground));
            }
        }
    }
    for j in 0..=NY {
        let y = j as f32 * DY + Y0;
        if y.abs() > SUB_HALF {
            continue;
        }
        for i in 0..NX {
            let x = (i as f32 + 0.5) * DX + X0;
            if x.abs() <= SUB_HALF {
                mat.set_pec(1, grid.ey_idx(i, j, k_ground));
            }
        }
    }

    // --- Volumetric ground: PEC Ez in the two Ez layers below the
    //     sheet plane (k_ground at z=-dz/2, k_ground-1 at z=-3dz/2) ---
    // A single-plane PEC sheet shares its adjacent H samples with the
    // air region below: the cavity's first Ex/Ey layers are updated
    // from H samples living under the sheet, so energy sloshes into
    // the below-sheet Ez layer (in the driven fixture it carried the
    // largest |Ez| in the grid and the cavity mode's z-profile peaked
    // UNDER the sheet). PEC Ez under the sheet decouples the two
    // sides (cavity_probe CAVITY_THICK_GROUND: below-sheet Ez energy
    // share dropped ~1500x). EFS_THICK_GROUND=0 restores the sheet.
    if std::env::var("EFS_THICK_GROUND").map_or(true, |v| v != "0") {
        for &kk in &[k_ground, k_ground - 1] {
            for j in 0..NY {
                let y = (j as f32 + 0.5) * DY + Y0;
                if y.abs() > SUB_HALF {
                    continue;
                }
                for i in 0..NX {
                    let x = (i as f32 + 0.5) * DX + X0;
                    if x.abs() <= SUB_HALF {
                        mat.set_pec(2, grid.ez_idx(i, j, kk));
                    }
                }
            }
        }
    }

    // --- Patch PEC sheet (z = 1.524 mm) ---
    // EFS_SHORT=1: extend the top sheet over the full substrate footprint
    // (shorts the feed post) — port-machinery test, expect Zin ≈ jωL_post.
    let short_top = std::env::var("EFS_SHORT").is_ok();
    let (patch_hx, patch_hy) = if short_top {
        (SUB_HALF, SUB_HALF)
    } else {
        (PATCH_HX, PATCH_HY)
    };
    for j in 0..NY {
        let y = (j as f32 + 0.5) * DY + Y0;
        if y.abs() > patch_hy {
            continue;
        }
        for i in 0..=NX {
            let x = i as f32 * DX + X0;
            if x.abs() <= patch_hx {
                mat.set_pec(0, grid.ex_idx(i, j, k_patch));
            }
        }
    }
    for j in 0..=NY {
        let y = j as f32 * DY + Y0;
        if y.abs() > patch_hy {
            continue;
        }
        for i in 0..NX {
            let x = (i as f32 + 0.5) * DX + X0;
            if x.abs() <= patch_hx {
                mat.set_pec(1, grid.ey_idx(i, j, k_patch));
            }
        }
    }

    // --- Feed: Ez column at (FEED_X, 0). Gap cell k_ground+1 (the
    //     substrate cell just above the ground plane) is the source;
    //     cells above it up to and including the patch plane are a PEC
    //     post (top cell's upper edge coincides with the patch plane,
    //     same convention as the Snyder probe). ---
    let src_i = ((FEED_X - X0) / DX - 0.5).round() as usize;
    let src_j = ((0.0f32 - Y0) / DY - 0.5).round() as usize;
    let feed_x_actual = (src_i as f32 + 0.5) * DX + X0;
    let feed_y_actual = (src_j as f32 + 0.5) * DY + Y0;
    println!(
        "feed Ez column ({src_i},{src_j}) at x={:.4} m, y={:.4} m (target −6 mm, 0)",
        feed_x_actual, feed_y_actual
    );
    assert!((feed_x_actual - FEED_X).abs() < 1e-6 && feed_y_actual.abs() < 1e-6);
    let k_src = k_ground + 1;
    // EFS_PORT_SPAN>1: openEMS-style distributed port over the whole
    // substrate column — no PEC pin (the port cells ARE the feed).
    let port_span: usize = std::env::var("EFS_PORT_SPAN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    // EFS_PIN_SHORT=1: stop the pin one cell below the patch (known-load
    // test — the port then sees the pin-top-to-plate capacitance).
    let pin_top = if std::env::var("EFS_PIN_SHORT").is_ok() {
        k_patch - 1
    } else {
        k_patch
    };
    if port_span <= 1 {
        for k in (k_src + 1)..=pin_top {
            mat.set_pec(2, grid.ez_idx(src_i, src_j, k));
        }
    }
    let li0 = src_i.saturating_sub(2);
    let li1 = (src_i + 3).min(NX);
    let lj0 = src_j.saturating_sub(2);
    let lj1 = (src_j + 3).min(NY);
    let feed = Feed {
        src_i,
        src_j,
        src_k: k_src,
        k_loop: k_src,
        hx_j_lo: lj0,
        hx_j_hi: lj1.min(NY),
        hx_i0: li0,
        hx_i1: li1,
        hy_i_lo: li0,
        hy_i_hi: li1.min(NX),
        hy_j0: lj0,
        hy_j1: lj1,
    };
    println!("source Ez cell ({src_i},{src_j},{k_src})");

    let cpml = Cpml::new(&grid, 12, dt); // npml 12: 8 let a PEC-channel grazing mode grow (see cpml.rs)
    let source = GaussianSource::new(2.2e9, 1.0);
    let rsource = ResistiveSource { rs: 50.0 };
    // Same analysis band as efs: 1–3 GHz at 5 MHz (401 points).
    let freqs: Vec<f64> = (0..=400).map(|m| 1.0e9 + m as f64 * 5e6).collect();
    let probe = DftProbe::new(freqs.clone(), dt);

    let mut sim = Fdtd::new(grid, mat, cpml, dt, feed, source, rsource, probe, None);
    sim.port_span = port_span;
    if std::env::var("DEBUG_PEC").is_ok() {
        let kmid = k_ground + subcells / 2;
        for k in (k_ground - 1)..=(k_patch + 1) {
            let ezi = sim.grid.ez_idx(src_i, src_j, k);
            println!(
                "post column k={k}: pec_z={} ca_z={:.3} cb_z={:.3e}",
                sim.mat.pec_z[ezi], sim.mat.ca_z[ezi], sim.mat.cb_z[ezi]
            );
        }
        println!(
            "ground Ex pec at (30,30,{k_ground}): {}  patch Ex pec at (30,30,{k_patch}): {}",
            sim.mat.pec_x[sim.grid.ex_idx(30, 30, k_ground)],
            sim.mat.pec_x[sim.grid.ex_idx(30, 30, k_patch)]
        );
        sim.run(3);
        for k in (k_ground - 1)..=(k_patch + 1) {
            let ezi = sim.grid.ez_idx(src_i, src_j, k);
            println!("after 3 steps: Ez(27,30,{k}) = {:.4e}", sim.grid.ez[ezi]);
        }
        let (v, i) = {
            let s = sim.probe.series();
            (s.0[2], s.1[2])
        };
        println!("probe step2: V={v:.4e} I={i:.4e}");
        // Compare the tight-loop current at the gap plane (k_src) with
        // the same loop at the post's mid-height (k=15): on a series
        // conductor they must agree in character.
        let feed15 = Feed {
            src_i,
            src_j,
            src_k: k_patch - 1,
            k_loop: k_patch - 1,
            hx_j_lo: lj0,
            hx_j_hi: lj1.min(NY),
            hx_i0: li0,
            hx_i1: li1,
            hy_i_lo: li0,
            hy_i_hi: li1.min(NX),
            hy_j0: lj0,
            hy_j1: lj1,
        };
        let dbg_steps = (2000.0 * 0.5 / cfl) as usize;
        let dump_series = std::env::var("DUMP_SERIES").is_ok();
        let dbg_steps = if dump_series { dbg_steps * 6 } else { dbg_steps };
        let mut series_out = String::new();
        for s in 0..dbg_steps {
            sim.step();
            if dump_series {
                let ser = sim.probe.series();
                let n = ser.0.len();
                series_out.push_str(&format!(
                    "{},{:.6e},{:.6e},{:.6e}\n",
                    n,
                    sim.grid.ez[sim.grid.ez_idx(35, 30, kmid)],
                    ser.0[n - 1],
                    ser.1[n - 1]
                ));
            }
            if s % 200 == 199 {
                // Patch voltage away from the pin: line integral of Ez
                // through the substrate at (35,30) and (45,30).
                let g = &sim.grid;
                let vcav = |i: usize| -> f32 {
                    ((k_ground + 1)..=k_patch).map(|k| g.ez[g.ez_idx(i, 30, k)] * DZ).sum()
                };
                println!(
                    "cavity audit step {}: -Vpin_path={:.4e} Vcav(35,30)={:.4e} Vcav(45,30)={:.4e}",
                    s + 1,
                    -((k_ground + 1)..=k_patch).map(|k| g.ez[g.ez_idx(27, 30, k)] * DZ).sum::<f32>(),
                    -vcav(35),
                    -vcav(45)
                );
                if s == 599 {
                    print!("Ez(35,30,k) column:");
                    for k in (k_ground - 1)..=(k_patch + 2) {
                        print!(" {:.3e}", g.ez[g.ez_idx(35, 30, k)]);
                    }
                    println!();
                    print!("Ez(27,30,k) column:");
                    for k in (k_ground - 1)..=(k_patch + 2) {
                        print!(" {:.3e}", g.ez[g.ez_idx(27, 30, k)]);
                    }
                    println!();
                    let mk_feed = |sk: usize| Feed {
                        src_i,
                        src_j,
                        src_k: sk,
                        k_loop: sk,
                        hx_j_lo: lj0,
                        hx_j_hi: lj1.min(NY),
                        hx_i0: li0,
                        hx_i1: li1,
                        hy_i_lo: li0,
                        hy_i_hi: li1.min(NX),
                        hy_j0: lj0,
                        hy_j1: lj1,
                    };
                    println!(
                        "loop currents: I(gap)={:.4e} I(top)={:.4e}",
                        mk_feed(k_ground + 1).measure_i_tight(g),
                        mk_feed(k_patch).measure_i_tight(g)
                    );
                }
            }
            if s == dbg_steps - 1 {
                // Cavity excitation audit: Ez across the patch footprint
                // at the substrate mid-layer (k=14) after the pulse.
                let g = &sim.grid;
                print!("Ez(i,30,kmid) i=20..41:");
                for i in 20..=41 {
                    print!(" {:.2e}", g.ez[g.ez_idx(i, 30, kmid)]);
                }
                println!();
                print!("Ez(27,j,kmid) j=20..41:");
                for j in 20..=41 {
                    print!(" {:.2e}", g.ez[g.ez_idx(27, j, kmid)]);
                }
                println!();
                let mut mx_cav = 0.0f32;
                for i in 15..=46 {
                    for j in 10..=51 {
                        let v = g.ez[g.ez_idx(i, j, kmid)].abs();
                        if v > mx_cav {
                            mx_cav = v;
                        }
                    }
                }
                println!(
                    "max|Ez| in substrate mid layer over patch region: {mx_cav:.3e}; Ez at pin (27,30,kmid) is PEC"
                );
            }
            {
                let ser = sim.probe.series();
                let n = ser.0.len();
                if n >= 440 && n <= 560 {
                    println!(
                        "step {}: Ez(src)={:.4e} V={:.4e} I={:.4e}",
                        n,
                        sim.grid.ez[sim.grid.ez_idx(src_i, src_j, k_src)],
                        ser.0[n - 1],
                        ser.1[n - 1]
                    );
                }
            }
            if s % 200 == 199 {
                let i15 = feed15.measure_i_tight(&sim.grid);
                let ser = sim.probe.series();
                let n = ser.0.len();
                println!(
                    "step {}: Ez(src)={:.4e} V={:.4e} I_gap={:.4e} I_post15={:.4e}",
                    n,
                    sim.grid.ez[sim.grid.ez_idx(src_i, src_j, k_src)],
                    ser.0[n - 1],
                    ser.1[n - 1],
                    i15
                );
            }
            if s == 602 {
                let g = &sim.grid;
                println!(
                    "mid-pulse fields: Ez(src)={:.4e} hx_bot={:.4e} hx_top={:.4e} hy_lft={:.4e} hy_rgt={:.4e}",
                    g.ez[g.ez_idx(src_i, src_j, k_src)],
                    g.hx[g.hx_idx(src_i, src_j, k_src)],
                    g.hx[g.hx_idx(src_i, src_j + 1, k_src)],
                    g.hy[g.hy_idx(src_i, src_j, k_src)],
                    g.hy[g.hy_idx(src_i + 1, src_j, k_src)],
                );
                // All H samples around the source cell, for loop audits:
                // labeling-A loop (faces of Ez(si,sj) per update_e pairing):
                // I_A = dy*(hy(si,sj)-hy(si-1,sj)) - dx*(hx(si,sj)-hx(si,sj-1))
                let hx_m = g.hx[g.hx_idx(src_i, src_j - 1, k_src)];
                let hx_0 = g.hx[g.hx_idx(src_i, src_j, k_src)];
                let hy_m = g.hy[g.hy_idx(src_i - 1, src_j, k_src)];
                let hy_0 = g.hy[g.hy_idx(src_i, src_j, k_src)];
                let i_a = DY * (hy_0 - hy_m) - DX * (hx_0 - hx_m);
                let i_a2 = DX * (hx_0 - hx_m) - DY * (hy_0 - hy_m);
                println!(
                    "loop audit: hx(sj-1)={hx_m:.4e} hx(sj)={hx_0:.4e} hy(si-1)={hy_m:.4e} hy(si)={hy_0:.4e} I_A={i_a:.4e} I_A2={i_a2:.4e}"
                );
                let maxh = g
                    .hx
                    .iter()
                    .chain(g.hy.iter())
                    .fold(0.0f32, |a, &x| a.max(x.abs()));
                let maxe = g.ez.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                println!("grid max|Hxy|={maxh:.4e} max|Ez|={maxe:.4e}");
            }
        }
        if dump_series {
            std::fs::write("/tmp/series.csv", series_out).expect("write series");
            println!("wrote /tmp/series.csv");
        }
        return;
    }
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
        let (max_ez, arg_ez) = sim
            .grid
            .ez
            .iter()
            .enumerate()
            .map(|(n, x)| (x.abs(), n))
            .fold((0.0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a });
        let (ai, aj, ak) = (
            arg_ez % NX,
            (arg_ez / NX) % NY,
            arg_ez / (NX * NY),
        );
        println!(
            "  step {done}/{n_steps}  t={:.2} ns  tail={:.2e}  max|Ez|={:.3e} at ({ai},{aj},{ak})  ({:.1}s, {:.0} steps/s)",
            sim.sim_time() * 1e9,
            sim.probe.tail_ratio(),
            max_ez,
            el,
            done as f64 / el,
        );
    }
    println!("wall: {:.1}s", t_start.elapsed().as_secs_f64());

    let (vf, cf) = sim.probe.dft();
    {
        // Port energy audit: for a passive load driven from rest,
        // integral of V*I dt (energy entering the load) must be >= 0.
        let (vs_, is_) = sim.probe.series();
        let e_port: f64 = vs_
            .iter()
            .zip(is_.iter())
            .map(|(v, i)| (*v as f64) * (*i as f64))
            .sum::<f64>()
            * dt as f64;
        let pk_v = vs_.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let pk_i = is_.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        println!("# port energy int(V*I)dt = {e_port:.4e} J   peak|V| = {pk_v:.3e}   peak|I| = {pk_i:.3e}");
    }
    let mut rows: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(freqs.len());
    for (m, &f) in freqs.iter().enumerate() {
        let (vr, vi) = vf[m];
        let (ir, ii) = cf[m];
        let (zr, zi) = cdiv(vr, vi, ir, ii);
        let (sr, si) = cdiv(zr - 50.0, zi, zr + 50.0, zi);
        let s11_db = 10.0 * (sr * sr + si * si).log10();
        rows.push((f, zr, zi, s11_db));
    }

    if let Some(path) = &csv_out {
        let mut out = String::from("freq_hz,zin_re,zin_im,s11_db\n");
        for (f, zr, zi, s) in &rows {
            out.push_str(&format!("{f:.0},{zr:.6},{zi:.6},{s:.4}\n"));
        }
        std::fs::write(path, out).expect("write csv");
        println!("wrote {path}");
    }

    // Resonance summary: min |S11| over the band.
    let (m0, r0) = rows
        .iter()
        .enumerate()
        .min_by(|a, b| a.1 .3.partial_cmp(&b.1 .3).unwrap())
        .unwrap();
    let _ = m0;
    println!(
        "# f0 (min S11) = {:.3} GHz   S11 = {:.2} dB   Zin = {:.2} {:+.2}j ohm",
        r0.0 / 1e9,
        r0.3,
        r0.1,
        r0.2
    );
    let band: Vec<f64> = rows
        .iter()
        .filter(|r| r.3 <= -10.0)
        .map(|r| r.0)
        .collect();
    if band.len() >= 2 {
        println!(
            "# −10 dB band: {:.3}–{:.3} GHz",
            band.first().unwrap() / 1e9,
            band.last().unwrap() / 1e9
        );
    } else {
        println!("# no −10 dB band in 1–3 GHz");
    }
    println!("# MHz  Re(Zin)  Im(Zin)  |S11|_dB   (2.35–2.52 GHz at 5 MHz, elsewhere 25 MHz)");
    for (f, zr, zi, s) in &rows {
        let fghz = f / 1e9;
        let fine = (2.35..=2.52).contains(&fghz);
        let coarse = ((f / 1e6) as i64 % 25) == 0;
        if fine || coarse {
            println!("{:.0}  {:.4}  {:+.4}  {:.2}", f / 1e6, zr, zi, s);
        }
    }
}
