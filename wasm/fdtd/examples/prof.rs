use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::cpml::Cpml;
use fdtd::geometry::{build_snyder, DX, DY, DZ, NX, NY, NZ};
use fdtd::solver::{update_h, update_e, apply_pec};
use std::time::Instant;

fn main() {
    let mut g = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let dt = g.courant_dt(0.99);
    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let _feed = build_snyder(&g, &mut mat, dt);
    let mut cpml = Cpml::new(&g, 8, dt);
    // put some nonzero fields in so we touch memory realistically
    for v in g.ex.iter_mut().take(100000) { *v = 1.0; }
    let t0 = Instant::now();
    for _ in 0..5 { update_h(&mut g, dt); }
    println!("update_h: {:.1} ms", t0.elapsed().as_secs_f64()*200.0);
    let t0 = Instant::now();
    for _ in 0..5 { update_e(&mut g, &mat); }
    println!("update_e: {:.1} ms", t0.elapsed().as_secs_f64()*200.0);
    let t0 = Instant::now();
    for _ in 0..5 { cpml.correct_h(&mut g, dt); }
    println!("correct_h: {:.1} ms", t0.elapsed().as_secs_f64()*200.0);
    let t0 = Instant::now();
    for _ in 0..5 { cpml.correct_e(&mut g, &mat.cb_x, &mat.cb_y, &mat.cb_z); }
    println!("correct_e: {:.1} ms", t0.elapsed().as_secs_f64()*200.0);
    let t0 = Instant::now();
    for _ in 0..5 { apply_pec(&mut g, &mat); }
    println!("apply_pec: {:.1} ms", t0.elapsed().as_secs_f64()*200.0);
    println!("threads: {}", rayon::current_num_threads());
}
