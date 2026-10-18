//! Coax wave-port solve with the AMS preconditioner: BiCGSTAB + AMS on a
//! request carrying `coax_port` metadata. The SGS fixture tool
//! (`coax_fixture`) needs ~11k iterations on a 68k-DOF fixture; the Snyder
//! coax mesh is ~500k tets, so production-size coax solves go through AMS.
//!
//! ```sh
//! cargo run --release --example coax_ams -- /path/to/2447.req
//! ```
//!
//! Prints: DOFs, port faces, AMS build stats, iterations, wall time,
//! residual, and Zin at the reference plane plus a short z-sweep of V/I/Z
//! down the coax section (a clean port shows Z constant along z).

use hcurl_fem::abi::PointRequest;
use hcurl_fem::ams::AmsPreconditioner;
use hcurl_fem::assembly::assemble_hcurl;
use hcurl_fem::budget::Budget;
use hcurl_fem::coax::{build_coax_rhs, coax_impedance, coax_vi, find_port_faces};
use hcurl_fem::edge::build_edge_system;
use hcurl_fem::krylov::{bicgstab, Preconditioner, DEFAULT_MAX_ITER, DEFAULT_TOL};
use hcurl_fem::mesh::Mesh;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: coax_ams <request.req>");
    let bytes = std::fs::read(&path).expect("read request file");
    let request = PointRequest::parse(&bytes).expect("parse request");
    let frequency_hz = request.frequency_hz;
    let budget = Budget::none();

    let mesh = Mesh::from_request(&request).expect("mesh");
    let cp = mesh.coax_port.expect("request must carry coax_port metadata");
    let sys = build_edge_system(&mesh).expect("edge system");
    let faces = find_port_faces(&mesh, &sys, &cp);
    let n_drive = faces.iter().filter(|f| !f.top).count();
    println!(
        "tets={} active={} dofs={} port faces: {} ({} drive, {} absorbing-only)",
        mesh.tet_count(),
        sys.active_tet_count,
        sys.free_dof_count,
        faces.len(),
        n_drive,
        faces.len() - n_drive
    );

    let t = Instant::now();
    let matrix = assemble_hcurl(&mesh, &sys, frequency_hz, &budget).expect("assembly");
    println!("assembly: {:.1}s nnz={}", t.elapsed().as_secs_f64(), matrix.nnz());

    let rhs = build_coax_rhs(&mesh, &sys, &cp, &faces, frequency_hz).expect("coax rhs");

    let t = Instant::now();
    let ams = AmsPreconditioner::build(&mesh, &sys, &matrix).expect("ams build");
    println!(
        "AMS build: {:.1}s, nodal dofs={} nodal nnz={}",
        t.elapsed().as_secs_f64(),
        ams.node_count(),
        ams.nodal_nnz()
    );

    let t = Instant::now();
    let res = bicgstab(
        &matrix,
        &rhs,
        DEFAULT_TOL,
        DEFAULT_MAX_ITER,
        Preconditioner::Ams(ams),
        &budget,
    )
    .expect("ams solve");
    println!(
        "AMS: iters={} conv={} residual={:.3e} wall={:.1}s",
        res.iterations,
        res.converged,
        res.residual_norm,
        t.elapsed().as_secs_f64()
    );

    let zin = coax_impedance(&mesh, &sys, &cp, &res.x, frequency_hz).expect("zin");
    println!("Zin(z_ref) = {:.4} {:+.4}j  |Z|={:.4}", zin.re, zin.im, zin.norm());
    for z_mm in [-0.25f64, -1.5, -3.0, -4.5, -5.5] {
        let mut probe = cp;
        probe.z_ref = z_mm * 1e-3;
        match coax_vi(&mesh, &sys, &probe, &res.x, frequency_hz) {
            Ok((v, i)) => println!(
                "z={:6.2} mm  V = {:9.4} {:+9.4}j  I = {:8.4} {:+8.4}j  Zin = {:9.4} {:+9.4}j",
                z_mm, v.re, v.im, i.re, i.im, (v / i).re, (v / i).im
            ),
            Err(e) => println!("z={:6.2} mm  extraction error: {e}", z_mm),
        }
    }
}
