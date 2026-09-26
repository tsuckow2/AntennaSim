//! AMS benchmark: solve the same H(curl) system with BiCGSTAB+SGS and with
//! BiCGSTAB+AMS (auxiliary-space Maxwell), then compare iterations, wall
//! time, residual, and port impedance on the physical port RHS.
//!
//! ```sh
//! cargo run --release --example ams_bench -- tests/fixtures/dipole_3ghz.req
//! RAYON_NUM_THREADS=2 cargo run --release --example ams_bench -- /path/to/2500.req
//! ```
//!
//! Prints per solver: iterations, solve wall time, relative residual,
//! Zin, and the AMS build time + nodal system size.

use hcurl_fem::abi::PointRequest;
use hcurl_fem::ams::AmsPreconditioner;
use hcurl_fem::assembly::assemble_hcurl;
use hcurl_fem::budget::Budget;
use hcurl_fem::edge::build_edge_system;
use hcurl_fem::krylov::{bicgstab, Preconditioner, DEFAULT_MAX_ITER, DEFAULT_TOL};
use hcurl_fem::mesh::Mesh;
use hcurl_fem::port::{build_port_rhs, port_voltage, select_gap_feed, terminals_from_facets};
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ams_bench <request.req>");
    let bytes = std::fs::read(&path).expect("read request file");
    let request = PointRequest::parse(&bytes).expect("parse request");
    let frequency_hz = request.frequency_hz;
    let budget = Budget::none();

    let mesh = Mesh::from_request(&request).expect("mesh");
    let sys = build_edge_system(&mesh).expect("edge system");
    let t = Instant::now();
    let matrix = assemble_hcurl(&mesh, &sys, frequency_hz, &budget).expect("assembly");
    let assembly_s = t.elapsed().as_secs_f64();

    let (positive, negative) = terminals_from_facets(&mesh).expect("terminals");
    let feed = select_gap_feed(&mesh, &sys, positive, negative).expect("feed");
    let rhs = build_port_rhs(&mesh, &sys, &feed, frequency_hz).expect("rhs");

    println!(
        "dofs={} nnz={} assembly={:.1}s",
        sys.free_dof_count,
        matrix.nnz(),
        assembly_s
    );

    // Baseline: BiCGSTAB + SGS.
    let t = Instant::now();
    let sgs = bicgstab(
        &matrix,
        &rhs,
        DEFAULT_TOL,
        DEFAULT_MAX_ITER,
        Preconditioner::Sgs,
        &budget,
    )
    .expect("sgs solve");
    let sgs_s = t.elapsed().as_secs_f64();
    let zin_sgs = port_voltage(&mesh, &sys, &feed, negative, &sgs.x).expect("zin");
    println!(
        "SGS: iters={} conv={} residual={:.3e} wall={:.1}s Zin={:.6}-j{:.6}",
        sgs.iterations, sgs.converged, sgs.residual_norm, sgs_s, zin_sgs.re, -zin_sgs.im
    );

    // AMS: build the auxiliary operator, then BiCGSTAB + AMS.
    let t = Instant::now();
    let ams = AmsPreconditioner::build(&mesh, &sys, &matrix).expect("ams build");
    let build_s = t.elapsed().as_secs_f64();
    println!(
        "AMS build: {:.1}s, nodal dofs={} nodal nnz={}",
        build_s,
        ams.node_count(),
        ams.nodal_nnz()
    );

    let t = Instant::now();
    let ams_res = bicgstab(
        &matrix,
        &rhs,
        DEFAULT_TOL,
        DEFAULT_MAX_ITER,
        Preconditioner::Ams(ams),
        &budget,
    )
    .expect("ams solve");
    let ams_s = t.elapsed().as_secs_f64();
    let zin_ams = port_voltage(&mesh, &sys, &feed, negative, &ams_res.x).expect("zin");
    println!(
        "AMS: iters={} conv={} residual={:.3e} wall={:.1}s Zin={:.6}-j{:.6}",
        ams_res.iterations,
        ams_res.converged,
        ams_res.residual_norm,
        ams_s,
        zin_ams.re,
        -zin_ams.im
    );
    println!(
        "delta Zin: {:.3e} + j{:.3e}",
        (zin_ams.re - zin_sgs.re).abs(),
        (zin_ams.im - zin_sgs.im).abs()
    );
}
