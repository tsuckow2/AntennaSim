//! AMS diagnostics on the dipole fixture with the physical port RHS.
//!
//! ```sh
//! cargo run --release --example ams_diag
//! ```
//!
//! 1. SGS baseline iterations.
//! 2. Standalone inner solve on `cn = Gᵀ·rhs` — does the nodal system solve cleanly?
//! 3. Outer AMS with a tight inner tolerance — does the *idea* work when the
//!    coarse solve is accurate?

use hcurl_fem::abi::PointRequest;
use hcurl_fem::ams::AmsPreconditioner;
use hcurl_fem::assembly::assemble_hcurl;
use hcurl_fem::budget::Budget;
use hcurl_fem::edge::build_edge_system;
use hcurl_fem::krylov::{bicgstab, Preconditioner, DEFAULT_TOL};
use hcurl_fem::mesh::Mesh;
use hcurl_fem::port::{build_port_rhs, select_gap_feed, terminals_from_facets};
use std::time::Instant;

fn main() {
    let budget = Budget::none();
    let bytes = include_bytes!("../tests/fixtures/dipole_3ghz.req");
    let request = PointRequest::parse(bytes).expect("parse");
    let frequency_hz = request.frequency_hz;
    let mesh = Mesh::from_request(&request).expect("mesh");
    let sys = build_edge_system(&mesh).expect("edges");
    let a = assemble_hcurl(&mesh, &sys, frequency_hz, &budget).expect("assembly");
    let (positive, _negative) = terminals_from_facets(&mesh).expect("terminals");
    let feed = select_gap_feed(&mesh, &sys, positive, _negative).expect("feed");
    let rhs = build_port_rhs(&mesh, &sys, &feed, frequency_hz).expect("rhs");
    println!("dofs={} nnz={}", sys.free_dof_count, a.nnz());

    // 1. SGS baseline.
    let t = Instant::now();
    let sgs = bicgstab(&a, &rhs, DEFAULT_TOL, 3000, Preconditioner::Sgs, &budget).unwrap();
    println!(
        "SGS: iters={} conv={} res={:.3e} wall={:.1}s",
        sgs.iterations,
        sgs.converged,
        sgs.residual_norm,
        t.elapsed().as_secs_f64()
    );

    // 2. AMS build + standalone inner solve on the true coarse RHS.
    let t = Instant::now();
    let mut ams = AmsPreconditioner::build(&mesh, &sys, &a).expect("ams");
    println!(
        "AMS build {:.1}s nodal_dofs={} nodal_nnz={}",
        t.elapsed().as_secs_f64(),
        ams.node_count(),
        ams.nodal_nnz()
    );
    let cn = ams.apply_g_t(&rhs);
    let cn_norm: f64 = cn.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    println!("||G^T rhs|| = {:.3e}", cn_norm);
    for tol in [1e-2, 1e-6] {
        let t = Instant::now();
        let inner = bicgstab(
            ams.nodal_matrix(),
            &cn,
            tol,
            2000,
            Preconditioner::Sgs,
            &budget,
        )
        .unwrap();
        println!(
            "inner tol={:e}: iters={} conv={} res={:.3e} wall={:.1}s",
            tol,
            inner.iterations,
            inner.converged,
            inner.residual_norm,
            t.elapsed().as_secs_f64()
        );
    }

    // 3. Outer AMS with an accurate coarse solve: does the idea work?
    ams.inner_sweeps = 0; // use inner BiCGSTAB
    ams.inner_tol = 1e-8;
    ams.inner_max_iter = 2000;
    let t = Instant::now();
    let out = bicgstab(
        &a,
        &rhs,
        DEFAULT_TOL,
        200,
        Preconditioner::Ams(ams),
        &budget,
    )
    .unwrap();
    println!(
        "AMS(tight inner): iters={} conv={} res={:.3e} wall={:.1}s",
        out.iterations,
        out.converged,
        out.residual_norm,
        t.elapsed().as_secs_f64()
    );
}
