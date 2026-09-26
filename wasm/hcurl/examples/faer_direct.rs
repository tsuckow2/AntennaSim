//! Direct-solve experiment: factor the assembled H(curl) system with faer's
//! sparse LU instead of BiCGSTAB/SGS, then compare impedance, time, memory.
//!
//! ```sh
//! cargo run --release --example faer_direct -- tests/fixtures/dipole_3ghz.req
//! RAYON_NUM_THREADS=2 cargo run --release --example faer_direct -- /path/to/2500.req
//! ```
//!
//! Prints: DOFs, nnz, factor nnz estimate, factor/solve wall time,
//! iterative residual of the direct solution, Zin from the direct solution,
//! and peak RSS.

use faer::linalg::solvers::SolveCore;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::{Conj, Mat};
use hcurl_fem::abi::PointRequest;
use hcurl_fem::assembly::assemble_hcurl;
use hcurl_fem::budget::Budget;
use hcurl_fem::csr::CsrMatrix;
use hcurl_fem::edge::build_edge_system;
use hcurl_fem::mesh::Mesh;
use hcurl_fem::port::{build_port_rhs, port_voltage, select_gap_feed, terminals_from_facets};
use num_complex::Complex64;
use std::time::Instant;

fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

/// Convert our CSR matrix to faer's CSC. Valid because the H(curl) system is
/// complex-symmetric: column j of A is row j of A with (col,row) swapped.
fn to_faer_csc(m: &CsrMatrix) -> SparseColMat<usize, faer::c64> {
    let n = m.nrows();
    assert_eq!(n, m.ncols());
    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = m.row(j).0.len();
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }
    let nnz = m.nnz();
    let mut row_idx = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    let mut next = col_ptr[..n].to_vec();
    for j in 0..n {
        let (cols, vs) = m.row(j);
        for (&k, &v) in cols.iter().zip(vs.iter()) {
            let p = next[j];
            next[j] += 1;
            debug_assert!(p < col_ptr[j + 1]);
            row_idx.push(k);
            vals.push(faer::c64::new(v.re, v.im));
        }
    }
    debug_assert_eq!(row_idx.len(), nnz);
    let symbolic = SymbolicSparseColMat::new_checked(n, n, col_ptr, None, row_idx);
    SparseColMat::<usize, faer::c64>::new(symbolic, vals)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: faer_direct <path.req>");
    let bytes = std::fs::read(&path).expect("request file must be readable");
    let request = PointRequest::parse(&bytes).expect("request must parse");
    let freq = request.frequency_hz;
    let rss0 = peak_rss_kb();

    let t0 = Instant::now();
    let mesh = Mesh::from_request(&request).expect("mesh");
    let sys = build_edge_system(&mesh).expect("edge system");
    let matrix = assemble_hcurl(&mesh, &sys, freq, &Budget::none()).expect("assembly");
    let (positive, negative) = terminals_from_facets(&mesh).expect("terminals");
    let feed = select_gap_feed(&mesh, &sys, positive, negative).expect("feed");
    let rhs = build_port_rhs(&mesh, &sys, &feed, freq).expect("rhs");
    let t_assemble = t0.elapsed();

    let n = sys.free_dof_count;
    println!(
        "dofs={n} nnz={} assembly_s={:.1}",
        matrix.nnz(),
        t_assemble.as_secs_f64()
    );

    let t1 = Instant::now();
    let a_faer = to_faer_csc(&matrix);
    let t_convert = t1.elapsed();

    let t2 = Instant::now();
    let lu = a_faer.as_ref().sp_lu().expect("sparse LU factorization");
    let t_factor = t2.elapsed();
    let rss1 = peak_rss_kb();

    // Solve A x = b in place.
    let mut x = Mat::<faer::c64>::from_fn(n, 1, |i, _| faer::c64::new(rhs[i].re, rhs[i].im));
    let t3 = Instant::now();
    lu.solve_in_place_with_conj(Conj::No, x.as_mut());
    let t_solve = t3.elapsed();

    // Back to Complex64 for the port-voltage extraction and residual check.
    let x_c: Vec<Complex64> = x
        .as_ref()
        .col(0)
        .iter()
        .map(|&v| Complex64::new(v.re, v.im))
        .collect();

    // Iterative residual ||A x - b|| / ||b|| through our own matvec.
    let ax = matrix.matvec(&x_c);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n {
        num += (ax[i] - rhs[i]).norm_sqr();
        den += rhs[i].norm_sqr();
    }
    let rel_res = (num / den).sqrt();

    let zin = port_voltage(&mesh, &sys, &feed, negative, &x_c).expect("port voltage");

    println!(
        "convert_s={:.2} factor_s={:.1} solve_s={:.2}",
        t_convert.as_secs_f64(),
        t_factor.as_secs_f64(),
        t_solve.as_secs_f64()
    );
    println!(
        "rel_residual={rel_res:.3e} zin={:.6}+j{:.6}",
        zin.re, zin.im
    );
    println!(
        "peak_rss_mb: start={} factor_done={}",
        rss0 / 1024,
        rss1 / 1024
    );
}
