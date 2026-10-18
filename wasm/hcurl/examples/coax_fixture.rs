//! Coax wave-port calibration fixture: solves a coax request and prints
//! V, I, Z = V/I at several planes along the section.
//!
//! Matched fixture (absorbing top face): expect a pure traveling wave —
//! |V| and Z constant along z, V phase rotating linearly with slope k_d.
//! Short fixture (copper cap): expect Z(z) = j·Z0·tan(k_d·|z|).
//!
//! ```sh
//! cargo run --release --example coax_fixture -- ~/workspace/coax_models/fixture-matched/2440.req
//! ```

use hcurl_fem::abi::PointRequest;
use hcurl_fem::assembly::assemble_hcurl;
use hcurl_fem::budget::Budget;
use hcurl_fem::coax::{build_coax_rhs, coax_vi, find_port_faces};
use hcurl_fem::csr::CsrMatrix;
use hcurl_fem::edge::build_edge_system;
use hcurl_fem::krylov::{bicgstab, Preconditioner, DEFAULT_MAX_ITER};
use hcurl_fem::mesh::Mesh;
use faer::linalg::solvers::SolveCore;
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
use faer::{Conj, Mat};
use num_complex::Complex64;

fn to_faer_csc(a: &CsrMatrix) -> SparseColMat<usize, faer::c64> {
    let n = a.nrows();
    let mut triplets: Vec<Triplet<usize, usize, faer::c64>> = Vec::with_capacity(a.nnz());
    for i in 0..n {
        let (cols, vs) = a.row(i);
        for (&j, &v) in cols.iter().zip(vs.iter()) {
            triplets.push(Triplet::new(i, j, faer::c64::new(v.re, v.im)));
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).expect("faer csc")
}

fn solve_direct(a: &CsrMatrix, rhs: &[Complex64]) -> Vec<Complex64> {
    let n = a.nrows();
    let a_faer = to_faer_csc(a);
    let lu = a_faer.as_ref().sp_lu().expect("sparse LU factorization");
    let mut x = Mat::<faer::c64>::from_fn(n, 1, |i, _| faer::c64::new(rhs[i].re, rhs[i].im));
    lu.solve_in_place_with_conj(Conj::No, x.as_mut());
    x.as_ref()
        .col(0)
        .iter()
        .map(|&v| Complex64::new(v.re, v.im))
        .collect()
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: coax_fixture <path.req>");
    let bytes = std::fs::read(&path).expect("request file must be readable");
    let request = PointRequest::parse(&bytes).expect("request must parse");
    let freq = request.frequency_hz;
    let mesh = Mesh::from_request(&request).expect("mesh");
    let cp = mesh.coax_port.expect("request must carry coax_port metadata");
    let sys = build_edge_system(&mesh).expect("edge system");
    let faces = find_port_faces(&mesh, &sys, &cp);
    let n_drive = faces.iter().filter(|f| !f.top).count();
    println!(
        "tets={} active={} dofs={} port faces: {} ({} drive, {} top)",
        mesh.tet_count(),
        sys.active_tet_count,
        sys.free_dof_count,
        faces.len(),
        n_drive,
        faces.len() - n_drive
    );
    let budget = Budget::none();
    let matrix = assemble_hcurl(&mesh, &sys, freq, &budget).expect("assembly");
    let rhs = build_coax_rhs(&mesh, &sys, &cp, &faces, freq).expect("rhs");
    let direct = std::env::args().any(|a| a == "--direct");
    let x = if direct {
        let x = solve_direct(&matrix, &rhs);
        let ax = matrix.matvec(&x);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..rhs.len() {
            num += (ax[i] - rhs[i]).norm_sqr();
            den += rhs[i].norm_sqr();
        }
        println!("direct solve: rel residual {:.3e}", (num / den).sqrt());
        x
    } else {
        let kry = bicgstab(
            &matrix,
            &rhs,
            request.metadata.krylov_tolerance(),
            DEFAULT_MAX_ITER,
            Preconditioner::Sgs,
            &budget,
        )
        .expect("solve");
        println!(
            "converged={} iters={} residual={:.3e}",
            kry.converged, kry.iterations, kry.residual_norm
        );
        kry.x
    };
    for z_mm in [-0.25f64, -1.5, -3.0, -4.5, -5.5] {
        let mut probe = cp;
        probe.z_ref = z_mm * 1e-3;
        match coax_vi(&mesh, &sys, &probe, &x, freq) {
            Ok((v, i)) => println!(
                "z={:6.2} mm  V = {:9.4} {:+9.4}j (|V|={:7.4}, {:7.1} deg)  I = {:8.4} {:+8.4}j  Zin = {:9.4} {:+9.4}j",
                z_mm, v.re, v.im, v.norm(), v.arg() * 180.0 / std::f64::consts::PI, i.re, i.im, (v / i).re, (v / i).im
            ),
            Err(e) => println!("z={:6.2} mm  extraction error: {e}", z_mm),
        }
    }
}
