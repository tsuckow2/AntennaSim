//! Native parity test against the TypeScript reference solver.
//!
//! Fixture: `fixtures/dipole_3ghz.req`, packed from the TS reference dipole
//! (`dipoleIR()`, `buildFEMMesh(ir, 0.004)`) by `tools/pack_dipole.ts`.
//! Reference values (TS, 2026-09-25):
//!   Zin = 66.637496 − j15.317936 Ω, S11 = 0.157179 − j0.110687,
//!   1117 iterations, residual 8.15e-8, 19,561 free DOFs.

use hcurl_fem::abi::{FarFieldBlob, PointRequest};
use hcurl_fem::budget::Budget;
use hcurl_fem::{assembly, edge, mesh};

fn load_request() -> PointRequest {
    let bytes = include_bytes!("fixtures/dipole_3ghz.req");
    PointRequest::parse(bytes).expect("fixture must parse")
}

#[test]
fn dipole_impedance_and_s11_match_typescript() {
    let request = load_request();
    let response = hcurl_fem::solve_point(&request, &Budget::none()).expect("solve must not error");
    assert_eq!(response.status, 0, "message: {:?}", response.message);

    let zin = response.zin.expect("Zin present");
    let s11 = response.s11.expect("S11 present");
    let iters = response.iteration_count.expect("iteration count present");
    let residual = response.residual_norm.expect("residual present");

    // Impedance within 2% of the TS reference (magnitude-relative).
    let dz = ((zin.re - 66.637496).powi(2) + (zin.im + 15.317936).powi(2)).sqrt();
    let zmag = (66.637496f64.powi(2) + 15.317936f64.powi(2)).sqrt();
    assert!(
        dz / zmag < 0.02,
        "Zin drifted: {zin:?} vs 66.637496-j15.317936"
    );
    let ds = ((s11.re - 0.157179).powi(2) + (s11.im + 0.110687).powi(2)).sqrt();
    assert!(ds < 0.01, "S11 drifted: {s11:?}");

    // Similar iteration count (same algorithm, same preconditioner).
    assert!(
        (500..=3000).contains(&iters),
        "iteration count {iters} far from TS reference 1117"
    );
    assert!(residual < 1e-6, "residual {residual:e} too large");

    // Far-field blob present and physical.
    assert!(!response.field_blob.is_empty(), "far-field blob missing");
    let blob = FarFieldBlob::parse(&response.field_blob).expect("blob must parse");
    assert!(
        (blob.max_gain_dbi - 2.7658).abs() < 0.15,
        "max gain {} dBi far from TS 2.7658",
        blob.max_gain_dbi
    );
    // The pattern has two symmetric broadside maxima (theta 45°/135°); TS
    // and Rust may pick different sides of the tie, so compare the gain
    // value and check the TS-side maximum is within noise of the Rust max.
    assert!(
        (blob.max_gain_theta_deg - 45.0).abs() <= 10.0
            || (blob.max_gain_theta_deg - 135.0).abs() <= 10.0,
        "max gain theta {}° is not broadside",
        blob.max_gain_theta_deg
    );
    let ti = blob
        .theta_deg
        .iter()
        .position(|&t| (t - 135.0).abs() < 1e-9)
        .expect("135° row present");
    let pj = blob
        .phi_deg
        .iter()
        .position(|&p| (p - 90.0).abs() < 1e-9)
        .expect("90° column present");
    let g_ts_side = blob.gain_dbi[ti * blob.phi_deg.len() + pj];
    assert!(
        blob.max_gain_dbi - g_ts_side < 0.05,
        "TS-side gain {g_ts_side} dBi differs from Rust max {} dBi",
        blob.max_gain_dbi
    );
    assert!(blob.radiated_power_w > 0.0);
    // The reference itself reports efficiency 1.1037 here (tight 0.2λ ABC
    // box, first-order ABC): this asserts port faithfulness, not physics.
    assert!(
        (blob.radiation_efficiency - 1.1037).abs() < 0.02,
        "radiation efficiency {} drifted from TS 1.1037",
        blob.radiation_efficiency
    );
}

#[test]
fn dipole_abc_anchors_match_typescript() {
    // TS anchors on the assembled matrix (2026-09-25): 14,946 imaginary
    // nnz, imaginary norm 0.03994385, imaginary diagonal in
    // [0.00016767, 0.00067067]. (The older [21, 68] / 2380.94 figures were
    // the MFEM prototype's differently-scaled ABC matrix, not the TS one.)
    let request = load_request();
    let mesh = mesh::Mesh::from_request(&request).unwrap();
    let sys = edge::build_edge_system(&mesh).unwrap();
    assert_eq!(sys.free_dof_count, 19_429, "free DOF count drifted");
    assert_eq!(sys.edge_count, 19_561, "global edge count drifted");

    let matrix = hcurl_fem::assembly::assemble_hcurl(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
    let mut nnz = 0usize;
    let mut norm2 = 0.0f64;
    let mut diag_min = f64::INFINITY;
    let mut diag_max = f64::NEG_INFINITY;
    for r in 0..matrix.nrows() {
        let (cols, vals) = matrix.row(r);
        for (k, v) in cols.iter().zip(vals.iter()) {
            if v.im != 0.0 {
                nnz += 1;
                norm2 += v.im * v.im;
                if *k == r {
                    diag_min = diag_min.min(v.im);
                    diag_max = diag_max.max(v.im);
                }
            }
        }
    }
    let norm = norm2.sqrt();
    assert_eq!(nnz, 14_946, "ABC imaginary nnz drifted");
    assert!(
        (norm - 0.03994385).abs() < 1e-6,
        "ABC imaginary norm {norm} drifted from 0.03994385"
    );
    assert!(
        (diag_min - 0.00016767).abs() < 1e-8 && (diag_max - 0.00067067).abs() < 1e-8,
        "ABC imaginary diagonal [{diag_min}, {diag_max}] drifted"
    );
}
