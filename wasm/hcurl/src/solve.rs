//! End-to-end single-frequency solve orchestration.
//!
//! Pipeline: request → validated mesh → global edges (PEC eliminated) →
//! H(curl) assembly (volume + Silver–Müller ABC) → one-amp gap feed →
//! BiCGSTAB/SGS → port voltage → Zin (= V at 1 A) → S11 → best-effort
//! Stratton–Chu far-field.
//!
//! Mirrors the TypeScript `solveFEMPoint` stage order and failure semantics:
//! hard input errors are [`Err`]; solver non-convergence and non-finite
//! impedance/S11 come back as [`Ok`] responses with a non-zero status code,
//! exactly like the reference's `"solver_error"` / `"non_finite"` results.

use num_complex::Complex64;

use crate::clock::Stopwatch;

use crate::abi::{FarFieldBlob, PointRequest, PointResponse, TimingMs};
use crate::assembly::assemble_hcurl;
use crate::budget::Budget;
use crate::edge::build_edge_system;
use crate::error::{Error, Result};
use crate::farfield::{compute_far_field, FarFieldOptions};
use crate::krylov::{bicgstab, Preconditioner, DEFAULT_MAX_ITER};
use crate::mesh::Mesh;
use crate::port::{
    build_port_rhs, port_voltage, reflection_coefficient, select_gap_feed, terminals_from_facets,
};

/// Solver-failure status code on the wire (matches the worker's
/// `"solver_error"` / `"non_finite"` mapping).
const STATUS_SOLVER_ERROR: u32 = 6;

fn failed_point(
    message: String,
    residual_norm: Option<f64>,
    iteration_count: Option<u32>,
    timing: TimingMs,
) -> PointResponse {
    PointResponse {
        status: STATUS_SOLVER_ERROR,
        residual_norm,
        iteration_count,
        zin: None,
        s11: None,
        timing,
        field_blob: Vec::new(),
        message: Some(message),
    }
}

/// Solve one driven frequency point; see the module docs for the pipeline.
pub fn solve_point(request: &PointRequest, budget: &Budget) -> Result<PointResponse> {
    let mut timing = TimingMs::default();

    let frequency_hz = request.frequency_hz;
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(Error::InvalidRequest(
            "frequency_hz must be a positive finite value".to_string(),
        ));
    }
    let z0 = request.metadata.reference_impedance_ohms();
    if !z0.is_finite() || z0 <= 0.0 {
        return Err(Error::InvalidRequest(
            "reference_impedance_ohms must be a positive finite value".to_string(),
        ));
    }

    let t0 = Stopwatch::start();
    let mesh = Mesh::from_request(request)?;
    budget.check()?;
    let sys = build_edge_system(&mesh)?;
    if sys.free_dof_count == 0 {
        return Err(Error::Mesh(
            "no free edge DOFs remain after PEC elimination".to_string(),
        ));
    }
    let matrix = assemble_hcurl(&mesh, &sys, frequency_hz, budget)?;
    timing.assembly = t0.elapsed_ms();
    let t1 = Stopwatch::start();

    let coax = mesh.coax_port;
    let rhs = if let Some(cp) = &coax {
        if !(cp.half_pin > 0.0 && cp.half_diel > cp.half_pin && cp.eps_diel >= 1.0) {
            return Err(Error::InvalidRequest(
                "coax_port: require half_diel > half_pin > 0 and eps_diel >= 1".to_string(),
            ));
        }
        let faces = crate::coax::find_port_faces(&mesh, &sys, cp);
        let n_drive = faces.iter().filter(|f| !f.top).count();
        if n_drive == 0 {
            return Err(Error::Mesh(
                "coax_port: no port faces found at z_port (check geometry/exclusions)".to_string(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!(
            "coax port: {} faces ({} drive, {} absorbing-only)",
            faces.len(),
            n_drive,
            faces.len() - n_drive
        );
        crate::coax::build_coax_rhs(&mesh, &sys, cp, &faces, frequency_hz)?
    } else {
        let (positive, negative) = terminals_from_facets(&mesh)?;
        let feed = select_gap_feed(&mesh, &sys, positive, negative)?;
        build_port_rhs(&mesh, &sys, &feed, frequency_hz)?
    };
    let kry = bicgstab(
        &matrix,
        &rhs,
        request.metadata.krylov_tolerance(),
        DEFAULT_MAX_ITER,
        Preconditioner::Sgs,
        budget,
    )?;
    timing.solve = t1.elapsed_ms();
    let t2 = Stopwatch::start();

    if !kry.converged {
        let residual = kry.residual_norm.is_finite().then_some(kry.residual_norm);
        return Ok(failed_point(
            format!(
                "BiCGSTAB did not converge (relative residual {:.3e})",
                kry.residual_norm
            ),
            residual,
            Some(kry.iterations),
            timing,
        ));
    }

    // Legacy path: the impressed current is exactly 1 A, so Zin = V.
    // Coax path: Zin = V/I at the reference plane from the total fields.
    let zin = if let Some(cp) = &coax {
        crate::coax::coax_impedance(&mesh, &sys, cp, &kry.x, frequency_hz)?
    } else {
        let (positive, negative) = terminals_from_facets(&mesh)?;
        let feed = select_gap_feed(&mesh, &sys, positive, negative)?;
        port_voltage(&mesh, &sys, &feed, negative, &kry.x)?
    };
    if !zin.re.is_finite() || !zin.im.is_finite() {
        return Ok(failed_point(
            "port impedance is non-finite".to_string(),
            Some(kry.residual_norm),
            Some(kry.iterations),
            timing,
        ));
    }
    let den = (zin.re + z0) * (zin.re + z0) + zin.im * zin.im;
    if !(den > 0.0) || !den.is_finite() {
        return Ok(failed_point(
            "S11 denominator is non-positive or non-finite".to_string(),
            Some(kry.residual_norm),
            Some(kry.iterations),
            timing,
        ));
    }
    let s11 = reflection_coefficient(zin, z0);
    if !s11.re.is_finite() || !s11.im.is_finite() {
        return Ok(failed_point(
            "S11 is non-finite".to_string(),
            Some(kry.residual_norm),
            Some(kry.iterations),
            timing,
        ));
    }

    // Far-field is best-effort: impedance/S11 stay valid when it is skipped
    // or fails, exactly like the reference.
    let mut field_blob = Vec::new();
    if let Some(ff_meta) = &request.metadata.far_field {
        if zin.re > 0.0 {
            let options = FarFieldOptions {
                theta_step_deg: ff_meta.theta_step_deg.unwrap_or(5.0),
                phi_step_deg: ff_meta.phi_step_deg.unwrap_or(5.0),
            };
            if let Ok(ff) = compute_far_field(
                &mesh,
                &sys,
                &kry.x,
                frequency_hz,
                0.5 * zin.re,
                &options,
                budget,
            ) {
                field_blob = FarFieldBlob::from_result(&ff).serialize();
            }
        }
    }

    timing.post = t2.elapsed_ms();

    Ok(PointResponse {
        status: 0,
        residual_norm: Some(kry.residual_norm),
        iteration_count: Some(kry.iterations),
        zin: Some(Complex64::new(zin.re, zin.im)),
        s11: Some(Complex64::new(s11.re, s11.im)),
        timing,
        field_blob,
        message: None,
    })
}
