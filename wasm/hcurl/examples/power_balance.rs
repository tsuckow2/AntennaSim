//! Power-balance discriminator (Diagnostic 2 of the FEM port audit).
//!
//! Solves one request with AMS, retains the field, and computes the accepted
//! power **independently of the port voltage path**:
//!
//! - `P_rad`: Stratton–Chu far-field pattern integrated over the sphere
//!   ([`compute_far_field`] is called with a dummy accepted power of 1 W;
//!   only its `radiated_power_w` — which never touches that input — is used).
//! - `P_loss`: volume integral `½·σ_eff·∫|E|² dV` per active tet, with
//!   `σ_eff = σ + ω·ε0·εr·tanδ` (the model's complex-permittivity loss).
//!   Conductors are PEC in this model (conductor tets are excluded), so
//!   there is no conductor-loss term to integrate.
//!
//! `R_power = 2·(P_rad + P_loss)/|I|²` at the 1 A feed. Agreement with
//! `Re(Zin)` convicts the driven port model; a value ≫ Re(Zin) toward
//! ~50 Ω would reopen the voltage path.
//!
//! ```sh
//! cargo run --release --example power_balance -- /path/to/2300.req
//! ```

use hcurl_fem::abi::PointRequest;
use hcurl_fem::ams::AmsPreconditioner;
use hcurl_fem::assembly::assemble_hcurl;
use hcurl_fem::basis::{edge_length, local_matrices};
use hcurl_fem::budget::Budget;
use hcurl_fem::constants::EPS0;
use hcurl_fem::edge::build_edge_system;
use hcurl_fem::farfield::{compute_far_field, FarFieldOptions};
use hcurl_fem::geometry::tet_geometry;
use hcurl_fem::krylov::{bicgstab, Preconditioner, DEFAULT_MAX_ITER, DEFAULT_TOL};
use hcurl_fem::mesh::Mesh;
use hcurl_fem::port::{build_port_rhs, port_voltage, select_gap_feed, terminals_from_facets};
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: power_balance <request.req>");
    let bytes = std::fs::read(&path).expect("read request file");
    let request = PointRequest::parse(&bytes).expect("parse request");
    let frequency_hz = request.frequency_hz;
    let budget = Budget::none();
    let omega = 2.0 * PI * frequency_hz;

    let mesh = Mesh::from_request(&request).expect("mesh");
    let sys = build_edge_system(&mesh).expect("edge system");
    let matrix = assemble_hcurl(&mesh, &sys, frequency_hz, &budget).expect("assembly");

    let (positive, negative) = terminals_from_facets(&mesh).expect("terminals");
    let feed = select_gap_feed(&mesh, &sys, positive, negative).expect("feed");
    let rhs = build_port_rhs(&mesh, &sys, &feed, frequency_hz).expect("rhs");

    let t = Instant::now();
    let ams = AmsPreconditioner::build(&mesh, &sys, &matrix).expect("ams build");
    let res = bicgstab(
        &matrix,
        &rhs,
        DEFAULT_TOL,
        DEFAULT_MAX_ITER,
        Preconditioner::Ams(ams),
        &budget,
    )
    .expect("ams solve");
    let wall = t.elapsed().as_secs_f64();
    assert!(res.converged, "AMS did not converge");
    let zin = port_voltage(&mesh, &sys, &feed, negative, &res.x).expect("zin");
    println!(
        "AMS: iters={} residual={:.3e} wall={:.1}s Zin={:.6}-j{:.6}",
        res.iterations, res.residual_norm, wall, zin.re, -zin.im
    );

    // --- Independent radiated power (far-field integral). ---
    let ff = compute_far_field(
        &mesh,
        &sys,
        &res.x,
        frequency_hz,
        1.0, // dummy: radiated_power_w does not depend on this input
        &FarFieldOptions::default(),
        &budget,
    )
    .expect("far field");
    let p_rad = ff.radiated_power_w;
    println!(
        "P_rad (far-field integral) = {:.6e} W  (max directivity {:.2} dBi)",
        p_rad, ff.max_directivity_dbi
    );

    // --- Independent dissipated power (volume loss integral). ---
    let mut p_loss_by_attr: BTreeMap<u32, f64> = BTreeMap::new();
    for (ord, &tet) in sys.active_tets.iter().enumerate() {
        let tet = tet as usize;
        let material = mesh.material_for_region(mesh.tet_regions[tet]);
        let sigma_eff = material.conductivity_s_m
            + omega * EPS0 * material.epsilon_r * material.loss_tangent;
        if !(sigma_eff > 0.0) {
            continue;
        }
        let geom = tet_geometry(&mesh, tet).expect("geom");
        let mut lengths = [0.0f64; 6];
        let mut signs = [0i8; 6];
        for e in 0..6 {
            lengths[e] = edge_length(&mesh, tet, e);
            signs[e] = sys.tet_edge_sign[ord * 6 + e];
        }
        let (_, mass) = local_matrices(&geom, &lengths, &signs);
        // ∫|E|² dV = Σ_ij conj(x_i)·M_ij·x_j over the tet's free edges.
        let mut e2 = 0.0f64;
        for i in 0..6 {
            let di = sys.dof_of_edge[sys.tet_edges[ord * 6 + i] as usize];
            if di < 0 {
                continue;
            }
            let xi = res.x[di as usize];
            for j in 0..6 {
                let dj = sys.dof_of_edge[sys.tet_edges[ord * 6 + j] as usize];
                if dj < 0 {
                    continue;
                }
                e2 += (xi.conj() * mass[i * 6 + j] * res.x[dj as usize]).re;
            }
        }
        *p_loss_by_attr.entry(material.attribute).or_insert(0.0) += 0.5 * sigma_eff * e2;
    }
    let p_loss: f64 = p_loss_by_attr.values().sum();
    for (attr, p) in &p_loss_by_attr {
        let m = mesh.material_for_region(*attr);
        println!(
            "P_loss attr {} (er={} sigma={:e} tand={}) = {:.6e} W",
            attr, m.epsilon_r, m.conductivity_s_m, m.loss_tangent, p
        );
    }

    let p_acc_fields = p_rad + p_loss;
    let r_power = 2.0 * p_acc_fields; // |I| = 1 A
    println!(
        "P_acc(fields) = {:.6e} W  ->  R_power = {:.4} ohm   vs   Re(Zin) = {:.4} ohm   ratio = {:.3}",
        p_acc_fields,
        r_power,
        zin.re,
        r_power / zin.re
    );
}
