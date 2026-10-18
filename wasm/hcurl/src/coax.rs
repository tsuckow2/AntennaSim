//! Coaxial wave port: geometry-found port faces, TEM absorbing term,
//! incident-wave drive, and V/I impedance extraction.
//!
//! The driven problem replaces the lumped gap feed with the physical
//! feed structure: a (square) coaxial section below the ground plane,
//! modeled in the mesh, whose port plane carries
//!   * an absorbing condition — the Silver–Müller tangential-mass term
//!     with `k → k_d = k0·√εr_diel`, exact for the coax TEM mode because
//!     its E/H ratio is the local wave impedance `η_d = η0/√εr_diel`
//!     pointwise on any cross-section; and
//!   * an incident TEM wave (round-coax `1/r` profile; the square
//!     section's non-TEM launch residue is evanescent and dies within a
//!     cross-section width of the port plane, and the extraction below
//!     is a field ratio, so the launch amplitude/profile only sets the
//!     overall scale).
//!
//! Impedance is extracted at the reference plane `z_ref` inside the
//! dielectric: `V = −∫E·dl` radially from outer wall to pin and
//! `I = ∮H·dl` around the pin with `H = ∇×E/(−jωμ0)` (the `e^{+jωt}`
//! convention), so `Zin = V/I` needs no analytic coax impedance and no
//! de-embedding rotation.

use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::abc::facet_tangential_mass;
use crate::abi::CoaxPortRequest;
use crate::basis::{curl_vector, edge_length, nedelec_at};
use crate::budget::Budget;
use crate::constants::{C0, ETA0, MU0};
use crate::edge::EdgeSystem;
use crate::error::{Error, Result};
use crate::geometry::tet_geometry;
use crate::mesh::Mesh;

/// One port face: active-tet ordinal + local facet index, and whether it
/// lies on the top (absorbing-only) plane.
#[derive(Debug, Clone, Copy)]
pub struct PortFace {
    pub ord: usize,
    pub local_facet: usize,
    pub top: bool,
}

fn sorted_triple(u: u32, v: u32, w: u32) -> (u32, u32, u32) {
    let mut s = [u, v, w];
    s.sort_unstable();
    (s[0], s[1], s[2])
}

/// Find the port faces geometrically: boundary faces (shared by exactly
/// one active tet) lying in a port plane, with centroid inside the
/// dielectric annulus, backed by a dielectric tet of the port's εr.
pub fn find_port_faces(mesh: &Mesh, sys: &EdgeSystem, cp: &CoaxPortRequest) -> Vec<PortFace> {
    // Face-use map over active tets: sorted vertex triple -> (count, first hit).
    let mut use_map: HashMap<(u32, u32, u32), (usize, usize, usize)> = HashMap::new();
    for (ord, &tet) in sys.active_tets.iter().enumerate() {
        let base = tet as usize * 4;
        let v = [
            mesh.tetrahedra[base],
            mesh.tetrahedra[base + 1],
            mesh.tetrahedra[base + 2],
            mesh.tetrahedra[base + 3],
        ];
        for f in 0..4 {
            let nodes: Vec<u32> = (0..4).filter(|&k| k != f).map(|k| v[k]).collect();
            let key = sorted_triple(nodes[0], nodes[1], nodes[2]);
            let e = use_map.entry(key).or_insert((0, ord, f));
            e.0 += 1;
        }
    }

    const Z_TOL: f64 = 1e-7;
    let mut faces = Vec::new();
    for &(count, ord, f) in use_map.values() {
        if count != 1 {
            continue;
        }
        let tet = sys.active_tets[ord] as usize;
        let base = tet * 4;
        let v = [
            mesh.tetrahedra[base],
            mesh.tetrahedra[base + 1],
            mesh.tetrahedra[base + 2],
            mesh.tetrahedra[base + 3],
        ];
        let nodes: Vec<u32> = (0..4).filter(|&k| k != f).map(|k| v[k]).collect();
        let pts: Vec<[f64; 3]> = nodes.iter().map(|&n| mesh.vertex(n)).collect();
        let z = pts[0][2];
        if pts.iter().any(|p| (p[2] - z).abs() > Z_TOL) {
            continue;
        }
        let top = if (z - cp.z_port).abs() <= Z_TOL {
            false
        } else if cp.z_port_top.map_or(false, |zt| (z - zt).abs() <= Z_TOL) {
            true
        } else {
            continue;
        };
        let cen = [
            (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0,
            (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0,
        ];
        let cheb = (cen[0] - cp.center[0]).abs().max((cen[1] - cp.center[1]).abs());
        if cheb < cp.half_pin - 1e-6 || cheb > cp.half_diel + 1e-6 {
            continue;
        }
        // The backing tet must be the port dielectric.
        let eps = mesh.material_for_region(mesh.tet_regions[tet]).epsilon_r;
        if (eps - cp.eps_diel).abs() > 0.02 * cp.eps_diel.max(1.0) {
            continue;
        }
        faces.push(PortFace {
            ord,
            local_facet: f,
            top,
        });
    }
    faces
}

/// Absorbing triplets for all port faces: `(di, dj, j·k_d·B_ij)`.
pub fn assemble_port_triplets(
    mesh: &Mesh,
    sys: &EdgeSystem,
    cp: &CoaxPortRequest,
    faces: &[PortFace],
    frequency_hz: f64,
    budget: &Budget,
) -> Result<Vec<(usize, usize, Complex64)>> {
    let k_d = 2.0 * PI * frequency_hz / C0 * cp.eps_diel.sqrt();
    let mut triplets = Vec::new();
    for (i, face) in faces.iter().enumerate() {
        if i % 256 == 0 {
            budget.check()?;
        }
        if let Some(mass) = facet_tangential_mass(mesh, sys, face.ord, face.local_facet)? {
            for (di, dj, bij) in mass.entries {
                triplets.push((di, dj, Complex64::new(0.0, k_d * bij)));
            }
        }
    }
    Ok(triplets)
}

/// Incident-wave RHS on the (bottom) port faces:
/// `b_i += 2·j·k_d·∫ (n̂×N_i)·(n̂×E_inc) dA`, evaluated with the same
/// 3-point edge-midpoint rule as the mass term. `E_inc` is the round-coax
/// TEM profile `E0·(a/r)·r̂` with `E0 = η_d/(2πa)` (incident current
/// ≈ 1 A around the pin; the exact amplitude is immaterial — the
/// extraction is a ratio of total fields).
pub fn build_coax_rhs(
    mesh: &Mesh,
    sys: &EdgeSystem,
    cp: &CoaxPortRequest,
    faces: &[PortFace],
    frequency_hz: f64,
) -> Result<Vec<Complex64>> {
    let k_d = 2.0 * PI * frequency_hz / C0 * cp.eps_diel.sqrt();
    let eta_d = ETA0 / cp.eps_diel.sqrt();
    let e0 = eta_d / (2.0 * PI * cp.half_pin);
    let coef = Complex64::new(0.0, 2.0 * k_d);
    let mut b = vec![Complex64::new(0.0, 0.0); sys.free_dof_count];
    for face in faces.iter().filter(|f| !f.top) {
        let mass = match facet_tangential_mass(mesh, sys, face.ord, face.local_facet)? {
            Some(m) => m,
            None => continue,
        };
        let n = mass.normal;
        for ei in 0..3 {
            let d = mass.dofs[ei];
            if d < 0 {
                continue;
            }
            let mut integ = 0.0f64;
            for q in 0..3 {
                let p = mass.midpoints[q];
                let rv = [p[0] - cp.center[0], p[1] - cp.center[1]];
                let r = (rv[0] * rv[0] + rv[1] * rv[1]).sqrt().max(0.5 * cp.half_pin);
                let e_inc = [
                    e0 * (cp.half_pin / r) * (rv[0] / r),
                    e0 * (cp.half_pin / r) * (rv[1] / r),
                    0.0,
                ];
                // s = n̂ × E_inc
                let s = [
                    n[1] * e_inc[2] - n[2] * e_inc[1],
                    n[2] * e_inc[0] - n[0] * e_inc[2],
                    n[0] * e_inc[1] - n[1] * e_inc[0],
                ];
                let t = [
                    mass.traces[q][ei * 3],
                    mass.traces[q][ei * 3 + 1],
                    mass.traces[q][ei * 3 + 2],
                ];
                integ += t[0] * s[0] + t[1] * s[1] + t[2] * s[2];
            }
            b[d as usize] += coef * (mass.area / 3.0) * integ;
        }
    }
    Ok(b)
}

/// Locate the active tet containing `p` among `candidates` (active-tet
/// ordinals), falling back to the nearest candidate centroid.
fn locate_tet(mesh: &Mesh, candidates: &[usize], sys: &EdgeSystem, p: [f64; 3]) -> Result<usize> {
    let mut best = usize::MAX;
    let mut best_d = f64::INFINITY;
    for &ord in candidates {
        let tet = sys.active_tets[ord] as usize;
        let g = tet_geometry(mesh, tet)?;
        let inside = (0..4).all(|m| {
            let lam = g.lambda(m, p);
            lam >= -1e-9 && lam <= 1.0 + 1e-9
        });
        if inside {
            return Ok(ord);
        }
        let c = g.centroid;
        let d = (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2) + (c[2] - p[2]).powi(2);
        if d < best_d {
            best_d = d;
            best = ord;
        }
    }
    if best == usize::MAX {
        return Err(Error::Mesh(
            "coax extraction: no candidate tet near the reference plane".to_string(),
        ));
    }
    Ok(best)
}

/// Electric field at `p` (in the tet located among `candidates`).
fn eval_e(
    mesh: &Mesh,
    sys: &EdgeSystem,
    candidates: &[usize],
    x: &[Complex64],
    p: [f64; 3],
) -> Result<[Complex64; 3]> {
    let ord = locate_tet(mesh, candidates, sys, p)?;
    let tet = sys.active_tets[ord] as usize;
    let geom = tet_geometry(mesh, tet)?;
    let lambda = [
        geom.lambda(0, p),
        geom.lambda(1, p),
        geom.lambda(2, p),
        geom.lambda(3, p),
    ];
    let mut e_vec = [Complex64::new(0.0, 0.0); 3];
    for edge in 0..6 {
        let d = sys.dof_of_edge[sys.tet_edges[ord * 6 + edge] as usize];
        if d < 0 {
            continue;
        }
        let nvec = nedelec_at(
            &geom,
            &lambda,
            edge,
            sys.tet_edge_sign[ord * 6 + edge],
            edge_length(mesh, tet, edge),
        );
        let xd = x[d as usize];
        for c in 0..3 {
            e_vec[c] += xd * nvec[c];
        }
    }
    Ok(e_vec)
}

/// `∇×E` at `p` (constant per tet).
fn eval_curl_e(
    mesh: &Mesh,
    sys: &EdgeSystem,
    candidates: &[usize],
    x: &[Complex64],
    p: [f64; 3],
) -> Result<[Complex64; 3]> {
    let ord = locate_tet(mesh, candidates, sys, p)?;
    let tet = sys.active_tets[ord] as usize;
    let geom = tet_geometry(mesh, tet)?;
    let mut curl = [Complex64::new(0.0, 0.0); 3];
    for edge in 0..6 {
        let d = sys.dof_of_edge[sys.tet_edges[ord * 6 + edge] as usize];
        if d < 0 {
            continue;
        }
        let cv = curl_vector(&geom, edge, edge_length(mesh, tet, edge));
        let s = sys.tet_edge_sign[ord * 6 + edge] as f64;
        let xd = x[d as usize] * s;
        for c in 0..3 {
            curl[c] += xd * cv[c];
        }
    }
    Ok(curl)
}

/// Voltage and current at the reference plane (see [`coax_impedance`]).
/// `V = −∫E·dl` from outer wall (negative) to pin (positive);
/// `I = ∮H·dl` around the pin, CCW from +z.
pub fn coax_vi(
    mesh: &Mesh,
    sys: &EdgeSystem,
    cp: &CoaxPortRequest,
    x: &[Complex64],
    frequency_hz: f64,
) -> Result<(Complex64, Complex64)> {
    let omega = 2.0 * PI * frequency_hz;
    // Candidate tets around the reference plane.
    let mut candidates: Vec<usize> = Vec::new();
    for (ord, &tet) in sys.active_tets.iter().enumerate() {
        let g = tet_geometry(mesh, tet as usize)?;
        let c = g.centroid;
        let cheb = (c[0] - cp.center[0]).abs().max((c[1] - cp.center[1]).abs());
        if (c[2] - cp.z_ref).abs() < 3e-3 && cheb < cp.half_diel + 2e-3 {
            candidates.push(ord);
        }
    }
    if candidates.is_empty() {
        return Err(Error::Mesh(
            "coax extraction: no tets around the reference plane".to_string(),
        ));
    }

    // V: −∫E·dl from the negative terminal (outer wall) to the positive
    // (pin) — accumulated here as +∫E·dl along pin→wall, the same thing.
    let nv = 24usize;
    let mut v = Complex64::new(0.0, 0.0);
    let dr = (cp.half_diel - cp.half_pin) / nv as f64;
    for k in 0..nv {
        let r = cp.half_pin + (k as f64 + 0.5) * dr;
        let p = [cp.center[0] + r, cp.center[1], cp.z_ref];
        let e = eval_e(mesh, sys, &candidates, x, p)?;
        v += e[0] * dr;
    }

    // I: square loop at half-width s, CCW viewed from +z.
    let s = 0.5 * (cp.half_pin + cp.half_diel);
    let hfac = Complex64::new(0.0, 1.0) / (omega * MU0); // 1/(−jωμ) = j/(ωμ)
    let ni = 16usize;
    let dl = 2.0 * s / ni as f64;
    let sides: [([f64; 2], [f64; 2]); 4] = [
        ([s, -s], [0.0, 1.0]),   // +x side, +y direction
        ([s, s], [-1.0, 0.0]),   // +y side, −x direction
        ([-s, s], [0.0, -1.0]),  // −x side, −y direction
        ([-s, -s], [1.0, 0.0]),  // −y side, +x direction
    ];
    let mut i_total = Complex64::new(0.0, 0.0);
    for (corner, t_hat) in sides {
        for k in 0..ni {
            // Sample point: fixed coordinate from the corner, swept
            // coordinate along t̂ from −s to +s.
            let sweep = -s + (k as f64 + 0.5) * dl;
            let (px, py) = if t_hat[1] != 0.0 {
                (cp.center[0] + corner[0], cp.center[1] + t_hat[1] * sweep)
            } else {
                (cp.center[0] + t_hat[0] * sweep, cp.center[1] + corner[1])
            };
            let p = [px, py, cp.z_ref];
            let curl = eval_curl_e(mesh, sys, &candidates, x, p)?;
            let h = [
                curl[0] * hfac,
                curl[1] * hfac,
                curl[2] * hfac,
            ];
            i_total += (h[0] * t_hat[0] + h[1] * t_hat[1]) * dl;
        }
    }
    if i_total.norm() == 0.0 {
        return Err(Error::Mesh("coax extraction: loop current is zero".to_string()));
    }
    Ok((v, i_total))
}

/// Input impedance at the reference plane: `Zin = V/I` with
/// `V = −∫E·dl` radially wall→pin (along +x) and `I = ∮H·dl`
/// around the pin on a square loop at half-width `(a+b)/2`,
/// `H = ∇×E/(−jωμ0)`.
pub fn coax_impedance(
    mesh: &Mesh,
    sys: &EdgeSystem,
    cp: &CoaxPortRequest,
    x: &[Complex64],
    frequency_hz: f64,
) -> Result<Complex64> {
    let (v, i) = coax_vi(mesh, sys, cp, x, frequency_hz)?;
    Ok(v / i)
}
