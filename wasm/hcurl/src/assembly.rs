//! H(curl) system assembly.
//!
//! Assembles the complex free-DOF system matrix
//!
//! ```text
//! A = K/μr − k0²·εc·Q + j·k0·B_abc
//! ```
//!
//! where `K`, `Q` are the real curl-curl and mass brackets, `εc` is the
//! complex relative permittivity, and `B_abc` is the real Silver–Müller
//! facet matrix ([`crate::abc`]). PEC-constrained edges are eliminated —
//! never assembled — so there is no identity-row surgery.
//!
//! Time convention `e^{+jωt}`: the ABC term enters as **+j·k0·B** (positive
//! imaginary part = dissipative).

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::abc::assemble_abc_triplets;
use crate::basis::{edge_length, local_matrices};
use crate::budget::Budget;
use crate::constants::{C0, EPS0};
use crate::csr::CsrMatrix;
use crate::edge::EdgeSystem;
use crate::error::Result;
use crate::geometry::tet_geometry;
use crate::mesh::Mesh;

/// Assemble the full complex system matrix at `frequency_hz`.
///
/// Returns the free-DOF matrix built from coordinate triplets (duplicates
/// summed by the CSR builder).
pub fn assemble_hcurl(
    mesh: &Mesh,
    sys: &EdgeSystem,
    frequency_hz: f64,
    budget: &Budget,
) -> Result<CsrMatrix> {
    let mut triplets = assemble_volume_triplets(mesh, sys, frequency_hz, budget)?;
    triplets.extend(assemble_abc_triplets(mesh, sys, frequency_hz, budget)?);
    let n = sys.free_dof_count;
    Ok(CsrMatrix::from_triplets(n, n, triplets))
}

/// Volume triplets: `(1/μr)·K + (−k0²·εc)·Q` per active tet.
///
/// Complex permittivity: `εc = εr·(1 − j·tanδ) − j·σ/(ω·ε0)`, so
/// `epsIm = −εr·tanδ − σ/(ω·ε0) ≤ 0` and the imaginary mass coefficient
/// `massIm = −k0²·epsIm ≥ 0` for lossy media.
pub fn assemble_volume_triplets(
    mesh: &Mesh,
    sys: &EdgeSystem,
    frequency_hz: f64,
    budget: &Budget,
) -> Result<Vec<(usize, usize, Complex64)>> {
    let omega = 2.0 * PI * frequency_hz;
    let k0 = omega / C0;

    // Per-tet triplet chunks computed in parallel. `map_indexed` preserves
    // tet order, so flattening below reproduces the serial triplet sequence
    // exactly and the CSR build is bit-identical. `Budget` is `Sync`
    // (atomic flag + `Instant`), so cancellation stays responsive.
    let chunks: Vec<Result<Vec<(usize, usize, Complex64)>>> =
        crate::par::map_indexed(&sys.active_tets, |ord, &tet| {
            if ord % 256 == 0 {
                budget.check()?;
            }
            let tet = tet as usize;
            let material = mesh.material_for_region(mesh.tet_regions[tet]);
            let geom = tet_geometry(mesh, tet)?;

            let mut lengths = [0.0f64; 6];
            let mut signs = [0i8; 6];
            for e in 0..6 {
                lengths[e] = edge_length(mesh, tet, e);
                signs[e] = sys.tet_edge_sign[ord * 6 + e];
            }
            let (curl, mass) = local_matrices(&geom, &lengths, &signs);

            let eps_re = material.epsilon_r;
            let eps_im = -material.epsilon_r * material.loss_tangent
                - material.conductivity_s_m / (omega * EPS0);
            let mass_re = -k0 * k0 * eps_re;
            let mass_im = -k0 * k0 * eps_im;
            let inv_mu = 1.0 / material.mu_r;

            let mut chunk = Vec::with_capacity(36);
            for i in 0..6 {
                let di = sys.dof_of_edge[sys.tet_edges[ord * 6 + i] as usize];
                if di < 0 {
                    continue;
                }
                for j in 0..6 {
                    let dj = sys.dof_of_edge[sys.tet_edges[ord * 6 + j] as usize];
                    if dj < 0 {
                        continue;
                    }
                    let kval = inv_mu * curl[i * 6 + j];
                    let qval = mass[i * 6 + j];
                    chunk.push((
                        di as usize,
                        dj as usize,
                        Complex64::new(kval + mass_re * qval, mass_im * qval),
                    ));
                }
            }
            Ok(chunk)
        });

    let mut triplets = Vec::new();
    for chunk in chunks {
        triplets.extend(chunk?);
    }
    Ok(triplets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::ATTR_CONDUCTOR_PEC;
    use crate::edge::build_edge_system;
    use crate::mesh::Material;

    fn unit_tet_mesh() -> Mesh {
        Mesh {
            vertices: vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
            tetrahedra: vec![0, 1, 2, 3],
            tet_regions: vec![1],
            materials: vec![Material::air()],
            boundary_facets: vec![],
            boundary_attributes: vec![],
            port_positive_facets: vec![],
            port_negative_facets: vec![],
        }
    }

    #[test]
    fn lossless_air_has_zero_imaginary_part() {
        let mesh = unit_tet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let a = assemble_hcurl(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
        assert_eq!((a.nrows(), a.ncols()), (6, 6));
        for i in 0..6 {
            let (_, vals) = a.row(i);
            assert!(vals.iter().all(|v| v.im == 0.0));
        }
        // A[0][0] = curl[0][0] − k0²·mass[0][0] = 4/3 − k0²/12.
        let k0 = 2.0 * PI * 3.0e9 / C0;
        let expected = 4.0 / 3.0 - k0 * k0 / 12.0;
        let d = a.diagonal();
        assert!((d[0].re - expected).abs() < 1e-9 * expected.abs());
    }

    #[test]
    fn conductivity_adds_positive_imaginary_mass() {
        let mut mesh = unit_tet_mesh();
        mesh.materials = vec![Material {
            attribute: 1,
            epsilon_r: 1.0,
            mu_r: 1.0,
            conductivity_s_m: 1.0, // lossy but not a conductor tet
            loss_tangent: 0.0,
        }];
        let sys = build_edge_system(&mesh).unwrap();
        let a = assemble_hcurl(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
        let d = a.diagonal();
        // imag = −k0²·epsIm·mass[0][0], epsIm = −σ/(ωε0) < 0 → imag > 0.
        assert!(d[0].im > 0.0);
        // Real part unchanged vs lossless.
        let k0 = 2.0 * PI * 3.0e9 / C0;
        assert!((d[0].re - (4.0 / 3.0 - k0 * k0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn pec_facet_eliminates_dofs_from_matrix() {
        let mut mesh = unit_tet_mesh();
        mesh.boundary_facets = vec![0, 1, 2];
        mesh.boundary_attributes = vec![ATTR_CONDUCTOR_PEC];
        let sys = build_edge_system(&mesh).unwrap();
        assert_eq!(sys.free_dof_count, 3); // edges (0,1),(0,2),(1,2) constrained
        let a = assemble_hcurl(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
        assert_eq!((a.nrows(), a.ncols()), (3, 3));
    }

    #[test]
    fn volume_matrix_is_complex_symmetric() {
        // A must equal its transpose (complex-symmetric, not Hermitian).
        let mesh = unit_tet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let a = assemble_hcurl(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
        for i in 0..6 {
            let (cols_i, vals_i) = a.row(i);
            for (&c, &v) in cols_i.iter().zip(vals_i.iter()) {
                let (cols_c, vals_c) = a.row(c);
                let pos = cols_c.iter().position(|&cc| cc == i).unwrap();
                assert!((v - vals_c[pos]).norm() < 1e-12);
            }
        }
    }
}
