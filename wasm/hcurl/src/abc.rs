//! First-order Silver–Müller absorbing boundary condition.
//!
//! Weak term **+j·k0·B** with `B_ij = ∫_facet (n×N_i)·(n×N_j) dA`, assembled
//! into the **imaginary** part of the system matrix. (Under the `e^{+jωt}`
//! convention the positive imaginary part is dissipative.)
//!
//! Evaluated with the 3-point edge-midpoint triangle rule (weight `area/3`
//! per point), which is exact for the quadratic integrand. The product
//! `(n×N_i)·(n×N_j)` is invariant under `n → −n`, so — unlike the far-field
//! code — the facet normal is **not** forced outward here.

use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::abi::ATTR_EXTERIOR_ABSORBING;
use crate::basis::{edge_length, TET_EDGES};
use crate::budget::Budget;
use crate::constants::C0;
use crate::edge::EdgeSystem;
use crate::error::Result;
use crate::geometry::tet_geometry;
use crate::mesh::Mesh;

fn sorted_triple(u: u32, v: u32, w: u32) -> (u32, u32, u32) {
    let mut s = [u, v, w];
    s.sort_unstable();
    (s[0], s[1], s[2])
}

/// ABC triplets: `(di, dj, j·k0·(area/3)·Σ_q t_q(i)·t_q(j))`.
pub fn assemble_abc_triplets(
    mesh: &Mesh,
    sys: &EdgeSystem,
    frequency_hz: f64,
    budget: &Budget,
) -> Result<Vec<(usize, usize, Complex64)>> {
    let k0 = 2.0 * PI * frequency_hz / C0;

    // Map each active tet's 4 faces (sorted vertex triple) to
    // (active ordinal, local facet opposite node f). Later tets overwrite
    // earlier ones; conforming meshes never collide.
    let mut facet_to_tet: HashMap<(u32, u32, u32), (usize, usize)> = HashMap::new();
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
            facet_to_tet.insert(sorted_triple(nodes[0], nodes[1], nodes[2]), (ord, f));
        }
    }

    // Per-facet triplet chunks in parallel; the ordered collect keeps the
    // triplet sequence identical to the serial loop.
    let chunks: Vec<Result<Vec<(usize, usize, Complex64)>>> =
        crate::par::map_indexed(&mesh.boundary_attributes, |bf, &attr| {
            if bf % 256 == 0 {
                budget.check()?;
            }
            let mut chunk = Vec::new();
            if attr != ATTR_EXTERIOR_ABSORBING {
                return Ok(chunk);
            }
            let base = bf * 3;
            let (u, v, w) = (
                mesh.boundary_facets[base],
                mesh.boundary_facets[base + 1],
                mesh.boundary_facets[base + 2],
            );
            let &(ord, local_facet) = match facet_to_tet.get(&sorted_triple(u, v, w)) {
                Some(hit) => hit,
                None => return Ok(chunk),
            };
            let tet = sys.active_tets[ord] as usize;
            let geom = tet_geometry(mesh, tet)?;

            // Facet normal from the stored (u,v,w) order; not forced outward.
            let pu = mesh.vertex(u);
            let pv = mesh.vertex(v);
            let pw = mesh.vertex(w);
            let vu = [pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]];
            let wu = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]];
            let mut n = [
                vu[1] * wu[2] - vu[2] * wu[1],
                vu[2] * wu[0] - vu[0] * wu[2],
                vu[0] * wu[1] - vu[1] * wu[0],
            ];
            let area2 = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if !(area2 > 0.0) {
                return Ok(chunk);
            }
            let area = area2 / 2.0;
            n[0] /= area2;
            n[1] /= area2;
            n[2] /= area2;

            // The 3 local edges lying on the facet, in TET_EDGES order.
            let local_nodes: Vec<usize> = (0..4).filter(|&k| k != local_facet).collect();
            let facet_edges: Vec<usize> = (0..6)
                .filter(|&e| {
                    let [a, b] = TET_EDGES[e];
                    local_nodes.contains(&a) && local_nodes.contains(&b)
                })
                .collect();
            if facet_edges.len() != 3 {
                return Ok(chunk);
            }

            // Tangential traces t_e(q) = n × N_e at the 3 edge midpoints q.
            // traces[q][3*ei + k]: q, ei index facet_edges.
            let mut traces = [[0.0f64; 9]; 3];
            for (qi, &qe) in facet_edges.iter().enumerate() {
                let [qa, qb] = TET_EDGES[qe];
                for (ei, &e) in facet_edges.iter().enumerate() {
                    let [a, b] = TET_EDGES[e];
                    let la = if a == qa || a == qb { 0.5 } else { 0.0 };
                    let lb = if b == qa || b == qb { 0.5 } else { 0.0 };
                    let len = edge_length(mesh, tet, e);
                    let f = sys.tet_edge_sign[ord * 6 + e] as f64 * len;
                    let ga = geom.grad_lambda(a);
                    let gb = geom.grad_lambda(b);
                    let nvec = [
                        f * (la * gb[0] - lb * ga[0]),
                        f * (la * gb[1] - lb * ga[1]),
                        f * (la * gb[2] - lb * ga[2]),
                    ];
                    traces[qi][ei * 3] = n[1] * nvec[2] - n[2] * nvec[1];
                    traces[qi][ei * 3 + 1] = n[2] * nvec[0] - n[0] * nvec[2];
                    traces[qi][ei * 3 + 2] = n[0] * nvec[1] - n[1] * nvec[0];
                }
            }

            let weight = area / 3.0;
            for (ii, &ei) in facet_edges.iter().enumerate() {
                let di = sys.dof_of_edge[sys.tet_edges[ord * 6 + ei] as usize];
                if di < 0 {
                    continue;
                }
                for (jj, &ej) in facet_edges.iter().enumerate() {
                    let dj = sys.dof_of_edge[sys.tet_edges[ord * 6 + ej] as usize];
                    if dj < 0 {
                        continue;
                    }
                    let mut dot = 0.0;
                    for q in 0..3 {
                        dot += traces[q][ii * 3] * traces[q][jj * 3]
                            + traces[q][ii * 3 + 1] * traces[q][jj * 3 + 1]
                            + traces[q][ii * 3 + 2] * traces[q][jj * 3 + 2];
                    }
                    chunk.push((
                        di as usize,
                        dj as usize,
                        Complex64::new(0.0, k0 * weight * dot),
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
    use crate::edge::build_edge_system;
    use crate::mesh::Material;

    fn abc_tet_mesh() -> Mesh {
        Mesh {
            vertices: vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
            tetrahedra: vec![0, 1, 2, 3],
            tet_regions: vec![1],
            materials: vec![Material::air()],
            boundary_facets: vec![0, 1, 2],
            boundary_attributes: vec![ATTR_EXTERIOR_ABSORBING],
            port_positive_facets: vec![],
            port_negative_facets: vec![],
        }
    }

    #[test]
    fn abc_lands_in_imaginary_part_only() {
        let mesh = abc_tet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let triplets = assemble_abc_triplets(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
        assert_eq!(triplets.len(), 9); // 3 facet edges × 3
                                       // Purely imaginary triplets; diagonal entries are positive
                                       // (off-diagonals may be zero or negative).
        assert!(triplets.iter().all(|&(_, _, v)| v.re == 0.0));
        for &(r, c, v) in &triplets {
            if r == c {
                assert!(v.im > 0.0, "diagonal {r} im={}", v.im);
            }
        }
    }

    #[test]
    fn abc_diagonal_matches_hand_computation() {
        // Facet (0,1,2), n = +z, area = 0.5, weight = area/3 = 1/6.
        // Edge (0,1) (local 0, global 0 → dof 0). N = s·l·(λ0∇λ1 − λ1∇λ0):
        //   q=(0,1) mid: N=(1,0.5,0.5), t=n×N=(−0.5,1,0),     |t|²=1.25
        //   q=(0,2) mid: N=(0.5,0,0),   t=(0,0.5,0),          |t|²=0.25
        //   q=(1,2) mid: N=(0.5,0.5,0.5), t=(−0.5,0.5,0),     |t|²=0.5
        // dot(0,0) = 2.0, so the triplet imag = k0·(1/6)·2 = k0/3.
        let mesh = abc_tet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let triplets = assemble_abc_triplets(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
        let k0 = 2.0 * PI * 3.0e9 / C0;
        let t00 = triplets
            .iter()
            .find(|&&(r, c, _)| r == 0 && c == 0)
            .unwrap();
        assert!((t00.2.im - k0 / 3.0).abs() < 1e-12 * k0, "im={}", t00.2.im);
    }

    #[test]
    fn abc_matrix_is_symmetric() {
        // B must be exactly symmetric: B_ij = B_ji by construction.
        let mesh = abc_tet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let triplets = assemble_abc_triplets(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
        for &(r, c, v) in &triplets {
            let mirror = triplets
                .iter()
                .find(|&&(rr, cc, _)| rr == c && cc == r)
                .unwrap();
            assert!((v.im - mirror.2.im).abs() < 1e-15);
        }
    }

    #[test]
    fn non_absorbing_facets_are_skipped() {
        let mut mesh = abc_tet_mesh();
        mesh.boundary_attributes = vec![crate::abi::ATTR_CONDUCTOR_PEC];
        let sys = build_edge_system(&mesh).unwrap();
        let triplets = assemble_abc_triplets(&mesh, &sys, 3.0e9, &Budget::none()).unwrap();
        assert!(triplets.is_empty());
    }
}
