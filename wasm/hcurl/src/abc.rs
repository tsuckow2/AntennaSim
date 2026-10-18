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

/// Tangential-trace mass data for one face of an active tet.
///
/// `entries` holds the real facet matrix `B_ij = (area/3)·Σ_q t_i(q)·t_j(q)`
/// with `t_e(q) = n̂ × N_e` the tangential traces at the 3 facet-edge
/// midpoints `q` (the 3-point edge-midpoint rule, exact for the quadratic
/// integrand). `normal` is the OUTWARD unit normal (away from the tet's
/// opposite vertex); `traces[q][ei*3 + c]` and `dofs[ei]` are indexed by
/// the facet-edge order of `entries`. Shared by the Silver–Müller ABC
/// (coefficient `j·k0`) and the coax wave port (`j·k_d`, plus an incident
/// RHS built from the same traces) — the mass is invariant under
/// `n̂ → −n̂`, so orienting it outward changes nothing for the ABC.
pub(crate) struct FacetMass {
    pub entries: Vec<(usize, usize, f64)>,
    pub normal: [f64; 3],
    pub area: f64,
    pub midpoints: [[f64; 3]; 3],
    pub traces: [[f64; 9]; 3],
    pub dofs: [i32; 3],
}

/// Tangential mass of the face of active tet `ord` opposite local node
/// `local_facet`. `None` when the face is degenerate.
pub(crate) fn facet_tangential_mass(
    mesh: &Mesh,
    sys: &EdgeSystem,
    ord: usize,
    local_facet: usize,
) -> Result<Option<FacetMass>> {
    let tet = sys.active_tets[ord] as usize;
    let geom = tet_geometry(mesh, tet)?;
    let base = tet * 4;
    let v = [
        mesh.tetrahedra[base],
        mesh.tetrahedra[base + 1],
        mesh.tetrahedra[base + 2],
        mesh.tetrahedra[base + 3],
    ];
    let local_nodes: Vec<usize> = (0..4).filter(|&k| k != local_facet).collect();
    let (pu, pv, pw) = (
        mesh.vertex(v[local_nodes[0]]),
        mesh.vertex(v[local_nodes[1]]),
        mesh.vertex(v[local_nodes[2]]),
    );
    let opp = mesh.vertex(v[local_facet]);
    let vu = [pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]];
    let wu = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]];
    let mut n = [
        vu[1] * wu[2] - vu[2] * wu[1],
        vu[2] * wu[0] - vu[0] * wu[2],
        vu[0] * wu[1] - vu[1] * wu[0],
    ];
    let area2 = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if !(area2 > 0.0) {
        return Ok(None);
    }
    let area = area2 / 2.0;
    n = [n[0] / area2, n[1] / area2, n[2] / area2];
    // Orient outward: away from the tet's opposite vertex.
    if n[0] * (opp[0] - pu[0]) + n[1] * (opp[1] - pu[1]) + n[2] * (opp[2] - pu[2]) > 0.0 {
        n = [-n[0], -n[1], -n[2]];
    }

    let facet_edges: Vec<usize> = (0..6)
        .filter(|&e| {
            let [a, b] = TET_EDGES[e];
            local_nodes.contains(&a) && local_nodes.contains(&b)
        })
        .collect();
    if facet_edges.len() != 3 {
        return Ok(None);
    }

    let mut traces = [[0.0f64; 9]; 3];
    let mut midpoints = [[0.0f64; 3]; 3];
    let mut dofs = [-1i32; 3];
    for (ei, &e) in facet_edges.iter().enumerate() {
        dofs[ei] = sys.dof_of_edge[sys.tet_edges[ord * 6 + e] as usize];
    }
    for (qi, &qe) in facet_edges.iter().enumerate() {
        let [qa, qb] = TET_EDGES[qe];
        let pa = mesh.vertex(v[qa]);
        let pb = mesh.vertex(v[qb]);
        midpoints[qi] = [
            0.5 * (pa[0] + pb[0]),
            0.5 * (pa[1] + pb[1]),
            0.5 * (pa[2] + pb[2]),
        ];
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
    let mut entries = Vec::with_capacity(9);
    for (ii, _) in facet_edges.iter().enumerate() {
        let di = dofs[ii];
        if di < 0 {
            continue;
        }
        for (jj, _) in facet_edges.iter().enumerate() {
            let dj = dofs[jj];
            if dj < 0 {
                continue;
            }
            let mut dot = 0.0;
            for q in 0..3 {
                dot += traces[q][ii * 3] * traces[q][jj * 3]
                    + traces[q][ii * 3 + 1] * traces[q][jj * 3 + 1]
                    + traces[q][ii * 3 + 2] * traces[q][jj * 3 + 2];
            }
            entries.push((di as usize, dj as usize, weight * dot));
        }
    }
    Ok(Some(FacetMass {
        entries,
        normal: n,
        area,
        midpoints,
        traces,
        dofs,
    }))
}

/// ABC triplets: `(di, dj, j·k0·B_ij)` over the absorbing facets.
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
            let mass = match facet_tangential_mass(mesh, sys, ord, local_facet)? {
                Some(m) => m,
                None => return Ok(chunk),
            };
            for (di, dj, bij) in mass.entries {
                chunk.push((di, dj, Complex64::new(0.0, k0 * bij)));
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
            coax_port: None,
            exclude_boxes: vec![],
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
