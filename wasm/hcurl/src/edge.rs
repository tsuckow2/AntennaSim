//! Global edge enumeration and PEC elimination.
//!
//! Global edges are oriented low→high vertex index and numbered in
//! first-seen order over the active (non-conductor) tets. Edges whose facet
//! edges lie on an attribute-20 (PEC) boundary facet are eliminated: they
//! never receive a free-DOF index and their rows/columns are never
//! assembled — there is no identity-row surgery.

use std::collections::HashMap;

use crate::abi::ATTR_CONDUCTOR_PEC;
use crate::basis::TET_EDGES;
use crate::error::{Error, Result};
use crate::mesh::Mesh;

/// Edge-based DOF layout for the assembled system.
#[derive(Debug, Clone)]
pub struct EdgeSystem {
    pub edge_count: usize,
    pub active_tet_count: usize,
    /// Active-tet ordinal → mesh tet index.
    pub active_tets: Vec<u32>,
    /// `[active_tet*6 + e]` → global edge id, `e` in [`TET_EDGES`] order.
    pub tet_edges: Vec<u32>,
    /// `+1` if the local edge direction agrees with the global low→high
    /// direction, else `−1`.
    pub tet_edge_sign: Vec<i8>,
    /// Global edge → free-DOF index, or `−1` when PEC-constrained.
    pub dof_of_edge: Vec<i32>,
    pub free_dof_count: usize,
    pub pec_edge_count: usize,
    pub conductor_tet_count: usize,
}

/// Build the edge system for `mesh`.
///
/// Errors when no active tets remain or when PEC elimination leaves no free
/// DOFs. Mirrors `buildEdgeSystem` in the TypeScript reference, including
/// per-facet PEC edge marking (a PEC facet marks its 3 edges; edges absent
/// from the active-tet map — e.g. on conductor tets — are skipped).
pub fn build_edge_system(mesh: &Mesh) -> Result<EdgeSystem> {
    let vertex_count = mesh.vertex_count();
    let tet_count = mesh.tet_count();
    if mesh.vertices.len() % 3 != 0 || mesh.tetrahedra.len() % 4 != 0 {
        return Err(Error::Mesh(
            "vertex/tet arrays have inconsistent lengths".to_string(),
        ));
    }

    let mut active_tets = Vec::new();
    let mut conductor_tet_count = 0usize;
    for t in 0..tet_count {
        if mesh.is_conductor_tet(t) {
            conductor_tet_count += 1;
        } else {
            active_tets.push(t as u32);
        }
    }
    if active_tets.is_empty() {
        return Err(Error::Mesh(
            "mesh has no active (non-conductor) tetrahedra".to_string(),
        ));
    }

    // Global edge key: u*nv + v with u < v (u64; exact for realistic meshes).
    let nv = vertex_count as u64;
    let key_of = |a: u32, b: u32| -> u64 {
        if a < b {
            a as u64 * nv + b as u64
        } else {
            b as u64 * nv + a as u64
        }
    };

    let active_tet_count = active_tets.len();
    let mut tet_edges = Vec::with_capacity(active_tet_count * 6);
    let mut tet_edge_sign = Vec::with_capacity(active_tet_count * 6);
    let mut edge_index: HashMap<u64, u32> = HashMap::new();
    let mut next_edge_id: u32 = 0;
    for &tet in &active_tets {
        let base = tet as usize * 4;
        for e in 0..6 {
            let [a, b] = TET_EDGES[e];
            let va = mesh.tetrahedra[base + a];
            let vb = mesh.tetrahedra[base + b];
            let id = *edge_index.entry(key_of(va, vb)).or_insert_with(|| {
                // First-seen order — deterministic.
                let id = next_edge_id;
                next_edge_id += 1;
                id
            });
            tet_edges.push(id);
            tet_edge_sign.push(if va < vb { 1 } else { -1 });
        }
    }
    let edge_count = edge_index.len();

    // PEC elimination: for each attribute-20 facet, mark its 3 facet edges.
    let mut pec_edge = vec![false; edge_count];
    for f in 0..mesh.facet_count() {
        if mesh.boundary_attributes[f] != ATTR_CONDUCTOR_PEC {
            continue;
        }
        let base = f * 3;
        let (u, v, w) = (
            mesh.boundary_facets[base],
            mesh.boundary_facets[base + 1],
            mesh.boundary_facets[base + 2],
        );
        for (a, b) in [(u, v), (v, w), (w, u)] {
            if let Some(&id) = edge_index.get(&key_of(a, b)) {
                pec_edge[id as usize] = true;
            }
        }
    }

    let mut dof_of_edge = vec![-1i32; edge_count];
    let mut free_dof_count = 0usize;
    let mut pec_edge_count = 0usize;
    for e in 0..edge_count {
        if pec_edge[e] {
            pec_edge_count += 1;
        } else {
            dof_of_edge[e] = free_dof_count as i32;
            free_dof_count += 1;
        }
    }
    if free_dof_count == 0 {
        return Err(Error::Mesh(
            "no free edge DOFs remain after PEC elimination".to_string(),
        ));
    }

    Ok(EdgeSystem {
        edge_count,
        active_tet_count,
        active_tets,
        tet_edges,
        tet_edge_sign,
        dof_of_edge,
        free_dof_count,
        pec_edge_count,
        conductor_tet_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tets sharing face (1,2,3): 12 local edges, 3 shared → 9 global.
    fn two_tet_mesh() -> Mesh {
        Mesh {
            vertices: vec![
                0., 0., 0., // 0
                1., 0., 0., // 1
                0., 1., 0., // 2
                0., 0., 1., // 3
                1., 1., 1., // 4
            ],
            tetrahedra: vec![0, 1, 2, 3, 1, 4, 2, 3],
            tet_regions: vec![1, 1],
            materials: vec![],
            boundary_facets: vec![0, 1, 2],
            boundary_attributes: vec![crate::abi::ATTR_EXTERIOR_ABSORBING],
            port_positive_facets: vec![0],
            port_negative_facets: vec![0],
        }
    }

    #[test]
    fn shared_face_edges_are_merged() {
        let sys = build_edge_system(&two_tet_mesh()).unwrap();
        assert_eq!(sys.active_tet_count, 2);
        assert_eq!(sys.edge_count, 9);
        assert_eq!(sys.free_dof_count, 9);
        // The shared face edges (1,2),(1,3),(2,3) are TET_EDGES[3,4,5] of tet 0
        // and TET_EDGES[1,2,5] of tet 1 ([1,4,2,3] local nodes).
        for k in 0..3 {
            assert_eq!(sys.tet_edges[3 + k], sys.tet_edges[6 + [1, 2, 5][k]]);
        }
    }

    #[test]
    fn pec_facet_marks_its_three_edges() {
        let mut mesh = two_tet_mesh();
        mesh.boundary_attributes[0] = ATTR_CONDUCTOR_PEC;
        let sys = build_edge_system(&mesh).unwrap();
        assert_eq!(sys.pec_edge_count, 3);
        assert_eq!(sys.free_dof_count, 6);
        assert!(sys.dof_of_edge.iter().filter(|&&d| d == -1).count() == 3);
        // Free DOFs are numbered densely in edge-id order.
        let mut seen: Vec<i32> = sys
            .dof_of_edge
            .iter()
            .copied()
            .filter(|&d| d >= 0)
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn conductor_tets_are_excluded() {
        let mut mesh = two_tet_mesh();
        mesh.materials = vec![crate::mesh::Material {
            attribute: 1,
            epsilon_r: 1.0,
            mu_r: 1.0,
            conductivity_s_m: 5.8e7,
            loss_tangent: 0.0,
        }];
        // Region 1 is now conducting: both tets excluded → error.
        assert!(build_edge_system(&mesh).is_err());
    }
}
