//! Tetrahedral element geometry.
//!
//! For tet `t` with nodes `v0..v3`, the Jacobian `J = [c0 c1 c2]` has columns
//! `ci = v_{i+1} − v0`. Its inverse rows are the barycentric gradients:
//!
//! ```text
//! ∇λ₁ = (c1×c2)/det,  ∇λ₂ = (c2×c0)/det,  ∇λ₃ = (c0×c1)/det,
//! ∇λ₀ = −(∇λ₁+∇λ₂+∇λ₃),   det = c0·(c1×c2)
//! ```
//!
//! Barycentric coordinates evaluate as `λm(p) = 1/4 + ∇λm·(p − centroid)`.

use crate::error::{Error, Result};
use crate::mesh::Mesh;

/// Degenerate-tet threshold on `|det J|`, mirroring the TypeScript reference.
pub const DEGENERATE_DET_THRESHOLD: f64 = 1e-30;

/// Geometry of one tetrahedron.
#[derive(Debug, Clone)]
pub struct TetGeometry {
    pub volume: f64,
    /// Barycentric gradients, `grad[3*m + k]` for node `m`, component `k`.
    pub grad: [f64; 12],
    pub centroid: [f64; 3],
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the geometry of mesh tet `t`. Errors on degenerate tets.
pub fn tet_geometry(mesh: &Mesh, t: usize) -> Result<TetGeometry> {
    let base = t * 4;
    let v = [
        mesh.vertex(mesh.tetrahedra[base]),
        mesh.vertex(mesh.tetrahedra[base + 1]),
        mesh.vertex(mesh.tetrahedra[base + 2]),
        mesh.vertex(mesh.tetrahedra[base + 3]),
    ];
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let c0 = sub(v[1], v[0]);
    let c1 = sub(v[2], v[0]);
    let c2 = sub(v[3], v[0]);

    let r0 = cross(c1, c2);
    let r1 = cross(c2, c0);
    let r2 = cross(c0, c1);
    let det = dot(c0, r0);
    if det.abs() < DEGENERATE_DET_THRESHOLD {
        return Err(Error::Mesh(format!(
            "degenerate tetrahedron {t}: |det J| = {det:e}"
        )));
    }

    let mut grad = [0.0f64; 12];
    for (m, r) in [r0, r1, r2].iter().enumerate() {
        // Nodes 1..3; node 0 is filled below as the negative sum.
        grad[(m + 1) * 3] = r[0] / det;
        grad[(m + 1) * 3 + 1] = r[1] / det;
        grad[(m + 1) * 3 + 2] = r[2] / det;
    }
    for k in 0..3 {
        grad[k] = -(grad[3 + k] + grad[6 + k] + grad[9 + k]);
    }

    let centroid = [
        (v[0][0] + v[1][0] + v[2][0] + v[3][0]) / 4.0,
        (v[0][1] + v[1][1] + v[2][1] + v[3][1]) / 4.0,
        (v[0][2] + v[1][2] + v[2][2] + v[3][2]) / 4.0,
    ];

    Ok(TetGeometry {
        volume: det.abs() / 6.0,
        grad,
        centroid,
    })
}

impl TetGeometry {
    /// Barycentric gradient of node `m` (0–3).
    pub fn grad_lambda(&self, m: usize) -> [f64; 3] {
        let b = m * 3;
        [self.grad[b], self.grad[b + 1], self.grad[b + 2]]
    }

    /// Barycentric coordinate of node `m` at point `p`.
    pub fn lambda(&self, m: usize, p: [f64; 3]) -> f64 {
        let g = self.grad_lambda(m);
        let d = [
            p[0] - self.centroid[0],
            p[1] - self.centroid[1],
            p[2] - self.centroid[2],
        ];
        0.25 + dot(g, d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn unit_tet_mesh() -> Mesh {
        // Nodes (0,0,0), (1,0,0), (0,1,0), (0,0,1).
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

    use crate::mesh::Material;

    #[test]
    fn reference_tet_geometry() {
        let mesh = unit_tet_mesh();
        let g = tet_geometry(&mesh, 0).unwrap();
        assert!((g.volume - 1.0 / 6.0).abs() < 1e-15);
        assert_eq!(g.grad_lambda(1), [1.0, 0.0, 0.0]);
        assert_eq!(g.grad_lambda(2), [0.0, 1.0, 0.0]);
        assert_eq!(g.grad_lambda(3), [0.0, 0.0, 1.0]);
        assert_eq!(g.grad_lambda(0), [-1.0, -1.0, -1.0]);
        // Barycentric coordinates recover the nodes.
        for m in 0..4 {
            let p = mesh.vertex(m as u32);
            for n in 0..4 {
                let expected = if m == n { 1.0 } else { 0.0 };
                assert!((g.lambda(n, p) - expected).abs() < 1e-12, "m={m} n={n}");
            }
        }
        // Partition of unity at an interior point.
        let p = [0.1, 0.2, 0.15];
        let sum: f64 = (0..4).map(|n| g.lambda(n, p)).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn degenerate_tet_is_rejected() {
        let mut mesh = unit_tet_mesh();
        mesh.vertices[3] = 0.0; // collapse node 1 onto node 0
        mesh.vertices[4] = 0.0;
        mesh.vertices[5] = 0.0;
        assert!(tet_geometry(&mesh, 0).is_err());
    }
}
