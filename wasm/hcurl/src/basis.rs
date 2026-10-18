//! First-order (Whitney-1) Nédélec edge-element basis.
//!
//! For local edge `e = (a, b)` with orientation sign `s_e` (±1, agreement of
//! the local direction with the global low→high direction) and physical
//! length `l_e`:
//!
//! ```text
//! N_e(p)   = s_e · l_e · ( λa(p)·∇λb − λb(p)·∇λa )
//! curl N_e = s_e · 2·l_e · ( ∇λa × ∇λb )          (constant over the tet)
//! ```
//!
//! Volume integrals are evaluated **analytically** from exact barycentric
//! moments — no quadrature is used for the volume terms:
//!
//! ```text
//! ∫ λp·λq dV = V/10  (p == q),  V/20  (p != q)
//! ∫ N_e dV   = s_e · l_e · (V/4) · (∇λb − ∇λa)
//! ```

use crate::geometry::TetGeometry;

/// Canonical local edge order; pairs always satisfy `a < b`.
pub const TET_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

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

/// Physical length of local edge `e` of mesh tet `t`.
pub fn edge_length(mesh: &crate::mesh::Mesh, tet: usize, e: usize) -> f64 {
    let [a, b] = TET_EDGES[e];
    let base = tet * 4;
    let pa = mesh.vertex(mesh.tetrahedra[base + a]);
    let pb = mesh.vertex(mesh.tetrahedra[base + b]);
    let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    dot(d, d).sqrt()
}

/// Unsigned curl vector `cv_e = 2·l_e·(∇λa × ∇λb)` (sign applied by caller).
pub fn curl_vector(geom: &TetGeometry, e: usize, length: f64) -> [f64; 3] {
    let [a, b] = TET_EDGES[e];
    let c = cross(geom.grad_lambda(a), geom.grad_lambda(b));
    [
        2.0 * length * c[0],
        2.0 * length * c[1],
        2.0 * length * c[2],
    ]
}

/// Local 6×6 element brackets (row-major), orientation signs applied.
///
/// - `curl[i][j] = s_i·s_j · V · (cv_i · cv_j)` — the caller scales by `1/μr`.
/// - `mass[i][j] = s_i·s_j · l_i·l_j · B̃_ij` with the analytic expansion of
///   `∫ (λai∇λbi − λbi∇λai)·(λaj∇λbj − λbj∇λaj) dV` — the caller scales by
///   `−k0²·εc` (complex).
pub fn local_matrices(
    geom: &TetGeometry,
    lengths: &[f64; 6],
    signs: &[i8; 6],
) -> ([f64; 36], [f64; 36]) {
    let v = geom.volume;
    let mut cv = [[0.0f64; 3]; 6];
    for e in 0..6 {
        cv[e] = curl_vector(geom, e, lengths[e]);
    }

    let mut curl = [0.0f64; 36];
    let mut mass = [0.0f64; 36];
    for i in 0..6 {
        let [ai, bi] = TET_EDGES[i];
        let si = signs[i] as f64;
        let gi_a = geom.grad_lambda(ai);
        let gi_b = geom.grad_lambda(bi);
        for j in 0..6 {
            let [aj, bj] = TET_EDGES[j];
            let sj = signs[j] as f64;
            let gj_a = geom.grad_lambda(aj);
            let gj_b = geom.grad_lambda(bj);

            curl[i * 6 + j] = si * sj * v * dot(cv[i], cv[j]);

            // Exact barycentric moment B(p,q): V/10 on the diagonal, V/20 off.
            // Pairing comes from expanding ∫(λai∇λbi − λbi∇λai)·(λaj∇λbj − λbj∇λaj):
            // (∇λbi·∇λbj) pairs with B(ai,aj), (∇λbi·∇λaj) with B(ai,bj), etc.
            let b = |p: usize, q: usize| {
                if p == q {
                    v / 10.0
                } else {
                    v / 20.0
                }
            };
            let bracket = dot(gi_b, gj_b) * b(ai, aj)
                - dot(gi_b, gj_a) * b(ai, bj)
                - dot(gi_a, gj_b) * b(bi, aj)
                + dot(gi_a, gj_a) * b(bi, bj);
            mass[i * 6 + j] = si * sj * lengths[i] * lengths[j] * bracket;
        }
    }
    (curl, mass)
}

/// Analytic integral `∫_tet N_e dV = s_e·l_e·(V/4)·(∇λb − ∇λa)`.
///
/// Used for the impressed-current RHS and the port-voltage path integral.
pub fn edge_field_integral(geom: &TetGeometry, e: usize, sign: i8, length: f64) -> [f64; 3] {
    let [a, b] = TET_EDGES[e];
    let ga = geom.grad_lambda(a);
    let gb = geom.grad_lambda(b);
    let f = sign as f64 * length * geom.volume / 4.0;
    [
        f * (gb[0] - ga[0]),
        f * (gb[1] - ga[1]),
        f * (gb[2] - ga[2]),
    ]
}

/// Evaluate the (signed) basis function `N_e` at barycentric coords `lambda`.
pub fn nedelec_at(
    geom: &TetGeometry,
    lambda: &[f64; 4],
    e: usize,
    sign: i8,
    length: f64,
) -> [f64; 3] {
    let [a, b] = TET_EDGES[e];
    let ga = geom.grad_lambda(a);
    let gb = geom.grad_lambda(b);
    let f = sign as f64 * length;
    [
        f * (lambda[a] * gb[0] - lambda[b] * ga[0]),
        f * (lambda[a] * gb[1] - lambda[b] * ga[1]),
        f * (lambda[a] * gb[2] - lambda[b] * ga[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::tet_geometry;
    use crate::mesh::Mesh;

    fn unit_tet() -> (Mesh, TetGeometry) {
        let mesh = Mesh {
            vertices: vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
            tetrahedra: vec![0, 1, 2, 3],
            tet_regions: vec![1],
            materials: vec![],
            boundary_facets: vec![],
            boundary_attributes: vec![],
            port_positive_facets: vec![],
            port_negative_facets: vec![],
            coax_port: None,
            exclude_boxes: vec![],
        };
        let geom = tet_geometry(&mesh, 0).unwrap();
        (mesh, geom)
    }

    #[test]
    fn curl_bracket_on_reference_tet() {
        // Edge (0,1): l = 1, cv = 2·((−1,−1,−1)×(1,0,0)) = (0,−2,2).
        // curl[0][0] = V·(cv·cv) = (1/6)·8 = 4/3.
        let (mesh, geom) = unit_tet();
        let lengths = [1.0, 1.0, 1.0, 2f64.sqrt(), 2f64.sqrt(), 2f64.sqrt()];
        let signs = [1i8; 6];
        let (curl, _) = local_matrices(&geom, &lengths, &signs);
        assert!(
            (curl[0] - 4.0 / 3.0).abs() < 1e-12,
            "curl[0][0]={}",
            curl[0]
        );
        // Symmetric.
        for i in 0..6 {
            for j in 0..6 {
                assert!((curl[i * 6 + j] - curl[j * 6 + i]).abs() < 1e-12);
            }
        }
        let _ = mesh;
    }

    #[test]
    fn mass_diagonal_on_reference_tet() {
        // mass[0][0] = l²·[(∇λ1·∇λ1)(V/10) − 2(∇λ1·∇λ0)(V/20) + (∇λ0·∇λ0)(V/10)]
        //           = V·(1/10 + 2/20 + 3/10) = V/2 = 1/12.
        let (_mesh, geom) = unit_tet();
        let lengths = [1.0, 1.0, 1.0, 2f64.sqrt(), 2f64.sqrt(), 2f64.sqrt()];
        let signs = [1i8; 6];
        let (_, mass) = local_matrices(&geom, &lengths, &signs);
        assert!(
            (mass[0] - 1.0 / 12.0).abs() < 1e-12,
            "mass[0][0]={}",
            mass[0]
        );
        for i in 0..6 {
            for j in 0..6 {
                assert!((mass[i * 6 + j] - mass[j * 6 + i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn mass_off_diagonal_pins_moment_pairing() {
        // i=0 (edge (0,1)), j=1 (edge (0,2)); l=1 both.
        // bracket = 0·B(0,0) −(−1)·B(0,2) −(−1)·B(1,0) + 3·B(1,2)
        //         = V/20 + V/20 + 3V/20 = V/4 = 1/24.
        // (A swapped B-pairing, e.g. B(bi,bj) on the first term, gives 1/15.)
        let (_mesh, geom) = unit_tet();
        let lengths = [1.0; 6];
        let signs = [1i8; 6];
        let (_, mass) = local_matrices(&geom, &lengths, &signs);
        assert!(
            (mass[1] - 1.0 / 24.0).abs() < 1e-12,
            "mass[0][1]={}",
            mass[1]
        );
    }

    #[test]
    fn orientation_sign_flips_bracket_signs() {
        let (_mesh, geom) = unit_tet();
        let lengths = [1.0; 6];
        let (c_pos, m_pos) = local_matrices(&geom, &lengths, &[1i8; 6]);
        let mut signs = [1i8; 6];
        signs[2] = -1;
        let (c_neg, m_neg) = local_matrices(&geom, &lengths, &signs);
        // Row/col 2 flip sign against every other row/col; diagonal unchanged.
        for j in 0..6 {
            let expected = if j == 2 { 1.0 } else { -1.0 };
            assert!((c_neg[2 * 6 + j] - expected * c_pos[2 * 6 + j]).abs() < 1e-12);
            assert!((m_neg[2 * 6 + j] - expected * m_pos[2 * 6 + j]).abs() < 1e-12);
        }
        assert!((c_neg[2 * 6 + 2] - c_pos[2 * 6 + 2]).abs() < 1e-12);
    }

    #[test]
    fn field_integral_matches_quadrature_free_identity() {
        // ∫N_e dV = s·l·(V/4)·(∇λb − ∇λa); edge (0,1): (0,−2,2)·(1/24)·…:
        // s·l·V/4 = 1·1·(1/6)/4 = 1/24; ∇λ1−∇λ0 = (2,1,1); → (1/12, 1/24, 1/24).
        let (_mesh, geom) = unit_tet();
        let iv = edge_field_integral(&geom, 0, 1, 1.0);
        assert!((iv[0] - 1.0 / 12.0).abs() < 1e-12, "{iv:?}");
        assert!((iv[1] - 1.0 / 24.0).abs() < 1e-12, "{iv:?}");
        assert!((iv[2] - 1.0 / 24.0).abs() < 1e-12, "{iv:?}");
    }
}
