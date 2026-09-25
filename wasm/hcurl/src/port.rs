//! Lumped-port feed: gap selection, 1 A impressed-current RHS, port voltage, S11.
//!
//! The packed ABI carries port **facet index sets**, not terminal
//! coordinates. Terminals are derived as the mean of all vertices of all
//! facets in each set (matching the prototype native solver; validated to
//! give exactly (∓0.001, 0, 0) on the reference dipole).
//!
//! The impressed current density is normalized for **exactly 1 A**:
//! `J = (L_gap/V_gap)·d̂` (prismatic-gap approximation, `A_cross = V_gap/L_gap`).
//! The missing-`L_gap` normalization inflated |Zin| ~500× in an earlier
//! revision — the `L_gap` factor is load-bearing and pinned by tests.

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::basis::{edge_field_integral, edge_length, nedelec_at};
use crate::constants::MU0;
use crate::edge::EdgeSystem;
use crate::error::{Error, Result};
use crate::geometry::{tet_geometry, TetGeometry};
use crate::mesh::Mesh;

/// Port terminals `(positive, negative)` from the port facet sets.
pub fn terminals_from_facets(mesh: &Mesh) -> Result<([f64; 3], [f64; 3])> {
    let centroid_of = |facets: &[u32], label: &str| -> Result<[f64; 3]> {
        let mut sum = [0.0f64; 3];
        let mut count = 0u64;
        for &f in facets {
            let base = f as usize * 3;
            for k in 0..3 {
                let p = mesh.vertex(mesh.boundary_facets[base + k]);
                sum[0] += p[0];
                sum[1] += p[1];
                sum[2] += p[2];
                count += 1;
            }
        }
        if count == 0 {
            return Err(Error::Mesh(format!("port {label} facet set is empty")));
        }
        let n = count as f64;
        Ok([sum[0] / n, sum[1] / n, sum[2] / n])
    };
    Ok((
        centroid_of(&mesh.port_positive_facets, "positive")?,
        centroid_of(&mesh.port_negative_facets, "negative")?,
    ))
}

/// Active tets carrying the impressed current around the terminal gap.
#[derive(Debug, Clone)]
pub struct GapFeed {
    /// Active-tet ordinals (indices into [`EdgeSystem::active_tets`]).
    pub gap_tets: Vec<usize>,
    pub gap_volume: f64,
    /// Unit vector from the negative to the positive terminal.
    pub direction: [f64; 3],
    pub gap_length: f64,
}

fn point_segment_distance_sq(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let mut t = 0.0;
    if len2 > 0.0 {
        t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / len2;
        t = t.clamp(0.0, 1.0);
    }
    let dx = p[0] - (a[0] + t * ab[0]);
    let dy = p[1] - (a[1] + t * ab[1]);
    let dz = p[2] - (a[2] + t * ab[2]);
    dx * dx + dy * dy + dz * dz
}

/// Select the active tets around the terminal gap.
///
/// Starts with radius `0.6·gap_length` around the terminal segment, doubling
/// up to 6 attempts; falls back to the active tet nearest the segment
/// midpoint.
pub fn select_gap_feed(
    mesh: &Mesh,
    sys: &EdgeSystem,
    positive: [f64; 3],
    negative: [f64; 3],
) -> Result<GapFeed> {
    let gv = [
        positive[0] - negative[0],
        positive[1] - negative[1],
        positive[2] - negative[2],
    ];
    let gap_length = (gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2]).sqrt();
    if !gap_length.is_finite() || gap_length <= 0.0 {
        return Err(Error::Mesh(
            "lumped-port terminals must be distinct points".to_string(),
        ));
    }
    let direction = [gv[0] / gap_length, gv[1] / gap_length, gv[2] / gap_length];

    let mut centroids = Vec::with_capacity(sys.active_tet_count);
    for &tet in &sys.active_tets {
        centroids.push(tet_geometry(mesh, tet as usize)?.centroid);
    }

    let mut selected: Vec<usize> = Vec::new();
    let mut radius = 0.6 * gap_length;
    for _ in 0..6 {
        if !selected.is_empty() {
            break;
        }
        selected.clear();
        let r2 = radius * radius;
        for (ord, c) in centroids.iter().enumerate() {
            if point_segment_distance_sq(*c, negative, positive) <= r2 {
                selected.push(ord);
            }
        }
        radius *= 2.0;
    }
    if selected.is_empty() {
        let mid = [
            negative[0] + gv[0] / 2.0,
            negative[1] + gv[1] / 2.0,
            negative[2] + gv[2] / 2.0,
        ];
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (ord, c) in centroids.iter().enumerate() {
            let d = (c[0] - mid[0]).powi(2) + (c[1] - mid[1]).powi(2) + (c[2] - mid[2]).powi(2);
            if d < best_d {
                best_d = d;
                best = ord;
            }
        }
        selected.push(best);
    }

    let mut gap_volume = 0.0;
    for &ord in &selected {
        gap_volume += tet_geometry(mesh, sys.active_tets[ord] as usize)?.volume;
    }
    if !(gap_volume > 0.0) {
        return Err(Error::Mesh("port feed region has no volume".to_string()));
    }
    Ok(GapFeed {
        gap_tets: selected,
        gap_volume,
        direction,
        gap_length,
    })
}

/// Right-hand side for the 1 A impressed current.
///
/// `b_i = −j·ω·μ0·L_gap·(d̂·∫N_i dV)/V_gap` — purely imaginary by
/// construction.
pub fn build_port_rhs(
    mesh: &Mesh,
    sys: &EdgeSystem,
    feed: &GapFeed,
    frequency_hz: f64,
) -> Result<Vec<Complex64>> {
    let omega = 2.0 * PI * frequency_hz;
    let coef = (-omega * MU0) * feed.gap_length / feed.gap_volume;
    let mut b = vec![Complex64::new(0.0, 0.0); sys.free_dof_count];
    for &ord in &feed.gap_tets {
        let tet = sys.active_tets[ord] as usize;
        let geom = tet_geometry(mesh, tet)?;
        for e in 0..6 {
            let d = sys.dof_of_edge[sys.tet_edges[ord * 6 + e] as usize];
            if d < 0 {
                continue;
            }
            let iv = edge_field_integral(&geom, e, sys.tet_edge_sign[ord * 6 + e], edge_length(mesh, tet, e));
            let proj = feed.direction[0] * iv[0] + feed.direction[1] * iv[1] + feed.direction[2] * iv[2];
            b[d as usize].im += coef * proj;
        }
    }
    Ok(b)
}

/// Port voltage from the line integral `V = −∫_{neg}^{pos} E·dl`,
/// sampled at 8 midpoints. Since the impressed current is exactly 1 A,
/// `Zin = V`.
pub fn port_voltage(
    mesh: &Mesh,
    sys: &EdgeSystem,
    feed: &GapFeed,
    negative: [f64; 3],
    x: &[Complex64],
) -> Result<Complex64> {
    // Gap-tet shapes for point location.
    let mut shapes: Vec<(usize, TetGeometry)> = Vec::with_capacity(feed.gap_tets.len());
    for &ord in &feed.gap_tets {
        let tet = sys.active_tets[ord] as usize;
        shapes.push((ord, tet_geometry(mesh, tet)?));
    }
    let locate = |p: [f64; 3]| -> usize {
        for (k, (_, g)) in shapes.iter().enumerate() {
            let inside = (0..4).all(|m| {
                let lam = g.lambda(m, p);
                lam >= -1e-9 && lam <= 1.0 + 1e-9
            });
            if inside {
                return k;
            }
        }
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (k, (_, g)) in shapes.iter().enumerate() {
            let c = g.centroid;
            let d = (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2) + (c[2] - p[2]).powi(2);
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        best
    };

    let gv = [
        feed.gap_length * feed.direction[0],
        feed.gap_length * feed.direction[1],
        feed.gap_length * feed.direction[2],
    ];
    let mut v = Complex64::new(0.0, 0.0);
    let samples = 8;
    for k in 0..samples {
        let t = (k as f64 + 0.5) / samples as f64;
        let p = [
            negative[0] + t * gv[0],
            negative[1] + t * gv[1],
            negative[2] + t * gv[2],
        ];
        let gi = locate(p);
        let (ord, geom) = &shapes[gi];
        let ord = *ord;
        let lambda = [geom.lambda(0, p), geom.lambda(1, p), geom.lambda(2, p), geom.lambda(3, p)];
        let tet = sys.active_tets[ord] as usize;
        let mut e_vec = [Complex64::new(0.0, 0.0); 3];
        for edge in 0..6 {
            let d = sys.dof_of_edge[sys.tet_edges[ord * 6 + edge] as usize];
            if d < 0 {
                continue;
            }
            let nvec = nedelec_at(
                geom,
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
        let proj = e_vec[0] * feed.direction[0] + e_vec[1] * feed.direction[1] + e_vec[2] * feed.direction[2];
        v -= proj * (feed.gap_length / samples as f64);
    }
    Ok(v)
}

/// Reflection coefficient `S11 = (Z − Z0)/(Z + Z0)`.
pub fn reflection_coefficient(zin: Complex64, z0: f64) -> Complex64 {
    let den = (zin.re + z0).powi(2) + zin.im.powi(2);
    Complex64::new(
        ((zin.re - z0) * (zin.re + z0) + zin.im * zin.im) / den,
        (2.0 * zin.im * z0) / den,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::build_edge_system;
    use crate::mesh::Material;

    fn unit_tet_mesh() -> Mesh {
        Mesh {
            vertices: vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
            tetrahedra: vec![0, 1, 2, 3],
            tet_regions: vec![1],
            materials: vec![Material::air()],
            boundary_facets: vec![0, 1, 2, 1, 2, 3],
            boundary_attributes: vec![10, 10],
            port_positive_facets: vec![0],
            port_negative_facets: vec![1],
        }
    }

    #[test]
    fn terminals_are_facet_vertex_means() {
        let mesh = unit_tet_mesh();
        let (pos, neg) = terminals_from_facets(&mesh).unwrap();
        // Facet 0 = (0,1,2): mean ((0+1+0)/3, (0+0+1)/3, 0).
        assert!((pos[0] - 1.0 / 3.0).abs() < 1e-15);
        assert!((pos[1] - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(pos[2], 0.0);
        // Facet 1 = (1,2,3): mean ((1+0+0)/3, (0+1+0)/3, (0+0+1)/3).
        assert!((neg[0] - 1.0 / 3.0).abs() < 1e-15);
        assert!((neg[1] - 1.0 / 3.0).abs() < 1e-15);
        assert!((neg[2] - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn gap_feed_selects_nearby_tet() {
        let mesh = unit_tet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let feed = select_gap_feed(&mesh, &sys, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
        assert_eq!(feed.gap_tets, vec![0]);
        assert!((feed.gap_volume - 1.0 / 6.0).abs() < 1e-15);
        assert!((feed.gap_length - 1.0).abs() < 1e-15);
        assert_eq!(feed.direction, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn rhs_is_purely_imaginary() {
        let mesh = unit_tet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let feed = select_gap_feed(&mesh, &sys, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
        let b = build_port_rhs(&mesh, &sys, &feed, 3.0e9).unwrap();
        assert_eq!(b.len(), 6);
        assert!(b.iter().all(|z| z.re == 0.0));
        assert!(b.iter().any(|z| z.im != 0.0));
        // The L_gap/V_gap normalization: b_im[0] = coef·(d̂·∫N_0 dV),
        // ∫N_0 dV = (1/12, 1/24, 1/24) (pinned in basis tests), d̂=(1,0,0).
        let coef = (-2.0 * PI * 3.0e9 * MU0) * 1.0 / (1.0 / 6.0);
        assert!((b[0].im - coef / 12.0).abs() < 1e-9 * coef.abs());
    }

    #[test]
    fn port_voltage_of_single_edge_field() {
        // x_0 = 1 on edge (0,1): E(t,0,0) = (1−t)∇λ1 − t∇λ0 = (1, t, t),
        // so E·x̂ = 1 along the segment (0,0,0)→(1,0,0) and V = −1 exactly
        // (the 8-point midpoint rule integrates the constant exactly).
        let mesh = unit_tet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let feed = GapFeed {
            gap_tets: vec![0],
            gap_volume: 1.0 / 6.0,
            direction: [1.0, 0.0, 0.0],
            gap_length: 1.0,
        };
        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let v = port_voltage(&mesh, &sys, &feed, [0.0, 0.0, 0.0], &x).unwrap();
        assert!((v.re + 1.0).abs() < 1e-12, "v={v}");
        assert!(v.im.abs() < 1e-12);
    }

    #[test]
    fn s11_of_reference_dipole_impedance() {
        let zin = Complex64::new(66.637496, -15.317936);
        let s = reflection_coefficient(zin, 50.0);
        assert!((s.re - 0.157179).abs() < 1e-6, "s={s}");
        assert!((s.im + 0.110687).abs() < 1e-6, "s={s}");
    }

    #[test]
    fn s11_of_matched_load_is_zero() {
        let s = reflection_coefficient(Complex64::new(50.0, 0.0), 50.0);
        assert!(s.norm() < 1e-15);
    }
}
