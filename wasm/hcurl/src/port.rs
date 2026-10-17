//! Lumped-port feed: gap selection, 1 A impressed-current RHS, port voltage, S11.
//!
//! The packed ABI carries port **facet index sets**, not terminal
//! coordinates. Terminals are derived as the mean of all vertices of all
//! facets in each set (matching the prototype native solver; validated to
//! give exactly (∓0.001, 0, 0) on the reference dipole).
//!
//! Gap tets are selected by **prism intersection**: the port footprint —
//! the convex hull of the port facet sets' vertices projected perpendicular
//! to the terminal direction — is swept between the two terminal planes,
//! and every active tet intersecting that prism carries the feed, weighted
//! by its intersection-volume fraction. The effective gap volume is then
//! the prism volume on every mesh, so `A_cross ≡ footprint area` by
//! construction and the source definition no longer depends on how tet
//! centroids happen to align with the port axis (the previous
//! centroid-radius selection silently flipped between the full prism and
//! axial slivers under refinement — see
//! `hidden_files/fem-port-audit-2026-10-17.md`). Degenerate facet data or
//! an empty intersection falls back to the legacy centroid-radius rule.
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
    /// Per-gap-tet weight: the fraction of the tet's volume inside the
    /// port prism (1.0 for fully covered tets and for the legacy fallback).
    pub weights: Vec<f64>,
    /// Effective feed volume: `Σ_tet vol(tet ∩ prism)` on the prism path
    /// (≈ the prism volume), summed selected-tet volume on the fallback.
    pub gap_volume: f64,
    /// Unit vector from the negative to the positive terminal.
    pub direction: [f64; 3],
    pub gap_length: f64,
    /// Area of the port footprint (convex hull of the port facet vertices
    /// projected ⊥ direction). Equals `gap_volume / gap_length` on the
    /// legacy fallback path.
    pub footprint_area: f64,
    /// Effective cross-section `gap_volume / gap_length`; identical to
    /// `footprint_area` (up to mesh coverage of the prism) on the prism
    /// path by construction.
    pub cross_area: f64,
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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

/// Orthonormal basis `(û, v̂)` spanning the plane ⊥ `dir`, with
/// `û × v̂ = dir̂` (so `v̂ = dir̂ × û`). `dir` need not be unit.
fn plane_basis(dir: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let dl = dot3(dir, dir).sqrt();
    let dir = [dir[0] / dl, dir[1] / dl, dir[2] / dl];
    let axis = (0..3)
        .min_by(|&a, &b| dir[a].abs().partial_cmp(&dir[b].abs()).unwrap())
        .unwrap();
    let mut u = [0.0; 3];
    u[axis] = 1.0;
    let d = dot3(u, dir);
    u = [u[0] - d * dir[0], u[1] - d * dir[1], u[2] - d * dir[2]];
    let n = dot3(u, u).sqrt();
    u = [u[0] / n, u[1] / n, u[2] / n];
    (u, cross3(dir, u))
}

/// Monotone-chain convex hull of 2D points; returns the hull vertices in
/// counter-clockwise order (no repeated endpoint) and the signed area
/// (positive for a CCW hull).
fn convex_hull_2d(points: &[[f64; 2]]) -> (Vec<[f64; 2]>, f64) {
    let mut pts: Vec<[f64; 2]> = points.to_vec();
    pts.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });
    pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-15 && (a[1] - b[1]).abs() < 1e-15);
    if pts.len() < 3 {
        return (Vec::new(), 0.0);
    }
    let cross2 = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut lower: Vec<[f64; 2]> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross2(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[f64; 2]> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross2(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    let hull: Vec<[f64; 2]> = lower.iter().chain(upper.iter()).copied().collect();
    let mut area2 = 0.0;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        area2 += a[0] * b[1] - b[0] * a[1];
    }
    (hull, area2 / 2.0)
}

/// The port prism: footprint hull swept between the terminal planes,
/// expressed as half-spaces `n·p ≤ c` plus an axis-aligned bounding box
/// for candidate rejection.
struct PortPrism {
    planes: Vec<([f64; 3], f64)>,
    aabb_lo: [f64; 3],
    aabb_hi: [f64; 3],
    footprint_area: f64,
}

impl PortPrism {
    /// Build the prism from the port facet sets. Returns `None` when the
    /// facet data is missing or its projection is degenerate.
    fn from_facets(
        mesh: &Mesh,
        positive: [f64; 3],
        negative: [f64; 3],
        direction: [f64; 3],
        gap_length: f64,
    ) -> Option<PortPrism> {
        if mesh.port_positive_facets.is_empty() || mesh.port_negative_facets.is_empty() {
            return None;
        }
        let (u, v) = plane_basis(direction);
        let mut pts2: Vec<[f64; 2]> = Vec::new();
        for facets in [&mesh.port_positive_facets, &mesh.port_negative_facets] {
            for &f in facets.iter() {
                let base = f as usize * 3;
                for k in 0..3 {
                    let p = mesh.vertex(mesh.boundary_facets[base + k]);
                    let rel = [p[0] - negative[0], p[1] - negative[1], p[2] - negative[2]];
                    pts2.push([dot3(rel, u), dot3(rel, v)]);
                }
            }
        }
        let (mut hull, area) = convex_hull_2d(&pts2);
        if hull.len() < 3 || !(area.abs() > 0.0) {
            return None;
        }
        if area < 0.0 {
            hull.reverse();
        }
        let footprint_area = area.abs();

        let mut planes: Vec<([f64; 3], f64)> = Vec::with_capacity(hull.len() + 2);
        // Axial half-spaces: 0 ≤ (p − neg)·dir ≤ gap_length.
        planes.push((
            [-direction[0], -direction[1], -direction[2]],
            -dot3(negative, direction),
        ));
        planes.push((direction, dot3(positive, direction)));
        // Lateral half-spaces, one per hull edge (CCW ⇒ interior is left of
        // A→B, i.e. (dir × (B−A))·(p − A) ≥ 0).
        let at = |q: [f64; 2]| {
            [
                negative[0] + q[0] * u[0] + q[1] * v[0],
                negative[1] + q[0] * u[1] + q[1] * v[1],
                negative[2] + q[0] * u[2] + q[1] * v[2],
            ]
        };
        for i in 0..hull.len() {
            let a3 = at(hull[i]);
            let b3 = at(hull[(i + 1) % hull.len()]);
            let e = [b3[0] - a3[0], b3[1] - a3[1], b3[2] - a3[2]];
            let n = cross3(direction, e);
            let nn = dot3(n, n).sqrt();
            let n = [n[0] / nn, n[1] / nn, n[2] / nn];
            planes.push(([-n[0], -n[1], -n[2]], -dot3(n, a3)));
        }

        let mut aabb_lo = [f64::INFINITY; 3];
        let mut aabb_hi = [f64::NEG_INFINITY; 3];
        for &q in &hull {
            for t in [0.0, gap_length] {
                let p = [
                    negative[0] + q[0] * u[0] + q[1] * v[0] + t * direction[0],
                    negative[1] + q[0] * u[1] + q[1] * v[1] + t * direction[1],
                    negative[2] + q[0] * u[2] + q[1] * v[2] + t * direction[2],
                ];
                for c in 0..3 {
                    aabb_lo[c] = aabb_lo[c].min(p[c]);
                    aabb_hi[c] = aabb_hi[c].max(p[c]);
                }
            }
        }
        Some(PortPrism {
            planes,
            aabb_lo,
            aabb_hi,
            footprint_area,
        })
    }

    /// Volume of `tet ∩ prism` for a tetrahedron given by its 4 vertices.
    fn intersection_volume(&self, verts: [[f64; 3]; 4]) -> f64 {
        // Initial polyhedron: the tet's 4 faces, each oriented outward
        // (normal pointing away from the opposite vertex).
        let mut faces: Vec<Vec<[f64; 3]>> = Vec::with_capacity(4);
        for (a, b, c, o) in [
            (0usize, 1usize, 2usize, 3usize),
            (0, 3, 1, 2),
            (0, 2, 3, 1),
            (1, 3, 2, 0),
        ] {
            let (pa, pb, pc, po) = (verts[a], verts[b], verts[c], verts[o]);
            let n = cross3(
                [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]],
                [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]],
            );
            let away = dot3(
                n,
                [po[0] - pa[0], po[1] - pa[1], po[2] - pa[2]],
            );
            if away > 0.0 {
                faces.push(vec![pa, pc, pb]);
            } else {
                faces.push(vec![pa, pb, pc]);
            }
        }
        for &(n, c) in &self.planes {
            faces = clip_polyhedron(faces, n, c);
            if faces.is_empty() {
                return 0.0;
            }
        }
        let mut vol6 = 0.0;
        for face in &faces {
            for i in 1..face.len().saturating_sub(1) {
                vol6 += dot3(face[0], cross3(face[i], face[i + 1]));
            }
        }
        (vol6 / 6.0).abs()
    }
}

/// Clip a convex polyhedron (faces as outward-oriented point lists) against
/// the half-space `n·p ≤ c`, adding the cap face on the clipping plane.
fn clip_polyhedron(faces: Vec<Vec<[f64; 3]>>, n: [f64; 3], c: f64) -> Vec<Vec<[f64; 3]>> {
    const TOL: f64 = 1e-12;
    let dist = |p: [f64; 3]| dot3(n, p) - c;
    let mut out: Vec<Vec<[f64; 3]>> = Vec::with_capacity(faces.len() + 1);
    let mut cap: Vec<[f64; 3]> = Vec::new();
    let mut any_outside = false;
    for face in faces {
        let m = face.len();
        if m == 0 {
            continue;
        }
        let mut poly: Vec<[f64; 3]> = Vec::with_capacity(m + 1);
        for i in 0..m {
            let cur = face[i];
            let nxt = face[(i + 1) % m];
            let dc = dist(cur);
            let dn = dist(nxt);
            if dc > TOL {
                any_outside = true;
            }
            if dc <= TOL {
                poly.push(cur);
                if dc.abs() <= TOL {
                    cap.push(cur);
                }
            }
            if (dc < -TOL && dn > TOL) || (dc > TOL && dn < -TOL) {
                let t = dc / (dc - dn);
                let p = [
                    cur[0] + t * (nxt[0] - cur[0]),
                    cur[1] + t * (nxt[1] - cur[1]),
                    cur[2] + t * (nxt[2] - cur[2]),
                ];
                poly.push(p);
                cap.push(p);
            }
        }
        if poly.len() >= 3 {
            out.push(poly);
        }
    }
    if out.is_empty() || !any_outside {
        // Nothing was cut away: no new cross-section face exists (a plane
        // merely touching an existing face must not duplicate it).
        return out;
    }
    // Cap: the cross-section polygon on the plane, ordered around its
    // centroid and oriented with outward normal +n.
    let mut uniq: Vec<[f64; 3]> = Vec::new();
    for p in cap {
        if !uniq
            .iter()
            .any(|q| (q[0] - p[0]).abs() < 1e-12 && (q[1] - p[1]).abs() < 1e-12 && (q[2] - p[2]).abs() < 1e-12)
        {
            uniq.push(p);
        }
    }
    if uniq.len() >= 3 {
        let mut cen = [0.0; 3];
        for p in &uniq {
            cen = [cen[0] + p[0], cen[1] + p[1], cen[2] + p[2]];
        }
        let m = uniq.len() as f64;
        cen = [cen[0] / m, cen[1] / m, cen[2] / m];
        let (e1, e2) = plane_basis(n);
        let angle = |p: [f64; 3]| {
            let d = [p[0] - cen[0], p[1] - cen[1], p[2] - cen[2]];
            dot3(d, e2).atan2(dot3(d, e1))
        };
        uniq.sort_by(|a, b| angle(*a).partial_cmp(&angle(*b)).unwrap());
        // Orient the cap so its Newell normal is +n (outward from the
        // kept half-space n·p ≤ c).
        let mut nn = [0.0; 3];
        for i in 0..uniq.len() {
            let a = uniq[i];
            let b = uniq[(i + 1) % uniq.len()];
            nn[0] += (a[1] - b[1]) * (a[2] + b[2]);
            nn[1] += (a[2] - b[2]) * (a[0] + b[0]);
            nn[2] += (a[0] - b[0]) * (a[1] + b[1]);
        }
        if dot3(nn, n) < 0.0 {
            uniq.reverse();
        }
        out.push(uniq);
    }
    out
}

/// Solve-time sanity log for the port cross-section (native builds only;
/// the WASM build keeps the fields on [`GapFeed`] instead of printing).
fn log_cross_section(feed: &GapFeed) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let a = feed.cross_area * 1e6;
        let f = feed.footprint_area * 1e6;
        let dev = if feed.footprint_area > 0.0 {
            (feed.cross_area - feed.footprint_area).abs() / feed.footprint_area
        } else {
            0.0
        };
        eprintln!(
            "gap feed: {} tets, A_cross = {a:.4} mm^2, footprint = {f:.4} mm^2, deviation = {:.1}%",
            feed.gap_tets.len(),
            dev * 100.0
        );
        if dev > 0.10 {
            eprintln!(
                "gap feed WARNING: A_cross deviates from the port footprint by {:.1}% (>10%)",
                dev * 100.0
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = feed;
    }
}

/// Select the active tets carrying the port feed.
///
/// Primary rule: prism intersection — every active tet intersecting the
/// swept port footprint contributes, weighted by its intersection-volume
/// fraction, so the effective gap volume is the prism volume on any mesh.
/// Falls back to the legacy centroid-radius rule (radius `0.6·gap_length`
/// around the terminal segment, doubling up to 6 attempts, then the
/// active tet nearest the segment midpoint) when the facet data cannot
/// define a prism or nothing intersects it.
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

    if let Some(prism) = PortPrism::from_facets(mesh, positive, negative, direction, gap_length) {
        let mut gap_tets: Vec<usize> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut gap_volume = 0.0f64;
        for (ord, &tet) in sys.active_tets.iter().enumerate() {
            let tet = tet as usize;
            let base = tet * 4;
            let verts = [
                mesh.vertex(mesh.tetrahedra[base]),
                mesh.vertex(mesh.tetrahedra[base + 1]),
                mesh.vertex(mesh.tetrahedra[base + 2]),
                mesh.vertex(mesh.tetrahedra[base + 3]),
            ];
            // Cheap AABB rejection against the prism bounds.
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for p in &verts {
                for c in 0..3 {
                    lo[c] = lo[c].min(p[c]);
                    hi[c] = hi[c].max(p[c]);
                }
            }
            if (0..3).any(|c| hi[c] < prism.aabb_lo[c] - 1e-12 || lo[c] > prism.aabb_hi[c] + 1e-12)
            {
                continue;
            }
            let v_int = prism.intersection_volume(verts);
            if v_int <= 1e-15 {
                continue;
            }
            let v_tet = tet_geometry(mesh, tet)?.volume;
            let w = (v_int / v_tet).clamp(0.0, 1.0);
            if w <= 1e-9 {
                continue;
            }
            gap_tets.push(ord);
            weights.push(w);
            gap_volume += v_int;
        }
        if !gap_tets.is_empty() && gap_volume > 0.0 {
            let feed = GapFeed {
                gap_tets,
                weights,
                gap_volume,
                direction,
                gap_length,
                footprint_area: prism.footprint_area,
                cross_area: gap_volume / gap_length,
            };
            log_cross_section(&feed);
            return Ok(feed);
        }
    }

    // Legacy fallback: centroid-radius selection around the terminal segment.
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
    let n = selected.len();
    let feed = GapFeed {
        gap_tets: selected,
        weights: vec![1.0; n],
        gap_volume,
        direction,
        gap_length,
        footprint_area: gap_volume / gap_length,
        cross_area: gap_volume / gap_length,
    };
    log_cross_section(&feed);
    Ok(feed)
}

/// Right-hand side for the 1 A impressed current.
///
/// `b_i = −j·ω·μ0·L_gap·(d̂·∫N_i dV)/V_gap` — purely imaginary by
/// construction. Each gap tet's moment is weighted by its prism
/// intersection fraction, so partially covered tets impress current only
/// over their in-prism part (approximating `∫_{tet∩prism} N_i dV` by
/// `w·∫_tet N_i dV`, exact for a tet fully inside or outside).
pub fn build_port_rhs(
    mesh: &Mesh,
    sys: &EdgeSystem,
    feed: &GapFeed,
    frequency_hz: f64,
) -> Result<Vec<Complex64>> {
    let omega = 2.0 * PI * frequency_hz;
    let coef = (-omega * MU0) * feed.gap_length / feed.gap_volume;
    let mut b = vec![Complex64::new(0.0, 0.0); sys.free_dof_count];
    for (gi, &ord) in feed.gap_tets.iter().enumerate() {
        let w = feed.weights.get(gi).copied().unwrap_or(1.0);
        if w == 0.0 {
            continue;
        }
        let tet = sys.active_tets[ord] as usize;
        let geom = tet_geometry(mesh, tet)?;
        for e in 0..6 {
            let d = sys.dof_of_edge[sys.tet_edges[ord * 6 + e] as usize];
            if d < 0 {
                continue;
            }
            let iv = edge_field_integral(
                &geom,
                e,
                sys.tet_edge_sign[ord * 6 + e],
                edge_length(mesh, tet, e),
            );
            let proj =
                feed.direction[0] * iv[0] + feed.direction[1] * iv[1] + feed.direction[2] * iv[2];
            b[d as usize].im += coef * w * proj;
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
        let lambda = [
            geom.lambda(0, p),
            geom.lambda(1, p),
            geom.lambda(2, p),
            geom.lambda(3, p),
        ];
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
        let proj = e_vec[0] * feed.direction[0]
            + e_vec[1] * feed.direction[1]
            + e_vec[2] * feed.direction[2];
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
            weights: vec![1.0],
            gap_volume: 1.0 / 6.0,
            direction: [1.0, 0.0, 0.0],
            gap_length: 1.0,
            footprint_area: 1.0 / 6.0,
            cross_area: 1.0 / 6.0,
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

    /// Unit cube split into the 6 Kuhn tets around the (0,0,0)–(1,1,1)
    /// diagonal. Vertex order: x + 4·(y + 2·z)-style bit packing
    /// (bit0 = x, bit1 = y, bit2 = z).
    fn kuhn_cube_mesh() -> Mesh {
        let mut vertices = Vec::new();
        for v in 0..8u32 {
            vertices.push((v & 1) as f64);
            vertices.push(((v >> 1) & 1) as f64);
            vertices.push(((v >> 2) & 1) as f64);
        }
        Mesh {
            vertices,
            tetrahedra: vec![
                0, 1, 3, 7, //
                0, 3, 2, 7, //
                0, 2, 6, 7, //
                0, 6, 4, 7, //
                0, 4, 5, 7, //
                0, 5, 1, 7,
            ],
            tet_regions: vec![1; 6],
            materials: vec![Material::air()],
            // Bottom (z=0): (0,1,3),(0,3,2); top (z=1): (4,6,7),(4,7,5).
            boundary_facets: vec![0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5],
            boundary_attributes: vec![10, 10, 10, 10],
            port_positive_facets: vec![2, 3],
            port_negative_facets: vec![0, 1],
        }
    }

    #[test]
    fn prism_selection_tiles_cube_exactly() {
        let mesh = kuhn_cube_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let (pos, neg) = terminals_from_facets(&mesh).unwrap();
        assert_eq!(pos, [0.5, 0.5, 1.0]);
        assert_eq!(neg, [0.5, 0.5, 0.0]);
        let feed = select_gap_feed(&mesh, &sys, pos, neg).unwrap();
        assert_eq!(feed.gap_tets.len(), 6);
        assert!(feed.weights.iter().all(|&w| (w - 1.0).abs() < 1e-12));
        // Prism = unit cube: A_cross must equal the footprint exactly.
        assert!((feed.footprint_area - 1.0).abs() < 1e-12);
        assert!((feed.gap_volume - 1.0).abs() < 1e-12);
        assert!((feed.cross_area - feed.footprint_area).abs() < 1e-12);
    }

    #[test]
    fn prism_selection_partial_coverage() {
        // Port facets = a single triangle on each face, covering the
        // (x,y) triangle (0,0),(1,0),(0,1): footprint 0.5, prism volume
        // 0.5. The x+y=1 prism wall is not a Kuhn face plane, so some
        // tets are genuinely partially covered — but the intersection
        // volumes must still sum to the prism volume exactly.
        let mut mesh = kuhn_cube_mesh();
        mesh.boundary_facets = vec![0, 1, 2, 4, 5, 6];
        mesh.boundary_attributes = vec![10, 10];
        mesh.port_positive_facets = vec![1];
        mesh.port_negative_facets = vec![0];
        let sys = build_edge_system(&mesh).unwrap();
        let (pos, neg) = terminals_from_facets(&mesh).unwrap();
        let feed = select_gap_feed(&mesh, &sys, pos, neg).unwrap();
        assert!((feed.footprint_area - 0.5).abs() < 1e-12);
        assert!((feed.gap_volume - 0.5).abs() < 1e-12, "V={}", feed.gap_volume);
        assert!((feed.cross_area - feed.footprint_area).abs() < 1e-12);
        assert!(feed
            .weights
            .iter()
            .any(|&w| w > 1e-6 && w < 1.0 - 1e-12));
        assert!(feed.weights.iter().all(|&w| w > 0.0 && w <= 1.0 + 1e-12));
    }

    #[test]
    fn prism_selection_ignores_outside_tets() {
        // Two Kuhn cubes side by side along x; the port facets cover only
        // the left cube's faces, so only its 6 tets may be selected.
        let mut vertices = Vec::new();
        for x in 0..3u32 {
            for y in 0..2u32 {
                for z in 0..2u32 {
                    vertices.push(x as f64);
                    vertices.push(y as f64);
                    vertices.push(z as f64);
                }
            }
        }
        // idx(x,y,z) = x*4 + y*2 + z.
        let idx = |x: u32, y: u32, z: u32| x * 4 + y * 2 + z;
        let kuhn = |o: u32| -> Vec<u32> {
            // Local corner map for cube with x-offset o (in idx units of 4).
            let c = |lx: u32, ly: u32, lz: u32| idx(o + lx, ly, lz);
            vec![
                c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), //
                c(0, 0, 0), c(1, 1, 0), c(0, 1, 0), c(1, 1, 1), //
                c(0, 0, 0), c(0, 1, 0), c(0, 1, 1), c(1, 1, 1), //
                c(0, 0, 0), c(0, 1, 1), c(0, 0, 1), c(1, 1, 1), //
                c(0, 0, 0), c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), //
                c(0, 0, 0), c(1, 0, 1), c(1, 0, 0), c(1, 1, 1),
            ]
        };
        let mut tets = kuhn(0);
        tets.extend(kuhn(1));
        let mesh = Mesh {
            vertices,
            tetrahedra: tets,
            tet_regions: vec![1; 12],
            materials: vec![Material::air()],
            // Left cube bottom (z=0): (0,4,6),(0,6,2) in idx() terms;
            // top (z=1): (1,3,7),(1,7,5).
            boundary_facets: vec![
                idx(0, 0, 0), idx(1, 0, 0), idx(1, 1, 0), //
                idx(0, 0, 0), idx(1, 1, 0), idx(0, 1, 0), //
                idx(0, 0, 1), idx(0, 1, 1), idx(1, 1, 1), //
                idx(0, 0, 1), idx(1, 1, 1), idx(1, 0, 1),
            ],
            boundary_attributes: vec![10, 10, 10, 10],
            port_positive_facets: vec![2, 3],
            port_negative_facets: vec![0, 1],
        };
        let sys = build_edge_system(&mesh).unwrap();
        let (pos, neg) = terminals_from_facets(&mesh).unwrap();
        let feed = select_gap_feed(&mesh, &sys, pos, neg).unwrap();
        assert_eq!(feed.gap_tets.len(), 6);
        assert!((feed.gap_volume - 1.0).abs() < 1e-12);
        assert!((feed.cross_area - feed.footprint_area).abs() < 1e-12);
        // Selected tets are exactly the left cube's (mesh tets 0..6).
        for &ord in &feed.gap_tets {
            assert!((sys.active_tets[ord] as usize) < 6);
        }
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
