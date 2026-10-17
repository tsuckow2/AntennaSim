//! Gap-feed diagnostic dump (no solve).
//!
//! For each packed request file passed on the command line, prints:
//! terminals, a per-attempt replay of the legacy centroid-radius
//! selection (verbatim copy of the pre-fix `select_gap_feed` loop),
//! the current library selection for comparison, gap volume /
//! cross-section, port facet-set projected extents, and the mesh
//! material table. (Written for the 2026-10-17 port audit; the replay
//! documents the legacy rule that the prism-intersection selection
//! replaced.)
//!
//! ```sh
//! cargo run --release --example gap_dump -- /path/to/2300.req [...]
//! ```

use hcurl_fem::abi::PointRequest;
use hcurl_fem::edge::build_edge_system;
use hcurl_fem::geometry::tet_geometry;
use hcurl_fem::mesh::Mesh;
use hcurl_fem::port::{select_gap_feed, terminals_from_facets};

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

/// Monotone-chain convex hull; returns hull vertices CCW and the area.
fn hull_area(points: &[[f64; 2]]) -> ([f64; 2], [f64; 2], f64, usize) {
    let mut pts: Vec<[f64; 2]> = points.to_vec();
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));
    pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-15 && (a[1] - b[1]).abs() < 1e-15);
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in &pts {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    if pts.len() < 3 {
        return (lo, hi, 0.0, pts.len());
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
    (lo, hi, (area2 / 2.0).abs(), hull.len())
}

fn facet_projection(mesh: &Mesh, facets: &[u32], origin: [f64; 3], u: [f64; 3], v: [f64; 3]) -> Vec<[f64; 2]> {
    let mut pts = Vec::new();
    for &f in facets {
        let base = f as usize * 3;
        for k in 0..3 {
            let p = mesh.vertex(mesh.boundary_facets[base + k]);
            let rel = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
            pts.push([dot(rel, u), dot(rel, v)]);
        }
    }
    pts
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    assert!(!paths.is_empty(), "usage: gap_dump <request.req> [...]");
    for path in &paths {
        println!("=== {path} ===");
        let bytes = std::fs::read(path).expect("read request");
        let request = PointRequest::parse(&bytes).expect("parse request");
        let mesh = Mesh::from_request(&request).expect("mesh");
        let sys = build_edge_system(&mesh).expect("edge system");
        let (positive, negative) = terminals_from_facets(&mesh).expect("terminals");
        let gv = [
            positive[0] - negative[0],
            positive[1] - negative[1],
            positive[2] - negative[2],
        ];
        let gap_length = dot(gv, gv).sqrt();
        let dir = [gv[0] / gap_length, gv[1] / gap_length, gv[2] / gap_length];
        println!(
            "freq={:.0} MHz tets={} active={} dofs={}",
            request.frequency_hz / 1e6,
            mesh.tet_count(),
            sys.active_tet_count,
            sys.free_dof_count
        );
        println!(
            "terminals: pos=({:.4},{:.4},{:.4}) mm neg=({:.4},{:.4},{:.4}) mm L={:.4} mm dir=({:.2},{:.2},{:.2})",
            positive[0] * 1e3, positive[1] * 1e3, positive[2] * 1e3,
            negative[0] * 1e3, negative[1] * 1e3, negative[2] * 1e3,
            gap_length * 1e3, dir[0], dir[1], dir[2]
        );

        // Verbatim replay of the legacy radius-doubling loop.
        let mut centroids = Vec::with_capacity(sys.active_tet_count);
        for &tet in &sys.active_tets {
            centroids.push(tet_geometry(&mesh, tet as usize).unwrap().centroid);
        }
        let mut radius = 0.6 * gap_length;
        let mut attempt_used: Option<usize> = None;
        let mut replay_selected: Vec<usize> = Vec::new();
        for attempt in 0..6 {
            let r2 = radius * radius;
            let mut sel: Vec<usize> = Vec::new();
            for (ord, c) in centroids.iter().enumerate() {
                if point_segment_distance_sq(*c, negative, positive) <= r2 {
                    sel.push(ord);
                }
            }
            println!(
                "  attempt {attempt}: radius={:.3} mm -> {} tets",
                radius * 1e3,
                sel.len()
            );
            if !sel.is_empty() {
                attempt_used = Some(attempt);
                replay_selected = sel;
                break;
            }
            radius *= 2.0;
        }

        // Current library selection vs the legacy replay above.
        let feed = select_gap_feed(&mesh, &sys, positive, negative).expect("feed");
        let same = feed.gap_tets == replay_selected;
        let a_cross = feed.gap_volume / feed.gap_length;
        println!(
            "current: legacy_attempt={:?} gap_tets={} V_gap={:.4} mm^3 A_cross={:.4} mm^2 legacy_match={}",
            attempt_used,
            feed.gap_tets.len(),
            feed.gap_volume * 1e9,
            a_cross * 1e6,
            same
        );
        // Selected-tet bbox (mm).
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &ord in &feed.gap_tets {
            let tet = sys.active_tets[ord] as usize;
            for k in 0..4 {
                let p = mesh.vertex(mesh.tetrahedra[tet * 4 + k]);
                for c in 0..3 {
                    lo[c] = lo[c].min(p[c]);
                    hi[c] = hi[c].max(p[c]);
                }
            }
        }
        println!(
            "selected bbox: x[{:.3},{:.3}] y[{:.3},{:.3}] z[{:.3},{:.3}] mm",
            lo[0] * 1e3, hi[0] * 1e3, lo[1] * 1e3, hi[1] * 1e3, lo[2] * 1e3, hi[2] * 1e3
        );

        // Facet-set projections ⊥ d̂.
        let axis = (0..3).min_by(|&a, &b| dir[a].abs().partial_cmp(&dir[b].abs()).unwrap()).unwrap();
        let mut u = [0.0; 3];
        u[axis] = 1.0;
        let d = dot(u, dir);
        u = [u[0] - d * dir[0], u[1] - d * dir[1], u[2] - d * dir[2]];
        let un = dot(u, u).sqrt();
        u = [u[0] / un, u[1] / un, u[2] / un];
        let v = cross(dir, u);
        for (label, facets) in [
            ("pos", &mesh.port_positive_facets),
            ("neg", &mesh.port_negative_facets),
        ] {
            let pts = facet_projection(&mesh, facets, negative, u, v);
            let (plo, phi, area, nhull) = hull_area(&pts);
            println!(
                "facets {label}: n={} proj-u[{:.3},{:.3}] proj-v[{:.3},{:.3}] mm hull_area={:.4} mm^2 hull_pts={}",
                facets.len(),
                plo[0] * 1e3, phi[0] * 1e3, plo[1] * 1e3, phi[1] * 1e3,
                area * 1e6, nhull
            );
        }
        let both: Vec<[f64; 2]> = facet_projection(&mesh, &mesh.port_positive_facets, negative, u, v)
            .into_iter()
            .chain(facet_projection(&mesh, &mesh.port_negative_facets, negative, u, v))
            .collect();
        let (_, _, area_union, _) = hull_area(&both);
        println!(
            "union hull area={:.4} mm^2  A_cross/union={:.3}",
            area_union * 1e6,
            a_cross / area_union
        );

        print!("materials:");
        for m in &mesh.materials {
            print!(
                " [attr {} er={} mu={} sigma={:e} tand={}]",
                m.attribute, m.epsilon_r, m.mu_r, m.conductivity_s_m, m.loss_tangent
            );
        }
        println!();
    }
}
