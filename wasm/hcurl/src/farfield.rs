//! Stratton–Chu near-to-far-field transform.
//!
//! Equivalent surface currents `J_s = n×H`, `M_s = E×n` are formed on the
//! absorbing outer boundary (with the normal forced **outward** — unlike the
//! ABC assembly) and radiated with the `e^{+jωt}` Stratton–Chu integrals:
//!
//! ```text
//! E_θ = −j·k0/(4π)·(η0·N_θ + L_φ),   E_φ = −j·k0/(4π)·(η0·N_φ − L_θ)
//! ```
//!
//! where `N = Σ aJ·e^{+jk·r̂·r′}`, `L = Σ aM·e^{+jk·r̂·r′}`.
//!
//! **Component-order warning:** a past revision swapped two `M_s = E×n`
//! components and pointed the gain lobe down through the ground plane. The
//! cross products live in [`surface_currents`], pinned by unit tests.

use crate::clock::Stopwatch;
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::abi::ATTR_EXTERIOR_ABSORBING;
use crate::basis::{edge_length, TET_EDGES};
use crate::budget::Budget;
use crate::constants::{C0, ETA0, MU0};
use crate::edge::EdgeSystem;
use crate::error::{Error, Result};
use crate::geometry::tet_geometry;
use crate::mesh::Mesh;

fn sorted_triple(u: u32, v: u32, w: u32) -> (u32, u32, u32) {
    let mut s = [u, v, w];
    s.sort_unstable();
    (s[0], s[1], s[2])
}

/// Sampling options for the near-to-far-field transform.
#[derive(Debug, Clone, Copy)]
pub struct FarFieldOptions {
    /// Polar step in degrees, (0, 30].
    pub theta_step_deg: f64,
    /// Azimuth step in degrees, (0, 60].
    pub phi_step_deg: f64,
}

impl Default for FarFieldOptions {
    fn default() -> Self {
        Self {
            theta_step_deg: 5.0,
            phi_step_deg: 5.0,
        }
    }
}

/// Far-field pattern and power summary.
#[derive(Debug, Clone)]
pub struct FarFieldResult {
    /// Polar angles in degrees, 0..=180.
    pub theta_deg: Vec<f64>,
    /// Azimuth angles in degrees, 0..<360.
    pub phi_deg: Vec<f64>,
    /// IEEE gain in dBi on the [theta][phi] grid (`4πU/P_accepted`).
    pub gain_dbi: Vec<Vec<f64>>,
    /// Directivity in dBi on the [theta][phi] grid (`4πU/P_radiated`).
    pub directivity_dbi: Vec<Vec<f64>>,
    pub max_gain_dbi: f64,
    pub max_gain_theta_deg: f64,
    pub max_gain_phi_deg: f64,
    pub max_directivity_dbi: f64,
    pub radiated_power_w: f64,
    pub accepted_power_w: f64,
    /// `radiated_power_w / accepted_power_w`.
    pub radiation_efficiency: f64,
    pub timing_ms: f64,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn cross_complex_vec(a: [f64; 3], b: [Complex64; 3]) -> [Complex64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn cross_vec_complex(a: [Complex64; 3], b: [f64; 3]) -> [Complex64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Equivalent surface currents from the volume field at a Huygens facet.
///
/// `J_s = n×H` with `H = j·curlE/(ωμ)`; `M_s = E×n`. The `E×n` component
/// order is pinned by [`surface_currents_order`] — do not "simplify".
fn surface_currents(
    e: [Complex64; 3],
    curl_e: [Complex64; 3],
    n: [f64; 3],
    omega: f64,
    mu: f64,
) -> ([Complex64; 3], [Complex64; 3]) {
    let h = [
        Complex64::new(0.0, 1.0) * curl_e[0] / (omega * mu),
        Complex64::new(0.0, 1.0) * curl_e[1] / (omega * mu),
        Complex64::new(0.0, 1.0) * curl_e[2] / (omega * mu),
    ];
    let j_s = cross_complex_vec(n, h); // n × H
    let m_s = cross_vec_complex(e, n); // E × n
    (j_s, m_s)
}

/// Compute the far-field pattern from a solved H(curl) field.
///
/// `accepted_power_w` is `P_acc = ½·Re(Zin)` for the 1 A feed. Errors when
/// there are no absorbing facets or the radiated power is non-positive.
#[allow(clippy::too_many_arguments)]
pub fn compute_far_field(
    mesh: &Mesh,
    sys: &EdgeSystem,
    x: &[Complex64],
    frequency_hz: f64,
    accepted_power_w: f64,
    options: &FarFieldOptions,
    budget: &Budget,
) -> Result<FarFieldResult> {
    let t0 = Stopwatch::start();
    if !(accepted_power_w > 0.0) || !accepted_power_w.is_finite() {
        return Err(Error::FarField(
            "acceptedPowerW must be a positive finite value".to_string(),
        ));
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(Error::FarField(
            "frequencyHz must be a positive finite value".to_string(),
        ));
    }
    if !(options.theta_step_deg > 0.0)
        || !(options.phi_step_deg > 0.0)
        || options.theta_step_deg > 30.0
        || options.phi_step_deg > 60.0
    {
        return Err(Error::FarField(
            "far-field sampling steps must lie in (0, 30] x (0, 60] degrees".to_string(),
        ));
    }
    let n_theta = (2usize).max((180.0 / options.theta_step_deg).round() as usize + 1);
    let n_phi = (2usize).max((360.0 / options.phi_step_deg).round() as usize);
    let omega = 2.0 * PI * frequency_hz;
    let k0 = omega / C0;

    // Expand the free-DOF solution onto all global edges (PEC edges read zero).
    let mut edge_x = vec![Complex64::new(0.0, 0.0); sys.edge_count];
    for e in 0..sys.edge_count {
        let d = sys.dof_of_edge[e];
        if d >= 0 {
            edge_x[e] = x[d as usize];
        }
    }

    // Active-tet faces → tet ordinal.
    let mut facet_to_tet: HashMap<(u32, u32, u32), usize> = HashMap::new();
    for (ord, &tet) in sys.active_tets.iter().enumerate() {
        let base = tet as usize * 4;
        let v = [
            mesh.tetrahedra[base],
            mesh.tetrahedra[base + 1],
            mesh.tetrahedra[base + 2],
            mesh.tetrahedra[base + 3],
        ];
        facet_to_tet.insert(sorted_triple(v[1], v[2], v[3]), ord);
        facet_to_tet.insert(sorted_triple(v[0], v[2], v[3]), ord);
        facet_to_tet.insert(sorted_triple(v[0], v[1], v[3]), ord);
        facet_to_tet.insert(sorted_triple(v[0], v[1], v[2]), ord);
    }

    // Huygens surface: area-weighted equivalent currents.
    let mut src_pos: Vec<[f64; 3]> = Vec::new();
    let mut src_j: Vec<[Complex64; 3]> = Vec::new();
    let mut src_m: Vec<[Complex64; 3]> = Vec::new();
    for bf in 0..mesh.facet_count() {
        if mesh.boundary_attributes[bf] != ATTR_EXTERIOR_ABSORBING {
            continue;
        }
        let base = bf * 3;
        let (u, v, w) = (
            mesh.boundary_facets[base],
            mesh.boundary_facets[base + 1],
            mesh.boundary_facets[base + 2],
        );
        let &ord = match facet_to_tet.get(&sorted_triple(u, v, w)) {
            Some(o) => o,
            None => continue,
        };
        let tet = sys.active_tets[ord] as usize;
        let geom = tet_geometry(mesh, tet)?;

        let pu = mesh.vertex(u);
        let pv = mesh.vertex(v);
        let pw = mesh.vertex(w);
        let mut n = cross(
            [pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]],
            [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]],
        );
        let area2 = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if !(area2 > 0.0) {
            continue;
        }
        let area = area2 / 2.0;
        n[0] /= area2;
        n[1] /= area2;
        n[2] /= area2;
        // Force the normal outward (away from the tet interior).
        let fc = [
            (pu[0] + pv[0] + pw[0]) / 3.0,
            (pu[1] + pv[1] + pw[1]) / 3.0,
            (pu[2] + pv[2] + pw[2]) / 3.0,
        ];
        let outward = [
            fc[0] - geom.centroid[0],
            fc[1] - geom.centroid[1],
            fc[2] - geom.centroid[2],
        ];
        if n[0] * outward[0] + n[1] * outward[1] + n[2] * outward[2] < 0.0 {
            n = [-n[0], -n[1], -n[2]];
        }

        // Reconstruct E and curl E at the facet centroid (λ = 1/4).
        let mut e = [Complex64::new(0.0, 0.0); 3];
        let mut curl_e = [Complex64::new(0.0, 0.0); 3];
        for edge in 0..6 {
            let ge = sys.tet_edges[ord * 6 + edge] as usize;
            let xd = edge_x[ge];
            if xd.re == 0.0 && xd.im == 0.0 {
                continue;
            }
            let [a, b] = TET_EDGES[edge];
            let s = sys.tet_edge_sign[ord * 6 + edge] as f64;
            let len = edge_length(mesh, tet, edge);
            let ga = geom.grad_lambda(a);
            let gb = geom.grad_lambda(b);
            // N_e(centroid) = s·l/4·(∇λb − ∇λa).
            let f = s * len / 4.0;
            let nv = [
                f * (gb[0] - ga[0]),
                f * (gb[1] - ga[1]),
                f * (gb[2] - ga[2]),
            ];
            // curl N_e = s·2·l·(∇λa × ∇λb).
            let g = s * 2.0 * len;
            let cv = [
                g * (ga[1] * gb[2] - ga[2] * gb[1]),
                g * (ga[2] * gb[0] - ga[0] * gb[2]),
                g * (ga[0] * gb[1] - ga[1] * gb[0]),
            ];
            for k in 0..3 {
                e[k] += xd * nv[k];
                curl_e[k] += xd * cv[k];
            }
        }
        let mu = MU0 * mesh.material_for_region(mesh.tet_regions[tet]).mu_r;
        let (j_s, m_s) = surface_currents(e, curl_e, n, omega, mu);
        src_pos.push(fc);
        src_j.push([area * j_s[0], area * j_s[1], area * j_s[2]]);
        src_m.push([area * m_s[0], area * m_s[1], area * m_s[2]]);
    }
    if src_pos.is_empty() {
        return Err(Error::FarField(
            "no absorbing-boundary facets found for the far-field transform".to_string(),
        ));
    }

    // Radiate the equivalent currents on the θ/φ grid.
    let theta_deg: Vec<f64> = (0..n_theta)
        .map(|ti| ti as f64 * 180.0 / (n_theta - 1) as f64)
        .collect();
    let phi_deg: Vec<f64> = (0..n_phi)
        .map(|pj| pj as f64 * 360.0 / n_phi as f64)
        .collect();
    let k0_over_4pi = k0 / (4.0 * PI);
    let d_theta = PI / (n_theta - 1) as f64;
    let d_phi = 2.0 * PI / n_phi as f64;
    let neg_j = Complex64::new(0.0, -1.0);

    // Each theta row is independent (read-only shared sources); the ordered
    // collect keeps `u_grid` bit-identical to the serial loop. The power
    // integral and gain/max scan below stay serial: they are cheap, and the
    // serial max-scan keeps tie-breaking deterministic.
    let rows: Vec<Result<Vec<f64>>> = crate::par::map_indexed(&theta_deg, |_, &theta_d| {
        budget.check()?;
        let theta = theta_d * PI / 180.0;
        let (st, ct) = theta.sin_cos();
        let mut u_row = Vec::with_capacity(n_phi);
        for &phi_d in &phi_deg {
            let phi = phi_d * PI / 180.0;
            let (sp, cp) = phi.sin_cos();
            let rhat = [st * cp, st * sp, ct];
            let that = [ct * cp, ct * sp, -st];
            let phat = [-sp, cp, 0.0];

            let mut n_vec = [Complex64::new(0.0, 0.0); 3];
            let mut l_vec = [Complex64::new(0.0, 0.0); 3];
            for q in 0..src_pos.len() {
                let phase = k0
                    * (rhat[0] * src_pos[q][0] + rhat[1] * src_pos[q][1] + rhat[2] * src_pos[q][2]);
                let (s, c) = phase.sin_cos();
                let ejk = Complex64::new(c, s);
                for k in 0..3 {
                    n_vec[k] += src_j[q][k] * ejk;
                    l_vec[k] += src_m[q][k] * ejk;
                }
            }
            let dot = |a: [Complex64; 3], b: [f64; 3]| -> Complex64 {
                a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            };
            let n_th = dot(n_vec, that);
            let n_ph = dot(n_vec, phat);
            let l_th = dot(l_vec, that);
            let l_ph = dot(l_vec, phat);
            // E_th = −j·k0/(4π)·(η0·N_th + L_ph), E_ph = −j·k0/(4π)·(η0·N_ph − L_th).
            let e_th = neg_j * k0_over_4pi * (ETA0 * n_th + l_ph);
            let e_ph = neg_j * k0_over_4pi * (ETA0 * n_ph - l_th);
            u_row.push((e_th.norm_sqr() + e_ph.norm_sqr()) / (2.0 * ETA0));
        }
        Ok(u_row)
    });

    let mut u_grid: Vec<Vec<f64>> = Vec::with_capacity(n_theta);
    for row in rows {
        u_grid.push(row?);
    }

    let mut radiated_power_w = 0.0;
    for (ti, &theta_d) in theta_deg.iter().enumerate() {
        let w = (theta_d * PI / 180.0).sin() * d_theta * d_phi;
        for &u in &u_grid[ti] {
            radiated_power_w += u * w;
        }
    }
    if !(radiated_power_w > 0.0) || !radiated_power_w.is_finite() {
        return Err(Error::FarField(
            "far-field power integral is non-positive or non-finite".to_string(),
        ));
    }

    let to_dbi = |x: f64| if x > 0.0 { 10.0 * x.log10() } else { -400.0 };
    let mut gain_dbi = Vec::with_capacity(n_theta);
    let mut directivity_dbi = Vec::with_capacity(n_theta);
    let mut max_gain_dbi = f64::NEG_INFINITY;
    let mut max_gain_theta_deg = 0.0;
    let mut max_gain_phi_deg = 0.0;
    let mut max_directivity_dbi = f64::NEG_INFINITY;
    for (ti, u_row) in u_grid.iter().enumerate() {
        let mut g_row = Vec::with_capacity(n_phi);
        let mut d_row = Vec::with_capacity(n_phi);
        for (pj, &u) in u_row.iter().enumerate() {
            let g = to_dbi(4.0 * PI * u / accepted_power_w);
            let d = to_dbi(4.0 * PI * u / radiated_power_w);
            if !g.is_finite() || !d.is_finite() {
                return Err(Error::FarField(
                    "far-field pattern is non-finite".to_string(),
                ));
            }
            if g > max_gain_dbi {
                max_gain_dbi = g;
                max_gain_theta_deg = theta_deg[ti];
                max_gain_phi_deg = phi_deg[pj];
            }
            if d > max_directivity_dbi {
                max_directivity_dbi = d;
            }
            g_row.push(g);
            d_row.push(d);
        }
        gain_dbi.push(g_row);
        directivity_dbi.push(d_row);
    }

    Ok(FarFieldResult {
        theta_deg,
        phi_deg,
        gain_dbi,
        directivity_dbi,
        max_gain_dbi,
        max_gain_theta_deg,
        max_gain_phi_deg,
        max_directivity_dbi,
        radiated_power_w,
        accepted_power_w,
        radiation_efficiency: radiated_power_w / accepted_power_w,
        timing_ms: t0.elapsed_ms(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::build_edge_system;
    use crate::mesh::Material;

    fn z(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// Pins the E×n component order: E=x̂, n=ẑ must give M_s = −ŷ.
    /// (The historical swap produced (0,0,−1) instead, tipping the lobe
    /// through the ground plane.)
    #[test]
    fn surface_currents_order() {
        let e = [z(1.0, 0.0), z(0.0, 0.0), z(0.0, 0.0)];
        let curl_e = [z(0.0, 0.0); 3];
        let (j_s, m_s) = surface_currents(e, curl_e, [0.0, 0.0, 1.0], 1.0, 1.0);
        assert!(j_s.iter().all(|v| v.norm() == 0.0));
        assert!((m_s[0].re).abs() < 1e-15);
        assert!((m_s[1].re + 1.0).abs() < 1e-15, "m_s={m_s:?}");
        assert!((m_s[2].re).abs() < 1e-15);
    }

    /// Pins the n×H order: curlE=2ŷ, ωμ=1 → H=j2ŷ, n=ẑ → J_s = −j2x̂.
    #[test]
    fn magnetic_current_from_curl() {
        let e = [z(0.0, 0.0); 3];
        let curl_e = [z(0.0, 0.0), z(2.0, 0.0), z(0.0, 0.0)];
        let (j_s, m_s) = surface_currents(e, curl_e, [0.0, 0.0, 1.0], 1.0, 1.0);
        assert!((j_s[0].im + 2.0).abs() < 1e-15, "j_s={j_s:?}");
        assert!(j_s[0].re.abs() < 1e-15);
        assert!(j_s[1].norm() < 1e-15 && j_s[2].norm() < 1e-15);
        assert!(m_s.iter().all(|v| v.norm() == 0.0));
    }

    fn single_facet_mesh() -> Mesh {
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
    fn grid_dimensions_follow_steps() {
        let mesh = single_facet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let x = vec![z(1.0, 0.0); sys.free_dof_count];
        let opts = FarFieldOptions {
            theta_step_deg: 30.0,
            phi_step_deg: 60.0,
        };
        let ff = compute_far_field(&mesh, &sys, &x, 3.0e9, 1.0, &opts, &Budget::none()).unwrap();
        assert_eq!(ff.theta_deg.len(), 7); // 0, 30, ..., 180
        assert_eq!(ff.phi_deg.len(), 6); // 0, 60, ..., 300
        assert_eq!(ff.gain_dbi.len(), 7);
        assert!(ff.gain_dbi.iter().all(|r| r.len() == 6));
        assert!(ff.max_gain_dbi.is_finite());
        assert!(ff.radiated_power_w > 0.0);
    }

    #[test]
    fn rejects_bad_options_and_power() {
        let mesh = single_facet_mesh();
        let sys = build_edge_system(&mesh).unwrap();
        let x = vec![z(1.0, 0.0); sys.free_dof_count];
        let bad = FarFieldOptions {
            theta_step_deg: 45.0,
            phi_step_deg: 5.0,
        };
        assert!(compute_far_field(&mesh, &sys, &x, 3.0e9, 1.0, &bad, &Budget::none()).is_err());
        let ok = FarFieldOptions::default();
        assert!(compute_far_field(&mesh, &sys, &x, 3.0e9, 0.0, &ok, &Budget::none()).is_err());
        assert!(compute_far_field(&mesh, &sys, &x, -1.0, 1.0, &ok, &Budget::none()).is_err());
    }

    #[test]
    fn no_absorbing_facets_is_an_error() {
        let mut mesh = single_facet_mesh();
        mesh.boundary_attributes = vec![crate::abi::ATTR_CONDUCTOR_PEC];
        let sys = build_edge_system(&mesh).unwrap();
        let x = vec![z(1.0, 0.0); sys.free_dof_count];
        let err = compute_far_field(
            &mesh,
            &sys,
            &x,
            3.0e9,
            1.0,
            &FarFieldOptions::default(),
            &Budget::none(),
        );
        assert!(matches!(err, Err(Error::FarField(_))));
    }
}
