//! Mesh and material data structures.
//!
//! [`Mesh`] is the validated domain representation built from a parsed
//! [`PointRequest`](crate::abi::PointRequest): flat arrays plus sanitized
//! materials. Conductor tets (σ ≥ 1e4 S/m) are identified here; their
//! exclusion from the edge system happens in [`crate::edge`].

use crate::abi::{self, PointRequest};
use crate::constants::CONDUCTOR_SIGMA_THRESHOLD;
use crate::error::{Error, Result};

/// Sanitized material for one region attribute.
///
/// Sanitization mirrors the TypeScript reference: non-positive μr/εr fall
/// back to 1, non-finite conductivity to 0, negative or non-finite loss
/// tangent to 0.
#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub attribute: u32,
    pub epsilon_r: f64,
    pub mu_r: f64,
    pub conductivity_s_m: f64,
    pub loss_tangent: f64,
}

impl Material {
    fn sanitize(raw: &abi::Material) -> Self {
        Self {
            attribute: raw.attribute,
            epsilon_r: if raw.epsilon_r > 0.0 {
                raw.epsilon_r
            } else {
                1.0
            },
            mu_r: if raw.mu_r > 0.0 { raw.mu_r } else { 1.0 },
            conductivity_s_m: if raw.conductivity_s_m.is_finite() {
                raw.conductivity_s_m
            } else {
                0.0
            },
            loss_tangent: if raw.loss_tangent.is_finite() {
                raw.loss_tangent.max(0.0)
            } else {
                0.0
            },
        }
    }

    /// Free-space fallback for region attributes missing from the table.
    pub fn air() -> Self {
        Self {
            attribute: 0,
            epsilon_r: 1.0,
            mu_r: 1.0,
            conductivity_s_m: 0.0,
            loss_tangent: 0.0,
        }
    }

    /// True when the region is excluded from the domain (PEC surface).
    pub fn is_conductor(&self) -> bool {
        self.conductivity_s_m >= CONDUCTOR_SIGMA_THRESHOLD
    }
}

/// Validated tetrahedral mesh with boundary and port data.
///
/// Layout is flat Structure-of-Arrays for cache-friendly assembly:
/// `vertices[3*v + {0,1,2}]`, `tetrahedra[4*t + {0..3}]`,
/// `boundary_facets[3*f + {0..2}]`.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<f64>,
    pub tetrahedra: Vec<u32>,
    pub tet_regions: Vec<u32>,
    /// Sanitized materials, sorted by region attribute (as packed).
    pub materials: Vec<Material>,
    pub boundary_facets: Vec<u32>,
    pub boundary_attributes: Vec<u32>,
    /// Indices into `boundary_facets`.
    pub port_positive_facets: Vec<u32>,
    /// Indices into `boundary_facets`.
    pub port_negative_facets: Vec<u32>,
}

impl Mesh {
    /// Build the domain mesh from a validated request.
    pub fn from_request(req: &PointRequest) -> Result<Self> {
        let vertex_count = req.vertices.len();
        let mut vertices = Vec::with_capacity(vertex_count * 3);
        for v in &req.vertices {
            vertices.extend_from_slice(v);
        }
        let mut tetrahedra = Vec::with_capacity(req.tets.len() * 4);
        for t in &req.tets {
            tetrahedra.extend_from_slice(t);
        }
        let mut boundary_facets = Vec::with_capacity(req.boundary_facets.len() * 3);
        for f in &req.boundary_facets {
            boundary_facets.extend_from_slice(f);
        }

        let facet_count = req.boundary_facets.len();
        for &fi in req.port_pos_facets.iter().chain(req.port_neg_facets.iter()) {
            if (fi as usize) >= facet_count {
                return Err(Error::Mesh(format!(
                    "port facet index {fi} out of bounds ({facet_count} facets)"
                )));
            }
        }

        Ok(Self {
            vertices,
            tetrahedra,
            tet_regions: req.tet_region.clone(),
            materials: req.materials.iter().map(Material::sanitize).collect(),
            boundary_facets,
            boundary_attributes: req.boundary_attr.clone(),
            port_positive_facets: req.port_pos_facets.clone(),
            port_negative_facets: req.port_neg_facets.clone(),
        })
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// Number of tetrahedra.
    pub fn tet_count(&self) -> usize {
        self.tetrahedra.len() / 4
    }

    /// Number of boundary facets.
    pub fn facet_count(&self) -> usize {
        self.boundary_facets.len() / 3
    }

    /// Material for a region attribute; unknown regions default to air.
    pub fn material_for_region(&self, attribute: u32) -> Material {
        self.materials
            .iter()
            .find(|m| m.attribute == attribute)
            .copied()
            .unwrap_or_else(Material::air)
    }

    /// True when tet `t` is a conductor tet (excluded from the domain).
    pub fn is_conductor_tet(&self, t: usize) -> bool {
        self.material_for_region(self.tet_regions[t]).is_conductor()
    }

    /// Coordinates of vertex `v`.
    pub fn vertex(&self, v: u32) -> [f64; 3] {
        let i = v as usize * 3;
        [self.vertices[i], self.vertices[i + 1], self.vertices[i + 2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_falls_back_to_air_like_values() {
        let raw = abi::Material {
            attribute: 7,
            epsilon_r: -2.0,
            mu_r: 0.0,
            conductivity_s_m: f64::NAN,
            loss_tangent: -0.5,
        };
        let m = Material::sanitize(&raw);
        assert_eq!(m.epsilon_r, 1.0);
        assert_eq!(m.mu_r, 1.0);
        assert_eq!(m.conductivity_s_m, 0.0);
        assert_eq!(m.loss_tangent, 0.0);
        assert!(!m.is_conductor());
    }

    #[test]
    fn conductor_threshold() {
        let raw = abi::Material {
            attribute: 1,
            epsilon_r: 1.0,
            mu_r: 1.0,
            conductivity_s_m: 5.8e7,
            loss_tangent: 0.0,
        };
        assert!(Material::sanitize(&raw).is_conductor());
    }

    #[test]
    fn unknown_region_defaults_to_air() {
        let mesh = Mesh {
            vertices: vec![],
            tetrahedra: vec![],
            tet_regions: vec![],
            materials: vec![],
            boundary_facets: vec![],
            boundary_attributes: vec![],
            port_positive_facets: vec![],
            port_negative_facets: vec![],
        };
        let m = mesh.material_for_region(999);
        assert_eq!((m.epsilon_r, m.mu_r), (1.0, 1.0));
    }
}
