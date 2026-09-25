//! Packed binary request/response wire format (ABI v1).
//!
//! The layout is shared with the TypeScript packer/unpacker in
//! `frontend/src/engine/wasm/mfem-abi.ts` and is **retained verbatim** from
//! the prototype solver so existing packed requests remain valid. All
//! multi-byte fields are little-endian.
//!
//! ## Request layout
//!
//! A 128-byte header followed by packed arrays:
//!
//! ```text
//! offset  size  field
//! ------  ----  -----
//!  0       4    abi_version            (must be 1)
//!  4       4    endianness_sentinel    (must be 0x12345678)
//!  8       4    reserved
//! 16       8    frequency_hz           (f64, > 0)
//! 24       4    vertex_count
//! 28       4    tet_count
//! 32       4    reserved
//! 36       4    boundary_facet_count
//! 40       4    port_facet_count       (total, pos + neg)
//! 44       4    port_pos_count
//! 48       4    vertices_offset        (f64 x3 per vertex)
//! 52       4    tets_offset            (u32 x4 per tet)
//! 56       4    tet_region_offset      (u32 per tet: material attribute)
//! 60       4    boundary_facets_offset (u32 x3 per facet)
//! 64       4    boundary_attr_offset   (u32 per facet)
//! 68       4    port_pos_offset        (u32 per facet: positive terminal facets)
//! 72       4    port_attr_offset       (u32 per port facet, pos-then-neg order)
//! 76       4    material_table_offset  (40-byte records, see Material)
//! 80       4    metadata_offset        (UTF-8 JSON, see RequestMetadata)
//! 84       4    vertices_byte_len
//! 88       4    tets_byte_len
//! 92       4    tet_region_byte_len
//! 96       4    boundary_facets_byte_len
//! 100      4    boundary_attr_byte_len
//! 104      4    port_pos_byte_len
//! 108      4    material_table_byte_len
//! 112      4    port_neg_offset        (u32 per facet: negative terminal facets)
//! 116      4    port_neg_byte_len
//! 120      4    port_attr_byte_len
//! 124      4    metadata_byte_len
//! ```
//!
//! ## Response layout
//!
//! A 104-byte header followed by the field blob, then the message bytes:
//!
//! ```text
//! offset  size  field
//! ------  ----  -----
//!  0       4    abi_version            (1)
//!  4       4    status_code            (see crate::error)
//!  8       4    reserved
//! 16       8    residual_norm          (f64, NaN when unavailable)
//! 24       4    iteration_count        (u32)
//! 28       4    reserved
//! 32       8    zin_real_ohms          (f64, NaN when unavailable)
//! 40       8    zin_imag_ohms          (f64, NaN when unavailable)
//! 48       8    s11_real               (f64, NaN when unavailable)
//! 56       8    s11_imag               (f64, NaN when unavailable)
//! 64       8    timing_assembly_ms     (f64)
//! 72       8    timing_solve_ms        (f64)
//! 80       8    timing_post_ms         (f64)
//! 88       4    field_blob_offset
//! 92       4    field_blob_len
//! 96       4    message_offset         (UTF-8)
//! 100      4    message_len
//! ```

use num_complex::Complex64;
use serde::Deserialize;

use crate::error::{Error, Result};

/// Wire ABI version accepted by this build.
pub const ABI_VERSION: u32 = 1;
/// Fixed request header size in bytes.
pub const REQUEST_HEADER_BYTES: usize = 128;
/// Fixed response header size in bytes.
pub const RESPONSE_HEADER_BYTES: usize = 104;
/// Little-endian sentinel; guards against byte-order confusion.
const ENDIANNESS_SENTINEL: u32 = 0x12345678;

/// Sanity limits mirrored from the TypeScript packer.
pub const MAX_VERTICES: usize = 350_000;
pub const MAX_TETS: usize = 2_000_000;
pub const MAX_REQUEST_BYTES: usize = 512 * 1024 * 1024;

/// Boundary attribute: exterior facet with Silver–Müller ABC.
pub const ATTR_EXTERIOR_ABSORBING: u32 = 10;
/// Boundary attribute: perfect electric conductor.
pub const ATTR_CONDUCTOR_PEC: u32 = 20;
/// Port attribute: positive terminal facet.
pub const ATTR_PORT_POSITIVE: u32 = 30;
/// Port attribute: negative terminal facet.
pub const ATTR_PORT_NEGATIVE: u32 = 31;

/// One 40-byte material table record: `u32 attribute, u32 reserved,`
/// `f64 epsilon_r, f64 mu_r, f64 conductivity_s_m, f64 loss_tangent`.
#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub attribute: u32,
    pub epsilon_r: f64,
    pub mu_r: f64,
    pub conductivity_s_m: f64,
    pub loss_tangent: f64,
}

/// Decoded request metadata JSON. All fields optional for forward
/// compatibility; only `reference_impedance_ohms` affects numerics.
/// Optional far-field request, nested under the `far_field` metadata key.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct FarFieldRequest {
    #[serde(default)]
    pub theta_step_deg: Option<f64>,
    #[serde(default)]
    pub phi_step_deg: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RequestMetadata {
    #[serde(default)]
    pub reference_impedance_ohms: Option<f64>,
    #[serde(default)]
    pub krylov_tolerance: Option<f64>,
    #[serde(default)]
    pub far_field: Option<FarFieldRequest>,
    /// Informational only; typed as `Value` because packers emit versions
    /// as either numbers or strings.
    #[serde(default)]
    pub mesh_artifact_version: Option<serde_json::Value>,
    #[serde(default)]
    pub model_ir_version: Option<serde_json::Value>,
    #[serde(default)]
    pub solver_formulation: Option<serde_json::Value>,
    #[serde(default)]
    pub absorbing_boundary: Option<serde_json::Value>,
}

impl RequestMetadata {
    fn parse(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| Error::InvalidRequest(format!("metadata JSON: {e}")))
    }

    /// Reference impedance for S11; defaults to 50 Ω (RF convention).
    pub fn reference_impedance_ohms(&self) -> f64 {
        self.reference_impedance_ohms.unwrap_or(50.0)
    }

    /// BiCGSTAB relative tolerance; defaults to 1e-7 (validated reference
    /// value). Non-positive or non-finite values fall back to the default.
    pub fn krylov_tolerance(&self) -> f64 {
        match self.krylov_tolerance {
            Some(t) if t.is_finite() && t > 0.0 => t,
            _ => crate::krylov::DEFAULT_TOL,
        }
    }
}

/// A validated single-frequency solve request.
#[derive(Debug, Clone)]
pub struct PointRequest {
    pub frequency_hz: f64,
    /// Vertex coordinates in meters, `[x, y, z]` per vertex.
    pub vertices: Vec<[f64; 3]>,
    /// Tetrahedra as vertex indices.
    pub tets: Vec<[u32; 4]>,
    /// Material attribute per tet (indexes [`Material::attribute`]).
    pub tet_region: Vec<u32>,
    /// Boundary triangles as vertex indices.
    pub boundary_facets: Vec<[u32; 3]>,
    /// Boundary attribute per facet ([`ATTR_EXTERIOR_ABSORBING`], …).
    pub boundary_attr: Vec<u32>,
    /// Positive-terminal port facets (indices into `boundary_facets`).
    pub port_pos_facets: Vec<u32>,
    /// Negative-terminal port facets.
    pub port_neg_facets: Vec<u32>,
    /// Port attributes in pos-then-neg facet order.
    pub port_attr: Vec<u32>,
    /// Material table, sorted by region attribute.
    pub materials: Vec<Material>,
    /// Decoded metadata.
    pub metadata: RequestMetadata,
}

fn read_u32(bytes: &[u8], offset: usize, label: &str) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::InvalidRequest(format!("{label}: header truncated")))
        .and_then(|s| {
            s.try_into()
                .map(u32::from_le_bytes)
                .map_err(|_| Error::InvalidRequest(format!("{label}: bad u32")))
        })
}

fn read_f64(bytes: &[u8], offset: usize, label: &str) -> Result<f64> {
    bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error::InvalidRequest(format!("{label}: header truncated")))
        .and_then(|s| {
            s.try_into()
                .map(f64::from_le_bytes)
                .map_err(|_| Error::InvalidRequest(format!("{label}: bad f64")))
        })
}

/// Copy a little-endian array region out of the request buffer.
fn read_region<T: Copy>(
    bytes: &[u8],
    offset: u32,
    byte_len: u32,
    width: usize,
    label: &str,
    decode: impl Fn(&[u8]) -> T,
) -> Result<Vec<T>> {
    let offset = offset as usize;
    let byte_len = byte_len as usize;
    if byte_len % width != 0 {
        return Err(Error::InvalidRequest(format!(
            "{label}: byte length {byte_len} not a multiple of {width}"
        )));
    }
    let region = bytes.get(offset..offset + byte_len).ok_or_else(|| {
        Error::InvalidRequest(format!("{label}: region extends past buffer bounds"))
    })?;
    Ok(region
        .chunks_exact(width)
        .map(decode)
        .collect())
}

fn decode_f64x3(s: &[u8]) -> [f64; 3] {
    [
        f64::from_le_bytes(s[0..8].try_into().unwrap()),
        f64::from_le_bytes(s[8..16].try_into().unwrap()),
        f64::from_le_bytes(s[16..24].try_into().unwrap()),
    ]
}

fn decode_u32(s: &[u8]) -> u32 {
    u32::from_le_bytes(s[0..4].try_into().unwrap())
}

fn decode_u32x3(s: &[u8]) -> [u32; 3] {
    [
        u32::from_le_bytes(s[0..4].try_into().unwrap()),
        u32::from_le_bytes(s[4..8].try_into().unwrap()),
        u32::from_le_bytes(s[8..12].try_into().unwrap()),
    ]
}

fn decode_u32x4(s: &[u8]) -> [u32; 4] {
    [
        u32::from_le_bytes(s[0..4].try_into().unwrap()),
        u32::from_le_bytes(s[4..8].try_into().unwrap()),
        u32::from_le_bytes(s[8..12].try_into().unwrap()),
        u32::from_le_bytes(s[12..16].try_into().unwrap()),
    ]
}

fn decode_material(s: &[u8]) -> Material {
    Material {
        attribute: u32::from_le_bytes(s[0..4].try_into().unwrap()),
        epsilon_r: f64::from_le_bytes(s[8..16].try_into().unwrap()),
        mu_r: f64::from_le_bytes(s[16..24].try_into().unwrap()),
        conductivity_s_m: f64::from_le_bytes(s[24..32].try_into().unwrap()),
        loss_tangent: f64::from_le_bytes(s[32..40].try_into().unwrap()),
    }
}

impl PointRequest {
    /// Parse and validate a packed v1 request buffer.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < REQUEST_HEADER_BYTES {
            return Err(Error::InvalidRequest(format!(
                "request is {} bytes, smaller than the {}-byte header",
                bytes.len(),
                REQUEST_HEADER_BYTES
            )));
        }
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(Error::InvalidRequest("request exceeds size limit".to_string()));
        }

        let abi_version = read_u32(bytes, 0, "abi_version")?;
        if abi_version != ABI_VERSION {
            return Err(Error::UnsupportedAbi {
                expected: ABI_VERSION,
                found: abi_version,
            });
        }
        if read_u32(bytes, 4, "endianness_sentinel")? != ENDIANNESS_SENTINEL {
            return Err(Error::EndiannessMismatch);
        }

        let frequency_hz = read_f64(bytes, 16, "frequency_hz")?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(Error::InvalidRequest(
                "frequency_hz must be positive and finite".to_string(),
            ));
        }

        let vertex_count = read_u32(bytes, 24, "vertex_count")? as usize;
        let tet_count = read_u32(bytes, 28, "tet_count")? as usize;
        let boundary_facet_count = read_u32(bytes, 36, "boundary_facet_count")? as usize;
        let port_facet_count = read_u32(bytes, 40, "port_facet_count")? as usize;
        let port_pos_count = read_u32(bytes, 44, "port_pos_count")? as usize;
        if vertex_count > MAX_VERTICES {
            return Err(Error::InvalidRequest(format!(
                "vertex_count {vertex_count} exceeds limit {MAX_VERTICES}"
            )));
        }
        if tet_count > MAX_TETS {
            return Err(Error::InvalidRequest(format!(
                "tet_count {tet_count} exceeds limit {MAX_TETS}"
            )));
        }
        if port_pos_count > port_facet_count {
            return Err(Error::InvalidRequest(
                "port_pos_count exceeds port_facet_count".to_string(),
            ));
        }

        let vertices: Vec<[f64; 3]> = read_region(
            bytes,
            read_u32(bytes, 48, "vertices_offset")?,
            read_u32(bytes, 84, "vertices_byte_len")?,
            24,
            "vertices",
            decode_f64x3,
        )?;
        let tets: Vec<[u32; 4]> = read_region(
            bytes,
            read_u32(bytes, 52, "tets_offset")?,
            read_u32(bytes, 88, "tets_byte_len")?,
            16,
            "tets",
            decode_u32x4,
        )?;
        let tet_region: Vec<u32> = read_region(
            bytes,
            read_u32(bytes, 56, "tet_region_offset")?,
            read_u32(bytes, 92, "tet_region_byte_len")?,
            4,
            "tet_region",
            decode_u32,
        )?;
        let boundary_facets: Vec<[u32; 3]> = read_region(
            bytes,
            read_u32(bytes, 60, "boundary_facets_offset")?,
            read_u32(bytes, 96, "boundary_facets_byte_len")?,
            12,
            "boundary_facets",
            decode_u32x3,
        )?;
        let boundary_attr: Vec<u32> = read_region(
            bytes,
            read_u32(bytes, 64, "boundary_attr_offset")?,
            read_u32(bytes, 100, "boundary_attr_byte_len")?,
            4,
            "boundary_attr",
            decode_u32,
        )?;
        let port_pos_facets: Vec<u32> = read_region(
            bytes,
            read_u32(bytes, 68, "port_pos_offset")?,
            read_u32(bytes, 104, "port_pos_byte_len")?,
            4,
            "port_pos_facets",
            decode_u32,
        )?;
        let port_neg_facets: Vec<u32> = read_region(
            bytes,
            read_u32(bytes, 112, "port_neg_offset")?,
            read_u32(bytes, 116, "port_neg_byte_len")?,
            4,
            "port_neg_facets",
            decode_u32,
        )?;
        let port_attr: Vec<u32> = read_region(
            bytes,
            read_u32(bytes, 72, "port_attr_offset")?,
            read_u32(bytes, 120, "port_attr_byte_len")?,
            4,
            "port_attr",
            decode_u32,
        )?;
        let materials: Vec<Material> = read_region(
            bytes,
            read_u32(bytes, 76, "material_table_offset")?,
            read_u32(bytes, 108, "material_table_byte_len")?,
            40,
            "material_table",
            decode_material,
        )?;
        let metadata_offset = read_u32(bytes, 80, "metadata_offset")? as usize;
        let metadata_len = read_u32(bytes, 124, "metadata_byte_len")? as usize;
        let metadata_bytes = bytes.get(metadata_offset..metadata_offset + metadata_len).ok_or_else(|| {
            Error::InvalidRequest("metadata region extends past buffer bounds".to_string())
        })?;
        let metadata = RequestMetadata::parse(metadata_bytes)?;

        // Cross-field consistency, mirroring the TypeScript packer checks.
        if vertices.len() != vertex_count {
            return Err(Error::InvalidRequest("vertices length mismatch".to_string()));
        }
        if tets.len() != tet_count {
            return Err(Error::InvalidRequest("tets length mismatch".to_string()));
        }
        if tet_region.len() != tet_count {
            return Err(Error::InvalidRequest(
                "tet_region length must equal tet_count".to_string(),
            ));
        }
        if boundary_facets.len() != boundary_facet_count {
            return Err(Error::InvalidRequest("boundary_facets length mismatch".to_string()));
        }
        if boundary_attr.len() != boundary_facet_count {
            return Err(Error::InvalidRequest(
                "boundary_attr length must equal boundary_facet_count".to_string(),
            ));
        }
        if port_pos_facets.len() != port_pos_count {
            return Err(Error::InvalidRequest("port_pos_facets length mismatch".to_string()));
        }
        if port_pos_facets.len() + port_neg_facets.len() != port_facet_count {
            return Err(Error::InvalidRequest(
                "port facet counts do not sum to port_facet_count".to_string(),
            ));
        }
        if port_attr.len() != port_facet_count {
            return Err(Error::InvalidRequest(
                "port_attr length must equal port_facet_count".to_string(),
            ));
        }
        if boundary_facets.is_empty() {
            return Err(Error::InvalidRequest(
                "boundary_facets must contain the explicit airbox boundary".to_string(),
            ));
        }
        if port_pos_facets.is_empty() || port_neg_facets.is_empty() {
            return Err(Error::InvalidRequest(
                "port facet sets must contain positive and negative facets".to_string(),
            ));
        }
        if materials.is_empty() {
            return Err(Error::InvalidRequest(
                "material_table must contain at least one material record".to_string(),
            ));
        }
        for tet in &tets {
            for &v in tet {
                if (v as usize) >= vertex_count {
                    return Err(Error::InvalidRequest("tet vertex index out of bounds".to_string()));
                }
            }
        }
        for facet in &boundary_facets {
            for &v in facet {
                if (v as usize) >= vertex_count {
                    return Err(Error::InvalidRequest("facet vertex index out of bounds".to_string()));
                }
            }
        }

        Ok(Self {
            frequency_hz,
            vertices,
            tets,
            tet_region,
            boundary_facets,
            boundary_attr,
            port_pos_facets,
            port_neg_facets,
            port_attr,
            materials,
            metadata,
        })
    }
}

/// Phase timings in milliseconds.
#[derive(Debug, Clone, Default)]
pub struct TimingMs {
    pub assembly: f64,
    pub solve: f64,
    pub post: f64,
}

/// Versioned far-field payload carried in the response `field_blob`.
///
/// Layout (all little-endian):
///
/// ```text
/// offset  size  field
/// ------  ----  -----
///  0       4    blob_version             (= 1)
///  4       4    n_theta                  (u32)
///  8       4    n_phi                    (u32)
/// 12       8    max_gain_dbi             (f64)
/// 20       8    max_gain_theta_deg       (f64)
/// 28       8    max_gain_phi_deg         (f64)
/// 36       8    max_directivity_dbi      (f64)
/// 44       8    radiated_power_w         (f64)
/// 52       8    accepted_power_w         (f64)
/// 60       8    radiation_efficiency     (f64)
/// 68      8*n_theta  theta_deg           (f64, 0..=180)
/// ..      8*n_phi    phi_deg             (f64, 0..<360)
/// ..      8*n_theta*n_phi  gain_dbi      (f64, row-major [theta][phi])
/// ..      8*n_theta*n_phi  directivity_dbi (f64, row-major [theta][phi])
/// ```
#[derive(Debug, Clone)]
pub struct FarFieldBlob {
    pub theta_deg: Vec<f64>,
    pub phi_deg: Vec<f64>,
    pub gain_dbi: Vec<f64>,
    pub directivity_dbi: Vec<f64>,
    pub max_gain_dbi: f64,
    pub max_gain_theta_deg: f64,
    pub max_gain_phi_deg: f64,
    pub max_directivity_dbi: f64,
    pub radiated_power_w: f64,
    pub accepted_power_w: f64,
    pub radiation_efficiency: f64,
}

/// Blob version written by [`FarFieldBlob::serialize`].
pub const FAR_FIELD_BLOB_VERSION: u32 = 1;

impl FarFieldBlob {
    /// Build from a [`crate::farfield::FarFieldResult`], flattening the
    /// `[theta][phi]` grids row-major.
    pub fn from_result(r: &crate::farfield::FarFieldResult) -> Self {
        Self {
            theta_deg: r.theta_deg.clone(),
            phi_deg: r.phi_deg.clone(),
            gain_dbi: r.gain_dbi.iter().flatten().copied().collect(),
            directivity_dbi: r.directivity_dbi.iter().flatten().copied().collect(),
            max_gain_dbi: r.max_gain_dbi,
            max_gain_theta_deg: r.max_gain_theta_deg,
            max_gain_phi_deg: r.max_gain_phi_deg,
            max_directivity_dbi: r.max_directivity_dbi,
            radiated_power_w: r.radiated_power_w,
            accepted_power_w: r.accepted_power_w,
            radiation_efficiency: r.radiation_efficiency,
        }
    }

    /// Serialize to the documented wire layout.
    pub fn serialize(&self) -> Vec<u8> {
        let n_theta = self.theta_deg.len();
        let n_phi = self.phi_deg.len();
        assert_eq!(self.gain_dbi.len(), n_theta * n_phi);
        assert_eq!(self.directivity_dbi.len(), n_theta * n_phi);
        let mut out = Vec::with_capacity(68 + 8 * (n_theta + n_phi + 2 * n_theta * n_phi));
        out.extend_from_slice(&FAR_FIELD_BLOB_VERSION.to_le_bytes());
        out.extend_from_slice(&(n_theta as u32).to_le_bytes());
        out.extend_from_slice(&(n_phi as u32).to_le_bytes());
        for v in [
            self.max_gain_dbi,
            self.max_gain_theta_deg,
            self.max_gain_phi_deg,
            self.max_directivity_dbi,
            self.radiated_power_w,
            self.accepted_power_w,
            self.radiation_efficiency,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in self
            .theta_deg
            .iter()
            .chain(self.phi_deg.iter())
            .chain(self.gain_dbi.iter())
            .chain(self.directivity_dbi.iter())
        {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Parse a blob produced by [`FarFieldBlob::serialize`].
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let rd_u32 = |off: usize| -> Result<u32> {
            bytes
                .get(off..off + 4)
                .and_then(|s| s.try_into().ok())
                .map(u32::from_le_bytes)
                .ok_or_else(|| Error::InvalidRequest("truncated far-field blob".to_string()))
        };
        let version = rd_u32(0)?;
        if version != FAR_FIELD_BLOB_VERSION {
            return Err(Error::InvalidRequest(format!(
                "unsupported far-field blob version {version}"
            )));
        }
        let n_theta = rd_u32(4)? as usize;
        let n_phi = rd_u32(8)? as usize;
        if n_theta == 0 || n_phi == 0 || n_theta > 1024 || n_phi > 4096 {
            return Err(Error::InvalidRequest(
                "far-field blob grid dimensions out of range".to_string(),
            ));
        }
        let mut off = 12usize;
        let mut rd_f64s = |n: usize| -> Result<Vec<f64>> {
            let end = off
                .checked_add(n.checked_mul(8).ok_or_else(|| {
                    Error::InvalidRequest("far-field blob grid too large".to_string())
                })?)
                .ok_or_else(|| Error::InvalidRequest("far-field blob overruns".to_string()))?;
            if end > bytes.len() {
                return Err(Error::InvalidRequest("truncated far-field blob".to_string()));
            }
            let vals: Vec<f64> = bytes[off..end]
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            off = end;
            Ok(vals)
        };
        let scalars = rd_f64s(7)?;
        let theta_deg = rd_f64s(n_theta)?;
        let phi_deg = rd_f64s(n_phi)?;
        let gain_dbi = rd_f64s(n_theta * n_phi)?;
        let directivity_dbi = rd_f64s(n_theta * n_phi)?;
        Ok(Self {
            theta_deg,
            phi_deg,
            gain_dbi,
            directivity_dbi,
            max_gain_dbi: scalars[0],
            max_gain_theta_deg: scalars[1],
            max_gain_phi_deg: scalars[2],
            max_directivity_dbi: scalars[3],
            radiated_power_w: scalars[4],
            accepted_power_w: scalars[5],
            radiation_efficiency: scalars[6],
        })
    }
}

/// A solved frequency point, ready for [`PointResponse::serialize`].
#[derive(Debug, Clone)]
pub struct PointResponse {    pub status: u32,
    pub residual_norm: Option<f64>,
    pub iteration_count: Option<u32>,
    pub zin: Option<Complex64>,
    pub s11: Option<Complex64>,
    pub timing: TimingMs,
    /// Carries a serialized [`FarFieldBlob`] when far-field was computed,
    /// empty otherwise.
    pub field_blob: Vec<u8>,
    pub message: Option<String>,
}

impl PointResponse {
    /// Build an error response: all numerics NaN, message attached.
    pub fn error(error: &Error) -> Self {
        Self {
            status: error.status_code(),
            residual_norm: None,
            iteration_count: None,
            zin: None,
            s11: None,
            timing: TimingMs::default(),
            field_blob: Vec::new(),
            message: Some(error.message()),
        }
    }

    /// Serialize to the v1 response wire format (header + blob + message).
    pub fn serialize(&self) -> Vec<u8> {
        let message_bytes = self.message.as_deref().unwrap_or("").as_bytes();
        let total = RESPONSE_HEADER_BYTES + self.field_blob.len() + message_bytes.len();
        let mut out = vec![0u8; total];

        let put_u32 = |out: &mut [u8], off: usize, v: u32| {
            out[off..off + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put_f64 = |out: &mut [u8], off: usize, v: f64| {
            out[off..off + 8].copy_from_slice(&v.to_le_bytes());
        };
        let nan = f64::NAN;

        put_u32(&mut out, 0, ABI_VERSION);
        put_u32(&mut out, 4, self.status);
        put_f64(&mut out, 16, self.residual_norm.unwrap_or(nan));
        put_u32(&mut out, 24, self.iteration_count.unwrap_or(0));
        put_f64(&mut out, 32, self.zin.map(|z| z.re).unwrap_or(nan));
        put_f64(&mut out, 40, self.zin.map(|z| z.im).unwrap_or(nan));
        put_f64(&mut out, 48, self.s11.map(|z| z.re).unwrap_or(nan));
        put_f64(&mut out, 56, self.s11.map(|z| z.im).unwrap_or(nan));
        put_f64(&mut out, 64, self.timing.assembly);
        put_f64(&mut out, 72, self.timing.solve);
        put_f64(&mut out, 80, self.timing.post);
        put_u32(&mut out, 88, RESPONSE_HEADER_BYTES as u32);
        put_u32(&mut out, 92, self.field_blob.len() as u32);
        put_u32(
            &mut out,
            96,
            (RESPONSE_HEADER_BYTES + self.field_blob.len()) as u32,
        );
        put_u32(&mut out, 100, message_bytes.len() as u32);

        out[RESPONSE_HEADER_BYTES..RESPONSE_HEADER_BYTES + self.field_blob.len()]
            .copy_from_slice(&self.field_blob);
        out[RESPONSE_HEADER_BYTES + self.field_blob.len()..].copy_from_slice(message_bytes);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal well-formed request: 1 vertex triple is not a mesh, but the
    /// parser only validates structure — enough to exercise the header path.
    fn minimal_request_bytes() -> Vec<u8> {
        let mut out = vec![0u8; REQUEST_HEADER_BYTES];
        let put_u32 = |out: &mut [u8], off: usize, v: u32| {
            out[off..off + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put_f64 = |out: &mut [u8], off: usize, v: f64| {
            out[off..off + 8].copy_from_slice(&v.to_le_bytes());
        };
        // Regions, in pack order: vertices, tets, tet_region, bfacets, battr,
        // port_pos, materials, metadata, port_neg, port_attr.
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        blobs.push(vec![0u8; 24]); // 1 vertex
        let mut tets = vec![0u8; 16];
        for i in 0..4 {
            tets[i * 4..i * 4 + 4].copy_from_slice(&(0u32).to_le_bytes());
        }
        blobs.push(tets);
        blobs.push(1u32.to_le_bytes().to_vec()); // tet_region
        let facet = vec![0u8; 12]; // 1 boundary facet
        blobs.push(facet);
        blobs.push(10u32.to_le_bytes().to_vec()); // absorbing attr
        blobs.push(0u32.to_le_bytes().to_vec()); // port_pos: facet 0
        let mut mat = vec![0u8; 40]; // 1 material
        mat[0..4].copy_from_slice(&1u32.to_le_bytes());
        mat[8..16].copy_from_slice(&1.0f64.to_le_bytes());
        mat[16..24].copy_from_slice(&1.0f64.to_le_bytes());
        blobs.push(mat);
        blobs.push(br#"{"reference_impedance_ohms": 50}"#.to_vec()); // metadata
        blobs.push(0u32.to_le_bytes().to_vec()); // port_neg: facet 0
        // port_attr is pos-then-neg CONTIGUOUS in the real packer; keep the
        // two halves adjacent so the single region read sees both.
        blobs.push(30u32.to_le_bytes().to_vec()); // port_attr pos
        blobs.push(31u32.to_le_bytes().to_vec()); // port_attr neg

        let mut offset = REQUEST_HEADER_BYTES;
        let mut offsets = Vec::new();
        for b in &blobs {
            offsets.push(offset);
            out.extend_from_slice(b);
            offset += b.len();
        }
        // blobs order: 0 verts,1 tets,2 tet_region,3 bfacets,4 battr,5 port_pos,
        // 6 materials,7 metadata,8 port_neg,9 port_attr_pos,10 port_attr_neg
        put_u32(&mut out, 0, ABI_VERSION);
        put_u32(&mut out, 4, ENDIANNESS_SENTINEL);
        put_f64(&mut out, 16, 3.0e9);
        put_u32(&mut out, 24, 1); // vertex_count
        put_u32(&mut out, 28, 1); // tet_count
        put_u32(&mut out, 36, 1); // boundary_facet_count
        put_u32(&mut out, 40, 2); // port_facet_count
        put_u32(&mut out, 44, 1); // port_pos_count
        put_u32(&mut out, 48, offsets[0] as u32);
        put_u32(&mut out, 52, offsets[1] as u32);
        put_u32(&mut out, 56, offsets[2] as u32);
        put_u32(&mut out, 60, offsets[3] as u32);
        put_u32(&mut out, 64, offsets[4] as u32);
        put_u32(&mut out, 68, offsets[5] as u32);
        put_u32(&mut out, 72, offsets[9] as u32);
        put_u32(&mut out, 76, offsets[6] as u32);
        put_u32(&mut out, 80, offsets[7] as u32);
        put_u32(&mut out, 84, blobs[0].len() as u32);
        put_u32(&mut out, 88, blobs[1].len() as u32);
        put_u32(&mut out, 92, blobs[2].len() as u32);
        put_u32(&mut out, 96, blobs[3].len() as u32);
        put_u32(&mut out, 100, blobs[4].len() as u32);
        put_u32(&mut out, 104, blobs[5].len() as u32);
        put_u32(&mut out, 108, blobs[6].len() as u32);
        put_u32(&mut out, 112, offsets[8] as u32);
        put_u32(&mut out, 116, blobs[8].len() as u32);
        put_u32(&mut out, 120, (blobs[9].len() + blobs[10].len()) as u32);
        put_u32(&mut out, 124, blobs[7].len() as u32);
        out
    }

    #[test]
    fn parses_minimal_well_formed_request() {
        let req = PointRequest::parse(&minimal_request_bytes()).expect("parse");
        assert!((req.frequency_hz - 3.0e9).abs() < 1e-6);
        assert_eq!(req.vertices.len(), 1);
        assert_eq!(req.tets.len(), 1);
        assert_eq!(req.tet_region, vec![1]);
        assert_eq!(req.boundary_facets.len(), 1);
        assert_eq!(req.boundary_attr, vec![ATTR_EXTERIOR_ABSORBING]);
        assert_eq!(req.port_pos_facets, vec![0]);
        assert_eq!(req.port_neg_facets, vec![0]);
        assert_eq!(req.port_attr, vec![ATTR_PORT_POSITIVE, ATTR_PORT_NEGATIVE]);
        assert_eq!(req.materials.len(), 1);
        assert_eq!(req.materials[0].attribute, 1);
        assert!((req.materials[0].epsilon_r - 1.0).abs() < 1e-12);
        assert!((req.metadata.reference_impedance_ohms() - 50.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_abi_version() {
        let mut bytes = minimal_request_bytes();
        bytes[0..4].copy_from_slice(&99u32.to_le_bytes());
        let err = PointRequest::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            Error::UnsupportedAbi {
                expected: 1,
                found: 99
            }
        );
    }

    #[test]
    fn response_round_trip_layout() {
        let resp = PointResponse {
            status: 0,
            residual_norm: Some(1e-8),
            iteration_count: Some(1117),
            zin: Some(Complex64::new(66.6375, -15.3179)),
            s11: Some(Complex64::new(0.157179, -0.110687)),
            timing: TimingMs {
                assembly: 10.0,
                solve: 20.0,
                post: 30.0,
            },
            field_blob: vec![9, 9, 9],
            message: Some("ok".to_string()),
        };
        let bytes = resp.serialize();
        assert_eq!(bytes.len(), RESPONSE_HEADER_BYTES + 3 + 2);
        let u32_at = |off: usize| {
            u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
        };
        let f64_at = |off: usize| {
            f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
        };
        assert_eq!(u32_at(0), 1);
        assert_eq!(u32_at(4), 0);
        assert_eq!(u32_at(24), 1117);
        assert!((f64_at(32) - 66.6375).abs() < 1e-12);
        assert!((f64_at(40) + 15.3179).abs() < 1e-12);
        assert_eq!(u32_at(88), RESPONSE_HEADER_BYTES as u32);
        assert_eq!(u32_at(92), 3);
        assert_eq!(u32_at(96), (RESPONSE_HEADER_BYTES + 3) as u32);
        assert_eq!(u32_at(100), 2);
        assert_eq!(&bytes[RESPONSE_HEADER_BYTES..RESPONSE_HEADER_BYTES + 3], &[9, 9, 9]);
        assert_eq!(&bytes[RESPONSE_HEADER_BYTES + 3..], b"ok");
    }

    #[test]
    fn error_response_uses_nan_and_message() {
        let resp = PointResponse::error(&Error::Cancelled);
        assert_eq!(resp.status, 7);
        let bytes = resp.serialize();
        let zin_re = f64::from_le_bytes(bytes[32..40].try_into().unwrap());
        assert!(zin_re.is_nan());
        assert_eq!(resp.message.as_deref(), Some("cancelled"));
    }

    #[test]
    fn far_field_blob_round_trip() {
        let blob = FarFieldBlob {
            theta_deg: vec![0.0, 90.0, 180.0],
            phi_deg: vec![0.0, 180.0],
            gain_dbi: vec![-50.0, -50.0, 2.45, 2.44, -50.0, -50.0],
            directivity_dbi: vec![-49.0, -49.0, 2.5, 2.49, -49.0, -49.0],
            max_gain_dbi: 2.45,
            max_gain_theta_deg: 90.0,
            max_gain_phi_deg: 0.0,
            max_directivity_dbi: 2.5,
            radiated_power_w: 0.97,
            accepted_power_w: 1.0,
            radiation_efficiency: 0.97,
        };
        let bytes = blob.serialize();
        assert_eq!(bytes.len(), 68 + 8 * (3 + 2 + 2 * 6));
        let back = FarFieldBlob::parse(&bytes).unwrap();
        assert_eq!(back.theta_deg, blob.theta_deg);
        assert_eq!(back.phi_deg, blob.phi_deg);
        assert_eq!(back.gain_dbi, blob.gain_dbi);
        assert_eq!(back.directivity_dbi, blob.directivity_dbi);
        assert_eq!(back.max_gain_dbi, 2.45);
        assert_eq!(back.max_gain_phi_deg, 0.0);
        assert_eq!(back.radiation_efficiency, 0.97);
        // Bad version and truncation are rejected.
        let mut bad = bytes.clone();
        bad[0] = 2;
        assert!(FarFieldBlob::parse(&bad).is_err());
        assert!(FarFieldBlob::parse(&bytes[..100]).is_err());
    }
}
