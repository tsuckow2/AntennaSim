//! wgpu backend: a GPU mirror of [`crate::solver::Fdtd`].
//!
//! Every kernel is a direct transcription of the CPU step sequence from
//! `Fdtd::step`:
//!
//! ```text
//! update_h → CPML-H (faces xlo,xhi,ylo,yhi,zlo,zhi) → wire current
//! → update_e → CPML-E (same face order) → wire deposit → PEC
//! → resistive source + V/I probe
//! ```
//!
//! One explicit bind group carries all buffers; dispatches run in order
//! inside a single compute pass, so face ordering and the leapfrog
//! sequence match the CPU code exactly. CPML faces keep the CPU's
//! xlo,xhi,ylo,yhi,zlo,zhi dispatch order (corner cells are corrected by
//! several faces sequentially — this is order-dependent).
//!
//! Initial state uploads from a CPU [`Fdtd`](crate::solver::Fdtd) (identical
//! initial conditions); the V/I time series downloads at the end and is
//! DFT'd on the CPU with the crate's own [`DftProbe`](crate::source::DftProbe).
//! Parity strategy: per-cell updates are independent, so the only
//! divergence sources vs the CPU are f32 rounding (GPU FMA contraction,
//! `exp`). Field-level agreement on a short run is the correctness gate.

use bytemuck::{Pod, Zeroable};

use crate::cpml::PmlFaceData;
use crate::solver::Fdtd;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

/// Static run parameters. 32 words, all f32/u32 words laid out so the WGSL
/// `GpuParams` struct matches field-for-field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    nx: u32,
    ny: u32,
    nz: u32,
    _pad0: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
    ch: f32, // dt / mu0 (H update + CPML-H)
    idx: f32,
    idy: f32,
    idz: f32,
    rs: f32,
    src_ez_idx: u32,
    has_wire: u32,
    _pad1: u32,
    // Tight Ampère loop indices (no-wire fallback): hx_bot, hx_top, hy_lft, hy_rgt.
    tl_hx_bot: u32,
    tl_hx_top: u32,
    tl_hy_lft: u32,
    tl_hy_rgt: u32,
    // Wire layout.
    wire_ez_base: u32,
    wire_ez_stride: u32, // ez index stride per k (= nx*ny)
    wire_nseg: u32,
    wire_nnodes: u32,
    wire_top_pec: u32,
    wire_dt_over_l: f32,
    wire_cap: f32,
    wire_deposit: f32, // legacy: was dt/(eps0*dx*dy); deposit now uses cb_z per cell (see wire_deposit WGSL)
    // Source waveform.
    src_v0: f32,
    src_t0: f32,
    src_tau: f32,
    wire_feed: u32, // 1 = Thévenin source attached at the wire base node
}

/// Per-face CPML descriptor, 2 × vec4u. Passed via a dynamic-offset uniform:
/// slot `f` (256-byte aligned) = face `f`.
/// word0 = (axis, lo, npml, n1), word1 = (n2, psi_base, coeff_base, 0).
/// psi_base/coeff_base are f32-word offsets into the psi / coeff buffers.
const FACE_SLOT_BYTES: u64 = 256;

// ---------------------------------------------------------------------------
// WGSL: H update (transcription of solver::update_h)
// ---------------------------------------------------------------------------

const SHADER_H: &str = r#"
struct Params {
    nx: u32, ny: u32, nz: u32, _pad0: u32,
    dx: f32, dy: f32, dz: f32, dt: f32,
    ch: f32, idx: f32, idy: f32, idz: f32,
    rs: f32, src_ez_idx: u32, has_wire: u32, _pad1: u32,
    tl_hx_bot: u32, tl_hx_top: u32, tl_hy_lft: u32, tl_hy_rgt: u32,
    wire_ez_base: u32, wire_ez_stride: u32, wire_nseg: u32, wire_nnodes: u32,
    wire_top_pec: u32, wire_dt_over_l: f32, wire_cap: f32, wire_deposit: f32,
    src_v0: f32, src_t0: f32, src_tau: f32, _pad2: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> ex: array<f32>;
@group(0) @binding(3) var<storage, read_write> ey: array<f32>;
@group(0) @binding(4) var<storage, read_write> ez: array<f32>;
@group(0) @binding(5) var<storage, read_write> hx: array<f32>;
@group(0) @binding(6) var<storage, read_write> hy: array<f32>;
@group(0) @binding(7) var<storage, read_write> hz: array<f32>;

fn ex_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }
fn ey_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }
fn ez_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * p.nx + i; }
fn hx_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }
fn hy_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }
fn hz_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * (p.nx + 1u) + i; }

// Hx: nx x (ny+1) x (nz+1).  Hx -= ch*((Ey(k+1)-Ey(k))/dz - (Ez(j+1)-Ez(j))/dy)
@compute @workgroup_size(8, 8, 4)
fn update_hx(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let j = gid.y; let k = gid.z;
    if (i >= p.nx || j > p.ny || k > p.nz) { return; }
    let ey_k = select(0.0, ey[ey_idx(i, j, k)], k < p.nz);
    let ey_kp = select(0.0, ey[ey_idx(i, j, k + 1u)], k + 1u < p.nz);
    let ez_j = select(0.0, ez[ez_idx(i, j, k)], j < p.ny);
    let ez_jp = select(0.0, ez[ez_idx(i, j + 1u, k)], j + 1u < p.ny);
    let h = hx_idx(i, j, k);
    hx[h] -= p.ch * ((ey_kp - ey_k) * p.idz - (ez_jp - ez_j) * p.idy);
}

// Hy: (nx+1) x ny x (nz+1).  Hy -= ch*((Ez(i+1)-Ez(i))/dx - (Ex(k+1)-Ex(k))/dz)
@compute @workgroup_size(8, 8, 4)
fn update_hy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let j = gid.y; let k = gid.z;
    if (i > p.nx || j >= p.ny || k > p.nz) { return; }
    let im = min(i, p.nx - 1u);
    let ez_i = select(0.0, ez[ez_idx(im, j, k)], i < p.nx);
    let ez_ip = select(0.0, ez[ez_idx(i + 1u, j, k)], i + 1u < p.nx);
    let ie = min(i, p.nx);
    let ex_k = select(0.0, ex[ex_idx(ie, j, k)], k < p.nz);
    let ex_kp = select(0.0, ex[ex_idx(ie, j, k + 1u)], k + 1u < p.nz);
    let h = hy_idx(i, j, k);
    hy[h] -= p.ch * ((ez_ip - ez_i) * p.idx - (ex_kp - ex_k) * p.idz);
}

// Hz: (nx+1) x (ny+1) x nz.  Hz -= ch*((Ex(j+1)-Ex(j))/dy - (Ey(i+1)-Ey(i))/dx)
@compute @workgroup_size(8, 8, 4)
fn update_hz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let j = gid.y; let k = gid.z;
    if (i > p.nx || j > p.ny || k >= p.nz) { return; }
    let ie = min(i, p.nx);
    let jm = min(j, p.ny - 1u);
    let ex_j = ex[ex_idx(ie, jm, k)];
    let ex_jp = select(0.0, ex[ex_idx(ie, j + 1u, k)], j + 1u < p.ny);
    let ie2 = min(i, p.nx - 1u);
    let ey_i = ey[ey_idx(ie2, j, k)];
    let ey_ip = select(0.0, ey[ey_idx(i + 1u, j, k)], i + 1u < p.nx);
    let h = hz_idx(i, j, k);
    hz[h] -= p.ch * ((ex_jp - ex_j) * p.idy - (ey_ip - ey_i) * p.idx);
}
"#;

// ---------------------------------------------------------------------------
// WGSL: E update (transcription of solver::update_e)
// ---------------------------------------------------------------------------

const SHADER_E: &str = r#"
struct Params {
    nx: u32, ny: u32, nz: u32, _pad0: u32,
    dx: f32, dy: f32, dz: f32, dt: f32,
    ch: f32, idx: f32, idy: f32, idz: f32,
    rs: f32, src_ez_idx: u32, has_wire: u32, _pad1: u32,
    tl_hx_bot: u32, tl_hx_top: u32, tl_hy_lft: u32, tl_hy_rgt: u32,
    wire_ez_base: u32, wire_ez_stride: u32, wire_nseg: u32, wire_nnodes: u32,
    wire_top_pec: u32, wire_dt_over_l: f32, wire_cap: f32, wire_deposit: f32,
    src_v0: f32, src_t0: f32, src_tau: f32, _pad2: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> ex: array<f32>;
@group(0) @binding(3) var<storage, read_write> ey: array<f32>;
@group(0) @binding(4) var<storage, read_write> ez: array<f32>;
@group(0) @binding(5) var<storage, read_write> hx: array<f32>;
@group(0) @binding(6) var<storage, read_write> hy: array<f32>;
@group(0) @binding(7) var<storage, read_write> hz: array<f32>;
@group(0) @binding(8) var<storage, read> ca_x: array<f32>;
@group(0) @binding(9) var<storage, read> ca_y: array<f32>;
@group(0) @binding(10) var<storage, read> ca_z: array<f32>;
@group(0) @binding(11) var<storage, read> cb_x: array<f32>;
@group(0) @binding(12) var<storage, read> cb_y: array<f32>;
@group(0) @binding(13) var<storage, read> cb_z: array<f32>;

fn ex_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }
fn ey_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }
fn ez_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * p.nx + i; }
fn hx_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }
fn hy_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }
fn hz_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * (p.nx + 1u) + i; }

// Ex = Ca*Ex + Cb*((Hy(k)-Hy(k-1))/dz - (Hz(j)-Hz(j-1))/dy)
@compute @workgroup_size(8, 8, 4)
fn update_ex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let j = gid.y; let k = gid.z;
    if (i > p.nx || j >= p.ny || k >= p.nz) { return; }
    let h = hz_idx(i, j, k);
    let hz_j = hz[h];
    let hz_jm = select(0.0, hz[hz_idx(i, j - 1u, k)], j > 0u);
    let hy_k = hy[hy_idx(i, j, k)];
    let hy_km = select(0.0, hy[hy_idx(i, j, k - 1u)], k > 0u);
    let e = ex_idx(i, j, k);
    ex[e] = ca_x[e] * ex[e] + cb_x[e] * ((hy_k - hy_km) * p.idz - (hz_j - hz_jm) * p.idy);
}

// Ey = Ca*Ey + Cb*((Hz(i)-Hz(i-1))/dx - (Hx(k)-Hx(k-1))/dz)
@compute @workgroup_size(8, 8, 4)
fn update_ey(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let j = gid.y; let k = gid.z;
    if (i >= p.nx || j > p.ny || k >= p.nz) { return; }
    let hx_k = hx[hx_idx(i, j, k)];
    let hx_km = select(0.0, hx[hx_idx(i, j, k - 1u)], k > 0u);
    let hz_i = hz[hz_idx(i, j, k)];
    let hz_im = select(0.0, hz[hz_idx(i - 1u, j, k)], i > 0u);
    let e = ey_idx(i, j, k);
    ey[e] = ca_y[e] * ey[e] + cb_y[e] * ((hz_i - hz_im) * p.idx - (hx_k - hx_km) * p.idz);
}

// Ez = Ca*Ez + Cb*((Hx(j)-Hx(j-1))/dy - (Hy(i)-Hy(i-1))/dx)
@compute @workgroup_size(8, 8, 4)
fn update_ez(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let j = gid.y; let k = gid.z;
    if (i >= p.nx || j >= p.ny || k > p.nz) { return; }
    let hy_i = hy[hy_idx(i, j, k)];
    let hy_im = select(0.0, hy[hy_idx(i - 1u, j, k)], i > 0u);
    let hx_j = hx[hx_idx(i, j, k)];
    let hx_jm = select(0.0, hx[hx_idx(i, j - 1u, k)], j > 0u);
    let e = ez_idx(i, j, k);
    ez[e] = ca_z[e] * ez[e] + cb_z[e] * ((hx_j - hx_jm) * p.idy - (hy_i - hy_im) * p.idx);
}
"#;

// ---------------------------------------------------------------------------
// WGSL: CPML corrections (transcription of cpml.rs correct_e_* / correct_h_*)
// Coefficient buffer per face: [b_e | a_e | b_h | a_h | kinv], each npml long.
// Psi buffer per face: [psi_a | psi_b | psi_c | psi_d], each npml*n1*n2.
// ---------------------------------------------------------------------------

const SHADER_CPML: &str = r#"
struct Params {
    nx: u32, ny: u32, nz: u32, _pad0: u32,
    dx: f32, dy: f32, dz: f32, dt: f32,
    ch: f32, idx: f32, idy: f32, idz: f32,
    rs: f32, src_ez_idx: u32, has_wire: u32, _pad1: u32,
    tl_hx_bot: u32, tl_hx_top: u32, tl_hy_lft: u32, tl_hy_rgt: u32,
    wire_ez_base: u32, wire_ez_stride: u32, wire_nseg: u32, wire_nnodes: u32,
    wire_top_pec: u32, wire_dt_over_l: f32, wire_cap: f32, wire_deposit: f32,
    src_v0: f32, src_t0: f32, src_tau: f32, _pad2: u32,
};
struct FaceU { d0: vec4u, d1: vec4u };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<uniform> face: FaceU;
@group(0) @binding(2) var<storage, read_write> ex: array<f32>;
@group(0) @binding(3) var<storage, read_write> ey: array<f32>;
@group(0) @binding(4) var<storage, read_write> ez: array<f32>;
@group(0) @binding(5) var<storage, read_write> hx: array<f32>;
@group(0) @binding(6) var<storage, read_write> hy: array<f32>;
@group(0) @binding(7) var<storage, read_write> hz: array<f32>;
@group(0) @binding(11) var<storage, read> cb_x: array<f32>;
@group(0) @binding(12) var<storage, read> cb_y: array<f32>;
@group(0) @binding(13) var<storage, read> cb_z: array<f32>;
@group(0) @binding(17) var<storage, read> coeff: array<f32>;
@group(0) @binding(18) var<storage, read_write> psi: array<f32>;

fn ex_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }
fn ey_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }
fn ez_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * p.nx + i; }
fn hx_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }
fn hy_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }
fn hz_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * (p.nx + 1u) + i; }

fn lo() -> u32 { return face.d0.y; }
fn npml() -> u32 { return face.d0.z; }
fn n1() -> u32 { return face.d0.w; }
fn n2() -> u32 { return face.d1.x; }
fn psi_base() -> u32 { return face.d1.y; }
fn coeff_base() -> u32 { return face.d1.z; }
fn be(pc: u32) -> f32 { return coeff[coeff_base() + pc]; }
fn ae(pc: u32) -> f32 { return coeff[coeff_base() + npml() + pc]; }
fn bh(pc: u32) -> f32 { return coeff[coeff_base() + 2u * npml() + pc]; }
fn ah(pc: u32) -> f32 { return coeff[coeff_base() + 3u * npml() + pc]; }
fn ki(pc: u32) -> f32 { return coeff[coeff_base() + 4u * npml() + pc]; }
fn psi_a(pi: u32) -> u32 { return psi_base() + pi; }
fn psi_b(pi: u32) -> u32 { return psi_base() + npml() * n1() * n2() + pi; }
fn psi_c(pi: u32) -> u32 { return psi_base() + 2u * npml() * n1() * n2() + pi; }
fn psi_d(pi: u32) -> u32 { return psi_base() + 3u * npml() * n1() * n2() + pi; }

// ---- E corrections ----

// x-normal: Ey via dHz/dx, Ez via dHy/dx.  dispatch (k, j, p).
@compute @workgroup_size(8, 8, 4)
fn cpml_e_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pc = gid.z; let j = gid.y; let k = gid.x;
    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }
    let ie = select(p.nx - 1u - pc, pc, lo() == 1u);
    let pi = (pc * n1() + j) * n2() + k;
    let b = be(pc); let a = ae(pc); let kv = ki(pc); let omk = 1.0 - kv;
    let hz_lo = select(0.0, hz[hz_idx(ie - 1u, j, k)], ie > 0u);
    let dhzdx = (hz[hz_idx(ie, j, k)] - hz_lo) / p.dx;
    let pa = psi_a(pi);
    let ps = b * psi[pa] + a * dhzdx;
    psi[pa] = ps;
    let ei = ey_idx(ie, j, k);
    ey[ei] += cb_y[ei] * (-omk * dhzdx + ps);
    let hy_lo = select(0.0, hy[hy_idx(ie - 1u, j, k)], ie > 0u);
    let dhydx = (hy[hy_idx(ie, j, k)] - hy_lo) / p.dx;
    let pb = psi_b(pi);
    let ps2 = b * psi[pb] + a * dhydx;
    psi[pb] = ps2;
    let ei2 = ez_idx(ie, j, k);
    ez[ei2] += cb_z[ei2] * (omk * dhydx - ps2);
}

// y-normal: Ex via dHz/dy, Ez via dHx/dy.  dispatch (k, i, p).
@compute @workgroup_size(8, 8, 4)
fn cpml_e_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pc = gid.z; let i = gid.y; let k = gid.x;
    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }
    let je = select(p.ny - 1u - pc, pc, lo() == 1u);
    let pi = (pc * n1() + i) * n2() + k;
    let b = be(pc); let a = ae(pc); let kv = ki(pc); let omk = 1.0 - kv;
    let hz_lo = select(0.0, hz[hz_idx(i, je - 1u, k)], je > 0u);
    let dhzdy = (hz[hz_idx(i, je, k)] - hz_lo) / p.dy;
    let pa = psi_a(pi);
    let ps = b * psi[pa] + a * dhzdy;
    psi[pa] = ps;
    let ei = ex_idx(i, je, k);
    ex[ei] += cb_x[ei] * (-omk * dhzdy + ps);
    let hx_lo = select(0.0, hx[hx_idx(i, je - 1u, k)], je > 0u);
    let dhxdy = (hx[hx_idx(i, je, k)] - hx_lo) / p.dy;
    let pb = psi_b(pi);
    let ps2 = b * psi[pb] + a * dhxdy;
    psi[pb] = ps2;
    let ei2 = ez_idx(i, je, k);
    ez[ei2] += cb_z[ei2] * (-omk * dhxdy + ps2);
}

// z-normal: Ex via dHy/dz, Ey via dHx/dz.  dispatch (j, i, p).
@compute @workgroup_size(8, 8, 4)
fn cpml_e_z(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pc = gid.z; let i = gid.y; let j = gid.x;
    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }
    let ke = select(p.nz - 1u - pc, pc, lo() == 1u);
    let pi = (pc * n1() + i) * n2() + j;
    let b = be(pc); let a = ae(pc); let kv = ki(pc); let omk = 1.0 - kv;
    let hy_lo = select(0.0, hy[hy_idx(i, j, ke - 1u)], ke > 0u);
    let dhydz = (hy[hy_idx(i, j, ke)] - hy_lo) / p.dz;
    let pa = psi_a(pi);
    let ps = b * psi[pa] + a * dhydz;
    psi[pa] = ps;
    let ei = ex_idx(i, j, ke);
    ex[ei] += cb_x[ei] * (-omk * dhydz + ps);
    let hx_lo = select(0.0, hx[hx_idx(i, j, ke - 1u)], ke > 0u);
    let dhxdz = (hx[hx_idx(i, j, ke)] - hx_lo) / p.dz;
    let pb = psi_b(pi);
    let ps2 = b * psi[pb] + a * dhxdz;
    psi[pb] = ps2;
    let ei2 = ey_idx(i, j, ke);
    ey[ei2] += cb_y[ei2] * (omk * dhxdz - ps2);
}

// ---- H corrections ----

// x-normal: Hy via dEz/dx, Hz via dEy/dx.  dispatch (k, j, p).
@compute @workgroup_size(8, 8, 4)
fn cpml_h_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pc = gid.z; let j = gid.y; let k = gid.x;
    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }
    let ih = select(p.nx - pc, pc, lo() == 1u);
    let pi = (pc * n1() + j) * n2() + k;
    let b = bh(pc); let a = ah(pc); let kv = ki(pc); let omk = 1.0 - kv;
    let e_hi = select(0.0, ez[ez_idx(ih + 1u, j, k)], ih + 1u < p.nx);
    let e_lo = select(0.0, ez[ez_idx(ih, j, k)], ih < p.nx);
    let dezdx = (e_hi - e_lo) / p.dx;
    let pcc = psi_c(pi);
    let ps = b * psi[pcc] + a * dezdx;
    psi[pcc] = ps;
    let hi = hy_idx(ih, j, k);
    hy[hi] -= p.ch * (-omk * dezdx + ps);
    let e2_hi = select(0.0, ey[ey_idx(ih + 1u, j, k)], ih + 1u < p.nx);
    let e2_lo = ey[ey_idx(min(ih, p.nx - 1u), j, k)];
    let deydx = (e2_hi - e2_lo) / p.dx;
    let pd = psi_d(pi);
    let ps2 = b * psi[pd] + a * deydx;
    psi[pd] = ps2;
    let hi2 = hz_idx(ih, j, k);
    hz[hi2] -= p.ch * (omk * deydx - ps2);
}

// y-normal: Hx via dEz/dy, Hz via dEx/dy.  dispatch (k, i, p).
@compute @workgroup_size(8, 8, 4)
fn cpml_h_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pc = gid.z; let i = gid.y; let k = gid.x;
    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }
    let jh = select(p.ny - pc, pc, lo() == 1u);
    let pi = (pc * n1() + i) * n2() + k;
    let b = bh(pc); let a = ah(pc); let kv = ki(pc); let omk = 1.0 - kv;
    let e_hi = select(0.0, ez[ez_idx(i, jh + 1u, k)], jh + 1u < p.ny);
    let e_lo = select(0.0, ez[ez_idx(i, jh, k)], jh < p.ny);
    let dezdy = (e_hi - e_lo) / p.dy;
    let pcc = psi_c(pi);
    let ps = b * psi[pcc] + a * dezdy;
    psi[pcc] = ps;
    let hi = hx_idx(i, jh, k);
    hx[hi] -= p.ch * (omk * dezdy - ps);
    let e2_hi = select(0.0, ex[ex_idx(i, jh + 1u, k)], jh + 1u < p.ny);
    let e2_lo = ex[ex_idx(i, min(jh, p.ny - 1u), k)];
    let dexdy = (e2_hi - e2_lo) / p.dy;
    let pd = psi_d(pi);
    let ps2 = b * psi[pd] + a * dexdy;
    psi[pd] = ps2;
    let hi2 = hz_idx(i, jh, k);
    hz[hi2] -= p.ch * (-omk * dexdy + ps2);
}

// z-normal: Hx via dEy/dz, Hy via dEx/dz.  dispatch (j, i, p).
@compute @workgroup_size(8, 8, 4)
fn cpml_h_z(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pc = gid.z; let i = gid.y; let j = gid.x;
    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }
    let kh = select(p.nz - pc, pc, lo() == 1u);
    let pi = (pc * n1() + i) * n2() + j;
    let b = bh(pc); let a = ah(pc); let kv = ki(pc); let omk = 1.0 - kv;
    let e_hi = select(0.0, ey[ey_idx(i, j, kh + 1u)], kh + 1u < p.nz);
    let e_lo = select(0.0, ey[ey_idx(i, j, kh)], kh < p.nz);
    let deydz = (e_hi - e_lo) / p.dz;
    let pcc = psi_c(pi);
    let ps = b * psi[pcc] + a * deydz;
    psi[pcc] = ps;
    let hi = hx_idx(i, j, kh);
    hx[hi] -= p.ch * (-omk * deydz + ps);
    let e2_hi = select(0.0, ex[ex_idx(i, j, kh + 1u)], kh + 1u < p.nz);
    let e2_lo = select(0.0, ex[ex_idx(i, j, kh)], kh < p.nz);
    let dexdz = (e2_hi - e2_lo) / p.dz;
    let pd = psi_d(pi);
    let ps2 = b * psi[pd] + a * dexdz;
    psi[pd] = ps2;
    let hi2 = hy_idx(i, j, kh);
    hy[hi2] -= p.ch * (omk * dexdz - ps2);
}
"#;

// ---------------------------------------------------------------------------
// WGSL: wire update/deposit, PEC, resistive source + probe.
// Transcriptions of thin_wire.rs (update_current, deposit_current),
// solver::apply_pec, and source.rs (GaussianSource::value,
// ResistiveSource::apply, Feed::measure_i_tight).
// ---------------------------------------------------------------------------

const SHADER_MISC: &str = r#"
struct Params {
    nx: u32, ny: u32, nz: u32, _pad0: u32,
    dx: f32, dy: f32, dz: f32, dt: f32,
    ch: f32, idx: f32, idy: f32, idz: f32,
    rs: f32, src_ez_idx: u32, has_wire: u32, _pad1: u32,
    tl_hx_bot: u32, tl_hx_top: u32, tl_hy_lft: u32, tl_hy_rgt: u32,
    wire_ez_base: u32, wire_ez_stride: u32, wire_nseg: u32, wire_nnodes: u32,
    wire_top_pec: u32, wire_dt_over_l: f32, wire_cap: f32, wire_deposit: f32,
    src_v0: f32, src_t0: f32, src_tau: f32, wire_feed: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> ex: array<f32>;
@group(0) @binding(3) var<storage, read_write> ey: array<f32>;
@group(0) @binding(4) var<storage, read_write> ez: array<f32>;
@group(0) @binding(5) var<storage, read_write> hx: array<f32>;
@group(0) @binding(6) var<storage, read_write> hy: array<f32>;
@group(0) @binding(14) var<storage, read> pec_x: array<u32>;
@group(0) @binding(15) var<storage, read> pec_y: array<u32>;
@group(0) @binding(16) var<storage, read> pec_z: array<u32>;
@group(0) @binding(13) var<storage, read> cb_z: array<f32>;
// wire state: currents[0..nseg] ++ charges[0..nnodes]
@group(0) @binding(19) var<storage, read_write> wire: array<f32>;
// step state: [0]=step counter, [1]=t (f32 bits), [2..4] pad
@group(0) @binding(20) var<storage, read_write> state: array<u32>;
// probe record: interleaved (v, i, v_field) per step — v_field is the
// gap-cell field voltage -Ez*dz at the wire's base cell, sampled after
// deposit+PEC; it discriminates wire-model node voltage from the
// physical terminal voltage in the recorded Zin.
@group(0) @binding(21) var<storage, read_write> record: array<f32>;

fn node_voltage(n: u32) -> f32 {
    if (p.wire_top_pec == 1u && n == p.wire_nnodes - 1u) { return 0.0; }
    return wire[p.wire_nseg + n] / p.wire_cap;
}

// Holland/Simpson current advance from E^n, then charge continuity.
// Does not touch the grid (deposit is a separate, later step). With a
// Thévenin source attached at the base node (wire_feed=1), node 0's
// charge update is the implicit source solve
// C*(V0^{n+1}-V0^n)/dt = (Vs - V0^{n+1})/Rs - I_0, with Vs evaluated
// at t+dt (state[1] still holds t; source_step advances it later) —
// mirroring ThinWire::update_current. (Rs=0 pins V0 = Vs.)
@compute @workgroup_size(1)
fn wire_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (p.has_wire == 0u) { return; }
    for (var s = 0u; s < p.wire_nseg; s++) {
        let ezi = p.wire_ez_base + s * p.wire_ez_stride;
        let emf = ez[ezi] * p.dz;
        let v_tail = node_voltage(s);
        let v_head = node_voltage(s + 1u);
        wire[s] += p.wire_dt_over_l * (emf + v_tail - v_head);
    }
    for (var n = 0u; n < p.wire_nnodes; n++) {
        if (n == 0u && p.wire_feed == 1u) { continue; }
        let i_out = select(0.0, wire[n], n < p.wire_nseg);
        let i_in = select(0.0, wire[n - 1u], n > 0u);
        var dq = -p.dt * (i_out - i_in);
        if (p.wire_top_pec == 1u && n == p.wire_nnodes - 1u) { dq = 0.0; }
        wire[p.wire_nseg + n] += dq;
    }
    if (p.wire_top_pec == 1u) {
        wire[p.wire_nseg + p.wire_nnodes - 1u] = 0.0;
    }
    if (p.wire_feed == 1u) {
        let t_next = bitcast<f32>(state[1]) + p.dt;
        let x = (t_next - p.src_t0) / p.src_tau;
        let vs = -p.src_v0 * x * exp(-x * x);
        let c = p.wire_cap;
        if (p.rs <= 0.0) {
            wire[p.wire_nseg] = c * vs;
        } else {
            let v0n = wire[p.wire_nseg] / c;
            let v0new = (c * v0n / p.dt + vs / p.rs - wire[0]) / (c / p.dt + 1.0 / p.rs);
            wire[p.wire_nseg] = c * v0new;
        }
    }
}

// Deposit I^{n+1/2} into Ez after the E update, through each segment
// cell's own Ampere coefficient: Ez -= cb_z*I/(dx*dy). (The original
// form used a constant dt/(eps0*dx*dy); cb_z is the consistent
// per-cell coefficient — identical for air cells.)
@compute @workgroup_size(1)
fn wire_deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (p.has_wire == 0u) { return; }
    for (var s = 0u; s < p.wire_nseg; s++) {
        let ezi = p.wire_ez_base + s * p.wire_ez_stride;
        ez[ezi] -= cb_z[ezi] / (p.dx * p.dy) * wire[s];
    }
}

@compute @workgroup_size(256)
fn pec_ex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= arrayLength(&ex)) { return; }
    if (pec_x[n] != 0u) { ex[n] = 0.0; }
}

@compute @workgroup_size(256)
fn pec_ey(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= arrayLength(&ey)) { return; }
    if (pec_y[n] != 0u) { ey[n] = 0.0; }
}

@compute @workgroup_size(256)
fn pec_ez(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= arrayLength(&ez)) { return; }
    if (pec_z[n] != 0u) { ez[n] = 0.0; }
}

// Terminal recording + the lumped-port source. Mirrors Fdtd::step's
// tail: t += dt; Vs = gaussian(t). The wire path records the base
// node's terminal pair (V_0 after its update in wire_update,
// currents[0]) — the Thévenin source lives on the node and the gap
// cell's Ez is NEVER overwritten (overwriting it erased the bottom
// segment's deposit every step and pumped the wire's top mode; see
// ThinWire::update_current). The lumped-port path uses the Norton
// impressed-current form (see ResistiveSource::apply_norton) and
// records (V_term, i_src) — the delivered current, not the Ampere
// loop (the loop around the impressed-source cell anti-correlates
// with it; see solver.rs).
@compute @workgroup_size(1)
fn source_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let step = state[0];
    var t = bitcast<f32>(state[1]);
    t += p.dt;
    let x = (t - p.src_t0) / p.src_tau;
    let vs = -p.src_v0 * x * exp(-x * x);
    var i_wire: f32;
    var v_term: f32;
    var v_field: f32;
    if (p.has_wire == 1u) {
        i_wire = wire[0];
        v_term = wire[p.wire_nseg] / p.wire_cap;
        v_field = -ez[p.wire_ez_base] * p.dz;
    } else {
        i_wire = (hx[p.tl_hx_bot] - hx[p.tl_hx_top]) * p.dx
               + (hy[p.tl_hy_rgt] - hy[p.tl_hy_lft]) * p.dy;
        let si = p.src_ez_idx;
        let v_now = -ez[si] * p.dz;
        let i_src = (vs - v_now) / p.rs;
        let eps_src = p.dt / cb_z[si];
        ez[si] -= p.dt * i_src / (eps_src * p.dx * p.dy);
        v_term = -ez[si] * p.dz;
        i_wire = i_src; // record the delivered current (see header)
        v_field = v_term;
    }
    record[3u * step] = v_term;
    record[3u * step + 1u] = i_wire;
    record[3u * step + 2u] = v_field;
    state[0] = step + 1u;
    state[1] = bitcast<u32>(t);
}
"#;

// ---------------------------------------------------------------------------
// Host implementation
// ---------------------------------------------------------------------------

struct FaceDispatch {
    axis: u8,
    /// Workgroup counts for the (n2, n1, npml) thread grid, (8,8,4) groups.
    disp: (u32, u32, u32),
}

/// Workgroup counts for a 3D thread grid with (8,8,4) workgroups.
fn wg3(x: u32, y: u32, z: u32) -> (u32, u32, u32) {
    (x.div_ceil(8), y.div_ceil(8), z.div_ceil(4))
}

pub struct GpuFdtd {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    bg: wgpu::BindGroup,
    pipe_hx: wgpu::ComputePipeline,
    pipe_hy: wgpu::ComputePipeline,
    pipe_hz: wgpu::ComputePipeline,
    pipe_ex: wgpu::ComputePipeline,
    pipe_ey: wgpu::ComputePipeline,
    pipe_ez: wgpu::ComputePipeline,
    pipe_ce: [wgpu::ComputePipeline; 3], // by axis
    pipe_ch: [wgpu::ComputePipeline; 3],
    pipe_wire_update: wgpu::ComputePipeline,
    pipe_wire_deposit: wgpu::ComputePipeline,
    pipe_pec: [wgpu::ComputePipeline; 3],
    pipe_source: wgpu::ComputePipeline,
    faces: Vec<FaceDispatch>,
    /// Workgroup counts per component kernel (8,8,4) groups.
    dims_e: [(u32, u32, u32); 3],
    dims_h: [(u32, u32, u32); 3],
    e_sizes: [u32; 3], // ex, ey, ez element counts
    h_sizes: [u32; 3], // hx, hy, hz element counts
    field_bufs: [wgpu::Buffer; 6],
    record_buf: wgpu::Buffer,
    wire_buf: wgpu::Buffer,
    wire_nseg: usize,
    wire_nnodes: usize,
    total_steps: usize,
    step: usize,
}

const EPS0: f32 = 8.854_187_812_8e-12;

fn wg(n: u32) -> u32 {
    n.div_ceil(256)
}

fn bentry<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

impl GpuFdtd {
    /// Create a GPU solver mirroring `sim`'s initial state.
    /// `total_steps` sizes the on-device V/I record buffer.
    /// Returns `None` if no GPU adapter is available.
    pub async fn try_new(sim: &Fdtd, total_steps: usize) -> Option<GpuFdtd> {
        let g = &sim.grid;
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let dt = sim.dt;
        let mu0 = 4.0e-7 * std::f32::consts::PI;

        // ---- adapter / device ----
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("fdtd-gpu"),
                    required_features: wgpu::Features::empty(),
                    // Adapter limits, not wgpu defaults: the single bind
                    // group carries ~20 storage buffers (wgpu's default
                    // limit is 8/stage). Native Vulkan (ANV) allows far
                    // more. A future browser build will need the layout
                    // split across bind groups to fit WebGPU's floor.
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;
        // Surface driver/device errors loudly: a silently dropped
        // submission otherwise looks like plausible-but-zero physics.
        device.on_uncaptured_error(Box::new(|e| {
            eprintln!("WGPU UNCAPTURED ERROR: {e:?}");
        }));

        // ---- params ----
        let feed = &sim.feed;
        let (v0, t0, tau) = sim.source.params();
        let mut params = GpuParams {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            _pad0: 0,
            dx: g.dx,
            dy: g.dy,
            dz: g.dz,
            dt,
            ch: dt / mu0,
            idx: 1.0 / g.dx,
            idy: 1.0 / g.dy,
            idz: 1.0 / g.dz,
            rs: sim.rsource.rs,
            src_ez_idx: g.ez_idx(feed.src_i, feed.src_j, feed.src_k) as u32,
            has_wire: 0,
            _pad1: 0,
            // Tight-loop faces of the source Ez cell (update_e pairing):
            // Hx(sj−1)/Hx(sj), Hy(si−1)/Hy(si). Feeds are interior
            // (si, sj ≥ 1) in every fixture; the CPU loop guards edges.
            tl_hx_bot: g.hx_idx(feed.src_i, feed.src_j - 1, feed.src_k) as u32,
            tl_hx_top: g.hx_idx(feed.src_i, feed.src_j, feed.src_k) as u32,
            tl_hy_lft: g.hy_idx(feed.src_i - 1, feed.src_j, feed.src_k) as u32,
            tl_hy_rgt: g.hy_idx(feed.src_i, feed.src_j, feed.src_k) as u32,
            wire_ez_base: 0,
            wire_ez_stride: (nx * ny) as u32,
            wire_nseg: 0,
            wire_nnodes: 0,
            wire_top_pec: 0,
            wire_dt_over_l: 0.0,
            wire_cap: 1.0,
            wire_deposit: 0.0,
            src_v0: v0,
            src_t0: t0,
            src_tau: tau,
            wire_feed: 0,
        };
        let mut wire_init: Vec<f32> = vec![0.0; 4];
        let (mut wire_nseg, mut wire_nnodes) = (0usize, 0usize);
        if let Some(tw) = &sim.thin_wire {
            params.has_wire = 1;
            params.wire_feed = tw.feed_rs.is_some() as u32;
            params.wire_ez_base = g.ez_idx(tw.i0, tw.j0, tw.k_lo) as u32;
            params.wire_nseg = tw.n_segments as u32;
            params.wire_nnodes = tw.n_nodes as u32;
            params.wire_top_pec = tw.top_pec as u32;
            params.wire_dt_over_l = dt / tw.seg_inductance;
            params.wire_cap = tw.node_capacitance;
            params.wire_deposit = dt / (EPS0 * g.dx * g.dy);
            wire_init = tw
                .currents
                .iter()
                .chain(tw.charges.iter())
                .copied()
                .collect();
            wire_nseg = tw.n_segments;
            wire_nnodes = tw.n_nodes;
        }

        // ---- CPML packing ----
        let faces_data: Vec<PmlFaceData> = sim.cpml.snapshot();
        let mut coeff: Vec<f32> = Vec::new();
        let mut psi: Vec<f32> = Vec::new();
        let mut face_slots: Vec<u8> = vec![0u8; (FACE_SLOT_BYTES as usize) * faces_data.len()];
        let mut faces: Vec<FaceDispatch> = Vec::new();
        for (fi, f) in faces_data.iter().enumerate() {
            let coeff_base = coeff.len() as u32;
            coeff.extend_from_slice(&f.b_e);
            coeff.extend_from_slice(&f.a_e);
            coeff.extend_from_slice(&f.b_h);
            coeff.extend_from_slice(&f.a_h);
            coeff.extend_from_slice(&f.kinv);
            let psi_base = psi.len() as u32;
            psi.extend_from_slice(&f.psi_a);
            psi.extend_from_slice(&f.psi_b);
            psi.extend_from_slice(&f.psi_c);
            psi.extend_from_slice(&f.psi_d);
            let words: [u32; 8] = [
                f.axis as u32,
                f.lo as u32,
                f.npml as u32,
                f.n1 as u32,
                f.n2 as u32,
                psi_base,
                coeff_base,
                0,
            ];
            let slot = &mut face_slots[fi * FACE_SLOT_BYTES as usize..][..32];
            slot.copy_from_slice(bytemuck::cast_slice(&words));
            faces.push(FaceDispatch {
                axis: f.axis,
                disp: wg3(f.n2 as u32, f.n1 as u32, f.npml as u32),
            });
        }

        // ---- buffers ----
        use wgpu::util::DeviceExt;
        let rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let field = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: rw,
            })
        };
        let matb = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: ro,
            })
        };
        let pec32 = |v: &[u8]| -> Vec<u32> { v.iter().map(|&x| x as u32).collect() };

        let field_bufs = [
            field("ex", &g.ex),
            field("ey", &g.ey),
            field("ez", &g.ez),
            field("hx", &g.hx),
            field("hy", &g.hy),
            field("hz", &g.hz),
        ];
        let ca_x = matb("ca_x", &sim.mat.ca_x);
        let ca_y = matb("ca_y", &sim.mat.ca_y);
        let ca_z = matb("ca_z", &sim.mat.ca_z);
        let cb_x = matb("cb_x", &sim.mat.cb_x);
        let cb_y = matb("cb_y", &sim.mat.cb_y);
        let cb_z = matb("cb_z", &sim.mat.cb_z);
        let pec_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pec_x"),
            contents: bytemuck::cast_slice(&pec32(&sim.mat.pec_x)),
            usage: ro,
        });
        let pec_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pec_y"),
            contents: bytemuck::cast_slice(&pec32(&sim.mat.pec_y)),
            usage: ro,
        });
        let pec_z = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pec_z"),
            contents: bytemuck::cast_slice(&pec32(&sim.mat.pec_z)),
            usage: ro,
        });
        let coeff_buf = matb("cpml_coeff", &coeff);
        let psi_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cpml_psi"),
            contents: bytemuck::cast_slice(&psi),
            usage: rw,
        });
        let wire_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wire"),
            contents: bytemuck::cast_slice(&wire_init),
            usage: rw,
        });
        let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("state"),
            contents: bytemuck::cast_slice(&[0u32, 0u32, 0u32, 0u32]),
            usage: rw,
        });
        let record_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("record"),
            contents: bytemuck::cast_slice(&vec![0.0f32; 3 * total_steps.max(1)]),
            usage: rw,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let face_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("face_uniforms"),
            contents: &face_slots,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ---- bind group (one explicit layout shared by all pipelines) ----
        let uni = |dynamic: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        };
        let stor = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let vis = wgpu::ShaderStages::COMPUTE;
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty,
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fdtd-bgl"),
            entries: &[
                entry(0, uni(false)),
                entry(1, uni(true)),
                entry(2, stor(false)),
                entry(3, stor(false)),
                entry(4, stor(false)),
                entry(5, stor(false)),
                entry(6, stor(false)),
                entry(7, stor(false)),
                entry(8, stor(true)),
                entry(9, stor(true)),
                entry(10, stor(true)),
                entry(11, stor(true)),
                entry(12, stor(true)),
                entry(13, stor(true)),
                entry(14, stor(true)),
                entry(15, stor(true)),
                entry(16, stor(true)),
                entry(17, stor(true)),
                entry(18, stor(false)),
                entry(19, stor(false)),
                entry(20, stor(false)),
                entry(21, stor(false)),
            ],
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fdtd-bg"),
            layout: &bgl,
            entries: &[
                bentry(0, &params_buf),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &face_buf,
                        offset: 0,
                        // One FaceU slot (2 x vec4u); the dynamic offset
                        // selects among the 256-byte-strided face slots.
                        size: std::num::NonZeroU64::new(32),
                    }),
                },
                bentry(2, &field_bufs[0]),
                bentry(3, &field_bufs[1]),
                bentry(4, &field_bufs[2]),
                bentry(5, &field_bufs[3]),
                bentry(6, &field_bufs[4]),
                bentry(7, &field_bufs[5]),
                bentry(8, &ca_x),
                bentry(9, &ca_y),
                bentry(10, &ca_z),
                bentry(11, &cb_x),
                bentry(12, &cb_y),
                bentry(13, &cb_z),
                bentry(14, &pec_x),
                bentry(15, &pec_y),
                bentry(16, &pec_z),
                bentry(17, &coeff_buf),
                bentry(18, &psi_buf),
                bentry(19, &wire_buf),
                bentry(20, &state_buf),
                bentry(21, &record_buf),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fdtd-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // ---- pipelines ----
        let mk = |src: &str, entry_point: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry_point),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe_hx = mk(SHADER_H, "update_hx");
        let pipe_hy = mk(SHADER_H, "update_hy");
        let pipe_hz = mk(SHADER_H, "update_hz");
        let pipe_ex = mk(SHADER_E, "update_ex");
        let pipe_ey = mk(SHADER_E, "update_ey");
        let pipe_ez = mk(SHADER_E, "update_ez");
        let pipe_ce = [
            mk(SHADER_CPML, "cpml_e_x"),
            mk(SHADER_CPML, "cpml_e_y"),
            mk(SHADER_CPML, "cpml_e_z"),
        ];
        let pipe_ch = [
            mk(SHADER_CPML, "cpml_h_x"),
            mk(SHADER_CPML, "cpml_h_y"),
            mk(SHADER_CPML, "cpml_h_z"),
        ];
        let pipe_wire_update = mk(SHADER_MISC, "wire_update");
        let pipe_wire_deposit = mk(SHADER_MISC, "wire_deposit");
        let pipe_pec = [
            mk(SHADER_MISC, "pec_ex"),
            mk(SHADER_MISC, "pec_ey"),
            mk(SHADER_MISC, "pec_ez"),
        ];
        let pipe_source = mk(SHADER_MISC, "source_step");

        Some(GpuFdtd {
            device,
            queue,
            adapter_name,
            bg,
            pipe_hx,
            pipe_hy,
            pipe_hz,
            pipe_ex,
            pipe_ey,
            pipe_ez,
            pipe_ce,
            pipe_ch,
            pipe_wire_update,
            pipe_wire_deposit,
            pipe_pec,
            pipe_source,
            faces,
            dims_e: [
                wg3(nx as u32 + 1, ny as u32, nz as u32),
                wg3(nx as u32, ny as u32 + 1, nz as u32),
                wg3(nx as u32, ny as u32, nz as u32 + 1),
            ],
            dims_h: [
                wg3(nx as u32, ny as u32 + 1, nz as u32 + 1),
                wg3(nx as u32 + 1, ny as u32, nz as u32 + 1),
                wg3(nx as u32 + 1, ny as u32 + 1, nz as u32),
            ],
            e_sizes: [
                g.ex.len() as u32,
                g.ey.len() as u32,
                g.ez.len() as u32,
            ],
            h_sizes: [
                g.hx.len() as u32,
                g.hy.len() as u32,
                g.hz.len() as u32,
            ],
            field_bufs,
            record_buf,
            wire_buf,
            wire_nseg,
            wire_nnodes,
            total_steps,
            step: 0,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn steps_done(&self) -> usize {
        self.step
    }

    /// Run `n_steps` (must not exceed the `total_steps` given at creation).
    /// Submits in chunks of 64 steps; blocks until the GPU finishes.
    pub fn run(&mut self, n_steps: usize) {
        assert!(
            self.step + n_steps <= self.total_steps,
            "GpuFdtd: run beyond the record buffer capacity"
        );
        // Steps per submission; overridable for debugging (GPU_CHUNK).
        let chunk_max: usize = std::env::var("GPU_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64)
            .max(1);
        let mut done = 0;
        while done < n_steps {
            let chunk = (n_steps - done).min(chunk_max);
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("fdtd-chunk"),
                },
            );
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                for _ in 0..chunk {
                    // H update
                    pass.set_pipeline(&self.pipe_hx);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(self.dims_h[0].0, self.dims_h[0].1, self.dims_h[0].2);
                    pass.set_pipeline(&self.pipe_hy);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(self.dims_h[1].0, self.dims_h[1].1, self.dims_h[1].2);
                    pass.set_pipeline(&self.pipe_hz);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(self.dims_h[2].0, self.dims_h[2].1, self.dims_h[2].2);
                    // CPML-H, CPU face order (xlo,xhi,ylo,yhi,zlo,zhi)
                    for (fi, f) in self.faces.iter().enumerate() {
                        pass.set_pipeline(&self.pipe_ch[f.axis as usize]);
                        pass.set_bind_group(0, &self.bg, &[(fi as u64 * FACE_SLOT_BYTES) as u32]);
                        pass.dispatch_workgroups(f.disp.0, f.disp.1, f.disp.2);
                    }
                    // wire current advance (E^n still current)
                    pass.set_pipeline(&self.pipe_wire_update);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(1, 1, 1);
                    // E update
                    pass.set_pipeline(&self.pipe_ex);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(self.dims_e[0].0, self.dims_e[0].1, self.dims_e[0].2);
                    pass.set_pipeline(&self.pipe_ey);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(self.dims_e[1].0, self.dims_e[1].1, self.dims_e[1].2);
                    pass.set_pipeline(&self.pipe_ez);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(self.dims_e[2].0, self.dims_e[2].1, self.dims_e[2].2);
                    // CPML-E, same face order
                    for (fi, f) in self.faces.iter().enumerate() {
                        pass.set_pipeline(&self.pipe_ce[f.axis as usize]);
                        pass.set_bind_group(0, &self.bg, &[(fi as u64 * FACE_SLOT_BYTES) as u32]);
                        pass.dispatch_workgroups(f.disp.0, f.disp.1, f.disp.2);
                    }
                    // wire deposit, PEC, source + probe
                    pass.set_pipeline(&self.pipe_wire_deposit);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(1, 1, 1);
                    for c in 0..3 {
                        pass.set_pipeline(&self.pipe_pec[c]);
                        pass.set_bind_group(0, &self.bg, &[0]);
                        pass.dispatch_workgroups(wg(self.e_sizes[c]), 1, 1);
                    }
                    pass.set_pipeline(&self.pipe_source);
                    pass.set_bind_group(0, &self.bg, &[0]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            self.step += chunk;
            done += chunk;
        }
    }

    async fn download_f32(&self, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let bytes = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }

    /// All six field arrays in order (ex, ey, ez, hx, hy, hz).
    pub async fn download_fields(&self) -> [Vec<f32>; 6] {
        let mut out: [Vec<f32>; 6] = Default::default();
        for c in 0..6 {
            let count = if c < 3 {
                self.e_sizes[c] as usize
            } else {
                self.h_sizes[c - 3] as usize
            };
            out[c] = self.download_f32(&self.field_bufs[c], count).await;
        }
        out
    }

    /// (V, I, V_field) time series for the steps run so far.
    pub async fn download_probe(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let raw = self.download_f32(&self.record_buf, 3 * self.step.max(1)).await;
        let mut v = Vec::with_capacity(self.step);
        let mut i = Vec::with_capacity(self.step);
        let mut v_field = Vec::with_capacity(self.step);
        for s in 0..self.step {
            v.push(raw[3 * s]);
            i.push(raw[3 * s + 1]);
            v_field.push(raw[3 * s + 2]);
        }
        (v, i, v_field)
    }

    /// (currents, charges) of the thin wire, if present.
    pub async fn download_wire(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        if self.wire_nseg == 0 {
            return None;
        }
        let raw = self
            .download_f32(&self.wire_buf, self.wire_nseg + self.wire_nnodes)
            .await;
        Some((
            raw[..self.wire_nseg].to_vec(),
            raw[self.wire_nseg..].to_vec(),
        ))
    }
}
