//! Snyder 2.44 GHz probe-fed patch voxelization.
//!
//! Geometry (from `frontend/scripts/snyder-layer-sweep.ts`):
//! - FR-4 substrate: 101.6 × 101.6 × 1.575 mm, εr = 4.4, tanδ = 0.02
//! - Ground: PEC at z = 0 (101.6 × 101.6 mm)
//! - Patch: zero-thickness PEC sheet at z = 1.575 mm (28.2084 × 37.5789 mm)
//! - Probe: 1.3 mm square copper pin at x = −6.5384 mm, z ∈ [0.525, 1.575] mm
//! - Feed gap: 1 cell (dz = 0.525 mm) between ground and probe bottom;
//!   the FEM's 0.5 mm gap is snapped to the grid.
//!
//! Two feed/ground constructions share the substrate and sheets:
//! - `build_snyder` (production): a grid-faithful PEC post (Ez PEC in a
//!   plus-cross at the probe column) driven by the Norton gap port, over
//!   a VOLUMETRIC ground (Ez PEC at k=30 and k=31). The original
//!   sheet ground mirrors the gap cell's Ez across its plane
//!   (Ez(k=31) ≈ −Ez(k=32)), cancelling the port voltage — the cavity
//!   saw ~1% of the drive (see snyder_diag / MEMORY.md). The post
//!   delivers ~370× the port energy of the thin wire on this grid.
//! - `build_snyder_wire`: the original Holland/Simpson thin-wire probe
//!   (0.1 mm filament — it cannot represent the 1.3 mm pin on a
//!   0.5 mm grid) over the zero-thickness sheet ground. Retained for
//!   comparison studies; the wire-era drivers use it explicitly.
//!
//! Grid: dx = dy = 0.5 mm, dz = 0.525 mm (3 cells = 1.575 mm exactly, so
//! both PEC sheets land on Ex/Ey sample planes). Domain ±58 mm in x/y,
//! z ∈ [−16.5375, 15.4875] mm (61 cells).

use crate::grid::YeeGrid;
use crate::material::Materials;
use crate::source::Feed;
use crate::thin_wire::ThinWire;

pub const DX: f32 = 0.5e-3;
pub const DY: f32 = 0.5e-3;
pub const DZ: f32 = 0.525e-3;
pub const NX: usize = 232;
pub const NY: usize = 232;
pub const NZ: usize = 61;
pub const X0: f32 = -58.0e-3;
pub const Y0: f32 = -58.0e-3;
/// Chosen so the ground sheet (z=0) lands on the Ex/Ey k=31 plane and the
/// patch sheet (z=1.575 mm) on k=34: z0 = −31.5·dz.
pub const Z0: f32 = -31.5 * DZ;

const SUB_XY: f32 = 101.6e-3;
const SUB_H: f32 = 1.575e-3;
const PATCH_L: f32 = 28.2084e-3;
const PATCH_W: f32 = 37.5789e-3;
const PROBE_X: f32 = -14.1042e-3 + 7.5658e-3; // −L/2 + FEED_FROM_EDGE
const PROBE_HALF: f32 = 0.65e-3;
const EPS_R: f32 = 4.4;
const LOSS_TANGENT: f32 = 0.02;
const F0: f32 = 2.5e9; // conductivity evaluated here

/// Voxelize the Snyder patch into `mat` (production construction):
/// PEC post + Norton gap port over a volumetric ground. Returns the
/// feed descriptor and no thin wire.
pub fn build_snyder(g: &YeeGrid, mat: &mut Materials, dt: f32) -> (Feed, Option<ThinWire>) {
    let (feed, _) = build_snyder_impl(g, mat, dt, true);
    (feed, None)
}

/// The original thin-wire construction: Holland/Simpson filament over
/// the zero-thickness sheet ground. Retained for comparison studies.
pub fn build_snyder_wire(g: &YeeGrid, mat: &mut Materials, dt: f32) -> (Feed, ThinWire) {
    let (feed, wire) = build_snyder_impl(g, mat, dt, false);
    (feed, wire.expect("wire construction returns a wire"))
}

fn build_snyder_impl(
    g: &YeeGrid,
    mat: &mut Materials,
    dt: f32,
    production: bool,
) -> (Feed, Option<ThinWire>) {
    assert!((g.dx - DX).abs() < 1e-9 && (g.dy - DY).abs() < 1e-9 && (g.dz - DZ).abs() < 1e-9);
    let eps0 = 8.854_187_817e-12f32;
    let sigma_sub = 2.0 * std::f32::consts::PI * F0 * eps0 * EPS_R * LOSS_TANGENT;

    let sub_half = SUB_XY / 2.0;
    let patch_hx = PATCH_L / 2.0;
    let patch_hy = PATCH_W / 2.0;

    // Probe footprint in Ez indices (cell-centered x,y).
    let pc_i = ((PROBE_X - X0) / DX - 0.5).round() as isize;
    let pc_j = ((0.0 - Y0) / DY - 0.5).round() as isize;
    // Thin-wire model: single-cell (1×1) wire instead of 3×3 block.
    // The Taflove/Umashankar correction accounts for the wire radius.
    let pr: isize = 0; // 1x1 thin wire (was 3x3 block)

    // k-planes of the PEC sheets (Ex/Ey sample planes).
    let k_ground = 31; // z = 0
    let k_patch = 34; // z = 1.575 mm
    // Probe z-range in Ez indices. Per research (Boonzaaier & Pistorius via
    // Douglas et al.): the via's top Ez=0 cell must have its upper edge
    // COINCIDENT with the patch Ex/Ey=0 plane. Ez(k) upper edge is at
    // (k+0.5)*dz+Z0; for k=34 that's exactly 1.575mm = patch plane.
    // (k=35's upper edge is at 2.1mm, which jumps OVER the plane.)
    let k_probe_lo = 33;
    let k_probe_hi = 34;
    // Gap/source cell: Ez k=32 (z=0.2625mm), between ground and via.
    let k_src = 32;

    // --- Substrate dielectric (all three E components) ---
    for k in 0..g.nz {
        let zc = (k as f32 + 0.5) * DZ + Z0; // Ex/Ey plane height
        if zc < 0.0 || zc > SUB_H {
            continue;
        }
        for j in 0..g.ny {
            let y = (j as f32 + 0.5) * DY + Y0;
            if y.abs() > sub_half {
                continue;
            }
            for i in 0..=g.nx {
                let x = i as f32 * DX + X0;
                if x.abs() > sub_half {
                    continue;
                }
                // Skip the air column under the probe (matches the FEM model).
                if (x - PROBE_X).abs() < PROBE_HALF + DX
                    && y.abs() < PROBE_HALF + DY
                    && k >= k_ground
                    && k <= k_patch
                {
                    continue;
                }
                mat.set_dielectric(0, g.ex_idx(i, j, k), EPS_R, sigma_sub, dt);
            }
        }
    }
    // Ey and Ez substrate: same footprint (sample positions differ by < dx/2;
    // the box test below is per-component for correctness).
    for k in 0..g.nz {
        for j in 0..=g.ny {
            let y = j as f32 * DY + Y0;
            if y.abs() > sub_half {
                continue;
            }
            let zc = (k as f32 + 0.5) * DZ + Z0;
            if zc < 0.0 || zc > SUB_H {
                continue;
            }
            for i in 0..g.nx {
                let x = (i as f32 + 0.5) * DX + X0;
                if x.abs() > sub_half {
                    continue;
                }
                if (x - PROBE_X).abs() < PROBE_HALF + DX
                    && y.abs() < PROBE_HALF + DY
                {
                    continue;
                }
                mat.set_dielectric(1, g.ey_idx(i, j, k), EPS_R, sigma_sub, dt);
            }
        }
    }
    for k in 0..=g.nz {
        let z = k as f32 * DZ + Z0;
        if z < 0.0 || z > SUB_H {
            continue;
        }
        for j in 0..g.ny {
            let y = (j as f32 + 0.5) * DY + Y0;
            if y.abs() > sub_half {
                continue;
            }
            for i in 0..g.nx {
                let x = (i as f32 + 0.5) * DX + X0;
                if x.abs() > sub_half {
                    continue;
                }
                if (x - PROBE_X).abs() < PROBE_HALF + DX
                    && y.abs() < PROBE_HALF + DY
                {
                    continue;
                }
                mat.set_dielectric(2, g.ez_idx(i, j, k), EPS_R, sigma_sub, dt);
            }
        }
    }

    // --- Ground PEC sheet (z=0): tangential E = 0 ---
    for j in 0..g.ny {
        let y = (j as f32 + 0.5) * DY + Y0;
        if y.abs() > sub_half {
            continue;
        }
        for i in 0..=g.nx {
            let x = i as f32 * DX + X0;
            if x.abs() <= sub_half {
                mat.set_pec(0, g.ex_idx(i, j, k_ground));
            }
        }
    }
    for j in 0..=g.ny {
        let y = j as f32 * DY + Y0;
        if y.abs() > sub_half {
            continue;
        }
        for i in 0..g.nx {
            let x = (i as f32 + 0.5) * DX + X0;
            if x.abs() <= sub_half {
                mat.set_pec(1, g.ey_idx(i, j, k_ground));
            }
        }
    }

    // --- Patch PEC sheet (z=1.575 mm) ---
    for j in 0..g.ny {
        let y = (j as f32 + 0.5) * DY + Y0;
        if y.abs() > patch_hy {
            continue;
        }
        for i in 0..=g.nx {
            let x = i as f32 * DX + X0;
            if x.abs() <= patch_hx {
                mat.set_pec(0, g.ex_idx(i, j, k_patch));
            }
        }
    }
    for j in 0..=g.ny {
        let y = j as f32 * DY + Y0;
        if y.abs() > patch_hy {
            continue;
        }
        for i in 0..g.nx {
            let x = (i as f32 + 0.5) * DX + X0;
            if x.abs() <= patch_hx {
                mat.set_pec(1, g.ey_idx(i, j, k_patch));
            }
        }
    }

    let thin_wire = if production {
        // --- PEC post (production feed) ---
        // Ez PEC in a plus-cross at the probe column, k_probe_lo..=hi —
        // the voxelized 1.3 mm pin. The Norton source drives the gap
        // cell below it (k_src) against the volumetric ground.
        for (di, dj) in [(0isize, 0isize), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            let ii = (pc_i + di) as usize;
            let jj = (pc_j + dj) as usize;
            for k in k_probe_lo..=k_probe_hi {
                mat.set_pec(2, g.ez_idx(ii, jj, k));
            }
        }
        // --- Volumetric ground ---
        // Ez PEC at k=30 and k=31 over the ground footprint: kills the
        // sheet's mirror image of the gap cell (see module docs).
        for j in 14..218usize {
            for i in 14..218usize {
                mat.set_pec(2, g.ez_idx(i, j, k_ground));
                mat.set_pec(2, g.ez_idx(i, j, k_ground - 1));
            }
        }
        None
    } else {
        // --- Probe thin wire (Holland/Simpson subcell model) ---
        // The wire is NOT a PEC cell. The Holland/Simpson model represents it as
        // an auxiliary 1D transmission-line system coupled bidirectionally to the
        // FDTD grid (samples Ez EMF, deposits current back). No side-wall Ex/Ey
        // PEC (thin wire has no resolved side walls).
        let pi0 = (pc_i - pr).max(0) as usize;
        let pj0 = (pc_j - pr).max(0) as usize;
        // Thin-wire model for the probe. Physical probe is 1.3mm square, but
        // the thin-wire subcell model requires a < Δ/2. Using a=0.1mm as a
        // starting point.
        const WIRE_RADIUS: f32 = 0.1e-3; // 0.1 mm
        Some(ThinWire::new(
            pi0, // i0 (pc_i since pr=0)
            pj0, // j0 (pc_j since pr=0)
            k_probe_lo,
            k_probe_hi,
            DX,
            DY,
            DZ,
            WIRE_RADIUS,
            true, // top terminates on the patch (PEC, V=0)
        ))
    };
    // --- Feed descriptor ---
    let src_i = pc_i as usize;
    let src_j = pc_j as usize;
    // Ampère loop at the gap H-plane, enclosing the probe.
    // (pc_i,pc_j) is the probe cell; loop extends ±2 cells around it.
    let pi0 = pc_i.max(0) as usize;
    let pj0 = pc_j.max(0) as usize;
    let li0 = pi0.saturating_sub(2);
    let li1 = (pi0 + 3).min(g.nx);
    let lj0 = pj0.saturating_sub(2);
    let lj1 = (pj0 + 3).min(g.ny);
    let feed = Feed {
        src_i,
        src_j,
        src_k: k_src,
        k_loop: k_src,
        hx_j_lo: lj0,
        hx_j_hi: lj1.min(g.ny),
        hx_i0: li0,
        hx_i1: li1,
        hy_i_lo: li0,
        hy_i_hi: li1.min(g.nx),
        hy_j0: lj0,
        hy_j1: lj1,
    };
    (feed, thin_wire)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sums(mat: &Materials) -> (f64, f64, usize, usize, usize) {
        let s = |v: &Vec<f32>| v.iter().map(|&x| x as f64).sum::<f64>();
        let c = |v: &Vec<u8>| v.iter().filter(|&&x| x != 0).count();
        (s(&mat.ca_x), s(&mat.cb_z), c(&mat.pec_x), c(&mat.pec_y), c(&mat.pec_z))
    }

    /// Production geometry must reproduce the validated reference
    /// construction (examples/snyder_ref.rs checksums at CFL 0.1,
    /// matched exactly by the WebGPU builder): PEC post + volumetric
    /// ground over the shared substrate/sheets.
    #[test]
    fn production_snyder_matches_reference_checksums() {
        let g = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
        let dt = g.courant_dt(0.1);
        let mut mat = Materials::vacuum(NX, NY, NZ, dt);
        let (feed, wire) = build_snyder(&g, &mut mat, dt);
        assert!(wire.is_none());
        assert_eq!((feed.src_i, feed.src_j, feed.src_k), (102, 115, 32));
        let (ca_x, cb_z, pec_x, pec_y, pec_z) = sums(&mat);
        assert_eq!((pec_x, pec_y, pec_z), (45744, 45612, 83242));
        assert!((ca_x - 3.297412e6).abs() / 3.297412e6 < 1e-5, "ca_x {ca_x}");
        assert!((cb_z - 3.580207e4).abs() / 3.580207e4 < 1e-5, "cb_z {cb_z}");
    }

    /// The wire variant keeps the sheet ground: no Ez PEC anywhere
    /// (the post's 10 cells + the thick ground's 83,232 are absent),
    /// and the same substrate/sheets/feed as production.
    #[test]
    fn wire_variant_keeps_sheet_ground() {
        let g = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
        let dt = g.courant_dt(0.1);
        let mut mat = Materials::vacuum(NX, NY, NZ, dt);
        let (feed, wire) = build_snyder_wire(&g, &mut mat, dt);
        assert_eq!((feed.src_i, feed.src_j, feed.src_k), (102, 115, 32));
        assert_eq!(wire.n_segments, 2);
        let (ca_x, cb_z, pec_x, pec_y, pec_z) = sums(&mat);
        assert_eq!((pec_x, pec_y, pec_z), (45744, 45612, 0));
        assert!((ca_x - 3.297412e6).abs() / 3.297412e6 < 1e-5, "ca_x {ca_x}");
        assert!((cb_z - 3.580207e4).abs() / 3.580207e4 < 1e-5, "cb_z {cb_z}");
    }
}
