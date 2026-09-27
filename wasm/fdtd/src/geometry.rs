//! Snyder 2.44 GHz probe-fed patch voxelization.
//!
//! Geometry (from `frontend/scripts/snyder-layer-sweep.ts`):
//! - FR-4 substrate: 101.6 × 101.6 × 1.575 mm, εr = 4.4, tanδ = 0.02
//! - Ground: zero-thickness PEC sheet at z = 0 (101.6 × 101.6 mm)
//! - Patch: zero-thickness PEC sheet at z = 1.575 mm (28.2084 × 37.5789 mm)
//! - Probe: 1.3 mm square copper pin at x = −6.5384 mm, z ∈ [0.525, 1.575] mm
//! - Feed gap: 1 cell (dz = 0.525 mm) between ground and probe bottom;
//!   the FEM's 0.5 mm gap is snapped to the grid.
//!
//! Grid: dx = dy = 0.5 mm, dz = 0.525 mm (3 cells = 1.575 mm exactly, so
//! both PEC sheets land on Ex/Ey sample planes). Domain ±66 mm in x/y,
//! z ∈ [−16.5375, 16.5375] mm (63 cells), 8-cell CPML on all faces.

use crate::grid::YeeGrid;
use crate::material::Materials;
use crate::source::Feed;

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

/// Voxelize the Snyder patch into `mat`. Returns the feed descriptor.
pub fn build_snyder(g: &YeeGrid, mat: &mut Materials, dt: f32) -> Feed {
    assert!((g.dx - DX).abs() < 1e-9 && (g.dy - DY).abs() < 1e-9 && (g.dz - DZ).abs() < 1e-9);
    let eps0 = 8.854_187_817e-12f32;
    let sigma_sub = 2.0 * std::f32::consts::PI * F0 * eps0 * EPS_R * LOSS_TANGENT;

    let sub_half = SUB_XY / 2.0;
    let patch_hx = PATCH_L / 2.0;
    let patch_hy = PATCH_W / 2.0;

    // Probe footprint in Ez indices (cell-centered x,y).
    let pc_i = ((PROBE_X - X0) / DX - 0.5).round() as isize;
    let pc_j = ((0.0 - Y0) / DY - 0.5).round() as isize;
    // 1.3 mm pin ≈ 2.6 cells; use a 3×3-cell staircased block.
    let pr: isize = 1; // 3x3 via

    // k-planes of the PEC sheets (Ex/Ey sample planes).
    let k_ground = 31; // z = 0
    let k_patch = 34; // z = 1.575 mm
    // Probe z-range in Ez indices. The via's top Ez=0 MUST overlap the
    // patch Ex/Ey=0 plane at k=34, so it extends to k=35 (z=1.8375mm,
    // half a cell above the patch). This half-cell extension is the
    // published fix for Yee wire-end staggering (Boonzaaier & Pistorius
    // 1992 via Douglas et al. 1998); without it the via is only
    // capacitively coupled to the patch (measured as −j1100 Ω).
    let k_probe_lo = 33;
    let k_probe_hi = 35;
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

    // --- Probe PEC block (Ez inside; side-wall Ex/Ey) ---
    let pi0 = (pc_i - pr).max(0) as usize;
    let pi1 = (pc_i + pr).min(g.nx as isize - 1) as usize;
    let pj0 = (pc_j - pr).max(0) as usize;
    let pj1 = (pc_j + pr).min(g.ny as isize - 1) as usize;
    for k in k_probe_lo..=k_probe_hi {
        for j in pj0..=pj1 {
            for i in pi0..=pi1 {
                mat.set_pec(2, g.ez_idx(i, j, k));
            }
        }
    }
    // Side walls: Ex at i = pi0/pi1+1 faces, Ey at j = pj0/pj1+1 faces.
    for k in k_ground..=k_patch {
        let zc = (k as f32 + 0.5) * DZ + Z0;
        if zc < 0.4e-3 || zc > SUB_H {
            continue;
        }
        for j in pj0..=pj1 {
            mat.set_pec(0, g.ex_idx(pi0, j, k));
            if pi1 + 1 <= g.nx {
                mat.set_pec(0, g.ex_idx(pi1 + 1, j, k));
            }
        }
        for i in pi0..=pi1 {
            mat.set_pec(1, g.ey_idx(i, pj0, k));
            if pj1 + 1 <= g.ny {
                mat.set_pec(1, g.ey_idx(i, pj1 + 1, k));
            }
        }
    }

    // --- Feed descriptor ---
    let src_i = pc_i as usize;
    let src_j = pc_j as usize;
    // Ampère loop at the gap H-plane, enclosing the probe block.
    let li0 = pi0.saturating_sub(2);
    let li1 = (pi1 + 3).min(g.nx);
    let lj0 = pj0.saturating_sub(2);
    let lj1 = (pj1 + 3).min(g.ny);
    Feed {
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
    }
}
