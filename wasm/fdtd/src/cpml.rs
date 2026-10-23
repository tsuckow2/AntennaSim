//! Convolutional PML (CPML) absorbing boundary, Gedney formulation.
//!
//! For a PML normal to x, the stretched-coordinate derivative is
//! `∂/∂x → (1/κ_x)·∂/∂x + ψ_x`, where the convolution term updates as
//! `ψ^{n+1} = b_x·ψ^n + a_x·(∂F/∂x)` with (CFS α included)
//! `b_x = exp(−(σ_x/κ_x + α_x)·Δt/ε₀)`,
//! `a_x = (b_x − 1)·σ_x/(κ_x·(σ_x + κ_x·α_x))`.
//! The H-side ψ coefficients equal the E-side ones: the matched
//! magnetic conductivity σ* = σ·μ₀/ε₀ has decay rate σ*/(κμ₀) =
//! σ/(κε₀). α is graded 0 at the outer boundary to α_max = 0.05 S/m
//! at the interface; it damps grazing/evanescent modes that make
//! α = 0 CPML late-time unstable.
//!
//! Only field components whose curl contains a derivative along the PML
//! normal need a ψ term (4 per face). ψ storage is face-local:
//! `npml × face_area` per term, not full-grid.
//!
//! Tangential-damping prototype (env `CPML_TANGENTIAL_RATIO`, default
//! 0 = off): the mode-2 slab-guided instability propagates *parallel*
//! to the PML faces inside the layer, where normal-direction CPML
//! absorption is ~zero. True M-PML (Meza-Fajardo & Papageorgiou 2008)
//! would stretch the tangential derivatives as well; as a first
//! prototype `Cpml::damp_tangential` instead applies a matched
//! electric+magnetic loss inside each face's shell, graded with the
//! face's own σ profile scaled by the ratio R (σ_t = R·σ). Matched
//! loss (σ*/μ₀ = σ/ε₀) keeps the vacuum impedance, so both E and H
//! decay by the same factor `exp(−σ_t·Δt/ε₀)` per step; overlapping
//! faces add conductivities. Known shortcomings of the surrogate vs
//! true M-PML: the loss is isotropic within the shell rather than
//! tied to the tangential stretching factors, it is applied as a
//! once-per-step exponential factor (leapfrog-centred only to O(Δt)),
//! and the depth index of staggered samples is taken from each
//! component's own outermost sample (half-cell offsets vs the ψ
//! machinery are ignored — immaterial for a smooth grading).
//!
//! True M-PML prototype (env `CPML_MPML_RATIO`, plus
//! `CPML_MPML_RATIO_Z` for the out-of-slab-plane tangential
//! direction; default off): inside a face with normal n, the
//! stretching factors of the two *tangential* derivative directions
//! u, v are no longer 1. Each tangential derivative inside the
//! face's shell gets its own ψ convolution, owned by the face, with
//! coefficients built from the face's tangential conductivity
//! profile σ_t(p) = R_t·σ(p) (same grading shape as the normal
//! profile, zero at the interface, R_t the Meza-Fajardo–
//! Papageorgiou "proportion coefficient" for that direction), the
//! face's CFS α profile, and κ_t = 1 — so the tangential correction
//! is the pure ψ term of the same (component, field, derivative)
//! template the normal corrections use, with the (1−1/κ) direct
//! term vanishing. E- and H-side tangential coefficients coincide
//! by the same impedance-matching identity as the normal ones.
//! Overlap partition rule: tangential corrections apply ONLY in
//! single-shell cells (face interiors). In an edge/corner overlap
//! a tangential derivative already carries the normal ψ of its own
//! axis's face; adding this face's tangential ψ for the same
//! derivative counts the unstretched derivative twice (at low
//! frequency each ψ → −∂F/∂a, so the pair sums to ≈ −∂F/∂a — a
//! sign flip, i.e. gain). The first prototype without the rule
//! blew up at step ~750 localised at a corner column (×1.10/step).
//! With the rule the replica still blows up (step ~1750,
//! ×1.03/step), now nucleating at the corners of the region where
//! the tangential operator switches on inside a face — hard
//! tangential switching is itself unstable in this architecture;
//! see the design note for the full evidence and the phase-2 plan
//! (tangentially ramped σ_t, or summed-profile single convolutions
//! in overlaps). See
//! goals/antennasim-fem-solver/hidden_files/
//! cpml-mpml-design-2026-10-21.md for the full derivation.
//!
//! Phase 2a (env `CPML_MPML_RAMP`, default 0 = off): the hard
//! partition switch is replaced by a tangential ramp. Inside the
//! single-shell region each tangential correction is scaled by
//! `w(q1)·w(q2)`, where `q` is the component's distance in cells
//! from the nearest switch boundary along that tangential axis
//! (0 at the first active cell) and `w` is the smoothstep
//! `t²(3−2t)`, `t = min(1, q/W)`, over the ramp width W =
//! `CPML_MPML_RAMP`. Both the ψ drive (`a_t·∂F`) and the applied
//! correction are scaled by `w` at apply time, per the design
//! note's prescription; the recursion coefficient `b_t` keeps its
//! normal-direction profile (its spatial variation is
//! second-order). The effective tangential operator therefore
//! turns on continuously from exactly 0 at the partition boundary
//! instead of jumping, while overlap cells stay excluded (no
//! double count). With the env unset (W = 0) `w ≡ 1` and the
//! phase-1 partition behaviour is reproduced bit-identically.
//! Phase 2a falsified the smoothing hypothesis: the nucleation
//! locus tracks the ramp top and the rate barely softens — any
//! tangential non-uniformity of the discrete operator inside a
//! face seeds the instability (design note §8).
//!
//! Phase 2b (env `CPML_MPML_FORM=summed`; default `partition`
//! keeps the phase-1/2a paths): the transition is removed by
//! construction. A single shell-volume pass owns ALL stretching
//! corrections (the per-face normal and tangential passes are
//! skipped): exactly one ψ convolution per (component,
//! derivative) template — the twelve templates cloned verbatim
//! from the face-normal correct_e_*/correct_h_* bodies — with
//! per-point coefficients built from the continuum's summed
//! directional profiles. At a shell point, direction u carries
//! σ_u^Σ = Σ_faces∋point σ_face,u (the owning axis face's normal
//! profile for its own direction, the Meza-Fajardo proportion
//! R_u·σ_face for the other faces), α^Σ = Σ_faces∋point α_face
//! (the design note's "α^Σ likewise"), and κ^Σ = the own-axis
//! face's κ profile, or 1 where the point is not in that face
//! (κ_t = 1, as in phase 1; κ is not summed). In single-shell
//! regions these reduce exactly to the existing normal /
//! tangential coefficient pairs; in edge/corner overlaps they
//! are the continuum M-PML's single summed stretching. The
//! coefficients are static, so they are precomputed once per
//! template over the corrected component's full index range
//! (ψ state likewise lives on component-shaped arrays — a
//! prototype layout; a production version would compress to
//! shell-indexed storage). Depths along each axis are taken from
//! the corrected component's own indices/extents, the same
//! convention the partition tests use. With M-PML off the
//! summed pass never runs and behaviour is bit-identical.
//!
//! Phase 2c (env `CPML_MPML_ALPHA`, summed form only; default
//! `sum` = the phase-2b rule): phase 2b still nucleated, in
//! lateral×z-lo edge blocks, and its mechanism read (§9 of the
//! design note) indicts the naive α sum — α is graded opposite
//! to σ, so α^Σ jumps ≈ α_max across a shell-interface plane
//! inside another face's shell while σ^Σ and κ^Σ change only by
//! grading-mediated amounts. Three alternative α-combination
//! rules are selectable: `weighted` (σ-weighted mean of the
//! faces' α, so the transition is σ-mediated), `dominant` (α
//! of the face contributing the largest σ_F,u; ties to the
//! own-axis face), and `tzero` (only the own-axis face's α
//! enters α^Σ; tangential contributions carry α = 0). All other
//! summed-form machinery is unchanged, and `sum` reproduces
//! phase 2b bit-identically.
//!
//! Phase 3a (env `CPML_FORM=ade`; default = the correction form
//! above): whole-volume ADE-CPML with NORMAL profiles only — the
//! phase-3 restructure scoped in §10 of the design note, built as
//! an equivalence foundation for the tangential (M-PML) terms of
//! phase 3b. The stretching is folded into the solver's update
//! loops: at each (component, derivative) sample inside a shell,
//! the plain difference `d` is replaced by the stretched
//! derivative `(1/κ_u)·d + ψ_u`, with `ψ_u ← b_u·ψ_u + a_u·d`
//! advanced at the same site (see `AdeCpml`). Per-point
//! coefficient fields (b_u, a_u, 1/κ_u) are built from the
//! per-direction TOTAL profiles — in 3a the tangential part of
//! every total profile is zero, so each direction's coefficients
//! are exactly its own axis face's normal profile, and the folded
//! operator is the correction form's operator term by term
//! (including the face passes' transverse fringe omissions: each
//! template's coefficients are masked to the component points the
//! corresponding face-local pass actually touches). The ψ state
//! and coefficients live on shell-indexed slots (two slabs of
//! depth npml per term), not component-shaped arrays: the
//! note's ~158 MB full-grid state estimate becomes ≈ 32 MB of ψ
//! (≈ 127 MB with the slot-replicated coefficient triples) on
//! the Snyder grid. Phase 3b will fill the builder's tangential
//! contributions from the summed profiles of §9–§10.
//!
//! Grading (depth `p = 0` at the outer boundary, `p = npml−1` at the
//! PML/interior interface):
//! `σ(p) = σ_max·((npml−p)/npml)^m`, `κ(p) = 1 + (κ_max−1)·((npml−p)/npml)^m`,
//! `σ_max = (m+1)/(150·π·Δ)`.
//!
//! Indexing convention: for a `lo` face the PML occupies the lowest
//! indices and `p = 0` is the outermost layer; for a `hi` face `p = 0`
//! is the outermost layer at the high-index end. Field values outside
//! the grid (past the outer boundary) are taken as 0 (PEC-backed PML).

use crate::grid::YeeGrid;

/// One PML face: normal axis, side, depth profiles, and 4 ψ arrays.
///
/// ψ assignment by axis (a,b for E-updates; c,d for H-updates):
/// - x-normal: a=ψ(H_z,x)→Ey, b=ψ(H_y,x)→Ez, c=ψ(E_z,x)→Hy, d=ψ(E_y,x)→Hz
/// - y-normal: a=ψ(H_z,y)→Ex, b=ψ(H_x,y)→Ez, c=ψ(E_z,y)→Hx, d=ψ(E_x,y)→Hz
/// - z-normal: a=ψ(H_y,z)→Ex, b=ψ(H_x,z)→Ey, c=ψ(E_y,z)→Hx, d=ψ(E_x,z)→Hy
struct PmlFace {
    axis: u8,
    lo: bool,
    npml: usize,
    b_e: Vec<f32>,
    a_e: Vec<f32>,
    b_h: Vec<f32>,
    a_h: Vec<f32>,
    kinv: Vec<f32>,
    /// σ profile per depth p (kept for the tangential-damping
    /// prototype; the ψ coefficients above are derived from it).
    sigma: Vec<f32>,
    /// CFS α and stretching κ profiles per depth p (kept for the
    /// phase-2b summed-profile coefficient build, which needs the
    /// raw profiles rather than the derived (b, a) pairs).
    alpha: Vec<f32>,
    kappa: Vec<f32>,
    psi_a: Vec<f32>,
    psi_b: Vec<f32>,
    psi_c: Vec<f32>,
    psi_d: Vec<f32>,
    /// M-PML tangential convolution coefficients for the two
    /// tangential derivative axes (u, v) in cyclic order around the
    /// face normal ((n,u,v) = (x,y,z), (y,z,x), or (z,x,y)):
    /// built from σ_t(p) = R_t·σ(p), the face's α profile, κ_t = 1.
    bt_u: Vec<f32>,
    at_u: Vec<f32>,
    bt_v: Vec<f32>,
    at_v: Vec<f32>,
    /// M-PML tangential ψ arrays, 8 per face, indexed
    /// (p, q1, q2) over (npml, n1+1, n2+1) — one row beyond the
    /// normal-ψ transverse extents so components whose transverse
    /// extent is n+1 are covered. Order (all faces, cyclic (n,u,v);
    /// "via F along a" = convolution of ∂F/∂a, profile of axis a):
    /// t0 = E_n via H_v along u, t1 = E_n via H_u along v,
    /// t2 = E_u via H_n along v, t3 = E_v via H_n along u,
    /// t4 = H_n via E_v along u, t5 = H_n via E_u along v,
    /// t6 = H_u via E_n along v, t7 = H_v via E_n along u.
    psi_t: Vec<Vec<f32>>,
    /// Face dimensions transverse to the normal: (n1, n2).
    n1: usize,
    n2: usize,
}

impl PmlFace {
    #[allow(clippy::too_many_arguments)]
    fn new(
        axis: u8,
        lo: bool,
        npml: usize,
        n1: usize,
        n2: usize,
        delta: f32,
        dt: f32,
        ratio_u: f32,
        ratio_v: f32,
    ) -> Self {
        let eps0 = 8.854_187_817e-12f32;
        // Grading parameters; env overrides exist for stability probes
        // (defaults are the production values and are what every
        // production build uses).
        let env_f32 = |key: &str, default: f32| -> f32 {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        };
        let m = env_f32("CPML_M", 3.0);
        let kappa_max = env_f32("CPML_KAPPA_MAX", 7.0);
        let sigma_max = (m + 1.0) / (150.0 * std::f32::consts::PI * delta);
        // CFS shift α: damps the grazing/evanescent late-time modes that
        // plain (α = 0) CPML is known to destabilize. Graded opposite σ:
        // 0 at the outer boundary (p = 0), α_max at the PML/interior
        // interface (p = npml−1). α_max = 0.3 S/m is empirical: at 0.05 a
        // PEC-channel grazing mode grew ×1.009/step in the slab probe
        // (examples/cpml_probe.rs, ground + slab blob); 0.3 suppresses
        // it over 6 ns, as does npml = 12 at α = 0.05.
        let alpha_max = env_f32("CPML_ALPHA_MAX", 0.3);

        let mut b_e = vec![0.0; npml];
        let mut a_e = vec![0.0; npml];
        let mut b_h = vec![0.0; npml];
        let mut a_h = vec![0.0; npml];
        let mut kinv = vec![0.0; npml];
        let mut sigma_prof = vec![0.0; npml];
        let mut alpha_prof = vec![0.0; npml];
        let mut kappa_prof = vec![0.0; npml];
        let mut bt_u = vec![0.0; npml];
        let mut at_u = vec![0.0; npml];
        let mut bt_v = vec![0.0; npml];
        let mut at_v = vec![0.0; npml];
        for p in 0..npml {
            let rho = (npml - p) as f32 / npml as f32;
            let sigma = sigma_max * rho.powf(m);
            sigma_prof[p] = sigma;
            let kappa = 1.0 + (kappa_max - 1.0) * rho.powf(m);
            let alpha = alpha_max * (p as f32 / npml as f32);
            alpha_prof[p] = alpha;
            kappa_prof[p] = kappa;
            // Gedney CPML coefficients. The H-side ψ uses the magnetic
            // conductivity σ* = σ·μ₀/ε₀ (impedance-matched PML), whose
            // decay rate σ*/(κ·μ₀) = σ/(κ·ε₀) is identical to the
            // E-side rate; hence b_h = b_e and a_h = a_e exactly.
            let b = (-(sigma / kappa + alpha) * dt / eps0).exp();
            let a = if sigma + kappa * alpha > 0.0 {
                (b - 1.0) * sigma / (kappa * (sigma + kappa * alpha))
            } else {
                0.0
            };
            b_e[p] = b;
            a_e[p] = a;
            b_h[p] = b;
            a_h[p] = a;
            kinv[p] = 1.0 / kappa;
            // M-PML tangential coefficients (κ_t = 1): same Gedney
            // recursion form with σ_t = R_t·σ(p) and the face's own
            // CFS α profile, so the tangential stretching pole is
            // regularised exactly like the normal one. E/H coincide
            // by the same matching identity as above. With R_t = 0,
            // a_t = 0 and the ψ_t arrays can never excite (the
            // correction pass is also skipped entirely; see Cpml).
            for (ratio, bt, at) in [
                (ratio_u, &mut bt_u, &mut at_u),
                (ratio_v, &mut bt_v, &mut at_v),
            ] {
                let st = ratio * sigma;
                let bt_p = (-(st + alpha) * dt / eps0).exp();
                bt[p] = bt_p;
                at[p] = if st + alpha > 0.0 {
                    (bt_p - 1.0) * st / (st + alpha)
                } else {
                    0.0
                };
            }
        }
        let area = npml * n1 * n2;
        let tarea = npml * (n1 + 1) * (n2 + 1);
        Self {
            axis,
            lo,
            npml,
            b_e,
            a_e,
            b_h,
            a_h,
            kinv,
            sigma: sigma_prof,
            alpha: alpha_prof,
            kappa: kappa_prof,
            psi_a: vec![0.0; area],
            psi_b: vec![0.0; area],
            psi_c: vec![0.0; area],
            psi_d: vec![0.0; area],
            bt_u,
            at_u,
            bt_v,
            at_v,
            psi_t: (0..8).map(|_| vec![0.0; tarea]).collect(),
            n1,
            n2,
        }
    }

    #[inline]
    fn psi_idx(&self, p: usize, q1: usize, q2: usize) -> usize {
        (p * self.n1 + q1) * self.n2 + q2
    }

    #[inline]
    fn psi_t_idx(&self, p: usize, q1: usize, q2: usize) -> usize {
        (p * (self.n1 + 1) + q1) * (self.n2 + 1) + q2
    }
}

/// Six-face CPML.
pub struct Cpml {
    faces: Vec<PmlFace>, // xlo, xhi, ylo, yhi, zlo, zhi
    /// Tangential-damping prototype ratio R (σ_t = R·σ inside each
    /// face's shell). 0 = off (production default; env
    /// `CPML_TANGENTIAL_RATIO` overrides for probes only).
    tangential_ratio: f32,
    /// M-PML proportion coefficients (Meza-Fajardo–Papageorgiou):
    /// tangential conductivity σ_t = R·σ for tangential derivative
    /// directions in the slab plane (x, y), and R_z for the z
    /// direction where it is tangential (x- and y-faces). Both 0 =
    /// off (production default; env `CPML_MPML_RATIO` /
    /// `CPML_MPML_RATIO_Z`, or `CPML_MPML=1` for the 0.25 default).
    mpml_ratio: f32,
    mpml_ratio_z: f32,
    /// Overlap-partition guard band (cells): tangential corrections
    /// are additionally withheld within this many cells of another
    /// axis's shell boundary (env `CPML_MPML_GUARD`, default 0).
    /// Probe for the interface-seam instability; see design note.
    mpml_guard: usize,
    /// Phase-2a tangential ramp width W (cells): the tangential
    /// operator turns on smoothly over this many cells from each
    /// switch boundary (env `CPML_MPML_RAMP`, default 0 = off =
    /// hard partition switch, bit-identical to phase 1).
    mpml_ramp: usize,
    /// M-PML overlap treatment (env `CPML_MPML_FORM`, default
    /// `partition`).
    mpml_form: MpmlForm,
    /// Phase-2c α-combination rule for the summed form (env
    /// `CPML_MPML_ALPHA`, default `sum` = the phase-2b rule).
    mpml_alpha: MpmlAlpha,
    /// Phase-2b summed-profile templates, in the fixed order
    /// documented at `build_summed`; empty unless the summed form
    /// is active with M-PML on.
    summed: Vec<SummedTemplate>,
    /// Phase-3a ADE form state (env `CPML_FORM=ade`); `None` =
    /// the correction form (production default). When present,
    /// the solver folds the stretching into its update loops and
    /// the face-local correction passes are skipped.
    ade: Option<AdeCpml>,
}

/// Owned snapshot of one PML face, for external backends (e.g. the GPU
/// solver) that need the grading coefficients and ψ state.
pub struct PmlFaceData {
    pub axis: u8,
    pub lo: bool,
    pub npml: usize,
    /// Transverse face dimensions.
    pub n1: usize,
    pub n2: usize,
    pub b_e: Vec<f32>,
    pub a_e: Vec<f32>,
    pub b_h: Vec<f32>,
    pub a_h: Vec<f32>,
    pub kinv: Vec<f32>,
    /// σ grading profile per depth p (for the tangential-damping
    /// factor computation in external backends; see damp_tangential).
    pub sigma: Vec<f32>,
    pub psi_a: Vec<f32>,
    pub psi_b: Vec<f32>,
    pub psi_c: Vec<f32>,
    pub psi_d: Vec<f32>,
    /// M-PML tangential coefficients (u, v cyclic around the normal)
    /// and ψ state; empty when the snapshot predates M-PML — readers
    /// must tolerate empty vecs (restore_psi skips them then).
    pub bt_u: Vec<f32>,
    pub at_u: Vec<f32>,
    pub bt_v: Vec<f32>,
    pub at_v: Vec<f32>,
    pub psi_t: Vec<Vec<f32>>,
}

impl Cpml {
    /// Clone all six faces' coefficients and ψ arrays (in face order
    /// xlo, xhi, ylo, yhi, zlo, zhi) for upload to an external backend.
    pub fn snapshot(&self) -> Vec<PmlFaceData> {
        self.faces
            .iter()
            .map(|f| PmlFaceData {
                axis: f.axis,
                lo: f.lo,
                npml: f.npml,
                n1: f.n1,
                n2: f.n2,
                b_e: f.b_e.clone(),
                a_e: f.a_e.clone(),
                b_h: f.b_h.clone(),
                a_h: f.a_h.clone(),
                kinv: f.kinv.clone(),
                sigma: f.sigma.clone(),
                psi_a: f.psi_a.clone(),
                psi_b: f.psi_b.clone(),
                psi_c: f.psi_c.clone(),
                psi_d: f.psi_d.clone(),
                bt_u: f.bt_u.clone(),
                at_u: f.at_u.clone(),
                bt_v: f.bt_v.clone(),
                at_v: f.at_v.clone(),
                psi_t: f.psi_t.clone(),
            })
            .collect()
    }

    /// Restore ψ state from a snapshot taken on an identically
    /// built Cpml (checkpoint/resume for long runs). Coefficients
    /// are deterministic from construction and are not touched.
    pub fn restore_psi(&mut self, snap: &[PmlFaceData]) {
        assert_eq!(snap.len(), self.faces.len());
        for (f, s) in self.faces.iter_mut().zip(snap.iter()) {
            assert_eq!(f.psi_a.len(), s.psi_a.len());
            f.psi_a.copy_from_slice(&s.psi_a);
            f.psi_b.copy_from_slice(&s.psi_b);
            f.psi_c.copy_from_slice(&s.psi_c);
            f.psi_d.copy_from_slice(&s.psi_d);
            // M-PML tangential ψ: snapshots written before M-PML
            // carry empty psi_t; leave the (zero) state then.
            if s.psi_t.len() == f.psi_t.len() {
                for (dst, src) in f.psi_t.iter_mut().zip(s.psi_t.iter()) {
                    if dst.len() == src.len() {
                        dst.copy_from_slice(src);
                    }
                }
            }
        }
    }
}

/// Is index `m` of a component axis with extent `extent` inside a
/// PML shell of depth `npml` (either side)? Used by the M-PML
/// tangential corrections to detect edge/corner overlap cells.
#[inline]
fn in_shell(m: usize, extent: usize, npml: usize) -> bool {
    m < npml || m + npml >= extent
}

/// Which M-PML overlap treatment is active (env `CPML_MPML_FORM`).
/// `Partition` is the phase-1/2a family (face-local tangential ψ,
/// hard switch or ramped switch in overlap cells). `Summed` is
/// phase 2b: one convolution per (component, derivative) with
/// summed-profile coefficients, driven by a shell-volume pass.
#[derive(Clone, Copy, PartialEq)]
enum MpmlForm {
    Partition,
    Summed,
}

/// Phase-2c α-combination rule for the summed form (env
/// `CPML_MPML_ALPHA`, default `sum`). Phase 2b summed α naively
/// (α^Σ = Σ α_F), but α is graded opposite to σ (α_max at the
/// interface), so α^Σ jumps ≈ α_max across a shell-interface
/// plane inside another face's shell while σ^Σ there is
/// grading-mediated — the surviving discontinuity and the 2b
/// nucleation seed (design note §9). The alternatives:
/// `Weighted` takes the σ-weighted mean of the faces' α (the
/// jump becomes σ-mediated); `Dominant` takes the α of the face
/// contributing the largest σ_F,u (ties → the own-axis face);
/// `Tzero` lets only the own-axis face's α enter α^Σ (tangential
/// contributions carry α = 0).
#[derive(Clone, Copy, PartialEq)]
enum MpmlAlpha {
    Sum,
    Weighted,
    Dominant,
    Tzero,
}

/// Phase-2b per-template state: one ψ convolution for one
/// (corrected component, derivative direction) pair, with static
/// per-point coefficients (b^Σ, a^Σ, 1 − 1/κ^Σ) laid out on the
/// corrected component's own array shape (flat index = the
/// component's grid index). Entries outside the PML shell are
/// zero, and the pass skips them (ψ there is identically 0).
struct SummedTemplate {
    b: Vec<f32>,
    a: Vec<f32>,
    omk: Vec<f32>,
    psi: Vec<f32>,
}

/// Depth of component index `m` (axis extent `extent`) inside a
/// PML shell of depth `npml`: `Some((lo, p))` with p = 0 at the
/// outer boundary — the same convention the ψ machinery uses
/// (E-side components have extent n along their normal, H-side
/// n + 1, and this formula reproduces both, since it is stated
/// purely in terms of the component's own extent).
#[inline]
fn shell_depth(m: usize, extent: usize, npml: usize) -> Option<(bool, usize)> {
    if m < npml {
        Some((true, m))
    } else if m + npml >= extent {
        Some((false, extent - 1 - m))
    } else {
        None
    }
}

/// Build the phase-2b summed-profile templates. Fixed template
/// order (component, derivative direction):
/// 0: Ey←∂/∂x, 1: Ez←∂/∂x, 2: Ex←∂/∂y, 3: Ez←∂/∂y,
/// 4: Ex←∂/∂z, 5: Ey←∂/∂z (E side);
/// 6: Hy←∂/∂x, 7: Hz←∂/∂x, 8: Hx←∂/∂y, 9: Hz←∂/∂y,
/// 10: Hx←∂/∂z, 11: Hy←∂/∂z (H side).
///
/// Per point of the corrected component (own indices/extents per
/// axis), the summed profiles are
///   σ_u^Σ = Σ_{faces F ∋ point} σ_F,u,  σ_F,u = σ_F(p_F) if F's
///           normal is u, else R_u·σ_F(p_F);
///   α^Σ   = combined from the faces' α_F(p_F) by `alpha_rule`
///           (phase 2b: plain sum; phase 2c alternatives — see
///           `MpmlAlpha`);
///   κ^Σ   = κ_F(p_F) for the face F whose normal is u, else 1;
/// and (b^Σ, a^Σ) follow the same Gedney recursion as the normal
/// coefficients, with the expression written identically so that
/// in single-shell regions the values reduce to the existing
/// normal (u = owning axis) and tangential (otherwise) pairs.
fn build_summed(
    faces: &[PmlFace],
    npml: usize,
    dt: f32,
    (nx, ny, nz): (usize, usize, usize),
    ratio: f32,
    ratio_z: f32,
    alpha_rule: MpmlAlpha,
) -> Vec<SummedTemplate> {
    let eps0 = 8.854_187_817e-12f32;
    // (direction u, component dims (ni, nj, nk)) per template.
    let spec: [(u8, (usize, usize, usize)); 12] = [
        (0, (nx, ny + 1, nz)),
        (0, (nx, ny, nz + 1)),
        (1, (nx + 1, ny, nz)),
        (1, (nx, ny, nz + 1)),
        (2, (nx + 1, ny, nz)),
        (2, (nx, ny + 1, nz)),
        (0, (nx + 1, ny, nz + 1)),
        (0, (nx + 1, ny + 1, nz)),
        (1, (nx, ny + 1, nz + 1)),
        (1, (nx + 1, ny + 1, nz)),
        (2, (nx, ny + 1, nz + 1)),
        (2, (nx + 1, ny, nz + 1)),
    ];
    let ratio_for = |u: u8| -> f32 { if u == 2 { ratio_z } else { ratio } };
    spec.iter()
        .map(|&(u, (ni, nj, nk))| {
            let n = ni * nj * nk;
            let mut b = vec![0.0f32; n];
            let mut a = vec![0.0f32; n];
            let mut omk = vec![0.0f32; n];
            let ru = ratio_for(u);
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        let idx = (k * nj + j) * ni + i;
                        let mut sig = 0.0f32;
                        let mut kap = 1.0f32;
                        let mut any = false;
                        // Per-axis face contributions to direction
                        // u: (σ_F,u, α_F), present only where the
                        // point is in that axis's shell. Collected
                        // in axis order so the `Sum` rule below
                        // performs the phase-2b additions in
                        // exactly the phase-2b order.
                        let mut contrib = [(0.0f32, 0.0f32); 3];
                        let mut present = [false; 3];
                        for (axis, (m, e)) in
                            [(i, ni), (j, nj), (k, nk)].into_iter().enumerate()
                        {
                            if let Some((lo, p)) = shell_depth(m, e, npml) {
                                any = true;
                                present[axis] = true;
                                let f = &faces[2 * axis + if lo { 0 } else { 1 }];
                                let csig = if axis as u8 == u {
                                    f.sigma[p]
                                } else {
                                    ru * f.sigma[p]
                                };
                                sig += csig;
                                contrib[axis] = (csig, f.alpha[p]);
                                if axis as u8 == u {
                                    kap = f.kappa[p];
                                }
                            }
                        }
                        if !any {
                            continue;
                        }
                        let alp = match alpha_rule {
                            MpmlAlpha::Sum => {
                                // Absent faces hold (0, 0), so this
                                // is Σ over present faces in axis
                                // order — the phase-2b sum.
                                contrib[0].1 + contrib[1].1 + contrib[2].1
                            }
                            MpmlAlpha::Weighted => {
                                if sig > 0.0 {
                                    (contrib[0].0 * contrib[0].1
                                        + contrib[1].0 * contrib[1].1
                                        + contrib[2].0 * contrib[2].1)
                                        / sig
                                } else {
                                    0.0
                                }
                            }
                            MpmlAlpha::Dominant => {
                                // Scan the own axis first; strict
                                // `>` keeps the earliest maximum, so
                                // ties go to the own-axis face.
                                let mut best: Option<(f32, f32)> = None;
                                for axis in
                                    [u as usize, (u as usize + 1) % 3, (u as usize + 2) % 3]
                                {
                                    if present[axis]
                                        && best.is_none_or(|b| contrib[axis].0 > b.0)
                                    {
                                        best = Some(contrib[axis]);
                                    }
                                }
                                best.map_or(0.0, |b| b.1)
                            }
                            MpmlAlpha::Tzero => contrib[u as usize].1,
                        };
                        let bv = (-(sig / kap + alp) * dt / eps0).exp();
                        b[idx] = bv;
                        a[idx] = if sig + kap * alp > 0.0 {
                            (bv - 1.0) * sig / (kap * (sig + kap * alp))
                        } else {
                            0.0
                        };
                        omk[idx] = 1.0 - 1.0 / kap;
                    }
                }
            }
            SummedTemplate {
                b,
                a,
                omk,
                psi: vec![0.0; n],
            }
        })
        .collect()
}

/// Phase-2a tangential ramp weight for one tangential axis:
/// smoothstep from 0 at the partition switch boundary to 1 at
/// `width` cells inside the single-shell region. Only called for
/// active cells (`!in_shell(m, extent, npg)`), so `m - npg` and
/// `extent - npg - 1 - m` are the cell's distances to the lo/hi
/// switch boundaries (0 = adjacent to the boundary). `width = 0`
/// disables the ramp (returns exactly 1.0, so the phase-1
/// partition path is bit-identical).
#[inline]
fn ramp_w(m: usize, extent: usize, npg: usize, width: usize) -> f32 {
    if width == 0 {
        return 1.0;
    }
    let q = (m - npg).min(extent - npg - 1 - m);
    if q >= width {
        1.0
    } else {
        let t = q as f32 / width as f32;
        t * t * (3.0 - 2.0 * t)
    }
}

/// Scale one field component's samples inside a face's shell by the
/// per-depth `factors` (tangential-damping prototype). `dims` are the
/// component's own array dimensions (row-major, x fastest); the depth
/// p of a sample is measured from that component's outermost sample
/// along the face normal (p = 0 outermost, as in the ψ machinery).
fn damp_component(
    arr: &mut [f32],
    dims: (usize, usize, usize),
    axis: u8,
    lo: bool,
    npml: usize,
    factors: &[f32],
) {
    let (ni, nj, nk) = dims;
    // (na, nb, nc): the shell axis first, then the two transverse axes
    // in (i, j, k) order.
    let (na, nb, nc) = match axis {
        0 => (ni, nj, nk),
        1 => (nj, ni, nk),
        _ => (nk, ni, nj),
    };
    for a in 0..na {
        let p = if lo {
            if a < npml {
                a
            } else {
                continue;
            }
        } else if a + npml >= na {
            na - 1 - a
        } else {
            continue;
        };
        let f = factors[p];
        if f == 1.0 {
            continue;
        }
        for c in 0..nc {
            for b in 0..nb {
                let (i, j, k) = match axis {
                    0 => (a, b, c),
                    1 => (b, a, c),
                    _ => (b, c, a),
                };
                arr[(k * nj + j) * ni + i] *= f;
            }
        }
    }
}

/// Term indices for the phase-3a ADE form: one stretched
/// derivative per (component, derivative direction) pair, in the
/// same order as `build_summed`'s template spec.
pub mod ade_terms {
    pub const EY_X: usize = 0;
    pub const EZ_X: usize = 1;
    pub const EX_Y: usize = 2;
    pub const EZ_Y: usize = 3;
    pub const EX_Z: usize = 4;
    pub const EY_Z: usize = 5;
    pub const HY_X: usize = 6;
    pub const HZ_X: usize = 7;
    pub const HX_Y: usize = 8;
    pub const HZ_Y: usize = 9;
    pub const HX_Z: usize = 10;
    pub const HY_Z: usize = 11;
    pub const COUNT: usize = 12;
}

/// One ADE term: the stretched derivative for one (component,
/// direction) pair, with coefficients and ψ state stored on
/// shell slots. Slot layout: `slot = (side·npml + p)·trans + t`,
/// where `side` 0 = lo shell, 1 = hi shell, `p` is the depth per
/// `shell_depth` on the component's own index/extent, and `t` is
/// the component's transverse flat index (for direction u: the
/// component coordinates along the other two axes, row-major with
/// the lower axis fastest — u=0: `k·dims.1 + j`; u=1:
/// `k·dims.0 + i`; u=2: `j·dims.0 + i`). The solver's update
/// loops compute the same `t` from their loop indices, so a term
/// touches each of its points exactly once per update.
pub struct AdeTerm {
    /// Component extent along the derivative axis (the
    /// shell-depth extent).
    extent: usize,
    /// Transverse slot count: component volume / extent.
    trans: usize,
    b: Vec<f32>,
    a: Vec<f32>,
    kinv: Vec<f32>,
    psi: Vec<f32>,
}

/// Phase-3a whole-volume ADE-CPML state (normal profiles only).
/// Coefficients follow the same Gedney recursion the face
/// templates use (`PmlFace::new`): `b = exp(−(σ/κ + α)·dt/ε₀)`,
/// `a = (b−1)·σ/(κ·(σ + κ·α))`; the E-side and H-side recursions
/// coincide through the impedance identity σ*/(κ·μ₀) =
/// σ/(κ·ε₀), so one (b, a, 1/κ) triple per (direction, point)
/// serves both field updates — the same identity the correction
/// form relies on for its shared face profiles. All twelve terms
/// carry the standard Gedney coefficients; term `EX_Y` is not
/// special-cased (it reproduced the correction form's
/// `correct_e_y` sign anomaly during phases 3a/3b, until the
/// anomaly itself was fixed in production — see `build_ade`).
pub struct AdeCpml {
    npml: usize,
    terms: Vec<AdeTerm>,
}

impl AdeCpml {
    /// Advance this term's ψ at the shell point addressed by
    /// (`m`, `trans`) — `m` the component index along the term's
    /// axis, `trans` the transverse flat index — and return the
    /// stretched derivative `(1/κ)·d + ψ_new`. Points outside the
    /// shell return `d` unchanged (interior updates untouched).
    #[inline]
    pub fn stretch(&mut self, term: usize, m: usize, trans: usize, d: f32) -> f32 {
        let npml = self.npml;
        let t = &mut self.terms[term];
        let (side, p) = if m < npml {
            (0usize, m)
        } else if m + npml >= t.extent {
            (1usize, t.extent - 1 - m)
        } else {
            return d;
        };
        let slot = (side * npml + p) * t.trans + trans;
        let psi = t.b[slot] * t.psi[slot] + t.a[slot] * d;
        t.psi[slot] = psi;
        t.kinv[slot] * d + psi
    }

    /// ψ state per term (checkpoint support).
    pub fn psi_snapshot(&self) -> Vec<Vec<f32>> {
        self.terms.iter().map(|t| t.psi.clone()).collect()
    }

    /// Restore ψ state from `psi_snapshot` output.
    pub fn restore_psi(&mut self, snap: &[Vec<f32>]) {
        assert_eq!(snap.len(), self.terms.len(), "ADE snapshot term count");
        for (t, s) in self.terms.iter_mut().zip(snap) {
            assert_eq!(s.len(), t.psi.len(), "ADE snapshot slot count");
            t.psi.copy_from_slice(s);
        }
    }

    /// Total ψ/coefficient storage in bytes (diagnostics).
    pub fn storage_bytes(&self) -> usize {
        self.terms
            .iter()
            .map(|t| (t.b.len() + t.a.len() + t.kinv.len() + t.psi.len()) * 4)
            .sum()
    }
}

/// Build the ADE coefficient/state fields for the grid from the
/// six faces' directional profiles. `tangential` gives, per
/// derivative direction u, the proportion of the OTHER faces'
/// profiles added into direction u's total (σ, α) — the phase-3b
/// hook. Phase 3a passes all zeros: each direction's total
/// profile is exactly its own axis face's normal profile, the
/// folded operator equals the correction form's operator, and
/// the builder is the only place 3b changes.
///
/// Coefficients are INACTIVE (b = 1, a = 0, 1/κ = 1, so `stretch`
/// returns `d` exactly) at the component points the correction
/// form's face-local pass for this term never touches: every
/// template loops its transverse component coordinates over
/// `[0, grid_extent)`, so the component plane at index
/// `grid_extent` on a transverse axis whose component extent is
/// `grid_extent + 1` is left plain (e.g. `correct_e_x` never
/// corrects Ey at j = ny). The mask reproduces that fringe
/// behaviour point for point.
fn build_ade(
    faces: &[PmlFace],
    npml: usize,
    dt: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    tangential: [f32; 3],
) -> AdeCpml {
    use ade_terms::*;
    let eps0 = 8.854_187_817e-12f32;
    let grid_ext = [nx, ny, nz];
    // (term, direction u, component dims) — build_summed's order.
    let spec: [(usize, usize, (usize, usize, usize)); COUNT] = [
        (EY_X, 0, (nx, ny + 1, nz)),
        (EZ_X, 0, (nx, ny, nz + 1)),
        (EX_Y, 1, (nx + 1, ny, nz)),
        (EZ_Y, 1, (nx, ny, nz + 1)),
        (EX_Z, 2, (nx + 1, ny, nz)),
        (EY_Z, 2, (nx, ny + 1, nz)),
        (HY_X, 0, (nx + 1, ny, nz + 1)),
        (HZ_X, 0, (nx + 1, ny + 1, nz)),
        (HX_Y, 1, (nx, ny + 1, nz + 1)),
        (HZ_Y, 1, (nx + 1, ny + 1, nz)),
        (HX_Z, 2, (nx, ny + 1, nz + 1)),
        (HY_Z, 2, (nx + 1, ny, nz + 1)),
    ];
    let mut terms: Vec<AdeTerm> = Vec::with_capacity(COUNT);
    for (slot_idx, &(term_id, u, dims)) in spec.iter().enumerate() {
        debug_assert_eq!(slot_idx, term_id);
        let dims3 = [dims.0, dims.1, dims.2];
        let extent = dims3[u];
        let vol = dims.0 * dims.1 * dims.2;
        let trans_size = vol / extent;
        let nslots = 2 * npml * trans_size;
        let mut b = vec![1.0f32; nslots];
        let mut a = vec![0.0f32; nslots];
        let mut kinv = vec![1.0f32; nslots];
        // Transverse axes in slot order: (first, second) with the
        // transverse index t = coord_second·dims_first + coord_first.
        let (w1, w2) = match u {
            0 => (1usize, 2usize),
            1 => (0usize, 2usize),
            _ => (0usize, 1usize),
        };
        for side in 0..2usize {
            for p in 0..npml {
                let m = if side == 0 { p } else { extent - 1 - p };
                // Own-axis face (faces are xlo, xhi, ylo, yhi,
                // zlo, zhi); the slot's p IS the shell_depth of m
                // on this extent, by construction.
                let own = &faces[2 * u + side];
                for t in 0..trans_size {
                    let c1 = t % dims3[w1];
                    let c2 = t / dims3[w1];
                    // Fringe mask (see fn docs): the face-local
                    // passes never touch transverse component
                    // index == grid extent.
                    if c1 >= grid_ext[w1] || c2 >= grid_ext[w2] {
                        continue;
                    }
                    // Component point in grid coordinates.
                    let mut pt = [0usize; 3];
                    pt[u] = m;
                    pt[w1] = c1;
                    pt[w2] = c2;
                    // Total directional profile at the point:
                    // own normal profile + tangential parts (zero
                    // in 3a). κ stays the own axis's, as in the
                    // summed form (§9).
                    let mut sigma = own.sigma[p];
                    let mut alpha = own.alpha[p];
                    let kappa = own.kappa[p];
                    let r = tangential[u];
                    if r != 0.0 {
                        for w in [w1, w2] {
                            if let Some((lo, pw)) = shell_depth(pt[w], grid_ext[w], npml) {
                                let f = &faces[2 * w + if lo { 0 } else { 1 }];
                                sigma += r * f.sigma[pw];
                                alpha += r * f.alpha[pw];
                            }
                        }
                    }
                    let slot = (side * npml + p) * trans_size + t;
                    let bv = (-(sigma / kappa + alpha) * dt / eps0).exp();
                    b[slot] = bv;
                    a[slot] = if sigma + kappa * alpha > 0.0 {
                        (bv - 1.0) * sigma / (kappa * (sigma + kappa * alpha))
                    } else {
                        0.0
                    };
                    // Every term carries the standard Gedney
                    // direct coefficient 1/κ. Term EX_Y is NOT
                    // special-cased: during phases 3a/3b its slots
                    // reproduced (or, under the retired
                    // CPML_ADE_TEXTBOOK toggle, corrected) the
                    // sign anomaly the correction form's
                    // `correct_e_y` then carried — total
                    // cb·(−(2 − 1/κ)·d + ψ) instead of
                    // cb·(−(d/κ + ψ)), the seed of the mode-2
                    // late-time instability. The anomaly was fixed
                    // in `correct_e_y` itself on 2026-10-22
                    // (design note §13), so the faithful form and
                    // the textbook form now coincide; the
                    // regression tests below pin EX_Y's slots to
                    // 1/κ so the special case cannot creep back.
                    kinv[slot] = 1.0 / kappa;
                }
            }
        }
        terms.push(AdeTerm {
            extent,
            trans: trans_size,
            b,
            a,
            kinv,
            psi: vec![0.0; nslots],
        });
    }
    AdeCpml { npml, terms }
}

impl Cpml {
    /// Build CPML for a grid. Each face's σ grading uses the cell size
    /// along that face's normal axis (grids may be anisotropic).
    pub fn new(g: &YeeGrid, npml: usize, dt: f32) -> Self {
        // M-PML proportion coefficients (probe-only env gates;
        // production default is off). `CPML_MPML=1` without an
        // explicit ratio selects 0.25, mid-range of the values the
        // M-PML literature reports (0.1–1.0).
        let env_f32 = |key: &str| -> Option<f32> {
            std::env::var(key).ok().and_then(|s| s.parse().ok())
        };
        let mpml_ratio = env_f32("CPML_MPML_RATIO").unwrap_or_else(|| {
            if env_f32("CPML_MPML").unwrap_or(0.0) != 0.0 {
                0.25
            } else {
                0.0
            }
        });
        let mpml_ratio_z = env_f32("CPML_MPML_RATIO_Z").unwrap_or(mpml_ratio);
        // Tangential proportion for a derivative along `axis`:
        // the z direction carries its own coefficient because our
        // slab/mode geometry is in-plane (xy); every other axis
        // uses the in-plane ratio.
        let ratio_for = |axis: u8| -> f32 {
            if axis == 2 {
                mpml_ratio_z
            } else {
                mpml_ratio
            }
        };
        // Cyclic tangential axes (u, v) per face normal n.
        let uv = [(1u8, 2u8), (2, 0), (0, 1)];
        let mut faces = Vec::with_capacity(6);
        faces.push(PmlFace::new(
            0,
            true,
            npml,
            g.ny,
            g.nz,
            g.dx,
            dt,
            ratio_for(uv[0].0),
            ratio_for(uv[0].1),
        ));
        faces.push(PmlFace::new(
            0,
            false,
            npml,
            g.ny,
            g.nz,
            g.dx,
            dt,
            ratio_for(uv[0].0),
            ratio_for(uv[0].1),
        ));
        faces.push(PmlFace::new(
            1,
            true,
            npml,
            g.nx,
            g.nz,
            g.dy,
            dt,
            ratio_for(uv[1].0),
            ratio_for(uv[1].1),
        ));
        faces.push(PmlFace::new(
            1,
            false,
            npml,
            g.nx,
            g.nz,
            g.dy,
            dt,
            ratio_for(uv[1].0),
            ratio_for(uv[1].1),
        ));
        faces.push(PmlFace::new(
            2,
            true,
            npml,
            g.nx,
            g.ny,
            g.dz,
            dt,
            ratio_for(uv[2].0),
            ratio_for(uv[2].1),
        ));
        faces.push(PmlFace::new(
            2,
            false,
            npml,
            g.nx,
            g.ny,
            g.dz,
            dt,
            ratio_for(uv[2].0),
            ratio_for(uv[2].1),
        ));
        let mpml_guard = std::env::var("CPML_MPML_GUARD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0usize);
        let mpml_ramp = std::env::var("CPML_MPML_RAMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0usize);
        let tangential_ratio = std::env::var("CPML_TANGENTIAL_RATIO")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let mpml_form = match std::env::var("CPML_MPML_FORM").ok().as_deref() {
            Some("summed") => MpmlForm::Summed,
            _ => MpmlForm::Partition,
        };
        let mpml_alpha = match std::env::var("CPML_MPML_ALPHA").ok().as_deref() {
            Some("weighted") => MpmlAlpha::Weighted,
            Some("dominant") => MpmlAlpha::Dominant,
            Some("tzero") => MpmlAlpha::Tzero,
            _ => MpmlAlpha::Sum,
        };
        let summed = if mpml_form == MpmlForm::Summed
            && (mpml_ratio != 0.0 || mpml_ratio_z != 0.0)
        {
            build_summed(
                &faces,
                npml,
                dt,
                (g.nx, g.ny, g.nz),
                mpml_ratio,
                mpml_ratio_z,
                mpml_alpha,
            )
        } else {
            Vec::new()
        };
        // Phase 3a: whole-volume ADE form (env `CPML_FORM=ade`),
        // normal profiles only — the tangential part of every
        // directional total profile is zero in this phase (see
        // `build_ade`). The correction form stays the default.
        // (The phase-3b `CPML_ADE_TEXTBOOK` toggle is retired:
        // since the 2026-10-22 `correct_e_y` fix the faithful
        // operator IS the textbook operator — §13.)
        let ade = if std::env::var("CPML_FORM").ok().as_deref() == Some("ade") {
            if mpml_ratio != 0.0 || mpml_ratio_z != 0.0 {
                eprintln!(
                    "warning: CPML_FORM=ade ignores CPML_MPML_* tangential terms \
                     in phase 3a (normal profiles only)"
                );
            }
            Some(build_ade(
                &faces,
                npml,
                dt,
                g.nx,
                g.ny,
                g.nz,
                [0.0; 3],
            ))
        } else {
            None
        };
        Self {
            faces,
            tangential_ratio,
            mpml_ratio,
            mpml_ratio_z,
            mpml_guard,
            mpml_ramp,
            mpml_form,
            mpml_alpha,
            summed,
            ade,
        }
    }

    /// CPML form this Cpml was built with: "ade" (phase 3a,
    /// env `CPML_FORM=ade`) or "correction" (default).
    pub fn form(&self) -> &'static str {
        if self.ade.is_some() {
            "ade"
        } else {
            "correction"
        }
    }

    /// Whether the phase-3a ADE form is active.
    pub fn ade_enabled(&self) -> bool {
        self.ade.is_some()
    }

    /// Mutable ADE state for the solver's folded update loops.
    pub fn ade_mut(&mut self) -> Option<&mut AdeCpml> {
        self.ade.as_mut()
    }

    /// ADE ψ snapshot (per term), for checkpointing; `None`
    /// unless the ADE form is active.
    pub fn ade_psi_snapshot(&self) -> Option<Vec<Vec<f32>>> {
        self.ade.as_ref().map(|a| a.psi_snapshot())
    }

    /// Restore ADE ψ state from `ade_psi_snapshot` output.
    /// Panics if the ADE form is not active — a checkpoint that
    /// carries ADE state must never be restored silently into a
    /// correction-form run (or vice versa).
    pub fn ade_restore_psi(&mut self, snap: &[Vec<f32>]) {
        self.ade
            .as_mut()
            .expect("ADE checkpoint state restored without CPML_FORM=ade")
            .restore_psi(snap);
    }

    /// Tangential-damping ratio R this Cpml was built with
    /// (0 = off; see `damp_tangential`). External backends read it
    /// to mirror the CPU step sequence.
    pub fn tangential_ratio(&self) -> f32 {
        self.tangential_ratio
    }

    /// M-PML proportion coefficients (in-plane, z) this Cpml was
    /// built with; (0, 0) = off.
    pub fn mpml_ratios(&self) -> (f32, f32) {
        (self.mpml_ratio, self.mpml_ratio_z)
    }

    /// Phase-2a tangential ramp width W (cells) this Cpml was
    /// built with; 0 = off (hard partition switch).
    pub fn mpml_ramp(&self) -> usize {
        self.mpml_ramp
    }

    /// M-PML overlap treatment this Cpml was built with:
    /// "partition" (phase 1/2a) or "summed" (phase 2b).
    pub fn mpml_form(&self) -> &'static str {
        match self.mpml_form {
            MpmlForm::Summed => "summed",
            MpmlForm::Partition => "partition",
        }
    }

    /// Phase-2c α-combination rule for the summed form:
    /// "sum" (phase 2b), "weighted", "dominant", or "tzero".
    pub fn mpml_alpha(&self) -> &'static str {
        match self.mpml_alpha {
            MpmlAlpha::Sum => "sum",
            MpmlAlpha::Weighted => "weighted",
            MpmlAlpha::Dominant => "dominant",
            MpmlAlpha::Tzero => "tzero",
        }
    }

    fn mpml_on(&self) -> bool {
        self.mpml_ratio != 0.0 || self.mpml_ratio_z != 0.0
    }

    /// Tangential-damping prototype: matched electric+magnetic loss
    /// inside each face's shell (see the module docs for the design
    /// and its shortcomings vs true M-PML). Every E and H sample in a
    /// shell is scaled by `exp(−R·σ(p)·Δt/ε₀)` once per step, with
    /// σ(p) the face's normal-direction profile at the sample's depth
    /// p; samples in edge/corner overlaps are scaled by each owning
    /// face in turn (conductivities add). No-op when R = 0, so the
    /// production path is bit-identical with the env unset.
    pub fn damp_tangential(&self, g: &mut YeeGrid, dt: f32) {
        let r = self.tangential_ratio;
        if r == 0.0 {
            return;
        }
        let eps0 = 8.854_187_817e-12f32;
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        for f in &self.faces {
            let factors: Vec<f32> = f
                .sigma
                .iter()
                .map(|s| (-r * s * dt / eps0).exp())
                .collect();
            damp_component(&mut g.ex, (nx + 1, ny, nz), f.axis, f.lo, f.npml, &factors);
            damp_component(&mut g.ey, (nx, ny + 1, nz), f.axis, f.lo, f.npml, &factors);
            damp_component(&mut g.ez, (nx, ny, nz + 1), f.axis, f.lo, f.npml, &factors);
            damp_component(&mut g.hx, (nx, ny + 1, nz + 1), f.axis, f.lo, f.npml, &factors);
            damp_component(&mut g.hy, (nx + 1, ny, nz + 1), f.axis, f.lo, f.npml, &factors);
            damp_component(&mut g.hz, (nx + 1, ny + 1, nz), f.axis, f.lo, f.npml, &factors);
        }
    }

    /// E-field PML correction after the standard E update.
    ///
    /// Per face/component: `ΔE = Cb·(s·((1/κ−1)·∂F/∂n) + s·ψ)`,
    /// i.e. the stretched derivative is `(1/κ)·∂F/∂n + ψ` with `ψ`
    /// entering UNSCALED; `ψ ← b·ψ + a·∂F/∂n`.
    pub fn correct_e(&mut self, g: &mut YeeGrid, cb_x: &[f32], cb_y: &[f32], cb_z: &[f32]) {
        if self.mpml_on() && self.mpml_form == MpmlForm::Summed {
            self.correct_e_summed(g, cb_x, cb_y, cb_z);
            return;
        }
        for fi in 0..self.faces.len() {
            let (axis, lo, npml, n1, n2) = {
                let f = &self.faces[fi];
                (f.axis, f.lo, f.npml, f.n1, f.n2)
            };
            match axis {
                0 => self.correct_e_x(fi, lo, npml, n1, n2, g, cb_y, cb_z),
                1 => self.correct_e_y(fi, lo, npml, n1, n2, g, cb_x, cb_z),
                _ => self.correct_e_z(fi, lo, npml, n1, n2, g, cb_x, cb_y),
            }
            if self.mpml_on() {
                let npg = npml + self.mpml_guard;
                let ramp = self.mpml_ramp;
                match axis {
                    0 => self.correct_e_mpml_x(fi, lo, npml, npg, ramp, g, cb_x, cb_y, cb_z),
                    1 => self.correct_e_mpml_y(fi, lo, npml, npg, ramp, g, cb_x, cb_y, cb_z),
                    _ => self.correct_e_mpml_z(fi, lo, npml, npg, ramp, g, cb_x, cb_y, cb_z),
                }
            }
        }
    }

    /// x-normal face: correct Ey (via dHz/dx) and Ez (via dHy/dx).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_x(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        ny: usize,
        nz: usize,
        g: &mut YeeGrid,
        cb_y: &[f32],
        cb_z: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let nx = g.nx;
        let dx = g.dx;
        for p in 0..npml {
            let be = f.b_e[p];
            let ae = f.a_e[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for j in 0..ny {
                for k in 0..nz {
                    // Ey has x-size nx; outermost layer is i=nx−1 (hi) or i=0 (lo).
                    let ie = if lo { p } else { nx - 1 - p };
                    // dHz/dx as update_e pairs it with Ey(ie):
                    // (hz[ie] − hz[ie−1])/dx, 0 outside.
                    let hz_lo = if ie > 0 { g.hz[g.hz_idx(ie - 1, j, k)] } else { 0.0 };
                    let dhzdx = (g.hz[g.hz_idx(ie, j, k)] - hz_lo) / dx;
                    let pi = f.psi_idx(p, j, k);
                    let psi = be * f.psi_a[pi] + ae * dhzdx;
                    f.psi_a[pi] = psi;
                    let ei = g.ey_idx(ie, j, k);
                    g.ey[ei] += cb_y[ei] * (-omk * dhzdx + psi);

                    // Ez: dHy/dx as update_e pairs it with Ez(ie):
                    // (hy[ie] − hy[ie−1])/dx, 0 outside.
                    let hy_lo = if ie > 0 { g.hy[g.hy_idx(ie - 1, j, k)] } else { 0.0 };
                    let dhydx = (g.hy[g.hy_idx(ie, j, k)] - hy_lo) / dx;
                    let psi2 = be * f.psi_b[pi] + ae * dhydx;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ez_idx(ie, j, k);
                    g.ez[ei2] += cb_z[ei2] * (omk * dhydx - psi2);
                }
            }
        }
    }

    /// y-normal face: correct Ex (via dHz/dy) and Ez (via dHx/dy).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_y(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        nx: usize,
        nz: usize,
        g: &mut YeeGrid,
        cb_x: &[f32],
        cb_z: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let ny = g.ny;
        let dy = g.dy;
        for p in 0..npml {
            let be = f.b_e[p];
            let ae = f.a_e[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for i in 0..nx {
                for k in 0..nz {
                    let je = if lo { p } else { ny - 1 - p };
                    // Ex: dHz/dy as update_e pairs it with Ex(je):
                    // (hz[je] − hz[je−1])/dy, 0 outside.
                    let hz_lo = if je > 0 { g.hz[g.hz_idx(i, je - 1, k)] } else { 0.0 };
                    let dhzdy = (g.hz[g.hz_idx(i, je, k)] - hz_lo) / dy;
                    let pi = f.psi_idx(p, i, k);
                    let psi = be * f.psi_a[pi] + ae * dhzdy;
                    f.psi_a[pi] = psi;
                    let ei = g.ex_idx(i, je, k);
                    // Ex: bracket has −∂Hz/∂y (s=−1): Δ = Cb·((1−1/κ)·d − ψ).
                    // (This term applied the s=+1 formula until the
                    // 2026-10-22 fix — the mode-2 anomaly, §13.)
                    g.ex[ei] += cb_x[ei] * (omk * dhzdy - psi);

                    let hx_lo = if je > 0 { g.hx[g.hx_idx(i, je - 1, k)] } else { 0.0 };
                    let dhxdy = (g.hx[g.hx_idx(i, je, k)] - hx_lo) / dy;
                    let psi2 = be * f.psi_b[pi] + ae * dhxdy;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ez_idx(i, je, k);
                    g.ez[ei2] += cb_z[ei2] * (-omk * dhxdy + psi2);
                }
            }
        }
    }

    /// z-normal face: correct Ex (via dHy/dz) and Ey (via dHx/dz).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_z(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        nx: usize,
        ny: usize,
        g: &mut YeeGrid,
        cb_x: &[f32],
        cb_y: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let nz = g.nz;
        let dz = g.dz;
        for p in 0..npml {
            let be = f.b_e[p];
            let ae = f.a_e[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for i in 0..nx {
                for j in 0..ny {
                    let ke = if lo { p } else { nz - 1 - p };
                    // dHy/dz as update_e pairs it with Ex(ke):
                    // (hy[ke] − hy[ke−1])/dz, 0 outside.
                    let hy_lo = if ke > 0 { g.hy[g.hy_idx(i, j, ke - 1)] } else { 0.0 };
                    let dhydz = (g.hy[g.hy_idx(i, j, ke)] - hy_lo) / dz;
                    let pi = f.psi_idx(p, i, j);
                    let psi = be * f.psi_a[pi] + ae * dhydz;
                    f.psi_a[pi] = psi;
                    let ei = g.ex_idx(i, j, ke);
                    g.ex[ei] += cb_x[ei] * (-omk * dhydz + psi);

                    let hx_lo = if ke > 0 { g.hx[g.hx_idx(i, j, ke - 1)] } else { 0.0 };
                    let dhxdz = (g.hx[g.hx_idx(i, j, ke)] - hx_lo) / dz;
                    let psi2 = be * f.psi_b[pi] + ae * dhxdz;
                    f.psi_b[pi] = psi2;
                    let ei2 = g.ey_idx(i, j, ke);
                    // Ey: bracket has −∂Hx/∂z (s=−1): Δ = Cb·((1−1/κ)·d − ψ).
                    g.ey[ei2] += cb_y[ei2] * (omk * dhxdz - psi2);
                }
            }
        }
    }

    /// H-field PML correction after the standard H update.
    ///
    /// Per face/component: `ΔH = −(dt/μ₀)·(−s·(1−1/κ)·∂F/∂n − s·ψ)`,
    /// where `s = ±1` is the sign of `∂F/∂n` in the bracket
    /// (`H −= c·[s·∂F/∂n + …]`) and `ψ` enters UNSCALED (the stretched
    /// derivative is `(1/κ)·∂F/∂n + ψ`); `ψ ← b·ψ + a·∂F/∂n`.
    pub fn correct_h(&mut self, g: &mut YeeGrid, dt: f32) {
        let mu0 = 4.0e-7 * std::f32::consts::PI;
        let c = dt / mu0;
        if self.mpml_on() && self.mpml_form == MpmlForm::Summed {
            self.correct_h_summed(g, c);
            return;
        }
        for fi in 0..self.faces.len() {
            let (axis, lo, npml, n1, n2) = {
                let f = &self.faces[fi];
                (f.axis, f.lo, f.npml, f.n1, f.n2)
            };
            match axis {
                0 => self.correct_h_x(fi, lo, npml, n1, n2, g, c),
                1 => self.correct_h_y(fi, lo, npml, n1, n2, g, c),
                _ => self.correct_h_z(fi, lo, npml, n1, n2, g, c),
            }
            if self.mpml_on() {
                let npg = npml + self.mpml_guard;
                let ramp = self.mpml_ramp;
                match axis {
                    0 => self.correct_h_mpml_x(fi, lo, npml, npg, ramp, g, c),
                    1 => self.correct_h_mpml_y(fi, lo, npml, npg, ramp, g, c),
                    _ => self.correct_h_mpml_z(fi, lo, npml, npg, ramp, g, c),
                }
            }
        }
    }

    /// x-normal: correct Hy (via dEz/dx) and Hz (via dEy/dx).
    fn correct_h_x(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        ny: usize,
        nz: usize,
        g: &mut YeeGrid,
        c: f32,
    ) {
        let f = &mut self.faces[fi];
        let nx = g.nx;
        let dx = g.dx;
        for p in 0..npml {
            let bh = f.b_h[p];
            let ah = f.a_h[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for j in 0..ny {
                for k in 0..nz {
                    // Hy has x-size nx+1; outermost is i=nx (hi) or i=0 (lo).
                    let ih = if lo { p } else { nx - p };
                    // dEz/dx as update_h pairs it with Hy(ih):
                    // (ez[ih+1] − ez[ih])/dx, out-of-range = 0 (mirrors
                    // update_h's guards: at ih=nx both samples are 0).
                    let e_hi = if ih + 1 < nx { g.ez[g.ez_idx(ih + 1, j, k)] } else { 0.0 };
                    let e_lo = if ih < nx { g.ez[g.ez_idx(ih, j, k)] } else { 0.0 };
                    // Hy: bracket has +∂Ez/∂x (s=+1): Δ = −c·(−(1−1/κ)·d + ψ).
                    let dezdx = (e_hi - e_lo) / dx;
                    let pi = f.psi_idx(p, j, k);
                    let psi = bh * f.psi_c[pi] + ah * dezdx;
                    f.psi_c[pi] = psi;
                    let hi = g.hy_idx(ih, j, k);
                    g.hy[hi] -= c * (-omk * dezdx + psi);

                    // Hz: dEy/dx as update_h pairs it with Hz(ih):
                    // (ey[ih+1] − ey[min(ih,nx−1)])/dx, out-of-range = 0.
                    let e2_hi = if ih + 1 < nx { g.ey[g.ey_idx(ih + 1, j, k)] } else { 0.0 };
                    let e2_lo = g.ey[g.ey_idx(ih.min(nx - 1), j, k)];
                    let deydx = (e2_hi - e2_lo) / dx;
                    let psi2 = bh * f.psi_d[pi] + ah * deydx;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hz_idx(ih, j, k);
                    g.hz[hi2] -= c * (omk * deydx - psi2);
                }
            }
        }
    }

    /// y-normal: correct Hx (via dEz/dy) and Hz (via dEx/dy).
    fn correct_h_y(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        nx: usize,
        nz: usize,
        g: &mut YeeGrid,
        c: f32,
    ) {
        let f = &mut self.faces[fi];
        let ny = g.ny;
        let dy = g.dy;
        for p in 0..npml {
            let bh = f.b_h[p];
            let ah = f.a_h[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for i in 0..nx {
                for k in 0..nz {
                    let jh = if lo { p } else { ny - p };
                    // dEz/dy as update_h pairs it with Hx(jh):
                    // (ez[jh+1] − ez[jh])/dy, out-of-range = 0.
                    let e_hi = if jh + 1 < ny { g.ez[g.ez_idx(i, jh + 1, k)] } else { 0.0 };
                    let e_lo = if jh < ny { g.ez[g.ez_idx(i, jh, k)] } else { 0.0 };
                    let dezdy = (e_hi - e_lo) / dy;
                    let pi = f.psi_idx(p, i, k);
                    let psi = bh * f.psi_c[pi] + ah * dezdy;
                    f.psi_c[pi] = psi;
                    let hi = g.hx_idx(i, jh, k);
                    g.hx[hi] -= c * (omk * dezdy - psi);

                    // dEx/dy as update_h pairs it with Hz(jh):
                    // (ex[jh+1] − ex[min(jh,ny−1)])/dy, out-of-range = 0.
                    let e2_hi = if jh + 1 < ny { g.ex[g.ex_idx(i, jh + 1, k)] } else { 0.0 };
                    let e2_lo = g.ex[g.ex_idx(i, jh.min(ny - 1), k)];
                    let dexdy = (e2_hi - e2_lo) / dy;
                    let psi2 = bh * f.psi_d[pi] + ah * dexdy;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hz_idx(i, jh, k);
                    // Hz: bracket has +∂Ex/∂y (s=+1): Δ = −c·(−(1−1/κ)·d + ψ).
                    g.hz[hi2] -= c * (-omk * dexdy + psi2);
                }
            }
        }
    }

    /// z-normal: correct Hx (via dEy/dz) and Hy (via dEx/dz).
    fn correct_h_z(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        nx: usize,
        ny: usize,
        g: &mut YeeGrid,
        c: f32,
    ) {
        let f = &mut self.faces[fi];
        let nz = g.nz;
        let dz = g.dz;
        for p in 0..npml {
            let bh = f.b_h[p];
            let ah = f.a_h[p];
            let ki = f.kinv[p];
            let omk = 1.0 - ki;
            for i in 0..nx {
                for j in 0..ny {
                    let kh = if lo { p } else { nz - p };
                    // dEy/dz as update_h pairs it with Hx(kh):
                    // (ey[kh+1] − ey[kh])/dz, out-of-range = 0.
                    let e_hi = if kh + 1 < nz { g.ey[g.ey_idx(i, j, kh + 1)] } else { 0.0 };
                    let e_lo = if kh < nz { g.ey[g.ey_idx(i, j, kh)] } else { 0.0 };
                    let deydz = (e_hi - e_lo) / dz;
                    let pi = f.psi_idx(p, i, j);
                    let psi = bh * f.psi_c[pi] + ah * deydz;
                    f.psi_c[pi] = psi;
                    let hi = g.hx_idx(i, j, kh);
                    // Hx: bracket has +∂Ey/∂z (s=+1): Δ = −c·(−(1−1/κ)·d + ψ).
                    g.hx[hi] -= c * (-omk * deydz + psi);

                    // dEx/dz as update_h pairs it with Hy(kh):
                    // (ex[kh+1] − ex[kh])/dz, out-of-range = 0.
                    let e2_hi = if kh + 1 < nz { g.ex[g.ex_idx(i, j, kh + 1)] } else { 0.0 };
                    let e2_lo = if kh < nz { g.ex[g.ex_idx(i, j, kh)] } else { 0.0 };
                    let dexdz = (e2_hi - e2_lo) / dz;
                    let psi2 = bh * f.psi_d[pi] + ah * dexdz;
                    f.psi_d[pi] = psi2;
                    let hi2 = g.hy_idx(i, j, kh);
                    g.hy[hi2] -= c * (omk * dexdz - psi2);
                }
            }
        }
    }

    // ---- M-PML tangential corrections (prototype; see module docs).
    //
    // Every term below is the tangential instance of an existing
    // (component, field, derivative) template from the correct_e_*/
    // correct_h_* functions above: identical derivative pairing,
    // identical ψ sign c2, but the ψ recursion uses THIS face's
    // tangential coefficients (bt/at of the derivative's axis) and
    // κ_t = 1 makes the (1−1/κ) direct term vanish, leaving
    // `Cb·c2·ψ_t` (E side) / `−c·c2·ψ_t` (H side). H-side derivative
    // pairings come in the same two boundary cases as the templates:
    // case A (forward difference, out-of-range samples 0) and case B
    // (forward difference with the low sample clamped to the last
    // in-range index).
    //
    // Overlap partition rule (see module docs): each block skips
    // cells whose updated component lies in another axis's shell
    // (`in_shell` on the component's own staggered indices/extents
    // along the two non-normal axes), so no derivative ever carries
    // two ψ convolutions.
    //
    // Phase 2a: when `ramp` > 0, each block additionally scales its
    // ψ drives and its applied correction by
    // `w = ramp_w(m1, e1, npg, ramp) * ramp_w(m2, e2, npg, ramp)`
    // for the component's tangential indices/extents, so the
    // tangential operator turns on smoothly from the switch
    // boundary instead of jumping (see module docs).

    /// M-PML E corrections, x-normal face (u = y, v = z).
    /// Ex via ∂Hz/∂y (+1, t0), Ex via ∂Hy/∂z (+1, t1),
    /// Ey via ∂Hx/∂z (−1, t2), Ez via ∂Hx/∂y (+1, t3).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_mpml_x(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        npg: usize,
        ramp: usize,
        g: &mut YeeGrid,
        cb_x: &[f32],
        cb_y: &[f32],
        cb_z: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let (dy, dz) = (g.dy, g.dz);
        for p in 0..npml {
            let ie = if lo { p } else { nx - 1 - p };
            let ih = if lo { p } else { nx - p };
            let (bu, au) = (f.bt_u[p], f.at_u[p]);
            let (bv, av) = (f.bt_v[p], f.at_v[p]);
            // Ex at (ih, j, k).
            for j in 0..ny {
                for k in 0..nz {
                    if in_shell(j, ny, npg) || in_shell(k, nz, npg) {
                        continue;
                    }
                    let w = ramp_w(j, ny, npg, ramp) * ramp_w(k, nz, npg, ramp);
                    let hz_lo = if j > 0 { g.hz[g.hz_idx(ih, j - 1, k)] } else { 0.0 };
                    let dhzdy = (g.hz[g.hz_idx(ih, j, k)] - hz_lo) / dy;
                    let pi = f.psi_t_idx(p, j, k);
                    let psi0 = bu * f.psi_t[0][pi] + w * au * dhzdy;
                    f.psi_t[0][pi] = psi0;
                    let hy_lo = if k > 0 { g.hy[g.hy_idx(ih, j, k - 1)] } else { 0.0 };
                    let dhydz = (g.hy[g.hy_idx(ih, j, k)] - hy_lo) / dz;
                    let psi1 = bv * f.psi_t[1][pi] + w * av * dhydz;
                    f.psi_t[1][pi] = psi1;
                    let ei = g.ex_idx(ih, j, k);
                    g.ex[ei] += cb_x[ei] * (w * (psi0 + psi1));
                }
            }
            // Ey at (ie, j, k).
            for j in 0..=ny {
                for k in 0..nz {
                    if in_shell(j, ny + 1, npg) || in_shell(k, nz, npg) {
                        continue;
                    }
                    let w = ramp_w(j, ny + 1, npg, ramp) * ramp_w(k, nz, npg, ramp);
                    let hx_lo = if k > 0 { g.hx[g.hx_idx(ie, j, k - 1)] } else { 0.0 };
                    let dhxdz = (g.hx[g.hx_idx(ie, j, k)] - hx_lo) / dz;
                    let pi = f.psi_t_idx(p, j, k);
                    let psi2 = bv * f.psi_t[2][pi] + w * av * dhxdz;
                    f.psi_t[2][pi] = psi2;
                    let ei = g.ey_idx(ie, j, k);
                    g.ey[ei] -= cb_y[ei] * (w * psi2);
                }
            }
            // Ez at (ie, j, k).
            for j in 0..ny {
                for k in 0..=nz {
                    if in_shell(j, ny, npg) || in_shell(k, nz + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(j, ny, npg, ramp) * ramp_w(k, nz + 1, npg, ramp);
                    let hx_lo = if j > 0 { g.hx[g.hx_idx(ie, j - 1, k)] } else { 0.0 };
                    let dhxdy = (g.hx[g.hx_idx(ie, j, k)] - hx_lo) / dy;
                    let pi = f.psi_t_idx(p, j, k);
                    let psi3 = bu * f.psi_t[3][pi] + w * au * dhxdy;
                    f.psi_t[3][pi] = psi3;
                    let ei = g.ez_idx(ie, j, k);
                    g.ez[ei] += cb_z[ei] * (w * psi3);
                }
            }
        }
    }

    /// M-PML E corrections, y-normal face (u = z, v = x).
    /// Ey via ∂Hx/∂z (−1, t0), Ey via ∂Hz/∂x (+1, t1),
    /// Ez via ∂Hy/∂x (−1, t2), Ex via ∂Hy/∂z (+1, t3).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_mpml_y(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        npg: usize,
        ramp: usize,
        g: &mut YeeGrid,
        cb_x: &[f32],
        cb_y: &[f32],
        cb_z: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let (dx, dz) = (g.dx, g.dz);
        for p in 0..npml {
            let je = if lo { p } else { ny - 1 - p };
            let jh = if lo { p } else { ny - p };
            let (bu, au) = (f.bt_u[p], f.at_u[p]);
            let (bv, av) = (f.bt_v[p], f.at_v[p]);
            // Ey at (i, jh, k).
            for i in 0..nx {
                for k in 0..nz {
                    if in_shell(i, nx, npg) || in_shell(k, nz, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx, npg, ramp) * ramp_w(k, nz, npg, ramp);
                    let hx_lo = if k > 0 { g.hx[g.hx_idx(i, jh, k - 1)] } else { 0.0 };
                    let dhxdz = (g.hx[g.hx_idx(i, jh, k)] - hx_lo) / dz;
                    let pi = f.psi_t_idx(p, i, k);
                    let psi0 = bu * f.psi_t[0][pi] + w * au * dhxdz;
                    f.psi_t[0][pi] = psi0;
                    let hz_lo = if i > 0 { g.hz[g.hz_idx(i - 1, jh, k)] } else { 0.0 };
                    let dhzdx = (g.hz[g.hz_idx(i, jh, k)] - hz_lo) / dx;
                    let psi1 = bv * f.psi_t[1][pi] + w * av * dhzdx;
                    f.psi_t[1][pi] = psi1;
                    let ei = g.ey_idx(i, jh, k);
                    g.ey[ei] += cb_y[ei] * (w * (-psi0 + psi1));
                }
            }
            // Ez at (i, je, k).
            for i in 0..nx {
                for k in 0..=nz {
                    if in_shell(i, nx, npg) || in_shell(k, nz + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx, npg, ramp) * ramp_w(k, nz + 1, npg, ramp);
                    let hy_lo = if i > 0 { g.hy[g.hy_idx(i - 1, je, k)] } else { 0.0 };
                    let dhydx = (g.hy[g.hy_idx(i, je, k)] - hy_lo) / dx;
                    let pi = f.psi_t_idx(p, i, k);
                    let psi2 = bv * f.psi_t[2][pi] + w * av * dhydx;
                    f.psi_t[2][pi] = psi2;
                    let ei = g.ez_idx(i, je, k);
                    g.ez[ei] -= cb_z[ei] * (w * psi2);
                }
            }
            // Ex at (i, je, k).
            for i in 0..=nx {
                for k in 0..nz {
                    if in_shell(i, nx + 1, npg) || in_shell(k, nz, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx + 1, npg, ramp) * ramp_w(k, nz, npg, ramp);
                    let hy_lo = if k > 0 { g.hy[g.hy_idx(i, je, k - 1)] } else { 0.0 };
                    let dhydz = (g.hy[g.hy_idx(i, je, k)] - hy_lo) / dz;
                    let pi = f.psi_t_idx(p, i, k);
                    let psi3 = bu * f.psi_t[3][pi] + w * au * dhydz;
                    f.psi_t[3][pi] = psi3;
                    let ei = g.ex_idx(i, je, k);
                    g.ex[ei] += cb_x[ei] * (w * psi3);
                }
            }
        }
    }

    /// M-PML E corrections, z-normal face (u = x, v = y).
    /// Ez via ∂Hy/∂x (−1, t0), Ez via ∂Hx/∂y (+1, t1),
    /// Ex via ∂Hz/∂y (+1, t2), Ey via ∂Hz/∂x (+1, t3).
    #[allow(clippy::too_many_arguments)]
    fn correct_e_mpml_z(
        &mut self,
        fi: usize,
        lo: bool,
        npml: usize,
        npg: usize,
        ramp: usize,
        g: &mut YeeGrid,
        cb_x: &[f32],
        cb_y: &[f32],
        cb_z: &[f32],
    ) {
        let f = &mut self.faces[fi];
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let (dx, dy) = (g.dx, g.dy);
        for p in 0..npml {
            let ke = if lo { p } else { nz - 1 - p };
            let kh = if lo { p } else { nz - p };
            let (bu, au) = (f.bt_u[p], f.at_u[p]);
            let (bv, av) = (f.bt_v[p], f.at_v[p]);
            // Ez at (i, j, kh).
            for i in 0..nx {
                for j in 0..ny {
                    if in_shell(i, nx, npg) || in_shell(j, ny, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx, npg, ramp) * ramp_w(j, ny, npg, ramp);
                    let hy_lo = if i > 0 { g.hy[g.hy_idx(i - 1, j, kh)] } else { 0.0 };
                    let dhydx = (g.hy[g.hy_idx(i, j, kh)] - hy_lo) / dx;
                    let pi = f.psi_t_idx(p, i, j);
                    let psi0 = bu * f.psi_t[0][pi] + w * au * dhydx;
                    f.psi_t[0][pi] = psi0;
                    let hx_lo = if j > 0 { g.hx[g.hx_idx(i, j - 1, kh)] } else { 0.0 };
                    let dhxdy = (g.hx[g.hx_idx(i, j, kh)] - hx_lo) / dy;
                    let psi1 = bv * f.psi_t[1][pi] + w * av * dhxdy;
                    f.psi_t[1][pi] = psi1;
                    let ei = g.ez_idx(i, j, kh);
                    g.ez[ei] += cb_z[ei] * (w * (-psi0 + psi1));
                }
            }
            // Ex at (i, j, ke).
            for i in 0..=nx {
                for j in 0..ny {
                    if in_shell(i, nx + 1, npg) || in_shell(j, ny, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx + 1, npg, ramp) * ramp_w(j, ny, npg, ramp);
                    let hz_lo = if j > 0 { g.hz[g.hz_idx(i, j - 1, ke)] } else { 0.0 };
                    let dhzdy = (g.hz[g.hz_idx(i, j, ke)] - hz_lo) / dy;
                    let pi = f.psi_t_idx(p, i, j);
                    let psi2 = bv * f.psi_t[2][pi] + w * av * dhzdy;
                    f.psi_t[2][pi] = psi2;
                    let ei = g.ex_idx(i, j, ke);
                    g.ex[ei] += cb_x[ei] * (w * psi2);
                }
            }
            // Ey at (i, j, ke).
            for i in 0..nx {
                for j in 0..=ny {
                    if in_shell(i, nx, npg) || in_shell(j, ny + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx, npg, ramp) * ramp_w(j, ny + 1, npg, ramp);
                    let hz_lo = if i > 0 { g.hz[g.hz_idx(i - 1, j, ke)] } else { 0.0 };
                    let dhzdx = (g.hz[g.hz_idx(i, j, ke)] - hz_lo) / dx;
                    let pi = f.psi_t_idx(p, i, j);
                    let psi3 = bu * f.psi_t[3][pi] + w * au * dhzdx;
                    f.psi_t[3][pi] = psi3;
                    let ei = g.ey_idx(i, j, ke);
                    g.ey[ei] += cb_y[ei] * (w * psi3);
                }
            }
        }
    }

    /// M-PML H corrections, x-normal face (u = y, v = z).
    /// Hx via ∂Ez/∂y (A, −1, t4), Hx via ∂Ey/∂z (A, +1, t5),
    /// Hy via ∂Ex/∂z (B, −1, t6), Hz via ∂Ex/∂y (B, +1, t7).
    #[allow(clippy::too_many_arguments)]
    fn correct_h_mpml_x(&mut self, fi: usize, lo: bool, npml: usize, npg: usize, ramp: usize, g: &mut YeeGrid, c: f32) {
        let f = &mut self.faces[fi];
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let (dy, dz) = (g.dy, g.dz);
        for p in 0..npml {
            let ie = if lo { p } else { nx - 1 - p };
            let ih = if lo { p } else { nx - p };
            let (bu, au) = (f.bt_u[p], f.at_u[p]);
            let (bv, av) = (f.bt_v[p], f.at_v[p]);
            // Hx at (ie, j, k).
            for j in 0..=ny {
                for k in 0..=nz {
                    if in_shell(j, ny + 1, npg) || in_shell(k, nz + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(j, ny + 1, npg, ramp) * ramp_w(k, nz + 1, npg, ramp);
                    let e_hi = if j + 1 < ny { g.ez[g.ez_idx(ie, j + 1, k)] } else { 0.0 };
                    let e_lo = if j < ny { g.ez[g.ez_idx(ie, j, k)] } else { 0.0 };
                    let dezdy = (e_hi - e_lo) / dy;
                    let pi = f.psi_t_idx(p, j, k);
                    let psi4 = bu * f.psi_t[4][pi] + w * au * dezdy;
                    f.psi_t[4][pi] = psi4;
                    let e2_hi = if k + 1 < nz { g.ey[g.ey_idx(ie, j, k + 1)] } else { 0.0 };
                    let e2_lo = if k < nz { g.ey[g.ey_idx(ie, j, k)] } else { 0.0 };
                    let deydz = (e2_hi - e2_lo) / dz;
                    let psi5 = bv * f.psi_t[5][pi] + w * av * deydz;
                    f.psi_t[5][pi] = psi5;
                    let hi = g.hx_idx(ie, j, k);
                    g.hx[hi] -= c * (w * (-psi4 + psi5));
                }
            }
            // Hy at (ih, j, k).
            for j in 0..ny {
                for k in 0..=nz {
                    if in_shell(j, ny, npg) || in_shell(k, nz + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(j, ny, npg, ramp) * ramp_w(k, nz + 1, npg, ramp);
                    let e_hi = if k + 1 < nz { g.ex[g.ex_idx(ih, j, k + 1)] } else { 0.0 };
                    let e_lo = g.ex[g.ex_idx(ih, j, k.min(nz - 1))];
                    let dexdz = (e_hi - e_lo) / dz;
                    let pi = f.psi_t_idx(p, j, k);
                    let psi6 = bv * f.psi_t[6][pi] + w * av * dexdz;
                    f.psi_t[6][pi] = psi6;
                    let hi = g.hy_idx(ih, j, k);
                    g.hy[hi] += c * (w * psi6);
                }
            }
            // Hz at (ih, j, k).
            for j in 0..=ny {
                for k in 0..nz {
                    if in_shell(j, ny + 1, npg) || in_shell(k, nz, npg) {
                        continue;
                    }
                    let w = ramp_w(j, ny + 1, npg, ramp) * ramp_w(k, nz, npg, ramp);
                    let e_hi = if j + 1 < ny { g.ex[g.ex_idx(ih, j + 1, k)] } else { 0.0 };
                    let e_lo = g.ex[g.ex_idx(ih, j.min(ny - 1), k)];
                    let dexdy = (e_hi - e_lo) / dy;
                    let pi = f.psi_t_idx(p, j, k);
                    let psi7 = bu * f.psi_t[7][pi] + w * au * dexdy;
                    f.psi_t[7][pi] = psi7;
                    let hi = g.hz_idx(ih, j, k);
                    g.hz[hi] -= c * (w * psi7);
                }
            }
        }
    }

    /// M-PML H corrections, y-normal face (u = z, v = x).
    /// Hy via ∂Ex/∂z (B, −1, t4), Hy via ∂Ez/∂x (A, +1, t5),
    /// Hz via ∂Ey/∂x (B, −1, t6), Hx via ∂Ey/∂z (A, +1, t7).
    #[allow(clippy::too_many_arguments)]
    fn correct_h_mpml_y(&mut self, fi: usize, lo: bool, npml: usize, npg: usize, ramp: usize, g: &mut YeeGrid, c: f32) {
        let f = &mut self.faces[fi];
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let (dx, dz) = (g.dx, g.dz);
        for p in 0..npml {
            let je = if lo { p } else { ny - 1 - p };
            let jh = if lo { p } else { ny - p };
            let (bu, au) = (f.bt_u[p], f.at_u[p]);
            let (bv, av) = (f.bt_v[p], f.at_v[p]);
            // Hy at (i, je, k).
            for i in 0..=nx {
                for k in 0..=nz {
                    if in_shell(i, nx + 1, npg) || in_shell(k, nz + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx + 1, npg, ramp) * ramp_w(k, nz + 1, npg, ramp);
                    let e_hi = if k + 1 < nz { g.ex[g.ex_idx(i, je, k + 1)] } else { 0.0 };
                    let e_lo = g.ex[g.ex_idx(i, je, k.min(nz - 1))];
                    let dexdz = (e_hi - e_lo) / dz;
                    let pi = f.psi_t_idx(p, i, k);
                    let psi4 = bu * f.psi_t[4][pi] + w * au * dexdz;
                    f.psi_t[4][pi] = psi4;
                    let e2_hi = if i + 1 < nx { g.ez[g.ez_idx(i + 1, je, k)] } else { 0.0 };
                    let e2_lo = if i < nx { g.ez[g.ez_idx(i, je, k)] } else { 0.0 };
                    let dezdx = (e2_hi - e2_lo) / dx;
                    let psi5 = bv * f.psi_t[5][pi] + w * av * dezdx;
                    f.psi_t[5][pi] = psi5;
                    let hi = g.hy_idx(i, je, k);
                    g.hy[hi] -= c * (w * (-psi4 + psi5));
                }
            }
            // Hz at (i, jh, k).
            for i in 0..=nx {
                for k in 0..nz {
                    if in_shell(i, nx + 1, npg) || in_shell(k, nz, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx + 1, npg, ramp) * ramp_w(k, nz, npg, ramp);
                    let e_hi = if i + 1 < nx { g.ey[g.ey_idx(i + 1, jh, k)] } else { 0.0 };
                    let e_lo = g.ey[g.ey_idx(i.min(nx - 1), jh, k)];
                    let deydx = (e_hi - e_lo) / dx;
                    let pi = f.psi_t_idx(p, i, k);
                    let psi6 = bv * f.psi_t[6][pi] + w * av * deydx;
                    f.psi_t[6][pi] = psi6;
                    let hi = g.hz_idx(i, jh, k);
                    g.hz[hi] += c * (w * psi6);
                }
            }
            // Hx at (i, jh, k).
            for i in 0..nx {
                for k in 0..=nz {
                    if in_shell(i, nx, npg) || in_shell(k, nz + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx, npg, ramp) * ramp_w(k, nz + 1, npg, ramp);
                    let e_hi = if k + 1 < nz { g.ey[g.ey_idx(i, jh, k + 1)] } else { 0.0 };
                    let e_lo = if k < nz { g.ey[g.ey_idx(i, jh, k)] } else { 0.0 };
                    let deydz = (e_hi - e_lo) / dz;
                    let pi = f.psi_t_idx(p, i, k);
                    let psi7 = bu * f.psi_t[7][pi] + w * au * deydz;
                    f.psi_t[7][pi] = psi7;
                    let hi = g.hx_idx(i, jh, k);
                    g.hx[hi] -= c * (w * psi7);
                }
            }
        }
    }

    /// M-PML H corrections, z-normal face (u = x, v = y).
    /// Hz via ∂Ey/∂x (B, −1, t4), Hz via ∂Ex/∂y (B, +1, t5),
    /// Hx via ∂Ez/∂y (A, −1, t6), Hy via ∂Ez/∂x (A, +1, t7).
    #[allow(clippy::too_many_arguments)]
    fn correct_h_mpml_z(&mut self, fi: usize, lo: bool, npml: usize, npg: usize, ramp: usize, g: &mut YeeGrid, c: f32) {
        let f = &mut self.faces[fi];
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let (dx, dy) = (g.dx, g.dy);
        for p in 0..npml {
            let ke = if lo { p } else { nz - 1 - p };
            let kh = if lo { p } else { nz - p };
            let (bu, au) = (f.bt_u[p], f.at_u[p]);
            let (bv, av) = (f.bt_v[p], f.at_v[p]);
            // Hz at (i, j, ke).
            for i in 0..=nx {
                for j in 0..=ny {
                    if in_shell(i, nx + 1, npg) || in_shell(j, ny + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx + 1, npg, ramp) * ramp_w(j, ny + 1, npg, ramp);
                    let e_hi = if i + 1 < nx { g.ey[g.ey_idx(i + 1, j, ke)] } else { 0.0 };
                    let e_lo = g.ey[g.ey_idx(i.min(nx - 1), j, ke)];
                    let deydx = (e_hi - e_lo) / dx;
                    let pi = f.psi_t_idx(p, i, j);
                    let psi4 = bu * f.psi_t[4][pi] + w * au * deydx;
                    f.psi_t[4][pi] = psi4;
                    let e2_hi = if j + 1 < ny { g.ex[g.ex_idx(i, j + 1, ke)] } else { 0.0 };
                    let e2_lo = g.ex[g.ex_idx(i, j.min(ny - 1), ke)];
                    let dexdy = (e2_hi - e2_lo) / dy;
                    let psi5 = bv * f.psi_t[5][pi] + w * av * dexdy;
                    f.psi_t[5][pi] = psi5;
                    let hi = g.hz_idx(i, j, ke);
                    g.hz[hi] -= c * (w * (-psi4 + psi5));
                }
            }
            // Hx at (i, j, kh).
            for i in 0..nx {
                for j in 0..=ny {
                    if in_shell(i, nx, npg) || in_shell(j, ny + 1, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx, npg, ramp) * ramp_w(j, ny + 1, npg, ramp);
                    let e_hi = if j + 1 < ny { g.ez[g.ez_idx(i, j + 1, kh)] } else { 0.0 };
                    let e_lo = if j < ny { g.ez[g.ez_idx(i, j, kh)] } else { 0.0 };
                    let dezdy = (e_hi - e_lo) / dy;
                    let pi = f.psi_t_idx(p, i, j);
                    let psi6 = bv * f.psi_t[6][pi] + w * av * dezdy;
                    f.psi_t[6][pi] = psi6;
                    let hi = g.hx_idx(i, j, kh);
                    g.hx[hi] += c * (w * psi6);
                }
            }
            // Hy at (i, j, kh).
            for i in 0..=nx {
                for j in 0..ny {
                    if in_shell(i, nx + 1, npg) || in_shell(j, ny, npg) {
                        continue;
                    }
                    let w = ramp_w(i, nx + 1, npg, ramp) * ramp_w(j, ny, npg, ramp);
                    let e_hi = if i + 1 < nx { g.ez[g.ez_idx(i + 1, j, kh)] } else { 0.0 };
                    let e_lo = if i < nx { g.ez[g.ez_idx(i, j, kh)] } else { 0.0 };
                    let dezdx = (e_hi - e_lo) / dx;
                    let pi = f.psi_t_idx(p, i, j);
                    let psi7 = bu * f.psi_t[7][pi] + w * au * dezdx;
                    f.psi_t[7][pi] = psi7;
                    let hi = g.hy_idx(i, j, kh);
                    g.hy[hi] -= c * (w * psi7);
                }
            }
        }
    }

    // ---- Phase-2b summed-profile corrections (see module docs).
    //
    // Each block is one (component, derivative) template cloned
    // from the corresponding face-normal body above (correct_e_* /
    // correct_h_*): identical derivative pairing and correction
    // expression, but with the per-point summed-profile
    // coefficients of the template's SummedTemplate and loops over
    // the component's full index range, acting wherever the
    // template is active (a^Σ ≠ 0 or the (1 − 1/κ^Σ) direct term
    // nonzero; elsewhere ψ is identically 0 and the correction
    // vanishes). E corrections read only H and H corrections read
    // only E, so template order within a pass is immaterial.

    /// Phase-2b E corrections: templates 0–5 of `build_summed`.
    fn correct_e_summed(&mut self, g: &mut YeeGrid, cb_x: &[f32], cb_y: &[f32], cb_z: &[f32]) {
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let (dx, dy, dz) = (g.dx, g.dy, g.dz);
        // T0: Ey via ∂Hz/∂x (clone of correct_e_x, Ey part).
        {
            let t = &mut self.summed[0];
            for k in 0..nz {
                for j in 0..=ny {
                    for ie in 0..nx {
                        let ei = g.ey_idx(ie, j, k);
                        let (bv, av, omk) = (t.b[ei], t.a[ei], t.omk[ei]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let hz_lo = if ie > 0 { g.hz[g.hz_idx(ie - 1, j, k)] } else { 0.0 };
                        let d = (g.hz[g.hz_idx(ie, j, k)] - hz_lo) / dx;
                        let psi = bv * t.psi[ei] + av * d;
                        t.psi[ei] = psi;
                        g.ey[ei] += cb_y[ei] * (-omk * d + psi);
                    }
                }
            }
        }
        // T1: Ez via ∂Hy/∂x (clone of correct_e_x, Ez part).
        {
            let t = &mut self.summed[1];
            for k in 0..=nz {
                for j in 0..ny {
                    for ie in 0..nx {
                        let ei = g.ez_idx(ie, j, k);
                        let (bv, av, omk) = (t.b[ei], t.a[ei], t.omk[ei]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let hy_lo = if ie > 0 { g.hy[g.hy_idx(ie - 1, j, k)] } else { 0.0 };
                        let d = (g.hy[g.hy_idx(ie, j, k)] - hy_lo) / dx;
                        let psi = bv * t.psi[ei] + av * d;
                        t.psi[ei] = psi;
                        g.ez[ei] += cb_z[ei] * (omk * d - psi);
                    }
                }
            }
        }
        // T2: Ex via ∂Hz/∂y (clone of correct_e_y, Ex part).
        {
            let t = &mut self.summed[2];
            for k in 0..nz {
                for je in 0..ny {
                    for i in 0..=nx {
                        let ei = g.ex_idx(i, je, k);
                        let (bv, av, omk) = (t.b[ei], t.a[ei], t.omk[ei]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let hz_lo = if je > 0 { g.hz[g.hz_idx(i, je - 1, k)] } else { 0.0 };
                        let d = (g.hz[g.hz_idx(i, je, k)] - hz_lo) / dy;
                        let psi = bv * t.psi[ei] + av * d;
                        t.psi[ei] = psi;
                        g.ex[ei] += cb_x[ei] * (-omk * d + psi);
                    }
                }
            }
        }
        // T3: Ez via ∂Hx/∂y (clone of correct_e_y, Ez part).
        {
            let t = &mut self.summed[3];
            for k in 0..=nz {
                for je in 0..ny {
                    for i in 0..nx {
                        let ei = g.ez_idx(i, je, k);
                        let (bv, av, omk) = (t.b[ei], t.a[ei], t.omk[ei]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let hx_lo = if je > 0 { g.hx[g.hx_idx(i, je - 1, k)] } else { 0.0 };
                        let d = (g.hx[g.hx_idx(i, je, k)] - hx_lo) / dy;
                        let psi = bv * t.psi[ei] + av * d;
                        t.psi[ei] = psi;
                        g.ez[ei] += cb_z[ei] * (-omk * d + psi);
                    }
                }
            }
        }
        // T4: Ex via ∂Hy/∂z (clone of correct_e_z, Ex part).
        {
            let t = &mut self.summed[4];
            for ke in 0..nz {
                for j in 0..ny {
                    for i in 0..=nx {
                        let ei = g.ex_idx(i, j, ke);
                        let (bv, av, omk) = (t.b[ei], t.a[ei], t.omk[ei]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let hy_lo = if ke > 0 { g.hy[g.hy_idx(i, j, ke - 1)] } else { 0.0 };
                        let d = (g.hy[g.hy_idx(i, j, ke)] - hy_lo) / dz;
                        let psi = bv * t.psi[ei] + av * d;
                        t.psi[ei] = psi;
                        g.ex[ei] += cb_x[ei] * (-omk * d + psi);
                    }
                }
            }
        }
        // T5: Ey via ∂Hx/∂z (clone of correct_e_z, Ey part).
        {
            let t = &mut self.summed[5];
            for ke in 0..nz {
                for j in 0..=ny {
                    for i in 0..nx {
                        let ei = g.ey_idx(i, j, ke);
                        let (bv, av, omk) = (t.b[ei], t.a[ei], t.omk[ei]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let hx_lo = if ke > 0 { g.hx[g.hx_idx(i, j, ke - 1)] } else { 0.0 };
                        let d = (g.hx[g.hx_idx(i, j, ke)] - hx_lo) / dz;
                        let psi = bv * t.psi[ei] + av * d;
                        t.psi[ei] = psi;
                        g.ey[ei] += cb_y[ei] * (omk * d - psi);
                    }
                }
            }
        }
    }

    /// Phase-2b H corrections: templates 6–11 of `build_summed`.
    fn correct_h_summed(&mut self, g: &mut YeeGrid, c: f32) {
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let (dx, dy, dz) = (g.dx, g.dy, g.dz);
        // T6: Hy via ∂Ez/∂x (clone of correct_h_x, Hy part; case A).
        {
            let t = &mut self.summed[6];
            for k in 0..=nz {
                for j in 0..ny {
                    for ih in 0..=nx {
                        let hi = g.hy_idx(ih, j, k);
                        let (bv, av, omk) = (t.b[hi], t.a[hi], t.omk[hi]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let e_hi = if ih + 1 < nx { g.ez[g.ez_idx(ih + 1, j, k)] } else { 0.0 };
                        let e_lo = if ih < nx { g.ez[g.ez_idx(ih, j, k)] } else { 0.0 };
                        let d = (e_hi - e_lo) / dx;
                        let psi = bv * t.psi[hi] + av * d;
                        t.psi[hi] = psi;
                        g.hy[hi] -= c * (-omk * d + psi);
                    }
                }
            }
        }
        // T7: Hz via ∂Ey/∂x (clone of correct_h_x, Hz part; case B).
        {
            let t = &mut self.summed[7];
            for k in 0..nz {
                for j in 0..=ny {
                    for ih in 0..=nx {
                        let hi = g.hz_idx(ih, j, k);
                        let (bv, av, omk) = (t.b[hi], t.a[hi], t.omk[hi]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let e_hi = if ih + 1 < nx { g.ey[g.ey_idx(ih + 1, j, k)] } else { 0.0 };
                        let e_lo = g.ey[g.ey_idx(ih.min(nx - 1), j, k)];
                        let d = (e_hi - e_lo) / dx;
                        let psi = bv * t.psi[hi] + av * d;
                        t.psi[hi] = psi;
                        g.hz[hi] -= c * (omk * d - psi);
                    }
                }
            }
        }
        // T8: Hx via ∂Ez/∂y (clone of correct_h_y, Hx part; case A).
        {
            let t = &mut self.summed[8];
            for k in 0..nz {
                for jh in 0..=ny {
                    for i in 0..nx {
                        let hi = g.hx_idx(i, jh, k);
                        let (bv, av, omk) = (t.b[hi], t.a[hi], t.omk[hi]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let e_hi = if jh + 1 < ny { g.ez[g.ez_idx(i, jh + 1, k)] } else { 0.0 };
                        let e_lo = if jh < ny { g.ez[g.ez_idx(i, jh, k)] } else { 0.0 };
                        let d = (e_hi - e_lo) / dy;
                        let psi = bv * t.psi[hi] + av * d;
                        t.psi[hi] = psi;
                        g.hx[hi] -= c * (omk * d - psi);
                    }
                }
            }
        }
        // T9: Hz via ∂Ex/∂y (clone of correct_h_y, Hz part; case B).
        {
            let t = &mut self.summed[9];
            for k in 0..nz {
                for jh in 0..=ny {
                    for i in 0..=nx {
                        let hi = g.hz_idx(i, jh, k);
                        let (bv, av, omk) = (t.b[hi], t.a[hi], t.omk[hi]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let e_hi = if jh + 1 < ny { g.ex[g.ex_idx(i, jh + 1, k)] } else { 0.0 };
                        let e_lo = g.ex[g.ex_idx(i, jh.min(ny - 1), k)];
                        let d = (e_hi - e_lo) / dy;
                        let psi = bv * t.psi[hi] + av * d;
                        t.psi[hi] = psi;
                        g.hz[hi] -= c * (-omk * d + psi);
                    }
                }
            }
        }
        // T10: Hx via ∂Ey/∂z (clone of correct_h_z, Hx part; case A).
        {
            let t = &mut self.summed[10];
            for kh in 0..=nz {
                for j in 0..=ny {
                    for i in 0..nx {
                        let hi = g.hx_idx(i, j, kh);
                        let (bv, av, omk) = (t.b[hi], t.a[hi], t.omk[hi]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let e_hi = if kh + 1 < nz { g.ey[g.ey_idx(i, j, kh + 1)] } else { 0.0 };
                        let e_lo = if kh < nz { g.ey[g.ey_idx(i, j, kh)] } else { 0.0 };
                        let d = (e_hi - e_lo) / dz;
                        let psi = bv * t.psi[hi] + av * d;
                        t.psi[hi] = psi;
                        g.hx[hi] -= c * (-omk * d + psi);
                    }
                }
            }
        }
        // T11: Hy via ∂Ex/∂z (clone of correct_h_z, Hy part; case A).
        {
            let t = &mut self.summed[11];
            for kh in 0..=nz {
                for j in 0..ny {
                    for i in 0..=nx {
                        let hi = g.hy_idx(i, j, kh);
                        let (bv, av, omk) = (t.b[hi], t.a[hi], t.omk[hi]);
                        if av == 0.0 && omk == 0.0 {
                            continue;
                        }
                        let e_hi = if kh + 1 < nz { g.ex[g.ex_idx(i, j, kh + 1)] } else { 0.0 };
                        let e_lo = if kh < nz { g.ex[g.ex_idx(i, j, kh)] } else { 0.0 };
                        let d = (e_hi - e_lo) / dz;
                        let psi = bv * t.psi[hi] + av * d;
                        t.psi[hi] = psi;
                        g.hy[hi] -= c * (omk * d - psi);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Materials;

    /// Regression guard for the mode-2 anomaly (design note
    /// §11–§13): `correct_e_y`'s Ex term (∂Hz/∂y, bracket sign
    /// −1) must apply Δ = Cb·((1−1/κ)·d − ψ), so plain update +
    /// correction totals Cb·(−(d/κ + ψ)). With ψ⁰ = 0 and a
    /// constant ∂Hz/∂y = d = 1, one correction pass must move
    /// Ex in a y-shell cell at depth p by exactly
    /// Cb·((1−1/κ) − a_e)·d. The pre-fix code applied the s=+1
    /// formula and produced the negative of the first term.
    #[test]
    fn correct_e_y_ex_term_matches_gedney() {
        let (nx, ny, nz) = (8usize, 8usize, 8usize);
        let dt = 1e-12f32;
        let mut g = YeeGrid::new(nx, ny, nz, 1.0, 1.0, 1.0);
        let mat = Materials::vacuum(nx, ny, nz, dt);
        let npml = 2;
        let mut cpml = Cpml::new(&g, npml, dt);
        // ∂Hz/∂y = 1 everywhere (Hz linear in j).
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let idx = g.hz_idx(i, j, k);
                    g.hz[idx] = j as f32;
                }
            }
        }
        cpml.correct_e(&mut g, &mat.cb_x, &mat.cb_y, &mat.cb_z);
        // y-lo face is faces[2]; depth p = 1 → je = 1.
        let (i, k, p) = (3usize, 3usize, 1usize);
        let ei = g.ex_idx(i, p, k);
        let face = &cpml.faces[2];
        let omk = 1.0 - face.kinv[p];
        let expected = mat.cb_x[ei] * (omk - face.a_e[p]);
        let got = g.ex[ei];
        assert!(
            (got - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "Ex correction at y-lo depth {p}: got {got}, expected {expected} \
             (Gedney Δ = Cb·((1−1/κ)·d − ψ))"
        );
    }

    /// Builder-level guard: every ADE term's slots carry the
    /// standard Gedney direct coefficient 1/κ — in particular
    /// EX_Y must never again carry the anomaly's (2 − 1/κ)
    /// (design note §13). Checked at transverse index t = 0,
    /// which the fringe mask never touches.
    #[test]
    fn ade_terms_all_carry_inverse_kappa() {
        let (nx, ny, nz) = (10usize, 10usize, 10usize);
        let dt = 1e-12f32;
        let g = YeeGrid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3);
        let npml = 3;
        let cpml = Cpml::new(&g, npml, dt);
        let ade = build_ade(&cpml.faces, npml, dt, nx, ny, nz, [0.0; 3]);
        // Derivative direction u per term, in ade_terms order.
        let us = [0usize, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2];
        for (term, &u) in us.iter().enumerate() {
            let t = &ade.terms[term];
            for side in 0..2usize {
                let kappa = &cpml.faces[2 * u + side].kappa;
                for p in 0..npml {
                    let slot = (side * npml + p) * t.trans;
                    let want = 1.0 / kappa[p];
                    assert!(
                        (t.kinv[slot] - want).abs() <= 1e-6,
                        "term {term} side {side} depth {p}: kinv {} != 1/κ {want}",
                        t.kinv[slot]
                    );
                }
            }
        }
    }
}
