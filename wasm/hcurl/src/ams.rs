//! Auxiliary-space Maxwell (AMS) preconditioner for the H(curl) system.
//!
//! Pointwise smoothers (Jacobi, SGS) are blind to the large near-nullspace of
//! the discrete curl-curl operator — discrete gradients — which is why
//! BiCGSTAB needs ~10⁴ iterations on production meshes. AMS (Hiptmair–Xu)
//! adds a coarse correction posed on the nodal (H¹) auxiliary space:
//!
//! ```text
//! M⁻¹ r = S⁻¹ r + G (Gᵀ A G)⁻¹ Gᵀ r
//! ```
//!
//! where `S⁻¹` is one SGS sweep on the edge system, `G` is the discrete
//! gradient (free-edge-DOFs × active nodes): the edge-DOF of ∇φ on the
//! low→high edge (u,v) is the length-normalized line integral
//! `(φᵥ − φᵤ)/l_e`, matching the length-scaled Nédélec basis
//! (`N_e = s_e·l_e·(λa∇λb − λb∇λa)`). The nodal space is the full H¹
//! auxiliary space (no Dirichlet condition on PEC): the nullspace of the
//! constrained curl-curl operator is gradients of H¹ functions constant
//! on each PEC component, and the full nodal space captures
//! inter-component modes (e.g. the differential potential between the
//! two dipole arms) that a PEC-Dirichlet coarse space would miss. The inner nodal
//! `Aₙ = GᵀAG` is Laplacian-like — the curl-curl part vanishes identically
//! (`GᵀKG = 0`, the discrete de Rham sequence) — so the same SGS smoother
//! works well on it and the inner solve is cheap.
//!
//! Only the gradient auxiliary space is used. The full AMS carries a second,
//! vector-valued auxiliary space for the solenoidal part; that is future work
//! if this stage alone does not cut iterations enough.

use num_complex::Complex64;

use crate::basis::TET_EDGES;
use crate::budget::Budget;
use crate::csr::CsrMatrix;
use crate::edge::EdgeSystem;
use crate::error::Result;
use crate::krylov::{bicgstab, Preconditioner, PreconditionerData};
use crate::mesh::Mesh;

/// Loose inner tolerance: the correction only needs roughly the right
/// direction; the outer Krylov iteration cleans up the rest.
const INNER_TOL: f64 = 1e-2;
const INNER_MAX_ITER: u32 = 200;
/// Default fixed SGS sweeps for the inner nodal approximation (see
/// `inner_sweeps`).
const INNER_SWEEPS: u32 = 3;

/// Precomputed AMS operator for one assembled H(curl) system.
///
/// Built once per matrix via [`AmsPreconditioner::build`]; application is
/// [`AmsPreconditioner::apply_correction`] (the SGS smoother part lives in
/// `krylov::PreconditionerData`, applied additively by the caller).
#[derive(Debug, Clone)]
pub struct AmsPreconditioner {
    /// Discrete gradient: free-edge-DOFs × active nodes.
    g: CsrMatrix,
    /// Nodal Galerkin matrix `GᵀAG`.
    a_nodal: CsrMatrix,
    node_count: usize,
    /// PEC-boundary node mask (diagnostics: the true curl-curl nullspace
    /// is gradients of potentials constant on PEC).
    pec_node: Vec<bool>,
    /// Precomputed SGS diagonal data for `a_nodal` (built once).
    nodal_sgs: PreconditionerData,
    edge_dof_count: usize,
    /// Inner solve tolerance (default 1e-2; loose is fine for a preconditioner).
    pub inner_tol: f64,
    /// Inner solve iteration cap (default 200).
    pub inner_max_iter: u32,
    /// Fixed SGS sweeps for the inner nodal approximation (default 3).
    /// When nonzero, the inner solve is replaced by this many SGS sweeps:
    /// far cheaper than inner BiCGSTAB, and a *fixed* linear operator
    /// (better for the outer Krylov method). Set to 0 to use inner
    /// BiCGSTAB instead.
    pub inner_sweeps: u32,
}

impl AmsPreconditioner {
    /// Build the discrete gradient and the nodal auxiliary matrix for the
    /// assembled system `a` (must be the `n_free × n_free` H(curl) matrix
    /// for `sys`).
    pub fn build(mesh: &Mesh, sys: &EdgeSystem, a: &CsrMatrix) -> Result<Self> {
        let n_free = sys.free_dof_count;
        assert_eq!(a.nrows(), n_free, "AMS needs the free-DOF system matrix");
        assert_eq!(a.ncols(), n_free, "AMS needs a square system matrix");
        let nv = mesh.vertex_count();

        // Edge id -> (u, v) with u < v (global low→high orientation).
        let mut edge_uv = vec![(0u32, 0u32); sys.edge_count];
        let mut seen = vec![false; sys.edge_count];
        for (ord, &tet) in sys.active_tets.iter().enumerate() {
            let base = tet as usize * 4;
            for e in 0..6 {
                let id = sys.tet_edges[ord * 6 + e] as usize;
                if !seen[id] {
                    seen[id] = true;
                    let [la, lb] = TET_EDGES[e];
                    let va = mesh.tetrahedra[base + la];
                    let vb = mesh.tetrahedra[base + lb];
                    edge_uv[id] = (va.min(vb), va.max(vb));
                }
            }
        }

        // Edge lengths (physical), for the length-normalized discrete gradient.
        let mut edge_len = vec![0.0f64; sys.edge_count];
        for (id, &(u, v)) in edge_uv.iter().enumerate() {
            let pu = mesh.vertex(u);
            let pv = mesh.vertex(v);
            let dx = pv[0] - pu[0];
            let dy = pv[1] - pu[1];
            let dz = pv[2] - pu[2];
            edge_len[id] = (dx * dx + dy * dy + dz * dz).sqrt();
        }

        // Free-DOF -> edge id (inverse of sys.dof_of_edge).
        let mut edge_of_dof = vec![0u32; n_free];
        for (id, &d) in sys.dof_of_edge.iter().enumerate() {
            if d >= 0 {
                edge_of_dof[d as usize] = id as u32;
            }
        }

        // Active nodal DOFs: incident to >= 1 free edge. Inactive nodes would
        // produce zero rows/columns in A_n (singular inner system).
        // Note: PEC-boundary nodes are *included* (no Dirichlet condition).
        // The nullspace of the constrained curl-curl operator is gradients
        // of H¹ functions constant on each PEC component — the full nodal
        // space captures inter-component modes (e.g. the differential
        // potential between the two dipole arms) that a PEC-Dirichlet
        // coarse space would miss. This is the Hiptmair-Xu auxiliary space.
        let mut node_active = vec![false; nv];
        for (id, &(u, v)) in edge_uv.iter().enumerate() {
            if sys.dof_of_edge[id] >= 0 {
                node_active[u as usize] = true;
                node_active[v as usize] = true;
            }
        }
        let mut node_dof = vec![-1i32; nv];
        let mut node_count = 0usize;
        for (i, &active) in node_active.iter().enumerate() {
            if active {
                node_dof[i] = node_count as i32;
                node_count += 1;
            }
        }

        // G triplets: row = free edge DOF; G[e,u] = -1/l_e, G[e,v] = +1/l_e.
        // (Length-normalized: the DOF functional is (1/l_e)·∫E·τdl, so the
        // edge-DOF of ∇φ is (φᵥ − φᵤ)/l_e and K·G = 0 on the nose.)
        // Every free edge has two active endpoints (PEC-boundary nodes are
        // active), so every row has exactly two entries.
        let mut g_trip = Vec::with_capacity(2 * n_free);
        for (id, &(u, v)) in edge_uv.iter().enumerate() {
            let d = sys.dof_of_edge[id];
            if d < 0 {
                continue;
            }
            let e = d as usize;
            let inv_l = 1.0 / edge_len[id];
            g_trip.push((
                e,
                node_dof[u as usize] as usize,
                Complex64::new(-inv_l, 0.0),
            ));
            g_trip.push((e, node_dof[v as usize] as usize, Complex64::new(inv_l, 0.0)));
        }
        let g = CsrMatrix::from_triplets(n_free, node_count, g_trip);

        // Active node -> incident (free edge DOF, G entry = ±1/l_e).
        let mut node_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nv];
        for (id, &(u, v)) in edge_uv.iter().enumerate() {
            let d = sys.dof_of_edge[id];
            if d < 0 {
                continue;
            }
            let e = d as usize;
            let inv_l = 1.0 / edge_len[id];
            node_edges[u as usize].push((e, -inv_l));
            node_edges[v as usize].push((e, inv_l));
        }

        // 1/l per free-edge DOF (for the G[f,k] = ±1/l_f factors below).
        let mut dof_inv_len = vec![0.0f64; n_free];
        for (id, &d) in sys.dof_of_edge.iter().enumerate() {
            if d >= 0 {
                dof_inv_len[d as usize] = 1.0 / edge_len[id];
            }
        }

        // A_n[i][j] = Σ_{e∋i} Σ_{(f,aval)∈A.row(e)} Σ_{k∈{uf,vf}}
        //            s(e,i)·s(f,k)·aval.  GᵀAG assembled row by row:
        // the raw contribution count is ~40× the final nnz, and holding
        // it as one triplet list OOM-killed a 1.5M-DOF build on a 7 GB
        // host; per-row lists keep peak memory at the final matrix size.
        let mut an_rows: Vec<Vec<(usize, Complex64)>> = vec![Vec::new(); node_count];
        for i_node in 0..nv {
            let i = node_dof[i_node];
            if i < 0 {
                continue;
            }
            let row = &mut an_rows[i as usize];
            for &(e, se) in &node_edges[i_node] {
                let (cols, vals) = a.row(e);
                for (&f, &aval) in cols.iter().zip(vals.iter()) {
                    let (uf, vf) = edge_uv[edge_of_dof[f] as usize];
                    // Endpoints of a free edge are always active nodes.
                    let ju = node_dof[uf as usize] as usize;
                    let jv = node_dof[vf as usize] as usize;
                    // G[e,i]·A[e,f]·G[f,j], with G[f,ju] = −1/l_f,
                    // G[f,jv] = +1/l_f.
                    let c = aval * se * dof_inv_len[f];
                    row.push((ju, -c));
                    row.push((jv, c));
                }
            }
        }
        let a_nodal = CsrMatrix::from_rows(node_count, node_count, an_rows);
        let nodal_sgs = PreconditionerData::new(&a_nodal);

        // PEC-boundary node mask (incident to >= 1 eliminated edge), over
        // the *nodal DOFs* — exposed for diagnostics: the true nullspace of
        // the constrained curl-curl operator is gradients of potentials
        // constant on PEC.
        let mut pec_nodal = vec![false; node_count];
        for (id, &(u, v)) in edge_uv.iter().enumerate() {
            if sys.dof_of_edge[id] < 0 {
                let du = node_dof[u as usize];
                if du >= 0 {
                    pec_nodal[du as usize] = true;
                }
                let dv = node_dof[v as usize];
                if dv >= 0 {
                    pec_nodal[dv as usize] = true;
                }
            }
        }

        Ok(Self {
            g,
            a_nodal,
            node_count,
            pec_node: pec_nodal,
            nodal_sgs,
            edge_dof_count: n_free,
            inner_tol: INNER_TOL,
            inner_max_iter: INNER_MAX_ITER,
            inner_sweeps: INNER_SWEEPS,
        })
    }

    /// Number of nodal auxiliary DOFs.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Nonzeros of the nodal auxiliary matrix (fill indicator).
    pub fn nodal_nnz(&self) -> usize {
        self.a_nodal.nnz()
    }

    /// PEC-boundary node mask over the nodal DOFs (diagnostics).
    pub fn pec_node_mask(&self) -> &[bool] {
        &self.pec_node
    }

    /// Nodal auxiliary matrix `GᵀAG` (for diagnostics).
    pub fn nodal_matrix(&self) -> &CsrMatrix {
        &self.a_nodal
    }

    /// Discrete gradient action: `w = G·phi` (edges ← nodes).
    pub fn apply_g(&self, phi: &[Complex64]) -> Vec<Complex64> {
        debug_assert_eq!(phi.len(), self.node_count);
        let mut w = vec![Complex64::new(0.0, 0.0); self.edge_dof_count];
        for e in 0..self.edge_dof_count {
            let (cols, vals) = self.g.row(e);
            let mut s = Complex64::new(0.0, 0.0);
            for (&j, &gv) in cols.iter().zip(vals.iter()) {
                s += gv * phi[j];
            }
            w[e] = s;
        }
        w
    }

    /// Adjoint action: `cn = Gᵀ·r` (nodes ← edges).
    pub fn apply_g_t(&self, r: &[Complex64]) -> Vec<Complex64> {
        debug_assert_eq!(r.len(), self.edge_dof_count);
        let mut cn = vec![Complex64::new(0.0, 0.0); self.node_count];
        for e in 0..self.edge_dof_count {
            let re = r[e];
            if re == Complex64::new(0.0, 0.0) {
                continue;
            }
            let (cols, vals) = self.g.row(e);
            for (&j, &gv) in cols.iter().zip(vals.iter()) {
                cn[j] += gv * re;
            }
        }
        cn
    }

    /// Inner nodal approximation `y ≈ Aₙ⁻¹·cn`.
    ///
    /// By default (`inner_sweeps > 0`) this is a fixed number of SGS sweeps:
    /// far cheaper than an inner Krylov solve, and a fixed linear operator.
    /// Set `inner_sweeps = 0` to use inner BiCGSTAB+SGS instead (diagnostics).
    pub fn solve_nodal(&self, cn: &[Complex64]) -> Vec<Complex64> {
        if self.inner_sweeps == 0 {
            return self.solve_nodal_bicgstab(cn);
        }
        let n = self.node_count;
        let mut y = vec![Complex64::new(0.0, 0.0); n];
        let mut residual = vec![Complex64::new(0.0, 0.0); n];
        let mut delta = vec![Complex64::new(0.0, 0.0); n];
        let mut work = vec![Complex64::new(0.0, 0.0); n];
        for _ in 0..self.inner_sweeps {
            // SGS sweep on the residual: y += M⁻¹·(cn − Aₙ·y).
            let ay = self.a_nodal.matvec(&y);
            for k in 0..n {
                residual[k] = cn[k] - ay[k];
            }
            self.nodal_sgs
                .sgs_apply(&self.a_nodal, &residual, &mut delta, &mut work);
            for k in 0..n {
                y[k] += delta[k];
            }
        }
        y
    }

    /// Inner BiCGSTAB+SGS solve (diagnostics; see `inner_sweeps`).
    pub fn solve_nodal_bicgstab(&self, cn: &[Complex64]) -> Vec<Complex64> {
        let budget = Budget::none();
        bicgstab(
            &self.a_nodal,
            cn,
            self.inner_tol,
            self.inner_max_iter,
            Preconditioner::Sgs,
            &budget,
        )
        .map(|r| r.x)
        .unwrap_or_else(|_| vec![Complex64::new(0.0, 0.0); self.node_count])
    }

    /// Additive coarse correction: `z += G (Aₙ⁻¹ (Gᵀ r))`.
    ///
    /// The caller applies the SGS smoother into `z` first; this adds the
    /// gradient-space correction on top.
    pub fn apply_correction(
        &self,
        r: &[Complex64],
        z: &mut [Complex64],
        budget: &Budget,
    ) -> Result<()> {
        debug_assert_eq!(r.len(), self.edge_dof_count);
        debug_assert_eq!(z.len(), self.edge_dof_count);
        let _ = budget;

        let cn = self.apply_g_t(r);
        let y = self.solve_nodal(&cn);
        for e in 0..self.edge_dof_count {
            let (cols, vals) = self.g.row(e);
            let mut s = Complex64::new(0.0, 0.0);
            for (&j, &gv) in cols.iter().zip(vals.iter()) {
                s += gv * y[j];
            }
            z[e] += s;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn test_g(&self) -> &CsrMatrix {
        &self.g
    }

    #[cfg(test)]
    pub(crate) fn test_a_nodal(&self) -> &CsrMatrix {
        &self.a_nodal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::PointRequest;
    use crate::edge::build_edge_system;

    fn dipole_system() -> (Mesh, EdgeSystem, CsrMatrix) {
        let bytes = include_bytes!("../tests/fixtures/dipole_3ghz.req");
        let request = PointRequest::parse(bytes).expect("fixture must parse");
        let mesh = Mesh::from_request(&request).expect("mesh");
        let sys = build_edge_system(&mesh).expect("edge system");
        let a =
            crate::assembly::assemble_hcurl(&mesh, &sys, 3.0e9, &Budget::none()).expect("assembly");
        (mesh, sys, a)
    }

    #[test]
    fn gradient_has_scaled_structure() {
        let (mesh, sys, a) = dipole_system();
        let ams = AmsPreconditioner::build(&mesh, &sys, &a).expect("build");
        let g = ams.test_g();
        assert_eq!(g.nrows(), sys.free_dof_count);
        // Exactly two entries per row: −1/l_e and +1/l_e (opposite signs,
        // equal magnitudes). The 1/l_e scaling is what puts range(G) in the
        // nullspace of the curl-curl operator for the length-scaled basis.
        for e in 0..g.nrows() {
            let (cols, vals) = g.row(e);
            assert_eq!(cols.len(), 2, "row {e} must have 2 entries");
            assert_ne!(cols[0], cols[1]);
            assert!(
                (vals[0].norm() - vals[1].norm()).abs() < 1e-15 * vals[0].norm(),
                "row {e}: magnitudes differ: {} vs {}",
                vals[0],
                vals[1]
            );
            assert!(
                vals[0].re * vals[1].re < 0.0,
                "row {e}: not opposite signs: {} vs {}",
                vals[0],
                vals[1]
            );
            assert_eq!(vals[0] + vals[1], Complex64::new(0.0, 0.0));
        }
    }

    /// Discrete de Rham: `K·G = 0`, where K is the curl-curl stiffness.
    ///
    /// K is extracted from two volume assemblies: with
    /// `T(ω) = (1/μr)·K − k0²·εc·Q`, the combination
    /// `[T(ω₁) − r·T(ω₂)]/(1−r)`, `r = (ω₁/ω₂)²`, isolates `(1/μr)·K`
    /// (the ABC lives in separate triplets and never enters).
    #[test]
    fn gradient_is_curl_free() {
        let (mesh, sys, _) = dipole_system();
        let n = sys.free_dof_count;
        let t1 = crate::assembly::assemble_volume_triplets(&mesh, &sys, 3.0e9, &Budget::none())
            .expect("volume triplets f1");
        let t2 = crate::assembly::assemble_volume_triplets(&mesh, &sys, 2.0e9, &Budget::none())
            .expect("volume triplets f2");
        assert_eq!(t1.len(), t2.len(), "same sparsity pattern expected");
        let r = (3.0f64 / 2.0f64).powi(2);
        let k_trip: Vec<(usize, usize, Complex64)> = t1
            .into_iter()
            .zip(t2.into_iter())
            .map(|((i1, j1, v1), (i2, j2, v2))| {
                assert_eq!((i1, j1), (i2, j2), "triplet order must match");
                (i1, j1, (v1 - v2 * r) / (1.0 - r))
            })
            .collect();
        let k = CsrMatrix::from_triplets(n, n, k_trip);
        let k_frob = k.frobenius_norm();
        assert!(k_frob > 0.0, "K must be nonzero");

        let a_full =
            crate::assembly::assemble_hcurl(&mesh, &sys, 3.0e9, &Budget::none()).expect("assembly");
        let ams = AmsPreconditioner::build(&mesh, &sys, &a_full).expect("build");

        // Deterministic pseudo-random nodal vector, vanishing on PEC:
        // the nullspace of the *constrained* curl-curl operator is
        // gradients of potentials constant on PEC (here: pinned to zero).
        let mut phi = vec![Complex64::new(0.0, 0.0); ams.node_count()];
        let mut s = 0x243F6A8885A308D3u64;
        for p in phi.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s as f64) / (u64::MAX as f64) - 0.5;
            *p = Complex64::new(u, 0.0);
        }
        for (p, &is_pec) in phi.iter_mut().zip(ams.pec_node_mask().iter()) {
            if is_pec {
                *p = Complex64::new(0.0, 0.0);
            }
        }
        let w = ams.apply_g(&phi);
        let w_norm: f64 = w.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(w_norm > 0.0);
        let kw = k.matvec(&w);
        let kw_norm: f64 = kw.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let rel = kw_norm / (k_frob * w_norm);
        assert!(
            rel < 1e-10,
            "K·Gφ not ~0: ||K·Gφ||/(||K||_F·||Gφ||) = {rel:e}"
        );
    }

    #[test]
    fn nodal_matrix_is_complex_symmetric() {
        let (mesh, sys, a) = dipole_system();
        let ams = AmsPreconditioner::build(&mesh, &sys, &a).expect("build");
        let an = ams.test_a_nodal();
        assert_eq!(an.nrows(), an.ncols());
        // Spot-check symmetry on a strided sample of rows.
        let frob = an.frobenius_norm();
        let mut asym = 0.0f64;
        let step = (an.nrows() / 64).max(1);
        for i in (0..an.nrows()).step_by(step) {
            let (cols, vals) = an.row(i);
            for (&j, &v) in cols.iter().zip(vals.iter()) {
                let (c2, v2) = an.row(j);
                let w = match c2.binary_search(&i) {
                    Ok(p) => v2[p],
                    Err(_) => Complex64::new(0.0, 0.0),
                };
                asym += (v - w).norm_sqr();
            }
        }
        assert!(
            asym.sqrt() < 1e-9 * frob,
            "A_n not symmetric: asym {asym:e}, frob {frob:e}"
        );
    }

    #[test]
    fn ams_beats_sgs_on_dipole() {
        let (mesh, sys, a) = dipole_system();
        let ams = AmsPreconditioner::build(&mesh, &sys, &a).expect("build");
        // Physical port RHS (3 GHz dipole fixture).
        let (positive, negative) = crate::port::terminals_from_facets(&mesh).expect("terminals");
        let feed = crate::port::select_gap_feed(&mesh, &sys, positive, negative).expect("feed");
        let b = crate::port::build_port_rhs(&mesh, &sys, &feed, 3.0e9).expect("rhs");
        let sgs = bicgstab(&a, &b, 1e-7, 3000, Preconditioner::Sgs, &Budget::none()).expect("sgs");
        let ams_res = bicgstab(
            &a,
            &b,
            1e-7,
            3000,
            Preconditioner::Ams(ams),
            &Budget::none(),
        )
        .expect("ams");
        assert!(ams_res.converged, "AMS run must converge");
        println!(
            "dipole: SGS {} iters (conv={}), AMS {} iters",
            sgs.iterations, sgs.converged, ams_res.iterations
        );
        assert!(
            ams_res.iterations < sgs.iterations,
            "AMS ({} iters) should beat SGS ({} iters)",
            ams_res.iterations,
            sgs.iterations
        );
    }
}
