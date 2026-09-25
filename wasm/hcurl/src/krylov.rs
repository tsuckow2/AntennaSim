//! Complex BiCGSTAB with Jacobi / symmetric Gauss–Seidel preconditioning.
//!
//! Solves the general complex (non-Hermitian, complex-symmetric) system
//! `A·x = b` using the Hermitian inner product `⟨x,y⟩ = Σ conj(xₖ)·yₖ`.
//! Mirrors `bicgstab` in the TypeScript reference, including the exact
//! floating-point operation order, so iteration counts reproduce closely.
//!
//! Default production parameters: [`DEFAULT_TOL`] = 1e-7,
//! [`DEFAULT_MAX_ITER`] = 12000, [`Preconditioner::Sgs`].

use num_complex::Complex64;

use crate::budget::Budget;
use crate::csr::CsrMatrix;
use crate::error::Result;

/// Default relative residual tolerance (validated value from the reference).
pub const DEFAULT_TOL: f64 = 1e-7;
/// Default iteration cap (headroom above the ~9.2k worst observed case).
pub const DEFAULT_MAX_ITER: u32 = 12000;
/// Zero-division guard shared with the reference implementation.
const TINY: f64 = 1e-300;

/// Available left preconditioners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preconditioner {
    /// Pointwise `x·conj(D)/|D|²` (kept for testing; pass-through on tiny diagonal).
    Jacobi,
    /// One symmetric sweep `M = (D+L)·D⁻¹·(D+U)` with the complex diagonal.
    Sgs,
}

/// Outcome of a BiCGSTAB run.
#[derive(Debug, Clone)]
pub struct KrylovResult {
    pub x: Vec<Complex64>,
    pub iterations: u32,
    /// Final relative residual `‖b − A·x‖₂/‖b‖₂`.
    pub residual_norm: f64,
    pub converged: bool,
}

/// Naive complex division with the reference's zero guard.
///
/// Note: this deliberately does **not** use a robust Smith-style division —
/// matching the reference's rounding behavior keeps iteration counts
/// comparable.
fn cdiv(a: Complex64, b: Complex64) -> Complex64 {
    let d = b.norm_sqr();
    if d < TINY {
        Complex64::new(0.0, 0.0)
    } else {
        Complex64::new(
            (a.re * b.re + a.im * b.im) / d,
            (a.im * b.re - a.re * b.im) / d,
        )
    }
}

fn norm2(v: &[Complex64]) -> f64 {
    // Pool-ordered reduction: a few ulps of drift vs the serial sum, well
    // inside the dipole parity tolerances.
    crate::par::sum_indexed(v.len(), |k| v[k].norm_sqr()).sqrt()
}

/// Hermitian inner product `Σ conj(aₖ)·bₖ`.
///
/// The parallel reduction order is pool-defined (see [`norm2`]).
fn dot_conj(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    crate::par::csum_indexed(a.len(), |k| {
        let (a1, a2) = (a[k].re, a[k].im);
        let (b1, b2) = (b[k].re, b[k].im);
        Complex64::new(a1 * b1 + a2 * b2, a1 * b2 - a2 * b1)
    })
}

struct PreconditionerData {
    diag: Vec<Complex64>,
    inv_diag: Vec<Complex64>, // conj(D)/|D|², or 1 on tiny diagonal
}

impl PreconditionerData {
    fn new(matrix: &CsrMatrix) -> Self {
        let n = matrix.nrows();
        let mut diag = vec![Complex64::new(0.0, 0.0); n];
        let mut inv_diag = vec![Complex64::new(1.0, 0.0); n];
        for r in 0..n {
            let (cols, vals) = matrix.row(r);
            if let Ok(pos) = cols.binary_search(&r) {
                let d = vals[pos];
                diag[r] = d;
                let den = d.norm_sqr();
                if den >= TINY {
                    inv_diag[r] = Complex64::new(d.re / den, -d.im / den);
                }
            }
        }
        Self { diag, inv_diag }
    }

    /// `y = M⁻¹·x` into `out`, using `scratch` for the SGS forward sweep.
    fn apply(
        &self,
        matrix: &CsrMatrix,
        pre: Preconditioner,
        x: &[Complex64],
        out: &mut [Complex64],
        scratch: &mut [Complex64],
    ) {
        let n = matrix.nrows();
        match pre {
            Preconditioner::Jacobi => {
                for k in 0..n {
                    out[k] = x[k] * self.inv_diag[k];
                }
            }
            Preconditioner::Sgs => {
                // Forward: (D + L)·y = x.
                for i in 0..n {
                    let mut s = x[i];
                    let (cols, vals) = matrix.row(i);
                    for (&j, &a) in cols.iter().zip(vals.iter()) {
                        if j < i {
                            s -= a * scratch[j];
                        }
                    }
                    let d = self.diag[i];
                    scratch[i] = if d.norm_sqr() < TINY {
                        s // pass through undivided
                    } else {
                        cdiv(s, d)
                    };
                }
                // Backward: (D + U)·z = D·y.
                for i in (0..n).rev() {
                    let d = self.diag[i];
                    let mut s = d * scratch[i];
                    let (cols, vals) = matrix.row(i);
                    for (&j, &a) in cols.iter().zip(vals.iter()) {
                        if j > i {
                            s -= a * out[j];
                        }
                    }
                    out[i] = if d.norm_sqr() < TINY {
                        s // pass through undivided
                    } else {
                        cdiv(s, d)
                    };
                }
            }
        }
    }
}

/// Run BiCGSTAB on `matrix·x = b`.
///
/// Returns [`Error::Cancelled`]/[`Error::Timeout`] if `budget` trips;
/// non-convergence is reported via `converged: false` (the caller maps it
/// to a solver error), never as `Err`.
pub fn bicgstab(
    matrix: &CsrMatrix,
    b: &[Complex64],
    tol: f64,
    max_iter: u32,
    pre: Preconditioner,
    budget: &Budget,
) -> Result<KrylovResult> {
    let n = matrix.nrows();
    assert_eq!(matrix.ncols(), n, "BiCGSTAB requires a square matrix");
    assert_eq!(b.len(), n, "RHS dimension mismatch");

    let done =
        |iterations: u32, residual_norm: f64, converged: bool, x: Vec<Complex64>| KrylovResult {
            x,
            iterations,
            residual_norm,
            converged,
        };

    let b_norm = norm2(b);
    if !(b_norm > 0.0) || !b_norm.is_finite() {
        return Ok(done(0, f64::NAN, false, vec![Complex64::new(0.0, 0.0); n]));
    }

    let pre_data = PreconditionerData::new(matrix);
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    let mut r = b.to_vec();
    let r_hat = b.to_vec();
    let mut p = vec![Complex64::new(0.0, 0.0); n];
    let mut v = vec![Complex64::new(0.0, 0.0); n];
    let mut s = vec![Complex64::new(0.0, 0.0); n];
    let mut t = vec![Complex64::new(0.0, 0.0); n];
    let mut z = vec![Complex64::new(0.0, 0.0); n];
    let mut ph = vec![Complex64::new(0.0, 0.0); n];
    let mut scratch = vec![Complex64::new(0.0, 0.0); n];

    let mut rho: Complex64;
    let mut rho_prev = Complex64::new(1.0, 0.0);
    let mut alpha = Complex64::new(1.0, 0.0);
    let mut omega = Complex64::new(1.0, 0.0);

    for iter in 1..=max_iter {
        budget.check()?;

        rho = dot_conj(&r_hat, &r);
        if rho.norm() < TINY {
            return Ok(done(iter, norm2(&r) / b_norm, false, x));
        }
        let beta = cdiv(rho, rho_prev) * cdiv(alpha, omega);
        // Each k is independent: disjoint writes, bit-identical to serial.
        crate::par::for_each_indexed_mut(&mut p, |k, pk| {
            *pk = r[k] + beta * (*pk - omega * v[k]);
        });
        pre_data.apply(matrix, pre, &p, &mut ph, &mut scratch);
        matrix.matvec_into(&ph, &mut v);
        alpha = cdiv(rho, dot_conj(&r_hat, &v));
        crate::par::for_each_indexed_mut(&mut s, |k, sk| {
            *sk = r[k] - alpha * v[k];
        });
        let s_norm = norm2(&s) / b_norm;
        if s_norm < tol && s_norm.is_finite() {
            crate::par::for_each_indexed_mut(&mut x, |k, xk| {
                *xk += alpha * ph[k];
            });
            return Ok(done(iter, s_norm, true, x));
        }
        pre_data.apply(matrix, pre, &s, &mut z, &mut scratch);
        matrix.matvec_into(&z, &mut t);
        omega = cdiv(dot_conj(&t, &s), dot_conj(&t, &t));
        crate::par::for_each_indexed_mut(&mut x, |k, xk| {
            *xk += alpha * ph[k] + omega * z[k];
        });
        crate::par::for_each_indexed_mut(&mut r, |k, rk| {
            *rk = s[k] - omega * t[k];
        });
        let r_norm = norm2(&r) / b_norm;
        if r_norm < tol && r_norm.is_finite() {
            return Ok(done(iter, r_norm, true, x));
        }
        if omega.norm() < TINY {
            return Ok(done(iter, r_norm, false, x));
        }
        rho_prev = rho;
    }
    Ok(done(max_iter, norm2(&r) / b_norm, false, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Budget;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// Small complex-symmetric, diagonally dominant system with a known
    /// solution x = [1+j, 2−j].
    fn small_system() -> (CsrMatrix, Vec<Complex64>, Vec<Complex64>) {
        let x_exact = vec![c(1.0, 1.0), c(2.0, -1.0)];
        let a = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, c(4.0, 1.0)),
                (0, 1, c(1.0, -1.0)),
                (1, 0, c(1.0, -1.0)),
                (1, 1, c(5.0, 2.0)),
            ],
        );
        let b = a.matvec(&x_exact);
        (a, b, x_exact)
    }

    #[test]
    fn converges_to_known_solution() {
        let (a, b, x_exact) = small_system();
        for pre in [Preconditioner::Jacobi, Preconditioner::Sgs] {
            let res = bicgstab(&a, &b, 1e-10, 100, pre, &Budget::none()).unwrap();
            assert!(res.converged, "pre={pre:?}");
            for (got, want) in res.x.iter().zip(x_exact.iter()) {
                assert!(
                    (got - want).norm() < 1e-8,
                    "pre={pre:?} got={got} want={want}"
                );
            }
            assert!(res.residual_norm < 1e-10);
        }
    }

    #[test]
    fn reports_non_convergence_instead_of_erroring() {
        let (a, b, _) = small_system();
        // One iteration is not enough for 1e-14 on this system.
        let res = bicgstab(&a, &b, 1e-14, 1, Preconditioner::Jacobi, &Budget::none()).unwrap();
        assert!(!res.converged);
        assert_eq!(res.iterations, 1);
    }

    #[test]
    fn degenerate_rhs_returns_not_converged() {
        let (a, _, _) = small_system();
        let b = vec![c(0.0, 0.0), c(0.0, 0.0)];
        let res = bicgstab(&a, &b, 1e-8, 100, Preconditioner::Jacobi, &Budget::none()).unwrap();
        assert!(!res.converged);
        assert!(res.residual_norm.is_nan());
    }

    #[test]
    fn sgs_sweep_matches_hand_computation() {
        // 2x2 real system; one SGS sweep on r = [3, 3]:
        // forward:  y0 = 3/4; y1 = (3 − 1·y0)/5 = 0.45
        // backward: z1 = 5·0.45/5 = 0.45; z0 = (4·0.75 − 1·0.45)/4 = 0.6375
        let a = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, c(4.0, 0.0)),
                (0, 1, c(1.0, 0.0)),
                (1, 0, c(1.0, 0.0)),
                (1, 1, c(5.0, 0.0)),
            ],
        );
        let pre = PreconditionerData::new(&a);
        let x = vec![c(3.0, 0.0), c(3.0, 0.0)];
        let mut out = vec![c(0.0, 0.0); 2];
        let mut scratch = vec![c(0.0, 0.0); 2];
        pre.apply(&a, Preconditioner::Sgs, &x, &mut out, &mut scratch);
        assert!((out[0].re - 0.6375).abs() < 1e-12, "z0={}", out[0]);
        assert!((out[1].re - 0.45).abs() < 1e-12, "z1={}", out[1]);
    }
}
