//! Complex compressed-sparse-row (CSR) matrix.
//!
//! The assembled H(curl) system is complex-symmetric (not Hermitian):
//!
//! ```text
//! [ A_r  -A_i ] [x_r]   [b_r]
//! [ A_i   A_r ] [x_i] = [b_i]
//! ```
//!
//! is never formed explicitly. Instead the complex matrix `A = A_r + j A_i`
//! is stored once in [`CsrMatrix`] and all solver kernels operate on
//! [`Complex64`](num_complex::Complex64) vectors directly.

use num_complex::Complex64;

/// Sparse matrix in compressed-sparse-row format with complex entries.
///
/// Rows are stored contiguously: row `i` occupies
/// `col_idx[row_ptr[i]..row_ptr[i+1]]` / `values[row_ptr[i]..row_ptr[i+1]]`,
/// with column indices sorted ascending within each row.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<Complex64>,
}

impl CsrMatrix {
    /// Build from coordinate triplets. Duplicate entries are summed;
    /// explicit zeros are dropped. Panics if any index is out of bounds.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        mut triplets: Vec<(usize, usize, Complex64)>,
    ) -> Self {
        for &(r, c, _) in &triplets {
            assert!(r < nrows, "triplet row {r} out of bounds ({nrows})");
            assert!(c < ncols, "triplet col {c} out of bounds ({ncols})");
        }
        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_idx = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());

        let mut iter = triplets.into_iter().peekable();
        while let Some((r, c, mut v)) = iter.next() {
            while let Some(&(r2, c2, v2)) = iter.peek() {
                if r2 == r && c2 == c {
                    v += v2;
                    iter.next();
                } else {
                    break;
                }
            }
            if v != Complex64::new(0.0, 0.0) {
                col_idx.push(c);
                values.push(v);
                row_ptr[r + 1] += 1;
            }
        }
        // Prefix sum: row_ptr[i] becomes the start offset of row i.
        for i in 0..nrows {
            row_ptr[i + 1] += row_ptr[i];
        }

        Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// All-zero matrix with the given shape.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored (nonzero) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// `(column_indices, values)` slice pair for row `i`.
    pub fn row(&self, i: usize) -> (&[usize], &[Complex64]) {
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        (&self.col_idx[start..end], &self.values[start..end])
    }

    /// `y = A x`. Panics on dimension mismatch.
    pub fn matvec(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.ncols, "matvec dimension mismatch");
        let mut y = vec![Complex64::new(0.0, 0.0); self.nrows];
        self.matvec_into(x, &mut y);
        y
    }

    /// `y += A x` accumulation form is not provided; this overwrites `y`.
    pub fn matvec_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(x.len(), self.ncols, "matvec x dimension mismatch");
        assert_eq!(y.len(), self.nrows, "matvec y dimension mismatch");
        for i in 0..self.nrows {
            let (cols, vals) = self.row(i);
            let mut acc = Complex64::new(0.0, 0.0);
            for (&c, &v) in cols.iter().zip(vals.iter()) {
                acc += v * x[c];
            }
            y[i] = acc;
        }
    }

    /// Diagonal entries (zero where structurally absent).
    pub fn diagonal(&self) -> Vec<Complex64> {
        let mut d = vec![Complex64::new(0.0, 0.0); self.nrows.min(self.ncols)];
        for i in 0..d.len() {
            let (cols, vals) = self.row(i);
            if let Ok(pos) = cols.binary_search(&i) {
                d[i] = vals[pos];
            }
        }
        d
    }

    /// Frobenius norm, useful for diagnostics and test assertions.
    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn matvec_matches_dense() {
        // [ 2+j    3  ]
        // [ 0    1-j  ]
        let a = CsrMatrix::from_triplets(
            2,
            2,
            vec![(0, 0, c(2.0, 1.0)), (0, 1, c(3.0, 0.0)), (1, 1, c(1.0, -1.0))],
        );
        let x = vec![c(1.0, 0.0), c(0.0, 1.0)];
        let y = a.matvec(&x);
        assert_eq!(y[0], c(2.0, 1.0) + c(3.0, 0.0) * c(0.0, 1.0));
        assert_eq!(y[1], c(1.0, -1.0) * c(0.0, 1.0));
    }

    #[test]
    fn duplicate_triplets_are_summed_and_zeros_dropped() {
        let a = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (1, 0, c(1.0, 0.0)),
                (0, 1, c(0.0, 0.0)), // explicit zero: dropped
                (1, 0, c(2.0, 0.0)),
            ],
        );
        assert_eq!(a.nnz(), 1);
        assert_eq!(a.diagonal(), vec![c(0.0, 0.0), c(0.0, 0.0)]);
        let (cols, vals) = a.row(1);
        assert_eq!(cols, &[0]);
        assert_eq!(vals, &[c(3.0, 0.0)]);
    }

    #[test]
    fn diagonal_extracts_structural_diagonal() {
        let a = CsrMatrix::from_triplets(
            3,
            3,
            vec![(0, 0, c(5.0, 0.0)), (2, 2, c(-2.0, 1.0)), (0, 2, c(1.0, 1.0))],
        );
        assert_eq!(a.diagonal(), vec![c(5.0, 0.0), c(0.0, 0.0), c(-2.0, 1.0)]);
    }
}
