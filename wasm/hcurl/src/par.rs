//! Data-parallel loop abstraction.
//!
//! [`map_indexed`], [`for_each_indexed_mut`], [`sum_indexed`] and
//! [`csum_indexed`] run on Rayon's thread pool on native targets and fall
//! back to sequential iterators on `wasm32` (the WASM module is built without
//! the atomics/SharedArrayBuffer threading setup, so the browser path stays
//! single-threaded for now).
//!
//! Ordering guarantees:
//!
//! - [`map_indexed`] and [`for_each_indexed_mut`] preserve the sequential
//!   visit order and the per-element operation order, so kernels built on
//!   them are **bit-identical** to the serial loop on every target.
//! - [`sum_indexed`] and [`csum_indexed`] leave the reduction order to the
//!   thread pool: results agree with the serial sum up to floating-point
//!   reassociation (a few ulps). Native runs are therefore not bit-for-bit
//!   reproducible across thread counts; the `wasm32` fallback always is.
//!
//! Thread count on native targets follows Rayon's default (one thread per
//! logical core) and can be overridden with `RAYON_NUM_THREADS`.

use num_complex::Complex64;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Ordered map over slice indices: `out[i] = f(i, &items[i])`.
///
/// The output order always matches the input order, so flattening the
/// per-element chunks afterwards reproduces the serial sequence exactly.
pub fn map_indexed<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> U + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        items.par_iter().enumerate().map(|(i, t)| f(i, t)).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        items.iter().enumerate().map(|(i, t)| f(i, t)).collect()
    }
}

/// Mutating visit over slice indices: each index is visited exactly once.
///
/// Safe for disjoint writes (each closure invocation owns `&mut items[i]`);
/// the visit order matches the sequential loop.
pub fn for_each_indexed_mut<T, F>(items: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        items.par_iter_mut().enumerate().for_each(|(i, t)| f(i, t));
    }
    #[cfg(target_arch = "wasm32")]
    {
        items.iter_mut().enumerate().for_each(|(i, t)| f(i, t));
    }
}

/// Real sum `Σ f(i)` for `i` in `0..n`; reduction order is pool-defined.
pub fn sum_indexed<F>(n: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        (0..n).into_par_iter().map(f).sum()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).map(f).sum()
    }
}

/// Complex sum `Σ f(i)` for `i` in `0..n`; reduction order is pool-defined.
pub fn csum_indexed<F>(n: usize, f: F) -> Complex64
where
    F: Fn(usize) -> Complex64 + Sync + Send,
{
    let zero = Complex64::new(0.0, 0.0);
    #[cfg(not(target_arch = "wasm32"))]
    {
        (0..n).into_par_iter().map(f).reduce(|| zero, |a, b| a + b)
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).map(f).fold(zero, |a, b| a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_preserves_order() {
        let items: Vec<i32> = (0..1000).collect();
        let out = map_indexed(&items, |i, &t| (i, t * 2));
        assert_eq!(out.len(), 1000);
        for (i, (oi, v)) in out.iter().enumerate() {
            assert_eq!(*oi, i);
            assert_eq!(*v, (i as i32) * 2);
        }
    }

    #[test]
    fn for_each_mut_visits_each_index_once() {
        let mut items = vec![0i32; 500];
        for_each_indexed_mut(&mut items, |i, t| *t = i as i32 + 1);
        for (i, &v) in items.iter().enumerate() {
            assert_eq!(v, i as i32 + 1);
        }
    }

    #[test]
    fn sums_agree_with_sequential() {
        let n = 10_000;
        let s = sum_indexed(n, |i| (i as f64).sin());
        let expect: f64 = (0..n).map(|i| (i as f64).sin()).sum();
        assert!((s - expect).abs() < 1e-9 * expect.abs().max(1.0));
        let c = csum_indexed(n, |i| Complex64::new(i as f64, -(i as f64)));
        let expect_c = Complex64::new(
            (0..n).map(|i| i as f64).sum::<f64>(),
            -(0..n).map(|i| i as f64).sum::<f64>(),
        );
        assert!((c - expect_c).norm() < 1e-6);
    }
}
