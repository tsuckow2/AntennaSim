# FDTD Yee Stencil Bug Fix

## Root Cause

The E-field update in `wasm/fdtd/src/solver.rs` had the curl terms swapped in all three components.

### The Bug

**Incorrect (old) code:**
```rust
// Ex: WRONG
ex = ca*ex + cb * ((hz_j - hz_jm) * idy - (hy_k - hy_km) * idz);
//                  ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
//                  (Hz term first)       (Hy term second)

// Ey: WRONG  
ey = ca*ey + cb * ((hx_k - hx_km) * idz - (hz_i - hz_im) * idx);

// Ez: WRONG
ez = ca*ez + cb * ((hy_i - hy_im) * idx - (hx_j - hx_jm) * idy);
```

**Correct (fixed) code:**
```rust
// Ex: CORRECT - (∇×H)_x = ∂Hy/∂z - ∂Hz/∂y
ex = ca*ex + cb * ((hy_k - hy_km) * idz - (hz_j - hz_jm) * idy);

// Ey: CORRECT - (∇×H)_y = ∂Hz/∂x - ∂Hx/∂z
ey = ca*ey + cb * ((hz_i - hz_im) * idx - (hx_k - hx_km) * idz);

// Ez: CORRECT - (∇×H)_z = ∂Hx/∂y - ∂Hy/∂x
ez = ca*ez + cb * ((hx_j - hx_jm) * idy - (hy_i - hy_im) * idx);
```

### Why It Caused Instability

The Yee scheme requires the discrete curl operators for E and H updates to be negative adjoints of each other for energy conservation in lossless media.

- H update: `H -= (dt/μ) * curl_h(E)` where `curl_h` uses forward differences
- E update: `E += (dt/ε) * curl_e(H)` where `curl_e` must use backward differences with the CORRECT term ordering

With the terms swapped, `curl_e` was not the adjoint of `curl_h`, breaking energy conservation and causing exponential growth (even at CFL=0.1, far below the theoretical limit of 0.577).

### Verification

1. **Unit tests pass:**
   - `single_cell_h_update`: Verifies Hx/Hy signs from single Ez impulse
   - `free_propagation_stable`: 20³ vacuum, 200 steps, CFL=0.5, max|Ez| stays bounded

2. **Snyder geometry stable:**
   - 232×232×61 grid, 500 steps with source, PEC, dielectric: stable
   - 1000 steps with conductive absorber: stable (max|Ez| ~13, decaying)

### CPML Status

The CPML in `src/cpml.rs` was written for the buggy Yee stencil. Its E-field corrections have inverted signs relative to the fixed Yee. The H-field corrections appear correct.

**Workaround:** Use a simple conductive absorber layer (8 cells, graded σ_max=15 S/m) instead of CPML. This is stable and provides adequate absorption for validation.

**TODO:** Rewrite CPML corrections to match the fixed Yee stencil, or verify the sign flips are complete.

## Files Changed

- `wasm/fdtd/src/solver.rs`: Fixed `update_h` and `update_e` (replaced with correct serial versions, added tests)
- `wasm/fdtd/src/cpml.rs`: Attempted E-field sign fix (incomplete, CPML still unstable)
- `wasm/fdtd/src/lib.rs`: Removed `yee_simple` module (consolidated into solver.rs)
- `wasm/fdtd/examples/snyder_absorber.rs`: New example using conductive absorber instead of CPML

## Test Results

```
running 4 tests
test grid::tests::sizes_match_layout ... ok
test solver::tests::single_cell_h_update ... ok
test source::tests::gaussian_peaks_near_target ... ok
test solver::tests::free_propagation_stable ... ok

test result: ok. 4 passed; 0 failed
```
