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

## Current Sign Fix (2026-09-26)

The Ampère loop in `Feed::measure_i` measures the downward (-z) gap current due to loop orientation. The antenna feed current (into the probe) is upward (+z). The example now negates the measured current:

```rust
// The Ampère loop measures downward (-z) gap current; the antenna
// feed current (into the probe) is upward (+z), so negate.
probe.push(vs, -i_avg);
```

**Validation (5ns Snyder run):**
- Before: Zin = -32.6 + j1387 Ω (negative R, nonphysical)
- After: Zin = +41.6 - j1391 Ω (positive R, capacitive X as expected below resonance)

## Feed Magnitude Issue (2026-09-26, OPEN)

The impedance magnitude is ~20x too large (|Z| ~1400Ω vs expected ~50Ω).

**Root cause:** The probe via (3×3 cells, Ez PEC at k=33-34, z=0.79-1.31mm) does not electrically connect to the patch (Ex/Ey PEC at k=34, z=1.575mm) due to Yee grid staggering. The half-cell gap (0.26mm) creates a capacitive disconnect, resulting in huge capacitive reactance (-j1000).

**Evidence:**
- Dipole calibration (no substrate, thin wire): ~100Ω magnitude (correct scale)
- Snyder with via: ~1400Ω (20x too big)
- Snyder without via: ~3200Ω (worse — via helps but doesn't fully connect)
- Raw current at step 100: 1.47µA for 0.6mV source (8x smaller than expected for 50Ω)

**Attempted fixes:**
- Extending via to k=35: crashed (out-of-bounds or instability)
- Removing via: worse (doubled reactance)

**Required:** Proper FDTD probe feed model that handles Yee stagger. Options:
1. Overlap via and patch by one cell with careful boundary handling
2. Use soft current source instead of hard voltage source
3. Model probe as thin wire with corrected radius
4. Use delta-gap feed directly on patch (no via)

This is a fundamental FDTD feed modeling issue, not a Yee stencil bug.

## Soft Source Fix (2026-09-26)

**Problem:** Hard source (`grid.ez[si] = -vs/dz`) clamps the E-field, preventing the patch from resonating. Results in huge capacitive reactance (-j1100).

**Solution:** Soft source (`grid.ez[si] += -vs/dz`) adds to the field without enforcing it, allowing the antenna to radiate.

**Validation (2ns test):**
- Hard source: Zin = 31 - j1132 Ω at 2.4 GHz
- Soft source: Zin = 411 + j4 Ω at 2.4 GHz (X≈0, resonance!)

The reactance is now near zero, indicating the patch is resonating. Resistance ~400Ω (8x higher than expected 50Ω) — feed position or scaling may need tuning, but the fundamental resonance is captured.

**Via extension:** k_probe_hi extended from 34 to 35 (z=1.84mm, above patch at 1.575mm) to ensure the via penetrates the patch plane through Yee staggering. Increased peak current 460x (1.47µA → 0.68mA).
