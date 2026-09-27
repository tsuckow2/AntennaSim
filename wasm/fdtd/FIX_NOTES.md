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

## Feed Impedance Status (2026-09-26, continued debugging)

**What works:**
- Yee stencil correct, stable (47ms/step serial)
- Current sign validated (R positive)
- Soft source (`+=`) achieves X≈0 (patch resonance!) vs hard source -j1100
- 17ns run: Zin ~410 + j3 Ω, S11 ~-2.1 dB across 2-3 GHz

**Open issue: Resistance 8x high (410Ω vs ~50Ω expected)**
- Free-space source alone gives ~750Ω (source self-impedance)
- Via geometry changes (k=34 vs 35) have zero effect → via not participating as expected
- Resistive Thevenin source (50Ω series) attempted but hard/soft source models
  bypass or overwhelm the conductivity
- Root cause: lumped source needs proper current-source formulation with
  correct dt scaling, not voltage addition. The `Ez += -vs/dz` is unphysical.

**For FEM frequency cross-check:**
The X≈0 indicates the patch resonates in FDTD. The absolute R value requires
proper port calibration (de-embedding source self-impedance). The resonance
frequency (where X crosses zero) is the key cross-check metric, not absolute R.

---

## Research-Corrected Feed Model (2026-09-27)

**Supersedes:** The "Soft Source Fix" and "Feed Impedance Status" sections above.
The 410+j3 Ω "resonance" was INVALID — it DFT'd the *intended* source waveform
(V_s) instead of the *actual* gap voltage. With correct V/I extraction, the soft
source gives the same highly capacitive impedance as the hard source. The
free-space ~750 Ω is grid self-impedance, not to be subtracted.

**Published feed model implemented (Luebbers & Langdon 1996):**
- **Resistive (Thevenin) source:** 50 Ω series resistor at the gap cell absorbs
  reflected waves (~32× faster convergence than hard source). Update replaces
  the E-update at the source cell: `E_z = −V_s/dz + I_s·Rs/dz`.
- **Tight 1-cell Ampère loop:** Four H-samples immediately adjacent to the source
  cell (CCW from +z, +I upward). The old wide loop picked up return currents.
- **Actual V, not V_s:** Record `V = −E_z·dz` (the gap voltage after the source
  update), never the prescribed waveform.
- **Via–patch overlap:** Via extends to k=35 (z=1.84mm), half a cell above the
  patch plane (z=1.575mm). This is the published Yee wire-end fix (Boonzaaier &
  Pistorius 1992); without it the via is only capacitively coupled.

**Code changes:**
- `src/source.rs`: Added `Feed::measure_i_tight()`, `ResistiveSource` struct with
  `apply()` method, unit tests for loop orientation and Thevenin equation.
- `src/solver.rs`: `Fdtd::step` now uses the Luebbers source (tight loop I at
  H-step, resistive update after E/PEC, records actual V/I).
- `src/geometry.rs`: `k_probe_hi = 35` (via overlaps patch plane).
- `examples/feed_test.rs`, `snyder_absorber.rs`, `snyder_fdtd.rs`: Updated to
  resistive source with correct V/I.

**Validation (2ns, Rs=50, 3x3 via):**
- Re(Zin) POSITIVE (42–67 Ω) — loop sign correct, no negation needed.
- X still highly capacitive (−j2932 at 2.4 GHz, −j2813 at 2.5 GHz) —
  via-patch connection incomplete.
- Via DOES affect Zin (no via: −j3490 at 2.5 GHz; 1x1 via: −j2380;
  3x3 via: −j2813), but does not create resonance. The patch is weakly
  excited (fields above patch nonzero but Hy currents tiny).
- Rs=0 gives identical results — the source is not the problem.

**Open:** The via-patch junction needs better conduction. The k=35 half-cell
extension (published fix) is insufficient in this geometry. Options: wider via
at junction, explicit 3D PEC connection, or thin-wire subcell model.

---

## Via Conduction Blocker (2026-09-27)

**Status:** The probe via does NOT conduct as a wire. Extending the via from
k=35 to k=45 (adding 5.25mm of PEC wire) has ZERO effect on Zin. The via acts
only as a near-field capacitive load (reducing X from −j3490 to −j2800), not
as a wire guiding current to the patch.

**Evidence:**
- Via to k=35: Zin = 52.0 − j2813 at 2.5 GHz
- Via to k=45: Zin = 52.0 − j2813 at 2.5 GHz (identical)
- No via: Zin = 29.1 − j3490 at 2.5 GHz (via provides capacitive loading)
- 1x1 via: Zin = 23.7 − j2380 (slightly better, still capacitive)
- 17ns run: No resonance, S11 ≈ 0 dB across 2–3 GHz

**Root cause:** Simple PEC-cell (E=0) vias do not model thin wires correctly
in FDTD. The literature uses subcell thin-wire models (Taflove) that modify
the H-field update around the wire to account for the wire radius and ensure
proper current conduction. Without this, the wire is "invisible" to propagating
fields.

**Definitive tests (2026-09-27):**
- Via k=33-35 vs k=33-34 vs k=33-45: ALL give identical Zin (52.0 − j2813).
  Via length has ZERO effect → via is disconnected.
- Via 3x3 vs 1x1: 3x3 gives 52−j2813, 1x1 gives 23.7−j2380. Width affects
  capacitive coupling but neither conducts as a wire.
- Source at k=32 vs k=33 (via base): minimal difference (23.7−j2380 vs
  22.5−j2536). Source-via coupling is not the main issue.
- Solid 3x3x2 PEC junction block at via-patch: ZERO effect.
- Distributed source driving via cells directly: WORSE (−j5073). Patch does
  not respond even with direct drive.
- Yee stagger: Ex/Ey at (k+½)dz, Ez at k·dz. Patch at 1.575mm (Ex/Ey k=34),
  via Ez at 1.3125mm (k=34) and 1.8375mm (k=35). No Ez exists AT the patch
  plane (would require k=34.5). The "upper edge coincident" rule (Boonzaaier
  & Pistorius) means via top at k=34 (upper edge at 1.575mm), not k=35.
  But even with correct alignment, the wire does not conduct.

**Impact:** FDTD cannot validate the Snyder patch resonance frequency until
a thin-wire model is implemented. The resistive source, tight loop, and V/I
extraction are correct and validated (Re(Z)>0, unit tests pass), but the feed
cannot deliver current to the patch.

**Next:** Implement Taflove thin-wire subcell model for the probe via, OR use
an alternative feed (e.g., aperture coupling) that doesn't require a wire.
