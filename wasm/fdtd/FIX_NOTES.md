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

## External Implementation Comparison (2026-09-27)

### witwin-maxwell — same failure mode

Inspected their patch antenna benchmark:
- Repo: https://github.com/witwin-ai/witwin-maxwell
- Assessment: docs/assessments/f2b-quasistatic-benches-acceptance-2026-07-21.md
- Patch source: benchmark/scenes/antenna/patch.py

Their geometry:
- Ground at z=0, one-cell lumped-port gap z=0 to dx, PEC probe from z=dx to patch
- Port-current surface at z=0.5dx in the source gap

Result: NOT a successful reference. Adding the via reduced reactance ~5x but
Re(Zin) < 4 Ω, still purely capacitive over 2-8 GHz. Patch did not resonate.
Their diagnosis: "lumped gap remained capacitively shorted by the adjacent PEC."
Proposed follow-up: "wire-bound clean-gap probe feed" (not implemented; their
matched-broadside test remains strict-xfail).

This independently reproduces our failure mode. Do not copy as a validated solution.

Note: witwin-maxwell v0.4.0 release notes mention "a subgrid thin-wire conductor
model with finite conductivity" — but their patch test is still xfail, so the
thin-wire model does not fix the probe-fed patch.

### openEMS — distributed port topology

From https://github.com/thliebig/openems-project/blob/HEAD/doc-src/concepts/ports.rst:

- "A lumped port is a small 2D surface or 3D cube filled by an electric field"
- "A lumped port uses a constant-value electric field as the excitation signal"
- "its physical size must be much smaller than the simulated structure"

Key insight: openEMS lumped ports are DISTRIBUTED (2D surface), not single-cell.
Our 1-cell source is smaller than openEMS's typical port.

However: openEMS does not publish a validated probe-fed patch benchmark with
quantitative Zin vs measurement. Their Horn_Antenna.py example uses a pin feed
through a waveguide, not a microstrip patch probe.

### Distributed 3x3 source experiment (2026-09-27)

Attempted to match openEMS topology: distributed 3x3 Ez source (matching the
3x3 via cross-section), wide Ampère loop enclosing the 3x3 for current measurement.

Result: NUMERICAL INSTABILITY. All runs (1ns, 10ns) produced NaN impedances.
The wide loop picks up displacement current, causing Luebbers Thevenin feedback
(v_term = Vs - I*Rs) to go unstable.

Tried:
- Wide loop at via plane (k=33): NaN
- Wide loop at source plane (k=32): NaN  
- Tight loop (1 cell beyond 3x3): NaN

The 1-cell source with tight 1-cell loop (e925999) is stable but gives wrong
impedance (-j2800, no resonance). The distributed source is conceptually correct
but numerically unstable with the Luebbers feedback.

**Verdict:** The probe feed is blocked in Yee FDTD without a proper thin-wire
subcell model (Taflove). The 3x3 PEC block is not a wire — it's a capacitive
obstacle. A distributed source does not fix the underlying geometry problem.

### gprMax

Antenna examples use transmission-line feeding, not probe feeds. Not applicable.

### Thin-wire subcell model (2026-09-27)

Implemented the Taflove/Umashankar thin-wire subcell model for the probe.

**Change:** Replaced the 3×3 PEC block via with a 1×1 thin wire (single Ez cell)
at (pc_i, pc_j, k=33..34). Added `src/thin_wire.rs` with the contour-path
correction factor:

```text
factor = ln(R/r0) / ln(R/a)
```

where R=Δ/2 (0.25mm), r0=0.23Δ (0.115mm, Yee cell equivalent radius),
a=0.1mm (wire radius). Factor ≈ 0.846.

The 4 H-field components surrounding the wire (Hx at (i0,j0) and (i0,j0-1),
Hy at (i0,j0) and (i0-1,j0)) have their Ez-difference term scaled by `factor`
in `update_h`. This accounts for the 1/r static field variation around the
thin wire, correcting the inductance from the Yee cell's natural r0 to the
physical wire radius a.

**Rationale:** The 3×3 block (1.5mm) does not conduct in the Yee grid —
changing its length has zero effect on Zin. The block is a capacitive
obstacle, not a wire. A 1×1 wire with thin-wire correction allows the
current to flow from the source (k=32) up through the wire (k=33-34) to
the patch, with the correct subcell inductance.

**Result (2ns feed_test):** Stable (finite=true), max|I|=5.37e-5 A (current
flows!), Re(Zin)=73-107 Ω (positive). Reactance still large capacitive
(-j10k to -j19k) — DFT not converged in 2ns, or wire radius needs tuning.

**Status:** 10ns validation run in progress. The thin-wire model is the
standard FDTD approach for probe feeds; the 3×3 block was the wrong geometry.

---

## Holland/Simpson Thin-Wire Model (2026-09-28)

### Why the coefficient method was abandoned

The Taflove/Umashankar coefficient-scaling approach (scaling H-field updates
by `factor = ln(R/r0)/ln(R/a) ≈ 0.8475`) produced a stable but unphysical
result: Zin ≈ 2700 Ω (purely capacitive, S11 ≈ -0.32 dB) on the Snyder
geometry. The 10ns Forge run produced NaN across 2-3 GHz.

The fundamental issue: the coefficient method keeps the wire as a PEC cell
(Ez=0), which blocks conduction current. The H-field correction adjusts
inductance but does not create a current path. The wire remains an open
circuit.

### Holland/Simpson formulation

Replaced with the Holland/Simpson (1981) auxiliary wire model, based on the
witwin-maxwell reference implementation (`AxisAlignedWireReference`):

- The wire is **NOT** a PEC cell. Ez at the wire location evolves normally.
- The wire is an auxiliary 1D transmission-line system:
  - Segment currents I (half-step) and node charges q (integer step)
  - Per-unit-length: L' = μ₀·ln(R/a)/(2π), C' = μ₀ε₀/L'
  - R = 0.23·dx (coupling distance), a = wire radius
- Bidirectional coupling:
  1. Sample EMF from Ez: `EMF = Ez·dz`
  2. Update current: `I^{n+1/2} = I^{n-1/2} + dt/L·(EMF + V_tail - V_head)`
  3. Deposit current into Ez: `Ez -= dt·I/(ε₀·dx·dy)` (as part of E^{n+1})
  4. Update charge: `q^{n+1} = q^n - dt·(I_out - I_in)`
- Top node tied to 0V when terminating on PEC (patch).
- Terminal current = bottom-segment wire current (not H-field Ampère loop).

### Stability

The coupled wire-FDTD system requires a stricter CFL than FDTD alone.
The witwin-maxwell reference computes the exact `2/ω_max` limit from the
coupled operator eigenvalues.

**Finding:** At 0.99× FDTD CFL, the coupled system is unstable (currents
blow up to 1e23 within ~130 steps, alternating signs = Nyquist mode).
At 0.1× FDTD CFL, the system is stable:
- Dipole fixture (40mm, 1V Gaussian): max|I| = 6.2e-6 A, finite, wire
  energy 2.8e-18 J. Wire alone (no FDTD) is stable at 0.5× CFL.
- Snyder 0.5ns test: 11-12 steps/s, no NaN, steady progress.

The 10× timestep reduction is the cost of the partitioned (operator-split)
coupling. A unified update (wire + E-field stepped together) would allow
larger steps but requires deeper solver changes.

### Files

- `wasm/fdtd/src/thin_wire.rs`: Rewritten with Holland/Simpson model.
  `ThinWire::new()` now takes (i0, j0, k_lo, k_hi, dx, dy, dz, radius,
  top_pec). Methods: `update_current()`, `deposit_current()`, `step()`
  (deprecated, use the two-phase version).
- `wasm/fdtd/src/solver.rs`: `Fdtd::step()` calls `wire.update_current()`
  after `update_h`, then `wire.deposit_current()` after `update_e`.
  `update_h()` no longer takes a thin-wire parameter (no H-correction).
  Terminal current is the wire current.
- `wasm/fdtd/src/geometry.rs`: Removed PEC marking at wire cells (wire
  is not PEC). Removed H-field correction. `build_snyder()` returns
  `(Feed, ThinWire)` with the new API.
- `wasm/fdtd/examples/dipole_fixture.rs`: Canonical dipole validation
  (two wire arms, center-fed, 0.1× CFL).
- `wasm/fdtd/examples/wire_isolated.rs`: Wire-alone stability test.

### Status

Implementation complete, unit tests pass (10/10). Dipole fixture validates
stability and reasonable current. Snyder geometry runs without NaN at
0.1× CFL. Full 10ns+ validation run pending (requires ~4 hours local at
11 steps/s, or ~28 hours on Forge at 1 step/s).
