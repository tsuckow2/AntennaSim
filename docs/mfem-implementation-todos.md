# MFEM-WASM Implementation Todos (SWR + Gain Rendering)

Goal: Implement the revised MFEM-WASM plan so browser FEM runs produce validated SWR output and a gain rendering workflow with explicit capability gating.

## Phase 1: ABI + Solver Artifact

- [ ] Add production MFEM solver target in `wasm/CMakeLists.txt` producing `mfem_solver.js` and `mfem_solver.wasm`.
- [ ] Export required C ABI symbols: `_mfem_solver_alloc`, `_mfem_solver_free`, `_mfem_solver_run`, optional `_mfem_solver_set_cancelled`.
- [ ] Implement/declare the fixed request/response header layouts from `docs/mfem-wasm-simulation-results-plan.md` (Frozen Contracts A/A1/A2).
- [ ] Add a TypeScript ABI pack/unpack layer with strict byte-length and offset validation.

Acceptance:
- Build produces loadable production solver artifact.
- ABI validation rejects malformed requests and reports structured status codes.

## Phase 2: Worker Migration + Cancellation

- [ ] Move production `simulateFEM()` to a single MFEM production worker path.
- [ ] Keep smoke diagnostics separate (`runMfemSmoke()` only).
- [ ] Implement progress events for stages + per-frequency completion.
- [ ] Implement M4 cancellation/timeout via worker terminate/recreate and generation IDs.
- [ ] Persist per-point `cancelled`/`timeout` status before rejecting aggregate run.

Acceptance:
- Production requests cannot hit smoke-only code.
- Stale worker messages are ignored.

## Phase 3: Conforming Mesh + Port Mapping

- [ ] Upgrade mesh bridge to produce conforming topology (shared interfaces) with explicit air domain.
- [ ] Enforce `box`-only volumetric solve path for M4; reject `sheetRect` with diagnostics.
- [ ] Emit boundary and port attributes (`exterior_absorbing`, `conductor_pec`, `port_positive`, `port_negative`).
- [ ] Implement deterministic terminal-to-facet mapping with tolerance and failure diagnostics.
- [ ] Emit `port_attr` payload aligned with facet ownership.

Acceptance:
- 918 MHz model yields deterministic artifact with air/material/boundary/port attributes.
- Invalid geometries/ports fail before solver invocation.

## Phase 4: Solver + Port Extraction

- [ ] Implement first-order H(curl) frequency-domain solve (M4 fixed formulation: `real_block_complex`).
- [ ] Implement first-order Silver-Muller absorbing boundary on outer airbox facets.
- [ ] Apply PEC only on non-port conductor-air facets.
- [ ] Implement frozen lumped-port convention:
  - 1 A current drive between positive/negative terminal facet sets,
  - voltage from line integral along centroid segment,
  - current from oriented flux through positive terminal set.
- [ ] Return structured per-point outputs (`Zin`, complex `S11`, timings, residual, status).

Acceptance:
- Per-point residual and convergence satisfy documented tolerances.
- Port convention produces reproducible sign and impedance.

## Phase 5: Frontend Result Contract + SWR

- [ ] Update frontend FEM result types to nullable numeric wire format with `point_status`.
- [ ] Ensure no `NaN`/`Infinity` in wire data; map to `null` + status.
- [ ] Derive SWR/Smith safely from validated S11 points.
- [ ] Keep/reference exact `reference_impedance_ohms` in each result.

Acceptance:
- JSON round-trip preserves invalid-point semantics.
- SWR and Smith rendering remain stable with sparse/invalid points.

## Phase 6: Gain Rendering Workflow

- [ ] Keep `capabilities.gain=false` until far-field outputs are available.
- [ ] Ensure gain UI rendering path works in both states:
  - disabled/unavailable messaging when `capabilities.gain=false`,
  - real plot rendering when gain arrays are present.
- [ ] Add explicit no-synthetic-gain rule for FEM mode (no NEC fallback/sentinel values).
- [ ] Add feature-ready contract for future far-field payload (`gain_dbi[]`, angles/frequency index).

Acceptance:
- Gain panel renders correctly without crashing in disabled and enabled states.
- No fabricated gain values are displayed.

## Phase 7: Benchmarks + Deployment

- [ ] Add fixture docs and executable tests for:
  - `benchmark_rect_waveguide_te10`
  - `benchmark_cavity_resonance`
  - `patch_918_petg_m4`
- [ ] Add ABI golden-buffer tests and worker protocol tests.
- [ ] Add static deployment smoke test via `docker-compose.wasm.yml`.

Acceptance:
- Benchmarks pass with documented tolerances.
- Browser-only deployment runs without backend.

## Definition of Done

- SWR simulation runs in browser via production MFEM path and renders in UI.
- Gain rendering path is production-safe and stable:
  - disabled state is explicit until far-field exists,
  - enabled state can render real gain data when provided.
- All validation, cancellation, and stale-result protections pass tests.
