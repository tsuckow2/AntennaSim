# MFEM-WASM Simulation Results Plan

## Goal

Enable the browser-only deployment to run a scripted ModelIR through an MFEM-backed frequency-domain FEM solver and show the resulting simulation data in the UI. The first useful result is a validated frequency sweep with complex input impedance and S11. SWR and Smith-chart views may be derived from a validated S11 result. Gain must remain unavailable until the solver produces validated radiated fields and power.

This plan is intended for a later implementation agent. It assumes `VITE_ENGINE=wasm` and no backend API server.

## Current State

Completed or partially available:

- Parametric script execution, schema form, ModelIR generation, and 3D preview.
- Finite-thickness copper boxes in the 918 MHz PETG example.
- ModelIR validation, serialization, and an initial Gmsh `.geo` exporter.
- A browser FEM worker and `WasmEngine.simulateFEM()` contract.
- A temporary dependency-free `fem1d` WASM solver. It is a 1D smoke test, not a physical antenna solver.
- MFEM v4.8 CPM integration in `wasm/CMakeLists.txt` and `wasm/build.sh`.
- An MFEM H(curl) smoke target source in `wasm/mfem_smoke.cpp` and a worker loader in `frontend/src/engine/wasm/worker-mfem.ts`.
- Existing result components: `ResultsTabs`, `ImpedanceChart`, `SWRChart`, `SmithChart`, and gain/pattern components.
- **Existing FEM scaffolding**: The repository already contains FEM request/result types, SWR derivation, capability flags, a Zustand FEM state slice, cancellation logic, a FEM results panel, and unit tests in `frontend/src/engine/fem.ts` and `frontend/src/engine/__tests__/fem.test.ts`.

Important gaps:

- MFEM artifacts must finish building and be copied to `frontend/public/wasm/`, then validated in a browser-worker smoke test.
- `Run FEM` is already exposed in the UI, but current `worker-fem.ts` only runs MFEM smoke and then fails due to missing production ABI.
- Current mesh conversion is scaffolding only (independent per-box meshing, no conforming shared topology, no air-domain construction, empty boundary/port attributes).
- Existing FEM store/panel/types must be migrated to production semantics rather than rebuilt.
- Existing NEC-oriented result pathways must not be used as a fallback for FEM gain/pattern output.

## Decisions Required Before Implementation

These decisions remove the largest sources of rework:

- **Meshing & Domain Contract**: Do not build a general tetrahedral mesher in this work package.
  The first implementation must use a pinned, WASM-compatible meshing library
  or a deliberately bounded structured box-to-tetrahedra converter. The
  bounded converter is allowed only for axis-aligned, non-overlapping boxes
  and must reject sheets, arbitrary rotations, and unsupported intersections.
  **Crucially, the converter must establish a conforming volume domain**,
  including explicit air volumes, shared material interfaces, and boundary topology.
  A general Gmsh/TetGen port is a separate dependency decision.
- **Geometry mapping**: Every finite-thickness `box` maps to a physical 3D
  volume with center and full XYZ size in metres. Conductive boxes remain
  volume regions with material attributes; they are not collapsed into shells.
  `sheetRect` is surface-only and is rejected by the volume FEM path unless a
  surface-impedance formulation is explicitly implemented.
- **Worker data transfer**: Use transferable `ArrayBuffer`s for mesh input,
  field/result output, and packed numeric tables. Use structured clone only for
  small metadata and diagnostics. Do not require `SharedArrayBuffer` or cross-
  origin isolation for the first release.
- **Solver Formulation**: Start with a single-process direct solve for the small benchmark
  mesh. Add an iterative solver only after the real-block complex formulation
  and convergence behavior are validated.
- **Sweep size**: Limit the first UI run to 201 frequency points. Keep the full
  packed result in the worker contract, but downsample only at the chart layer
  if a later workflow permits larger sweeps.
- **Port Mapping**: Define a deterministic terminal-to-mesh mapping rule, including
  tolerance, required conductor contact, positive/negative facet orientation,
  and failure diagnostics for terminals on valid conductors, outside the mesh,
  inside dielectric, or ambiguous mappings.
- **ABI Specification**: Freeze a concrete request/response schema with ABI version,
  byte lengths, offsets, allocation/free functions, status codes, and
  endianness validation before implementing the solver.
- **Validation Criteria**: Define explicit numerical reference values and tolerances
  for the first result. Establish a canonical cavity or waveguide reference for
  field/error norms and a port impedance/reference-power check.
- **Result Representation**: Define a wire representation for non-finite values
  (e.g., `null` with a status flag) to ensure deterministic JSON serialization.
- **Solver Migration**: The production MFEM worker must become the sole implementation
  of `simulateFEM()`. Decide whether `worker-fem.ts` is removed, renamed as a
  mesh/solver adapter, or retained only for an explicit diagnostic path.

## Frozen Contracts (M4)

This section is normative for the first production handoff. If any field changes,
increment `abi_version` and `mesh_artifact_version` and update migration tests.

### A. Native ABI contract (v1)

- `abi_version`: `1` (uint32 in header)
- Endianness: little-endian only. Header includes sentinel `0x12345678`.
- Scalar widths: `uint32` and `float64` only.
- All numeric payload lengths are in bytes and must be multiples of scalar width.
- Byte payloads (`metadata_json`, response `message`) are raw UTF-8 and are not required to be scalar-aligned.
- Native exports:
  - `_mfem_solver_alloc(size_bytes: number): number`
  - `_mfem_solver_free(ptr: number): void`
  - `_mfem_solver_set_cancelled(flag: number): void`
  - `_mfem_solver_run(req_ptr: number, req_len: number, rsp_ptr_out: number, rsp_len_out: number): number`
- Call model:
  - `_mfem_solver_run` processes exactly one frequency point per call.
  - Frequency sweeps are orchestrated in the worker by repeated single-frequency calls.
- Status codes:
  - `0`: success
  - `1`: invalid_header
  - `2`: invalid_offsets
  - `3`: unsupported_version
  - `4`: invalid_geometry
  - `5`: invalid_port
  - `6`: solver_failure
  - `7`: cancelled
  - `8`: timeout
- Request header fields (ordered):
  - `abi_version`, `endianness_sentinel`, `flags`, `frequency_hz`
  - `vertex_count`, `tet_count`, `region_attr_count`, `boundary_facet_count`, `port_facet_count`, `port_pos_facet_count`
  - `vertices_offset`, `tets_offset`, `tet_region_offset`, `boundary_facets_offset`, `boundary_attr_offset`
  - `port_pos_facets_offset`, `port_neg_facets_offset`, `port_attr_offset`, `material_table_offset`, `metadata_json_offset`, `metadata_json_len`
- Request header byte layout (fixed 128 bytes, little-endian):
  - `0`: `abi_version` (u32)
  - `4`: `endianness_sentinel` (u32)
  - `8`: `flags` (u32)
  - `12`: reserved (u32)
  - `16`: `frequency_hz` (f64)
  - `24`: `vertex_count` (u32)
  - `28`: `tet_count` (u32)
  - `32`: `region_attr_count` (u32)
  - `36`: `boundary_facet_count` (u32)
  - `40`: `port_facet_count` (u32)
  - `44`: `port_pos_facet_count` (u32)
  - `48`: `vertices_offset` (u32)
  - `52`: `tets_offset` (u32)
  - `56`: `tet_region_offset` (u32)
  - `60`: `boundary_facets_offset` (u32)
  - `64`: `boundary_attr_offset` (u32)
  - `68`: `port_pos_facets_offset` (u32)
  - `72`: `port_attr_offset` (u32)
  - `76`: `material_table_offset` (u32)
  - `80`: `metadata_json_offset` (u32)
  - `84`: `vertices_len` (u32)
  - `88`: `tets_len` (u32)
  - `92`: `tet_region_len` (u32)
  - `96`: `boundary_facets_len` (u32)
  - `100`: `boundary_attr_len` (u32)
  - `104`: `port_pos_facets_len` (u32)
  - `108`: `material_table_len` (u32)
  - `112`: `port_neg_facets_offset` (u32)
  - `116`: `port_neg_facets_len` (u32)
  - `120`: `port_attr_len` (u32)
  - `124`: `metadata_json_len` (u32)
- Response header fields (ordered):
  - `abi_version`, `status_code`, `flags`, `residual_norm`, `iteration_count`
  - `zin_real_ohms`, `zin_imag_ohms`, `s11_real`, `s11_imag`
  - `timing_assembly_ms`, `timing_solve_ms`, `timing_post_ms`
  - `field_blob_offset`, `field_blob_len`, `message_offset`, `message_len`
- Response header byte layout (fixed 104 bytes, little-endian):
  - `0`: `abi_version` (u32)
  - `4`: `status_code` (u32)
  - `8`: `flags` (u32)
  - `12`: reserved (u32)
  - `16`: `residual_norm` (f64)
  - `24`: `iteration_count` (u32)
  - `28`: reserved (u32)
  - `32`: `zin_real_ohms` (f64)
  - `40`: `zin_imag_ohms` (f64)
  - `48`: `s11_real` (f64)
  - `56`: `s11_imag` (f64)
  - `64`: `timing_assembly_ms` (f64)
  - `72`: `timing_solve_ms` (f64)
  - `80`: `timing_post_ms` (f64)
  - `88`: `field_blob_offset` (u32)
  - `92`: `field_blob_len` (u32)
  - `96`: `message_offset` (u32)
  - `100`: `message_len` (u32)
- Ownership:
  - TS allocates request buffer via `_mfem_solver_alloc`, copies data, calls run, then frees request ptr.
  - Native allocates response buffer; TS copies/reads it and calls `_mfem_solver_free`.
  - Worker thread is sole owner of transferred `ArrayBuffer`s after transfer.
- Max limits (hard-fail with structured error before native call):
  - `max_vertices = 350000`
  - `max_tets = 2000000`
  - `max_true_dofs = 1500000`
  - `max_points = 201`
  - `max_request_bytes = 512 MiB`
  - `per_frequency_timeout_ms = 30000`
- Out-parameter semantics:
  - `rsp_ptr_out` and `rsp_len_out` are pointers to 32-bit slots in WASM memory.
  - Worker allocates these slots with `_mfem_solver_alloc(8)` and passes a 4-byte aligned pointer.
  - Native writes the response pointer and response byte length to those slots before returning.
  - Worker reads both 32-bit values then frees the slot buffer via `_mfem_solver_free`.
  - On non-success status, response header + message may still be present and must be read.
- Cancellation semantics:
  - M4 cancellation/timeout mechanism is worker termination and restart (authoritative behavior).
  - On cancel/timeout, main thread terminates the MFEM worker, rejects all pending requests,
    creates a fresh worker instance, and marks in-flight points as `cancelled` or `timeout`.
  - Main-thread coordinator must persist per-point status (`7`/`8`) before rejecting the aggregate run.
  - Worker instances use a generation ID; responses from older generations are ignored as stale.
  - `_mfem_solver_set_cancelled(1)` remains optional for future cooperative cancellation,
    but is not required for M4 correctness.
  - Cancelled solve status maps to `7`; timeout maps to `8` at the result protocol layer.

### A1. Payload binary layouts (v1)

- `vertices` payload: packed `float64[x, y, z]`, length `vertex_count * 3 * 8`.
- `tets` payload: packed `uint32[v0, v1, v2, v3]`, length `tet_count * 4 * 4`.
- `tet_region` payload: packed `uint32[region_attr]`, length `tet_count * 4`.
- `boundary_facets` payload: packed `uint32[v0, v1, v2]`, length `boundary_facet_count * 3 * 4`.
- `boundary_attr` payload: packed `uint32[attr]`, length `boundary_facet_count * 4`.
- `port_pos_facets` and `port_neg_facets`: packed `uint32[facet_index]` arrays, total count `port_facet_count`.
- `port_neg_facet_count` is derived as `port_facet_count - port_pos_facet_count` and must be `>= 1`.
- `port_attr` payload: packed `uint32[attr]`, one per port facet index entry.
- `material_table` payload: fixed 40-byte records:
  - `0`: `region_attr` (u32)
  - `4`: reserved (u32)
  - `8`: `epsilon_r` (f64)
  - `16`: `mu_r` (f64)
  - `24`: `conductivity_s_m` (f64)
  - `32`: `loss_tangent` (f64)
- `metadata_json` payload: UTF-8 JSON with required keys:
  - `mesh_artifact_version`
  - `model_ir_version`
  - `reference_impedance_ohms`
  - `points_total`
  - `point_index`
  - `solver_formulation`
  - `absorbing_boundary`

### A2. Fixed M4 choices (v1)

- Solver formulation: `real_block_complex`.
- Absorbing boundary: `silver_muller_first_order` on outer airbox facets.
- Sweep count field name: `points` (sample count) across UI, worker, and tests.
- Field blob for M4 is optional; if absent, response must set `field_blob_offset = 0` and `field_blob_len = 0`.

### B. Mesh artifact contract (v1)

- `mesh_artifact_version`: `1`
- Supported geometry input for M4:
  - `box` only for volumetric solve path.
  - `sheetRect` rejected for M4 with explicit diagnostic.
- Air-domain rule:
  - Use `ModelIR.boundaries[*].padding_m` to build an explicit outer air volume.
  - If no boundary entry exists, fail validation with `missing_air_domain`.
  - All solid regions must be fully embedded within this air volume.
- Conforming topology requirements:
  - Shared vertices/facets at touching interfaces.
  - No duplicated coincident facets with conflicting ownership.
  - Deterministic vertex/tet ordering for stable serialization.
- Attribute table (reserved IDs):
  - Region attributes:
    - `1`: air
    - `1000+`: user materials in deterministic material-name order
  - Boundary attributes:
    - `10`: exterior_absorbing
    - `20`: conductor_pec
    - `30`: port_positive
    - `31`: port_negative

### C. Port mapping and sign convention (v1)

- Port artifact representation: facet sets (`port_positive_facets`, `port_negative_facets`).
- Excitation model (frozen for M4): 1 A lumped-current drive between positive and negative facet sets.
- Weak-form source term for port drive:
  - Add to RHS: `+∫_{S+} Jp · v dS - ∫_{S-} Jp · v dS`
  - `Jp` is uniform tangential surface current density on each terminal set, scaled so total driven current is exactly `1 A`.
- Terminal-to-facet mapping:
  - Candidate facets are conductor-air interface facets only.
  - Point-to-facet tolerance: `tol = max(1e-6 m, 0.25 * local_edge_length)`.
  - If zero candidates: `invalid_port_no_contact`.
  - If multiple disconnected candidates: `invalid_port_ambiguous_contact`.
  - If facet not on conductor-air interface: `invalid_port_material_contact`.
- Orientation/sign:
  - Current is positive from `positive` terminal facet set to `negative` terminal facet set.
  - Port facets are excluded from PEC constraints; non-port conductor-air facets receive PEC constraints.
  - Voltage is computed as `V = ∫_L E · dl`, where `L` is the straight reference segment from positive-facet centroid to negative-facet centroid.
  - Current is extracted as oriented current flux through positive terminal facets (must match `1 A` within 1% in acceptance tests).
  - Reference plane is the centroid plane midway between positive/negative facet-set centroids.
  - `Zin = V / I` under that sign convention.
  - S11 uses `reference_impedance_ohms` from request.

### D. Result-invalid-value wire format (v1)

- Wire format must not use JSON `Infinity`/`NaN`.
- For each frequency point:
  - numeric fields may be `null` only when `point_status != "ok"`.
  - `point_status` required: `ok | non_finite | solver_error | cancelled | timeout`.
  - `point_message` optional human-readable detail.
- UI normalization:
  - Charts skip invalid points and show warnings.
  - Capability flags control tab visibility. Gain/far-field tabs hidden when false.
- Native status to `point_status` mapping:
  - `0 -> ok`
  - `6 -> solver_error`
  - `7 -> cancelled`
  - `8 -> timeout`
  - non-finite numeric payload -> `non_finite`

### E. Benchmark fixture and tolerances (v1)

- Canonical fixture set:
  - `benchmark_rect_waveguide_te10`
  - `benchmark_cavity_resonance`
  - `patch_918_petg_m4` (integration fixture)
- Fixture definitions must be checked in under:
  - `docs/benchmarks/mfem-wasm/benchmark_rect_waveguide_te10.md`
  - `docs/benchmarks/mfem-wasm/benchmark_cavity_resonance.md`
  - `docs/benchmarks/mfem-wasm/patch_918_petg_m4.md`
- Each fixture doc is required to include:
  - geometry dimensions (meters)
  - material parameters (`epsilon_r`, `mu_r`, conductivity/loss tangent)
  - airbox padding and boundary kind
  - explicit port terminal coordinates and mapped facet-attribute IDs
  - mesh refinement levels used in convergence checks
  - probe definitions (field and/or impedance probes with coordinates)
  - reference numeric values and tolerance bands
- Minimum numeric acceptance:
  - Relative residual per frequency: `<= 1e-6`
  - Two-mesh impedance convergence (real and imag): `<= 5%` difference
  - S11 magnitude difference between two meshes: `<= 0.05`
  - Port power-sign consistency check must pass at all accepted points.
- Fixture numeric targets for M4 release gate:
  - `benchmark_rect_waveguide_te10`:
    - frequency: `9.5 GHz`, reference impedance `50 ohms`
    - waveguide dimensions: `a = 22.86 mm`, `b = 10.16 mm`, length `50 mm`
    - port plane centered at `z = 0`, opposing terminal facets on waveguide feed cross section
    - target `|S11| <= 0.20`
    - target `Re(Zin)` in `[35, 65] ohms`
    - target `Im(Zin)` in `[-20, 20] ohms`
  - `benchmark_cavity_resonance`:
    - cavity dimensions: `100 mm x 80 mm x 60 mm` PEC enclosure with air fill
    - probe point: cavity center `(0, 0, 0)` in local fixture coordinates
    - first resonance frequency error `<= 3%` against fixture reference
    - field-energy relative error `<= 5%` at reference probe set
  - `patch_918_petg_m4`:
    - use checked-in model from `frontend/src/dsl/examples/patch_918_petg.js`
    - port terminals and mesh mapping recorded in fixture doc
    - sweep includes `918 MHz`
    - resonance in `[880, 950] MHz`
    - minimum `S11_dB <= -6 dB`

### F. UI migration boundary (v1)

- `toNECFrequencyResults()` is non-production for FEM and must not be on the production execution path.
- Synthetic gain values (for example sentinel NEC gain fields) are forbidden in FEM mode.
- Production `Run FEM` enablement requires:
  - production solver capability true,
  - mesh/domain validation pass,
  - port mapping validation pass.
- Capability source:
  - `mfemSmokeAvailable` is diagnostic-only and must not unlock production runs.
  - `mfemProductionAvailable` unlocks production `Run FEM`.
  - UI must show explicit reason when production capability is false.


## Scope and Physical Honesty

### Production gating rule

- Keep the current FEM UI and controls as scaffolding.
- Keep `Run FEM` disabled for production runs until the worker path is switched from smoke-only execution to a real MFEM solver ABI.
- Provide a separate explicit `Run MFEM Smoke` diagnostic action for build/runtime validation.

### First release target

A browser-worker FEM run must:

1. Accept validated ModelIR and a frequency sweep.
2. Build or load a tetrahedral volume mesh.
3. Construct first-order Nedelec H(curl) spaces in MFEM.
4. Assemble a frequency-domain Maxwell system with material coefficients.
5. Apply PEC conductor treatment, a documented absorbing boundary, and one lumped port.
6. Solve each frequency point in a worker.
7. Extract port voltage/current and return complex input impedance and S11.
8. Render S11, impedance, resonance, and SWR/Smith-derived views in the UI.

### Explicitly deferred

- Do not show gain from the 1D solver or from an unvalidated field solution.
- Do not show radiation efficiency or directivity until radiated power and far-field sampling are implemented and benchmarked.
- Do not label the temporary `fem1d` output as production FEM.
- Do not silently fall back from MFEM to `fem1d` for user-facing antenna results. A fallback may be retained for developer smoke tests only.

## Work Packages

### 1. Lock the MFEM-WASM build

Files:

- `wasm/CMakeLists.txt`
- `wasm/build.sh`
- `wasm/mfem_smoke.cpp`
- `frontend/public/wasm/`

Tasks:

- Finish a clean CPM-backed MFEM v4.8 build with optional native backends disabled.
- Produce `mfem_smoke.js` and `mfem_smoke.wasm`.
- Run the exported H(curl) smoke function and verify a positive true-DOF count.
- Copy artifacts into the Vite public WASM directory.
- Keep `CPM_SOURCE_CACHE` configurable and preserve `MFEM_SOURCE_DIR` for offline builds.
- Document the required emsdk and first-build network requirement.
- Add production MFEM solver artifact target in CMake:
  - output module name: `mfem_solver.js` / `mfem_solver.wasm`
  - export factory: `createMfemSolver`
  - export functions: `_mfem_solver_alloc`, `_mfem_solver_free`, `_mfem_solver_set_cancelled`, `_mfem_solver_run`

Acceptance:

- `./wasm/build.sh` completes from a clean build directory.
- The smoke module loads in a browser worker and returns a positive DOF count.
- The production solver module loads in a browser worker and exposes required symbols.
- No backend service is required.

### 2. Add a real MFEM worker boundary

Files:

- `frontend/src/engine/wasm/worker-mfem.ts`
- `frontend/src/engine/wasm/mfem-module.d.ts`
- `frontend/src/engine/wasm/index.ts`
- `frontend/src/engine/types.ts`

Tasks:

- Keep `runMfemSmoke()` for diagnostics and development validation.
- Migrate production `simulateFEM()` to a single worker path using the new MFEM solver ABI.
- Retire or repurpose `worker-fem.ts` so production runs cannot execute smoke-only code paths.
- Add worker cancellation, timeout, and structured progress diagnostics:
  `loading`, `meshing`, `assembly`, `solving`, `postprocess`, `complete`, `failed`, `cancelled`.
- Ensure every pending request is rejected when the worker errors, times out, or terminates.
- Keep the MFEM worker protocol separate from NEC worker protocol and types.

Acceptance:

- A unit/integration test verifies production requests do not hit smoke-only APIs.
- A browser smoke action reports MFEM load and a positive DOF count.
- Production worker protocol emits progress by stage and per-frequency completion.
- Large buffers are transferred as transferables without clone fallback.

### 3. Replace the initial Gmsh-only handoff with a mesh artifact contract

Files:

- `frontend/src/dsl/gmsh.ts`
- `frontend/src/dsl/model-ir.ts`
- New mesh bridge module under `frontend/src/engine/` or `frontend/src/dsl/`

Tasks:

- Define a versioned conforming mesh artifact containing:
  vertices, tetrahedra, region/material attributes, boundary attributes,
  port attributes, deterministic ordering metadata, and quality statistics.
- Define one complete domain construction rule before solver work:
  explicit air volume generation/recognition, shared interfaces between touching regions,
  exterior boundary facets, conductor boundary facets, and port facet ownership.
- Ensure finite-thickness `box` entities are volume regions.
- Keep `sheetRect` as explicit surface-only primitive and reject it for volume FEM unless a surface-impedance path exists.
- Map ModelIR material and boundary names to stable integer attributes.
- Define deterministic terminal-to-mesh mapping:
  entity type (face/edge), tolerance relative to local mesh size,
  required conductor contact, orientation convention, and failure diagnostics.
- Add diagnostics for unsupported overlaps, disconnected domains, degenerate elements,
  missing air domain, unresolved ports, ambiguous port mapping, and same-facet terminal collapse.

Implementation choice:

- Prefer a pinned, browser-compatible Gmsh or TetGen WASM module if one is
  selected and validated.
- The default first implementation is a bounded structured converter for
  axis-aligned, non-overlapping `box` entities: subdivide the Cartesian box
  arrangement and split each hexahedral cell into tetrahedra deterministically.
  It must reject `sheetRect`, rotated boxes, arbitrary intersections, and
  geometry that cannot be represented without ambiguous material ownership.
- Do not implement a general tetrahedral mesher from scratch in this plan.
- Do not pass `.geo` text directly to MFEM and assume meshing happened.

Acceptance:

- The 918 MHz finite-thickness patch model produces a deterministic conforming artifact
  with air region, dielectric/conductor regions, exterior boundary facets, and resolvable port attributes.
- Mesh statistics include element count, node count, minimum quality, material/boundary counts, and port mapping status.
- Invalid geometry or unresolved port mapping fails before MFEM is called.

### 4. Implement the MFEM frequency-domain solver ABI

Files:

- New C++ solver source under `wasm/`, for example `mfem_solver.cpp`.
- `wasm/CMakeLists.txt`.
- New TypeScript declarations and worker adapter.

Tasks:

- Accept a mesh artifact, material table, port definition, boundary definition, and one frequency in a compact ABI.
- Freeze ABI details before C++ implementation:
  version number, byte lengths, byte offsets, scalar widths, units,
  status/error encoding, allocation/free functions, max sizes, and endianness marker.
- Use packed `Uint32Array` and `Float64Array` payloads with explicit little-endian layout,
  and a strict metadata header (`abi_version`, counts, offsets, units, flags).
- Define ownership transfer semantics for each buffer crossing worker/native boundary.
- Use first-order Nedelec elements.
- Assemble the curl-curl and frequency-scaled mass terms.
- Choose and document one formulation up front (real block-complex for first release).
- Include relative permittivity, relative permeability, dielectric loss tangent, and conductor treatment in the coefficients.
- Choose and document conductor model for first release:
  Mode A (default): PEC on conductor-air interfaces.
  Mode B (deferred): finite conductivity volume loss model.
- Implement one documented absorbing boundary first and define where it applies
  (outer airbox facets with normal orientation); add PML only after baseline stability.
- Implement a physically defined lumped port and extract both voltage and current.
- Return solver status, residual/iteration data, field availability, and stage timings.
- Reuse mesh and symbolic assembly data across a frequency sweep where possible.

Acceptance:

- A canonical waveguide/cavity benchmark passes with explicit numeric tolerances recorded in repo docs.
- Port voltage/current produce finite `Zin` and S11 with consistent sign convention and reference impedance.
- Two mesh resolutions meet a defined convergence threshold on impedance and residual.
- ABI error cases return structured errors (not NaN-filled points).

### 5. Define a frontend FEM result contract

Files:

- `frontend/src/engine/fem.ts`
- `frontend/src/engine/types.ts`
- New result adapter/types module as needed.

Suggested result shape:

```ts
interface FEMSimulationResult {
  simulation_id: string;
  engine: "mfem-wasm";
  mesh: {
    elements: number;
    nodes: number;
    min_quality?: number;
    warnings: string[];
  };
  reference_impedance_ohms: number;
  frequency_data: Array<{
    frequency_mhz: number;
    impedance_real_ohms: number | null;
    impedance_imag_ohms: number | null;
    s11_real: number | null;
    s11_imag: number | null;
    s11_db: number | null;
    swr: number | null;
    point_status: "ok" | "non_finite" | "solver_error" | "cancelled" | "timeout";
    point_message?: string;
  }>;
  resonant_frequency_mhz: number | null;
  capabilities: {
    impedance: boolean;
    s11: boolean;
    swr: boolean;
    smith_chart: boolean;
    near_field: boolean;
    far_field: boolean;
    gain: boolean;
  };
  warnings: string[];
}
```

Tasks:

- Compute SWR only from validated S11 and emit `point_status` when `|S11| >= 1` or values are non-finite.
- Keep complex impedance and S11 as separate values; do not reconstruct one from the other unless the reference impedance is recorded.
- Include reference impedance and solver/mesh provenance.
- Include `reference_impedance_ohms` in every result and preserve the exact
  value used to calculate S11.
- Mark gain and far-field capabilities false until implemented.
- Define `Infinity`, `-Infinity`, and `NaN` handling at wire level: represent as `null` with status/message.
- Return at most 201 points from the first UI run; chart downsampling is a
  presentation concern, not a change to solver data.

Acceptance:

- Result serialization is deterministic and test-covered.
- S11 dB, SWR, and Smith coordinates agree with existing chart expectations.
- Unsupported capabilities are visibly disabled rather than shown as zeros.
- Invalid/non-finite points survive round-trip serialization without ambiguity.

### 6. Connect the Parametric UI to FEM execution

Files:

- `frontend/src/pages/ParametricEditorPage.tsx`
- `frontend/src/stores/dslStore.ts`
- New FEM result store or store slice.
- `frontend/src/components/dsl/ScriptEditorPanel.tsx`
- `frontend/src/components/dsl/ModelIRPreview.tsx` if run status is shown there.

Tasks:

- Keep existing frequency sweep controls and rename `steps` to `points` in UI/protocol
  (or explicitly define `steps` as point count everywhere) to remove ambiguity.
- Add a `Run FEM` action enabled only when script diagnostics and ModelIR validation are clean.
- Gate `Run FEM` on production solver capability flag in addition to validation.
- Show stages: validating, meshing, loading MFEM, assembling, solving, complete, failed, cancelled.
- Preserve the last successful result while a new run is in progress, with clear provenance.
- Add cancellation and stale-result protection when parameters or script source change.
- Keep NEC simulation controls and FEM controls separate.

Acceptance:

- A user can run the example patch from the parametric page without a backend once production solver capability is enabled.
- The UI shows progress, errors, mesh information, and completion time.
- Changing parameters invalidates or marks the previous result stale.

### 7. Render FEM results in the existing results UI

Files:

- `frontend/src/components/results/ResultsTabs.tsx`
- `frontend/src/components/results/ImpedanceChart.tsx`
- `frontend/src/components/results/SWRChart.tsx`
- `frontend/src/components/results/SmithChart.tsx`
- New FEM results panel/page if existing result tabs are too NEC-specific.

Tasks:

- Add an FEM result mode or adapter rather than forcing FEM data into unrelated NEC fields.
- Render S11 dB versus frequency as the primary chart.
- Render real and imaginary impedance versus frequency.
- Render SWR derived from S11.
- Render Smith chart coordinates derived from complex S11.
- Show resonant frequency and mesh summary.
- Display warnings and capability badges, especially that gain is not yet available.
- Add empty, loading, failed, and stale-result states.
- Do not surface synthetic NEC gain values in FEM result mode.

Acceptance:

- The completed browser run visibly produces S11, impedance, SWR, Smith chart, and resonance output.
- No gain chart appears unless `capabilities.gain` is true.
- Charts handle one-point sweeps, invalid points, and failed runs without crashing.

### 8. Validation and release gate

Tasks:

- Add unit tests for mesh mapping, material mapping, port mapping, SWR, Smith coordinates, and result capability flags.
- Add worker protocol tests for success, timeout, cancellation, and malformed WASM responses.
- Add worker protocol compatibility tests for migration from old FEM API:
  - `steps` field is rejected in favor of `points`.
  - no NEC-shape fallback is used for production FEM result rendering.
  - non-finite values are encoded as `null` + `point_status`.
- Add browser smoke coverage for loading `mfem_smoke.js` and the production solver module.
- Add canonical benchmark tests with explicit expected values/tolerances and mesh-convergence checks.
- Add resource-limit tests for max mesh sizes, max DOFs, memory budget, and per-frequency timeout behavior.
- Run the full frontend suite and a clean WASM build.
- Verify static hosting with `docker-compose.wasm.yml` and no backend container.

M4 completion criteria:

- A scripted finite-thickness 918 MHz model runs entirely in a browser worker.
- A real MFEM H(curl) solver, not `fem1d`, produces a frequency sweep.
- The UI displays validated S11 and impedance results plus derived SWR/Smith views.
- Mesh provenance, warnings, and unsupported gain capability are visible.
- A benchmark and at least one mesh-convergence check pass.
- Resource-limit and cancellation behaviors return structured errors without browser lockups.

## Recommended Execution Order

1. Finish and smoke-test the CPM MFEM build.
2. Expose the MFEM worker smoke diagnostic.
3. Freeze mesh and FEM result contracts.
4. Freeze exact ABI, port-mapping, and invalid-value wire semantics.
5. Implement conforming domain mesh conversion (or validated external mesher path).
6. Implement MFEM H(curl) single-frequency solve with chosen formulation.
7. Add frequency sweep reuse, progress, cancellation, and `WasmEngine` migration.
8. Migrate existing FEM state/panel to production capability gating.
9. Adapt charts in FEM mode and remove NEC-gain fallback behavior.
10. Run benchmark, convergence, resource-limit, static deployment, and full regression checks.

## Risks and Decisions for the Implementing Agent

- MFEM source compilation makes the WASM build substantially heavier; preserve CPM caching and avoid shipping MFEM headers or source to the frontend.
- Browser memory limits may require small meshes, chunked frequency sweeps, and explicit cancellation.
- Complex sparse solves may require a real block system and a carefully chosen solver that MFEM can build without HYPRE.
- Gmsh itself may be too heavy for the first browser build; keep a narrow box/tetrahedral mesher fallback only if it is validated.
- A visually complete results panel must not imply that gain exists before far-field and power calculations are implemented.
- Keep the backend FEM prototype isolated; it is useful as a numerical reference but is not part of the WASM deployment path.
