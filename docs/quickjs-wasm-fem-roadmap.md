# QuickJS-WASM + FEM Roadmap

## Goal

Build a browser-first, scriptable workflow for designing directional 918 MHz
antennas. Users will define arbitrary parameters and generate reproducible
antenna geometry with QuickJS running in WASM. Generated geometry will be
validated, converted to a solver-neutral intermediate representation, meshed,
and evaluated with a frequency-domain FEM pipeline.

The first representative design is a 918 MHz microstrip patch or directional
parasitic antenna in or near a PETG enclosure. S11 and resonance shift are
first-class outputs.

## Scope

### In scope

- QuickJS-WASM execution in a browser sandbox.
- Schema-first arbitrary parameters: number, integer, boolean, string, enum,
  and vector values.
- Script-generated geometry, materials, ports, boundaries, and mesh hints.
- A versioned ModelIR independent of the scripting runtime and solver.
- Gmsh conversion and mesh diagnostics.
- Frequency-domain FEM integration with frequency sweeps and S11 plots.
- PETG-aware example models and documented material assumptions.

### Out of scope for the first release

- STEP/STL import and a full CAD kernel.
- Thermal or mechanical coupling.
- Nonlinear materials.
- Distributed cloud optimization.
- Full far-field, efficiency, and calibration workflows before the core
  S11 pipeline is validated.

## Architecture

```text
QuickJS-WASM script
        |
        v
defineParams() --> ParamSchema --> schema-driven web form
        |
        v
buildModel(params, api)
        |
        v
validated ModelIR --> Gmsh mesh --> FEM solver --> S11/results UI
```

The script runtime must not access the filesystem or network. It must enforce
deterministic execution, CPU/time limits, and memory limits. Scripts emit IR
through a constrained geometry API rather than directly controlling the
mesher or solver.

## Implementation phases

### Phase 0: Scope and contracts

- Confirm the 918 MHz directional antenna workflow and S11-first MVP.
- Freeze the versioning strategy for ParamSchema and ModelIR.
- Define browser-worker and future remote-solver boundaries.

### Phase 1: Script runtime and schema contract

- Run QuickJS-WASM in a Web Worker.
- Expose `defineParams()` and `buildModel(params, api)`.
- Disable network and filesystem capabilities.
- Add timeout, memory, cancellation, and deterministic execution guards.
- Return structured diagnostics with source location when available.

### Phase 2: Arbitrary parameter system

- Validate schema types, defaults, ranges, steps, units, and required values.
- Support custom cross-parameter constraints.
- Generate forms from the schema, including grouped fields and vector inputs.
- Map validation and script errors back to parameter keys.
- Persist and restore parameter sets as JSON presets.

### Phase 2b: Parametric script editor UI ✅ Complete

- Add a `/parametric` route and a top-level `ParametricEditorPage` (its own
  tab in the navbar, alongside the Wire Editor).
- Create a `dslStore` holding script source, param values, the last
  `ScriptExecution` result, run status, and diagnostics.
- Add a `ScriptEditorPanel` with an editable script textarea, a Run button
  that calls `runParametricScript` in a Web Worker, and inline
  phase-labelled diagnostics below the editor.
- Add a `ParamSchemaForm` component that renders the schema returned by
  `defineParams()`: number/int fields use sliders or number inputs, bool
  uses a checkbox, string uses a text input, enum uses a select, and vector
  uses a row of number inputs. Group headings from `schema.groups` are
  shown. Constraint violations are surfaced inline. Changing any value
  re-runs `buildModel` (debounced) and updates the IR in the store.
- Add a `ModelIRPreview` Three.js scene that visualises the current IR:
  `box` entities as translucent box meshes, `sheetRect` as thin planes,
  lumped ports as marker spheres, and the airbox/PML as a wireframe
  bounding cube. Materials are colour-coded.
- Support saving and restoring param values as JSON presets (reuse the
  `ProjectActions` pattern).
- Mobile layout: bottom-sheet tabs for Script, Params, Preview, and
  Diagnostics.

### Phase 3: Geometry API and ModelIR compiler

- Add v1 primitives: `box`, `sheetRect`, materials, lumped ports, airbox,
  radiation/PML boundaries, and mesh hints.
- Define canonical entities, region tags, materials, ports, boundaries, and
  mesh directives.
- Add checks for degenerate geometry, zero thickness, invalid references,
  topology problems, and unsupported overlaps.

### Phase 4: Meshing bridge

- Convert ModelIR to Gmsh geometry and physical groups.
- Apply local refinement around feeds, conductor edges, dielectric interfaces,
  and PETG surfaces.
- Return mesh quality statistics and actionable failure diagnostics.

### Phase 5: FEM solver integration

- Implement a frequency-domain electromagnetic solver interface.
- Start with first-order edge elements, PEC conductors, lumped excitation,
  and an absorbing outer boundary.
- Support mesh reuse and frequency sweeps.
- Produce input impedance, S11, resonant frequency, and optional field slices.
- Keep the interface compatible with local workers and a future remote solver.

### Phase 5a: MFEM-WASM feasibility (in progress)

- Use MFEM v4.8 as the finite-element core instead of expanding the temporary
  hand-written `fem1d` solver.
- Build MFEM as a single-process Emscripten module with MPI, HYPRE, METIS,
  SuiteSparse, OpenMP, and other optional native dependencies disabled.
- Compile a browser worker-compatible H(curl) smoke target that creates a
  tetrahedral mesh, constructs first-order Nedelec elements, and assembles a
  curl-curl plus mass bilinear form.
- Load MFEM through CPM.cmake at a pinned `v4.8` tag, using a reusable
  `CPM_SOURCE_CACHE`; keep the dependency source outside the repository.
- Allow `MFEM_SOURCE_DIR` as a local/offline override for patched or pre-fetched
  MFEM sources.
- Do not expose SWR, gain, or physical impedance as production results until
  the port excitation and field postprocessing are validated.
- Replace `fem1d` with the MFEM solver only after the feasibility target runs
  in the browser and a canonical benchmark is available.

### Phase 6: 918 MHz templates

- Add `patch_918_petg.js` with documented parameters and defaults.
- Add directional starter variants such as a Yagi and patch with parasitics.
- Document the target band, expected tradeoffs, and PETG uncertainty.

### Phase 7: Sweep and optimization

- Implement single-parameter, grid, and random sweeps.
- Add objectives and constraints for S11, resonance, gain, and
  front-to-back ratio.
- Save run history, the best candidate, and parameter provenance.

### Phase 8: PETG calibration

- Add calibrated material profiles with effective relative permittivity and
  loss tangent.
- Support measured resonance-shift input and fitting of effective PETG
  properties.
- Record printer, infill, layer direction, moisture, and measurement metadata.

### Phase 9: Validation and release gate

- Add schema, sandbox, IR, serializer, and determinism tests.
- Validate a canonical cavity or waveguide benchmark.
- Validate a patch without an enclosure against analytical/reference results.
- Verify the expected PETG resonance and bandwidth shift.
- Require mesh-convergence checks for S11 minima.

## Milestones

- **M1:** A browser script executes and returns a validated parameter schema.
- **M2 ✅:** The parametric editor tab is live: the example script runs, the
  schema-driven form renders, parameters update the IR in real time, and
  the 3D ModelIR preview is visible.
- **M3:** A script compiles to valid ModelIR and a mesh preview.
- **M4:** A scripted model produces an end-to-end frequency sweep and S11 plot
  using a validated MFEM H(curl) solver; the current `fem1d` WASM target is
  only a temporary integration smoke test.
- **M5:** A 918 MHz template can be swept and reports the best result.

## Initial implementation issues and dependency order

1. Define ParamSchema specification and JSON Schema.
2. Implement the QuickJS-WASM sandbox with timeout and memory limits.
3. Add the `defineParams`/`buildModel` loader and structured error reporting.
4. ✅ Add a `/parametric` route, `ParametricEditorPage`, and `dslStore`.
5. ✅ Add `ScriptEditorPanel` with inline diagnostics and worker integration.
6. ✅ Build the `ParamSchemaForm` schema-driven parameter form renderer.
7. ✅ Add `ModelIRPreview` Three.js scene for geometry visualisation.
8. ✅ Support JSON preset save/restore for parameter values.
9. Implement Geometry API v1: `box`, `sheetRect`, `material`, and
   `lumpedPort`.
10. Define ModelIR v1 and its serializer.
11. Add IR validation for topology, materials, references, and ports.
12. Implement IR-to-Gmsh conversion and mesh diagnostics.
13. Prove the MFEM-WASM H(curl) feasibility target on a tetrahedral benchmark.
14. Integrate the MFEM solver client for frequency sweeps and validated S11.
15. Add the documented `patch_918_petg.js` example.

The sequence is the default blocking order. Work may overlap when an interface
is stable, but later work must not bypass validation or silently couple the
QuickJS runtime to the FEM implementation.

## Proposed package boundaries

```text
packages/dsl-runtime   QuickJS-WASM host and sandbox
packages/dsl-api       Geometry, material, port, and boundary API
packages/param-schema  Types and schema validation
packages/model-ir      IR types, validation, and serialization
packages/mesh-bridge   ModelIR-to-Gmsh conversion
packages/solver-client FEM job and result interface
apps/web               Script editor, forms, previews, and results
examples/              918 MHz parametric antenna scripts
```

## Technical decisions

- Use a two-pass script contract: load and validate `defineParams()` first,
  then execute `buildModel()` with validated values.
- Keep parameter values as a dynamic dictionary while validating every value
  against the user-defined schema.
- Keep ModelIR solver-neutral so future script, GUI, or CAD frontends can share
  the FEM pipeline.
- Start with Gmsh and MFEM first-order edge elements for small models; use a
  real block-complex formulation or MFEM complex operators for frequency
  domain assembly, then add iterative methods and PML after the baseline is
  validated.
- Keep the hand-written `fem1d` implementation as a temporary browser ABI and
  numerical smoke test only; it is not a physical antenna solver.
- Treat PETG properties as uncertain defaults and prefer user-calibrated
  values, since print settings can change effective dielectric behavior.
