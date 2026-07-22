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
- **M2:** Arbitrary parameters generate a working form and validated preset.
- **M3:** A script compiles to valid ModelIR and a mesh preview.
- **M4:** A scripted model produces an end-to-end frequency sweep and S11 plot.
- **M5:** A 918 MHz template can be swept and reports the best result.

## Initial implementation issues and dependency order

1. Define ParamSchema specification and JSON Schema.
2. Implement the QuickJS-WASM sandbox with timeout and memory limits.
3. Add the `defineParams`/`buildModel` loader and structured error reporting.
4. Build the schema-driven parameter form renderer in the web UI.
5. Implement Geometry API v1: `box`, `sheetRect`, `material`, and
   `lumpedPort`.
6. Define ModelIR v1 and its serializer.
7. Add IR validation for topology, materials, references, and ports.
8. Implement IR-to-Gmsh conversion and mesh diagnostics.
9. Integrate the solver client for frequency sweeps and S11 plots.
10. Add the documented `patch_918_petg.js` example.

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
- Start with Gmsh and a direct sparse solve for small models; add iterative
  methods and PML after the baseline is validated.
- Treat PETG properties as uncertain defaults and prefer user-calibrated
  values, since print settings can change effective dielectric behavior.
