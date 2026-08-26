# MFEM-WASM Implementation Plan

## Phase 1: Result Contract and Types
- [ ] Update `frontend/src/engine/types.ts` to include the new `FEMSimulationResult` shape.
- [ ] Update `frontend/src/engine/fem.ts` to handle the new result shape, including nullable impedance fields and `point_status`.
- [ ] Verify: Ensure results from the solver correctly map to the new frontend types.

## Phase 2: Worker Implementation (Port & Solver)
- [ ] Implement the new `mfem_solver.js` / `mfem_solver.wasm` production artifact.
- [ ] Implement the lumped-port excitation (1A drive) and extraction logic in the worker.
- [ ] Implement the worker generation ID for handling stale messages during cancellation/timeout.
- [ ] Verify: Test a single-frequency solve to ensure `Zin` and `S11` are produced correctly for the new port convention.

## Phase 3: Mesh and Validation
- [ ] Update mesh converter to support air-domain padding and port facet mapping.
- [ ] Implement validation checks for `missing_air_domain` and `invalid_port` conditions.
- [ ] Verify: Ensure invalid meshes are caught by the validator with correct diagnostic messages.

## Phase 4: UI Integration and Benchmarks
- [ ] Connect the Parametric UI to the new `Run FEM` production path.
- [ ] Implement the benchmark fixtures (waveguide, cavity, patch) with concrete geometry and reference data.
- [ ] Verify: Run the waveguide benchmark and verify it meets the target $|S11| \le 0.20$.
