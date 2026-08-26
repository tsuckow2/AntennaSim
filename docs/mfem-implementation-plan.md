# MFEM-WASM Implementation Plan

This plan details the steps required to implement a functional frequency-domain FEM solver using MFEM in the browser, ultimately enabling SWR and Gain plots.

## Phase 1: Core Data Contracts

### 1.1 Mesh Artifact Contract

Define a versioned contract for a tetrahedral mesh. This is the primary data structure passed from the geometry engine to the solver.

- **Vertices**: `Float64Array` of XYZ coordinates.
- **Elements**: `Uint32Array` of 4 indices per tetrahedron.
- **Materials**: Mapping of region IDs to material properties (permittivity, permeability, loss tangent).
- **Boundaries**: Mapping of surface IDs to types (PEC, ABC, Port).
- **Ports**: Mapping of port IDs to specific facet pairs and lumped terminal properties.

### 1.2 Solver ABI

Define the `SimulationRequest` and `SimulationResponse` types for the Web Worker boundary.

- **Request**: `meshArtifact`, `frequency` (float), `materialTable`, `portDefinitions`.
- **Response**: `status` (success/error), `portResults` (complex voltage/current), `diagnostics` (iterations, residuals).

### 1.3 Frontend Result Contract

Define a unified `FEMResult` object for the frontend state.

- It should contain a collection of `PortResult` objects, each indexed by port ID, and a global status object.

## Phase 2: Meshing & Geometry

### 2.1 Bounded Structured Converter

Implement a C++/WASM module that:

- Takes a set of axis-aligned boxes.
- Subdivides boxes into a regular grid of hexahedra.
- Splits each hexahedron into 5 or 6 tetrahedra deterministically.
- Ensures no overlapping boxes are processed.

### 2.2 Validation

Implement a pre-flight check that rejects:

- Rotated boxes (non-axis-aligned).
- Overlapping boxes (intersecting volumes).
- Sheets (zero-thickness boxes).

## Phase 3: C++ Solver Implementation

### 3.1 MFEM Solver Core

Implement the curl-curl system assembly using first-order Nedelec elements.

### 3.2 Boundary Conditions

- **PEC**: Zero-impedance boundary condition on conductor surfaces.
- **ABC**: Simple first-order absorbing boundary condition.

### 3.3 Lumped Port

Implement the extraction of $V$ and $I$ from a port facet.

### 3.4 Frequency Sweeps

Implement logic to solve for multiple frequencies, optimizing by reusing the symbolic assembly of the curl-curl operator.

## Phase 4: Worker Integration

### 4.1 Worker Adapter

Update `frontend/src/engine/wasm/worker-mfem.ts` to handle the new ABI.

### 4.2 Data Transfer

Ensure `meshArtifact` and `portResults` are transferred as `ArrayBuffer`s to minimize cloning overhead.

## Phase 5: UI & Visualization

### 5.1 S11 & Z-Parameter Calculation

Implement utilities to compute S11 from $V$ and $I$ at the port.

### 5.2 SWR & Gain Plots

Create React components to render SWR and Gain based on the calculated S11 and field data.
