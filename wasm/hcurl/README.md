# hcurl-fem

A pure-Rust H(curl)-conforming finite-element solver for time-harmonic
electromagnetics, compiled to WebAssembly for in-browser antenna simulation.
It solves the vector wave equation on tetrahedral meshes with first-order
Nédélec edge elements:

```text
curl (μr⁻¹ curl E) − k₀² εc E = −jωμ₀ J_imp        in Ω
```

with a first-order Silver–Müller absorbing boundary, a 1 A impressed-current
lumped port, and a Stratton–Chu near-to-far-field transform.

This crate is a faithful port of the validated TypeScript reference solver
(`frontend/src/engine/fem-solver.ts`). The numerical core has **zero
WASM-specific code**: the same `solve_point` entry point runs natively under
`cargo test` and inside the browser, so validation is fast and native/WASM
parity is structural rather than hoped-for.

> **Status:** complete and validated. 47 Rust tests pass (44 unit + 2
> integration + 1 doc-test); native and real-WASM FFI runs agree with the
> TypeScript reference to ~1e-9 Ω on the 3 GHz dipole fixture.

## Physical and sign conventions

| Item | Convention |
|------|-----------|
| Time dependence | `e^{+jωt}` (engineering) |
| PDE | `curl(μr⁻¹ curl E) − k₀² εc E = −jωμ₀ J_imp` |
| Absorbing boundary | Silver–Müller: `+j k₀ B` added to the LHS |
| Far-field phase | `e^{+jk r̂·r′}` (outgoing, consistent with `e^{+jωt}`) |
| Magnetic field | `H = j curl E / (ωμ)` |
| Equivalent currents | `J_s = n × H`, `M_s = E × n` (Stratton–Chu) |
| Port excitation | exact 1 A impressed current: `J = L_gap·d̂ / V_gap` |
| Terminal voltage | 8-midpoint line integral across the gap |
| Input impedance | `Zin = V` (current is exactly 1 A by construction) |
| Conductor threshold | `σ ≥ 1e4 S/m` treated as PEC |

## Layout

```text
wasm/hcurl/
├── Cargo.toml            # package hcurl-fem, lib hcurl_fem (cdylib + rlib)
├── README.md             # this file
├── src/
│   ├── lib.rs            # crate docs, solve_point entry point
│   ├── abi.rs            # packed binary request/response (wire format v1)
│   │                     # + FarFieldBlob v1 (versioned far-field payload)
│   ├── ffi.rs            # extern "C" exports (raw WASM boundary)
│   ├── error.rs          # Error enum + ABI status codes
│   ├── budget.rs         # cooperative cancellation + timeout
│   ├── clock.rs          # native/WASM-safe timing (Instant panics on WASM)
│   ├── constants.rs      # physical constants (SI, e^{+jωt})
│   ├── mesh.rs           # validated mesh + sanitized materials
│   ├── edge.rs           # global edge enumeration, PEC elimination
│   ├── geometry.rs       # tet Jacobian, barycentric gradients
│   ├── basis.rs          # Nédélec basis, analytic element brackets
│   ├── assembly.rs       # complex volume assembly
│   ├── abc.rs            # Silver–Müller facet assembly
│   ├── csr.rs            # complex CSR sparse matrix
│   ├── krylov.rs         # complex BiCGSTAB + symmetric Gauss–Seidel
│   ├── port.rs           # gap source, V/I, Zin, S11
│   ├── farfield.rs       # Stratton–Chu near-to-far-field transform
│   └── solve.rs          # end-to-end point orchestration
├── js/
│   └── hcurl_solver.js   # loader shim: createHcurlSolver factory
├── tools/
│   ├── pack_dipole.ts    # builds the dipole parity fixture (see below)
│   └── wasm_smoke.mjs    # real-WASM FFI smoke test via the loader shim
└── tests/
    ├── dipole_parity.rs  # TS-vs-Rust parity integration test
    └── fixtures/
        └── dipole_3ghz.req  # packed dipole request (420,157 bytes)
```

### Module responsibilities

- **`mesh`** — validates the packed mesh (vertex/tet counts, index ranges),
  sanitizes the material table (NaN/inf rejection, conductor threshold).
- **`edge`** — canonical global edge enumeration from tets; eliminates PEC
  edges (n×E = 0) from the free-DOF set.
- **`geometry` / `basis`** — per-tet Jacobian, barycentric gradients, and the
  analytic 6×6 Nédélec curl-curl / mass element brackets (with a corrected
  mass-bracket expansion that a cross-product regression test pins down).
- **`assembly` / `abc`** — complex volume assembly plus the first-order
  Silver–Müller term `+j k₀ B` on radiation facets; both poll the
  cancellation budget.
- **`csr`** — complex compressed-sparse-row matrix (`from_triplets`,
  `matvec`, Hermitian products).
- **`krylov`** — complex BiCGSTAB with full-complex symmetric Gauss–Seidel
  preconditioner `M = (D+L)D⁻¹(D+U)`; defaults: tolerance `1e-7`, max
  `12000` iterations.
- **`port`** — adaptive gap-tet selection from the port facets, exact 1 A
  impressed-current RHS, 8-midpoint terminal voltage, `Zin`, S11 against the
  request's reference impedance.
- **`farfield`** — Stratton–Chu reconstruction of equivalent currents on the
  absorbing boundary → IEEE gain/directivity grids on the requested θ/φ
  lattice, radiated/accepted power, efficiency.
- **`solve`** — end-to-end orchestration of one frequency point with phase
  timings.

## Building

Prerequisites: a recent stable Rust toolchain (`rustup`).

```sh
# Native library + full test suite (fast iteration; numerics validated here)
cargo test

# WebAssembly module
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
# → target/wasm32-unknown-unknown/release/hcurl_fem.wasm  (~279 KB)
```

`js/hcurl_solver.js` + `hcurl_fem.wasm` (renamed `hcurl_solver.wasm`) are the
two deployable artifacts; copy both to `frontend/public/wasm/` (see
*Browser integration*).

## Parallelism

On native targets the hot loops run on Rayon's thread pool through the
small `par` module (`src/par.rs`): element assembly (volume + ABC), the
CSR matvec, the BiCGSTAB vector updates and reductions, and the far-field
θ grid. The thread count follows Rayon's default (one per logical core);
override it with `RAYON_NUM_THREADS`.

Two guarantees are load-bearing:

- **Bit-identical kernels.** Assembly, matvec, vector updates, and the
  far-field grid preserve the serial visit and operation order, so they
  reproduce the serial results exactly — the dipole parity test pins the
  assembled-matrix anchors to 1e-8 and still passes.
- **Pool-ordered reductions.** The `norm2` / `dot_conj` sums reduce in
  thread-pool order: a few ulps of drift, so native iteration counts can
  wobble by a handful between runs. The parity tolerances (Zin 2%,
  S11 0.01 absolute) absorb it comfortably.

The symmetric Gauss–Seidel preconditioner stays serial — triangular sweeps
are inherently sequential — and it is roughly two-thirds of the
per-iteration work, which caps solve-phase speedup (Amdahl). Measured on
the dipole fixture (19,429 free DOFs, release build): 7.36 s pinned to one
thread vs 6.27 s on 2 cores (**1.17×**), identical Zin to 6 decimals.
Assembly and the far-field grid scale near-linearly; the solve phase sees
diminishing returns past a few cores until SGS is replaced or parallelized.

```sh
cargo run --release --example dipole_bench
RAYON_NUM_THREADS=1 cargo run --release --example dipole_bench
```

On `wasm32` the `par` helpers compile to plain sequential iterators — the
WASM module is built without the atomics/`SharedArrayBuffer` threading
setup, so the browser module is unchanged and stays single-threaded.
Enabling true WASM threads later needs no numerical-code changes (the
kernels already go through `par`): build with
`-C target-feature=+atomics`, initialize a `wasm-bindgen-rayon` thread
pool from the JS loader, and serve COOP/COEP headers.

## API

### Rust

```rust
use hcurl_fem::abi::PointRequest;

let request = PointRequest::parse(&request_bytes)?; // validates wire format v1
let response = hcurl_fem::solve_point(&request)?;   // one frequency point
let wire_bytes: Vec<u8> = response.serialize();     // packed v1 response
```

Key types:

| Item | Description |
|------|-------------|
| `abi::PointRequest::parse` | Validates/decodes a packed v1 request: mesh, materials, port facets, frequency, metadata |
| `hcurl_fem::solve_point` | End-to-end solve of one frequency point → `abi::PointResponse` |
| `abi::PointResponse::serialize` | 104-byte header + far-field blob + message |
| `abi::FarFieldBlob::from_result` / `parse` / `serialize` | Versioned far-field payload (v1) |
| `csr::CsrMatrix` | Complex CSR: `from_triplets`, `matvec`, `diagonal` |
| `mesh::Mesh::from_request` | Validated domain mesh + sanitized materials |
| `edge::build_edge_system` | Global edge enumeration, PEC elimination |
| `basis::local_matrices` | Analytic 6×6 curl-curl / mass element brackets |
| `krylov::bicgstab` | Complex BiCGSTAB + SGS, `KrylovOptions { tolerance, max_iterations }` |
| `port::PortResult` | Gap source data, `Zin`, S11 |
| `error::Error` | Typed failures; `status_code()` maps to the response header |
| `budget::Budget` | Cooperative cancellation + wall-clock timeout for long loops |

Request metadata (JSON, optional) understood by the solver:

| Key | Effect |
|-----|--------|
| `krylov_tolerance` | Overrides the default `1e-7` BiCGSTAB tolerance |
| `far_field.theta_step_deg` | θ lattice step (default 5°); omit `far_field` to skip the transform |
| `far_field.phi_step_deg` | φ lattice step (default 5°) |

### Raw WASM FFI

The module exposes these `extern "C"` functions (see `src/ffi.rs`). They are
compiled **unconditionally** — there is no feature-gated WASM API; the same
symbols are present in native test builds of the cdylib.

| Export | Signature | Description |
|--------|-----------|-------------|
| `hcurl_solver_alloc` | `(size: usize) -> *mut u8` | Allocate `size` bytes in WASM linear memory |
| `hcurl_solver_free` | `(ptr: *mut u8)` | Free a pointer from `alloc` or `run` |
| `hcurl_solver_run` | `(req_ptr, req_len, rsp_ptr_out, rsp_len_out) -> i32` | Solve one point; writes response ptr/len to the two out-slots; returns 0 on success |
| `hcurl_solver_set_cancelled` | `(flag: i32)` | Cooperative cancel flag (polled by assembly, Krylov, far-field) |
| `hcurl_solver_last_error_len` | `() -> usize` | Length of the last captured Rust panic message |
| `hcurl_solver_last_error_copy` | `(dst: *mut u8, cap: usize) -> usize` | Copy that message out |

A Rust panic inside `run` is caught by a panic hook that records the message
for `hcurl_solver_last_error_*` instead of trapping with `unreachable`.

### JavaScript loader

`js/hcurl_solver.js` adapts the raw exports to the Emscripten-style interface
the frontend worker expects:

```js
const module = await createHcurlSolver({ locateFile });
const reqPtr = module._hcurl_solver_alloc(requestBytes.length);
module.HEAPU8.set(requestBytes, reqPtr);
const outPtr = module._hcurl_solver_alloc(8);
const status = module._hcurl_solver_run(reqPtr, requestBytes.length, outPtr, outPtr + 4);
// read response ptr/len from HEAPU8, copy out, then:
module._hcurl_solver_free(rspPtr);
module._hcurl_solver_free(reqPtr);
module._hcurl_solver_free(outPtr);
// cancellation from the UI thread:
module._hcurl_solver_set_cancelled(1);
// last Rust panic message ("" when none):
module._hcurl_solver_last_error();
```

Note the loader returns the module object directly from `createHcurlSolver`
(it is not a factory-of-factories).

### Packed wire format (ABI v1)

The outer request/response layout is byte-for-byte identical to the original
prototype format, so existing packed requests stay valid. Field-by-field
documentation lives in `src/abi.rs`; the TypeScript mirror is
`frontend/src/engine/wasm/hcurl-abi.ts`.

- Request: 128-byte header (endianness sentinel `0x12345678`, counts,
  section offsets, `f64` frequency) + vertex/tet/region/boundary/port
  sections + material table + metadata JSON.
- Response: 104-byte header (`u32` ABI version, `u32` status code, `f64`
  residual, `u32` iterations, `f64` Zin re/im, `f64` S11 re/im, three `f64`
  phase timings, field-blob offset/len, message offset/len) + field blob +
  message.

#### Far-field blob v1 (`field_blob`)

Little-endian:

```text
u32 version (=1), u32 n_theta, u32 n_phi
7× f64: max_gain_dbi, max_gain_theta_deg, max_gain_phi_deg,
        max_directivity_dbi, radiated_power_w, accepted_power_w,
        radiation_efficiency
n_theta × f64 theta_deg, n_phi × f64 phi_deg
(n_theta·n_phi) × f64 gain_dbi        (row-major [theta][phi])
(n_theta·n_phi) × f64 directivity_dbi (row-major [theta][phi])
```

The TypeScript decoder is `unpackHcurlFarFieldBlob` in
`frontend/src/engine/wasm/hcurl-abi.ts`; it rejects version mismatches,
truncation, out-of-range grids, and non-finite values.

## Cancellation and timeouts

Long loops (assembly, Krylov iterations, far-field θ rows) poll a
`budget::Budget` that carries a cooperative cancel flag and an optional
deadline. `std::time::Instant` panics on bare `wasm32-unknown-unknown`, so
`src/clock.rs` abstracts timing: native builds record real phase timings and
enforce the deadline; WASM builds report `0.0` phase timings and skip the
internal deadline — the frontend worker's 15-minute watchdog remains the WASM
timeout mechanism. Cancellation always works on both targets via
`hcurl_solver_set_cancelled`.

## Browser integration

1. Build the WASM module (above) and copy both artifacts:
   ```sh
   cp target/wasm32-unknown-unknown/release/hcurl_fem.wasm \
      frontend/public/wasm/hcurl_solver.wasm
   cp js/hcurl_solver.js frontend/public/wasm/hcurl_solver.js
   ```
2. `frontend/src/engine/wasm/worker-fem.ts` loads `wasm/hcurl_solver.js`
   (`createHcurlSolver`), packs each frequency point with `packHcurlPointRequest`,
   decodes the response with `unpackHcurlPointResponse`, and decodes the
   far-field blob with `unpackHcurlFarFieldBlob`. Gain/far-field capabilities
   are advertised only when a blob decodes successfully; otherwise the result
   carries `gain_data.available: false` and a warning, and the worker falls
   back to the TypeScript reference solver if the module fails to load.

## Testing

```sh
cargo test            # 44 unit + 2 integration + 1 doc-test
cargo test --release  # with optimizations, for timing-sensitive checks
node tools/wasm_smoke.mjs   # real WASM FFI end-to-end via the loader shim
```

### Dipole parity validation (3 GHz, 0.42λ center-fed dipole)

The fixture (`tests/fixtures/dipole_3ghz.req`, built by `tools/pack_dipole.ts`
from the real frontend mesher + TypeScript solver) anchors the port:

| | TypeScript reference | Rust native | Rust WASM (FFI) |
|---|---|---|---|
| Zin | 66.6374961714 − j15.3179359547 Ω | 66.6374961971 − j15.3179360539 Ω | 66.6374961971 − j15.3179360539 Ω |
| S11 | 0.1571792582 − j0.1106871681 | 0.1571792585 − j0.1106871687 | 0.1571792585 − j0.1106871687 |
| iterations | 1,117 | 1,230 | 1,230 |
| residual | 8.15e-8 | 9.41e-8 | 9.41e-8 |
| max gain | 2.7658427 dBi | 2.7658 dBi | 2.7658 dBi |
| P_rad / P_acc | 36.7745808 / 33.3187481 W | 36.7746 / 33.3187 W | 36.7746 / 33.3187 W |

The dipole's broadside maxima are symmetric: TypeScript reports θ=135°/φ=90°,
Rust reports the equivalent opposite lobe θ=45°/φ=270°; the parity test
accepts both. Native and WASM FFI runs are bit-identical.

## Current limitations

- First-order Nédélec elements and a first-order Silver–Müller ABC: accuracy
  is comparable to the TypeScript reference, not to higher-order codes.
- Radiation efficiency can exceed 1 on electrically small ABC boxes (the
  3 GHz fixture uses a tight 0.2λ box) — the TypeScript reference behaves the
  same way, so parity tests compare against the reference rather than
  asserting efficiency ≤ 1.
- Single lumped port, single frequency point per `run` call; sweeps are
  orchestrated by the frontend worker.
- Phase timings are `0.0` on WASM (no `Instant`); wall-clock timing is the
  worker's job.
- The packed ABI is v1 and fixed; the far-field payload is independently
  versioned (blob v1) so it can evolve without breaking the outer layout.

## License

GPL-3.0-or-later, following the repository license.
