# fdtd — Yee-grid FDTD solver

A modular, native-speed finite-difference time-domain solver for broadband
antenna impedance, written in Rust. The first comparison fixture is Snyder's
2.44 GHz probe-fed patch; the crate itself is geometry-agnostic.

## Architecture

```
src/
  grid.rs      Yee grid: field arrays, Courant dt, index helpers
  material.rs  Per-component Ca/Cb update coefficients, PEC masks, lossy dielectrics
  cpml.rs      Six-face convolutional PML (correction pass after the Yee update)
  source.rs    Differentiated-Gaussian hard source, DFT probe (V/I → Zin/S11)
  geometry.rs  Snyder patch fixture (grid constants + build_snyder)
  solver.rs    Fdtd: fused H/E updates, PEC, source injection, time loop
```

- **Fields** are `f32`; the DFT accumulates in `f64`.
- **Parallelism** via Rayon: the H update (Hx/Hy/Hz) and E update (Ex/Ey/Ez)
  are each fused into a single z-slab-parallel pass. CPML and PEC are also
  parallel. On wasm32, Rayon compiles to sequential iterators (no threads).
- **CPML** is applied as a correction Δ = (CPML − standard) after the vanilla
  Yee update, so the hot loops stay simple. Six faces, face-local ψ storage,
  CFS grading (κ_max=7, σ_max from Taflove).
- **Source**: differentiated-Gaussian hard voltage source at the feed gap,
  `V(t) = −(t−t₀)/τ · exp(−(t−t₀)²/τ²)`, τ chosen for ~2–3 GHz bandwidth.
- **Impedance**: Ampère-loop feed current (∮H·dl around the probe at the gap
  plane) sampled at H-steps, averaged to E-step time; V and I are DFT'd
  on the fly, then Zin(f) = V(f)/I(f), S11 w.r.t. 50 Ω.

## Build & test

```sh
cargo build --release
cargo test --release          # 2 unit tests (grid layout, source spectrum)
cargo run --release --example snyder_fdtd -- [t_end_ns]   # default 17 ns
cargo run --release --example prof                        # per-kernel timing
```

wasm32 (no threads; sequential fallback):
```sh
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown
```

## Validation status (2026-09-26)

- [x] No-panic smoke test (3 ns, 232×232×61 grid)
- [x] CPML sign audit (6 corrections had flipped signs; fixed)
- [ ] Snyder 17 ns broadband run → compare Zin/S11 vs FEM sweep
- [ ] Grid-refinement check (0.5 mm vs 0.35 mm in-plane)
- [ ] CPML reflection quantified (vs larger domain)

## Assumptions & caveats

- The PML region must be vacuum (no PEC/dielectric intersecting it).
- Hard source: the gap E is overwritten, not added; keep the gap ≥1 cell.
- Lossy dielectric uses a first-order semi-implicit conductivity term.
- The Ampère loop assumes the probe is z-directed at the gap plane.
- FDTD is ~200× more unknowns than FEM per wavelength; the 17 ns Snyder
  run takes ~1 h on 2 cores. This is a validation tool, not a replacement.
