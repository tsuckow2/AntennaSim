# WebGPU FDTD bench

A browser page that runs this repo's FDTD solver (`wasm/fdtd/src/gpu.rs`)
on any machine with a WebGPU-capable browser, plus a relay that lets the
page be driven remotely as an agent. Deployed at
https://webgpu.muse-workspace.g.codingwell.net (Forge, `/var/www/webgpu`).

## What's here

- `index.html` — the assembled, deployable page (v5.7). Do not edit by
  hand; rebuild it (below).
- `build/template.html` — the page shell (UI, agent long-poll, manual
  parity/bench jobs).
- `build/gen_shaders.py` — extracts the four WGSL modules from
  `../../wasm/fdtd/src/gpu.rs`, renumbers @group (fields 0, materials 1,
  CPML 2, misc 3), computes per-entry-point binding usage, and rewrites
  the PEC mask kernels as grid-stride loops (their flat dispatch exceeds
  WebGPU's 65535 workgroups-per-dimension cap on large grids).
- `build/solver_core.js` — the `solver` job: a JS port of the Snyder
  geometry/CPML builders (f32-faithful, checksums bit-match the Rust
  references in `wasm/fdtd/examples/snyder_ref*.rs`), per-entry-point
  minimal bind-group layouts (≤8 storage buffers — the WebGPU floor),
  native step order, and a Goertzel DFT producing Zin/S11 tables.
  Config: `{post, thick, zref|zref2|zref3}` selects the feed/ground
  variant and the z-refinement ladder rung.
- `build/build_v5.py` — assembles `index.html` from the template +
  shaders + core.
- `relay/` — `relay.py` (job relay service on Forge, 127.0.0.1:8177,
  proxied at `/agent/` + token-guarded `/api/`), `webgpu-ctl` (on-Forge
  CLI), `webgpu-relay.service` (systemd unit).
- `ctl` — local wrapper that runs `webgpu-ctl` on Forge over SSH:
  `ctl sessions`, `ctl submit <session> <job.json>`, `ctl job <id>`,
  `ctl parity`, `ctl bench`, `ctl reload`.
- `webgpu.nginx.conf`, `version.txt` — vhost and the self-update
  version stamp (open tabs poll it and reload themselves when it
  changes; a `reload` job forces it).

## Rebuild & deploy

```sh
cd build && python3 gen_shaders.py && python3 build_v5.py
# deploy: copy ../index.html to Forge /var/www/webgpu/index.html and
# bump version.txt; open tabs self-update within ~60 s.
```

## Results provenance

Full-solver rates: ~1.5–2.1 Gcell/s on an NVIDIA Turing laptop GPU
(cw-r10) — a 10 ns Snyder run takes ~4 min at the base grid, ~28 min
at dz/4, ~98 min at dz/8. CPU = native GPU = WebGPU series agree to
~1e-6 (6000-step aggregates). The Snyder dz ladder measured with this
bench: 2.015 → 2.19 → 2.31 → 2.35 GHz (FEM anchor ~2.44 GHz).
