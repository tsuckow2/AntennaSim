
'use strict';
// ---------------------------------------------------------------------------
// FDTD Yee kernels — WGSL copied verbatim from the native gpu-spike test
// (wgpu 24, validated against the CPU solver: max abs diff 2.98e-8).
// ---------------------------------------------------------------------------
const SHADER_H = /* wgsl */`
struct Params {
    nx: u32, ny: u32, nz: u32, _pad: u32,
    ch: f32, ca: f32, cb: f32, _pad2: f32,
    idx: f32, idy: f32, idz: f32, _pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> ex: array<f32>;
@group(0) @binding(2) var<storage, read> ey: array<f32>;
@group(0) @binding(3) var<storage, read> ez: array<f32>;
@group(0) @binding(4) var<storage, read_write> hx: array<f32>;
@group(0) @binding(5) var<storage, read_write> hy: array<f32>;
@group(0) @binding(6) var<storage, read_write> hz: array<f32>;

fn id3(i: u32, j: u32, k: u32) -> u32 { return (i * p.ny + j) * p.nz + k; }

@compute @workgroup_size(256)
fn update_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    let total = p.nx * p.ny * p.nz;
    if (n >= total) { return; }
    let k = n % p.nz;
    let j = (n / p.nz) % p.ny;
    let i = n / (p.ny * p.nz);
    if (i + 1u >= p.nx || j + 1u >= p.ny || k + 1u >= p.nz) { return; }
    let c = id3(i, j, k);
    let jp = c + p.nz;
    let kp = c + 1u;
    hx[c] -= p.ch * ((ez[jp] - ez[c]) * p.idy - (ey[kp] - ey[c]) * p.idz);
    hy[c] -= p.ch * ((ex[kp] - ex[c]) * p.idz - (ez[c + p.ny * p.nz] - ez[c]) * p.idx);
    hz[c] -= p.ch * ((ey[c + p.ny * p.nz] - ey[c]) * p.idx - (ex[jp] - ex[c]) * p.idy);
}
`;
const SHADER_E = /* wgsl */`
struct Params {
    nx: u32, ny: u32, nz: u32, _pad: u32,
    ch: f32, ca: f32, cb: f32, _pad2: f32,
    idx: f32, idy: f32, idz: f32, _pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> ex: array<f32>;
@group(0) @binding(2) var<storage, read_write> ey: array<f32>;
@group(0) @binding(3) var<storage, read_write> ez: array<f32>;
@group(0) @binding(4) var<storage, read> hx: array<f32>;
@group(0) @binding(5) var<storage, read> hy: array<f32>;
@group(0) @binding(6) var<storage, read> hz: array<f32>;

fn id3(i: u32, j: u32, k: u32) -> u32 { return (i * p.ny + j) * p.nz + k; }

@compute @workgroup_size(256)
fn update_e(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    let total = p.nx * p.ny * p.nz;
    if (n >= total) { return; }
    let k = n % p.nz;
    let j = (n / p.nz) % p.ny;
    let i = n / (p.ny * p.nz);
    if (j == 0u || k == 0u || i == 0u) { return; }
    let c = id3(i, j, k);
    let jm = c - p.nz;
    let km = c - 1u;
    let im = c - p.ny * p.nz;
    ex[c] = p.ca * ex[c] + p.cb * ((hz[c] - hz[jm]) * p.idy - (hy[c] - hy[km]) * p.idz);
    ey[c] = p.ca * ey[c] + p.cb * ((hx[c] - hx[km]) * p.idz - (hz[c] - hz[im]) * p.idx);
    ez[c] = p.ca * ez[c] + p.cb * ((hy[c] - hy[im]) * p.idx - (hx[c] - hx[jm]) * p.idy);
}
`;

// Physical constants / grid spacings — identical to gpu-spike make_params().
function makeParams(nx, ny, nz) {
  const dx = 5.0e-4, dy = 5.0e-4, dz = 5.25e-4;
  const c0 = 299792458.0;
  const dt = 0.5 / (c0 * Math.sqrt(1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz)));
  const mu0 = 1.25663706212e-6, eps0 = 8.854187817e-12;
  return { nx, ny, nz, ch: dt / mu0, ca: 1.0, cb: dt / eps0,
           idx: 1 / dx, idy: 1 / dy, idz: 1 / dz };
}

function paramsBuffer(p) {
  const buf = new ArrayBuffer(48);
  const v = new DataView(buf);
  v.setUint32(0, p.nx, true); v.setUint32(4, p.ny, true); v.setUint32(8, p.nz, true);
  v.setFloat32(16, p.ch, true); v.setFloat32(20, p.ca, true); v.setFloat32(24, p.cb, true);
  v.setFloat32(32, p.idx, true); v.setFloat32(36, p.idy, true); v.setFloat32(40, p.idz, true);
  return buf;
}

// Gaussian bump in Ez at the grid centre — identical to gpu-spike bump_init().
function bumpInit(nx, ny, nz) {
  const n = nx * ny * nz;
  const fields = [];
  for (let f = 0; f < 6; f++) fields.push(new Float32Array(n));
  const ez = fields[2];
  const ci = Math.floor(nx / 2), cj = Math.floor(ny / 2), ck = Math.floor(nz / 2);
  for (let di = -2; di <= 2; di++) for (let dj = -2; dj <= 2; dj++) for (let dk = -2; dk <= 2; dk++) {
    const r2 = di * di + dj * dj + dk * dk;
    ez[((ci + di) * ny + (cj + dj)) * nz + (ck + dk)] = Math.exp(-r2 / 3.0);
  }
  return fields;
}

// CPU mirror of the two kernels (JS numbers are f64; the native mirror is f32,
// so expect browser parity diffs in the 1e-7..1e-6 range rather than 3e-8).
function cpuStep(f, nx, ny, nz, p) {
  const [ex, ey, ez, hx, hy, hz] = f;
  const id = (i, j, k) => (i * ny + j) * nz + k;
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    if (i + 1 >= nx || j + 1 >= ny || k + 1 >= nz) continue;
    const c = id(i, j, k), jp = c + nz, kp = c + 1, ip = c + ny * nz;
    hx[c] -= p.ch * ((ez[jp] - ez[c]) * p.idy - (ey[kp] - ey[c]) * p.idz);
    hy[c] -= p.ch * ((ex[kp] - ex[c]) * p.idz - (ez[ip] - ez[c]) * p.idx);
    hz[c] -= p.ch * ((ey[ip] - ey[c]) * p.idx - (ex[jp] - ex[c]) * p.idy);
  }
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    if (j === 0 || k === 0 || i === 0) continue;
    const c = id(i, j, k), jm = c - nz, km = c - 1, im = c - ny * nz;
    ex[c] = p.ca * ex[c] + p.cb * ((hz[c] - hz[jm]) * p.idy - (hy[c] - hy[km]) * p.idz);
    ey[c] = p.ca * ey[c] + p.cb * ((hx[c] - hx[km]) * p.idz - (hz[c] - hz[im]) * p.idx);
    ez[c] = p.ca * ez[c] + p.cb * ((hy[c] - hy[im]) * p.idx - (hx[c] - hx[jm]) * p.idy);
  }
}

// ---------------------------------------------------------------------------
// Page state
// ---------------------------------------------------------------------------
const $ = id => document.getElementById(id);
const PAGE_VERSION = 'v5.7';
const results = { page: 'webgpu-fdtd-test-bench', version: 2, pageVersion: PAGE_VERSION, runs: [] };
let adapter = null, device = null, adapterSnapshot = null;
let stopRequested = false;
let jobRunning = false;
let eventSink = null; // when set (remote job), log lines stream to the relay

function log(msg) {
  const el = $('log');
  const t = new Date().toLocaleTimeString();
  el.textContent += `[${t}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
  if (eventSink) { try { eventSink(msg); } catch (e) { /* relay hiccup: keep going */ } }
}
function fmt(n) { return n.toLocaleString('en-US'); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function setAdapterStatus(kind, text) {
  const el = $('adapterStatus');
  el.className = 'pill ' + kind;
  el.textContent = text;
}

async function refreshAdapter() {
  setAdapterStatus('idle', 'checking…');
  $('checkBtn').disabled = true;
  $('benchBtn').disabled = true;
  if (!navigator.gpu) {
    setAdapterStatus('bad', 'WebGPU not available');
    $('adapterInfo').innerHTML =
      '<p>This browser did not expose WebGPU (<span class="mono">navigator.gpu</span> is missing). ' +
      'Use a current Chrome or Edge on desktop/Android, or Safari 26+. The page must be served ' +
      'over HTTPS (it is) and WebGPU must not be disabled in browser flags.</p>';
    log('navigator.gpu missing — WebGPU unavailable in this browser/context.');
    return;
  }
  const pref = $('powerPref').value;
  const opts = {};
  if (pref) opts.powerPreference = pref;
  try {
    adapter = await navigator.gpu.requestAdapter(opts);
  } catch (e) {
    adapter = null;
    log('requestAdapter threw: ' + e.message);
  }
  if (!adapter) {
    setAdapterStatus('bad', 'no adapter');
    $('adapterInfo').innerHTML = '<p><span class="mono">requestAdapter()</span> returned null — ' +
      'no WebGPU-capable GPU was offered. Try the other power preference, or a different browser.</p>';
    log('requestAdapter returned null (powerPreference=' + (pref || 'default') + ').');
    return;
  }
  const info = adapter.info || {};
  const limits = adapter.limits;
  adapterSnapshot = {
    vendor: info.vendor || '', architecture: info.architecture || '',
    device: info.device || '', description: info.description || '',
    isFallbackAdapter: !!info.isFallbackAdapter,
    powerPreference: pref || 'default',
    features: [...adapter.features],
    limits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
      maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    },
    userAgent: navigator.userAgent,
    when: new Date().toISOString(),
  };
  device = await adapter.requestDevice();
  device.lost.then(d => {
    log('DEVICE LOST: ' + d.message + ' (reason: ' + d.reason + '). Refresh the adapter to recover.');
    setAdapterStatus('bad', 'device lost');
    $('checkBtn').disabled = true;
    $('benchBtn').disabled = true;
  });
  device.addEventListener('uncapturederror', ev => log('GPU error: ' + ev.error.message));

  const label = info.description || info.device || info.architecture || 'GPU';
  setAdapterStatus(info.isFallbackAdapter ? 'warn' : 'ok',
                   info.isFallbackAdapter ? 'software fallback adapter' : 'ready');
  $('adapterInfo').innerHTML =
    '<table>' +
    '<tr><td class="k">Adapter</td><td class="mono">' + escapeHtml(label) + '</td></tr>' +
    '<tr><td class="k">Vendor / architecture</td><td class="mono">' +
      escapeHtml(info.vendor || '?') + ' / ' + escapeHtml(info.architecture || '?') + '</td></tr>' +
    '<tr><td class="k">Fallback (software)</td><td>' + (info.isFallbackAdapter ? 'yes — results will be slow' : 'no') + '</td></tr>' +
    '<tr><td class="k">Power preference</td><td>' + (pref || 'default') + '</td></tr>' +
    '<tr><td class="k">Max buffer size</td><td class="mono">' + fmt(limits.maxBufferSize) + ' bytes</td></tr>' +
    '<tr><td class="k">Max storage binding</td><td class="mono">' + fmt(limits.maxStorageBufferBindingSize) + ' bytes</td></tr>' +
    '<tr><td class="k">Max workgroups / dim</td><td class="mono">' + fmt(limits.maxComputeWorkgroupsPerDimension) + '</td></tr>' +
    '</table>' +
    '<details><summary>Features (' + adapterSnapshot.features.length + ')</summary>' +
    '<div class="mono">' + adapterSnapshot.features.join(', ') + '</div></details>';
  log('Adapter ready: ' + label + ' (vendor=' + (info.vendor || '?') +
      ', fallback=' + !!info.isFallbackAdapter + ').');
  $('checkBtn').disabled = false;
  $('benchBtn').disabled = false;
}

// ---------------------------------------------------------------------------
// GPU runner: builds buffers/pipelines for one FDTD grid, runs steps chunked.
// ---------------------------------------------------------------------------
async function runGpu(nx, ny, nz) {
  const params = makeParams(nx, ny, nz);
  const cells = nx * ny * nz;
  const bytes = cells * 4;
  if (bytes > adapter.limits.maxStorageBufferBindingSize)
    throw new Error('field buffer ' + fmt(bytes) + ' B exceeds maxStorageBufferBindingSize ' +
                    fmt(adapter.limits.maxStorageBufferBindingSize) + ' B');
  const init = bumpInit(nx, ny, nz);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const bufs = init.map((data, i) => {
    const b = device.createBuffer({ label: 'field' + i, size: bytes, usage });
    device.queue.writeBuffer(b, 0, data);
    return b;
  });
  const paramsBuf = device.createBuffer({
    label: 'params', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuf, 0, paramsBuffer(params));
  const staging = device.createBuffer({
    label: 'staging', size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const modH = device.createShaderModule({ code: SHADER_H });
  const modE = device.createShaderModule({ code: SHADER_E });
  const pipeH = device.createComputePipeline({
    layout: 'auto', compute: { module: modH, entryPoint: 'update_h' } });
  const pipeE = device.createComputePipeline({
    layout: 'auto', compute: { module: modE, entryPoint: 'update_e' } });
  const mkBg = pipe => device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: paramsBuf } },
              ...bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))],
  });
  const bgH = mkBg(pipeH), bgE = mkBg(pipeE);
  const wg = Math.ceil(cells / 256);

  async function runSteps(total, chunk, onProgress) {
    let done = 0;
    while (done < total) {
      if (stopRequested) break;
      const n = Math.min(chunk, total - done);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let s = 0; s < n; s++) {
        pass.setPipeline(pipeH); pass.setBindGroup(0, bgH); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeE); pass.setBindGroup(0, bgE); pass.dispatchWorkgroups(wg);
      }
      pass.end();
      device.queue.submit([enc.finish()]);
      done += n;
      if (onProgress) onProgress(done, total);
      await device.queue.onSubmittedWorkDone();
    }
    return done;
  }

  async function readField(which) {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bufs[which], 0, staging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return out;
  }

  function destroy() {
    bufs.forEach(b => b.destroy());
    paramsBuf.destroy(); staging.destroy();
  }
  return { runSteps, readField, destroy, cells };
}

// ---------------------------------------------------------------------------
// Job cores — shared by the manual buttons and the remote agent.
// ---------------------------------------------------------------------------
async function coreParity(n, steps) {
  log(`Parity check: grid ${n}^3, ${steps} steps…`);
  const gpu = await runGpu(n, n, n);
  await gpu.runSteps(steps, steps);
  const params = makeParams(n, n, n);
  const cpu = bumpInit(n, n, n);
  for (let s = 0; s < steps; s++) cpuStep(cpu, n, n, n, params);
  let worst = 0;
  for (let f = 0; f < 6; f++) {
    const got = await gpu.readField(f);
    for (let i = 0; i < got.length; i++) {
      const d = Math.abs(got[i] - cpu[f][i]);
      if (d > worst) worst = d;
    }
  }
  gpu.destroy();
  const pass = worst < 1e-5;
  log(`Parity check done: max abs diff ${worst.toExponential(3)} over 6 fields × ${steps} steps → ` +
      (pass ? 'PASS' : 'FAIL'));
  const rec = { job: 'parity-check', grid: [n, n, n], steps, maxAbsDiff: worst, pass,
                adapter: adapterSnapshot, when: new Date().toISOString() };
  results.runs.push(rec);
  return rec;
}

async function coreBench(nx, ny, nz, steps, repeats, onProgress) {
  const rates = [];
  const runs = [];
  for (let r = 0; r < repeats; r++) {
    log(`Benchmark run ${r + 1}/${repeats}: grid ${nx}×${ny}×${nz}, ${steps} steps (+10 warmup)…`);
    const gpu = await runGpu(nx, ny, nz);
    await gpu.runSteps(10, 10); // warmup, untimed
    const t0 = performance.now();
    const doneSteps = await gpu.runSteps(steps, 25, onProgress);
    await device.queue.onSubmittedWorkDone();
    const el = (performance.now() - t0) / 1000;
    const ez = await gpu.readField(2);
    let checksum = 0;
    for (let i = 0; i < ez.length; i++) checksum += Math.abs(ez[i]);
    gpu.destroy();
    if (doneSteps < steps) { log('Benchmark stopped after ' + doneSteps + ' steps.'); break; }
    const rate = gpu.cells * doneSteps / el / 1e6;
    rates.push(rate);
    log(`Run ${r + 1}: ${el.toFixed(3)} s → ${rate.toFixed(1)} Mcell/s (Ez |sum| checksum ${checksum.toExponential(6)})`);
    const rec = { job: 'benchmark', grid: [nx, ny, nz], steps: doneSteps, seconds: el,
                  mcellPerSec: rate, ezAbsSum: checksum,
                  timing: 'wall clock around queue completion, 10-step warmup excluded',
                  adapter: adapterSnapshot, when: new Date().toISOString() };
    results.runs.push(rec);
    runs.push(rec);
  }
  let median = null;
  if (rates.length) {
    const sorted = [...rates].sort((a, b) => a - b);
    median = sorted[Math.floor(sorted.length / 2)];
  }
  return { job: 'benchmark', grid: [nx, ny, nz], steps, repeats, rates, medianMcellPerSec: median, runs };
}

// Custom WGSL job contract (remote jobs pushed by Muse):
//   entry point `main`, @workgroup_size(256), guard `if (n >= total) { return; }`
//   @binding(0) uniform { nx: u32, ny: u32, nz: u32, step: u32 }
//   @binding(1..6) var<storage, read_write> fields ex, ey, ez, hx, hy, hz
// Fields start from the standard bump init. `step` is incremented per dispatch.
async function coreCustom(job) {
  const [nx, ny, nz] = job.grid;
  const steps = job.steps || 10;
  const cells = nx * ny * nz;
  const bytes = cells * 4;
  if (bytes > adapter.limits.maxStorageBufferBindingSize)
    throw new Error('field buffer ' + fmt(bytes) + ' B exceeds maxStorageBufferBindingSize');
  const mod = device.createShaderModule({ code: job.wgsl });
  const info = await mod.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === 'error');
  info.messages.filter(m => m.type === 'warning')
    .forEach(m => log('WGSL warning (line ' + m.lineNum + '): ' + m.message));
  if (errs.length)
    throw new Error('WGSL compile error: ' +
      errs.map(m => 'line ' + m.lineNum + ': ' + m.message).join('; '));
  const init = bumpInit(nx, ny, nz);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const bufs = init.map((data, i) => {
    const b = device.createBuffer({ label: 'cfield' + i, size: bytes, usage });
    device.queue.writeBuffer(b, 0, data);
    return b;
  });
  const uni = device.createBuffer({
    label: 'cuniform', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uniBase = new Uint32Array([nx, ny, nz, 0]);
  device.queue.writeBuffer(uni, 0, uniBase);
  const staging = device.createBuffer({
    label: 'cstaging', size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const pipe = device.createComputePipeline({
    layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uni } },
              ...bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))],
  });
  const wg = Math.ceil(cells / 256);
  log(`Custom job '${job.name || 'custom'}': grid ${nx}×${ny}×${nz}, ${steps} steps…`);
  const t0 = performance.now();
  for (let s = 0; s < steps; s++) {
    if (stopRequested) { log('Custom job stopped at step ' + s + '.'); break; }
    device.queue.writeBuffer(uni, 12, new Uint32Array([s]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wg);
    pass.end();
    device.queue.submit([enc.finish()]);
    if ((s + 1) % 25 === 0 || s + 1 === steps) {
      await device.queue.onSubmittedWorkDone();
      log(`Custom job: step ${s + 1}/${steps}`);
    }
  }
  await device.queue.onSubmittedWorkDone();
  const el = (performance.now() - t0) / 1000;
  let fieldAbsSums = null;
  if (job.readback !== false) {
    fieldAbsSums = [];
    for (let f = 0; f < 6; f++) {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(bufs[f], 0, staging, 0, bytes);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(staging.getMappedRange());
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += Math.abs(data[i]);
      staging.unmap();
      fieldAbsSums.push(sum);
    }
  }
  bufs.forEach(b => b.destroy()); uni.destroy(); staging.destroy();
  const rate = cells * steps / el / 1e6;
  log(`Custom job done: ${el.toFixed(3)} s → ${rate.toFixed(1)} Mcell-steps/s`);
  const rec = { job: 'custom', name: job.name || 'custom', grid: [nx, ny, nz], steps,
                seconds: el, mcellPerSec: rate, fieldAbsSums,
                adapter: adapterSnapshot, when: new Date().toISOString() };
  results.runs.push(rec);
  return rec;
}

// AUTO-GENERATED by build/gen_shaders.py from wasm/fdtd/src/gpu.rs — do not edit by hand.
const SOLVER_SHADERS = {
  h: "\nstruct Params {\n    nx: u32, ny: u32, nz: u32, _pad0: u32,\n    dx: f32, dy: f32, dz: f32, dt: f32,\n    ch: f32, idx: f32, idy: f32, idz: f32,\n    rs: f32, src_ez_idx: u32, has_wire: u32, _pad1: u32,\n    tl_hx_bot: u32, tl_hx_top: u32, tl_hy_lft: u32, tl_hy_rgt: u32,\n    wire_ez_base: u32, wire_ez_stride: u32, wire_nseg: u32, wire_nnodes: u32,\n    wire_top_pec: u32, wire_dt_over_l: f32, wire_cap: f32, wire_deposit: f32,\n    src_v0: f32, src_t0: f32, src_tau: f32, _pad2: u32,\n};\n@group(0) @binding(0) var<uniform> p: Params;\n@group(0) @binding(2) var<storage, read_write> ex: array<f32>;\n@group(0) @binding(3) var<storage, read_write> ey: array<f32>;\n@group(0) @binding(4) var<storage, read_write> ez: array<f32>;\n@group(0) @binding(5) var<storage, read_write> hx: array<f32>;\n@group(0) @binding(6) var<storage, read_write> hy: array<f32>;\n@group(0) @binding(7) var<storage, read_write> hz: array<f32>;\n\nfn ex_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }\nfn ey_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }\nfn ez_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * p.nx + i; }\nfn hx_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }\nfn hy_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }\nfn hz_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * (p.nx + 1u) + i; }\n\n// Hx: nx x (ny+1) x (nz+1).  Hx -= ch*((Ey(k+1)-Ey(k))/dz - (Ez(j+1)-Ez(j))/dy)\n@compute @workgroup_size(8, 8, 4)\nfn update_hx(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x; let j = gid.y; let k = gid.z;\n    if (i >= p.nx || j > p.ny || k > p.nz) { return; }\n    let ey_k = select(0.0, ey[ey_idx(i, j, k)], k < p.nz);\n    let ey_kp = select(0.0, ey[ey_idx(i, j, k + 1u)], k + 1u < p.nz);\n    let ez_j = select(0.0, ez[ez_idx(i, j, k)], j < p.ny);\n    let ez_jp = select(0.0, ez[ez_idx(i, j + 1u, k)], j + 1u < p.ny);\n    let h = hx_idx(i, j, k);\n    hx[h] -= p.ch * ((ey_kp - ey_k) * p.idz - (ez_jp - ez_j) * p.idy);\n}\n\n// Hy: (nx+1) x ny x (nz+1).  Hy -= ch*((Ez(i+1)-Ez(i))/dx - (Ex(k+1)-Ex(k))/dz)\n@compute @workgroup_size(8, 8, 4)\nfn update_hy(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x; let j = gid.y; let k = gid.z;\n    if (i > p.nx || j >= p.ny || k > p.nz) { return; }\n    let im = min(i, p.nx - 1u);\n    let ez_i = select(0.0, ez[ez_idx(im, j, k)], i < p.nx);\n    let ez_ip = select(0.0, ez[ez_idx(i + 1u, j, k)], i + 1u < p.nx);\n    let ie = min(i, p.nx);\n    let ex_k = select(0.0, ex[ex_idx(ie, j, k)], k < p.nz);\n    let ex_kp = select(0.0, ex[ex_idx(ie, j, k + 1u)], k + 1u < p.nz);\n    let h = hy_idx(i, j, k);\n    hy[h] -= p.ch * ((ez_ip - ez_i) * p.idx - (ex_kp - ex_k) * p.idz);\n}\n\n// Hz: (nx+1) x (ny+1) x nz.  Hz -= ch*((Ex(j+1)-Ex(j))/dy - (Ey(i+1)-Ey(i))/dx)\n@compute @workgroup_size(8, 8, 4)\nfn update_hz(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x; let j = gid.y; let k = gid.z;\n    if (i > p.nx || j > p.ny || k >= p.nz) { return; }\n    let ie = min(i, p.nx);\n    let jm = min(j, p.ny - 1u);\n    let ex_j = ex[ex_idx(ie, jm, k)];\n    let ex_jp = select(0.0, ex[ex_idx(ie, j + 1u, k)], j + 1u < p.ny);\n    let ie2 = min(i, p.nx - 1u);\n    let ey_i = ey[ey_idx(ie2, j, k)];\n    let ey_ip = select(0.0, ey[ey_idx(i + 1u, j, k)], i + 1u < p.nx);\n    let h = hz_idx(i, j, k);\n    hz[h] -= p.ch * ((ex_jp - ex_j) * p.idy - (ey_ip - ey_i) * p.idx);\n}\n",
  e: "\nstruct Params {\n    nx: u32, ny: u32, nz: u32, _pad0: u32,\n    dx: f32, dy: f32, dz: f32, dt: f32,\n    ch: f32, idx: f32, idy: f32, idz: f32,\n    rs: f32, src_ez_idx: u32, has_wire: u32, _pad1: u32,\n    tl_hx_bot: u32, tl_hx_top: u32, tl_hy_lft: u32, tl_hy_rgt: u32,\n    wire_ez_base: u32, wire_ez_stride: u32, wire_nseg: u32, wire_nnodes: u32,\n    wire_top_pec: u32, wire_dt_over_l: f32, wire_cap: f32, wire_deposit: f32,\n    src_v0: f32, src_t0: f32, src_tau: f32, _pad2: u32,\n};\n@group(0) @binding(0) var<uniform> p: Params;\n@group(0) @binding(2) var<storage, read_write> ex: array<f32>;\n@group(0) @binding(3) var<storage, read_write> ey: array<f32>;\n@group(0) @binding(4) var<storage, read_write> ez: array<f32>;\n@group(0) @binding(5) var<storage, read_write> hx: array<f32>;\n@group(0) @binding(6) var<storage, read_write> hy: array<f32>;\n@group(0) @binding(7) var<storage, read_write> hz: array<f32>;\n@group(1) @binding(8) var<storage, read> ca_x: array<f32>;\n@group(1) @binding(9) var<storage, read> ca_y: array<f32>;\n@group(1) @binding(10) var<storage, read> ca_z: array<f32>;\n@group(1) @binding(11) var<storage, read> cb_x: array<f32>;\n@group(1) @binding(12) var<storage, read> cb_y: array<f32>;\n@group(1) @binding(13) var<storage, read> cb_z: array<f32>;\n\nfn ex_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }\nfn ey_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }\nfn ez_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * p.nx + i; }\nfn hx_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }\nfn hy_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }\nfn hz_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * (p.nx + 1u) + i; }\n\n// Ex = Ca*Ex + Cb*((Hy(k)-Hy(k-1))/dz - (Hz(j)-Hz(j-1))/dy)\n@compute @workgroup_size(8, 8, 4)\nfn update_ex(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x; let j = gid.y; let k = gid.z;\n    if (i > p.nx || j >= p.ny || k >= p.nz) { return; }\n    let h = hz_idx(i, j, k);\n    let hz_j = hz[h];\n    let hz_jm = select(0.0, hz[hz_idx(i, j - 1u, k)], j > 0u);\n    let hy_k = hy[hy_idx(i, j, k)];\n    let hy_km = select(0.0, hy[hy_idx(i, j, k - 1u)], k > 0u);\n    let e = ex_idx(i, j, k);\n    ex[e] = ca_x[e] * ex[e] + cb_x[e] * ((hy_k - hy_km) * p.idz - (hz_j - hz_jm) * p.idy);\n}\n\n// Ey = Ca*Ey + Cb*((Hz(i)-Hz(i-1))/dx - (Hx(k)-Hx(k-1))/dz)\n@compute @workgroup_size(8, 8, 4)\nfn update_ey(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x; let j = gid.y; let k = gid.z;\n    if (i >= p.nx || j > p.ny || k >= p.nz) { return; }\n    let hx_k = hx[hx_idx(i, j, k)];\n    let hx_km = select(0.0, hx[hx_idx(i, j, k - 1u)], k > 0u);\n    let hz_i = hz[hz_idx(i, j, k)];\n    let hz_im = select(0.0, hz[hz_idx(i - 1u, j, k)], i > 0u);\n    let e = ey_idx(i, j, k);\n    ey[e] = ca_y[e] * ey[e] + cb_y[e] * ((hz_i - hz_im) * p.idx - (hx_k - hx_km) * p.idz);\n}\n\n// Ez = Ca*Ez + Cb*((Hx(j)-Hx(j-1))/dy - (Hy(i)-Hy(i-1))/dx)\n@compute @workgroup_size(8, 8, 4)\nfn update_ez(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x; let j = gid.y; let k = gid.z;\n    if (i >= p.nx || j >= p.ny || k > p.nz) { return; }\n    let hy_i = hy[hy_idx(i, j, k)];\n    let hy_im = select(0.0, hy[hy_idx(i - 1u, j, k)], i > 0u);\n    let hx_j = hx[hx_idx(i, j, k)];\n    let hx_jm = select(0.0, hx[hx_idx(i, j - 1u, k)], j > 0u);\n    let e = ez_idx(i, j, k);\n    ez[e] = ca_z[e] * ez[e] + cb_z[e] * ((hx_j - hx_jm) * p.idy - (hy_i - hy_im) * p.idx);\n}\n",
  cpml: "\nstruct Params {\n    nx: u32, ny: u32, nz: u32, _pad0: u32,\n    dx: f32, dy: f32, dz: f32, dt: f32,\n    ch: f32, idx: f32, idy: f32, idz: f32,\n    rs: f32, src_ez_idx: u32, has_wire: u32, _pad1: u32,\n    tl_hx_bot: u32, tl_hx_top: u32, tl_hy_lft: u32, tl_hy_rgt: u32,\n    wire_ez_base: u32, wire_ez_stride: u32, wire_nseg: u32, wire_nnodes: u32,\n    wire_top_pec: u32, wire_dt_over_l: f32, wire_cap: f32, wire_deposit: f32,\n    src_v0: f32, src_t0: f32, src_tau: f32, _pad2: u32,\n};\nstruct FaceU { d0: vec4u, d1: vec4u };\n@group(0) @binding(0) var<uniform> p: Params;\n@group(2) @binding(1) var<uniform> face: FaceU;\n@group(0) @binding(2) var<storage, read_write> ex: array<f32>;\n@group(0) @binding(3) var<storage, read_write> ey: array<f32>;\n@group(0) @binding(4) var<storage, read_write> ez: array<f32>;\n@group(0) @binding(5) var<storage, read_write> hx: array<f32>;\n@group(0) @binding(6) var<storage, read_write> hy: array<f32>;\n@group(0) @binding(7) var<storage, read_write> hz: array<f32>;\n@group(1) @binding(11) var<storage, read> cb_x: array<f32>;\n@group(1) @binding(12) var<storage, read> cb_y: array<f32>;\n@group(1) @binding(13) var<storage, read> cb_z: array<f32>;\n@group(2) @binding(17) var<storage, read> coeff: array<f32>;\n@group(2) @binding(18) var<storage, read_write> psi: array<f32>;\n\nfn ex_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }\nfn ey_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }\nfn ez_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * p.nx + i; }\nfn hx_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * p.nx + i; }\nfn hy_idx(i: u32, j: u32, k: u32) -> u32 { return (k * p.ny + j) * (p.nx + 1u) + i; }\nfn hz_idx(i: u32, j: u32, k: u32) -> u32 { return (k * (p.ny + 1u) + j) * (p.nx + 1u) + i; }\n\nfn lo() -> u32 { return face.d0.y; }\nfn npml() -> u32 { return face.d0.z; }\nfn n1() -> u32 { return face.d0.w; }\nfn n2() -> u32 { return face.d1.x; }\nfn psi_base() -> u32 { return face.d1.y; }\nfn coeff_base() -> u32 { return face.d1.z; }\nfn be(pc: u32) -> f32 { return coeff[coeff_base() + pc]; }\nfn ae(pc: u32) -> f32 { return coeff[coeff_base() + npml() + pc]; }\nfn bh(pc: u32) -> f32 { return coeff[coeff_base() + 2u * npml() + pc]; }\nfn ah(pc: u32) -> f32 { return coeff[coeff_base() + 3u * npml() + pc]; }\nfn ki(pc: u32) -> f32 { return coeff[coeff_base() + 4u * npml() + pc]; }\nfn psi_a(pi: u32) -> u32 { return psi_base() + pi; }\nfn psi_b(pi: u32) -> u32 { return psi_base() + npml() * n1() * n2() + pi; }\nfn psi_c(pi: u32) -> u32 { return psi_base() + 2u * npml() * n1() * n2() + pi; }\nfn psi_d(pi: u32) -> u32 { return psi_base() + 3u * npml() * n1() * n2() + pi; }\n\n// ---- E corrections ----\n\n// x-normal: Ey via dHz/dx, Ez via dHy/dx.  dispatch (k, j, p).\n@compute @workgroup_size(8, 8, 4)\nfn cpml_e_x(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let pc = gid.z; let j = gid.y; let k = gid.x;\n    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }\n    let ie = select(p.nx - 1u - pc, pc, lo() == 1u);\n    let pi = (pc * n1() + j) * n2() + k;\n    let b = be(pc); let a = ae(pc); let kv = ki(pc); let omk = 1.0 - kv;\n    let hz_lo = select(0.0, hz[hz_idx(ie - 1u, j, k)], ie > 0u);\n    let dhzdx = (hz[hz_idx(ie, j, k)] - hz_lo) / p.dx;\n    let pa = psi_a(pi);\n    let ps = b * psi[pa] + a * dhzdx;\n    psi[pa] = ps;\n    let ei = ey_idx(ie, j, k);\n    ey[ei] += cb_y[ei] * (-omk * dhzdx + ps);\n    let hy_lo = select(0.0, hy[hy_idx(ie - 1u, j, k)], ie > 0u);\n    let dhydx = (hy[hy_idx(ie, j, k)] - hy_lo) / p.dx;\n    let pb = psi_b(pi);\n    let ps2 = b * psi[pb] + a * dhydx;\n    psi[pb] = ps2;\n    let ei2 = ez_idx(ie, j, k);\n    ez[ei2] += cb_z[ei2] * (omk * dhydx - ps2);\n}\n\n// y-normal: Ex via dHz/dy, Ez via dHx/dy.  dispatch (k, i, p).\n@compute @workgroup_size(8, 8, 4)\nfn cpml_e_y(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let pc = gid.z; let i = gid.y; let k = gid.x;\n    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }\n    let je = select(p.ny - 1u - pc, pc, lo() == 1u);\n    let pi = (pc * n1() + i) * n2() + k;\n    let b = be(pc); let a = ae(pc); let kv = ki(pc); let omk = 1.0 - kv;\n    let hz_lo = select(0.0, hz[hz_idx(i, je - 1u, k)], je > 0u);\n    let dhzdy = (hz[hz_idx(i, je, k)] - hz_lo) / p.dy;\n    let pa = psi_a(pi);\n    let ps = b * psi[pa] + a * dhzdy;\n    psi[pa] = ps;\n    let ei = ex_idx(i, je, k);\n    ex[ei] += cb_x[ei] * (-omk * dhzdy + ps);\n    let hx_lo = select(0.0, hx[hx_idx(i, je - 1u, k)], je > 0u);\n    let dhxdy = (hx[hx_idx(i, je, k)] - hx_lo) / p.dy;\n    let pb = psi_b(pi);\n    let ps2 = b * psi[pb] + a * dhxdy;\n    psi[pb] = ps2;\n    let ei2 = ez_idx(i, je, k);\n    ez[ei2] += cb_z[ei2] * (-omk * dhxdy + ps2);\n}\n\n// z-normal: Ex via dHy/dz, Ey via dHx/dz.  dispatch (j, i, p).\n@compute @workgroup_size(8, 8, 4)\nfn cpml_e_z(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let pc = gid.z; let i = gid.y; let j = gid.x;\n    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }\n    let ke = select(p.nz - 1u - pc, pc, lo() == 1u);\n    let pi = (pc * n1() + i) * n2() + j;\n    let b = be(pc); let a = ae(pc); let kv = ki(pc); let omk = 1.0 - kv;\n    let hy_lo = select(0.0, hy[hy_idx(i, j, ke - 1u)], ke > 0u);\n    let dhydz = (hy[hy_idx(i, j, ke)] - hy_lo) / p.dz;\n    let pa = psi_a(pi);\n    let ps = b * psi[pa] + a * dhydz;\n    psi[pa] = ps;\n    let ei = ex_idx(i, j, ke);\n    ex[ei] += cb_x[ei] * (-omk * dhydz + ps);\n    let hx_lo = select(0.0, hx[hx_idx(i, j, ke - 1u)], ke > 0u);\n    let dhxdz = (hx[hx_idx(i, j, ke)] - hx_lo) / p.dz;\n    let pb = psi_b(pi);\n    let ps2 = b * psi[pb] + a * dhxdz;\n    psi[pb] = ps2;\n    let ei2 = ey_idx(i, j, ke);\n    ey[ei2] += cb_y[ei2] * (omk * dhxdz - ps2);\n}\n\n// ---- H corrections ----\n\n// x-normal: Hy via dEz/dx, Hz via dEy/dx.  dispatch (k, j, p).\n@compute @workgroup_size(8, 8, 4)\nfn cpml_h_x(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let pc = gid.z; let j = gid.y; let k = gid.x;\n    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }\n    let ih = select(p.nx - pc, pc, lo() == 1u);\n    let pi = (pc * n1() + j) * n2() + k;\n    let b = bh(pc); let a = ah(pc); let kv = ki(pc); let omk = 1.0 - kv;\n    let e_hi = select(0.0, ez[ez_idx(ih + 1u, j, k)], ih + 1u < p.nx);\n    let e_lo = select(0.0, ez[ez_idx(ih, j, k)], ih < p.nx);\n    let dezdx = (e_hi - e_lo) / p.dx;\n    let pcc = psi_c(pi);\n    let ps = b * psi[pcc] + a * dezdx;\n    psi[pcc] = ps;\n    let hi = hy_idx(ih, j, k);\n    hy[hi] -= p.ch * (-omk * dezdx + ps);\n    let e2_hi = select(0.0, ey[ey_idx(ih + 1u, j, k)], ih + 1u < p.nx);\n    let e2_lo = ey[ey_idx(min(ih, p.nx - 1u), j, k)];\n    let deydx = (e2_hi - e2_lo) / p.dx;\n    let pd = psi_d(pi);\n    let ps2 = b * psi[pd] + a * deydx;\n    psi[pd] = ps2;\n    let hi2 = hz_idx(ih, j, k);\n    hz[hi2] -= p.ch * (omk * deydx - ps2);\n}\n\n// y-normal: Hx via dEz/dy, Hz via dEx/dy.  dispatch (k, i, p).\n@compute @workgroup_size(8, 8, 4)\nfn cpml_h_y(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let pc = gid.z; let i = gid.y; let k = gid.x;\n    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }\n    let jh = select(p.ny - pc, pc, lo() == 1u);\n    let pi = (pc * n1() + i) * n2() + k;\n    let b = bh(pc); let a = ah(pc); let kv = ki(pc); let omk = 1.0 - kv;\n    let e_hi = select(0.0, ez[ez_idx(i, jh + 1u, k)], jh + 1u < p.ny);\n    let e_lo = select(0.0, ez[ez_idx(i, jh, k)], jh < p.ny);\n    let dezdy = (e_hi - e_lo) / p.dy;\n    let pcc = psi_c(pi);\n    let ps = b * psi[pcc] + a * dezdy;\n    psi[pcc] = ps;\n    let hi = hx_idx(i, jh, k);\n    hx[hi] -= p.ch * (omk * dezdy - ps);\n    let e2_hi = select(0.0, ex[ex_idx(i, jh + 1u, k)], jh + 1u < p.ny);\n    let e2_lo = ex[ex_idx(i, min(jh, p.ny - 1u), k)];\n    let dexdy = (e2_hi - e2_lo) / p.dy;\n    let pd = psi_d(pi);\n    let ps2 = b * psi[pd] + a * dexdy;\n    psi[pd] = ps2;\n    let hi2 = hz_idx(i, jh, k);\n    hz[hi2] -= p.ch * (-omk * dexdy + ps2);\n}\n\n// z-normal: Hx via dEy/dz, Hy via dEx/dz.  dispatch (j, i, p).\n@compute @workgroup_size(8, 8, 4)\nfn cpml_h_z(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let pc = gid.z; let i = gid.y; let j = gid.x;\n    if (pc >= npml() || gid.y >= n1() || gid.x >= n2()) { return; }\n    let kh = select(p.nz - pc, pc, lo() == 1u);\n    let pi = (pc * n1() + i) * n2() + j;\n    let b = bh(pc); let a = ah(pc); let kv = ki(pc); let omk = 1.0 - kv;\n    let e_hi = select(0.0, ey[ey_idx(i, j, kh + 1u)], kh + 1u < p.nz);\n    let e_lo = select(0.0, ey[ey_idx(i, j, kh)], kh < p.nz);\n    let deydz = (e_hi - e_lo) / p.dz;\n    let pcc = psi_c(pi);\n    let ps = b * psi[pcc] + a * deydz;\n    psi[pcc] = ps;\n    let hi = hx_idx(i, j, kh);\n    hx[hi] -= p.ch * (-omk * deydz + ps);\n    let e2_hi = select(0.0, ex[ex_idx(i, j, kh + 1u)], kh + 1u < p.nz);\n    let e2_lo = select(0.0, ex[ex_idx(i, j, kh)], kh < p.nz);\n    let dexdz = (e2_hi - e2_lo) / p.dz;\n    let pd = psi_d(pi);\n    let ps2 = b * psi[pd] + a * dexdz;\n    psi[pd] = ps2;\n    let hi2 = hy_idx(i, j, kh);\n    hy[hi2] -= p.ch * (omk * dexdz - ps2);\n}\n",
  misc: "\nstruct Params {\n    nx: u32, ny: u32, nz: u32, _pad0: u32,\n    dx: f32, dy: f32, dz: f32, dt: f32,\n    ch: f32, idx: f32, idy: f32, idz: f32,\n    rs: f32, src_ez_idx: u32, has_wire: u32, _pad1: u32,\n    tl_hx_bot: u32, tl_hx_top: u32, tl_hy_lft: u32, tl_hy_rgt: u32,\n    wire_ez_base: u32, wire_ez_stride: u32, wire_nseg: u32, wire_nnodes: u32,\n    wire_top_pec: u32, wire_dt_over_l: f32, wire_cap: f32, wire_deposit: f32,\n    src_v0: f32, src_t0: f32, src_tau: f32, wire_feed: u32,\n};\n@group(0) @binding(0) var<uniform> p: Params;\n@group(0) @binding(2) var<storage, read_write> ex: array<f32>;\n@group(0) @binding(3) var<storage, read_write> ey: array<f32>;\n@group(0) @binding(4) var<storage, read_write> ez: array<f32>;\n@group(0) @binding(5) var<storage, read_write> hx: array<f32>;\n@group(0) @binding(6) var<storage, read_write> hy: array<f32>;\n@group(1) @binding(14) var<storage, read> pec_x: array<u32>;\n@group(1) @binding(15) var<storage, read> pec_y: array<u32>;\n@group(1) @binding(16) var<storage, read> pec_z: array<u32>;\n@group(1) @binding(13) var<storage, read> cb_z: array<f32>;\n// wire state: currents[0..nseg] ++ charges[0..nnodes]\n@group(3) @binding(19) var<storage, read_write> wire: array<f32>;\n// step state: [0]=step counter, [1]=t (f32 bits), [2..4] pad\n@group(3) @binding(20) var<storage, read_write> state: array<u32>;\n// probe record: interleaved (v, i, v_field) per step \u2014 v_field is the\n// gap-cell field voltage -Ez*dz at the wire's base cell, sampled after\n// deposit+PEC; it discriminates wire-model node voltage from the\n// physical terminal voltage in the recorded Zin.\n@group(3) @binding(21) var<storage, read_write> record: array<f32>;\n\nfn node_voltage(n: u32) -> f32 {\n    if (p.wire_top_pec == 1u && n == p.wire_nnodes - 1u) { return 0.0; }\n    return wire[p.wire_nseg + n] / p.wire_cap;\n}\n\n// Holland/Simpson current advance from E^n, then charge continuity.\n// Does not touch the grid (deposit is a separate, later step). With a\n// Th\u00e9venin source attached at the base node (wire_feed=1), node 0's\n// charge update is the implicit source solve\n// C*(V0^{n+1}-V0^n)/dt = (Vs - V0^{n+1})/Rs - I_0, with Vs evaluated\n// at t+dt (state[1] still holds t; source_step advances it later) \u2014\n// mirroring ThinWire::update_current. (Rs=0 pins V0 = Vs.)\n@compute @workgroup_size(1)\nfn wire_update(@builtin(global_invocation_id) gid: vec3<u32>) {\n    if (p.has_wire == 0u) { return; }\n    for (var s = 0u; s < p.wire_nseg; s++) {\n        let ezi = p.wire_ez_base + s * p.wire_ez_stride;\n        let emf = ez[ezi] * p.dz;\n        let v_tail = node_voltage(s);\n        let v_head = node_voltage(s + 1u);\n        wire[s] += p.wire_dt_over_l * (emf + v_tail - v_head);\n    }\n    for (var n = 0u; n < p.wire_nnodes; n++) {\n        if (n == 0u && p.wire_feed == 1u) { continue; }\n        let i_out = select(0.0, wire[n], n < p.wire_nseg);\n        let i_in = select(0.0, wire[n - 1u], n > 0u);\n        var dq = -p.dt * (i_out - i_in);\n        if (p.wire_top_pec == 1u && n == p.wire_nnodes - 1u) { dq = 0.0; }\n        wire[p.wire_nseg + n] += dq;\n    }\n    if (p.wire_top_pec == 1u) {\n        wire[p.wire_nseg + p.wire_nnodes - 1u] = 0.0;\n    }\n    if (p.wire_feed == 1u) {\n        let t_next = bitcast<f32>(state[1]) + p.dt;\n        let x = (t_next - p.src_t0) / p.src_tau;\n        let vs = -p.src_v0 * x * exp(-x * x);\n        let c = p.wire_cap;\n        if (p.rs <= 0.0) {\n            wire[p.wire_nseg] = c * vs;\n        } else {\n            let v0n = wire[p.wire_nseg] / c;\n            let v0new = (c * v0n / p.dt + vs / p.rs - wire[0]) / (c / p.dt + 1.0 / p.rs);\n            wire[p.wire_nseg] = c * v0new;\n        }\n    }\n}\n\n// Deposit I^{n+1/2} into Ez after the E update, through each segment\n// cell's own Ampere coefficient: Ez -= cb_z*I/(dx*dy). (The original\n// form used a constant dt/(eps0*dx*dy); cb_z is the consistent\n// per-cell coefficient \u2014 identical for air cells.)\n@compute @workgroup_size(1)\nfn wire_deposit(@builtin(global_invocation_id) gid: vec3<u32>) {\n    if (p.has_wire == 0u) { return; }\n    for (var s = 0u; s < p.wire_nseg; s++) {\n        let ezi = p.wire_ez_base + s * p.wire_ez_stride;\n        ez[ezi] -= cb_z[ezi] / (p.dx * p.dy) * wire[s];\n    }\n}\n\n@compute @workgroup_size(256)\nfn pec_ex(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let total = arrayLength(&ex);\n    for (var n = gid.x; n < total; n += 16776960u) {\n        if (pec_x[n] != 0u) { ex[n] = 0.0; }\n    }\n}\n\n@compute @workgroup_size(256)\nfn pec_ey(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let total = arrayLength(&ey);\n    for (var n = gid.x; n < total; n += 16776960u) {\n        if (pec_y[n] != 0u) { ey[n] = 0.0; }\n    }\n}\n\n@compute @workgroup_size(256)\nfn pec_ez(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let total = arrayLength(&ez);\n    for (var n = gid.x; n < total; n += 16776960u) {\n        if (pec_z[n] != 0u) { ez[n] = 0.0; }\n    }\n}\n\n// Terminal recording + the lumped-port source. Mirrors Fdtd::step's\n// tail: t += dt; Vs = gaussian(t). The wire path records the base\n// node's terminal pair (V_0 after its update in wire_update,\n// currents[0]) \u2014 the Th\u00e9venin source lives on the node and the gap\n// cell's Ez is NEVER overwritten (overwriting it erased the bottom\n// segment's deposit every step and pumped the wire's top mode; see\n// ThinWire::update_current). The lumped-port path uses the Norton\n// impressed-current form (see ResistiveSource::apply_norton) and\n// records (V_term, i_src) \u2014 the delivered current, not the Ampere\n// loop (the loop around the impressed-source cell anti-correlates\n// with it; see solver.rs).\n@compute @workgroup_size(1)\nfn source_step(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let step = state[0];\n    var t = bitcast<f32>(state[1]);\n    t += p.dt;\n    let x = (t - p.src_t0) / p.src_tau;\n    let vs = -p.src_v0 * x * exp(-x * x);\n    var i_wire: f32;\n    var v_term: f32;\n    var v_field: f32;\n    if (p.has_wire == 1u) {\n        i_wire = wire[0];\n        v_term = wire[p.wire_nseg] / p.wire_cap;\n        v_field = -ez[p.wire_ez_base] * p.dz;\n    } else {\n        i_wire = (hx[p.tl_hx_bot] - hx[p.tl_hx_top]) * p.dx\n               + (hy[p.tl_hy_rgt] - hy[p.tl_hy_lft]) * p.dy;\n        let si = p.src_ez_idx;\n        let v_now = -ez[si] * p.dz;\n        let i_src = (vs - v_now) / p.rs;\n        let eps_src = p.dt / cb_z[si];\n        ez[si] -= p.dt * i_src / (eps_src * p.dx * p.dy);\n        v_term = -ez[si] * p.dz;\n        i_wire = i_src; // record the delivered current (see header)\n        v_field = v_term;\n    }\n    record[3u * step] = v_term;\n    record[3u * step + 1u] = i_wire;\n    record[3u * step + 2u] = v_field;\n    state[0] = step + 1u;\n    state[1] = bitcast<u32>(t);\n}\n",
};
const SOLVER_DECL = {"p": {"binding": 0, "kind": "uniform", "rw": false}, "ex": {"binding": 2, "kind": "storage", "rw": true}, "ey": {"binding": 3, "kind": "storage", "rw": true}, "ez": {"binding": 4, "kind": "storage", "rw": true}, "hx": {"binding": 5, "kind": "storage", "rw": true}, "hy": {"binding": 6, "kind": "storage", "rw": true}, "hz": {"binding": 7, "kind": "storage", "rw": true}, "ca_x": {"binding": 8, "kind": "storage", "rw": false}, "ca_y": {"binding": 9, "kind": "storage", "rw": false}, "ca_z": {"binding": 10, "kind": "storage", "rw": false}, "cb_x": {"binding": 11, "kind": "storage", "rw": false}, "cb_y": {"binding": 12, "kind": "storage", "rw": false}, "cb_z": {"binding": 13, "kind": "storage", "rw": false}, "face": {"binding": 1, "kind": "uniform", "rw": false}, "coeff": {"binding": 17, "kind": "storage", "rw": false}, "psi": {"binding": 18, "kind": "storage", "rw": true}, "pec_x": {"binding": 14, "kind": "storage", "rw": false}, "pec_y": {"binding": 15, "kind": "storage", "rw": false}, "pec_z": {"binding": 16, "kind": "storage", "rw": false}, "wire": {"binding": 19, "kind": "storage", "rw": true}, "state": {"binding": 20, "kind": "storage", "rw": true}, "record": {"binding": 21, "kind": "storage", "rw": true}};
const SOLVER_USAGE = {"h": {"update_hx": ["ey", "ez", "hx", "p"], "update_hy": ["ex", "ez", "hy", "p"], "update_hz": ["ex", "ey", "hz", "p"]}, "e": {"update_ex": ["ca_x", "cb_x", "ex", "hy", "hz", "p"], "update_ey": ["ca_y", "cb_y", "ey", "hx", "hz", "p"], "update_ez": ["ca_z", "cb_z", "ez", "hx", "hy", "p"]}, "cpml": {"cpml_e_x": ["cb_y", "cb_z", "coeff", "ey", "ez", "face", "hy", "hz", "p", "psi"], "cpml_e_y": ["cb_x", "cb_z", "coeff", "ex", "ez", "face", "hx", "hz", "p", "psi"], "cpml_e_z": ["cb_x", "cb_y", "coeff", "ex", "ey", "face", "hx", "hy", "p", "psi"], "cpml_h_x": ["coeff", "ey", "ez", "face", "hy", "hz", "p", "psi"], "cpml_h_y": ["coeff", "ex", "ez", "face", "hx", "hz", "p", "psi"], "cpml_h_z": ["coeff", "ex", "ey", "face", "hx", "hy", "p", "psi"]}, "misc": {"wire_update": ["ez", "p", "state", "wire"], "wire_deposit": ["cb_z", "ez", "p", "wire"], "pec_ex": ["ex", "pec_x"], "pec_ey": ["ey", "pec_y"], "pec_ez": ["ez", "pec_z"], "source_step": ["cb_z", "ez", "hx", "hy", "p", "record", "state", "wire"]}};

// ---------------------------------------------------------------------------
// Full-solver core for the WebGPU bench (page v5): runs the REAL AntennaSim
// fdtd solver (gpu.rs WGSL, bind groups split per entry point) on the
// Snyder patch. The geometry/materials/CPML builder below is a faithful
// f32 port of geometry.rs `build_snyder`, material.rs, cpml.rs and the
// gpu.rs host packing; build checksums in the job result are verified
// against the Rust reference (examples/snyder_ref.rs).
// ---------------------------------------------------------------------------
const F = Math.fround;
// Snyder geometry variants (z-convergence ladder). Base: dz = 0.525 mm
// (3 substrate layers); zref: dz/2 (6 layers); zref2: dz/4 (12);
// zref3: dz/8 (24). k-planes follow Z0 = -(k_ground + 0.5) * dz; the
// post starts one Ez sample above the source cell in every variant.
function snyderGeom(variant) {
  const LADDER = {
    '': { dz: 5.25e-4, nz: 61, kg: 31, layers: 3 },
    zref: { dz: 2.625e-4, nz: 123, kg: 63, layers: 6 },
    zref2: { dz: 1.3125e-4, nz: 245, kg: 127, layers: 12 },
    zref3: { dz: 6.5625e-5, nz: 492, kg: 255, layers: 24 },
  };
  const L = LADDER[variant || ''];
  const DX = F(5.0e-4), DY = F(5.0e-4);
  const DZ = F(L.dz);
  const NX = 232, NY = 232, NZ = L.nz;
  const X0 = F(-58.0e-3), Y0 = F(-58.0e-3);
  const kGround = L.kg, subLayers = L.layers;
  const Z0 = F(-(kGround + 0.5) * DZ);
  const EPS0 = F(8.854187817e-12);
  const inv = F(Math.sqrt(F(F(1 / F(DX * DX)) + F(1 / F(DY * DY))) + F(1 / F(DZ * DZ))));
  const dt = F(0.1 / F(F(299792458.0) * inv));
  return {
    DX, DY, DZ, NX, NY, NZ, X0, Y0, Z0, EPS0, dt, kGround,
    kPatch: kGround + subLayers, kSrc: kGround + 1,
    postLo: kGround + 2, postHi: kGround + subLayers,
    thickA: kGround, thickB: kGround - 1,
  };
}
const SNY = snyderGeom('');
const exIdx = (i, j, k) => (k * SNY.NY + j) * (SNY.NX + 1) + i;
const eyIdx = (i, j, k) => (k * (SNY.NY + 1) + j) * SNY.NX + i;
const ezIdx = (i, j, k) => (k * SNY.NY + j) * SNY.NX + i;
const hxIdx = (i, j, k) => (k * (SNY.NY + 1) + j) * SNY.NX + i;
const hyIdx = (i, j, k) => (k * SNY.NY + j) * (SNY.NX + 1) + i;
const hzIdx = (i, j, k) => (k * (SNY.NY + 1) + j) * (SNY.NX + 1) + i;

function buildSnyderSolver(cfg) {
  const G = snyderGeom(cfg.zref3 ? 'zref3' : cfg.zref2 ? 'zref2' : cfg.zref ? 'zref' : '');
  const { NX, NY, NZ, DX, DY, DZ, X0, Y0, Z0, EPS0, dt } = G;
  const SUB_H = F(1.575e-3), subHalf = F(F(101.6e-3) / 2);
  const patchHx = F(F(28.2084e-3) / 2), patchHy = F(F(37.5789e-3) / 2);
  const PROBE_X = F(F(-14.1042e-3) + F(7.5658e-3)), PROBE_HALF = F(0.65e-3);
  const EPS_R = F(4.4);
  const sigmaSub = F(F(F(F(F(2.0 * F(Math.PI)) * F(2.5e9)) * EPS0) * EPS_R) * F(0.02));
  const kGround = G.kGround, kPatch = G.kPatch, kSrc = G.kSrc;
  const pcI = Math.round(F(F(F(PROBE_X - X0) / DX) - 0.5));
  const pcJ = Math.round(F(F(F(0.0 - Y0) / DY) - 0.5));

  const nEx = (NX + 1) * NY * NZ, nEy = NX * (NY + 1) * NZ, nEz = NX * NY * (NZ + 1);
  const cbVac = F(dt / EPS0);
  const ca = [new Float32Array(nEx).fill(1), new Float32Array(nEy).fill(1), new Float32Array(nEz).fill(1)];
  const cb = [new Float32Array(nEx).fill(cbVac), new Float32Array(nEy).fill(cbVac), new Float32Array(nEz).fill(cbVac)];
  const pec = [new Uint32Array(nEx), new Uint32Array(nEy), new Uint32Array(nEz)];
  const idxFns = [exIdx, eyIdx, ezIdx];
  function setDielectric(comp, idx, epsR, sigma) {
    const eps = F(EPS0 * epsR);
    const s = F(F(sigma * dt) / F(2 * eps));
    ca[comp][idx] = F(F(1 - s) / F(1 + s));
    cb[comp][idx] = F(F(dt / eps) / F(1 + s));
  }
  // Substrate dielectric — Ex loop (channel skip includes k range test).
  for (let k = 0; k < NZ; k++) {
    const zc = F(F((k + 0.5) * DZ) + Z0);
    if (zc < 0 || zc > SUB_H) continue;
    for (let j = 0; j < NY; j++) {
      const y = F(F((j + 0.5) * DY) + Y0);
      if (Math.abs(y) > subHalf) continue;
      for (let i = 0; i <= NX; i++) {
        const x = F(F(i * DX) + X0);
        if (Math.abs(x) > subHalf) continue;
        if (Math.abs(F(x - PROBE_X)) < F(PROBE_HALF + DX) && Math.abs(y) < F(PROBE_HALF + DY) &&
            k >= kGround && k <= kPatch) continue;
        setDielectric(0, exIdx(i, j, k), EPS_R, sigmaSub);
      }
    }
  }
  // Ey substrate.
  for (let k = 0; k < NZ; k++) {
    for (let j = 0; j <= NY; j++) {
      const y = F(F(j * DY) + Y0);
      if (Math.abs(y) > subHalf) continue;
      const zc = F(F((k + 0.5) * DZ) + Z0);
      if (zc < 0 || zc > SUB_H) continue;
      for (let i = 0; i < NX; i++) {
        const x = F(F((i + 0.5) * DX) + X0);
        if (Math.abs(x) > subHalf) continue;
        if (Math.abs(F(x - PROBE_X)) < F(PROBE_HALF + DX) && Math.abs(y) < F(PROBE_HALF + DY)) continue;
        setDielectric(1, eyIdx(i, j, k), EPS_R, sigmaSub);
      }
    }
  }
  // Ez substrate.
  for (let k = 0; k <= NZ; k++) {
    const z = F(F(k * DZ) + Z0);
    if (z < 0 || z > SUB_H) continue;
    for (let j = 0; j < NY; j++) {
      const y = F(F((j + 0.5) * DY) + Y0);
      if (Math.abs(y) > subHalf) continue;
      for (let i = 0; i < NX; i++) {
        const x = F(F((i + 0.5) * DX) + X0);
        if (Math.abs(x) > subHalf) continue;
        if (Math.abs(F(x - PROBE_X)) < F(PROBE_HALF + DX) && Math.abs(y) < F(PROBE_HALF + DY)) continue;
        setDielectric(2, ezIdx(i, j, k), EPS_R, sigmaSub);
      }
    }
  }
  // Ground sheet (Ex/Ey at k=31), patch sheet (k=34).
  for (let j = 0; j < NY; j++) {
    const y = F(F((j + 0.5) * DY) + Y0);
    if (Math.abs(y) > subHalf) continue;
    for (let i = 0; i <= NX; i++) {
      const x = F(F(i * DX) + X0);
      if (Math.abs(x) <= subHalf) pec[0][exIdx(i, j, kGround)] = 1;
    }
  }
  for (let j = 0; j <= NY; j++) {
    const y = F(F(j * DY) + Y0);
    if (Math.abs(y) > subHalf) continue;
    for (let i = 0; i < NX; i++) {
      const x = F(F((i + 0.5) * DX) + X0);
      if (Math.abs(x) <= subHalf) pec[1][eyIdx(i, j, kGround)] = 1;
    }
  }
  for (let j = 0; j < NY; j++) {
    const y = F(F((j + 0.5) * DY) + Y0);
    if (Math.abs(y) > patchHy) continue;
    for (let i = 0; i <= NX; i++) {
      const x = F(F(i * DX) + X0);
      if (Math.abs(x) <= patchHx) pec[0][exIdx(i, j, kPatch)] = 1;
    }
  }
  for (let j = 0; j <= NY; j++) {
    const y = F(F(j * DY) + Y0);
    if (Math.abs(y) > patchHy) continue;
    for (let i = 0; i < NX; i++) {
      const x = F(F((i + 0.5) * DX) + X0);
      if (Math.abs(x) <= patchHx) pec[1][eyIdx(i, j, kPatch)] = 1;
    }
  }
  // PEC post (config.post): plus-cross of Ez columns over the probe span.
  if (cfg.post) {
    for (const [di, dj] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
      for (let k = G.postLo; k <= G.postHi; k++) pec[2][ezIdx(pcI + di, pcJ + dj, k)] = 1;
    }
  }
  // Volumetric ground (config.thick): Ez PEC at the two layers at/below
  // the ground sheet over the ground footprint.
  if (cfg.thick) {
    for (let j = 14; j < 218; j++) for (let i = 14; i < 218; i++) {
      pec[2][ezIdx(i, j, G.thickA)] = 1;
      pec[2][ezIdx(i, j, G.thickB)] = 1;
    }
  }
  // CPML faces (xlo, xhi, ylo, yhi, zlo, zhi), npml=12.
  const NPML = 12;
  const faceDims = [
    { axis: 0, lo: 1, n1: NY, n2: NZ, delta: DX }, { axis: 0, lo: 0, n1: NY, n2: NZ, delta: DX },
    { axis: 1, lo: 1, n1: NX, n2: NZ, delta: DY }, { axis: 1, lo: 0, n1: NX, n2: NZ, delta: DY },
    { axis: 2, lo: 1, n1: NX, n2: NY, delta: DZ }, { axis: 2, lo: 0, n1: NX, n2: NY, delta: DZ },
  ];
  const coeff = [];
  const faceWords = [];
  let psiTotal = 0;
  for (const fd of faceDims) {
    const m = 3.0, kappaMax = 7.0, alphaMax = 0.3;
    const sigmaMax = F(F(m + 1) / F(F(150.0 * F(Math.PI)) * fd.delta));
    const be = [], ae = [], kinv = [];
    for (let p = 0; p < NPML; p++) {
      const rho = F(F(NPML - p) / NPML);
      const sigma = F(sigmaMax * Math.pow(rho, m));
      const kappa = F(1 + F(F(kappaMax - 1) * Math.pow(rho, m)));
      const alpha = F(alphaMax * F(p / NPML));
      const b = F(Math.exp(F(F(-F(F(sigma / kappa) + alpha)) * dt) / EPS0));
      const a = (sigma + F(kappa * alpha)) > 0
        ? F(F(F(b - 1) * sigma) / F(kappa * F(sigma + F(kappa * alpha)))) : 0;
      be.push(b); ae.push(a); kinv.push(F(1 / kappa));
    }
    const coeffBase = coeff.length;
    coeff.push(...be, ...ae, ...be, ...ae, ...kinv); // b_e, a_e, b_h(=b_e), a_h(=a_e), kinv
    const psiBase = psiTotal;
    psiTotal += 4 * NPML * fd.n1 * fd.n2;
    faceWords.push([fd.axis, fd.lo, NPML, fd.n1, fd.n2, psiBase, coeffBase, 0]);
  }
  // Gaussian source (2.5 GHz peak, v0 = 1) — GaussianSource::new.
  const tau = F(1 / F(F(F(Math.PI) * F(2.5e9)) * F(Math.SQRT2)));
  const t0 = F(4 * tau);
  // Params (GpuParams layout, 128 bytes written by caller).
  const mu0 = F(4.0e-7 * Math.PI);
  const params = {
    nx: NX, ny: NY, nz: NZ, dx: DX, dy: DY, dz: DZ, dt,
    ch: F(dt / mu0), idx: F(1 / DX), idy: F(1 / DY), idz: F(1 / DZ),
    rs: 50.0, srcEzIdx: ezIdx(pcI, pcJ, kSrc), hasWire: 0,
    tlHxBot: hxIdx(pcI, pcJ - 1, kSrc), tlHxTop: hxIdx(pcI, pcJ, kSrc),
    tlHyLft: hyIdx(pcI - 1, pcJ, kSrc), tlHyRgt: hyIdx(pcI, pcJ, kSrc),
    wireEzBase: 0, wireEzStride: NX * NY, wireNseg: 0, wireNnodes: 0,
    wireTopPec: 0, wireDtOverL: 0.0, wireCap: 1.0, wireDeposit: 0.0,
    srcV0: 1.0, srcT0: t0, srcTau: tau, wireFeed: 0,
  };
  // Checksums (mirror examples/snyder_ref.rs).
  const wsum = a => {
    let s = 0, w = 0;
    for (let i = 0; i < a.length; i++) { s += a[i]; w += a[i] * (i + 1); }
    return [s, w];
  };
  const checksums = {};
  const names = ['ca_x', 'ca_y', 'ca_z'];
  const cbn = ['cb_x', 'cb_y', 'cb_z'];
  for (let c = 0; c < 3; c++) {
    checksums[names[c]] = wsum(ca[c]);
    checksums[cbn[c]] = wsum(cb[c]);
    let cnt = 0;
    for (let i = 0; i < pec[c].length; i++) cnt += pec[c][i];
    checksums['pec_' + 'xyz'[c]] = cnt;
  }
  checksums.coeff = wsum(Float32Array.from(coeff));
  checksums.psiLen = psiTotal;
  checksums.srcCell = [pcI, pcJ, kSrc];
  checksums.dt = dt;
  return {
    ca, cb, pec, coeff: Float32Array.from(coeff), psiLen: psiTotal,
    faceWords, params, checksums,
    dims: { NX, NY, NZ, dt },
    sizes: { ex: nEx, ey: nEy, ez: nEz, hx: NX * (NY + 1) * (NZ + 1), hy: (NX + 1) * NY * (NZ + 1), hz: (NX + 1) * (NY + 1) * NZ },
  };
}

function paramsBytes(p) {
  const buf = new ArrayBuffer(128);
  const v = new DataView(buf);
  const u = (o, x) => v.setUint32(o, x, true);
  const f = (o, x) => v.setFloat32(o, x, true);
  u(0, p.nx); u(4, p.ny); u(8, p.nz);
  f(16, p.dx); f(20, p.dy); f(24, p.dz); f(28, p.dt);
  f(32, p.ch); f(36, p.idx); f(40, p.idy); f(44, p.idz);
  f(48, p.rs); u(52, p.srcEzIdx); u(56, p.hasWire);
  u(64, p.tlHxBot); u(68, p.tlHxTop); u(72, p.tlHyLft); u(76, p.tlHyRgt);
  u(80, p.wireEzBase); u(84, p.wireEzStride); u(88, p.wireNseg); u(92, p.wireNnodes);
  u(96, p.wireTopPec); f(100, p.wireDtOverL); f(104, p.wireCap); f(108, p.wireDeposit);
  f(112, p.srcV0); f(116, p.srcT0); f(120, p.srcTau); u(124, p.wireFeed);
  return buf;
}

// ---------------------------------------------------------------------------
// Solver runner: builds buffers + pipelines from SOLVER_SHADERS/SOLVER_USAGE
// (one minimal bind-group layout per entry point), then steps in the exact
// native order (gpu.rs GpuFdtd::run): H, CPML-H (face order), wire_update,
// E, CPML-E, wire_deposit, PEC, source. Chunks of 64 steps per submission.
// ---------------------------------------------------------------------------
const SOLVER_ENTRIES = {
  h: ['update_hx', 'update_hy', 'update_hz'],
  e: ['update_ex', 'update_ey', 'update_ez'],
  cpml: ['cpml_e_x', 'cpml_e_y', 'cpml_e_z', 'cpml_h_x', 'cpml_h_y', 'cpml_h_z'],
  misc: ['wire_update', 'wire_deposit', 'pec_ex', 'pec_ey', 'pec_ez', 'source_step'],
};
const BINDING_GROUP = b => (b === 0 || (b >= 2 && b <= 7)) ? 0 : (b >= 8 && b <= 16) ? 1 : (b === 1 || b === 17 || b === 18) ? 2 : 3;

async function coreSolver(job) {
  const cfg = Object.assign({ post: true, thick: true }, job.config || {});
  const steps = job.steps || 1000;
  const chunkMax = job.chunk || 64;
  log(`Solver job: Snyder post=${cfg.post} thick=${cfg.thick}, ${steps} steps — building geometry…`);
  const tBuild = performance.now();
  const B = buildSnyderSolver(cfg);
  log(`Geometry built in ${((performance.now() - tBuild) / 1000).toFixed(1)} s. src cell ${B.checksums.srcCell}, dt=${B.checksums.dt.toExponential(4)}`);
  const { NX, NY, NZ } = B.dims;
  const S = B.sizes;
  const usageRW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const usageRO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mkBuf = (label, size, usage) => device.createBuffer({ label, size, usage });
  const putBuf = (label, data, usage) => {
    const b = mkBuf(label, data.byteLength, usage);
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const fieldBufs = [2, 3, 4, 5, 6, 7].map((b, i) =>
    mkBuf('field' + b, [S.ex, S.ey, S.ez, S.hx, S.hy, S.hz][i] * 4, usageRW));
  const bufByBinding = {};
  [2, 3, 4, 5, 6, 7].forEach((b, i) => { bufByBinding[b] = fieldBufs[i]; });
  bufByBinding[8] = putBuf('ca_x', B.ca[0], usageRO);
  bufByBinding[9] = putBuf('ca_y', B.ca[1], usageRO);
  bufByBinding[10] = putBuf('ca_z', B.ca[2], usageRO);
  bufByBinding[11] = putBuf('cb_x', B.cb[0], usageRO);
  bufByBinding[12] = putBuf('cb_y', B.cb[1], usageRO);
  bufByBinding[13] = putBuf('cb_z', B.cb[2], usageRO);
  bufByBinding[14] = putBuf('pec_x', B.pec[0], usageRO);
  bufByBinding[15] = putBuf('pec_y', B.pec[1], usageRO);
  bufByBinding[16] = putBuf('pec_z', B.pec[2], usageRO);
  bufByBinding[17] = putBuf('coeff', B.coeff, usageRO);
  bufByBinding[18] = mkBuf('psi', B.psiLen * 4, usageRW);
  bufByBinding[19] = putBuf('wire', new Float32Array(4), usageRW);
  bufByBinding[20] = putBuf('state', new Uint32Array(4), usageRW);
  bufByBinding[21] = mkBuf('record', 3 * steps * 4, usageRW);
  const paramsBuf = mkBuf('params', 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(paramsBuf, 0, paramsBytes(B.params));
  bufByBinding[0] = paramsBuf;
  const faceBuf = mkBuf('faces', 6 * 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  {
    const fb = new ArrayBuffer(6 * 256);
    const dv = new DataView(fb);
    B.faceWords.forEach((w, fi) => w.forEach((x, wi) => dv.setUint32(fi * 256 + wi * 4, x, true)));
    device.queue.writeBuffer(faceBuf, 0, fb);
  }
  bufByBinding[1] = faceBuf;

  // Shader modules + compilation check.
  const modules = {};
  for (const key of Object.keys(SOLVER_ENTRIES)) {
    const mod = device.createShaderModule({ code: SOLVER_SHADERS[key], label: 'solver-' + key });
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter(m => m.type === 'error');
    info.messages.filter(m => m.type === 'warning')
      .forEach(m => log(`WGSL[${key}] warning (line ${m.lineNum}): ${m.message}`));
    if (errs.length) throw new Error(`WGSL[${key}] compile error: ` +
      errs.map(m => 'line ' + m.lineNum + ': ' + m.message).join('; '));
    modules[key] = mod;
  }
  // Pipelines with per-entry-point minimal layouts.
  const emptyLayout = device.createBindGroupLayout({ entries: [] });
  const pipes = {};
  for (const [key, entries] of Object.entries(SOLVER_ENTRIES)) {
    for (const entry of entries) {
      const names = SOLVER_USAGE[key][entry];
      const groupLayouts = [], groupEntries = [[], [], [], []];
      for (let g = 0; g < 4; g++) {
        const inGroup = names.filter(n => BINDING_GROUP(SOLVER_DECL[n].binding) === g);
        if (!inGroup.length) { groupLayouts.push(emptyLayout); continue; }
        groupLayouts.push(device.createBindGroupLayout({
          entries: inGroup.map(n => {
            const d = SOLVER_DECL[n];
            return {
              binding: d.binding, visibility: GPUShaderStage.COMPUTE,
              buffer: d.kind === 'uniform'
                ? (d.binding === 1 ? { type: 'uniform', hasDynamicOffset: true } : { type: 'uniform' })
                : { type: d.rw ? 'storage' : 'read-only-storage' },
            };
          }),
        }));
        for (const n of inGroup) {
          const d = SOLVER_DECL[n];
          groupEntries[g].push({
            binding: d.binding,
            resource: d.binding === 1
              ? { buffer: faceBuf, offset: 0, size: 32 }
              : { buffer: bufByBinding[d.binding] },
          });
        }
      }
      const layout = device.createPipelineLayout({ bindGroupLayouts: groupLayouts });
      const pipe = device.createComputePipeline({
        layout, compute: { module: modules[key], entryPoint: entry } });
      const groups = groupEntries.map((ents, g) => ents.length
        ? device.createBindGroup({ layout: groupLayouts[g], entries: ents }) : null);
      pipes[entry] = { pipe, groups };
    }
  }
  const setP = (pass, name, faceIdx) => {
    const P = pipes[name];
    pass.setPipeline(P.pipe);
    for (let g = 0; g < 4; g++) {
      if (!P.groups[g]) continue;
      if (g === 2 && faceIdx !== undefined) pass.setBindGroup(g, P.groups[g], [faceIdx * 256]);
      else pass.setBindGroup(g, P.groups[g]);
    }
  };
  const wg3 = (x, y, z) => [Math.ceil(x / 8), Math.ceil(y / 8), Math.ceil(z / 4)];
  const dimsH = [wg3(NX, NY + 1, NZ + 1), wg3(NX + 1, NY, NZ + 1), wg3(NX + 1, NY + 1, NZ)];
  const dimsE = [wg3(NX + 1, NY, NZ), wg3(NX, NY + 1, NZ), wg3(NX, NY, NZ + 1)];
  const faceDisp = B.faceWords.map(w => wg3(w[4], w[3], w[2]));
  const pecDisp = [S.ex, S.ey, S.ez].map(s => [Math.min(Math.ceil(s / 256), 65535), 1, 1]);
  const disp = (pass, d) => pass.dispatchWorkgroups(d[0], d[1], d[2]);

  log(`Solver: running ${steps} steps…`);
  const t0 = performance.now();
  let done = 0;
  while (done < steps) {
    if (stopRequested) { log(`Solver stopped at step ${done}.`); break; }
    const n = Math.min(chunkMax, steps - done);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let s = 0; s < n; s++) {
      setP(pass, 'update_hx'); disp(pass, dimsH[0]);
      setP(pass, 'update_hy'); disp(pass, dimsH[1]);
      setP(pass, 'update_hz'); disp(pass, dimsH[2]);
      for (let fi = 0; fi < 6; fi++) { setP(pass, 'cpml_h_' + 'xyz'[B.faceWords[fi][0]], fi); disp(pass, faceDisp[fi]); }
      setP(pass, 'wire_update'); pass.dispatchWorkgroups(1, 1, 1);
      setP(pass, 'update_ex'); disp(pass, dimsE[0]);
      setP(pass, 'update_ey'); disp(pass, dimsE[1]);
      setP(pass, 'update_ez'); disp(pass, dimsE[2]);
      for (let fi = 0; fi < 6; fi++) { setP(pass, 'cpml_e_' + 'xyz'[B.faceWords[fi][0]], fi); disp(pass, faceDisp[fi]); }
      setP(pass, 'wire_deposit'); pass.dispatchWorkgroups(1, 1, 1);
      setP(pass, 'pec_ex'); disp(pass, pecDisp[0]);
      setP(pass, 'pec_ey'); disp(pass, pecDisp[1]);
      setP(pass, 'pec_ez'); disp(pass, pecDisp[2]);
      setP(pass, 'source_step'); pass.dispatchWorkgroups(1, 1, 1);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
    done += n;
    await device.queue.onSubmittedWorkDone();
    const el = (performance.now() - t0) / 1000;
    log(`Solver: step ${done}/${steps} (${(NX * NY * NZ * done / el / 1e6).toFixed(1)} Mcell/s)`);
  }
  const seconds = (performance.now() - t0) / 1000;
  // Readback: record buffer + Ez checksum.
  async function readback(buf, count) {
    const staging = device.createBuffer({
      size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, count * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    return out;
  }
  const rec = await readback(bufByBinding[21], 3 * done);
  const ez = await readback(bufByBinding[4], S.ez);
  let ezAbsSum = 0;
  for (let i = 0; i < ez.length; i++) ezAbsSum += Math.abs(ez[i]);
  const V = new Float64Array(done), I = new Float64Array(done);
  let sumAbsV = 0, sumAbsI = 0, maxAbsV = 0;
  for (let s = 0; s < done; s++) {
    V[s] = rec[3 * s]; I[s] = rec[3 * s + 1];
    sumAbsV += Math.abs(V[s]); sumAbsI += Math.abs(I[s]);
    if (Math.abs(V[s]) > maxAbsV) maxAbsV = Math.abs(V[s]);
  }
  let tailMax = 0;
  for (let s = Math.max(0, done - 100); s < done; s++) tailMax = Math.max(tailMax, Math.abs(V[s]));
  // DFT (Goertzel recurrence, mirrors DftProbe::dft); band from the job
  // (default 2000..3000 MHz in 101 points).
  const fA = (job.freqs && job.freqs[0]) || 2.0e9;
  const fB = (job.freqs && job.freqs[1]) || 3.0e9;
  const nF = (job.freqs && job.freqs[2]) || 101;
  const zin = [];
  for (let m = 0; m < nF; m++) {
    const fhz = fA + (fB - fA) * (nF === 1 ? 0 : m / (nF - 1));
    const wdt = 2 * Math.PI * fhz * B.dims.dt;
    const cw = Math.cos(wdt), sw = Math.sin(wdt);
    let c = 1, s = 0, vr = 0, vi = 0, ir = 0, ii = 0;
    for (let n = 0; n < done; n++) {
      vr += V[n] * c; vi -= V[n] * s; ir += I[n] * c; ii -= I[n] * s;
      const cn = c * cw - s * sw; s = s * cw + c * sw; c = cn;
    }
    const den = ir * ir + ii * ii;
    const zr = den > 0 ? (vr * ir + vi * ii) / den : 0;
    const zi = den > 0 ? (vi * ir - vr * ii) / den : 0;
    const gr = (zr - 50) * (zr + 50) + zi * zi, gi = 2 * 50 * zi; // (Z-50)/(Z+50) num
    const gd = (zr + 50) * (zr + 50) + zi * zi;
    const gmag = gd > 0 ? Math.sqrt((gr * gr + gi * gi)) / gd : 0;
    zin.push([+(fhz / 1e6).toFixed(0), +zr.toFixed(3), +zi.toFixed(3),
              +(20 * Math.log10(Math.max(gmag, 1e-12))).toFixed(2)]);
  }
  Object.values(bufByBinding).forEach(b => b.destroy());
  const result = {
    job: 'solver', config: cfg, steps: done, seconds,
    mcellPerSec: NX * NY * NZ * done / seconds / 1e6,
    checksums: B.checksums,
    series: { sumAbsV, sumAbsI, maxAbsV, lastV: V[done - 1], lastI: I[done - 1],
              tailRatio: maxAbsV > 0 ? tailMax / maxAbsV : 0 },
    ezAbsSum, zin,
    adapter: adapterSnapshot, when: new Date().toISOString(),
  };
  results.runs.push(result);
  log(`Solver done: ${seconds.toFixed(1)} s, ${result.mcellPerSec.toFixed(1)} Mcell/s, ` +
      `sumAbsV=${sumAbsV.toExponential(4)}, ezAbsSum=${ezAbsSum.toExponential(4)}`);
  return result;
}

// ---------------------------------------------------------------------------
// Manual job handlers
// ---------------------------------------------------------------------------
async function runCheck() {
  if (jobRunning) return;
  jobRunning = true;
  const n = parseInt($('checkGrid').value, 10);
  const steps = Math.max(1, parseInt($('checkSteps').value, 10) || 10);
  $('checkBtn').disabled = true;
  $('checkResult').innerHTML = '';
  try {
    const rec = await coreParity(n, steps);
    $('checkResult').innerHTML = '<span class="pill ' + (rec.pass ? 'ok' : 'bad') + '">' +
      (rec.pass ? 'PASS' : 'FAIL') + '</span> <span class="mono">max abs diff = ' +
      rec.maxAbsDiff.toExponential(3) + '</span>';
  } catch (e) {
    log('Parity check failed: ' + e.message);
    $('checkResult').innerHTML = '<span class="pill bad">error</span> ' + escapeHtml(e.message);
  }
  jobRunning = false;
  $('checkBtn').disabled = false;
}

async function runBench() {
  if (jobRunning) return;
  jobRunning = true;
  stopRequested = false;
  const [nx, ny, nz] = $('benchGrid').value.split(',').map(Number);
  const steps = Math.max(10, parseInt($('benchSteps').value, 10) || 200);
  const repeats = Math.min(9, Math.max(1, parseInt($('benchRepeats').value, 10) || 1));
  $('benchBtn').disabled = true; $('checkBtn').disabled = true;
  $('stopBtn').classList.remove('hidden');
  const prog = $('benchProgress');
  prog.classList.remove('hidden'); prog.value = 0;
  $('benchResult').innerHTML = '';
  try {
    const out = await coreBench(nx, ny, nz, steps, repeats, (d, t) => { prog.value = (d / t) * 100; });
    if (out.rates.length) {
      $('benchResult').innerHTML = '<span class="pill ok">done</span> ' +
        '<span class="mono">median ' + out.medianMcellPerSec.toFixed(1) + ' Mcell/s' +
        (out.rates.length > 1 ? ' (runs: ' + out.rates.map(x => x.toFixed(1)).join(', ') + ')' : '') +
        '</span>';
    }
  } catch (e) {
    log('Benchmark failed: ' + e.message);
    $('benchResult').innerHTML = '<span class="pill bad">error</span> ' + escapeHtml(e.message);
  }
  prog.classList.add('hidden');
  $('stopBtn').classList.add('hidden');
  jobRunning = false;
  $('benchBtn').disabled = false; $('checkBtn').disabled = false;
}

// ---------------------------------------------------------------------------
// Remote agent — long-polls the relay, executes jobs, streams results back.
// ---------------------------------------------------------------------------
const agent = { connected: false, sessionId: null, agentKey: null, abortCtrl: null, wakeLock: null };

function setAgentStatus(kind, text) {
  const el = $('agentStatus');
  el.className = 'pill ' + kind;
  el.textContent = text;
}

async function agentPost(path, obj, signal) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj),
    signal: signal,
  });
  if (!r.ok) throw new Error(path + ' → HTTP ' + r.status);
  return r.json();
}

async function agentRegister() {
  const label = $('agentLabel').value.trim() ||
    (adapterSnapshot && (adapterSnapshot.description || adapterSnapshot.architecture)) ||
    'browser';
  let saved = null;
  try { saved = JSON.parse(sessionStorage.getItem('webgpu-session') || 'null'); } catch (e) {}
  const payload = { label: label, adapter: adapterSnapshot, userAgent: navigator.userAgent,
                    pageVersion: PAGE_VERSION };
  if (saved && saved.sessionId && saved.agentKey) {
    payload.resumeSessionId = saved.sessionId;
    payload.agentKey = saved.agentKey;
  }
  const res = await agentPost('/agent/register', payload);
  agent.sessionId = res.sessionId;
  agent.agentKey = res.agentKey;
  try {
    sessionStorage.setItem('webgpu-session',
      JSON.stringify({ sessionId: res.sessionId, agentKey: res.agentKey }));
  } catch (e) {}
  $('agentSession').textContent = res.sessionId + ' (' + label + ')';
  log('Remote control connected. Session ' + res.sessionId + ', label "' + label + '"' +
      (res.resumed ? ' (resumed — queued jobs kept)' : '') + '.');
}

function agentEvent(jobId, line) {
  agentPost('/agent/event', {
    sessionId: agent.sessionId, agentKey: agent.agentKey, jobId: jobId, line: line,
  }).catch(() => {});
}

async function agentPostResult(jobId, status, result) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await agentPost('/agent/result', {
        sessionId: agent.sessionId, agentKey: agent.agentKey,
        jobId: jobId, status: status, result: result,
      });
      return;
    } catch (e) { await sleep(2000); }
  }
  log('Could not deliver result for job ' + jobId + ' after 3 attempts.');
}

async function executeRemote(job) {
  log('Remote job received: ' + job.type + ' (id ' + job.id + ')');
  if (job.type === 'reload') {
    // Works even without a GPU device. Report first, then reload; after the
    // reload the page resumes this same session (sessionStorage) so queued
    // jobs and identity survive.
    await agentPostResult(job.id, 'done', { reloading: true, pageVersion: PAGE_VERSION });
    log('Reloading page now…');
    setTimeout(() => location.reload(), 400);
    return;
  }
  if (!device) {
    await agentPostResult(job.id, 'error', { error: 'no WebGPU device on this page' });
    return;
  }
  if (jobRunning) {
    await agentPostResult(job.id, 'error', { error: 'page busy with another job' });
    return;
  }
  jobRunning = true;
  stopRequested = false;
  eventSink = line => agentEvent(job.id, line);
  try {
    let rec = null;
    if (job.type === 'parity') {
      const n = Array.isArray(job.grid) ? job.grid[0] : (job.grid || 24);
      rec = await coreParity(n, job.steps || 10);
    } else if (job.type === 'bench') {
      const g = job.grid || [232, 232, 61];
      rec = await coreBench(g[0], g[1], g[2], job.steps || 200, job.repeats || 1, null);
    } else if (job.type === 'custom') {
      rec = await coreCustom(job);
    } else if (job.type === 'solver') {
      rec = await coreSolver(job);
    } else {
      throw new Error('unknown job type: ' + job.type);
    }
    await agentPostResult(job.id, 'done', rec);
    log('Remote job ' + job.id + ' done, result sent.');
  } catch (e) {
    log('Remote job ' + job.id + ' failed: ' + e.message);
    await agentPostResult(job.id, 'error', { error: e.message });
  }
  eventSink = null;
  jobRunning = false;
  if (pendingUpdate) {
    log('Reloading for the pending page update…');
    location.reload();
  }
}

async function pollLoop() {
  while (agent.connected) {
    try {
      agent.abortCtrl = new AbortController();
      const timer = setTimeout(() => agent.abortCtrl.abort(), 45000);
      const res = await agentPost('/agent/poll',
        { sessionId: agent.sessionId, agentKey: agent.agentKey }, agent.abortCtrl.signal);
      clearTimeout(timer);
      if (res.job) await executeRemote(res.job);
    } catch (e) {
      if (!agent.connected) break;
      if (String(e.message).includes('401')) {
        // Relay forgot this session (e.g. it restarted): register fresh.
        log('Relay dropped the session — registering a new one…');
        try { sessionStorage.removeItem('webgpu-session'); } catch (e2) {}
        try { await agentRegister(); } catch (e2) { await sleep(5000); }
      } else {
        // Transient failure — most often the browser froze/aborted the poll
        // for a background tab. Keep the SAME session so jobs queued for it
        // are still delivered when the tab wakes up.
        await sleep(3000);
      }
    }
  }
}

async function keepAwake() {
  try {
    agent.wakeLock = await navigator.wakeLock.request('screen');
    agent.wakeLock.addEventListener('release', () => { agent.wakeLock = null; });
  } catch (e) { /* wake lock is best-effort */ }
}

async function connectAgent() {
  if (agent.connected) return;
  try {
    await agentRegister();
  } catch (e) {
    setAgentStatus('bad', 'relay unreachable');
    log('Could not reach the relay: ' + e.message);
    return;
  }
  agent.connected = true;
  setAgentStatus('ok', 'connected');
  $('connectBtn').disabled = true;
  $('disconnectBtn').disabled = false;
  keepAwake();
  pollLoop();
}

function disconnectAgent() {
  agent.connected = false;
  if (agent.abortCtrl) agent.abortCtrl.abort();
  if (agent.wakeLock) { agent.wakeLock.release(); agent.wakeLock = null; }
  try { sessionStorage.removeItem('webgpu-session'); } catch (e) {}
  agent.sessionId = null;
  $('agentSession').textContent = '—';
  setAgentStatus('idle', 'disconnected');
  $('connectBtn').disabled = false;
  $('disconnectBtn').disabled = true;
  log('Remote control disconnected.');
}

// ---------------------------------------------------------------------------
// Self-update — poll version.txt; when the deployed version differs from the
// running PAGE_VERSION, reload (immediately when idle, deferred while a job
// runs). This lets Muse push new page code to every open tab remotely:
// deploy a new index.html + version.txt and tabs pick it up within ~60 s.
// ---------------------------------------------------------------------------
let pendingUpdate = false;
async function checkForUpdate() {
  try {
    const r = await fetch('version.txt', { cache: 'no-store' });
    if (!r.ok) return;
    const v = (await r.text()).trim();
    if (!v || v === PAGE_VERSION) return;
    if (jobRunning) {
      if (!pendingUpdate) log('Page update ' + v + ' available — will reload when the current job finishes.');
      pendingUpdate = true;
    } else {
      log('Page update ' + v + ' available — reloading…');
      location.reload();
    }
  } catch (e) { /* offline or missing version.txt: ignore */ }
}
setInterval(checkForUpdate, 60000);

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
$('refreshBtn').addEventListener('click', refreshAdapter);
$('checkBtn').addEventListener('click', runCheck);
$('benchBtn').addEventListener('click', runBench);
$('stopBtn').addEventListener('click', () => { stopRequested = true; log('Stop requested…'); });
$('clearLogBtn').addEventListener('click', () => { $('log').textContent = ''; });
$('connectBtn').addEventListener('click', connectAgent);
$('disconnectBtn').addEventListener('click', disconnectAgent);
$('agentLabel').value = localStorage.getItem('webgpu-label') || '';
$('agentLabel').addEventListener('change', () => {
  localStorage.setItem('webgpu-label', $('agentLabel').value.trim());
});
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && agent.connected && !agent.wakeLock) keepAwake();
});
$('copyBtn').addEventListener('click', async () => {
  results.exportedAt = new Date().toISOString();
  results.url = location.href;
  try {
    await navigator.clipboard.writeText(JSON.stringify(results, null, 2));
    $('copyNote').textContent = 'Copied — paste it back to Muse.';
  } catch (e) {
    $('copyNote').textContent = 'Clipboard blocked; use Download instead.';
  }
});
$('downloadBtn').addEventListener('click', () => {
  results.exportedAt = new Date().toISOString();
  results.url = location.href;
  const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'webgpu-fdtd-results.json';
  a.click();
  URL.revokeObjectURL(a.href);
});

log('Page loaded. Detecting adapter…');
(async () => { await refreshAdapter(); connectAgent(); })();
