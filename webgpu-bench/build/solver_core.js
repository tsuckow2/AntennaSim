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
function snyderGeom(variant, cfl) {
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
  const dt = F((cfl || 0.2) / F(F(299792458.0) * inv));
  return {
    DX, DY, DZ, NX, NY, NZ, X0, Y0, Z0, EPS0, dt, kGround,
    kPatch: kGround + subLayers, kSrc: kGround + 1,
    postLo: kGround + 2, postHi: kGround + subLayers,
    thickA: kGround, thickB: kGround - 1,
  };
}
const SNY = snyderGeom('', 0.1);
const exIdx = (i, j, k) => (k * SNY.NY + j) * (SNY.NX + 1) + i;
const eyIdx = (i, j, k) => (k * (SNY.NY + 1) + j) * SNY.NX + i;
const ezIdx = (i, j, k) => (k * SNY.NY + j) * SNY.NX + i;
const hxIdx = (i, j, k) => (k * (SNY.NY + 1) + j) * SNY.NX + i;
const hyIdx = (i, j, k) => (k * SNY.NY + j) * (SNY.NX + 1) + i;
const hzIdx = (i, j, k) => (k * (SNY.NY + 1) + j) * (SNY.NX + 1) + i;

function buildSnyderSolver(cfg) {
  const G = snyderGeom(cfg.zref3 ? 'zref3' : cfg.zref2 ? 'zref2' : cfg.zref ? 'zref' : '', cfg.cfl);
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
