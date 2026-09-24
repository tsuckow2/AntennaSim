import { describe, expect, it } from "vitest";
import type { ModelIR } from "../../dsl/model-ir";
import {
  buildFEMMesh,
  FEM_ATTR_CONDUCTOR_PEC,
  FEM_ATTR_EXTERIOR_ABSORBING,
  type FEMMeshArtifact,
} from "../mesh";
import {
  assembleHcurl,
  assembleRealBrackets,
  bicgstab,
  buildEdgeSystem,
  selectGapFeed,
  solveFEMPoint,
  type ComplexCSR,
  type RealCSR,
  type Vec3,
} from "../fem-solver";

// Copper patch on an FR4 substrate, floating in a generated air volume
// (same fixture as the mesh pipeline tests).
function patchIR(): ModelIR {
  return {
    version: 1,
    paramSchemaVersion: 1,
    materials: [
      { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 },
      { name: "fr4", epsilon_r: 4.4, mu_r: 1, loss_tangent: 0.02 },
    ],
    entities: [
      { id: "substrate", kind: "box", center: [0, 0, 0], size: [0.02, 0.02, 0.002], material: "fr4" },
      { id: "patch", kind: "box", center: [0, 0, 0.0015], size: [0.012, 0.012, 0.001], material: "copper" },
    ],
    ports: [
      { id: "feed", kind: "lumpedPort", positive: [0, 0, 0.002], negative: [0.004, 0, 0.002], impedance_ohms: 50 },
    ],
    boundaries: [{ kind: "radiation", padding_m: 0.01 }],
    meshHints: [],
  };
}

// Center-fed copper dipole in free space: two arms with a 2 mm air gap.
// The lumped port bridges the gap, so the impressed current drives a real
// feed gap (unlike a port sitting on top of a PEC patch, which is shorted).
function dipoleIR(): ModelIR {
  return {
    version: 1,
    paramSchemaVersion: 1,
    materials: [{ name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 }],
    entities: [
      { id: "arm1", kind: "box", center: [-0.011, 0, 0], size: [0.02, 0.002, 0.002], material: "copper" },
      { id: "arm2", kind: "box", center: [0.011, 0, 0], size: [0.02, 0.002, 0.002], material: "copper" },
    ],
    ports: [
      { id: "feed", kind: "lumpedPort", positive: [-0.001, 0, 0], negative: [0.001, 0, 0], impedance_ohms: 50 },
    ],
    boundaries: [{ kind: "radiation", padding_m: 0.02 }],
    meshHints: [],
  };
}

const DIPOLE_POS: Vec3 = [-0.001, 0, 0];
const DIPOLE_NEG: Vec3 = [0.001, 0, 0];

function toComplex(real: RealCSR): ComplexCSR {
  return {
    n: real.n,
    rowPtr: real.rowPtr,
    colIdx: real.colIdx,
    re: Float64Array.from(real.values),
    im: new Float64Array(real.values.length),
  };
}

function realMatVec(a: RealCSR, x: Float64Array): Float64Array {
  const y = new Float64Array(a.n);
  for (let r = 0; r < a.n; r += 1) {
    let s = 0;
    const end = a.rowPtr[r + 1] ?? 0;
    for (let k = a.rowPtr[r] ?? 0; k < end; k += 1) s += (a.values[k] ?? 0) * (x[a.colIdx[k] ?? 0] ?? 0);
    y[r] = s;
  }
  return y;
}

function realDot(x: Float64Array, y: Float64Array): number {
  let s = 0;
  for (let k = 0; k < x.length; k += 1) s += (x[k] ?? 0) * (y[k] ?? 0);
  return s;
}

/** Deterministic PRNG for the eigenvalue test start vector. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("FEM H(curl) solver", () => {
  it("builds a deterministic edge system with conductor exclusion and PEC elimination", () => {
    const mesh = buildFEMMesh(patchIR(), 0.005);
    const first = buildEdgeSystem(mesh);
    const second = buildEdgeSystem(mesh);
    expect(first.edgeCount).toBeGreaterThan(0);
    expect(first.freeDofCount).toBeGreaterThan(0);
    expect(first.freeDofCount).toBeLessThan(first.edgeCount);
    expect(first.pecEdgeCount).toBeGreaterThan(0);
    // The copper patch volume is meshed but excluded from the domain.
    expect(first.conductorTetCount).toBeGreaterThan(0);
    expect(first.activeTetCount + first.conductorTetCount).toBe(mesh.tetrahedra.length / 4);
    expect([...first.dofOfEdge]).toEqual([...second.dofOfEdge]);
    expect([...first.tetEdges]).toEqual([...second.tetEdges]);
    // Every free DOF index is dense in [0, freeDofCount).
    const seen = new Set<number>();
    for (const dof of first.dofOfEdge) if (dof >= 0) seen.add(dof);
    expect(seen.size).toBe(first.freeDofCount);
  });

  it("assembles a complex-symmetric, deterministic system matrix", () => {
    const mesh = buildFEMMesh(patchIR(), 0.005);
    const a = assembleHcurl(mesh, 2.4e9).matrix;
    const b = assembleHcurl(mesh, 2.4e9).matrix;
    expect(a.n).toBeGreaterThan(0);
    expect(a.colIdx.length).toBe(a.re.length);
    // Deterministic: identical sparsity and values across assemblies.
    expect([...a.rowPtr]).toEqual([...b.rowPtr]);
    expect([...a.colIdx]).toEqual([...b.colIdx]);
    expect([...a.re]).toEqual([...b.re]);
    expect([...a.im]).toEqual([...b.im]);
    // Complex-symmetric: A[i,j] == A[j,i] for both parts (K, Q symmetric; ABC symmetric).
    const lookup = new Map<number, number>();
    for (let r = 0; r < a.n; r += 1) {
      const end = a.rowPtr[r + 1] ?? 0;
      for (let k = a.rowPtr[r] ?? 0; k < end; k += 1) lookup.set(r * a.n + (a.colIdx[k] ?? 0), k);
    }
    let checked = 0;
    let maxAsym = 0;
    for (let r = 0; r < a.n; r += 1) {
      const end = a.rowPtr[r + 1] ?? 0;
      for (let k = a.rowPtr[r] ?? 0; k < end; k += 1) {
        const c = a.colIdx[k] ?? 0;
        const mirror = lookup.get(c * a.n + r);
        expect(mirror).toBeDefined();
        const kk = mirror ?? 0;
        maxAsym = Math.max(
          maxAsym,
          Math.abs((a.re[k] ?? 0) - (a.re[kk] ?? 0)),
          Math.abs((a.im[k] ?? 0) - (a.im[kk] ?? 0)),
        );
        checked += 1;
      }
    }
    expect(checked).toBeGreaterThan(0);
    expect(maxAsym).toBeLessThan(1e-9);
  });

  it("solves the driven dipole problem with a physical impedance and S11", () => {
    const mesh = buildFEMMesh(dipoleIR(), 0.004);
    const result = solveFEMPoint({
      mesh,
      frequencyHz: 3e9,
      referenceImpedanceOhms: 50,
      portPositive: DIPOLE_POS,
      portNegative: DIPOLE_NEG,
    });
    expect(result.status).toBe("ok");
    expect(result.converged).toBe(true);
    expect(result.residualNorm ?? 1).toBeLessThan(1e-6);
    expect(result.freeDofCount).toBeGreaterThan(0);
    const z = result.impedanceOhms;
    expect(z).not.toBeNull();
    expect(Number.isFinite(z?.re ?? Number.NaN)).toBe(true);
    expect(Number.isFinite(z?.im ?? Number.NaN)).toBe(true);
    // Passive system: Re(Zin) >= 0.
    expect(z?.re ?? -1).toBeGreaterThanOrEqual(0);
    const s11 = result.s11;
    expect(s11).not.toBeNull();
    const mag = Math.hypot(s11?.re ?? 2, s11?.im ?? 0);
    // Passive system: |S11| <= 1.
    expect(mag).toBeLessThanOrEqual(1 + 1e-9);
    // S11 identity: S11 == (Z - Z0) / (Z + Z0).
    const z0 = 50;
    const den = ((z?.re ?? 0) + z0) ** 2 + ((z?.im ?? 0)) ** 2;
    expect(((z?.re ?? 0) - z0) * ((z?.re ?? 0) + z0) + (z?.im ?? 0) ** 2).toBeCloseTo(
      (s11?.re ?? 0) * den,
      6,
    );
    expect(2 * (z?.im ?? 0) * z0).toBeCloseTo((s11?.im ?? 0) * den, 6);
    // Timing fields are populated.
    expect(result.timingMs.assembly).toBeGreaterThanOrEqual(0);
    expect(result.timingMs.solve).toBeGreaterThanOrEqual(0);
    // Physical plausibility for a 0.42-wavelength dipole at 3 GHz: the
    // radiation resistance is tens of ohms and the reactance is small.
    // (Guards against port-current normalization regressions, which
    // previously inflated |Zin| by ~500x.)
    expect(z?.re ?? -1).toBeGreaterThanOrEqual(5);
    expect(z?.re ?? 1e9).toBeLessThanOrEqual(500);
    expect(Math.abs(z?.im ?? 1e9)).toBeLessThanOrEqual(200);
  }, 120000);

  it("computes a physical far-field pattern for the dipole", () => {
    const mesh = buildFEMMesh(dipoleIR(), 0.004);
    const result = solveFEMPoint({
      mesh,
      frequencyHz: 3e9,
      referenceImpedanceOhms: 50,
      portPositive: DIPOLE_POS,
      portNegative: DIPOLE_NEG,
      farField: { thetaStepDeg: 5, phiStepDeg: 5 },
    });
    expect(result.status).toBe("ok");
    const ff = result.farField;
    expect(ff).not.toBeNull();
    if (!ff) return;
    // Grid shape matches the requested sampling.
    expect(ff.thetaDeg.length).toBe(37);
    expect(ff.phiDeg.length).toBe(72);
    expect(ff.gainDbi.length).toBe(37);
    expect(ff.gainDbi[0]?.length).toBe(72);
    // A ~half-wave dipole has ~2.15 dBi directivity; allow solver/ABC margin.
    expect(ff.maxDirectivityDbi).toBeGreaterThanOrEqual(1.5);
    expect(ff.maxDirectivityDbi).toBeLessThanOrEqual(2.9);
    // The max must lie broadside to the x-directed dipole (|r.x| small).
    const thetaR = (ff.maxGainThetaDeg * Math.PI) / 180;
    const phiR = (ff.maxGainPhiDeg * Math.PI) / 180;
    expect(Math.abs(Math.sin(thetaR) * Math.cos(phiR))).toBeLessThan(0.5);
    // Deep null along the dipole axis (+x): theta=90, phi=0.
    const ti = ff.thetaDeg.findIndex((t) => Math.abs(t - 90) < 1e-9);
    const pj = ff.phiDeg.findIndex((p) => Math.abs(p - 0) < 1e-9);
    expect(ff.gainDbi[ti]?.[pj] ?? 0).toBeLessThan(-20);
    // Power balance: efficiency is physical and gain/directivity are consistent.
    expect(ff.radiationEfficiency).toBeGreaterThan(0.5);
    expect(ff.radiationEfficiency).toBeLessThan(1.1);
    expect(ff.maxGainDbi).toBeCloseTo(
      ff.maxDirectivityDbi + 10 * Math.log10(ff.radiationEfficiency),
      2,
    );
    // Accepted power matches 1 A into Re(Zin).
    expect(ff.acceptedPowerW).toBeCloseTo(0.5 * (result.impedanceOhms?.re ?? 0), 9);
  }, 120000);

  it("is bit-deterministic across repeated solves", () => {
    const mesh = buildFEMMesh(dipoleIR(), 0.004);
    const args = {
      mesh,
      frequencyHz: 3e9,
      referenceImpedanceOhms: 50,
      portPositive: DIPOLE_POS,
      portNegative: DIPOLE_NEG,
    };
    const first = solveFEMPoint(args);
    const second = solveFEMPoint(args);
    expect(first.status).toBe("ok");
    expect(second.impedanceOhms?.re).toBe(first.impedanceOhms?.re);
    expect(second.impedanceOhms?.im).toBe(first.impedanceOhms?.im);
    expect(second.s11?.re).toBe(first.s11?.re);
    expect(second.s11?.im).toBe(first.s11?.im);
  }, 120000);

  it("reports solver_error instead of throwing on bad input", () => {
    const mesh = buildFEMMesh(patchIR(), 0.005);
    const badFreq = solveFEMPoint({
      mesh,
      frequencyHz: -100,
      referenceImpedanceOhms: 50,
      portPositive: DIPOLE_POS,
      portNegative: DIPOLE_NEG,
    });
    expect(badFreq.status).toBe("solver_error");
    expect(badFreq.message).toBeTruthy();
    const badPort = solveFEMPoint({
      mesh,
      frequencyHz: 2.4e9,
      referenceImpedanceOhms: 50,
      portPositive: DIPOLE_POS,
      portNegative: DIPOLE_POS,
    });
    expect(badPort.status).toBe("solver_error");
  });

  it("selects gap feed tets near the terminal segment", () => {
    const mesh = buildFEMMesh(dipoleIR(), 0.004);
    const system = buildEdgeSystem(mesh);
    const feed = selectGapFeed(mesh, system, DIPOLE_POS, DIPOLE_NEG);
    expect(feed.gapTets.length).toBeGreaterThan(0);
    expect(feed.gapVolume).toBeGreaterThan(0);
    expect(feed.gapLength).toBeCloseTo(0.002, 9);
    const dirLen = Math.hypot(feed.direction[0] ?? 0, feed.direction[1] ?? 0, feed.direction[2] ?? 0);
    expect(dirLen).toBeCloseTo(1, 12);
  });

  it("reproduces the analytic TE101 resonance of a PEC cavity", () => {
    // Rectangular air cavity, all exterior facets relabeled PEC. The smallest
    // eigenvalue of (K, Q) is the TE101 mode: k^2 = pi^2 ((1/Lx)^2 + (1/Lz)^2)
    // with Lx, Lz taken from the actual meshed domain (entity box plus the
    // radiation padding shell).
    const a = 0.12;
    const b = 0.06;
    const c = 0.09;
    const ir: ModelIR = {
      version: 1,
      paramSchemaVersion: 1,
      materials: [{ name: "air", epsilon_r: 1, mu_r: 1 }],
      entities: [{ id: "cavity", kind: "box", center: [0, 0, 0], size: [a, b, c], material: "air" }],
      ports: [],
      boundaries: [{ kind: "radiation", padding_m: 0.004 }],
      meshHints: [],
    };
    const meshed = buildFEMMesh(ir, 0.015);
    const pecAttributes = meshed.boundaryAttributes.map((attr) =>
      attr === FEM_ATTR_EXTERIOR_ABSORBING ? FEM_ATTR_CONDUCTOR_PEC : attr,
    );
    const mesh: FEMMeshArtifact = { ...meshed, boundaryAttributes: pecAttributes };
    expect(mesh.boundaryAttributes.every((attr) => attr === FEM_ATTR_CONDUCTOR_PEC)).toBe(true);

    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (let k = 0; k < mesh.vertices.length; k += 3) {
      const x = mesh.vertices[k] ?? 0;
      const z = mesh.vertices[k + 2] ?? 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    const Lx = maxX - minX;
    const Lz = maxZ - minZ;
    expect(Lx).toBeCloseTo(a + 0.008, 9);
    expect(Lz).toBeCloseTo(c + 0.008, 9);

    const { stiffness, massBracket } = assembleRealBrackets(mesh);
    const n = stiffness.n;
    expect(n).toBeGreaterThan(100);

    // Shift-invert inverse iteration around 0.9x the expected eigenvalue, so
    // the shift is unambiguously closer to TE101 than to the gradient null
    // space at 0.
    const kAnalytic = Math.PI * Math.sqrt((1 / Lx) ** 2 + (1 / Lz) ** 2);
    const lambdaAnalytic = kAnalytic * kAnalytic;
    const sigma = 0.9 * lambdaAnalytic;
    const shifted = toComplex(stiffness);
    {
      // A_shifted = K - sigma * Q, assembled by axpy on the CSR structures
      // (both share sparsity only if assembled identically; do it densely-safe via triplets).
      const q = massBracket;
      const map = new Map<number, number>();
      for (let r = 0; r < n; r += 1) {
        const end = shifted.rowPtr[r + 1] ?? 0;
        for (let k = shifted.rowPtr[r] ?? 0; k < end; k += 1) map.set(r * n + (shifted.colIdx[k] ?? 0), k);
      }
      for (let r = 0; r < n; r += 1) {
        const end = q.rowPtr[r + 1] ?? 0;
        for (let k = q.rowPtr[r] ?? 0; k < end; k += 1) {
          const kk = map.get(r * n + (q.colIdx[k] ?? 0));
          expect(kk).toBeDefined();
          shifted.re[kk ?? 0] = (shifted.re[kk ?? 0] ?? 0) - sigma * (q.values[k] ?? 0);
        }
      }
    }

    const rand = mulberry32(1234);
    const xInit = new Float64Array(n);
    for (let k = 0; k < n; k += 1) xInit[k] = rand() - 0.5;
    let x: Float64Array = xInit;
    const normalizeQ = (v: Float64Array): Float64Array => {
      const qv = realMatVec(massBracket, v);
      const s = Math.sqrt(Math.max(realDot(v, qv), 1e-300));
      const out = new Float64Array(v.length);
      for (let k = 0; k < v.length; k += 1) out[k] = (v[k] ?? 0) / s;
      return out;
    };
    x = normalizeQ(x);
    const zero = new Float64Array(n);
    for (let iter = 0; iter < 20; iter += 1) {
      const qx = realMatVec(massBracket, x);
      const sol = bicgstab(shifted, qx, zero, 1e-10, 5000, "sgs");
      expect(sol.converged).toBe(true);
      x = normalizeQ(sol.xRe);
    }
    const kx = realMatVec(stiffness, x);
    const qx = realMatVec(massBracket, x);
    const lambda = realDot(x, kx) / realDot(x, qx);
    expect(lambda).toBeGreaterThan(0);
    const relativeError = Math.abs(Math.sqrt(lambda) - kAnalytic) / kAnalytic;
    expect(relativeError).toBeLessThan(0.25);
  }, 120000);

  it("keeps the default 908 MHz patch example within the reference solver's reach", async () => {
    // Regression: a 5 mm patch refine hint doubled the free DOFs (98k -> 179k)
    // and pushed BiCGSTAB past convergence. The example must stay small enough
    // for the SGS-preconditioned solver to converge within the iteration cap.
    const { readFileSync } = await import("node:fs");
    const { executeParametricScript } = await import("../../dsl/executor");
    const script = readFileSync(
      new URL("../../dsl/examples/patch_908_directional.js", import.meta.url),
      "utf8",
    );
    const exec = await executeParametricScript(script);
    expect(exec.diagnostics).toHaveLength(0);
    const port = exec.ir!.ports[0]!;
    const mesh = buildFEMMesh(exec.ir!, 0.015, {
      id: port.id as string,
      positive: port.positive as Vec3,
      negative: port.negative as Vec3,
    });
    const system = buildEdgeSystem(mesh);
    // 98k DOFs converges in ~9.2k iterations; 179k did not converge at all.
    // Keep a comfortable margin under the 250k hard cap.
    expect(system.freeDofCount).toBeGreaterThan(0);
    expect(system.freeDofCount).toBeLessThan(120000);
  }, 120000);
});
