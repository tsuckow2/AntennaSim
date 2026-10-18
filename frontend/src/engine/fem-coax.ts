/**
 * Coaxial wave port — TypeScript mirror of `wasm/hcurl/src/coax.rs`:
 * geometry-found port faces, TEM absorbing term, incident-wave drive,
 * and V/I impedance extraction.
 *
 * The driven problem replaces the lumped gap feed with the physical
 * feed structure: a (square) coaxial section below the ground plane,
 * modeled in the mesh, whose port plane carries
 *   * an absorbing condition — the Silver–Müller tangential-mass term
 *     with `k -> k_d = k0 * sqrt(eps_diel)`, exact for the coax TEM mode
 *     because its E/H ratio is the local wave impedance
 *     `eta_d = eta0 / sqrt(eps_diel)` pointwise on any cross-section; and
 *   * an incident TEM wave (round-coax `1/r` profile; the square
 *     section's non-TEM launch residue is evanescent and dies within a
 *     cross-section width of the port plane, and the extraction below is
 *     a field ratio, so the launch amplitude/profile only sets the
 *     overall scale).
 *
 * Impedance is extracted at the reference plane `z_ref` inside the
 * dielectric: `V = -int E . dl` radially from outer wall to pin and
 * `I = oint H . dl` around the pin with `H = curl E / (-j omega mu0)`
 * (the `e^{+j omega t}` convention), so `Zin = V / I` needs no analytic
 * coax impedance and no de-embedding rotation.
 */

import type { FEMMeshArtifact } from "./mesh";
import { TET_EDGES, tetGeometry, type EdgeSystem, type Vec3 } from "./fem-solver";

const C0 = 299792458;
const MU0 = 4 * Math.PI * 1e-7;
const EPS0 = 8.8541878128e-12;
const ETA0 = Math.sqrt(MU0 / EPS0);

/** Coax port specification (the request metadata `coax_port` object). */
export interface CoaxPortSpec {
  /** Coax axis position `[x, y]` in meters. */
  center: [number, number];
  /** Port plane z (drive + absorbing condition). */
  zPort: number;
  /** Reference plane z for the V/I extraction (inside the dielectric). */
  zRef: number;
  /** Inner conductor half-width (square) in meters. */
  halfPin: number;
  /** Dielectric outer half-width (square) in meters. */
  halfDiel: number;
  /** Dielectric relative permittivity (TEM wave impedance eta0/sqrt(eps_r)). */
  epsDiel: number;
  /**
   * Optional second absorbing-only plane (calibration fixtures: a
   * matched termination at the far end of a coax section).
   */
  zPortTop?: number;
}

/** One port face: active-tet ordinal + local facet index, and whether it
 * lies on the top (absorbing-only) plane. */
export interface PortFace {
  ord: number;
  localFacet: number;
  top: boolean;
}

export interface FacetMass {
  /** Tangential-mass entries `(di, dj, B_ij)` over the facet's free DOFs. */
  entries: Array<readonly [number, number, number]>;
  /** Outward unit normal. */
  normal: Vec3;
  area: number;
  /** Edge midpoints at the 3 quadrature points, one per facet edge. */
  midpoints: [Vec3, Vec3, Vec3];
  /** Tangential traces `n x N_i` at the quadrature points: 3 x 9, row-major. */
  traces: number[][];
  /** Free-DOF index per facet edge, or -1 when PEC-constrained. */
  dofs: Int32Array;
}

function vertexCoords(mesh: FEMMeshArtifact, v: number): Vec3 {
  return [mesh.vertices[v * 3] ?? 0, mesh.vertices[v * 3 + 1] ?? 0, mesh.vertices[v * 3 + 2] ?? 0];
}

/**
 * Tangential facet mass for one tet facet — mirror of
 * `facet_tangential_mass` in `wasm/hcurl/src/abc.rs`:
 * `B_ij = int_facet (n x N_i) . (n x N_j) dA` with the 3-point
 * edge-midpoint rule (exact for the quadratic integrand), plus the
 * outward normal, quadrature midpoints, and tangential traces the coax
 * incident-wave RHS needs. Returns null for a degenerate facet or a
 * facet without exactly 3 edges.
 */
export function facetTangentialMass(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  ord: number,
  localFacet: number,
): FacetMass | null {
  const tet = sys.activeTets[ord] ?? 0;
  const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
  const v = [0, 1, 2, 3].map((k) => mesh.tetrahedra[tet * 4 + k] ?? 0);
  const localNodes = [0, 1, 2, 3].filter((k) => k !== localFacet);
  const pu = vertexCoords(mesh, v[localNodes[0] ?? 0] ?? 0);
  const pv = vertexCoords(mesh, v[localNodes[1] ?? 0] ?? 0);
  const pw = vertexCoords(mesh, v[localNodes[2] ?? 0] ?? 0);
  const opp = vertexCoords(mesh, v[localFacet] ?? 0);
  const vu: Vec3 = [pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]];
  const wu: Vec3 = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]];
  let nx = vu[1] * wu[2] - vu[2] * wu[1];
  let ny = vu[2] * wu[0] - vu[0] * wu[2];
  let nz = vu[0] * wu[1] - vu[1] * wu[0];
  const area2 = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (!(area2 > 0)) return null;
  const area = area2 / 2;
  nx /= area2;
  ny /= area2;
  nz /= area2;
  // Orient outward: away from the tet's opposite vertex.
  if (nx * (opp[0] - pu[0]) + ny * (opp[1] - pu[1]) + nz * (opp[2] - pu[2]) > 0) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }
  const normal: Vec3 = [nx, ny, nz];

  const facetEdges: number[] = [];
  for (let e = 0; e < 6; e += 1) {
    const pair = TET_EDGES[e] ?? [0, 1];
    if (localNodes.includes(pair[0] ?? 0) && localNodes.includes(pair[1] ?? 0)) facetEdges.push(e);
  }
  if (facetEdges.length !== 3) return null;

  const grad = geom.grad;
  const edgeLen = (e: number): number => {
    const pair = TET_EDGES[e] ?? [0, 1];
    const pa = vertexCoords(mesh, v[pair[0] ?? 0] ?? 0);
    const pb = vertexCoords(mesh, v[pair[1] ?? 0] ?? 0);
    return Math.sqrt((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2 + (pb[2] - pa[2]) ** 2);
  };

  const dofs = new Int32Array(3).fill(-1);
  facetEdges.forEach((e, ei) => {
    dofs[ei] = sys.dofOfEdge[sys.tetEdges[ord * 6 + e] ?? 0] ?? -1;
  });
  const midpoints = facetEdges.map((qe) => {
    const pair = TET_EDGES[qe] ?? [0, 1];
    const pa = vertexCoords(mesh, v[pair[0] ?? 0] ?? 0);
    const pb = vertexCoords(mesh, v[pair[1] ?? 0] ?? 0);
    return [
      0.5 * (pa[0] + pb[0]),
      0.5 * (pa[1] + pb[1]),
      0.5 * (pa[2] + pb[2]),
    ] as Vec3;
  }) as [Vec3, Vec3, Vec3];

  const traces: number[][] = [];
  for (const qe of facetEdges) {
    const qpair = TET_EDGES[qe] ?? [0, 1];
    const qa = qpair[0] ?? 0;
    const qb = qpair[1] ?? 0;
    const row: number[] = [];
    for (const e of facetEdges) {
      const pair = TET_EDGES[e] ?? [0, 1];
      const a = pair[0] ?? 0;
      const b = pair[1] ?? 0;
      const la = a === qa || a === qb ? 0.5 : 0;
      const lb = b === qa || b === qb ? 0.5 : 0;
      const f = (sys.tetEdgeSign[ord * 6 + e] ?? 1) * edgeLen(e);
      const nvecx = f * (la * (grad[b * 3] ?? 0) - lb * (grad[a * 3] ?? 0));
      const nvecy = f * (la * (grad[b * 3 + 1] ?? 0) - lb * (grad[a * 3 + 1] ?? 0));
      const nvecz = f * (la * (grad[b * 3 + 2] ?? 0) - lb * (grad[a * 3 + 2] ?? 0));
      // n x N
      row.push(ny * nvecz - nz * nvecy, nz * nvecx - nx * nvecz, nx * nvecy - ny * nvecx);
    }
    traces.push(row);
  }

  const weight = area / 3;
  const entries: Array<readonly [number, number, number]> = [];
  for (let ii = 0; ii < 3; ii += 1) {
    const di = dofs[ii] ?? -1;
    if (di < 0) continue;
    for (let jj = 0; jj < 3; jj += 1) {
      const dj = dofs[jj] ?? -1;
      if (dj < 0) continue;
      let dot = 0;
      for (let q = 0; q < 3; q += 1) {
        const trace = traces[q] ?? [];
        dot +=
          (trace[ii * 3] ?? 0) * (trace[jj * 3] ?? 0) +
          (trace[ii * 3 + 1] ?? 0) * (trace[jj * 3 + 1] ?? 0) +
          (trace[ii * 3 + 2] ?? 0) * (trace[jj * 3 + 2] ?? 0);
      }
      entries.push([di, dj, weight * dot]);
    }
  }
  return { entries, normal, area, midpoints, traces, dofs };
}

/**
 * Find the port faces geometrically: boundary faces (shared by exactly
 * one active tet) lying in a port plane, with centroid inside the
 * dielectric annulus, backed by a dielectric tet of the port's eps_r.
 */
export function findPortFaces(mesh: FEMMeshArtifact, sys: EdgeSystem, cp: CoaxPortSpec): PortFace[] {
  const keyOf = (a: number, b: number, c: number): string => {
    const s = [a, b, c].sort((x, y) => x - y);
    return `${s[0] ?? 0},${s[1] ?? 0},${s[2] ?? 0}`;
  };
  const useMap = new Map<string, { count: number; ord: number; localFacet: number }>();
  for (let ord = 0; ord < sys.activeTetCount; ord += 1) {
    const tet = sys.activeTets[ord] ?? 0;
    const vids = [0, 1, 2, 3].map((k) => mesh.tetrahedra[tet * 4 + k] ?? 0);
    for (let f = 0; f < 4; f += 1) {
      const nodes = vids.filter((_, k) => k !== f);
      const key = keyOf(nodes[0] ?? 0, nodes[1] ?? 0, nodes[2] ?? 0);
      const entry = useMap.get(key);
      if (entry) {
        entry.count += 1;
      } else {
        useMap.set(key, { count: 1, ord, localFacet: f });
      }
    }
  }

  const epsByRegion = new Map<number, number>();
  for (const entry of mesh.materials) {
    epsByRegion.set(entry.attribute, entry.epsilon_r > 0 ? entry.epsilon_r : 1);
  }

  const Z_TOL = 1e-7;
  const faces: PortFace[] = [];
  for (const { count, ord, localFacet } of useMap.values()) {
    if (count !== 1) continue;
    const tet = sys.activeTets[ord] ?? 0;
    const vids = [0, 1, 2, 3].map((k) => mesh.tetrahedra[tet * 4 + k] ?? 0);
    const nodes = vids.filter((_, k) => k !== localFacet);
    const pts = nodes.map((n) => vertexCoords(mesh, n));
    const z = pts[0]?.[2] ?? 0;
    if (pts.some((p) => Math.abs(p[2] - z) > Z_TOL)) continue;
    let top: boolean;
    if (Math.abs(z - cp.zPort) <= Z_TOL) {
      top = false;
    } else if (cp.zPortTop !== undefined && Math.abs(z - cp.zPortTop) <= Z_TOL) {
      top = true;
    } else {
      continue;
    }
    const cenX = ((pts[0]?.[0] ?? 0) + (pts[1]?.[0] ?? 0) + (pts[2]?.[0] ?? 0)) / 3;
    const cenY = ((pts[0]?.[1] ?? 0) + (pts[1]?.[1] ?? 0) + (pts[2]?.[1] ?? 0)) / 3;
    const cheb = Math.max(Math.abs(cenX - cp.center[0]), Math.abs(cenY - cp.center[1]));
    if (cheb < cp.halfPin - 1e-6 || cheb > cp.halfDiel + 1e-6) continue;
    // The backing tet must be the port dielectric.
    const eps = epsByRegion.get(mesh.tetRegions[tet] ?? 0) ?? 1;
    if (Math.abs(eps - cp.epsDiel) > 0.02 * Math.max(cp.epsDiel, 1)) continue;
    faces.push({ ord, localFacet, top });
  }
  return faces;
}

/** Absorbing triplets for all port faces: `(di, dj, j * k_d * B_ij)`,
 * returned as parallel row/col/im arrays (real parts are zero). */
export function assemblePortTriplets(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  cp: CoaxPortSpec,
  faces: readonly PortFace[],
  frequencyHz: number,
): { rows: number[]; cols: number[]; im: number[] } {
  const kD = ((2 * Math.PI * frequencyHz) / C0) * Math.sqrt(cp.epsDiel);
  const rows: number[] = [];
  const cols: number[] = [];
  const im: number[] = [];
  for (const face of faces) {
    const mass = facetTangentialMass(mesh, sys, face.ord, face.localFacet);
    if (!mass) continue;
    for (const [di, dj, bij] of mass.entries) {
      rows.push(di);
      cols.push(dj);
      im.push(kD * bij);
    }
  }
  return { rows, cols, im };
}

/**
 * Incident-wave RHS on the (bottom) port faces:
 * `b_i += 2 j k_d int (n x N_i) . (n x E_inc) dA`, evaluated with the
 * same 3-point edge-midpoint rule as the mass term. `E_inc` is the
 * round-coax TEM profile `E0 (a/r) r_hat` with `E0 = eta_d / (2 pi a)`
 * (incident current ~1 A around the pin; the exact amplitude is
 * immaterial — the extraction is a ratio of total fields).
 */
export function buildCoaxRhs(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  cp: CoaxPortSpec,
  faces: readonly PortFace[],
  frequencyHz: number,
): { re: Float64Array; im: Float64Array } {
  const kD = ((2 * Math.PI * frequencyHz) / C0) * Math.sqrt(cp.epsDiel);
  const etaD = ETA0 / Math.sqrt(cp.epsDiel);
  const e0 = etaD / (2 * Math.PI * cp.halfPin);
  const coefIm = 2 * kD; // the drive coefficient 2 j k_d, split into parts
  const re = new Float64Array(sys.freeDofCount);
  const im = new Float64Array(sys.freeDofCount);
  for (const face of faces) {
    if (face.top) continue;
    const mass = facetTangentialMass(mesh, sys, face.ord, face.localFacet);
    if (!mass) continue;
    const n = mass.normal;
    for (let ei = 0; ei < 3; ei += 1) {
      const d = mass.dofs[ei] ?? -1;
      if (d < 0) continue;
      let integ = 0;
      for (let q = 0; q < 3; q += 1) {
        const p = mass.midpoints[q] ?? [0, 0, 0];
        const rvx = p[0] - cp.center[0];
        const rvy = p[1] - cp.center[1];
        const r = Math.max(Math.sqrt(rvx * rvx + rvy * rvy), 0.5 * cp.halfPin);
        const eInc: Vec3 = [
          e0 * (cp.halfPin / r) * (rvx / r),
          e0 * (cp.halfPin / r) * (rvy / r),
          0,
        ];
        // s = n x E_inc
        const sx = n[1] * eInc[2] - n[2] * eInc[1];
        const sy = n[2] * eInc[0] - n[0] * eInc[2];
        const sz = n[0] * eInc[1] - n[1] * eInc[0];
        const trace = mass.traces[q] ?? [];
        integ +=
          (trace[ei * 3] ?? 0) * sx + (trace[ei * 3 + 1] ?? 0) * sy + (trace[ei * 3 + 2] ?? 0) * sz;
      }
      im[d] = (im[d] ?? 0) + coefIm * (mass.area / 3) * integ;
    }
  }
  return { re, im };
}

/* ------------------------------------------------------------------ */
/* V/I extraction at the reference plane                                 */
/* ------------------------------------------------------------------ */

interface CVec3 {
  re: Vec3;
  im: Vec3;
}

function lambdaAt(
  geom: ReturnType<typeof tetGeometry>,
  m: number,
  p: Vec3,
): number {
  const g = geom.grad;
  const c = geom.centroid;
  return (
    0.25 +
    (g[m * 3] ?? 0) * (p[0] - c[0]) +
    (g[m * 3 + 1] ?? 0) * (p[1] - c[1]) +
    (g[m * 3 + 2] ?? 0) * (p[2] - c[2])
  );
}

/** Locate the active tet containing `p` among `candidates` (active-tet
 * ordinals), falling back to the nearest candidate centroid. */
function locateTet(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  candidates: readonly number[],
  p: Vec3,
): number {
  let best = -1;
  let bestD = Infinity;
  for (const ord of candidates) {
    const tet = sys.activeTets[ord] ?? 0;
    const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
    let inside = true;
    for (let m = 0; m < 4; m += 1) {
      const lam = lambdaAt(geom, m, p);
      if (lam < -1e-9 || lam > 1 + 1e-9) {
        inside = false;
        break;
      }
    }
    if (inside) return ord;
    const c = geom.centroid;
    const d = (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2;
    if (d < bestD) {
      bestD = d;
      best = ord;
    }
  }
  if (best < 0) {
    throw new Error("coax extraction: no candidate tet near the reference plane");
  }
  return best;
}

function tetEdgeLength(mesh: FEMMeshArtifact, tet: number, e: number): number {
  const pair = TET_EDGES[e] ?? [0, 1];
  const pa = vertexCoords(mesh, mesh.tetrahedra[tet * 4 + (pair[0] ?? 0)] ?? 0);
  const pb = vertexCoords(mesh, mesh.tetrahedra[tet * 4 + (pair[1] ?? 1)] ?? 0);
  return Math.sqrt((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2 + (pb[2] - pa[2]) ** 2);
}

/** Electric field at `p` (in the tet located among `candidates`). */
function evalE(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  candidates: readonly number[],
  xRe: Float64Array,
  xIm: Float64Array,
  p: Vec3,
): CVec3 {
  const ord = locateTet(mesh, sys, candidates, p);
  const tet = sys.activeTets[ord] ?? 0;
  const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
  const lambda = [0, 1, 2, 3].map((m) => lambdaAt(geom, m, p));
  const out: CVec3 = { re: [0, 0, 0], im: [0, 0, 0] };
  for (let edge = 0; edge < 6; edge += 1) {
    const d = sys.dofOfEdge[sys.tetEdges[ord * 6 + edge] ?? 0] ?? -1;
    if (d < 0) continue;
    const pair = TET_EDGES[edge] ?? [0, 1];
    const a = pair[0] ?? 0;
    const b = pair[1] ?? 0;
    const f = (sys.tetEdgeSign[ord * 6 + edge] ?? 1) * tetEdgeLength(mesh, tet, edge);
    const la = lambda[a] ?? 0;
    const lb = lambda[b] ?? 0;
    const grad = geom.grad;
    const xr = xRe[d] ?? 0;
    const xi = xIm[d] ?? 0;
    for (let c = 0; c < 3; c += 1) {
      const nvec = f * (la * (grad[b * 3 + c] ?? 0) - lb * (grad[a * 3 + c] ?? 0));
      out.re[c] = (out.re[c] ?? 0) + xr * nvec;
      out.im[c] = (out.im[c] ?? 0) + xi * nvec;
    }
  }
  return out;
}

/** `curl E` at `p` (constant per tet). */
function evalCurlE(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  candidates: readonly number[],
  xRe: Float64Array,
  xIm: Float64Array,
  p: Vec3,
): CVec3 {
  const ord = locateTet(mesh, sys, candidates, p);
  const tet = sys.activeTets[ord] ?? 0;
  const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
  const out: CVec3 = { re: [0, 0, 0], im: [0, 0, 0] };
  for (let edge = 0; edge < 6; edge += 1) {
    const d = sys.dofOfEdge[sys.tetEdges[ord * 6 + edge] ?? 0] ?? -1;
    if (d < 0) continue;
    const pair = TET_EDGES[edge] ?? [0, 1];
    const a = pair[0] ?? 0;
    const b = pair[1] ?? 0;
    const grad = geom.grad;
    const gax = grad[a * 3] ?? 0;
    const gay = grad[a * 3 + 1] ?? 0;
    const gaz = grad[a * 3 + 2] ?? 0;
    const gbx = grad[b * 3] ?? 0;
    const gby = grad[b * 3 + 1] ?? 0;
    const gbz = grad[b * 3 + 2] ?? 0;
    const len2 = 2 * tetEdgeLength(mesh, tet, edge);
    const cvx = len2 * (gay * gbz - gaz * gby);
    const cvy = len2 * (gaz * gbx - gax * gbz);
    const cvz = len2 * (gax * gby - gay * gbx);
    const s = sys.tetEdgeSign[ord * 6 + edge] ?? 1;
    const xr = (xRe[d] ?? 0) * s;
    const xi = (xIm[d] ?? 0) * s;
    out.re[0] = (out.re[0] ?? 0) + xr * cvx;
    out.re[1] = (out.re[1] ?? 0) + xr * cvy;
    out.re[2] = (out.re[2] ?? 0) + xr * cvz;
    out.im[0] = (out.im[0] ?? 0) + xi * cvx;
    out.im[1] = (out.im[1] ?? 0) + xi * cvy;
    out.im[2] = (out.im[2] ?? 0) + xi * cvz;
  }
  return out;
}

/**
 * Voltage and current at the reference plane.
 * `V = -int E . dl` from outer wall (negative) to pin (positive) —
 * accumulated as `+int E . dl` along pin->wall, the same thing;
 * `I = oint H . dl` around the pin, CCW viewed from +z.
 */
export function coaxVI(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  cp: CoaxPortSpec,
  xRe: Float64Array,
  xIm: Float64Array,
  frequencyHz: number,
): { v: { re: number; im: number }; i: { re: number; im: number } } {
  const omega = 2 * Math.PI * frequencyHz;
  // Candidate tets around the reference plane.
  const candidates: number[] = [];
  for (let ord = 0; ord < sys.activeTetCount; ord += 1) {
    const tet = sys.activeTets[ord] ?? 0;
    const c = tetGeometry(mesh.vertices, mesh.tetrahedra, tet).centroid;
    const cheb = Math.max(Math.abs(c[0] - cp.center[0]), Math.abs(c[1] - cp.center[1]));
    if (Math.abs(c[2] - cp.zRef) < 3e-3 && cheb < cp.halfDiel + 2e-3) {
      candidates.push(ord);
    }
  }
  if (candidates.length === 0) {
    throw new Error("coax extraction: no tets around the reference plane");
  }

  // V: radial line integral along +x at the reference plane.
  const nv = 24;
  let vRe = 0;
  let vIm = 0;
  const dr = (cp.halfDiel - cp.halfPin) / nv;
  for (let k = 0; k < nv; k += 1) {
    const r = cp.halfPin + (k + 0.5) * dr;
    const p: Vec3 = [cp.center[0] + r, cp.center[1], cp.zRef];
    const e = evalE(mesh, sys, candidates, xRe, xIm, p);
    vRe += (e.re[0] ?? 0) * dr;
    vIm += (e.im[0] ?? 0) * dr;
  }

  // I: square loop at half-width s, CCW viewed from +z.
  const s = 0.5 * (cp.halfPin + cp.halfDiel);
  const hGain = 1 / (omega * MU0); // H = curl E * j / (omega mu0)
  const ni = 16;
  const dl = (2 * s) / ni;
  const sides: ReadonlyArray<readonly [readonly [number, number], readonly [number, number]]> = [
    [[s, -s], [0, 1]], // +x side, +y direction
    [[s, s], [-1, 0]], // +y side, -x direction
    [[-s, s], [0, -1]], // -x side, -y direction
    [[-s, -s], [1, 0]], // -y side, +x direction
  ];
  let iRe = 0;
  let iIm = 0;
  for (const [corner, tHat] of sides) {
    for (let k = 0; k < ni; k += 1) {
      const sweep = -s + (k + 0.5) * dl;
      const px =
        tHat[1] !== 0 ? cp.center[0] + (corner[0] ?? 0) : cp.center[0] + (tHat[0] ?? 0) * sweep;
      const py =
        tHat[1] !== 0 ? cp.center[1] + (tHat[1] ?? 0) * sweep : cp.center[1] + (corner[1] ?? 0);
      const curl = evalCurlE(mesh, sys, candidates, xRe, xIm, [px, py, cp.zRef]);
      // h = curl * (j * hGain): re = -hGain * curl.im, im = hGain * curl.re.
      const hxRe = -hGain * (curl.im[0] ?? 0);
      const hxIm = hGain * (curl.re[0] ?? 0);
      const hyRe = -hGain * (curl.im[1] ?? 0);
      const hyIm = hGain * (curl.re[1] ?? 0);
      iRe += (hxRe * (tHat[0] ?? 0) + hyRe * (tHat[1] ?? 0)) * dl;
      iIm += (hxIm * (tHat[0] ?? 0) + hyIm * (tHat[1] ?? 0)) * dl;
    }
  }
  if (Math.hypot(iRe, iIm) === 0) {
    throw new Error("coax extraction: loop current is zero");
  }
  return { v: { re: vRe, im: vIm }, i: { re: iRe, im: iIm } };
}

/**
 * Input impedance at the reference plane: `Zin = V / I` with
 * `V = -int E . dl` radially wall->pin (along +x) and `I = oint H . dl`
 * around the pin on a square loop at half-width `(a + b) / 2`,
 * `H = curl E / (-j omega mu0)`.
 */
export function coaxImpedance(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  cp: CoaxPortSpec,
  xRe: Float64Array,
  xIm: Float64Array,
  frequencyHz: number,
): { re: number; im: number } {
  const { v, i } = coaxVI(mesh, sys, cp, xRe, xIm, frequencyHz);
  const den = i.re * i.re + i.im * i.im;
  return {
    re: (v.re * i.re + v.im * i.im) / den,
    im: (v.im * i.re - v.re * i.im) / den,
  };
}
