/**
 * TypeScript reference H(curl) finite-element solver for the FEM pipeline.
 *
 * This module implements the same single-frequency driven formulation the
 * production native MFEM solver is specified against, so it can act as a
 * validated fallback inside worker-fem while the native `mfem_solver`
 * artifact is unpublished:
 *
 *   curl( (1/mu_r) curl E ) - k0^2 eps_c E = -j omega mu0 J_imp   in the air/dielectric volume
 *   n x E = 0                                                    on PEC facets (attribute 20)
 *   n x curl E + j k0 n x (n x E) = 0                            on absorbing facets (attribute 10)
 *
 * Discretization: lowest-order Nedelec (Whitney-1) edge elements on the
 * conforming tetrahedral mesh produced by buildFEMMesh. Tets inside
 * conductor volumes (conductivity >= FEM_CONDUCTOR_CONDUCTIVITY_S_M) are
 * excluded from the computational domain; their surface facets carry the
 * PEC attribute. Complex permittivity per material:
 *
 *   eps_c = eps_r (1 - j tan(d)) - j sigma / (omega eps0)
 *
 * The lumped port is an impressed electric current of 1 A spread uniformly
 * over the active tets near the terminal gap, directed from the negative to
 * the positive terminal. Port voltage is the line integral of E from the
 * negative to the positive terminal, so Zin = V / 1A and
 * S11 = (Zin - Z0) / (Zin + Z0).
 *
 * Time convention is e^{j omega t} throughout. The first-order Silver-Muller
 * term enters the weak form as +j k0 B, which is the dissipative (absorbing)
 * choice under this convention.
 *
 * Limits: this is a reference implementation, not a performance solver.
 * Meshes above FEM_TS_SOLVER_MAX_FREE_DOFS free edge DOFs are rejected so a
 * caller can fall back to (or require) the native solver instead of hanging
 * the worker.
 */

import {
  FEM_ATTR_CONDUCTOR_PEC,
  FEM_ATTR_EXTERIOR_ABSORBING,
  FEM_CONDUCTOR_CONDUCTIVITY_S_M,
  type FEMMeshArtifact,
} from "./mesh";

const EPS0 = 8.8541878128e-12;
const MU0 = 4 * Math.PI * 1e-7;

/** Local node pairs for the six edges of a tetrahedron, in canonical order. */
const TET_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 2],
  [1, 3],
  [2, 3],
];

/** Upper bound on free edge DOFs handled by this reference solver. */
export const FEM_TS_SOLVER_MAX_FREE_DOFS = 250000;

export type Vec3 = [number, number, number];

export interface RealCSR {
  n: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  values: Float64Array;
}

export interface ComplexCSR {
  n: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  re: Float64Array;
  im: Float64Array;
}

export interface EdgeSystem {
  edgeCount: number;
  activeTetCount: number;
  /** Active-tet ordinal -> index into mesh.tetrahedra / mesh.tetRegions. */
  activeTets: Uint32Array;
  /** Six global edge ids per active tet, in TET_EDGES order. */
  tetEdges: Uint32Array;
  /**
   * Orientation sign per tet edge: +1 when the tet-local edge direction
   * (local node a -> b, a < b) agrees with the global edge direction
   * (lower -> higher global vertex index), -1 otherwise.
   */
  tetEdgeSign: Int8Array;
  /** Global edge id -> free-DOF index, or -1 when PEC-constrained. */
  dofOfEdge: Int32Array;
  freeDofCount: number;
  pecEdgeCount: number;
  conductorTetCount: number;
}

interface RegionMaterial {
  muR: number;
  epsR: number;
  sigma: number;
  tanD: number;
  conductor: boolean;
}

function regionMaterials(mesh: FEMMeshArtifact): Map<number, RegionMaterial> {
  const map = new Map<number, RegionMaterial>();
  for (const entry of mesh.materials) {
    const sigma = Number.isFinite(entry.conductivity_s_m) ? entry.conductivity_s_m : 0;
    map.set(entry.attribute, {
      muR: entry.mu_r > 0 ? entry.mu_r : 1,
      epsR: entry.epsilon_r > 0 ? entry.epsilon_r : 1,
      sigma,
      tanD: Number.isFinite(entry.loss_tangent) ? Math.max(0, entry.loss_tangent) : 0,
      conductor: sigma >= FEM_CONDUCTOR_CONDUCTIVITY_S_M,
    });
  }
  return map;
}

/**
 * Enumerate global edges over the active (non-conductor) tets, orient each
 * edge from the lower to the higher global vertex index, and eliminate the
 * PEC-constrained edges (both endpoints on an attribute-20 facet).
 */
export function buildEdgeSystem(mesh: FEMMeshArtifact): EdgeSystem {
  const vertexCount = mesh.vertices.length / 3;
  const tetCount = mesh.tetrahedra.length / 4;
  if (!Number.isInteger(vertexCount) || !Number.isInteger(tetCount)) {
    throw new Error("FEM mesh vertex/tet arrays have inconsistent lengths");
  }
  const materials = regionMaterials(mesh);
  const air: RegionMaterial = { muR: 1, epsR: 1, sigma: 0, tanD: 0, conductor: false };

  const active: number[] = [];
  let conductorTetCount = 0;
  for (let t = 0; t < tetCount; t += 1) {
    const region = mesh.tetRegions[t] ?? 0;
    const material = materials.get(region) ?? air;
    if (material.conductor) {
      conductorTetCount += 1;
    } else {
      active.push(t);
    }
  }
  if (active.length === 0) throw new Error("FEM mesh has no active (non-conductor) tetrahedra");

  const activeTets = Uint32Array.from(active);
  const activeTetCount = active.length;
  const tetEdges = new Uint32Array(activeTetCount * 6);
  const tetEdgeSign = new Int8Array(activeTetCount * 6);
  const edgeIndex = new Map<number, number>();
  const keyOf = (u: number, v: number): number => (u < v ? u * vertexCount + v : v * vertexCount + u);

  for (let t = 0; t < activeTetCount; t += 1) {
    const tet = activeTets[t] ?? 0;
    for (let e = 0; e < 6; e += 1) {
      const pair = TET_EDGES[e] ?? [0, 1];
      const va = mesh.tetrahedra[tet * 4 + (pair[0] ?? 0)] ?? 0;
      const vb = mesh.tetrahedra[tet * 4 + (pair[1] ?? 1)] ?? 0;
      const key = keyOf(va, vb);
      let edgeId = edgeIndex.get(key);
      if (edgeId === undefined) {
        edgeId = edgeIndex.size;
        edgeIndex.set(key, edgeId);
      }
      tetEdges[t * 6 + e] = edgeId;
      tetEdgeSign[t * 6 + e] = va < vb ? 1 : -1;
    }
  }

  const edgeCount = edgeIndex.size;
  const pecEdge = new Uint8Array(edgeCount);
  const facetCount = mesh.boundaryFacets.length / 3;
  for (let f = 0; f < facetCount; f += 1) {
    if ((mesh.boundaryAttributes[f] ?? 0) !== FEM_ATTR_CONDUCTOR_PEC) continue;
    const u = mesh.boundaryFacets[f * 3] ?? 0;
    const v = mesh.boundaryFacets[f * 3 + 1] ?? 0;
    const w = mesh.boundaryFacets[f * 3 + 2] ?? 0;
    for (const [a, b] of [[u, v], [v, w], [w, u]] as const) {
      const edgeId = edgeIndex.get(keyOf(a, b));
      if (edgeId !== undefined) pecEdge[edgeId] = 1;
    }
  }

  const dofOfEdge = new Int32Array(edgeCount).fill(-1);
  let freeDofCount = 0;
  let pecEdgeCount = 0;
  for (let e = 0; e < edgeCount; e += 1) {
    if (pecEdge[e] === 1) {
      pecEdgeCount += 1;
    } else {
      dofOfEdge[e] = freeDofCount;
      freeDofCount += 1;
    }
  }

  return {
    edgeCount,
    activeTetCount,
    activeTets,
    tetEdges,
    tetEdgeSign,
    dofOfEdge,
    freeDofCount,
    pecEdgeCount,
    conductorTetCount,
  };
}

/* ------------------------------------------------------------------ */
/* Element geometry                                                      */
/* ------------------------------------------------------------------ */

interface TetGeometry {
  volume: number;
  /** Barycentric gradients, 4 x 3. */
  grad: Float64Array;
  centroid: Vec3;
}

function tetGeometry(vertices: Float64Array, tetrahedra: Uint32Array, tet: number): TetGeometry {
  const px = new Float64Array(4);
  const py = new Float64Array(4);
  const pz = new Float64Array(4);
  for (let k = 0; k < 4; k += 1) {
    const vid = tetrahedra[tet * 4 + k] ?? 0;
    px[k] = vertices[vid * 3] ?? 0;
    py[k] = vertices[vid * 3 + 1] ?? 0;
    pz[k] = vertices[vid * 3 + 2] ?? 0;
  }
  // Edge vectors from node 0; columns of the Jacobian.
  const c0x = (px[1] ?? 0) - (px[0] ?? 0);
  const c0y = (py[1] ?? 0) - (py[0] ?? 0);
  const c0z = (pz[1] ?? 0) - (pz[0] ?? 0);
  const c1x = (px[2] ?? 0) - (px[0] ?? 0);
  const c1y = (py[2] ?? 0) - (py[0] ?? 0);
  const c1z = (pz[2] ?? 0) - (pz[0] ?? 0);
  const c2x = (px[3] ?? 0) - (px[0] ?? 0);
  const c2y = (py[3] ?? 0) - (py[0] ?? 0);
  const c2z = (pz[3] ?? 0) - (pz[0] ?? 0);
  // Rows of the inverse Jacobian are (c_{i+1} x c_{i+2}) / det.
  const r0x = (c1y ?? 0) * (c2z ?? 0) - (c1z ?? 0) * (c2y ?? 0);
  const r0y = (c1z ?? 0) * (c2x ?? 0) - (c1x ?? 0) * (c2z ?? 0);
  const r0z = (c1x ?? 0) * (c2y ?? 0) - (c1y ?? 0) * (c2x ?? 0);
  const r1x = (c2y ?? 0) * (c0z ?? 0) - (c2z ?? 0) * (c0y ?? 0);
  const r1y = (c2z ?? 0) * (c0x ?? 0) - (c2x ?? 0) * (c0z ?? 0);
  const r1z = (c2x ?? 0) * (c0y ?? 0) - (c2y ?? 0) * (c0x ?? 0);
  const r2x = (c0y ?? 0) * (c1z ?? 0) - (c0z ?? 0) * (c1y ?? 0);
  const r2y = (c0z ?? 0) * (c1x ?? 0) - (c0x ?? 0) * (c1z ?? 0);
  const r2z = (c0x ?? 0) * (c1y ?? 0) - (c0y ?? 0) * (c1x ?? 0);
  const det = (c0x ?? 0) * (r0x ?? 0) + (c0y ?? 0) * (r0y ?? 0) + (c0z ?? 0) * (r0z ?? 0);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-30) {
    throw new Error(`Degenerate tetrahedron encountered during FEM assembly (tet ${tet})`);
  }
  const grad = new Float64Array(12);
  grad[3] = (r0x ?? 0) / (det ?? 1);
  grad[4] = (r0y ?? 0) / (det ?? 1);
  grad[5] = (r0z ?? 0) / (det ?? 1);
  grad[6] = (r1x ?? 0) / (det ?? 1);
  grad[7] = (r1y ?? 0) / (det ?? 1);
  grad[8] = (r1z ?? 0) / (det ?? 1);
  grad[9] = (r2x ?? 0) / (det ?? 1);
  grad[10] = (r2y ?? 0) / (det ?? 1);
  grad[11] = (r2z ?? 0) / (det ?? 1);
  grad[0] = -(grad[3] ?? 0) - (grad[6] ?? 0) - (grad[9] ?? 0);
  grad[1] = -(grad[4] ?? 0) - (grad[7] ?? 0) - (grad[10] ?? 0);
  grad[2] = -(grad[5] ?? 0) - (grad[8] ?? 0) - (grad[11] ?? 0);
  return {
    volume: Math.abs(det ?? 0) / 6,
    grad,
    centroid: [
      ((px[0] ?? 0) + (px[1] ?? 0) + (px[2] ?? 0) + (px[3] ?? 0)) / 4,
      ((py[0] ?? 0) + (py[1] ?? 0) + (py[2] ?? 0) + (py[3] ?? 0)) / 4,
      ((pz[0] ?? 0) + (pz[1] ?? 0) + (pz[2] ?? 0) + (pz[3] ?? 0)) / 4,
    ],
  };
}

/* ------------------------------------------------------------------ */
/* CSR construction from triplets                                        */
/* ------------------------------------------------------------------ */

function csrFromTriplets(n: number, rows: number[], cols: number[], values: number[]): RealCSR {
  const order = rows.map((_, k) => k).sort((x, y) => {
    const rx = rows[x] ?? 0;
    const ry = rows[y] ?? 0;
    if (rx !== ry) return rx - ry;
    return (cols[x] ?? 0) - (cols[y] ?? 0);
  });
  const rowPtr = new Uint32Array(n + 1);
  const colIdx: number[] = [];
  const vals: number[] = [];
  let prevRow = -1;
  let prevCol = -1;
  const seqRows: number[] = [];
  for (const k of order) {
    const r = rows[k] ?? 0;
    const c = cols[k] ?? 0;
    const v = values[k] ?? 0;
    if (r === prevRow && c === prevCol) {
      vals[vals.length - 1] = (vals[vals.length - 1] ?? 0) + v;
    } else {
      colIdx.push(c);
      vals.push(v);
      seqRows.push(r);
      prevRow = r;
      prevCol = c;
    }
  }
  const rowCounts = new Uint32Array(n);
  for (const r of seqRows) rowCounts[r] = (rowCounts[r] ?? 0) + 1;
  for (let r = 0; r < n; r += 1) {
    rowPtr[r + 1] = (rowPtr[r] ?? 0) + (rowCounts[r] ?? 0);
  }
  return { n, rowPtr, colIdx: Uint32Array.from(colIdx), values: Float64Array.from(vals) };
}

function complexCsrFromTriplets(
  n: number,
  rows: number[],
  cols: number[],
  re: number[],
  im: number[],
): ComplexCSR {
  const order = rows.map((_, k) => k).sort((x, y) => {
    const rx = rows[x] ?? 0;
    const ry = rows[y] ?? 0;
    if (rx !== ry) return rx - ry;
    return (cols[x] ?? 0) - (cols[y] ?? 0);
  });
  const rowPtr = new Uint32Array(n + 1);
  const colIdx: number[] = [];
  const reVals: number[] = [];
  const imVals: number[] = [];
  let prevRow = -1;
  let prevCol = -1;
  const seqRows: number[] = [];
  for (const k of order) {
    const r = rows[k] ?? 0;
    const c = cols[k] ?? 0;
    if (r === prevRow && c === prevCol) {
      reVals[reVals.length - 1] = (reVals[reVals.length - 1] ?? 0) + (re[k] ?? 0);
      imVals[imVals.length - 1] = (imVals[imVals.length - 1] ?? 0) + (im[k] ?? 0);
    } else {
      colIdx.push(c);
      reVals.push(re[k] ?? 0);
      imVals.push(im[k] ?? 0);
      seqRows.push(r);
      prevRow = r;
      prevCol = c;
    }
  }
  const rowCounts = new Uint32Array(n);
  for (const r of seqRows) rowCounts[r] = (rowCounts[r] ?? 0) + 1;
  for (let r = 0; r < n; r += 1) rowPtr[r + 1] = (rowPtr[r] ?? 0) + (rowCounts[r] ?? 0);
  return {
    n,
    rowPtr,
    colIdx: Uint32Array.from(colIdx),
    re: Float64Array.from(reVals),
    im: Float64Array.from(imVals),
  };
}

/* ------------------------------------------------------------------ */
/* Local Nedelec element matrices                                        */
/* ------------------------------------------------------------------ */

interface TetLocalMatrices {
  /** 6x6 curl-curl bracket (1/mu_r applied by caller), row-major. */
  curl: Float64Array;
  /** 6x6 mass bracket: integral of N_i . N_j over the tet, row-major. */
  mass: Float64Array;
}

/**
 * Lowest-order Nedelec basis on a tet: N_e = s_e l_e (lambda_a grad lambda_b
 * - lambda_b grad lambda_a), curl N_e = 2 s_e l_e (grad lambda_a x grad lambda_b).
 * The mass bracket uses the exact barycentric integrals
 *   integral lambda_p lambda_q dV = V/10 (p = q), V/20 (p != q).
 */
function tetLocalMatrices(
  geom: TetGeometry,
  tetrahedra: Uint32Array,
  vertices: Float64Array,
  tet: number,
  signs: Int8Array,
  tetOrdinal: number,
): TetLocalMatrices {
  const curl = new Float64Array(36);
  const mass = new Float64Array(36);
  const grad = geom.grad;
  const volume = geom.volume;
  const curlVec = new Float64Array(18);
  const edgeLen = new Float64Array(6);
  for (let e = 0; e < 6; e += 1) {
    const pair = TET_EDGES[e] ?? [0, 1];
    const a = pair[0] ?? 0;
    const b = pair[1] ?? 0;
    const va = tetrahedra[tet * 4 + a] ?? 0;
    const vb = tetrahedra[tet * 4 + b] ?? 0;
    const dx = (vertices[vb * 3] ?? 0) - (vertices[va * 3] ?? 0);
    const dy = (vertices[vb * 3 + 1] ?? 0) - (vertices[va * 3 + 1] ?? 0);
    const dz = (vertices[vb * 3 + 2] ?? 0) - (vertices[va * 3 + 2] ?? 0);
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    edgeLen[e] = len;
    const gax = grad[a * 3] ?? 0;
    const gay = grad[a * 3 + 1] ?? 0;
    const gaz = grad[a * 3 + 2] ?? 0;
    const gbx = grad[b * 3] ?? 0;
    const gby = grad[b * 3 + 1] ?? 0;
    const gbz = grad[b * 3 + 2] ?? 0;
    // Unsigned curl: 2 l_e (grad_a x grad_b); the orientation sign is applied below.
    curlVec[e * 3] = 2 * len * ((gay ?? 0) * (gbz ?? 0) - (gaz ?? 0) * (gby ?? 0));
    curlVec[e * 3 + 1] = 2 * len * ((gaz ?? 0) * (gbx ?? 0) - (gax ?? 0) * (gbz ?? 0));
    curlVec[e * 3 + 2] = 2 * len * ((gax ?? 0) * (gby ?? 0) - (gay ?? 0) * (gbx ?? 0));
  }
  const baryon = (p: number, q: number): number => (p === q ? volume / 10 : volume / 20);
  for (let i = 0; i < 6; i += 1) {
    const si = signs[tetOrdinal * 6 + i] ?? 1;
    const pairI = TET_EDGES[i] ?? [0, 1];
    const ai = pairI[0] ?? 0;
    const bi = pairI[1] ?? 0;
    for (let j = 0; j < 6; j += 1) {
      const sj = signs[tetOrdinal * 6 + j] ?? 1;
      const pairJ = TET_EDGES[j] ?? [0, 1];
      const aj = pairJ[0] ?? 0;
      const bj = pairJ[1] ?? 0;
      const dot =
        (curlVec[i * 3] ?? 0) * (curlVec[j * 3] ?? 0) +
        (curlVec[i * 3 + 1] ?? 0) * (curlVec[j * 3 + 1] ?? 0) +
        (curlVec[i * 3 + 2] ?? 0) * (curlVec[j * 3 + 2] ?? 0);
      curl[i * 6 + j] = (si * sj * volume * dot) as number;
      const gai = [grad[ai * 3] ?? 0, grad[ai * 3 + 1] ?? 0, grad[ai * 3 + 2] ?? 0] as const;
      const gbi = [grad[bi * 3] ?? 0, grad[bi * 3 + 1] ?? 0, grad[bi * 3 + 2] ?? 0] as const;
      const gaj = [grad[aj * 3] ?? 0, grad[aj * 3 + 1] ?? 0, grad[aj * 3 + 2] ?? 0] as const;
      const gbj = [grad[bj * 3] ?? 0, grad[bj * 3 + 1] ?? 0, grad[bj * 3 + 2] ?? 0] as const;
      const dotBB = (gbi[0] ?? 0) * (gbj[0] ?? 0) + (gbi[1] ?? 0) * (gbj[1] ?? 0) + (gbi[2] ?? 0) * (gbj[2] ?? 0);
      const dotBA = (gbi[0] ?? 0) * (gaj[0] ?? 0) + (gbi[1] ?? 0) * (gaj[1] ?? 0) + (gbi[2] ?? 0) * (gaj[2] ?? 0);
      const dotAB = (gai[0] ?? 0) * (gbj[0] ?? 0) + (gai[1] ?? 0) * (gbj[1] ?? 0) + (gai[2] ?? 0) * (gbj[2] ?? 0);
      const dotAA = (gai[0] ?? 0) * (gaj[0] ?? 0) + (gai[1] ?? 0) * (gaj[1] ?? 0) + (gai[2] ?? 0) * (gaj[2] ?? 0);
      const bracket =
        dotBB * baryon(ai, aj) - dotBA * baryon(ai, bj) - dotAB * baryon(bi, aj) + dotAA * baryon(bi, bj);
      mass[i * 6 + j] = (si * sj * (edgeLen[i] ?? 0) * (edgeLen[j] ?? 0) * bracket) as number;
    }
  }
  return { curl, mass };
}

/* ------------------------------------------------------------------ */
/* Global assembly                                                       */
/* ------------------------------------------------------------------ */

const C0 = 299792458;

export interface HcurlBrackets {
  system: EdgeSystem;
  /** Curl-curl stiffness divided by mu_r: real, PEC-eliminated. */
  stiffness: RealCSR;
  /** Mass bracket (integral of N_i . N_j): real, PEC-eliminated. The caller
   * applies the per-material complex factor -k0^2 eps_c. */
  massBracket: RealCSR;
}

/** Assemble the real curl-curl stiffness and mass brackets over free DOFs. */
export function assembleRealBrackets(mesh: FEMMeshArtifact, system?: EdgeSystem): HcurlBrackets {
  const sys = system ?? buildEdgeSystem(mesh);
  const materials = regionMaterials(mesh);
  const air: RegionMaterial = { muR: 1, epsR: 1, sigma: 0, tanD: 0, conductor: false };
  const n = sys.freeDofCount;
  const rows: number[] = [];
  const cols: number[] = [];
  const kVals: number[] = [];
  const qVals: number[] = [];
  for (let t = 0; t < sys.activeTetCount; t += 1) {
    const tet = sys.activeTets[t] ?? 0;
    const region = mesh.tetRegions[tet] ?? 0;
    const material = materials.get(region) ?? air;
    const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
    const local = tetLocalMatrices(geom, mesh.tetrahedra, mesh.vertices, tet, sys.tetEdgeSign, t);
    const invMu = 1 / material.muR;
    for (let i = 0; i < 6; i += 1) {
      const di = sys.dofOfEdge[sys.tetEdges[t * 6 + i] ?? 0] ?? -1;
      if (di < 0) continue;
      for (let j = 0; j < 6; j += 1) {
        const dj = sys.dofOfEdge[sys.tetEdges[t * 6 + j] ?? 0] ?? -1;
        if (dj < 0) continue;
        rows.push(di);
        cols.push(dj);
        kVals.push(invMu * (local.curl[i * 6 + j] ?? 0));
        qVals.push(local.mass[i * 6 + j] ?? 0);
      }
    }
  }
  return {
    system: sys,
    stiffness: csrFromTriplets(n, rows, cols, kVals),
    massBracket: csrFromTriplets(n, rows, cols, qVals),
  };
}

export interface HcurlAssembly {
  system: EdgeSystem;
  /** Free-DOF system matrix: K/mu_r - k0^2 eps_c Q + j k0 B(abc). */
  matrix: ComplexCSR;
}

function facetKey(u: number, v: number, w: number): string {
  const s = [u, v, w].sort((x, y) => x - y);
  return `${s[0] ?? 0},${s[1] ?? 0},${s[2] ?? 0}`;
}

/**
 * First-order Silver-Muller boundary term: +j k0 B with
 * B_ij = integral_facet (n x N_i) . (n x N_j) dA, evaluated with the
 * 3-point (edge-midpoint) triangle rule, which is exact for the quadratic
 * integrand. The (n x N) product is invariant under flipping n.
 */
function assembleAbcTriplets(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  k0: number,
  rows: number[],
  cols: number[],
  im: number[],
): void {
  const facetToTet = new Map<string, { ordinal: number; localFacet: number }>();
  for (let t = 0; t < sys.activeTetCount; t += 1) {
    const tet = sys.activeTets[t] ?? 0;
    const vids = [0, 1, 2, 3].map((k) => mesh.tetrahedra[tet * 4 + k] ?? 0);
    for (let f = 0; f < 4; f += 1) {
      const nodes = vids.filter((_, k) => k !== f);
      facetToTet.set(facetKey(nodes[0] ?? 0, nodes[1] ?? 0, nodes[2] ?? 0), { ordinal: t, localFacet: f });
    }
  }
  const facetCount = mesh.boundaryFacets.length / 3;
  for (let bf = 0; bf < facetCount; bf += 1) {
    if ((mesh.boundaryAttributes[bf] ?? 0) !== FEM_ATTR_EXTERIOR_ABSORBING) continue;
    const u = mesh.boundaryFacets[bf * 3] ?? 0;
    const v = mesh.boundaryFacets[bf * 3 + 1] ?? 0;
    const w = mesh.boundaryFacets[bf * 3 + 2] ?? 0;
    const hit = facetToTet.get(facetKey(u, v, w));
    if (!hit) continue;
    const tet = sys.activeTets[hit.ordinal] ?? 0;
    const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
    const grad = geom.grad;
    const pux = mesh.vertices[u * 3] ?? 0;
    const puy = mesh.vertices[u * 3 + 1] ?? 0;
    const puz = mesh.vertices[u * 3 + 2] ?? 0;
    const vux = (mesh.vertices[v * 3] ?? 0) - pux;
    const vuy = (mesh.vertices[v * 3 + 1] ?? 0) - puy;
    const vuz = (mesh.vertices[v * 3 + 2] ?? 0) - puz;
    const wux = (mesh.vertices[w * 3] ?? 0) - pux;
    const wuy = (mesh.vertices[w * 3 + 1] ?? 0) - puy;
    const wuz = (mesh.vertices[w * 3 + 2] ?? 0) - puz;
    let nx = vuy * wuz - vuz * wuy;
    let ny = vuz * wux - vux * wuz;
    let nz = vux * wuy - vuy * wux;
    const area2 = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(area2 > 0)) continue;
    const area = area2 / 2;
    nx /= area2;
    ny /= area2;
    nz /= area2;
    const localNodes = [0, 1, 2, 3].filter((k) => k !== hit.localFacet);
    const facetEdges: number[] = [];
    for (let e = 0; e < 6; e += 1) {
      const pair = TET_EDGES[e] ?? [0, 1];
      if (localNodes.includes(pair[0] ?? 0) && localNodes.includes(pair[1] ?? 0)) facetEdges.push(e);
    }
    if (facetEdges.length !== 3) continue;
    // Tangential traces t_i(q) = n x N_i evaluated at the 3 edge midpoints q.
    const traces: number[][] = [];
    for (const qi of facetEdges) {
      const qpair = TET_EDGES[qi] ?? [0, 1];
      const qa = qpair[0] ?? 0;
      const qb = qpair[1] ?? 0;
      const row: number[] = [];
      for (const e of facetEdges) {
        const pair = TET_EDGES[e] ?? [0, 1];
        const a = pair[0] ?? 0;
        const b = pair[1] ?? 0;
        const la = a === qa || a === qb ? 0.5 : 0;
        const lb = b === qa || b === qb ? 0.5 : 0;
        const va = mesh.tetrahedra[tet * 4 + a] ?? 0;
        const vb = mesh.tetrahedra[tet * 4 + b] ?? 0;
        const len = Math.sqrt(
          ((mesh.vertices[vb * 3] ?? 0) - (mesh.vertices[va * 3] ?? 0)) ** 2 +
            ((mesh.vertices[vb * 3 + 1] ?? 0) - (mesh.vertices[va * 3 + 1] ?? 0)) ** 2 +
            ((mesh.vertices[vb * 3 + 2] ?? 0) - (mesh.vertices[va * 3 + 2] ?? 0)) ** 2,
        );
        const s = sys.tetEdgeSign[hit.ordinal * 6 + e] ?? 1;
        const f = s * len;
        const nvecx = f * (la * (grad[b * 3] ?? 0) - lb * (grad[a * 3] ?? 0));
        const nvecy = f * (la * (grad[b * 3 + 1] ?? 0) - lb * (grad[a * 3 + 1] ?? 0));
        const nvecz = f * (la * (grad[b * 3 + 2] ?? 0) - lb * (grad[a * 3 + 2] ?? 0));
        // n x N
        row.push(
          ny * nvecz - nz * nvecy,
          nz * nvecx - nx * nvecz,
          nx * nvecy - ny * nvecx,
        );
      }
      traces.push(row);
    }
    const weight = area / 3;
    for (let ii = 0; ii < 3; ii += 1) {
      const ei = facetEdges[ii] ?? 0;
      const di = sys.dofOfEdge[sys.tetEdges[hit.ordinal * 6 + ei] ?? 0] ?? -1;
      if (di < 0) continue;
      for (let jj = 0; jj < 3; jj += 1) {
        const ej = facetEdges[jj] ?? 0;
        const dj = sys.dofOfEdge[sys.tetEdges[hit.ordinal * 6 + ej] ?? 0] ?? -1;
        if (dj < 0) continue;
        let dot = 0;
        for (const trace of traces) {
          dot +=
            (trace[ii * 3] ?? 0) * (trace[jj * 3] ?? 0) +
            (trace[ii * 3 + 1] ?? 0) * (trace[jj * 3 + 1] ?? 0) +
            (trace[ii * 3 + 2] ?? 0) * (trace[jj * 3 + 2] ?? 0);
        }
        rows.push(di);
        cols.push(dj);
        im.push(k0 * weight * dot);
      }
    }
  }
}

/** Assemble the full complex free-DOF system matrix at a given frequency. */
export function assembleHcurl(
  mesh: FEMMeshArtifact,
  frequencyHz: number,
  system?: EdgeSystem,
): HcurlAssembly {
  if (!Number.isFinite(frequencyHz) || frequencyHz <= 0) {
    throw new Error("frequencyHz must be a positive finite value");
  }
  const sys = system ?? buildEdgeSystem(mesh);
  const materials = regionMaterials(mesh);
  const air: RegionMaterial = { muR: 1, epsR: 1, sigma: 0, tanD: 0, conductor: false };
  const omega = 2 * Math.PI * frequencyHz;
  const k0 = omega / C0;
  const n = sys.freeDofCount;
  const rows: number[] = [];
  const cols: number[] = [];
  const re: number[] = [];
  const im: number[] = [];
  for (let t = 0; t < sys.activeTetCount; t += 1) {
    const tet = sys.activeTets[t] ?? 0;
    const region = mesh.tetRegions[tet] ?? 0;
    const material = materials.get(region) ?? air;
    const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
    const local = tetLocalMatrices(geom, mesh.tetrahedra, mesh.vertices, tet, sys.tetEdgeSign, t);
    // Complex relative permittivity: eps_r (1 - j tan d) - j sigma / (omega eps0).
    const epsRe = material.epsR;
    const epsIm = -material.epsR * material.tanD - material.sigma / (omega * EPS0);
    const massRe = -k0 * k0 * epsRe;
    const massIm = -k0 * k0 * epsIm;
    const invMu = 1 / material.muR;
    for (let i = 0; i < 6; i += 1) {
      const di = sys.dofOfEdge[sys.tetEdges[t * 6 + i] ?? 0] ?? -1;
      if (di < 0) continue;
      for (let j = 0; j < 6; j += 1) {
        const dj = sys.dofOfEdge[sys.tetEdges[t * 6 + j] ?? 0] ?? -1;
        if (dj < 0) continue;
        const kval = invMu * (local.curl[i * 6 + j] ?? 0);
        const qval = local.mass[i * 6 + j] ?? 0;
        rows.push(di);
        cols.push(dj);
        re.push(kval + massRe * qval);
        im.push(massIm * qval);
      }
    }
  }
  // Silver-Muller absorbing term contributes +j k0 B to the imaginary part.
  const abcIm: number[] = [];
  const abcRows: number[] = [];
  const abcCols: number[] = [];
  assembleAbcTriplets(mesh, sys, k0, abcRows, abcCols, abcIm);
  for (let k = 0; k < abcRows.length; k += 1) {
    rows.push(abcRows[k] ?? 0);
    cols.push(abcCols[k] ?? 0);
    re.push(0);
    im.push(abcIm[k] ?? 0);
  }
  return { system: sys, matrix: complexCsrFromTriplets(n, rows, cols, re, im) };
}

/* ------------------------------------------------------------------ */
/* Complex BiCGSTAB with Jacobi preconditioning                          */
/* ------------------------------------------------------------------ */

export interface KrylovResult {
  xRe: Float64Array;
  xIm: Float64Array;
  iterations: number;
  /** Final relative residual norm ||b - A x|| / ||b||. */
  residualNorm: number;
  converged: boolean;
}

export function complexMatVec(
  matrix: ComplexCSR,
  xRe: Float64Array,
  xIm: Float64Array,
  yRe: Float64Array,
  yIm: Float64Array,
): void {
  yRe.fill(0);
  yIm.fill(0);
  for (let r = 0; r < matrix.n; r += 1) {
    let sr = 0;
    let si = 0;
    const end = matrix.rowPtr[r + 1] ?? 0;
    for (let k = matrix.rowPtr[r] ?? 0; k < end; k += 1) {
      const c = matrix.colIdx[k] ?? 0;
      const ar = matrix.re[k] ?? 0;
      const ai = matrix.im[k] ?? 0;
      const xr = xRe[c] ?? 0;
      const xi = xIm[c] ?? 0;
      sr += ar * xr - ai * xi;
      si += ar * xi + ai * xr;
    }
    yRe[r] = sr;
    yIm[r] = si;
  }
}

/**
 * BiCGSTAB for the general complex (non-Hermitian, complex-symmetric) system.
 * Uses the Hermitian inner product <x, y> = sum conj(x_k) y_k.
 */
export function bicgstab(
  matrix: ComplexCSR,
  bRe: Float64Array,
  bIm: Float64Array,
  tol = 1e-8,
  maxIter = 5000,
  preconditioner: "jacobi" | "sgs" = "jacobi",
): KrylovResult {
  const n = matrix.n;
  const xRe = new Float64Array(n);
  const xIm = new Float64Array(n);
  const done = (iterations: number, residualNorm: number, converged: boolean): KrylovResult => ({
    xRe,
    xIm,
    iterations,
    residualNorm,
    converged,
  });
  const norm = (vr: Float64Array, vi: Float64Array): number => {
    let s = 0;
    for (let k = 0; k < n; k += 1) {
      const a = vr[k] ?? 0;
      const b = vi[k] ?? 0;
      s += a * a + b * b;
    }
    return Math.sqrt(s);
  };
  const bNorm = norm(bRe, bIm);
  if (!(bNorm > 0) || !Number.isFinite(bNorm)) return done(0, Number.NaN, false);

  // Jacobi preconditioner: pointwise inverse of the diagonal.
  const invRe = new Float64Array(n);
  const invIm = new Float64Array(n);
  // Complex diagonal (kept for the SGS sweep).
  const diagRe = new Float64Array(n);
  const diagIm = new Float64Array(n);
  for (let r = 0; r < n; r += 1) {
    let dr = 0;
    let di = 0;
    const end = matrix.rowPtr[r + 1] ?? 0;
    for (let k = matrix.rowPtr[r] ?? 0; k < end; k += 1) {
      if ((matrix.colIdx[k] ?? 0) === r) {
        dr = matrix.re[k] ?? 0;
        di = matrix.im[k] ?? 0;
        break;
      }
    }
    diagRe[r] = dr;
    diagIm[r] = di;
    const d = dr * dr + di * di;
    if (d < 1e-300) {
      invRe[r] = 1;
      invIm[r] = 0;
    } else {
      invRe[r] = dr / d;
      invIm[r] = -di / d;
    }
  }
  const yRe = new Float64Array(n);
  const yIm = new Float64Array(n);
  const applyPre = (xr: Float64Array, xi: Float64Array, yr: Float64Array, yi: Float64Array): void => {
    if (preconditioner === "sgs") {
      // One symmetric Gauss-Seidel sweep: M z = r with
      // M = (D + L) D^{-1} (D + U). Forward sweep solves (D + L) y = r,
      // backward sweep solves (D + U) z = D y. CSR rows give both the
      // lower (j < i) and upper (j > i) couplings.
      for (let i = 0; i < n; i += 1) {
        let sr = xr[i] ?? 0;
        let si = xi[i] ?? 0;
        const end = matrix.rowPtr[i + 1] ?? 0;
        for (let k = matrix.rowPtr[i] ?? 0; k < end; k += 1) {
          const j = matrix.colIdx[k] ?? 0;
          if (j < i) {
            const ar = matrix.re[k] ?? 0;
            const ai = matrix.im[k] ?? 0;
            sr -= ar * (yRe[j] ?? 0) - ai * (yIm[j] ?? 0);
            si -= ar * (yIm[j] ?? 0) + ai * (yRe[j] ?? 0);
          }
        }
        const dr = diagRe[i] ?? 0;
        const di = diagIm[i] ?? 0;
        const d = dr * dr + di * di;
        if (d < 1e-300) {
          yRe[i] = sr;
          yIm[i] = si;
        } else {
          yRe[i] = (sr * dr + si * di) / d;
          yIm[i] = (si * dr - sr * di) / d;
        }
      }
      for (let i = n - 1; i >= 0; i -= 1) {
        const dr = diagRe[i] ?? 0;
        const di = diagIm[i] ?? 0;
        let sr = dr * (yRe[i] ?? 0) - di * (yIm[i] ?? 0);
        let si = dr * (yIm[i] ?? 0) + di * (yRe[i] ?? 0);
        const end = matrix.rowPtr[i + 1] ?? 0;
        for (let k = matrix.rowPtr[i] ?? 0; k < end; k += 1) {
          const j = matrix.colIdx[k] ?? 0;
          if (j > i) {
            const ar = matrix.re[k] ?? 0;
            const ai = matrix.im[k] ?? 0;
            sr -= ar * (yr[j] ?? 0) - ai * (yi[j] ?? 0);
            si -= ar * (yi[j] ?? 0) + ai * (yr[j] ?? 0);
          }
        }
        const d = dr * dr + di * di;
        if (d < 1e-300) {
          yr[i] = sr;
          yi[i] = si;
        } else {
          yr[i] = (sr * dr + si * di) / d;
          yi[i] = (si * dr - sr * di) / d;
        }
      }
      return;
    }
    for (let k = 0; k < n; k += 1) {
      const a = xr[k] ?? 0;
      const b = xi[k] ?? 0;
      const c = invRe[k] ?? 0;
      const d = invIm[k] ?? 0;
      yr[k] = a * c - b * d;
      yi[k] = a * d + b * c;
    }
  };
  const dotConj = (
    ar: Float64Array,
    ai: Float64Array,
    br: Float64Array,
    bi: Float64Array,
  ): [number, number] => {
    let sr = 0;
    let si = 0;
    for (let k = 0; k < n; k += 1) {
      const a1 = ar[k] ?? 0;
      const a2 = ai[k] ?? 0;
      const b1 = br[k] ?? 0;
      const b2 = bi[k] ?? 0;
      sr += a1 * b1 + a2 * b2;
      si += a1 * b2 - a2 * b1;
    }
    return [sr, si];
  };
  const cdiv = (ar: number, ai: number, br: number, bi: number): [number, number] => {
    const d = br * br + bi * bi;
    if (d < 1e-300) return [0, 0];
    return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
  };

  const rRe = Float64Array.from(bRe);
  const rIm = Float64Array.from(bIm);
  const rHatRe = Float64Array.from(bRe);
  const rHatIm = Float64Array.from(bIm);
  const pRe = new Float64Array(n);
  const pIm = new Float64Array(n);
  const vRe = new Float64Array(n);
  const vIm = new Float64Array(n);
  const sRe = new Float64Array(n);
  const sIm = new Float64Array(n);
  const tRe = new Float64Array(n);
  const tIm = new Float64Array(n);
  const zRe = new Float64Array(n);
  const zIm = new Float64Array(n);
  const phRe = new Float64Array(n);
  const phIm = new Float64Array(n);

  let rhoRe = 1;
  let rhoIm = 0;
  let rhoPrevRe = 1;
  let rhoPrevIm = 0;
  let alphaRe = 1;
  let alphaIm = 0;
  let omegaRe = 1;
  let omegaIm = 0;

  for (let iter = 1; iter <= maxIter; iter += 1) {
    const [rrRe, rrIm] = dotConj(rHatRe, rHatIm, rRe, rIm);
    rhoRe = rrRe;
    rhoIm = rrIm;
    if (Math.hypot(rhoRe, rhoIm) < 1e-300) return done(iter, norm(rRe, rIm) / bNorm, false);
    const [t1r, t1i] = cdiv(rhoRe, rhoIm, rhoPrevRe, rhoPrevIm);
    const [t2r, t2i] = cdiv(alphaRe, alphaIm, omegaRe, omegaIm);
    const betaRe = t1r * t2r - t1i * t2i;
    const betaIm = t1r * t2i + t1i * t2r;
    for (let k = 0; k < n; k += 1) {
      const ovr = omegaRe * (vRe[k] ?? 0) - omegaIm * (vIm[k] ?? 0);
      const ovi = omegaRe * (vIm[k] ?? 0) + omegaIm * (vRe[k] ?? 0);
      const dr = (pRe[k] ?? 0) - ovr;
      const di = (pIm[k] ?? 0) - ovi;
      pRe[k] = (rRe[k] ?? 0) + betaRe * dr - betaIm * di;
      pIm[k] = (rIm[k] ?? 0) + betaRe * di + betaIm * dr;
    }
    applyPre(pRe, pIm, phRe, phIm);
    complexMatVec(matrix, phRe, phIm, vRe, vIm);
    const [rvr, rvi] = dotConj(rHatRe, rHatIm, vRe, vIm);
    const [alr, ali] = cdiv(rhoRe, rhoIm, rvr, rvi);
    alphaRe = alr;
    alphaIm = ali;
    for (let k = 0; k < n; k += 1) {
      const avr = alphaRe * (vRe[k] ?? 0) - alphaIm * (vIm[k] ?? 0);
      const avi = alphaRe * (vIm[k] ?? 0) + alphaIm * (vRe[k] ?? 0);
      sRe[k] = (rRe[k] ?? 0) - avr;
      sIm[k] = (rIm[k] ?? 0) - avi;
    }
    const sNorm = norm(sRe, sIm) / bNorm;
    if (sNorm < tol && Number.isFinite(sNorm)) {
      for (let k = 0; k < n; k += 1) {
        xRe[k] = (xRe[k] ?? 0) + alphaRe * (phRe[k] ?? 0) - alphaIm * (phIm[k] ?? 0);
        xIm[k] = (xIm[k] ?? 0) + alphaRe * (phIm[k] ?? 0) + alphaIm * (phRe[k] ?? 0);
      }
      return done(iter, sNorm, true);
    }
    applyPre(sRe, sIm, zRe, zIm);
    complexMatVec(matrix, zRe, zIm, tRe, tIm);
    const [tsr, tsi] = dotConj(tRe, tIm, sRe, sIm);
    const [ttr, tti] = dotConj(tRe, tIm, tRe, tIm);
    const [omr, omi] = cdiv(tsr, tsi, ttr, tti);
    omegaRe = omr;
    omegaIm = omi;
    for (let k = 0; k < n; k += 1) {
      const axr = alphaRe * (phRe[k] ?? 0) - alphaIm * (phIm[k] ?? 0);
      const axi = alphaRe * (phIm[k] ?? 0) + alphaIm * (phRe[k] ?? 0);
      const ozr = omegaRe * (zRe[k] ?? 0) - omegaIm * (zIm[k] ?? 0);
      const ozi = omegaRe * (zIm[k] ?? 0) + omegaIm * (zRe[k] ?? 0);
      xRe[k] = (xRe[k] ?? 0) + axr + ozr;
      xIm[k] = (xIm[k] ?? 0) + axi + ozi;
      const otr = omegaRe * (tRe[k] ?? 0) - omegaIm * (tIm[k] ?? 0);
      const oti = omegaRe * (tIm[k] ?? 0) + omegaIm * (tRe[k] ?? 0);
      rRe[k] = (sRe[k] ?? 0) - otr;
      rIm[k] = (sIm[k] ?? 0) - oti;
    }
    const rNorm = norm(rRe, rIm) / bNorm;
    if (rNorm < tol && Number.isFinite(rNorm)) return done(iter, rNorm, true);
    if (Math.hypot(omegaRe, omegaIm) < 1e-300) return done(iter, rNorm, false);
    rhoPrevRe = rhoRe;
    rhoPrevIm = rhoIm;
  }
  return done(maxIter, norm(rRe, rIm) / bNorm, false);
}

/* ------------------------------------------------------------------ */
/* Lumped-port feed                                                      */
/* ------------------------------------------------------------------ */

export interface GapFeed {
  /** Active-tet ordinals carrying the impressed current. */
  gapTets: Uint32Array;
  gapVolume: number;
  /** Unit vector from the negative to the positive terminal. */
  direction: Vec3;
  gapLength: number;
}

function pointSegmentDistanceSq(p: Vec3, a: Vec3, b: Vec3): number {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const abz = b[2] - a[2];
  const len2 = abx * abx + aby * aby + abz * abz;
  let t = 0;
  if (len2 > 0) {
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby + (p[2] - a[2]) * abz) / len2;
    t = Math.min(1, Math.max(0, t));
  }
  const dx = p[0] - (a[0] + t * abx);
  const dy = p[1] - (a[1] + t * aby);
  const dz = p[2] - (a[2] + t * abz);
  return dx * dx + dy * dy + dz * dz;
}

/**
 * Select the active tets around the port terminal gap that carry the
 * impressed-current feed. Starts with a radius of 0.6 gap lengths around the
 * terminal segment and grows it until at least one tet is found.
 */
export function selectGapFeed(
  mesh: FEMMeshArtifact,
  system: EdgeSystem,
  positive: Vec3,
  negative: Vec3,
): GapFeed {
  const gx = positive[0] - negative[0];
  const gy = positive[1] - negative[1];
  const gz = positive[2] - negative[2];
  const gapLength = Math.sqrt(gx * gx + gy * gy + gz * gz);
  if (!Number.isFinite(gapLength) || gapLength <= 0) {
    throw new Error("Lumped-port terminals must be distinct points");
  }
  const direction: Vec3 = [gx / gapLength, gy / gapLength, gz / gapLength];
  let selected: number[] = [];
  let radius = 0.6 * gapLength;
  for (let attempt = 0; attempt < 6 && selected.length === 0; attempt += 1) {
    selected = [];
    const r2 = radius * radius;
    for (let t = 0; t < system.activeTetCount; t += 1) {
      const tet = system.activeTets[t] ?? 0;
      const centroid = tetGeometry(mesh.vertices, mesh.tetrahedra, tet).centroid;
      if (pointSegmentDistanceSq(centroid, negative, positive) <= r2) selected.push(t);
    }
    radius *= 2;
  }
  if (selected.length === 0) {
    // Fall back to the active tet nearest the segment midpoint.
    const mid: Vec3 = [negative[0] + gx / 2, negative[1] + gy / 2, negative[2] + gz / 2];
    let best = 0;
    let bestD = Number.POSITIVE_INFINITY;
    for (let t = 0; t < system.activeTetCount; t += 1) {
      const tet = system.activeTets[t] ?? 0;
      const centroid = tetGeometry(mesh.vertices, mesh.tetrahedra, tet).centroid;
      const dx = centroid[0] - mid[0];
      const dy = centroid[1] - mid[1];
      const dz = centroid[2] - mid[2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = t;
      }
    }
    selected = [best];
  }
  let gapVolume = 0;
  for (const t of selected) {
    const tet = system.activeTets[t] ?? 0;
    gapVolume += tetGeometry(mesh.vertices, mesh.tetrahedra, tet).volume;
  }
  if (!(gapVolume > 0)) throw new Error("Port feed region has no volume");
  return { gapTets: Uint32Array.from(selected), gapVolume, direction, gapLength };
}

/**
 * Right-hand side for a 1 A impressed current spread uniformly over the gap
 * tets: b_i = -j omega mu0 L_gap (d . integral N_i dV) / V_gap.
 * With integral N_e dV = s_e l_e V/4 (grad lambda_b - grad lambda_a) the
 * right-hand side is purely imaginary.
 */
function buildPortRhs(
  mesh: FEMMeshArtifact,
  system: EdgeSystem,
  feed: GapFeed,
  omega: number,
): { re: Float64Array; im: Float64Array } {
  const n = system.freeDofCount;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  // The impressed density J = d_hat / A_cross (1 A through any gap
  // cross-section) with A_cross = V_gap / L_gap for the prismatic gap, so
  // J = L_gap * d_hat / V_gap and the port current is exactly 1 A.
  const coef = ((-omega * MU0) * feed.gapLength) / feed.gapVolume;
  for (const t of feed.gapTets) {
    const tet = system.activeTets[t] ?? 0;
    const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
    const grad = geom.grad;
    for (let e = 0; e < 6; e += 1) {
      const edgeId = system.tetEdges[t * 6 + e] ?? 0;
      const d = system.dofOfEdge[edgeId] ?? -1;
      if (d < 0) continue;
      const s = system.tetEdgeSign[t * 6 + e] ?? 1;
      const pair = TET_EDGES[e] ?? [0, 1];
      const a = pair[0] ?? 0;
      const b = pair[1] ?? 0;
      const va = mesh.tetrahedra[tet * 4 + a] ?? 0;
      const vb = mesh.tetrahedra[tet * 4 + b] ?? 0;
      const dx = (mesh.vertices[vb * 3] ?? 0) - (mesh.vertices[va * 3] ?? 0);
      const dy = (mesh.vertices[vb * 3 + 1] ?? 0) - (mesh.vertices[va * 3 + 1] ?? 0);
      const dz = (mesh.vertices[vb * 3 + 2] ?? 0) - (mesh.vertices[va * 3 + 2] ?? 0);
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const f = (s * len * geom.volume) / 4;
      const ix = f * ((grad[b * 3] ?? 0) - (grad[a * 3] ?? 0));
      const iy = f * ((grad[b * 3 + 1] ?? 0) - (grad[a * 3 + 1] ?? 0));
      const iz = f * ((grad[b * 3 + 2] ?? 0) - (grad[a * 3 + 2] ?? 0));
      im[d] =
        (im[d] ?? 0) +
        coef * ((feed.direction[0] ?? 0) * ix + (feed.direction[1] ?? 0) * iy + (feed.direction[2] ?? 0) * iz);
    }
  }
  return { re, im };
}

interface GapTetShape {
  grad: Float64Array;
  centroid: Vec3;
}

/**
 * Port voltage from the line integral of E along the terminal gap:
 * V = -integral_{neg}^{pos} E . dl, sampled at 8 midpoints.
 */
function portVoltage(
  mesh: FEMMeshArtifact,
  system: EdgeSystem,
  feed: GapFeed,
  negative: Vec3,
  xRe: Float64Array,
  xIm: Float64Array,
): { re: number; im: number } {
  const shapes: GapTetShape[] = [];
  for (const t of feed.gapTets) {
    const tet = system.activeTets[t] ?? 0;
    const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
    shapes.push({ grad: geom.grad, centroid: geom.centroid });
  }
  const locate = (p: Vec3): number => {
    for (let k = 0; k < shapes.length; k += 1) {
      const shape = shapes[k];
      if (!shape) continue;
      let inside = true;
      for (let m = 0; m < 4; m += 1) {
        const lam =
          0.25 +
          (shape.grad[m * 3] ?? 0) * (p[0] - shape.centroid[0]) +
          (shape.grad[m * 3 + 1] ?? 0) * (p[1] - shape.centroid[1]) +
          (shape.grad[m * 3 + 2] ?? 0) * (p[2] - shape.centroid[2]);
        if (lam < -1e-9 || lam > 1 + 1e-9) {
          inside = false;
          break;
        }
      }
      if (inside) return k;
    }
    let best = 0;
    let bestD = Number.POSITIVE_INFINITY;
    for (let k = 0; k < shapes.length; k += 1) {
      const shape = shapes[k];
      if (!shape) continue;
      const dx = shape.centroid[0] - p[0];
      const dy = shape.centroid[1] - p[1];
      const dz = shape.centroid[2] - p[2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    }
    return best;
  };
  const gx = feed.gapLength * (feed.direction[0] ?? 0);
  const gy = feed.gapLength * (feed.direction[1] ?? 0);
  const gz = feed.gapLength * (feed.direction[2] ?? 0);
  let vre = 0;
  let vim = 0;
  const samples = 8;
  for (let k = 0; k < samples; k += 1) {
    const tt = (k + 0.5) / samples;
    const p: Vec3 = [negative[0] + tt * gx, negative[1] + tt * gy, negative[2] + tt * gz];
    const gi = locate(p);
    const shape = shapes[gi];
    if (!shape) continue;
    const tetOrdinal = feed.gapTets[gi] ?? 0;
    const tet = system.activeTets[tetOrdinal] ?? 0;
    const lam = [0, 1, 2, 3].map(
      (m) =>
        0.25 +
        (shape.grad[m * 3] ?? 0) * (p[0] - shape.centroid[0]) +
        (shape.grad[m * 3 + 1] ?? 0) * (p[1] - shape.centroid[1]) +
        (shape.grad[m * 3 + 2] ?? 0) * (p[2] - shape.centroid[2]),
    );
    let erx = 0;
    let ery = 0;
    let erz = 0;
    let eix = 0;
    let eiy = 0;
    let eiz = 0;
    for (let e = 0; e < 6; e += 1) {
      const edgeId = system.tetEdges[tetOrdinal * 6 + e] ?? 0;
      const d = system.dofOfEdge[edgeId] ?? -1;
      if (d < 0) continue;
      const s = system.tetEdgeSign[tetOrdinal * 6 + e] ?? 1;
      const pair = TET_EDGES[e] ?? [0, 1];
      const a = pair[0] ?? 0;
      const b = pair[1] ?? 0;
      const va = mesh.tetrahedra[tet * 4 + a] ?? 0;
      const vb = mesh.tetrahedra[tet * 4 + b] ?? 0;
      const dx = (mesh.vertices[vb * 3] ?? 0) - (mesh.vertices[va * 3] ?? 0);
      const dy = (mesh.vertices[vb * 3 + 1] ?? 0) - (mesh.vertices[va * 3 + 1] ?? 0);
      const dz = (mesh.vertices[vb * 3 + 2] ?? 0) - (mesh.vertices[va * 3 + 2] ?? 0);
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const f = s * len;
      const nvecx = f * ((lam[a] ?? 0) * (shape.grad[b * 3] ?? 0) - (lam[b] ?? 0) * (shape.grad[a * 3] ?? 0));
      const nvecy =
        f * ((lam[a] ?? 0) * (shape.grad[b * 3 + 1] ?? 0) - (lam[b] ?? 0) * (shape.grad[a * 3 + 1] ?? 0));
      const nvecz =
        f * ((lam[a] ?? 0) * (shape.grad[b * 3 + 2] ?? 0) - (lam[b] ?? 0) * (shape.grad[a * 3 + 2] ?? 0));
      const xr = xRe[d] ?? 0;
      const xi = xIm[d] ?? 0;
      erx += xr * nvecx;
      ery += xr * nvecy;
      erz += xr * nvecz;
      eix += xi * nvecx;
      eiy += xi * nvecy;
      eiz += xi * nvecz;
    }
    const dl = feed.gapLength / samples;
    vre -= (erx * (feed.direction[0] ?? 0) + ery * (feed.direction[1] ?? 0) + erz * (feed.direction[2] ?? 0)) * dl;
    vim -= (eix * (feed.direction[0] ?? 0) + eiy * (feed.direction[1] ?? 0) + eiz * (feed.direction[2] ?? 0)) * dl;
  }
  return { re: vre, im: vim };
}

/* ------------------------------------------------------------------ */
/* Top-level driven solve                                                */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Far-field (near-to-far-field transform)                               */
/* ------------------------------------------------------------------ */

/**
 * Options for the near-to-far-field transform. The equivalent electric and
 * magnetic surface currents (J_s = n x H, M_s = E x n) are formed on the
 * absorbing outer boundary and radiated to the far field with the
 * Stratton-Chu integrals under the e^{jwt} convention.
 */
export interface FEMFarFieldOptions {
  /** Polar sampling step in degrees, (0, 30]. Defaults to 5. */
  thetaStepDeg?: number;
  /** Azimuth sampling step in degrees, (0, 60]. Defaults to 5. */
  phiStepDeg?: number;
}

export interface FEMFarFieldResult {
  /** Polar angles in degrees, 0..180. */
  thetaDeg: number[];
  /** Azimuth angles in degrees, 0..<360. */
  phiDeg: number[];
  /** IEEE gain in dBi on the [theta][phi] grid (4*pi*U / P_accepted). */
  gainDbi: number[][];
  /** Directivity in dBi on the [theta][phi] grid (4*pi*U / P_radiated). */
  directivityDbi: number[][];
  maxGainDbi: number;
  maxGainThetaDeg: number;
  maxGainPhiDeg: number;
  maxDirectivityDbi: number;
  radiatedPowerW: number;
  acceptedPowerW: number;
  /** radiatedPowerW / acceptedPowerW. */
  radiationEfficiency: number;
  timingMs: number;
}

const ETA0 = Math.sqrt(MU0 / EPS0);

function muRofTet(mesh: FEMMeshArtifact, tet: number): number {
  const region = mesh.tetRegions[tet] ?? 0;
  for (const entry of mesh.materials) {
    if (entry.attribute === region) return entry.mu_r > 0 ? entry.mu_r : 1;
  }
  return 1;
}

/**
 * Compute the far-field pattern from a solved H(curl) field via equivalent
 * surface currents on the absorbing boundary. Throws on invalid input or
 * degenerate results; callers map failures to a null far-field payload.
 */
export function computeFarField(
  mesh: FEMMeshArtifact,
  system: EdgeSystem,
  xRe: Float64Array,
  xIm: Float64Array,
  frequencyHz: number,
  acceptedPowerW: number,
  options?: FEMFarFieldOptions,
): FEMFarFieldResult {
  const t0 = performance.now();
  if (!(acceptedPowerW > 0) || !Number.isFinite(acceptedPowerW)) {
    throw new Error("acceptedPowerW must be a positive finite value");
  }
  if (!Number.isFinite(frequencyHz) || frequencyHz <= 0) {
    throw new Error("frequencyHz must be a positive finite value");
  }
  const thetaStep = options?.thetaStepDeg ?? 5;
  const phiStep = options?.phiStepDeg ?? 5;
  if (!(thetaStep > 0) || !(phiStep > 0) || thetaStep > 30 || phiStep > 60) {
    throw new Error("far-field sampling steps must lie in (0, 30] x (0, 60] degrees");
  }
  const nTheta = Math.max(2, Math.round(180 / thetaStep) + 1);
  const nPhi = Math.max(2, Math.round(360 / phiStep));
  const omega = 2 * Math.PI * frequencyHz;
  const k0 = omega / C0;

  // Expand the free-DOF solution onto all global edges (PEC edges read zero).
  const edgeRe = new Float64Array(system.edgeCount);
  const edgeIm = new Float64Array(system.edgeCount);
  for (let e = 0; e < system.edgeCount; e += 1) {
    const d = system.dofOfEdge[e] ?? -1;
    if (d >= 0) {
      edgeRe[e] = xRe[d] ?? 0;
      edgeIm[e] = xIm[d] ?? 0;
    }
  }

  // Map each active-tet face to its tet ordinal so boundary facets resolve
  // to the adjacent volume element.
  const facetToTet = new Map<string, number>();
  for (let t = 0; t < system.activeTetCount; t += 1) {
    const tet = system.activeTets[t] ?? 0;
    const v0 = mesh.tetrahedra[tet * 4] ?? 0;
    const v1 = mesh.tetrahedra[tet * 4 + 1] ?? 0;
    const v2 = mesh.tetrahedra[tet * 4 + 2] ?? 0;
    const v3 = mesh.tetrahedra[tet * 4 + 3] ?? 0;
    facetToTet.set(facetKey(v1, v2, v3), t);
    facetToTet.set(facetKey(v0, v2, v3), t);
    facetToTet.set(facetKey(v0, v1, v3), t);
    facetToTet.set(facetKey(v0, v1, v2), t);
  }

  // Huygens surface: area-weighted equivalent currents on the absorbing boundary.
  const srcX: number[] = [];
  const srcY: number[] = [];
  const srcZ: number[] = [];
  const aJre: number[] = [];
  const aJim: number[] = [];
  const aMre: number[] = [];
  const aMim: number[] = [];
  const facetCount = mesh.boundaryFacets.length / 3;
  for (let bf = 0; bf < facetCount; bf += 1) {
    if ((mesh.boundaryAttributes[bf] ?? 0) !== FEM_ATTR_EXTERIOR_ABSORBING) continue;
    const u = mesh.boundaryFacets[bf * 3] ?? 0;
    const v = mesh.boundaryFacets[bf * 3 + 1] ?? 0;
    const w = mesh.boundaryFacets[bf * 3 + 2] ?? 0;
    const ord = facetToTet.get(facetKey(u, v, w));
    if (ord === undefined) continue;
    const tet = system.activeTets[ord] ?? 0;
    const geom = tetGeometry(mesh.vertices, mesh.tetrahedra, tet);
    const grad = geom.grad;
    const pux = mesh.vertices[u * 3] ?? 0;
    const puy = mesh.vertices[u * 3 + 1] ?? 0;
    const puz = mesh.vertices[u * 3 + 2] ?? 0;
    const pvx = mesh.vertices[v * 3] ?? 0;
    const pvy = mesh.vertices[v * 3 + 1] ?? 0;
    const pvz = mesh.vertices[v * 3 + 2] ?? 0;
    const pwx = mesh.vertices[w * 3] ?? 0;
    const pwy = mesh.vertices[w * 3 + 1] ?? 0;
    const pwz = mesh.vertices[w * 3 + 2] ?? 0;
    let nx = (pvy - puy) * (pwz - puz) - (pvz - puz) * (pwy - puy);
    let ny = (pvz - puz) * (pwx - pux) - (pvx - pux) * (pwz - puz);
    let nz = (pvx - pux) * (pwy - puy) - (pvy - puy) * (pwx - pux);
    const area2 = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(area2 > 0)) continue;
    const area = area2 / 2;
    nx /= area2;
    ny /= area2;
    nz /= area2;
    // Force the normal outward (away from the tet interior).
    const ox = (pux + pvx + pwx) / 3 - (geom.centroid[0] ?? 0);
    const oy = (puy + pvy + pwy) / 3 - (geom.centroid[1] ?? 0);
    const oz = (puz + pvz + pwz) / 3 - (geom.centroid[2] ?? 0);
    if (nx * ox + ny * oy + nz * oz < 0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    // Reconstruct E and curl E at the facet centroid (barycentric 1/4).
    // Basis: N_e = s_e * l_e * (lambda_a grad_b - lambda_b grad_a),
    // curl N_e = s_e * 2 * l_e * (grad_a x grad_b).
    let eReX = 0;
    let eReY = 0;
    let eReZ = 0;
    let eImX = 0;
    let eImY = 0;
    let eImZ = 0;
    let cReX = 0;
    let cReY = 0;
    let cReZ = 0;
    let cImX = 0;
    let cImY = 0;
    let cImZ = 0;
    for (let e = 0; e < 6; e += 1) {
      const pair = TET_EDGES[e] ?? [0, 1];
      const a = pair[0] ?? 0;
      const b = pair[1] ?? 0;
      const s = system.tetEdgeSign[ord * 6 + e] ?? 1;
      const ge = system.tetEdges[ord * 6 + e] ?? 0;
      const er = edgeRe[ge] ?? 0;
      const ei = edgeIm[ge] ?? 0;
      if (er === 0 && ei === 0) continue;
      const va = mesh.tetrahedra[tet * 4 + a] ?? 0;
      const vb = mesh.tetrahedra[tet * 4 + b] ?? 0;
      const len = Math.sqrt(
        ((mesh.vertices[vb * 3] ?? 0) - (mesh.vertices[va * 3] ?? 0)) ** 2 +
          ((mesh.vertices[vb * 3 + 1] ?? 0) - (mesh.vertices[va * 3 + 1] ?? 0)) ** 2 +
          ((mesh.vertices[vb * 3 + 2] ?? 0) - (mesh.vertices[va * 3 + 2] ?? 0)) ** 2,
      );
      const gax = grad[a * 3] ?? 0;
      const gay = grad[a * 3 + 1] ?? 0;
      const gaz = grad[a * 3 + 2] ?? 0;
      const gbx = grad[b * 3] ?? 0;
      const gby = grad[b * 3 + 1] ?? 0;
      const gbz = grad[b * 3 + 2] ?? 0;
      const f = (s * len) / 4;
      const nvx = f * (gbx - gax);
      const nvy = f * (gby - gay);
      const nvz = f * (gbz - gaz);
      eReX += er * nvx;
      eReY += er * nvy;
      eReZ += er * nvz;
      eImX += ei * nvx;
      eImY += ei * nvy;
      eImZ += ei * nvz;
      const g = s * 2 * len;
      const cvx = g * (gay * gbz - gaz * gby);
      const cvy = g * (gaz * gbx - gax * gbz);
      const cvz = g * (gax * gby - gay * gbx);
      cReX += er * cvx;
      cReY += er * cvy;
      cReZ += er * cvz;
      cImX += ei * cvx;
      cImY += ei * cvy;
      cImZ += ei * cvz;
    }
    // H = j * curlE / (omega * mu).
    const invWmu = 1 / (omega * MU0 * muRofTet(mesh, tet));
    const hReX = -cImX * invWmu;
    const hReY = -cImY * invWmu;
    const hReZ = -cImZ * invWmu;
    const hImX = cReX * invWmu;
    const hImY = cReY * invWmu;
    const hImZ = cReZ * invWmu;
    // J_s = n x H, M_s = E x n.
    const jReX = ny * hReZ - nz * hReY;
    const jReY = nz * hReX - nx * hReZ;
    const jReZ = nx * hReY - ny * hReX;
    const jImX = ny * hImZ - nz * hImY;
    const jImY = nz * hImX - nx * hImZ;
    const jImZ = nx * hImY - ny * hImX;
    const mReX = eReY * nz - eReZ * ny;
    const mReY = eReZ * nx - eReX * nz;
    const mReZ = eReX * ny - eReY * nx;
    const mImX = eImY * nz - eImZ * ny;
    const mImY = eImZ * nx - eImX * nz;
    const mImZ = eImX * ny - eImY * nx;
    srcX.push((pux + pvx + pwx) / 3);
    srcY.push((puy + pvy + pwy) / 3);
    srcZ.push((puz + pvz + pwz) / 3);
    aJre.push(area * jReX, area * jReY, area * jReZ);
    aJim.push(area * jImX, area * jImY, area * jImZ);
    aMre.push(area * mReX, area * mReY, area * mReZ);
    aMim.push(area * mImX, area * mImY, area * mImZ);
  }
  const srcCount = srcX.length;
  if (srcCount === 0) throw new Error("no absorbing-boundary facets found for the far-field transform");

  // Radiate the equivalent currents: N = sum aJ e^{jk r^.r'}, L = sum aM e^{jk r^.r'}.
  const thetaDeg: number[] = [];
  for (let ti = 0; ti < nTheta; ti += 1) thetaDeg.push((ti * 180) / (nTheta - 1));
  const phiDeg: number[] = [];
  for (let pj = 0; pj < nPhi; pj += 1) phiDeg.push((pj * 360) / nPhi);
  const k0Over4Pi = k0 / (4 * Math.PI);
  const dTheta = Math.PI / (nTheta - 1);
  const dPhi = (2 * Math.PI) / nPhi;
  const uGrid: number[][] = [];
  for (let ti = 0; ti < nTheta; ti += 1) {
    const theta = ((thetaDeg[ti] ?? 0) * Math.PI) / 180;
    const st = Math.sin(theta);
    const ct = Math.cos(theta);
    const uRow: number[] = [];
    for (let pj = 0; pj < nPhi; pj += 1) {
      const phi = ((phiDeg[pj] ?? 0) * Math.PI) / 180;
      const sp = Math.sin(phi);
      const cp = Math.cos(phi);
      const rhx = st * cp;
      const rhy = st * sp;
      const rhz = ct;
      const thx = ct * cp;
      const thy = ct * sp;
      const thz = -st;
      const phx = -sp;
      const phy = cp;
      let nRx = 0;
      let nRy = 0;
      let nRz = 0;
      let nIx = 0;
      let nIy = 0;
      let nIz = 0;
      let lRx = 0;
      let lRy = 0;
      let lRz = 0;
      let lIx = 0;
      let lIy = 0;
      let lIz = 0;
      for (let q = 0; q < srcCount; q += 1) {
        const phase = k0 * (rhx * (srcX[q] ?? 0) + rhy * (srcY[q] ?? 0) + rhz * (srcZ[q] ?? 0));
        const c = Math.cos(phase);
        const s = Math.sin(phase);
        const j3 = q * 3;
        const jr0 = aJre[j3] ?? 0;
        const jr1 = aJre[j3 + 1] ?? 0;
        const jr2 = aJre[j3 + 2] ?? 0;
        const ji0 = aJim[j3] ?? 0;
        const ji1 = aJim[j3 + 1] ?? 0;
        const ji2 = aJim[j3 + 2] ?? 0;
        const mr0 = aMre[j3] ?? 0;
        const mr1 = aMre[j3 + 1] ?? 0;
        const mr2 = aMre[j3 + 2] ?? 0;
        const mi0 = aMim[j3] ?? 0;
        const mi1 = aMim[j3 + 1] ?? 0;
        const mi2 = aMim[j3 + 2] ?? 0;
        nRx += jr0 * c - ji0 * s;
        nRy += jr1 * c - ji1 * s;
        nRz += jr2 * c - ji2 * s;
        nIx += jr0 * s + ji0 * c;
        nIy += jr1 * s + ji1 * c;
        nIz += jr2 * s + ji2 * c;
        lRx += mr0 * c - mi0 * s;
        lRy += mr1 * c - mi1 * s;
        lRz += mr2 * c - mi2 * s;
        lIx += mr0 * s + mi0 * c;
        lIy += mr1 * s + mi1 * c;
        lIz += mr2 * s + mi2 * c;
      }
      // Spherical components of N and L (basis vectors are real).
      const nTr = nRx * thx + nRy * thy + nRz * thz;
      const nTi = nIx * thx + nIy * thy + nIz * thz;
      const nPr = nRx * phx + nRy * phy;
      const nPi = nIx * phx + nIy * phy;
      const lTr = lRx * thx + lRy * thy + lRz * thz;
      const lTi = lIx * thx + lIy * thy + lIz * thz;
      const lPr = lRx * phx + lRy * phy;
      const lPi = lIx * phx + lIy * phy;
      // r-normalized far E-field (Balanis): E_th = -j k0/(4 pi) (eta0 N_th + L_ph),
      // E_ph = -j k0/(4 pi) (eta0 N_ph - L_th). Note the crossed L components.
      const etTr = k0Over4Pi * (ETA0 * nTi + lPi);
      const etTi = -k0Over4Pi * (ETA0 * nTr + lPr);
      const etPr = k0Over4Pi * (ETA0 * nPi - lTi);
      const etPi = -k0Over4Pi * (ETA0 * nPr - lTr);
      const u = (etTr * etTr + etTi * etTi + etPr * etPr + etPi * etPi) / (2 * ETA0);
      uRow.push(u);
    }
    uGrid.push(uRow);
  }

  let radiatedPowerW = 0;
  for (let ti = 0; ti < nTheta; ti += 1) {
    const theta = ((thetaDeg[ti] ?? 0) * Math.PI) / 180;
    const w = Math.sin(theta) * dTheta * dPhi;
    const row = uGrid[ti] ?? [];
    for (let pj = 0; pj < nPhi; pj += 1) radiatedPowerW += (row[pj] ?? 0) * w;
  }
  if (!(radiatedPowerW > 0) || !Number.isFinite(radiatedPowerW)) {
    throw new Error("far-field power integral is non-positive or non-finite");
  }

  const toDbi = (x: number): number => (x > 0 ? 10 * Math.log10(x) : -400);
  const gainDbi: number[][] = [];
  const directivityDbi: number[][] = [];
  let maxGainDbi = -Infinity;
  let maxGainThetaDeg = 0;
  let maxGainPhiDeg = 0;
  let maxDirectivityDbi = -Infinity;
  for (let ti = 0; ti < nTheta; ti += 1) {
    const gRow: number[] = [];
    const dRow: number[] = [];
    const uRow = uGrid[ti] ?? [];
    for (let pj = 0; pj < nPhi; pj += 1) {
      const u = uRow[pj] ?? 0;
      const gDbi = toDbi((4 * Math.PI * u) / acceptedPowerW);
      const dDbi = toDbi((4 * Math.PI * u) / radiatedPowerW);
      if (!Number.isFinite(gDbi) || !Number.isFinite(dDbi)) {
        throw new Error("far-field pattern is non-finite");
      }
      gRow.push(gDbi);
      dRow.push(dDbi);
      if (gDbi > maxGainDbi) {
        maxGainDbi = gDbi;
        maxGainThetaDeg = thetaDeg[ti] ?? 0;
        maxGainPhiDeg = phiDeg[pj] ?? 0;
      }
      if (dDbi > maxDirectivityDbi) maxDirectivityDbi = dDbi;
    }
    gainDbi.push(gRow);
    directivityDbi.push(dRow);
  }
  return {
    thetaDeg,
    phiDeg,
    gainDbi,
    directivityDbi,
    maxGainDbi,
    maxGainThetaDeg,
    maxGainPhiDeg,
    maxDirectivityDbi,
    radiatedPowerW,
    acceptedPowerW,
    radiationEfficiency: radiatedPowerW / acceptedPowerW,
    timingMs: performance.now() - t0,
  };
}

export interface FEMSolverInput {
  mesh: FEMMeshArtifact;
  frequencyHz: number;
  referenceImpedanceOhms: number;
  portPositive: Vec3;
  portNegative: Vec3;
  /** When present, compute the far-field pattern via equivalent surface currents. */
  farField?: FEMFarFieldOptions;
  /**
   * Relative BiCGSTAB convergence tolerance. Defaults to 1e-7 (the validated
   * reference value). Larger models may use 1e-5..1e-6 to trade a small amount
   * of accuracy for much shorter solves; the default is unchanged.
   */
  krylovTolerance?: number;
}

export type FEMSolverStatus = "ok" | "solver_error" | "non_finite";

export interface FEMSolverResult {
  status: FEMSolverStatus;
  message?: string;
  impedanceOhms: { re: number; im: number } | null;
  s11: { re: number; im: number } | null;
  residualNorm: number | null;
  iterationCount: number | null;
  converged: boolean;
  freeDofCount: number;
  /** Far-field pattern; null when not requested, inapplicable, or failed. */
  farField: FEMFarFieldResult | null;
  timingMs: { assembly: number; solve: number; post: number };
}

/**
 * Solve one driven frequency point with the reference H(curl) solver.
 * Never throws: failures are reported through the result status.
 */
export function solveFEMPoint(input: FEMSolverInput): FEMSolverResult {
  const timingMs = { assembly: 0, solve: 0, post: 0 };
  const blank = {
    impedanceOhms: null as { re: number; im: number } | null,
    s11: null as { re: number; im: number } | null,
    residualNorm: null as number | null,
    iterationCount: null as number | null,
    converged: false,
    freeDofCount: 0,
    farField: null as FEMFarFieldResult | null,
    timingMs,
  };
  const fail = (status: FEMSolverStatus, message: string): FEMSolverResult => ({
    ...blank,
    status,
    message,
  });
  try {
    if (!Number.isFinite(input.frequencyHz) || input.frequencyHz <= 0) {
      return fail("solver_error", "frequencyHz must be a positive finite value");
    }
    if (!Number.isFinite(input.referenceImpedanceOhms) || input.referenceImpedanceOhms <= 0) {
      return fail("solver_error", "referenceImpedanceOhms must be a positive finite value");
    }
    const t0 = performance.now();
    const system = buildEdgeSystem(input.mesh);
    if (system.freeDofCount <= 0) {
      return fail("solver_error", "no free edge DOFs remain after PEC elimination");
    }
    if (system.freeDofCount > FEM_TS_SOLVER_MAX_FREE_DOFS) {
      return fail(
        "solver_error",
        `mesh has ${system.freeDofCount} free edge DOFs, above the TypeScript reference solver limit of ${FEM_TS_SOLVER_MAX_FREE_DOFS}`,
      );
    }
    const { matrix } = assembleHcurl(input.mesh, input.frequencyHz, system);
    const t1 = performance.now();
    const omega = 2 * Math.PI * input.frequencyHz;
    const feed = selectGapFeed(input.mesh, system, input.portPositive, input.portNegative);
    const rhs = buildPortRhs(input.mesh, system, feed, omega);
    const krylovTol =
      input.krylovTolerance != null && Number.isFinite(input.krylovTolerance) && input.krylovTolerance > 0
        ? input.krylovTolerance
        : 1e-7;
    // The 908 MHz patch needs ~9.2k iterations at 98k DOFs; cap at 12k for headroom.
    const kry = bicgstab(matrix, rhs.re, rhs.im, krylovTol, 12000, "sgs");
    const t2 = performance.now();
    timingMs.assembly = t1 - t0;
    timingMs.solve = t2 - t1;
    if (!kry.converged) {
      return {
        ...blank,
        status: "solver_error",
        message: `BiCGSTAB did not converge (relative residual ${kry.residualNorm})`,
        freeDofCount: system.freeDofCount,
        residualNorm: Number.isFinite(kry.residualNorm) ? kry.residualNorm : null,
        iterationCount: kry.iterations,
      };
    }
    const v = portVoltage(input.mesh, system, feed, input.portNegative, kry.xRe, kry.xIm);
    // The impressed current is normalized to 1 A, so Zin = V.
    const zRe = v.re;
    const zIm = v.im;
    if (!Number.isFinite(zRe) || !Number.isFinite(zIm)) {
      return {
        ...blank,
        status: "non_finite",
        message: "port impedance is non-finite",
        freeDofCount: system.freeDofCount,
        residualNorm: kry.residualNorm,
        iterationCount: kry.iterations,
      };
    }
    const z0 = input.referenceImpedanceOhms;
    const den = (zRe + z0) * (zRe + z0) + zIm * zIm;
    if (!(den > 0) || !Number.isFinite(den)) {
      return {
        ...blank,
        status: "non_finite",
        message: "S11 denominator is non-positive or non-finite",
        freeDofCount: system.freeDofCount,
        residualNorm: kry.residualNorm,
        iterationCount: kry.iterations,
      };
    }
    const sRe = ((zRe - z0) * (zRe + z0) + zIm * zIm) / den;
    const sIm = (2 * zIm * z0) / den;
    if (!Number.isFinite(sRe) || !Number.isFinite(sIm)) {
      return {
        ...blank,
        status: "non_finite",
        message: "S11 is non-finite",
        freeDofCount: system.freeDofCount,
        residualNorm: kry.residualNorm,
        iterationCount: kry.iterations,
      };
    }
    // Far-field is best-effort: impedance/S11 stay valid when it is skipped.
    let farField: FEMFarFieldResult | null = null;
    if (input.farField !== undefined && zRe > 0) {
      try {
        farField = computeFarField(
          input.mesh,
          system,
          kry.xRe,
          kry.xIm,
          input.frequencyHz,
          0.5 * zRe,
          input.farField,
        );
      } catch {
        farField = null;
      }
    }
    const t3 = performance.now();
    timingMs.post = t3 - t2;
    return {
      ...blank,
      status: "ok",
      freeDofCount: system.freeDofCount,
      impedanceOhms: { re: zRe, im: zIm },
      s11: { re: sRe, im: sIm },
      residualNorm: kry.residualNorm,
      iterationCount: kry.iterations,
      converged: true,
      farField,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return fail("solver_error", message);
  }
}
