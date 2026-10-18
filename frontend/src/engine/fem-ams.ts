/**
 * Auxiliary-space Maxwell (AMS) preconditioner for the H(curl) system —
 * TypeScript mirror of `wasm/hcurl/src/ams.rs`.
 *
 * Pointwise smoothers (Jacobi, SGS) are blind to the large near-nullspace
 * of the discrete curl-curl operator — discrete gradients — which is why
 * BiCGSTAB needs ~10^4 iterations on production meshes. AMS
 * (Hiptmair–Xu) adds a coarse correction posed on the nodal (H1)
 * auxiliary space:
 *
 *   M^-1 r = S^-1 r + G (G^T A G)^-1 G^T r
 *
 * where `S^-1` is one SGS sweep on the edge system (applied by the
 * caller, in `bicgstab`), `G` is the discrete gradient (free-edge-DOFs x
 * active nodes): the edge-DOF of grad(phi) on the low->high edge (u,v)
 * is the length-normalized line integral `(phi_v - phi_u)/l_e`, matching
 * the length-scaled Nedelec basis. The nodal space is the full H1
 * auxiliary space (no Dirichlet condition on PEC): the nullspace of the
 * constrained curl-curl operator is gradients of H1 functions constant
 * on each PEC component, and the full nodal space captures
 * inter-component modes (e.g. the differential potential between the two
 * dipole arms) that a PEC-Dirichlet coarse space would miss. The inner
 * nodal `A_n = G^T A G` is Laplacian-like — the curl-curl part vanishes
 * identically (`G^T K G = 0`, the discrete de Rham sequence) — so the
 * same SGS smoother works well on it and the inner solve is cheap.
 *
 * The inner nodal approximation is a fixed number (3) of SGS sweeps on
 * the residual, exactly as in the Rust default: far cheaper than an
 * inner Krylov solve, and a fixed linear operator (better for the outer
 * Krylov method).
 */

import type { FEMMeshArtifact } from "./mesh";
import { TET_EDGES, type ComplexCSR, type EdgeSystem } from "./fem-solver";

/** Fixed SGS sweeps for the inner nodal approximation (Rust INNER_SWEEPS). */
const INNER_SWEEPS = 3;
/** Diagonal magnitude floor, shared with the Rust krylov module. */
const TINY = 1e-300;

export interface AmsPreconditioner {
  /** Number of nodal auxiliary DOFs. */
  nodeCount: number;
  /** Nonzeros of the nodal auxiliary matrix (diagnostics). */
  nodalNnz: number;
  /**
   * Additive coarse correction: `z += G (A_n^-1 (G^T r))`.
   * The caller applies the SGS smoother into `z` first; this adds the
   * gradient-space correction on top, using the *original* residual `r`.
   */
  applyCorrection: (
    rRe: Float64Array,
    rIm: Float64Array,
    zRe: Float64Array,
    zIm: Float64Array,
  ) => void;
}

/**
 * One symmetric Gauss-Seidel sweep `M z = x` with
 * `M = (D + L) D^-1 (D + U)` — the same smoother as `bicgstab`'s SGS
 * branch and the Rust `PreconditionerData::sgs_apply`. `y` is scratch.
 */
function sgsApply(
  matrix: ComplexCSR,
  diagRe: Float64Array,
  diagIm: Float64Array,
  xRe: Float64Array,
  xIm: Float64Array,
  outRe: Float64Array,
  outIm: Float64Array,
  yRe: Float64Array,
  yIm: Float64Array,
): void {
  const n = matrix.n;
  // Forward: (D + L) y = x.
  for (let i = 0; i < n; i += 1) {
    let sr = xRe[i] ?? 0;
    let si = xIm[i] ?? 0;
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
    if (d < TINY) {
      yRe[i] = sr;
      yIm[i] = si;
    } else {
      yRe[i] = (sr * dr + si * di) / d;
      yIm[i] = (si * dr - sr * di) / d;
    }
  }
  // Backward: (D + U) z = D y.
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
        sr -= ar * (outRe[j] ?? 0) - ai * (outIm[j] ?? 0);
        si -= ar * (outIm[j] ?? 0) + ai * (outRe[j] ?? 0);
      }
    }
    const d = dr * dr + di * di;
    if (d < TINY) {
      outRe[i] = sr;
      outIm[i] = si;
    } else {
      outRe[i] = (sr * dr + si * di) / d;
      outIm[i] = (si * dr - sr * di) / d;
    }
  }
}

/** Diagonal of a complex CSR matrix, split into re/im arrays. */
function csrDiagonal(matrix: ComplexCSR): { diagRe: Float64Array; diagIm: Float64Array } {
  const diagRe = new Float64Array(matrix.n);
  const diagIm = new Float64Array(matrix.n);
  for (let r = 0; r < matrix.n; r += 1) {
    const end = matrix.rowPtr[r + 1] ?? 0;
    for (let k = matrix.rowPtr[r] ?? 0; k < end; k += 1) {
      if ((matrix.colIdx[k] ?? 0) === r) {
        diagRe[r] = matrix.re[k] ?? 0;
        diagIm[r] = matrix.im[k] ?? 0;
        break;
      }
    }
  }
  return { diagRe, diagIm };
}

/**
 * Build the discrete gradient and the nodal auxiliary matrix for the
 * assembled system `matrix` (the `n_free x n_free` H(curl) matrix for
 * `sys`). Mirrors `AmsPreconditioner::build`, including the row-by-row
 * `G^T A G` assembly (a single triplet list would hold ~40x the final
 * nnz and OOM-killed the Rust build at 1.5M DOFs).
 */
export function buildAmsPreconditioner(
  mesh: FEMMeshArtifact,
  sys: EdgeSystem,
  matrix: ComplexCSR,
): AmsPreconditioner {
  const nFree = sys.freeDofCount;
  if (matrix.n !== nFree) {
    throw new Error("AMS needs the free-DOF system matrix");
  }
  const vertexCount = mesh.vertices.length / 3;

  // Edge id -> (u, v) with u < v (global low->high orientation).
  const edgeU = new Int32Array(sys.edgeCount);
  const edgeV = new Int32Array(sys.edgeCount);
  const seen = new Uint8Array(sys.edgeCount);
  for (let t = 0; t < sys.activeTetCount; t += 1) {
    const tet = sys.activeTets[t] ?? 0;
    for (let e = 0; e < 6; e += 1) {
      const id = sys.tetEdges[t * 6 + e] ?? 0;
      if (seen[id] === 1) continue;
      seen[id] = 1;
      const pair = TET_EDGES[e] ?? [0, 1];
      const va = mesh.tetrahedra[tet * 4 + (pair[0] ?? 0)] ?? 0;
      const vb = mesh.tetrahedra[tet * 4 + (pair[1] ?? 1)] ?? 0;
      edgeU[id] = Math.min(va, vb);
      edgeV[id] = Math.max(va, vb);
    }
  }

  // Edge lengths (physical), for the length-normalized discrete gradient.
  const edgeLen = new Float64Array(sys.edgeCount);
  for (let id = 0; id < sys.edgeCount; id += 1) {
    const u = edgeU[id] ?? 0;
    const v = edgeV[id] ?? 0;
    const dx = (mesh.vertices[v * 3] ?? 0) - (mesh.vertices[u * 3] ?? 0);
    const dy = (mesh.vertices[v * 3 + 1] ?? 0) - (mesh.vertices[u * 3 + 1] ?? 0);
    const dz = (mesh.vertices[v * 3 + 2] ?? 0) - (mesh.vertices[u * 3 + 2] ?? 0);
    edgeLen[id] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  // Free-DOF -> edge id (inverse of sys.dofOfEdge).
  const edgeOfDof = new Int32Array(nFree);
  for (let id = 0; id < sys.edgeCount; id += 1) {
    const d = sys.dofOfEdge[id] ?? -1;
    if (d >= 0) edgeOfDof[d] = id;
  }

  // Active nodal DOFs: incident to >= 1 free edge. Inactive nodes would
  // produce zero rows/columns in A_n (singular inner system). PEC-boundary
  // nodes are *included* (no Dirichlet condition) — see the module doc.
  const nodeActive = new Uint8Array(vertexCount);
  for (let id = 0; id < sys.edgeCount; id += 1) {
    if ((sys.dofOfEdge[id] ?? -1) >= 0) {
      nodeActive[edgeU[id] ?? 0] = 1;
      nodeActive[edgeV[id] ?? 0] = 1;
    }
  }
  const nodeDof = new Int32Array(vertexCount).fill(-1);
  let nodeCount = 0;
  for (let i = 0; i < vertexCount; i += 1) {
    if (nodeActive[i] === 1) {
      nodeDof[i] = nodeCount;
      nodeCount += 1;
    }
  }

  // G per free edge DOF: endpoints (as nodal DOFs) and 1/l_e.
  // G[e, u] = -1/l_e, G[e, v] = +1/l_e.
  const dofNodeU = new Int32Array(nFree);
  const dofNodeV = new Int32Array(nFree);
  const dofInvLen = new Float64Array(nFree);
  for (let id = 0; id < sys.edgeCount; id += 1) {
    const d = sys.dofOfEdge[id] ?? -1;
    if (d < 0) continue;
    dofNodeU[d] = nodeDof[edgeU[id] ?? 0] ?? -1;
    dofNodeV[d] = nodeDof[edgeV[id] ?? 0] ?? -1;
    dofInvLen[d] = 1 / (edgeLen[id] ?? 1);
  }

  // Active node (by vertex) -> incident (free edge DOF, G entry = ±1/l_e).
  const nodeEdges: Array<Array<readonly [number, number]>> = Array.from(
    { length: vertexCount },
    () => [],
  );
  for (let id = 0; id < sys.edgeCount; id += 1) {
    const d = sys.dofOfEdge[id] ?? -1;
    if (d < 0) continue;
    const invL = 1 / (edgeLen[id] ?? 1);
    nodeEdges[edgeU[id] ?? 0]?.push([d, -invL]);
    nodeEdges[edgeV[id] ?? 0]?.push([d, invL]);
  }

  // A_n[i][j] = sum over incident edges e and A-row entries (f, aval) of
  // G[e,i] * A[e,f] * G[f,j], assembled row by row (see the build doc).
  const aRowPtr = new Uint32Array(nodeCount + 1);
  const aColIdx: number[] = [];
  const aRe: number[] = [];
  const aIm: number[] = [];
  for (let vtx = 0; vtx < vertexCount; vtx += 1) {
    const i = nodeDof[vtx] ?? -1;
    if (i < 0) continue;
    const rowMap = new Map<number, [number, number]>();
    for (const [e, se] of nodeEdges[vtx] ?? []) {
      const end = matrix.rowPtr[e + 1] ?? 0;
      for (let k = matrix.rowPtr[e] ?? 0; k < end; k += 1) {
        const f = matrix.colIdx[k] ?? 0;
        const fId = edgeOfDof[f] ?? 0;
        const ju = nodeDof[edgeU[fId] ?? 0] ?? -1;
        const jv = nodeDof[edgeV[fId] ?? 0] ?? -1;
        // c = A[e,f] * G[e,i] * (1/l_f); G[f,ju] = -1/l_f, G[f,jv] = +1/l_f.
        const scale = se * (dofInvLen[f] ?? 0);
        const cr = (matrix.re[k] ?? 0) * scale;
        const ci = (matrix.im[k] ?? 0) * scale;
        const accU = rowMap.get(ju);
        if (accU) {
          accU[0] -= cr;
          accU[1] -= ci;
        } else {
          rowMap.set(ju, [-cr, -ci]);
        }
        const accV = rowMap.get(jv);
        if (accV) {
          accV[0] += cr;
          accV[1] += ci;
        } else {
          rowMap.set(jv, [cr, ci]);
        }
      }
    }
    const cols = [...rowMap.keys()].sort((a, b) => a - b);
    for (const c of cols) {
      const val = rowMap.get(c) ?? [0, 0];
      // CsrMatrix::from_rows drops explicit zeros after summation.
      if (val[0] === 0 && val[1] === 0) continue;
      aColIdx.push(c);
      aRe.push(val[0]);
      aIm.push(val[1]);
    }
    aRowPtr[i + 1] = aColIdx.length;
  }
  const aNodal: ComplexCSR = {
    n: nodeCount,
    rowPtr: aRowPtr,
    colIdx: Uint32Array.from(aColIdx),
    re: Float64Array.from(aRe),
    im: Float64Array.from(aIm),
  };
  const { diagRe: nodalDiagRe, diagIm: nodalDiagIm } = csrDiagonal(aNodal);

  // Work arrays for applyCorrection / the inner nodal solve.
  const cnRe = new Float64Array(nodeCount);
  const cnIm = new Float64Array(nodeCount);
  const yRe = new Float64Array(nodeCount);
  const yIm = new Float64Array(nodeCount);
  const resRe = new Float64Array(nodeCount);
  const resIm = new Float64Array(nodeCount);
  const delRe = new Float64Array(nodeCount);
  const delIm = new Float64Array(nodeCount);
  const ayRe = new Float64Array(nodeCount);
  const ayIm = new Float64Array(nodeCount);
  const sgsYRe = new Float64Array(nodeCount);
  const sgsYIm = new Float64Array(nodeCount);

  /** Inner nodal approximation `y ~= A_n^-1 cn`: fixed SGS sweeps on the residual. */
  const solveNodal = (): void => {
    yRe.fill(0);
    yIm.fill(0);
    for (let sweep = 0; sweep < INNER_SWEEPS; sweep += 1) {
      // residual = cn - A_n y
      ayRe.fill(0);
      ayIm.fill(0);
      for (let r = 0; r < nodeCount; r += 1) {
        let sr = 0;
        let si = 0;
        const end = aNodal.rowPtr[r + 1] ?? 0;
        for (let k = aNodal.rowPtr[r] ?? 0; k < end; k += 1) {
          const c = aNodal.colIdx[k] ?? 0;
          const ar = aNodal.re[k] ?? 0;
          const ai = aNodal.im[k] ?? 0;
          const xr = yRe[c] ?? 0;
          const xi = yIm[c] ?? 0;
          sr += ar * xr - ai * xi;
          si += ar * xi + ai * xr;
        }
        ayRe[r] = sr;
        ayIm[r] = si;
      }
      for (let k = 0; k < nodeCount; k += 1) {
        resRe[k] = (cnRe[k] ?? 0) - (ayRe[k] ?? 0);
        resIm[k] = (cnIm[k] ?? 0) - (ayIm[k] ?? 0);
      }
      sgsApply(aNodal, nodalDiagRe, nodalDiagIm, resRe, resIm, delRe, delIm, sgsYRe, sgsYIm);
      for (let k = 0; k < nodeCount; k += 1) {
        yRe[k] = (yRe[k] ?? 0) + (delRe[k] ?? 0);
        yIm[k] = (yIm[k] ?? 0) + (delIm[k] ?? 0);
      }
    }
  };

  const applyCorrection = (
    rRe: Float64Array,
    rIm: Float64Array,
    zRe: Float64Array,
    zIm: Float64Array,
  ): void => {
    // cn = G^T r (nodes <- edges).
    cnRe.fill(0);
    cnIm.fill(0);
    for (let e = 0; e < nFree; e += 1) {
      const rr = rRe[e] ?? 0;
      const ri = rIm[e] ?? 0;
      if (rr === 0 && ri === 0) continue;
      const invL = dofInvLen[e] ?? 0;
      const u = dofNodeU[e] ?? 0;
      const v = dofNodeV[e] ?? 0;
      cnRe[u] = (cnRe[u] ?? 0) - invL * rr;
      cnIm[u] = (cnIm[u] ?? 0) - invL * ri;
      cnRe[v] = (cnRe[v] ?? 0) + invL * rr;
      cnIm[v] = (cnIm[v] ?? 0) + invL * ri;
    }
    solveNodal();
    // z += G y (edges <- nodes).
    for (let e = 0; e < nFree; e += 1) {
      const invL = dofInvLen[e] ?? 0;
      const u = dofNodeU[e] ?? 0;
      const v = dofNodeV[e] ?? 0;
      zRe[e] = (zRe[e] ?? 0) + invL * ((yRe[v] ?? 0) - (yRe[u] ?? 0));
      zIm[e] = (zIm[e] ?? 0) + invL * ((yIm[v] ?? 0) - (yIm[u] ?? 0));
    }
  };

  return {
    nodeCount,
    nodalNnz: aColIdx.length,
    applyCorrection,
  };
}
