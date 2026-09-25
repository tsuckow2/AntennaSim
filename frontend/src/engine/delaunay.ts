/**
 * Incremental 3D Delaunay triangulation (Bowyer-Watson).
 *
 * Given a set of points, computes the Delaunay tetrahedralization:
 * no point lies inside the circumsphere of any tetrahedron.
 *
 * Used as the alternative mesher for the AntennaSim FEM pipeline.
 * The triangulation is deterministic for a fixed input point order
 * (ties broken by insertion order and a fixed epsilon).
 */

/** A tetrahedron: 4 vertex indices, positively oriented (orient3d > 0). */
export type Tet = [number, number, number, number];

/** Signed volume * 6 of tet (a,b,c,d). Positive if d is above plane abc. */
export function orient3d(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
  dx: number, dy: number, dz: number,
): number {
  const m11 = bx - ax, m12 = by - ay, m13 = bz - az;
  const m21 = cx - ax, m22 = cy - ay, m23 = cz - az;
  const m31 = dx - ax, m32 = dy - ay, m33 = dz - az;
  return (
    m11 * (m22 * m33 - m23 * m32) -
    m12 * (m21 * m33 - m23 * m31) +
    m13 * (m21 * m32 - m22 * m31)
  );
}

/**
 * In-sphere test. For a positively-oriented tet (a,b,c,d), returns
 * positive if p is strictly inside the circumsphere, negative if outside.
 * Computed via the circumsphere center (3x3 solve).
 */
export function inSphere(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
  dx: number, dy: number, dz: number,
  px: number, py: number, pz: number,
): number {
  // Solve 2(b-a)·c = |b|²-|a|², etc. for circumcenter.
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const adx = dx - ax, ady = dy - ay, adz = dz - az;
  const rhs1 = (bx * bx + by * by + bz * bz - ax * ax - ay * ay - az * az) / 2;
  const rhs2 = (cx * cx + cy * cy + cz * cz - ax * ax - ay * ay - az * az) / 2;
  const rhs3 = (dx * dx + dy * dy + dz * dz - ax * ax - ay * ay - az * az) / 2;

  // 3x3 determinant (Cramer's rule).
  const det =
    abx * (acy * adz - acz * ady) -
    aby * (acx * adz - acz * adx) +
    abz * (acx * ady - acy * adx);
  if (Math.abs(det) < 1e-300) return 0; // Degenerate tet.

  const detX =
    rhs1 * (acy * adz - acz * ady) -
    aby * (rhs2 * adz - acz * rhs3) +
    abz * (rhs2 * ady - acy * rhs3);
  const detY =
    abx * (rhs2 * adz - acz * rhs3) -
    rhs1 * (acx * adz - acz * adx) +
    abz * (acx * rhs3 - rhs2 * adx);
  const detZ =
    abx * (acy * rhs3 - rhs2 * ady) -
    aby * (acx * rhs3 - rhs2 * adx) +
    rhs1 * (acx * ady - acy * adx);
  const ccx = detX / det, ccy = detY / det, ccz = detZ / det;

  const r2 = (ax - ccx) ** 2 + (ay - ccy) ** 2 + (az - ccz) ** 2;
  const d2 = (px - ccx) ** 2 + (py - ccy) ** 2 + (pz - ccz) ** 2;
  // Scale-aware epsilon: relative to the circumradius.
  const eps = 1e-9 * r2;
  if (Math.abs(r2 - d2) <= eps) return 0;
  return r2 - d2; // > 0 if p inside.
}

/** Face key: sorted vertex indices joined. */
function faceKey(a: number, b: number, c: number): string {
  if (a > b) [a, b] = [b, a];
  if (b > c) [b, c] = [c, b];
  if (a > b) [a, b] = [b, a];
  return `${a},${b},${c}`;
}

/** The 4 faces of a tet: each is the 3 verts opposite one vertex. */
function tetFaces(t: Tet): [[number, number, number], [number, number, number], [number, number, number], [number, number, number]] {
  const [v0, v1, v2, v3] = t;
  return [
    [v1, v2, v3], // opposite v0
    [v0, v2, v3], // opposite v1
    [v0, v1, v3], // opposite v2
    [v0, v1, v2], // opposite v3
  ];
}

/**
 * Compute the Delaunay tetrahedralization of points.
 * @param points Flat [x0,y0,z0, x1,y1,z1, ...].
 * @returns Flat tet indices [t0v0,t0v1,t0v2,t0v3, t1v0,...], positively oriented.
 */
export function delaunay3d(points: Float64Array): Uint32Array {
  const n = points.length / 3;
  if (n < 4) throw new Error("delaunay3d needs at least 4 points");

  // Bounding box.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = points[i * 3]!, y = points[i * 3 + 1]!, z = points[i * 3 + 2]!;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  const R = Math.sqrt((maxX - minX) ** 2 + (maxY - minY) ** 2 + (maxZ - minZ) ** 2) / 2;
  const D = 10 * R + 1;

  // Supertetrahedron vertices (regular tet, inradius D*10/3 > R, contains all points).
  const s = D / Math.sqrt(3);
  const superVerts: [number, number, number][] = [
    [cx + s, cy + s, cz + s],
    [cx + s, cy - s, cz - s],
    [cx - s, cy + s, cz - s],
    [cx - s, cy - s, cz + s],
  ];
  // Ensure positive orientation.
  const s0 = superVerts[0]!;
  const s1 = superVerts[1]!;
  const s2 = superVerts[2]!;
  const s3 = superVerts[3]!;
  const o = orient3d(s0[0], s0[1], s0[2], s1[0], s1[1], s1[2], s2[0], s2[1], s2[2], s3[0], s3[1], s3[2]);
  const sv0 = n, sv1 = n + 1, sv2 = n + 2, sv3 = n + 3;
  const superTet: Tet = o > 0 ? [sv0, sv1, sv2, sv3] : [sv0, sv1, sv3, sv2];

  // All points (real + super) for predicate access.
  const allPoints = new Float64Array((n + 4) * 3);
  allPoints.set(points);
  for (let i = 0; i < 4; i++) {
    allPoints[(n + i) * 3] = superVerts[i]![0];
    allPoints[(n + i) * 3 + 1] = superVerts[i]![1];
    allPoints[(n + i) * 3 + 2] = superVerts[i]![2];
  }
  const px = (i: number) => allPoints[i * 3]!;
  const py = (i: number) => allPoints[i * 3 + 1]!;
  const pz = (i: number) => allPoints[i * 3 + 2]!;

  // Tet storage. Dead tets are marked (alive=false); indices stay stable.
  const tets: Tet[] = [superTet];
  const alive: boolean[] = [true];

  // Face -> tet indices (only alive tets). Updated incrementally.
  const faceMap = new Map<string, number[]>();
  const addFace = (a: number, b: number, c: number, t: number) => {
    const k = faceKey(a, b, c);
    const lst = faceMap.get(k);
    if (lst) lst.push(t);
    else faceMap.set(k, [t]);
  };
  const removeFace = (a: number, b: number, c: number, t: number) => {
    const k = faceKey(a, b, c);
    const lst = faceMap.get(k);
    if (!lst) return;
    const idx = lst.indexOf(t);
    if (idx >= 0) lst.splice(idx, 1);
    if (lst.length === 0) faceMap.delete(k);
  };
  for (const [a, b, c] of tetFaces(superTet)) addFace(a, b, c, 0);

  const addTet = (t: Tet): number => {
    const idx = tets.length;
    tets.push(t);
    alive.push(true);
    for (const [a, b, c] of tetFaces(t)) addFace(a, b, c, idx);
    return idx;
  };
  const killTet = (idx: number) => {
    if (!alive[idx]) return;
    alive[idx] = false;
    for (const [a, b, c] of tetFaces(tets[idx]!)) removeFace(a, b, c, idx);
  };

  // Uniform grid for seed search: cell -> tet indices (rebuilt periodically).
  let grid = new Map<string, number[]>();
  let gridCell = 1;
  let gridValidUntil = 0; // rebuild when tets.length exceeds this
  const rebuildGrid = () => {
    grid = new Map();
    // Cell size from current tet count and bbox volume.
    const vol = (maxX - minX) * (maxY - minY) * (maxZ - minZ);
    const avgTetVol = vol / Math.max(1, tets.length);
    gridCell = Math.cbrt(avgTetVol) * 2;
    if (!isFinite(gridCell) || gridCell <= 0) gridCell = R / 10;
    for (let t = 0; t < tets.length; t++) {
      if (!alive[t]) continue;
      const [v0, v1, v2, v3] = tets[t]!;
      const gx = Math.floor(((px(v0) + px(v1) + px(v2) + px(v3)) / 4 - minX) / gridCell);
      const gy = Math.floor(((py(v0) + py(v1) + py(v2) + py(v3)) / 4 - minY) / gridCell);
      const gz = Math.floor(((pz(v0) + pz(v1) + pz(v2) + pz(v3)) / 4 - minZ) / gridCell);
      const k = `${gx},${gy},${gz}`;
      const lst = grid.get(k);
      if (lst) lst.push(t);
      else grid.set(k, [t]);
    }
    gridValidUntil = tets.length + Math.max(100, tets.length >> 2);
  };

  const inSphereTet = (t: number, pi: number): number => {
    const [v0, v1, v2, v3] = tets[t]!;
    return inSphere(
      px(v0), py(v0), pz(v0),
      px(v1), py(v1), pz(v1),
      px(v2), py(v2), pz(v2),
      px(v3), py(v3), pz(v3),
      px(pi), py(pi), pz(pi),
    );
  };

  // Insert points.
  for (let pi = 0; pi < n; pi++) {
    if (tets.length > gridValidUntil) rebuildGrid();

    // Seed: find a tet whose circumsphere contains pi.
    let seed = -1;
    const gx = Math.floor((px(pi) - minX) / gridCell);
    const gy = Math.floor((py(pi) - minY) / gridCell);
    const gz = Math.floor((pz(pi) - minZ) / gridCell);
    outer: for (let r = 0; r <= 3 && seed < 0; r++) {
      for (let dx = -r; dx <= r && seed < 0; dx++) {
        for (let dy = -r; dy <= r && seed < 0; dy++) {
          for (let dz = -r; dz <= r && seed < 0; dz++) {
            if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
            const lst = grid.get(`${gx + dx},${gy + dy},${gz + dz}`);
            if (!lst) continue;
            for (const t of lst) {
              if (!alive[t]) continue;
              if (inSphereTet(t, pi) > 0) {
                seed = t;
                break outer;
              }
            }
          }
        }
      }
    }
    if (seed < 0) {
      // Fallback: linear scan (should be rare).
      for (let t = 0; t < tets.length; t++) {
        if (!alive[t]) continue;
        if (inSphereTet(t, pi) > 0) {
          seed = t;
          break;
        }
      }
    }
    if (seed < 0) throw new Error(`delaunay3d: no seed tet found for point ${pi}`);

    // Cavity: BFS over tets with inSphere > 0.
    const bad = new Set<number>();
    const queue = [seed];
    bad.add(seed);
    while (queue.length > 0) {
      const t = queue.pop()!;
      for (const [a, b, c] of tetFaces(tets[t]!)) {
        const lst = faceMap.get(faceKey(a, b, c));
        if (!lst) continue;
        for (const nb of lst) {
          if (nb === t || !alive[nb] || bad.has(nb)) continue;
          if (inSphereTet(nb, pi) > 0) {
            bad.add(nb);
            queue.push(nb);
          }
        }
      }
    }

    // Boundary faces: exactly one adjacent tet is bad.
    const boundaryFaces: [number, number, number][] = [];
    for (const t of bad) {
      for (const [a, b, c] of tetFaces(tets[t]!)) {
        const lst = faceMap.get(faceKey(a, b, c)) ?? [];
        let badCount = 0;
        for (const nb of lst) if (bad.has(nb)) badCount++;
        // Face is on cavity boundary if t is the only bad tet adjacent.
        // (lst has 1 or 2 tets; t is bad; boundary if no other bad.)
        if (badCount === 1) boundaryFaces.push([a, b, c]);
      }
    }

    // Remove bad tets, add new tets (pi, face).
    for (const t of bad) killTet(t);
    const pix = px(pi), piy = py(pi), piz = pz(pi);
    for (const [a, b, c] of boundaryFaces) {
      const ax = px(a), ay = py(a), az = pz(a);
      const bx = px(b), by = py(b), bz = pz(b);
      const cx = px(c), cy = py(c), cz = pz(c);
      const o2 = orient3d(pix, piy, piz, ax, ay, az, bx, by, bz, cx, cy, cz);
      if (Math.abs(o2) < 1e-300) continue; // Degenerate; skip.
      const nt: Tet = o2 > 0 ? [pi, a, b, c] : [pi, a, c, b];
      addTet(nt);
    }
  }

  // Collect alive tets that don't use super vertices.
  const out: number[] = [];
  for (let t = 0; t < tets.length; t++) {
    if (!alive[t]) continue;
    const [v0, v1, v2, v3] = tets[t]!;
    if (v0 >= n || v1 >= n || v2 >= n || v3 >= n) continue;
    out.push(v0, v1, v2, v3);
  }
  return new Uint32Array(out);
}
