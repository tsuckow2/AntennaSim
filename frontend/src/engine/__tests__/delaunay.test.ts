import { describe, expect, it } from "vitest";
import { delaunay3d, inSphere, orient3d } from "../delaunay";

describe("delaunay3d", () => {
  it("triangulates a single tetrahedron", () => {
    const pts = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const tets = delaunay3d(pts);
    expect(tets.length).toBe(4);
    const sorted = [...tets].sort((a, b) => a - b);
    expect(sorted).toEqual([0, 1, 2, 3]);
  });

  it("triangulates a cube into 5 or 6 tets", () => {
    const pts = new Float64Array([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
      0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
    ]);
    const tets = delaunay3d(pts);
    const nt = tets.length / 4;
    expect(nt).toBeGreaterThanOrEqual(5);
    expect(nt).toBeLessThanOrEqual(6);
    // All tets positively oriented.
    for (let t = 0; t < nt; t++) {
      const [a, b, c, d] = [tets[t * 4]!, tets[t * 4 + 1]!, tets[t * 4 + 2]!, tets[t * 4 + 3]!];
      const o = orient3d(
        pts[a * 3]!, pts[a * 3 + 1]!, pts[a * 3 + 2]!,
        pts[b * 3]!, pts[b * 3 + 1]!, pts[b * 3 + 2]!,
        pts[c * 3]!, pts[c * 3 + 1]!, pts[c * 3 + 2]!,
        pts[d * 3]!, pts[d * 3 + 1]!, pts[d * 3 + 2]!,
      );
      expect(o).toBeGreaterThan(0);
    }
  });

  it("satisfies the Delaunay empty-sphere property on a grid", () => {
    // 3x3x3 grid.
    const pts: number[] = [];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) pts.push(i, j, k);
    const parr = new Float64Array(pts);
    const tets = delaunay3d(parr);
    const nt = tets.length / 4;
    expect(nt).toBeGreaterThan(0);
    // Spot-check: no point strictly inside the circumsphere of the first tet.
    const [a, b, c, d] = [tets[0]!, tets[1]!, tets[2]!, tets[3]!];
    for (let p = 0; p < 27; p++) {
      if (p === a || p === b || p === c || p === d) continue;
      const s = inSphere(
        parr[a * 3]!, parr[a * 3 + 1]!, parr[a * 3 + 2]!,
        parr[b * 3]!, parr[b * 3 + 1]!, parr[b * 3 + 2]!,
        parr[c * 3]!, parr[c * 3 + 1]!, parr[c * 3 + 2]!,
        parr[d * 3]!, parr[d * 3 + 1]!, parr[d * 3 + 2]!,
        parr[p * 3]!, parr[p * 3 + 1]!, parr[p * 3 + 2]!,
      );
      expect(s).toBeLessThanOrEqual(0);
    }
  });
});
