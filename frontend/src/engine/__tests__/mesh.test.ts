import { describe, expect, it } from "vitest";
import { buildFEMMesh } from "../mesh";
import type { ModelIR } from "../../dsl/model-ir";

const base: ModelIR = {
  version: 1,
  paramSchemaVersion: 1,
  materials: [{ name: "air", epsilon_r: 1, mu_r: 1 }],
  entities: [{ id: "box", kind: "box", center: [0, 0, 0], size: [0.1, 0.1, 0.1], material: "air" }],
  ports: [{ id: "feed", kind: "lumpedPort", positive: [0, 0, 0.04], negative: [0, 0, -0.04], impedance_ohms: 50 }],
  boundaries: [],
  meshHints: [],
};

describe("bounded FEM mesh conversion", () => {
  it("maps a finite-thickness box to deterministic tetrahedra in metres", () => {
    const first = buildFEMMesh(base, 0.05);
    const second = buildFEMMesh(base, 0.05);
    expect(first.statistics.elements).toBe(40);
    expect(first.statistics.nodes).toBe(27);
    expect(Array.from(first.vertices)).toEqual(Array.from(second.vertices));
    expect(Array.from(first.tetrahedra)).toEqual(Array.from(second.tetrahedra));
  });

  it("rejects surface sheets and overlapping volumes before MFEM", () => {
    expect(() => buildFEMMesh({ ...base, entities: [{ id: "sheet", kind: "sheetRect", center: [0, 0, 0], size: [1, 1], normal: [0, 0, 1], material: "air" }] }, 0.1)).toThrow(/sheetRect/);
    expect(() => buildFEMMesh({ ...base, entities: [base.entities[0]!, { ...base.entities[0]!, id: "overlap", center: [0.01, 0, 0] }] }, 0.1)).toThrow(/Overlapping/);
  });
});