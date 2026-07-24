import { describe, expect, it } from "vitest";
import { exportModelIRToGmsh } from "../gmsh";
import type { ModelIR, SheetRectEntity } from "../model-ir";

const ir: ModelIR = {
  version: 1,
  paramSchemaVersion: 1,
  materials: [
    { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 58000000 },
    { name: "air", epsilon_r: 1, mu_r: 1 },
  ],
  entities: [
    { id: "patch", kind: "sheetRect", center: [0, 0, 0], size: [0.04, 0.02], normal: [0, 0, 1], material: "copper" },
    { id: "volume", kind: "box", center: [0, 0, 0], size: [0.1, 0.1, 0.1], material: "air" },
  ],
  ports: [{ id: "feed", kind: "lumpedPort", positive: [0, 0, 0.001], negative: [0, 0, -0.001], impedance_ohms: 50 }],
  boundaries: [{ kind: "radiation", padding_m: 0.1 }],
  meshHints: [{ region: "patch", max_size_m: 0.005 }],
};

describe("exportModelIRToGmsh", () => {
  it("exports deterministic geometry, physical groups, and port endpoints", () => {
    const result = exportModelIRToGmsh(ir);
    expect(result.diagnostics).toEqual([]);
    expect(result.geo).toContain('SetFactory("OpenCASCADE");');
    expect(result.geo).toContain('Physical Volume("airbox")');
    expect(result.geo).toContain('Physical Surface("copper")');
    expect(result.geo).toContain('Physical Point("feed_positive")');
    expect(result.geo).toContain('Physical Point("feed_negative")');
    expect(result.geo).toContain("Mesh.CharacteristicLengthMax = 0.005;");
  });

  it("exports finite-thickness boxes as volume groups", () => {
    const result = exportModelIRToGmsh({
      ...ir,
      entities: [{ id: "patch", kind: "box", center: [0, 0, 0.001], size: [0.04, 0.02, 0.000035], material: "copper" }],
      meshHints: [{ region: "patch", max_size_m: 0.005 }],
    });
    expect(result.diagnostics).toEqual([]);
    expect(result.geo).toContain('Physical Volume("copper")');
    expect(result.geo).not.toContain('Physical Surface("copper")');
  });

  it("reports unsupported sheet orientations", () => {
    const result = exportModelIRToGmsh({
      ...ir,
      entities: [{ ...(ir.entities[0] as SheetRectEntity), normal: [1, 0, 0] }],
    });
    expect(result.diagnostics).toContainEqual(expect.objectContaining({ path: "entities[0].normal", severity: "error" }));
  });
});