import { describe, expect, it } from "vitest";
import { selectFEMPort } from "../fem";
import type { FEMSimulationRequest } from "../fem";

const request: FEMSimulationRequest = {
  model_ir: {
    version: 1,
    paramSchemaVersion: 1,
    materials: [{ name: "air", epsilon_r: 1, mu_r: 1 }],
    entities: [{ id: "volume", kind: "box", center: [0, 0, 0], size: [0.1, 0.1, 0.1], material: "air" }],
    ports: [{ id: "feed", kind: "lumpedPort", positive: [0, 0, 0.002], negative: [0, 0, 0], impedance_ohms: 50 }],
    boundaries: [],
    meshHints: [],
  },
  start_mhz: 900,
  stop_mhz: 936,
  steps: 5,
  mesh_size_m: 0.0005,
  reference_impedance_ohms: 50,
};

describe("browser FEM contract", () => {
  it("selects a validated lumped port", () => {
    expect(selectFEMPort(request).id).toBe("feed");
  });

  it("rejects an invalid ModelIR before entering WASM", () => {
    expect(() => selectFEMPort({
      ...request,
      model_ir: {
        ...request.model_ir,
        entities: [{ ...request.model_ir.entities[0]!, material: "missing" }],
      },
    })).toThrow(/entities.*material/);
  });
});