import { describe, expect, it } from "vitest";
import { executeParametricScript } from "../executor";
import { serializeModelIR, validateModelIR } from "../model-ir";

const script = `
function defineParams() {
  return { version: 1, params: {
    width: { type: "number", default: 0.04, min: 0.01, max: 0.1 }
  }};
}
function buildModel(params, api) {
  api.material("copper", { epsilon_r: 1, conductivity_s_m: 58000000 });
  api.sheetRect("patch", { center: [0, 0, 0], size: [params.width, 0.02], material: "copper" });
  api.lumpedPort("feed", { positive: [0, 0, 0.001], negative: [0, 0, -0.001] });
  api.airbox({ padding_m: 0.1 });
}
`;

describe("executeParametricScript", () => {
  it("loads schema and compiles a model into validated IR", async () => {
    const result = await executeParametricScript(script);
    expect(result.schema?.params.width?.type).toBe("number");
    expect(result.ir?.entities[0]?.id).toBe("patch");
    expect(result.diagnostics).toHaveLength(0);
    expect(serializeModelIR(result.ir!)).toContain('"version":1');
  });

  it("reports schema and model failures without throwing", async () => {
    expect((await executeParametricScript("function defineParams() { return {}; }")).diagnostics[0]).toMatchObject({ phase: "schema" });
    const result = await executeParametricScript(`${script}\nfunction buildModel() { throw new Error("bad geometry"); }`);
    expect(result.diagnostics[0]).toMatchObject({ phase: "model", message: "bad geometry" });
  });
});

describe("validateModelIR", () => {
  it("rejects missing material references and degenerate geometry", () => {
    const errors = validateModelIR({
      version: 1,
      paramSchemaVersion: 1,
      materials: [],
      entities: [{ id: "patch", kind: "box", center: [0, 0, 0], size: [0, 1, 1], material: "missing" }],
      ports: [],
      boundaries: [],
      meshHints: [],
    });
    expect(errors.map((error) => error.path)).toEqual(expect.arrayContaining(["entities[0].material", "entities[0].size"]));
  });
});
