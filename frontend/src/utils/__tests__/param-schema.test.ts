import {
  PARAM_SCHEMA_JSON_SCHEMA,
  PARAM_SCHEMA_VERSION,
  validateParamSchema,
} from "../param-schema";

describe("validateParamSchema", () => {
  it("accepts a valid mixed-type schema", () => {
    const parsed = validateParamSchema({
      version: PARAM_SCHEMA_VERSION,
      title: "Patch 918",
      params: {
        freq: { type: "number", default: 918e6, min: 100e6, max: 6e9, unit: "Hz" },
        patchLen: { type: "number", default: 0.045, min: 0.005, max: 0.2, unit: "m" },
        directors: { type: "int", default: 3, min: 1, max: 8, step: 1 },
        petgEnabled: { type: "bool", default: true },
        topology: { type: "enum", values: ["patch", "yagi"], default: "patch" },
        notes: { type: "string", default: "v1", minLength: 1, maxLength: 32 },
        offset: { type: "vector", size: 3, default: [0, 0, 0.008], min: -1, max: 1 },
      },
      groups: [
        { title: "Geometry", params: ["patchLen", "offset"] },
        { title: "Simulation", params: ["freq", "directors", "topology"] },
      ],
      constraints: [
        { expr: "directors <= 8" },
        { expr: "freq >= 100e6", message: "Frequency too low" },
      ],
    });

    expect(parsed.version).toBe(PARAM_SCHEMA_VERSION);
    expect(parsed.params.topology?.type).toBe("enum");
  });

  it("rejects unknown type", () => {
    expect(() => validateParamSchema({
      version: PARAM_SCHEMA_VERSION,
      params: {
        foo: { type: "float" },
      },
    })).toThrow("type must be one of");
  });

  it("rejects invalid param key", () => {
    expect(() => validateParamSchema({
      version: PARAM_SCHEMA_VERSION,
      params: {
        "bad-key": { type: "number" },
      },
    })).toThrow("invalid parameter key");
  });

  it("rejects enum default not in values", () => {
    expect(() => validateParamSchema({
      version: PARAM_SCHEMA_VERSION,
      params: {
        topology: { type: "enum", values: ["patch", "yagi"], default: "dipole" },
      },
    })).toThrow("must be included");
  });

  it("rejects vector default length mismatch", () => {
    expect(() => validateParamSchema({
      version: PARAM_SCHEMA_VERSION,
      params: {
        offset: { type: "vector", size: 3, default: [0, 0] },
      },
    })).toThrow("length must match");
  });

  it("rejects group references to unknown params", () => {
    expect(() => validateParamSchema({
      version: PARAM_SCHEMA_VERSION,
      params: {
        freq: { type: "number", default: 918e6 },
      },
      groups: [{ title: "Simulation", params: ["missing"] }],
    })).toThrow("unknown parameter");
  });

  it("rejects unsupported schema version", () => {
    expect(() => validateParamSchema({
      version: 2,
      params: {
        freq: { type: "number", default: 918e6 },
      },
    })).toThrow("version must be");
  });

  it("rejects invalid regex pattern in string definition", () => {
    expect(() => validateParamSchema({
      version: PARAM_SCHEMA_VERSION,
      params: {
        label: { type: "string", pattern: "[" },
      },
    })).toThrow("valid regular expression");
  });
});

describe("PARAM_SCHEMA_JSON_SCHEMA", () => {
  it("exports a draft-2020-12 object schema", () => {
    expect(PARAM_SCHEMA_JSON_SCHEMA.$schema).toBe("https://json-schema.org/draft/2020-12/schema");
    expect(PARAM_SCHEMA_JSON_SCHEMA.type).toBe("object");
    expect(PARAM_SCHEMA_JSON_SCHEMA.required).toEqual(["version", "params"]);
  });
});
