/**
 * ParamSchema v1 for QuickJS-WASM parametric model definitions.
 *
 * This schema describes arbitrary user-defined parameters that can be rendered
 * in UI forms and validated before passing into buildModel().
 */

export const PARAM_SCHEMA_VERSION = 1;

export type ParamType = "number" | "int" | "bool" | "string" | "enum" | "vector";

interface ParamDefinitionBase {
  /** Optional UI label */
  title?: string;
  /** Optional UI help text */
  description?: string;
  /** Optional units hint (Hz, m, mm, etc.) */
  unit?: string;
  /** Whether this parameter must be provided. Defaults to true when omitted. */
  required?: boolean;
}

export interface NumberParamDefinition extends ParamDefinitionBase {
  type: "number";
  default?: number;
  min?: number;
  max?: number;
  step?: number;
}

export interface IntParamDefinition extends ParamDefinitionBase {
  type: "int";
  default?: number;
  min?: number;
  max?: number;
  step?: number;
}

export interface BoolParamDefinition extends ParamDefinitionBase {
  type: "bool";
  default?: boolean;
}

export interface StringParamDefinition extends ParamDefinitionBase {
  type: "string";
  default?: string;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
}

export interface EnumParamDefinition extends ParamDefinitionBase {
  type: "enum";
  values: string[];
  default?: string;
}

export interface VectorParamDefinition extends ParamDefinitionBase {
  type: "vector";
  default?: number[];
  size?: number;
  min?: number;
  max?: number;
  step?: number;
}

export type ParamDefinition =
  | NumberParamDefinition
  | IntParamDefinition
  | BoolParamDefinition
  | StringParamDefinition
  | EnumParamDefinition
  | VectorParamDefinition;

export interface ParamConstraint {
  /** Boolean expression evaluated against params, e.g. "directorSpacing < 0.5 * lambda" */
  expr: string;
  /** Optional user-facing message when constraint fails */
  message?: string;
}

export interface ParamGroup {
  /** Group title shown in UI */
  title: string;
  /** Ordered param keys in this group */
  params: string[];
}

export interface ParamSchema {
  version: typeof PARAM_SCHEMA_VERSION;
  title?: string;
  description?: string;
  params: Record<string, ParamDefinition>;
  groups?: ParamGroup[];
  constraints?: ParamConstraint[];
}

const KEY_RE = /^[A-Za-z][A-Za-z0-9_]*$/;

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(`Invalid ParamSchema: ${message}`);
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function assertOptionalFiniteNumber(value: unknown, path: string): void {
  if (value !== undefined) {
    assert(isFiniteNumber(value), `${path} must be a finite number`);
  }
}

function assertOptionalInteger(value: unknown, path: string): void {
  if (value !== undefined) {
    assert(typeof value === "number" && Number.isInteger(value), `${path} must be an integer`);
  }
}

function assertNumericParam(path: string, def: Record<string, unknown>, integerOnly: boolean): void {
  const assertNum = integerOnly ? assertOptionalInteger : assertOptionalFiniteNumber;
  assertNum(def.min, `${path}.min`);
  assertNum(def.max, `${path}.max`);
  assertNum(def.step, `${path}.step`);
  assertNum(def.default, `${path}.default`);

  if (def.min !== undefined && def.max !== undefined) {
    assert((def.min as number) <= (def.max as number), `${path}.min must be <= ${path}.max`);
  }
  if (def.step !== undefined) {
    assert((def.step as number) > 0, `${path}.step must be > 0`);
  }

  if (def.default !== undefined) {
    const defaultValue = def.default as number;
    if (def.min !== undefined) assert(defaultValue >= (def.min as number), `${path}.default must be >= ${path}.min`);
    if (def.max !== undefined) assert(defaultValue <= (def.max as number), `${path}.default must be <= ${path}.max`);
  }
}

function validateDefinition(path: string, value: unknown): void {
  assert(isObject(value), `${path} must be an object`);

  const def = value as Record<string, unknown>;
  assert(typeof def.type === "string", `${path}.type is required`);
  const type = def.type as string;
  assert(["number", "int", "bool", "string", "enum", "vector"].includes(type), `${path}.type must be one of number|int|bool|string|enum|vector`);

  if (def.title !== undefined) assert(typeof def.title === "string", `${path}.title must be a string`);
  if (def.description !== undefined) assert(typeof def.description === "string", `${path}.description must be a string`);
  if (def.unit !== undefined) assert(typeof def.unit === "string", `${path}.unit must be a string`);
  if (def.required !== undefined) assert(typeof def.required === "boolean", `${path}.required must be a boolean`);

  switch (type) {
    case "number":
      assertNumericParam(path, def, false);
      break;

    case "int":
      assertNumericParam(path, def, true);
      break;

    case "bool":
      if (def.default !== undefined) {
        assert(typeof def.default === "boolean", `${path}.default must be a boolean`);
      }
      break;

    case "string": {
      if (def.default !== undefined) {
        assert(typeof def.default === "string", `${path}.default must be a string`);
      }
      assertOptionalInteger(def.minLength, `${path}.minLength`);
      assertOptionalInteger(def.maxLength, `${path}.maxLength`);
      if (def.minLength !== undefined) {
        assert((def.minLength as number) >= 0, `${path}.minLength must be >= 0`);
      }
      if (def.maxLength !== undefined) {
        assert((def.maxLength as number) >= 0, `${path}.maxLength must be >= 0`);
      }
      if (def.minLength !== undefined && def.maxLength !== undefined) {
        assert((def.minLength as number) <= (def.maxLength as number), `${path}.minLength must be <= ${path}.maxLength`);
      }
      if (def.pattern !== undefined) {
        assert(typeof def.pattern === "string", `${path}.pattern must be a string`);
        const pattern = def.pattern as string;
        try {
          new RegExp(pattern);
        } catch {
          throw new Error(`Invalid ParamSchema: ${path}.pattern is not a valid regular expression`);
        }
      }
      if (def.default !== undefined) {
        const defaultValue = def.default as string;
        if (def.minLength !== undefined) {
          assert(defaultValue.length >= (def.minLength as number), `${path}.default length must be >= ${path}.minLength`);
        }
        if (def.maxLength !== undefined) {
          assert(defaultValue.length <= (def.maxLength as number), `${path}.default length must be <= ${path}.maxLength`);
        }
      }
      break;
    }

    case "enum": {
      assert(Array.isArray(def.values), `${path}.values must be an array of strings`);
      const values = def.values as unknown[];
      assert(values.length > 0, `${path}.values must not be empty`);
      const seen = new Set<string>();
      for (let i = 0; i < values.length; i++) {
        const option = values[i] as unknown;
        assert(typeof option === "string", `${path}.values[${i}] must be a string`);
        const optionValue = option as string;
        assert(optionValue.length > 0, `${path}.values[${i}] must not be empty`);
        assert(!seen.has(optionValue), `${path}.values contains duplicate value "${optionValue}"`);
        seen.add(optionValue);
      }
      if (def.default !== undefined) {
        assert(typeof def.default === "string", `${path}.default must be a string`);
        assert(seen.has(def.default as string), `${path}.default must be included in ${path}.values`);
      }
      break;
    }

    case "vector": {
      if (def.default !== undefined) {
        assert(Array.isArray(def.default), `${path}.default must be an array of numbers`);
        const defaultValue = def.default as unknown[];
        assert(defaultValue.length > 0, `${path}.default must contain at least one component`);
        for (let i = 0; i < defaultValue.length; i++) {
          assert(isFiniteNumber(defaultValue[i]), `${path}.default[${i}] must be a finite number`);
        }
      }
      assertOptionalInteger(def.size, `${path}.size`);
      if (def.size !== undefined) {
        assert((def.size as number) >= 2, `${path}.size must be >= 2`);
      }
      assertOptionalFiniteNumber(def.min, `${path}.min`);
      assertOptionalFiniteNumber(def.max, `${path}.max`);
      assertOptionalFiniteNumber(def.step, `${path}.step`);
      if (def.min !== undefined && def.max !== undefined) {
        assert((def.min as number) <= (def.max as number), `${path}.min must be <= ${path}.max`);
      }
      if (def.step !== undefined) {
        assert((def.step as number) > 0, `${path}.step must be > 0`);
      }
      if (def.default !== undefined) {
        if (def.size !== undefined) {
          assert((def.default as number[]).length === (def.size as number), `${path}.default length must match ${path}.size`);
        }
        const d = def.default as number[];
        for (let i = 0; i < d.length; i++) {
          if (def.min !== undefined) assert(d[i]! >= (def.min as number), `${path}.default[${i}] must be >= ${path}.min`);
          if (def.max !== undefined) assert(d[i]! <= (def.max as number), `${path}.default[${i}] must be <= ${path}.max`);
        }
      }
      break;
    }
  }
}

/**
 * Validate a candidate object against ParamSchema v1 rules.
 * Throws an Error with a user-facing message if invalid.
 */
export function validateParamSchema(value: unknown): ParamSchema {
  assert(isObject(value), "schema must be an object");

  const schema = value as Record<string, unknown>;
  const version = schema.version;
  assert(version === PARAM_SCHEMA_VERSION, `version must be ${PARAM_SCHEMA_VERSION}`);
  if (schema.title !== undefined) assert(typeof schema.title === "string", "title must be a string");
  if (schema.description !== undefined) assert(typeof schema.description === "string", "description must be a string");

  const params = schema.params as Record<string, unknown>;
  assert(isObject(params), "params must be an object");
  const entries = Object.entries(params);
  assert(entries.length > 0, "params must define at least one parameter");

  for (const [key, def] of entries) {
    assert(KEY_RE.test(key), `invalid parameter key "${key}"`);
    validateDefinition(`params.${key}`, def);
  }

  if (schema.groups !== undefined) {
    assert(Array.isArray(schema.groups), "groups must be an array");
    const groups = schema.groups as unknown[];
    const knownParams = new Set(entries.map(([k]) => k));
    for (let i = 0; i < groups.length; i++) {
      const group = groups[i] as Record<string, unknown>;
      assert(isObject(group), `groups[${i}] must be an object`);
      assert(typeof group.title === "string" && group.title.length > 0, `groups[${i}].title must be a non-empty string`);
      assert(Array.isArray(group.params), `groups[${i}].params must be an array of parameter keys`);
      const groupParams = group.params as unknown[];
      for (let p = 0; p < groupParams.length; p++) {
        const paramKey = groupParams[p];
        assert(typeof paramKey === "string", `groups[${i}].params[${p}] must be a string`);
        const paramName = paramKey as string;
        assert(knownParams.has(paramName), `groups[${i}].params[${p}] references unknown parameter "${paramName}"`);
      }
    }
  }

  if (schema.constraints !== undefined) {
    assert(Array.isArray(schema.constraints), "constraints must be an array");
    const constraints = schema.constraints as unknown[];
    for (let i = 0; i < constraints.length; i++) {
      const c = constraints[i] as Record<string, unknown>;
      assert(isObject(c), `constraints[${i}] must be an object`);
      assert(typeof c.expr === "string" && c.expr.length > 0, `constraints[${i}].expr must be a non-empty string`);
      if (c.message !== undefined) {
        assert(typeof c.message === "string", `constraints[${i}].message must be a string`);
      }
    }
  }

  return schema as unknown as ParamSchema;
}

/**
 * JSON Schema (Draft 2020-12) for ParamSchema v1.
 */
export const PARAM_SCHEMA_JSON_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "https://antennasim.dev/schemas/param-schema-v1.json",
  title: "AntennaSim ParamSchema v1",
  type: "object",
  additionalProperties: false,
  required: ["version", "params"],
  properties: {
    version: { const: PARAM_SCHEMA_VERSION },
    title: { type: "string" },
    description: { type: "string" },
    params: {
      type: "object",
      minProperties: 1,
      propertyNames: { pattern: "^[A-Za-z][A-Za-z0-9_]*$" },
      additionalProperties: { $ref: "#/$defs/paramDefinition" },
    },
    groups: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["title", "params"],
        properties: {
          title: { type: "string", minLength: 1 },
          params: {
            type: "array",
            items: { type: "string", pattern: "^[A-Za-z][A-Za-z0-9_]*$" },
          },
        },
      },
    },
    constraints: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["expr"],
        properties: {
          expr: { type: "string", minLength: 1 },
          message: { type: "string" },
        },
      },
    },
  },
  $defs: {
    base: {
      type: "object",
      properties: {
        title: { type: "string" },
        description: { type: "string" },
        unit: { type: "string" },
        required: { type: "boolean" },
      },
    },
    numberDef: {
      allOf: [
        { $ref: "#/$defs/base" },
        {
          type: "object",
          additionalProperties: false,
          required: ["type"],
          properties: {
            type: { const: "number" },
            default: { type: "number" },
            min: { type: "number" },
            max: { type: "number" },
            step: { type: "number", exclusiveMinimum: 0 },
          },
        },
      ],
    },
    intDef: {
      allOf: [
        { $ref: "#/$defs/base" },
        {
          type: "object",
          additionalProperties: false,
          required: ["type"],
          properties: {
            type: { const: "int" },
            default: { type: "integer" },
            min: { type: "integer" },
            max: { type: "integer" },
            step: { type: "integer", exclusiveMinimum: 0 },
          },
        },
      ],
    },
    boolDef: {
      allOf: [
        { $ref: "#/$defs/base" },
        {
          type: "object",
          additionalProperties: false,
          required: ["type"],
          properties: {
            type: { const: "bool" },
            default: { type: "boolean" },
          },
        },
      ],
    },
    stringDef: {
      allOf: [
        { $ref: "#/$defs/base" },
        {
          type: "object",
          additionalProperties: false,
          required: ["type"],
          properties: {
            type: { const: "string" },
            default: { type: "string" },
            minLength: { type: "integer", minimum: 0 },
            maxLength: { type: "integer", minimum: 0 },
            pattern: { type: "string" },
          },
        },
      ],
    },
    enumDef: {
      allOf: [
        { $ref: "#/$defs/base" },
        {
          type: "object",
          additionalProperties: false,
          required: ["type", "values"],
          properties: {
            type: { const: "enum" },
            values: {
              type: "array",
              minItems: 1,
              items: { type: "string", minLength: 1 },
              uniqueItems: true,
            },
            default: { type: "string" },
          },
        },
      ],
    },
    vectorDef: {
      allOf: [
        { $ref: "#/$defs/base" },
        {
          type: "object",
          additionalProperties: false,
          required: ["type"],
          properties: {
            type: { const: "vector" },
            default: {
              type: "array",
              minItems: 1,
              items: { type: "number" },
            },
            size: { type: "integer", minimum: 2 },
            min: { type: "number" },
            max: { type: "number" },
            step: { type: "number", exclusiveMinimum: 0 },
          },
        },
      ],
    },
    paramDefinition: {
      oneOf: [
        { $ref: "#/$defs/numberDef" },
        { $ref: "#/$defs/intDef" },
        { $ref: "#/$defs/boolDef" },
        { $ref: "#/$defs/stringDef" },
        { $ref: "#/$defs/enumDef" },
        { $ref: "#/$defs/vectorDef" },
      ],
    },
  },
} as const;
