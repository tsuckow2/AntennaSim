import { validateParamSchema, type ParamSchema } from "../utils/param-schema";
import { createGeometryApi } from "./geometry-api";
import { validateModelIR, type IRDiagnostic, type ModelIR } from "./model-ir";

export interface ScriptDiagnostic {
  phase: "schema" | "model";
  message: string;
}

export interface ScriptExecution {
  schema?: ParamSchema;
  ir?: ModelIR;
  diagnostics: ScriptDiagnostic[] | IRDiagnostic[];
}

function compile(code: string): { defineParams: () => unknown; buildModel: (params: Record<string, unknown>, api: unknown) => unknown } {
  if (code.length > 1_000_000) throw new Error("Script exceeds the 1 MB limit");
  const factory = new Function(`"use strict"; const globalThis = undefined; const fetch = undefined; const XMLHttpRequest = undefined; const require = undefined; const process = undefined; ${code}; return { defineParams, buildModel };`);
  const exports = factory() as { defineParams?: unknown; buildModel?: unknown };
  if (typeof exports.defineParams !== "function" || typeof exports.buildModel !== "function") {
    throw new Error("Script must define defineParams() and buildModel(params, api)");
  }
  return exports as { defineParams: () => unknown; buildModel: (params: Record<string, unknown>, api: unknown) => unknown };
}

export function executeParametricScript(code: string, values?: Record<string, unknown>): ScriptExecution {
  let compiled: ReturnType<typeof compile>;
  try { compiled = compile(code); } catch (error) { return { diagnostics: [{ phase: "schema", message: error instanceof Error ? error.message : String(error) }] }; }
  let schema: ParamSchema;
  try { schema = validateParamSchema(compiled.defineParams()); } catch (error) {
    return { diagnostics: [{ phase: "schema", message: error instanceof Error ? error.message : String(error) }] };
  }
  const params = values ?? Object.fromEntries(Object.entries(schema.params).map(([key, definition]) => [key, "default" in definition ? definition.default : undefined]));
  const { api, ir } = createGeometryApi(schema.version);
  try { compiled.buildModel(params, api); } catch (error) {
    return { schema, diagnostics: [{ phase: "model", message: error instanceof Error ? error.message : String(error) }] };
  }
  const diagnostics = validateModelIR(ir);
  return diagnostics.some((diagnostic) => diagnostic.severity === "error") ? { schema, diagnostics } : { schema, ir, diagnostics };
}
