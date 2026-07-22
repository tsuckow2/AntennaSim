import { validateParamSchema, type ParamSchema } from "../utils/param-schema";
import { createGeometryApi } from "./geometry-api";
import { validateModelIR, type IRDiagnostic, type ModelIR } from "./model-ir";
import { getQuickJS } from "quickjs-emscripten";

export interface ScriptDiagnostic {
  phase: "schema" | "model";
  message: string;
}

export interface ScriptExecution {
  schema?: ParamSchema;
  ir?: ModelIR;
  diagnostics: ScriptDiagnostic[] | IRDiagnostic[];
}

export async function executeParametricScript(code: string, values?: Record<string, unknown>): Promise<ScriptExecution> {
  if (code.length > 1_000_000) return { diagnostics: [{ phase: "schema", message: "Script exceeds the 1 MB limit" }] };
  const QuickJS = await getQuickJS();
  const runtime = QuickJS.newRuntime();
  runtime.setMemoryLimit(16 * 1024 * 1024);
  let interruptCycles = 0;
  runtime.setInterruptHandler(() => ++interruptCycles > 5_000_000);
  const vm = runtime.newContext();
  const addApi = (name: string, callback: (...args: unknown[]) => void, apiObject: ReturnType<typeof vm.newObject>) => {
    vm.newFunction(name, (...args) => {
      callback(...args.map(vm.dump));
      return vm.undefined;
    }).consume((handle) => vm.setProp(apiObject, name, handle));
  };
  let schema: ParamSchema;
  try {
    const evaluated = vm.unwrapResult(vm.evalCode(`${code}\ndefineParams()`));
    const dumped = evaluated.consume(vm.dump);
    schema = validateParamSchema(dumped);
  } catch (error) {
    vm.dispose();
    runtime.dispose();
    return { diagnostics: [{ phase: "schema", message: error instanceof Error ? error.message : String(error) }] };
  }
  const params = values ?? Object.fromEntries(Object.entries(schema.params).map(([key, definition]) => [key, "default" in definition ? definition.default : undefined]));
  const { api, ir } = createGeometryApi(schema.version);
  const apiObject = vm.newObject();
  addApi("material", (name, spec) => api.material(String(name), spec as Parameters<typeof api.material>[1]), apiObject);
  addApi("box", (id, options) => api.box(String(id), options as Parameters<typeof api.box>[1]), apiObject);
  addApi("sheetRect", (id, options) => api.sheetRect(String(id), options as Parameters<typeof api.sheetRect>[1]), apiObject);
  addApi("lumpedPort", (id, options) => api.lumpedPort(String(id), options as Parameters<typeof api.lumpedPort>[1]), apiObject);
  addApi("airbox", (options) => api.airbox(options as Parameters<typeof api.airbox>[0]), apiObject);
  addApi("pml", (options) => api.pml(options as Parameters<typeof api.pml>[0]), apiObject);
  addApi("refine", (options) => api.refine(options as Parameters<typeof api.refine>[0]), apiObject);
  const paramsHandle = vm.unwrapResult(vm.evalCode(`(${JSON.stringify(params)})`));
  vm.setProp(vm.global, "__params", paramsHandle);
  paramsHandle.dispose();
  apiObject.consume((handle) => vm.setProp(vm.global, "api", handle));
  try {
    vm.unwrapResult(vm.evalCode(`${code}\nbuildModel(__params, api)`)).dispose();
  } catch (error) {
    vm.dispose();
    runtime.dispose();
    return { schema, diagnostics: [{ phase: "model", message: error instanceof Error ? error.message : String(error) }] };
  }
  vm.dispose();
  runtime.dispose();
  const diagnostics = validateModelIR(ir);
  return diagnostics.some((diagnostic) => diagnostic.severity === "error") ? { schema, diagnostics } : { schema, ir, diagnostics };
}
