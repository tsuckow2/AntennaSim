/**
 * DSL store — state for the Parametric Script Editor page.
 *
 * Holds the current script source, user-edited parameter values, the last
 * ScriptExecution result, and run status.  The store does NOT auto-run on
 * parameter changes; callers schedule debounced re-runs themselves.
 */

import { create } from "zustand";
import type { ScriptExecution } from "../dsl/executor";
import { runParametricScript } from "../dsl/run-in-worker";
import { runBrowserFEM, cancelBrowserFEM } from "../engine/wasm";
import {
  FEMRunAbortedError,
  validateFEMSweep,
  type FEMSimulationRequest,
  type FEMSimulationResult,
} from "../engine/fem";
import type { ModelIR } from "../dsl/model-ir";
import PATCH_SRC from "../dsl/examples/patch_908_directional.js?raw";

export type DslRunStatus = "idle" | "running" | "success" | "error";
export type FemRunStatus =
  | "idle"
  | "running"
  | "success"
  | "error"
  | "cancelled";

interface DslState {
  /** Current script source */
  script: string;
  /** User-set parameter values (keyed by param key) */
  paramValues: Record<string, unknown>;
  /** Run lifecycle status */
  status: DslRunStatus;
  /** Last execution result (schema and/or IR may be absent on failure) */
  result: ScriptExecution | null;
  /** Worker-level error message (timeout, crash — separate from diagnostics) */
  runError: string | null;
  femStatus: FemRunStatus;
  femResult: FEMSimulationResult | null;
  femError: string | null;
  femRequestToken: number;
  /** Latest worker progress while a FEM run is in flight (null otherwise). */
  femProgress: {
    pointIndex: number;
    pointsTotal: number;
    stage: string;
  } | null;

  /** Replace the entire script source and reset execution state */
  setScript: (script: string) => void;
  /** Update a single param value (does not trigger a re-run) */
  setParamValue: (key: string, value: unknown) => void;
  /** Replace all param values at once */
  setParamValues: (values: Record<string, unknown>) => void;
  /** Execute the script inside a Web Worker and update state */
  runScript: () => Promise<void>;
  runFEM: (options: Omit<FEMSimulationRequest, "model_ir">) => Promise<void>;
  cancelFEM: () => void;
}

export const useDslStore = create<DslState>((set, get) => ({
  script: PATCH_SRC,
  paramValues: {},
  status: "idle",
  result: null,
  runError: null,
  femStatus: "idle",
  femResult: null,
  femError: null,
  femRequestToken: 0,
  femProgress: null,

  setScript: (script) =>
    set((state) => ({
      script,
      paramValues: {},
      result: null,
      status: "idle",
      runError: null,
      femStatus:
        state.femStatus === "running" || state.femResult ? "cancelled" : "idle",
      femError: null,
      femProgress: null,
      femRequestToken: state.femRequestToken + 1,
    })),

  setParamValue: (key, value) =>
    set((s) => ({ paramValues: { ...s.paramValues, [key]: value } })),

  setParamValues: (values) => set({ paramValues: values }),

  runScript: async () => {
    const { script, paramValues } = get();
    set({ status: "running", runError: null });
    try {
      const hasValues = Object.keys(paramValues).length > 0;
      const result = await runParametricScript(
        { code: script, values: hasValues ? paramValues : undefined },
        { timeoutMs: 5000 },
      );
      // Populate defaults for params not yet in paramValues (first run only)
      const updated = { ...paramValues };
      let populated = false;
      if (result.schema) {
        for (const [key, def] of Object.entries(result.schema.params)) {
          if (!(key in updated) && "default" in def) {
            updated[key] = def.default as unknown;
            populated = true;
          }
        }
      }
      set({
        status: result.ir !== undefined ? "success" : "error",
        result,
        runError: null,
        ...(populated ? { paramValues: updated } : {}),
      });
    } catch (err) {
      set({
        status: "error",
        runError: err instanceof Error ? err.message : String(err),
      });
    }
  },
  runFEM: async (options) => {
    const state = get();
    const modelIR = state.result?.ir;
    if (!modelIR) {
      set({
        femStatus: "error",
        femError: "Run the script successfully before starting FEM",
      });
      return;
    }
    try {
      validateFEMSweep(options.start_mhz, options.stop_mhz, options.steps);
      if (!Number.isFinite(options.mesh_size_m) || options.mesh_size_m <= 0) {
        throw new Error("FEM mesh size must be a positive finite value");
      }
      if (
        !Number.isFinite(options.reference_impedance_ohms) ||
        options.reference_impedance_ohms <= 0
      ) {
        throw new Error(
          "FEM reference impedance must be a positive finite value",
        );
      }
    } catch (error) {
      set({
        femStatus: "error",
        femError: error instanceof Error ? error.message : String(error),
      });
      return;
    }
    const token = state.femRequestToken + 1;
    set({ femStatus: "running", femError: null, femProgress: null, femRequestToken: token });
    try {
      // FEM always runs in the browser via the FEM worker (native hcurl WASM
      // first, TypeScript H(curl) fallback flagged in the result warnings),
      // independent of the VITE_ENGINE selection used for NEC2 simulations.
      const result = await runBrowserFEM(
        {
          ...options,
          model_ir: modelIR as ModelIR,
        },
        (progress) => {
          if (get().femRequestToken !== token) return;
          set({ femProgress: progress });
        },
      );
      if (get().femRequestToken !== token) return;
      set({ femStatus: "success", femResult: result, femError: null, femProgress: null });
    } catch (error) {
      if (get().femRequestToken !== token) return;
      if (error instanceof FEMRunAbortedError) {
        set({
          femStatus: error.kind === "cancelled" ? "cancelled" : "error",
          femResult: error.partialResult,
          femError: error.message,
          femProgress: null,
        });
        return;
      }
      set({
        femStatus: "error",
        femError: error instanceof Error ? error.message : String(error),
        femProgress: null,
      });
    }
  },
  cancelFEM: () => {
    cancelBrowserFEM();
    set({ femStatus: "cancelled", femError: "FEM run cancelled", femProgress: null });
  },
}));
