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
import PATCH_SRC from "../dsl/examples/patch_918_petg.js?raw";

export type DslRunStatus = "idle" | "running" | "success" | "error";

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

  /** Replace the entire script source and reset execution state */
  setScript: (script: string) => void;
  /** Update a single param value (does not trigger a re-run) */
  setParamValue: (key: string, value: unknown) => void;
  /** Replace all param values at once */
  setParamValues: (values: Record<string, unknown>) => void;
  /** Execute the script inside a Web Worker and update state */
  runScript: () => Promise<void>;
}

export const useDslStore = create<DslState>((set, get) => ({
  script: PATCH_SRC,
  paramValues: {},
  status: "idle",
  result: null,
  runError: null,

  setScript: (script) =>
    set({ script, paramValues: {}, result: null, status: "idle", runError: null }),

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
      set({ status: "error", runError: err instanceof Error ? err.message : String(err) });
    }
  },
}));
