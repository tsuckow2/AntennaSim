import type { ScriptExecution } from "./executor";
import type { ScriptWorkerRequest, ScriptWorkerResponse } from "./worker";

export interface ScriptRunOptions {
  timeoutMs?: number;
}

/**
 * Execute user code away from the UI thread. Terminating the worker is the
 * hard timeout boundary; callers should use this entry point for untrusted
 * scripts rather than calling executeParametricScript directly.
 */
export function runParametricScript(
  request: ScriptWorkerRequest,
  options: ScriptRunOptions = {},
): Promise<ScriptExecution> {
  const timeoutMs = options.timeoutMs ?? 5000;
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    const timeout = globalThis.setTimeout(() => {
      worker.terminate();
      reject(new Error(`Script timed out after ${timeoutMs} ms`));
    }, timeoutMs);
    worker.onmessage = (event: MessageEvent<ScriptWorkerResponse>) => {
      globalThis.clearTimeout(timeout);
      worker.terminate();
      resolve(event.data.result);
    };
    worker.onerror = (event) => {
      globalThis.clearTimeout(timeout);
      worker.terminate();
      reject(new Error(event.message || "Script worker failed"));
    };
    worker.postMessage(request);
  });
}
