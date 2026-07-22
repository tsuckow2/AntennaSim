import { executeParametricScript } from "./executor";

export interface ScriptWorkerRequest {
  code: string;
  values?: Record<string, unknown>;
}

export interface ScriptWorkerResponse {
  result: ReturnType<typeof executeParametricScript>;
}

self.onmessage = (event: MessageEvent<ScriptWorkerRequest>) => {
  try {
    const result = executeParametricScript(event.data.code, event.data.values);
    (self as unknown as Worker).postMessage({ result } satisfies ScriptWorkerResponse);
  } catch (error) {
    (self as unknown as Worker).postMessage({ result: { diagnostics: [{ phase: "model", message: String(error) }] } });
  }
};
