import { executeParametricScript, type ScriptExecution } from "./executor";

export interface ScriptWorkerRequest {
  code: string;
  values?: Record<string, unknown>;
}

export interface ScriptWorkerResponse {
  result: ScriptExecution;
}

self.onmessage = (event: MessageEvent<ScriptWorkerRequest>) => {
  void executeParametricScript(event.data.code, event.data.values).then(
    (result) => (self as unknown as Worker).postMessage({ result } satisfies ScriptWorkerResponse),
    (error: unknown) => (self as unknown as Worker).postMessage({ result: { diagnostics: [{ phase: "model", message: String(error) }] } }),
  );
};
