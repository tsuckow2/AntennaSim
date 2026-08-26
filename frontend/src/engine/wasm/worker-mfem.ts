import type { MfemSmokeModule } from "./mfem-module";

export interface MfemSmokeRequest {
  type: "mfem_smoke";
  id: string;
}

export interface MfemSmokeSuccess {
  type: "success";
  id: string;
  trueDofs: number;
}

export interface MfemSmokeError {
  type: "error";
  id: string;
  message: string;
}

export type MfemSmokeResponse = MfemSmokeSuccess | MfemSmokeError;
type MfemFactory = (options?: Record<string, unknown>) => Promise<MfemSmokeModule>;
let factory: MfemFactory | null = null;

async function loadFactory(): Promise<MfemFactory> {
  if (factory) return factory;
  const response = await fetch(`${import.meta.env.BASE_URL}wasm/mfem_smoke.js`);
  if (!response.ok) throw new Error(`Failed to fetch mfem_smoke.js (${response.status})`);
  (0, eval)(await response.text());
  const loaded = (self as unknown as Record<string, unknown>).createMfemSmoke as MfemFactory | undefined;
  if (!loaded) throw new Error("mfem_smoke.js loaded but createMfemSmoke was not found");
  factory = loaded;
  return loaded;
}

self.onmessage = async (event: MessageEvent<MfemSmokeRequest>) => {
  if (event.data.type !== "mfem_smoke") return;
  try {
    const module = await (await loadFactory())({
      locateFile: (path: string) => path.endsWith(".wasm") ? `${import.meta.env.BASE_URL}wasm/mfem_smoke.wasm` : path,
    });
    const trueDofs = module._mfem_hcurl_smoke();
    if (trueDofs <= 0) throw new Error(`MFEM H(curl) smoke test failed with code ${trueDofs}`);
    self.postMessage({ type: "success", id: event.data.id, trueDofs } satisfies MfemSmokeSuccess);
  } catch (error) {
    self.postMessage({ type: "error", id: event.data.id, message: error instanceof Error ? error.message : String(error) } satisfies MfemSmokeError);
  }
};
