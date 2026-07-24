import type { ModelIR } from "../../dsl/model-ir";
import { selectFEMPort, type FEMSimulationRequest, type FEMSimulationResult } from "../fem";
import type { Fem1dModule } from "./fem1d-module";

export interface FEMWorkerMessage {
  type: "fem_simulate";
  id: string;
  request: FEMSimulationRequest;
}

export interface FEMWorkerSuccess {
  type: "success";
  id: string;
  result: FEMSimulationResult;
}

export interface FEMWorkerError {
  type: "error";
  id: string;
  message: string;
}

export type FEMWorkerResponse = FEMWorkerSuccess | FEMWorkerError;

type FemFactory = (options?: Record<string, unknown>) => Promise<Fem1dModule>;
let factory: FemFactory | null = null;

async function loadFactory(): Promise<FemFactory> {
  if (factory) return factory;
  const base = import.meta.env.BASE_URL;
  const response = await fetch(`${base}wasm/fem1d.js`);
  if (!response.ok) throw new Error(`Failed to fetch fem1d.js (${response.status}). Build the FEM WASM artifact first.`);
  (0, eval)(await response.text());
  const loaded = (self as unknown as Record<string, unknown>).createFem1d as FemFactory | undefined;
  if (!loaded) throw new Error("fem1d.js loaded but createFem1d was not found");
  factory = loaded;
  return loaded;
}

function portLength(modelIR: ModelIR, portId: string | undefined): number {
  const port = selectFEMPort({
    model_ir: modelIR,
    start_mhz: 1,
    stop_mhz: 1,
    steps: 1,
    mesh_size_m: 1,
    reference_impedance_ohms: 50,
    port_id: portId,
  });
  return Math.hypot(
    port.positive[0] - port.negative[0],
    port.positive[1] - port.negative[1],
    port.positive[2] - port.negative[2],
  );
}

async function solve(request: FEMSimulationRequest): Promise<FEMSimulationResult> {
  const started = performance.now();
  const lengthM = portLength(request.model_ir, request.port_id);
  if (!Number.isFinite(lengthM) || lengthM <= 0) throw new Error("Lumped-port terminals must be distinct");
  const steps = Math.max(1, Math.floor(request.steps));
  const outputPointerBytes = steps * 6 * Float64Array.BYTES_PER_ELEMENT;
  const module = await (await loadFactory())({
    locateFile: (path: string) => path.endsWith(".wasm") ? `${import.meta.env.BASE_URL}wasm/fem1d.wasm` : path,
  });
  const outputPointer = module._malloc(outputPointerBytes);
  if (!outputPointer) throw new Error("Unable to allocate FEM output buffer");
  try {
    const elements = module._fem_solve(
      lengthM,
      request.start_mhz,
      request.stop_mhz,
      steps,
      request.mesh_size_m,
      request.reference_impedance_ohms,
      outputPointer,
    );
    if (elements < 0) throw new Error(`FEM WASM solver failed with code ${elements}`);
    const start = outputPointer / Float64Array.BYTES_PER_ELEMENT;
    const values = module.HEAPF64.slice(start, start + steps * 6);
    const frequencyData = Array.from({ length: steps }, (_, index) => {
      const offset = index * 6;
      return {
        frequency_mhz: values[offset]!,
        impedance_real_ohms: values[offset + 1]!,
        impedance_imag_ohms: values[offset + 2]!,
        s11_real: values[offset + 3]!,
        s11_imag: values[offset + 4]!,
        s11_db: values[offset + 5]!,
      };
    });
    const resonant = frequencyData.reduce((best, point) =>
      Math.abs(point.impedance_imag_ohms) < Math.abs(best.impedance_imag_ohms) ? point : best,
    frequencyData[0]!);
    return {
      simulation_id: `wasm-fem-${Date.now().toString(36)}`,
      engine: "wasm-fem1d",
      computed_in_ms: Math.round(performance.now() - started),
      mesh: {
        elements,
        nodes: elements + 1,
        length_m: lengthM,
        warnings: [],
      },
      frequency_data: frequencyData,
      resonant_frequency_mhz: resonant.frequency_mhz,
      warnings: ["wasm-fem1d is a 1D port-axis linear-element model; 3D H(curl) edge elements remain a future solver milestone."],
    };
  } finally {
    module._free(outputPointer);
  }
}

self.onmessage = async (event: MessageEvent<FEMWorkerMessage>) => {
  const message = event.data;
  if (message.type !== "fem_simulate") return;
  try {
    self.postMessage({ type: "success", id: message.id, result: await solve(message.request) } satisfies FEMWorkerSuccess);
  } catch (error) {
    self.postMessage({ type: "error", id: message.id, message: error instanceof Error ? error.message : String(error) } satisfies FEMWorkerError);
  }
};
