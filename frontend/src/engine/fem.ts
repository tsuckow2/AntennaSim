import type { ModelIR } from "../dsl/model-ir";
import { validateModelIR } from "../dsl/model-ir";

export interface FEMFrequencyPoint {
  frequency_mhz: number;
  impedance_real_ohms: number;
  impedance_imag_ohms: number;
  s11_real: number;
  s11_imag: number;
  s11_db: number;
}

export interface FEMSimulationRequest {
  model_ir: ModelIR;
  start_mhz: number;
  stop_mhz: number;
  steps: number;
  mesh_size_m: number;
  reference_impedance_ohms: number;
  port_id?: string;
}

export interface FEMSimulationResult {
  simulation_id: string;
  engine: "wasm-fem1d";
  computed_in_ms: number;
  mesh: { elements: number; nodes: number; length_m: number; warnings: string[] };
  frequency_data: FEMFrequencyPoint[];
  resonant_frequency_mhz: number | null;
  warnings: string[];
}

export function selectFEMPort(request: FEMSimulationRequest): { positive: [number, number, number]; negative: [number, number, number]; id: string } {
  const diagnostics = validateModelIR(request.model_ir);
  if (diagnostics.some((diagnostic) => diagnostic.severity === "error")) {
    const first = diagnostics.find((diagnostic) => diagnostic.severity === "error")!;
    throw new Error(`Invalid ModelIR: ${first.path} ${first.message}`);
  }
  const port = request.model_ir.ports.find((candidate) => !request.port_id || candidate.id === request.port_id);
  if (!port) throw new Error(`Unknown lumped port: ${request.port_id ?? "first port"}`);
  return port;
}
