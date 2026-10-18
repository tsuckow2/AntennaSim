import type { ModelIR } from "../dsl/model-ir";
import { validateModelIR } from "../dsl/model-ir";
import type { FEMMeshArtifact } from "./mesh";
import type { FrequencyResult } from "../api/nec";
import { identifyResonance, type ReactanceCrossing } from "../utils/resonance";

export type FEMPointStatus = "ok" | "non_finite" | "solver_error" | "cancelled" | "timeout";

export interface FEMGainPoint {
  frequency_mhz: number;
  point_status: FEMPointStatus;
  point_message?: string;
  gain_dbi: number | null;
  theta_deg: number | null;
  phi_deg: number | null;
}

/** Full far-field pattern for one frequency point (gain-plot data). */
export interface FEMGainPattern {
  frequency_mhz: number;
  point_status: FEMPointStatus;
  /** Polar angles in degrees, 0..180. */
  theta_deg: number[];
  /** Azimuth angles in degrees, 0..<360. */
  phi_deg: number[];
  /** IEEE gain in dBi on the [theta][phi] grid. */
  gain_dbi: number[][];
  /** Directivity in dBi on the [theta][phi] grid. */
  directivity_dbi: number[][];
  max_gain_dbi: number;
  max_gain_theta_deg: number;
  max_gain_phi_deg: number;
  max_directivity_dbi: number;
  radiated_power_w: number;
  accepted_power_w: number;
  radiation_efficiency: number;
}

export interface FEMGainData {
  available: boolean;
  message?: string;
  points?: FEMGainPoint[];
  /** Per-frequency far-field patterns for gain plots, when computed. */
  patterns?: FEMGainPattern[];
}

export interface FEMFrequencyPoint {
  frequency_mhz: number;
  point_status: FEMPointStatus;
  point_message?: string;
  impedance_real_ohms: number | null;
  impedance_imag_ohms: number | null;
  s11_real: number | null;
  s11_imag: number | null;
  s11_db: number | null;
  swr: number | null;
}

export interface FEMSimulationRequest {
  model_ir: ModelIR;
  start_mhz: number;
  stop_mhz: number;
  steps: number;
  mesh_size_m: number;
  reference_impedance_ohms: number;
  port_id?: string;
  /**
   * Far-field request for the TypeScript fallback solver. When omitted the
   * fallback computes a 5-degree pattern per frequency point; set enabled to
   * false to skip it. Ignored by the native hcurl path.
   */
  far_field?: {
    enabled: boolean;
    theta_step_deg?: number;
    phi_step_deg?: number;
  };
}

export interface FEMSimulationResult {
  simulation_id: string;
  engine: "hcurl-wasm" | "fem-typescript";
  computed_in_ms: number;
  /**
   * Rayon worker thread count used by a native H(curl) solve. Absent for the
   * TypeScript fallback (and for results produced before this field existed).
   */
  threads?: number;
  mesh: FEMMeshArtifact["statistics"] & { warnings: string[] };
  reference_impedance_ohms: number;
  frequency_data: FEMFrequencyPoint[];
  resonant_frequency_mhz: number | null;
  warnings: string[];
  gain_data?: FEMGainData;
  capabilities: {
    impedance: boolean;
    s11: boolean;
    swr: boolean;
    smith_chart: boolean;
    near_field: boolean;
    far_field: boolean;
    gain: boolean;
    port_status: boolean;
    mesh_status: boolean;
  };
  port_status?: {
    port_id: string;
    positive: [number, number, number];
    negative: [number, number, number];
    status: "ok" | "error" | "missing";
    message?: string;
  }[];
  mesh_status?: {
    status: "ok" | "error" | "missing";
    message?: string;
  };
}

export class FEMRunAbortedError extends Error {
  kind: "cancelled" | "timeout";
  partialResult: FEMSimulationResult;

  constructor(kind: "cancelled" | "timeout", message: string, partialResult: FEMSimulationResult) {
    super(message);
    this.name = "FEMRunAbortedError";
    this.kind = kind;
    this.partialResult = partialResult;
  }
}

function finiteOrNull(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) ? value : null;
}

function deriveS11Db(s11Real: number, s11Imag: number): number | null {
  const magnitude = Math.hypot(s11Real, s11Imag);
  if (!Number.isFinite(magnitude) || magnitude <= 0) return null;
  const db = 20 * Math.log10(magnitude);
  return Number.isFinite(db) ? db : null;
}

export function computeFEMSWR(s11Real: number, s11Imag: number): number | null {
  const magnitude = Math.hypot(s11Real, s11Imag);
  if (!Number.isFinite(magnitude) || magnitude >= 1) return null;
  return (1 + magnitude) / (1 - magnitude);
}

export function adaptFEMFrequencyData(
  points: FEMFrequencyPoint[],
): FEMFrequencyPoint[] {
  return points.map((point) => {
    const pointStatus = point.point_status;
    const re = finiteOrNull(point.impedance_real_ohms);
    const im = finiteOrNull(point.impedance_imag_ohms);
    const s11Real = finiteOrNull(point.s11_real);
    const s11Imag = finiteOrNull(point.s11_imag);
    const invalid = pointStatus === "ok" && (re == null || im == null || s11Real == null || s11Imag == null);
    if (pointStatus !== "ok" || invalid) {
      return {
        ...point,
        point_status: pointStatus === "ok" ? "non_finite" : pointStatus,
        impedance_real_ohms: null,
        impedance_imag_ohms: null,
        s11_real: null,
        s11_imag: null,
        s11_db: null,
        swr: null,
      };
    }
    const safeS11Real = s11Real as number;
    const safeS11Imag = s11Imag as number;
    return {
      ...point,
      impedance_real_ohms: re,
      impedance_imag_ohms: im,
      s11_real: safeS11Real,
      s11_imag: safeS11Imag,
      s11_db: point.s11_db ?? deriveS11Db(safeS11Real, safeS11Imag),
      swr: computeFEMSWR(safeS11Real, safeS11Imag),
    };
  });
}

/**
 * Identify the sweep's resonant frequency from its solved points, or
 * null when no reactance crossing qualifies as a resonance. Delegates
 * to `identifyResonance` (utils/resonance): a crossing must carry at
 * least a quarter of the sweep's peak Re(Z) — rejecting the shallow
 * series-type crossing a patch shows above its parallel resonance —
 * and, among qualifying crossings, the one nearest the minimum-|S11|
 * point wins. Only "ok" points with finite impedance participate.
 */
export function identifyFEMResonance(
  points: readonly FEMFrequencyPoint[],
): ReactanceCrossing | null {
  const samples = points.flatMap((point) => {
    if (
      point.point_status !== "ok"
      || point.impedance_real_ohms == null
      || point.impedance_imag_ohms == null
    ) {
      return [];
    }
    return [{
      frequencyMhz: point.frequency_mhz,
      resistanceOhms: point.impedance_real_ohms,
      reactanceOhms: point.impedance_imag_ohms,
      s11Db: point.s11_db,
    }];
  });
  return identifyResonance(samples);
}

export function toNECFrequencyResults(result: FEMSimulationResult): FrequencyResult[] {
  return result.frequency_data.flatMap((point) => {
    if (
      point.point_status !== "ok"
      || point.impedance_real_ohms == null
      || point.impedance_imag_ohms == null
      || point.s11_real == null
      || point.s11_imag == null
      || point.swr == null
    ) {
      return [];
    }
    return [{
    frequency_mhz: point.frequency_mhz,
    impedance: { real: point.impedance_real_ohms, imag: point.impedance_imag_ohms },
    swr_50: point.swr,
    gain_max_dbi: Number.NaN,
    gain_max_theta: 0,
    gain_max_phi: 0,
    front_to_back_db: null,
    beamwidth_e_deg: null,
    beamwidth_h_deg: null,
    efficiency_percent: null,
    pattern: null,
    currents: null,
  }];
  });
}

export function validateFEMSweep(startMhz: number, stopMhz: number, steps: number): void {
  if (![startMhz, stopMhz, steps].every(Number.isFinite)) throw new Error("FEM sweep values must be finite");
  if (startMhz <= 0 || stopMhz < startMhz) throw new Error("FEM sweep must have positive, ascending frequencies");
  if (!Number.isInteger(steps) || steps < 1 || steps > 201) throw new Error("FEM UI sweeps must contain between 1 and 201 points");
}

export const HCURL_CAPABILITIES: FEMSimulationResult["capabilities"] = {
  impedance: true,
  s11: true,
  swr: true,
  smith_chart: true,
  near_field: false,
  far_field: false,
  gain: false,
  port_status: true,
  mesh_status: true,
};

export function buildFEMFrequencies(startMhz: number, stopMhz: number, steps: number): number[] {
  validateFEMSweep(startMhz, stopMhz, steps);
  if (steps === 1) return [startMhz];
  const out: number[] = [];
  const span = stopMhz - startMhz;
  for (let index = 0; index < steps; index += 1) {
    out.push(startMhz + (span * index) / (steps - 1));
  }
  return out;
}

export function createStatusOnlyFEMResult(
  request: FEMSimulationRequest,
  status: "cancelled" | "timeout",
  message: string,
): FEMSimulationResult {
  const frequency_data = adaptFEMFrequencyData(
    buildFEMFrequencies(request.start_mhz, request.stop_mhz, request.steps).map((frequency_mhz) => ({
      frequency_mhz,
      point_status: status,
      point_message: message,
      impedance_real_ohms: null,
      impedance_imag_ohms: null,
      s11_real: null,
      s11_imag: null,
      s11_db: null,
      swr: null,
    })),
  );
  return {
    simulation_id: `hcurl-${status}-${Date.now().toString(36)}`,
    engine: "hcurl-wasm",
    computed_in_ms: 0,
    mesh: {
      nodes: 0,
      elements: 0,
      min_quality: 0,
      material_count: 0,
      boundary_count: 0,
      warnings: [message],
    },
    reference_impedance_ohms: request.reference_impedance_ohms,
    frequency_data,
    resonant_frequency_mhz: null,
    warnings: [message],
    gain_data: {
      available: false,
      message: "Gain unavailable for this FEM run.",
    },
    capabilities: HCURL_CAPABILITIES,
  };
}

export function getFEMGainAvailabilityMessage(result: FEMSimulationResult): string {
  if (!result.capabilities.gain) return "Gain is unavailable for FEM mode until far-field outputs are validated.";
  if (!result.gain_data?.available) return result.gain_data?.message ?? "Gain capability is enabled but no far-field payload was provided.";
  const renderable = getRenderableFEMGainPoints(result);
  if (renderable.length === 0) return "Gain capability is enabled, but no valid gain points are available for rendering.";
  return "Gain data available.";
}

export function getRenderableFEMGainPoints(result: FEMSimulationResult): Array<{ frequency_mhz: number; gain_dbi: number; theta_deg: number; phi_deg: number }> {
  if (!result.capabilities.gain || !result.gain_data?.available || !result.gain_data.points) return [];
  return result.gain_data.points.flatMap((point) => {
    if (
      point.point_status !== "ok"
      || point.gain_dbi == null
      || point.theta_deg == null
      || point.phi_deg == null
      || !Number.isFinite(point.gain_dbi)
      || !Number.isFinite(point.theta_deg)
      || !Number.isFinite(point.phi_deg)
    ) {
      return [];
    }
    return [{
      frequency_mhz: point.frequency_mhz,
      gain_dbi: point.gain_dbi,
      theta_deg: point.theta_deg,
      phi_deg: point.phi_deg,
    }];
  });
}

/**
 * Convert a FEM far-field pattern to the NEC PatternData shape expected by
 * RadiationPattern3D. Returns null when the pattern grid is missing,
 * non-uniform, or degenerate. The FEM [theta][phi] gain layout matches the
 * NEC component's indexing, and both use theta-from-+Z conventions.
 */
export function femGainPatternToNecPattern(pattern: FEMGainPattern): import("../api/nec").PatternData | null {
  const thetaDeg = pattern.theta_deg;
  const phiDeg = pattern.phi_deg;
  const gainDbi = pattern.gain_dbi;
  if (thetaDeg.length < 2 || phiDeg.length < 2) return null;
  if (gainDbi.length !== thetaDeg.length) return null;
  for (const row of gainDbi) {
    if (!row || row.length !== phiDeg.length) return null;
  }
  const thetaStep = (thetaDeg[1] ?? 0) - (thetaDeg[0] ?? 0);
  const phiStep = (phiDeg[1] ?? 0) - (phiDeg[0] ?? 0);
  if (!(thetaStep > 0) || !(phiStep > 0)) return null;
  // FEM grids are generated uniformly; reject anything else rather than resampling silently.
  for (let i = 2; i < thetaDeg.length; i += 1) {
    if (Math.abs(((thetaDeg[i] ?? 0) - (thetaDeg[i - 1] ?? 0)) - thetaStep) > 1e-9) return null;
  }
  for (let j = 2; j < phiDeg.length; j += 1) {
    if (Math.abs(((phiDeg[j] ?? 0) - (phiDeg[j - 1] ?? 0)) - phiStep) > 1e-9) return null;
  }
  return {
    theta_start: thetaDeg[0] ?? 0,
    theta_step: thetaStep,
    theta_count: thetaDeg.length,
    phi_start: phiDeg[0] ?? 0,
    phi_step: phiStep,
    phi_count: phiDeg.length,
    gain_dbi: gainDbi.map((row) => [...(row ?? [])]),
  };
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
