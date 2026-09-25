import { describe, expect, it } from "vitest";
import {
  adaptFEMFrequencyData,
  buildFEMFrequencies,
  computeFEMSWR,
  createStatusOnlyFEMResult,
  femGainPatternToNecPattern,
  getFEMGainAvailabilityMessage,
  getRenderableFEMGainPoints,
  HCURL_CAPABILITIES,
  selectFEMPort,
  toNECFrequencyResults,
  validateFEMSweep,
} from "../fem";
import type { FEMGainPattern, FEMSimulationRequest } from "../fem";

const request: FEMSimulationRequest = {
  model_ir: {
    version: 1,
    paramSchemaVersion: 1,
    materials: [{ name: "air", epsilon_r: 1, mu_r: 1 }],
    entities: [{ id: "volume", kind: "box", center: [0, 0, 0], size: [0.1, 0.1, 0.1], material: "air" }],
    ports: [{ id: "feed", kind: "lumpedPort", positive: [0, 0, 0.002], negative: [0, 0, 0], impedance_ohms: 50 }],
    boundaries: [],
    meshHints: [],
  },
  start_mhz: 900,
  stop_mhz: 936,
  steps: 5,
  mesh_size_m: 0.0005,
  reference_impedance_ohms: 50,
};

describe("browser FEM contract", () => {
  it("selects a validated lumped port", () => {
    expect(selectFEMPort(request).id).toBe("feed");
  });

  it("rejects an invalid ModelIR before entering WASM", () => {
    expect(() => selectFEMPort({
      ...request,
      model_ir: {
        ...request.model_ir,
        entities: [{ ...request.model_ir.entities[0]!, material: "missing" }],
      },
    })).toThrow(/entities.*material/);
  });

  it("derives SWR from S11 and handles an invalid magnitude explicitly", () => {
    expect(computeFEMSWR(0, 0)).toBe(1);
    expect(computeFEMSWR(0.5, 0)).toBeCloseTo(3);
    expect(computeFEMSWR(1, 0)).toBeNull();
    expect(adaptFEMFrequencyData([{ ...requestPoint }])[0]!.swr).toBe(1);
  });

  it("maps invalid wire numbers to point_status=non_finite with null payloads", () => {
    const mapped = adaptFEMFrequencyData([{
      ...requestPoint,
      point_status: "ok",
      s11_real: Number.NaN,
      s11_imag: 0,
    }]);
    expect(mapped[0]!.point_status).toBe("non_finite");
    expect(mapped[0]!.s11_real).toBeNull();
    expect(mapped[0]!.swr).toBeNull();
  });

  it("skips invalid FEM points when adapting chart inputs and never fabricates gain", () => {
    const frequency_data = adaptFEMFrequencyData([
      { ...requestPoint, point_status: "ok" },
      {
        ...requestPoint,
        frequency_mhz: 919,
        point_status: "solver_error",
        impedance_real_ohms: null,
        impedance_imag_ohms: null,
        s11_real: null,
        s11_imag: null,
        s11_db: null,
        swr: null,
      },
    ]);
    const chartData = toNECFrequencyResults({
      simulation_id: "sim",
      engine: "hcurl-wasm",
      computed_in_ms: 1,
      mesh: { nodes: 1, elements: 1, min_quality: 1, material_count: 1, boundary_count: 0, warnings: [] },
      reference_impedance_ohms: 50,
      frequency_data,
      resonant_frequency_mhz: 918,
      warnings: [],
      capabilities: HCURL_CAPABILITIES,
    });
    expect(chartData).toHaveLength(1);
    expect(Number.isNaN(chartData[0]!.gain_max_dbi)).toBe(true);
  });

  it("creates cancellation and timeout status-only results without NaN/Infinity on the wire", () => {
    const cancelled = createStatusOnlyFEMResult(request, "cancelled", "cancelled by user");
    const timedOut = createStatusOnlyFEMResult(request, "timeout", "timed out");
    expect(cancelled.frequency_data.every((point) => point.point_status === "cancelled")).toBe(true);
    expect(timedOut.frequency_data.every((point) => point.point_status === "timeout")).toBe(true);
    expect(JSON.stringify(cancelled)).not.toContain("Infinity");
    expect(JSON.stringify(cancelled)).not.toContain("NaN");
  });

  it("keeps gain rendering capability-gated and supports enabled payloads", () => {
    const disabled = createStatusOnlyFEMResult(request, "cancelled", "cancelled by user");
    expect(getFEMGainAvailabilityMessage(disabled)).toMatch(/unavailable/i);
    expect(getRenderableFEMGainPoints(disabled)).toHaveLength(0);

    const enabled = {
      ...disabled,
      capabilities: { ...disabled.capabilities, gain: true },
      gain_data: {
        available: true,
        points: [
          {
            frequency_mhz: 918,
            point_status: "ok" as const,
            gain_dbi: 5.2,
            theta_deg: 15,
            phi_deg: 90,
          },
        ],
      },
    };
    expect(getFEMGainAvailabilityMessage(enabled)).toMatch(/available/i);
    expect(getRenderableFEMGainPoints(enabled)).toHaveLength(1);
  });

  it("keeps the first UI sweep bounded and capabilities honest", () => {
    expect(() => validateFEMSweep(900, 936, 202)).toThrow(/201/);
    expect(buildFEMFrequencies(900, 936, 5)).toHaveLength(5);
    expect(HCURL_CAPABILITIES.gain).toBe(false);
    expect(HCURL_CAPABILITIES.far_field).toBe(false);
  });

  it("converts FEM far-field patterns to NEC PatternData for the 3D view", () => {
    const pattern: FEMGainPattern = {
      frequency_mhz: 908,
      point_status: "ok",
      theta_deg: [0, 90, 180],
      phi_deg: [0, 180],
      gain_dbi: [
        [5, 5],
        [3, 3],
        [-10, -10],
      ],
      directivity_dbi: [
        [5, 5],
        [3, 3],
        [-10, -10],
      ],
      max_gain_dbi: 5,
      max_gain_theta_deg: 0,
      max_gain_phi_deg: 0,
      max_directivity_dbi: 5,
      radiated_power_w: 1,
      accepted_power_w: 1,
      radiation_efficiency: 1,
    };
    const nec = femGainPatternToNecPattern(pattern);
    expect(nec).not.toBeNull();
    expect(nec?.theta_start).toBe(0);
    expect(nec?.theta_step).toBe(90);
    expect(nec?.theta_count).toBe(3);
    expect(nec?.phi_start).toBe(0);
    expect(nec?.phi_step).toBe(180);
    expect(nec?.phi_count).toBe(2);
    // [theta][phi] layout preserved.
    expect(nec?.gain_dbi[0]?.[0]).toBe(5);
    expect(nec?.gain_dbi[2]?.[1]).toBe(-10);
    // Rejects degenerate grids.
    expect(femGainPatternToNecPattern({ ...pattern, theta_deg: [0] })).toBeNull();
    expect(femGainPatternToNecPattern({ ...pattern, gain_dbi: [[1]] })).toBeNull();
  });
});

const requestPoint = {
  frequency_mhz: 918,
  point_status: "ok" as const,
  impedance_real_ohms: 50,
  impedance_imag_ohms: 0,
  s11_real: 0,
  s11_imag: 0,
  s11_db: null,
  swr: null,
};