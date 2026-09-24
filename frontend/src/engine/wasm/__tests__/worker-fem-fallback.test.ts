import { describe, expect, it, vi } from "vitest";
import type { FEMSimulationRequest } from "../../fem";
import type { FEMWorkerMessage, FEMWorkerResponse } from "../worker-fem";

// Exercise the TypeScript fallback path in worker-fem.ts by running the
// worker module in Node with `self` stubbed and the native fetch forced to fail.
describe("worker-fem TypeScript fallback", () => {
  it("falls back to solveFEMPoint when the native module is unavailable", async () => {
    const posted: FEMWorkerResponse[] = [];
    const selfStub = {
      postMessage: (response: FEMWorkerResponse) => {
        posted.push(response);
      },
      onmessage: null as ((event: MessageEvent<FEMWorkerMessage>) => void) | null,
    };
    vi.stubGlobal("self", selfStub);
    // Force native load failure.
    vi.stubGlobal("fetch", () => Promise.reject(new Error("native unavailable")));

    await import("../worker-fem");
    expect(typeof selfStub.onmessage).toBe("function");

    const request: FEMSimulationRequest = {
      model_ir: {
        version: 1,
        paramSchemaVersion: 1,
        materials: [{ name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 }],
        entities: [
          { id: "arm1", kind: "box", center: [-0.011, 0, 0], size: [0.02, 0.002, 0.002], material: "copper" },
          { id: "arm2", kind: "box", center: [0.011, 0, 0], size: [0.02, 0.002, 0.002], material: "copper" },
        ],
        ports: [
          { id: "feed", kind: "lumpedPort", positive: [-0.001, 0, 0], negative: [0.001, 0, 0], impedance_ohms: 50 },
        ],
        boundaries: [{ kind: "radiation", padding_m: 0.02 }],
        meshHints: [],
      },
      start_mhz: 3000,
      stop_mhz: 3000,
      steps: 1,
      mesh_size_m: 0.006,
      reference_impedance_ohms: 50,
    };

    selfStub.onmessage?.({
      data: { type: "fem_simulate", id: "test-1", generation: 1, request } as FEMWorkerMessage,
    } as MessageEvent<FEMWorkerMessage>);

    // Wait for the async worker to post its response.
    const deadline = Date.now() + 120000;
    while (posted.length === 0 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    expect(posted.length).toBe(1);
    const response = posted[0];
    expect(response?.type).toBe("success");
    if (response?.type !== "success") return;
    expect(response.result.engine).toBe("fem-typescript");
    // Fallback warning is surfaced.
    expect(response.result.warnings.some((w) => w.includes("TypeScript"))).toBe(true);
    // One frequency point with mapped impedance/S11.
    const points = response.result.frequency_data;
    expect(points.length).toBe(1);
    const point = points[0];
    expect(point?.point_status).toBe("ok");
    expect(Number.isFinite(point?.impedance_real_ohms ?? Number.NaN)).toBe(true);
    expect(Number.isFinite(point?.impedance_imag_ohms ?? Number.NaN)).toBe(true);
    // Far-field is computed by default and mapped into gain_data.
    expect(response.result.capabilities.gain).toBe(true);
    expect(response.result.capabilities.far_field).toBe(true);
    const gainData = response.result.gain_data;
    expect(gainData?.available).toBe(true);
    expect(gainData?.patterns?.length).toBe(1);
    const pattern = gainData?.patterns?.[0];
    expect(pattern?.frequency_mhz).toBe(3000);
    expect(Number.isFinite(pattern?.max_gain_dbi ?? Number.NaN)).toBe(true);
    expect(pattern?.theta_deg.length).toBeGreaterThan(0);
    expect(pattern?.gain_dbi.length).toBe(pattern?.theta_deg.length);
    // Backwards-compatible max-gain points are populated for existing UI.
    expect(gainData?.points?.length).toBe(1);
    expect(gainData?.points?.[0]?.gain_dbi).toBe(pattern?.max_gain_dbi);

    vi.unstubAllGlobals();
  }, 180000);
});
