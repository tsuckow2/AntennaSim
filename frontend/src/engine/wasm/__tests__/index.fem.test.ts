import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { type FEMSimulationRequest, type FEMSimulationResult, MFEM_CAPABILITIES } from "../../fem";
import type { FEMWorkerMessage, FEMWorkerResponse } from "../worker-fem";

class MockWorker {
  static instances: MockWorker[] = [];

  terminated = false;
  lastPosted: unknown;
  private listeners: { message: Array<(event: MessageEvent<FEMWorkerResponse>) => void>; error: Array<(event: ErrorEvent) => void> } = {
    message: [],
    error: [],
  };

  constructor(..._args: unknown[]) {
    MockWorker.instances.push(this);
  }

  addEventListener(type: "message" | "error", callback: ((event: MessageEvent<FEMWorkerResponse>) => void) | ((event: ErrorEvent) => void)): void {
    if (type === "message") this.listeners.message.push(callback as (event: MessageEvent<FEMWorkerResponse>) => void);
    else this.listeners.error.push(callback as (event: ErrorEvent) => void);
  }

  removeEventListener(_type: "message" | "error", _callback: unknown): void {
    // No-op for this harness.
  }

  postMessage(message: unknown): void {
    this.lastPosted = message;
  }

  terminate(): void {
    this.terminated = true;
  }

  emitMessage(data: FEMWorkerResponse): void {
    const event = { data } as MessageEvent<FEMWorkerResponse>;
    for (const listener of this.listeners.message) listener(event);
  }

  emitError(message: string): void {
    const event = { message } as ErrorEvent;
    for (const listener of this.listeners.error) listener(event);
  }
}

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
  stop_mhz: 900,
  steps: 1,
  mesh_size_m: 0.001,
  reference_impedance_ohms: 50,
};

const okResult: FEMSimulationResult = {
  simulation_id: "sim",
  engine: "mfem-wasm",
  computed_in_ms: 1,
  mesh: { nodes: 1, elements: 1, min_quality: 1, material_count: 1, boundary_count: 0, warnings: [] },
  reference_impedance_ohms: 50,
  frequency_data: [{
    frequency_mhz: 900,
    point_status: "ok",
    impedance_real_ohms: 50,
    impedance_imag_ohms: 0,
    s11_real: 0,
    s11_imag: 0,
    s11_db: -30,
    swr: 1,
  }],
  resonant_frequency_mhz: 900,
  warnings: [],
  capabilities: MFEM_CAPABILITIES,
};

describe("WasmEngine FEM cancellation/timeout coordinator", () => {
  beforeEach(() => {
    vi.resetModules();
    MockWorker.instances = [];
    vi.stubGlobal("Worker", MockWorker as unknown as typeof Worker);
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("terminates/recreates FEM worker and rejects with cancelled partial result", async () => {
    const { WasmEngine } = await import("../index");
    const { FEMRunAbortedError } = await import("../../fem");
    const engine = new WasmEngine();
    const promise = engine.simulateFEM!(request);
    void promise.catch(() => undefined);
    expect(MockWorker.instances.length).toBe(1);

    engine.cancelFEM!();
    await expect(promise).rejects.toBeInstanceOf(FEMRunAbortedError);
    const aborted = await promise.catch((error) => error as InstanceType<typeof FEMRunAbortedError>);
    if (!(aborted instanceof FEMRunAbortedError)) throw new Error("expected FEMRunAbortedError");
    expect(aborted.kind).toBe("cancelled");
    expect(aborted.partialResult.frequency_data[0]!.point_status).toBe("cancelled");

    expect(MockWorker.instances[0]!.terminated).toBe(true);
    expect(MockWorker.instances.length).toBe(2);
  });

  it("ignores stale generation messages after cancellation and resolves fresh generation", async () => {
    const { WasmEngine } = await import("../index");
    const { FEMRunAbortedError } = await import("../../fem");
    const engine = new WasmEngine();
    const first = engine.simulateFEM!(request);
    void first.catch(() => undefined);
    const worker1 = MockWorker.instances[0]!;
    const msg1 = worker1.lastPosted as FEMWorkerMessage;

    engine.cancelFEM!();
    await expect(first).rejects.toBeInstanceOf(FEMRunAbortedError);

    worker1.emitMessage({ type: "success", id: msg1.id, generation: msg1.generation, result: okResult });

    const second = engine.simulateFEM!(request);
    const worker2 = MockWorker.instances[1]!;
    const msg2 = worker2.lastPosted as FEMWorkerMessage;
    worker2.emitMessage({ type: "success", id: msg2.id, generation: msg2.generation, result: okResult });

    await expect(second).resolves.toEqual(okResult);
  });

  it("marks pending requests as timeout and recreates worker", async () => {
    vi.useFakeTimers();
    const { WasmEngine } = await import("../index");
    const { FEMRunAbortedError } = await import("../../fem");
    const engine = new WasmEngine();
    const promise = engine.simulateFEM!(request);
    void promise.catch(() => undefined);

    await vi.advanceTimersByTimeAsync(30001);
    await expect(promise).rejects.toBeInstanceOf(FEMRunAbortedError);
    expect(MockWorker.instances[0]!.terminated).toBe(true);
    expect(MockWorker.instances.length).toBe(2);
  });

  it("propagates current-generation worker errors", async () => {
    const { WasmEngine } = await import("../index");
    const engine = new WasmEngine();
    const promise = engine.simulateFEM!(request);
    void promise.catch(() => undefined);
    const worker = MockWorker.instances[0]!;
    worker.emitError("boom");
    await expect(promise).rejects.toThrow(/FEM worker error: boom/);
  });
});
