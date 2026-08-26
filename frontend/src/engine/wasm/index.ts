/**
 * WasmEngine — SimulationEngine implementation using nec2c compiled to WebAssembly.
 *
 * Runs simulations entirely in the browser via Web Workers:
 * - `simulate` / `simulateAdvanced` → worker.ts
 * - `optimize` → worker-optimizer.ts
 * - `importFile` / `exportFile` → inline (no WASM needed, pure TS parsers)
 *
 * Used when VITE_ENGINE=wasm (GitHub Pages / static deployment).
 */

import type { SimulationResult } from "../../api/nec";
import type {
  SimulationEngine,
  SimulateRequest,
  SimulateAdvancedRequest,
  ImportResult,
  ExportData,
  OptimizationRequest,
  OptimizationProgress,
  OptimizationResult,
} from "../types";
import { parseNecFile } from "../parsers/nec-file";
import { parseMaa } from "../parsers/maa-import";
import { exportMaa } from "../parsers/maa-export";
import { buildCardDeck } from "../parsers/nec-input";
import { FEMRunAbortedError, createStatusOnlyFEMResult, type FEMSimulationRequest, type FEMSimulationResult } from "../fem";
import type { FEMWorkerMessage, FEMWorkerResponse } from "./worker-fem";
import type { MfemSmokeRequest, MfemSmokeResponse } from "./worker-mfem";
import type {
  WorkerRequest,
  WorkerResponse,
} from "./worker";
import type {
  OptimizerWorkerRequest,
  OptimizerWorkerResponse,
} from "./worker-optimizer";
import {
  assertSupportedFrequencyMhz,
  assertSupportedFrequencyRange,
} from "../limits";

const MFEM_PER_POINT_TIMEOUT_MS = 30000;

// ---------------------------------------------------------------------------
// Worker pool (single worker per type, lazy-initialized)
// ---------------------------------------------------------------------------

let simWorker: Worker | null = null;
let optWorker: Worker | null = null;
let femWorker: Worker | null = null;
let mfemWorker: Worker | null = null;
let femWorkerGeneration = 0;

function getSimWorker(): Worker {
  if (!simWorker) {
    simWorker = new Worker(
      new URL("./worker.ts", import.meta.url),
      { type: "module" },
    );
  }
  return simWorker;
}

function getOptWorker(): Worker {
  if (!optWorker) {
    optWorker = new Worker(
      new URL("./worker-optimizer.ts", import.meta.url),
      { type: "module" },
    );
  }
  return optWorker;
}

function getFemWorker(): Worker {
  if (!femWorker) {
    femWorker = new Worker(new URL("./worker-fem.ts", import.meta.url), { type: "module" });
    femWorkerGeneration += 1;
  }
  return femWorker;
}

function resetFemWorker(): void {
  femWorker?.terminate();
  femWorker = null;
  femWorkerListenerAttached = false;
  getFemWorker();
}

function getMfemWorker(): Worker {
  if (!mfemWorker) mfemWorker = new Worker(new URL("./worker-mfem.ts", import.meta.url), { type: "module" });
  return mfemWorker;
}

// ---------------------------------------------------------------------------
// Pending request tracking
// ---------------------------------------------------------------------------

/** Map from request ID to { resolve, reject } for pending simulation requests. */
const pendingRequests = new Map<
  string,
  { resolve: (r: SimulationResult) => void; reject: (e: Error) => void }
>();

/** Counter for generating unique request IDs */
let nextRequestId = 0;

function generateId(): string {
  return `req-${++nextRequestId}-${Date.now().toString(36)}`;
}

const pendingFEMRequests = new Map<string, {
  generation: number;
  request: FEMSimulationRequest;
  resolve: (result: FEMSimulationResult) => void;
  reject: (error: Error) => void;
  timeoutId: ReturnType<typeof setTimeout>;
}>();
let femWorkerListenerAttached = false;
const pendingMfemSmokeRequests = new Map<string, { resolve: (trueDofs: number) => void; reject: (error: Error) => void }>();
let mfemWorkerListenerAttached = false;

function ensureFEMWorkerListener(): void {
  if (femWorkerListenerAttached) return;
  femWorkerListenerAttached = true;
  getFemWorker().addEventListener("message", (event: MessageEvent<FEMWorkerResponse>) => {
    if (event.data.generation !== femWorkerGeneration) return;
    const pending = pendingFEMRequests.get(event.data.id);
    if (!pending) return;
    clearTimeout(pending.timeoutId);
    pendingFEMRequests.delete(event.data.id);
    if (event.data.type === "success") pending.resolve(event.data.result);
    else pending.reject(new Error(event.data.message));
  });
  getFemWorker().addEventListener("error", (event) => {
    const error = new Error(`FEM worker error: ${event.message}`);
    for (const [id, pending] of pendingFEMRequests) {
      clearTimeout(pending.timeoutId);
      pending.reject(error);
      pendingFEMRequests.delete(id);
    }
    resetFemWorker();
  });
}

function rejectPendingFemRequestsAsAborted(kind: "cancelled" | "timeout", message: string): void {
  for (const [id, pending] of pendingFEMRequests) {
    clearTimeout(pending.timeoutId);
    pending.reject(new FEMRunAbortedError(kind, message, createStatusOnlyFEMResult(pending.request, kind, message)));
    pendingFEMRequests.delete(id);
  }
}

function ensureMfemWorkerListener(): void {
  if (mfemWorkerListenerAttached) return;
  mfemWorkerListenerAttached = true;
  getMfemWorker().addEventListener("message", (event: MessageEvent<MfemSmokeResponse>) => {
    const pending = pendingMfemSmokeRequests.get(event.data.id);
    if (!pending) return;
    pendingMfemSmokeRequests.delete(event.data.id);
    if (event.data.type === "success") pending.resolve(event.data.trueDofs);
    else pending.reject(new Error(event.data.message));
  });
  getMfemWorker().addEventListener("error", (event) => {
    const error = new Error(`MFEM worker error: ${event.message}`);
    for (const [id, pending] of pendingMfemSmokeRequests) { pending.reject(error); pendingMfemSmokeRequests.delete(id); }
    mfemWorker = null;
    mfemWorkerListenerAttached = false;
  });
}

// ---------------------------------------------------------------------------
// Worker message handlers (set up lazily)
// ---------------------------------------------------------------------------

let simWorkerListenerAttached = false;

function ensureSimWorkerListener(): void {
  if (simWorkerListenerAttached) return;
  simWorkerListenerAttached = true;

  getSimWorker().addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
    const msg = event.data;
    const pending = pendingRequests.get(msg.id);
    if (!pending) return;

    pendingRequests.delete(msg.id);

    if (msg.type === "success") {
      pending.resolve(msg.result);
    } else {
      pending.reject(new Error(msg.message));
    }
  });

  getSimWorker().addEventListener("error", (event) => {
    // If the worker crashes, reject all pending requests
    const error = new Error(`Simulation worker error: ${event.message}`);
    for (const [id, pending] of pendingRequests) {
      pending.reject(error);
      pendingRequests.delete(id);
    }
    // Reset worker so it can be re-created
    simWorker = null;
    simWorkerListenerAttached = false;
  });
}

// ---------------------------------------------------------------------------
// WasmEngine
// ---------------------------------------------------------------------------

export class WasmEngine implements SimulationEngine {
  cancelFEM(): void {
    rejectPendingFemRequestsAsAborted("cancelled", "FEM run cancelled");
    resetFemWorker();
  }

  runMfemSmoke(): Promise<number> {
    ensureMfemWorkerListener();
    const id = generateId();
    const message: MfemSmokeRequest = { type: "mfem_smoke", id };
    return new Promise<number>((resolve, reject) => {
      pendingMfemSmokeRequests.set(id, { resolve, reject });
      getMfemWorker().postMessage(message);
    });
  }

  async simulateFEM(request: FEMSimulationRequest): Promise<FEMSimulationResult> {
    ensureFEMWorkerListener();
    const id = generateId();
    const generation = femWorkerGeneration;
    const timeoutMs = Math.max(MFEM_PER_POINT_TIMEOUT_MS, request.steps * MFEM_PER_POINT_TIMEOUT_MS);
    const message: FEMWorkerMessage = { type: "fem_simulate", id, generation, request };
    return new Promise<FEMSimulationResult>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        const pending = pendingFEMRequests.get(id);
        if (!pending || pending.generation !== generation) return;
        rejectPendingFemRequestsAsAborted("timeout", `FEM run timed out after ${timeoutMs} ms`);
        resetFemWorker();
      }, timeoutMs);
      pendingFEMRequests.set(id, { generation, request, resolve, reject, timeoutId });
      getFemWorker().postMessage(message);
    });
  }

  /**
   * V1: Basic simulation (template mode, single excitation).
   *
   * Converts to a SimulateAdvancedRequest and delegates to simulateAdvanced.
   */
  async simulate(request: SimulateRequest): Promise<SimulationResult> {
    const advancedRequest: SimulateAdvancedRequest = {
      wires: request.wires,
      excitations: [
        {
          wire_tag: request.excitation.wire_tag,
          segment: request.excitation.segment,
          voltage_real: request.excitation.voltage_real,
          voltage_imag: request.excitation.voltage_imag,
        },
      ],
      ground: request.ground,
      frequency: request.frequency,
      frequencySegments: request.frequencySegments,
      compute_currents: true,
      near_field: {
        plane: "horizontal",
        height_m: 1.8,
        extent_m: 20.0,
        resolution_m: 0.5,
      },
      pattern_step: request.patternStep,
      comment: "AntennaSim simulation",
    };

    return this.simulateAdvanced(advancedRequest);
  }

  /**
   * V2: Advanced simulation (editor mode).
   *
   * Sends the request to the simulation Web Worker and awaits the result.
   */
  async simulateAdvanced(
    request: SimulateAdvancedRequest,
  ): Promise<SimulationResult> {
    assertSupportedFrequencyRange(request.frequency, request.frequencySegments);
    ensureSimWorkerListener();

    const id = generateId();

    const message: WorkerRequest = {
      type: "simulate",
      id,
      request,
    };

    return new Promise<SimulationResult>((resolve, reject) => {
      pendingRequests.set(id, { resolve, reject });
      getSimWorker().postMessage(message);
    });
  }

  /**
   * Import a .nec or .maa file.
   *
   * Runs entirely on the main thread using the TS parsers (no WASM needed).
   */
  async importFile(
    content: string,
    format: "nec" | "maa",
  ): Promise<ImportResult> {
    if (format === "nec") {
      return parseNecFile(content);
    } else {
      return parseMaa(content);
    }
  }

  /**
   * Export to .nec or .maa format.
   *
   * Runs entirely on the main thread using the TS generators.
   */
  async exportFile(data: ExportData, format: "nec" | "maa"): Promise<string> {
    if (format === "maa") {
      const centerFreq =
        (data.frequency_start_mhz + data.frequency_stop_mhz) / 2;
      return exportMaa(
        data.title,
        data.wires,
        data.excitations,
        data.loads,
        centerFreq,
      );
    }

    // NEC format: use buildCardDeck to generate the card deck
    const cardDeck = buildCardDeck({
      wires: data.wires,
      excitations: data.excitations,
      ground: data.ground,
      frequency: {
        start_mhz: data.frequency_start_mhz,
        stop_mhz: data.frequency_stop_mhz,
        steps: data.frequency_steps,
      },
      loads: data.loads,
      transmission_lines: data.transmission_lines,
      compute_currents: true,
      comment: data.title || "AntennaSim export",
    });
    return cardDeck;
  }

  /**
   * Run optimizer with progress callback.
   *
   * Spawns a dedicated Web Worker that runs Nelder-Mead with WASM simulations.
   * Returns a cancel function and a result promise.
   */
  async optimize(
    request: OptimizationRequest,
    onProgress: (progress: OptimizationProgress) => void,
  ): Promise<{ result: Promise<OptimizationResult>; cancel: () => void }> {
    assertSupportedFrequencyMhz(
      request.frequency_start_mhz,
      "Optimizer start frequency",
    );
    assertSupportedFrequencyMhz(
      request.frequency_stop_mhz,
      "Optimizer stop frequency",
    );
    if (request.target_frequency_mhz !== undefined) {
      assertSupportedFrequencyMhz(
        request.target_frequency_mhz,
        "Optimizer target frequency",
      );
    }
    // Create a fresh optimizer worker for each run
    // (we terminate the old one if present to free resources)
    if (optWorker) {
      optWorker.terminate();
      optWorker = null;
    }

    const worker = getOptWorker();
    const id = generateId();

    let cancel = () => {
      const cancelMsg: OptimizerWorkerRequest = { type: "cancel", id };
      worker.postMessage(cancelMsg);
    };

    const resultPromise = new Promise<OptimizationResult>(
      (resolve, reject) => {
        const handleMessage = (event: MessageEvent<OptimizerWorkerResponse>) => {
          const msg = event.data;
          if (msg.id !== id) return;

          switch (msg.type) {
            case "progress":
              onProgress(msg.data);
              break;
            case "result":
              cleanup();
              resolve(msg.data);
              break;
            case "error":
              cleanup();
              reject(new Error(msg.message));
              break;
          }
        };

        const handleError = (event: ErrorEvent) => {
          cleanup();
          reject(new Error(`Optimizer worker error: ${event.message}`));
        };

        const cleanup = () => {
          worker.removeEventListener("message", handleMessage);
          worker.removeEventListener("error", handleError);
        };

        worker.addEventListener("message", handleMessage);
        worker.addEventListener("error", handleError);

        cancel = () => {
          const cancelMsg: OptimizerWorkerRequest = { type: "cancel", id };
          worker.postMessage(cancelMsg);
          cleanup();
          reject(new Error("Optimization cancelled"));
        };
      },
    );

    // Send the start message
    const startMsg: OptimizerWorkerRequest = {
      type: "start",
      id,
      request,
    };
    worker.postMessage(startMsg);

    return { result: resultPromise, cancel };
  }
}
