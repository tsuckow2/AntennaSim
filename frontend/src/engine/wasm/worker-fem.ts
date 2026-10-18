import { buildFEMMesh, FEM_ATTR_PORT_NEGATIVE, FEM_ATTR_PORT_POSITIVE } from "../mesh";
import { adaptFEMFrequencyData, buildFEMCoaxRunConfig, buildFEMFrequencies, HCURL_CAPABILITIES, identifyFEMResonance, selectFEMPort, type FEMPointStatus, type FEMSimulationRequest, type FEMSimulationResult } from "../fem";
import { solveFEMPoint, type FEMSolverInput } from "../fem-solver";
import type { HcurlMtPackage, HcurlSolverModule } from "./hcurl-module";
import { HCURL_ABI_VERSION, packHcurlPointRequest, unpackHcurlFarFieldBlob, unpackHcurlPointResponse } from "./hcurl-abi";

export interface FEMWorkerMessage {
  type: "fem_simulate";
  id: string;
  generation: number;
  request: FEMSimulationRequest;
}

export interface FEMWorkerSuccess {
  type: "success";
  id: string;
  generation: number;
  result: FEMSimulationResult;
}

export interface FEMWorkerError {
  type: "error";
  id: string;
  generation: number;
  message: string;
  stage?: "validating" | "meshing" | "loading-native" | "assembly" | "solving";
  code?: "unsupported_solver_abi" | "invalid_mesh" | "worker_failure";
}

export interface FEMWorkerProgress {
  type: "progress";
  id: string;
  generation: number;
  /** 1-based count of fully solved frequency points (0 while meshing/loading). */
  point_index: number;
  points_total: number;
  stage: "meshing" | "loading-native" | "solving";
}

export type FEMWorkerResponse = FEMWorkerSuccess | FEMWorkerError | FEMWorkerProgress;

/** Emscripten-style raw exports of the classic single-threaded module. */
interface HcurlClassicModule {
  HEAPU8: Uint8Array;
  _hcurl_solver_alloc(sizeBytes: number): number;
  _hcurl_solver_free(ptr: number): void;
  _hcurl_solver_set_cancelled?(flag: number): void;
  _hcurl_solver_run(reqPtr: number, reqLen: number, rspPtrOut: number, rspLenOut: number): number;
  _hcurl_solver_last_error(): string;
}

type HcurlClassicFactory = (options?: Record<string, unknown>) => Promise<HcurlClassicModule>;

/**
 * Threaded build: wasm-pack `--target web` glue plus the wasm-bindgen-rayon
 * thread pool. Needs SharedArrayBuffer, i.e. a cross-origin-isolated page
 * (COOP/COEP response headers, set in nginx).
 */
async function loadThreadedModule(): Promise<HcurlSolverModule> {
  if (!crossOriginIsolated) {
    throw new Error("page is not cross-origin isolated; the threaded solver needs COOP/COEP headers");
  }
  const base = import.meta.env.BASE_URL;
  const pkg = (await import(/* @vite-ignore */ `${base}wasm/hcurl_solver_mt.js`)) as unknown as HcurlMtPackage;
  // Pass the .wasm URL explicitly: the published glue file is renamed from
  // the wasm-pack default, so the glue's built-in filename is stale.
  await pkg.default(`${base}wasm/hcurl_solver_mt_bg.wasm`);
  const threadCount = Math.max(1, navigator.hardwareConcurrency || 1);
  // Spawns the Rayon worker pool; must resolve before the first solve call.
  await pkg.initThreadPool(threadCount);
  return {
    solvePoint: (requestBytes) => pkg.hcurlSolvePoint(requestBytes),
    setCancelled: (flag) => pkg.hcurlSetCancelled(flag),
    lastError: () => pkg.hcurlLastError(),
    threadCount,
  };
}

/** Classic single-threaded build, adapted to the v2 module interface. */
async function loadClassicModule(): Promise<HcurlSolverModule> {
  const base = import.meta.env.BASE_URL;
  const response = await fetch(`${base}wasm/hcurl_solver.js`);
  if (!response.ok) throw new Error(`Failed to fetch hcurl_solver.js (${response.status}). Build and publish the Rust H(curl) solver artifact (see wasm/hcurl/README.md).`);
  (0, eval)(await response.text());
  const factory = (self as unknown as Record<string, unknown>).createHcurlSolver as HcurlClassicFactory | undefined;
  if (!factory) throw new Error("hcurl_solver.js loaded but createHcurlSolver was not found");
  const raw = await factory({
    locateFile: (path: string) => path.endsWith(".wasm") ? `${base}wasm/hcurl_solver.wasm` : path,
  });
  if (typeof raw._hcurl_solver_alloc !== "function" || typeof raw._hcurl_solver_run !== "function") {
    throw new Error("Native H(curl) module is missing its solver exports");
  }
  const readU32 = (ptr: number): number => {
    const heap = raw.HEAPU8;
    return (heap[ptr] ?? 0)
      | ((heap[ptr + 1] ?? 0) << 8)
      | ((heap[ptr + 2] ?? 0) << 16)
      | ((heap[ptr + 3] ?? 0) << 24);
  };
  return {
    solvePoint(requestBytes: Uint8Array): Uint8Array {
      const reqPtr = raw._hcurl_solver_alloc(requestBytes.byteLength);
      const outSlotsPtr = raw._hcurl_solver_alloc(8);
      if (!reqPtr || !outSlotsPtr) throw new Error("H(curl) solver failed to allocate request/output slots");
      try {
        raw.HEAPU8.set(requestBytes, reqPtr);
        raw._hcurl_solver_run(reqPtr, requestBytes.byteLength, outSlotsPtr, outSlotsPtr + 4);
        const rspPtr = readU32(outSlotsPtr);
        const rspLen = readU32(outSlotsPtr + 4);
        if (!rspPtr || !rspLen) {
          const detail = raw._hcurl_solver_last_error();
          throw new Error(`H(curl) solver returned an empty response${detail ? `: ${detail}` : ""}`);
        }
        const responseBytes = raw.HEAPU8.slice(rspPtr, rspPtr + rspLen);
        raw._hcurl_solver_free(rspPtr);
        return responseBytes;
      } finally {
        raw._hcurl_solver_free(reqPtr);
        raw._hcurl_solver_free(outSlotsPtr);
      }
    },
    setCancelled: (flag) => raw._hcurl_solver_set_cancelled?.(flag ? 1 : 0),
    lastError: () => raw._hcurl_solver_last_error(),
    threadCount: 1,
  };
}

type NativeLoadResult =
  | { module: HcurlSolverModule; error: null }
  | { module: null; error: string };

/**
 * Prefer the threaded Rust H(curl) module, fall back to the classic
 * single-threaded module. The TypeScript reference solver remains the final
 * fallback when neither native module loads.
 */
async function loadNativeModule(): Promise<NativeLoadResult> {
  let threadedError: unknown = null;
  try {
    return { module: await loadThreadedModule(), error: null };
  } catch (error) {
    threadedError = error;
  }
  try {
    return { module: await loadClassicModule(), error: null };
  } catch (classicError) {
    const detail = [
      threadedError instanceof Error ? threadedError.message : String(threadedError),
      classicError instanceof Error ? classicError.message : String(classicError),
    ].join(" | ");
    return { module: null, error: detail };
  }
}

function invokePoint(
  module: HcurlSolverModule,
  requestBytes: Uint8Array,
): ReturnType<typeof unpackHcurlPointResponse> {
  const responseBytes = module.solvePoint(requestBytes);
  if (!responseBytes || responseBytes.byteLength === 0) {
    const detail = module.lastError();
    throw new Error(`H(curl) solver returned an empty response${detail ? `: ${detail}` : ""}`);
  }
  return unpackHcurlPointResponse(responseBytes);
}

type FEMProgressCallback = (
  pointIndex: number,
  pointsTotal: number,
  stage: FEMWorkerProgress["stage"],
) => void;

async function solve(
  request: FEMSimulationRequest,
  onProgress: FEMProgressCallback,
): Promise<FEMSimulationResult> {
  const port = selectFEMPort(request);
  if (port.positive.every((value, axis) => value === port.negative[axis])) throw new Error("Port terminals must be distinct");
  const coaxConfig = buildFEMCoaxRunConfig(request);
  onProgress(0, 0, "meshing");
  const mesh = buildFEMMesh(request.model_ir, request.mesh_size_m, port);
  const frequencies = buildFEMFrequencies(request.start_mhz, request.stop_mhz, request.steps);
  const startedAt = performance.now();
  onProgress(0, frequencies.length, "loading-native");

  // Prefer the threaded native Rust H(curl) module, then the classic
  // single-threaded one; otherwise fall back to the TypeScript H(curl)
  // reference solver so FEM keeps working.
  let module: HcurlSolverModule | null = null;
  let nativeError: string | null = null;
  let threadCount: number | null = null;
  const nativeLoad = await loadNativeModule();
  if (nativeLoad.module) {
    module = nativeLoad.module;
    threadCount = nativeLoad.module.threadCount;
  } else {
    nativeError = nativeLoad.error;
  }

  const toPoint = (
    frequencyMhz: number,
    pointStatus: FEMPointStatus,
    pointMessage: string | undefined,
    zinRe: number | null,
    zinIm: number | null,
    s11Re: number | null,
    s11Im: number | null,
  ) => ({
    frequency_mhz: frequencyMhz,
    point_status: pointStatus,
    point_message: pointMessage,
    impedance_real_ohms: zinRe,
    impedance_imag_ohms: zinIm,
    s11_real: s11Re,
    s11_imag: s11Im,
    s11_db: null,
    swr: null,
  });

  let points: ReturnType<typeof toPoint>[];
  let engine: FEMSimulationResult["engine"];
  let gainData: FEMSimulationResult["gain_data"];
  let capabilities: FEMSimulationResult["capabilities"];
  const warnings = [...mesh.warnings];
  if (module) {
    // Port attributes follow the positive-then-negative facet ordering.
    const portAttr = new Uint32Array(mesh.portPositiveFacets.length + mesh.portNegativeFacets.length);
    portAttr.fill(FEM_ATTR_PORT_POSITIVE, 0, mesh.portPositiveFacets.length);
    portAttr.fill(FEM_ATTR_PORT_NEGATIVE, mesh.portPositiveFacets.length);
    const ffEnabled = request.far_field?.enabled ?? true;
    const solved = frequencies.map((frequencyMhz, pointIndex) => {
      const packed = packHcurlPointRequest({
        frequencyHz: frequencyMhz * 1_000_000,
        vertices: mesh.vertices,
        tets: mesh.tetrahedra,
        tetRegion: mesh.tetRegions,
        boundaryFacets: mesh.boundaryFacets,
        boundaryAttr: mesh.boundaryAttributes,
        portPosFacets: mesh.portPositiveFacets,
        portNegFacets: mesh.portNegativeFacets,
        portAttr,
        materialTable: mesh.materialTable,
        metadataJson: JSON.stringify({
          mesh_artifact_version: mesh.version,
          model_ir_version: request.model_ir.version,
          reference_impedance_ohms: request.reference_impedance_ohms,
          points_total: frequencies.length,
          point_index: pointIndex,
          solver_formulation: "hcurl_nedelec_first_order",
          absorbing_boundary: "silver_muller_first_order",
          krylov_tolerance: 1e-7,
          far_field: ffEnabled
            ? {
              theta_step_deg: request.far_field?.theta_step_deg ?? 5,
              phi_step_deg: request.far_field?.phi_step_deg ?? 5,
            }
            : undefined,
          // Coax wave port (native coax.rs path): coax_port + exclude_boxes.
          ...(coaxConfig ? coaxConfig.metadata : {}),
          abi_version: HCURL_ABI_VERSION,
        }),
      });
      const rsp = invokePoint(module as HcurlSolverModule, packed);
      onProgress(pointIndex + 1, frequencies.length, "solving");
      let farField: ReturnType<typeof unpackHcurlFarFieldBlob> | null = null;
      if (rsp.fieldBlob) {
        try {
          farField = unpackHcurlFarFieldBlob(rsp.fieldBlob);
        } catch (error) {
          warnings.push(
            `Point ${frequencyMhz} MHz: native far-field blob failed to decode (${error instanceof Error ? error.message : String(error)}); gain unavailable for this point.`,
          );
        }
      }
      return {
        point: toPoint(
          frequencyMhz,
          rsp.pointStatus,
          rsp.pointMessage,
          rsp.zinRealOhms,
          rsp.zinImagOhms,
          rsp.s11Real,
          rsp.s11Imag,
        ),
        farField,
        frequencyMhz,
      };
    });
    points = solved.map((s) => s.point);
    const patterns = solved.flatMap((s) => {
      const ff = s.farField;
      if (!ff || s.point.point_status !== "ok") return [];
      return [{
        frequency_mhz: s.frequencyMhz,
        point_status: s.point.point_status,
        theta_deg: [...ff.thetaDeg],
        phi_deg: [...ff.phiDeg],
        gain_dbi: ff.gainDbi.map((row) => [...row]),
        directivity_dbi: ff.directivityDbi.map((row) => [...row]),
        max_gain_dbi: ff.maxGainDbi,
        max_gain_theta_deg: ff.maxGainThetaDeg,
        max_gain_phi_deg: ff.maxGainPhiDeg,
        max_directivity_dbi: ff.maxDirectivityDbi,
        radiated_power_w: ff.radiatedPowerW,
        accepted_power_w: ff.acceptedPowerW,
        radiation_efficiency: ff.radiationEfficiency,
      }];
    });
    engine = "hcurl-wasm";
    gainData = patterns.length > 0
      ? {
        available: true,
        points: patterns.map((p) => ({
          frequency_mhz: p.frequency_mhz,
          point_status: p.point_status,
          gain_dbi: p.max_gain_dbi,
          theta_deg: p.max_gain_theta_deg,
          phi_deg: p.max_gain_phi_deg,
        })),
        patterns,
      }
      : {
        available: false,
        message: "Far-field computation was disabled or did not produce a pattern for this run.",
      };
    capabilities = {
      ...HCURL_CAPABILITIES,
      far_field: patterns.length > 0,
      gain: patterns.length > 0,
    };
  } else {
    // TypeScript H(curl) fallback: same mesh, same port, one point at a time.
    warnings.push(
      `Native H(curl) solver unavailable (${nativeError ?? "unknown load failure"}); ` +
      "used the TypeScript H(curl) reference solver instead.",
    );
    const solverInputBase: Omit<FEMSolverInput, "frequencyHz" | "farField"> = {
      mesh,
      referenceImpedanceOhms: request.reference_impedance_ohms,
      portPositive: [port.positive[0] ?? 0, port.positive[1] ?? 0, port.positive[2] ?? 0],
      portNegative: [port.negative[0] ?? 0, port.negative[1] ?? 0, port.negative[2] ?? 0],
      // Coax wave port for the TypeScript fallback (fem-coax.ts path).
      ...(coaxConfig ? { coaxPort: coaxConfig.solver.coaxPort, excludeBoxes: coaxConfig.solver.excludeBoxes } : {}),
    };
    const ffEnabled = request.far_field?.enabled ?? true;
    const ffOptions = ffEnabled
      ? {
        thetaStepDeg: request.far_field?.theta_step_deg,
        phiStepDeg: request.far_field?.phi_step_deg,
      }
      : undefined;
    const solved = frequencies.map((frequencyMhz, pointIndex) => {
      const result = solveFEMPoint({
        ...solverInputBase,
        frequencyHz: frequencyMhz * 1_000_000,
        ...(ffOptions ? { farField: ffOptions } : {}),
      });
      onProgress(pointIndex + 1, frequencies.length, "solving");
      return {
        point: toPoint(
          frequencyMhz,
          result.status,
          result.message,
          result.impedanceOhms?.re ?? null,
          result.impedanceOhms?.im ?? null,
          result.s11?.re ?? null,
          result.s11?.im ?? null,
        ),
        farField: result.farField,
        frequencyMhz,
      };
    });
    points = solved.map((s) => s.point);
    const patterns = solved.flatMap((s) => {
      const ff = s.farField;
      if (!ff) return [];
      return [{
        frequency_mhz: s.frequencyMhz,
        point_status: s.point.point_status,
        theta_deg: [...ff.thetaDeg],
        phi_deg: [...ff.phiDeg],
        gain_dbi: ff.gainDbi.map((row) => [...row]),
        directivity_dbi: ff.directivityDbi.map((row) => [...row]),
        max_gain_dbi: ff.maxGainDbi,
        max_gain_theta_deg: ff.maxGainThetaDeg,
        max_gain_phi_deg: ff.maxGainPhiDeg,
        max_directivity_dbi: ff.maxDirectivityDbi,
        radiated_power_w: ff.radiatedPowerW,
        accepted_power_w: ff.acceptedPowerW,
        radiation_efficiency: ff.radiationEfficiency,
      }];
    });
    gainData = patterns.length > 0
      ? {
        available: true,
        points: patterns.map((p) => ({
          frequency_mhz: p.frequency_mhz,
          point_status: p.point_status,
          gain_dbi: p.max_gain_dbi,
          theta_deg: p.max_gain_theta_deg,
          phi_deg: p.max_gain_phi_deg,
        })),
        patterns,
      }
      : {
        available: false,
        message: "Far-field computation was disabled or did not produce a pattern for this run.",
      };
    capabilities = {
      ...HCURL_CAPABILITIES,
      far_field: patterns.length > 0,
      gain: patterns.length > 0,
    };
    engine = "fem-typescript";
  }

  const frequencyData = adaptFEMFrequencyData(points);
  const resonance = identifyFEMResonance(frequencyData);

  return {
    simulation_id: `hcurl-${Date.now().toString(36)}`,
    engine,
    computed_in_ms: Math.round(performance.now() - startedAt),
    // Rayon worker thread count for native solves; absent for the
    // TypeScript fallback.
    ...(threadCount !== null ? { threads: threadCount } : {}),
    mesh: {
      ...mesh.statistics,
      warnings: [...mesh.warnings],
    },
    reference_impedance_ohms: request.reference_impedance_ohms,
    frequency_data: frequencyData,
    resonant_frequency_mhz: resonance?.frequencyMhz ?? null,
    warnings,
    gain_data: gainData,
    capabilities,
  };
}

self.onmessage = async (event: MessageEvent<FEMWorkerMessage>) => {
  const message = event.data;
  if (message.type !== "fem_simulate") return;
  const postProgress: FEMProgressCallback = (point_index, points_total, stage) =>
    self.postMessage({
      type: "progress",
      id: message.id,
      generation: message.generation,
      point_index,
      points_total,
      stage,
    } satisfies FEMWorkerProgress);
  try {
    self.postMessage({ type: "success", id: message.id, generation: message.generation, result: await solve(message.request, postProgress) } satisfies FEMWorkerSuccess);
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    self.postMessage({
      type: "error",
      id: message.id,
      generation: message.generation,
      message: text,
      stage: text.includes("fetch hcurl_solver") || text.includes("createHcurlSolver") || text.includes("missing _hcurl_solver") ? "loading-native" : (text.includes("boundary/port") ? "assembly" : "meshing"),
      code: text.includes("fetch hcurl_solver") || text.includes("createHcurlSolver") || text.includes("missing _hcurl_solver") ? "unsupported_solver_abi" : "invalid_mesh",
    } satisfies FEMWorkerError);
  }
};
