import { buildFEMMesh, FEM_ATTR_PORT_NEGATIVE, FEM_ATTR_PORT_POSITIVE } from "../mesh";
import { adaptFEMFrequencyData, buildFEMFrequencies, HCURL_CAPABILITIES, selectFEMPort, type FEMPointStatus, type FEMSimulationRequest, type FEMSimulationResult } from "../fem";
import { solveFEMPoint, type FEMSolverInput } from "../fem-solver";
import type { HcurlSolverModule } from "./hcurl-module";
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

type HcurlFactory = (options?: Record<string, unknown>) => Promise<HcurlSolverModule>;
let factory: HcurlFactory | null = null;

async function loadFactory(): Promise<HcurlFactory> {
  if (factory) return factory;
  const base = import.meta.env.BASE_URL;
  const response = await fetch(`${base}wasm/hcurl_solver.js`);
  if (!response.ok) throw new Error(`Failed to fetch hcurl_solver.js (${response.status}). Build and publish the Rust H(curl) solver artifact (see wasm/hcurl/README.md).`);
  (0, eval)(await response.text());
  const loaded = (self as unknown as Record<string, unknown>).createHcurlSolver as HcurlFactory | undefined;
  if (!loaded) throw new Error("hcurl_solver.js loaded but createHcurlSolver was not found");
  factory = loaded;
  return loaded;
}

function readU32(module: HcurlSolverModule, ptr: number): number {
  const heap = module.HEAPU8;
  return (heap[ptr] ?? 0)
    | ((heap[ptr + 1] ?? 0) << 8)
    | ((heap[ptr + 2] ?? 0) << 16)
    | ((heap[ptr + 3] ?? 0) << 24);
}

function assertModuleExports(module: HcurlSolverModule): void {
  if (typeof module._hcurl_solver_alloc !== "function") throw new Error("Native H(curl) module is missing _hcurl_solver_alloc");
  if (typeof module._hcurl_solver_free !== "function") throw new Error("Native H(curl) module is missing _hcurl_solver_free");
  if (typeof module._hcurl_solver_run !== "function") throw new Error("Native H(curl) module is missing _hcurl_solver_run");
}

function invokePoint(
  module: HcurlSolverModule,
  requestBytes: Uint8Array,
): ReturnType<typeof unpackHcurlPointResponse> {
  const reqPtr = module._hcurl_solver_alloc(requestBytes.byteLength);
  const outSlotsPtr = module._hcurl_solver_alloc(8);
  if (!reqPtr || !outSlotsPtr) throw new Error("H(curl) solver failed to allocate request/output slots");
  try {
    module.HEAPU8.set(requestBytes, reqPtr);
    const status = module._hcurl_solver_run(reqPtr, requestBytes.byteLength, outSlotsPtr, outSlotsPtr + 4);
    const rspPtr = readU32(module, outSlotsPtr);
    const rspLen = readU32(module, outSlotsPtr + 4);
    if (!rspPtr || !rspLen) {
      const detail = module._hcurl_solver_last_error();
      throw new Error(`H(curl) solver returned empty response pointer/length (status ${status})${detail ? `: ${detail}` : ""}`);
    }
    const responseBytes = module.HEAPU8.slice(rspPtr, rspPtr + rspLen);
    module._hcurl_solver_free(rspPtr);
    return unpackHcurlPointResponse(responseBytes);
  } finally {
    module._hcurl_solver_free(reqPtr);
    module._hcurl_solver_free(outSlotsPtr);
  }
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
  if (port.positive.every((value, axis) => value === port.negative[axis])) throw new Error("Lumped-port terminals must be distinct");
  onProgress(0, 0, "meshing");
  const mesh = buildFEMMesh(request.model_ir, request.mesh_size_m, port);
  const frequencies = buildFEMFrequencies(request.start_mhz, request.stop_mhz, request.steps);
  const startedAt = performance.now();
  onProgress(0, frequencies.length, "loading-native");

  // Prefer the native Rust H(curl) module when it is published; otherwise
  // fall back to the TypeScript H(curl) reference solver so FEM keeps working.
  let module: HcurlSolverModule | null = null;
  let nativeError: string | null = null;
  try {
    const loaded = await (await loadFactory())({
      locateFile: (path: string) => path.endsWith(".wasm") ? `${import.meta.env.BASE_URL}wasm/hcurl_solver.wasm` : path,
    });
    assertModuleExports(loaded);
    module = loaded;
  } catch (error) {
    nativeError = error instanceof Error ? error.message : String(error);
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

  return {
    simulation_id: `hcurl-${Date.now().toString(36)}`,
    engine,
    computed_in_ms: Math.round(performance.now() - startedAt),
    mesh: {
      ...mesh.statistics,
      warnings: [...mesh.warnings],
    },
    reference_impedance_ohms: request.reference_impedance_ohms,
    frequency_data: adaptFEMFrequencyData(points),
    resonant_frequency_mhz: null,
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
