import { buildFEMMesh, FEM_ATTR_PORT_NEGATIVE, FEM_ATTR_PORT_POSITIVE } from "../mesh";
import { adaptFEMFrequencyData, buildFEMFrequencies, MFEM_CAPABILITIES, selectFEMPort, type FEMSimulationRequest, type FEMSimulationResult } from "../fem";
import type { MfemSolverModule } from "./mfem-module";
import { MFEM_ABI_VERSION, packMfemPointRequest, unpackMfemPointResponse } from "./mfem-abi";

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
  stage?: "validating" | "meshing" | "loading-mfem" | "assembly" | "solving";
  code?: "unsupported_solver_abi" | "invalid_mesh" | "worker_failure";
}

export type FEMWorkerResponse = FEMWorkerSuccess | FEMWorkerError;

type MfemFactory = (options?: Record<string, unknown>) => Promise<MfemSolverModule>;
let factory: MfemFactory | null = null;

async function loadFactory(): Promise<MfemFactory> {
  if (factory) return factory;
  const base = import.meta.env.BASE_URL;
  const response = await fetch(`${base}wasm/mfem_solver.js`);
  if (!response.ok) throw new Error(`Failed to fetch mfem_solver.js (${response.status}). Build and publish the production MFEM solver artifact.`);
  (0, eval)(await response.text());
  const loaded = (self as unknown as Record<string, unknown>).createMfemSolver as MfemFactory | undefined;
  if (!loaded) throw new Error("mfem_solver.js loaded but createMfemSolver was not found");
  factory = loaded;
  return loaded;
}

function readU32(module: MfemSolverModule, ptr: number): number {
  const heap = module.HEAPU8;
  return (heap[ptr] ?? 0)
    | ((heap[ptr + 1] ?? 0) << 8)
    | ((heap[ptr + 2] ?? 0) << 16)
    | ((heap[ptr + 3] ?? 0) << 24);
}

function assertModuleExports(module: MfemSolverModule): void {
  if (typeof module._mfem_solver_alloc !== "function") throw new Error("Production MFEM module is missing _mfem_solver_alloc");
  if (typeof module._mfem_solver_free !== "function") throw new Error("Production MFEM module is missing _mfem_solver_free");
  if (typeof module._mfem_solver_run !== "function") throw new Error("Production MFEM module is missing _mfem_solver_run");
}

function invokePoint(
  module: MfemSolverModule,
  requestBytes: Uint8Array,
): ReturnType<typeof unpackMfemPointResponse> {
  const reqPtr = module._mfem_solver_alloc(requestBytes.byteLength);
  const outSlotsPtr = module._mfem_solver_alloc(8);
  if (!reqPtr || !outSlotsPtr) throw new Error("MFEM solver failed to allocate request/output slots");
  try {
    module.HEAPU8.set(requestBytes, reqPtr);
    const status = module._mfem_solver_run(reqPtr, requestBytes.byteLength, outSlotsPtr, outSlotsPtr + 4);
    const rspPtr = readU32(module, outSlotsPtr);
    const rspLen = readU32(module, outSlotsPtr + 4);
    if (!rspPtr || !rspLen) throw new Error(`MFEM solver returned empty response pointer/length (status ${status})`);
    const responseBytes = module.HEAPU8.slice(rspPtr, rspPtr + rspLen);
    module._mfem_solver_free(rspPtr);
    return unpackMfemPointResponse(responseBytes);
  } finally {
    module._mfem_solver_free(reqPtr);
    module._mfem_solver_free(outSlotsPtr);
  }
}

async function solve(request: FEMSimulationRequest): Promise<FEMSimulationResult> {
  const port = selectFEMPort(request);
  if (port.positive.every((value, axis) => value === port.negative[axis])) throw new Error("Lumped-port terminals must be distinct");
  const mesh = buildFEMMesh(request.model_ir, request.mesh_size_m, port);

  const module = await (await loadFactory())({
    locateFile: (path: string) => path.endsWith(".wasm") ? `${import.meta.env.BASE_URL}wasm/mfem_solver.wasm` : path,
  });
  assertModuleExports(module);

  const frequencies = buildFEMFrequencies(request.start_mhz, request.stop_mhz, request.steps);
  const startedAt = performance.now();
  // Port attributes follow the positive-then-negative facet ordering.
  const portAttr = new Uint32Array(mesh.portPositiveFacets.length + mesh.portNegativeFacets.length);
  portAttr.fill(FEM_ATTR_PORT_POSITIVE, 0, mesh.portPositiveFacets.length);
  portAttr.fill(FEM_ATTR_PORT_NEGATIVE, mesh.portPositiveFacets.length);
  const points = frequencies.map((frequencyMhz, pointIndex) => {
    const packed = packMfemPointRequest({
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
        solver_formulation: "real_block_complex",
        absorbing_boundary: "silver_muller_first_order",
        abi_version: MFEM_ABI_VERSION,
      }),
    });
    const rsp = invokePoint(module, packed);
    return {
      frequency_mhz: frequencyMhz,
      point_status: rsp.pointStatus,
      point_message: rsp.pointMessage,
      impedance_real_ohms: rsp.zinRealOhms,
      impedance_imag_ohms: rsp.zinImagOhms,
      s11_real: rsp.s11Real,
      s11_imag: rsp.s11Imag,
      s11_db: null,
      swr: null,
    };
  });

  return {
    simulation_id: `mfem-${Date.now().toString(36)}`,
    engine: "mfem-wasm",
    computed_in_ms: Math.round(performance.now() - startedAt),
    mesh: {
      ...mesh.statistics,
      warnings: [...mesh.warnings],
    },
    reference_impedance_ohms: request.reference_impedance_ohms,
    frequency_data: adaptFEMFrequencyData(points),
    resonant_frequency_mhz: null,
    warnings: [...mesh.warnings],
    gain_data: {
      available: false,
      message: "Gain remains unavailable in FEM mode until validated far-field payloads are provided by the native solver.",
    },
    capabilities: MFEM_CAPABILITIES,
  };
}

self.onmessage = async (event: MessageEvent<FEMWorkerMessage>) => {
  const message = event.data;
  if (message.type !== "fem_simulate") return;
  try {
    self.postMessage({ type: "success", id: message.id, generation: message.generation, result: await solve(message.request) } satisfies FEMWorkerSuccess);
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    self.postMessage({
      type: "error",
      id: message.id,
      generation: message.generation,
      message: text,
      stage: text.includes("fetch mfem_solver") || text.includes("createMfemSolver") || text.includes("missing _mfem_solver") ? "loading-mfem" : (text.includes("boundary/port") ? "assembly" : "meshing"),
      code: text.includes("fetch mfem_solver") || text.includes("createMfemSolver") || text.includes("missing _mfem_solver") ? "unsupported_solver_abi" : "invalid_mesh",
    } satisfies FEMWorkerError);
  }
};
