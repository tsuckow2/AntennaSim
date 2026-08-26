import type { FEMPointStatus } from "../fem";

export const MFEM_ABI_VERSION = 1;
export const MFEM_REQUEST_HEADER_BYTES = 128;
export const MFEM_RESPONSE_HEADER_BYTES = 104;
export const MFEM_ENDIANNESS_SENTINEL = 0x12345678;

export const MFEM_LIMITS = {
  maxVertices: 350000,
  maxTets: 2000000,
  maxPoints: 201,
  maxRequestBytes: 512 * 1024 * 1024,
  perFrequencyTimeoutMs: 30000,
} as const;

export type MfemNativeStatusCode = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

export interface MfemPointResponse {
  statusCode: MfemNativeStatusCode;
  pointStatus: FEMPointStatus;
  pointMessage?: string;
  residualNorm: number | null;
  iterationCount: number | null;
  zinRealOhms: number | null;
  zinImagOhms: number | null;
  s11Real: number | null;
  s11Imag: number | null;
  timingAssemblyMs: number | null;
  timingSolveMs: number | null;
  timingPostMs: number | null;
  fieldBlob: Uint8Array | null;
}

export interface MfemPointRequest {
  frequencyHz: number;
  vertices: Float64Array;
  tets: Uint32Array;
  tetRegion: Uint32Array;
  boundaryFacets: Uint32Array;
  boundaryAttr: Uint32Array;
  portPosFacets: Uint32Array;
  portNegFacets: Uint32Array;
  portAttr: Uint32Array;
  materialTable: Uint8Array;
  metadataJson: string;
}

function assertOffsetAndLength(
  offset: number,
  len: number,
  totalBytes: number,
  label: string,
): void {
  if (!Number.isInteger(offset) || offset < 0) throw new Error(`${label}_offset is invalid`);
  if (!Number.isInteger(len) || len < 0) throw new Error(`${label}_len is invalid`);
  if (offset + len > totalBytes) throw new Error(`${label} extends past response buffer bounds`);
}

function ensureMul(value: number, width: number, label: string): void {
  if (value % width !== 0) throw new Error(`${label} byte length must be a multiple of ${width}`);
}

function finiteOrNull(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

function mapPointStatus(code: MfemNativeStatusCode, nonFinite: boolean): FEMPointStatus {
  if (nonFinite) return "non_finite";
  switch (code) {
    case 0:
      return "ok";
    case 6:
      return "solver_error";
    case 7:
      return "cancelled";
    case 8:
      return "timeout";
    default:
      return "solver_error";
  }
}

function writeBytes(target: Uint8Array, offset: number, source: Uint8Array): number {
  target.set(source, offset);
  return offset + source.byteLength;
}

function utf8Bytes(value: string): Uint8Array {
  return new TextEncoder().encode(value);
}

export function packMfemPointRequest(payload: MfemPointRequest): Uint8Array {
  if (!Number.isFinite(payload.frequencyHz) || payload.frequencyHz <= 0) throw new Error("frequency_hz must be a positive finite value");
  if (payload.vertices.length / 3 > MFEM_LIMITS.maxVertices) throw new Error(`vertex_count exceeds ${MFEM_LIMITS.maxVertices}`);
  if (payload.tets.length / 4 > MFEM_LIMITS.maxTets) throw new Error(`tet_count exceeds ${MFEM_LIMITS.maxTets}`);

  const verticesBytes = new Uint8Array(payload.vertices.buffer, payload.vertices.byteOffset, payload.vertices.byteLength);
  const tetsBytes = new Uint8Array(payload.tets.buffer, payload.tets.byteOffset, payload.tets.byteLength);
  const tetRegionBytes = new Uint8Array(payload.tetRegion.buffer, payload.tetRegion.byteOffset, payload.tetRegion.byteLength);
  const boundaryFacetsBytes = new Uint8Array(payload.boundaryFacets.buffer, payload.boundaryFacets.byteOffset, payload.boundaryFacets.byteLength);
  const boundaryAttrBytes = new Uint8Array(payload.boundaryAttr.buffer, payload.boundaryAttr.byteOffset, payload.boundaryAttr.byteLength);
  const portPosBytes = new Uint8Array(payload.portPosFacets.buffer, payload.portPosFacets.byteOffset, payload.portPosFacets.byteLength);
  const portNegBytes = new Uint8Array(payload.portNegFacets.buffer, payload.portNegFacets.byteOffset, payload.portNegFacets.byteLength);
  const portAttrBytes = new Uint8Array(payload.portAttr.buffer, payload.portAttr.byteOffset, payload.portAttr.byteLength);
  const materialTableBytes = payload.materialTable;
  const metadataBytes = utf8Bytes(payload.metadataJson);

  ensureMul(verticesBytes.byteLength, 8, "vertices");
  ensureMul(tetsBytes.byteLength, 4, "tets");
  ensureMul(tetRegionBytes.byteLength, 4, "tet_region");
  ensureMul(boundaryFacetsBytes.byteLength, 4, "boundary_facets");
  ensureMul(boundaryAttrBytes.byteLength, 4, "boundary_attr");
  ensureMul(portPosBytes.byteLength, 4, "port_pos_facets");
  ensureMul(portNegBytes.byteLength, 4, "port_neg_facets");
  ensureMul(portAttrBytes.byteLength, 4, "port_attr");

  const vertexCount = payload.vertices.length / 3;
  const tetCount = payload.tets.length / 4;
  const boundaryFacetCount = payload.boundaryFacets.length / 3;
  const portPosCount = payload.portPosFacets.length;
  const portNegCount = payload.portNegFacets.length;
  const portFacetCount = portPosCount + portNegCount;
  if (payload.tetRegion.length !== tetCount) throw new Error("tet_region length must equal tet_count");
  if (payload.boundaryAttr.length !== boundaryFacetCount) throw new Error("boundary_attr length must equal boundary_facet_count");
  if (payload.portAttr.length !== portFacetCount) throw new Error("port_attr length must equal total port facet count");

  const totalBytes = MFEM_REQUEST_HEADER_BYTES
    + verticesBytes.byteLength
    + tetsBytes.byteLength
    + tetRegionBytes.byteLength
    + boundaryFacetsBytes.byteLength
    + boundaryAttrBytes.byteLength
    + portPosBytes.byteLength
    + materialTableBytes.byteLength
    + metadataBytes.byteLength
    + portNegBytes.byteLength
    + portAttrBytes.byteLength;

  if (totalBytes > MFEM_LIMITS.maxRequestBytes) throw new Error(`request exceeds ${MFEM_LIMITS.maxRequestBytes} bytes`);

  const buffer = new Uint8Array(totalBytes);
  const view = new DataView(buffer.buffer);

  let offset = MFEM_REQUEST_HEADER_BYTES;
  const verticesOffset = offset;
  offset = writeBytes(buffer, offset, verticesBytes);
  const tetsOffset = offset;
  offset = writeBytes(buffer, offset, tetsBytes);
  const tetRegionOffset = offset;
  offset = writeBytes(buffer, offset, tetRegionBytes);
  const boundaryFacetsOffset = offset;
  offset = writeBytes(buffer, offset, boundaryFacetsBytes);
  const boundaryAttrOffset = offset;
  offset = writeBytes(buffer, offset, boundaryAttrBytes);
  const portPosOffset = offset;
  offset = writeBytes(buffer, offset, portPosBytes);
  const materialTableOffset = offset;
  offset = writeBytes(buffer, offset, materialTableBytes);
  const metadataOffset = offset;
  offset = writeBytes(buffer, offset, metadataBytes);
  const portNegOffset = offset;
  offset = writeBytes(buffer, offset, portNegBytes);
  const portAttrOffset = offset;
  writeBytes(buffer, offset, portAttrBytes);

  view.setUint32(0, MFEM_ABI_VERSION, true);
  view.setUint32(4, MFEM_ENDIANNESS_SENTINEL, true);
  view.setUint32(8, 0, true);
  view.setFloat64(16, payload.frequencyHz, true);
  view.setUint32(24, vertexCount, true);
  view.setUint32(28, tetCount, true);
  view.setUint32(32, 0, true);
  view.setUint32(36, boundaryFacetCount, true);
  view.setUint32(40, portFacetCount, true);
  view.setUint32(44, portPosCount, true);
  view.setUint32(48, verticesOffset, true);
  view.setUint32(52, tetsOffset, true);
  view.setUint32(56, tetRegionOffset, true);
  view.setUint32(60, boundaryFacetsOffset, true);
  view.setUint32(64, boundaryAttrOffset, true);
  view.setUint32(68, portPosOffset, true);
  view.setUint32(72, portAttrOffset, true);
  view.setUint32(76, materialTableOffset, true);
  view.setUint32(80, metadataOffset, true);
  view.setUint32(84, verticesBytes.byteLength, true);
  view.setUint32(88, tetsBytes.byteLength, true);
  view.setUint32(92, tetRegionBytes.byteLength, true);
  view.setUint32(96, boundaryFacetsBytes.byteLength, true);
  view.setUint32(100, boundaryAttrBytes.byteLength, true);
  view.setUint32(104, portPosBytes.byteLength, true);
  view.setUint32(108, materialTableBytes.byteLength, true);
  view.setUint32(112, portNegOffset, true);
  view.setUint32(116, portNegBytes.byteLength, true);
  view.setUint32(120, portAttrBytes.byteLength, true);
  view.setUint32(124, metadataBytes.byteLength, true);

  return buffer;
}

export function unpackMfemPointResponse(raw: Uint8Array): MfemPointResponse {
  if (raw.byteLength < MFEM_RESPONSE_HEADER_BYTES) throw new Error("MFEM response is smaller than fixed header");
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const abiVersion = view.getUint32(0, true);
  if (abiVersion !== MFEM_ABI_VERSION) throw new Error(`Unsupported MFEM ABI version in response: ${abiVersion}`);

  const statusCode = view.getUint32(4, true) as MfemNativeStatusCode;
  const residualNorm = view.getFloat64(16, true);
  const iterationCount = view.getUint32(24, true);
  const zinRealOhms = view.getFloat64(32, true);
  const zinImagOhms = view.getFloat64(40, true);
  const s11Real = view.getFloat64(48, true);
  const s11Imag = view.getFloat64(56, true);
  const timingAssemblyMs = view.getFloat64(64, true);
  const timingSolveMs = view.getFloat64(72, true);
  const timingPostMs = view.getFloat64(80, true);
  const fieldBlobOffset = view.getUint32(88, true);
  const fieldBlobLen = view.getUint32(92, true);
  const messageOffset = view.getUint32(96, true);
  const messageLen = view.getUint32(100, true);

  assertOffsetAndLength(fieldBlobOffset, fieldBlobLen, raw.byteLength, "field_blob");
  assertOffsetAndLength(messageOffset, messageLen, raw.byteLength, "message");

  const fieldBlob = fieldBlobLen > 0 ? raw.slice(fieldBlobOffset, fieldBlobOffset + fieldBlobLen) : null;
  const message = messageLen > 0
    ? new TextDecoder().decode(raw.slice(messageOffset, messageOffset + messageLen))
    : undefined;

  const hasNonFinite = [zinRealOhms, zinImagOhms, s11Real, s11Imag, residualNorm, timingAssemblyMs, timingSolveMs, timingPostMs].some((value) => !Number.isFinite(value));

  return {
    statusCode,
    pointStatus: mapPointStatus(statusCode, hasNonFinite),
    pointMessage: message,
    residualNorm: finiteOrNull(residualNorm),
    iterationCount: Number.isFinite(iterationCount) ? iterationCount : null,
    zinRealOhms: finiteOrNull(zinRealOhms),
    zinImagOhms: finiteOrNull(zinImagOhms),
    s11Real: finiteOrNull(s11Real),
    s11Imag: finiteOrNull(s11Imag),
    timingAssemblyMs: finiteOrNull(timingAssemblyMs),
    timingSolveMs: finiteOrNull(timingSolveMs),
    timingPostMs: finiteOrNull(timingPostMs),
    fieldBlob,
  };
}
