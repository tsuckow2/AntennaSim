import { describe, expect, it } from "vitest";
import {
  MFEM_ABI_VERSION,
  MFEM_RESPONSE_HEADER_BYTES,
  packMfemPointRequest,
  unpackMfemPointResponse,
} from "../mfem-abi";

describe("MFEM ABI pack/unpack", () => {
  it("rejects malformed tet_region length", () => {
    expect(() => packMfemPointRequest({
      frequencyHz: 918e6,
      vertices: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
      tets: new Uint32Array([0, 1, 2, 3]),
      tetRegion: new Uint32Array([]),
      boundaryFacets: new Uint32Array([]),
      boundaryAttr: new Uint32Array([]),
      portPosFacets: new Uint32Array([]),
      portNegFacets: new Uint32Array([]),
      portAttr: new Uint32Array([]),
      materialTable: new Uint8Array([]),
      metadataJson: "{}",
    })).toThrow(/tet_region/);
  });

  it("parses success response payload with optional message and field blob", () => {
    const msg = new TextEncoder().encode("ok");
    const field = new Uint8Array([1, 2, 3]);
    const total = MFEM_RESPONSE_HEADER_BYTES + field.byteLength + msg.byteLength;
    const raw = new Uint8Array(total);
    const view = new DataView(raw.buffer);
    view.setUint32(0, MFEM_ABI_VERSION, true);
    view.setUint32(4, 0, true);
    view.setFloat64(16, 1e-9, true);
    view.setUint32(24, 4, true);
    view.setFloat64(32, 50, true);
    view.setFloat64(40, 2, true);
    view.setFloat64(48, 0.1, true);
    view.setFloat64(56, -0.2, true);
    view.setFloat64(64, 1, true);
    view.setFloat64(72, 2, true);
    view.setFloat64(80, 3, true);
    view.setUint32(88, MFEM_RESPONSE_HEADER_BYTES, true);
    view.setUint32(92, field.byteLength, true);
    view.setUint32(96, MFEM_RESPONSE_HEADER_BYTES + field.byteLength, true);
    view.setUint32(100, msg.byteLength, true);
    raw.set(field, MFEM_RESPONSE_HEADER_BYTES);
    raw.set(msg, MFEM_RESPONSE_HEADER_BYTES + field.byteLength);

    const parsed = unpackMfemPointResponse(raw);
    expect(parsed.pointStatus).toBe("ok");
    expect(parsed.zinRealOhms).toBe(50);
    expect(parsed.fieldBlob).toEqual(field);
    expect(parsed.pointMessage).toBe("ok");
  });

  it("maps non-finite numeric payloads to point_status=non_finite", () => {
    const raw = new Uint8Array(MFEM_RESPONSE_HEADER_BYTES);
    const view = new DataView(raw.buffer);
    view.setUint32(0, MFEM_ABI_VERSION, true);
    view.setUint32(4, 0, true);
    view.setFloat64(16, Number.NaN, true);
    view.setFloat64(32, Number.POSITIVE_INFINITY, true);
    view.setFloat64(40, 0, true);
    view.setFloat64(48, 0, true);
    view.setFloat64(56, 0, true);
    view.setFloat64(64, 0, true);
    view.setFloat64(72, 0, true);
    view.setFloat64(80, 0, true);
    view.setUint32(88, 0, true);
    view.setUint32(92, 0, true);
    view.setUint32(96, 0, true);
    view.setUint32(100, 0, true);

    const parsed = unpackMfemPointResponse(raw);
    expect(parsed.pointStatus).toBe("non_finite");
    expect(parsed.zinRealOhms).toBeNull();
    expect(parsed.residualNorm).toBeNull();
  });
});
