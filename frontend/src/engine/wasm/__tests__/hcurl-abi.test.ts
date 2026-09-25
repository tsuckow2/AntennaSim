import { describe, expect, it } from "vitest";
import {
  HCURL_ABI_VERSION,
  HCURL_FAR_FIELD_BLOB_VERSION,
  HCURL_RESPONSE_HEADER_BYTES,
  packHcurlPointRequest,
  unpackHcurlPointResponse,
} from "../hcurl-abi";

describe("H(curl) ABI pack/unpack", () => {
  it("rejects malformed tet_region length", () => {
    expect(() => packHcurlPointRequest({
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
    const total = HCURL_RESPONSE_HEADER_BYTES + field.byteLength + msg.byteLength;
    const raw = new Uint8Array(total);
    const view = new DataView(raw.buffer);
    view.setUint32(0, HCURL_ABI_VERSION, true);
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
    view.setUint32(88, HCURL_RESPONSE_HEADER_BYTES, true);
    view.setUint32(92, field.byteLength, true);
    view.setUint32(96, HCURL_RESPONSE_HEADER_BYTES + field.byteLength, true);
    view.setUint32(100, msg.byteLength, true);
    raw.set(field, HCURL_RESPONSE_HEADER_BYTES);
    raw.set(msg, HCURL_RESPONSE_HEADER_BYTES + field.byteLength);

    const parsed = unpackHcurlPointResponse(raw);
    expect(parsed.pointStatus).toBe("ok");
    expect(parsed.zinRealOhms).toBe(50);
    expect(parsed.fieldBlob).toEqual(field);
    expect(parsed.pointMessage).toBe("ok");
  });

  it("maps non-finite numeric payloads to point_status=non_finite", () => {
    const raw = new Uint8Array(HCURL_RESPONSE_HEADER_BYTES);
    const view = new DataView(raw.buffer);
    view.setUint32(0, HCURL_ABI_VERSION, true);
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

    const parsed = unpackHcurlPointResponse(raw);
    expect(parsed.pointStatus).toBe("non_finite");
    expect(parsed.zinRealOhms).toBeNull();
    expect(parsed.residualNorm).toBeNull();
  });
});

describe("unpackHcurlFarFieldBlob", () => {
  function makeBlob(overrides?: { version?: number; gain?: number; nTheta?: number; nPhi?: number }): Uint8Array {
    const nTheta = overrides?.nTheta ?? 2;
    const nPhi = overrides?.nPhi ?? 3;
    const scalars = [2.76, 90, 45, 2.45, 36.77, 33.32, 1.104];
    const theta = [0, 90];
    const phi = [0, 90, 180];
    const gain = Array.from({ length: nTheta * nPhi }, (_, i) => overrides?.gain ?? (-10 + i));
    const directivity = gain.map((g) => g + 0.3);
    const parts: number[] = [...scalars, ...theta.slice(0, nTheta), ...phi.slice(0, nPhi), ...gain.slice(0, nTheta * nPhi), ...directivity.slice(0, nTheta * nPhi)];
    const raw = new Uint8Array(12 + parts.length * 8);
    const view = new DataView(raw.buffer);
    view.setUint32(0, overrides?.version ?? HCURL_FAR_FIELD_BLOB_VERSION, true);
    view.setUint32(4, nTheta, true);
    view.setUint32(8, nPhi, true);
    parts.forEach((v, i) => view.setFloat64(12 + i * 8, v, true));
    return raw;
  }

  it("decodes a valid v1 blob with row-major grids", async () => {
    const { unpackHcurlFarFieldBlob } = await import("../hcurl-abi");
    const ff = unpackHcurlFarFieldBlob(makeBlob());
    expect(ff.thetaDeg).toEqual([0, 90]);
    expect(ff.phiDeg).toEqual([0, 90, 180]);
    expect(ff.gainDbi).toHaveLength(2);
    expect(ff.gainDbi[1]?.[2]).toBeCloseTo(-5, 12);
    expect(ff.directivityDbi[0]?.[0]).toBeCloseTo(-9.7, 12);
    expect(ff.maxGainDbi).toBeCloseTo(2.76, 12);
    expect(ff.radiatedPowerW).toBeCloseTo(36.77, 12);
    expect(ff.radiationEfficiency).toBeCloseTo(1.104, 12);
  });

  it("rejects wrong version, truncation, and non-finite values", async () => {
    const { unpackHcurlFarFieldBlob } = await import("../hcurl-abi");
    expect(() => unpackHcurlFarFieldBlob(makeBlob({ version: 2 }))).toThrow(/version/);
    expect(() => unpackHcurlFarFieldBlob(makeBlob().slice(0, 40))).toThrow();
    expect(() => unpackHcurlFarFieldBlob(makeBlob({ gain: Number.NaN }))).toThrow(/non-finite/);
    expect(() => unpackHcurlFarFieldBlob(new Uint8Array(4))).toThrow(/header/);
  });
});
