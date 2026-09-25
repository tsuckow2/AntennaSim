import { describe, expect, it } from "vitest";
import type { ModelIR } from "../../dsl/model-ir";
import {
  buildFEMMesh,
  FEM_ATTR_CONDUCTOR_PEC,
  FEM_ATTR_EXTERIOR_ABSORBING,
  FEM_ATTR_PORT_NEGATIVE,
  FEM_ATTR_PORT_POSITIVE,
  type FEMMeshArtifact,
} from "../mesh";
import { packHcurlPointRequest } from "../wasm/hcurl-abi";

// Copper patch on an FR4 substrate, floating in a generated air volume.
function patchIR(): ModelIR {
  return {
    version: 1,
    paramSchemaVersion: 1,
    materials: [
      { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 },
      { name: "fr4", epsilon_r: 4.4, mu_r: 1, loss_tangent: 0.02 },
    ],
    entities: [
      { id: "substrate", kind: "box", center: [0, 0, 0], size: [0.02, 0.02, 0.002], material: "fr4" },
      { id: "patch", kind: "box", center: [0, 0, 0.0015], size: [0.012, 0.012, 0.001], material: "copper" },
    ],
    ports: [
      { id: "feed", kind: "lumpedPort", positive: [0, 0, 0.002], negative: [0.004, 0, 0.002], impedance_ohms: 50 },
    ],
    boundaries: [{ kind: "radiation", padding_m: 0.01 }],
    meshHints: [],
  };
}

function regionHistogram(mesh: FEMMeshArtifact): Map<number, number> {
  const histogram = new Map<number, number>();
  for (const region of mesh.tetRegions) histogram.set(region, (histogram.get(region) ?? 0) + 1);
  return histogram;
}

describe("FEM mesh pipeline", () => {
  it("builds a deterministic conforming mesh with an explicit air volume", () => {
    const first = buildFEMMesh(patchIR(), 0.005);
    const second = buildFEMMesh(patchIR(), 0.005);
    for (const key of ["vertices", "tetrahedra", "tetRegions", "boundaryFacets", "boundaryAttributes"] as const) {
      expect([...first[key]]).toEqual([...second[key]]);
    }
    expect(first.statistics).toMatchObject({ nodes: 700, elements: 2916, material_count: 3, boundary_count: 816 });
    expect(first.statistics.min_quality).toBeGreaterThan(0);
    // Region attributes follow the frozen scheme: 1 = airbox, 1000 + sorted material name.
    expect(regionHistogram(first)).toEqual(new Map([[1, 2712], [1000, 54], [1001, 150]]));
    // Every absorbing facet lies on the airbox exterior.
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    for (let i = 0; i < first.vertices.length; i += 3) {
      xs.push(first.vertices[i]!);
      ys.push(first.vertices[i + 1]!);
      zs.push(first.vertices[i + 2]!);
    }
    const bounds = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys), Math.min(...zs), Math.max(...zs)];
    for (let f = 0; f < first.boundaryAttributes.length; f += 1) {
      if (first.boundaryAttributes[f] !== FEM_ATTR_EXTERIOR_ABSORBING) continue;
      const cx = (first.vertices[first.boundaryFacets[f * 3]! * 3]! + first.vertices[first.boundaryFacets[f * 3 + 1]! * 3]! + first.vertices[first.boundaryFacets[f * 3 + 2]! * 3]!) / 3;
      const cy = (first.vertices[first.boundaryFacets[f * 3]! * 3 + 1]! + first.vertices[first.boundaryFacets[f * 3 + 1]! * 3 + 1]! + first.vertices[first.boundaryFacets[f * 3 + 2]! * 3 + 1]!) / 3;
      const cz = (first.vertices[first.boundaryFacets[f * 3]! * 3 + 2]! + first.vertices[first.boundaryFacets[f * 3 + 1]! * 3 + 2]! + first.vertices[first.boundaryFacets[f * 3 + 2]! * 3 + 2]!) / 3;
      const onExterior =
        Math.abs(cx - bounds[0]!) < 1e-12 || Math.abs(cx - bounds[1]!) < 1e-12 ||
        Math.abs(cy - bounds[2]!) < 1e-12 || Math.abs(cy - bounds[3]!) < 1e-12 ||
        Math.abs(cz - bounds[4]!) < 1e-12 || Math.abs(cz - bounds[5]!) < 1e-12;
      expect(onExterior).toBe(true);
    }
  });

  it("tags conductor interfaces as PEC and emits 40-byte material records", () => {
    const mesh = buildFEMMesh(patchIR(), 0.005);
    const pec = [...mesh.boundaryAttributes].filter((attr) => attr === FEM_ATTR_CONDUCTOR_PEC);
    expect(pec.length).toBe(60);
    expect(mesh.materials.map((entry) => [entry.attribute, entry.name])).toEqual([
      [1, "airbox"],
      [1000, "copper"],
      [1001, "fr4"],
    ]);
    expect(mesh.materialTable.byteLength).toBe(120);
    const view = new DataView(mesh.materialTable.buffer);
    const record = (index: number) => ({
      attribute: view.getUint32(index * 40, true),
      epsilon_r: view.getFloat64(index * 40 + 8, true),
      mu_r: view.getFloat64(index * 40 + 16, true),
      conductivity: view.getFloat64(index * 40 + 24, true),
      loss_tangent: view.getFloat64(index * 40 + 32, true),
    });
    expect(record(0)).toMatchObject({ attribute: 1, epsilon_r: 1, mu_r: 1, conductivity: 0, loss_tangent: 0 });
    expect(record(1)).toMatchObject({ attribute: 1000, conductivity: 5.8e7 });
    expect(record(2)).toMatchObject({ attribute: 1001, epsilon_r: 4.4, loss_tangent: 0.02 });
  });

  it("shares vertices at touching material interfaces", () => {
    const mesh = buildFEMMesh(patchIR(), 0.005);
    const regionsByVertex = new Map<number, Set<number>>();
    mesh.tetrahedra.forEach((vertex, index) => {
      const region = mesh.tetRegions[Math.floor(index / 4)]!;
      let set = regionsByVertex.get(vertex);
      if (!set) {
        set = new Set();
        regionsByVertex.set(vertex, set);
      }
      set.add(region);
    });
    // Patch bottom (z = 1 mm) sits exactly on the substrate top.
    const interfaceVertices: number[] = [];
    for (let v = 0; v < mesh.statistics.nodes; v += 1) {
      const x = mesh.vertices[v * 3]!;
      const y = mesh.vertices[v * 3 + 1]!;
      const z = mesh.vertices[v * 3 + 2]!;
      if (Math.abs(z - 0.001) < 1e-12 && Math.abs(x) <= 0.006 + 1e-12 && Math.abs(y) <= 0.006 + 1e-12) {
        interfaceVertices.push(v);
      }
    }
    expect(interfaceVertices.length).toBeGreaterThan(0);
    for (const vertex of interfaceVertices) {
      const regions = regionsByVertex.get(vertex)!;
      expect(regions.has(1000)).toBe(true);
      expect(regions.has(1001)).toBe(true);
    }
  });

  it("maps lumped-port terminals to conductor-air facet sets", () => {
    const ir = patchIR();
    const mesh = buildFEMMesh(ir, 0.005, ir.ports[0]);
    expect(mesh.portPositiveFacets.length).toBe(2);
    expect(mesh.portNegativeFacets.length).toBe(2);
    const positive = new Set(mesh.portPositiveFacets);
    for (const facet of mesh.portNegativeFacets) expect(positive.has(facet)).toBe(false);
    for (const facet of [...mesh.portPositiveFacets, ...mesh.portNegativeFacets]) {
      expect(facet).toBeLessThan(mesh.statistics.boundary_count);
      expect(mesh.boundaryAttributes[facet]).toBe(FEM_ATTR_CONDUCTOR_PEC);
    }
    // The artifact packs cleanly into the frozen native ABI request.
    const portAttr = new Uint32Array(mesh.portPositiveFacets.length + mesh.portNegativeFacets.length);
    portAttr.fill(FEM_ATTR_PORT_POSITIVE, 0, mesh.portPositiveFacets.length);
    portAttr.fill(FEM_ATTR_PORT_NEGATIVE, mesh.portPositiveFacets.length);
    expect(() =>
      packHcurlPointRequest({
        frequencyHz: 918e6,
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
          mesh_artifact_version: 1,
          model_ir_version: 1,
          reference_impedance_ohms: 50,
          points_total: 1,
          point_index: 0,
          solver_formulation: "real_block_complex",
          absorbing_boundary: "silver_muller_first_order",
        }),
      }),
    ).not.toThrow();
  });

  it("rejects a missing air domain", () => {
    const ir = patchIR();
    ir.boundaries = [];
    expect(() => buildFEMMesh(ir, 0.005)).toThrow(/missing_air_domain/);
  });

  it("rejects overlapping volumes before MFEM", () => {
    const ir = patchIR();
    expect(() =>
      buildFEMMesh({ ...ir, entities: [ir.entities[0]!, { ...ir.entities[0]!, id: "copy" }] }, 0.005),
    ).toThrow(/Overlapping/);
  });

  it("meshes a zero-thickness PEC sheetRect as conductor facets", () => {
    const ir = patchIR();
    // Replace the 1 mm copper patch box with a zero-thickness PEC sheet.
    ir.entities = [
      ir.entities[0]!,
      { id: "patch", kind: "sheetRect", center: [0, 0, 0.002], size: [0.012, 0.012], normal: [0, 0, 1], material: "copper" },
    ];
    const mesh = buildFEMMesh(ir, 0.005);
    // No copper volume region: only airbox (1) and fr4 (1001).
    expect(regionHistogram(mesh)).toEqual(new Map([[1, 2766], [1001, 150]]));
    // The sheet contributes PEC boundary facets on the z=0.002 plane.
    let pecOnSheet = 0;
    for (let f = 0; f < mesh.boundaryFacets.length / 3; f += 1) {
      if (mesh.boundaryAttributes[f] !== FEM_ATTR_CONDUCTOR_PEC) continue;
      const z0 = mesh.vertices[mesh.boundaryFacets[f * 3]! * 3 + 2]!;
      const z1 = mesh.vertices[mesh.boundaryFacets[f * 3 + 1]! * 3 + 2]!;
      const z2 = mesh.vertices[mesh.boundaryFacets[f * 3 + 2]! * 3 + 2]!;
      if (Math.abs(z0 - 0.002) < 1e-9 && Math.abs(z1 - 0.002) < 1e-9 && Math.abs(z2 - 0.002) < 1e-9) pecOnSheet += 1;
    }
    expect(pecOnSheet).toBeGreaterThan(0);
  });

  it("rejects a non-conductor sheetRect", () => {
    const ir = patchIR();
    expect(() =>
      buildFEMMesh({ ...ir, entities: [{ id: "sheet", kind: "sheetRect", center: [0, 0, 0], size: [0.01, 0.01], normal: [0, 0, 1], material: "fr4" }] }, 0.005),
    ).toThrow(/non-conductor/);
  });

  it("reports invalid_port_no_contact for a terminal floating in air", () => {
    const ir = patchIR();
    // In the air region, further than tol from every mesh surface.
    const port = { ...ir.ports[0]!, positive: [0.015, 0, 0.005] as [number, number, number] };
    expect(() => buildFEMMesh(ir, 0.005, port)).toThrow(/invalid_port_no_contact/);
  });

  it("reports invalid_port_material_contact for a terminal on a dielectric surface", () => {
    const ir = patchIR();
    // On top of the FR4 substrate, clear of the copper patch.
    const port = { ...ir.ports[0]!, positive: [0.008, 0, 0.001] as [number, number, number] };
    expect(() => buildFEMMesh(ir, 0.005, port)).toThrow(/invalid_port_material_contact/);
  });

  it("reports invalid_port_ambiguous_contact for a terminal between two conductors", () => {
    const ir = patchIR();
    ir.entities = [
      { id: "blockA", kind: "box", center: [-0.006, 0, 0], size: [0.01, 0.01, 0.002], material: "copper" },
      { id: "blockB", kind: "box", center: [0.006, 0, 0], size: [0.01, 0.01, 0.002], material: "copper" },
    ];
    const port = {
      id: "feed",
      positive: [0, 0, 0.001] as [number, number, number],
      negative: [-0.006, 0, 0.001] as [number, number, number],
    };
    expect(() => buildFEMMesh(ir, 0.005, port)).toThrow(/invalid_port_ambiguous_contact/);
  });

  it("resolves a short probe-feed port without false ambiguous contact", () => {
    // Mirrors the 908 MHz patch example: a 1 mm probe gap above the ground
    // plane with a 15 mm nominal mesh. The old mesh-size-only tolerance
    // (3.75 mm) saw both the probe bottom and the ground top and threw
    // invalid_port_ambiguous_contact.
    const ir: ModelIR = {
      version: 1,
      paramSchemaVersion: 1,
      materials: [{ name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 }],
      entities: [
        { id: "ground", kind: "box", center: [0, 0, -0.001], size: [0.35, 0.35, 0.002], material: "copper" },
        { id: "probe", kind: "box", center: [0, 0, 0.002], size: [0.004, 0.004, 0.002], material: "copper" },
      ],
      ports: [],
      boundaries: [{ kind: "radiation", padding_m: 0.02 }],
      meshHints: [],
    };
    const port = {
      id: "port1",
      positive: [0, 0, 0.001] as [number, number, number],
      negative: [0, 0, 0] as [number, number, number],
    };
    const mesh = buildFEMMesh(ir, 0.015, port);
    expect(mesh.portPositiveFacets.length).toBeGreaterThan(0);
    expect(mesh.portNegativeFacets.length).toBeGreaterThan(0);
    const positive = new Set(mesh.portPositiveFacets);
    for (const facet of mesh.portNegativeFacets) expect(positive.has(facet)).toBe(false);
    // Positive facets are on the probe bottom (z = 1 mm), negative on the ground top (z = 0).
    const facetZ = (f: number) =>
      (mesh.vertices[mesh.boundaryFacets[f * 3]! * 3 + 2]! +
        mesh.vertices[mesh.boundaryFacets[f * 3 + 1]! * 3 + 2]! +
        mesh.vertices[mesh.boundaryFacets[f * 3 + 2]! * 3 + 2]!) / 3;
    for (const f of mesh.portPositiveFacets) expect(Math.abs(facetZ(f) - 0.001)).toBeLessThan(1e-9);
    for (const f of mesh.portNegativeFacets) expect(Math.abs(facetZ(f))).toBeLessThan(1e-9);
  });

  it("honors mesh hints with finer local subdivision", () => {
    const plain = buildFEMMesh(patchIR(), 0.01);
    const ir = patchIR();
    ir.meshHints = [{ region: "patch", max_size_m: 0.0025 }];
    const refined = buildFEMMesh(ir, 0.01);
    expect(refined.statistics.nodes).toBeGreaterThan(plain.statistics.nodes);
  });

  it("leaves port facet sets empty with a warning when no port is given", () => {
    const mesh = buildFEMMesh(patchIR(), 0.005);
    expect(mesh.portPositiveFacets.length).toBe(0);
    expect(mesh.portNegativeFacets.length).toBe(0);
    expect(mesh.warnings.some((warning) => /No lumped port/.test(warning))).toBe(true);
  });
});
