import type { BoxEntity, ModelIR, Vec3 } from "../dsl/model-ir";
import { validateModelIR } from "../dsl/model-ir";

export interface FEMMeshArtifact {
  version: 1;
  units: "m";
  vertices: Float64Array;
  tetrahedra: Uint32Array;
  elementAttributes: Uint32Array;
  materialNames: string[];
  boundaryAttributes: Uint32Array;
  portAttributes: Uint32Array;
  statistics: {
    nodes: number;
    elements: number;
    min_quality: number;
    material_count: number;
    boundary_count: number;
  };
  warnings: string[];
}

interface Bounds { min: Vec3; max: Vec3 }

function bounds(box: BoxEntity): Bounds {
  return {
    min: [box.center[0] - box.size[0] / 2, box.center[1] - box.size[1] / 2, box.center[2] - box.size[2] / 2],
    max: [box.center[0] + box.size[0] / 2, box.center[1] + box.size[1] / 2, box.center[2] + box.size[2] / 2],
  };
}

function overlaps(a: Bounds, b: Bounds): boolean {
  return a.min.every((value, axis) => value < b.max[axis]! && a.max[axis]! > b.min[axis]!);
}

function cellQuality(size: Vec3): number {
  const shortest = Math.min(...size);
  const longest = Math.max(...size);
  return longest > 0 ? shortest / longest : 0;
}

/** Build the deliberately bounded FEM mesh: independent Cartesian boxes split into five tetrahedra per cell. */
export function buildFEMMesh(ir: ModelIR, meshSizeM: number): FEMMeshArtifact {
  const diagnostics = validateModelIR(ir);
  const firstError = diagnostics.find((item) => item.severity === "error");
  if (firstError) throw new Error(`Invalid ModelIR: ${firstError.path} ${firstError.message}`);
  if (!Number.isFinite(meshSizeM) || meshSizeM <= 0) throw new Error("mesh_size_m must be a positive finite value");
  if (ir.entities.some((entity) => entity.kind === "sheetRect")) {
    throw new Error("Volume FEM does not support sheetRect; use finite-thickness box entities");
  }
  if (ir.boundaries.length === 0) {
    throw new Error("missing_air_domain: a radiation or PML boundary with padding_m is required for volume FEM");
  }
  const boxes = ir.entities.filter((entity): entity is BoxEntity => entity.kind === "box");
  if (boxes.length === 0) throw new Error("Volume FEM requires at least one box entity");
  const boxBounds = boxes.map(bounds);
  for (let left = 0; left < boxes.length; left += 1) {
    for (let right = left + 1; right < boxes.length; right += 1) {
      if (overlaps(boxBounds[left]!, boxBounds[right]!)) {
        throw new Error(`Overlapping box volumes are ambiguous: ${boxes[left]!.id} and ${boxes[right]!.id}`);
      }
    }
  }

  const vertexValues: number[] = [];
  const vertexIndex = new Map<string, number>();
  const tetrahedra: number[] = [];
  const attributes: number[] = [];
  const materialIndex = new Map(ir.materials.map((material, index) => [material.name, index + 1]));
  let minimumQuality = 1;

  const getVertex = (vertex: Vec3): number => {
    const key = vertex.map((value) => value.toPrecision(15)).join(",");
    const existing = vertexIndex.get(key);
    if (existing !== undefined) return existing;
    const index = vertexValues.length / 3;
    vertexValues.push(...vertex);
    vertexIndex.set(key, index);
    return index;
  };

  boxes.forEach((box, boxIndex) => {
    const lower = boxBounds[boxIndex]!.min;
    const nx = Math.max(1, Math.ceil(box.size[0] / meshSizeM));
    const ny = Math.max(1, Math.ceil(box.size[1] / meshSizeM));
    const nz = Math.max(1, Math.ceil(box.size[2] / meshSizeM));
    const dx = box.size[0] / nx;
    const dy = box.size[1] / ny;
    const dz = box.size[2] / nz;
    minimumQuality = Math.min(minimumQuality, cellQuality([dx, dy, dz]));
    const indexAt = (x: number, y: number, z: number) => getVertex([
      lower[0] + x * dx,
      lower[1] + y * dy,
      lower[2] + z * dz,
    ]);
    const attribute = materialIndex.get(box.material);
    if (!attribute) throw new Error(`Unknown material for box ${box.id}: ${box.material}`);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const v000 = indexAt(x, y, z); const v100 = indexAt(x + 1, y, z);
      const v010 = indexAt(x, y + 1, z); const v110 = indexAt(x + 1, y + 1, z);
      const v001 = indexAt(x, y, z + 1); const v101 = indexAt(x + 1, y, z + 1);
      const v011 = indexAt(x, y + 1, z + 1); const v111 = indexAt(x + 1, y + 1, z + 1);
      const cells = [[v000, v100, v110, v111], [v000, v110, v010, v111], [v000, v010, v011, v111], [v000, v011, v001, v111], [v000, v001, v101, v111]];
      cells.forEach((tetrahedron) => { tetrahedra.push(...tetrahedron); attributes.push(attribute); });
    }
  });

  return {
    version: 1,
    units: "m",
    vertices: new Float64Array(vertexValues),
    tetrahedra: new Uint32Array(tetrahedra),
    elementAttributes: new Uint32Array(attributes),
    materialNames: ir.materials.map((material) => material.name),
    boundaryAttributes: new Uint32Array(),
    portAttributes: new Uint32Array(),
    statistics: {
      nodes: vertexValues.length / 3,
      elements: attributes.length,
      min_quality: minimumQuality,
      material_count: ir.materials.length,
      boundary_count: 0,
    },
    warnings: ["Bounded box converter: boxes are meshed independently and no airbox or absorbing boundary is generated."],
  };
}