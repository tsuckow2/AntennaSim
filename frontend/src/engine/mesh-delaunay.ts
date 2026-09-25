/**
 * Alternative unstructured mesher: Delaunay tetrahedralization.
 *
 * Uses the same geometry input (boxes, sheets, airbox) and the same
 * tensor-grid points as the Kuhn mesher, but triangulates them with
 * Bowyer-Watson Delaunay instead of Kuhn subdivision. The post-triangulation
 * assembly (facet inventory, boundary classification, sheet PEC tagging,
 * port mapping, material table) mirrors mesh.ts.
 *
 * Purpose: controlled comparison of triangulation quality (Delaunay maximizes
 * the minimum solid angle; Kuhn on anisotropic grids produces slivers) and
 * its effect on solver conditioning vs the tensor mesher.
 */

import type { BoxEntity, Material, ModelIR, SheetRectEntity, Vec3 } from "../dsl/model-ir";
import { validateModelIR } from "../dsl/model-ir";
import { delaunay3d } from "./delaunay";
import {
  FEM_ATTR_CONDUCTOR_PEC,
  FEM_ATTR_EXTERIOR_ABSORBING,
  FEM_CONDUCTOR_CONDUCTIVITY_S_M,
  FEM_REGION_AIR,
  FEM_REGION_MATERIAL_BASE,
  type FEMMeshArtifact,
  type FEMMeshMaterialEntry,
  type FEMPortTerminals,
  type MeshedBox,
  type MeshedSheet,
} from "./mesh";

/** Grid planes closer than this are merged. */
const COORD_SNAP_M = 1e-12;
const snap = (value: number): number => Math.round(value / COORD_SNAP_M) * COORD_SNAP_M;

function isConductor(material: Material): boolean {
  return (
    material.conductivity_s_m != null &&
    Number.isFinite(material.conductivity_s_m) &&
    material.conductivity_s_m >= FEM_CONDUCTOR_CONDUCTIVITY_S_M
  );
}

function isAirLikeMaterial(material: Material): boolean {
  return (
    material.epsilon_r === 1 &&
    material.mu_r === 1 &&
    !(material.conductivity_s_m != null && material.conductivity_s_m > 0) &&
    !(material.loss_tangent != null && material.loss_tangent > 0)
  );
}

function boxFromEntity(entity: BoxEntity, attribute: number, material: Material): MeshedBox {
  return {
    id: entity.id,
    min: [snap(entity.center[0] - entity.size[0] / 2), snap(entity.center[1] - entity.size[1] / 2), snap(entity.center[2] - entity.size[2] / 2)],
    max: [snap(entity.center[0] + entity.size[0] / 2), snap(entity.center[1] + entity.size[1] / 2), snap(entity.center[2] + entity.size[2] / 2)],
    attribute,
    conductor: isConductor(material),
    airLike: isAirLikeMaterial(material),
  };
}

function sheetFromEntity(entity: SheetRectEntity, material: Material): MeshedSheet {
  if (!isConductor(material)) {
    throw new Error(`sheetRect ${entity.id} uses non-conductor material ${entity.material}; sheets must be conductive (PEC)`);
  }
  const n = entity.normal;
  let axis: 0 | 1 | 2;
  if (n[0] === 1 || n[0] === -1) axis = 0;
  else if (n[1] === 1 || n[1] === -1) axis = 1;
  else if (n[2] === 1 || n[2] === -1) axis = 2;
  else throw new Error(`sheetRect ${entity.id} has non-axis-aligned normal [${n}]; only ±x, ±y, ±z are supported`);
  const axes = [0, 1, 2].filter((a) => a !== axis) as [0 | 1 | 2, 0 | 1 | 2];
  const uAxis = axes[0]!;
  const vAxis = axes[1]!;
  return {
    id: entity.id,
    axis,
    plane: snap(entity.center[axis]),
    uAxis,
    uMin: snap(entity.center[uAxis]! - entity.size[0]! / 2),
    uMax: snap(entity.center[uAxis]! + entity.size[0]! / 2),
    vAxis,
    vMin: snap(entity.center[vAxis]! - entity.size[1]! / 2),
    vMax: snap(entity.center[vAxis]! + entity.size[1]! / 2),
  };
}

function buildMaterialTable(entries: FEMMeshMaterialEntry[]): Uint8Array {
  const buffer = new Uint8Array(entries.length * 40);
  const view = new DataView(buffer.buffer);
  entries.forEach((entry, index) => {
    const offset = index * 40;
    view.setUint32(offset, entry.attribute, true);
    view.setUint32(offset + 4, 0, true);
    view.setFloat64(offset + 8, entry.epsilon_r, true);
    view.setFloat64(offset + 16, entry.mu_r, true);
    view.setFloat64(offset + 24, entry.conductivity_s_m, true);
    view.setFloat64(offset + 32, entry.loss_tangent, true);
  });
  return buffer;
}

function pointTriangleDistanceSq(p: Vec3, a: Vec3, b: Vec3, c: Vec3): number {
  // Barycentric projection; standard implementation.
  const ab: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const ap: Vec3 = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
  const d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];
  if (d1 <= 0 && d2 <= 0) return ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2];
  const bp: Vec3 = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
  const d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
  const d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];
  if (d3 >= 0 && d4 <= d3) return bp[0] * bp[0] + bp[1] * bp[1] + bp[2] * bp[2];
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const proj: Vec3 = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
    const dx = p[0] - proj[0], dy = p[1] - proj[1], dz = p[2] - proj[2];
    return dx * dx + dy * dy + dz * dz;
  }
  const cp: Vec3 = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
  const d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
  const d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];
  if (d6 >= 0 && d5 <= d6) return cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2];
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const proj: Vec3 = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
    const dx = p[0] - proj[0], dy = p[1] - proj[1], dz = p[2] - proj[2];
    return dx * dx + dy * dy + dz * dz;
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + d5 - d6);
    const proj: Vec3 = [b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])];
    const dx = p[0] - proj[0], dy = p[1] - proj[1], dz = p[2] - proj[2];
    return dx * dx + dy * dy + dz * dz;
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom;
  const proj: Vec3 = [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w];
  const dx = p[0] - proj[0], dy = p[1] - proj[1], dz = p[2] - proj[2];
  return dx * dx + dy * dy + dz * dz;
}

/**
 * Build a Delaunay-triangulated FEM mesh from the same geometry as buildFEMMesh.
 * Uses the tensor-grid points (snapped to box faces and sheet planes) but
 * triangulates via Bowyer-Watson Delaunay instead of Kuhn subdivision.
 */
export function buildDelaunayMesh(ir: ModelIR, meshSizeM: number, port?: FEMPortTerminals): FEMMeshArtifact {
  const diagnostics = validateModelIR(ir);
  const firstError = diagnostics.find((item) => item.severity === "error");
  if (firstError) throw new Error(`Invalid ModelIR: ${firstError.path} ${firstError.message}`);
  if (!Number.isFinite(meshSizeM) || meshSizeM <= 0) throw new Error("mesh_size_m must be a positive finite value");
  const boxEntities = ir.entities.filter((entity): entity is BoxEntity => entity.kind === "box");
  const sheetEntities = ir.entities.filter((entity): entity is SheetRectEntity => entity.kind === "sheetRect");
  if (boxEntities.length === 0 && sheetEntities.length === 0) {
    throw new Error("Volume FEM requires at least one box or sheetRect entity");
  }

  const materialByName = new Map(ir.materials.map((material) => [material.name, material]));
  const sortedNames = [...materialByName.keys()].sort();
  const attributeByMaterial = new Map(sortedNames.map((name, index) => [name, FEM_REGION_MATERIAL_BASE + index]));
  const boxes: MeshedBox[] = boxEntities.map((entity) => {
    const material = materialByName.get(entity.material);
    if (!material) throw new Error(`Unknown material for box ${entity.id}: ${entity.material}`);
    return boxFromEntity(entity, attributeByMaterial.get(entity.material)!, material);
  });
  const sheets: MeshedSheet[] = sheetEntities.map((entity) => {
    const material = materialByName.get(entity.material);
    if (!material) throw new Error(`Unknown material for sheetRect ${entity.id}: ${entity.material}`);
    return sheetFromEntity(entity, material);
  });

  for (let left = 0; left < boxes.length; left += 1) {
    for (let right = left + 1; right < boxes.length; right += 1) {
      const a = boxes[left]!;
      const b = boxes[right]!;
      const overlap = ([0, 1, 2] as const).every((axis) => a.min[axis] < b.max[axis] && a.max[axis] > b.min[axis]);
      if (overlap) throw new Error(`Overlapping box volumes are ambiguous: ${a.id} and ${b.id}`);
    }
  }

  if (ir.boundaries.length === 0) {
    throw new Error("missing_air_domain: a radiation or PML boundary with padding_m is required for volume FEM");
  }
  const padding = Math.max(...ir.boundaries.map((boundary) => boundary.padding_m));

  const airMin: Vec3 = [Infinity, Infinity, Infinity];
  const airMax: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const box of boxes) {
    for (let axis = 0 as 0 | 1 | 2; axis < 3; axis += 1) {
      airMin[axis] = Math.min(airMin[axis], box.min[axis]!);
      airMax[axis] = Math.max(airMax[axis], box.max[axis]!);
    }
  }
  for (const sheet of sheets) {
    const axes = [sheet.axis, sheet.uAxis, sheet.vAxis] as const;
    const mins = [sheet.plane, sheet.uMin, sheet.vMin] as const;
    const maxs = [sheet.plane, sheet.uMax, sheet.vMax] as const;
    for (let i = 0; i < 3; i += 1) {
      const axis = axes[i]!;
      airMin[axis] = Math.min(airMin[axis]!, mins[i]!);
      airMax[axis] = Math.max(airMax[axis]!, maxs[i]!);
    }
  }
  const airBox: MeshedBox = {
    id: "airbox",
    min: [snap(airMin[0]! - padding), snap(airMin[1]! - padding), snap(airMin[2]! - padding)],
    max: [snap(airMax[0]! + padding), snap(airMax[1]! + padding), snap(airMax[2]! + padding)],
    attribute: FEM_REGION_AIR,
    conductor: false,
    airLike: true,
  };
  const allBoxes = [...boxes, airBox];
  const boxByAttribute = new Map(allBoxes.map((box) => [box.attribute, box]));

  // Same tensor-grid planes as buildFEMMesh (box faces + sheet planes/footprints).
  const hintByRegion = new Map(ir.meshHints.map((hint) => [hint.region, hint.max_size_m]));
  const planes: number[][] = [[], [], []];
  for (let axis = 0 as 0 | 1 | 2; axis < 3; axis += 1) {
    const faceSet = new Set<number>(allBoxes.flatMap((box) => [box.min[axis], box.max[axis]]));
    for (const sheet of sheets) {
      if (sheet.axis === axis) faceSet.add(sheet.plane);
      if (sheet.uAxis === axis) {
        faceSet.add(sheet.uMin);
        faceSet.add(sheet.uMax);
      }
      if (sheet.vAxis === axis) {
        faceSet.add(sheet.vMin);
        faceSet.add(sheet.vMax);
      }
    }
    const faces = [...faceSet].sort((x, y) => x - y);
    const axisPlanes: number[] = [];
    for (let i = 0; i < faces.length - 1; i += 1) {
      const a = faces[i]!;
      const b = faces[i + 1]!;
      let required = meshSizeM;
      for (const box of allBoxes) {
        const hint = hintByRegion.get(box.id);
        if (hint != null && box.min[axis]! <= a + COORD_SNAP_M && box.max[axis]! >= b - COORD_SNAP_M) {
          required = Math.min(required, hint);
        }
      }
      const count = Math.max(1, Math.ceil((b - a) / required - 1e-9));
      for (let k = 0; k < count; k += 1) axisPlanes.push(a + ((b - a) * k) / count);
    }
    axisPlanes.push(faces[faces.length - 1]!);
    planes[axis] = [...new Set(axisPlanes.map(snap))].sort((x, y) => x - y);
  }

  // Tensor-product points.
  const nx = planes[0]!.length;
  const ny = planes[1]!.length;
  const nz = planes[2]!.length;
  const vertices = new Float64Array(nx * ny * nz * 3);
  let vi = 0;
  for (let iz = 0; iz < nz; iz += 1) {
    for (let iy = 0; iy < ny; iy += 1) {
      for (let ix = 0; ix < nx; ix += 1) {
        vertices[vi++] = planes[0]![ix]!;
        vertices[vi++] = planes[1]![iy]!;
        vertices[vi++] = planes[2]![iz]!;
      }
    }
  }

  // Delaunay triangulation.
  const t0 = Date.now();
  const tets = delaunay3d(vertices);
  const meshTimeMs = Date.now() - t0;
  const tetCount = tets.length / 4;

  // Classify tets by centroid (innermost box wins; boxes never overlap).
  const tetRegions = new Uint32Array(tetCount);
  for (let t = 0; t < tetCount; t += 1) {
    const v0 = tets[t * 4]! * 3;
    const v1 = tets[t * 4 + 1]! * 3;
    const v2 = tets[t * 4 + 2]! * 3;
    const v3 = tets[t * 4 + 3]! * 3;
    const cx = (vertices[v0]! + vertices[v1]! + vertices[v2]! + vertices[v3]!) / 4;
    const cy = (vertices[v0 + 1]! + vertices[v1 + 1]! + vertices[v2 + 1]! + vertices[v3 + 1]!) / 4;
    const cz = (vertices[v0 + 2]! + vertices[v1 + 2]! + vertices[v2 + 2]! + vertices[v3 + 2]!) / 4;
    let region = FEM_REGION_AIR;
    for (const box of boxes) {
      if (cx > box.min[0]! && cx < box.max[0]! && cy > box.min[1]! && cy < box.max[1]! && cz > box.min[2]! && cz < box.max[2]!) {
        region = box.attribute;
        break;
      }
    }
    tetRegions[t] = region;
  }

  // Tet quality: radius ratio (inradius/circumradius * 3), 1 = equilateral.
  let minQuality = 1;
  for (let t = 0; t < tetCount; t += 1) {
    const v = [tets[t * 4]!, tets[t * 4 + 1]!, tets[t * 4 + 2]!, tets[t * 4 + 3]!];
    const p = v.map((idx) => [vertices[idx * 3]!, vertices[idx * 3 + 1]!, vertices[idx * 3 + 2]!] as Vec3);
    // Volume.
    const ax = p[1]![0] - p[0]![0], ay = p[1]![1] - p[0]![1], az = p[1]![2] - p[0]![2];
    const bx = p[2]![0] - p[0]![0], by = p[2]![1] - p[0]![1], bz = p[2]![2] - p[0]![2];
    const cx2 = p[3]![0] - p[0]![0], cy2 = p[3]![1] - p[0]![1], cz2 = p[3]![2] - p[0]![2];
    const vol = Math.abs(ax * (by * cz2 - bz * cy2) - ay * (bx * cz2 - bz * cx2) + az * (bx * cy2 - by * cx2)) / 6;
    if (vol < 1e-30) continue;
    // Face areas.
    const faceArea = (a: Vec3, b: Vec3, c: Vec3): number => {
      const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
      const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
      const nx2 = uy * vz - uz * vy, ny2 = uz * vx - ux * vz, nz2 = ux * vy - uy * vx;
      return Math.sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2) / 2;
    };
    const s = faceArea(p[0]!, p[1]!, p[2]!) + faceArea(p[0]!, p[1]!, p[3]!) + faceArea(p[0]!, p[2]!, p[3]!) + faceArea(p[1]!, p[2]!, p[3]!);
    const inR = (3 * vol) / s;
    // Circumradius via edge lengths (Cayley-Menger would be overkill; use max edge / sqrt(8/3) approx? No — compute properly).
    // For quality metric, use inradius / (max edge length). Equilateral: inR = edge * sqrt(6)/12 ≈ 0.204*edge.
    let maxEdge = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        const dx = p[i]![0] - p[j]![0], dy = p[i]![1] - p[j]![1], dz = p[i]![2] - p[j]![2];
        maxEdge = Math.max(maxEdge, Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
    // Normalized: 1 for equilateral (inR = maxEdge * sqrt(6)/12).
    const q = (inR / maxEdge) / (Math.sqrt(6) / 12);
    minQuality = Math.min(minQuality, q);
  }

  const warnings: string[] = [`Delaunay triangulation took ${meshTimeMs} ms.`];
  if (minQuality < 0.2) {
    warnings.push(`Mesh contains low-quality tets (min radius ratio ${minQuality.toFixed(3)}).`);
  }

  // Facet inventory.
  const TET_FACES = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] as const;
  const facetKey = (a: number, b: number, c: number): string => {
    const s = [a, b, c].sort((x, y) => x - y);
    return `${s[0]},${s[1]},${s[2]}`;
  };
  const facetVerts: number[] = [];
  const facetRegionA: number[] = [];
  const facetRegionB: number[] = [];
  const facetIndexByKey = new Map<string, number>();
  for (let t = 0; t < tetCount; t += 1) {
    const v = [tets[t * 4]!, tets[t * 4 + 1]!, tets[t * 4 + 2]!, tets[t * 4 + 3]!];
    for (const face of TET_FACES) {
      const key = facetKey(v[face[0]]!, v[face[1]]!, v[face[2]]!);
      const existing = facetIndexByKey.get(key);
      if (existing === undefined) {
        facetIndexByKey.set(key, facetVerts.length / 3);
        facetVerts.push(v[face[0]]!, v[face[1]]!, v[face[2]]!);
        facetRegionA.push(tetRegions[t]!);
        facetRegionB.push(-1);
      } else {
        facetRegionB[existing] = tetRegions[t]!;
      }
    }
  }

  // Boundary classification.
  const facetCount = facetVerts.length / 3;
  const boundaryFacets: number[] = [];
  const boundaryAttributes: number[] = [];
  const boundaryIndexOfFacet = new Array<number>(facetCount).fill(-1);
  const facetIsConductorAir = new Array<boolean>(facetCount).fill(false);
  const facetIsSurface = new Array<boolean>(facetCount).fill(false);
  const vertexCoord = (index: number): Vec3 => [vertices[index * 3]!, vertices[index * 3 + 1]!, vertices[index * 3 + 2]!];
  let nonAirExterior = false;
  for (let f = 0; f < facetCount; f += 1) {
    const regionA = boxByAttribute.get(facetRegionA[f]!)!;
    const regionB = facetRegionB[f]! === -1 ? null : boxByAttribute.get(facetRegionB[f]!);
    let attribute: number | null = null;
    let conductorAir = false;
    if (regionB === null) {
      attribute = FEM_ATTR_EXTERIOR_ABSORBING;
      facetIsSurface[f] = true;
      if (!regionA.airLike) nonAirExterior = true;
    } else if (facetRegionA[f] !== facetRegionB[f]) {
      const regionBnn = regionB!;
      const conductorSide = regionA.conductor ? regionA : regionBnn.conductor ? regionBnn : null;
      const otherSide = conductorSide === regionA ? regionBnn : regionA;
      if (conductorSide && !otherSide.conductor) {
        attribute = FEM_ATTR_CONDUCTOR_PEC;
        conductorAir = otherSide.airLike;
      }
      facetIsSurface[f] = true;
    }
    if (attribute !== null) {
      boundaryIndexOfFacet[f] = boundaryFacets.length / 3;
      boundaryFacets.push(facetVerts[f * 3]!, facetVerts[f * 3 + 1]!, facetVerts[f * 3 + 2]!);
      boundaryAttributes.push(attribute);
      if (conductorAir) facetIsConductorAir[f] = true;
    }
  }
  if (nonAirExterior) {
    warnings.push("A non-air material reaches the airbox exterior; the absorbing boundary was applied anyway.");
  }

  // Sheet PEC tagging.
  const SHEET_PLANE_TOL_M = 1e-9;
  for (const sheet of sheets) {
    let tagged = 0;
    for (let f = 0; f < facetCount; f += 1) {
      if (boundaryIndexOfFacet[f] !== -1) continue;
      const va = vertexCoord(facetVerts[f * 3]!);
      const vb = vertexCoord(facetVerts[f * 3 + 1]!);
      const vc = vertexCoord(facetVerts[f * 3 + 2]!);
      if (
        Math.abs(va[sheet.axis]! - sheet.plane) > SHEET_PLANE_TOL_M ||
        Math.abs(vb[sheet.axis]! - sheet.plane) > SHEET_PLANE_TOL_M ||
        Math.abs(vc[sheet.axis]! - sheet.plane) > SHEET_PLANE_TOL_M
      ) {
        continue;
      }
      const inside = (v: Vec3): boolean =>
        v[sheet.uAxis]! >= sheet.uMin - SHEET_PLANE_TOL_M &&
        v[sheet.uAxis]! <= sheet.uMax + SHEET_PLANE_TOL_M &&
        v[sheet.vAxis]! >= sheet.vMin - SHEET_PLANE_TOL_M &&
        v[sheet.vAxis]! <= sheet.vMax + SHEET_PLANE_TOL_M;
      if (!inside(va) || !inside(vb) || !inside(vc)) continue;
      boundaryIndexOfFacet[f] = boundaryFacets.length / 3;
      boundaryFacets.push(facetVerts[f * 3]!, facetVerts[f * 3 + 1]!, facetVerts[f * 3 + 2]!);
      boundaryAttributes.push(FEM_ATTR_CONDUCTOR_PEC);
      facetIsConductorAir[f] = true;
      facetIsSurface[f] = true;
      tagged += 1;
    }
    if (tagged === 0) {
      warnings.push(`sheetRect ${sheet.id} tagged no mesh facets.`);
    }
  }

  // Port mapping (same as mesh.ts contract C).
  let portPositiveFacets = new Uint32Array();
  let portNegativeFacets = new Uint32Array();
  if (port) {
    const portLengthM = Math.hypot(
      port.positive[0] - port.negative[0],
      port.positive[1] - port.negative[1],
      port.positive[2] - port.negative[2],
    );
    const resolveTerminal = (terminal: Vec3, label: string): number[] => {
      const tol = Math.max(1e-6, 0.25 * Math.min(meshSizeM, portLengthM));
      const tolSq = tol * tol;
      const facetDistanceSq = (f: number): number => {
        const a = vertexCoord(facetVerts[f * 3]!);
        const b = vertexCoord(facetVerts[f * 3 + 1]!);
        const c = vertexCoord(facetVerts[f * 3 + 2]!);
        return pointTriangleDistanceSq(terminal, a, b, c);
      };
      const candidates: number[] = [];
      for (let f = 0; f < facetCount; f += 1) {
        if (!facetIsConductorAir[f]) continue;
        if (facetDistanceSq(f) <= tolSq) candidates.push(boundaryIndexOfFacet[f]!);
      }
      if (candidates.length === 0) {
        let touchedSurface = false;
        for (let f = 0; f < facetCount && !touchedSurface; f += 1) {
          if (facetIsSurface[f] && facetDistanceSq(f) <= tolSq) touchedSurface = true;
        }
        if (touchedSurface) {
          throw new Error(`invalid_port_material_contact: ${label} terminal of port ${port.id} is not on a conductor-air interface`);
        }
        throw new Error(`invalid_port_no_contact: ${label} terminal of port ${port.id} touches no mesh facet within tolerance`);
      }
      const candidateSet = new Set(candidates);
      const edgeToFacets = new Map<string, number[]>();
      for (const f of candidates) {
        const vs = [boundaryFacets[f * 3]!, boundaryFacets[f * 3 + 1]!, boundaryFacets[f * 3 + 2]!].sort((x, y) => x - y);
        for (const [p, q] of [[vs[0], vs[1]], [vs[1], vs[2]], [vs[0], vs[2]]] as const) {
          const key = `${p},${q}`;
          const list = edgeToFacets.get(key) ?? [];
          list.push(f);
          edgeToFacets.set(key, list);
        }
      }
      const visited = new Set<number>();
      const components: number[][] = [];
      for (const start of candidates) {
        if (visited.has(start)) continue;
        const component: number[] = [];
        const stack = [start];
        visited.add(start);
        while (stack.length > 0) {
          const f = stack.pop()!;
          component.push(f);
          const vs = [boundaryFacets[f * 3]!, boundaryFacets[f * 3 + 1]!, boundaryFacets[f * 3 + 2]!].sort((x, y) => x - y);
          for (const [p, q] of [[vs[0], vs[1]], [vs[1], vs[2]], [vs[0], vs[2]]] as const) {
            for (const neighbor of edgeToFacets.get(`${p},${q}`) ?? []) {
              if (candidateSet.has(neighbor) && !visited.has(neighbor)) {
                visited.add(neighbor);
                stack.push(neighbor);
              }
            }
          }
        }
        components.push(component);
      }
      if (components.length > 1) {
        throw new Error(`invalid_port_ambiguous_contact: ${label} terminal of port ${port.id} touches ${components.length} disconnected conductor-air patches`);
      }
      return components[0]!.sort((x, y) => x - y);
    };
    const positive = resolveTerminal(port.positive, "positive");
    const negative = resolveTerminal(port.negative, "negative");
    const overlap = positive.filter((f) => negative.includes(f));
    if (overlap.length > 0) {
      throw new Error(`invalid_port_ambiguous_contact: port ${port.id} terminal facet sets overlap`);
    }
    portPositiveFacets = new Uint32Array(positive);
    portNegativeFacets = new Uint32Array(negative);
  } else {
    warnings.push("No lumped port was mapped; port facet sets are empty.");
  }

  const materials: FEMMeshMaterialEntry[] = [
    {
      attribute: FEM_REGION_AIR,
      name: "airbox",
      epsilon_r: 1,
      mu_r: 1,
      conductivity_s_m: 0,
      loss_tangent: 0,
    },
    ...sortedNames.map((name) => {
      const material = materialByName.get(name)!;
      return {
        attribute: attributeByMaterial.get(name)!,
        name,
        epsilon_r: material.epsilon_r,
        mu_r: material.mu_r,
        conductivity_s_m: material.conductivity_s_m ?? 0,
        loss_tangent: material.loss_tangent ?? 0,
      };
    }),
  ];

  return {
    version: 1,
    units: "m",
    vertices,
    tetrahedra: tets,
    tetRegions,
    materialTable: buildMaterialTable(materials),
    materials,
    boundaryFacets: new Uint32Array(boundaryFacets),
    boundaryAttributes: new Uint32Array(boundaryAttributes),
    portPositiveFacets,
    portNegativeFacets,
    statistics: {
      nodes: vertices.length / 3,
      elements: tetCount,
      min_quality: minQuality,
      material_count: materials.length,
      boundary_count: boundaryFacets.length / 3,
    },
    warnings,
  };
}
