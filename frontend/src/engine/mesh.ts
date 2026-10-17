import type { BoxEntity, Material, ModelIR, SheetRectEntity, Vec3 } from "../dsl/model-ir";
import { validateModelIR } from "../dsl/model-ir";

/**
 * Conforming tetrahedral mesh pipeline for the M4 volume FEM path.
 *
 * Implements frozen contracts B (mesh artifact v1) and C (port mapping) from
 * docs/mfem-wasm-simulation-results-plan.md:
 *
 * - Box and zero-thickness `sheetRect` geometry. Sheets must be conductive
 *   (PEC) and axis-aligned; they contribute a grid plane plus footprint
 *   boundaries, and their interior facets are tagged as conductor PEC.
 * - The outer air volume is generated from `ModelIR.boundaries[*].padding_m`
 *   (largest padding wins); a missing boundary fails with `missing_air_domain`.
 * - All boxes are placed on one tensor-product grid whose planes are snapped to
 *   every box face, so touching interfaces share vertices/facets exactly
 *   (conforming topology, deterministic ordering).
 * - Region attributes: 1 = airbox, 1000 + n = user materials in sorted-name order.
 * - Boundary attributes: 10 = exterior_absorbing, 20 = conductor_pec,
 *   30 = port_positive, 31 = port_negative.
 * - Port facet sets are indices into `boundaryFacets`. A port facet keeps its
 *   boundary attribute (20: it is a conductor-air interface); the matching
 *   `port_attr` entry (30/31) takes precedence when the solver applies
 *   boundary conditions, per contract C ("port facets are excluded from PEC").
 * - The 40-byte material records follow contract A1.
 *
 * Deliberate simplifications vs. the frozen text (documented, not silent):
 * - "Local edge length" for the port tolerance is approximated by the
 *   requested mesh size; grid snapping only ever makes cells smaller, so the
 *   resulting tolerance is conservative. It is additionally capped at
 *   0.25 * port length so a short port (e.g. a 1 mm probe feed) cannot
 *   resolve the opposite terminal's conductor surface as its own.
 * - A port terminal's facet set is the within-tolerance seed completed to
 *   the whole connected, coplanar conductor-air patch containing it (see
 *   resolveTerminal): point distance alone under-covers a terminal face
 *   whose triangles' nearest points are a quad diagonal away from the
 *   terminal point, which made the resolved footprint mesh-density
 *   dependent.
 * - A material counts as a conductor when `conductivity_s_m >= 1e4`; a
 *   material counts as air-like for port candidacy when it is the airbox or
 *   has epsilon_r == 1, mu_r == 1 and no conductivity/loss.
 * - Mesh hints refine whole tensor slabs covering the hinted region's axis
 *   range (tensor grids cannot refine a single brick in isolation).
 */

export const FEM_REGION_AIR = 1;
export const FEM_REGION_MATERIAL_BASE = 1000;
export const FEM_ATTR_EXTERIOR_ABSORBING = 10;
export const FEM_ATTR_CONDUCTOR_PEC = 20;
export const FEM_ATTR_PORT_POSITIVE = 30;
export const FEM_ATTR_PORT_NEGATIVE = 31;
/** Conductivity at/above which a material is treated as a conductor (PEC). */
export const FEM_CONDUCTOR_CONDUCTIVITY_S_M = 1e4;
/** Grid planes closer than this are merged (float noise from center +/- size/2). */
const COORD_SNAP_M = 1e-12;

export interface FEMMeshMaterialEntry {
  attribute: number;
  name: string;
  epsilon_r: number;
  mu_r: number;
  conductivity_s_m: number;
  loss_tangent: number;
}

/** Minimal lumped-port terminal description needed for facet mapping. */
export interface FEMPortTerminals {
  id: string;
  positive: Vec3;
  negative: Vec3;
}

export interface FEMMeshArtifact {
  version: 1;
  units: "m";
  vertices: Float64Array;
  tetrahedra: Uint32Array;
  /** Per-tet region attribute: 1 (airbox) or 1000 + material index. */
  tetRegions: Uint32Array;
  /** 40-byte material records per contract A1, sorted by region attribute. */
  materialTable: Uint8Array;
  materials: FEMMeshMaterialEntry[];
  /** Packed vertex triples, one facet per 3 entries. */
  boundaryFacets: Uint32Array;
  /** Per-facet attribute: 10 (exterior_absorbing) or 20 (conductor_pec). */
  boundaryAttributes: Uint32Array;
  /** Indices into boundaryFacets for the positive / negative terminal sets. */
  portPositiveFacets: Uint32Array;
  portNegativeFacets: Uint32Array;
  statistics: {
    nodes: number;
    elements: number;
    min_quality: number;
    material_count: number;
    boundary_count: number;
  };
  warnings: string[];
}

export interface MeshedBox {
  id: string;
  min: Vec3;
  max: Vec3;
  attribute: number;
  conductor: boolean;
  airLike: boolean;
}

/**
 * A zero-thickness PEC sheet (sheetRect entity).
 * - axis: the normal axis (0=x, 1=y, 2=z).
 * - plane: coordinate along the normal axis.
 * - uAxis/uMin/uMax, vAxis/vMin/vMax: in-plane extents.
 * Only axis-aligned sheets of conductive material are supported.
 */
export interface MeshedSheet {
  id: string;
  axis: 0 | 1 | 2;
  plane: number;
  uAxis: 0 | 1 | 2;
  uMin: number;
  uMax: number;
  vAxis: 0 | 1 | 2;
  vMin: number;
  vMax: number;
}

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

/**
 * Convert a sheetRect entity to a MeshedSheet.
 * Only axis-aligned normals (±x, ±y, ±z) and conductive materials are supported.
 * The in-plane axes (u, v) are the two axes orthogonal to the normal, in
 * ascending axis order (e.g. normal=z → u=x, v=y). size=[su, sv] maps to
 * [u extent, v extent].
 */
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
  const plane = snap(entity.center[axis]);
  const uMin = snap(entity.center[uAxis]! - entity.size[0]! / 2);
  const uMax = snap(entity.center[uAxis]! + entity.size[0]! / 2);
  const vMin = snap(entity.center[vAxis]! - entity.size[1]! / 2);
  const vMax = snap(entity.center[vAxis]! + entity.size[1]! / 2);
  return { id: entity.id, axis, plane, uAxis, uMin, uMax, vAxis, vMin, vMax };
}

function buildMaterialTable(entries: FEMMeshMaterialEntry[]): Uint8Array {
  const buffer = new Uint8Array(entries.length * 40);
  const view = new DataView(buffer.buffer);
  entries.forEach((entry, index) => {
    const offset = index * 40;
    view.setUint32(offset, entry.attribute, true);
    view.setUint32(offset + 4, 0, true); // reserved
    view.setFloat64(offset + 8, entry.epsilon_r, true);
    view.setFloat64(offset + 16, entry.mu_r, true);
    view.setFloat64(offset + 24, entry.conductivity_s_m, true);
    view.setFloat64(offset + 32, entry.loss_tangent, true);
  });
  return buffer;
}

/** Squared distance from point p to triangle (a, b, c). */
function pointTriangleDistanceSq(p: Vec3, a: Vec3, b: Vec3, c: Vec3): number {
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
    const q: Vec3 = [a[0] + v * ab[0] - p[0], a[1] + v * ab[1] - p[1], a[2] + v * ab[2] - p[2]];
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
  }
  const cp: Vec3 = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
  const d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
  const d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];
  if (d6 >= 0 && d5 <= d6) return cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2];
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const q: Vec3 = [a[0] + w * ac[0] - p[0], a[1] + w * ac[1] - p[1], a[2] + w * ac[2] - p[2]];
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    const q: Vec3 = [b[0] + w * (c[0] - b[0]) - p[0], b[1] + w * (c[1] - b[1]) - p[1], b[2] + w * (c[2] - b[2]) - p[2]];
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  const q: Vec3 = [a[0] + ab[0] * v + ac[0] * w - p[0], a[1] + ab[1] * v + ac[1] * w - p[1], a[2] + ab[2] * v + ac[2] * w - p[2]];
  return q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
}

/**
 * Build a conforming tetrahedral volume mesh for the M4 FEM path.
 *
 * @param ir validated parametric model IR (box entities only)
 * @param meshSizeM target maximum cell edge length in metres
 * @param port optional lumped port; when given, its terminal facet sets are
 *             resolved per contract C, otherwise the port sets stay empty
 */
export function buildFEMMesh(ir: ModelIR, meshSizeM: number, port?: FEMPortTerminals): FEMMeshArtifact {
  const diagnostics = validateModelIR(ir);
  const firstError = diagnostics.find((item) => item.severity === "error");
  if (firstError) throw new Error(`Invalid ModelIR: ${firstError.path} ${firstError.message}`);
  if (!Number.isFinite(meshSizeM) || meshSizeM <= 0) throw new Error("mesh_size_m must be a positive finite value");
  const boxEntities = ir.entities.filter((entity): entity is BoxEntity => entity.kind === "box");
  const sheetEntities = ir.entities.filter((entity): entity is SheetRectEntity => entity.kind === "sheetRect");
  if (boxEntities.length === 0 && sheetEntities.length === 0) {
    throw new Error("Volume FEM requires at least one box or sheetRect entity");
  }
  const entities = boxEntities;

  const materialByName = new Map(ir.materials.map((material) => [material.name, material]));
  const sortedNames = [...materialByName.keys()].sort();
  const attributeByMaterial = new Map(sortedNames.map((name, index) => [name, FEM_REGION_MATERIAL_BASE + index]));
  const boxes: MeshedBox[] = entities.map((entity) => {
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

  // Explicit outer air volume enclosing every solid region and sheet.
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
  for (const box of boxes) {
    for (let axis = 0 as 0 | 1 | 2; axis < 3; axis += 1) {
      if (box.min[axis]! < airBox.min[axis]! || box.max[axis]! > airBox.max[axis]!) {
        throw new Error(`Box ${box.id} is not embedded in the generated air volume`);
      }
    }
  }
  const allBoxes = [...boxes, airBox];
  const boxByAttribute = new Map(allBoxes.map((box) => [box.attribute, box]));

  // Tensor-product grid: planes snapped to every box face, then subdivided so
  // no interval exceeds the requested size (or an applicable mesh hint).
  // Zero-thickness sheets contribute their normal plane plus their in-plane
  // footprint boundaries, so sheet facets align exactly with the footprint.
  const hintByRegion = new Map(ir.meshHints.map((hint) => [hint.region, hint]));
  const axisName = ["x", "y", "z"] as const;
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
          // Per-axis override lets thin slabs (e.g. substrates) refine only
          // through their thickness without exploding the in-plane grid.
          const axisSize = hint.max_size_by_axis_m?.[axisName[axis]] ?? hint.max_size_m;
          required = Math.min(required, axisSize);
        }
      }
      const count = Math.max(1, Math.ceil((b - a) / required - 1e-9));
      for (let k = 0; k < count; k += 1) axisPlanes.push(a + ((b - a) * k) / count);
    }
    axisPlanes.push(faces[faces.length - 1]!);
    planes[axis] = [...new Set(axisPlanes.map(snap))].sort((x, y) => x - y);
  }

  const nx = planes[0]!.length - 1;
  const ny = planes[1]!.length - 1;
  const nz = planes[2]!.length - 1;
  const vertexAt = (ix: number, iy: number, iz: number): number => iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix;
  const vertices = new Float64Array((nx + 1) * (ny + 1) * (nz + 1) * 3);
  for (let iz = 0; iz <= nz; iz += 1) {
    for (let iy = 0; iy <= ny; iy += 1) {
      for (let ix = 0; ix <= nx; ix += 1) {
        const v = vertexAt(ix, iy, iz) * 3;
        vertices[v] = planes[0]![ix]!;
        vertices[v + 1] = planes[1]![iy]!;
        vertices[v + 2] = planes[2]![iz]!;
      }
    }
  }
  const vertexCoord = (index: number): Vec3 => [vertices[index * 3]!, vertices[index * 3 + 1]!, vertices[index * 3 + 2]!];

  // Cells: each brick belongs to exactly one region (innermost box wins; the
  // airbox catches the rest). Boxes never overlap, so this is unambiguous.
  const cellRegion: number[] = [];
  let minQuality = 1;
  const warnings: string[] = [];
  for (let iz = 0; iz < nz; iz += 1) {
    for (let iy = 0; iy < ny; iy += 1) {
      for (let ix = 0; ix < nx; ix += 1) {
        const x0 = planes[0]![ix]!;
        const x1 = planes[0]![ix + 1]!;
        const y0 = planes[1]![iy]!;
        const y1 = planes[1]![iy + 1]!;
        const z0 = planes[2]![iz]!;
        const z1 = planes[2]![iz + 1]!;
        const centroid: Vec3 = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
        let region = FEM_REGION_AIR;
        for (const box of boxes) {
          if (
            centroid[0] > box.min[0]! && centroid[0] < box.max[0]! &&
            centroid[1] > box.min[1]! && centroid[1] < box.max[1]! &&
            centroid[2] > box.min[2]! && centroid[2] < box.max[2]!
          ) {
            region = box.attribute;
            break;
          }
        }
        const dx = x1 - x0;
        const dy = y1 - y0;
        const dz = z1 - z0;
        minQuality = Math.min(minQuality, Math.min(dx, dy, dz) / Math.max(dx, dy, dz));
        cellRegion.push(region);
      }
    }
  }
  if (minQuality < 0.2) {
    warnings.push(
      `Mesh contains high-aspect-ratio cells (min quality ${minQuality.toFixed(3)}); thin features may degrade solver conditioning.`,
    );
  }

  // Six tetrahedra per cell (Kuhn subdivision along the v000->v111 diagonal).
  // All cells share the same diagonal direction, so face triangulations match
  // across neighboring cells and the mesh is conforming.
  const tets: number[] = [];
  const tetRegions: number[] = [];
  const cellIndex = (ix: number, iy: number, iz: number): number => iz * ny * nx + iy * nx + ix;
  for (let iz = 0; iz < nz; iz += 1) {
    for (let iy = 0; iy < ny; iy += 1) {
      for (let ix = 0; ix < nx; ix += 1) {
        const v000 = vertexAt(ix, iy, iz);
        const v100 = vertexAt(ix + 1, iy, iz);
        const v010 = vertexAt(ix, iy + 1, iz);
        const v110 = vertexAt(ix + 1, iy + 1, iz);
        const v001 = vertexAt(ix, iy, iz + 1);
        const v101 = vertexAt(ix + 1, iy, iz + 1);
        const v011 = vertexAt(ix, iy + 1, iz + 1);
        const v111 = vertexAt(ix + 1, iy + 1, iz + 1);
        const pattern = [
          [v000, v100, v110, v111],
          [v000, v110, v010, v111],
          [v000, v010, v011, v111],
          [v000, v011, v001, v111],
          [v000, v001, v101, v111],
          [v000, v101, v100, v111],
        ];
        const region = cellRegion[cellIndex(ix, iy, iz)]!;
        for (const tet of pattern) {
          tets.push(...tet);
          tetRegions.push(region);
        }
      }
    }
  }

  // Facet inventory: every tet face, deduplicated by sorted vertex triple.
  // Interior faces shared by two tets appear once; exterior faces once.
  const TET_FACES = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] as const;
  const facetKey = (a: number, b: number, c: number): string => {
    const s = [a, b, c].sort((x, y) => x - y);
    return `${s[0]},${s[1]},${s[2]}`;
  };
  const facetVerts: number[] = [];
  const facetRegionA: number[] = [];
  const facetRegionB: number[] = []; // -1 for exterior faces
  const facetIndexByKey = new Map<string, number>();
  const tetCount = tetRegions.length;
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

  // Boundary classification per contract B.
  const facetCount = facetVerts.length / 3;
  const boundaryFacets: number[] = [];
  const boundaryAttributes: number[] = [];
  // All-facet index -> index into boundaryFacets (-1 when the facet is untagged).
  const boundaryIndexOfFacet = new Array<number>(facetCount).fill(-1);
  // True for conductor-air PEC facets: the only valid port candidates (contract C).
  const facetIsConductorAir = new Array<boolean>(facetCount).fill(false);
  // True for facets on the domain exterior or a material interface. Interior
  // same-region faces (including Kuhn diagonal faces) are not surfaces.
  const facetIsSurface = new Array<boolean>(facetCount).fill(false);
  let nonAirExterior = false;
  for (let f = 0; f < facetCount; f += 1) {
    const regionA = boxByAttribute.get(facetRegionA[f]!)!;
    const regionB = facetRegionB[f] === -1 ? null : boxByAttribute.get(facetRegionB[f]!)!;
    let attribute: number | null = null;
    let conductorAir = false;
    if (regionB === null) {
      // Exterior face of the air volume.
      facetIsSurface[f] = true;
      if (regionA.conductor) {
        attribute = FEM_ATTR_CONDUCTOR_PEC;
        conductorAir = true;
      } else {
        attribute = FEM_ATTR_EXTERIOR_ABSORBING;
        if (!regionA.airLike) nonAirExterior = true;
      }
    } else if (regionA.attribute !== regionB.attribute) {
      // Material interface inside the volume.
      facetIsSurface[f] = true;
      const conductorSide = regionA.conductor ? regionA : regionB.conductor ? regionB : null;
      const otherSide = conductorSide === regionA ? regionB : regionA;
      if (conductorSide && !otherSide.conductor) {
        attribute = FEM_ATTR_CONDUCTOR_PEC;
        conductorAir = otherSide.airLike;
      }
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

  // Zero-thickness PEC sheets: tag facets lying on the sheet plane within the
  // sheet footprint as conductor PEC. These are interior facets (shared by two
  // tets) that the volume classification above skips; the solver's PEC edge
  // elimination applies to them exactly like conductor-volume surfaces.
  // Facets must be fully inside the footprint (all three vertices), not just
  // centroid-inside, since the grid planes now align with footprint edges.
  const SHEET_PLANE_TOL_M = 1e-9;
  for (const sheet of sheets) {
    let tagged = 0;
    for (let f = 0; f < facetCount; f += 1) {
      if (boundaryIndexOfFacet[f] !== -1) continue; // Already tagged (e.g. probe contact).
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
      // All vertices must lie within the footprint (with tolerance).
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
      warnings.push(`sheetRect ${sheet.id} tagged no mesh facets; it may lie outside the meshed volume or miss the grid planes.`);
    }
  }

  // Port terminal -> facet-set mapping per contract C.
  let portPositiveFacets = new Uint32Array();
  let portNegativeFacets = new Uint32Array();
  if (port) {
    // The port gap bounds the terminal tolerance: a terminal may legitimately
    // sit within one port length of the opposite terminal's conductor face
    // (e.g. a 1 mm probe feed over a ground plane), so the tolerance must be
    // a fraction of the port length or the validator invents ambiguity.
    const portLengthM = Math.hypot(
      port.positive[0] - port.negative[0],
      port.positive[1] - port.negative[1],
      port.positive[2] - port.negative[2],
    );
    const portDir: Vec3 = [
      (port.positive[0] - port.negative[0]) / portLengthM,
      (port.positive[1] - port.negative[1]) / portLengthM,
      (port.positive[2] - port.negative[2]) / portLengthM,
    ];
    const resolveTerminal = (terminal: Vec3, label: string): number[] => {
      // Tolerance per contract C: tol = max(1e-6 m, 0.25 * local edge length).
      // Local edge length is approximated by the requested mesh size; grid
      // snapping only ever makes cells smaller, so this is conservative.
      // It is additionally capped at 0.25 * port length so a short lumped
      // port cannot "see" the opposite terminal's conductor surface.
      const tol = Math.max(1e-6, 0.25 * Math.min(meshSizeM, portLengthM));
      const tolSq = tol * tol;
      const facetDistanceSq = (f: number): number => {
        const a = vertexCoord(facetVerts[f * 3]!);
        const b = vertexCoord(facetVerts[f * 3 + 1]!);
        const c = vertexCoord(facetVerts[f * 3 + 2]!);
        return pointTriangleDistanceSq(terminal, a, b, c);
      };
      // Candidate facets are conductor-air interfaces only (contract C),
      // collected as indices into boundaryFacets for the ABI port arrays.
      // The point-distance test only *seeds* the terminal patch: a triangle
      // of the terminal's own face can lie farther than tol from the
      // terminal point. With the tensor grid's shared Kuhn diagonal, the
      // two triangles across the diagonal from a centred terminal have
      // their nearest point on the quad diagonal, half a cell diagonal
      // away (0.46 mm vs the 0.125 mm tol on the xy = 1.25 mm Snyder mesh),
      // which silently dropped 2 of the 8 triangles tiling the 1.3 mm
      // probe face and made the resolved footprint mesh-density dependent.
      // The seed is therefore completed to the full connected coplanar
      // conductor-air patch below; the tolerance still decides *which*
      // patch a terminal belongs to, and the opposite terminal's face (a
      // port length away, never edge-connected) stays out.
      const seedFacets: number[] = [];
      for (let f = 0; f < facetCount; f += 1) {
        if (!facetIsConductorAir[f]) continue;
        if (facetDistanceSq(f) <= tolSq) seedFacets.push(f);
      }
      const candidates: number[] = seedFacets.map((f) => boundaryIndexOfFacet[f]!);
      if (candidates.length === 0) {
        // Distinguish "touched the wrong surface" from "touched nothing".
        // Interior same-region faces (e.g. Kuhn diagonal faces) are pure
        // meshing artifacts and do not count as contact.
        let touchedSurface = false;
        for (let f = 0; f < facetCount && !touchedSurface; f += 1) {
          if (facetIsSurface[f] && facetDistanceSq(f) <= tolSq) touchedSurface = true;
        }
        if (touchedSurface) {
          throw new Error(`invalid_port_material_contact: ${label} terminal of port ${port.id} is not on a conductor-air interface`);
        }
        throw new Error(`invalid_port_no_contact: ${label} terminal of port ${port.id} touches no mesh facet within tolerance`);
      }
      // Patch completion: grow the seed across edge-adjacent conductor-air
      // facets lying in the seed face's plane. Growth eligibility is
      // stricter than seed candidacy: a volume conductor against an
      // air-like region, an exterior conductor face, or a zero-thickness
      // sheet facet with air on *both* sides. In particular a sheet facet
      // backed by a dielectric (the ground sheet under the substrate) is
      // not eligible, so a ground terminal cannot flood its whole sheet.
      const growthEligible = (f: number): boolean => {
        const regionA = boxByAttribute.get(facetRegionA[f]!)!;
        const other = facetRegionB[f]!;
        if (other === -1) return regionA.conductor;
        const regionB = boxByAttribute.get(other)!;
        if (regionA.conductor !== regionB.conductor) {
          return (regionA.conductor ? regionB : regionA).airLike;
        }
        return regionA.airLike && regionB.airLike && boundaryIndexOfFacet[f] !== -1;
      };
      // Plane of the seed facet nearest the terminal; grid vertices are
      // snapped, so coplanar facets are coplanar to float noise.
      const COPLANAR_TOL_M = 1e-9;
      let nearest = seedFacets[0]!;
      let nearestD = Infinity;
      for (const f of seedFacets) {
        const d = facetDistanceSq(f);
        if (d < nearestD) {
          nearestD = d;
          nearest = f;
        }
      }
      const pa = vertexCoord(facetVerts[nearest * 3]!);
      const pb = vertexCoord(facetVerts[nearest * 3 + 1]!);
      const pc = vertexCoord(facetVerts[nearest * 3 + 2]!);
      const cross: Vec3 = [
        (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1]),
        (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2]),
        (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]),
      ];
      const crossLen = Math.hypot(cross[0], cross[1], cross[2]);
      if (crossLen > 0) {
        const planeNormal: Vec3 = [cross[0] / crossLen, cross[1] / crossLen, cross[2] / crossLen];
        // Completion applies only when the seed face fronts the port gap,
        // i.e. its normal is (near-)parallel to the port axis, as for the
        // facing faces of a probe feed. A terminal on a face that *contains*
        // the port axis (a coplanar gap port, both terminals on one face)
        // keeps the seed-only set: completing there would flood the shared
        // face from both terminals until the sets overlap.
        const frontsGap =
          Math.abs(
            planeNormal[0] * portDir[0] + planeNormal[1] * portDir[1] + planeNormal[2] * portDir[2],
          ) > 0.5;
        if (frontsGap) {
          const inPlane = (f: number): boolean => {
            for (let k = 0; k < 3; k += 1) {
              const v = vertexCoord(facetVerts[f * 3 + k]!);
              const d =
                (v[0] - pa[0]) * planeNormal[0] +
                (v[1] - pa[1]) * planeNormal[1] +
                (v[2] - pa[2]) * planeNormal[2];
              if (Math.abs(d) > COPLANAR_TOL_M) return false;
            }
            return true;
          };
          // Edge adjacency among growth-eligible facets (facetVerts indices).
          const growthEdgeToFacets = new Map<string, number[]>();
          for (let f = 0; f < facetCount; f += 1) {
            if (!growthEligible(f)) continue;
            const vs = [facetVerts[f * 3]!, facetVerts[f * 3 + 1]!, facetVerts[f * 3 + 2]!].sort((x, y) => x - y);
            for (const [p, q] of [[vs[0], vs[1]], [vs[1], vs[2]], [vs[0], vs[2]]] as const) {
              const key = `${p},${q}`;
              const list = growthEdgeToFacets.get(key) ?? [];
              list.push(f);
              growthEdgeToFacets.set(key, list);
            }
          }
          const grownVisited = new Set<number>(seedFacets);
          const grownStack = [...seedFacets];
          while (grownStack.length > 0) {
            const f = grownStack.pop()!;
            const vs = [facetVerts[f * 3]!, facetVerts[f * 3 + 1]!, facetVerts[f * 3 + 2]!].sort((x, y) => x - y);
            for (const [p, q] of [[vs[0], vs[1]], [vs[1], vs[2]], [vs[0], vs[2]]] as const) {
              for (const neighbor of growthEdgeToFacets.get(`${p},${q}`) ?? []) {
                if (grownVisited.has(neighbor) || !inPlane(neighbor)) continue;
                grownVisited.add(neighbor);
                candidates.push(boundaryIndexOfFacet[neighbor]!);
                grownStack.push(neighbor);
              }
            }
          }
        }
      }
      // Group into edge-connected components; more than one is ambiguous.
      // Note: candidates are indices into boundaryFacets, not facetVerts.
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
    tetrahedra: new Uint32Array(tets),
    tetRegions: new Uint32Array(tetRegions),
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
