import type { Boundary, CoaxPort, GeometryEntity, LumpedPort, Material, MeshHint, ModelIR, Vec3 } from "./model-ir";

export interface GeometryApi {
  material(name: string, spec: Omit<Material, "name">): Material;
  box(id: string, options: { center: Vec3; size: Vec3; material: string }): void;
  sheetRect(id: string, options: { center: Vec3; size: [number, number]; normal?: Vec3; material: string }): void;
  lumpedPort(id: string, options: { positive: Vec3; negative: Vec3; impedance_ohms?: number }): void;
  /**
   * Coaxial wave-port feed (see CoaxPort in model-ir): the mesher
   * injects the pin/dielectric/frame geometry and holes the feed
   * column out of the substrate and ground, so author those whole.
   */
  coaxPort(id: string, options: Omit<CoaxPort, "id" | "kind" | "impedance_ohms"> & { impedance_ohms?: number }): void;
  airbox(options: { padding_m: number }): void;
  pml(options: { padding_m: number; layers?: number }): void;
  refine(options: MeshHint): void;
}

export function createGeometryApi(paramSchemaVersion: 1): { api: GeometryApi; ir: ModelIR } {
  const ir: ModelIR = { version: 1, paramSchemaVersion, materials: [], entities: [], ports: [], boundaries: [], meshHints: [] };
  const ids = new Set<string>();
  const requireNewId = (id: string) => {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(id) || ids.has(id)) throw new Error(`Invalid geometry id "${id}"`);
    ids.add(id);
  };
  const requireMaterial = (name: string) => {
    if (!ir.materials.some((material) => material.name === name)) throw new Error(`Unknown material "${name}"`);
  };
  const api: GeometryApi = {
    material(name, spec) {
      if (!name || ir.materials.some((material) => material.name === name)) throw new Error(`Duplicate or empty material "${name}"`);
      if (!Number.isFinite(spec.epsilon_r) || spec.epsilon_r < 1) throw new Error(`Invalid epsilon_r for "${name}"`);
      const material = { ...spec, name, mu_r: spec.mu_r ?? 1 };
      ir.materials.push(material);
      return material;
    },
    box(id, options) {
      requireNewId(id); requireMaterial(options.material);
      ir.entities.push({ id, kind: "box", ...options });
    },
    sheetRect(id, options) {
      requireNewId(id); requireMaterial(options.material);
      ir.entities.push({ id, kind: "sheetRect", normal: options.normal ?? [0, 0, 1], ...options });
    },
    lumpedPort(id, options) {
      requireNewId(id);
      ir.ports.push({ id, kind: "lumpedPort", impedance_ohms: 50, ...options });
    },
    coaxPort(id, options) {
      requireNewId(id);
      ir.ports.push({ id, kind: "coaxPort", impedance_ohms: 50, ...options });
    },
    airbox(options) { ir.boundaries.push({ kind: "radiation", ...options }); },
    pml(options) { ir.boundaries.push({ kind: "pml", layers: 8, ...options }); },
    refine(options) { ir.meshHints.push(options); },
  };
  return { api, ir };
}

export type { Boundary, CoaxPort, GeometryEntity, LumpedPort };
