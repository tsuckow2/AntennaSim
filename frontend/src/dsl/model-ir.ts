import type { ParamSchema } from "../utils/param-schema";

export const MODEL_IR_VERSION = 1 as const;

export type Vec3 = [number, number, number];

export interface Material {
  name: string;
  epsilon_r: number;
  mu_r: number;
  conductivity_s_m?: number;
  loss_tangent?: number;
}

export interface BoxEntity {
  id: string;
  kind: "box";
  center: Vec3;
  size: Vec3;
  material: string;
}

export interface SheetRectEntity {
  id: string;
  kind: "sheetRect";
  center: Vec3;
  size: [number, number];
  normal: Vec3;
  material: string;
}

export type GeometryEntity = BoxEntity | SheetRectEntity;

export interface LumpedPort {
  id: string;
  kind: "lumpedPort";
  positive: Vec3;
  negative: Vec3;
  impedance_ohms: number;
}

export interface Boundary {
  kind: "radiation" | "pml";
  padding_m: number;
  layers?: number;
}

export interface MeshHint {
  region: string;
  max_size_m: number;
  /** Optional per-axis overrides; axes not listed fall back to max_size_m. */
  max_size_by_axis_m?: { x?: number; y?: number; z?: number };
}

export interface ModelIR {
  version: typeof MODEL_IR_VERSION;
  paramSchemaVersion: ParamSchema["version"];
  materials: Material[];
  entities: GeometryEntity[];
  ports: LumpedPort[];
  boundaries: Boundary[];
  meshHints: MeshHint[];
}

export interface IRDiagnostic {
  path: string;
  message: string;
  severity: "error" | "warning";
}

const finite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);
const vector = (v: unknown, size: number): v is number[] =>
  Array.isArray(v) && v.length === size && v.every(finite);

function diagnostic(path: string, message: string): IRDiagnostic {
  return { path, message, severity: "error" };
}

export function validateModelIR(ir: unknown): IRDiagnostic[] {
  const errors: IRDiagnostic[] = [];
  if (!ir || typeof ir !== "object") return [diagnostic("", "IR must be an object")];
  const value = ir as Record<string, unknown>;
  if (value.version !== MODEL_IR_VERSION) errors.push(diagnostic("version", `must be ${MODEL_IR_VERSION}`));
  for (const key of ["materials", "entities", "ports", "boundaries", "meshHints"]) {
    if (!Array.isArray(value[key])) errors.push(diagnostic(key, "must be an array"));
  }
  if (errors.length) return errors;

  const materials = value.materials as Material[];
  const materialNames = new Set<string>();
  materials.forEach((material, i) => {
    if (!material || typeof material !== "object") {
      errors.push(diagnostic(`materials[${i}]`, "must be an object"));
      return;
    }
    if (!material.name || materialNames.has(material.name)) {
      errors.push(diagnostic(`materials[${i}].name`, "must be unique and non-empty"));
    }
    materialNames.add(material.name);
    if (!finite(material.epsilon_r) || material.epsilon_r < 1) {
      errors.push(diagnostic(`materials[${i}].epsilon_r`, "must be finite and >= 1"));
    }
    if (!finite(material.mu_r) || material.mu_r <= 0) {
      errors.push(diagnostic(`materials[${i}].mu_r`, "must be finite and > 0"));
    }
  });

  const ids = new Set<string>();
  const entities = value.entities as GeometryEntity[];
  entities.forEach((entity, i) => {
    const path = `entities[${i}]`;
    if (!entity || typeof entity !== "object" || !entity.id || ids.has(entity.id)) {
      errors.push(diagnostic(`${path}.id`, "must be unique and non-empty"));
      return;
    }
    ids.add(entity.id);
    if (!materialNames.has(entity.material)) errors.push(diagnostic(`${path}.material`, "references an unknown material"));
    if (entity.kind === "box") {
      if (!vector(entity.center, 3)) errors.push(diagnostic(`${path}.center`, "must be a finite 3-vector"));
      if (!vector(entity.size, 3) || entity.size.some((n) => n <= 0)) errors.push(diagnostic(`${path}.size`, "must contain three positive values"));
    } else if (entity.kind === "sheetRect") {
      if (!vector(entity.center, 3) || !vector(entity.normal, 3)) errors.push(diagnostic(path, "center and normal must be finite 3-vectors"));
      if (!vector(entity.size, 2) || entity.size.some((n) => n <= 0)) errors.push(diagnostic(`${path}.size`, "must contain two positive values"));
    } else {
      errors.push(diagnostic(`${path}.kind`, "unsupported geometry kind"));
    }
  });

  const ports = value.ports as LumpedPort[];
  ports.forEach((port, i) => {
    if (!port.id || ids.has(port.id)) errors.push(diagnostic(`ports[${i}].id`, "must be unique and non-empty"));
    ids.add(port.id);
    if (!vector(port.positive, 3) || !vector(port.negative, 3)) errors.push(diagnostic(`ports[${i}]`, "terminals must be finite 3-vectors"));
    if (!finite(port.impedance_ohms) || port.impedance_ohms <= 0) errors.push(diagnostic(`ports[${i}].impedance_ohms`, "must be > 0"));
  });

  (value.meshHints as MeshHint[]).forEach((hint, i) => {
    if (!hint.region || !ids.has(hint.region) && hint.region !== "airbox") errors.push(diagnostic(`meshHints[${i}].region`, "references an unknown region"));
    if (!finite(hint.max_size_m) || hint.max_size_m <= 0) errors.push(diagnostic(`meshHints[${i}].max_size_m`, "must be > 0"));
    const byAxis = hint.max_size_by_axis_m;
    if (byAxis !== undefined) {
      for (const axis of ["x", "y", "z"] as const) {
        const v = byAxis[axis];
        if (v !== undefined && (!finite(v) || v <= 0)) errors.push(diagnostic(`meshHints[${i}].max_size_by_axis_m.${axis}`, "must be > 0"));
      }
    }
  });
  (value.boundaries as Boundary[]).forEach((boundary, i) => {
    if (boundary.kind !== "radiation" && boundary.kind !== "pml") errors.push(diagnostic(`boundaries[${i}].kind`, "must be radiation or pml"));
    if (!finite(boundary.padding_m) || boundary.padding_m <= 0) errors.push(diagnostic(`boundaries[${i}].padding_m`, "must be > 0"));
    if (boundary.kind === "pml" && (!Number.isInteger(boundary.layers) || boundary.layers! < 1)) errors.push(diagnostic(`boundaries[${i}].layers`, "must be a positive integer for PML"));
  });
  return errors;
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.keys(value as object).sort().map((key) => [key, sortKeys((value as Record<string, unknown>)[key])]));
}

export function serializeModelIR(ir: ModelIR): string {
  const errors = validateModelIR(ir);
  if (errors.some((error) => error.severity === "error")) throw new Error(`Invalid ModelIR: ${errors[0]!.path} ${errors[0]!.message}`);
  return JSON.stringify(sortKeys(ir));
}
