import type { BoxEntity, ModelIR, SheetRectEntity, Vec3 } from "./model-ir";
import { resolveCoaxPort, validateModelIR } from "./model-ir";

export interface GmshDiagnostic {
  path: string;
  message: string;
  severity: "error" | "warning";
}

export interface GmshExport {
  geo: string;
  diagnostics: GmshDiagnostic[];
}

const diagnostic = (path: string, message: string, severity: GmshDiagnostic["severity"] = "error"): GmshDiagnostic => ({
  path,
  message,
  severity,
});

const format = (value: number): string => Number(value.toFixed(12)).toString();
const point = (value: Vec3): string => value.map(format).join(", ");
const quote = (value: string): string => JSON.stringify(value);

function boxBounds(entity: BoxEntity): [Vec3, Vec3] {
  const half = entity.size.map((value) => value / 2) as Vec3;
  return [
    entity.center.map((value, index) => value - half[index]!) as Vec3,
    entity.center.map((value, index) => value + half[index]!) as Vec3,
  ];
}

function sheetRectCommand(entity: SheetRectEntity, tag: number): string | undefined {
  const [width, height] = entity.size;
  const [nx, ny, nz] = entity.normal;
  if (nx === 0 && ny === 0 && Math.abs(nz) === 1) {
    return `Rectangle(${tag}) = {${format(entity.center[0] - width / 2)}, ${format(entity.center[1] - height / 2)}, ${format(entity.center[2])}, ${format(width)}, ${format(height)}};`;
  }
  return undefined;
}

function airboxBounds(ir: ModelIR): [Vec3, Vec3] | undefined {
  const boxes = ir.entities.filter((entity): entity is BoxEntity => entity.kind === "box");
  if (boxes.length === 0 || ir.boundaries.length === 0) return undefined;
  const padding = Math.max(...ir.boundaries.map((boundary) => boundary.padding_m));
  const bounds = boxes.map(boxBounds);
  const min: Vec3 = [
    Math.min(...bounds.map(([lower]) => lower[0])) - padding,
    Math.min(...bounds.map(([lower]) => lower[1])) - padding,
    Math.min(...bounds.map(([lower]) => lower[2])) - padding,
  ];
  const max: Vec3 = [
    Math.max(...bounds.map(([, upper]) => upper[0])) + padding,
    Math.max(...bounds.map(([, upper]) => upper[1])) + padding,
    Math.max(...bounds.map(([, upper]) => upper[2])) + padding,
  ];
  return [min, max];
}

export function exportModelIRToGmsh(ir: ModelIR): GmshExport {
  const diagnostics: GmshDiagnostic[] = validateModelIR(ir).map((error) => ({ ...error }));
  if (diagnostics.some((error) => error.severity === "error")) return { geo: "", diagnostics };

  const lines = [
    "SetFactory(\"OpenCASCADE\");",
    "Mesh.CharacteristicLengthExtendFromBoundary = 0;",
  ];
  const volumeGroups = new Map<string, number[]>();
  const surfaceGroups = new Map<string, number[]>();
  let tag = 1;

  ir.entities.forEach((entity, index) => {
    if (entity.kind === "box") {
      const [lower] = boxBounds(entity);
      lines.push(`Box(${tag}) = {${point(lower)}, ${entity.size.map(format).join(", ")}};`);
      volumeGroups.set(entity.material, [...(volumeGroups.get(entity.material) ?? []), tag]);
      tag += 1;
      return;
    }

    const command = sheetRectCommand(entity, tag);
    if (!command) {
      diagnostics.push(diagnostic(`entities[${index}].normal`, "only sheetRect normals aligned with the z axis are supported by the Gmsh exporter"));
      return;
    }
    lines.push(command);
    surfaceGroups.set(entity.material, [...(surfaceGroups.get(entity.material) ?? []), tag]);
    tag += 1;
  });

  const airbox = airboxBounds(ir);
  if (airbox) {
    const [lower, upper] = airbox;
    const size = upper.map((value, index) => value - lower[index]!) as Vec3;
    lines.push(`Box(${tag}) = {${point(lower)}, ${size.map(format).join(", ")}};`);
    lines.push(`Physical Volume(\"airbox\") = {${tag}};`);
    tag += 1;
  } else if (ir.boundaries.length > 0) {
    diagnostics.push(diagnostic("boundaries", "an airbox requires at least one box entity", "warning"));
  }

  volumeGroups.forEach((entities, material) => {
    lines.push(`Physical Volume(${quote(material)}) = {${entities.join(", ")}};`);
  });
  surfaceGroups.forEach((entities, material) => {
    lines.push(`Physical Surface(${quote(material)}) = {${entities.join(", ")}};`);
  });

  ir.ports.forEach((port, index) => {
    // A coax port exports its nominal terminals: pin top / port plane.
    let positive: Vec3;
    let negative: Vec3;
    if (port.kind === "coaxPort") {
      const spec = resolveCoaxPort(port);
      positive = [spec.center[0], spec.center[1], spec.pinTopZ];
      negative = [spec.center[0], spec.center[1], spec.portZ];
    } else {
      positive = port.positive;
      negative = port.negative;
    }
    const positiveTag = tag++;
    const negativeTag = tag++;
    lines.push(`Point(${positiveTag}) = {${point(positive)}, 0.001};`);
    lines.push(`Point(${negativeTag}) = {${point(negative)}, 0.001};`);
    lines.push(`Physical Point(${quote(`${port.id}_positive`)}) = {${positiveTag}};`);
    lines.push(`Physical Point(${quote(`${port.id}_negative`)}) = {${negativeTag}};`);
    if (positive.every((value, coordinate) => value === negative[coordinate])) {
      diagnostics.push(diagnostic(`ports[${index}]`, "positive and negative terminals must be distinct"));
    }
  });

  ir.meshHints.forEach((hint) => {
    lines.push(`Mesh.CharacteristicLengthMax = ${format(hint.max_size_m)};`);
  });
  lines.push("Mesh 3;");
  return { geo: `${lines.join("\n")}\n`, diagnostics };
}