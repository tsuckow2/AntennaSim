/**
 * Coax wave-port model generator: calibration fixtures + Snyder patch.
 *
 * Modes (COAX_MODE env):
 *   fixture-matched — coax section only, wave port at the bottom,
 *     absorbing face at the top: a pure traveling wave must result.
 *   fixture-short — coax section capped with copper at z=0:
 *     Zin(z_ref) must equal j·Z0·tan(k_d · |z_ref|).
 *   snyder — the full Snyder patch with the coax feed (no lumped gap):
 *     pin continuous from the port plane to the patch, PTFE dielectric,
 *     outer-conductor frame meeting a holed ground sheet.
 *
 * Writes packed native point requests + manifest.json under
 * ~/workspace/coax_models/<mode>/ (persistent; /tmp is wiped).
 *
 * Usage:
 *   npx esbuild scripts/snyder-coax-gen.ts --bundle --platform=node \
 *     --format=cjs --outfile=/tmp/coax-gen.cjs && COAX_MODE=snyder node /tmp/coax-gen.cjs
 */
import * as fs from "node:fs";
import * as path from "node:path";
import type { ModelIR } from "../src/dsl/model-ir";
import { buildFEMMesh, FEM_ATTR_PORT_NEGATIVE, FEM_ATTR_PORT_POSITIVE } from "../src/engine/mesh";
import { packHcurlPointRequest } from "../src/engine/wasm/hcurl-abi";

const MODE = process.env.COAX_MODE ?? "fixture-matched";
// Coax cross-section (square), meters.
const HALF_PIN = 0.00065; // 1.3 mm square pin (matches the Snyder probe)
const HALF_DIEL = 0.0022; // PTFE outer half-width
const HALF_OUTER = 0.0032; // outer-conductor frame outer half-width
const COAX_L = 0.006; // port plane at z = -COAX_L
const EPS_PTFE = 2.1;
// Snyder geometry (design values, as in snyder-layer-sweep.ts).
const H = 0.001575;
const G = 0.1016;
const L = 0.0282084;
const W = 0.0375789;
const FEED_FROM_EDGE = 0.0075658;
const probeX = -L / 2 + FEED_FROM_EDGE;
const subHalf = G / 2;
const MESH_M = 0.005;
const PADDING_M = 0.03;

type Ent = ModelIR["entities"][number];
type Hint = NonNullable<ModelIR["meshHints"]>[number];

function splitAround(
  prefix: string,
  cx: number,
  innerHalf: number,
  outerHalf: number,
  zCenter: number,
  zSize: number,
  material: string,
): Ent[] {
  // Four boxes tiling [-outerHalf, +outerHalf]^2 around a square hole of
  // half-width innerHalf centered on (cx, 0), all at the given z span.
  const mk = (id: string, center: [number, number, number], size: [number, number, number]): Ent => ({
    id,
    kind: "box",
    center,
    size,
    material,
  });
  const mid = (innerHalf + outerHalf) / 2;
  return [
    mk(`${prefix}_l`, [cx - mid, 0, zCenter], [outerHalf - innerHalf, outerHalf * 2, zSize]),
    mk(`${prefix}_r`, [cx + mid, 0, zCenter], [outerHalf - innerHalf, outerHalf * 2, zSize]),
    mk(`${prefix}_f`, [cx, -mid, zCenter], [innerHalf * 2, outerHalf - innerHalf, zSize]),
    mk(`${prefix}_b`, [cx, mid, zCenter], [innerHalf * 2, outerHalf - innerHalf, zSize]),
  ];
}

function coaxEntities(cx: number, zTop: number, withCap: boolean): Ent[] {
  const ents: Ent[] = [];
  // Pin: one copper box from the port plane to zTop.
  ents.push({
    id: "probe",
    kind: "box",
    center: [cx, 0, (zTop - COAX_L) / 2],
    size: [HALF_PIN * 2, HALF_PIN * 2, zTop + COAX_L],
    material: "copper",
  });
  // PTFE annulus around the pin, z in [-L, 0].
  ents.push(...splitAround("ptfe", cx, HALF_PIN, HALF_DIEL, -COAX_L / 2, COAX_L, "ptfe"));
  // Outer-conductor frame, z in [-L, 0].
  ents.push(...splitAround("frame", cx, HALF_DIEL, HALF_OUTER, -COAX_L / 2, COAX_L, "copper"));
  if (withCap) {
    ents.push({
      id: "cap",
      kind: "box",
      center: [cx, 0, 0.001],
      size: [HALF_OUTER * 2, HALF_OUTER * 2, 0.002],
      material: "copper",
    });
  }
  return ents;
}

function coaxHints(): Hint[] {
  const hints: Hint[] = [];
  const fine = { x: 0.0005, y: 0.0005, z: 0.001 };
  hints.push({ region: "probe", max_size_m: MESH_M, max_size_by_axis_m: fine });
  for (const p of ["ptfe_l", "ptfe_r", "ptfe_f", "ptfe_b"]) {
    hints.push({ region: p, max_size_m: MESH_M, max_size_by_axis_m: fine });
  }
  for (const p of ["frame_l", "frame_r", "frame_f", "frame_b"]) {
    hints.push({ region: p, max_size_m: MESH_M, max_size_by_axis_m: { x: 0.001, y: 0.001, z: 0.0015 } });
  }
  return hints;
}

function excludeBelow(cx: number): number[] {
  return [cx - HALF_OUTER, -HALF_OUTER, -PADDING_M - 0.002, cx + HALF_OUTER, HALF_OUTER, -COAX_L];
}
function excludeAbove(cx: number): number[] {
  return [cx - HALF_OUTER, -HALF_OUTER, 0, cx + HALF_OUTER, HALF_OUTER, PADDING_M + 0.002];
}

function pack(
  ir: ModelIR,
  freqsMHz: number[],
  metadata: Record<string, unknown>,
  outDir: string,
  cx: number,
) {
  const mesh = buildFEMMesh(ir, MESH_M, undefined);
  const tets = mesh.tetrahedra.length / 4;
  console.log(
    `${MODE}: tets=${tets} nodes=${mesh.statistics.nodes} warnings=${mesh.warnings.length}`,
  );
  for (const w of mesh.warnings) console.log(`  warn: ${w}`);
  // Dummy port facets for the packer (the coax path never reads them):
  // positive = pin bottom face facets, negative = frame bottom facets,
  // both in the port plane.
  const pos: number[] = [];
  const neg: number[] = [];
  const nf = mesh.boundaryFacets.length / 3;
  for (let f = 0; f < nf; f++) {
    let cz = 0;
    let cxm = 0;
    let cym = 0;
    for (let k = 0; k < 3; k++) {
      const v = mesh.boundaryFacets[f * 3 + k]!;
      cxm += mesh.vertices[v * 3]! / 3;
      cym += mesh.vertices[v * 3 + 1]! / 3;
      cz += mesh.vertices[v * 3 + 2]! / 3;
    }
    if (Math.abs(cz + COAX_L) > 1e-9) continue;
    const cheb = Math.max(Math.abs(cxm - cx), Math.abs(cym));
    if (cheb <= HALF_PIN + 1e-9) pos.push(f);
    else if (cheb >= HALF_DIEL - 1e-9 && cheb <= HALF_OUTER + 1e-9) neg.push(f);
  }
  console.log(`dummy port facets: ${pos.length}+${neg.length}`);
  if (pos.length === 0 || neg.length === 0) throw new Error("dummy port facet selection failed");
  const portAttr = new Uint32Array(pos.length + neg.length);
  portAttr.fill(FEM_ATTR_PORT_POSITIVE, 0, pos.length);
  portAttr.fill(FEM_ATTR_PORT_NEGATIVE, pos.length);
  fs.mkdirSync(outDir, { recursive: true });
  for (const mhz of freqsMHz) {
    const req = packHcurlPointRequest({
      frequencyHz: mhz * 1e6,
      vertices: mesh.vertices,
      tets: mesh.tetrahedra,
      tetRegion: mesh.tetRegions,
      boundaryFacets: mesh.boundaryFacets,
      boundaryAttr: mesh.boundaryAttributes,
      portPosFacets: new Uint32Array(pos),
      portNegFacets: new Uint32Array(neg),
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
        ...metadata,
      }),
    });
    fs.writeFileSync(path.join(outDir, `${mhz}.req`), req);
  }
  fs.writeFileSync(
    path.join(outDir, "manifest.json"),
    JSON.stringify({ mode: MODE, tets, nodes: mesh.statistics.nodes, metadata }, null, 2),
  );
  console.log("wrote", outDir);
}

const outBase = `${process.env.HOME}/workspace/coax_models`;

if (MODE === "fixture-matched" || MODE === "fixture-short") {
  const short = MODE === "fixture-short";
  const ir: ModelIR = {
    version: 1,
    paramSchemaVersion: 1,
    materials: [
      { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 },
      { name: "ptfe", epsilon_r: EPS_PTFE, mu_r: 1 },
    ],
    entities: coaxEntities(0, 0, short),
    ports: [],
    boundaries: [{ kind: "radiation", padding_m: PADDING_M }],
    meshHints: coaxHints(),
  };
  const metadata: Record<string, unknown> = {
    coax_port: {
      center: [0, 0],
      z_port: -COAX_L,
      z_ref: -0.00025,
      half_pin: HALF_PIN,
      half_diel: HALF_DIEL,
      eps_diel: EPS_PTFE,
      ...(short ? {} : { z_port_top: 0 }),
    },
    exclude_boxes: short
      ? [excludeBelow(0)]
      : [excludeBelow(0), excludeAbove(0)],
  };
  pack(ir, [2440], metadata, path.join(outBase, MODE), 0);
} else if (MODE === "snyder") {
  const ER = parseFloat(process.env.SNYDER_ER ?? "4.4");
  const XY_MM = parseFloat(process.env.SNYDER_XY_HINT_MM ?? "2.5");
  const LAYERS = parseInt(process.env.SNYDER_LAYERS ?? "1", 10);
  const zSize = H / LAYERS;
  const subIds = ["substrate_l", "substrate_r", "substrate_f", "substrate_b"];
  const ents: Ent[] = [
    {
      id: "substrate_l",
      kind: "box",
      center: [(-subHalf + probeX - HALF_PIN) / 2, 0, H / 2],
      size: [probeX - HALF_PIN + subHalf, G, H],
      material: "fr4",
    },
    {
      id: "substrate_r",
      kind: "box",
      center: [(probeX + HALF_PIN + subHalf) / 2, 0, H / 2],
      size: [subHalf - probeX - HALF_PIN, G, H],
      material: "fr4",
    },
    {
      id: "substrate_f",
      kind: "box",
      center: [probeX, (-subHalf - HALF_PIN) / 2, H / 2],
      size: [HALF_PIN * 2, subHalf - HALF_PIN, H],
      material: "fr4",
    },
    {
      id: "substrate_b",
      kind: "box",
      center: [probeX, (HALF_PIN + subHalf) / 2, H / 2],
      size: [HALF_PIN * 2, subHalf - HALF_PIN, H],
      material: "fr4",
    },
    ...coaxEntities(probeX, H, false),
    // Ground: four sheets around a square hole (half HALF_OUTER) at the feed.
    {
      id: "ground_l",
      kind: "sheetRect",
      center: [(probeX - HALF_OUTER - subHalf) / 2 + 0, 0, 0],
      size: [subHalf + probeX - HALF_OUTER, G],
      normal: [0, 0, 1],
      material: "copper",
    },
    {
      id: "ground_r",
      kind: "sheetRect",
      center: [(probeX + HALF_OUTER + subHalf) / 2, 0, 0],
      size: [subHalf - probeX - HALF_OUTER, G],
      normal: [0, 0, 1],
      material: "copper",
    },
    {
      id: "ground_f",
      kind: "sheetRect",
      center: [probeX, (-subHalf - HALF_OUTER) / 2, 0],
      size: [HALF_OUTER * 2, subHalf - HALF_OUTER],
      normal: [0, 0, 1],
      material: "copper",
    },
    {
      id: "ground_b",
      kind: "sheetRect",
      center: [probeX, (HALF_OUTER + subHalf) / 2, 0],
      size: [HALF_OUTER * 2, subHalf - HALF_OUTER],
      normal: [0, 0, 1],
      material: "copper",
    },
    {
      id: "patch",
      kind: "sheetRect",
      center: [0, 0, H],
      size: [L, W],
      normal: [0, 0, 1],
      material: "copper",
    },
  ];
  const ir: ModelIR = {
    version: 1,
    paramSchemaVersion: 1,
    materials: [
      { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 },
      { name: "fr4", epsilon_r: ER, mu_r: 1, loss_tangent: 0.02 },
      { name: "ptfe", epsilon_r: EPS_PTFE, mu_r: 1 },
    ],
    entities: ents,
    ports: [],
    boundaries: [{ kind: "radiation", padding_m: PADDING_M }],
    meshHints: [
      ...subIds.map((region) => ({
        region,
        max_size_m: MESH_M,
        max_size_by_axis_m: { x: XY_MM / 1000, y: XY_MM / 1000, z: zSize },
      })),
      ...coaxHints(),
    ],
  };
  const metadata: Record<string, unknown> = {
    coax_port: {
      center: [probeX, 0],
      z_port: -COAX_L,
      z_ref: -0.00025,
      half_pin: HALF_PIN,
      half_diel: HALF_DIEL,
      eps_diel: EPS_PTFE,
    },
    exclude_boxes: [excludeBelow(probeX)],
  };
  const freqs = (process.env.COAX_FREQS ?? "2300,2400,2447,2500").split(",").map((s) => parseInt(s, 10));
  pack(ir, freqs, metadata, path.join(outBase, `snyder_xy${XY_MM}_L${LAYERS}`), probeX);
} else {
  throw new Error(`unknown COAX_MODE ${MODE}`);
}
