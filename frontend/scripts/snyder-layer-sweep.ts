/**
 * Snyder 2.44 GHz probe-fed patch — substrate layer convergence sweep.
 *
 * Builds the patch with zero-thickness PEC sheets (ground + patch), a small
 * copper probe volume, and the FR-4 substrate split into boxes around the
 * probe. The substrate gets a z-only mesh hint so the 1.575 mm slab is cut
 * into `layers` element layers through its thickness while x/y stay at the
 * 5 mm nominal size.
 *
 * Usage:
 *   npx esbuild scripts/snyder-layer-sweep.ts --bundle --platform=node \
 *     --format=cjs --outfile=/tmp/snyder-gen.cjs && node /tmp/snyder-gen.cjs
 *
 * Writes packed native point requests to /tmp/snyder_sweep/L<layers>/<mhz>.req
 * plus a manifest.json with mesh statistics per layer count.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import type { ModelIR } from "../src/dsl/model-ir";
import {
  buildFEMMesh,
  FEM_ATTR_PORT_NEGATIVE,
  FEM_ATTR_PORT_POSITIVE,
} from "../src/engine/mesh";
import { packHcurlPointRequest } from "../src/engine/wasm/hcurl-abi";

// Snyder thesis, Cal Poly 2010 (HFSS-optimized, design values).
const H = 0.001575; // substrate thickness (m)
const G = 0.1016; // ground/substrate size (m)
const L = 0.0282084; // patch resonant length, x (m)
const W = 0.0375789; // patch width, y (m)
const FEED_FROM_EDGE = 0.0075658; // probe inset from -x patch edge (m)
const PROBE_HALF = 0.00065; // ~1.3 mm square SMA pin
const GAP = 0.0005; // lumped-port gap between probe bottom and ground
const MESH_M = 0.005;
const PADDING_M = 0.03;

const probeX = -L / 2 + FEED_FROM_EDGE;
const subHalf = G / 2;

function snyderIR(layers: number): ModelIR {
  const zSize = H / layers;
  const zHint = (region: string) => ({
    region,
    max_size_m: MESH_M,
    max_size_by_axis_m: { z: zSize },
  });
  const subIds = ["substrate_l", "substrate_r", "substrate_f", "substrate_b"];
  return {
    version: 1,
    paramSchemaVersion: 1,
    materials: [
      { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 },
      { name: "fr4", epsilon_r: 4.4, mu_r: 1, loss_tangent: 0.02 },
    ],
    entities: [
      // Substrate split into boxes around the probe footprint (no overlaps).
      {
        id: "substrate_l",
        kind: "box",
        center: [(-subHalf + probeX - PROBE_HALF) / 2, 0, H / 2],
        size: [probeX - PROBE_HALF + subHalf, G, H],
        material: "fr4",
      },
      {
        id: "substrate_r",
        kind: "box",
        center: [(probeX + PROBE_HALF + subHalf) / 2, 0, H / 2],
        size: [subHalf - probeX - PROBE_HALF, G, H],
        material: "fr4",
      },
      {
        id: "substrate_f",
        kind: "box",
        center: [probeX, (-subHalf - PROBE_HALF) / 2, H / 2],
        size: [PROBE_HALF * 2, subHalf - PROBE_HALF, H],
        material: "fr4",
      },
      {
        id: "substrate_b",
        kind: "box",
        center: [probeX, (PROBE_HALF + subHalf) / 2, H / 2],
        size: [PROBE_HALF * 2, subHalf - PROBE_HALF, H],
        material: "fr4",
      },
      // Substrate column under the probe is left as air: the lumped port
      // spans this feed gap (probe bottom -> ground), mirroring the
      // 908 MHz example, so the probe bottom is a conductor-air interface.
      // Coax-style probe pin (real 3D feature, kept as a volume).
      {
        id: "probe",
        kind: "box",
        center: [probeX, 0, (GAP + H) / 2],
        size: [PROBE_HALF * 2, PROBE_HALF * 2, H - GAP],
        material: "copper",
      },
      // Zero-thickness PEC sheets for ground and patch.
      {
        id: "ground",
        kind: "sheetRect",
        center: [0, 0, 0],
        size: [G, G],
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
    ],
    ports: [
      {
        id: "feed",
        kind: "lumpedPort",
        positive: [probeX, 0, GAP],
        negative: [probeX, 0, 0],
        impedance_ohms: 50,
      },
    ],
    boundaries: [{ kind: "radiation", padding_m: PADDING_M }],
    meshHints: subIds.map(zHint),
  };
}

const LAYERS = [1, 2, 3, 4, 6];
const FREQS_MHZ = [2300, 2400, 2500, 2600, 2700, 2800];
// Persistent workspace dir: /tmp is wiped on VM replacement.
// Override with SNYDER_SWEEP_OUT.
const OUT = process.env.SNYDER_SWEEP_OUT ?? `${process.env.HOME}/workspace/snyder_sweep`;

const manifest: Record<string, unknown> = {};
for (const layers of LAYERS) {
  const ir = snyderIR(layers);
  const mesh = buildFEMMesh(ir, MESH_M, ir.ports[0]);
  const tets = mesh.tetrahedra.length / 4;
  console.log(
    `layers=${layers} tets=${tets} nodes=${mesh.statistics.nodes} ` +
      `portFacets=${mesh.portPositiveFacets.length}+${mesh.portNegativeFacets.length} ` +
      `warnings=${mesh.warnings.length}`,
  );
  for (const w of mesh.warnings) console.log(`  warn: ${w}`);
  manifest[`L${layers}`] = { tets, nodes: mesh.statistics.nodes };

  const portAttr = new Uint32Array(
    mesh.portPositiveFacets.length + mesh.portNegativeFacets.length,
  );
  portAttr.fill(FEM_ATTR_PORT_POSITIVE, 0, mesh.portPositiveFacets.length);
  portAttr.fill(FEM_ATTR_PORT_NEGATIVE, mesh.portPositiveFacets.length);

  const dir = path.join(OUT, `L${layers}`);
  fs.mkdirSync(dir, { recursive: true });
  for (const mhz of FREQS_MHZ) {
    const req = packHcurlPointRequest({
      frequencyHz: mhz * 1e6,
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
    });
    fs.writeFileSync(path.join(dir, `${mhz}.req`), req);
  }
}
fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log("wrote", OUT);
