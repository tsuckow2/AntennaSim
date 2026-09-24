import type { BoxEntity, ModelIR, Vec3 } from "../dsl/model-ir";

/**
 * Parametric 908 MHz microstrip patch models for the volume FEM path.
 *
 * Two builders share one stack-up so a single patch can be tuned cheaply
 * before the full array is meshed:
 *
 * - `singlePatchIR`: one inset-fed patch on a grounded Rogers-type substrate.
 *   Used to tune the resonant length and inset depth for 908 MHz.
 * - `patchArray2x2IR`: 2x2 broadside array of the same patch, driven through
 *   a corporate microstrip feed (50-ohm input -> two 100-ohm branches ->
 *   per-column 100-ohm lines -> inset patch feeds) from a single lumped port.
 *
 * Stack-up (all dimensions in metres):
 * - ground plane: 2 mm copper at z in [-0.002, 0]
 * - substrate: 10 mm, epsilon_r = 2.2, loss tangent 0.001 at z in [0, 0.010]
 * - patches and feed lines: 2 mm copper at z in [0.010, 0.012]
 *
 * Feed line widths are for the 10 mm / er=2.2 stack-up: ~30 mm for 50 ohm,
 * ~8 mm for 100 ohm. The inset notch is 13 mm wide (8 mm line + 2.5 mm gaps).
 */

export const PATCH_DESIGN_FREQ_HZ = 908e6;
export const PATCH_SUBSTRATE_ER = 2.2;
export const PATCH_SUBSTRATE_H_M = 0.01;
/** Copper thickness (m). Kept at 4 mm so thin-slab cells stay well-conditioned
 *  on a 12-15 mm tensor mesh (aspect ratio >= ~0.27, no quality warnings). */
export const PATCH_COPPER_T_M = 0.004;
export const PATCH_GROUND_T_M = 0.004;
export const PATCH_FEED_W100_M = 0.008;
export const PATCH_FEED_W50_M = 0.03;
/** Half-width of the inset notch (m): 8 mm line + 4 mm gaps each side. */
export const PATCH_NOTCH_HALF_W_M = 0.008;

export interface SinglePatchParams {
  /** Resonant dimension along x (m). Theory: ~0.106 for 908 MHz. */
  lengthM?: number;
  /** Patch width along y (m). */
  widthM?: number;
  /** Inset depth of the 100-ohm feed point from the patch edge (m). */
  insetM?: number;
  /** Reference impedance for S11 (ohm); the inset feed targets 100 ohm. */
  referenceOhms?: number;
}

export interface PatchArrayParams {
  lengthM?: number;
  widthM?: number;
  insetM?: number;
  /** Element centre spacing (m); default 0.7 * lambda0 at 908 MHz. */
  spacingM?: number;
  referenceOhms?: number;
}

interface PatchPieces {
  boxes: BoxEntity[];
  /** Point where the feed line meets the patch (inner end of the notch). */
  feedPoint: Vec3;
  /** Centre of the notch opening on the patch edge. */
  edgePoint: Vec3;
}

function copperBox(id: string, cx: number, cy: number, sx: number, sy: number): BoxEntity {
  return {
    id,
    kind: "box",
    center: [cx, cy, PATCH_SUBSTRATE_H_M + PATCH_COPPER_T_M / 2],
    size: [sx, sy, PATCH_COPPER_T_M],
    material: "copper",
  };
}

/**
 * Build the three copper boxes forming an inset-fed patch whose feed enters
 * from the -x edge (single patch) or the +x edge (array, mirrored by the
 * caller via `mirrorX`).
 */
function insetPatch(
  idPrefix: string,
  cx: number,
  cy: number,
  lengthM: number,
  widthM: number,
  insetM: number,
  feedFromNegativeX: boolean,
): PatchPieces {
  const halfL = lengthM / 2;
  const halfW = widthM / 2;
  // Notch: from the feed edge (xEdge) inward by insetM, half-width notchHalf.
  const xEdge = feedFromNegativeX ? cx - halfL : cx + halfL;
  const xInner = feedFromNegativeX ? xEdge + insetM : xEdge - insetM;
  const xA0 = Math.min(xEdge, xInner);
  const xA1 = Math.max(xEdge, xInner);
  const boxes: BoxEntity[] = [];
  // Box A: full-width part of the patch behind the notch.
  const backX0 = feedFromNegativeX ? xInner : cx - halfL;
  const backX1 = feedFromNegativeX ? cx + halfL : xInner;
  boxes.push(
    copperBox(
      `${idPrefix}-a`,
      (backX0 + backX1) / 2,
      cy,
      backX1 - backX0,
      widthM,
    ),
  );
  // Boxes B/C: flank the notch.
  const notchW = xA1 - xA0;
  const flankW = halfW - PATCH_NOTCH_HALF_W_M;
  boxes.push(
    copperBox(`${idPrefix}-b`, (xA0 + xA1) / 2, cy + PATCH_NOTCH_HALF_W_M + flankW / 2, notchW, flankW),
    copperBox(`${idPrefix}-c`, (xA0 + xA1) / 2, cy - PATCH_NOTCH_HALF_W_M - flankW / 2, notchW, flankW),
  );
  return {
    boxes,
    feedPoint: [xInner, cy, PATCH_SUBSTRATE_H_M + PATCH_COPPER_T_M],
    edgePoint: [xEdge, cy, PATCH_SUBSTRATE_H_M + PATCH_COPPER_T_M],
  };
}

/** Single inset-fed patch, feed entering from the -x edge. */
export function singlePatchIR(params: SinglePatchParams = {}): ModelIR {
  const lengthM = params.lengthM ?? 0.106;
  const widthM = params.widthM ?? 0.115;
  const insetM = params.insetM ?? 0.03;
  const referenceOhms = params.referenceOhms ?? 100;

  const patch = insetPatch("patch", 0, 0, lengthM, widthM, insetM, true);
  // 100-ohm feed line from the substrate edge region into the notch, with a
  // 4 mm series gap for the lumped port (a port riding on continuous copper
  // is shorted and reads Z ~ 0).
  const lineX0 = -0.1;
  const lineX1 = patch.feedPoint[0];
  const gapX = -0.07;
  const gapHalf = 0.002;
  const feedBoxes = [
    copperBox("feedline-a", (lineX0 + gapX - gapHalf) / 2, 0, gapX - gapHalf - lineX0, PATCH_FEED_W100_M),
    copperBox("feedline-b", (gapX + gapHalf + lineX1) / 2, 0, lineX1 - gapX - gapHalf, PATCH_FEED_W100_M),
  ];
  // Lumped port across the series gap, terminals inside the gap.
  const portZ = PATCH_SUBSTRATE_H_M + PATCH_COPPER_T_M / 2;
  const port: Vec3 = [gapX + 0.0015, 0, portZ];
  const portNeg: Vec3 = [gapX - 0.0015, 0, portZ];

  return {
    version: 1,
    paramSchemaVersion: 1,
    materials: [
      { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 },
      { name: "rogers", epsilon_r: PATCH_SUBSTRATE_ER, mu_r: 1, loss_tangent: 0.001 },
    ],
    entities: [
      {
        id: "ground",
        kind: "box",
        center: [0, 0, -PATCH_GROUND_T_M / 2],
        size: [0.26, 0.26, PATCH_GROUND_T_M],
        material: "copper",
      },
      {
        id: "substrate",
        kind: "box",
        center: [0, 0, PATCH_SUBSTRATE_H_M / 2],
        size: [0.22, 0.22, PATCH_SUBSTRATE_H_M],
        material: "rogers",
      },
      ...patch.boxes,
      ...feedBoxes,
    ],
    ports: [
      {
        id: "port1",
        kind: "lumpedPort",
        positive: [port[0], port[1], port[2]],
        negative: [portNeg[0], portNeg[1], portNeg[2]],
        impedance_ohms: referenceOhms,
      },
    ],
    boundaries: [{ kind: "radiation", padding_m: 0.06 }],
    meshHints: [],
  };
}

export interface ProbePatchParams {
  lengthM?: number;
  widthM?: number;
  /** Probe position measured from the -x patch edge (m). ~0.039 targets 50 ohm. */
  probeInsetM?: number;
  referenceOhms?: number;
  /** Ground plane size (m). Default 0.26; use larger for better F/B. */
  groundSizeM?: number;
}

/**
 * Probe-fed single patch: solid patch, vertical 4x4 mm probe from 1 mm above
 * the ground plane up to the patch underside. The lumped port is a 1 mm
 * vertical segment in the substrate driving probe-vs-ground (classic coax
 * probe feed). The substrate is built from four boxes around the probe so no
 * volumes overlap.
 */
export function probePatchIR(params: ProbePatchParams = {}): ModelIR {
  const lengthM = params.lengthM ?? 0.106;
  const widthM = params.widthM ?? 0.115;
  const probeInsetM = params.probeInsetM ?? 0.039;
  const referenceOhms = params.referenceOhms ?? 50;
  const groundSizeM = params.groundSizeM ?? 0.26;

  const probeX = -lengthM / 2 + probeInsetM;
  const probeHalf = 0.002;
  const subZ0 = 0;
  const subZ1 = PATCH_SUBSTRATE_H_M;
  const subHalf = 0.11;

  const substrateBoxes: BoxEntity[] = [
    {
      id: "substrate-l",
      kind: "box",
      center: [(-subHalf + probeX - probeHalf) / 2, 0, (subZ0 + subZ1) / 2],
      size: [probeX - probeHalf + subHalf, subHalf * 2, subZ1 - subZ0],
      material: "rogers",
    },
    {
      id: "substrate-r",
      kind: "box",
      center: [(probeX + probeHalf + subHalf) / 2, 0, (subZ0 + subZ1) / 2],
      size: [subHalf - probeX - probeHalf, subHalf * 2, subZ1 - subZ0],
      material: "rogers",
    },
    {
      id: "substrate-f",
      kind: "box",
      center: [probeX, (-subHalf - probeHalf) / 2, (subZ0 + subZ1) / 2],
      size: [probeHalf * 2, subHalf - probeHalf, subZ1 - subZ0],
      material: "rogers",
    },
    {
      id: "substrate-b",
      kind: "box",
      center: [probeX, (probeHalf + subHalf) / 2, (subZ0 + subZ1) / 2],
      size: [probeHalf * 2, subHalf - probeHalf, subZ1 - subZ0],
      material: "rogers",
    },
  ];

  const probeZ0 = 0.001;
  const probeZ1 = PATCH_SUBSTRATE_H_M; // touches patch underside, no overlap
  const probe: BoxEntity = {
    id: "probe",
    kind: "box",
    center: [probeX, 0, (probeZ0 + probeZ1) / 2],
    size: [probeHalf * 2, probeHalf * 2, probeZ1 - probeZ0],
    material: "copper",
  };

  const patch: BoxEntity = {
    id: "patch",
    kind: "box",
    center: [0, 0, PATCH_SUBSTRATE_H_M + PATCH_COPPER_T_M / 2],
    size: [lengthM, widthM, PATCH_COPPER_T_M],
    material: "copper",
  };

  return {
    version: 1,
    paramSchemaVersion: 1,
    materials: [
      { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 },
      { name: "rogers", epsilon_r: PATCH_SUBSTRATE_ER, mu_r: 1, loss_tangent: 0.001 },
    ],
    entities: [
      {
        id: "ground",
        kind: "box",
        center: [0, 0, -PATCH_GROUND_T_M / 2],
        size: [groundSizeM, groundSizeM, PATCH_GROUND_T_M],
        material: "copper",
      },
      ...substrateBoxes,
      patch,
      probe,
    ],
    ports: [
      {
        id: "port1",
        kind: "lumpedPort",
        positive: [probeX, 0, probeZ0],
        negative: [probeX, 0, 0],
        impedance_ohms: referenceOhms,
      },
    ],
    boundaries: [{ kind: "radiation", padding_m: 0.06 }],
    meshHints: [],
  };
}

/** 2x2 broadside patch array with a corporate microstrip feed. */
export function patchArray2x2IR(params: PatchArrayParams = {}): ModelIR {  const lengthM = params.lengthM ?? 0.106;
  const widthM = params.widthM ?? 0.115;
  const insetM = params.insetM ?? 0.03;
  const spacingM = params.spacingM ?? 0.231;
  const referenceOhms = params.referenceOhms ?? 50;

  const entities: BoxEntity[] = [
    {
      id: "ground",
      kind: "box",
      center: [0, -0.01, -PATCH_GROUND_T_M / 2],
      size: [0.48, 0.46, PATCH_GROUND_T_M],
      material: "copper",
    },
    {
      id: "substrate",
      kind: "box",
      center: [0, -0.01, PATCH_SUBSTRATE_H_M / 2],
      size: [0.44, 0.42, PATCH_SUBSTRATE_H_M],
      material: "rogers",
    },
  ];

  const half = spacingM / 2;
  // Column feed lines sit outside the patch columns (100-ohm, 8 mm wide).
  const colX = half + lengthM / 2 + 0.0225;
  const colHalfY = half + PATCH_FEED_W100_M / 2;
  for (const sx of [1, -1]) {
    entities.push(
      copperBox(
        `col-${sx > 0 ? "p" : "n"}`,
        sx * (colX + PATCH_FEED_W100_M / 2),
        0,
        PATCH_FEED_W100_M,
        2 * colHalfY,
      ),
    );
    for (const sy of [1, -1]) {
      const cx = sx * half;
      const cy = sy * half;
      // Array patches are fed from their outer x-edge (away from centre).
      const patch = insetPatch(
        `p${sx > 0 ? "p" : "n"}${sy > 0 ? "p" : "n"}`,
        cx,
        cy,
        lengthM,
        widthM,
        insetM,
        sx < 0,
      );
      entities.push(...patch.boxes);
      // 100-ohm line from the column into the notch (butts into the column).
      const feedX0 = patch.feedPoint[0];
      const feedX1 = sx * colX;
      entities.push(
        copperBox(
          `f${sx > 0 ? "p" : "n"}${sy > 0 ? "p" : "n"}`,
          (feedX0 + feedX1) / 2,
          cy,
          Math.abs(feedX1 - feedX0),
          PATCH_FEED_W100_M,
        ),
      );
    }
  }

  // Main tee: one 100-ohm arm across the middle, butting into both columns.
  // (The 100-ohm -> 50-ohm step at each column tee is left unmatched in v1;
  // it is symmetric, so the pattern is unaffected and only S11 suffers.)
  entities.push(
    copperBox("arm", 0, 0, 2 * colX, PATCH_FEED_W100_M),
  );
  // 50-ohm input line from the board edge to the tee, with a 4 mm series
  // gap for the lumped port.
  const inY0 = -0.2;
  const inY1 = -PATCH_FEED_W100_M / 2;
  const gapY = -0.18;
  const gapHalf = 0.002;
  entities.push(
    copperBox("input-a", 0, (inY0 + gapY - gapHalf) / 2, PATCH_FEED_W50_M, gapY - gapHalf - inY0),
    copperBox("input-b", 0, (gapY + gapHalf + inY1) / 2, PATCH_FEED_W50_M, inY1 - gapY - gapHalf),
  );

  const portZ = PATCH_SUBSTRATE_H_M + PATCH_COPPER_T_M / 2;
  const portPos: Vec3 = [0, gapY + 0.0015, portZ];
  const portNeg: Vec3 = [0, gapY - 0.0015, portZ];

  return {
    version: 1,
    paramSchemaVersion: 1,
    materials: [
      { name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 },
      { name: "rogers", epsilon_r: PATCH_SUBSTRATE_ER, mu_r: 1, loss_tangent: 0.001 },
    ],
    entities,
    ports: [
      {
        id: "port1",
        kind: "lumpedPort",
        positive: portPos,
        negative: portNeg,
        impedance_ohms: referenceOhms,
      },
    ],
    boundaries: [{ kind: "radiation", padding_m: 0.06 }],
    meshHints: [],
  };
}
