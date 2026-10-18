import { describe, expect, it } from "vitest";
import type { ModelIR } from "../../dsl/model-ir";
import { buildFEMMesh, type FEMMeshArtifact } from "../mesh";
import { buildEdgeSystem, solveFEMPoint, type ExcludeBox } from "../fem-solver";
import { findPortFaces, type CoaxPortSpec } from "../fem-coax";

/**
 * Coax wave-port parity tests for the TypeScript reference solver
 * (fem-coax.ts + fem-ams.ts, mirrors of the Rust `coax.rs` / `ams.rs`).
 *
 * The fixtures are the calibration models of
 * `frontend/scripts/snyder-coax-gen.ts` (modes fixture-matched /
 * fixture-short), rebuilt here through the same `buildFEMMesh` call, so
 * the meshes are the ones the native solver was calibrated on:
 *   matched: Zin = 44.32 - j1.49 ohm (the square-coax Z0 ~ 44.3 ohm),
 *            42 AMS iterations, 448 port faces (224 drive + 224 top);
 *   short:   Zin(z_ref) = j Z0 tan(k_d * d) — +j0.70 ohm at d = 0.25 mm
 *            and +j19.37 ohm at d = 5.5 mm from the cap.
 */

const HALF_PIN = 0.00065; // 1.3 mm square pin
const HALF_DIEL = 0.0022; // PTFE outer half-width
const HALF_OUTER = 0.0032; // outer-conductor frame outer half-width
const COAX_L = 0.006; // port plane at z = -COAX_L
const EPS_PTFE = 2.1;
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
  ents.push({
    id: "probe",
    kind: "box",
    center: [cx, 0, (zTop - COAX_L) / 2],
    size: [HALF_PIN * 2, HALF_PIN * 2, zTop + COAX_L],
    material: "copper",
  });
  ents.push(...splitAround("ptfe", cx, HALF_PIN, HALF_DIEL, -COAX_L / 2, COAX_L, "ptfe"));
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

function fixtureIR(short: boolean): ModelIR {
  return {
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
}

const EXCLUDE_BELOW: ExcludeBox = [
  -HALF_OUTER, -HALF_OUTER, -PADDING_M - 0.002, HALF_OUTER, HALF_OUTER, -COAX_L,
];
const EXCLUDE_ABOVE: ExcludeBox = [
  -HALF_OUTER, -HALF_OUTER, 0, HALF_OUTER, HALF_OUTER, PADDING_M + 0.002,
];

const MATCHED_SPEC: CoaxPortSpec = {
  center: [0, 0],
  zPort: -COAX_L,
  zRef: -0.00025,
  halfPin: HALF_PIN,
  halfDiel: HALF_DIEL,
  epsDiel: EPS_PTFE,
  zPortTop: 0,
};

let matchedMeshCache: FEMMeshArtifact | undefined;
function matchedMesh(): FEMMeshArtifact {
  matchedMeshCache ??= buildFEMMesh(fixtureIR(false), MESH_M);
  return matchedMeshCache;
}

let shortMeshCache: FEMMeshArtifact | undefined;
function shortMesh(): FEMMeshArtifact {
  shortMeshCache ??= buildFEMMesh(fixtureIR(true), MESH_M);
  return shortMeshCache;
}

describe("FEM coax wave port (TypeScript mirror)", () => {
  it("exclusion boxes make the port planes true boundaries and the faces findable", () => {
    const mesh = matchedMesh();
    const boxes = [EXCLUDE_BELOW, EXCLUDE_ABOVE];
    const sys = buildEdgeSystem(mesh, boxes);
    // Native-solver counts for this exact mesh (fixture-verify log).
    expect(sys.activeTetCount).toBe(54294);
    expect(sys.freeDofCount).toBe(68111);
    const faces = findPortFaces(mesh, sys, MATCHED_SPEC);
    expect(faces).toHaveLength(448);
    expect(faces.filter((f) => !f.top)).toHaveLength(224);
    expect(faces.filter((f) => f.top)).toHaveLength(224);
    // Without the exclusion boxes the port planes are interior (the air
    // stub behind them is active), so no port faces exist and more tets
    // are active.
    const sysAll = buildEdgeSystem(mesh);
    expect(sysAll.activeTetCount).toBeGreaterThan(sys.activeTetCount);
    expect(findPortFaces(mesh, sysAll, MATCHED_SPEC)).toHaveLength(0);
  }, 120000);

  it("matched fixture: Zin is the square-coax Z0, constant and ~44.32 - j1.49 ohm", () => {
    const result = solveFEMPoint({
      mesh: matchedMesh(),
      frequencyHz: 2440e6,
      referenceImpedanceOhms: 50,
      portPositive: [0, 0, 0],
      portNegative: [0, 0, -COAX_L],
      coaxPort: MATCHED_SPEC,
      excludeBoxes: [EXCLUDE_BELOW, EXCLUDE_ABOVE],
    });
    expect(result.status).toBe("ok");
    expect(result.converged).toBe(true);
    // AMS parity: the native solver converges this fixture in 42 iterations.
    expect(result.iterationCount ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(100);
    const z = result.impedanceOhms;
    expect(z).not.toBeNull();
    expect(Math.abs((z?.re ?? 0) - 44.32)).toBeLessThan(0.5);
    expect(Math.abs((z?.im ?? 0) + 1.49)).toBeLessThan(0.5);
  }, 300000);

  it("short fixture: Zin follows j Z0 tan(k_d d) at two extraction planes", () => {
    const mesh = shortMesh();
    const base: CoaxPortSpec = {
      center: [0, 0],
      zPort: -COAX_L,
      zRef: -0.00025,
      halfPin: HALF_PIN,
      halfDiel: HALF_DIEL,
      epsDiel: EPS_PTFE,
    };
    // d = distance from the extraction plane to the copper cap at z = 0.
    // Native calibration: +j0.70 ohm at d = 0.25 mm, +j19.37 at d = 5.5 mm.
    for (const [zRef, expectedIm] of [[-0.00025, 0.7], [-0.0055, 19.37]] as const) {
      const result = solveFEMPoint({
        mesh,
        frequencyHz: 2440e6,
        referenceImpedanceOhms: 50,
        portPositive: [0, 0, 0],
        portNegative: [0, 0, -COAX_L],
        coaxPort: { ...base, zRef },
        excludeBoxes: [EXCLUDE_BELOW],
      });
      expect(result.status).toBe("ok");
      const z = result.impedanceOhms;
      expect(z).not.toBeNull();
      expect(Math.abs(z?.re ?? 99)).toBeLessThan(0.5);
      expect(Math.abs((z?.im ?? 0) - expectedIm)).toBeLessThan(0.75);
    }
  }, 300000);
});
