import { describe, expect, it } from "vitest";
import { buildFEMMesh } from "../mesh";
import { buildEdgeSystem } from "../fem-solver";
import { findPortFaces } from "../fem-coax";
import {
  buildFEMCoaxRunConfig,
  selectFEMPort,
  type FEMSimulationRequest,
} from "../fem";

/**
 * App-side coax feed, end to end: the snyder_coax.js DSL example
 * authors the Snyder patch with a coaxPort (substrate and ground
 * whole); the mesher must inject the feed geometry and reproduce the
 * validated generator mesh (scripts/snyder-coax-gen.ts, mode snyder,
 * xy2.5 L1: 499,200 tets), and fem.ts must derive the exact run
 * configuration the generator packs (coax_port metadata + the
 * exclude box behind the port plane).
 */
describe("coax feed app path (DSL -> mesh -> run config)", () => {
  it("builds the Snyder coax model from the example script", async () => {
    const { readFileSync } = await import("node:fs");
    const { executeParametricScript } = await import("../../dsl/executor");
    const script = readFileSync(
      new URL("../../dsl/examples/snyder_coax.js", import.meta.url),
      "utf8",
    );
    const exec = await executeParametricScript(script);
    expect(exec.diagnostics).toHaveLength(0);
    const ir = exec.ir!;
    const probeX = -0.0282084 / 2 + 0.0075658;

    const request: FEMSimulationRequest = {
      model_ir: ir,
      start_mhz: 1900,
      stop_mhz: 2500,
      steps: 13,
      mesh_size_m: 0.005,
      reference_impedance_ohms: 50,
    };

    // Nominal terminals: pin top / port plane.
    const port = selectFEMPort(request);
    expect(port.id).toBe("feed");
    expect(port.positive).toEqual([probeX, 0, 0.001575]);
    expect(port.negative).toEqual([probeX, 0, -0.006]);

    // Run configuration: the generator's metadata, verbatim values.
    const config = buildFEMCoaxRunConfig(request);
    expect(config).not.toBeNull();
    expect(config!.metadata.coax_port).toEqual({
      center: [probeX, 0],
      z_port: -0.006,
      z_ref: -0.00025,
      half_pin: 0.00065,
      half_diel: 0.0022,
      eps_diel: 2.1,
    });
    expect(config!.metadata.exclude_boxes).toEqual([
      [probeX - 0.0032, -0.0032, -0.032, probeX + 0.0032, 0.0032, -0.006],
    ]);
    expect(config!.solver.coaxPort.zPort).toBe(-0.006);
    expect(config!.solver.excludeBoxes).toEqual(config!.metadata.exclude_boxes);

    // Mesh: identical to the hand-authored generator mesh.
    const mesh = buildFEMMesh(ir, 0.005, port);
    expect(mesh.tetrahedra.length / 4).toBe(499200);
    expect(mesh.portPositiveFacets.length).toBeGreaterThan(0);
    expect(mesh.portNegativeFacets.length).toBeGreaterThan(0);

    // The port faces the solver will drive: 224 at the port plane.
    const system = buildEdgeSystem(mesh, config!.solver.excludeBoxes);
    const faces = findPortFaces(mesh, system, config!.solver.coaxPort);
    expect(faces).toHaveLength(224);
    expect(faces.every((f) => !f.top)).toBe(true);
  }, 300000);

  it("returns no coax config for a lumped-port request", async () => {
    const { readFileSync } = await import("node:fs");
    const { executeParametricScript } = await import("../../dsl/executor");
    const script = readFileSync(
      new URL("../../dsl/examples/patch_908_directional.js", import.meta.url),
      "utf8",
    );
    const exec = await executeParametricScript(script);
    expect(exec.diagnostics).toHaveLength(0);
    const request: FEMSimulationRequest = {
      model_ir: exec.ir!,
      start_mhz: 850,
      stop_mhz: 950,
      steps: 5,
      mesh_size_m: 0.015,
      reference_impedance_ohms: 50,
    };
    expect(buildFEMCoaxRunConfig(request)).toBeNull();
  }, 120000);
});
