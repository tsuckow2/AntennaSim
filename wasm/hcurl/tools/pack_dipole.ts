/**
 * Dev tool: pack the 3 GHz reference dipole into a v1 request and record the
 * TypeScript reference result for Rust parity checks.
 *
 * Bundle with esbuild and run with node:
 *   ./node_modules/.bin/esbuild tools/pack_dipole.ts --bundle --platform=node \
 *     --format=cjs --outfile=/tmp/pack_dipole.cjs && node /tmp/pack_dipole.cjs
 */
import { writeFileSync } from "node:fs";
import { buildFEMMesh } from "../../../frontend/src/engine/mesh";
import { solveFEMPoint, buildEdgeSystem, assembleHcurl } from "../../../frontend/src/engine/fem-solver";
import { packHcurlPointRequest } from "../../../frontend/src/engine/wasm/hcurl-abi";
import type { ModelIR } from "../../../frontend/src/dsl/model-ir";

function dipoleIR(): ModelIR {
  return {
    version: 1,
    paramSchemaVersion: 1,
    materials: [{ name: "copper", epsilon_r: 1, mu_r: 1, conductivity_s_m: 5.8e7 }],
    entities: [
      { id: "arm1", kind: "box", center: [-0.011, 0, 0], size: [0.02, 0.002, 0.002], material: "copper" },
      { id: "arm2", kind: "box", center: [0.011, 0, 0], size: [0.02, 0.002, 0.002], material: "copper" },
    ],
    ports: [
      { id: "feed", kind: "lumpedPort", positive: [-0.001, 0, 0], negative: [0.001, 0, 0], impedance_ohms: 50 },
    ],
    boundaries: [{ kind: "radiation", padding_m: 0.02 }],
    meshHints: [],
  };
}

const mesh = buildFEMMesh(dipoleIR(), 0.004, {
  positive: [-0.001, 0, 0],
  negative: [0.001, 0, 0],
});
console.log("tets:", mesh.statistics.elements, "vertices:", mesh.statistics.nodes);
const tsSys = buildEdgeSystem(mesh);
console.log("TS freeDofCount:", tsSys.freeDofCount, "edgeCount:", tsSys.edgeCount);
{
  // ABC anchor diagnostics on the assembled TS matrix.
  const { matrix } = assembleHcurl(mesh, 3.0e9, tsSys);
  let nnz = 0;
  let norm2 = 0;
  let dmin = Infinity;
  let dmax = -Infinity;
  for (let r = 0; r < matrix.n; r++) {
    const end = matrix.rowPtr[r + 1] ?? 0;
    for (let k = matrix.rowPtr[r] ?? 0; k < end; k++) {
      const im = matrix.im[k] ?? 0;
      if (im !== 0) {
        nnz++;
        norm2 += im * im;
        if (matrix.colIdx[k] === r) {
          if (im < dmin) dmin = im;
          if (im > dmax) dmax = im;
        }
      }
    }
  }
  console.log("TS ABC imag nnz:", nnz, "norm:", Math.sqrt(norm2), "diag:", dmin, dmax);
}
const ts = solveFEMPoint({
  mesh,
  frequencyHz: 3.0e9,
  referenceImpedanceOhms: 50,
  portPositive: [-0.001, 0, 0],
  portNegative: [0.001, 0, 0],
  farField: { thetaStepDeg: 5, phiStepDeg: 5 },
});
console.log("TS status:", ts.status);
console.log("TS Zin:", ts.impedanceOhms);
console.log("TS S11:", ts.s11);
console.log("TS iters:", ts.iterationCount, "residual:", ts.residualNorm);
console.log("TS far-field max gain:", ts.farField?.maxGainDbi, "dBi @", ts.farField?.maxGainThetaDeg, "°,", ts.farField?.maxGainPhiDeg, "°");
console.log("TS P_rad:", ts.farField?.radiatedPowerW, "P_acc:", ts.farField?.acceptedPowerW, "eff:", ts.farField?.radiationEfficiency);
console.log("TS timing:", ts.timingMs);

const portAttr = new Uint32Array(mesh.portPositiveFacets.length + mesh.portNegativeFacets.length);
portAttr.fill(30, 0, mesh.portPositiveFacets.length);
portAttr.fill(31, mesh.portPositiveFacets.length);

const packed = packHcurlPointRequest({
  frequencyHz: 3.0e9,
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
    mesh_artifact_version: mesh.version,
    reference_impedance_ohms: 50,
    krylov_tolerance: 1e-7,
    far_field: { theta_step_deg: 5, phi_step_deg: 5 },
    solver_formulation: "hcurl_nedelec_first_order",
    absorbing_boundary: "silver_muller_first_order",
    abi_version: 1,
  }),
});
writeFileSync("/tmp/dipole_3ghz.req", packed);
console.log("wrote /tmp/dipole_3ghz.req", packed.byteLength, "bytes");
