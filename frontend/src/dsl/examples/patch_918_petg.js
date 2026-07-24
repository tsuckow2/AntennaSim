// Parametric 918 MHz patch in a PETG enclosure.
// Parameter lengths are millimeters; ModelIR geometry is emitted in metres.
function defineParams() {
  return {
    version: 1,
    title: "918 MHz PETG patch",
    params: {
      patchWidth: { type: "number", default: 45, min: 20, max: 80, step: 0.5, unit: "mm" },
      patchLength: { type: "number", default: 35, min: 20, max: 80, step: 0.5, unit: "mm" },
      copperThickness: { type: "number", default: 0.035, min: 0.01, max: 0.2, step: 0.005, unit: "mm" },
      caseOffset: { type: "number", default: 8, min: 2, max: 30, step: 0.5, unit: "mm" },
      includeCase: { type: "bool", default: true },
    },
    groups: [
      { title: "Patch", params: ["patchWidth", "patchLength", "copperThickness"] },
      { title: "Enclosure", params: ["caseOffset", "includeCase"] },
    ],
  };
}

function buildModel(params, api) {
  const mm = (value) => value / 1000;
  api.material("Copper", { epsilon_r: 1, conductivity_s_m: 58000000 });
  api.material("PETG", { epsilon_r: 2.9, loss_tangent: 0.015 });
  api.material("Air", { epsilon_r: 1 });
  api.box("substrate", { center: [0, 0, 0], size: [0.06, 0.06, 0.0016], material: "Air" });
  const copperThickness = mm(params.copperThickness);
  const groundCenterZ = 0.0016 + copperThickness / 2;
  const patchCenterZ = 0.0032 + copperThickness / 2;
  api.box("ground", { center: [0, 0, groundCenterZ], size: [0.06, 0.06, copperThickness], material: "Copper" });
  api.box("patch", { center: [0, 0, patchCenterZ], size: [mm(params.patchWidth), mm(params.patchLength), copperThickness], material: "Copper" });
  api.lumpedPort("feed", { positive: [0, 0, 0.0032 + copperThickness], negative: [0, 0, 0.0016 + copperThickness] });
  if (params.includeCase) {
    api.box("case", { center: [0, 0, 0.0032 + mm(params.caseOffset)], size: [0.09, 0.09, 0.003], material: "PETG" });
  }
  api.airbox({ padding_m: 0.1 });
  api.refine({ region: "patch", max_size_m: 0.005 });
}
