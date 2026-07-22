// Parametric 918 MHz patch in a PETG enclosure.
// Lengths are metres; PETG properties are defaults and should be calibrated.
function defineParams() {
  return {
    version: 1,
    title: "918 MHz PETG patch",
    params: {
      patchWidth: { type: "number", default: 0.045, min: 0.02, max: 0.08, step: 0.0005, unit: "m" },
      patchLength: { type: "number", default: 0.035, min: 0.02, max: 0.08, step: 0.0005, unit: "m" },
      caseOffset: { type: "number", default: 0.008, min: 0.002, max: 0.03, step: 0.0005, unit: "m" },
      includeCase: { type: "bool", default: true },
    },
    groups: [
      { title: "Patch", params: ["patchWidth", "patchLength"] },
      { title: "Enclosure", params: ["caseOffset", "includeCase"] },
    ],
  };
}

function buildModel(params, api) {
  api.material("Copper", { epsilon_r: 1, conductivity_s_m: 58000000 });
  api.material("PETG", { epsilon_r: 2.9, loss_tangent: 0.015 });
  api.material("Air", { epsilon_r: 1 });
  api.box("substrate", { center: [0, 0, 0], size: [0.06, 0.06, 0.0016], material: "Air" });
  api.sheetRect("ground", { center: [0, 0, 0.0016], size: [0.06, 0.06], material: "Copper" });
  api.sheetRect("patch", { center: [0, 0, 0.0032], size: [params.patchWidth, params.patchLength], material: "Copper" });
  api.lumpedPort("feed", { positive: [0, 0, 0.0032], negative: [0, 0, 0.0016] });
  if (params.includeCase) {
    api.box("case", { center: [0, 0, 0.0032 + params.caseOffset], size: [0.09, 0.09, 0.003], material: "PETG" });
  }
  api.airbox({ padding_m: 0.1 });
  api.refine({ region: "patch", max_size_m: 0.005 });
}
