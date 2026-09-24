// Parametric 908 MHz probe-fed directional patch antenna.
//
// Single microstrip patch over a ground plane, fed by a coaxial probe inset
// from the radiating edge. The ground plane makes the pattern directional
// (broadside, toward +z) with front-to-back set by the ground size.
//
// Defaults mirror the validated FEM model: 92 mm resonant length, 34 mm
// probe inset for ~50 ohm match, 10 mm Rogers-5880-like substrate
// (er 2.2), 350 mm ground plane.
//
// Parameter lengths are millimeters; ModelIR geometry is emitted in metres.
function defineParams() {
  return {
    version: 1,
    title: "908 MHz directional patch",
    params: {
      patchLength: { type: "number", default: 92, min: 60, max: 120, step: 0.5, unit: "mm" },
      patchWidth: { type: "number", default: 106, min: 60, max: 140, step: 0.5, unit: "mm" },
      probeInset: { type: "number", default: 34, min: 8, max: 55, step: 0.5, unit: "mm" },
      substrateHeight: { type: "number", default: 10, min: 3, max: 20, step: 0.5, unit: "mm" },
      groundSize: { type: "number", default: 350, min: 200, max: 600, step: 5, unit: "mm" },
    },
    groups: [
      { title: "Patch", params: ["patchLength", "patchWidth", "probeInset"] },
      { title: "Stackup", params: ["substrateHeight", "groundSize"] },
    ],
  };
}

function buildModel(params, api) {
  const mm = (value) => value / 1000;
  const L = mm(params.patchLength);
  const W = mm(params.patchWidth);
  const inset = mm(params.probeInset);
  const H = mm(params.substrateHeight);
  const G = mm(params.groundSize);
  const copperT = 0.004;
  const probeHalf = 0.002; // 4x4 mm probe
  const probeX = -L / 2 + inset; // measured from the -x patch edge
  const subHalf = G / 2;

  api.material("Copper", { epsilon_r: 1, conductivity_s_m: 58000000 });
  api.material("Rogers5880", { epsilon_r: 2.2, loss_tangent: 0.001 });

  // Ground plane
  api.box("ground", { center: [0, 0, -copperT / 2], size: [G, G, copperT], material: "Copper" });

  // Substrate: four boxes around the probe so no volumes overlap
  const subZ = H / 2;
  api.box("substrate_l", {
    center: [(-subHalf + probeX - probeHalf) / 2, 0, subZ],
    size: [probeX - probeHalf + subHalf, G, H],
    material: "Rogers5880",
  });
  api.box("substrate_r", {
    center: [(probeX + probeHalf + subHalf) / 2, 0, subZ],
    size: [subHalf - probeX - probeHalf, G, H],
    material: "Rogers5880",
  });
  api.box("substrate_f", {
    center: [probeX, (-subHalf - probeHalf) / 2, subZ],
    size: [probeHalf * 2, subHalf - probeHalf, H],
    material: "Rogers5880",
  });
  api.box("substrate_b", {
    center: [probeX, (probeHalf + subHalf) / 2, subZ],
    size: [probeHalf * 2, subHalf - probeHalf, H],
    material: "Rogers5880",
  });

  // Probe: 1 mm above the ground plane up to the patch underside
  const probeZ0 = 0.001;
  api.box("probe", {
    center: [probeX, 0, (probeZ0 + H) / 2],
    size: [probeHalf * 2, probeHalf * 2, H - probeZ0],
    material: "Copper",
  });

  // Patch
  api.box("patch", {
    center: [0, 0, H + copperT / 2],
    size: [L, W, copperT],
    material: "Copper",
  });

  // Coax-style probe feed: 1 mm vertical lumped port, probe vs ground
  api.lumpedPort("port1", {
    positive: [probeX, 0, probeZ0],
    negative: [probeX, 0, 0],
    impedance_ohms: 50,
  });

  api.airbox({ padding_m: 0.06 });
  // Note: a finer patch refine hint was removed (2026-09-24). It doubled the
  // free DOFs (98k -> 179k) and pushed the problem past what the TypeScript
  // reference solver's SGS-preconditioned BiCGSTAB can converge. The uniform
  // 15 mm mesh (lambda/22 at 908 MHz) solves in ~9k iterations.
}
