// Snyder 2.44 GHz micropatch with a physical coax wave-port feed.
//
// The validated Snyder model (Cal Poly SLO senior project, 2010):
// 28.2084 x 37.5789 mm patch on 62-mil FR-4, 101.6 mm ground, probe
// inset 7.5658 mm from the radiating edge. HFSS reference: 2.437 GHz,
// 50 ohm match.
//
// Unlike the lumped-port examples, the feed is the physical coax: a
// 1.3 mm square pin in a PTFE annulus (er 2.1) inside a copper frame,
// driven by the FEM solver's TEM wave port below the ground plane.
// The substrate and ground are authored whole — the mesher holes the
// feed column out of them and injects the coax geometry.
//
// Parameter lengths are in the units noted; ModelIR geometry is metres.
function defineParams() {
  return {
    version: 1,
    title: "Snyder patch — coax wave-port feed",
    params: {
      er: { type: "number", default: 4.4, min: 2, max: 12, step: 0.1, unit: "" },
      xyMm: { type: "number", default: 2.5, min: 1, max: 5, step: 0.25, unit: "mm" },
    },
    groups: [
      { title: "Model", params: ["er", "xyMm"] },
    ],
  };
}

function buildModel(params, api) {
  const H = 0.001575; // 62-mil substrate
  const G = 0.1016; // 4-inch substrate / ground
  const L = 0.0282084; // patch length (x)
  const W = 0.0375789; // patch width (y)
  const FEED_FROM_EDGE = 0.0075658;
  const probeX = -L / 2 + FEED_FROM_EDGE;

  api.material("copper", { epsilon_r: 1, conductivity_s_m: 5.8e7 });
  api.material("fr4", { epsilon_r: params.er, loss_tangent: 0.02 });
  api.material("ptfe", { epsilon_r: 2.1 });

  // Substrate and ground, authored whole: the coax feed column is
  // holed out of both by the mesher.
  api.box("substrate", { center: [0, 0, H / 2], size: [G, G, H], material: "fr4" });
  api.sheetRect("ground", { center: [0, 0, 0], size: [G, G], normal: [0, 0, 1], material: "copper" });
  api.sheetRect("patch", { center: [0, 0, H], size: [L, W], normal: [0, 0, 1], material: "copper" });

  api.coaxPort("feed", {
    center: [probeX, 0],
    ground_z_m: 0,
    pin_top_z_m: H,
  });

  api.airbox({ padding_m: 0.03 });
  api.refine({
    region: "substrate",
    max_size_m: 0.005,
    max_size_by_axis_m: { x: params.xyMm / 1000, y: params.xyMm / 1000, z: H },
  });
}
