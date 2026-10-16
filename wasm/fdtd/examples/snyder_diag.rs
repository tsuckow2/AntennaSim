//! Snyder dual-recording diagnostic (2026-10-15).
//!
//! The fixed (base-node source) solver is stable over 10 ns, but the
//! recorded terminal pair (V_base, I_bottom) yields a nonphysical Zin
//! (4.68 - j3334 ohm @ 2300 MHz vs the FEM anchor 28.95 - j17.46) and
//! the ringdown dies within ~1 ns. This run records TWO terminal pairs
//! simultaneously on the identical simulation:
//!
//!   node pair : (V_base node voltage, I_bottom wire current) — what
//!               Fdtd::step's probe records.
//!   field pair: (V = -sum Ez*dz over the probe column k=32..=34
//!               from ground plane to patch plane, I = tight Ampere
//!               loop around the probe at the gap cell k=32).
//!
//! If the field pair lands near the FEM anchor, the physics is right
//! and the node-pair definition is the defect. If both are ~-j3300,
//! the drive never couples into the patch mode and the hunt moves to
//! the geometry (wire top / patch junction, ground return).
//!
//! Periodic audits print the cavity voltage at the patch-center
//! column, the wire energy, and max|Ez| inside the substrate slab,
//! so we can see whether the cavity ever charges.
//!
//! Usage: `cargo run --release --example snyder_diag -- [t_end_ns]`

use fdtd::cpml::Cpml;
use fdtd::geometry::{build_snyder, DX, DY, DZ, NX, NY, NZ};
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;
use fdtd::solver::Fdtd;
use fdtd::source::{cdiv, DftProbe, GaussianSource, ResistiveSource};

fn main() {
    let t_end_ns: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);
    let t_end = t_end_ns * 1e-9;

    let grid = YeeGrid::new(NX, NY, NZ, DX, DY, DZ);
    let cfl: f32 = std::env::var("SNYDER_CFL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    let dt = grid.courant_dt(cfl);
    let mut mat = Materials::vacuum(NX, NY, NZ, dt);
    let (feed, thin_wire) = build_snyder(&grid, &mut mat, dt);
    let (pi, pj) = (feed.src_i, feed.src_j);
    // SNYDER_POST=1: replace the Holland/Simpson filament (a=0.1mm —
    // the physical probe is a 1.3mm square pin, unrepresentable in
    // the thin-wire parametrization on this grid) with a grid-faithful
    // PEC post: Ez PEC over the plus-cross of columns whose centers
    // fall inside the 1.3mm pin, k=33..=34, driven by the Norton
    // lumped port in the gap cell k=32 (the efs-validated parts).
    let use_post = std::env::var("SNYDER_POST").is_ok();
    // SNYDER_THICKGND=1: volumetric ground — Ez PEC at k=30,31 over
    // the ground footprint. The regional audit showed the gap cell's
    // field mirrored by an equal-and-opposite Ez in the k=31 layer
    // whose edge straddles the ground sheet (820 vs -810 V/m): the
    // port voltage cancels across the sheet and the cavity sees ~1%.
    // Shorting the straddle layer kills the mirror (the efs lesson).
    if std::env::var("SNYDER_THICKGND").is_ok() {
        for j in 14..218usize {
            for i in 14..218usize {
                mat.set_pec(2, grid.ez_idx(i, j, 31));
                mat.set_pec(2, grid.ez_idx(i, j, 30));
            }
        }
        println!("SNYDER_THICKGND: Ez PEC at k=30,31 over the ground footprint");
    }
    let wire_arg = if use_post {
        for (di, dj) in [(0isize, 0isize), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (ii, jj) = ((pi as isize + di) as usize, (pj as isize + dj) as usize);
            for k in 33..=34 {
                mat.set_pec(2, grid.ez_idx(ii, jj, k));
            }
        }
        println!("SNYDER_POST: PEC post (5-col cross, k=33..=34) + Norton gap port");
        None
    } else {
        Some(thin_wire)
    };
    println!(
        "probe column ({pi},{pj})  wire k {}..={}  dt={dt:.4e}",
        wire_arg.as_ref().map(|w| w.k_lo).unwrap_or(0),
        wire_arg.as_ref().map(|w| w.k_hi).unwrap_or(0)
    );

    let cpml = Cpml::new(&grid, 12, dt);
    let source = GaussianSource::new(2.5e9, 1.0);
    let rsource = ResistiveSource { rs: 50.0 };
    let freqs: Vec<f64> = (0..=200).map(|m| 1.0e9 + m as f64 * 10e6).collect();
    let probe = DftProbe::new(freqs.clone(), dt);
    let mut field_probe = DftProbe::new(freqs.clone(), dt);

    let mut sim = Fdtd::new(
        grid,
        mat,
        cpml,
        dt,
        feed,
        source,
        rsource,
        probe,
        wire_arg,
    );
    let n_steps = (t_end / dt).ceil() as usize;
    println!("running {n_steps} steps ({t_end_ns} ns)...");
    let t_start = std::time::Instant::now();

    // TM10 (x-resonant, L=28.2mm): Ez NODE at patch center x=0,
    // ANTINODES at the radiating edges x=+-14.1mm. Audit the two edge
    // columns and the node column, plus the x-profile of column
    // voltages at i = 88 (edge), 102 (feed), 116 (node), 130, 143
    // (edge). (The first diagnostic run audited the center column and
    // a scan window covering only |x|<8mm — it was blind to the mode.)
    let prof_i = [88usize, 102, 116, 130, 143];
    let cj = 116usize;
    let (mut ev_node, mut ev_field) = (0.0f64, 0.0f64);
    let audit_every = 1024usize;

    for s in 0..n_steps {
        sim.step();
        // Field pair at the same time convention as the solver probe:
        // V from E^{n+1}, I from H^{n+1/2} via the tight loop at k=32.
        let g = &sim.grid;
        let v_f: f32 = (32..=34).map(|k| -g.ez[g.ez_idx(pi, pj, k)] * DZ).sum();
        let i_f = sim.feed.measure_i_tight(g);
        field_probe.push(v_f, i_f);

        if s % audit_every == audit_every - 1 || s + 1 == n_steps {
            let (vn, in_) = {
                let ser = sim.probe.series();
                let n = ser.0.len();
                (ser.0[n - 1], ser.1[n - 1])
            };
            ev_node += 0.0; // (energies accumulated from DFT pairs below instead)
            let _ = ev_node;
            let vcav: f32 = (32..=34).map(|k| -g.ez[g.ez_idx(prof_i[2], cj, k)] * DZ).sum();
            let prof: Vec<String> = prof_i
                .iter()
                .map(|&ii| {
                    let v: f32 = (32..=34).map(|k| -g.ez[g.ez_idx(ii, cj, k)] * DZ).sum();
                    format!("{v:+.2e}")
                })
                .collect();
            // max |Ez| inside the substrate slab under the patch
            // (patch footprint: i 84..148, j 78..154)
            let (mut mx, mut mloc) = (0.0f32, (0, 0, 0));
            for k in 32..=34 {
                for j in 78..154 {
                    for i in 84..148 {
                        let v = g.ez[g.ez_idx(i, j, k)].abs();
                        if v > mx {
                            mx = v;
                            mloc = (i, j, k);
                        }
                    }
                }
            }
            let w_e = sim.thin_wire.as_ref().map(|w| w.energy()).unwrap_or(0.0);
            // Regional |Ez| maxima: below ground (k=20..30), slab
            // (32..34), above patch (36..44) — is the drive energy
            // leaking into the air box under the ground sheet?
            let reg = |k0: usize, k1: usize| {
                let mut m = 0.0f32;
                for k in k0..=k1 {
                    for j in 90..142 {
                        for i in 90..142 {
                            m = m.max(g.ez[g.ez_idx(i, j, k)].abs());
                        }
                    }
                }
                m
            };
            println!(
                "    regions: maxEz below-gnd={:.3e}  slab={:.3e}  above-patch={:.3e}  Ezcol(feed, k=28..38)=[{}]",
                reg(20, 30),
                reg(32, 34),
                reg(36, 44),
                (28..=38)
                    .map(|k| format!("{:.1e}", g.ez[g.ez_idx(pi, pj, k)]))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            println!(
                "  step {}/{n_steps} t={:.2} ns  node(V={vn:+.3e},I={in_:+.3e})  field(V={v_f:+.3e},I={i_f:+.3e})  Vprof[88,102,116,130,143]=[{}]  wireE={w_e:.3e}  maxEz_slab={mx:.3e} at {mloc:?}  ({:.0} steps/s)",
                s + 1,
                sim.sim_time() * 1e9,
                prof.join(","),
                (s + 1) as f64 / t_start.elapsed().as_secs_f64(),
            );
            let _ = vcav;
        }
    }
    let _ = ev_field;
    println!("wall: {:.1}s", t_start.elapsed().as_secs_f64());

    let (nvf, ncf) = sim.probe.dft();
    let (fvf, fcf) = field_probe.dft();
    // Port energies from the recorded series.
    let (sv, si) = sim.probe.series();
    let e_node: f64 = sv
        .iter()
        .zip(si.iter())
        .map(|(v, i)| (*v as f64) * (*i as f64))
        .sum::<f64>()
        * dt as f64;
    println!("node-pair port energy int(V*I)dt = {e_node:.4e} J");
    println!("# MHz   Zin_node (re, im, |S11|dB)      Zin_field (re, im, |S11|dB)");
    for (m, &f) in freqs.iter().enumerate() {
        let (zr1, zi1) = cdiv(nvf[m].0, nvf[m].1, ncf[m].0, ncf[m].1);
        let (zr2, zi2) = cdiv(fvf[m].0, fvf[m].1, fcf[m].0, fcf[m].1);
        let s11 = |zr: f64, zi: f64| {
            let (nr, ni) = (zr - 50.0, zi);
            let (dr, di) = (zr + 50.0, zi);
            10.0 * ((nr * nr + ni * ni) / (dr * dr + di * di)).log10()
        };
        if m % 10 == 0 || (f - 2.3e9).abs() < 1.0 {
            println!(
                "{:.0}  ({:+9.2},{:+10.2},{:+6.2})   ({:+9.2},{:+10.2},{:+6.2})",
                f / 1e6,
                zr1,
                zi1,
                s11(zr1, zi1),
                zr2,
                zi2,
                s11(zr2, zi2)
            );
        }
    }
}
