import { describe, expect, it } from "vitest";
import { findReactanceCrossings, identifyResonance, type ReactanceSample } from "../resonance";
import { identifyFEMResonance, type FEMFrequencyPoint } from "../../engine/fem";

/**
 * The Snyder patch coax sweep at xy2.5 (FEM, AMS) that resolved the
 * "3.7 ohm anomaly": the patch's true parallel resonance is at
 * ~2.288 GHz with Rin ~35-40 ohm, and the shallow rising crossing near
 * 2.385 GHz (R ~5 ohm, probe inductance cancelling the capacitive tail)
 * is NOT a resonance. Frequencies in MHz, impedance in ohms.
 */
const SNYDER_SWEEP: ReadonlyArray<readonly [number, number, number]> = [
  [1900, 0.36, 11.76],
  [2000, 0.66, 14.03],
  [2100, 1.61, 17.81],
  [2200, 7.74, 26.81],
  [2240, 21.27, 31.39],
  [2260, 34.48, 25.39],
  [2280, 39.22, 6.71],
  [2290, 34.55, -1.57],
  [2300, 28.0, -6.1],
  [2320, 17.02, -7.33],
  [2350, 8.69, -3.7],
  [2400, 3.89, 1.57],
  [2447, 2.32, 4.69],
  [2500, 1.54, 7.08],
];

function s11Db(re: number, im: number, z0 = 50): number {
  const num = Math.hypot(re - z0, im);
  const den = Math.hypot(re + z0, im);
  return 20 * Math.log10(num / den);
}

function samples(rows: ReadonlyArray<readonly [number, number, number]>): ReactanceSample[] {
  return rows.map(([frequencyMhz, resistanceOhms, reactanceOhms]) => ({
    frequencyMhz,
    resistanceOhms,
    reactanceOhms,
    s11Db: s11Db(resistanceOhms, reactanceOhms),
  }));
}

describe("findReactanceCrossings", () => {
  it("classifies the Snyder sweep's two crossings by direction, with interpolated R", () => {
    const crossings = findReactanceCrossings(samples(SNYDER_SWEEP));
    expect(crossings).toHaveLength(2);
    const [parallel, series] = crossings as [typeof crossings[0], typeof crossings[0]];
    expect(parallel.kind).toBe("parallel");
    expect(parallel.frequencyMhz).toBeCloseTo(2288.1, 0);
    expect(parallel.resistanceOhms).toBeCloseTo(35.4, 0);
    expect(series.kind).toBe("series");
    expect(series.frequencyMhz).toBeCloseTo(2385.1, 0);
    expect(series.resistanceOhms).toBeCloseTo(5.3, 0);
  });

  it("returns no crossings when X never changes sign", () => {
    expect(findReactanceCrossings(samples([[100, 50, 10], [200, 50, 20]]))).toHaveLength(0);
  });
});

describe("identifyResonance", () => {
  it("picks the parallel resonance, not the shallow series crossing (the anomaly regression)", () => {
    const resonance = identifyResonance(samples(SNYDER_SWEEP));
    expect(resonance).not.toBeNull();
    expect(resonance?.kind).toBe("parallel");
    expect(resonance?.frequencyMhz).toBeCloseTo(2288.1, 0);
    expect(resonance?.resistanceOhms).toBeCloseTo(35.4, 0);
  });

  it("reports no resonance for a sweep that starts above f0 (only the tail crossing is in range)", () => {
    const aboveF0 = SNYDER_SWEEP.filter(([f]) => f >= 2300);
    expect(identifyResonance(samples(aboveF0))).toBeNull();
  });

  it("identifies a series-resonant antenna (dipole) by its rising crossing", () => {
    const dipole = samples([
      [2800, 40, -80],
      [2900, 55, -45],
      [2950, 66, -20],
      [3000, 73, -2],
      [3050, 71, 18],
      [3100, 60, 40],
      [3200, 45, 75],
    ]);
    const resonance = identifyResonance(dipole);
    expect(resonance).not.toBeNull();
    expect(resonance?.kind).toBe("series");
    expect(resonance?.frequencyMhz).toBeCloseTo(3005, 0);
    expect(resonance?.resistanceOhms).toBeCloseTo(72.8, 0);
  });

  it("returns null with no crossings or no samples", () => {
    expect(identifyResonance(samples([[100, 50, 10], [200, 50, 20]]))).toBeNull();
    expect(identifyResonance([])).toBeNull();
  });
});

describe("identifyFEMResonance", () => {
  function femPoints(rows: ReadonlyArray<readonly [number, number, number]>): FEMFrequencyPoint[] {
    return rows.map(([frequency_mhz, re, im]) => {
      const den = (re + 50) ** 2 + im ** 2;
      const s11_real = ((re - 50) * (re + 50) + im * im) / den;
      const s11_imag = (2 * im * 50) / den;
      return {
        frequency_mhz,
        point_status: "ok" as const,
        impedance_real_ohms: re,
        impedance_imag_ohms: im,
        s11_real,
        s11_imag,
        s11_db: s11Db(re, im),
        swr: (1 + Math.hypot(s11_real, s11_imag)) / (1 - Math.hypot(s11_real, s11_imag)),
      };
    });
  }

  it("finds the Snyder parallel resonance from FEM points", () => {
    const resonance = identifyFEMResonance(femPoints(SNYDER_SWEEP));
    expect(resonance?.kind).toBe("parallel");
    expect(resonance?.frequencyMhz).toBeCloseTo(2288.1, 0);
  });

  it("ignores non-ok and non-finite points", () => {
    const points = femPoints(SNYDER_SWEEP);
    for (const p of points) {
      if (p.frequency_mhz < 2260) {
        p.point_status = "solver_error";
        p.impedance_real_ohms = null;
        p.impedance_imag_ohms = null;
      }
    }
    // The falling crossing is still bracketed by the surviving points
    // (2260 +j25.39 -> 2290 -j1.57), so a resonance is still found,
    // interpolated across the wider gap.
    const resonance = identifyFEMResonance(points);
    expect(resonance?.kind).toBe("parallel");
    expect(resonance?.frequencyMhz).toBeGreaterThan(2280);
    expect(resonance?.frequencyMhz).toBeLessThan(2290);
  });
});
