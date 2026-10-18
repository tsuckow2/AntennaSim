/**
 * Reactance-crossing classification and resonance identification.
 *
 * Every X = 0 crossing is not "the resonance". A driven sweep can cross
 * zero reactance more than once, with opposite meanings:
 *
 * - **Parallel resonance** — X falls through zero (+ to -) as frequency
 *   rises, with Re(Z) near a local peak. This is the micropatch cavity
 *   resonance (and the usual design target).
 * - **Series crossing** — X rises through zero (- to +). For a
 *   series-resonant antenna (dipole, monopole) this *is* the resonance;
 *   but a patch sweep continued above its parallel resonance also
 *   produces a shallow rising crossing where the probe inductance
 *   cancels the cavity's capacitive tail, at a resistance of only a few
 *   ohms. Reporting that crossing as "the resonance" misstates the
 *   antenna's operating point (it is how a 2.447 GHz / 3.7 ohm point was
 *   once mistaken for a patch resonance whose true parallel resonance
 *   sat at 2.288 GHz / ~37 ohm).
 *
 * Identification rule: among the crossings whose resistance is at
 * least a quarter of the sweep's maximum Re(Z) (which rejects the
 * shallow tail crossings), the resonance is the one nearest the
 * sweep's minimum-|S11| frequency. When no crossing qualifies, there
 * is no identified resonance — the sweep may not bracket it at all.
 */

export interface ReactanceSample {
  frequencyMhz: number;
  resistanceOhms: number;
  reactanceOhms: number;
  /** S11 magnitude in dB (negative), when known. */
  s11Db?: number | null;
}

export interface ReactanceCrossing {
  frequencyMhz: number;
  kind: "parallel" | "series";
  /** Interpolated Re(Z) at the crossing, ohms. */
  resistanceOhms: number;
}

/** Minimum fraction of the sweep's peak Re(Z) for a crossing to count
 * as a resonance candidate. */
const MIN_RESISTANCE_FRACTION = 0.25;

/**
 * All reactance zero crossings in the sweep, linearly interpolated,
 * each classified by direction and carrying its interpolated Re(Z).
 */
export function findReactanceCrossings(
  samples: readonly ReactanceSample[],
): ReactanceCrossing[] {
  const sorted = [...samples].sort((a, b) => a.frequencyMhz - b.frequencyMhz);
  const crossings: ReactanceCrossing[] = [];
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (!a || !b) continue;
    const x0 = a.reactanceOhms;
    const x1 = b.reactanceOhms;
    let kind: "parallel" | "series" | null = null;
    if (x0 >= 0 && x1 < 0) kind = "parallel";
    else if (x0 < 0 && x1 >= 0) kind = "series";
    if (kind === null) continue;
    const denom = Math.abs(x0) + Math.abs(x1);
    const t = denom > 0 ? Math.abs(x0) / denom : 0;
    crossings.push({
      frequencyMhz: a.frequencyMhz + t * (b.frequencyMhz - a.frequencyMhz),
      kind,
      resistanceOhms: a.resistanceOhms + t * (b.resistanceOhms - a.resistanceOhms),
    });
  }
  return crossings;
}

/**
 * Identify the sweep's resonance, or return null when no crossing
 * qualifies (see the module doc for the rule).
 */
export function identifyResonance(
  samples: readonly ReactanceSample[],
): ReactanceCrossing | null {
  if (samples.length === 0) return null;
  const crossings = findReactanceCrossings(samples);
  if (crossings.length === 0) return null;
  let maxR = -Infinity;
  let s11MinFreq: number | null = null;
  let s11MinDb = Infinity;
  for (const s of samples) {
    if (Number.isFinite(s.resistanceOhms)) maxR = Math.max(maxR, s.resistanceOhms);
    if (s.s11Db !== null && s.s11Db !== undefined && Number.isFinite(s.s11Db) && s.s11Db < s11MinDb) {
      s11MinDb = s.s11Db;
      s11MinFreq = s.frequencyMhz;
    }
  }
  const threshold = MIN_RESISTANCE_FRACTION * maxR;
  const candidates = crossings.filter((c) => c.resistanceOhms >= threshold);
  if (candidates.length === 0) return null;
  if (s11MinFreq === null) {
    // No S11 evidence: take the strongest (highest-R) crossing.
    return candidates.reduce((best, c) => (c.resistanceOhms > best.resistanceOhms ? c : best));
  }
  const target = s11MinFreq;
  return candidates.reduce((best, c) => {
    const dBest = Math.abs(best.frequencyMhz - target);
    const dC = Math.abs(c.frequencyMhz - target);
    if (dC !== dBest) return dC < dBest ? c : best;
    return c.resistanceOhms > best.resistanceOhms ? c : best;
  });
}
