//! Lumped feed: resistive (Thevenin) voltage source, V/I probes, on-the-fly DFT.
//!
//! The feed is a 1-cell z-gap between the ground plane and the probe.
//! A Luebbers & Langdon (1996) resistive source (series Rs = 50 Ω, the
//! coax Z0) drives the gap cell. Per timestep, with H^{n+½} known:
//!
//! ```text
//! I_s = [Hx(is,js,ks) − Hx(is,js+1,ks)]·dx           (tight 1-cell
//!     + [Hy(is+1,js,ks) − Hy(is,js,ks)]·dy           Ampère loop around
//!                                                    the source cell,
//!                                                    CCW viewed from +z)
//! E_z(is,js,ks) = −V_s/dz + I_s·Rs/dz               (replaces E-update
//!                                                    at source cell)
//! ```
//!
//! (+I_s flows in +z, upward into the probe; terminal voltage
//! V = −E_z·dz = V_s − I_s·Rs.) The resistor absorbs reflected waves,
//! so the run converges ~32× faster than a hard (Rs=0) source, and
//! Zin(f) = DFT(V)/DFT(I) needs no de-embedding. Record the ACTUAL
//! gap voltage (−E_z·dz), never the prescribed V_s(t).

use crate::grid::YeeGrid;

/// Differentiated-Gaussian source: `V_s(t) = −V₀·((t−t₀)/τ)·exp(−((t−t₀)/τ)²)`.
pub struct GaussianSource {
    v0: f32,
    t0: f32,
    tau: f32,
}

impl GaussianSource {
    /// `peak_freq_hz`: the spectrum peaks here. `t0 = 4τ`.
    pub fn new(peak_freq_hz: f32, v0: f32) -> Self {
        let tau = 1.0 / (std::f32::consts::PI * peak_freq_hz * std::f32::consts::SQRT_2);
        Self {
            v0,
            t0: 4.0 * tau,
            tau,
        }
    }

    pub fn value(&self, t: f32) -> f32 {
        let x = (t - self.t0) / self.tau;
        -self.v0 * x * (-x * x).exp()
    }

    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Raw waveform parameters `(v0, t0, tau)` for external backends
    /// (e.g. the GPU solver) that evaluate the pulse on-device.
    pub fn params(&self) -> (f32, f32, f32) {
        (self.v0, self.t0, self.tau)
    }
}

/// Feed geometry: gap (source) cell and Ampère-loop ranges.
pub struct Feed {
    /// Gap cell (source): Ez index.
    pub src_i: usize,
    pub src_j: usize,
    pub src_k: usize,
    /// Ampère loop at H-step k_loop: Hx rows j_lo/j_hi over i in i0..i1,
    /// Hy cols i_lo/i_hi over j in j0..j1.
    pub k_loop: usize,
    pub hx_j_lo: usize,
    pub hx_j_hi: usize,
    pub hx_i0: usize,
    pub hx_i1: usize,
    pub hy_i_lo: usize,
    pub hy_i_hi: usize,
    pub hy_j0: usize,
    pub hy_j1: usize,
}

impl Feed {
    /// Measure I at the current H-step (call right after the H update).
    pub fn measure_i(&self, g: &YeeGrid) -> f32 {
        let k = self.k_loop;
        let mut sum_hx_lo = 0.0f32;
        let mut sum_hx_hi = 0.0f32;
        for i in self.hx_i0..self.hx_i1 {
            sum_hx_lo += g.hx[g.hx_idx(i, self.hx_j_lo, k)];
            sum_hx_hi += g.hx[g.hx_idx(i, self.hx_j_hi, k)];
        }
        let mut sum_hy_lo = 0.0f32;
        let mut sum_hy_hi = 0.0f32;
        for j in self.hy_j0..self.hy_j1 {
            sum_hy_lo += g.hy[g.hy_idx(self.hy_i_lo, j, k)];
            sum_hy_hi += g.hy[g.hy_idx(self.hy_i_hi, j, k)];
        }
        (sum_hx_lo - sum_hx_hi) * g.dx + (sum_hy_hi - sum_hy_lo) * g.dy
    }

    /// Tight 1-cell Ampère loop around the source Ez cell
    /// (`src_i`, `src_j`, `src_k`), counter-clockwise viewed from +z.
    /// By the right-hand rule +I flows in +z (upward, into the probe).
    /// This is the terminal current for the resistive source; do NOT use
    /// the wide loop above here (it picks up return/displacement currents).
    pub fn measure_i_tight(&self, g: &YeeGrid) -> f32 {
        let (si, sj, sk) = (self.src_i, self.src_j, self.src_k);
        // The four H faces of the Ez cell itself, per update_e's pairing:
        // Hx(si,sj−1)/Hx(si,sj) are its −y/+y faces, Hy(si−1,sj)/Hy(si,sj)
        // its −x/+x faces. (An earlier version sampled (sj, sj+1, si,
        // si+1) — the loop around the cell CORNER — which cancels the
        // circulating field and returns a small wrong-sign residual.)
        // Missing faces at the domain edge read 0, mirroring update_e.
        let hx_bot = if sj > 0 { g.hx[g.hx_idx(si, sj - 1, sk)] } else { 0.0 };
        let hx_top = g.hx[g.hx_idx(si, sj, sk)];
        let hy_lft = if si > 0 { g.hy[g.hy_idx(si - 1, sj, sk)] } else { 0.0 };
        let hy_rgt = g.hy[g.hy_idx(si, sj, sk)];
        (hx_bot - hx_top) * g.dx + (hy_rgt - hy_lft) * g.dy
    }
}

/// Accumulate V(t), I(t) and DFT them at the requested frequencies.
pub struct DftProbe {
    pub freqs_hz: Vec<f64>,
    v_series: Vec<f32>,
    i_series: Vec<f32>,
    dt: f64,
}

impl DftProbe {
    pub fn new(freqs_hz: Vec<f64>, dt: f32) -> Self {
        Self {
            freqs_hz,
            v_series: Vec::new(),
            i_series: Vec::new(),
            dt: dt as f64,
        }
    }

    pub fn push(&mut self, v: f32, i: f32) {
        self.v_series.push(v);
        self.i_series.push(i);
    }

    /// The recorded (V, I) time series (for cross-backend comparisons).
    pub fn series(&self) -> (&[f32], &[f32]) {
        (&self.v_series, &self.i_series)
    }

    pub fn len(&self) -> usize {
        self.v_series.len()
    }

    /// Last |V| relative to max |V|: ringdown diagnostic.
    pub fn tail_ratio(&self) -> f32 {
        if self.v_series.is_empty() {
            return 1.0;
        }
        let max_v = self
            .v_series
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-30);
        let n = self.v_series.len();
        let tail = self.v_series[n - n.min(100)..]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        tail / max_v
    }

    /// DFT of V and I at each frequency. Returns (V(f), I(f)).
    pub fn dft(&self) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let n = self.v_series.len();
        let mut vf = Vec::with_capacity(self.freqs_hz.len());
        let mut cf = Vec::with_capacity(self.freqs_hz.len());
        for &fhz in &self.freqs_hz {
            let mut vr = 0.0;
            let mut vi = 0.0;
            let mut ir = 0.0;
            let mut ii = 0.0;
            let omega_dt = 2.0 * std::f64::consts::PI * fhz * self.dt;
            // Goertzel-like recurrence: cheaper than exp() per sample.
            let (mut c, mut s) = (1.0f64, 0.0f64);
            let (cw, sw) = (omega_dt.cos(), omega_dt.sin());
            for m in 0..n {
                let v = self.v_series[m] as f64;
                let ci = self.i_series[m] as f64;
                vr += v * c;
                vi -= v * s;
                ir += ci * c;
                ii -= ci * s;
                let cn = c * cw - s * sw;
                s = s * cw + c * sw;
                c = cn;
            }
            vf.push((vr, vi));
            cf.push((ir, ii));
        }
        (vf, cf)
    }
}

/// Complex division: (a+ib)/(c+id).
pub fn cdiv(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let den = c * c + d * d;
    ((a * c + b * d) / den, (b * c - a * d) / den)
}

/// Luebbers & Langdon (1996) resistive (Thevenin) voltage source.
///
/// A series resistor Rs (the feedline characteristic impedance, 50 Ω)
/// placed at the source cell absorbs reflected waves, so the run settles
/// ~32× faster than a hard (Rs=0) source, and Zin = V/I needs no
/// de-embedding. The update replaces the ordinary FDTD E-update at the
/// source cell:
///
/// ```text
/// E_z(is,js,ks) = −V_s/dz + I_s·Rs/dz
/// ```
///
/// where `I_s` is the tight-loop current (upward/+z positive, measured at
/// the H-step just before) and `V_s(t)` the prescribed waveform. Terminal
/// voltage `V = −E_z·dz = V_s − I_s·Rs`. Record `(V, I_s)` in the probe —
/// the ACTUAL gap voltage, never the prescribed `V_s(t)`.
///
/// Sign convention: our hard source used `Ez = −V_s/dz`, so the resistive
/// update is the negated Luebbers form. If a validation run returns
/// Re(Zin) < 0, the loop orientation is flipped for your grid — negate
/// `I_s` (the physics is unchanged, only the sign reference).
pub struct ResistiveSource {
    /// Series resistance in ohms (50.0 for a coax-fed antenna).
    pub rs: f32,
}

impl ResistiveSource {
    /// Apply the source at the feed gap cell. Call after the E update
    /// (and PEC enforcement); `i_up` is `Feed::measure_i_tight` from the
    /// H-step just before. Returns `(V_terminal, I_up)` for the V/I probe.
    pub fn apply(&self, g: &mut YeeGrid, feed: &Feed, vs: f32, i_up: f32) -> (f32, f32) {
        let si = g.ez_idx(feed.src_i, feed.src_j, feed.src_k);
        let dz = g.dz;
        g.ez[si] = -vs / dz + i_up * self.rs / dz;
        let v_term = -g.ez[si] * dz;
        (v_term, i_up)
    }

    /// Norton-form resistive source for the lumped-port (no-wire) path.
    ///
    /// The overwrite form (`apply`) feeds the measured loop current back
    /// through Rs. That current contains the gap cell's own displacement
    /// current, whose feedback gain ω·C_gap·Rs exceeds 1 at grid
    /// frequencies and pumps a discrete resonance (observed ×1.02/step
    /// growth at ω·dt ≈ 0.46 on a 0.381 mm gap cell, at any CFL). The
    /// Norton form injects the source as an impressed current
    /// I_src = (Vs − V)/Rs into the cell's natural E update instead:
    /// explicit negative feedback with gain dt/(C_gap·Rs) ≪ 1, and the
    /// terminal relation V = Vs − I·Rs holds through the cell dynamics
    /// (exactly at DC). `cb_src` is the source cell's Cb (= dt/ε),
    /// used to recover ε for the deposit. Returns V_terminal after the
    /// deposit; the probe records it with the loop current.
    pub fn apply_norton(&self, g: &mut YeeGrid, feed: &Feed, vs: f32, dt: f32, cb_src: f32) -> f32 {
        let si = g.ez_idx(feed.src_i, feed.src_j, feed.src_k);
        let dz = g.dz;
        let v_now = -g.ez[si] * dz;
        let i_src = (vs - v_now) / self.rs;
        let eps = dt / cb_src;
        g.ez[si] -= dt * i_src / (eps * g.dx * g.dy);
        -g.ez[si] * dz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tight_loop_sign_ccw_from_pos_z() {
        // H curling CCW viewed from +z (right-hand rule) encloses a +z
        // (upward) current; the tight loop must report I > 0.
        let mut g = YeeGrid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let feed = Feed {
            src_i: 1,
            src_j: 1,
            src_k: 1,
            k_loop: 1,
            hx_j_lo: 0,
            hx_j_hi: 2,
            hx_i0: 0,
            hx_i1: 2,
            hy_i_lo: 0,
            hy_i_hi: 2,
            hy_j0: 0,
            hy_j1: 2,
        };
        // CCW from +z: bottom +x, top −x, right +y, left −y.
        // Faces of Ez(1,1,1) per update_e's pairing: Hx(1,0)=bottom,
        // Hx(1,1)=top, Hy(0,1)=left, Hy(1,1)=right.
        let (ib, it) = (g.hx_idx(1, 0, 1), g.hx_idx(1, 1, 1));
        let (il, ir) = (g.hy_idx(0, 1, 1), g.hy_idx(1, 1, 1));
        g.hx[ib] = 1.0; // bottom
        g.hx[it] = -1.0; // top
        g.hy[il] = -1.0; // left
        g.hy[ir] = 1.0; // right
        let i = feed.measure_i_tight(&g);
        assert!(i > 0.0, "CCW H must give +I (upward), got {i}");
        assert!((i - 4.0).abs() < 1e-6, "expected 2·dx+2·dy=4, got {i}");
    }

    #[test]
    fn resistive_source_thevenin() {
        // Thevenin: V_terminal = V_s − I·Rs; E_z = −V/dz.
        let mut g = YeeGrid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let feed = Feed {
            src_i: 1,
            src_j: 1,
            src_k: 1,
            k_loop: 1,
            hx_j_lo: 0,
            hx_j_hi: 2,
            hx_i0: 0,
            hx_i1: 2,
            hy_i_lo: 0,
            hy_i_hi: 2,
            hy_j0: 0,
            hy_j1: 2,
        };
        let rsrc = ResistiveSource { rs: 50.0 };
        // I_up = 0.02 A upward, V_s = 2 V → V_term = 2 − 1 = 1 V.
        let (v, i) = rsrc.apply(&mut g, &feed, 2.0, 0.02);
        assert!(
            (v - 1.0).abs() < 1e-6,
            "V_term should be V_s−I·Rs=1, got {v}"
        );
        assert!((i - 0.02).abs() < 1e-9, "I should pass through, got {i}");
        let ez = g.ez[g.ez_idx(1, 1, 1)];
        assert!(
            (ez - (-1.0)).abs() < 1e-6,
            "E_z should be −V/dz=−1, got {ez}"
        );
    }

    #[test]
    fn resistive_source_norton_charges_toward_vs() {
        // Norton form: from Ez=0 with Vs=2, Rs=50, the deposit must
        // drive V_terminal positive toward Vs (partial step: the cell
        // capacitance charges through Rs), never past it in one step.
        let mut g = YeeGrid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let feed = Feed {
            src_i: 1,
            src_j: 1,
            src_k: 1,
            k_loop: 1,
            hx_j_lo: 0,
            hx_j_hi: 2,
            hx_i0: 0,
            hx_i1: 2,
            hy_i_lo: 0,
            hy_i_hi: 2,
            hy_j0: 0,
            hy_j1: 2,
        };
        let rsrc = ResistiveSource { rs: 50.0 };
        let dt = 1e-12f32;
        let eps0 = 8.854_187_817e-12f32;
        let cb_src = dt / eps0; // vacuum cell
        let v = rsrc.apply_norton(&mut g, &feed, 2.0, dt, cb_src);
        assert!(v > 0.0 && v < 2.0, "V should charge toward Vs=2, got {v}");
        // Repeated application with no load current settles at Vs
        // (RC: tau = Rs * eps0 * A/dz = 443 steps at this dt).
        let mut v = v;
        for _ in 0..20000 {
            v = rsrc.apply_norton(&mut g, &feed, 2.0, dt, cb_src);
        }
        assert!((v - 2.0).abs() < 1e-3, "V should settle at Vs, got {v}");
    }

    #[test]
    fn gaussian_peaks_near_target() {
        // Numerical check: the differentiated Gaussian's spectrum should
        // peak near the requested frequency.
        let src = GaussianSource::new(2.5e9, 1.0);
        // Sample V(t) finely and DFT around the peak.
        let dt = 1e-12;
        let n = 20000;
        let mut vs = Vec::with_capacity(n);
        for m in 0..n {
            vs.push(src.value(m as f32 * dt as f32));
        }
        let mut probe = DftProbe::new(vec![2.0e9, 2.5e9, 3.0e9], dt as f32);
        for &v in &vs {
            probe.push(v, 0.0);
        }
        let (vf, _) = probe.dft();
        let mag: Vec<f64> = vf.iter().map(|(r, i)| r * r + i * i).collect();
        assert!(
            mag[1] > mag[0] && mag[1] > mag[2],
            "spectrum should peak at 2.5 GHz: {mag:?}"
        );
    }
}
