//! Holland/Simpson thin-wire subcell model for a z-directed wire.
//!
//! Unlike the simple coefficient-scaling approach (which treats the wire as
//! a PEC cell with modified H-field updates), the Holland/Simpson model
//! represents the wire as an auxiliary 1D transmission-line system coupled
//! bidirectionally to the 3D FDTD grid:
//!
//! 1. Sample the E-field along the wire to get the electromotive force (EMF)
//!    on each segment.
//! 2. Advance the wire current using the telegrapher's equation:
//!    `I^{n+1/2} = I^{n-1/2} + dt/L_seg * (EMF + V_tail - V_head)`
//! 3. Deposit the wire current back into the E-field as a current source:
//!    `Ez -= dt * I / (eps0 * dx * dy)`
//! 4. Update the node charges via continuity:
//!    `q^{n+1} = q^n - dt * (I_out - I_in)`
//!
//! The wire is NOT a PEC cell. The Ez field at the wire location evolves
//! normally, with the wire current deposited as a source term.
//!
//! Per-unit-length parameters (from the wire kernel):
//! - `L' = mu0 * ln(R/a) / (2*pi)` (inductance per unit length)
//! - `C' = mu0 * eps0 / L'` (capacitance per unit length)
//! where `R` is the coupling distance (≈ 0.23*dx for a Yee cell) and `a`
//! is the physical wire radius.
//!
//! Reference: Holland & Simpson, "Finite-difference analysis of EMP
//! coupling to thin struts and wires," IEEE Trans. EMC, 1981.

use crate::grid::YeeGrid;

/// Physical constants.
const MU0: f32 = 1.25663706212e-6;
const EPS0: f32 = 8.8541878128e-12;

/// Thin-wire model for a z-directed wire using the Holland/Simpson formulation.
pub struct ThinWire {
    /// Ez index of the wire (i0, j0).
    pub i0: usize,
    /// Ez index of the wire (i0, j0).
    pub j0: usize,
    /// First k-index of the wire (inclusive).
    pub k_lo: usize,
    /// Last k-index of the wire (inclusive).
    pub k_hi: usize,
    /// Physical wire radius in meters.
    pub radius: f32,
    /// Number of wire segments (one per Ez cell).
    pub n_segments: usize,
    /// Number of wire nodes (segments + 1).
    pub n_nodes: usize,
    /// Segment inductance (L' * dz) in Henries.
    pub seg_inductance: f32,
    /// Node capacitance (C' * dz) in Farads (for interior nodes).
    pub node_capacitance: f32,
    /// Wire currents at half-step (one per segment), in Amperes.
    /// Positive = flowing in +z (upward).
    pub currents: Vec<f32>,
    /// Node charges at integer step (one per node), in Coulombs.
    pub charges: Vec<f32>,
    /// Cell sizes (for deposition).
    dx: f32,
    dy: f32,
    dz: f32,
    /// If true, the top node is tied to 0V (PEC termination, e.g., patch).
    /// If false, the top is open (current = 0 at the end).
    pub top_pec: bool,
}

impl ThinWire {
    /// Create a Holland/Simpson thin-wire model.
    ///
    /// * `i0, j0`: Ez indices of the wire axis.
    /// * `k_lo, k_hi`: inclusive k-range of the wire (one segment per k).
    /// * `dx, dy, dz`: cell sizes in meters.
    /// * `radius`: physical wire radius in meters.
    /// * `top_pec`: if true, the top node voltage is fixed at 0 (wire
    ///   terminates on a PEC like a patch); if false, the top is open.
    pub fn new(
        i0: usize,
        j0: usize,
        k_lo: usize,
        k_hi: usize,
        dx: f32,
        dy: f32,
        dz: f32,
        radius: f32,
        top_pec: bool,
    ) -> Self {
        assert!(k_hi >= k_lo, "wire must span at least one cell");
        let n_segments = k_hi - k_lo + 1;
        let n_nodes = n_segments + 1;

        // Coupling distance: equivalent radius of the Yee cell.
        // For a square cell, R ≈ 0.23 * dx (Holland & Simpson).
        let r_coupling = 0.23 * dx;
        assert!(
            radius < r_coupling,
            "wire radius must be below coupling distance"
        );

        // Per-unit-length parameters.
        let l_prime = MU0 * (r_coupling / radius).ln() / (2.0 * std::f32::consts::PI);
        let c_prime = MU0 * EPS0 / l_prime;

        // Length-integrated coefficients.
        let seg_inductance = l_prime * dz;
        let node_capacitance = c_prime * dz;

        Self {
            i0,
            j0,
            k_lo,
            k_hi,
            radius,
            n_segments,
            n_nodes,
            seg_inductance,
            node_capacitance,
            currents: vec![0.0; n_segments],
            charges: vec![0.0; n_nodes],
            dx,
            dy,
            dz,
            top_pec,
        }
    }

    /// Node voltage in Volts. The top node is 0V if `top_pec`.
    fn node_voltage(&self, n: usize) -> f32 {
        if self.top_pec && n == self.n_nodes - 1 {
            0.0
        } else {
            self.charges[n] / self.node_capacitance
        }
    }

    /// Advance the wire current by one step (half-step update).
    ///
    /// Samples EMF from `grid.ez` (at integer time step), updates `currents`
    /// to the next half-step, and updates `charges` via continuity.
    /// Does NOT modify the grid — the caller must deposit the returned
    /// current into Ez as part of the E-field update.
    ///
    /// Returns the bottom-segment current (terminal current, +z upward).
    pub fn update_current(&mut self, grid: &YeeGrid, dt: f32) -> f32 {
        let dt_over_l = dt / self.seg_inductance;

        // 1. Sample EMF and update currents.
        //    I^{n+1/2} = I^{n-1/2} + dt/L * (EMF + V_tail - V_head)
        for s in 0..self.n_segments {
            let k = self.k_lo + s;
            let ez_idx = grid.ez_idx(self.i0, self.j0, k);
            let emf = grid.ez[ez_idx] * self.dz;

            let v_tail = self.node_voltage(s);
            let v_head = self.node_voltage(s + 1);

            self.currents[s] += dt_over_l * (emf + v_tail - v_head);
        }

        // 2. Update charges via continuity.
        //    q^{n+1} = q^n - dt * (I_out - I_in)
        let mut dq = vec![0.0f32; self.n_nodes];
        for n in 0..self.n_nodes {
            let i_out = if n < self.n_segments {
                self.currents[n]
            } else {
                0.0
            };
            let i_in = if n > 0 {
                self.currents[n - 1]
            } else {
                0.0
            };
            dq[n] = -dt * (i_out - i_in);
        }
        // Don't accumulate charge on the PEC-terminated top node.
        if self.top_pec {
            dq[self.n_nodes - 1] = 0.0;
            self.charges[self.n_nodes - 1] = 0.0;
        }
        for n in 0..self.n_nodes {
            self.charges[n] += dq[n];
        }

        // Return the bottom-segment current (terminal current).
        self.currents[0]
    }

    /// Deposit the wire current into the Ez field as a current source.
    ///
    /// Must be called after the FDTD E-update, as part of computing E^{n+1}:
    /// `Ez -= dt * I / (eps0 * dx * dy)` for each wire segment.
    pub fn deposit_current(&self, grid: &mut YeeGrid, dt: f32) {
        let deposit_coeff = dt / (EPS0 * self.dx * self.dy);
        for s in 0..self.n_segments {
            let k = self.k_lo + s;
            let ez_idx = grid.ez_idx(self.i0, self.j0, k);
            grid.ez[ez_idx] -= deposit_coeff * self.currents[s];
        }
    }

    /// Advance the wire state by one step, coupled to the E-field.
    ///
    /// DEPRECATED: Use `update_current` + `deposit_current` separately to
    /// ensure the deposition happens at the correct point in the leapfrog
    /// sequence (after the FDTD E-update, not before).
    pub fn step(&mut self, grid: &mut YeeGrid, dt: f32) -> f32 {
        let i = self.update_current(grid, dt);
        self.deposit_current(grid, dt);
        i
    }

    /// Maximum stable timestep for the wire subsystem alone.
    /// From the reference: dt_max = 2 / omega_max, where omega_max is the
    /// highest eigenfrequency of the coupled wire operator.
    /// For a uniform wire, this is approximately dz / v_wire, where
    /// v_wire = 1/sqrt(L'C') is the wave speed on the wire.
    pub fn max_stable_dt(&self) -> f32 {
        let l_prime = self.seg_inductance / self.dz;
        let c_prime = self.node_capacitance / self.dz;
        let v_wire = 1.0 / (l_prime * c_prime).sqrt();
        // CFL for the 1D wave equation: dt < dz / v.
        // Use a safety factor of 0.9.
        0.9 * self.dz / v_wire
    }

    /// Total wire energy (for validation): 0.5*L*I^2 + 0.5*q^2/C.
    pub fn energy(&self) -> f32 {
        let mut e = 0.0;
        for &i in &self.currents {
            e += 0.5 * self.seg_inductance * i * i;
        }
        for n in 0..self.n_nodes {
            if !(self.top_pec && n == self.n_nodes - 1) {
                e += 0.5 * self.charges[n] * self.charges[n] / self.node_capacitance;
            }
        }
        e
    }

    /// Reset the wire state (zero currents and charges).
    pub fn reset(&mut self) {
        for i in &mut self.currents {
            *i = 0.0;
        }
        for q in &mut self.charges {
            *q = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_parameters_positive() {
        let tw = ThinWire::new(10, 10, 5, 8, 0.5e-3, 0.5e-3, 0.525e-3, 0.1e-3, true);
        assert!(tw.seg_inductance > 0.0);
        assert!(tw.node_capacitance > 0.0);
        assert_eq!(tw.n_segments, 4);
        assert_eq!(tw.n_nodes, 5);
    }

    #[test]
    fn wire_wave_speed_near_c() {
        // The wave speed on the wire should be close to c (it's in vacuum).
        let tw = ThinWire::new(10, 10, 5, 8, 0.5e-3, 0.5e-3, 0.525e-3, 0.1e-3, true);
        let l_prime = tw.seg_inductance / 0.525e-3;
        let c_prime = tw.node_capacitance / 0.525e-3;
        let v = 1.0 / (l_prime * c_prime).sqrt();
        let c = 1.0 / (MU0 * EPS0).sqrt();
        // Should be within 1% of c (the model is quasi-static, not exact).
        assert!((v - c).abs() / c < 0.01, "v={} c={}", v, c);
    }

    #[test]
    fn top_pec_voltage_zero() {
        let mut tw = ThinWire::new(10, 10, 5, 6, 0.5e-3, 0.5e-3, 0.525e-3, 0.1e-3, true);
        tw.charges[2] = 1e-12; // Try to put charge on top node
        assert_eq!(tw.node_voltage(2), 0.0);
        // Interior node has non-zero voltage
        tw.charges[1] = 1e-12;
        assert!(tw.node_voltage(1) != 0.0);
    }

    #[test]
    fn max_stable_dt_positive() {
        let tw = ThinWire::new(10, 10, 5, 8, 0.5e-3, 0.5e-3, 0.525e-3, 0.1e-3, true);
        let dt_max = tw.max_stable_dt();
        assert!(dt_max > 0.0);
        // Should be on the order of dz/c ≈ 1.75ps
        assert!(dt_max < 10e-12 && dt_max > 0.1e-12, "dt_max={}", dt_max);
    }
}
