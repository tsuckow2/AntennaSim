//! Test the ThinWire in isolation (no FDTD grid coupling).
//! Drives the wire with a prescribed EMF and checks for stability.

use fdtd::thin_wire::ThinWire;
use fdtd::grid::YeeGrid;
use fdtd::material::Materials;

fn main() {
    let nx = 10;
    let ny = 10;
    let nz = 30;
    let dx = 1.0e-3;
    let dy = 1.0e-3;
    let dz = 1.0e-3;

    let mut grid = YeeGrid::new(nx, ny, nz, dx, dy, dz);
    
    // Wire: 20 segments, open at both ends.
    let mut wire = ThinWire::new(5, 5, 5, 24, dx, dy, dz, 0.1e-3, false);
    
    let c0 = 299792458.0;
    let dt = 0.5 * dz / c0;
    let mat = Materials::vacuum(nx, ny, nz, dt);
    println!("dt = {:.3e}, wire max dt = {:.3e}", dt, wire.max_stable_dt());
    
    // Prescribed EMF: Gaussian pulse on the middle segment.
    let t0 = 1.0e-9;
    let sigma = 0.2e-9;
    
    for step in 0..2000 {
        let t = step as f32 * dt;
        
        // Set EMF by directly setting Ez (wire samples it).
        // Clear first.
        for k in 5..=24 {
            let idx = grid.ez_idx(5, 5, k);
            grid.ez[idx] = 0.0;
        }
        // Drive middle segment (k=14) with Gaussian.
        let vs = (-((t - t0).powi(2)) / (2.0 * sigma * sigma)).exp();
        let k_drive = 14;
        let idx = grid.ez_idx(5, 5, k_drive);
        grid.ez[idx] = vs / dz; // EMF = Ez*dz = Vs
        
        // Wire step (no deposition needed for isolation test, but do it).
        let _i = wire.update_current(&grid, dt);
        wire.deposit_current(&mut grid, &mat);
        
        if step % 200 == 0 {
            let max_i = wire.currents.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let max_q = wire.charges.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            println!("step {}: max|I|={:.3e}, max|q|={:.3e}, E={:.3e}",
                     step, max_i, max_q, wire.energy());
            if !max_i.is_finite() {
                println!("UNSTABLE at step {}", step);
                return;
            }
        }
    }
    println!("Wire alone is STABLE for 2000 steps.");
}
