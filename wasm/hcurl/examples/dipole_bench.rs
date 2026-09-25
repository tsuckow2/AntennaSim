//! Native benchmark: full dipole `solve_point`, timed.
//!
//! ```sh
//! cargo run --release --example dipole_bench
//! RAYON_NUM_THREADS=1 cargo run --release --example dipole_bench
//! ```
//!
//! The first form uses one Rayon thread per logical core; the second pins
//! the pool to a single thread, which approximates the pre-Rayon serial
//! code (same kernels, one worker).

use hcurl_fem::abi::PointRequest;
use hcurl_fem::budget::Budget;
use std::time::Instant;

fn main() {
    let bytes = include_bytes!("../tests/fixtures/dipole_3ghz.req");
    let request = PointRequest::parse(bytes).expect("fixture must parse");

    let t0 = Instant::now();
    let response =
        hcurl_fem::solve_point(&request, &Budget::none()).expect("solve must not error");
    let dt = t0.elapsed();
    assert_eq!(response.status, 0, "message: {:?}", response.message);

    let zin = response.zin.expect("Zin present");
    println!("threads   : {}", rayon::current_num_threads());
    println!("Zin       : {zin:.6} Ω");
    println!("iterations: {}", response.iteration_count.unwrap());
    println!("residual  : {:.3e}", response.residual_norm.unwrap());
    println!("wall time : {dt:.2?}");
}
