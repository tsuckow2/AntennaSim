//! Native point-request solver: reads one packed `.req` file, solves, prints one
//! machine-readable result line.
//!
//! ```sh
//! cargo run --release --example solve_req_file -- /tmp/snyder_sweep/L2/2400.req
//! RAYON_NUM_THREADS=2 cargo run --release --example solve_req_file -- path.req
//! ```
//!
//! Output (single line):
//! `freq_mhz=2400 status=0 zin_re=.. zin_im=.. s11_db=.. iters=.. residual=..e-.. wall_s=..`

use hcurl_fem::abi::PointRequest;
use hcurl_fem::budget::Budget;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: solve_req_file <path.req>");
    let bytes = std::fs::read(&path).expect("request file must be readable");
    let request = PointRequest::parse(&bytes).expect("request must parse");

    let stem = std::path::Path::new(&path)
        .file_stem()
        .expect("file stem")
        .to_string_lossy()
        .to_string();

    let t0 = Instant::now();
    let response = hcurl_fem::solve_point(&request, &Budget::none()).expect("solve must not error");
    let wall_s = t0.elapsed().as_secs_f64();

    let zin = response.zin.map(|z| (z.re, z.im)).unwrap_or((f64::NAN, f64::NAN));
    let s11_db = response
        .s11
        .map(|s| 20.0 * s.norm().log10())
        .unwrap_or(f64::NAN);
    println!(
        "freq_mhz={} status={} zin_re={:.4} zin_im={:.4} s11_db={:.2} iters={} residual={:.3e} wall_s={:.1}",
        stem,
        response.status,
        zin.0,
        zin.1,
        s11_db,
        response.iteration_count.unwrap_or(0),
        response.residual_norm.unwrap_or(f64::NAN),
        wall_s,
    );
    if let Some(msg) = response.message {
        println!("message: {msg}");
    }
}
