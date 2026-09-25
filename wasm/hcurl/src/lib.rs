//! # hcurl-fem
//!
//! An H(curl)-conforming finite-element solver for time-harmonic Maxwell
//! equations, built on first-order Nédélec edge elements.
//!
//! The formulation solves the vector wave equation
//!
//! ```text
//! curl (mu^-1 curl E) - omega^2 epsilon_c E = -j omega J    in Omega
//! ```
//!
//! with a first-order Silver–Müller absorbing boundary condition on the
//! exterior boundary, a 1 A impressed-current lumped port, and an optional
//! Stratton–Chu near-to-far-field transform for gain patterns.
//!
//! ## Design
//!
//! The crate is a pure-Rust library with **no WebAssembly-specific code in
//! the numerical core**. The WASM interface is a thin FFI layer ([`ffi`])
//! over the same [`solve_point`] entry point used natively, so every
//! algorithm is validated with ordinary `cargo test` runs and the WASM build
//! is parity-by-construction.
//!
//! Time-harmonic convention: fields vary as `e^{jωt}`.
//!
//! ## Modules
//!
//! - [`constants`] — physical constants (SI, `e^{+jωt}`)
//! - [`mesh`] — validated mesh + sanitized materials
//! - [`edge`] — global edge enumeration, PEC elimination
//! - [`geometry`] — tetrahedral Jacobian, barycentric gradients
//! - [`basis`] — first-order Nédélec basis, analytic element brackets
//! - [`assembly`] — curl-curl / mass / ABC assembly
//! - [`krylov`] — complex BiCGSTAB + symmetric Gauss–Seidel
//! - [`port`] — gap source, V/I extraction, S11
//! - [`farfield`] — Stratton–Chu near-to-far-field transform
//! - [`par`] — data-parallel loops (Rayon natively, sequential on wasm32)
//! - [`solve`] — end-to-end single-point orchestration
//! - [`csr`] — complex compressed-sparse-row matrix
//! - [`abi`] — packed binary request/response wire format (v1)
//! - [`ffi`] — `extern "C"` exports for the WASM module
//! - [`budget`] — cooperative cancellation + wall-clock timeout
//! - [`error`] — error types
//!
//! ## Quick start
//!
//! ```no_run
//! use hcurl_fem::abi::PointRequest;
//! use hcurl_fem::budget::Budget;
//!
//! # fn main() -> hcurl_fem::Result<()> {
//! let request_bytes: &[u8] = todo!("packed v1 request, see abi module docs");
//! let request = PointRequest::parse(request_bytes)?;
//! let response = hcurl_fem::solve_point(&request, &Budget::none())?;
//! let wire_bytes = response.serialize();
//! # Ok(())
//! # }
//! ```

pub mod abc;
pub mod abi;
pub mod assembly;
pub mod basis;
pub mod budget;
pub mod clock;
pub mod constants;
pub mod csr;
pub mod edge;
pub mod error;
pub mod farfield;
pub mod ffi;
pub mod geometry;
pub mod krylov;
pub mod mesh;
pub mod par;
pub mod port;
pub mod solve;

pub use abi::{PointRequest, PointResponse};
pub use csr::CsrMatrix;
pub use error::{Error, Result};

/// Solve one frequency point end to end.
///
/// This is the single entry point shared by the native library API and the
/// WASM FFI. It parses nothing and allocates nothing on the FFI boundary:
/// it takes a validated [`PointRequest`] and a [`Budget`], and returns a
/// [`PointResponse`] ready for [`PointResponse::serialize`].
pub use solve::solve_point;
