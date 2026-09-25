//! `#[wasm_bindgen]` API surface for the threaded browser build.
//!
//! Compiled only for `wasm32` with the `wasm-threads` feature. Unlike the
//! raw `extern "C"` FFI in [`crate::ffi`], these wrappers copy byte buffers
//! across the boundary, so the host never manages WASM linear memory:
//!
//! ```js
//! import init, { initThreadPool, hcurlSolvePoint } from './hcurl_solver_mt.js';
//! await init(wasmUrl);                                // wasm-bindgen init
//! await initThreadPool(navigator.hardwareConcurrency); // Rayon pool
//! const responseBytes = hcurlSolvePoint(requestBytes);  // packed ABI v1
//! ```
//!
//! `initThreadPool` itself is re-exported from `wasm-bindgen-rayon` at the
//! crate root (see `lib.rs`); it must resolve before the first
//! [`solve_point_bytes`] call so Rayon's global pool is the worker-backed
//! one instead of the (unusable on wasm) default.

use std::sync::atomic::Ordering;

use wasm_bindgen::prelude::*;

use crate::abi::PointResponse;
use crate::ffi::{self, CANCELLED};

/// Solve one packed v1 point request; returns the packed v1 response.
///
/// Mirrors [`crate::ffi::hcurl_solver_run`] but moves buffers by value
/// instead of raw pointers. Graceful failures (bad request, solver error)
/// serialize into an error response. A Rust panic aborts the module
/// (panic=abort on wasm) after the panic hook captures the message for
/// [`last_error`].
#[wasm_bindgen(js_name = hcurlSolvePoint)]
pub fn solve_point_bytes(req_bytes: &[u8]) -> Vec<u8> {
    CANCELLED.store(false, Ordering::SeqCst);
    ffi::install_panic_hook();

    let response = match ffi::run_point(req_bytes) {
        Ok(resp) => resp,
        Err(e) => PointResponse::error(&e),
    };
    response.serialize()
}

/// Request cooperative cancellation (`true`) or clear it (`false`).
///
/// Polled by the solver [`Budget`][crate::budget::Budget] between BiCGSTAB
/// iterations. Best-effort: the frontend currently cancels by terminating
/// the worker, which also tears down the thread pool.
#[wasm_bindgen(js_name = hcurlSetCancelled)]
pub fn set_cancelled(flag: bool) {
    CANCELLED.store(flag, Ordering::SeqCst);
}

/// Last captured Rust panic message (`""` when none captured yet).
#[wasm_bindgen(js_name = hcurlLastError)]
pub fn last_error() -> String {
    let len = ffi::hcurl_solver_last_error_len();
    if len == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; len];
    let copied = ffi::hcurl_solver_last_error_copy(buf.as_mut_ptr(), len);
    buf.truncate(copied);
    String::from_utf8_lossy(&buf).into_owned()
}
