//! `extern "C"` exports for the WebAssembly module.
//!
//! The frontend worker (`frontend/src/engine/wasm/worker-fem.ts`) drives the
//! module through four functions, bridged by `js/hcurl_solver.js`:
//!
//! ```text
//! ptr = hcurl_solver_alloc(size_bytes)
//! hcurl_solver_free(ptr)
//! hcurl_solver_set_cancelled(0 | 1)
//! status = hcurl_solver_run(req_ptr, req_len, rsp_ptr_out, rsp_len_out)
//! ```
//!
//! `hcurl_solver_run` parses the packed v1 request, runs
//! [`crate::solve_point`], and writes the serialized response into freshly
//! allocated WASM memory, storing its pointer/length into `rsp_ptr_out` /
//! `rsp_len_out`. The host copies the response out and frees it with
//! `hcurl_solver_free`.
//!
//! All allocation goes through a size-prefixed [`std::alloc`] scheme so
//! `free` needs no out-of-band bookkeeping. Panics are caught at the
//! boundary and converted into error responses — a panic must never unwind
//! across the FFI.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use crate::abi::{PointRequest, PointResponse};
use crate::budget::Budget;
use crate::error::{Error, Result};

/// Set by [`hcurl_solver_set_cancelled`]; polled by [`Budget`].
pub(crate) static CANCELLED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Panic-message capture.
//
// `catch_unwind` cannot intercept panics on `wasm32-unknown-unknown`
// (panic=abort), so the message would otherwise be lost. The hook below
// saves the last panic message into a fixed buffer the host can read back
// with `hcurl_solver_last_error_{len,copy}` after a failed `run`.
// Single-threaded WASM: no locking needed beyond the atomic length.
// ---------------------------------------------------------------------------

const LAST_ERROR_CAP: usize = 1024;
static mut LAST_ERROR_BUF: [u8; LAST_ERROR_CAP] = [0; LAST_ERROR_CAP];
static LAST_ERROR_LEN: AtomicUsize = AtomicUsize::new(0);

fn install_panic_hook() {
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| {
                info.payload()
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
            })
            .unwrap_or("<non-string panic payload>");
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".to_string());
        let message = format!("{location}: {payload}");
        let bytes = message.as_bytes();
        let n = bytes.len().min(LAST_ERROR_CAP);
        unsafe {
            LAST_ERROR_BUF[..n].copy_from_slice(&bytes[..n]);
        }
        LAST_ERROR_LEN.store(n, Ordering::SeqCst);
        previous(info);
    }));
}

/// Byte length of the last captured panic message (0 if none).
#[no_mangle]
pub extern "C" fn hcurl_solver_last_error_len() -> usize {
    LAST_ERROR_LEN.load(Ordering::SeqCst)
}

/// Copy up to `cap` bytes of the last panic message into `dst`; returns
/// bytes copied. Host allocates `dst` (e.g. via `hcurl_solver_alloc`).
///
/// # Safety
/// `dst` must point to `cap` writable bytes.
#[no_mangle]
pub extern "C" fn hcurl_solver_last_error_copy(dst: *mut u8, cap: usize) -> usize {
    if dst.is_null() {
        return 0;
    }
    let n = LAST_ERROR_LEN.load(Ordering::SeqCst).min(cap);
    unsafe {
        std::ptr::copy_nonoverlapping(std::ptr::addr_of!(LAST_ERROR_BUF).cast::<u8>(), dst, n);
    }
    n
}

/// Wall-clock budget per frequency point, mirroring the worker watchdog.
/// Wall-clock cap for one `run` call. `None` on WASM: `std::time::Instant`
/// panics on `wasm32-unknown-unknown`, and the host worker already enforces
/// its own 15-minute watchdog with progress resets; in-WASM cancellation
/// polling via [`hcurl_solver_set_cancelled`] stays live on all platforms.
#[cfg(not(target_arch = "wasm32"))]
pub const SOLVE_TIMEOUT: Option<Duration> = Some(Duration::from_secs(15 * 60));
#[cfg(target_arch = "wasm32")]
pub const SOLVE_TIMEOUT: Option<Duration> = None;

/// Allocate `size` bytes in WASM linear memory; returns null on OOM.
#[no_mangle]
pub extern "C" fn hcurl_solver_alloc(size: usize) -> *mut u8 {
    let total = match size.checked_add(8) {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    let layout = match std::alloc::Layout::from_size_align(total, 8) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };
    let raw = unsafe { std::alloc::alloc(layout) };
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        (raw as *mut u64).write(total as u64);
        raw.add(8)
    }
}

/// Free a pointer previously returned by [`hcurl_solver_alloc`].
///
/// # Safety
/// `ptr` must be a live pointer from `hcurl_solver_alloc` (or null).
#[no_mangle]
pub extern "C" fn hcurl_solver_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let raw = ptr.sub(8);
        let total = (raw as *const u64).read() as usize;
        if let Ok(layout) = std::alloc::Layout::from_size_align(total, 8) {
            std::alloc::dealloc(raw, layout);
        }
    }
}

/// Request cooperative cancellation (`flag != 0`) or clear it (`flag == 0`).
#[no_mangle]
pub extern "C" fn hcurl_solver_set_cancelled(flag: i32) {
    CANCELLED.store(flag != 0, Ordering::SeqCst);
}

fn run_point(req_bytes: &[u8]) -> Result<PointResponse> {
    let budget = Budget::new(&CANCELLED, SOLVE_TIMEOUT);
    let request = PointRequest::parse(req_bytes)?;
    crate::solve_point(&request, &budget)
}

/// Solve one packed request.
///
/// Returns the numeric status code (also present in the response header).
/// On success `*rsp_ptr_out`/`*rsp_len_out` describe a freshly allocated
/// response buffer the host must free with [`hcurl_solver_free`].
///
/// # Safety
/// `req_ptr` must point to `req_len` readable bytes; `rsp_ptr_out` and
/// `rsp_len_out` must be writable `u32` slots.
#[no_mangle]
pub extern "C" fn hcurl_solver_run(
    req_ptr: *const u8,
    req_len: usize,
    rsp_ptr_out: *mut u32,
    rsp_len_out: *mut u32,
) -> i32 {
    // A fresh point starts un-cancelled; the host sets the flag mid-solve.
    CANCELLED.store(false, Ordering::SeqCst);
    install_panic_hook();

    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if req_ptr.is_null() || rsp_ptr_out.is_null() || rsp_len_out.is_null() {
            return Err(Error::InvalidRequest("null pointer argument".to_string()));
        }
        let req_bytes = unsafe { std::slice::from_raw_parts(req_ptr, req_len) };
        run_point(req_bytes)
    }));

    let response = match outcome {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => PointResponse::error(&e),
        Err(_) => PointResponse::error(&Error::Internal(
            "solver panicked; see console for details".to_string(),
        )),
    };
    let status = response.status as i32;

    let bytes = response.serialize();
    let ptr = hcurl_solver_alloc(bytes.len());
    if ptr.is_null() {
        return -2; // host-visible OOM marker; no response buffer
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        rsp_ptr_out.write(ptr as u32);
        rsp_len_out.write(bytes.len() as u32);
    }
    status
}
