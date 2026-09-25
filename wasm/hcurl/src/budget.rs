//! Cooperative cancellation and wall-clock budgets.
//!
//! Long-running phases (assembly, the BiCGSTAB loop) poll a [`Budget`].
//! Cancellation is requested from the host through
//! [`crate::ffi::hcurl_solver_set_cancelled`]; the timeout is fixed per
//! [`solve_point`](crate::solve_point) invocation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};

/// Backing flag for [`Budget::none`]: never set.
static NEVER_CANCELLED: AtomicBool = AtomicBool::new(false);

/// Pollable cancellation flag + optional deadline.
#[derive(Debug)]
pub struct Budget {
    cancelled: &'static AtomicBool,
    deadline: Option<Instant>,
}

impl Budget {
    /// Create a budget backed by `cancelled`, expiring `timeout` after creation.
    pub fn new(cancelled: &'static AtomicBool, timeout: Option<Duration>) -> Self {
        Self {
            cancelled,
            deadline: timeout.map(|d| Instant::now() + d),
        }
    }

    /// A budget that never cancels and never times out. For tests and
    /// contexts where cancellation is handled elsewhere.
    pub fn none() -> Self {
        Self {
            cancelled: &NEVER_CANCELLED,
            deadline: None,
        }
    }

    /// Returns [`Error::Cancelled`] if cancellation was requested, or
    /// [`Error::Timeout`] if the deadline passed. Call from inner loops at a
    /// granularity that keeps UI cancellation responsive without dominating
    /// runtime (e.g. once per solver iteration, once per few hundred elements).
    pub fn check(&self) -> Result<()> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(Error::Cancelled);
        }
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(Error::Timeout);
            }
        }
        Ok(())
    }
}
