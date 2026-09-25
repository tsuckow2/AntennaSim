//! Error types for hcurl-fem.
//!
//! Every failure in the solve pipeline maps to a [`Error`] variant, which in
//! turn maps to a numeric status code on the packed response ABI:
//!
//! | Code | Variant | Meaning |
//! |------|---------|---------|
//! | 0 | — | ok |
//! | 1 | `InvalidRequest` / `UnsupportedAbi` / `EndiannessMismatch` | malformed request |
//! | 2 | `Mesh` | mesh validation failure |
//! | 3 | `Assembly` | matrix assembly failure |
//! | 6 | `Solver` / `Internal` | solver or internal failure |
//! | 7 | `Cancelled` | cooperative cancellation |
//! | 8 | `Timeout` | wall-clock budget exhausted |
//!
//! Codes 4–5 are reserved. The frontend maps any unlisted code to a generic
//! solver error and surfaces [`Error::message`] to the user.

use std::fmt;

/// Failure modes of the H(curl) solve pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The packed request failed validation (bad offsets, lengths, values…).
    InvalidRequest(String),
    /// Request ABI version is not supported by this build.
    UnsupportedAbi { expected: u32, found: u32 },
    /// Endianness sentinel mismatch (all multi-byte fields are little-endian).
    EndiannessMismatch,
    /// Mesh-level validation failure (degenerate tets, missing boundary…).
    Mesh(String),
    /// System assembly failure (basis, quadrature, ABC…).
    Assembly(String),
    /// The iterative solver did not converge.
    Solver {
        iterations: u32,
        residual: f64,
        message: String,
    },
    /// Cancelled via [`crate::ffi::hcurl_solver_set_cancelled`].
    Cancelled,
    /// The wall-clock solve budget was exhausted.
    Timeout,
    /// Near-to-far-field transform failure (no Huygens surface, bad power…).
    FarField(String),
    /// Anything else; always a bug, never user input.
    Internal(String),
}

impl Error {
    /// Numeric status code written into the packed response header.
    pub fn status_code(&self) -> u32 {
        match self {
            Error::InvalidRequest(_) | Error::UnsupportedAbi { .. } | Error::EndiannessMismatch => {
                1
            }
            Error::Mesh(_) => 2,
            Error::Assembly(_) => 3,
            Error::Solver { .. } | Error::FarField(_) | Error::Internal(_) => 6,
            Error::Cancelled => 7,
            Error::Timeout => 8,
        }
    }

    /// Human-readable detail, surfaced to the user via the response message.
    pub fn message(&self) -> String {
        match self {
            Error::InvalidRequest(detail) => format!("invalid request: {detail}"),
            Error::UnsupportedAbi { expected, found } => {
                format!("unsupported ABI version {found} (expected {expected})")
            }
            Error::EndiannessMismatch => "endianness sentinel mismatch".to_string(),
            Error::Mesh(detail) => format!("mesh error: {detail}"),
            Error::Assembly(detail) => format!("assembly error: {detail}"),
            Error::Solver {
                iterations,
                residual,
                message,
            } => {
                format!("solver failed after {iterations} iterations (residual {residual:.3e}): {message}")
            }
            Error::Cancelled => "cancelled".to_string(),
            Error::Timeout => "solve timed out".to_string(),
            Error::FarField(detail) => format!("far-field error: {detail}"),
            Error::Internal(detail) => format!("internal error: {detail}"),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for Error {}

/// Fallible result used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
