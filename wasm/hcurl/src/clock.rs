//! Monotonic stopwatch for solver phase timings.
//!
//! `std::time::Instant` panics on `wasm32-unknown-unknown`, so the WASM
//! build reports `0.0` for every phase and the host measures wall-clock
//! itself (the worker already times each point). Native builds report real
//! elapsed milliseconds.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::time::Instant;

    /// Real monotonic stopwatch (native targets).
    pub struct Stopwatch {
        start: Instant,
    }

    impl Stopwatch {
        /// Start measuring from now.
        pub fn start() -> Self {
            Self {
                start: Instant::now(),
            }
        }

        /// Milliseconds elapsed since [`Stopwatch::start`].
        pub fn elapsed_ms(&self) -> f64 {
            self.start.elapsed().as_secs_f64() * 1000.0
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod stub {
    /// Zero stopwatch: WASM has no `std::time::Instant`; every phase
    /// reports `0.0` ms and the host measures wall-clock instead.
    pub struct Stopwatch;

    impl Stopwatch {
        /// Start measuring (no-op on WASM).
        pub fn start() -> Self {
            Self
        }

        /// Always `0.0` on WASM.
        pub fn elapsed_ms(&self) -> f64 {
            0.0
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::Stopwatch;
#[cfg(target_arch = "wasm32")]
pub use stub::Stopwatch;
