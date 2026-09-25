export interface HcurlSmokeModule {
  _hcurl_smoke(): number;
}

/**
 * v2 native H(curl) solver module interface.
 *
 * Both the threaded wasm-bindgen build (`hcurl_solver_mt.js`, Rayon pool)
 * and the classic single-threaded build (`hcurl_solver.js`, Emscripten-style
 * raw exports wrapped by the loader) are adapted to this interface, so the
 * worker never manages WASM linear memory directly.
 */
export interface HcurlSolverModule {
  /** Solve one packed v1 point request; returns packed v1 response bytes. */
  solvePoint(requestBytes: Uint8Array): Uint8Array;
  /** Cooperative cancellation flag (best-effort; worker reset is the hard cancel). */
  setCancelled(flag: boolean): void;
  /** Last captured Rust panic message ("" when none). */
  lastError(): string;
  /** Rayon worker thread count (1 for the single-threaded module). */
  readonly threadCount: number;
}

/** Minimal typings for the wasm-pack `--target web` glue (public/wasm/hcurl_solver_mt.js). */
export interface HcurlMtPackage {
  /** wasm-bindgen init; pass the .wasm URL explicitly (the glue file is renamed on publish). */
  default: (wasmUrl: string) => Promise<unknown>;
  /** wasm-bindgen-rayon pool init; must resolve before the first solvePoint call. */
  initThreadPool: (threadCount: number) => Promise<unknown>;
  hcurlSolvePoint: (requestBytes: Uint8Array) => Uint8Array;
  hcurlSetCancelled: (flag: boolean) => void;
  hcurlLastError: () => string;
}
