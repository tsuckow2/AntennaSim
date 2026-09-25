export interface HcurlSmokeModule {
  _hcurl_smoke(): number;
}

export interface HcurlSolverModule {
  HEAPU8: Uint8Array;
  _hcurl_solver_alloc(size_bytes: number): number;
  _hcurl_solver_free(ptr: number): void;
  _hcurl_solver_set_cancelled?(flag: number): void;
  _hcurl_solver_run(req_ptr: number, req_len: number, rsp_ptr_out: number, rsp_len_out: number): number;
  /** Last Rust panic message captured by the in-WASM hook ("" when none). */
  _hcurl_solver_last_error(): string;
}
