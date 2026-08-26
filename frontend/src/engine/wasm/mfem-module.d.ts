export interface MfemSmokeModule {
  _mfem_hcurl_smoke(): number;
}

export interface MfemSolverModule {
  HEAPU8: Uint8Array;
  _mfem_solver_alloc(size_bytes: number): number;
  _mfem_solver_free(ptr: number): void;
  _mfem_solver_set_cancelled?(flag: number): void;
  _mfem_solver_run(req_ptr: number, req_len: number, rsp_ptr_out: number, rsp_len_out: number): number;
}
