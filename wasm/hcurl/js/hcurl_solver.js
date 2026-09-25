/* hcurl_solver.js — loader shim for the hcurl-fem WebAssembly module.
 *
 * Adapts the raw WASM exports (`hcurl_solver_alloc/free/run`, `memory`) to
 * the Emscripten-style interface expected by
 * frontend/src/engine/wasm/worker-fem.ts:
 *
 *   const module = await createHcurlSolver({ locateFile });
 *   module.HEAPU8            // Uint8Array view of WASM linear memory
 *   module._hcurl_solver_alloc(size)
 *   module._hcurl_solver_free(ptr)
 *   module._hcurl_solver_set_cancelled(flag)
 *   module._hcurl_solver_run(reqPtr, reqLen, rspPtrOut, rspLenOut)
 *   module._hcurl_solver_last_error()   // last Rust panic message, "" if none
 *
 * The worker loads this file with (0, eval) and reads `createHcurlSolver`
 * off the global scope, so this script assigns it there.
 *
 * Build: `cargo build --release --target wasm32-unknown-unknown` produces
 * `target/wasm32-unknown-unknown/release/hcurl_fem.wasm`; copy it next to
 * this file as `hcurl_solver.wasm` (see ../README.md).
 */
(function (global) {
  "use strict";

  async function createHcurlSolver(options) {
    var opts = options || {};
    var locateFile =
      typeof opts.locateFile === "function" ? opts.locateFile : function (p) { return p; };
    var wasmUrl = locateFile("hcurl_solver.wasm");

    var response = await fetch(wasmUrl);
    if (!response.ok) {
      throw new Error("Failed to fetch " + wasmUrl + " (" + response.status + ")");
    }
    // ArrayBuffer instantiation: no application/wasm MIME requirement,
    // at the cost of one extra copy. Fine for a <2 MB module.
    var bytes = await response.arrayBuffer();
    var instantiated = await WebAssembly.instantiate(bytes, {});
    var instance = instantiated.instance;
    var memory = instance.exports.memory;
    var ex = instance.exports;

    for (var name of [
      "hcurl_solver_alloc",
      "hcurl_solver_free",
      "hcurl_solver_run",
    ]) {
      if (typeof ex[name] !== "function") {
        throw new Error("hcurl_solver.wasm is missing export " + name);
      }
    }

    var heapU8 = new Uint8Array(memory.buffer);
    function refreshHeap() {
      // memory.grow() detaches the old buffer; re-create the view lazily.
      if (heapU8.buffer !== memory.buffer) heapU8 = new Uint8Array(memory.buffer);
    }
    function callExport(name, a, b, c, d) {
      var result = ex[name](a, b, c, d);
      refreshHeap();
      return result;
    }

    var module = {
      get HEAPU8() {
        refreshHeap();
        return heapU8;
      },
      _hcurl_solver_alloc: function (size) {
        return callExport("hcurl_solver_alloc", size >>> 0);
      },
      _hcurl_solver_free: function (ptr) {
        callExport("hcurl_solver_free", ptr >>> 0);
      },
      _hcurl_solver_set_cancelled: function (flag) {
        callExport("hcurl_solver_set_cancelled", flag ? 1 : 0);
      },
      _hcurl_solver_run: function (reqPtr, reqLen, rspPtrOut, rspLenOut) {
        return callExport(
          "hcurl_solver_run",
          reqPtr >>> 0,
          reqLen >>> 0,
          rspPtrOut >>> 0,
          rspLenOut >>> 0,
        );
      },
      // Last Rust panic message captured by the in-WASM panic hook
      // (empty string when no panic has been captured yet).
      _hcurl_solver_last_error: function () {
        var len = callExport("hcurl_solver_last_error_len");
        if (len === 0) return "";
        var bufPtr = callExport("hcurl_solver_alloc", len);
        var copied = callExport("hcurl_solver_last_error_copy", bufPtr, len);
        var bytes = module.HEAPU8.slice(bufPtr, bufPtr + copied);
        callExport("hcurl_solver_free", bufPtr);
        return new TextDecoder().decode(bytes);
      },
    };
    return module;
  }

  global.createHcurlSolver = createHcurlSolver;
})(typeof self !== "undefined" ? self : this);
