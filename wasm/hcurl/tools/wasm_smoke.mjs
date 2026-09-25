// Node smoke test: drive the real WASM module through js/hcurl_solver.js.
// Usage:
//   cargo build --release --target wasm32-unknown-unknown
//   cp target/wasm32-unknown-unknown/release/hcurl_fem.wasm /tmp/hcurl_solver.wasm
//   node tools/wasm_smoke.mjs [/path/to/request.req]
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(root, "..", "js", "hcurl_solver.js"), "utf8");

const sandbox = {};
sandbox.self = sandbox; // shadow Node's `self` so the shim binds to the vm global
sandbox.TextDecoder = TextDecoder;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: "hcurl_solver.js" });
const createHcurlSolver = sandbox.createHcurlSolver;

const wasmBytes = fs.readFileSync("/tmp/hcurl_solver.wasm");
sandbox.fetch = async () => ({
  ok: true,
  arrayBuffer: async () =>
    wasmBytes.buffer.slice(
      wasmBytes.byteOffset,
      wasmBytes.byteOffset + wasmBytes.byteLength,
    ),
});

const module = await createHcurlSolver({ locateFile: (p) => p });
console.log("HEAPU8 bytes:", module.HEAPU8.length);

const reqPath =
  process.argv[2] ?? path.join(root, "..", "tests", "fixtures", "dipole_3ghz.req");
const req = fs.readFileSync(reqPath);
const t0 = Date.now();
const reqPtr = module._hcurl_solver_alloc(req.length);
module.HEAPU8.set(req, reqPtr);
const outPtr = module._hcurl_solver_alloc(8);
let status;
try {
  status = module._hcurl_solver_run(reqPtr, req.length, outPtr, outPtr + 4);
} catch (e) {
  console.log("RUN PANIC:", JSON.stringify(module._hcurl_solver_last_error()));
  process.exit(1);
}
const dv = new DataView(module.HEAPU8.buffer);
const rspPtr = dv.getUint32(outPtr, true);
const rspLen = dv.getUint32(outPtr + 4, true);
const rsp = module.HEAPU8.slice(rspPtr, rspPtr + rspLen);
module._hcurl_solver_free(reqPtr);
module._hcurl_solver_free(outPtr);
module._hcurl_solver_free(rspPtr);
console.log("run status:", status, "rsp bytes:", rspLen, "ms:", Date.now() - t0);

const rdv = new DataView(rsp.buffer, rsp.byteOffset, rsp.byteLength);
console.log("abi:", rdv.getUint32(0, true), "code:", rdv.getUint32(4, true));
console.log("zin:", rdv.getFloat64(32, true), rdv.getFloat64(40, true));
console.log("s11:", rdv.getFloat64(48, true), rdv.getFloat64(56, true));
console.log(
  "iters:",
  rdv.getUint32(24, true),
  "residual:",
  rdv.getFloat64(16, true),
);
console.log("blob len:", rdv.getUint32(92, true), "msg len:", rdv.getUint32(100, true));
