/**
 * ScriptEditorPanel — script textarea, Run button, and inline diagnostics.
 *
 * The script source is kept in dslStore.  Running the script executes it in
 * a Web Worker.  Diagnostics from both the schema and model phases are shown
 * below the editor with their phase label.
 */

import { useCallback, useState } from "react";
import { Button } from "../ui/Button";
import { useDslStore } from "../../stores/dslStore";
import type { ScriptDiagnostic } from "../../dsl/executor";
import type { IRDiagnostic } from "../../dsl/model-ir";

function isScriptDiagnostic(d: ScriptDiagnostic | IRDiagnostic): d is ScriptDiagnostic {
  return "phase" in d;
}

export function ScriptEditorPanel() {
  const [femSweep, setFemSweep] = useState({ start: 900, stop: 936, steps: 5, mesh: 0.005, z0: 50, portId: "" });
  const script = useDslStore((s) => s.script);
  const setScript = useDslStore((s) => s.setScript);
  const status = useDslStore((s) => s.status);
  const result = useDslStore((s) => s.result);
  const runError = useDslStore((s) => s.runError);
  const runScript = useDslStore((s) => s.runScript);
  const runFEM = useDslStore((s) => s.runFEM);
  const cancelFEM = useDslStore((s) => s.cancelFEM);
  const femStatus = useDslStore((s) => s.femStatus);
  const femError = useDslStore((s) => s.femError);
  const hasModel = Boolean(result?.ir);
  const ports = result?.ir?.ports ?? [];

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => setScript(e.target.value),
    [setScript],
  );

  const handleRun = useCallback(() => {
    void runScript();
  }, [runScript]);

  const isRunning = status === "running";
  const isFemRunning = femStatus === "running";
  const diagnostics = result?.diagnostics ?? [];

  return (
    <div className="flex flex-col h-full">
      {/* Textarea */}
      <div className="flex-1 min-h-0 p-2">
        <textarea
          value={script}
          onChange={handleChange}
          spellCheck={false}
          className="w-full h-full resize-none rounded border border-border bg-background
            font-mono text-xs text-text-primary p-2 focus:outline-none
            focus:border-accent/50 leading-relaxed"
          aria-label="Script source"
        />
      </div>

      {/* Diagnostics */}
      {(diagnostics.length > 0 || runError) && (
        <div className="shrink-0 max-h-40 overflow-y-auto border-t border-border px-3 py-2 space-y-1">
          {runError && (
            <p className="text-[11px] font-mono text-swr-bad">
              <span className="font-semibold">worker:</span> {runError}
            </p>
          )}
          {diagnostics.map((d, i) => {
            if (isScriptDiagnostic(d)) {
              return (
                <p key={i} className="text-[11px] font-mono text-swr-bad">
                  <span className="text-text-secondary font-semibold">[{d.phase}]</span>{" "}
                  {d.message}
                </p>
              );
            }
            const color = d.severity === "error" ? "text-swr-bad" : "text-swr-warning";
            return (
              <p key={i} className={`text-[11px] font-mono ${color}`}>
                <span className="text-text-secondary font-semibold">[{d.severity}]</span>{" "}
                {d.path ? <span className="text-text-secondary">{d.path}: </span> : null}
                {d.message}
              </p>
            );
          })}
        </div>
      )}

      {/* Footer: status + Run button */}
      <div className="shrink-0 border-t border-border px-3 py-2 space-y-2">
        <div className="grid grid-cols-3 gap-1">
          <label className="text-[10px] text-text-secondary">Start MHz<input aria-label="FEM start frequency" type="number" value={femSweep.start} onChange={(event) => setFemSweep({ ...femSweep, start: Number(event.target.value) })} className="w-full bg-background border border-border rounded px-1 py-1 text-xs" /></label>
          <label className="text-[10px] text-text-secondary">Stop MHz<input aria-label="FEM stop frequency" type="number" value={femSweep.stop} onChange={(event) => setFemSweep({ ...femSweep, stop: Number(event.target.value) })} className="w-full bg-background border border-border rounded px-1 py-1 text-xs" /></label>
          <label className="text-[10px] text-text-secondary">Points<input aria-label="FEM sweep points" type="number" min={1} max={201} value={femSweep.steps} onChange={(event) => setFemSweep({ ...femSweep, steps: Number(event.target.value) })} className="w-full bg-background border border-border rounded px-1 py-1 text-xs" /></label>
          <label className="text-[10px] text-text-secondary">Mesh m<input aria-label="FEM mesh size" type="number" min={0.000001} step={0.0001} value={femSweep.mesh} onChange={(event) => setFemSweep({ ...femSweep, mesh: Number(event.target.value) })} className="w-full bg-background border border-border rounded px-1 py-1 text-xs" /></label>
          <label className="text-[10px] text-text-secondary">Z0 ohms<input aria-label="FEM reference impedance" type="number" min={0.001} step={1} value={femSweep.z0} onChange={(event) => setFemSweep({ ...femSweep, z0: Number(event.target.value) })} className="w-full bg-background border border-border rounded px-1 py-1 text-xs" /></label>
          <label className="text-[10px] text-text-secondary">Port<select aria-label="FEM port" value={femSweep.portId || ports[0]?.id || ""} onChange={(event) => setFemSweep({ ...femSweep, portId: event.target.value })} className="w-full bg-background border border-border rounded px-1 py-1 text-xs"><option value="" disabled>Select port</option>{ports.map((port) => <option key={port.id} value={port.id}>{port.id}</option>)}</select></label>
        </div>
        <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-text-secondary font-mono">
          {status === "idle" && "Ready"}
          {status === "running" && "Running…"}
          {status === "success" && (
            <span className="text-emerald-400">✓ Success</span>
          )}
          {status === "error" && (
            <span className="text-swr-bad">✗ Error</span>
          )}
        </span>
        <div className="flex gap-2">
          <Button size="sm" onClick={handleRun} loading={isRunning} disabled={isRunning}>Run</Button>
          {isFemRunning ? (
            <Button size="sm" variant="danger" onClick={cancelFEM}>Cancel FEM</Button>
          ) : (
            <Button size="sm" variant="secondary" disabled={!hasModel || isRunning || ports.length === 0} onClick={() => void runFEM({ start_mhz: femSweep.start, stop_mhz: femSweep.stop, steps: femSweep.steps, mesh_size_m: femSweep.mesh, reference_impedance_ohms: femSweep.z0, port_id: femSweep.portId || ports[0]?.id })}>Run FEM</Button>
          )}
        </div>
        </div>
        {femStatus === "cancelled" && <p className="text-[11px] text-text-secondary">FEM run cancelled; previous result is stale.</p>}
        {femError && <p className="text-[11px] text-swr-bad">{femError}</p>}
      </div>
    </div>
  );
}
