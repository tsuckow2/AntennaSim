/**
 * ScriptEditorPanel — script textarea, Run button, and inline diagnostics.
 *
 * The script source is kept in dslStore.  Running the script executes it in
 * a Web Worker.  Diagnostics from both the schema and model phases are shown
 * below the editor with their phase label.
 */

import { useCallback } from "react";
import { Button } from "../ui/Button";
import { useDslStore } from "../../stores/dslStore";
import type { ScriptDiagnostic } from "../../dsl/executor";
import type { IRDiagnostic } from "../../dsl/model-ir";

function isScriptDiagnostic(d: ScriptDiagnostic | IRDiagnostic): d is ScriptDiagnostic {
  return "phase" in d;
}

export function ScriptEditorPanel() {
  const script = useDslStore((s) => s.script);
  const setScript = useDslStore((s) => s.setScript);
  const status = useDslStore((s) => s.status);
  const result = useDslStore((s) => s.result);
  const runError = useDslStore((s) => s.runError);
  const runScript = useDslStore((s) => s.runScript);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => setScript(e.target.value),
    [setScript],
  );

  const handleRun = useCallback(() => {
    void runScript();
  }, [runScript]);

  const isRunning = status === "running";
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
      <div className="shrink-0 flex items-center justify-between gap-2 px-3 py-2 border-t border-border">
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
        <Button size="sm" onClick={handleRun} loading={isRunning} disabled={isRunning}>
          Run
        </Button>
      </div>
    </div>
  );
}
