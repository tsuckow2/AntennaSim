/**
 * ParametricEditorPage — QuickJS-WASM parametric script editor.
 *
 * Desktop layout (3-panel):
 *   [Left: Script editor + diagnostics] [Center: 3D ModelIR preview] [Right: Param form]
 *
 * Mobile layout:
 *   [3D preview (flex-1)] [Bottom sheet: Script | Params | Diagnostics tabs]
 */

import { useCallback, useEffect, useReducer, useRef } from "react";
import { Navbar } from "../components/layout/Navbar";
import { ErrorBoundary } from "../components/common/ErrorBoundary";
import { ScriptEditorPanel } from "../components/dsl/ScriptEditorPanel";
import { ParamSchemaForm } from "../components/dsl/ParamSchemaForm";
import { ModelIRPreview } from "../components/dsl/ModelIRPreview";
import { FEMResultsPanel } from "../components/dsl/FEMResultsPanel";
import { SegmentedControl } from "../components/ui/SegmentedControl";
import { useDslStore } from "../stores/dslStore";
import type { ScriptDiagnostic } from "../dsl/executor";
import type { IRDiagnostic } from "../dsl/model-ir";

const MOBILE_SEGMENTS = [
  { key: "script", label: "Script" },
  { key: "params", label: "Params" },
  { key: "diagnostics", label: "Diagnostics" },
] as const;

type MobileTab = (typeof MOBILE_SEGMENTS)[number]["key"];

export function ParametricEditorPage() {
  const paramValues = useDslStore((s) => s.paramValues);
  const setParamValue = useDslStore((s) => s.setParamValue);
  const runScript = useDslStore((s) => s.runScript);
  const result = useDslStore((s) => s.result);
  const status = useDslStore((s) => s.status);

  // Mobile tab state (local to this page)
  const [mobileTab, setMobileTab] = useReducer(
    (_prev: MobileTab, next: MobileTab) => next,
    "script" as MobileTab,
  );

  // Run the script once on mount
  useEffect(() => {
    void runScript();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-switch to params tab on mobile after first successful run
  useEffect(() => {
    if (status === "success" && mobileTab === "script") {
      setMobileTab("params");
    }
  }, [status]); // eslint-disable-line react-hooks/exhaustive-deps

  // Debounce re-run on param changes
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleParamChange = useCallback(
    (key: string, value: unknown) => {
      setParamValue(key, value);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => void runScript(), 300);
    },
    [setParamValue, runScript],
  );

  const schema = result?.schema;
  const ir = result?.ir;

  return (
    <div className="flex flex-col h-dvh bg-background">
      <Navbar />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* === LEFT PANEL — Script editor (desktop only) === */}
        <aside className="hidden lg:flex flex-col w-80 xl:w-96 border-r border-border bg-surface shrink-0">
          <div className="flex items-center px-3 py-2 border-b border-border shrink-0">
            <h2 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Script Editor
            </h2>
          </div>
          <div className="flex-1 min-h-0">
            <ScriptEditorPanel />
          </div>
        </aside>

        {/* === CENTER — 3D preview === */}
        <main className="flex-1 relative min-w-0 min-h-0 flex flex-col">
          <div className="flex-1 min-h-0">
            <ErrorBoundary label="ModelIR Preview">
              <ModelIRPreview ir={ir} />
            </ErrorBoundary>
          </div>
          <FEMResultsPanel />
        </main>

        {/* === RIGHT PANEL — Param form (desktop only) === */}
        <aside className="hidden lg:flex flex-col w-80 xl:w-96 border-l border-border bg-surface shrink-0 overflow-y-auto">
          <div className="flex items-center px-3 py-2 border-b border-border shrink-0">
            <h2 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Parameters
            </h2>
          </div>
          <div className="flex-1 p-3">
            {schema ? (
              <ParamSchemaForm
                schema={schema}
                values={paramValues}
                onChange={handleParamChange}
              />
            ) : (
              <p className="text-xs text-text-secondary">
                {status === "running"
                  ? "Evaluating script…"
                  : "Run the script to see parameters."}
              </p>
            )}
          </div>
        </aside>
      </div>

      {/* === MOBILE BOTTOM SHEET === */}
      <div className="lg:hidden border-t border-border bg-surface flex flex-col max-h-[55%]">
        <div className="px-3 pt-2 pb-1 shrink-0">
          <SegmentedControl
            segments={MOBILE_SEGMENTS as unknown as { key: string; label: string }[]}
            activeKey={mobileTab}
            onChange={(key) => setMobileTab(key as MobileTab)}
          />
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto">
          {mobileTab === "script" && (
            <div className="h-full">
              <ScriptEditorPanel />
            </div>
          )}

          {mobileTab === "params" && (
            <div className="p-3">
              {schema ? (
                <ParamSchemaForm
                  schema={schema}
                  values={paramValues}
                  onChange={handleParamChange}
                />
              ) : (
                <p className="text-xs text-text-secondary p-2">
                  {status === "running"
                    ? "Evaluating script…"
                    : "Run the script to see parameters."}
                </p>
              )}
            </div>
          )}

          {mobileTab === "diagnostics" && (
            <div className="p-3 space-y-1">
              <DiagnosticsPanel />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Diagnostics panel (used in mobile "Diagnostics" tab)
// ---------------------------------------------------------------------------

function isScriptDiagnostic(d: ScriptDiagnostic | IRDiagnostic): d is ScriptDiagnostic {
  return "phase" in d;
}

function DiagnosticsPanel() {
  const result = useDslStore((s) => s.result);
  const runError = useDslStore((s) => s.runError);
  const status = useDslStore((s) => s.status);
  const diagnostics = result?.diagnostics ?? [];

  if (status === "running") {
    return <p className="text-xs text-text-secondary">Evaluating…</p>;
  }
  if (!runError && diagnostics.length === 0) {
    return <p className="text-xs text-text-secondary">No diagnostics.</p>;
  }

  return (
    <>
      {runError && (
        <p className="text-[11px] font-mono text-swr-bad">
          <span className="font-semibold">worker:</span> {runError}
        </p>
      )}
      {diagnostics.map((d, i) => {
        if (isScriptDiagnostic(d)) {
          return (
            <p key={i} className="text-[11px] font-mono text-swr-bad">
              <span className="text-text-secondary font-semibold">[{d.phase}]</span> {d.message}
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
    </>
  );
}
