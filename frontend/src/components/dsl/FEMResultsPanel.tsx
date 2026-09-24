import { ImpedanceChart } from "../results/ImpedanceChart";
import { SWRChart } from "../results/SWRChart";
import { SmithChart } from "../results/SmithChart";
import { useDslStore } from "../../stores/dslStore";
import { getFEMGainAvailabilityMessage, getRenderableFEMGainPoints, toNECFrequencyResults } from "../../engine/fem";

export function FEMResultsPanel() {
  const fem = useDslStore((state) => state.femResult);
  const femStatus = useDslStore((state) => state.femStatus);
  const femError = useDslStore((state) => state.femError);
  const femProgress = useDslStore((state) => state.femProgress);
  if (femStatus === "idle" && !fem) return null;
  if (femStatus === "running") {
    const detail =
      femProgress && femProgress.pointsTotal > 0 && femProgress.pointIndex > 0
        ? ` — point ${femProgress.pointIndex} of ${femProgress.pointsTotal}`
        : femProgress?.stage === "meshing"
          ? " — meshing"
          : "";
    return <div className="p-3 text-xs text-text-secondary">FEM run in progress{detail}...</div>;
  }
  if (femStatus === "error" && !fem) return <div className="p-3 text-xs text-swr-bad">FEM run failed: {femError}</div>;
  if (!fem) return null;
  const data = toNECFrequencyResults(fem);
  const invalidPoints = fem.frequency_data.filter((point) => point.point_status !== "ok").length;
  const gainPoints = getRenderableFEMGainPoints(fem);
  const gainMessage = getFEMGainAvailabilityMessage(fem);
  return (
    <section className="border-t border-border bg-surface p-3 overflow-y-auto max-h-[48%]">
      <div className="flex items-center justify-between gap-3 mb-2">
        <div>
          <h2 className="text-xs font-semibold text-text-primary">
            FEM results <span className="font-normal text-text-secondary">· {fem.engine === "fem-typescript" ? "TypeScript H(curl) solver" : "MFEM WASM"}</span>
          </h2>
          <p className="text-[11px] text-text-secondary">{fem.mesh.elements} elements, {fem.mesh.nodes} nodes, Z0 = {fem.reference_impedance_ohms} ohms</p>
        </div>
        <span className="text-[10px] text-swr-warning">{gainMessage}</span>
      </div>
      {femStatus === "error" && femError && <p className="text-[11px] text-swr-bad mb-1">FEM run failed: {femError}</p>}
      {invalidPoints > 0 && <p className="text-[11px] text-swr-warning mb-1">Skipped {invalidPoints} invalid FEM points while rendering charts.</p>}
      {fem.warnings.map((warning) => <p key={warning} className="text-[11px] text-swr-warning mb-1">{warning}</p>)}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        {fem.capabilities.impedance && <ImpedanceChart data={data} heightClass="h-48" />}
        {fem.capabilities.swr && <SWRChart data={data} heightClass="h-48" />}
        {fem.capabilities.smith_chart && <SmithChart data={data} z0={fem.reference_impedance_ohms} responsive size={240} />}
      </div>
      {fem.capabilities.gain && gainPoints.length > 0 && (
        <div className="mt-3">
          <h3 className="text-xs font-semibold text-text-primary">Gain (native far-field)</h3>
          <div className="mt-1 space-y-0.5">
            {gainPoints.slice(0, 5).map((point) => (
              <p key={point.frequency_mhz} className="text-[11px] text-text-secondary font-mono">
                {point.frequency_mhz.toFixed(3)} MHz: {point.gain_dbi.toFixed(2)} dBi @ {point.theta_deg.toFixed(1)} deg el, {point.phi_deg.toFixed(1)} deg az
              </p>
            ))}
          </div>
        </div>
      )}
      <p className="text-[11px] text-text-secondary mt-2">Resonant frequency: {fem.resonant_frequency_mhz === null ? "not available" : `${fem.resonant_frequency_mhz.toFixed(3)} MHz`}</p>
    </section>
  );
}