/**
 * FEMPattern3D — 3D radiation pattern viewer for FEM far-field results.
 *
 * Converts the FEM far-field pattern (per-frequency) to the NEC PatternData
 * shape and renders it with the shared RadiationPattern3D component.
 */

import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { RadiationPattern3D } from "../three/RadiationPattern3D";
import {
  femGainPatternToNecPattern,
  type FEMSimulationResult,
} from "../../engine/fem";

export function FEMPattern3D({ fem }: { fem: FEMSimulationResult }) {
  const patterns = fem.gain_data?.patterns ?? [];
  const [selectedIndex, setSelectedIndex] = useState(0);
  const safeIndex = Math.min(selectedIndex, Math.max(0, patterns.length - 1));
  const selected = patterns[safeIndex];
  const necPattern = useMemo(
    () => (selected ? femGainPatternToNecPattern(selected) : null),
    [selected],
  );

  if (patterns.length === 0 || !necPattern || !selected) return null;

  return (
    <div className="mt-3">
      <div className="flex items-center justify-between gap-3 mb-1">
        <h3 className="text-xs font-semibold text-text-primary">
          3D radiation pattern
          <span className="font-normal text-text-secondary">
            {" "}· {selected.frequency_mhz.toFixed(3)} MHz · max {selected.max_gain_dbi.toFixed(2)} dBi
          </span>
        </h3>
        {patterns.length > 1 && (
          <select
            className="text-[11px] bg-surface border border-border rounded px-1 py-0.5"
            value={safeIndex}
            onChange={(event) => setSelectedIndex(Number(event.target.value))}
            aria-label="Select frequency for 3D pattern"
          >
            {patterns.map((pattern, index) => (
              <option key={pattern.frequency_mhz} value={index}>
                {pattern.frequency_mhz.toFixed(1)} MHz
              </option>
            ))}
          </select>
        )}
      </div>
      <div className="h-64 w-full rounded border border-border overflow-hidden bg-black">
        <Canvas camera={{ position: [2.2, 1.6, 2.2], fov: 45 }}>
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 5, 5]} intensity={0.8} />
          <RadiationPattern3D
            pattern={necPattern}
            scale={1.2}
            opacity={0.85}
            center={[0, 0, 0]}
          />
          <OrbitControls enablePan={false} minDistance={1} maxDistance={10} />
        </Canvas>
      </div>
      <p className="text-[10px] text-text-secondary mt-1">
        Drag to rotate · scroll to zoom. Lobe radius ∝ gain; color blue→red = low→high.
      </p>
    </div>
  );
}
