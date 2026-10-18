/**
 * Impedance (R + jX) vs Frequency chart.
 *
 * Features:
 * - R (resistance) as solid blue line, X (reactance) as solid orange line
 * - 50-ohm reference line (dashed gray with label)
 * - Zero-reactance reference line
 * - Reactance-crossing markers, classified by direction: a falling
 *   crossing (X: + to -) is a parallel resonance (green); a rising
 *   crossing is a series crossing (gray) — the resonance of a
 *   series-resonant antenna, but on a patch swept above its parallel
 *   resonance it is only the probe inductance cancelling the capacitive
 *   tail, and is not labeled a resonance. Each marker shows the
 *   interpolated R at the crossing.
 * - Crosshair tooltip showing frequency, R, X, and |Z|
 * - Clear legend with color-coded labels
 */

import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import type { FrequencyResult } from "../../api/nec";
import type { MatchingConfig } from "../../utils/units";
import { applyMatching, DEFAULT_MATCHING } from "../../utils/units";
import { findReactanceCrossings } from "../../utils/resonance";
import { useChartTheme } from "../../hooks/useChartTheme";

interface ImpedanceChartProps {
  data: FrequencyResult[];
  /** Matching config for impedance transformation */
  matching?: MatchingConfig;
  /** Height class override (default: h-48) */
  heightClass?: string;
}

export function ImpedanceChart({ data, matching = DEFAULT_MATCHING, heightClass = "h-56" }: ImpedanceChartProps) {
  const chartData = useMemo(
    () =>
      data
        .map((d) => {
          const m = applyMatching(d.impedance.real, d.impedance.imag, matching);
          return {
            freq: d.frequency_mhz,
            r: m.real,
            x: m.imag,
          };
        })
        .sort((a, b) => a.freq - b.freq),
    [data, matching]
  );

  const freqRange = useMemo(() => {
    if (chartData.length === 0) return { min: 0, max: 1 };
    return {
      min: chartData[0]!.freq,
      max: chartData[chartData.length - 1]!.freq,
    };
  }, [chartData]);

  // Calculate Y axis bounds
  const yBounds = useMemo(() => {
    if (chartData.length === 0) return { min: -100, max: 200 };
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const d of chartData) {
      minVal = Math.min(minVal, d.r, d.x);
      maxVal = Math.max(maxVal, d.r, d.x);
    }
    const padding = Math.max(20, (maxVal - minVal) * 0.1);
    return {
      min: Math.floor((minVal - padding) / 10) * 10,
      max: Math.ceil((maxVal + padding) / 10) * 10,
    };
  }, [chartData]);

  // Reactance zero crossings, classified by direction (parallel vs
  // series) with the interpolated resistance at each crossing.
  const crossings = useMemo(
    () =>
      findReactanceCrossings(
        chartData.map((d) => ({
          frequencyMhz: d.freq,
          resistanceOhms: d.r,
          reactanceOhms: d.x,
        })),
      ),
    [chartData],
  );

  const ct = useChartTheme();

  if (data.length === 0) return null;

  return (
    <div className={`w-full ${heightClass} flex flex-col`}>
      <div className="flex-1 min-h-0">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, bottom: 5, left: -5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={ct.grid}
            strokeOpacity={0.5}
          />

          <XAxis
            dataKey="freq"
            type="number"
            domain={[freqRange.min, freqRange.max]}
            tick={{ fill: ct.tick, fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
            tickFormatter={(v: number) => v.toFixed(3)}
            stroke={ct.axis}
          />

          <YAxis
            domain={[yBounds.min, yBounds.max]}
            tick={{ fill: ct.tick, fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
            stroke={ct.axis}
            tickFormatter={(v: number) => `${v}`}
          />

          {/* Reference impedance line (matches feedline Z0) */}
          <ReferenceLine
            y={matching.feedlineZ0}
            stroke="#6B7280"
            strokeDasharray="6 3"
            strokeOpacity={0.5}
            label={{ value: `${matching.feedlineZ0}\u03A9`, position: "right", fill: "#6B7280", fontSize: 9, fontFamily: "JetBrains Mono, monospace" }}
          />

          {/* Zero reactance reference (resonance line) */}
          <ReferenceLine y={0} stroke={ct.axis} strokeOpacity={0.6} />

          {/* Reactance-crossing markers: parallel resonances in green,
              series crossings in gray, each labeled with its R. */}
          {crossings.map((crossing, i) => {
            const parallel = crossing.kind === "parallel";
            const color = parallel ? "#10B981" : "#9CA3AF";
            return (
              <ReferenceLine
                key={`res-${i}`}
                x={crossing.frequencyMhz}
                stroke={color}
                strokeDasharray="3 3"
                strokeOpacity={0.5}
                label={{
                  value: `${parallel ? "res " : ""}${crossing.frequencyMhz.toFixed(3)} MHz · R ${crossing.resistanceOhms.toFixed(0)}Ω`,
                  position: "top",
                  fill: color,
                  fontSize: 8,
                  fontFamily: "JetBrains Mono, monospace",
                }}
              />
            );
          })}

          <Tooltip
            contentStyle={{
              backgroundColor: ct.tooltipBg,
              border: `1px solid ${ct.tooltipBorder}`,
              borderRadius: "6px",
              fontSize: "11px",
              fontFamily: "JetBrains Mono, monospace",
            }}
            labelStyle={{ color: ct.tooltipLabel }}
            labelFormatter={(v) => `${Number(v).toFixed(3)} MHz`}
            formatter={(value, name) => {
              const label = name === "r" ? "R" : "jX";
              const color = name === "r" ? "#3B82F6" : "#F59E0B";
              return [
                <span key={String(name)} style={{ color }}>
                  {Number(value).toFixed(1)} {"\u03A9"}
                </span>,
                label,
              ];
            }}
            cursor={{ stroke: ct.cursor, strokeWidth: 1 }}
          />

          <Legend
            iconType="line"
            wrapperStyle={{ fontSize: "10px", fontFamily: "JetBrains Mono, monospace", paddingTop: "4px" }}
            formatter={(value: string) => (
              <span style={{ color: ct.tick }}>
                {value === "r" ? "R  Resistance (\u03A9)" : "jX  Reactance (\u03A9)"}
              </span>
            )}
          />

          {/* Resistance — solid blue */}
          <Line
            type="monotone"
            dataKey="r"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3, fill: "#3B82F6" }}
            name="r"
            animationDuration={300}
          />

          {/* Reactance — solid orange */}
          <Line
            type="monotone"
            dataKey="x"
            stroke="#F59E0B"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3, fill: "#F59E0B" }}
            name="x"
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
      </div>
      {/* Supplementary legend for reference lines */}
      <div className="flex items-center justify-center gap-3 pt-1 shrink-0" style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "9px" }}>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-0 border-t border-dashed" style={{ borderColor: "#6B7280" }} />
          <span style={{ color: ct.tick }}>{matching.feedlineZ0}{"\u03A9"} ref</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-0 border-t border-dashed" style={{ borderColor: "#10B981" }} />
          <span style={{ color: ct.tick }}>Parallel resonance (jX=0, falling)</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-0 border-t border-dashed" style={{ borderColor: "#9CA3AF" }} />
          <span style={{ color: ct.tick }}>Series crossing (jX=0, rising)</span>
        </span>
      </div>
    </div>
  );
}
