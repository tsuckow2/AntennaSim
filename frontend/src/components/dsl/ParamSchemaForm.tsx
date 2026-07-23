/**
 * ParamSchemaForm — schema-driven parameter form for parametric scripts.
 *
 * Renders each parameter in the schema as an appropriate input:
 *   number / int  → Slider when min/max/step are all set, else NumberInput
 *   bool          → checkbox
 *   string        → text input
 *   enum          → <select>
 *   vector        → row of NumberInputs (one per component)
 *
 * When schema.groups is present, parameters are rendered under their group
 * headings.  Parameters not listed in any group are rendered last.
 */

import { useCallback } from "react";
import type { ParamSchema, ParamDefinition, VectorParamDefinition } from "../../utils/param-schema";
import { Slider } from "../ui/Slider";
import { NumberInput } from "../ui/NumberInput";

interface ParamSchemaFormProps {
  schema: ParamSchema;
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}

// ---------------------------------------------------------------------------
// Per-type helpers
// ---------------------------------------------------------------------------

function inferDecimals(step: number | undefined, fallback = 4): number {
  if (!step) return fallback;
  const s = step.toString();
  const dot = s.indexOf(".");
  return dot < 0 ? 0 : s.length - dot - 1;
}

function NumericField({
  paramKey,
  def,
  value,
  onChange,
  integer,
}: {
  paramKey: string;
  def: { min?: number; max?: number; step?: number; unit?: string; title?: string; description?: string };
  value: number;
  onChange: (key: string, value: unknown) => void;
  integer: boolean;
}) {
  const hasRange = def.min !== undefined && def.max !== undefined && def.step !== undefined;
  const decimals = integer ? 0 : inferDecimals(def.step);
  const handleChange = useCallback((v: number) => onChange(paramKey, v), [paramKey, onChange]);

  if (hasRange) {
    return (
      <Slider
        label={def.title ?? paramKey}
        value={value}
        min={def.min!}
        max={def.max!}
        step={def.step!}
        unit={def.unit ?? ""}
        decimals={decimals}
        description={def.description}
        onChange={handleChange}
      />
    );
  }
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs text-text-secondary flex-1 truncate" title={def.description}>
        {def.title ?? paramKey}
      </span>
      <NumberInput
        value={value}
        onChange={handleChange}
        min={def.min}
        max={def.max}
        decimals={decimals}
        unit={def.unit}
      />
    </div>
  );
}

function VectorField({
  paramKey,
  def,
  value,
  onChange,
}: {
  paramKey: string;
  def: VectorParamDefinition;
  value: number[];
  onChange: (key: string, value: unknown) => void;
}) {
  const size = def.size ?? value.length ?? 3;
  const handleCompChange = useCallback(
    (i: number, v: number) => {
      const next = [...value];
      next[i] = v;
      onChange(paramKey, next);
    },
    [paramKey, value, onChange],
  );

  return (
    <div className="space-y-0.5">
      <span className="text-xs text-text-secondary block" title={def.description}>
        {def.title ?? paramKey}
        {def.unit ? <span className="ml-1 text-[10px]">({def.unit})</span> : null}
      </span>
      <div className="flex gap-1 flex-wrap">
        {Array.from({ length: size }, (_, i) => (
          <NumberInput
            key={i}
            value={value[i] ?? 0}
            onChange={(v) => handleCompChange(i, v)}
            min={def.min}
            max={def.max}
            decimals={inferDecimals(def.step)}
            label={(["x", "y", "z", "w"] as const)[i] ?? String(i)}
            size="sm"
          />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-param renderer
// ---------------------------------------------------------------------------

function ParamField({
  paramKey,
  def,
  value,
  onChange,
}: {
  paramKey: string;
  def: ParamDefinition;
  value: unknown;
  onChange: (key: string, value: unknown) => void;
}) {
  switch (def.type) {
    case "number":
    case "int": {
      const num = typeof value === "number" ? value : (def.default as number | undefined) ?? 0;
      return (
        <NumericField
          paramKey={paramKey}
          def={def}
          value={num}
          onChange={onChange}
          integer={def.type === "int"}
        />
      );
    }

    case "bool": {
      const checked = typeof value === "boolean" ? value : (def.default as boolean | undefined) ?? false;
      return (
        <label className="flex items-center gap-2 cursor-pointer select-none" title={def.description}>
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onChange(paramKey, e.target.checked)}
            className="rounded border-border accent-accent"
          />
          <span className="text-xs text-text-secondary">{def.title ?? paramKey}</span>
        </label>
      );
    }

    case "string": {
      const str = typeof value === "string" ? value : (def.default as string | undefined) ?? "";
      return (
        <div className="space-y-0.5">
          <label className="text-xs text-text-secondary" title={def.description}>
            {def.title ?? paramKey}
          </label>
          <input
            type="text"
            value={str}
            onChange={(e) => onChange(paramKey, e.target.value)}
            className="w-full text-xs font-mono bg-background border border-border rounded
              px-2 py-1 text-text-primary focus:outline-none focus:border-accent/50"
          />
        </div>
      );
    }

    case "enum": {
      const sel = typeof value === "string" ? value : (def.default as string | undefined) ?? def.values[0];
      return (
        <div className="space-y-0.5">
          <label className="text-xs text-text-secondary" title={def.description}>
            {def.title ?? paramKey}
          </label>
          <select
            value={sel}
            onChange={(e) => onChange(paramKey, e.target.value)}
            className="w-full text-xs bg-background border border-border rounded
              px-2 py-1 text-text-primary focus:outline-none focus:border-accent/50"
          >
            {def.values.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
      );
    }

    case "vector": {
      const arr = Array.isArray(value) ? (value as number[]) : ((def.default as number[] | undefined) ?? [0, 0, 0]);
      return <VectorField paramKey={paramKey} def={def} value={arr} onChange={onChange} />;
    }

    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ParamSchemaForm({ schema, values, onChange }: ParamSchemaFormProps) {
  const paramEntries = Object.entries(schema.params);
  const groups = schema.groups ?? [];

  // Track which params are covered by a group
  const grouped = new Set(groups.flatMap((g) => g.params));
  const ungrouped = paramEntries.filter(([k]) => !grouped.has(k));

  return (
    <div className="space-y-4">
      {schema.title && (
        <div>
          <h2 className="text-sm font-semibold text-text-primary">{schema.title}</h2>
          {schema.description && (
            <p className="text-[11px] text-text-secondary mt-0.5">{schema.description}</p>
          )}
        </div>
      )}

      {groups.map((group) => (
        <div key={group.title} className="space-y-2">
          <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wider">
            {group.title}
          </h3>
          {group.params.map((key) => {
            const def = schema.params[key];
            if (!def) return null;
            return (
              <ParamField
                key={key}
                paramKey={key}
                def={def}
                value={values[key]}
                onChange={onChange}
              />
            );
          })}
        </div>
      ))}

      {ungrouped.length > 0 && (
        <div className="space-y-2">
          {groups.length > 0 && (
            <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wider">
              Other
            </h3>
          )}
          {ungrouped.map(([key, def]) => (
            <ParamField
              key={key}
              paramKey={key}
              def={def}
              value={values[key]}
              onChange={onChange}
            />
          ))}
        </div>
      )}

      {schema.constraints && schema.constraints.length > 0 && (
        <div className="space-y-1 border-t border-border pt-2">
          <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wider">
            Constraints
          </h3>
          {schema.constraints.map((c, i) => (
            <p key={i} className="text-[10px] text-text-secondary font-mono">
              {c.message ?? c.expr}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
