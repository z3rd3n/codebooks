import type { ParamSpec } from "../../api/types";

interface ParamFieldProps {
  spec: ParamSpec;
  value: unknown;
  onChange: (key: string, value: unknown) => void;
}

/** Renders one dynamic ParamSpec control (choice / int / float / bool). */
export function ParamField({ spec, value, onChange }: ParamFieldProps) {
  const current = value ?? spec.default ?? null;

  return (
    <div className="field">
      <label className="field-label" htmlFor={`param-${spec.key}`}>
        {spec.label}
      </label>
      {renderControl(spec, current, onChange)}
      {spec.help ? <span className="field-help">{spec.help}</span> : null}
    </div>
  );
}

function renderControl(
  spec: ParamSpec,
  current: unknown,
  onChange: (key: string, value: unknown) => void,
) {
  switch (spec.type) {
    case "bool":
      return (
        <div className="checkbox-row">
          <input
            id={`param-${spec.key}`}
            type="checkbox"
            checked={Boolean(current)}
            onChange={(e) => onChange(spec.key, e.target.checked)}
          />
          <span className="text-sm text-secondary">{current ? "Enabled" : "Disabled"}</span>
        </div>
      );

    case "choice": {
      const choices = spec.choices ?? [];
      if (choices.length === 0) {
        return <div className="text-sm text-muted">No options available.</div>;
      }
      if (choices.length <= 6) {
        return (
          <div className="choice-grid">
            {choices.map((c) => {
              const selected = String(c.value) === String(current);
              return (
                <button
                  key={String(c.value)}
                  type="button"
                  className={`choice-card${selected ? " selected" : ""}`}
                  onClick={() => onChange(spec.key, c.value)}
                  aria-pressed={selected}
                >
                  <span className="choice-card-label">{c.label}</span>
                  {c.description ? (
                    <span className="choice-card-desc">{c.description}</span>
                  ) : null}
                </button>
              );
            })}
          </div>
        );
      }
      return (
        <select
          id={`param-${spec.key}`}
          className="select"
          value={String(current ?? "")}
          onChange={(e) => {
            const raw = e.target.value;
            const match = choices.find((c) => String(c.value) === raw);
            onChange(spec.key, match ? match.value : raw);
          }}
        >
          {choices.map((c) => (
            <option key={String(c.value)} value={String(c.value)}>
              {c.label}
            </option>
          ))}
        </select>
      );
    }

    case "int":
    case "float": {
      const num = typeof current === "number" ? current : Number(current) || 0;
      return (
        <input
          id={`param-${spec.key}`}
          className="input"
          type="number"
          value={Number.isFinite(num) ? num : ""}
          min={spec.min ?? undefined}
          max={spec.max ?? undefined}
          step={spec.step ?? (spec.type === "int" ? 1 : 0.1)}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "") return onChange(spec.key, null);
            onChange(spec.key, spec.type === "int" ? parseInt(v, 10) : parseFloat(v));
          }}
        />
      );
    }

    default:
      return null;
  }
}

/** Evaluate a ParamSpec's `visibleIf` condition against current param values. */
export function isParamVisible(spec: ParamSpec, params: Record<string, unknown>): boolean {
  if (!spec.visibleIf) return true;
  const actual = params[spec.visibleIf.key];
  return String(actual) === String(spec.visibleIf.value);
}
