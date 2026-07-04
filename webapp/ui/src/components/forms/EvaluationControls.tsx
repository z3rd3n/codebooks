import { Icon } from "../Icon";

interface RankStepperProps {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
}

export function RankStepper({ value, onChange, min, max }: RankStepperProps) {
  return (
    <div className="stepper" role="group" aria-label="Rank">
      <button type="button" onClick={() => onChange(Math.max(min, value - 1))} disabled={value <= min} aria-label="Decrease rank">
        &minus;
      </button>
      <span className="stepper-value tabular-nums">{value}</span>
      <button type="button" onClick={() => onChange(Math.min(max, value + 1))} disabled={value >= max} aria-label="Increase rank">
        +
      </button>
    </div>
  );
}

interface DropsSliderProps {
  value: number;
  onChange: (v: number) => void;
}

export function DropsSlider({ value, onChange }: DropsSliderProps) {
  return (
    <div className="col col-gap-2">
      <input
        className="range"
        type="range"
        min={1}
        max={64}
        step={1}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        aria-label="Monte-Carlo drops"
      />
      <span className="text-sm text-secondary tabular-nums">{value} drop{value === 1 ? "" : "s"}</span>
    </div>
  );
}

interface SnrRangeProps {
  value: number[];
  onChange: (v: number[]) => void;
}

/** SNR range editor: start/stop/step, capped at 12 points per the API contract. */
export function SnrRange({ value, onChange }: SnrRangeProps) {
  const start = value.length ? value[0] : -10;
  const stop = value.length ? value[value.length - 1] : 30;
  const step = value.length > 1 ? value[1] - value[0] : 5;

  function rebuild(newStart: number, newStop: number, newStep: number) {
    if (newStep <= 0) return;
    const points: number[] = [];
    for (let v = newStart; v <= newStop + 1e-9 && points.length < 12; v += newStep) {
      points.push(Math.round(v * 100) / 100);
    }
    onChange(points.length ? points : [newStart]);
  }

  return (
    <div className="col col-gap-2">
      <div className="grid-3">
        <div className="field" style={{ marginBottom: 0 }}>
          <label className="field-label" htmlFor="snr-start">Start (dB)</label>
          <input
            id="snr-start"
            className="input"
            type="number"
            value={start}
            onChange={(e) => rebuild(parseFloat(e.target.value) || 0, stop, step)}
          />
        </div>
        <div className="field" style={{ marginBottom: 0 }}>
          <label className="field-label" htmlFor="snr-stop">Stop (dB)</label>
          <input
            id="snr-stop"
            className="input"
            type="number"
            value={stop}
            onChange={(e) => rebuild(start, parseFloat(e.target.value) || 0, step)}
          />
        </div>
        <div className="field" style={{ marginBottom: 0 }}>
          <label className="field-label" htmlFor="snr-step">Step (dB)</label>
          <input
            id="snr-step"
            className="input"
            type="number"
            min={0.5}
            value={step}
            onChange={(e) => rebuild(start, stop, parseFloat(e.target.value) || 1)}
          />
        </div>
      </div>
      <span className="text-sm text-muted tabular-nums">
        {value.length} point{value.length === 1 ? "" : "s"}: {value.join(", ")} dB
      </span>
      {value.length > 12 && (
        <span className="text-sm" style={{ color: "var(--danger)" }}>
          <Icon name="warning" size={14} /> Maximum 12 SNR points; extra points will be dropped.
        </span>
      )}
    </div>
  );
}

interface SeedInputProps {
  value: number;
  onChange: (v: number) => void;
}

export function SeedInput({ value, onChange }: SeedInputProps) {
  return (
    <input
      className="input"
      type="number"
      value={value}
      onChange={(e) => onChange(parseInt(e.target.value, 10) || 0)}
      aria-label="Random seed"
      style={{ maxWidth: 140 }}
    />
  );
}
