import { useState } from "react";
import type { ChannelConfig } from "../../api/types";

interface ChannelPreset {
  id: string;
  label: string;
  description: string;
  config: Required<Pick<ChannelConfig, "n_paths" | "max_delay" | "max_doppler">>;
}

const PRESETS: ChannelPreset[] = [
  {
    id: "sparse-urban",
    label: "Sparse urban",
    description: "4 paths, moderate delay spread — typical macro-cell NLOS.",
    config: { n_paths: 4, max_delay: 3.0, max_doppler: 0.0 },
  },
  {
    id: "rich-scattering",
    label: "Rich scattering",
    description: "12 paths, wide delay spread — dense indoor/urban multipath.",
    config: { n_paths: 12, max_delay: 6.0, max_doppler: 0.0 },
  },
  {
    id: "near-los",
    label: "Near line-of-sight",
    description: "2 paths, short delay spread — clear line-of-sight dominant path.",
    config: { n_paths: 2, max_delay: 1.0, max_doppler: 0.0 },
  },
  {
    id: "mobile-user",
    label: "Mobile user",
    description: "4 paths with Doppler spread — required for Doppler-aware and predicted codebooks.",
    config: { n_paths: 4, max_delay: 3.0, max_doppler: 0.5 },
  },
];

interface ChannelPickerProps {
  value: ChannelConfig;
  onChange: (channel: ChannelConfig) => void;
  nRx: number;
  onNRxChange: (n: number) => void;
}

export function ChannelPicker({ value, onChange, nRx, onNRxChange }: ChannelPickerProps) {
  const activePresetId = value.preset ?? null;
  const [customOpen, setCustomOpen] = useState(activePresetId === null || activePresetId === "custom");

  function selectPreset(preset: ChannelPreset) {
    onChange({ preset: preset.id, ...preset.config });
    setCustomOpen(false);
  }

  function openCustom() {
    setCustomOpen(true);
    onChange({
      preset: "custom",
      n_paths: value.n_paths ?? 4,
      max_delay: value.max_delay ?? 3.0,
      max_doppler: value.max_doppler ?? 0.0,
    });
  }

  return (
    <div className="col col-gap-3">
      <div className="choice-grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))" }}>
        {PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            className={`choice-card${activePresetId === p.id && !customOpen ? " selected" : ""}`}
            onClick={() => selectPreset(p)}
            aria-pressed={activePresetId === p.id && !customOpen}
          >
            <span className="choice-card-label">{p.label}</span>
            <span className="choice-card-desc">{p.description}</span>
          </button>
        ))}
        <button
          type="button"
          className={`choice-card${customOpen ? " selected" : ""}`}
          onClick={openCustom}
          aria-pressed={customOpen}
        >
          <span className="choice-card-label">Custom</span>
          <span className="choice-card-desc">Set paths, delay spread, and Doppler manually.</span>
        </button>
      </div>

      {customOpen && (
        <div className="grid-3">
          <div className="field">
            <label className="field-label" htmlFor="ch-npaths">Number of paths</label>
            <input
              id="ch-npaths"
              className="input"
              type="number"
              min={1}
              max={32}
              value={value.n_paths ?? 4}
              onChange={(e) => onChange({ ...value, preset: "custom", n_paths: parseInt(e.target.value, 10) || 1 })}
            />
          </div>
          <div className="field">
            <label className="field-label" htmlFor="ch-delay">Max delay spread</label>
            <input
              id="ch-delay"
              className="input"
              type="number"
              step={0.1}
              min={0}
              value={value.max_delay ?? 3.0}
              onChange={(e) => onChange({ ...value, preset: "custom", max_delay: parseFloat(e.target.value) || 0 })}
            />
          </div>
          <div className="field">
            <label className="field-label" htmlFor="ch-doppler">Max Doppler</label>
            <input
              id="ch-doppler"
              className="input"
              type="number"
              step={0.05}
              min={0}
              value={value.max_doppler ?? 0.0}
              onChange={(e) => onChange({ ...value, preset: "custom", max_doppler: parseFloat(e.target.value) || 0 })}
            />
          </div>
        </div>
      )}

      <div className="field" style={{ maxWidth: 220 }}>
        <label className="field-label" htmlFor="ch-nrx">Receive antennas (n_rx)</label>
        <input
          id="ch-nrx"
          className="input"
          type="number"
          min={1}
          max={8}
          value={nRx}
          onChange={(e) => onNRxChange(parseInt(e.target.value, 10) || 1)}
        />
      </div>
    </div>
  );
}
