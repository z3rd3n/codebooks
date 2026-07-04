import { useEffect, useMemo, useRef, useState } from "react";
import type { CodebookDetail, RunRequest } from "../../api/types";
import { api, ApiError } from "../../api/client";
import { ParamField, isParamVisible } from "./ParamField";
import { AntennaPicker } from "./AntennaPicker";
import { ChannelPicker } from "./ChannelPicker";
import { RankStepper, DropsSlider, SnrRange, SeedInput } from "./EvaluationControls";
import { Icon } from "../Icon";

export function buildDefaultRequest(detail: CodebookDetail): RunRequest {
  const params: Record<string, unknown> = {};
  for (const p of detail.params ?? []) {
    params[p.key] = p.default;
  }
  // Prefer a mid-size (~16-port) array: the smallest geometries (4 ports) are
  // incompatible with many families' default parameter combinations, so the
  // out-of-the-box config would otherwise be invalid.
  const pairs = detail.antenna?.pairs ?? [];
  const antennaPair = pairs.length
    ? pairs.reduce((best, p) => (Math.abs(p.ports - 16) < Math.abs(best.ports - 16) ? p : best), pairs[0])
    : undefined;
  const [rankLo, rankHi] = detail.ranks ?? [1, 1];
  return {
    codebook_id: detail.id,
    params,
    antenna: {
      n1: antennaPair?.n1 ?? 4,
      n2: antennaPair?.n2 ?? 2,
      ng: antennaPair?.ng ?? 1,
    },
    n3: 8,
    rank: rankLo,
    channel: { preset: "sparse-urban", n_rx: 2, n_paths: 4, max_delay: 3.0, max_doppler: 0.0 },
    snr_db: [-10, -5, 0, 5, 10, 15, 20, 25, 30],
    drops: 8,
    seed: 0,
    ...(rankHi < rankLo ? {} : {}),
  };
}

interface PlaygroundFormProps {
  detail: CodebookDetail;
  request: RunRequest;
  onChange: (req: RunRequest) => void;
  onRun: (req: RunRequest) => void;
  running: boolean;
}

export function PlaygroundForm({ detail, request, onChange, onRun, running }: PlaygroundFormProps) {
  const [validation, setValidation] = useState<{ ok: boolean; error?: string | null } | null>(null);
  const [validating, setValidating] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reqRef = useRef(request);
  reqRef.current = request;

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setValidating(true);
      api
        .validate(reqRef.current)
        .then((res) => setValidation(res))
        .catch((err) => {
          const message = err instanceof ApiError ? err.message : "Could not validate this configuration.";
          setValidation({ ok: false, error: message });
        })
        .finally(() => setValidating(false));
    }, 400);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(request)]);

  const [rankLo, rankHi] = detail.ranks ?? [1, 8];

  const visibleParams = useMemo(
    () => (detail.params ?? []).filter((p) => isParamVisible(p, request.params)),
    [detail.params, request.params],
  );

  function setParam(key: string, value: unknown) {
    onChange({ ...request, params: { ...request.params, [key]: value } });
  }

  return (
    <div className="col col-gap-5">
      {(detail.params ?? []).length > 0 && (
        <section className="card">
          <h3 className="card-title">Codebook settings</h3>
          <p className="panel-subtitle" style={{ marginBottom: 16 }}>
            Parameters specific to {detail.shortName || detail.name}.
          </p>
          {visibleParams.map((p) => (
            <ParamField key={p.key} spec={p} value={request.params[p.key]} onChange={setParam} />
          ))}
        </section>
      )}

      <section className="card">
        <h3 className="card-title">Antenna array</h3>
        <p className="panel-subtitle" style={{ marginBottom: 16 }}>
          The base station's dual-polarized N1 &times; N2 antenna grid.
        </p>
        <AntennaPicker
          spec={detail.antenna}
          value={request.antenna}
          onChange={(pair) =>
            onChange({ ...request, antenna: { n1: pair.n1, n2: pair.n2, ng: pair.ng ?? 1 } })
          }
        />
      </section>

      <section className="card">
        <h3 className="card-title">Channel</h3>
        <p className="panel-subtitle" style={{ marginBottom: 16 }}>
          Synthetic multipath channel used to evaluate the codebook.
        </p>
        <ChannelPicker
          value={request.channel}
          onChange={(channel) => onChange({ ...request, channel: { ...channel, n_rx: request.channel.n_rx } })}
          nRx={request.channel.n_rx ?? 2}
          onNRxChange={(n_rx) => onChange({ ...request, channel: { ...request.channel, n_rx } })}
        />
      </section>

      <section className="card">
        <h3 className="card-title">Evaluation</h3>
        <p className="panel-subtitle" style={{ marginBottom: 16 }}>
          Rank, frequency granularity, Monte-Carlo drops, SNR sweep, and seed.
        </p>
        <div className="grid-2">
          <div className="field">
            <label className="field-label">Rank (layers)</label>
            <RankStepper
              value={request.rank}
              min={rankLo}
              max={rankHi}
              onChange={(rank) => onChange({ ...request, rank })}
            />
          </div>
          <div className="field">
            <label className="field-label" htmlFor="n3">Frequency units (N3)</label>
            <input
              id="n3"
              className="input"
              type="number"
              min={1}
              max={64}
              value={request.n3}
              onChange={(e) => onChange({ ...request, n3: parseInt(e.target.value, 10) || 1 })}
            />
          </div>
        </div>
        <div className="grid-2" style={{ marginTop: 4 }}>
          <div className="field">
            <label className="field-label">Monte-Carlo drops</label>
            <DropsSlider value={request.drops} onChange={(drops) => onChange({ ...request, drops })} />
          </div>
          <div className="field">
            <label className="field-label">Seed</label>
            <SeedInput value={request.seed} onChange={(seed) => onChange({ ...request, seed })} />
          </div>
        </div>
        <div className="field">
          <label className="field-label">SNR sweep</label>
          <SnrRange value={request.snr_db} onChange={(snr_db) => onChange({ ...request, snr_db })} />
        </div>
      </section>

      {validation && !validation.ok && (
        <div className="error-banner" role="alert">
          <Icon name="warning" />
          <div className="error-banner-body">
            <div className="error-banner-title">This configuration is invalid</div>
            <div>{validation.error}</div>
            <div className="error-banner-hint">Adjust the parameters above and it will re-validate automatically.</div>
          </div>
        </div>
      )}

      <div className="row row-gap-3">
        <button
          className="btn btn-primary"
          disabled={running || (validation !== null && !validation.ok)}
          onClick={() => onRun(request)}
        >
          {running ? <Icon name="spinner" className="spin" /> : <Icon name="play" />}
          {running ? "Running…" : "Run"}
        </button>
        {validating && <span className="text-sm text-muted">Validating…</span>}
        {validation?.ok && !validating && (
          <span className="text-sm" style={{ color: "var(--success)" }}>
            <Icon name="check" size={14} /> Configuration looks valid
          </span>
        )}
      </div>
    </div>
  );
}
