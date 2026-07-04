import { useEffect, useMemo, useRef, useState } from "react";
import type { AntennaSpec, FigureInfo, FigureJobResult, FiguresRunRequest, JobStatus, Meta } from "../api/types";
import { api, ApiError } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { AntennaPicker } from "../components/forms/AntennaPicker";
import { SkeletonGrid } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";
import { EmptyState } from "../components/EmptyState";
import { Icon } from "../components/Icon";
import { fmtSeconds } from "../utils/format";

// Display names must match the backend's registry.FAMILIES keys.
const FAMILY_NAMES = [
  "Type I (R15)",
  "Type II (R15)",
  "eType II (R16)",
  "feType II PS (R17)",
  "eType II Doppler (R18)",
];
const CDL_MODELS = ["A", "B", "C", "D", "E"];

interface FigConfig {
  channel: "synthetic" | "cdl";
  families: string[];
  n1: number;
  n2: number;
  n3: number;
  n_rx: number;
  n_paths: number;
  max_delay: number;
  cdl_model: string;
  cdl_speed: number;
  cdl_delay_spread_ns: number;
  fast: boolean;
  drops: number;
  seed: number;
}

const DEFAULT_CONFIG: FigConfig = {
  channel: "synthetic",
  families: [...FAMILY_NAMES],
  n1: 4,
  n2: 2,
  n3: 8,
  n_rx: 2,
  n_paths: 4,
  max_delay: 3.0,
  cdl_model: "C",
  cdl_speed: 3.0,
  cdl_delay_spread_ns: 100.0,
  fast: true,
  drops: 100,
  seed: 0,
};

export default function FigureLab() {
  const figures = useApiData<FigureInfo[]>(() => api.figures(), []);
  const meta = useApiData<Meta>(() => api.meta(), []);
  const sionnaAvailable = meta.data?.sionna_available ?? false;

  const [selected, setSelected] = useState<Set<string>>(new Set(["fig_01_se_vs_snr"]));
  const [config, setConfig] = useState<FigConfig>(DEFAULT_CONFIG);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<string | null>(null);
  const [dataModal, setDataModal] = useState<FigureJobResult | null>(null);

  const figuresBySlug = useMemo(() => {
    const m = new Map<string, FigureInfo>();
    for (const f of figures.data ?? []) m.set(f.slug, f);
    return m;
  }, [figures.data]);

  const running = job != null && (job.status === "queued" || job.status === "running");

  // A figure is unavailable when the CDL channel is chosen but it has no CDL twin.
  const isBlocked = (slug: string) =>
    config.channel === "cdl" && figuresBySlug.get(slug)?.cdlAvailable === false;

  const antennaSpec: AntennaSpec = {
    mode: "multi",
    pairs: meta.data?.antennas ?? [{ n1: 4, n2: 2, ports: 16 }],
  };

  // Poll the job every second until it finishes.
  const jobRef = useRef<string | null>(null);
  jobRef.current = jobId;
  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer: number;
    const poll = async () => {
      try {
        const j = await api.job(jobId);
        if (!active || jobRef.current !== jobId) return;
        setJob(j);
        if (j.status === "done" || j.status === "error") return;
      } catch {
        // transient poll error — keep trying
      }
      if (active) timer = window.setTimeout(poll, 1000);
    };
    timer = window.setTimeout(poll, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [jobId]);

  function toggle(slug: string) {
    if (isBlocked(slug)) return;
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) next.delete(slug);
      else next.add(slug);
      return next;
    });
  }

  function switchChannel(channel: "synthetic" | "cdl") {
    setConfig((c) => ({ ...c, channel }));
    if (channel === "cdl") {
      // Drop any selected figures that have no CDL variant.
      setSelected((prev) => {
        const next = new Set<string>();
        for (const slug of prev) {
          if (figuresBySlug.get(slug)?.cdlAvailable !== false) next.add(slug);
        }
        return next;
      });
    }
  }

  function toggleFamily(name: string) {
    setConfig((c) => ({
      ...c,
      families: c.families.includes(name)
        ? c.families.filter((f) => f !== name)
        : [...c.families, name],
    }));
  }

  async function run() {
    if (selected.size === 0) return;
    setError(null);
    setJob(null);
    const body: FiguresRunRequest = {
      slugs: [...selected],
      channel: config.channel,
      families: config.families,
      antenna: { n1: config.n1, n2: config.n2 },
      n3: config.n3,
      n_rx: config.n_rx,
      n_paths: config.n_paths,
      max_delay: config.max_delay,
      cdl_model: config.cdl_model,
      cdl_speed: config.cdl_speed,
      cdl_delay_spread_ns: config.cdl_delay_spread_ns,
      drops: config.drops,
      seed: config.seed,
      fast: config.fast,
    };
    try {
      const { job_id } = await api.runFigures(body);
      setJobId(job_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start the figure job.");
    }
  }

  const est = useMemo(() => {
    let total = 0;
    for (const slug of selected) {
      const f = figuresBySlug.get(slug);
      if (f) total += config.fast ? Math.min(f.estSeconds, 10) : f.estSeconds;
    }
    return total;
  }, [selected, figuresBySlug, config.fast]);

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Figure Lab</h1>
        <p className="page-subtitle">
          Reproduce the benchmark figures from the project on a channel of your choosing. Pick the
          figures, set up the channel and antenna, and the app runs the exact scripts that generate
          them.
        </p>
      </div>

      <div className="figlab-grid">
        {/* Figure selection */}
        <section>
          <h3 className="card-title" style={{ marginBottom: 12 }}>Figures</h3>
          {figures.loading && <SkeletonGrid count={6} />}
          {figures.error && (
            <ErrorBanner message={figures.error} hint="Check that the backend is running on port 8787." />
          )}
          {!figures.loading && !figures.error && (
            <div className="col col-gap-2">
              {(figures.data ?? []).map((f) => {
                const on = selected.has(f.slug);
                const blocked = isBlocked(f.slug);
                return (
                  <button
                    key={f.slug}
                    type="button"
                    className={`figure-select-card${on ? " selected" : ""}${blocked ? " disabled" : ""}`}
                    onClick={() => toggle(f.slug)}
                    aria-pressed={on}
                    disabled={blocked}
                    title={blocked ? "No 3GPP CDL version of this figure — available on the synthetic channel only." : undefined}
                  >
                    <span className="figure-check">{on && <Icon name="check" size={13} />}</span>
                    <span style={{ flex: 1 }}>
                      <span className="row row-gap-2" style={{ justifyContent: "space-between" }}>
                        <strong className="text-sm">{f.title}</strong>
                        <span className="text-sm text-muted tabular-nums">~{f.estSeconds}s</span>
                      </span>
                      {f.question && <span className="figure-question">{f.question}</span>}
                      <span className="row row-gap-2 row-wrap" style={{ marginTop: 6 }}>
                        {f.honorsFamilies === false && (
                          <span className="chip chip-sm">ignores family selection</span>
                        )}
                        {blocked && <span className="chip chip-sm">synthetic only</span>}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </section>

        {/* Config panel */}
        <aside className="figlab-config">
          <section className="card">
            <h3 className="card-title">Channel</h3>
            <div className="segmented" role="tablist" style={{ marginBottom: 14 }}>
              <button
                role="tab"
                aria-selected={config.channel === "synthetic"}
                className={`segmented-btn${config.channel === "synthetic" ? " active" : ""}`}
                onClick={() => switchChannel("synthetic")}
              >
                Synthetic
              </button>
              <button
                role="tab"
                aria-selected={config.channel === "cdl"}
                className={`segmented-btn${config.channel === "cdl" ? " active" : ""}`}
                onClick={() => sionnaAvailable && switchChannel("cdl")}
                disabled={!sionnaAvailable}
                title={sionnaAvailable ? undefined : "Sionna (TensorFlow) is not installed. Install the optional sionna extra to use 3GPP CDL channels."}
              >
                3GPP CDL {!sionnaAvailable && <Icon name="info" size={12} />}
              </button>
            </div>

            {config.channel === "synthetic" ? (
              <div className="grid-2">
                <div className="field">
                  <label className="field-label">Multipath rays</label>
                  <input
                    className="input"
                    type="number"
                    min={1}
                    max={32}
                    value={config.n_paths}
                    onChange={(e) => setConfig((c) => ({ ...c, n_paths: parseInt(e.target.value, 10) || 1 }))}
                  />
                </div>
                <div className="field">
                  <label className="field-label">Max ray delay</label>
                  <input
                    className="input"
                    type="number"
                    step={0.5}
                    min={0}
                    value={config.max_delay}
                    onChange={(e) => setConfig((c) => ({ ...c, max_delay: parseFloat(e.target.value) || 0 }))}
                  />
                </div>
              </div>
            ) : (
              <div className="grid-2">
                <div className="field">
                  <label className="field-label">CDL model</label>
                  <select
                    className="select"
                    value={config.cdl_model}
                    onChange={(e) => setConfig((c) => ({ ...c, cdl_model: e.target.value }))}
                  >
                    {CDL_MODELS.map((m) => (
                      <option key={m} value={m}>CDL-{m}</option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label className="field-label">UE speed (km/h)</label>
                  <input
                    className="input"
                    type="number"
                    min={0}
                    value={config.cdl_speed}
                    onChange={(e) => setConfig((c) => ({ ...c, cdl_speed: parseFloat(e.target.value) || 0 }))}
                  />
                </div>
                <div className="field">
                  <label className="field-label">Delay spread (ns)</label>
                  <input
                    className="input"
                    type="number"
                    min={1}
                    value={config.cdl_delay_spread_ns}
                    onChange={(e) => setConfig((c) => ({ ...c, cdl_delay_spread_ns: parseFloat(e.target.value) || 1 }))}
                  />
                </div>
              </div>
            )}
            <div className="field" style={{ marginTop: 4 }}>
              <label className="field-label">Receive antennas (n_rx)</label>
              <input
                className="input"
                type="number"
                min={1}
                max={8}
                value={config.n_rx}
                onChange={(e) => setConfig((c) => ({ ...c, n_rx: parseInt(e.target.value, 10) || 1 }))}
                style={{ maxWidth: 120 }}
              />
            </div>
          </section>

          <section className="card">
            <h3 className="card-title">Antenna &amp; frequency</h3>
            <div className="field">
              <label className="field-label">Antenna array (N1 &times; N2)</label>
              <AntennaPicker
                spec={antennaSpec}
                value={{ n1: config.n1, n2: config.n2 }}
                onChange={(pair) => setConfig((c) => ({ ...c, n1: pair.n1, n2: pair.n2 }))}
              />
            </div>
            <div className="field" style={{ maxWidth: 160, marginTop: 8 }}>
              <label className="field-label">Frequency units (N3)</label>
              <input className="input" type="number" min={1} max={64} value={config.n3}
                onChange={(e) => setConfig((c) => ({ ...c, n3: parseInt(e.target.value, 10) || 1 }))} />
            </div>
          </section>

          <section className="card">
            <h3 className="card-title">Codebook families</h3>
            <p className="panel-subtitle" style={{ marginBottom: 12 }}>
              Which families to include (figures marked “ignores family selection” always show their own set).
            </p>
            <div className="col col-gap-2">
              {FAMILY_NAMES.map((name) => (
                <label key={name} className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={config.families.includes(name)}
                    onChange={() => toggleFamily(name)}
                  />
                  <span className="text-sm">{name}</span>
                </label>
              ))}
            </div>
          </section>

          <section className="card">
            <h3 className="card-title">Run settings</h3>
            <label className="checkbox-row" style={{ marginBottom: 12 }}>
              <input
                type="checkbox"
                checked={config.fast}
                onChange={(e) => setConfig((c) => ({ ...c, fast: e.target.checked }))}
              />
              <span className="text-sm">Fast preview (fewer drops — quick but noisier)</span>
            </label>
            <div className="grid-2">
              <div className="field">
                <label className="field-label">Drops</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  max={1000}
                  value={config.drops}
                  disabled={config.fast}
                  onChange={(e) => setConfig((c) => ({ ...c, drops: parseInt(e.target.value, 10) || 1 }))}
                />
              </div>
              <div className="field">
                <label className="field-label">Seed</label>
                <input
                  className="input"
                  type="number"
                  value={config.seed}
                  onChange={(e) => setConfig((c) => ({ ...c, seed: parseInt(e.target.value, 10) || 0 }))}
                />
              </div>
            </div>
          </section>

          <button className="btn btn-primary" onClick={run} disabled={running || selected.size === 0} style={{ width: "100%" }}>
            {running ? <Icon name="spinner" className="spin" /> : <Icon name="figures" />}
            {running ? "Generating…" : `Generate ${selected.size} figure${selected.size === 1 ? "" : "s"}`}
          </button>
          {selected.size > 0 && !running && (
            <div className="text-sm text-muted" style={{ marginTop: 8, textAlign: "center" }}>
              Estimated ~{est}s{config.channel === "cdl" ? " (plus TensorFlow load on first run)" : ""}
            </div>
          )}
        </aside>
      </div>

      {error && (
        <div style={{ marginTop: 24 }}>
          <ErrorBanner message={error} title="Could not start job" />
        </div>
      )}

      {job && (
        <section style={{ marginTop: 32 }}>
          <div className="row row-gap-3" style={{ justifyContent: "space-between", marginBottom: 12 }}>
            <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 650 }}>Results</h2>
            <span className="text-sm text-muted">{job.message}</span>
          </div>
          <div className="progress-track" aria-hidden>
            <div
              className="progress-fill"
              style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
            />
          </div>

          <div className="figure-gallery" style={{ marginTop: 20 }}>
            {(job.results ?? []).map((r) => (
              <FigureResultCard
                key={r.slug}
                res={r}
                info={figuresBySlug.get(r.slug)}
                onZoom={setLightbox}
                onData={setDataModal}
              />
            ))}
            {running && (job.results?.length ?? 0) < selected.size && (
              <div className="card figure-pending">
                <Icon name="spinner" className="spin" size={20} />
                <span className="text-sm text-muted">Waiting for remaining figures…</span>
              </div>
            )}
          </div>
        </section>
      )}

      {!job && !error && (
        <div style={{ marginTop: 32 }}>
          <EmptyState
            icon="figures"
            title="No figures generated yet"
            body="Select one or more figures on the left, configure the channel, and press Generate."
          />
        </div>
      )}

      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)} role="dialog" aria-label="Figure preview">
          <img src={lightbox} alt="Figure full size" />
        </div>
      )}

      {dataModal && <DataModal res={dataModal} info={figuresBySlug.get(dataModal.slug)} onClose={() => setDataModal(null)} />}
    </div>
  );
}

function FigureResultCard({
  res,
  info,
  onZoom,
  onData,
}: {
  res: FigureJobResult;
  info?: FigureInfo;
  onZoom: (src: string) => void;
  onData: (r: FigureJobResult) => void;
}) {
  return (
    <div className="card figure-card">
      <div className="row row-gap-2" style={{ justifyContent: "space-between", marginBottom: 10 }}>
        <strong className="text-sm">{info?.title ?? res.slug}</strong>
        {res.seconds != null && <span className="text-sm text-muted tabular-nums">{fmtSeconds(res.seconds)}</span>}
      </div>

      {res.ok && res.png_url ? (
        <>
          <button
            type="button"
            className="figure-thumb"
            onClick={() => onZoom(res.png_url!)}
            aria-label="Enlarge figure"
          >
            <img src={res.png_url} alt={info?.title ?? res.slug} loading="lazy" />
            <span className="figure-thumb-zoom"><Icon name="expand" size={16} /></span>
          </button>
          {info?.howToRead && (
            <p className="text-sm text-secondary" style={{ lineHeight: 1.6, marginTop: 12 }}>
              <strong>How to read it:</strong> {info.howToRead}
            </p>
          )}
          <div className="row row-gap-2" style={{ marginTop: 12, flexWrap: "wrap" }}>
            <a className="btn btn-sm" href={res.png_url} download>
              <Icon name="download" size={13} /> PNG
            </a>
            {res.json_url && (
              <a className="btn btn-sm" href={res.json_url} download>
                <Icon name="download" size={13} /> JSON
              </a>
            )}
            {res.data && (
              <button className="btn btn-sm btn-ghost" onClick={() => onData(res)}>
                <Icon name="grid" size={13} /> View data
              </button>
            )}
          </div>
        </>
      ) : (
        <div>
          <div className="error-banner" role="alert" style={{ marginBottom: 0 }}>
            <Icon name="warning" />
            <div className="error-banner-body">
              <div className="error-banner-title">This figure failed to generate</div>
              {res.error && <div className="text-sm">{res.error}</div>}
            </div>
          </div>
          {res.log && (
            <details className="about-toggle" style={{ marginTop: 10 }}>
              <summary><Icon name="info" size={14} /> Show log</summary>
              <pre className="log-block">{res.log}</pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

/** Best-effort flat table from a figure's JSON, else pretty JSON. */
function asTable(data: Record<string, unknown>): { cols: string[]; rows: (number | string)[][] } | null {
  const lists: Record<string, number[]> = {};
  for (const [k, v] of Object.entries(data)) {
    if (Array.isArray(v) && v.length > 0 && v.every((x) => typeof x === "number")) {
      lists[k] = v as number[];
    }
  }
  const cols = Object.keys(lists);
  if (cols.length === 0) return null;
  const lengths = new Set(cols.map((c) => lists[c].length));
  if (lengths.size !== 1) return null;
  const n = lists[cols[0]].length;
  const rows: (number | string)[][] = [];
  for (let i = 0; i < n; i++) rows.push(cols.map((c) => lists[c][i]));
  return { cols, rows };
}

function DataModal({ res, info, onClose }: { res: FigureJobResult; info?: FigureInfo; onClose: () => void }) {
  const table = res.data ? asTable(res.data) : null;
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Figure data">
        <div className="modal-head">
          <strong>{info?.title ?? res.slug} — data</strong>
          <button type="button" className="btn-icon" aria-label="Close" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body">
          {table ? (
            <div style={{ overflowX: "auto" }}>
              <table className="table">
                <thead>
                  <tr>{table.cols.map((c) => <th key={c}>{c}</th>)}</tr>
                </thead>
                <tbody>
                  {table.rows.map((row, i) => (
                    <tr key={i}>
                      {row.map((v, j) => (
                        <td key={j} className="tabular-nums">{typeof v === "number" ? v.toFixed(4) : v}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <pre className="log-block">{JSON.stringify(res.data, null, 2)}</pre>
          )}
        </div>
      </div>
    </div>
  );
}
