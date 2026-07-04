import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import type {
  AntennaSpec,
  ChannelConfig,
  CodebookDetail,
  CodebookSummary,
  CompareRequest,
  CompareResult,
  CompareSharedConfig,
} from "../api/types";
import { api, ApiError } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { AntennaPicker } from "../components/forms/AntennaPicker";
import { ChannelPicker } from "../components/forms/ChannelPicker";
import { RankStepper, DropsSlider, SnrRange, SeedInput } from "../components/forms/EvaluationControls";
import { ParamField, isParamVisible } from "../components/forms/ParamField";
import { Chart } from "../components/Chart";
import { Panel } from "../components/results/Panel";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { Skeleton } from "../components/Skeleton";
import { Icon } from "../components/Icon";
import { fmtNum, fmtSE, fmtPct, fmtSeconds } from "../utils/format";
import { seriesColors, chartTextStyle, chartGrid, cssVar } from "../utils/chartTheme";
import { loadJson, saveJson, STORAGE_KEYS } from "../utils/storage";

// A generous ceiling (there are only ~12 codebooks; duplicates with different
// params are allowed) that guards against runaway requests without limiting
// real use.
const MAX_SCHEMES = 24;

// Fallback shared-antenna geometries shown before any codebook's supported
// geometries have loaded. Once schemes are added, the real options are the
// intersection of what every added codebook supports (see sharedAntennaSpec).
const FALLBACK_ANTENNA: AntennaSpec = {
  mode: "multi",
  pairs: [
    { n1: 2, n2: 1, ports: 4 },
    { n1: 4, n2: 1, ports: 8 },
    { n1: 2, n2: 2, ports: 8 },
    { n1: 4, n2: 2, ports: 16 },
    { n1: 8, n2: 2, ports: 32 },
    { n1: 4, n2: 4, ports: 32 },
  ],
};

const antennaKey = (p: { n1: number; n2: number; ng?: number | null }) =>
  `${p.ng ?? 1}-${p.n1}-${p.n2}`;

const DEFAULT_SHARED: CompareSharedConfig = {
  antenna: { n1: 4, n2: 2, ng: 1 },
  n3: 8,
  rank: 1,
  channel: { preset: "sparse-urban", n_rx: 2, n_paths: 4, max_delay: 3.0, max_doppler: 0.0 },
  snr_db: [-10, -5, 0, 5, 10, 15, 20, 25, 30],
  drops: 16,
  seed: 0,
};

interface SchemeEntry {
  uid: string;
  codebook_id: string;
  label: string;
  params: Record<string, unknown>;
}

interface PersistedState {
  schemes: SchemeEntry[];
  shared: CompareSharedConfig;
}

function defaultParams(detail: CodebookDetail): Record<string, unknown> {
  const p: Record<string, unknown> = {};
  for (const spec of detail.params ?? []) p[spec.key] = spec.default;
  return p;
}

let uidCounter = 0;
const nextUid = () => `s${Date.now().toString(36)}${uidCounter++}`;

export default function Compare() {
  const list = useApiData<CodebookSummary[]>(() => api.codebooks(), []);

  const persisted = useMemo(() => loadJson<PersistedState>(STORAGE_KEYS.compareState), []);
  const [schemes, setSchemes] = useState<SchemeEntry[]>(persisted?.schemes ?? []);
  const [shared, setShared] = useState<CompareSharedConfig>(persisted?.shared ?? DEFAULT_SHARED);
  const [details, setDetails] = useState<Record<string, CodebookDetail>>({});
  const [openUid, setOpenUid] = useState<string | null>(null);
  const [adding, setAdding] = useState("");

  const [result, setResult] = useState<CompareResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Persist config so a comparison survives a reload.
  useEffect(() => {
    saveJson<PersistedState>(STORAGE_KEYS.compareState, { schemes, shared });
  }, [schemes, shared]);

  // Lazily fetch the detail (params) for any scheme's codebook we haven't cached.
  useEffect(() => {
    const missing = Array.from(new Set(schemes.map((s) => s.codebook_id))).filter((id) => !details[id]);
    for (const id of missing) {
      api
        .codebook(id)
        .then((d) => setDetails((prev) => ({ ...prev, [id]: d })))
        .catch(() => undefined);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schemes]);

  const colors = seriesColors();

  // Shared-antenna options = the geometries EVERY added codebook supports.
  // Single-panel and multi-panel codebooks have disjoint supported arrays, so
  // mixing them yields an empty set (surfaced as a clear message below) rather
  // than a raw "requires Ng in {2,4}" error.
  const loadedDetails = schemes
    .map((s) => details[s.codebook_id])
    .filter((d): d is CodebookDetail => Boolean(d));
  const detailsPending = loadedDetails.length < schemes.length;

  const sharedAntennaSpec = useMemo<AntennaSpec>(() => {
    if (loadedDetails.length === 0) return FALLBACK_ANTENNA;
    let common: typeof FALLBACK_ANTENNA.pairs | null = null;
    for (const d of loadedDetails) {
      const pairs = d.antenna?.pairs ?? [];
      if (common === null) {
        common = [...pairs];
      } else {
        const keys = new Set(pairs.map(antennaKey));
        common = common.filter((p) => keys.has(antennaKey(p)));
      }
    }
    return { mode: "multi", pairs: common ?? [] };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(loadedDetails.map((d) => d.id))]);

  const antennaIncompatible = schemes.length > 0 && !detailsPending && sharedAntennaSpec.pairs.length === 0;

  // Keep the shared antenna valid for the current scheme set: if the selection
  // is no longer supported by all schemes, snap to a compatible ~16-port one.
  useEffect(() => {
    const pairs = sharedAntennaSpec.pairs;
    if (pairs.length === 0) return;
    if (pairs.some((p) => antennaKey(p) === antennaKey(shared.antenna))) return;
    const pref = pairs.reduce((b, p) => (Math.abs(p.ports - 16) < Math.abs(b.ports - 16) ? p : b), pairs[0]);
    setShared((s) => ({ ...s, antenna: { n1: pref.n1, n2: pref.n2, ng: pref.ng ?? 1 } }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sharedAntennaSpec]);

  function addScheme(codebookId: string) {
    if (!codebookId || schemes.length >= MAX_SCHEMES) return;
    const summary = list.data?.find((c) => c.id === codebookId);
    const uid = nextUid();
    // Seed params from the cached detail if available; otherwise fetch and fill.
    const cached = details[codebookId];
    const entry: SchemeEntry = {
      uid,
      codebook_id: codebookId,
      label: summary?.shortName || summary?.name || codebookId,
      params: cached ? defaultParams(cached) : {},
    };
    setSchemes((prev) => [...prev, entry]);
    setAdding("");
    if (!cached) {
      api
        .codebook(codebookId)
        .then((d) => {
          setDetails((prev) => ({ ...prev, [codebookId]: d }));
          setSchemes((prev) =>
            prev.map((s) => (s.uid === uid ? { ...s, params: defaultParams(d) } : s)),
          );
        })
        .catch(() => undefined);
    }
  }

  function updateScheme(uid: string, patch: Partial<SchemeEntry>) {
    setSchemes((prev) => prev.map((s) => (s.uid === uid ? { ...s, ...patch } : s)));
  }

  function removeScheme(uid: string) {
    setSchemes((prev) => prev.filter((s) => s.uid !== uid));
    if (openUid === uid) setOpenUid(null);
  }

  async function run() {
    if (schemes.length === 0) return;
    setRunning(true);
    setError(null);
    const req: CompareRequest = {
      schemes: schemes.map((s) => ({ codebook_id: s.codebook_id, params: s.params, label: s.label })),
      shared,
    };
    try {
      const res = await api.compare(req);
      setResult(res);
      if (!res.ok && res.error) setError(res.error);
    } catch (err) {
      setResult(null);
      setError(err instanceof ApiError ? err.message : "The comparison failed unexpectedly.");
    } finally {
      setRunning(false);
    }
  }

  const available = (list.data ?? []).filter(() => true);

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Compare codebooks</h1>
        <p className="page-subtitle">
          Put as many codebooks as you like on the same channel and antenna and see the trade-off
          between feedback cost and accuracy side by side.
        </p>
      </div>

      <section className="card" style={{ marginBottom: 24 }}>
        <h3 className="card-title">Schemes</h3>
        <p className="panel-subtitle" style={{ marginBottom: 16 }}>
          Add codebooks to compare. Click a scheme to rename it or change its parameters.
        </p>

        <div className="row row-gap-2 row-wrap" style={{ marginBottom: 16 }}>
          {schemes.length === 0 && (
            <span className="text-sm text-muted">No schemes added yet.</span>
          )}
          {schemes.map((s, i) => {
            const res = result?.results?.find((r) => r.label === s.label);
            const failed = res && !res.ok;
            return (
              <div className="popover-anchor" key={s.uid}>
                <div
                  className="compare-chip"
                  style={{ borderColor: failed ? "var(--danger)" : "var(--border)" }}
                >
                  <span
                    aria-hidden
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background: colors[i % colors.length],
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ textDecoration: failed ? "line-through" : "none" }}>{s.label}</span>
                  <button
                    type="button"
                    className="btn-icon btn-sm"
                    aria-label="Edit scheme"
                    onClick={() => setOpenUid(openUid === s.uid ? null : s.uid)}
                  >
                    <Icon name="expand" size={13} />
                  </button>
                  <button
                    type="button"
                    className="btn-icon btn-sm"
                    aria-label="Remove scheme"
                    onClick={() => removeScheme(s.uid)}
                  >
                    <Icon name="close" size={13} />
                  </button>
                </div>
                {openUid === s.uid && (
                  <SchemePopover
                    entry={s}
                    detail={details[s.codebook_id]}
                    onChange={(patch) => updateScheme(s.uid, patch)}
                    onClose={() => setOpenUid(null)}
                  />
                )}
                {failed && res?.error && (
                  <div className="text-sm" style={{ color: "var(--danger)", marginTop: 4, maxWidth: 220 }}>
                    {res.error}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {schemes.length < MAX_SCHEMES && (
          <div className="row row-gap-2">
            <select
              className="select"
              value={adding}
              onChange={(e) => {
                setAdding(e.target.value);
                if (e.target.value) addScheme(e.target.value);
              }}
              style={{ maxWidth: 320 }}
              disabled={list.loading}
            >
              <option value="">+ Add a codebook…</option>
              {available.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.release} · {c.name}
                </option>
              ))}
            </select>
          </div>
        )}
      </section>

      {antennaIncompatible && (
        <div style={{ marginBottom: 16 }}>
          <ErrorBanner
            title="These codebooks don't share an antenna"
            message="The selected codebooks support different antenna array types (for example, multi-panel codebooks need a multi-panel array that single-panel codebooks can't use), so there's no single geometry to compare them on."
            hint="Compare multi-panel codebooks with each other, and single-panel codebooks with each other."
          />
        </div>
      )}

      <SharedConfigForm spec={sharedAntennaSpec} shared={shared} onChange={setShared} />

      <div className="row row-gap-3" style={{ margin: "20px 0 8px" }}>
        <button
          className="btn btn-primary"
          onClick={run}
          disabled={running || schemes.length === 0 || antennaIncompatible}
        >
          {running ? <Icon name="spinner" className="spin" /> : <Icon name="play" />}
          {running ? "Comparing…" : `Compare ${schemes.length || ""} scheme${schemes.length === 1 ? "" : "s"}`.trim()}
        </button>
        {result?.ok && !running && (
          <span className="text-sm text-muted row row-gap-2">
            <Icon name="check" size={14} /> Done in {fmtSeconds(result.seconds)}
          </span>
        )}
      </div>

      {error && (
        <div style={{ marginTop: 16 }}>
          <ErrorBanner message={error} title="Comparison problem" hint="Adjust the schemes or shared settings and try again." />
        </div>
      )}

      {running && (
        <div className="col col-gap-3" style={{ marginTop: 24 }}>
          <Skeleton height={180} width="100%" />
          <Skeleton height={280} width="100%" />
        </div>
      )}

      {!running && result?.results && result.results.length > 0 && (
        <CompareResults results={result.results} snrDb={shared.snr_db} colors={colors} />
      )}

      {!running && !result && !error && schemes.length === 0 && (
        <div style={{ marginTop: 24 }}>
          <EmptyState
            icon="compare"
            title="Nothing to compare yet"
            body="Add two or more codebooks above, choose a shared channel and antenna, then run the comparison."
          />
        </div>
      )}
    </div>
  );
}

function SchemePopover({
  entry,
  detail,
  onChange,
  onClose,
}: {
  entry: SchemeEntry;
  detail: CodebookDetail | undefined;
  onChange: (patch: Partial<SchemeEntry>) => void;
  onClose: () => void;
}) {
  const visibleParams = (detail?.params ?? []).filter((p) => isParamVisible(p, entry.params));
  return (
    <div className="popover" role="dialog" aria-label="Edit scheme">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <strong className="text-sm">Edit scheme</strong>
        <button type="button" className="btn-icon btn-sm" aria-label="Close" onClick={onClose}>
          <Icon name="close" size={13} />
        </button>
      </div>
      <div className="field">
        <label className="field-label">Label</label>
        <input
          className="input"
          value={entry.label}
          onChange={(e) => onChange({ label: e.target.value })}
        />
      </div>
      {!detail && <div className="text-sm text-muted">Loading parameters…</div>}
      {detail && visibleParams.length === 0 && (
        <div className="text-sm text-muted">This codebook has no adjustable parameters.</div>
      )}
      {detail &&
        visibleParams.map((spec) => (
          <ParamField
            key={spec.key}
            spec={spec}
            value={entry.params[spec.key]}
            onChange={(key, value) => onChange({ params: { ...entry.params, [key]: value } })}
          />
        ))}
    </div>
  );
}

function SharedConfigForm({
  spec,
  shared,
  onChange,
}: {
  spec: AntennaSpec;
  shared: CompareSharedConfig;
  onChange: (c: CompareSharedConfig) => void;
}) {
  return (
    <section className="card">
      <h3 className="card-title">Shared settings</h3>
      <p className="panel-subtitle" style={{ marginBottom: 16 }}>
        Every scheme is evaluated on this same antenna, channel, and SNR sweep, so the curves are
        directly comparable.
      </p>

      <div className="field">
        <label className="field-label">Antenna array</label>
        {spec.pairs.length === 0 ? (
          <div className="text-sm text-muted">
            No antenna geometry is shared by all selected codebooks — see the note above.
          </div>
        ) : (
          <AntennaPicker
            spec={spec}
            value={shared.antenna}
            onChange={(pair) =>
              onChange({ ...shared, antenna: { n1: pair.n1, n2: pair.n2, ng: pair.ng ?? 1 } })
            }
          />
        )}
      </div>

      <div className="field" style={{ marginTop: 8 }}>
        <label className="field-label">Channel</label>
        <ChannelPicker
          value={shared.channel}
          onChange={(channel) => onChange({ ...shared, channel: { ...channel, n_rx: shared.channel.n_rx } })}
          nRx={shared.channel.n_rx ?? 2}
          onNRxChange={(n_rx) => onChange({ ...shared, channel: { ...shared.channel, n_rx } })}
        />
      </div>

      <div className="grid-3" style={{ marginTop: 8 }}>
        <div className="field">
          <label className="field-label">Rank (layers)</label>
          <RankStepper value={shared.rank} min={1} max={8} onChange={(rank) => onChange({ ...shared, rank })} />
        </div>
        <div className="field">
          <label className="field-label" htmlFor="cmp-n3">Frequency units (N3)</label>
          <input
            id="cmp-n3"
            className="input"
            type="number"
            min={1}
            max={64}
            value={shared.n3}
            onChange={(e) => onChange({ ...shared, n3: parseInt(e.target.value, 10) || 1 })}
          />
        </div>
        <div className="field">
          <label className="field-label">Seed</label>
          <SeedInput value={shared.seed} onChange={(seed) => onChange({ ...shared, seed })} />
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 4 }}>
        <div className="field">
          <label className="field-label">Monte-Carlo drops</label>
          <DropsSlider value={shared.drops} onChange={(drops) => onChange({ ...shared, drops })} />
        </div>
      </div>
      <div className="field">
        <label className="field-label">SNR sweep</label>
        <SnrRange value={shared.snr_db} onChange={(snr_db) => onChange({ ...shared, snr_db })} />
      </div>
    </section>
  );
}

function CompareResults({
  results,
  snrDb,
  colors,
}: {
  results: CompareResult["results"];
  snrDb: number[];
  colors: string[];
}) {
  const ok = results.filter((r) => r.ok);

  const scatterOption = useMemo<EChartsOption>(() => {
    return {
      textStyle: chartTextStyle(),
      tooltip: {
        trigger: "item",
        formatter: (p: unknown) => {
          const point = p as { data: [number, number]; name: string };
          return `${point.name}<br/>${fmtNum(point.data[0])} bits · SGCS ${fmtNum(point.data[1])}`;
        },
      },
      grid: chartGrid({ left: 56 }),
      xAxis: {
        type: "value",
        name: "Feedback bits",
        nameLocation: "middle",
        nameGap: 30,
        axisLabel: { color: cssVar("--text-muted") },
        splitLine: { lineStyle: { color: cssVar("--border") } },
      },
      yAxis: {
        type: "value",
        name: "SGCS (accuracy)",
        nameLocation: "middle",
        nameGap: 44,
        max: 1,
        axisLabel: { color: cssVar("--text-muted") },
        splitLine: { lineStyle: { color: cssVar("--border") } },
      },
      series: [
        {
          type: "scatter",
          symbolSize: 16,
          data: ok.map((r, i) => ({
            name: r.label,
            value: [r.total_bits ?? 0, r.sgcs ?? 0],
            itemStyle: { color: colors[results.indexOf(r) % colors.length] },
          })),
          label: {
            show: true,
            position: "top",
            formatter: (p: unknown) => (p as { name: string }).name,
            color: cssVar("--text-secondary"),
            fontSize: 11,
          },
        },
      ],
    };
  }, [ok, results, colors]);

  const barOption = useMemo<EChartsOption>(() => {
    return {
      textStyle: chartTextStyle(),
      tooltip: { trigger: "axis", valueFormatter: (v) => `${Number(v).toFixed(3)} bit/s/Hz` },
      legend: { data: ["Achieved @10 dB", "Eigen bound @10 dB"], top: 0, textStyle: chartTextStyle() },
      grid: chartGrid({ bottom: 60 }),
      xAxis: {
        type: "category",
        data: ok.map((r) => r.label),
        axisLabel: { color: cssVar("--text-muted"), interval: 0, rotate: ok.length > 3 ? 20 : 0 },
        axisLine: { lineStyle: { color: cssVar("--border-strong") } },
      },
      yAxis: {
        type: "value",
        name: "bit/s/Hz",
        nameLocation: "middle",
        nameGap: 40,
        axisLabel: { color: cssVar("--text-muted") },
        splitLine: { lineStyle: { color: cssVar("--border") } },
      },
      series: [
        {
          name: "Achieved @10 dB",
          type: "bar",
          data: ok.map((r, i) => ({
            value: r.se_at_10db ?? 0,
            itemStyle: { color: colors[results.indexOf(r) % colors.length] },
          })),
        },
        {
          name: "Eigen bound @10 dB",
          type: "bar",
          data: ok.map((r) => r.bound_at_10db ?? 0),
          itemStyle: { color: cssVar("--series-bound"), opacity: 0.5 },
        },
      ],
    };
  }, [ok, results, colors]);

  const overlayOption = useMemo<EChartsOption>(() => {
    return {
      textStyle: chartTextStyle(),
      tooltip: { trigger: "axis", valueFormatter: (v) => `${Number(v).toFixed(3)} bit/s/Hz` },
      legend: { data: ok.map((r) => r.label), top: 0, type: "scroll", textStyle: chartTextStyle() },
      grid: chartGrid({ top: 36 }),
      xAxis: {
        type: "category",
        data: (ok[0]?.snr_db ?? snrDb).map((v) => v.toFixed(0)),
        name: "SNR (dB)",
        nameLocation: "middle",
        nameGap: 28,
        axisLabel: { color: cssVar("--text-muted") },
        axisLine: { lineStyle: { color: cssVar("--border-strong") } },
      },
      yAxis: {
        type: "value",
        name: "bit/s/Hz",
        nameLocation: "middle",
        nameGap: 40,
        axisLabel: { color: cssVar("--text-muted") },
        splitLine: { lineStyle: { color: cssVar("--border") } },
      },
      series: ok.map((r) => ({
        name: r.label,
        type: "line",
        data: r.se ?? [],
        symbolSize: 5,
        lineStyle: { color: colors[results.indexOf(r) % colors.length], width: 2 },
        itemStyle: { color: colors[results.indexOf(r) % colors.length] },
      })),
    };
  }, [ok, snrDb, results, colors]);

  return (
    <div className="col col-gap-4" style={{ marginTop: 24 }}>
      <Panel
        title="Summary"
        subtitle="Accuracy and cost for each scheme on the shared channel. SGCS is how faithfully the reported precoder matches the ideal one (1.0 = perfect)."
      >
        <div style={{ overflowX: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th>Scheme</th>
                <th>Codebook</th>
                <th style={{ textAlign: "right" }}>SGCS</th>
                <th style={{ textAlign: "right" }}>Subspace SGCS</th>
                <th style={{ textAlign: "right" }}>Feedback bits</th>
                <th style={{ textAlign: "right" }}>SE @10 dB</th>
                <th style={{ textAlign: "right" }}>% of bound</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => {
                const pct = r.se_at_10db != null && r.bound_at_10db ? r.se_at_10db / r.bound_at_10db : null;
                return (
                  <tr key={`${r.label}-${i}`}>
                    <td>
                      <span className="row row-gap-2">
                        {r.ok && (
                          <span
                            aria-hidden
                            style={{
                              width: 9,
                              height: 9,
                              borderRadius: "50%",
                              background: colors[i % colors.length],
                            }}
                          />
                        )}
                        <span style={{ textDecoration: r.ok ? "none" : "line-through" }}>{r.label}</span>
                      </span>
                    </td>
                    <td className="text-sm text-muted">
                      {r.ok ? r.scheme_name ?? "—" : <span style={{ color: "var(--danger)" }}>{r.error}</span>}
                    </td>
                    <td className="tabular-nums" style={{ textAlign: "right" }}>{r.ok ? fmtNum(r.sgcs) : "—"}</td>
                    <td className="tabular-nums" style={{ textAlign: "right" }}>{r.ok ? fmtNum(r.subspace_sgcs) : "—"}</td>
                    <td className="tabular-nums" style={{ textAlign: "right" }}>{r.ok ? fmtNum(r.total_bits) : "—"}</td>
                    <td className="tabular-nums" style={{ textAlign: "right" }}>{r.ok ? fmtSE(r.se_at_10db) : "—"}</td>
                    <td className="tabular-nums" style={{ textAlign: "right" }}>{pct != null ? fmtPct(pct) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      {ok.length > 0 && (
        <Panel
          title="Cost vs accuracy"
          subtitle="Feedback bits (horizontal) against reconstruction accuracy (vertical). The ideal scheme sits toward the top-left: high accuracy for few bits."
          about="Each dot is one scheme. Moving right costs more uplink feedback; moving up means the reported precoder is closer to the ideal eigen-beamformer. This is the rate–distortion trade-off at the heart of codebook design."
        >
          <Chart option={scatterOption} height={340} ariaLabel="Feedback bits versus SGCS accuracy" />
        </Panel>
      )}

      {ok.length > 0 && (
        <Panel
          title="Spectral efficiency at 10 dB"
          subtitle="Achieved SE per scheme at 10 dB SNR, next to the ideal-eigenvector upper bound."
          about="The colored bar is what each codebook actually delivers; the faint bar is the ceiling if the base station knew the exact channel. A taller colored bar (closer to the faint one) is better."
        >
          <Chart option={barOption} height={320} ariaLabel="Spectral efficiency at 10 dB by scheme" />
        </Panel>
      )}

      {ok.length > 0 && (
        <Panel
          title="Spectral efficiency vs SNR"
          subtitle="The full SE-vs-SNR curve for every scheme, overlaid."
          about="Each line is one scheme across the SNR sweep. Lines that sit higher deliver more throughput at the same signal strength."
        >
          <Chart option={overlayOption} height={340} ariaLabel="Spectral efficiency versus SNR overlay" />
        </Panel>
      )}
    </div>
  );
}
