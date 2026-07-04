import { useMemo, useState } from "react";
import type { CodebookDetail, RunRequest, RunResult } from "../../api/types";
import { api, ApiError } from "../../api/client";
import { PlaygroundForm, buildDefaultRequest } from "../../components/forms/PlaygroundForm";
import { ResultsDashboard } from "../../components/results/ResultsDashboard";
import { ErrorBanner } from "../../components/ErrorBanner";
import { EmptyState } from "../../components/EmptyState";
import { Icon } from "../../components/Icon";
import { loadJson, saveJson } from "../../utils/storage";
import { fmtNum } from "../../utils/format";

interface RunTabProps {
  detail: CodebookDetail;
  /** localStorage key for persisting the last request (playground only). */
  persistKey?: string;
}

/** Plain-language field descriptions keyed by PMI field name, for the
 * overhead-breakdown hover text. Derived from the curated content. */
function buildFieldDescriptions(detail: CodebookDetail): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of detail.content?.whatIsReported ?? []) {
    if (f.field) out[f.field] = f.plain;
  }
  return out;
}

/** The "Run" tab: the guided playground form + results dashboard, scoped to a
 * single codebook. Reused verbatim by the standalone Playground page. */
export function RunTab({ detail, persistKey }: RunTabProps) {
  const [request, setRequest] = useState<RunRequest>(() => {
    if (persistKey) {
      const saved = loadJson<RunRequest>(persistKey);
      if (saved && saved.codebook_id === detail.id) return saved;
    }
    return buildDefaultRequest(detail);
  });
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const fieldDescriptions = useMemo(() => buildFieldDescriptions(detail), [detail]);
  const estSeconds = Math.max(1, Math.round(request.drops * 0.25));

  async function run(req: RunRequest) {
    setRunning(true);
    setError(null);
    if (persistKey) saveJson(persistKey, req);
    const t0 = performance.now();
    setElapsed(0);
    const timer = window.setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    try {
      const res = await api.run(req);
      setResult(res);
      setError(res.ok ? null : res.error ?? "The run did not complete.");
    } catch (err) {
      setResult(null);
      setError(err instanceof ApiError ? err.message : "The run failed unexpectedly.");
    } finally {
      window.clearInterval(timer);
      setRunning(false);
    }
  }

  return (
    <div className="run-layout">
      <div className="run-form">
        <PlaygroundForm
          detail={detail}
          request={request}
          onChange={setRequest}
          onRun={run}
          running={running}
        />
      </div>

      <div className="run-results">
        {running && (
          <div className="panel run-progress">
            <Icon name="spinner" className="spin" />
            <div>
              <div style={{ fontWeight: 600 }}>Running {detail.shortName || detail.name}…</div>
              <div className="text-sm text-muted tabular-nums">
                {fmtNum(elapsed)} s elapsed · about {estSeconds}s for {request.drops} drops
              </div>
            </div>
          </div>
        )}

        {!running && error && (
          <ErrorBanner
            message={error}
            title="This run could not complete"
            hint="Adjust the codebook or channel settings on the left and run again."
          />
        )}

        {!running && !error && result?.ok && (
          <ResultsDashboard result={result} fieldDescriptions={fieldDescriptions} />
        )}

        {!running && !error && !result && (
          <EmptyState
            icon="playground"
            title="No results yet"
            body="Configure the codebook, antenna, channel, and evaluation on the left, then press Run to see spectral efficiency, feedback overhead, and the reconstructed precoder."
          />
        )}
      </div>
    </div>
  );
}
