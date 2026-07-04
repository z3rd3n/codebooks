import type { RunMetrics } from "../../api/types";
import { fmtBits, fmtNum, fmtPct } from "../../utils/format";

interface MetricCardsProps {
  metrics: RunMetrics;
}

function seAt10db(snr: number[], se: number[]): number | null {
  if (!snr?.length || !se?.length) return null;
  let bestIdx = 0;
  let bestDiff = Infinity;
  snr.forEach((v, i) => {
    const diff = Math.abs(v - 10);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestIdx = i;
    }
  });
  return se[bestIdx] ?? null;
}

export function MetricCards({ metrics }: MetricCardsProps) {
  const se10 = seAt10db(metrics.snr_db ?? [], metrics.se ?? []);
  const bound10 = seAt10db(metrics.snr_db ?? [], metrics.se_upper_bound ?? []);
  const pctOfBound = se10 !== null && bound10 ? se10 / bound10 : null;

  return (
    <div className="metric-grid">
      <div className="metric-card">
        <div className="metric-label">SGCS</div>
        <div className="metric-value tabular-nums">{fmtNum(metrics.sgcs)}</div>
        <div className="metric-sub">Cosine similarity to ideal precoder</div>
      </div>
      <div className="metric-card">
        <div className="metric-label">Subspace SGCS</div>
        <div className="metric-value tabular-nums">{fmtNum(metrics.subspace_sgcs)}</div>
        <div className="metric-sub">Multi-layer subspace fidelity</div>
      </div>
      <div className="metric-card">
        <div className="metric-label">Feedback bits</div>
        <div className="metric-value tabular-nums">{fmtBits(metrics.total_bits)}</div>
        <div className="metric-sub">Total PMI report size</div>
      </div>
      <div className="metric-card">
        <div className="metric-label">SE @ 10 dB</div>
        <div className="metric-value tabular-nums">{fmtNum(se10)}</div>
        <div className="metric-sub">
          bit/s/Hz{pctOfBound !== null ? ` — ${fmtPct(pctOfBound)} of eigen bound` : ""}
        </div>
      </div>
    </div>
  );
}
