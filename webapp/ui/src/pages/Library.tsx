import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useApiData } from "../hooks/useApiData";
import { api } from "../api/client";
import type { CodebookSummary } from "../api/types";
import { SkeletonGrid } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";
import { EmptyState } from "../components/EmptyState";
import { ReleaseBadge } from "../components/Badge";

const RELEASE_ORDER = ["R15", "R16", "R17", "R18", "R19"];

export default function Library() {
  const { data, loading, error } = useApiData<CodebookSummary[]>(() => api.codebooks(), []);

  const grouped = useMemo(() => {
    const map = new Map<string, CodebookSummary[]>();
    for (const cb of data ?? []) {
      const key = cb.release || "Other";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(cb);
    }
    for (const arr of map.values()) arr.sort((a, b) => (a.position ?? 0) - (b.position ?? 0));
    return map;
  }, [data]);

  const orderedReleases = useMemo(() => {
    const keys = Array.from(grouped.keys());
    return keys.sort((a, b) => {
      const ia = RELEASE_ORDER.indexOf(a);
      const ib = RELEASE_ORDER.indexOf(b);
      return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    });
  }, [grouped]);

  const lineageChain = useMemo(() => {
    return [...(data ?? [])].sort((a, b) => (a.position ?? 0) - (b.position ?? 0));
  }, [data]);

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Codebook library</h1>
        <p className="page-subtitle">
          Every PMI codebook implemented in this project, grouped by 3GPP release. Each one builds on the
          ideas of the release before it.
        </p>
      </div>

      {!loading && !error && lineageChain.length > 1 && (
        <div className="card card-tight" style={{ marginBottom: 32, overflowX: "auto" }}>
          <div className="row row-gap-2" style={{ minWidth: "max-content" }}>
            {lineageChain.map((cb, i) => (
              <span key={cb.id} className="row row-gap-2">
                <Link to={`/codebooks/${cb.id}`} className="chip" style={{ textDecoration: "none" }}>
                  <ReleaseBadge release={cb.release} />
                  {cb.shortName || cb.name}
                </Link>
                {i < lineageChain.length - 1 && <span className="text-muted">&rarr;</span>}
              </span>
            ))}
          </div>
        </div>
      )}

      {loading && <SkeletonGrid count={8} />}
      {!loading && error && (
        <ErrorBanner message={error} hint="Check that the backend server is running on port 8787." />
      )}
      {!loading && !error && (data?.length ?? 0) === 0 && (
        <EmptyState title="No codebooks found" body="The catalog appears to be empty." />
      )}

      {!loading &&
        !error &&
        orderedReleases.map((release) => (
          <section key={release} style={{ marginBottom: 40 }}>
            <div className="row row-gap-2" style={{ marginBottom: 16 }}>
              <ReleaseBadge release={release} />
              <h2 style={{ fontSize: "var(--fs-lg)", fontWeight: 600 }}>{release}</h2>
            </div>
            <div className="card-grid">
              {grouped.get(release)!.map((cb) => (
                <CodebookCard key={cb.id} cb={cb} />
              ))}
            </div>
          </section>
        ))}
    </div>
  );
}

function CodebookCard({ cb }: { cb: CodebookSummary }) {
  return (
    <Link to={`/codebooks/${cb.id}`} className="card card-hover" style={{ textDecoration: "none", color: "inherit" }}>
      <div className="row row-gap-2" style={{ marginBottom: 10, justifyContent: "space-between" }}>
        <ReleaseBadge release={cb.release} />
        <span className="text-sm text-muted">{cb.specClause}</span>
      </div>
      <div className="card-title" style={{ marginBottom: 6 }}>
        {cb.name}
      </div>
      <p className="text-secondary text-sm" style={{ lineHeight: 1.6, marginBottom: 12 }}>
        {cb.tagline || "No description yet."}
      </p>
      <div className="row row-gap-2 row-wrap">
        <span className="chip">
          rank {cb.ranks?.[0] ?? "?"}&ndash;{cb.ranks?.[1] ?? "?"}
        </span>
        <span className="chip">
          {cb.portRange?.[0] ?? "?"}&ndash;{cb.portRange?.[1] ?? "?"} ports
        </span>
      </div>
      {cb.lineage?.adds && (
        <div className="text-sm text-muted mt-3" style={{ borderTop: "1px solid var(--border)", paddingTop: 10 }}>
          Adds over {cb.lineage.parent ?? "parent"}: {cb.lineage.adds}
        </div>
      )}
    </Link>
  );
}
