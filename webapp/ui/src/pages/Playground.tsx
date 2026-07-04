import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import type { CodebookDetail, CodebookSummary } from "../api/types";
import { api } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { RunTab } from "./codebook-tabs/RunTab";
import { ReleaseBadge } from "../components/Badge";
import { Skeleton, SkeletonGrid } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";
import { Icon } from "../components/Icon";
import { STORAGE_KEYS } from "../utils/storage";

export default function Playground() {
  const [params, setParams] = useSearchParams();
  const selectedId = params.get("codebook");

  const list = useApiData<CodebookSummary[]>(() => api.codebooks(), []);

  function select(id: string | null) {
    const next = new URLSearchParams(params);
    if (id) next.set("codebook", id);
    else next.delete("codebook");
    setParams(next, { replace: false });
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Playground</h1>
        <p className="page-subtitle">
          Pick a codebook, describe an antenna and a radio channel, and run it. You&apos;ll see how
          accurately it reconstructs the channel, how many feedback bits it costs, and exactly what
          the phone reports — no coding required.
        </p>
      </div>

      {!selectedId && (
        <CodebookPicker list={list} onSelect={select} />
      )}

      {selectedId && <SelectedRunner id={selectedId} onBack={() => select(null)} list={list.data ?? []} />}
    </div>
  );
}

function CodebookPicker({
  list,
  onSelect,
}: {
  list: ReturnType<typeof useApiData<CodebookSummary[]>>;
  onSelect: (id: string) => void;
}) {
  if (list.loading) return <SkeletonGrid count={9} />;
  if (list.error) return <ErrorBanner message={list.error} hint="Check that the backend is running on port 8787." />;

  const ordered = [...(list.data ?? [])].sort((a, b) => (a.position ?? 0) - (b.position ?? 0));

  return (
    <div className="card-grid">
      {ordered.map((cb) => (
        <button
          key={cb.id}
          type="button"
          className="card card-hover"
          style={{ textAlign: "left", cursor: "pointer" }}
          onClick={() => onSelect(cb.id)}
        >
          <div className="row row-gap-2" style={{ marginBottom: 10, justifyContent: "space-between" }}>
            <ReleaseBadge release={cb.release} />
            <span className="text-sm text-muted">{cb.specClause}</span>
          </div>
          <div className="card-title" style={{ marginBottom: 6 }}>{cb.name}</div>
          <p className="text-secondary text-sm" style={{ lineHeight: 1.6, marginBottom: 12 }}>
            {cb.tagline || "No description yet."}
          </p>
          <span className="row row-gap-2 text-sm" style={{ color: "var(--accent)", fontWeight: 600 }}>
            Configure &amp; run <Icon name="arrow-right" size={14} />
          </span>
        </button>
      ))}
    </div>
  );
}

function SelectedRunner({
  id,
  onBack,
  list,
}: {
  id: string;
  onBack: () => void;
  list: CodebookSummary[];
}) {
  const { data, loading, error } = useApiData<CodebookDetail>(() => api.codebook(id), [id]);
  const summary = useMemo(() => list.find((c) => c.id === id), [list, id]);

  return (
    <div>
      <button type="button" className="btn btn-ghost btn-sm" onClick={onBack} style={{ marginBottom: 16 }}>
        <Icon name="chevron-left" size={14} /> Choose a different codebook
      </button>

      {loading && (
        <div className="col col-gap-3">
          <Skeleton height={28} width="45%" />
          <Skeleton height={200} width="100%" />
        </div>
      )}

      {!loading && error && (
        <ErrorBanner message={error} hint="This codebook id may not exist. Go back and pick another." />
      )}

      {!loading && !error && data && (
        <>
          <header className="page-header" style={{ marginBottom: 20 }}>
            <div className="row row-gap-2" style={{ marginBottom: 10, flexWrap: "wrap" }}>
              <ReleaseBadge release={data.release} />
              <span className="chip">{data.specClause}</span>
              <span className="chip">
                rank {data.ranks?.[0] ?? "?"}&ndash;{data.ranks?.[1] ?? "?"}
              </span>
            </div>
            <h2 className="page-title" style={{ fontSize: "var(--fs-2xl)" }}>{data.name}</h2>
            <p className="page-subtitle">{summary?.tagline || data.tagline}</p>
          </header>
          <RunTab key={data.id} detail={data} persistKey={STORAGE_KEYS.playgroundRequest} />
        </>
      )}
    </div>
  );
}
