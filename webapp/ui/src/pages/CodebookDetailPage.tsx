import { useState } from "react";
import { useParams } from "react-router-dom";
import { useApiData } from "../hooks/useApiData";
import { api } from "../api/client";
import type { CodebookDetail } from "../api/types";
import { Skeleton } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";
import { ReleaseBadge } from "../components/Badge";
import { UnderstandTab } from "./codebook-tabs/UnderstandTab";
import { RunTab } from "./codebook-tabs/RunTab";
import { DeepDiveTab } from "./codebook-tabs/DeepDiveTab";

type TabKey = "understand" | "run" | "deep-dive";

export default function CodebookDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [tab, setTab] = useState<TabKey>("understand");
  const { data, loading, error } = useApiData<CodebookDetail>(
    () => api.codebook(id ?? ""),
    [id],
  );

  return (
    <div className="page">
      {loading && (
        <div>
          <Skeleton height={32} width="40%" />
          <div className="mt-3">
            <Skeleton height={16} width="70%" />
          </div>
        </div>
      )}

      {!loading && error && (
        <>
          <div className="page-header">
            <h1 className="page-title">Codebook not found</h1>
          </div>
          <ErrorBanner message={error} hint="This codebook id may not exist. Return to the library and pick one from the list." />
        </>
      )}

      {!loading && !error && data && (
        <>
          <header className="page-header">
            <div className="row row-gap-3" style={{ marginBottom: 10, flexWrap: "wrap" }}>
              <ReleaseBadge release={data.release} />
              <span className="chip">{data.specClause}</span>
              <span className="chip">
                rank {data.ranks?.[0] ?? "?"}&ndash;{data.ranks?.[1] ?? "?"}
              </span>
              <span className="chip">
                {data.portRange?.[0] ?? "?"}&ndash;{data.portRange?.[1] ?? "?"} ports
              </span>
            </div>
            <h1 className="page-title">{data.name}</h1>
            <p className="page-subtitle">{data.tagline}</p>
          </header>

          <div className="tabs" role="tablist">
            <button
              role="tab"
              aria-selected={tab === "understand"}
              className={`tab${tab === "understand" ? " active" : ""}`}
              onClick={() => setTab("understand")}
            >
              Understand
            </button>
            <button
              role="tab"
              aria-selected={tab === "run"}
              className={`tab${tab === "run" ? " active" : ""}`}
              onClick={() => setTab("run")}
            >
              Run
            </button>
            <button
              role="tab"
              aria-selected={tab === "deep-dive"}
              className={`tab${tab === "deep-dive" ? " active" : ""}`}
              onClick={() => setTab("deep-dive")}
            >
              Deep dive
            </button>
          </div>

          {tab === "understand" && <UnderstandTab detail={data} />}
          {tab === "run" && <RunTab detail={data} />}
          {tab === "deep-dive" && <DeepDiveTab detail={data} />}
        </>
      )}
    </div>
  );
}
