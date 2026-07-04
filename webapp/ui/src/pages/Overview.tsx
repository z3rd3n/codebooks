import { Link } from "react-router-dom";
import { useApiData } from "../hooks/useApiData";
import { api } from "../api/client";
import type { CodebookSummary, HomeContent } from "../api/types";
import { Skeleton } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";
import { ReleaseBadge } from "../components/Badge";
import { Icon } from "../components/Icon";
import { Markdown } from "../components/Markdown";

export default function Overview() {
  const home = useApiData<HomeContent>(() => api.home(), []);
  const codebooks = useApiData<CodebookSummary[]>(() => api.codebooks(), []);

  const hero = home.data?.hero;
  const story = home.data?.story ?? [];
  const concepts = home.data?.concepts ?? [];
  const timeline = home.data?.timeline ?? [];

  return (
    <div className="page">
      <section style={{ marginBottom: 48 }}>
        {home.loading ? (
          <>
            <Skeleton height={40} width="60%" />
            <div className="mt-3">
              <Skeleton height={16} width="80%" />
            </div>
          </>
        ) : home.error ? (
          <ErrorBanner message={home.error} hint="The overview content couldn't be loaded. Try refreshing." />
        ) : (
          <>
            <h1 className="page-title" style={{ fontSize: "var(--fs-3xl)" }}>
              {hero?.title || "CSI Codebook Studio"}
            </h1>
            <p className="page-subtitle" style={{ fontSize: "var(--fs-lg)", maxWidth: "72ch" }}>
              {hero?.subtitle ||
                "Explore, understand, and run 3GPP NR PMI codebooks — no terminal, no spec-reading required."}
            </p>
            <div className="row row-gap-3 mt-4">
              <Link to="/codebooks" className="btn btn-primary">
                Browse the library
                <Icon name="arrow-right" size={14} />
              </Link>
              <Link to="/playground" className="btn">
                Open the playground
              </Link>
            </div>
          </>
        )}
      </section>

      {!home.loading && !home.error && story.length > 0 && (
        <section className="card" style={{ marginBottom: 40, maxWidth: 760 }}>
          <div className="col col-gap-3">
            {story.map((para, i) => (
              <Markdown key={i} text={para} />
            ))}
          </div>
        </section>
      )}

      {!home.loading && !home.error && concepts.length > 0 && (
        <section style={{ marginBottom: 48 }}>
          <h2 style={{ fontSize: "var(--fs-xl)", fontWeight: 650, marginBottom: 16 }}>Key concepts</h2>
          <div className="card-grid">
            {concepts.map((c, i) => (
              <div className="card card-hover" key={i}>
                <div className="card-title">{c.title}</div>
                <div className="text-secondary" style={{ lineHeight: 1.6 }}>
                  <Markdown text={c.body} />
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section>
        <h2 style={{ fontSize: "var(--fs-xl)", fontWeight: 650, marginBottom: 16 }}>Release timeline</h2>
        {codebooks.loading || home.loading ? (
          <div className="row row-gap-3">
            {Array.from({ length: 5 }, (_, i) => (
              <Skeleton key={i} height={90} width={180} />
            ))}
          </div>
        ) : timeline.length === 0 ? (
          <div className="empty-state">
            <Icon name="empty" size={28} />
            <span className="text-sm">No timeline data available yet.</span>
          </div>
        ) : (
          <Timeline entries={timeline} />
        )}
      </section>
    </div>
  );
}

function Timeline({ entries }: { entries: NonNullable<HomeContent["timeline"]> }) {
  return (
    <div
      className="row row-gap-3"
      style={{ overflowX: "auto", paddingBottom: 8 }}
      role="list"
      aria-label="Codebook release timeline"
    >
      {entries.map((e, i) => (
        <Link
          key={`${e.id}-${i}`}
          to={`/codebooks/${e.id}`}
          role="listitem"
          className="card card-hover"
          style={{
            minWidth: 220,
            flexShrink: 0,
            textDecoration: "none",
            color: "inherit",
            position: "relative",
          }}
        >
          <div className="row row-gap-2" style={{ marginBottom: 8 }}>
            <ReleaseBadge release={e.release} />
            <span className="text-sm text-muted">{e.year}</span>
          </div>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>{e.name}</div>
          <div className="text-sm text-secondary">{e.oneLiner}</div>
        </Link>
      ))}
    </div>
  );
}
