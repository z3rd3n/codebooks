import type { CodebookDetail, DocResponse } from "../../api/types";
import { api } from "../../api/client";
import { useApiData } from "../../hooks/useApiData";
import { Markdown } from "../../components/Markdown";
import { Skeleton } from "../../components/Skeleton";
import { ErrorBanner } from "../../components/ErrorBanner";
import { Icon } from "../../components/Icon";

/** The "Deep dive" tab: the full spec-faithful documentation chapter for this
 * codebook, rendered with markdown + KaTeX. */
export function DeepDiveTab({ detail }: { detail: CodebookDetail }) {
  const { data, loading, error } = useApiData<DocResponse>(
    () => api.codebookDoc(detail.id),
    [detail.id],
  );

  return (
    <div style={{ maxWidth: 820 }}>
      <div className="info-banner" style={{ marginBottom: 24 }}>
        <Icon name="info" />
        <div>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Technical reference</div>
          <div className="text-sm">
            This is the complete, spec-faithful chapter for {detail.shortName || detail.name} (
            {detail.specClause}). It quotes TS 38.214 and the implementing code directly. For the
            plain-language tour, use the <strong>Understand</strong> tab.
          </div>
        </div>
      </div>

      {loading && (
        <div className="col col-gap-3">
          <Skeleton height={22} width="55%" />
          <Skeleton height={14} width="90%" />
          <Skeleton height={14} width="85%" />
          <Skeleton height={14} width="70%" />
        </div>
      )}

      {!loading && error && (
        <ErrorBanner
          message={error}
          title="Documentation unavailable"
          hint="The technical chapter for this codebook could not be loaded."
        />
      )}

      {!loading && !error && data?.markdown && (
        <article className="doc-body">
          <Markdown text={data.markdown} flattenRelativeLinks />
        </article>
      )}

      {!loading && !error && !data?.markdown && (
        <div className="text-sm text-muted">No documentation chapter is available for this codebook.</div>
      )}
    </div>
  );
}
