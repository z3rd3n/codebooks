import type { CodebookDetail } from "../../api/types";
import { Markdown } from "../../components/Markdown";
import { GlossaryChip } from "../../components/GlossaryChip";
import { EmptyState } from "../../components/EmptyState";
import { Icon } from "../../components/Icon";

/** The "Understand" tab: curated plain-language explanation of the codebook. */
export function UnderstandTab({ detail }: { detail: CodebookDetail }) {
  const content = detail.content;

  if (!content) {
    return (
      <EmptyState
        title="Explanation not written yet"
        body="The curated explanation for this codebook hasn't been added. The Run and Deep dive tabs still work."
      />
    );
  }

  const overview = content.overview ?? [];
  const steps = content.howItWorks ?? [];
  const reported = content.whatIsReported ?? [];
  const paramsExplained = content.parametersExplained ?? [];
  const strengths = content.strengths ?? [];
  const limitations = content.limitations ?? [];
  const glossaryTerms = content.glossary ?? [];

  return (
    <div className="col col-gap-5" style={{ maxWidth: 860 }}>
      {overview.length > 0 && (
        <section>
          {overview.map((para, i) => (
            <Markdown key={i} text={para} />
          ))}
        </section>
      )}

      {steps.length > 0 && (
        <section className="card">
          <h3 className="card-title">How it works</h3>
          <p className="panel-subtitle" style={{ marginBottom: 20 }}>
            The feedback loop, step by step.
          </p>
          <ol className="step-flow">
            {steps.map((s, i) => (
              <li key={i} className="step-item">
                <span className="step-number tabular-nums">{i + 1}</span>
                <div>
                  <div className="step-title">{s.title}</div>
                  <div className="step-body">
                    <Markdown text={s.body} />
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}

      {content.mathHighlight?.latex && (
        <section className="card" style={{ background: "var(--accent-soft)", borderColor: "var(--accent-soft-border)" }}>
          <h3 className="card-title">The key formula</h3>
          <div style={{ overflowX: "auto", padding: "8px 0" }}>
            <Markdown text={`$$${content.mathHighlight.latex}$$`} />
          </div>
          {content.mathHighlight.caption && (
            <p className="text-secondary" style={{ lineHeight: 1.6 }}>
              <Markdown text={content.mathHighlight.caption} />
            </p>
          )}
        </section>
      )}

      {reported.length > 0 && (
        <section className="card">
          <h3 className="card-title">What the phone reports</h3>
          <p className="panel-subtitle" style={{ marginBottom: 16 }}>
            Every field in this codebook's PMI feedback message.
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Field</th>
                <th>In plain words</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {reported.map((f, i) => (
                <tr key={`${f.field}-${i}`}>
                  <td style={{ fontFamily: "var(--font-mono)", fontSize: 12.5, fontWeight: 600, whiteSpace: "nowrap" }}>
                    {f.field}
                  </td>
                  <td>
                    <Markdown text={f.plain} />
                  </td>
                  <td className="text-secondary">{f.detail ? <Markdown text={f.detail} /> : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {paramsExplained.length > 0 && (
        <section className="card">
          <h3 className="card-title">Choosing the parameters</h3>
          <div className="col col-gap-4 mt-3">
            {paramsExplained.map((p, i) => (
              <div key={`${p.key}-${i}`}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 12.5, fontWeight: 600, marginBottom: 4 }}>
                  {p.key}
                </div>
                <Markdown text={p.plain} />
                {p.guidance && (
                  <div className="text-sm text-secondary" style={{ marginTop: 4 }}>
                    <Markdown text={p.guidance} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {(strengths.length > 0 || limitations.length > 0) && (
        <section className="grid-2">
          <div className="card">
            <h3 className="card-title" style={{ color: "var(--success)" }}>Strengths</h3>
            <ul className="col col-gap-2 mt-2">
              {strengths.length === 0 && <li className="text-sm text-muted">None listed.</li>}
              {strengths.map((s, i) => (
                <li key={i} className="row row-gap-2" style={{ alignItems: "flex-start" }}>
                  <span style={{ color: "var(--success)", flexShrink: 0, marginTop: 2 }}>
                    <Icon name="check" size={14} />
                  </span>
                  <Markdown text={s} />
                </li>
              ))}
            </ul>
          </div>
          <div className="card">
            <h3 className="card-title" style={{ color: "var(--warning)" }}>Limitations</h3>
            <ul className="col col-gap-2 mt-2">
              {limitations.length === 0 && <li className="text-sm text-muted">None listed.</li>}
              {limitations.map((s, i) => (
                <li key={i} className="row row-gap-2" style={{ alignItems: "flex-start" }}>
                  <span style={{ color: "var(--warning)", flexShrink: 0, marginTop: 2 }}>
                    <Icon name="warning" size={14} />
                  </span>
                  <Markdown text={s} />
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {content.whenToUse && (
        <section className="info-banner">
          <Icon name="info" />
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>When to use it</div>
            <Markdown text={content.whenToUse} />
          </div>
        </section>
      )}

      {glossaryTerms.length > 0 && (
        <section>
          <h3 style={{ fontSize: "var(--fs-lg)", fontWeight: 600, marginBottom: 12 }}>Related terms</h3>
          <div className="row row-gap-2 row-wrap">
            {glossaryTerms.map((term) => (
              <span key={term} className="chip">
                <GlossaryChip term={term} />
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
