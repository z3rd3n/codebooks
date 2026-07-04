import { useMemo, useState } from "react";
import { useGlossary } from "../hooks/useGlossary";
import { EmptyState } from "../components/EmptyState";
import { Skeleton } from "../components/Skeleton";
import { Icon } from "../components/Icon";

export default function Glossary() {
  const { terms, loading } = useGlossary();
  const [query, setQuery] = useState("");

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = terms.filter(
      (t) =>
        !q ||
        t.term.toLowerCase().includes(q) ||
        t.short.toLowerCase().includes(q) ||
        t.long.toLowerCase().includes(q),
    );
    const sorted = [...filtered].sort((a, b) => a.term.localeCompare(b.term));
    const map = new Map<string, typeof terms>();
    for (const t of sorted) {
      const letter = (t.term[0] || "#").toUpperCase();
      const key = /[A-Z]/.test(letter) ? letter : "#";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(t);
    }
    return map;
  }, [terms, query]);

  const letters = Array.from(groups.keys());

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Glossary</h1>
        <p className="page-subtitle">
          Plain-language definitions for the acronyms and terms used throughout this project — from
          PMI and CSI to spectral efficiency and DFT beams.
        </p>
      </div>

      <div className="field" style={{ maxWidth: 420, marginBottom: 28 }}>
        <div className="search-box">
          <Icon name="search" size={16} />
          <input
            className="input"
            type="search"
            placeholder="Search terms…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search glossary"
            style={{ border: "none", paddingLeft: 8, background: "transparent" }}
          />
        </div>
      </div>

      {loading && (
        <div className="col col-gap-3">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} height={54} width="100%" />
          ))}
        </div>
      )}

      {!loading && letters.length === 0 && (
        <EmptyState
          icon="search"
          title="No matching terms"
          body={query ? `Nothing matches “${query}”. Try a different search.` : "The glossary is empty."}
        />
      )}

      {!loading &&
        letters.map((letter) => (
          <section key={letter} style={{ marginBottom: 28 }}>
            <h2 className="glossary-letter">{letter}</h2>
            <div className="col col-gap-3">
              {groups.get(letter)!.map((t) => (
                <div key={t.term} className="card glossary-entry">
                  <div className="row row-gap-3" style={{ alignItems: "baseline", flexWrap: "wrap" }}>
                    <h3 className="glossary-term">{t.term}</h3>
                    <span className="text-sm text-muted">{t.short}</span>
                  </div>
                  <p className="text-secondary" style={{ lineHeight: 1.65, marginTop: 8, marginBottom: 0 }}>
                    {t.long}
                  </p>
                </div>
              ))}
            </div>
          </section>
        ))}
    </div>
  );
}
