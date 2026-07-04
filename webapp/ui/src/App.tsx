import { useMemo } from "react";
import { HashRouter, Routes, Route } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { GlossaryContext } from "./hooks/useGlossary";
import { useApiData } from "./hooks/useApiData";
import { api } from "./api/client";
import type { GlossaryTerm } from "./api/types";

import Overview from "./pages/Overview";
import Library from "./pages/Library";
import CodebookDetailPage from "./pages/CodebookDetailPage";
import Playground from "./pages/Playground";
import Compare from "./pages/Compare";
import FigureLab from "./pages/FigureLab";
import Glossary from "./pages/Glossary";
import NotFound from "./pages/NotFound";

export default function App() {
  const { data } = useApiData<GlossaryTerm[]>(() => api.glossary(), []);
  const terms = data ?? [];
  const byTerm = useMemo(() => {
    const m = new Map<string, GlossaryTerm>();
    for (const t of terms) {
      if (t?.term) m.set(t.term.toLowerCase(), t);
    }
    return m;
  }, [terms]);

  return (
    <GlossaryContext.Provider value={{ terms, byTerm, loading: data === null }}>
      <HashRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/codebooks" element={<Library />} />
            <Route path="/codebooks/:id" element={<CodebookDetailPage />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/figures" element={<FigureLab />} />
            <Route path="/glossary" element={<Glossary />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AppShell>
      </HashRouter>
    </GlossaryContext.Provider>
  );
}
