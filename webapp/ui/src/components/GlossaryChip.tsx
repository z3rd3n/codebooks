import type { ReactNode } from "react";
import { useGlossary } from "../hooks/useGlossary";

interface GlossaryChipProps {
  term: string;
  children?: ReactNode;
}

/** Wraps a glossary term with a hover/focus tooltip showing its short definition. */
export function GlossaryChip({ term, children }: GlossaryChipProps) {
  const { byTerm } = useGlossary();
  const entry = byTerm.get(term.toLowerCase());

  if (!entry) {
    return <>{children ?? term}</>;
  }

  return (
    <span className="glossary-chip" tabIndex={0}>
      {children ?? term}
      <span className="tooltip" role="tooltip">
        {entry.short}
      </span>
    </span>
  );
}
