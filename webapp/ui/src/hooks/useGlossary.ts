import { createContext, useContext } from "react";
import type { GlossaryTerm } from "../api/types";

export interface GlossaryContextValue {
  terms: GlossaryTerm[];
  byTerm: Map<string, GlossaryTerm>;
  loading: boolean;
}

export const GlossaryContext = createContext<GlossaryContextValue>({
  terms: [],
  byTerm: new Map(),
  loading: true,
});

export function useGlossary(): GlossaryContextValue {
  return useContext(GlossaryContext);
}
