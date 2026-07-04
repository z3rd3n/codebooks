import { useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

export interface ApiDataState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/** Fetch-on-mount helper: runs `fetcher` once (or when `deps` change), tracks
 * loading/error state, and ignores results from stale/aborted runs. */
export function useApiData<T>(fetcher: () => Promise<T>, deps: unknown[] = []): ApiDataState<T> {
  const [state, setState] = useState<ApiDataState<T>>({ data: null, loading: true, error: null });
  const seq = useRef(0);

  useEffect(() => {
    const mySeq = ++seq.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetcher()
      .then((data) => {
        if (seq.current !== mySeq) return;
        setState({ data, loading: false, error: null });
      })
      .catch((err) => {
        if (seq.current !== mySeq) return;
        const message = err instanceof ApiError ? err.message : "Unexpected error. Please try again.";
        setState({ data: null, loading: false, error: message });
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
