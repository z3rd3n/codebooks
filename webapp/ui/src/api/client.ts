import type {
  CodebookDetail,
  CodebookSummary,
  CompareRequest,
  CompareResult,
  DocResponse,
  FigureInfo,
  FiguresRunRequest,
  FiguresRunResponse,
  GlossaryTerm,
  HomeContent,
  JobStatus,
  Meta,
  RunRequest,
  RunResult,
  ValidateResult,
} from "./types";

/** Thrown for any non-2xx response; carries the backend's friendly message. */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
      ...init,
    });
  } catch {
    throw new ApiError(
      "Could not reach the server. Is the backend running on port 8787?",
      0,
    );
  }

  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body && typeof body.error === "string") {
        message = body.error;
      }
    } catch {
      // response wasn't JSON; keep the generic message
    }
    throw new ApiError(message, res.status);
  }

  // Some endpoints (rare) could return empty bodies; guard against JSON parse errors.
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiError("Server returned an unreadable response.", res.status);
  }
}

export const api = {
  meta: () => request<Meta>("/api/meta"),

  codebooks: () => request<CodebookSummary[]>("/api/codebooks"),

  codebook: (id: string) =>
    request<CodebookDetail>(`/api/codebooks/${encodeURIComponent(id)}`),

  codebookDoc: (id: string) =>
    request<DocResponse>(`/api/codebooks/${encodeURIComponent(id)}/doc`),

  home: () => request<HomeContent>("/api/content/home"),

  glossary: () => request<GlossaryTerm[]>("/api/content/glossary"),

  foundations: () => request<DocResponse>("/api/content/foundations"),

  validate: (body: RunRequest) =>
    request<ValidateResult>("/api/validate", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  run: (body: RunRequest) =>
    request<RunResult>("/api/run", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  compare: (body: CompareRequest) =>
    request<CompareResult>("/api/compare", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  figures: () => request<FigureInfo[]>("/api/figures"),

  runFigures: (body: FiguresRunRequest) =>
    request<FiguresRunResponse>("/api/figures/run", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  job: (id: string) => request<JobStatus>(`/api/jobs/${encodeURIComponent(id)}`),
};
