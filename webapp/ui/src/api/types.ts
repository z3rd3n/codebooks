// Single source of truth for the API contract described in webapp/SPEC.md §3.
// Every field that the backend may omit or send as null/empty is typed as
// optional / nullable so the UI can render defensively.

export type Release = "R15" | "R16" | "R17" | "R18" | "R19";

// ---------------------------------------------------------------------------
// §1 / meta
// ---------------------------------------------------------------------------

export interface Meta {
  version: string;
  sionna_available: boolean;
  antennas?: AntennaPair[];
}

// ---------------------------------------------------------------------------
// §2 catalog / ParamSpec
// ---------------------------------------------------------------------------

export type ParamType = "choice" | "int" | "float" | "bool";

export interface ParamChoice {
  value: string | number | boolean;
  label: string;
  description?: string | null;
}

export interface ParamVisibleIf {
  key: string;
  value: string | number | boolean;
}

export interface ParamSpec {
  key: string;
  label: string;
  type: ParamType;
  default: string | number | boolean | null;
  choices?: ParamChoice[] | null;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  visibleIf?: ParamVisibleIf | null;
  help?: string | null;
}

export interface AntennaPair {
  n1: number;
  n2: number;
  ports: number;
  ng?: number | null;
}

export type AntennaMode = "single" | "multi" | "fixed";

export interface AntennaSpec {
  mode: AntennaMode;
  pairs: AntennaPair[];
}

// ---------------------------------------------------------------------------
// Codebook summary / detail
// ---------------------------------------------------------------------------

export interface CodebookSummary {
  id: string;
  name: string;
  shortName: string;
  release: Release | string;
  specClause: string;
  tagline: string;
  ranks: [number, number];
  portRange: [number, number];
  position: number;
  lineage?: CodebookLineage | null;
}

export interface CodebookLineage {
  parent?: string | null;
  adds?: string | null;
}

export interface CodebookDetail extends CodebookSummary {
  params: ParamSpec[];
  antenna: AntennaSpec;
  defaults?: Record<string, unknown> | null;
  content?: CodebookContent | null;
}

// ---------------------------------------------------------------------------
// §5 content schema
// ---------------------------------------------------------------------------

export interface HowItWorksStep {
  title: string;
  body: string;
}

export interface WhatIsReportedField {
  field: string;
  plain: string;
  detail?: string | null;
}

export interface ParameterExplained {
  key: string;
  plain: string;
  guidance?: string | null;
}

export interface MathHighlight {
  caption: string;
  latex: string;
}

export interface CodebookContent {
  id: string;
  name: string;
  shortName: string;
  release: Release | string;
  specClause: string;
  docFile: string;
  position: number;
  lineage?: CodebookLineage | null;
  tagline: string;
  overview?: string[] | null;
  howItWorks?: HowItWorksStep[] | null;
  whatIsReported?: WhatIsReportedField[] | null;
  parametersExplained?: ParameterExplained[] | null;
  strengths?: string[] | null;
  limitations?: string[] | null;
  whenToUse?: string | null;
  mathHighlight?: MathHighlight | null;
  glossary?: string[] | null;
}

export interface DocResponse {
  markdown: string;
}

// ---------------------------------------------------------------------------
// home.json / glossary.json
// ---------------------------------------------------------------------------

export interface TimelineEntry {
  release: Release | string;
  year: number;
  id: string;
  name: string;
  oneLiner: string;
}

export interface ConceptCard {
  title: string;
  body: string;
}

export interface HomeContent {
  hero?: { title: string; subtitle: string } | null;
  story?: string[] | null;
  timeline?: TimelineEntry[] | null;
  concepts?: ConceptCard[] | null;
}

export interface GlossaryTerm {
  term: string;
  short: string;
  long: string;
}

// ---------------------------------------------------------------------------
// §3 validate / run / compare requests
// ---------------------------------------------------------------------------

export interface ChannelConfig {
  preset?: string | null;
  n_rx?: number;
  n_paths?: number;
  max_delay?: number;
  max_doppler?: number;
  inter_trp_delay?: number;
}

export interface AntennaConfig {
  n1: number;
  n2: number;
  ng?: number;
}

export interface RunRequest {
  codebook_id: string;
  params: Record<string, unknown>;
  antenna: AntennaConfig;
  n3: number;
  rank: number;
  channel: ChannelConfig;
  snr_db: number[];
  drops: number;
  seed: number;
}

export type ValidateRequest = RunRequest;

export interface ValidateResult {
  ok: boolean;
  error?: string | null;
}

// ---------------------------------------------------------------------------
// RunResult
// ---------------------------------------------------------------------------

export interface RunMetrics {
  sgcs?: number | null;
  subspace_sgcs?: number | null;
  snr_db: number[];
  se: number[];
  se_upper_bound: number[];
  overhead_bits?: Record<string, number> | null;
  total_bits?: number | null;
}

export interface PmiField {
  name: string;
  value: string;
  bits?: number | null;
  description?: string | null;
}

export interface PmiReport {
  fields: PmiField[];
}

export interface HeatmapMatrix {
  abs: number[][];
  rows?: string | null;
  cols?: string | null;
}

export interface PrecoderLayer {
  layer: number;
  abs: number[][];
  phase: number[][];
}

export interface BeamGrid {
  g1: number;
  g2: number;
  selected: [number, number][];
}

export interface RunViz {
  channel?: HeatmapMatrix | null;
  eigen_spectrum?: number[] | null;
  precoder?: PrecoderLayer[] | null;
  beam_grid?: BeamGrid | null;
}

export interface RunResult {
  ok: boolean;
  error?: string | null;
  seconds?: number | null;
  scheme_name?: string | null;
  config_echo?: Record<string, unknown> | null;
  python_snippet?: string | null;
  metrics?: RunMetrics | null;
  pmi?: PmiReport | null;
  viz?: RunViz | null;
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

export interface CompareSchemeRequest {
  codebook_id: string;
  params: Record<string, unknown>;
  label: string;
}

export interface CompareSharedConfig {
  antenna: AntennaConfig;
  n3: number;
  rank: number;
  channel: ChannelConfig;
  snr_db: number[];
  drops: number;
  seed: number;
}

export interface CompareRequest {
  schemes: CompareSchemeRequest[];
  shared: CompareSharedConfig;
}

export interface CompareSchemeResult {
  label: string;
  scheme_name?: string | null;
  ok: boolean;
  error?: string | null;
  sgcs?: number | null;
  subspace_sgcs?: number | null;
  total_bits?: number | null;
  se?: number[] | null;
  se_upper_bound?: number[] | null;
  snr_db?: number[] | null;
  se_at_10db?: number | null;
  bound_at_10db?: number | null;
}

export interface CompareResult {
  ok: boolean;
  error?: string | null;
  seconds?: number | null;
  results: CompareSchemeResult[];
}

// ---------------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------------

export interface FigureInfo {
  slug: string;
  title: string;
  estSeconds: number;
  honorsFamilies: boolean;
  cdlAvailable?: boolean;
  swept?: string[] | null;
  blurb?: string | null;
  question?: string | null;
  howToRead?: string | null;
  whatToLookFor?: string | null;
  caveats?: string | null;
}

export interface FiguresRunRequest {
  slugs: string[];
  channel: "synthetic" | "cdl";
  families?: string[];
  antenna: { n1: number; n2: number };
  n3: number;
  n_rx: number;
  n_paths?: number;
  max_delay?: number;
  cdl_model?: string;
  cdl_speed?: number;
  cdl_delay_spread_ns?: number;
  drops: number;
  seed: number;
  fast: boolean;
}

export interface FiguresRunResponse {
  job_id: string;
}

export type JobStatusState = "queued" | "running" | "done" | "error";

export interface FigureJobResult {
  slug: string;
  ok: boolean;
  png_url?: string | null;
  json_url?: string | null;
  data?: Record<string, unknown> | null;
  log?: string | null;
  seconds?: number | null;
  error?: string | null;
}

export interface JobStatus {
  id: string;
  kind: string;
  status: JobStatusState;
  progress?: number | null;
  message?: string | null;
  results?: FigureJobResult[] | null;
}

// ---------------------------------------------------------------------------
// Generic API error envelope
// ---------------------------------------------------------------------------

export interface ApiErrorBody {
  error: string;
}
