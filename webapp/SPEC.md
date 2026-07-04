# CSI Codebook Studio — Design & Build Spec

A professional web UI for exploring, understanding, and running the 3GPP NR PMI
codebooks implemented in `src/nr_csi/`. Target audience: **someone who cannot
code and has never read TS 38.214** — a student, a manager, a researcher from an
adjacent field. They must be able to (a) learn what each codebook does, (b) run
it on a realistic channel with one click, (c) investigate the results, and
(d) generate the benchmark figures — all without touching a terminal beyond
starting the app.

This replaces the Streamlit app (`webapp/app.py` + `webapp/registry.py`), which
stays untouched until the new app is verified, then gets deleted.

---

## 1. Architecture

```
webapp/
  SPEC.md                  # this file
  README.md                # how to run (written by backend agent)
  run.sh                   # build UI if needed + start uvicorn (backend agent)
  server/                  # FastAPI backend  (BACKEND AGENT owns this dir)
    __init__.py
    main.py                # app factory; /api routes; serves ui/dist statically
    catalog.py             # codebook registry: metadata, param schemas, factories
    runner.py              # playground runs: channel + select/precoder + metrics + viz
    figures.py             # figure-script subprocess jobs (port of registry.py)
    jobs.py                # tiny thread-based job manager (in-memory store)
    content.py             # loads content/ JSON + docs markdown
    selftest.py            # TestClient smoke test of every endpoint
    content/               # curated explanations  (CONTENT AGENT owns this dir)
      codebooks/<id>.json  # one per catalog entry (schema in §5)
      glossary.json
      figures.json
      home.json
  ui/                      # Vite + React + TS  (FRONTEND AGENT owns this dir)
    package.json  vite.config.ts  tsconfig.json  index.html
    src/ ...
```

* Python deps: `fastapi`, `uvicorn` (already installed in `.venv`; the repo
  `.venv/bin/python` is the interpreter for everything). `httpx` is installed
  for the TestClient selftest.
* Server port **8787**. Vite dev server proxies `/api` and `/artifacts` to 8787.
* Production mode: `npm run build` → `webapp/ui/dist`, served by FastAPI
  (`StaticFiles(html=True)` mounted at `/`, after the `/api` and `/artifacts`
  routes). If `dist` is missing, `GET /` returns a plain-text hint to build.
* No database. Job store is an in-memory dict (single-user local tool). Figure
  artifacts live where the scripts already write them (`results/webapp`,
  `results/sionna_cdl_gallery`).

## 2. Codebook catalog (locked IDs)

`catalog.py` defines one entry per row. **These IDs are a contract** shared by
backend, content files, and frontend routing — do not rename.

| id | class(es) in `nr_csi.codebooks` | doc chapter | key user-facing params |
|---|---|---|---|
| `type1-2port` | `TwoPortType1Codebook` | 01 | rank 1–2 (P fixed = 2, no antenna picker) |
| `type1-sp` | `Type1Codebook` | 01 | codebookMode 1/2, rank 1–8 |
| `type1-mp` | `Type1MultiPanelCodebook` | 02 | (Ng,N1,N2) from `SUPPORTED_NG_N1N2`, codebookMode, rank 1–4 |
| `type2-r15` | `R15Type2Codebook` | 03 | L, subband amplitude (if ctor has it), port-selection variant (+d), rank 1–2 |
| `etype2-r16` | `R16Type2Codebook` | 04 | paramCombination 1–8 (PS: 1–6 + d), port-selection toggle, R 1/2, rank 1–4 |
| `fetype2-r17` | `R17Type2Codebook` | 05 | paramCombination 1–8, rank 1–4 (inherently port-selection) |
| `predicted-ps-r18` | `R18PredictedPortSelectionCodebook` | 05 | see ctor |
| `etype2-doppler-r18` | `R18Type2Codebook` | 06 | paramCombination-Doppler 1–9, N4; **eval uses `n_slots=N4`** and a moving channel |
| `cjt-r18` | `R18CJTCodebook`, `R18CJTPortSelectionCodebook` | 08 | N_TRP, per-spec L combo, PS variant toggle |
| `refined-type1-r19` | `RefinedType1SinglePanelCodebook` | 07 | mode modeA/modeB; allows `SUPPORTED_N1N2_R19` large arrays |
| `refined-type1mp-r19` | `RefinedType1MultiPanelCodebook` | 07 | `SUPPORTED_NG_N1N2_R19` geometries |
| `refined-type2-r19` | `RefinedEType2Codebook` / `RefinedFeType2PortSelectionCodebook` / `RefinedPredictedEType2Codebook` | 07 | variant choice: regular / PS / predicted, + their params |

Backend agent: derive **exact** constructor arguments, valid ranges, and
config bars from the source files (`src/nr_csi/codebooks/*.py`,
`src/nr_csi/config.py`) and from real usage in
`scripts/figures/fig_13_new_codebooks.py`, `scripts/figures/fig_05_mobility.py`,
`src/nr_csi/figtools/figlib.py` (`standard_schemes`), and `tests/`. Constructor
`ValueError`s are already spec-quoting and friendly — catch them and return
their message in a structured 400.

Each catalog entry declares a **param schema** the frontend renders dynamically:

```jsonc
// ParamSpec
{ "key": "param_combination", "label": "Parameter combination",
  "type": "choice",              // "choice" | "int" | "float" | "bool"
  "default": 4,
  "choices": [ {"value": 1, "label": "1 — L=2, p=1/4, β=1/4", "description": "Cheapest: 2 beams, few taps"} ],
  "min": null, "max": null, "step": null,
  "visibleIf": {"key": "port_selection", "value": true},   // optional, single condition
  "help": "one plain sentence (content agent may override via parametersExplained)" }
```

Antenna geometry is a shared control, not a ParamSpec: each entry exposes
`"antenna": {"mode": "single"|"multi"|"fixed", "pairs": [{"n1":4,"n2":2,"ports":16}...]}`
computed from the config tables (+ R19 tables where the class accepts them).

## 3. API contract (locked)

All under `/api`. Errors: `{"error": "<friendly message>"}` with 400/404/500.

```
GET  /api/meta            -> {"version": "...", "sionna_available": bool}
GET  /api/codebooks       -> [CodebookSummary]   # id, name, shortName, release,
                             # specClause, tagline, ranks [lo,hi], portRange, position, lineage
GET  /api/codebooks/{id}  -> CodebookDetail      # summary + params: [ParamSpec]
                             # + antenna spec + defaults + content (the full content JSON, §5)
GET  /api/codebooks/{id}/doc -> {"markdown": "..."}   # docs/codebooks/<docFile> raw text
GET  /api/content/home       -> home.json
GET  /api/content/glossary   -> glossary.json
GET  /api/content/foundations-> {"markdown": "..."}   # docs/codebooks/00-foundations.md
POST /api/validate        -> {"ok": true} | {"ok": false, "error": "..."}
                             # instantiate scheme from a RunRequest without running
POST /api/run             -> RunResult            # synchronous; cap drops<=64, snr points<=12
POST /api/compare         -> CompareResult        # synchronous; cap schemes<=6, drops<=32
GET  /api/figures         -> [FigureInfo]         # slug, title, estSeconds, honorsFamilies,
                             # swept, + explanation fields merged from content/figures.json
POST /api/figures/run     -> {"job_id": "..."}
GET  /api/jobs/{job_id}   -> JobStatus
GET  /artifacts/**        -> StaticFiles mount of results/ (read-only)
```

### RunRequest

```jsonc
{ "codebook_id": "etype2-r16",
  "params": {"param_combination": 4, "port_selection": false},
  "antenna": {"n1": 4, "n2": 2, "ng": 1},
  "n3": 8, "rank": 2,
  "channel": {"preset": "sparse-urban",          // preset fills the fields below; explicit fields win
              "n_rx": 2, "n_paths": 4, "max_delay": 3.0, "max_doppler": 0.0},
  "snr_db": [-10,-5,0,5,10,15,20,25,30],
  "drops": 8, "seed": 0 }
```

Channel presets (runner.py): `sparse-urban` (4 paths, delay 3.0, doppler 0),
`rich-scattering` (12, 6.0, 0), `near-los` (2, 1.0, 0),
`mobile-user` (4, 3.0, doppler 0.5 — required default for `etype2-doppler-r18`
and `predicted` variants). Channel is `RandomRayChannel` (synthetic only; CDL
lives in the Figure Lab — playground must stay TensorFlow-free and fast).

### RunResult

```jsonc
{ "ok": true, "seconds": 1.8, "scheme_name": "R16 eType II",
  "config_echo": { ...the request, resolved... },
  "python_snippet": "from nr_csi... # runnable code equivalent to this run",
  "metrics": {
    "sgcs": 0.93, "subspace_sgcs": 0.95,
    "snr_db": [...], "se": [...], "se_upper_bound": [...],
    "overhead_bits": {"i11 q1,q2": 4, ...},    // scheme.overhead_bits(pmi) of the last drop
    "total_bits": 289 },
  "pmi": { "fields": [ {"name": "i11", "value": "…summarised…", "bits": 4,
                        "description": "…from content whatIsReported, if matched…"} ] },
  "viz": {
    "channel":  {"abs": [[...]], "rows": "N3 frequency units", "cols": "P ports"},  // last drop, slot 0, rx 0
    "eigen_spectrum": [ ... ],                       // mean singular values of H across N3
    "precoder": [ {"layer": 1, "abs": [[...]], "phase": [[...]]} ],  // W[0] per layer, N3 x P
    "beam_grid": {"g1": 16, "g2": 8, "selected": [[l,m], ...]}       // optional; omit if N/A
  } }
```

Numeric matrices are plain nested lists of rounded floats (≤4 decimals); cap any
axis at 64 entries (decimate evenly if larger). `beam_grid` implemented at least
for `type1-sp` and the Type II family (from the reported q1/q2 + beam indices);
omit where it doesn't apply.

For multi-interval schemes (`etype2-doppler-r18`, predicted variants), run
`evaluate(..., n_slots=N4)` following `eval/harness.py` and fig_05/fig_13 usage;
viz uses interval 0. For CJT, build the channel as fig_13's `panel_cjt_delay`
does (per-TRP `RandomRayChannel`s concatenated on the port axis with a delay
offset per TRP; expose `inter_trp_delay` as a channel field, default 0.5).

### CompareRequest / CompareResult

```jsonc
// request
{ "schemes": [ {"codebook_id": "type2-r15", "params": {...}, "label": "Type II L=4"} ],
  "shared": { "antenna": {...}, "n3": 8, "rank": 1, "channel": {...},
              "snr_db": [...], "drops": 16, "seed": 0 } }
// result
{ "ok": true, "seconds": ...,
  "results": [ { "label": "...", "scheme_name": "...", "ok": true, "error": null,
                 "sgcs": ..., "subspace_sgcs": ..., "total_bits": ...,
                 "se": [...], "se_upper_bound": [...], "snr_db": [...],
                 "se_at_10db": ..., "bound_at_10db": ... } ] }
```

Per-scheme failures (e.g. rank unsupported) must not fail the whole compare —
return `ok:false, error` for that scheme only. Schemes that need special
channels (Doppler/CJT) are allowed in compare only when the shared channel is
compatible; otherwise return a per-scheme error explaining why.

### Figures

`POST /api/figures/run` body mirrors the old Streamlit cfg (port
`registry.py`'s `build_env`/`run_figure` faithfully — env vars, `--fast`,
freshness checks, CDL output dir quirk):

```jsonc
{ "slugs": ["fig_01_se_vs_snr"], "channel": "synthetic"|"cdl",
  "families": ["Type I (R15)", ...],            // same display names as registry.FAMILIES
  "antenna": {"n1":4,"n2":2}, "n3": 8, "n_rx": 2,
  "n_paths": 4, "max_delay": 3.0,               // synthetic
  "cdl_model": "C", "cdl_speed": 3.0, "cdl_delay_spread_ns": 100.0,   // cdl
  "drops": 100, "seed": 0, "fast": true }
```

JobStatus:

```jsonc
{ "id": "...", "kind": "figures", "status": "queued"|"running"|"done"|"error",
  "progress": 0.42, "message": "Running SE vs SNR (2/5)…",
  "results": [ {"slug": "...", "ok": true, "png_url": "/artifacts/webapp/fig_01_se_vs_snr.png",
                "json_url": null|"...", "data": {...}|null, "log": "...", "seconds": 12.3} ] }
```

Jobs run in one worker thread, figures sequential, results appended as they
finish (so the UI can show partial results). `sionna_available` in `/api/meta` =
`importlib.util.find_spec("sionna") is not None`.

## 4. Playground semantics

`runner.py` uses `nr_csi.eval.harness.evaluate` for metrics (it already returns
se, bound, sgcs, subspace_sgcs, mean bits) and does **one extra single drop**
with the request seed to produce the PMI/viz payloads (select → precoder →
overhead_bits on a fresh channel). Rank guard: call `select_rank`-style checks
or just let ctor/select raise and translate. Keep total runtime for the default
request under ~3 s (drops=8); `etype2-doppler-r18` and 128-port R19 may take
longer — that's fine, the frontend shows a spinner with the estimate.

`python_snippet` must be honest: imports, AntennaConfig.standard(...), channel,
scheme ctor with the resolved params, `evaluate(...)` call — copy-pasteable.

## 5. Content schema (locked)

`content/codebooks/<id>.json` — written for a smart newcomer. Every acronym
expanded at first use. Analogy-first. Strings may contain **markdown + inline
math** `$...$` (frontend renders markdown+KaTeX inside content strings).

```jsonc
{ "id": "etype2-r16",
  "name": "Enhanced Type II (eType II)", "shortName": "eType II",
  "release": "R16", "specClause": "TS 38.214 §5.2.2.2.5–6",
  "docFile": "04-etype2-r16.md", "position": 6,
  "lineage": {"parent": "type2-r15", "adds": "delay-domain (frequency) compression"},
  "tagline": "one plain sentence — what it is and why it exists",
  "overview": ["2–4 paragraphs, plain language, analogy first"],
  "howItWorks": [ {"title": "step", "body": "…"} ],        // 4–6 steps: UE measure → … → gNB rebuild
  "whatIsReported": [ {"field": "i11", "plain": "…", "detail": "…"} ],
  "parametersExplained": [ {"key": "param_combination", "plain": "…", "guidance": "which value to pick when"} ],
  "strengths": ["…"], "limitations": ["…"],
  "whenToUse": "…",
  "mathHighlight": {"caption": "plain-language walkthrough of the formula", "latex": "W^l_t = …"},
  "glossary": ["PMI", "subband"] }
```

`glossary.json`: `[ {"term": "PMI", "short": "≤15 words", "long": "2–4 sentences"} ]`
— 25–40 terms (PMI, RI, CQI, CSI, CSI-RS, gNB, UE, precoder, beamforming, DFT
beam, oversampling, subband, N3, rank/layer, SGCS, spectral efficiency, SNR,
overhead bits, port, dual polarization, UPA, delay tap, Doppler, TRP/CJT, port
selection, param combination, eigenvector, Monte-Carlo drop, …).

`figures.json`: keyed by slug: `{"title", "question"` (what question the figure
answers)`, "howToRead", "whatToLookFor", "caveats"}` for all 13 slugs in
`webapp/registry.py::FIGURES`.

`home.json`:

```jsonc
{ "hero": {"title": "...", "subtitle": "..."},
  "story": ["3–5 paragraphs: what CSI feedback is, the elevator pitch of the
             codebook problem (describe the base station / phone loop)"],
  "timeline": [ {"release": "R15", "year": 2018, "id": "type1-sp",
                 "name": "Type I", "oneLiner": "..."} , ... ordered ... ],
  "concepts": [ {"title": "...", "body": "..."} ]  }   // 3–4 key ideas: beams, compression, overhead-vs-accuracy, the UE/gNB split
```

Ground every claim in `docs/codebooks/*.md` (read the chapter before writing)
— do not invent numbers or clause references. It's fine to simplify; it's not
fine to be wrong.

## 6. Frontend

Vite + React 18 + TypeScript, `react-router-dom`, `echarts` (imported directly,
thin own wrapper component), `react-markdown` + `remark-gfm` + `remark-math` +
`rehype-katex` + `katex` CSS. No Tailwind, no UI kit — a hand-rolled design
system in plain CSS (CSS variables), which keeps it professional and dependency-light.

### Design language ("professional grade")

* Light theme default, dark theme toggle (CSS variables, `data-theme` on root,
  persisted in localStorage).
* **No emoji anywhere in the chrome.** Inline SVG icons only (hand-drawn 16/20 px
  strokes, one shared Icon component).
* Type: system font stack; `font-variant-numeric: tabular-nums` for metrics;
  clear hierarchy (13px UI base, 15px content, generous line-height for prose).
* Color: near-white background, 1px hairline borders (no heavy shadows), one
  indigo accent; release badges color-coded: R15 slate, R16 teal, R17 amber,
  R18 violet, R19 rose — used consistently everywhere (library, timeline, compare).
* Layout: fixed left sidebar (220px, collapsible) + content column max ~1180px;
  cards 10px radius; consistent 8px spacing grid.
* States: skeleton loaders, friendly empty states, error banners that show the
  backend's message verbatim plus a "what to try" hint. Charts get an
  "About this chart" toggle beneath (content from figure/codebook JSON or
  hardcoded one-liners for playground panels).
* Numbers: 3 significant digits, units always shown (bits, bit/s/Hz, dB).

### Pages (react-router)

* `/` **Overview** — hero, the story paragraphs, key-concept cards, and the
  release **timeline** (horizontal, badge-colored, click → codebook page). Data
  from `/api/content/home` + `/api/codebooks`.
* `/codebooks` **Library** — grid of cards grouped by release; card: name,
  release badge, spec clause, tagline, ports/ranks chips, "adds over parent"
  lineage hint. A slim lineage strip (Type I → II → R16 → R17 → R18 → R19) at top.
* `/codebooks/:id` **Codebook page** — header (name, badges: release, clause,
  ranks, ports) + three tabs:
  * **Understand** — overview paras, howItWorks as a numbered step flow
    (styled), mathHighlight in a KaTeX card with the plain-language caption,
    whatIsReported table, strengths/limitations two-column, whenToUse callout,
    inline glossary chips (hover → tooltip with `short`).
  * **Run** — the playground form scoped to this codebook (shared component)
    with the results dashboard inline.
  * **Deep dive** — `/api/codebooks/:id/doc` markdown rendered with KaTeX
    (this is the full spec-faithful chapter; a banner explains it's the
    technical reference and links relative paths to GitHub-style code refs as
    plain code text, not links).
* `/playground` — codebook picker (cards with tagline) then the same form +
  dashboard; `?codebook=<id>` preselects. Form sections: **Codebook settings**
  (dynamic ParamSpecs), **Antenna** (visual: a little SVG grid of N1×N2
  dual-pol elements per option, port count labeled), **Channel** (preset cards
  with one-line descriptions + "custom" reveal for n_paths/max_delay/…),
  **Evaluation** (rank stepper limited to entry's range, drops slider 1–64,
  SNR range, seed). Debounced `POST /api/validate` on change → inline error
  banner. Run button with elapsed-time indicator.
  **Results dashboard**: metric cards (SGCS, subspace SGCS, feedback bits,
  SE@10 dB and % of bound) → SE-vs-SNR line chart with shaded gap to the eigen
  bound → overhead breakdown (horizontal stacked bar, per-field, hover =
  description) → heatmap row: channel |H| and per-layer |W| (layer tabs) →
  beam-grid dot plot when present → PMI report table (field, bits, summary,
  description) → collapsible "Python equivalent" code block with copy button.
  Every panel has a one-line "what am I looking at" subtitle.
  Last run request persisted in localStorage.
* `/compare` — scheme chips (add up to 6; each opens a small param popover,
  label editable), shared config (same form components), run → results: summary
  table, bits-vs-SGCS scatter (the rate–distortion view, badge-colored, bound
  lines), grouped SE@10dB bars vs bound, SE curves overlay. Per-scheme errors
  render as struck-through chips with the message.
* `/figures` **Figure Lab** — left: figure cards (title, question, est.
  runtime, "ignores family selection" chip where `honors_families` is false)
  with checkboxes; right: config panel (channel synthetic/CDL — CDL disabled
  with an explanatory tooltip when `!sionna_available`, families, antenna/N3,
  fast toggle defaulting on, drops, seed) and Run. Progress panel polls
  `/api/jobs/:id` every 1 s: overall bar + per-figure status; results stream in
  as gallery cards: PNG (click → full-size lightbox), "how to read" text from
  content, elapsed, buttons: download PNG / download JSON / view data (modal
  table for flat JSON), log expander on failure.
* `/glossary` — searchable term list, grouped alphabetically.

### Dev/build

`npm run dev` (proxy to 8787), `npm run build` must pass clean, plus
`npx tsc --noEmit`. Keep components in `src/components`, pages in `src/pages`,
API client + types in `src/api/` (types transcribed from §3 — single source).

## 7. Verification bar

* `webapp/server/selftest.py` (run with the repo `.venv/bin/python -m webapp.server.selftest`):
  hits every GET endpoint, POST /api/validate (good + bad config),
  POST /api/run for at least `type1-sp`, `etype2-r16`, `etype2-doppler-r18`,
  `cjt-r18`, POST /api/compare with 3 schemes, and a figures job with
  `fig_03_overhead_breakdown` (fast, ~3 s) polled to completion. Asserts shapes
  per §3. Prints PASS/FAIL per check.
* Frontend: `npm run build` + `tsc --noEmit` clean.
* Content: every JSON parses; codebook ids exactly match §2; every figure slug
  in registry.FIGURES has an entry.
* Final integration (done by the orchestrator): launch on 8787, click through
  every page, run a playground run and a figure job from the browser, zero
  console errors.
