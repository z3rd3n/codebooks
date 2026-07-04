# CSI Codebook Studio

A web UI for exploring, running, and benchmarking the 3GPP NR PMI codebooks
implemented in `src/nr_csi/`. See `SPEC.md` for the full design contract.

## Running the backend

The repo's `.venv` already has the runtime dependencies installed. If you are
starting from a clean environment, install the `web` extra first:

```bash
pip install -e ".[web]"
```

Then either:

```bash
./webapp/run.sh
```

which builds `webapp/ui/dist` (if it's missing and `npm` is available) and
starts the API + static file server, or run uvicorn directly:

```bash
.venv/bin/python -m uvicorn webapp.server.main:app --port 8787
```

The server listens on **port 8787**. Open http://localhost:8787/ once the UI
is built; before that, `/` returns a plain-text build hint and every `/api/*`
route still works (useful for backend-only development and the selftest).

## Dev mode (frontend + backend together)

Run the backend as above, then in a second terminal:

```bash
cd webapp/ui
npm install
npm run dev
```

The Vite dev server proxies `/api` and `/artifacts` to `http://localhost:8787`,
so you get hot-reloading on the frontend while the backend serves real data.

## Verifying the backend

```bash
.venv/bin/python -m webapp.server.selftest
```

This hits every `/api` endpoint with FastAPI's `TestClient`, including a real
figures job (`fig_03_overhead_breakdown`, ~3s) polled to completion and real
`/api/run` calls for `type1-sp`, `etype2-r16`, `etype2-doppler-r18`, and
`cjt-r18` (these execute the actual PMI selection/reconstruction numerics, so
expect the run to take on the order of ten seconds total). Prints one
`PASS`/`FAIL` line per check and a final summary.

## Layout

* `server/` — FastAPI backend (routes, codebook catalog, playground runner,
  figure-script job runner, content loader).
* `server/content/` — curated explanations (JSON) consumed by
  `server/content.py`; the server tolerates this directory being partial or
  absent and falls back to bare catalog metadata.
* `ui/` — Vite + React + TypeScript frontend (built separately; see its own
  `package.json`).
