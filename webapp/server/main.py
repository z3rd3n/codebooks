"""FastAPI app factory: /api routes; serves ui/dist statically at /.

Run with:  .venv/bin/python -m uvicorn webapp.server.main:app --port 8787
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import catalog, content, figures, jobs, runner

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
UI_DIST = pathlib.Path(__file__).resolve().parents[1] / "ui" / "dist"

app = FastAPI(title="CSI Codebook Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(runner.ValidationError)
async def _validation_error_handler(request: Request, exc: runner.ValidationError):
    return _error(400, str(exc))


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError):
    return _error(400, str(exc))


# --------------------------------------------------------------------- meta


@app.get("/api/meta")
def get_meta():
    try:
        version = importlib.metadata.version("nr-csi")
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0"
    from nr_csi.config import SUPPORTED_N1N2

    antennas = [
        {"n1": n1, "n2": n2, "ports": 2 * n1 * n2}
        for (n1, n2) in SUPPORTED_N1N2
    ]
    antennas.sort(key=lambda a: (a["ports"], a["n1"]))
    return {
        "version": version,
        "sionna_available": importlib.util.find_spec("sionna") is not None,
        "antennas": antennas,
    }


# ---------------------------------------------------------------- catalog


@app.get("/api/codebooks")
def list_codebooks():
    return [entry.summary_dict() for entry in catalog.CATALOG]


@app.get("/api/codebooks/{codebook_id}")
def get_codebook(codebook_id: str):
    entry = catalog.get_entry(codebook_id)
    if entry is None:
        return _error(404, f"unknown codebook id '{codebook_id}'")
    body = entry.summary_dict()
    body["params"] = [p.to_dict() for p in entry.params]
    body["antenna"] = entry.antenna.to_dict()
    body["defaults"] = {p.key: p.default for p in entry.params}
    body["content"] = content.codebook_content(codebook_id)
    return body


@app.get("/api/codebooks/{codebook_id}/doc")
def get_codebook_doc(codebook_id: str):
    entry = catalog.get_entry(codebook_id)
    if entry is None:
        return _error(404, f"unknown codebook id '{codebook_id}'")
    return {"markdown": content.doc_markdown(entry.doc_file)}


# ---------------------------------------------------------------- content


@app.get("/api/content/home")
def get_home_content():
    return content.home_content()


@app.get("/api/content/glossary")
def get_glossary():
    return content.glossary()


@app.get("/api/content/foundations")
def get_foundations():
    return {"markdown": content.foundations_markdown()}


# ---------------------------------------------------------------- playground


@app.post("/api/validate")
def post_validate(req: dict):
    try:
        runner.validate_request(req)
    except runner.ValidationError as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


@app.post("/api/run")
def post_run(req: dict):
    req = dict(req)
    req["drops"] = min(int(req.get("drops", 8)), 64)
    req["snr_db"] = list(req.get("snr_db") or [-10, -5, 0, 5, 10, 15, 20, 25, 30])[:12]
    result = runner.run_playground(req)
    return result


@app.post("/api/compare")
def post_compare(req: dict):
    import time

    started = time.time()
    # Safety ceiling only (schemes run sequentially); far above the ~12 codebooks.
    schemes_req = list(req.get("schemes") or [])[:32]
    shared = dict(req.get("shared") or {})
    shared_drops = min(int(shared.get("drops", 16)), 32)

    results = []
    for entry_req in schemes_req:
        label = entry_req.get("label") or entry_req.get("codebook_id", "scheme")
        run_req = {
            "codebook_id": entry_req.get("codebook_id"),
            "params": entry_req.get("params") or {},
            "antenna": shared.get("antenna") or {},
            "n3": shared.get("n3", 8),
            "rank": shared.get("rank", 1),
            "channel": shared.get("channel") or {},
            "snr_db": shared.get("snr_db") or [-10, -5, 0, 5, 10, 15, 20, 25, 30],
            "drops": shared_drops,
            "seed": shared.get("seed", 0),
        }
        try:
            single = runner.run_playground(run_req)
        except runner.ValidationError as exc:
            results.append({
                "label": label, "scheme_name": None, "ok": False, "error": str(exc),
                "sgcs": None, "subspace_sgcs": None, "total_bits": None,
                "se": None, "se_upper_bound": None, "snr_db": None,
                "se_at_10db": None, "bound_at_10db": None,
            })
            continue
        except ValueError as exc:
            results.append({
                "label": label, "scheme_name": None, "ok": False, "error": str(exc),
                "sgcs": None, "subspace_sgcs": None, "total_bits": None,
                "se": None, "se_upper_bound": None, "snr_db": None,
                "se_at_10db": None, "bound_at_10db": None,
            })
            continue
        m = single["metrics"]
        results.append({
            "label": label,
            "scheme_name": single["scheme_name"],
            "ok": True,
            "error": None,
            "sgcs": m["sgcs"],
            "subspace_sgcs": m["subspace_sgcs"],
            "total_bits": m["total_bits"],
            "se": m["se"],
            "se_upper_bound": m["se_upper_bound"],
            "snr_db": m["snr_db"],
            "se_at_10db": m["se_at_10db"],
            "bound_at_10db": m["bound_at_10db"],
        })

    return {"ok": True, "seconds": round(time.time() - started, 3), "results": results}


# ------------------------------------------------------------------ figures


@app.get("/api/figures")
def list_figures():
    explain = content.figures_content()
    out = []
    for slug, fig in figures.FIGURES.items():
        extra = explain.get(slug, {}) if isinstance(explain, dict) else {}
        out.append({
            "slug": slug,
            "title": fig.title,
            "estSeconds": fig.est_seconds,
            "honorsFamilies": fig.honors_families,
            "cdlAvailable": figures.cdl_twin_exists(slug),
            "swept": list(fig.swept),
            "blurb": fig.blurb,
            **extra,
        })
    return out


@app.post("/api/figures/run")
def post_figures_run(req: dict):
    slugs = [s for s in (req.get("slugs") or []) if s in figures.FIGURES]
    if not slugs:
        return _error(400, "no valid figure slugs requested")

    def target(job):
        figures.run_figures_job(job, req)

    job_id = jobs.start_job("figures", target)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return _error(404, f"unknown job id '{job_id}'")
    return job.to_dict()


# --------------------------------------------------------------------- static


app.mount("/artifacts", StaticFiles(directory=str(RESULTS_DIR), check_dir=False), name="artifacts")

if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")
else:

    @app.get("/")
    def _no_ui():
        return PlainTextResponse(
            "CSI Codebook Studio UI is not built yet.\n\n"
            "Run:\n  cd webapp/ui && npm install && npm run build\n\n"
            "Then restart the server, or use `webapp/run.sh`.\n"
            "The API is live at /api (see /api/meta).",
        )
