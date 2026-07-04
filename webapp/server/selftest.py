"""TestClient smoke test of every endpoint (SPEC.md §7).

Run:  .venv/bin/python -m webapp.server.selftest
"""

from __future__ import annotations

import sys
import time

from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"PASS  {name}")
    else:
        _FAIL += 1
        print(f"FAIL  {name}  {detail}")


def run() -> int:
    # ---------------------------------------------------------------- meta
    r = client.get("/api/meta")
    check("GET /api/meta 200", r.status_code == 200, r.text)
    body = r.json()
    check("meta has version+sionna_available",
          "version" in body and "sionna_available" in body, body)

    # ------------------------------------------------------------ codebooks
    r = client.get("/api/codebooks")
    check("GET /api/codebooks 200", r.status_code == 200, r.text)
    summaries = r.json()
    check("codebooks: 12 entries", len(summaries) == 12, len(summaries))
    expected_ids = {
        "type1-2port", "type1-sp", "type1-mp", "type2-r15", "etype2-r16",
        "fetype2-r17", "predicted-ps-r18", "etype2-doppler-r18", "cjt-r18",
        "refined-type1-r19", "refined-type1mp-r19", "refined-type2-r19",
    }
    got_ids = {s["id"] for s in summaries}
    check("codebooks: ids match SPEC §2", got_ids == expected_ids, got_ids ^ expected_ids)
    for s in summaries:
        for key in ("id", "name", "shortName", "release", "specClause", "tagline",
                    "ranks", "portRange", "position", "lineage"):
            check(f"summary[{s['id']}] has {key}", key in s, s)

    for cb_id in expected_ids:
        r = client.get(f"/api/codebooks/{cb_id}")
        check(f"GET /api/codebooks/{cb_id} 200", r.status_code == 200, r.text)
        detail = r.json()
        for key in ("params", "antenna", "defaults", "content"):
            check(f"detail[{cb_id}] has {key}", key in detail, detail.keys())
        check(f"detail[{cb_id}].antenna has mode+pairs",
              "mode" in detail["antenna"] and "pairs" in detail["antenna"], detail["antenna"])

        r_doc = client.get(f"/api/codebooks/{cb_id}/doc")
        check(f"GET /api/codebooks/{cb_id}/doc 200", r_doc.status_code == 200, r_doc.text)
        check(f"doc[{cb_id}] has markdown key", "markdown" in r_doc.json())

    r = client.get("/api/codebooks/does-not-exist")
    check("GET /api/codebooks/<bad> 404", r.status_code == 404, r.status_code)

    # -------------------------------------------------------------- content
    r = client.get("/api/content/home")
    check("GET /api/content/home 200", r.status_code == 200, r.text)
    r = client.get("/api/content/glossary")
    check("GET /api/content/glossary 200", r.status_code == 200, r.text)
    check("glossary is a list", isinstance(r.json(), list), type(r.json()))
    r = client.get("/api/content/foundations")
    check("GET /api/content/foundations 200", r.status_code == 200, r.text)
    check("foundations has markdown key", "markdown" in r.json())

    # ------------------------------------------------------------- validate
    good_req = {
        "codebook_id": "type1-sp",
        "params": {"codebook_mode": 1},
        "antenna": {"n1": 4, "n2": 2},
        "n3": 8, "rank": 1,
        "channel": {"preset": "sparse-urban"},
        "snr_db": [0, 10, 20], "drops": 4, "seed": 0,
    }
    r = client.post("/api/validate", json=good_req)
    check("POST /api/validate (good) 200", r.status_code == 200, r.text)
    check("validate (good) ok=True", r.json().get("ok") is True, r.json())

    bad_req = dict(good_req, codebook_id="etype2-r16",
                    params={"param_combination": 8}, antenna={"n1": 2, "n2": 1})
    r = client.post("/api/validate", json=bad_req)
    check("POST /api/validate (bad) 200", r.status_code == 200, r.text)
    check("validate (bad) ok=False with error", r.json().get("ok") is False and "error" in r.json(), r.json())

    # ------------------------------------------------------------------ run
    run_specs = [
        ("type1-sp", {"codebook_mode": 1}, {"n1": 4, "n2": 2}, {}),
        ("etype2-r16", {"param_combination": 4, "port_selection": False}, {"n1": 4, "n2": 2}, {}),
        ("etype2-doppler-r18", {"param_combination": 3, "N4": 2}, {"n1": 4, "n2": 2}, {"preset": "mobile-user"}),
        ("cjt-r18", {"n_trp": 2, "param_combination_L": 1, "param_combination": 1}, {"n1": 4, "n2": 2}, {}),
    ]
    run_results = {}
    for cb_id, params, antenna, channel in run_specs:
        req = {
            "codebook_id": cb_id, "params": params, "antenna": antenna,
            "n3": 8, "rank": 1, "channel": channel,
            "snr_db": [0, 10, 20], "drops": 4, "seed": 0,
        }
        t0 = time.time()
        r = client.post("/api/run", json=req)
        dt = time.time() - t0
        check(f"POST /api/run[{cb_id}] 200 ({dt:.1f}s)", r.status_code == 200, r.text[:800])
        if r.status_code != 200:
            continue
        body = r.json()
        run_results[cb_id] = body
        check(f"run[{cb_id}] ok=True", body.get("ok") is True, body)
        for key in ("seconds", "scheme_name", "config_echo", "python_snippet", "metrics", "pmi", "viz"):
            check(f"run[{cb_id}] has {key}", key in body, body.keys())
        m = body.get("metrics", {})
        for key in ("sgcs", "subspace_sgcs", "snr_db", "se", "se_upper_bound",
                    "overhead_bits", "total_bits"):
            check(f"run[{cb_id}].metrics has {key}", key in m, m.keys())
        viz = body.get("viz", {})
        for key in ("channel", "eigen_spectrum", "precoder"):
            check(f"run[{cb_id}].viz has {key}", key in viz, viz.keys())
        check(f"run[{cb_id}].pmi has fields list", isinstance(body.get("pmi", {}).get("fields"), list))

    check("run[type1-sp].viz has beam_grid", "beam_grid" in run_results.get("type1-sp", {}).get("viz", {}))

    # -------------------------------------------------------------- compare
    compare_req = {
        "schemes": [
            {"codebook_id": "type2-r15", "params": {"L": 4}, "label": "Type II L=4"},
            {"codebook_id": "etype2-r16", "params": {"param_combination": 4}, "label": "eType II pc4"},
            {"codebook_id": "type1-sp", "params": {"codebook_mode": 1}, "label": "Type I"},
        ],
        "shared": {
            "antenna": {"n1": 4, "n2": 2}, "n3": 8, "rank": 1,
            "channel": {"preset": "sparse-urban"}, "snr_db": [0, 10, 20],
            "drops": 4, "seed": 0,
        },
    }
    r = client.post("/api/compare", json=compare_req)
    check("POST /api/compare 200", r.status_code == 200, r.text[:800])
    body = r.json()
    check("compare ok=True", body.get("ok") is True, body)
    check("compare has 3 results", len(body.get("results", [])) == 3, body.get("results"))
    for res in body.get("results", []):
        check(f"compare result[{res.get('label')}] has ok key", "ok" in res, res)

    # ------------------------------------------------------------- figures
    r = client.get("/api/figures")
    check("GET /api/figures 200", r.status_code == 200, r.text)
    fig_list = r.json()
    check("figures: 13 entries", len(fig_list) == 13, len(fig_list))
    for f in fig_list:
        for key in ("slug", "title", "estSeconds", "honorsFamilies", "swept"):
            check(f"figure[{f.get('slug')}] has {key}", key in f, f)

    fig_req = {
        "slugs": ["fig_03_overhead_breakdown"], "channel": "synthetic",
        "families": list(figures_families()),
        "antenna": {"n1": 4, "n2": 2}, "n3": 8, "n_rx": 2,
        "n_paths": 4, "max_delay": 3.0,
        "drops": 8, "seed": 0, "fast": True,
    }
    r = client.post("/api/figures/run", json=fig_req)
    check("POST /api/figures/run 200", r.status_code == 200, r.text)
    job_id = r.json().get("job_id")
    check("figures/run returns job_id", bool(job_id), r.json())

    status = None
    if job_id:
        deadline = time.time() + 60
        while time.time() < deadline:
            r = client.get(f"/api/jobs/{job_id}")
            check_status = r.status_code == 200
            if not check_status:
                break
            status = r.json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        check("job reaches done", status is not None and status.get("status") == "done", status)
        if status:
            for key in ("id", "kind", "status", "progress", "message", "results"):
                check(f"job status has {key}", key in status, status.keys())
            results = status.get("results", [])
            check("job has 1 figure result", len(results) == 1, results)
            if results:
                res0 = results[0]
                check("figure result ok=True", res0.get("ok") is True, res0)
                check("figure result has png_url", bool(res0.get("png_url")), res0)

    r = client.get(f"/api/jobs/does-not-exist-{time.time()}")
    check("GET /api/jobs/<bad> 404", r.status_code == 404, r.status_code)

    # ---------------------------------------------------------- summary
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


def figures_families():
    from . import figures

    return figures.FAMILIES.keys()


if __name__ == "__main__":
    sys.exit(run())
