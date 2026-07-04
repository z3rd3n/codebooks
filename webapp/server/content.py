"""Loads curated content JSON (written by the content agent) and docs markdown.

Tolerant of missing/partial content: every accessor falls back to an empty
but well-shaped structure so the API never 500s just because
``webapp/server/content/`` is incomplete or absent (a second agent authors
those files in parallel with this backend).
"""

from __future__ import annotations

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTENT_DIR = pathlib.Path(__file__).resolve().parent / "content"
DOCS_DIR = REPO_ROOT / "docs" / "codebooks"


def _read_json(path: pathlib.Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def codebook_content(codebook_id: str) -> dict:
    """content/codebooks/<id>.json, or an empty-but-shaped dict if missing."""
    data = _read_json(CONTENT_DIR / "codebooks" / f"{codebook_id}.json")
    if isinstance(data, dict):
        return data
    return {
        "overview": [],
        "howItWorks": [],
        "whatIsReported": [],
        "parametersExplained": [],
        "strengths": [],
        "limitations": [],
        "whenToUse": "",
        "mathHighlight": None,
        "glossary": [],
    }


def glossary() -> list:
    data = _read_json(CONTENT_DIR / "glossary.json")
    return data if isinstance(data, list) else []


def figures_content() -> dict:
    data = _read_json(CONTENT_DIR / "figures.json")
    return data if isinstance(data, dict) else {}


def home_content() -> dict:
    data = _read_json(CONTENT_DIR / "home.json")
    if isinstance(data, dict):
        return data
    return {"hero": {"title": "", "subtitle": ""}, "story": [], "timeline": [], "concepts": []}


def doc_markdown(doc_file: str) -> str:
    """Raw markdown text of a docs/codebooks/<doc_file> chapter."""
    path = DOCS_DIR / doc_file
    if not path.exists():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def foundations_markdown() -> str:
    return doc_markdown("00-foundations.md")
