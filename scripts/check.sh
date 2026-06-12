#!/usr/bin/env bash
# Full local verification: lint, fast suite, slow suite, optional Sionna suite.
# Usage: scripts/check.sh [--with-sionna] [--cov]
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_SIONNA=0
WITH_COV=0
for arg in "$@"; do
    case "$arg" in
        --with-sionna) WITH_SIONNA=1 ;;
        --cov) WITH_COV=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "== ruff =="
ruff check src tests scripts

COV_ARGS=()
if [ "$WITH_COV" -eq 1 ]; then
    COV_ARGS=(--cov=nr_csi --cov-report=term-missing)
fi

echo "== fast suite =="
python -m pytest -m "not slow and not sionna" -q "${COV_ARGS[@]}"

echo "== slow suite =="
python -m pytest -m slow -q

if [ "$WITH_SIONNA" -eq 1 ]; then
    echo "== sionna integration =="
    python -m pytest -m sionna -q
fi

echo "all checks passed"
