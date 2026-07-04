#!/usr/bin/env bash
# Build the UI (if missing and npm is available) and start the backend.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d "webapp/ui/dist" ] && command -v npm >/dev/null 2>&1 && [ -f "webapp/ui/package.json" ]; then
  echo "Building webapp/ui (dist missing)..."
  (cd webapp/ui && npm install && npm run build)
fi

exec .venv/bin/python -m uvicorn webapp.server.main:app --port 8787
