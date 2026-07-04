#!/usr/bin/env bash
#
# Share CSI Codebook Studio with anyone, temporarily.
#
# Double-click this file (or run ./webapp/share.sh) to:
#   1. start the app on your own computer, and
#   2. open a temporary public link (via Cloudflare) that you can send to
#      anyone. The link works only while this window stays open.
#
# Press Ctrl-C (or just close the window) to stop sharing.

set -euo pipefail

PORT=8787
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- checks -----------------------------------------------------------------
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Install it once with:  brew install cloudflared"
  exit 1
fi

# --- 1. start the app (only if it isn't already running) --------------------
started_here=0
if curl -s -o /dev/null "http://localhost:${PORT}/api/meta"; then
  echo "✓ The app is already running on port ${PORT}."
else
  echo "Starting the app on port ${PORT} (first run builds the interface — this can take a minute)…"
  # Build the UI if needed, then run the server in the background.
  if [ ! -d "webapp/ui/dist" ] && command -v npm >/dev/null 2>&1 && [ -f "webapp/ui/package.json" ]; then
    (cd webapp/ui && npm install && npm run build)
  fi
  .venv/bin/python -m uvicorn webapp.server.main:app --port "${PORT}" >/tmp/codebook-studio.log 2>&1 &
  APP_PID=$!
  # Stop the app too when this script exits.
  trap 'echo; echo "Stopping…"; kill "${APP_PID}" 2>/dev/null || true' EXIT INT TERM
  # Wait until it answers (up to ~60s).
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "http://localhost:${PORT}/api/meta"; then break; fi
    sleep 1
  done
  if ! curl -s -o /dev/null "http://localhost:${PORT}/api/meta"; then
    echo "The app did not start. See /tmp/codebook-studio.log for details."
    exit 1
  fi
  echo "✓ App is up."
fi

# --- 2. open the public link ------------------------------------------------
echo
echo "Opening a public link… watch for the https://<something>.trycloudflare.com"
echo "address below — that is what you send to people. Keep this window open."
echo "-------------------------------------------------------------------------"
exec cloudflared tunnel --url "http://localhost:${PORT}"
