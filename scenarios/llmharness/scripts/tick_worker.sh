#!/usr/bin/env bash
# Background worker tick. Iterates every session inbox and runs one
# summarize+detect pass. Run from cron or a long-running daemon (see install.sh).
set -euo pipefail

ROOT="${LLMHARNESS_ROOT:-${PWD}/.harness}"
PYTHON="${LLMHARNESS_PYTHON:-python3}"
PKG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

INBOX_DIR="${ROOT}/inbox"
[[ -d "$INBOX_DIR" ]] || exit 0

shopt -s nullglob
for f in "$INBOX_DIR"/*.jsonl; do
  sid="$(basename "$f" .jsonl)"
  PYTHONPATH="${PKG_SRC}${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m llmharness tick --root "$ROOT" --session "$sid" || true
done
