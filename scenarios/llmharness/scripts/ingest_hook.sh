#!/usr/bin/env bash
# PostToolUse / Stop hook: append transcript delta to the inbox, return immediately.
set -euo pipefail

ROOT="${LLMHARNESS_ROOT:-${PWD}/.harness}"
PYTHON="${LLMHARNESS_PYTHON:-python3}"
PKG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

PYTHONPATH="${PKG_SRC}${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m llmharness --root "$ROOT" ingest --from-hook >/dev/null
