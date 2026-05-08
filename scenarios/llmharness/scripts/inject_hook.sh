#!/usr/bin/env bash
# UserPromptSubmit hook: emit and consume any pending reminder for this session.
set -euo pipefail

ROOT="${LLMHARNESS_ROOT:-${PWD}/.harness}"
PYTHON="${LLMHARNESS_PYTHON:-python3}"
PKG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

PYTHONPATH="${PKG_SRC}${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m llmharness --root "$ROOT" inject --from-hook
