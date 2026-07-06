#!/usr/bin/env bash
# Compatibility wrapper for the workflow-first Aegis injection campaign.
# Prefer calling `agentm workflow run contrib/scenarios/injection/workflow.py`
# directly from automation; this script keeps the older loop.sh entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKFLOW_PATH="${SCRIPT_DIR}/workflow.py"

SYSTEM=""
ROUNDS=999
SLEEP=900
MODEL=""
STATE_BASE="${HOME}/.aegisctl/injection-author"
EXTRA_INSTRUCTION=""
AEGISCTL_BIN="/tmp/aegisctl"
SOURCE_DIR=""
HEARTBEAT_DIR=""
MAX_PARALLEL=1
VALIDATION_MODE=false
PROJECT="${AEGIS_PROJECT:-pair_diagnosis}"

usage() {
  sed -n '2,18p' "$0" >&2
  exit "${1:-2}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds) ROUNDS="$2"; shift 2 ;;
    --sleep) SLEEP="$2"; shift 2 ;;
    --model|--worker-model) MODEL="$2"; shift 2 ;;
    --state-dir) STATE_BASE="$2"; shift 2 ;;
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --extra-instruction) EXTRA_INSTRUCTION="$2"; shift 2 ;;
    --aegisctl-bin) AEGISCTL_BIN="$2"; shift 2 ;;
    --heartbeat-dir) HEARTBEAT_DIR="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --validation-mode) VALIDATION_MODE=true; shift ;;
    --escalate-after|--max-consecutive-failures|--max-turns)
      echo "warning: $1 is ignored by the workflow wrapper" >&2
      shift 2 ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown flag: $1" >&2; usage 2 ;;
    *)
      if [[ -z "$SYSTEM" ]]; then
        SYSTEM="$1"; shift
      else
        echo "unexpected positional: $1" >&2; usage 2
      fi
      ;;
  esac
done

if [[ -z "$SYSTEM" ]]; then
  echo "error: <system> is required (e.g. ts, hs, sn, mm, tea, sockshop, otel-demo)" >&2
  usage 2
fi
if [[ ! -x "$AEGISCTL_BIN" ]]; then
  echo "error: aegisctl binary not executable at $AEGISCTL_BIN — pass --aegisctl-bin /abs/path/to/aegisctl" >&2
  exit 127
fi
if [[ -n "$SOURCE_DIR" && ! -d "$SOURCE_DIR" ]]; then
  echo "error: --source-dir $SOURCE_DIR is not a directory" >&2
  exit 2
fi

STATE_DIR="${STATE_BASE}/${SYSTEM}"
mkdir -p "${STATE_DIR}/rounds"
LOG="${STATE_DIR}/loop.log"

ARGS_JSON=$(python3 - <<'PY' \
  "$SYSTEM" "$ROUNDS" "$SLEEP" "$STATE_DIR" "$AEGISCTL_BIN" "$PROJECT" \
  "$SOURCE_DIR" "$EXTRA_INSTRUCTION" "$MODEL" "$HEARTBEAT_DIR" "$MAX_PARALLEL" "$VALIDATION_MODE"
import json, sys
(
    system, rounds, sleep, state_dir, aegisctl_bin, project,
    source_dir, extra_instruction, model, heartbeat_dir, max_parallel,
    validation_mode,
) = sys.argv[1:]
payload = {
    "system": system,
    "rounds": int(rounds),
    "sleep": int(sleep),
    "state_dir": state_dir,
    "aegisctl_bin": aegisctl_bin,
    "project": project,
    "source_dir": source_dir or None,
    "extra_instruction": extra_instruction,
    "worker_model": model or None,
    "heartbeat_dir": heartbeat_dir or None,
    "max_parallel": int(max_parallel),
    "validation_mode": validation_mode.lower() == "true",
}
print(json.dumps(payload, ensure_ascii=False))
PY
)

{
  echo "[$(date -Iseconds)] running injection workflow: system=${SYSTEM} rounds=${ROUNDS} sleep=${SLEEP}s state=${STATE_DIR}"
  cd "$REPO_ROOT"
  uv run agentm workflow run "$WORKFLOW_PATH" --cwd "$STATE_DIR" --args "$ARGS_JSON"
} 2>&1 | tee -a "$LOG"
