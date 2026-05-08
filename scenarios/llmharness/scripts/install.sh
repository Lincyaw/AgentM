#!/usr/bin/env bash
# Start the llmharness background worker for the current project.
#
# This plugin's hooks are wired automatically by Claude Code via
# hooks/hooks.json once the plugin is installed. The worker that consumes
# the inbox is *not* run by Claude Code — it must be started here.
#
# Usage:
#   scripts/install.sh                  # start worker, polling every 5s
#   scripts/install.sh --no-worker      # do nothing (hooks alone are wired by CC)
#
# Env (also picked up by the hooks at runtime):
#   LLMHARNESS_ROOT       harness storage root (default: $PWD/.harness)
#   LLMHARNESS_PROVIDER   rule | agentm   (default: rule)
#   LLMHARNESS_AGENTM_BIN agentm CLI binary (default: agentm)
#   LLMHARNESS_AGENTM_CWD AgentM checkout root (where scenarios/ lives)
#   LLMHARNESS_AGENTM_MODEL provider model id, e.g. claude-sonnet-4-6
#   LLMHARNESS_DISTILL_DIR  if set, dump (input,output) pairs for each call
set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_ROOT="${LLMHARNESS_ROOT:-${PWD}/.harness}"
PID_FILE="${HARNESS_ROOT}/worker.pid"
LOG_FILE="${HARNESS_ROOT}/worker.log"

START_WORKER=1
for arg in "$@"; do
  case "$arg" in
    --no-worker) START_WORKER=0 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
  esac
done

mkdir -p "$HARNESS_ROOT"

if [[ "$START_WORKER" -eq 1 ]]; then
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "stopping previous worker $(cat "$PID_FILE")"
    kill "$(cat "$PID_FILE")" || true
    sleep 0.2
  fi
  LLMHARNESS_ROOT="$HARNESS_ROOT" \
    nohup bash -c '
      while true; do
        bash "'"$PLUGIN_ROOT"'/scripts/tick_worker.sh" || true
        sleep 5
      done
    ' >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "started worker pid $(cat "$PID_FILE") (log: $LOG_FILE)"
fi

cat <<EOF
done.
- harness root: $HARNESS_ROOT
- to stop the worker: kill \$(cat $PID_FILE)
EOF
