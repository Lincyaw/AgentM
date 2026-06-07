#!/usr/bin/env bash
#
# update.sh — pull latest code and restart the AgentM gateway with minimal
# user-visible disruption.
#
# Default policy: restart ONLY the gateway. The feishu client auto-reconnects
# to the restarted gateway (Phase 1) and the in-flight reply is replayed from
# its outbox, so the feishu daemon almost never needs a restart. Pass
# --with-feishu when the *feishu client code itself* changed.
#
# Usage:
#   ./update.sh                # git pull + uv sync + restart gateway only
#   ./update.sh --with-feishu  # also restart the feishu client
#   ./update.sh --all-packages # uv sync --all-packages (if feishu unit uses
#                              # plain `uv run agentm-feishu` instead of
#                              # `uv run --package agentm-feishu`)
#
# Run as the operator who can `sudo systemctl restart` (or set SUDO="" and run
# this whole script as a user with the rights, e.g. for `systemctl --user`).

set -euo pipefail

# --- locate and verify the repo root -----------------------------------------
# This script lives at contrib/gateway-peers/deploy/update.sh, so the repo root
# is three levels up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if ! grep -q '\[tool.uv.workspace\]' "${REPO_ROOT}/pyproject.toml" 2>/dev/null; then
  echo "ERROR: ${REPO_ROOT} does not look like the AgentM repo root" \
       "(no [tool.uv.workspace] in pyproject.toml). Aborting." >&2
  exit 1
fi

# --- parse flags --------------------------------------------------------------
WITH_FEISHU=0
SYNC_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --with-feishu)  WITH_FEISHU=1 ;;
    --all-packages) SYNC_ARGS+=(--all-packages) ;;
    *) echo "ERROR: unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

# Units are user units (installed by `agentm gateway --install-systemd`), so
# restarts go through `systemctl --user` (no sudo).
SCTL=(systemctl --user)

echo "==> Repo root: ${REPO_ROOT}"
BEFORE="$(git rev-parse --short HEAD)"
echo "==> Current commit (before): ${BEFORE}"

# --- pull (fast-forward only; never auto-merge) -------------------------------
echo "==> git pull --ff-only"
git pull --ff-only

AFTER="$(git rev-parse --short HEAD)"
echo "==> Current commit (after):  ${AFTER}"
if [[ "${BEFORE}" == "${AFTER}" ]]; then
  echo "==> No new commits pulled — restarting anyway to pick up any local/env changes."
fi

# --- sync dependencies --------------------------------------------------------
echo "==> uv sync ${SYNC_ARGS[*]:-}"
uv sync "${SYNC_ARGS[@]}"

# --- restart only what is necessary -------------------------------------------
# The gateway always restarts (it holds the changed code paths and sessions).
# The feishu client is left running by default: it re-dials the new gateway and
# replays its outbox, so users perceive at most a brief pause, not a restart.
SERVICES=(agentm-gateway)
if [[ "${WITH_FEISHU}" -eq 1 ]]; then
  SERVICES+=(agentm-feishu)
  echo "==> --with-feishu: feishu client WILL also be restarted (its code changed)."
else
  echo "==> feishu client left running (auto-reconnects + outbox replay)."
fi

echo "==> ${SCTL[*]} restart ${SERVICES[*]}"
"${SCTL[@]}" restart "${SERVICES[@]}"

echo "==> Done. ${BEFORE} -> ${AFTER}. Restarted: ${SERVICES[*]}"
echo "    Tail logs with:  journalctl --user -u agentm-gateway -f"
