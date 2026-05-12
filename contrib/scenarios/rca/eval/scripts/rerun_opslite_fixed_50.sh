#!/usr/bin/env bash
# Re-run the frozen 50-sample ops-lite eval against AgentM rca:harness.sync.
#
# Use this every time a strategy / atom / prompt change should be
# compared against the 2026-05-11 baseline. The 50 case names are
# pinned in ``contrib/scenarios/rca/eval/fixtures/ops-lite-fixed-50.txt``;
# this script re-tags eval.db to match the file (idempotent) before
# running, so even if rcabench-platform's DB ordering changes, the
# same 50 cases stay in scope.
#
# Required env (sourced from .env if present): AGENTM_PROVIDER,
#   OPENAI_BASE_URL, WARPGATE_TICKET, OPENAI_VERIFY_SSL. Override
#   ``RCA_DATASET_ROOT`` to point at a different ops-lite checkout;
#   override ``MODEL_NAME`` / ``JUDGE_MODEL`` to compare different LLMs
#   against the same case set.
#
# Usage:
#   bash contrib/scenarios/rca/eval/scripts/rerun_opslite_fixed_50.sh \
#       [<exp_id_suffix>] [<scenario>]
#
#   exp_id_suffix: free-form tag appended to the experiment id so each
#                  iteration shows up as a distinct row in eval.db
#                  (default: ``baseline``). Use e.g. ``after-prompt-tweak``.
#   scenario:      AgentM scenario name (default: ``rca:harness.sync``).
#                  Pass ``rca:harness`` or ``rca:baseline`` to compare
#                  across audit modes.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$repo_root"

# Load .env (without clobbering already-exported vars).
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

exp_suffix="${1:-baseline}"
scenario="${2:-rca:harness.sync}"

export RCA_DATASET_ROOT="${RCA_DATASET_ROOT:-$repo_root/datasets/ops-lite/cases}"
export MODEL_NAME="${MODEL_NAME:-${AGENTM_MODEL:-Doubao-Seed-2.0-pro}}"
export JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_NAME}"

# 1. Make eval.db agree with the on-disk fixture (idempotent).
uv run --no-sync python contrib/scenarios/rca/eval/scripts/retag_fixed_set.py

# 2. Render the env-substituted config.
out_yaml="$(mktemp /tmp/opslite-fixed-50-XXXXXX.yaml)"
trap 'rm -f "$out_yaml"' EXIT
envsubst < contrib/scenarios/rca/eval/config.ops-lite-fixed-50.yaml > "$out_yaml"

# 3. Run. The exp_id makes each iteration a distinct row in eval.db so
#    you can diff metrics across attempts with a plain SQL query.
exp_id="agentm-rca-opslite-fixed-50-${exp_suffix}"
echo "running fixed-50 eval: exp_id=$exp_id  scenario=$scenario  model=$MODEL_NAME"
uv run --no-sync rca llm-eval run "$out_yaml" \
    -a agentm \
    --ak "scenario=$scenario" \
    -n 5 \
    --exp-id "$exp_id"

echo
echo "done. metrics row:"
uv run --no-sync python - <<PY
import json, sqlite3
con = sqlite3.connect("eval.db")
rows = con.execute(
    "select count(*) filter (where correct = 1), count(*) "
    "from evaluation_data where exp_id=? and stage='judged'",
    ("$exp_id",),
).fetchone()
print(f"  $exp_id: {rows[0]}/{rows[1]} correct")
PY
