#!/usr/bin/env bash
# One-shot launcher: chained_fork eval on a baseline-FAILED narrowing case.
# GT = {search, rate, recommendation}; clean baseline found only {search}.
# Tests whether the auditor reminder recovers the missing root causes.
# Pairs chained_fork with opinions10 (clean baseline, no live injection).

set -euo pipefail

cd "$(dirname "$0")"

set -a
source .env
set +a

export RCA_DATASET_ROOT="${RCA_DATASET_ROOT:-$PWD/datasets/ops-lite/cases}"
export MODEL_NAME="${MODEL_NAME:-Doubao-Seed-2.0-pro}"
export JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_NAME}"

TAG="ops-lite-narrow-case"

# 1. Make sure eval.db has the single-case tag.
uv run --no-sync python contrib/scenarios/rca/eval/scripts/retag_fixed_set.py \
    --fixture contrib/scenarios/rca/eval/fixtures/ops-lite-narrow-case.txt \
    --dataset ops-lite \
    --tag "$TAG"

# 2. Render config with substituted env + remap tag.
out=$(mktemp /tmp/opslite-narrow-XXXXXX.yaml)
trap 'rm -f "$out"' EXIT
envsubst < contrib/scenarios/rca/eval/config.ops-lite-fixed-50.yaml \
  | sed "s/ops-lite-fixed-50/$TAG/g" > "$out"

exp_id="agentm-chained-narrow-promptv2-$(date +%H%M)"
echo
echo ">>> exp_id=$exp_id  scenario=rca:harness.sync.opinions10  chained_fork=true  max_interventions=5"
echo

uv run --no-sync rca llm-eval run "$out" \
    -a agentm \
    --ak scenario=rca:harness.sync.opinions10 \
    --ak chained_fork=true \
    --ak max_interventions=5 \
    -n 1 -l 1 \
    --exp-id "$exp_id"

echo
echo "=== judge result ==="
uv run --no-sync python - <<PY
import sqlite3
con = sqlite3.connect("eval.db")
row = con.execute(
    "select count(*) filter (where correct=1), count(*) "
    "from evaluation_data where exp_id=? and stage='judged'",
    ("$exp_id",),
).fetchone()
print(f"  {row[0]}/{row[1]} correct")
PY

echo
echo "=== artifacts ==="
echo "  observability JSONL: .agentm/observability/<sid>.jsonl"
echo "  chained sidecar:     .agentm/audit_replay/*.chained.jsonl (look for __chain_header__)"
echo "  experiment id:       $exp_id"
