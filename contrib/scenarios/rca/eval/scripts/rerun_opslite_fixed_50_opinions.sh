#!/usr/bin/env bash
# Run the frozen 50-sample ops-lite eval against
# ``rca:harness.sync.opinions``: synchronous extractor + auditor child
# sessions whose verdicts are recorded but NEVER injected back into the
# main agent (``enable_reminders: false``). Use this when you want
# per-case audit data for offline review without letting the audit
# influence trajectories.
#
# After the run finishes, aggregate the replay sidecars into
# ``./cases-opinions/<case>/`` with ``llmharness-aggregate``.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

exp_suffix="${1:-opinions}"
scenario="${2:-rca:harness.sync.opinions}"
out_dir="${3:-$repo_root/cases-opinions}"

export RCA_DATASET_ROOT="${RCA_DATASET_ROOT:-$repo_root/datasets/ops-lite/cases}"
export MODEL_NAME="${MODEL_NAME:-${AGENTM_MODEL:-Doubao-Seed-2.0-pro}}"
export JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_NAME}"

uv run --no-sync python contrib/scenarios/rca/eval/scripts/retag_fixed_set.py

rendered_yaml="$(mktemp /tmp/opslite-fixed-50-opinions-XXXXXX.yaml)"
trap 'rm -f "$rendered_yaml"' EXIT
envsubst < contrib/scenarios/rca/eval/config.ops-lite-fixed-50.yaml > "$rendered_yaml"

exp_id="agentm-rca-opslite-fixed-50-${exp_suffix}"
echo "[1/2] running eval: exp_id=$exp_id  scenario=$scenario  model=$MODEL_NAME"
uv run --no-sync rca llm-eval run "$rendered_yaml" \
    -a agentm \
    --ak "scenario=$scenario" \
    -n 5 \
    --exp-id "$exp_id"

echo
echo "[2/2] aggregating replay sidecars to $out_dir/"
mkdir -p "$out_dir"

# Pair each rolled-out trace with its case name from eval.db, then drive
# llmharness-aggregate once per (root_session_id, sample_id) pair so the
# case directory is keyed by the dataset case name rather than the bare
# session id.
mapfile -t pairs < <(
  uv run --no-sync python - <<PY "$exp_id"
import sys, sqlite3
exp_id = sys.argv[1]
con = sqlite3.connect("eval.db")
for trace_id, source in con.execute(
    "select trace_id, source from evaluation_data "
    "where exp_id=? and stage='rolled_out' and trace_id is not null",
    (exp_id,),
):
    print(f"{trace_id}\t{source}")
PY
)

if [[ ${#pairs[@]} -eq 0 ]]; then
    echo "no rollout rows found for $exp_id — check eval.db" >&2
    exit 2
fi

for line in "${pairs[@]}"; do
    sid="${line%%$'\t'*}"
    case_name="${line##*$'\t'}"
    [[ -z "$sid" || -z "$case_name" ]] && continue
    if [[ ! -f ".agentm/audit_replay/${sid}.jsonl" ]]; then
        echo "  skip $case_name — no replay sidecar at .agentm/audit_replay/${sid}.jsonl" >&2
        continue
    fi
    uv run --no-sync llmharness-aggregate \
        --cwd "$repo_root" \
        --root-session-id "$sid" \
        --sample-id "$case_name" \
        --dataset-name "ops-lite" \
        --dataset-path "$RCA_DATASET_ROOT" \
        --out "$out_dir"
done

echo
echo "done. $(find "$out_dir" -mindepth 1 -maxdepth 1 -type d | wc -l) case directories under $out_dir/"
