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

# Pair each rolled-out session with its case name from eval.db, then
# drive llmharness-aggregate once per (session_id, sample_id) pair so the
# case directory is keyed by the dataset case name rather than the bare
# session id.
#
# Identity bridge: ``evaluation_data.trace_id`` stores the OTel *trace_id*
# (= ``session.root_session_id``), but the replay sidecar is keyed by the
# main agent's *session_id* (``.agentm/audit_replay/<session_id>.jsonl``,
# = the ``.agentm/observability/<session_id>.jsonl`` filename). The two
# differ for a real root session and eval.db does NOT persist the
# session_id, so we recover it from the observability log: every session's
# ``session.start`` record carries both ids. ``TraceReader.first_session_identity``
# is the in-process form of the ``agentm trace index`` verb (CLI form:
# ``agentm trace index --format ndjson | jq 'select(.trace_id==…)'``).
mapfile -t pairs < <(
  uv run --no-sync python - <<PY "$exp_id" "$repo_root"
import sys, sqlite3
from pathlib import Path
from agentm.core.abi import TraceReader

exp_id, repo_root = sys.argv[1], sys.argv[2]

# Build trace_id -> main session_id from the observability tree (the only
# place both ids coexist). purpose "root"/None is the main agent session;
# spawned extractor/auditor children share the trace_id but are not it.
sid_of_trace: dict[str, str] = {}
obs_dir = Path(repo_root) / ".agentm" / "observability"
for p in obs_dir.glob("*.jsonl"):
    ident = TraceReader(p).first_session_identity()
    if ident and ident.trace_id and ident.purpose in (None, "root"):
        sid_of_trace[ident.trace_id] = ident.session_id or ""

con = sqlite3.connect("eval.db")
for trace_id, source in con.execute(
    "select trace_id, source from evaluation_data "
    "where exp_id=? and stage='rolled_out' and trace_id is not null",
    (exp_id,),
):
    # Fall back to the trace_id only if the bridge missed (observability
    # pruned); the sidecar-exists guard below then skips it rather than
    # aggregating the wrong file.
    print(f"{sid_of_trace.get(trace_id) or trace_id}\t{source}")
PY
)

if [[ ${#pairs[@]} -eq 0 ]]; then
    echo "no rollout rows found for $exp_id — check eval.db" >&2
    exit 2
fi

for line in "${pairs[@]}"; do
    session_id="${line%%$'\t'*}"
    case_name="${line##*$'\t'}"
    [[ -z "$session_id" || -z "$case_name" ]] && continue
    if [[ ! -f ".agentm/audit_replay/${session_id}.jsonl" ]]; then
        echo "  skip $case_name — no replay sidecar at .agentm/audit_replay/${session_id}.jsonl" >&2
        continue
    fi
    uv run --no-sync llmharness-aggregate replay \
        --cwd "$repo_root" \
        --session-id "$session_id" \
        --sample-id "$case_name" \
        --dataset-name "ops-lite" \
        --dataset-path "$RCA_DATASET_ROOT" \
        --out "$out_dir"
done

echo
echo "done. $(find "$out_dir" -mindepth 1 -maxdepth 1 -type d | wc -l) case directories under $out_dir/"
