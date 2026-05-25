#!/usr/bin/env bash
# A/B over 100 ops-lite cases, both single runs (NO fork):
#   A = rca:baseline          (no auditor, no intervention at all)
#   B = rca:harness.sync      (auditor injects live every 5 turns; agent always accepts)
# Same 100 cases (ops-lite-fixed-100), paired → accuracy + flip rate.

set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

export RCA_DATASET_ROOT="${RCA_DATASET_ROOT:-$PWD/datasets/ops-lite/cases}"
export MODEL_NAME="${MODEL_NAME:-Doubao-Seed-2.0-pro}"
export JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_NAME}"

STAMP="$(date +%m%d-%H%M)"
EXP_A="agentm-ab100-baseline-$STAMP"
EXP_B="agentm-ab100-intervene-$STAMP"
CONC="${CONC:-5}"

# guard: snapshot eval.db before a multi-hour run
cp eval.db "eval.db.ab100-prerun-$STAMP" 2>/dev/null || true

# 1. tag the 100-case set
uv run --no-sync python contrib/scenarios/rca/eval/scripts/retag_fixed_set.py \
    --fixture contrib/scenarios/rca/eval/fixtures/ops-lite-fixed-100.txt \
    --dataset ops-lite --tag ops-lite-fixed-100

# 2. render config
out=$(mktemp /tmp/ab100-XXXXXX.yaml); trap 'rm -f "$out"' EXIT
envsubst < contrib/scenarios/rca/eval/config.ops-lite-fixed-100.yaml > "$out"

run_arm() {  # $1=exp_id  $2=scenario
  echo; echo ">>> ARM $2  exp_id=$1  (n=$CONC, 100 cases)"; echo
  uv run --no-sync rca llm-eval run "$out" \
      -a agentm --ak scenario="$2" \
      -n "$CONC" -l 100 --exp-id "$1"
}

# 3. Arm A then Arm B (sequential — avoid doubling provider load)
run_arm "$EXP_A" "rca:baseline"
run_arm "$EXP_B" "rca:harness.sync"

# 4. paired comparison + flip rate
echo; echo "=== A/B RESULT (paired on source) ==="
uv run --no-sync python - "$EXP_A" "$EXP_B" <<'PY'
import sqlite3, sys
a,b = sys.argv[1], sys.argv[2]
con = sqlite3.connect("eval.db")
def acc(exp):
    r = con.execute("select count(*), sum(correct) from evaluation_data where exp_id=? and stage='judged'",(exp,)).fetchone()
    return r[0] or 0, r[1] or 0
na,ca = acc(a); nb,cb = acc(b)
print(f"  A baseline   {a}: {ca}/{na}  acc={ca/na:.3f}" if na else f"  A {a}: no judged rows")
print(f"  B intervene  {b}: {cb}/{nb}  acc={cb/nb:.3f}" if nb else f"  B {b}: no judged rows")
# paired flip rate
rows = con.execute("""
  select A.source, A.correct, B.correct
  from   evaluation_data A join evaluation_data B on A.source=B.source
  where  A.exp_id=? and B.exp_id=? and A.stage='judged' and B.stage='judged'
""",(a,b)).fetchall()
paired=len(rows)
to_correct = sum(1 for _,ca_,cb_ in rows if not ca_ and cb_)   # wrong→right (auditor helped)
to_wrong   = sum(1 for _,ca_,cb_ in rows if ca_ and not cb_)   # right→wrong (auditor hurt)
same       = paired - to_correct - to_wrong
print(f"\n  paired cases: {paired}")
print(f"  flip wrong→right (auditor helped): {to_correct}")
print(f"  flip right→wrong (auditor hurt):   {to_wrong}")
print(f"  unchanged: {same}")
print(f"  net flip: {to_correct - to_wrong:+d}  | flip rate: {(to_correct+to_wrong)/paired:.3f}" if paired else "")
PY

echo; echo "exp_ids: A=$EXP_A  B=$EXP_B"
