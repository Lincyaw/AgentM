#!/usr/bin/env bash
# Progress / review helper for the 100-case A/B run (baseline vs always-accept-auditor).
# Usage: bash ab100_status.sh [STAMP]   (STAMP defaults to 0525-0847)
set -euo pipefail
cd "$(dirname "$0")"
STAMP="${1:-0525-0847}"
A="agentm-ab100-baseline-$STAMP"
B="agentm-ab100-intervene-$STAMP"

echo "=== run: $STAMP ==="
echo "  A (baseline)  exp_id=$A   scenario=rca:baseline"
echo "  B (intervene) exp_id=$B   scenario=rca:harness.sync"
echo

echo "=== process ==="
ps -eo pid,etime,cmd | grep "ab100-.*-$STAMP" | grep -v grep | head -2 || echo "  (no live rca llm-eval process — finished or not started)"
echo

echo "=== progress (per arm, per stage) ==="
sqlite3 eval.db "select exp_id, stage, count(*) n, sum(correct) correct
                 from evaluation_data
                 where exp_id in ('$A','$B') group by exp_id, stage order by exp_id, stage;" 2>&1
echo

echo "=== accuracy + paired flip rate (only meaningful once both arms judged) ==="
uv run --no-sync python - "$A" "$B" <<'PY'
import sqlite3, sys
a,b=sys.argv[1],sys.argv[2]
con=sqlite3.connect("eval.db")
def acc(e):
    r=con.execute("select count(*),sum(correct) from evaluation_data where exp_id=? and stage='judged'",(e,)).fetchone()
    return r[0] or 0, r[1] or 0
na,ca=acc(a); nb,cb=acc(b)
print(f"  A baseline  judged={na} correct={ca}" + (f" acc={ca/na:.3f}" if na else ""))
print(f"  B intervene judged={nb} correct={cb}" + (f" acc={cb/nb:.3f}" if nb else ""))
rows=con.execute("""select A.source,A.correct,B.correct from evaluation_data A
   join evaluation_data B on A.source=B.source
   where A.exp_id=? and B.exp_id=? and A.stage='judged' and B.stage='judged'""",(a,b)).fetchall()
p=len(rows)
if p:
    h=sum(1 for _,x,y in rows if not x and y)   # wrong->right
    w=sum(1 for _,x,y in rows if x and not y)   # right->wrong
    print(f"  paired={p}  helped(wrong->right)={h}  hurt(right->wrong)={w}  unchanged={p-h-w}")
    print(f"  net={h-w:+d}  flip_rate={(h+w)/p:.3f}")
    # list the flipped cases for review
    fl=[(s,'W->R' if (not x and y) else 'R->W') for s,x,y in rows if x!=y]
    if fl:
        print("  flipped cases:")
        for s,d in fl: print(f"    {d}  {s}")
else:
    print("  (paired comparison pending — B not judged yet)")
PY
echo
echo "=== artifacts ==="
echo "  log:        /tmp/ab100.log"
echo "  eval-data:  eval-data/$A/  eval-data/$B/"
echo "  trace map:  uv run --no-sync python contrib/scenarios/rca/eval/scripts/build_trace_map.py --exp-id $A --out eval-data/$A/trace_map.json"
echo "  per-case trajectory: sqlite3 eval.db \"select source,trace_id from evaluation_data where exp_id='$A'\" then grep trace_id in .agentm/observability/"
