#!/usr/bin/env bash
# extractor-iter.sh — one full iteration of the prompt-tuning loop.
#
# Usage:
#   scripts/extractor-iter.sh <iter-name> [--prompt PATH] [--rows PATH] [--source PATH] [--concurrency N]
#
# Steps:
#   1. Re-run the extractor with the chosen prompt on the pinned 10 rows
#      (auditor records pass through; see scripts/rerun_extractor.py).
#   2. Fold the per-row records into the canonical cases/ layout via
#      `llmharness-aggregate eval-db`.
#   3. Upload that cases/ dir to aegis-blob under
#      shared/cases/iter-<iter-name>/.
#   4. Print the bucket/prefix to paste into the /cases UI Settings page.
#
# Idempotent: re-running with the same <iter-name> overwrites the local
# work dir and the blob prefix.
#
# Defaults:
#   --prompt       src/llmharness/audit/extractor/prompts/extractor_default.md
#   --rows         runs/iters/pinned-10.txt
#   --source       runs/eval_db/openrca-2-lite-n500-t20
#   --concurrency  10

set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat >&2 <<EOF
usage: $(basename "$0") <iter-name> [--prompt PATH] [--rows PATH] [--source PATH] [--concurrency N]

iter-name is a kebab-case slug, e.g. "baseline" or "external-refs-v2".
The cases land in blob bucket=shared prefix=cases/iter-<iter-name>/.
EOF
    exit 2
fi

ITER="$1"; shift
case "$ITER" in
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *[!a-z0-9_-]*|"") echo "iter-name must be kebab-case [a-z0-9_-]+ (got $ITER)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_PROMPT="$REPO_ROOT/src/llmharness/audit/extractor/prompts/extractor_default.md"
DEFAULT_ROWS="$REPO_ROOT/runs/iters/pinned-10.txt"
DEFAULT_SOURCE="$REPO_ROOT/runs/eval_db/openrca-2-lite-n500-t20"

# Source AgentM/.env if present so OPENAI_API_KEY / OPENAI_BASE_URL etc.
# reach the replayed extractor LLM client. Without these the OpenAI SDK
# refuses to even attempt the call ("api_key client option must be set")
# and every replay record fails with status=prompt_error — silently,
# from the orchestrator's point of view.
AGENTM_ENV="$REPO_ROOT/../../../.env"
if [[ -f "$AGENTM_ENV" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$AGENTM_ENV"
    set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY not set — source AgentM/.env or export it before running." >&2
    exit 3
fi

PROMPT="$DEFAULT_PROMPT"
ROWS="$DEFAULT_ROWS"
SOURCE="$DEFAULT_SOURCE"
CONC=10
FULL_TRAJECTORY=0
MAX_OUTPUT_TOKENS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)              PROMPT="$2"; shift 2 ;;
        --rows)                ROWS="$2"; shift 2 ;;
        --source)              SOURCE="$2"; shift 2 ;;
        --concurrency)         CONC="$2"; shift 2 ;;
        --full-trajectory)     FULL_TRAJECTORY=1; shift ;;
        --max-output-tokens)   MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

for f in "$PROMPT" "$ROWS"; do
    [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 2; }
done
[[ -d "$SOURCE" ]] || { echo "missing source dir: $SOURCE" >&2; exit 2; }

WORK="$REPO_ROOT/runs/iters/$ITER"
RAW="$WORK/raw"
CASES="$WORK/cases"
EXP_ID="$(basename "$SOURCE")"
BLOB_PREFIX="cases/iter-${ITER}/"

# Resolve python; prefer the AgentM venv that already has agentm + llmharness.
PY="${LLMH_PY:-/home/ddq/AoyangSpace/AgentM/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    PY="python3"
fi

echo "=> iter:     $ITER"
echo "=> prompt:   $PROMPT"
echo "=> rows:     $ROWS"
echo "=> source:   $SOURCE  (exp_id=$EXP_ID)"
echo "=> workdir:  $WORK"
echo "=> blob:     shared/$BLOB_PREFIX"
echo "=> python:   $PY"
echo

# 1. Re-run extractor with the chosen prompt.
mkdir -p "$RAW/$EXP_ID"
echo "==> 1/3 rerun extractor"
RERUN_EXTRA=()
if [[ "$FULL_TRAJECTORY" -eq 1 ]]; then
    RERUN_EXTRA+=("--full-trajectory")
fi
if [[ -n "$MAX_OUTPUT_TOKENS" ]]; then
    RERUN_EXTRA+=("--max-output-tokens" "$MAX_OUTPUT_TOKENS")
fi
"$PY" "$REPO_ROOT/scripts/rerun_extractor.py" \
    --src-root "$SOURCE" \
    --out-root "$RAW/$EXP_ID" \
    --rows-file "$ROWS" \
    --prompt "$PROMPT" \
    --concurrency "$CONC" \
    "${RERUN_EXTRA[@]}"

# 2. Aggregate into canonical cases/ layout.
echo
echo "==> 2/3 aggregate"
rm -rf "$CASES"
mkdir -p "$CASES"
# Pass --id filters so the aggregator picks up only our pinned set even
# if the raw dir contains stale unrelated rows.
ID_ARGS=()
while IFS= read -r line; do
    s="${line%%#*}"
    s="$(echo -n "$s" | tr -d '[:space:]')"
    [[ -n "$s" ]] && ID_ARGS+=("--id" "$s")
done < "$ROWS"
"$PY" -m llmharness.aggregate.cli eval-db \
    --src "$RAW/$EXP_ID" \
    --out "$CASES" \
    "${ID_ARGS[@]}"

# 3. Upload to blob.
echo
echo "==> 3/3 upload"
"$PY" "$REPO_ROOT/scripts/upload_cases_to_blob.py" \
    --src "$CASES" \
    --bucket shared \
    --prefix "$BLOB_PREFIX" \
    --concurrency "$CONC"

cat <<EOF

================================================================
Iteration "$ITER" complete.
  local cases:  $CASES
  blob prefix:  shared/$BLOB_PREFIX

Open https://118.196.98.178:8082/cases and in Settings set:
  bucket: shared
  prefix: $BLOB_PREFIX

Compare against the previous iter by switching prefix back. The case
ids are stable across iterations (e.g. openrca-2-lite-n500-t20-row1)
so deep-links survive.
================================================================
EOF
