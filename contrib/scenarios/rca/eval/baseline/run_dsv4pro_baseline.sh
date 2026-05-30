#!/usr/bin/env bash
set -euo pipefail

# DSv4pro baseline over ops-lite-clean (500 cases), concurrency 15.
# Run from repo root:
#   bash contrib/scenarios/rca/eval/baseline/run_dsv4pro_baseline.sh

export AGENTM_PROVIDER=openai
export AGENTM_MODEL=DeepSeek-V4-pro
export AGENTM_PROVIDER_NAME=litellm-dsv4pro
export OPENAI_BASE_URL=http://101.96.215.245:8088/v1
export OPENAI_API_KEY=sk-DLbuXPx8tzeb29atiigWEIXoU9P0xkh_2amQNqJHpMk
export OPENAI_VERIFY_SSL=false

exec uv run rca llm-eval run \
  contrib/scenarios/rca/eval/baseline/config_ops_lite_clean_dsv4pro.yaml \
  -a agentm \
  --ak scenario=rca:baseline \
  -n 15 \
  --timeout 1800 \
  --max-steps 60 \
  "$@"
