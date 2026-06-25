#!/bin/bash
# Build and push terminal-bench task images to a container registry.
#
# Works with any terminal-bench-format benchmark (longcli-bench,
# terminal-bench core, custom task sets) — any directory containing
# task subdirectories with a Dockerfile + INSTRUCTION.md.
#
# Usage:
#   bash contrib/evals/longcli/setup_images.sh [OPTIONS]
#
# Options:
#   --tasks-dir  DIR     Directory containing task subdirectories (required)
#   --base-dir   DIR     Directory with base Dockerfiles (optional; auto-builds
#                        tb/make-pytest:v0 and tb/c-env:v0 if found)
#   --registry   REG     Registry prefix (default: opspai)
#   --prefix     PREFIX  Image name prefix (default: longcli)
#   --tag        TAG     Image tag (default: v0)
#   --push               Push images to registry after building
#   --load-kind  CLUSTER Also load images into a kind cluster
#
# Examples:
#   # LongCLI-Bench
#   bash setup_images.sh --tasks-dir ~/longcli-bench/tasks_long_cli \
#     --base-dir ~/longcli-bench/longcli_dockerImage --push
#
#   # Terminal-Bench core
#   bash setup_images.sh --tasks-dir ~/terminal-bench/tasks \
#     --prefix tb-core --push
#
#   # Custom task set
#   bash setup_images.sh --tasks-dir ./my-tasks --prefix custom --push
#
# ARL usage (no build needed on the target cluster):
#   AGENTM_AGENT_ENV_IMAGE="opspai/longcli-cs61_fa24_hog:v0" \
#   uv run agentm --scenario terminal_bench_arl --model glm47 -p "..."

set -euo pipefail

TASKS_DIR=""
BASE_DIR=""
REGISTRY="opspai"
PREFIX="longcli"
TAG="v0"
PUSH=false
KIND_CLUSTER=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --tasks-dir)  TASKS_DIR="$2"; shift 2 ;;
    --base-dir)   BASE_DIR="$2"; shift 2 ;;
    --registry)   REGISTRY="$2"; shift 2 ;;
    --prefix)     PREFIX="$2"; shift 2 ;;
    --tag)        TAG="$2"; shift 2 ;;
    --push)       PUSH=true; shift ;;
    --load-kind)  KIND_CLUSTER="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "$TASKS_DIR" ]; then
  echo "Error: --tasks-dir is required."
  echo "Usage: bash setup_images.sh --tasks-dir <path> [--push]"
  exit 1
fi

if [ ! -d "$TASKS_DIR" ]; then
  echo "Error: $TASKS_DIR not found."
  exit 1
fi

# --- Step 1: Build base images (if base-dir provided) ---
if [ -n "$BASE_DIR" ] && [ -d "$BASE_DIR" ]; then
  echo "=== Building base images from $BASE_DIR ==="
  for pattern in Dockerfile.*-base Dockerfile.*_base; do
    for dockerfile in "$BASE_DIR"/$pattern; do
      [ ! -f "$dockerfile" ] && continue
      # Extract name: Dockerfile.make-pytest-base -> make-pytest
      base_name=$(basename "$dockerfile" | sed 's/Dockerfile\.\(.*\)-base/\1/' | sed 's/Dockerfile\.\(.*\)_base/\1/')
      echo "  Building tb/$base_name:v0 ..."
      docker build -f "$dockerfile" -t "tb/$base_name:v0" "$BASE_DIR" -q >/dev/null
    done
  done
  echo ""
fi

# --- Step 2: Build + tag task images ---
echo "=== Building task images from $TASKS_DIR ==="
IMAGES=()
SKIPPED=0
for task_dir in "$TASKS_DIR"/*/; do
  [ ! -d "$task_dir" ] && continue
  task=$(basename "$task_dir")
  [ ! -f "$task_dir/Dockerfile" ] && { SKIPPED=$((SKIPPED+1)); continue; }
  # Must have a FROM line
  grep -q "^FROM" "$task_dir/Dockerfile" || { SKIPPED=$((SKIPPED+1)); continue; }

  registry_tag="$REGISTRY/$PREFIX-$task:$TAG"
  echo "  $task -> $registry_tag"
  docker build -t "$registry_tag" "$task_dir" -q >/dev/null
  IMAGES+=("$registry_tag")
done

echo ""
echo "Built ${#IMAGES[@]} images (skipped $SKIPPED)."

# --- Step 3: Push ---
if $PUSH; then
  echo ""
  echo "=== Pushing to $REGISTRY ==="
  for img in "${IMAGES[@]}"; do
    echo "  $img"
    docker push "$img" -q
  done
  echo "Pushed ${#IMAGES[@]} images."
fi

# --- Step 4: Load into kind (optional) ---
if [ -n "$KIND_CLUSTER" ]; then
  if kind get clusters 2>/dev/null | grep -q "^${KIND_CLUSTER}$"; then
    echo ""
    echo "=== Loading into kind cluster '$KIND_CLUSTER' ==="
    kind load docker-image "${IMAGES[@]}" --name "$KIND_CLUSTER"
  else
    echo "Warning: kind cluster '$KIND_CLUSTER' not found."
  fi
fi

echo ""
echo "=== Summary ==="
for img in "${IMAGES[@]}"; do
  echo "  $img"
done
