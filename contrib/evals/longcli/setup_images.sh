#!/bin/bash
# Build all longcli task Docker images and load into kind cluster.
# Usage: bash contrib/evals/longcli/setup_images.sh [LONGCLI_DIR] [KIND_CLUSTER]
#
# Prerequisites:
#   - Docker
#   - kind
#   - longcli-bench repo cloned locally
#   - Base images already built:
#       cd <longcli-dir>/longcli_dockerImage
#       docker build -f Dockerfile.make-pytest-base -t tb/make-pytest:v0 .
#       docker build -f Dockerfile.c-env-base -t tb/c-env:v0 .

set -euo pipefail

LONGCLI_DIR="${1:-$HOME/AoyangSpace/longcli-bench}"
KIND_CLUSTER="${2:-arl-agentm}"
TASKS_DIR="$LONGCLI_DIR/tasks_long_cli"

if [ ! -d "$TASKS_DIR" ]; then
  echo "Error: $TASKS_DIR not found. Pass the longcli-bench repo path as first argument."
  exit 1
fi

# Check base images exist
for base in tb/make-pytest:v0 tb/c-env:v0; do
  if ! docker image inspect "$base" >/dev/null 2>&1; then
    echo "Error: base image $base not found. Build it first:"
    echo "  cd $LONGCLI_DIR/longcli_dockerImage"
    echo "  docker build -f Dockerfile.make-pytest-base -t tb/make-pytest:v0 ."
    echo "  docker build -f Dockerfile.c-env-base -t tb/c-env:v0 ."
    exit 1
  fi
done

echo "Building task images from $TASKS_DIR..."
IMAGES=()
for task_dir in "$TASKS_DIR"/*/; do
  task=$(basename "$task_dir")
  [ "$task" = "build_task.md" ] && continue
  # Skip tasks without a valid Dockerfile
  [ ! -f "$task_dir/Dockerfile" ] && continue
  head -1 "$task_dir/Dockerfile" | grep -q "^FROM" || continue

  tag="tb/$task:v0"
  echo "  Building $tag..."
  docker build -t "$tag" "$task_dir" -q >/dev/null
  IMAGES+=("$tag")
done
echo "Built ${#IMAGES[@]} images."

# Load into kind if cluster exists
if kind get clusters 2>/dev/null | grep -q "^${KIND_CLUSTER}$"; then
  echo "Loading into kind cluster '$KIND_CLUSTER'..."
  kind load docker-image "${IMAGES[@]}" --name "$KIND_CLUSTER"
  echo "Done."
else
  echo "Kind cluster '$KIND_CLUSTER' not found. Images built locally only."
  echo "Load manually: kind load docker-image ${IMAGES[*]} --name <cluster>"
fi
