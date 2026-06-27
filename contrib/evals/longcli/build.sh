#!/bin/bash
# Build (and optionally push) all longcli task environment images via Skaffold.
#
# Usage:
#   bash contrib/evals/longcli/build.sh             # build only
#   bash contrib/evals/longcli/build.sh --push      # build + push to registry
#   bash contrib/evals/longcli/build.sh -b longcli-cs61-fa24-hog --push  # single task
#
# Environment:
#   LONGCLI_DIR  Path to longcli-bench repo (default: ../longcli-bench)
#   REGISTRY     Image registry prefix (default: opspai)
#   TAG          Image tag (default: v0)
#
# The script renders skaffold.yaml.tmpl with envsubst, then runs skaffold build.
# All extra arguments are passed through to skaffold build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LONGCLI_DIR="${LONGCLI_DIR:-$HOME/AoyangSpace/longcli-bench}"
export REGISTRY="${REGISTRY:-opspai}"
export TAG="${TAG:-v0}"

if [ ! -d "$LONGCLI_DIR/tasks_long_cli" ]; then
  echo "Error: $LONGCLI_DIR/tasks_long_cli not found."
  echo "Set LONGCLI_DIR to your longcli-bench repo path."
  exit 1
fi

# Build base images if missing
for base_name in make-pytest c-env; do
  if ! docker image inspect "tb/$base_name:v0" >/dev/null 2>&1; then
    dockerfile="$LONGCLI_DIR/longcli_dockerImage/Dockerfile.${base_name}-base"
    [ ! -f "$dockerfile" ] && dockerfile="$LONGCLI_DIR/longcli_dockerImage/Dockerfile.${base_name//-/_}-base"
    if [ -f "$dockerfile" ]; then
      echo "Building base image tb/$base_name:v0 ..."
      docker build -f "$dockerfile" -t "tb/$base_name:v0" "$LONGCLI_DIR/longcli_dockerImage" -q >/dev/null
    fi
  fi
done

# Render template
RENDERED=$(mktemp)
# Escape Go template tags so envsubst doesn't eat them
envsubst '${LONGCLI_DIR} ${REGISTRY} ${TAG}' < "$SCRIPT_DIR/skaffold.yaml.tmpl" > "$RENDERED"

echo "LONGCLI_DIR=$LONGCLI_DIR"
echo "REGISTRY=$REGISTRY"
echo "TAG=$TAG"
echo ""

skaffold build -f "$RENDERED" "$@"
rm -f "$RENDERED"
