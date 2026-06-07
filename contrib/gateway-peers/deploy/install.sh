#!/usr/bin/env bash
#
# install.sh — auto-install the AgentM gateway + Feishu client as systemd
# services. Detects the repo path, the running user, the `uv` binary, and the
# repo .env, renders both unit files with those real values, installs them, and
# enables + starts them. No manual placeholder editing.
#
# Usage:
#   ./install.sh                 # auto: system units if root, else user units
#   sudo ./install.sh            # force system units (/etc/systemd/system)
#   ./install.sh --user          # force user units (~/.config/systemd/user)
#   ./install.sh --system        # force system units
#   ./install.sh --run-as NAME   # (system mode) account the services run as
#   ./install.sh --no-start      # install + enable, but don't start now
#
# Re-running is safe: it overwrites the unit files and reloads systemd.

set -euo pipefail

# --- locate + verify the repo root -------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if ! grep -q '\[tool.uv.workspace\]' "${REPO_ROOT}/pyproject.toml" 2>/dev/null; then
  echo "ERROR: ${REPO_ROOT} is not the AgentM repo root (no [tool.uv.workspace])." >&2
  exit 1
fi

# --- parse flags --------------------------------------------------------------
MODE=""            # "system" | "user" | "" (auto)
RUN_AS=""          # system-mode account; default detected below
START=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)  MODE="system" ;;
    --user)    MODE="user" ;;
    --run-as)  RUN_AS="${2:?--run-as needs a username}"; shift ;;
    --no-start) START=0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- detect uv ----------------------------------------------------------------
UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" ]]; then
  echo "ERROR: 'uv' not found on PATH. Install uv first (https://docs.astral.sh/uv/)." >&2
  exit 1
fi
UV_DIR="$(dirname "${UV_BIN}")"
# Give the unit a PATH that contains uv plus the usual system dirs.
UNIT_PATH="${UV_DIR}:/usr/local/bin:/usr/bin:/bin"

# --- pick mode ----------------------------------------------------------------
if [[ -z "${MODE}" ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then MODE="system"; else MODE="user"; fi
fi

ENV_FILE="${REPO_ROOT}/.env"
GATEWAY="agentm-gateway.service"
FEISHU="agentm-feishu.service"

if [[ "${MODE}" == "system" ]]; then
  # system units run without a login session → pin an absolute socket under
  # /run/agentm (systemd creates it via RuntimeDirectory).
  SOCK="unix:///run/agentm/gw.sock"
  RUNTIME_LINE="RuntimeDirectory=agentm"
  UNIT_DIR="/etc/systemd/system"
  RUN_AS="${RUN_AS:-${SUDO_USER:-root}}"
  USER_LINES=$'User='"${RUN_AS}"$'\nGroup='"${RUN_AS}"
  SCTL=(systemctl)
  WANTED_BY="multi-user.target"
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: system mode writes to ${UNIT_DIR} — re-run with sudo." >&2
    exit 1
  fi
else
  # user units: %t expands to $XDG_RUNTIME_DIR (the per-user /run/user/<uid>).
  SOCK="unix://%t/agentm/gw.sock"
  RUNTIME_LINE="RuntimeDirectory=agentm"
  UNIT_DIR="${HOME}/.config/systemd/user"
  USER_LINES=""        # a user unit already runs as you
  SCTL=(systemctl --user)
  WANTED_BY="default.target"
  RUN_AS="$(whoami)"
fi

echo "==> Mode:          ${MODE}"
echo "==> Repo root:     ${REPO_ROOT}"
echo "==> Run as:        ${RUN_AS}"
echo "==> uv:            ${UV_BIN}"
echo "==> Socket:        ${SOCK}"
echo "==> Unit dir:      ${UNIT_DIR}"
echo "==> EnvironmentFile: ${ENV_FILE} $([[ -f ${ENV_FILE} ]] && echo '(found)' || echo '(MISSING — see note below)')"

mkdir -p "${UNIT_DIR}"

# --- render units -------------------------------------------------------------
cat > "${UNIT_DIR}/${GATEWAY}" <<EOF
[Unit]
Description=AgentM gateway (single-process chat session host)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
${USER_LINES}
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=-${ENV_FILE}
${RUNTIME_LINE}
RuntimeDirectoryMode=0750
Environment=AGENTM_SOCKET=${SOCK}
Environment=PATH=${UNIT_PATH}
ExecStart=${UV_BIN} run agentm gateway --bind ${SOCK}
Restart=always
RestartSec=2

[Install]
WantedBy=${WANTED_BY}
EOF

cat > "${UNIT_DIR}/${FEISHU}" <<EOF
[Unit]
Description=AgentM Feishu/Lark chat client
After=${GATEWAY}
Wants=${GATEWAY}

[Service]
Type=simple
${USER_LINES}
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=-${ENV_FILE}
Environment=PATH=${UNIT_PATH}
ExecStart=${UV_BIN} run --package agentm-feishu agentm-feishu --connect ${SOCK}
Restart=always
RestartSec=3

[Install]
WantedBy=${WANTED_BY}
EOF

echo "==> Wrote ${UNIT_DIR}/${GATEWAY}"
echo "==> Wrote ${UNIT_DIR}/${FEISHU}"

# --- reload + enable (+ start) ------------------------------------------------
"${SCTL[@]}" daemon-reload
if [[ "${START}" -eq 1 ]]; then
  "${SCTL[@]}" enable --now "${GATEWAY}" "${FEISHU}"
  echo "==> Enabled + started both services."
else
  "${SCTL[@]}" enable "${GATEWAY}" "${FEISHU}"
  echo "==> Enabled (not started; --no-start)."
fi

# --- post-install notes -------------------------------------------------------
echo
echo "Done."
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "NOTE: ${ENV_FILE} is missing. The Feishu client needs LARK_APP_ID /"
  echo "      LARK_APP_SECRET there (and model creds for the gateway). Run"
  echo "      'agentm onboard' or create ${ENV_FILE}, then:"
  echo "        ${SCTL[*]} restart ${FEISHU}"
fi
if [[ "${MODE}" == "user" ]]; then
  echo "NOTE: user units stop at logout unless lingering is on. Enable it with:"
  echo "        sudo loginctl enable-linger ${RUN_AS}"
fi
echo "Logs:   ${SCTL[*]} status ${GATEWAY}"
echo "        journalctl $([[ ${MODE} == user ]] && echo --user) -u agentm-gateway -f"
echo "Update: ${SCRIPT_DIR}/update.sh   (add --with-feishu when feishu code changed)"
