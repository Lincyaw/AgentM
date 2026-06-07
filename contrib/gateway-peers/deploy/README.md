# Deploying the AgentM gateway + Feishu client with systemd

Deployment tooling to run the [single-process gateway](../../../.claude/designs/single-process-gateway.md)
and the Feishu (Lark) chat client as managed daemons: auto-restart on crash,
start on boot, and a one-command "pull latest + restart" with minimal
user-visible disruption.

Files in this directory:

| File | Purpose |
|---|---|
| `agentm-gateway.service` | systemd unit for the gateway daemon |
| `agentm-feishu.service`  | systemd unit for the Feishu client daemon |
| `update.sh`              | pull latest code + restart only what changed |
| `README.md`              | this file |

## Architecture in one breath

The **gateway** holds all chat sessions in memory and serves chat-client peers
over a unix socket. The **Feishu client** is a peer process that bridges Lark
events to that socket. They are deliberately separate processes (vendor-SDK
isolation), connected over `unix:///run/agentm/gw.sock`.

## One-time install (system units — primary form)

The primary form is **system units** under `/etc/systemd/system/`, run as a
dedicated unprivileged user. (The target box happens to be root with the repo
at `~/w/AgentM`; even there, prefer a dedicated service user. A `systemctl
--user` variant is described at the end.)

1. **Copy (or symlink) the units** into systemd's path:

   ```bash
   sudo cp contrib/gateway-peers/deploy/agentm-gateway.service /etc/systemd/system/
   sudo cp contrib/gateway-peers/deploy/agentm-feishu.service  /etc/systemd/system/
   ```

   (Symlinking instead — `sudo ln -s "$PWD/contrib/.../agentm-gateway.service"
   /etc/systemd/system/` — lets `git pull` ship unit changes, but you must
   still `daemon-reload` after a pull that touches them.)

2. **Edit the placeholders** in both copied units. Every machine-specific value
   is a clearly-marked placeholder:

   - `WorkingDirectory=` → your repo checkout, e.g. `/root/w/AgentM`.
   - `EnvironmentFile=` → the repo's `.env` path, e.g. `/root/w/AgentM/.env`
     (keep the leading `-` so a missing file doesn't fail the unit).
   - `User=` / `Group=` → the service account. **Both units must use the same
     user** (the gateway socket only accepts its own uid by default).
   - `Documentation=` paths (cosmetic) → your repo path.
   - If `uv` isn't on `/usr/local/bin:/usr/bin:/bin`, fix the `Environment=PATH=`
     line (e.g. add `/root/.local/bin` for a per-user uv install).

3. **Provide credentials via `.env`** at the `EnvironmentFile=` path. It must
   carry at least `LARK_APP_ID`, `LARK_APP_SECRET`, and your model credentials
   (`AGENTM_*` / provider keys). **Never inline secrets in the unit files.** A
   populated `.env` is git-ignored and must not be committed.

4. **Make sure the feishu peer is installed.** The units use
   `uv run --package agentm-feishu …`, which works after a plain `uv sync`
   (the `--package` selector resolves the workspace member). If you prefer the
   units to call plain `uv run agentm-feishu`, run `uv sync --all-packages`
   once (and use `update.sh --all-packages` thereafter).

5. **Reload, enable, start:**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now agentm-gateway agentm-feishu
   ```

   `enable` wires boot-start; `--now` starts them immediately. The gateway
   comes up first (the feishu unit `After=`/`Wants=` it), creates
   `/run/agentm/` via `RuntimeDirectory=`, and binds the socket.

6. **Verify:**

   ```bash
   systemctl status agentm-gateway agentm-feishu
   journalctl -u agentm-gateway -f
   ```

## Day-to-day: updating

From anywhere in the repo (the script finds the root itself):

```bash
contrib/gateway-peers/deploy/update.sh
```

It: verifies the repo root, `git pull --ff-only` (fails loudly rather than
auto-merging), `uv sync`, then `sudo systemctl restart agentm-gateway` **only**.
It echoes the before/after commit and which services it restarted.

- **Feishu code changed too?** `update.sh --with-feishu` also restarts the
  client.
- **Units use plain `uv run agentm-feishu`?** `update.sh --all-packages`.
- **Managing user units / no sudo?** `SUDO="" update.sh` (pair with
  `systemctl --user` — adjust the script's restart target if needed).

## Why the gateway restarts but feishu usually doesn't

Phase 1 gave the feishu client **auto-reconnect**: when its gateway link drops,
it re-dials with the **same `peer_name`**, and the gateway **replays that peer's
outbox** — so an in-flight reply that was being streamed when the gateway went
down is re-delivered after it comes back. The user perceives at most a short
pause, not a dropped conversation.

That's why:

- The feishu unit uses `Wants=`/`After=` the gateway, **not** `Requires=`. A
  `Requires` would cascade-restart feishu every time the gateway restarts,
  which is exactly what we're avoiding. `Wants` keeps them loosely coupled:
  start-order preference only, no restart coupling.
- `update.sh` restarts only the gateway by default. The feishu daemon keeps
  running, notices the socket drop, reconnects to the new gateway, and replays.
- You only restart feishu (`--with-feishu`) when the **feishu client's own
  code** changed and the new code must actually load.

## Why the socket path is pinned (not the XDG default)

`default_socket_url()` returns `$XDG_RUNTIME_DIR/agentm-gw.sock` when that env
var is set, else `/tmp/agentm-gw-<uid>.sock`. Under systemd there is no login
session, so `XDG_RUNTIME_DIR` is typically unset and the path silently falls
back to the uid-suffixed `/tmp` form — ambiguous, and easy for the two units to
disagree on. So both units **pin** `unix:///run/agentm/gw.sock`:

- `RuntimeDirectory=agentm` makes systemd create `/run/agentm/` on start (mode
  `0750`) and clean it up on stop, so the socket has a stable, tidy home.
- Both units set `Environment=AGENTM_SOCKET=unix:///run/agentm/gw.sock` and pass
  the same value on the command line (`--bind` / `--connect`), so there is one
  unambiguous path.

## Logs and exit codes

- Follow gateway logs: `journalctl -u agentm-gateway -f`
- Follow feishu logs: `journalctl -u agentm-feishu -f`

The feishu CLI uses these exit codes (relevant because `Restart=always` will
relaunch on any non-graceful exit — check the log to see which fired):

| Code | Meaning |
|---|---|
| 0 | clean exit |
| 1 | generic / unexpected error |
| 2 | usage error (bad flags/args) |
| 4 | authentication error (e.g. gateway rejected the reconnect handshake) |
| 6 | interrupted (SIGINT) |
| 7 | connect error (couldn't reach the gateway socket — check the path / that the gateway is up / that both units run as the same uid) |

A repeated exit 7 in the feishu log usually means a socket-path or uid mismatch
between the two units; exit 4 means the gateway is up but rejecting auth.

## `systemctl --user` variant

To run without root, install under `~/.config/systemd/user/` instead, then:

- **Drop** `User=`/`Group=` from both units (a user unit already runs as you).
- Replace the pinned `/run/agentm` socket with the per-user runtime dir using
  the `%t` specifier (expands to `$XDG_RUNTIME_DIR` for user units):
  - `RuntimeDirectory=agentm` → `%t/agentm`
  - `--bind` / `--connect` / `AGENTM_SOCKET` → `unix://%t/agentm/gw.sock`
- Then `systemctl --user daemon-reload` and
  `systemctl --user enable --now agentm-gateway agentm-feishu`.
- For the daemons to survive logout, enable lingering once:
  `sudo loginctl enable-linger <youruser>`.
- For updates, run `SUDO="" contrib/gateway-peers/deploy/update.sh` and change
  the restart line to `systemctl --user restart …` (or just restart by hand).

## Note on verification

The unit files were checked with `systemd-analyze verify` and `update.sh` with
`bash -n` at authoring time. If you modify them, re-run:

```bash
systemd-analyze verify /etc/systemd/system/agentm-gateway.service
systemd-analyze verify /etc/systemd/system/agentm-feishu.service
bash -n contrib/gateway-peers/deploy/update.sh
```
