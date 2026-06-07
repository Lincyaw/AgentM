# Deploying the AgentM gateway + Feishu client with systemd

Run the [single-process gateway](../../../.claude/designs/single-process-gateway.md)
and the Feishu (Lark) chat client as managed daemons: auto-restart on crash,
start on boot, and a one-command "pull latest + restart" with minimal
user-visible disruption.

There is a **single** install mechanism — built into the `agentm` CLI. It
writes BOTH unit files (gateway + feishu), in system or user mode, with all
machine-specific values resolved for you. The old shell installer and the
hand-edited `.service` templates are gone; do not look for them.

Files in this directory:

| File | Purpose |
|---|---|
| `update.sh`  | pull latest code + restart only what changed |
| `README.md`  | this file |

## Install (recommended): `agentm onboard`

`agentm onboard` walks you through model creds, workspace, persona, and Feishu
credentials, then offers a final step:

```
== 5. systemd services (optional) ==
Set up systemd services now (auto-restart + start on boot)? [y/N]
```

Answering yes installs and starts both services, configured to run the gateway
in the workspace you just set up (its `--cwd` and `EnvironmentFile=<workspace>/.env`
are baked in). That is the whole deployment.

## Install directly

If you already have a configured workspace, install without re-running onboard:

```sh
# as root → system units in /etc/systemd/system, run as $SUDO_USER:
sudo agentm gateway --cwd /path/to/workspace --install-systemd

# or as a normal user → user units in ~/.config/systemd/user:
agentm gateway --cwd /path/to/workspace --install-systemd
#   then, so they survive logout / start on boot:
sudo loginctl enable-linger "$(whoami)"
```

What `--install-systemd` does:

- **Detects mode by EUID.** Root → system units (`/etc/systemd/system`,
  `User=`/`Group=$SUDO_USER`, `WantedBy=multi-user.target`). Non-root → user
  units (`~/.config/systemd/user`, no `User=`, `WantedBy=default.target`).
- **Writes both units.** `agentm-gateway.service` and `agentm-feishu.service`.
  If `agentm-feishu` can't be resolved (on `$PATH` or next to the `agentm`
  binary), it skips the feishu unit and prints a note — run
  `uv sync --all-packages`, then re-run.
- **Bakes the current invocation** into the gateway `ExecStart` (your
  `--cwd`, `--scenario`, `--model`, etc.), overriding `--bind` to the pinned
  socket so the two units cannot disagree.
- **Points `EnvironmentFile=` at `<workspace>/.env`** (the `--cwd` value, else
  the current dir) — where onboard writes `LARK_*` + model creds. The leading
  `-` makes a missing file non-fatal.
- **Sets a sane `PATH`** (the resolved bin's dir + `/usr/local/bin:/usr/bin:/bin`)
  so a sparse systemd environment still finds everything.
- **Reloads, enables, and starts** via `systemctl` (or `systemctl --user`).

Uninstall (mirrors the install mode):

```sh
sudo agentm gateway --uninstall-systemd     # system units
agentm gateway --uninstall-systemd          # user units
```

It stops, disables, and removes both units, then `daemon-reload`s.

## Architecture in one breath

The **gateway** holds all chat sessions in memory and serves chat-client peers
over a unix socket. The **Feishu client** is a peer process that bridges Lark
events to that socket. They are deliberately separate processes (vendor-SDK
isolation), connected over the pinned socket below.

## Day-to-day: updating

From anywhere in the repo (the script finds the root itself):

```bash
contrib/gateway-peers/deploy/update.sh
```

It verifies the repo root, `git pull --ff-only` (fails loudly rather than
auto-merging), `uv sync`, then `sudo systemctl restart agentm-gateway` **only**.
It echoes the before/after commit and which services it restarted.

- **Feishu code changed too?** `update.sh --with-feishu` also restarts the
  client.
- **Units use plain `uv run agentm-feishu`?** `update.sh --all-packages`.
- **Managing user units / no sudo?** `SUDO="" update.sh` (pair with
  `systemctl --user` — adjust the script's restart target if needed).

## Why the gateway restarts but feishu usually doesn't

The feishu client has **auto-reconnect**: when its gateway link drops, it
re-dials with the **same `peer_name`**, and the gateway **replays that peer's
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
disagree on. So `--install-systemd` **pins** the socket and writes the same
value into both units (gateway `--bind` == feishu `--connect`):

- **System units**: `unix:///run/agentm/gw.sock`, with `RuntimeDirectory=agentm`
  (mode `0750`) so systemd creates `/run/agentm/` on start and cleans it on stop.
- **User units**: `unix://%t/agentm/gw.sock`, where `%t` expands to the per-user
  runtime dir (`$XDG_RUNTIME_DIR`, i.e. `/run/user/<uid>`).

## Logs and exit codes

- Follow gateway logs: `journalctl -u agentm-gateway -f`
  (user units: `journalctl --user -u agentm-gateway -f`)
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
