# Deploying the AgentM gateway + Feishu client with systemd

Run the [single-process gateway](../../../.claude/designs/single-process-gateway.md)
and the Feishu (Lark) chat client as managed daemons: auto-restart on crash,
start on boot, and a one-command "pull latest + restart" with minimal
user-visible disruption.

There is a **single** install mechanism — built into the `agentm` CLI. It
writes BOTH user unit files (gateway + feishu) into `~/.config/systemd/user`,
with all machine-specific values resolved for you. The old shell installer and
the hand-edited `.service` templates are gone; do not look for them.

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
agentm gateway --cwd /path/to/workspace --install-systemd
#   then, so they survive logout / start on boot:
loginctl enable-linger "$(whoami)"
```

These are **user** units (`~/.config/systemd/user` + `systemctl --user`) — even
when run as root. No system units, nothing in `/etc`.

What `--install-systemd` does:

- **Writes both user units.** `agentm-gateway.service` and
  `agentm-feishu.service`. If `agentm-feishu` can't be resolved (on `$PATH` or
  next to the `agentm` binary), it skips the feishu unit and prints a note —
  run `uv sync --all-packages`, then re-run.
- **Bakes the current invocation** into the gateway `ExecStart` (your `--cwd`,
  `--scenario`, `--model`, etc.), overriding `--bind` to the pinned socket so
  the two units cannot disagree. The workspace is absolutized, so a relative
  `--cwd` is fine.
- **Points `EnvironmentFile=` at `<workspace>/.env`** (the `--cwd` value, else
  the current dir) — where onboard writes `LARK_*` + model creds. The leading
  `-` makes a missing file non-fatal.
- **Sets a sane `PATH`** (the resolved bin's dir + `/usr/local/bin:/usr/bin:/bin`)
  so a sparse systemd environment still finds everything.
- **Reloads, enables, and starts** via `systemctl --user`.

Uninstall:

```sh
agentm gateway --uninstall-systemd
```

It stops, disables, and removes both units, then `daemon-reload`s.

> **Upgrading from the old gateway-only unit?** Earlier installs used
> `EnvironmentFile=~/.config/agentm/gateway.env`; the workspace `.env` is now
> the single source. That old file is no longer referenced — you can delete it
> after moving any vars into `<workspace>/.env`.

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
auto-merging), `uv sync`, then `systemctl --user restart agentm-gateway`
**only**. It echoes the before/after commit and which services it restarted.

- **Feishu code changed too?** `update.sh --with-feishu` also restarts the
  client.
- **Units need the feishu workspace member?** `update.sh --all-packages`
  (runs `uv sync --all-packages` so `agentm-feishu` is installed).

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

## Why the socket path is pinned

The local default socket is scoped by `$AGENTM_RUNTIME_DIR` when set, otherwise
by a per-user temp directory whose name includes a hash of `$AGENTM_HOME`.
That is the right default for ad hoc local CLIs, but managed user units should
not depend on whichever environment systemd happened to provide. So
`--install-systemd` **pins** the socket and writes the same value into both
units (gateway `--bind` == feishu `--connect`):
`unix://%t/agentm/gw.sock`, where `%t` expands to the per-user runtime dir
(`$XDG_RUNTIME_DIR`, i.e. `/run/user/<uid>`).

## Logs and exit codes

- Follow gateway logs: `journalctl --user -u agentm-gateway -f`
- Follow feishu logs: `journalctl --user -u agentm-feishu -f`

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
