---
name: deployment-awareness
description: Understand how you yourself are deployed and operated — your systemd services, workspace, config and credentials, where your observability/trace data lives, how to read back your own behaviour, and how to spot operational problems. Also covers multi-bot configuration (adding/removing/modifying feishu bots via config.toml), per-channel workspace routing, and persona management. Use when the user reports that you misbehaved, asks "why did you do that", says the bot is down/slow/silent, asks about your own deployment/logs/how to update you, or wants to add a new bot, change a bot's config, set up workspace isolation, or manage bot personas.
---

# deployment-awareness

You are a running deployment, not just a chat model. When something about
your own behaviour or availability is in question, inspect the deployment
before guessing. This skill is the map; `trace-analysis` and `self-debug`
are the detailed tools it points to.

## What you are

You run inside an **`agentm gateway`** process — a single-process host that
holds chat sessions in memory. The Feishu/Lark client is a *separate*
process that connects to the gateway over WebSocket and relays messages.
Both are systemd **user** units (not system units):

| Unit | Process | Role |
|------|---------|------|
| `agentm-gateway.service` | `agentm gateway --bind ws://0.0.0.0:8765 --cwd <workspace>` | hosts your sessions (this is where *you* run) |
| `agentm-feishu.service` | `agentm-feishu --connect ws://127.0.0.1:8765` | the Feishu/Lark chat client peer (may host multiple bots) |

Both are `Restart=always` and managed with `systemctl --user`. The feishu
unit is loosely coupled (`Wants=`, not `Requires=`) so it reconnects on its
own when the gateway restarts.

## Where everything lives

| Thing | Path |
|-------|------|
| Your workspace (your file/shell tools operate here) | the gateway's `--cwd` |
| Your character files | `<workspace>/persona.md`, `SOUL.md`, `IDENTITY.md`, `USER.md` |
| Model / provider / key | `~/.agentm/config.toml` |
| Feishu bot configs | `~/.agentm/config.toml` `[feishu.bots.*]` section |
| Gateway workspace routing | `~/.agentm/config.toml` `[gateway]` section |
| Per-bot workspaces | `~/.agentm/workspaces/{channel_name}/` (auto-created by gateway) |
| Your observability trace | `<workspace>/.agentm/observability/<session_id>.jsonl` (one file per session) |
| Example config | `config.toml.example` at repo root |

Character / persona files can be read and edited with your own tools; edits
reload into your prompt on your next reply (no restart). `config.toml` is
read at process start — changing it needs a restart (below).

## Multi-bot configuration

A single `agentm-feishu` process can host multiple Feishu bots. Each bot
is a separate Feishu app (its own `app_id`) connecting to the gateway as an
independent peer with its own `channel_name` and `scenario`.

### Adding a new bot

1. Create a Feishu app in the Feishu Open Platform (get `app_id` + `app_secret`)
2. Add a `[feishu.bots.<name>]` section to `~/.agentm/config.toml`:

```toml
[feishu.bots.newbot]
app_id = "cli_zzzzzzzzzzzz"
app_secret_file = "/run/secrets/newbot"   # or app_secret = "..." for dev
channel_name = "newbot"                    # defaults to the key name
scenario = "chatbot"                       # which scenario to run
# session_scope = "chat"                   # "chat" (default) or "user"
# allow_from = ["*"]                       # sender allowlist
```

3. Restart `agentm-feishu`: `systemctl --user restart agentm-feishu`

Secret precedence per bot: `app_secret_file` > `app_secret` > `LARK_APP_SECRET_{NAME_UPPER}` env.

### Modifying a bot

Edit the corresponding `[feishu.bots.<name>]` section in config.toml, then
restart `agentm-feishu`. The gateway does NOT need a restart — it sees the
new peer reconnect automatically.

### Removing a bot

Delete the `[feishu.bots.<name>]` section from config.toml, then restart
`agentm-feishu`. The gateway's sessions for that channel remain in memory
until the gateway restarts (harmless).

### Single-bot mode (backward compat)

If `--app-id` is passed on the CLI (or `LARK_APP_ID` is in env), the
`[feishu.bots]` section is ignored and the old single-bot path runs.

## Per-channel workspace routing

When `[gateway] workspace_root` is set, each channel gets its own isolated
workspace directory. Without it, all bots share the gateway's `--cwd`.

```toml
[gateway]
workspace_root = "~/.agentm/workspaces"
```

The gateway resolves: channel `"assistant"` → `~/.agentm/workspaces/assistant/`,
auto-creating the directory on first message. Each workspace has its own
session store, `.agentm/` state, and observability traces.

Explicit overrides for specific channels:

```toml
[gateway.workspaces]
researcher = "/mnt/data/rca-workspace"    # only when convention path won't do
```

### Persona management

Each bot's workspace can contain a `persona.md` that the scenario reads as
the bot's character/role description. The bot manages this file itself —
users tell the bot who it should be through conversation, and the bot
writes/updates `persona.md` in its own workspace. No manual file editing
needed; no config.toml entry. This is the same self-modification pattern as
Claude Code's `CLAUDE.md`.

### Listing active bots

```bash
# From config.toml:
grep -A3 '\[feishu\.bots\.' ~/.agentm/config.toml

# Running workspaces:
ls ~/.agentm/workspaces/
```

## Observe yourself

Every turn you take is recorded as an OTLP/JSON trace. **Never parse the
JSONL directly** — query it with the `agentm trace` CLI through `bash`
(`agentm` is on your PATH). `--latest` selects the most recent session,
which is you:

```bash
agentm trace messages --latest    # your conversation trajectory
agentm trace tools --latest       # every tool call + args + result
agentm trace turns --latest       # per-turn tokens + stop reason
agentm trace usage --latest       # token economics
agentm trace index                # all sessions, to drill into a past one
```

(There is no `tui_snapshot` here — that tool exists only for the terminal
client, not the Feishu deployment.) For the full atomic-command reference
and trace-tree composition idioms, load the **`trace-analysis`** skill; to
reason about debugging your own session step by step, load **`self-debug`**.

## Find and debug problems

Match the symptom to where the evidence is:

| Symptom | Where to look |
|---------|---------------|
| "Why did you do X?" / wrong behaviour | `agentm trace tools --latest` then `messages --latest` — the trace is the source of truth; the conversation is a lossy view |
| Bot silent / not replying / "down" | `systemctl --user status agentm-gateway agentm-feishu`, then `journalctl --user -u agentm-feishu -n 80 --no-pager` |
| Replies slow / a tool hangs | long tools auto-move to the background (~120s); check `agentm trace tools --latest` for a background ticket |
| Something broke right after an update | `journalctl --user -u agentm-gateway -n 120 --no-pager` for a crash/traceback at start |
| New bot not responding | check `--check-config` (`agentm-feishu --check-config`), then `journalctl --user -u agentm-feishu` for "skipped" warnings (missing secret / app_id) |
| Bot replies landing in wrong thread | verify `channel_name` is unique per bot in config.toml; check `workspace_root` is set if bots need isolated workspaces |

Work from evidence to hypothesis to fix — in that order.

## Lifecycle: restart, update, logs

```bash
systemctl --user status  agentm-gateway agentm-feishu      # health
systemctl --user restart agentm-gateway                    # restart yourself (drops in-memory sessions)
journalctl --user -u agentm-gateway -u agentm-feishu -f    # follow logs
```

Updating to newer code (from the source checkout):

```bash
cd <repo> && git pull --ff-only && uv sync --all-packages
# if the gateway/wire code changed, reinstall the units:
agentm gateway --uninstall-systemd
agentm gateway --cwd <workspace> --install-systemd
```

Restarting or reinstalling **terminates the running gateway**, ending the
current conversation — never do it casually mid-task. If the user asks you
to update yourself, confirm first, and expect the session to drop.

## Quick start: first-time multi-bot setup

For a user setting up from scratch with two bots:

```bash
# 1. Copy the example config and edit credentials
cp config.toml.example ~/.agentm/config.toml
vim ~/.agentm/config.toml   # fill in app_id, app_secret, model keys

# 2. Start the gateway (one process, workspace routing enabled)
agentm gateway --bind ws://0.0.0.0:8765

# 3. In another terminal, start the feishu peer (reads [feishu.bots] automatically)
agentm-feishu --connect ws://127.0.0.1:8765 --verbose

# 4. Validate config without connecting
agentm-feishu --check-config
```

After this, @-mention each bot in a Feishu group. The gateway auto-creates
`~/.agentm/workspaces/{channel_name}/` on the first message. Tell the bot
its role through conversation — it writes `persona.md` in its own workspace.
