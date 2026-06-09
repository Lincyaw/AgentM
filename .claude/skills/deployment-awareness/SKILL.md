---
name: deployment-awareness
description: AgentM deployment topology, configuration, and operations guide — deployment modes (standalone, gateway+feishu, multi-bot), config.toml structure, per-channel workspace routing, persona management, observability, and troubleshooting. Use when the user asks about deployment, adds/removes/modifies bots, sets up workspace isolation, debugs a silent/slow/misbehaving bot, or manages bot personas. Also trigger when diagnosing which deployment mode is active.
---

# deployment-awareness

Reference for deploying and operating AgentM. Covers the deployment modes,
how to detect which mode is running, configuration, multi-bot setup,
workspace isolation, and troubleshooting.

## Deployment modes

AgentM supports three deployment modes. Each is a superset of the previous:

| Mode | Processes | When to use |
|------|-----------|-------------|
| **Standalone** | `agentm -p "..."` or `agentm` (TUI) | local dev, one-shot prompts, terminal chat |
| **Gateway + client** | `agentm gateway` + `agentm-feishu` (or `agentm-terminal`) | single bot served over Feishu/Lark |
| **Multi-bot** | `agentm gateway` + `agentm-feishu` (N bots from config.toml) | multiple agents in one group chat |

### Detecting the current mode

```bash
# Is a gateway running?
systemctl --user status agentm-gateway 2>/dev/null || pgrep -af 'agentm gateway'

# Is a feishu peer running? How many bots?
systemctl --user status agentm-feishu 2>/dev/null || pgrep -af agentm-feishu
agentm-feishu --check-config 2>&1    # lists resolved bots

# Is workspace routing active?
grep 'workspace_root' ~/.agentm/config.toml 2>/dev/null

# What workspaces exist?
ls ~/.agentm/workspaces/ 2>/dev/null
```

## Architecture

```
┌─────────────────┐         WS          ┌──────────────────┐
│  agentm-feishu  │◄──────────────────►│  agentm gateway   │
│  (N bots, each  │   v2 wire protocol  │  (sessions,       │
│   a FeishuChannel│                    │   workspace routing│
│   + WireClient)  │                    │   per channel)     │
└─────────────────┘                     └──────────────────┘
       │                                        │
       │ Feishu WS                              │ session cwd
       ▼                                        ▼
  Feishu servers                         ~/.agentm/workspaces/{channel}/
```

- **Gateway** holds sessions in memory, routes outbound by `channel_name`.
- **Feishu peer** maps Feishu events ↔ wire envelopes. In multi-bot mode,
  each bot is an independent `FeishuChannel` + `WireClient` pair.
- **Workspace routing**: when `[gateway] workspace_root` is set, the gateway
  resolves each channel to `{root}/{channel}/`, auto-creating on first use.

When deployed as systemd user units:

| Unit | Command |
|------|---------|
| `agentm-gateway.service` | `agentm gateway --bind ws://0.0.0.0:8765` |
| `agentm-feishu.service` | `agentm-feishu --connect ws://127.0.0.1:8765` |

Both are `Restart=always`. The feishu unit is loosely coupled (`Wants=`,
not `Requires=`) — it reconnects on its own when the gateway restarts.

## Configuration

All long-lived config lives in `~/.agentm/config.toml`
(`$AGENTM_HOME/config.toml`). See `config.toml.example` at repo root for
the full annotated reference.

| Section | What it controls | Read by |
|---------|-----------------|---------|
| `default_model` / `[models.*]` | LLM provider, model, API key | gateway |
| `[gateway]` | `workspace_root` | gateway |
| `[gateway.workspaces]` | explicit per-channel workspace overrides | gateway |
| `[feishu.bots.*]` | per-bot app_id, secret, channel, scenario | agentm-feishu |

Config is read at process start — changes need a restart of the affected
process (`systemctl --user restart agentm-feishu` / `agentm-gateway`).

### Key paths

| Thing | Path |
|-------|------|
| Central config | `~/.agentm/config.toml` |
| Per-bot workspaces | `~/.agentm/workspaces/{channel_name}/` |
| Persona / character files | `<workspace>/persona.md` |
| Observability trace | `<workspace>/.agentm/observability/<session_id>.jsonl` |
| Example config | `config.toml.example` at repo root |

## Multi-bot configuration

A single `agentm-feishu` process can host N Feishu bots. Each bot is a
separate Feishu app with its own `channel_name` and `scenario`.

### Adding a bot

1. Create a Feishu app in the Open Platform (get `app_id` + `app_secret`)
2. Add a `[feishu.bots.<name>]` section to config.toml:

```toml
[feishu.bots.newbot]
app_id = "cli_zzzzzzzzzzzz"
app_secret_file = "/run/secrets/newbot"   # or app_secret = "..." for dev
channel_name = "newbot"                    # defaults to the key name
scenario = "chatbot"                       # which scenario to run
# session_scope = "chat"                   # "chat" (default) or "user"
# allow_from = ["*"]                       # sender allowlist
```

3. Restart: `systemctl --user restart agentm-feishu`

Secret precedence: `app_secret_file` > `app_secret` > `LARK_APP_SECRET_{NAME_UPPER}` env.

### Modifying a bot

Edit the `[feishu.bots.<name>]` section, restart `agentm-feishu`. The
gateway does not need a restart — it sees the new peer reconnect.

### Removing a bot

Delete the section, restart `agentm-feishu`.

### Single-bot backward compat

When `--app-id` is on the CLI or `LARK_APP_ID` is in env, the
`[feishu.bots]` section is ignored.

## Per-channel workspace routing

```toml
[gateway]
workspace_root = "~/.agentm/workspaces"
```

Channel `"assistant"` → `~/.agentm/workspaces/assistant/` (auto-created).
Each workspace has its own session store, `.agentm/` state, and traces.

Explicit overrides when the convention path doesn't fit:

```toml
[gateway.workspaces]
researcher = "/mnt/data/rca-workspace"
```

Without `workspace_root`, all channels share the gateway's `--cwd`.

### Persona management

Each workspace can contain a `persona.md` that the scenario reads as role
description. The bot manages this file through conversation — users tell
the bot who it should be, the bot writes `persona.md` in its workspace.

### Listing bots and workspaces

```bash
grep -A3 '\[feishu\.bots\.' ~/.agentm/config.toml
ls ~/.agentm/workspaces/
```

## Observability

Every turn is recorded as OTLP/JSON. Query with `agentm trace` (never
parse JSONL directly):

```bash
agentm trace messages --latest    # conversation trajectory
agentm trace tools --latest       # tool calls + args + result
agentm trace turns --latest       # per-turn tokens + stop reason
agentm trace usage --latest       # token economics
agentm trace index                # all sessions
```

For the full reference, load the `trace-analysis` skill; for step-by-step
session debugging, load `self-debug`.

## Troubleshooting

| Symptom | Where to look |
|---------|---------------|
| Wrong behaviour / "why did you do X?" | `agentm trace tools --latest` then `messages --latest` |
| Bot silent / down | `systemctl --user status agentm-gateway agentm-feishu`, then `journalctl --user -u agentm-feishu -n 80` |
| Replies slow / tool hangs | `agentm trace tools --latest` for background tickets |
| Broke after update | `journalctl --user -u agentm-gateway -n 120` for crash traceback |
| New bot not responding | `agentm-feishu --check-config`, then journal for "skipped" warnings |
| Replies in wrong chat/thread | verify `channel_name` is unique per bot; check `workspace_root` is set |

## Lifecycle

```bash
systemctl --user status  agentm-gateway agentm-feishu
systemctl --user restart agentm-gateway                    # drops in-memory sessions
systemctl --user restart agentm-feishu                     # reconnects, no session loss
journalctl --user -u agentm-gateway -u agentm-feishu -f
```

Updating from source:

```bash
cd <repo> && git pull --ff-only && uv sync --all-packages
agentm gateway --uninstall-systemd
agentm gateway --cwd <workspace> --install-systemd
```

Restarting the gateway terminates in-memory sessions. Confirm before doing
it mid-task.

## Quick start: multi-bot from scratch

```bash
# 1. Copy example and fill in credentials
cp config.toml.example ~/.agentm/config.toml
vim ~/.agentm/config.toml

# 2. Start the gateway
agentm gateway --bind ws://0.0.0.0:8765

# 3. Start the feishu peer (auto-reads [feishu.bots])
agentm-feishu --connect ws://127.0.0.1:8765 --verbose

# 4. Validate without connecting
agentm-feishu --check-config
```

@-mention each bot in Feishu. The gateway creates workspaces on first
message. Tell the bot its role through conversation.
