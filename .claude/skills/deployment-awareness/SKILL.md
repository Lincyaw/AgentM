---
name: deployment-awareness
description: Understand how you yourself are deployed and operated — your systemd services, workspace, config and credentials, where your observability/trace data lives, how to read back your own behaviour, and how to spot operational problems. Use when the user reports that you misbehaved, asks "why did you do that", says the bot is down/slow/silent, or asks about your own deployment, logs, or how to update you.
---

# deployment-awareness

You are a running deployment, not just a chat model. When something about
your own behaviour or availability is in question, inspect the deployment
before guessing. This skill is the map; `trace-analysis` and `self-debug`
are the detailed tools it points to.

## What you are

You run inside an **`agentm gateway`** process — a single-process host that
holds chat sessions in memory. The Feishu/Lark client is a *separate*
process that connects to the gateway over a Unix socket and relays
messages. Both are systemd **user** units (not system units):

| Unit | Process | Role |
|------|---------|------|
| `agentm-gateway.service` | `agentm gateway --bind unix://$XDG_RUNTIME_DIR/agentm/gw.sock --cwd <workspace>` | hosts your sessions (this is where *you* run) |
| `agentm-feishu.service` | `agentm-feishu --connect <same socket>` | the Feishu/Lark chat client peer |

Both are `Restart=always` and managed with `systemctl --user`. The feishu
unit is loosely coupled (`Wants=`, not `Requires=`) so it reconnects on its
own when the gateway restarts.

## Where everything lives

| Thing | Path |
|-------|------|
| Your workspace (your file/shell tools operate here) | the gateway's `--cwd` |
| Your character files | `<workspace>/SOUL.md`, `IDENTITY.md`, `USER.md` |
| Model / provider / key | `~/.agentm/config.toml` |
| Feishu credentials | `<workspace>/.env` (`LARK_APP_ID` / `LARK_APP_SECRET` / `LARK_ALLOW_FROM`) |
| Your observability trace | `<workspace>/.agentm/observability/<session_id>.jsonl` (one file per session) |

You can read and edit the character files with your own tools; edits reload
into your prompt on your next reply (no restart). `config.toml` and `.env`
are read at process start — changing them needs a restart (below).

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
