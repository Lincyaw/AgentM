# agentm-feishu-channel

Feishu / Lark gateway channel for AgentM. A long-running daemon that
mediates between a Feishu chat (via `lark_oapi.channel.FeishuChannel`)
and one `agentm.harness.AgentSession` per chat / thread.

## What it is

This package is one of several **presenters** alongside the AgentM CLI
and Textual TUI. AgentM itself has no formal "channel" abstraction
today — this package introduces the shape under `contrib/channels/` so
future Slack / Discord adapters can settle into the same seam.

```
        Feishu chat                                 AgentM session
─────────────────────────────              ────────────────────────────
  user message ─┐                                ┌──── EventBus ────┐
                ▼                                ▼                  │
        ┌──────────────┐                ┌──────────────────┐        │
        │  ChatSource  │  ──────────►   │  FeishuGateway   │        │
        │  (lark_oapi) │     msgs       │  per-route       │        │
        └──────────────┘                │  AgentSession    │        │
                ▲                       └──────────────────┘        │
        cards / text │                              │                │
                     └──────  outbound   ◄──────────┘                │
                                                    └─ tool_call ────┤
                                                       approval card │
                                                                     ▼
                                                      ChatSource.send_card
```

## Install

```bash
pip install agentm-feishu-channel[feishu]   # pulls lark-oapi
# or, in the AgentM workspace root:
uv sync --package agentm-feishu-channel
```

The `[feishu]` extra is **optional**; the gateway core, stub chat
source, and tests work without `lark_oapi`.

## Run

Minimal:

```bash
export LARK_APP_ID=cli_xxxx
export LARK_APP_SECRET=xxxxxxxx
export ANTHROPIC_API_KEY=...
agentm-feishu-gateway --cwd ./workspace --scenario feishu_chat
```

With a config file (`gateway.yaml`):

```yaml
app_id: cli_xxxx
app_secret: xxxxxxxx
scenario: feishu_chat
cwd: ./workspace
state_dir: ./workspace/.agentm/feishu

approval:
  always_allow:   [read, grep, find, ls]
  always_block:   []
  require_approval:
    - bash         # every shell call asks the user (lark-cli writes,
                   # lark-cli reads, anything else)
    - lark         # if a future "lark" tool exists alongside bash
  timeout_seconds: 300
```

```bash
agentm-feishu-gateway --config gateway.yaml
```

## Feishu app setup

The gateway needs a Feishu / Lark app (custom app or store app) with:

1. **Long-connection event subscription** enabled (the SDK uses WSS;
   no public callback URL required).
2. The bot present in every chat the gateway should serve (DM the bot,
   or `@` it in a group; it must be a member to receive messages).
3. Permissions:
   - `im:message`            (read incoming messages)
   - `im:message:send_as_bot`(send replies as the bot)
   - `im:resource`           (download forwarded files / images, optional)
   - `cardkit:card:write`    (post and update interactive cards used by
                              the approval bridge)
4. **Card callback**: subscribe `card.action.trigger` so button clicks
   on approval cards flow back into the gateway.

Check `auth status` / `auth check` from `lark-cli` on the same host to
confirm scopes are granted before launching the gateway in a chat.

## Architecture

| File | Role |
|------|------|
| `chat_source.py` | Abstract transport seam — `ChatSource` Protocol + `StubChatSource` (in-memory, used by tests). |
| `feishu_source.py` | Production adapter from `lark_oapi.channel.FeishuChannel` to `ChatSource`. Imports `lark_oapi` lazily. |
| `gateway.py` | Owns chat→session map, EventBus → outbound rendering, message routing, lifecycle. |
| `approval.py` | `ToolCallEvent` async handler that posts approval cards and awaits the user's button click. |
| `cards.py` | Card body templates (approval card, resolved card, streaming assistant card). |
| `chat_session_map.py` | Disk-persisted `(chat_id, thread_id) → session_id` map so gateway restarts resume the same agent. |
| `cli.py` | `agentm-feishu-gateway` console script. |

## Why a separate package and not an atom

A gateway is an **external driver of `AgentSession`**, not an atom
installed inside one. Atoms run per-session; the gateway runs *across*
sessions, holding HTTP / WSS connections, mapping chat IDs to sessions,
and rendering EventBus events back to a chat surface. It belongs at the
same architectural level as `agentm.cli` and `agentm.modes.textual_app`,
which is why this lives at `contrib/channels/feishu/` rather than
`contrib/extensions/`.

## Why bash + approval bridge instead of a `lark` atom

Earlier iteration shipped a `contrib/extensions/lark_cli` atom that
gated reads vs writes inside the tool. That gate was illusory: an
agent with `bash` could call `lark-cli` directly and bypass it. The
gateway's approval bridge fixes the gate at the right layer — every
**bash invocation** surfaces as a card the human approves before it
runs. The `feishu-cli` skill (`contrib/skills/feishu-cli/SKILL.md`)
teaches the agent how to use `lark-cli` over `bash` safely; the gate
enforces it.

## Status

This package is intentionally **not** integrated into `agentm`'s
default scenarios. It opts in: install the workspace member, point a
Feishu app at it, and run the daemon. The non-Feishu surface (gateway
core, stub source, approval bridge, chat→session map) is unit-tested;
the Feishu adapter (`feishu_source.py`) is integration-tested only by
running against a real Feishu app.
