# agentm-channels

Multi-channel chat gateway for AgentM. Wraps Feishu / Slack / Discord /
… behind a single message bus so the same agent can serve any chat
surface, and adding a new channel is one file.

Architecture (mirrors HKUDS/nanobot's `nanobot.bus` + `nanobot.channels`):

```
       ┌────────────────────────────┐                       ┌────────────────────────┐
       │       BaseChannel(s)       │                       │       Gateway          │
       │                            │   inbound queue       │                        │
   ↑ Feishu  ─┐                     │ ─────────────────►    │ session_key →          │
   ↑ Slack   ─┼─ register via name  │   MessageBus          │  AgentSession.prompt   │
   ↑ Stub    ─┘   (pkgutil scan)    │ ◄─────────────────    │  EventBus → outbound   │
       │                            │   outbound queue      │                        │
       └────────────────────────────┘                       └────────────────────────┘
                  ▲                                                     │
                  │  ChannelManager.send (with retry)                   │
                  └─────────────────────────────────────────────────────┘
```

## Layout

```
src/agentm_channels/
├── bus.py               # MessageBus, InboundMessage, OutboundMessage
├── base.py              # BaseChannel(ABC) — single contract every channel implements
├── registry.py          # auto-discover via pkgutil + entry_points
├── manager.py           # ChannelManager — lifecycle + outbound dispatch
├── gateway.py           # MessageBus ↔ AgentSession bridge
├── approval.py          # channel-agnostic ApprovalBridge (typed Button + button_value round-trip)
├── chat_session_map.py  # session_key → session_id persistence
├── cli.py               # `agentm-gateway` console entry
└── channels/
    ├── stub.py          # in-memory channel for tests
    └── feishu.py        # FeishuChannel (lark_oapi.channel adapter)
```

## Run (one-liner)

```bash
LARK_APP_ID=cli_xxxx \
LARK_APP_SECRET=xxxx \
uv run agentm-gateway --cwd ./workspace --scenario feishu_chat
```

`.env` in the cwd or AgentM workspace root is auto-loaded
(`AGENTM_PROVIDER` / `AGENTM_MODEL` / `OPENAI_API_KEY` etc).

## Run (config file)

```yaml
# gateway.yaml
cwd: ./workspace
scenario: feishu_chat
provider: openai
model: Doubao-Seed-2.0-pro

approval:
  require_approval: [bash]
  always_allow: [read, grep, find, ls]
  timeout_seconds: 300

channels:
  feishu:
    enabled: true
    app_id: ${LARK_APP_ID}
    app_secret: ${LARK_APP_SECRET}
    allow_from: ['*']        # or: [ou_xxxx, ou_yyyy]
```

```bash
uv run agentm-gateway --config gateway.yaml
```

## Adding a channel (e.g. Slack)

1. Create `src/agentm_channels/channels/slack.py`:

   ```python
   from ..base import BaseChannel
   from ..bus import OutboundMessage

   class SlackChannel(BaseChannel):
       name = "slack"
       async def start(self): ...     # subscribe to events
       async def stop(self): ...
       async def send(self, msg: OutboundMessage): ...
   ```

2. Add a `channels.slack` section to your gateway YAML with
   `enabled: true`. The registry picks it up automatically.

External plugins use `pyproject.toml` entry points:

```toml
[project.entry-points."agentm_channels.channels"]
discord = "my_pkg.discord:DiscordChannel"
```

## Approval flow

`ToolCallEvent` for any tool listed in `approval.require_approval` is
intercepted. The bridge publishes an `OutboundMessage` carrying two
typed `Button`s (`Approve` / `Deny`, with `style` set so each channel
can render the visual hint without label-string heuristics). Each
channel renders the buttons in its native UI (Feishu schema-2.0
interactive card with markdown body, Slack action block, …). When the
user clicks, the channel publishes an `InboundMessage` whose
`button_value` field carries the typed button payload — the encoding
of approval id + decision is private to the bridge. The gateway hands
every inbound with a `button_value` to
`ApprovalBridge.try_resolve_inbound`; the bridge resolves the pending
Future when the click matches a live request and the agent's loop
unblocks.

Identity is checked: only the user who triggered the tool call may
approve their own call.

## What lands in the chat

- **Assistant text** (every `TurnEndEvent`) — rendered as a markdown
  card so Chinese, code fences, lists, and inline emphasis all show up.
- **Diagnostics** (`warning` / `error`) — surfaced as plain markdown
  cards so the user sees infrastructure problems instead of a silent
  hang.
- **Tool calls / tool results** — *not* echoed to chat. Tool I/O is
  noisy and exposes raw payloads; the agent's next assistant turn
  summarises whatever the user actually needs to know. Operators can
  always inspect the trajectory JSONL for the full picture.

## Tests

```bash
uv run --package agentm-channels pytest tests/
```

The stub channel + fake-session E2E proves the full flow without
lark-oapi or a real LLM. The Feishu channel itself is exercised
end-to-end against a real bot — the wire layer (connect, message
receive, text send, card update) was validated during initial
development; later refactors are covered by the type checker plus
the `lark_oapi` SDK shape.
