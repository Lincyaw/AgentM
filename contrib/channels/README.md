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
├── approval.py          # channel-agnostic ApprovalBridge (uses OutboundMessage.buttons)
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
intercepted. The bridge publishes an `OutboundMessage` with two buttons
(`Approve` / `Deny`); each channel renders these in its native UI
(Feishu interactive card, Slack action block, …). When the user
clicks, the channel publishes an `InboundMessage` whose
`metadata.button_value` carries `<approval_id>:approve|deny`. The
gateway routes it to the bridge instead of the agent — the bridge
resolves the pending Future and the agent's loop unblocks.

Identity is checked: only the user who triggered the tool call may
approve their own call.

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
