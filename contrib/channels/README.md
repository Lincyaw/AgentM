# agentm-channels

Multi-channel chat gateway for AgentM. Wraps Feishu / Slack / Discord /
… behind a single message bus so the same agent can serve any chat
surface, and adding a new channel is one file.

Architecture (two-queue message bus + channel registry):

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

## Local validation (terminal mode)

Two-process invocation is the only documented contract: the gateway
runs as a wire-protocol daemon, each chat platform runs as a separate
client process and talks to the gateway over a Unix socket.

```bash
# Terminal A:
uv run agentm-gateway --bind unix:///tmp/gw.sock \
    --cwd ./workspace --scenario general_purpose

# Terminal B:
uv run agentm-terminal --connect unix:///tmp/gw.sock
```

You get a stdin/stdout REPL backed by the same gateway + commands +
approvals as production. Useful for testing `/help`, `/skill:foo`,
`/new`, custom markdown commands, and approval flows before pointing
Feishu at the daemon. `Ctrl-D` on the terminal client ends the
session.

Approval-button round-trips in the terminal client: when the agent
posts an approval card you'll see numbered `[1] Approve` / `[2] Deny`
options and the literal `value=…` next to each. Type `=<value>` (e.g.
`=approval-deadbeef:approve`) to click. Less ergonomic than a Feishu
button, but exercises the exact same code path.

## Cross-host (WebSocket) quickstart

The gateway also speaks WebSocket so workers / clients can connect over
the network. Authentication is by bearer token — each client must
present a token from the gateway's allow-list.

```bash
# Pick / generate tokens, one per line. Blank lines and lines starting
# with '#' are ignored.
install -m 0600 /dev/stdin /etc/agentm/tokens <<'EOF'
# alice's worker
2c8f9d3a1e4b6f7c2a5d9e1f3b4c7a2d
# bob's terminal
8e1d2c4b6a9f3e5d7c1b4a6e9d2f5c8b
EOF

# Plaintext (terminate TLS in nginx or another reverse proxy):
agentm-gateway --bind ws://0.0.0.0:7777/agentm \
    --bind-token-file /etc/agentm/tokens

# Or wss directly:
agentm-gateway --bind wss://0.0.0.0:7777/agentm \
    --bind-token-file /etc/agentm/tokens \
    --tls-cert /etc/agentm/cert.pem --tls-key /etc/agentm/key.pem

# Clients pass the token via env (avoids leaking through `ps`):
AGENTM_TOKEN="$(sed -n '/^[^#]/p;q' /etc/agentm/tokens)" \
  agentm-worker --connect ws://gw.example.com:7777/agentm
```

Reverse-proxying through nginx is supported out of the box because
each WebSocket binary frame carries exactly one already-framed wire
envelope — the proxy only needs the standard `proxy_set_header
Upgrade`/`Connection` block and TLS termination. No example config is
shipped; the standard
[`websockets`](https://websockets.readthedocs.io/en/stable/topics/deployment.html)
recipe applies.

> **Deprecated:** the v0 in-process `channels:` yaml block still loads
> for backwards compatibility, but emits a `DeprecationWarning` (and
> an `agentm-gateway: warn: …` line on stderr at startup) and will be
> removed in a future minor. New deployments should always use
> `--bind unix:///path` + separate client processes.

## Run (one-liner)

```bash
LARK_APP_ID=cli_xxxx \
LARK_APP_SECRET=xxxx \
uv run agentm-gateway --cwd ./workspace --scenario general_purpose
```

`.env` in the cwd or AgentM workspace root is auto-loaded
(`AGENTM_PROVIDER` / `AGENTM_MODEL` / `OPENAI_API_KEY` etc).

## Run (config file)

```yaml
# gateway.yaml
cwd: ./workspace
scenario: general_purpose
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

## Slash commands

Messages starting with `/` are intercepted before the agent sees them.
Design lives in `.claude/designs/command-routing.md`.

Builtin control commands (no LLM call):

| Command | Effect |
|---|---|
| `/new` | Discard the AgentSession for this chat; next message starts fresh. |
| `/end` | Discard the session **and** forget the persistent mapping (cold restart on next message). |
| `/status` | Report session id, turn count, pending approvals. |
| `/help` | List every discovered command, grouped by source. |

Custom prompt commands (Claude-Code-style markdown) live in
`<cwd>/.agentm/commands/*.md`:

```markdown
---
name: standup
summary: Generate today's standup
---
Read the last 24h of commits and write a 3-bullet standup for
$ARGUMENTS.
```

`/standup polish` resolves `$ARGUMENTS` to `polish` and feeds the
expanded body to the agent as the user turn.

Skills as commands. Any SKILL.md under `<cwd>/.claude/skills/` or
`~/.claude/skills/` is also reachable as `/skill:<name> [args]`. The
skill body is injected as a one-turn instruction, so the user can
force-activate a skill even when the LLM has not selected it from the
auto-loaded skill index.

Unknown commands (`/foo` with no matching handler) are rejected with a
visible error — they **never** reach the LLM. This avoids the failure
mode of typos leaking sensitive prompts into the model when the user
clearly intended a command.

Atom commands (`/atom:install <name>`, `/atom:uninstall <name>`) are
designed but not implemented in this PR — see
`.claude/designs/command-routing.md` §8.

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
