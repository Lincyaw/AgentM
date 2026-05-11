# Design: Gateway / Client Process Split (channels v1)

**Status**: PROPOSED
**Created**: 2026-05-11
**Supersedes**: `gateway-channels.md` (which now describes v0 — the
in-process design shipped in PR #137).

This is a **design-only** document. Implementation lands in subsequent
PRs guided by `plans/2026-05-11-gateway-client-server.md`. Open
questions that need user sign-off are tagged `[OPEN]`.

---

## 1. Why split

The v0 design ships gateway and every channel (Feishu, terminal, stub)
as one Python process. That has worked, but it has structural
problems we've now hit or are about to hit:

1. **Crash isolation.** A bug in `lark_oapi` WS handling — connection
   reset, threadpool exhaustion, unhandled exception during a card
   render — currently takes down `Gateway` + every active session.
2. **Hot restart.** Restarting "the Feishu adapter" today means
   restarting the entire daemon. Every chat session is reset, every
   pending approval future is dropped.
3. **Auth surface.** The gateway daemon needs Lark `app_secret` only
   because Feishu lives in the same process. A pure operator who
   never wants to expose secrets to the gateway has no way to slot in
   a Feishu adapter she trusts.
4. **Testability.** Channel implementations and the gateway can be
   unit-tested separately, but **integration** testing today requires
   spinning up the whole stack. With a process boundary, an
   integration test client is one socket and a JSON dialect away.
5. **Multi-language clients.** Today every channel is Python because
   it has to be importable into the gateway process. Once the wire is
   a serialized protocol over IPC, the next Slack adapter can be a
   Go binary, the next WeChat adapter a Node script. The gateway
   doesn't care.

The pivot: **the gateway becomes a server**, channels become
**clients**, and the existing in-process `Bus` / `BaseChannel`
abstraction lives on as the *server-side* internal handler that
demultiplexes connected clients.

---

## 2. First principles

1. **Wire protocol is the contract.** Same status as the §11 atom
   contract or the `ExtensionAPI` Protocol. Versioned. Schema-locked.
   Breaking changes are semver-major.
2. **One transport family, multiple bind targets.** JSON over a
   length-framed stream socket. Unix socket by default for local
   deployments, TCP optional behind a feature flag for remote
   deployments. No HTTP, no gRPC in v1 — both pull in framework
   dependencies that the kernel does not need to take on.
3. **Clients are dumb adapters.** A client speaks one chat platform
   ("speak lark_oapi to/from Feishu") and the gateway protocol. It
   owns no session state, no approval state, no command routing. If
   it crashes mid-conversation, the gateway carries on; the client
   reconnects and resumes.
4. **No silent reach-arounds.** The gateway never opens a network
   connection to a chat platform. Every byte from the world reaches
   the gateway through a client process speaking the wire. This is
   the same boundary as §11 — kernel never touches scenario code
   directly.
5. **Backwards compatibility is finite.** v0 in-process channels stay
   functional during the migration window but are tagged deprecated
   and removed after every shipping channel has a v1 client. No
   permanent dual-stack.

---

## 3. Topology

```
                    ┌──────────────────────────────┐
                    │  agentm-gateway (daemon)     │
                    │                              │
                    │   ChatSessionMap             │
                    │   ApprovalBridge             │
                    │   CommandRouter              │
                    │   AgentSession × N           │
                    │   ClientRegistry             │
                    │   ┌─────────────┐            │
                    │   │ SocketServer│ ◄─── unix:///var/run/agentm.sock
                    │   └─────────────┘            │
                    └──────────────────────────────┘
                               ▲   ▲   ▲
                          IPC  │   │   │  IPC
              ┌────────────────┘   │   └─────────────────┐
              │                    │                     │
   ┌─────────────────────┐   ┌─────────────┐  ┌────────────────────────┐
   │ agentm-feishu (proc)│   │ agentm-     │  │ agentm-terminal (proc) │
   │                     │   │ http (proc) │  │  stdin / stdout        │
   │  lark_oapi WS       │   │ webhooks    │  │                        │
   │  → InboundMessage   │   │             │  │                        │
   │  ← OutboundMessage  │   │             │  │                        │
   │     → card render   │   │             │  │                        │
   └─────────────────────┘   └─────────────┘  └────────────────────────┘
            │
   Feishu open platform
```

The gateway runs on one host as a single process. Every client runs
as its own process. Communication is one bidirectional stream socket
per client connection.

---

## 4. Wire protocol v0

### 4.1 Framing

Length-prefixed JSON. Each message:

```
4-byte big-endian length     N
N bytes UTF-8 JSON body
```

Why not line-delimited (`\n`-terminated) JSON like the in-process
JSON terminal mode? Two reasons:

* Embedded newlines in message content (markdown, code blocks) would
  need escaping; explicit length avoids the question entirely.
* Length-prefixed framing trivially supports binary payloads when v2
  needs them (media uploads, audio); line-delimited would mean
  reframing later.

### 4.2 Message envelope

Every JSON body wraps:

```json
{
  "v": "v0",
  "id": "client-msg-12",
  "kind": "<kind>",
  "ts": "2026-05-11T14:32:18Z",
  "body": { ... }
}
```

* `v` — protocol version. Servers reject unknown majors.
* `id` — sender-assigned, opaque. Used to correlate replies/acks.
* `kind` — discriminator (one of the kinds in §4.3).
* `ts` — ISO 8601 UTC, advisory; servers use their own clock for
  ordering when authoritative.
* `body` — kind-specific payload.

### 4.3 Kinds

**Client → Server**

| kind | Direction | Purpose |
|---|---|---|
| `hello` | C→S, first message | Client identifies itself: channel name, version, capabilities, optional auth token. Server replies `welcome` or `error`. |
| `inbound` | C→S | Forwarded user message: sender_id, chat_id, content, media, metadata, session_key_override, button_value. Mirrors current `InboundMessage`. |
| `presence` | C→S | (Optional) "user is typing", "user opened chat". Advisory; server may ignore. |
| `ack` | C→S | Confirms receipt of a server-originated message id. Servers use this for delivery guarantees in §4.5. |
| `bye` | C→S, last | Graceful shutdown. Server flushes pending outbound for this client, then closes. |

**Server → Client**

| kind | Direction | Purpose |
|---|---|---|
| `welcome` | S→C, reply to `hello` | Server accepted the client; carries server version, negotiated protocol version, gateway session id. |
| `outbound` | S→C | Render-and-send-this-to-the-user. Mirrors current `OutboundMessage` (content, buttons, kind=message|turn_complete, metadata). |
| `error` | S→C | Protocol-level error (bad hello, auth failure, malformed inbound). Carries a code + message + whether the connection survives. |
| `ping` | S→C | Liveness check. Client replies `pong`. Drift threshold for stale-client cleanup. |

`pong` (C→S) is the symmetric reply to `ping`.

### 4.4 hello / welcome handshake

```json
// C→S
{
  "v":"v0", "id":"h1", "kind":"hello", "ts":"...",
  "body":{
    "client_name":"feishu",
    "client_version":"0.1.0",
    "wire_versions":["v0"],
    "auth": {"method":"token","token":"..."} ,
    "capabilities":["streaming","buttons","markdown"]
  }
}

// S→C
{
  "v":"v0", "id":"w1", "kind":"welcome", "ts":"...", "in_reply_to":"h1",
  "body":{
    "server_version":"0.2.0",
    "wire_version":"v0",
    "session_resume": ["feishu:c123","feishu:c456"]
  }
}
```

`session_resume` lets a reconnecting client know which routes the
server still believes belong to it — important for crash recovery
(§7).

### 4.5 Delivery semantics

* Server → Client `outbound` is **at-least-once**. The server retains
  unacked outbound for a small bounded window (default 100 messages
  per client) and replays them after reconnect.
* Client → Server `inbound` is **at-most-once** within a session — a
  duplicate `inbound.id` from the same client is dropped after
  logging. Clients responsible for not re-sending after a confirmed
  ack.
* Out-of-order tolerated: the server reorders nothing; each `outbound`
  carries enough metadata (chat_id, turn marker) for the client to
  render in receipt order.

### 4.6 Why JSON, not protobuf / msgpack / cap'n proto

* JSON is debuggable with `tcpdump | jq`. For a chat substrate this
  matters more than throughput.
* No code generation; clients in any language can hand-write a
  client without a toolchain dependency.
* The current `OutboundMessage`/`Button`/etc dataclasses already
  serialize naturally to JSON. We're not inventing schemas.

Trade-off: ~5–10× the bandwidth of msgpack. For chat-rate traffic
(messages per second, not thousand-per-second) this is irrelevant.

---

## 5. Transport

### 5.1 Default: Unix socket

* Path: `/var/run/agentm/gateway.sock` (configurable).
* Permission: 0660 owned by `agentm:agentm` (or whatever group the
  operator chooses).
* Trust: peer-credential auth on Linux/macOS — the kernel tells the
  server the connecting client's uid/gid. If it matches the
  configured `peer_uid`/`peer_gid`, no token needed.

### 5.2 Optional: TCP

* Behind `--bind tcp://host:port`. Off by default.
* Token auth mandatory in TCP mode. Plain `Bearer` in the `hello`
  body; the user supplies the token via env var to both gateway
  (allow-list) and client (presented).
* No TLS termination in the gateway — operator is expected to put
  this behind nginx / stunnel / WireGuard. This is the standard
  Unix-tool decomposition; SMTP/Postfix is the reference model. If
  TLS-in-gateway becomes a real requirement we'll add it explicitly.

### 5.3 Why not HTTP/WebSocket?

* Adds a framework dependency the kernel does not currently take.
* Browser-driven clients are not on the v1 roadmap (a v2 web client
  would need WebSocket; we can add a WebSocket-on-the-same-socket
  upgrade in v2 without breaking v1).
* Length-prefixed JSON is one screen of code per language.

### 5.4 Why not gRPC

* Pulls in protobuf and a runtime per language.
* The whole point of the v1 wire is human-debuggable. gRPC works
  against that.
* If someone wants to write a gRPC adapter on top of our v0 wire,
  they can — but the gateway speaks JSON.

---

## 6. Auth and identity

### 6.1 Client identity (who is connecting)

Three options, all supported:

1. **Unix peer-cred** (default for `unix://`). Kernel provides the
   client's uid; matches against `peer_uid` allow-list.
2. **Token**. Bearer in `hello.auth.token`. Required for TCP.
   Optional for Unix when peer-cred is in use (defense in depth).
3. **mTLS via stunnel**. Operator concern, gateway doesn't see it.

### 6.2 Channel identity (what platform are we)

`hello.body.client_name` is the channel name (`feishu`, `slack`,
`terminal`, …). Combined with the gateway's `clients.<name>` config:

```yaml
clients:
  feishu:
    auth: { method: peer_uid, allow: [1000] }
    can_send_as: ["feishu"]      # channel namespaces this client may
                                  # set on InboundMessage.channel
  terminal:
    auth: { method: token, allow: ["${TERM_TOKEN}"] }
    can_send_as: ["terminal"]
```

A client connecting and trying to send `inbound.channel = "slack"`
when its config only allows `feishu` is rejected with `error`
(`code = forbidden_channel`).

### 6.3 User identity (who sent the message)

Unchanged from v0: `inbound.sender_id` is the platform user id. The
client is responsible for canonicalising. The gateway never
authenticates the *user*, only the *client*.

`[OPEN]` Do we want per-user ACL at the gateway? Today `allow_from`
is per-channel. v1 keeps that — but in the future we may want
"these sender_ids may use atom commands" etc. Defer to a follow-up.

---

## 7. Crash, reconnect, multi-client semantics

### 7.1 Gateway crash

Sessions are persisted by `ChatSessionMap` (already exists). On
restart, the gateway has no live clients; any client connecting after
gets a `welcome.session_resume` listing the routes the gateway still
remembers. The client decides whether to "claim" them by resending
pending outbound state (typing indicators etc.).

### 7.2 Client crash

The gateway notices the socket close. Per-route state stays alive
(session, pending approval futures), but **outbound destined for a
disconnected client is queued up to a bounded ring** (`outbound_buffer
= 100` per route). When the client reconnects with the same
`client_name` + identity, the gateway replays the queue.

If two clients try to register the same `client_name` simultaneously,
the second is rejected (`error code = duplicate_client`). This is
deliberate: chat platforms have one canonical adapter at a time. If
the operator wants HA, they front the gateway with a leader-elect
sidecar.

### 7.3 Multi-client to the same chat

`[OPEN]` This is the hardest semantic question. Two reasonable
behaviours:

**(A) Single-writer per chat (recommended)**: at any moment, exactly
one client process is the adapter for a given `(channel, chat_id)`.
Trivial routing: outbound goes to whichever client connected with
`client_name == route.channel`. This is what we get for free with
"one client per channel name".

**(B) Pool of adapters**: multiple clients may register the same
`client_name` and the gateway round-robins or sticky-routes among
them. Enables horizontal scaling but explodes the consistency story.

Recommendation: **ship (A)**. If/when (B) becomes a real requirement,
we add a sticky-route field to `hello` ("I am `feishu` adapter
instance `aZ`, route shards 0–127 to me"). Not v1.

### 7.4 Approval round-trip across the wire

A `ToolCallEvent` causes the gateway's `ApprovalBridge` to publish
one `outbound` to the chat's client with `kind = approval_request`
and two typed `Button`s. The button-value encoding is unchanged from
v0 (`<approval_id>:<approve|deny>`).

The user clicks. The client (Feishu adapter) receives the card
callback, sends an `inbound` with `button_value = "<...>:approve"`.
The gateway's existing `Gateway._dispatch` button-value branch picks
it up and resolves the approval future.

Identity check still applies: the bridge stores `sender_id` of the
requester at request time; a click from a different `sender_id` is
ignored.

**Network-partition concern.** If the client connection drops between
sending the approval card and receiving the click, the click is
delayed until reconnect+replay. Pending approval future times out as
usual; user sees "approval timed out" the same as if they sat on
their hands. The gateway never sees the click → it's a deny. This is
acceptable failure semantics.

---

## 8. Migration plan (channels v0 → v1)

Four phases, each shippable independently.

### Phase 1 — Protocol + server, in-tree client lib

* Implement `agentm_channels.wire` (framing + envelope + kinds).
* Implement `SocketServer` inside the existing gateway daemon.
* Implement `agentm_channels.client` Python lib (used by tests and
  by Phase 2 clients).
* `BaseChannel` stays. The gateway can run *either* in-process
  channels (v0) *or* a `SocketServer` (v1) selected by config.
* Wire test: in-tree client lib drives the running gateway through a
  Unix socket; existing 61 tests pass + new wire tests.

### Phase 2 — Terminal extracted to `agentm-terminal`

* New process: `agentm-terminal`. Wraps the wire client lib + the
  stdin/stdout JSON dialect already in PR #137. Same UX (`--format
  json|text`).
* Gateway no longer ships a `TerminalChannel` in-process — `--terminal`
  flag becomes "start `agentm-terminal` as a subprocess and connect
  to the gateway". Operators who used the old `--terminal` get a
  deprecation warning and a translation note.
* `[OPEN]` Should `--terminal` keep working as a single-process
  convenience (spawn both as a subprocess pair from one command)?
  Recommend yes; it's the smoke-test path.

### Phase 3 — Feishu extracted to `agentm-feishu`

* New process: `agentm-feishu`. Holds the `lark_oapi` dep. Reads
  `app_id`/`app_secret` from its own config. Connects to the
  gateway over Unix socket. The existing FeishuChannel code moves
  here almost verbatim; the only behavioural change is that
  outbound now arrives as `outbound` over a socket instead of as
  `OutboundMessage` from a Queue.
* Gateway pyproject no longer requires `lark_oapi` (it moves to
  `agentm-feishu`'s pyproject). Smaller deps for everyone not
  running Feishu.
* `gateway.yaml`:

  ```yaml
  clients:
    feishu:
      socket: unix:///var/run/agentm/gateway.sock
      app_id: ...
      app_secret: ...
  ```

  becomes the new shape (config moved out of the gateway YAML).

### Phase 4 — Deprecate in-process channels

* `BaseChannel` and the `channels/` directory get a "Use clients
  instead" deprecation note.
* StubChannel stays for now — it's the test fixture, in-process by
  necessity.
* `gateway.yaml` `channels:` key emits a deprecation warning on
  startup; supported until the next minor.

---

## 9. What stays out of v1

* **Web client.** A WebSocket upgrade on the same socket would be
  natural in v2.
* **Per-user ACL beyond `allow_from`.** Mentioned in §6.3.
* **HA / multi-replica gateway.** §7.3 (B).
* **TLS in the gateway.** §5.2.
* **Streaming token deltas.** Already deferred at v0; no new
  obstacle here, but no implementation either.
* **Audio / image inline payloads.** Wire framing supports binary in
  v2 (length-prefixed makes it cheap); kinds + schema deferred.

---

## 10. Open questions for review

Each marked `[OPEN]` above. Summarised here for the design PR
checkout:

1. **Multi-client to same chat (§7.3).** Recommend (A) single-writer
   per channel name. Confirm or argue for (B).
2. **`--terminal` convenience wrapper (Phase 2).** Recommend keeping
   it as a "spawn pair" convenience. Confirm or remove.
3. **Per-user ACL (§6.3).** Defer to a follow-up — confirm OK.
4. **Token rotation / cred refresh** for TCP mode. Today `hello`
   carries the static token. Live rotation needs a kind like
   `reauth`. Defer to v2 unless explicitly needed.

---

## 11. Cross-references

* `gateway-channels.md` — v0 design, in-process. Stays as-is during
  the migration window with a "superseded by" pointer at the top.
* `command-routing.md` — slash command layer. Lives **server-side**,
  unchanged. Clients never see commands; they just forward
  `inbound.content`. Server intercepts before LLM dispatch.
* `pluggable-architecture.md` — the IPC wire is at the same boundary
  layer as `BaseChannel` was. It does not replace any internal port
  in the SDK.
