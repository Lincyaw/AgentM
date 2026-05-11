# Plan: Gateway-Client process split (channels v1)

Date: 2026-05-11
Branch (this design PR): `feature/gateway-client-server-design`
Designs: `designs/client-server-architecture.md` (new),
`designs/gateway-channels.md` (now annotated as v0).

This plan covers only the **design PR**. Each implementation phase
gets its own plan file once the design is locked, so review on each
phase stays scoped.

## Scope of the design PR

- New design doc: `client-server-architecture.md` (proposed v1).
- Annotation on `gateway-channels.md` (v0, kept during migration).
- Index entries: add `client_server_architecture`, mark
  `gateway_channels` `status: superseded_during_migration`.
- This plan file.

**No code changes.** No new tests. No new wire-protocol implementation
yet. The design PR is for getting alignment on the open questions
listed in `client-server-architecture.md` §10:

1. Multi-client to same chat — recommend single-writer per channel name.
2. `--terminal` convenience wrapper after Phase 2 — recommend keep.
3. Per-user ACL beyond `allow_from` — recommend defer.
4. TCP token rotation — recommend defer.

## Phases (implementation, in subsequent PRs)

Each phase is one PR. Order matters; each unlocks the next.

### Phase 1 — Wire protocol + server, in-tree client lib

- `contrib/channels/src/agentm_channels/wire/{__init__.py,framing.py,envelope.py,kinds.py}` — pure functions, no I/O. Length-frame encode/decode, envelope validation, kind enum.
- `contrib/channels/src/agentm_channels/server.py` — asyncio Unix socket server bound to `Gateway`. Accepts hello, registers a `ClientRegistry` entry, fans inbound onto the existing `MessageBus`, drains outbound to the connected client.
- `contrib/channels/src/agentm_channels/client.py` — Python client lib used by tests + Phase 2/3. Connects, handshakes, sends inbound, consumes outbound.
- New CLI flag `--bind unix:///path` on `agentm-gateway`. When set, runs the socket server alongside (or instead of) in-process channels.
- New tests:
  - Wire encode/decode round-trip (fuzz on random envelopes).
  - hello/welcome handshake.
  - Inbound→outbound echo through the socket using a stub session factory.
  - Client crash mid-conversation → reconnect → replay (bounded buffer).
  - Auth failure path (wrong token / wrong peer uid).
- The Phase 1 PR adds no removals — v0 channels keep working.

### Phase 2 — `agentm-terminal` extracted

- New package: `contrib/channels-clients/terminal/` (own pyproject).
- Move `channels/terminal.py` content into the new process, wrap with the wire client lib.
- `agentm-gateway --terminal` becomes a convenience that spawns `agentm-terminal` as a subprocess and connects it to a same-process socket server. Deprecation warning on the legacy in-process terminal channel; removed in Phase 4.
- Tests: subprocess-driven integration test mirroring `test_gateway_e2e.py` over the wire.

### Phase 3 — `agentm-feishu` extracted

- New package: `contrib/channels-clients/feishu/` (own pyproject; this is where `lark_oapi` moves).
- Move `channels/feishu.py` (including the WS-loop monkey-patch and ACK-emoji handling).
- `gateway.yaml` `channels.feishu` deprecated; new shape:

  ```yaml
  # gateway side
  bind: unix:///var/run/agentm/gateway.sock
  clients:
    feishu:
      auth: {method: peer_uid, allow: [1000]}

  # feishu-client side
  socket: unix:///var/run/agentm/gateway.sock
  app_id: ${LARK_APP_ID}
  app_secret: ${LARK_APP_SECRET}
  allow_from: ['*']
  ```

- The gateway pyproject drops `lark_oapi` as a runtime dep; only the feishu client carries it. Smaller image / smaller venv for operators not using Feishu.

### Phase 4 — Deprecate v0 in-process channels

- Mark `BaseChannel` / `ChannelManager` deprecated. Keep `StubChannel` (test fixture only).
- `gateway.yaml` `channels:` block emits deprecation warning on startup.
- Move `gateway-channels.md` to `designs/historical/`.
- Removal in the next minor after Phase 4 lands.

## Out of scope (forever, or v2+)

- WebSocket / HTTP transport (v2 if a browser client materialises).
- TLS in the gateway (operator concern — stunnel / nginx).
- gRPC / protobuf (defeats the debuggability premise).
- Multi-replica gateway / leader election (deferred until real HA pressure exists).
- Streaming token deltas (orthogonal — blocked on SDK AgentM emitting them).

## Acceptance for THIS design PR

1. The two design docs (new + annotated) plus this plan and index
   updates land cleanly.
2. The four `[OPEN]` decisions in `client-server-architecture.md` §10
   have explicit user approval / counter-proposals captured in the PR
   thread before any Phase 1 work begins.
