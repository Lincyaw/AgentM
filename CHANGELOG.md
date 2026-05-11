# Changelog

## v0.2.0 (pending)

### Deprecated

- `agentm_channels.BaseChannel` — v0 in-process channel adapters. Subclassing
  it now emits `DeprecationWarning`; `StubChannel` (test fixture) and the
  bridge's synthetic `_WireChannel` are exempt. Run
  `agentm-gateway --bind unix:///path` + a separate client process
  (`agentm-terminal`, `agentm-feishu`) instead.
- `agentm_channels.ChannelManager` — when constructed with a non-stub
  v0 channel set, emits `DeprecationWarning`. Wire-bridge usage via
  `inject_channel` is exempt.
- `gateway.yaml`'s `channels:` block — `agentm-gateway` prints a structured
  warning on stderr at startup when this block is non-empty. Will be
  removed in the next minor.

### Added

- `_WireChannel` now serializes `OutboundMessage.buttons` and
  `OutboundMessage.metadata` on the wire; `WireBridge.handle_inbound`
  forwards `envelope.body["button_value"]` into `InboundMessage.button_value`.
  Closes the approval-button round-trip for wire-mode clients.

## Unreleased

- Changed Claude Code compatibility loading from the old flat atom modules
  `contrib.extensions.cc.cc_agents`, `contrib.extensions.cc.cc_commands`, and
  `contrib.extensions.cc.cc_plugins` to the explicit package mount
  `contrib.extensions.cc`. Existing scenario manifests should replace the
  three old dotted paths with one `--extension contrib.extensions.cc` or
  manifest entry for `contrib.extensions.cc`.
