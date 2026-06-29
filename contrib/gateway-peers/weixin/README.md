# agentm-weixin

WeChat (微信) chat-client peer for the AgentM gateway, using the iLink Bot API.

## Quick start

```bash
# Login (QR scan):
agentm-weixin login

# Production-style local supervisor: starts gateway + adapter together.
agentm-weixin serve

# Or connect to an already managed gateway. For systemd installs, use the
# pinned user-runtime socket written by `agentm gateway --install-systemd`.
agentm-weixin run --connect unix://${XDG_RUNTIME_DIR}/agentm/gw.sock
```

Use `unix:///tmp/gw.sock` only for ad-hoc development when you also started the
gateway manually with the same path.
