# agentm-feishu

Feishu / Lark client process for the AgentM channels gateway.

Connects to an `agentm-gateway --bind unix://…` over the v1 wire
protocol and bridges inbound Feishu messages / card-button clicks into
the gateway, then renders gateway outbounds as Feishu interactive
cards. Replaces the legacy in-process `FeishuChannel` driver.

```sh
agentm-gateway --bind unix:///tmp/gw.sock &
agentm-feishu \
  --connect unix:///tmp/gw.sock \
  --app-id cli_xxxx \
  --app-secret /run/secrets/feishu_app_secret
```

The app secret is read from a file (or `LARK_APP_SECRET` env). It is
never accepted as a CLI argument — secrets must not appear in argv
(per `autoharness:cli-design` rule group 5).

Exit codes (per `cli-design` rule group 3):

| Code | Meaning |
|------|---------|
| 0 | Clean shutdown |
| 1 | Generic unexpected error |
| 2 | Argument / usage error |
| 4 | Auth rejected by gateway or Feishu credentials invalid |
| 6 | User interrupt (SIGINT) |
| 7 | Cannot connect to socket |
