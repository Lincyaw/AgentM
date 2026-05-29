# agentm-feishu

Feishu / Lark client process for the AgentM channels gateway.

Connects to an `agentm gateway --bind unix://…` over the v2 wire
protocol and bridges inbound Feishu messages / card-button clicks into
the gateway, then renders gateway outbounds as Feishu interactive
cards. Replaces the legacy in-process `FeishuChannel` driver.

```sh
agentm gateway --bind unix:///tmp/gw.sock &
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

## Card rendering

The gateway fans out the full session event surface (`turn_start`,
`stream_text`, `tool_call`, `usage`, …). Rather than post a card per
event — which would flood the chat — the adapter collapses one agent run
into a single **live card** that is updated in place:

- an **activity line** shows the latest 1-2 operations in real time (⏳
  running → ✅/❌ done, falling back to 思考中 / ✅ 完成) so the chat
  perceives ongoing progress — there is no generic "正在回答" state;
- the answer body is filled by the final `assistant_text`;
- the full tool-step history accumulates in a **collapsible panel**
  (auto-expanded while working, collapsed once the answer lands) so detail
  is available without crowding the reply.

`approval_request` and `diagnostic_error` / `diagnostic_warning` get
their own standalone cards (approvals carry interactive buttons and must
persist). Runtime/observability kinds (`usage`, `child_*`, `extension_*`,
`api_*`, `session_ready`, …) are dropped — they belong in the terminal
TUI, not a chat.
