# Local OTel collector for AgentM

`agentm.extensions.builtin.otlp_export` ships AgentM's OTLP spans + logs
to a remote collector in real time (in addition to the per-session
ndjson file written by `observability`). This directory gives you a
one-command collector + Jaeger UI to receive them locally.

## Start

```bash
docker compose -f tools/otel/docker-compose.yaml up -d
uv run agentm --extension agentm.extensions.builtin.otlp_export "hello"
open http://localhost:16686       # Jaeger UI, service = "agentm"
```

Stop with `docker compose -f tools/otel/docker-compose.yaml down`.

After the first traces arrive, run the index init script once to add
materialized columns and bloom filter indexes for fast session-level queries:

```bash
clickhouse-client --host localhost --query "$(cat tools/otel/init-db.sql)"
# or via HTTP:
curl http://localhost:8123/ --data-binary @tools/otel/init-db.sql
```

Without these indexes, `agentm trace export-dataset` and other
session-filtered queries do full table scans on every call.

## What you get

Span tree per session (GenAI semconv aligned):

```
invoke_agent <scenario>
├── agentm.turn (index=0)
│   ├── chat <model>            (gen_ai.* attributes)
│   └── execute_tool <name>     (gen_ai.tool.call.{arguments,result})
└── agentm.turn (index=1)
    └── ...
```

Lifecycle events (`extension.install`, `atom.reload`, `api.register`,
`api.send_user_message`, `agentm.session.start`, `agentm.session.end`,
`agentm.message.appended`, `agentm.turn.summary`, `agentm.diagnostic`)
are emitted as **log records** on the same OTel pipeline, not spans.

## Cross-session linkage

Sub-agent + cognitive-audit child sessions share the parent's
`trace_id` (W3C TraceContext propagation). Each session's
`invoke_agent` span additionally carries `agentm.session.root_id` and
`agentm.session.parent_id` attributes for stable Lucene/SQL grouping
when traces span processes.

## Config knobs (all optional)

| Env var | Atom config | Default |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `endpoint` | `http://localhost:4317` (gRPC) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `protocol` | `grpc` (or `http/protobuf`) |
| `OTEL_EXPORTER_OTLP_HEADERS` | `headers` | none |
| `OTEL_EXPORTER_OTLP_INSECURE` | `insecure` | `true` (grpc only) |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | `timeout` | `10` (seconds) |

Send to a remote collector (Grafana Cloud / Honeycomb / etc.) by setting
`OTEL_EXPORTER_OTLP_ENDPOINT` + `OTEL_EXPORTER_OTLP_HEADERS` — the
docker-compose here is for local dev only.

## Useful Jaeger queries

* `gen_ai.request.model = "<your-model>"` — every LLM call.
* `gen_ai.tool.name = "submit_verdict"` — only cognitive-audit verdicts.
* tag `agentm.session.root_id = <hex>` — every span in one agent tree
  (parent + sub-agents + audit children).
