# Local persistence and observability services

This compose stack provides two independent data planes for development:

| Service | Role | Data ownership |
| --- | --- | --- |
| PostgreSQL | Optional authoritative `TrajectoryStore` and `TrajectoryNodeStore` backend | Complete resumable sessions, turns, nodes, heads, cache, and compaction state |
| OTel collector | OTLP transport | No durable AgentM ownership; forwards diagnostic logs and spans |
| ClickHouse | Collector-managed observability query backend | Standard `otel_logs` and `otel_traces` only |

Deployments select one authoritative trajectory backend, such as JSONL or
PostgreSQL. OTLP is optional and never substitutes for a failed trajectory
write. ClickHouse does not contain an AgentM-specific trajectory mirror.

## Start

```bash
docker compose -f tools/otel/docker-compose.yaml up -d
docker compose -f tools/otel/docker-compose.yaml ps
```

For a CLI session whose scenario installs the `observability` atom:

```bash
AGENTM_TRAJECTORY_DSN=postgresql://agentm:agentm@localhost:55432/agentm_test \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
OTEL_EXPORTER_OTLP_INSECURE=true \
uv run agentm run --scenario chat --message "hello"
```

The first variable selects PostgreSQL as the one authoritative trajectory
store. Without it, the CLI deliberately writes JSONL under the current
project's `.agentm/trajectory/` directory instead.

Open `http://localhost:8123/play` to query ClickHouse. PostgreSQL is available
at `postgresql://agentm:agentm@localhost:55432/agentm_test`.

Stop the services without deleting their volumes:

```bash
docker compose -f tools/otel/docker-compose.yaml down
```

## ClickHouse indexes

After the collector creates its standard tables, apply the idempotent
session-id indexes:

```bash
docker exec -i agentm-clickhouse clickhouse-client --multiquery \
  < tools/otel/init-db.sql
```

The indexes accelerate filters on the shared `agentm.session.id` attribute.
They do not create session or turn storage.

## Signals

| Signal | Examples | Query source |
| --- | --- | --- |
| Lifecycle logs | `agentm.session.start`, `agentm.session.ready`, `agentm.session.end` | `otel.otel_logs` |
| Runtime logs | `agentm.turn.committed`, `agentm.run.end`, `agentm.event.dispatch` | `otel.otel_logs` |
| Spans | `agentm.session <purpose>`, `agentm.turn`, `chat <model>`, `execute_tool <name>` | `otel.otel_traces` |

All records carry `agentm.session.id`; session lifecycle records also carry
root and parent ids. These ids correlate observability with the selected
trajectory store without copying complete turns into OTLP.

## Configuration

| Setting | Meaning | Default |
| --- | --- | --- |
| observability atom `export` | `auto`, `local_file`, or `otlp` | `auto` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector endpoint | `http://localhost:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` or `http/protobuf` | `grpc` |
| `OTEL_EXPORTER_OTLP_HEADERS` | Remote collector headers | none |
| `OTEL_EXPORTER_OTLP_INSECURE` | Disable TLS for gRPC | inferred for local endpoint |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | Export timeout in seconds | `10` |

`export=auto` uses OTLP when the collector is reachable and otherwise writes
the local per-session OTLP JSONL diagnostic file. `export=otlp` does not turn
OTLP into trajectory persistence; resume still requires the configured
`TrajectoryStore`.
