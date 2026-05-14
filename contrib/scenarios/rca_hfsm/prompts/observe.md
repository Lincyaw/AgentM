# OBSERVE state

The symptoms are recorded. Before you propose a hypothesis, gather enough
L1 facts that any hypothesis you propose has at least one falsifiable
prediction tied to evidence the trace can re-check.

Start by querying the case's parquet fixtures: call `list_tables` to see
the schema, then `query_sql` to read concrete facts (trace timings,
error spans, metric series, log rows). The standard rca fixture exposes
tables such as `abnormal_traces` / `normal_traces`, `abnormal_logs` /
`normal_logs`, `metrics_sum`, and pod/phase tables; columns include
`service_name`, `parent_span_id`, `attr.http.response.status_code`, and
nanosecond `duration`. **You cannot reason about a system you haven't
queried.** Do not skip this — the symptoms are alerts, not evidence.

Use `record_observation` to append the citable facts you find — log
snippets, query result rows, file contents. Each observation should
name its `source_tool_call` (the tool that produced it) and link to the
symptoms it speaks to via `related_symptoms`. The observation cache
memoises idempotent tool calls, so you can re-issue a probing query
without paying for it twice.

Resist the temptation to jump to a hypothesis on the first plausible
correlation. A hypothesis proposed without grounding observations cannot
declare meaningful negative predictions, and the gate will reject it.

Move to HYPOTHESIZE by calling `propose_hypothesis` once you have enough
facts to articulate at least one prediction that, if observed, would
**rule the hypothesis out**.

Available tools: `list_tables`, `query_sql`, `record_observation`,
`record_symptom`, `propose_hypothesis`.
