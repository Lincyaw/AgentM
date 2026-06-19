# AgentM Glossary

Terms with precise meaning inside this project. Implementation details belong elsewhere — this file is a glossary, not a spec.

## Entry point

The CLI is the supported entry: `agentm` (one-shot prompt, plus the `agentm gateway` and `agentm trace` subcommands) and the separate chat-client peer binaries `agentm-feishu` / `agentm-terminal` (vendor-SDK isolation only). `AgentSession.create` is the substrate-level constructor those CLIs call. Notebook / embedder use is not a current priority — there is no `agentm.Session` façade.

Power-users override defaults by passing fields into `AgentSessionConfig`.

## Substrate axiom (aspirational, not enforced)

The design doc `pluggable-architecture.md` says: *"substrate provides registration hooks and asserts at freeze time that the required services have been registered; it never instantiates a default."*

Today the code violates this — `build_extension_api_scope` and `session_factory` auto-instantiate `GitBackedResourceWriter`, `default_project_layout`, `default_catalog_service`, `InMemorySessionManager`, `InMemoryResourceLoader`, and `LastRegisteredWins`. Closing this gap was scoped, planned (see the grilling-session record in chat history), and then deliberately deferred: the cleanup is large, the user-visible benefit is zero, and the partially-done Phase 1 explored on branch `worktree-agent-phase1-retry` showed the constitution-path machinery (`is_constitution_path` / `load_core_manifest`) blocks any atom-based `GitBackedResourceWriter` until those helpers move out of `_internal/`.

If this work is revived, the load-bearing prerequisites are:
1. Move constitution-path helpers from `core/_internal/catalog/manifest.py` to `core/lib/` or `core/abi/`.
2. Decide JSONL vs in-memory as the floor `SessionManager` (CLI needs JSONL persistence; the in-memory default in `worktree-agent-phase1-retry` was a Phase 1 stub).
3. Promote `SessionManager` / `ResourceLoader` / `ProjectLayout` from `core.runtime` concrete classes to `core.abi` Protocols so atoms can satisfy them structurally.
4. Phase the substrate auto-instantiation removal so the CLI doesn't break mid-migration.

## Substrate-only (kernel singletons)

Two pieces are not pluggable by design and have no atom-replacement axis: **CatalogService** (`.agentm/catalog/` is constitution-layer) and **ProviderResolver** ("last-registered wins" is universal). The boundary axiom does not bind these.

## llmharness / TEL working context

`contrib/extensions/llmharness` is the cognitive-audit extension. The live
adapter runs extractor and auditor child sessions beside a main AgentM session:
the extractor turns the trajectory into an event graph, and the auditor looks
for drift, blind spots, premature conclusions, or reminders worth surfacing.

The current working area is the TELBench / TEL-agent path, not the main
adapter. TEL means trajectory error localization: a trajectory is split into
ordered spans, and the agent must submit the span ids where the original agent
actually committed an unwarranted action. The 2-pass TEL workflow uses
`agents/tel/prompts/notepad.md` for a first-pass attention index and
`agents/tel/prompts/reason.md` for critic verification and final
`submit_error_spans`.

The current uncommitted prompt edits are intentionally narrower than the old
"flag everything suspicious" wording. They emphasize evidence available at the
time, exact task constraints, commitment strength, and separating the first
defective support from later carrier spans. The expected effect is better
TELBench precision without losing the causal-origin recall that matters for
span-level F1 and first-error accuracy.

Do not rebuild the session/fork/workflow substrate for this work. AgentM already
has the needed primitives:

1. `agentm --fork` and `agentm fork` can continue from an existing session at a
   message id, turn id, turn index, or message-count prefix.
2. `agentm trace messages` / `agentm trace turns` expose the selectors needed to
   choose a fork point.
3. `agentm workflow run` can run a complete case workflow, spawn child agents,
   pass `atom_config`, restrict tools, set `trace_label`, and report child
   session ids.
4. RCA already has a scenario-local replay-fork reference implementation:
   baseline session -> offline audit -> surfaced reminder -> fork continuation
   -> judge. That code is reference material, not something that needs to be
   folded into the main `agentm` command.

## RCA scenario working context

For single-agent RCA debugging, use the baseline manifest:
`agentm --scenario rca:baseline`. It includes
`contrib/scenarios/rca/overlays/base-investigator.yaml`, which mounts local
operations, `rca.default.duckdb_sql`, `rca.default.finalize`, observability,
the rcabench contract, prompt loader, and tool index.

Local case data can be mounted with `AGENTM_RCA_DATA_DIR`. On this machine,
`/Users/bytedance/dataset/<case-id>` contains RCA parquet case directories.
Case `1` has the standard files: abnormal/normal logs, metrics, metric sums,
histograms, traces, plus `env.json`, `injection.json`, and
`conclusion.parquet`.

Full smoke run recorded on case `1`:

```bash
AGENTM_RCA_DATA_DIR=/Users/bytedance/dataset/1 \
AGENTM_DUCKDB_THREADS=1 \
uv run agentm --cwd /tmp/agentm-rca-case1-full \
  --scenario rca:baseline \
  -p 'Investigate the incident using the RCA telemetry dataset mounted via AGENTM_RCA_DATA_DIR. Identify the root cause service and fault kind from the observability data, then submit the final RCA report via the required tool.'
```

The first attempt timed out on the provider after loading skills and calling
`list_tables`, then resumed successfully with:

```bash
uv run agentm --cwd /tmp/agentm-rca-case1-full \
  --scenario rca:baseline \
  --resume 7e176647595544d48e31cdb83fa34163 \
  -p 'Continue the RCA investigation from the existing trajectory. Use the telemetry tables to identify the root cause service and fault kind, then submit the final report via submit_final_report.'
```

Final session `7e176647595544d48e31cdb83fa34163` completed naturally:
20 turns, 33 tool calls, 600,058 input tokens, 9,400 output tokens. Final
submission named `mysql` with `fault_kind: pod_failure`; evidence included
`k8s.statefulset.ready_pods` dropping to 0 in the abnormal window, MySQL
startup/init logs, and downstream `ts-auth-service` / `ts-train-service`
missing-table errors.

`rca:baseline` now excludes `conclusion.parquet` through
`contrib/scenarios/rca/overlays/base-investigator.yaml`, matching the clean RCA
eval expectation. A quick `list_tables` smoke test on case `1` confirmed the
`conclusion` view is no longer exposed.

Hard case recorded on local case `1188`:

```bash
AGENTM_RCA_DATA_DIR=/Users/bytedance/dataset/1188 \
AGENTM_DUCKDB_THREADS=1 \
uv run agentm --cwd /tmp/agentm-rca-case1188-full \
  --scenario rca:baseline \
  -p 'Investigate the incident using the RCA telemetry files. Identify the root cause service and fault kind from the observability data, then submit the final RCA report via the required tool.'
```

Final session `aeb6719f76e5496c8114cf49d70d46c8` completed naturally:
67 turns, 74 tool calls, 2,996,965 input tokens, 32,108 output tokens. The
injection is an HTTP request-delay on
`ts-route-plan-service` -> `ts-travel2-service`; the strict RCA judge expects
`ts-route-plan-service` with `fault_kind: http_slow`. The agent found strong
edge evidence (`routeplan_to_travel2_GET` abnormal gap >1s on 100% of pairs,
normal 0%), but submitted `ts-route-plan-service` with `fault_kind:
network_delay`. This is a useful wrong/hard baseline trajectory for later
RCA-agent debugging.

Reminder fork smoke tests on this baseline:

- `9bdb5ae7ac1944e4b705b95ebb6717a2`: forked from turn 65 with a
  downstream-victim reminder. Output flipped to
  `ts-travel2-service:network_delay`; exact_match stayed false and secondary
  service-hit metrics got worse.
- `a3595aaaea0148bba561793873fc84f7`: forked from turn 65 with a
  fault-kind-disambiguation reminder. Output flipped to
  `ts-route-plan-service:http_slow`; exact_match became true.

Local case-study tables were exported to
`runs/replay-fork/case-study-1188-reminder.csv` and
`runs/replay-fork/case-study-1188-reminder.md` (under ignored `runs/`).
For current traces, the exporter starts from the baseline/root session and
discovers fork variants from `agentm trace index --children-of` plus each
child's lineage metadata:

```bash
uv run python scripts/export_reminder_case_study.py \
  --baseline-session aeb6719f76e5496c8114cf49d70d46c8 \
  --out-prefix runs/replay-fork/case-study-1188-reminder
```

If a historical baseline/fork pair predates the parent/lineage metadata needed
by `trace index`, rerun that baseline and its forks instead of adding legacy
query fallback. `--variant`, `--case-id`, `--data-dir`, and `--hypothesis`
remain manual overrides for historical exports or nicer labels.

## Session lineage metadata for reminder case studies

AgentM now has a harness-independent metadata path for ablation/case-study
analysis:

- `AgentSessionConfig.lineage` records structural provenance such as root,
  fork, sub-agent, workflow worker, source session id, parent session id, and
  fork point.
- `AgentSessionConfig.experiment` records study context such as
  `kind=reminder_injection`, case id, baseline session id, insertion turn or
  message id, reminder id, and reminder text.
- The observability atom writes both dictionaries into the
  `agentm.session.start` body and projects stable short fields into
  `LogAttributes` for ClickHouse SQL. Reminder text itself is not indexed as an
  attribute; `agentm.session.experiment.reminder_text_hash` is emitted instead.
- Main covered entry points: `agentm -p` root sessions, CLI/gateway forks,
  builtin `sub_agent` children, workflow workers, llmharness child sessions,
  and the RCA replay-fork reminder continuation path.

This should let a ClickHouse view reconstruct a table like:
baseline session -> forked reminder session -> insertion point -> child
extractor/auditor sessions -> final outcome, without coupling that table to
llmharness internals.
