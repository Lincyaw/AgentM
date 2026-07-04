# AgentM Glossary

Terms with precise meaning inside this project. Implementation details belong elsewhere — this file is a glossary, not a spec.

## Entry point

The CLI is the supported entry: `agentm` (one-shot prompt, plus the `agentm gateway` and `agentm trace` subcommands) and the separate chat-client peer binaries `agentm-feishu` / `ag` (vendor-SDK isolation only). `AgentSession.create` is the substrate-level constructor those CLIs call. Notebook / embedder use is not a current priority — there is no `agentm.Session` façade.

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

Do not rebuild the session/fork/workflow substrate for this work. AgentM already
has the needed primitives:

1. `agentm --fork` and `agentm fork` can continue from an existing session at a
   message id, turn id, turn index, or message-count prefix.
2. `agentm trace messages` / `agentm trace turns` expose the selectors needed to
   choose a fork point.
3. `agentm workflow run` can run a complete case workflow, spawn child agents,
   pass `atom_config`, restrict tools, set `trace_label`, and report child
   session ids.
4. Rescue-window experiments live in the standalone `agentm-rescue-window`
   eval package: baseline session -> policy decision -> SDK-backed fork
   continuation -> scenario adapter/export. RCA only owns the scoring/export
   adapter, not the fork runtime.

## RCA scenario working context

For single-agent RCA debugging, use the baseline manifest:
`agentm --scenario rca:baseline`. It includes
`contrib/scenarios/rca/overlays/base-investigator.yaml`, which mounts local
operations, `rca.default.duckdb_sql`, `rca.default.finalize`, observability,
the rcabench contract, prompt loader, and tool index.

`rca:baseline` now excludes `conclusion.parquet` through
`contrib/scenarios/rca/overlays/base-investigator.yaml`, matching the clean
RCA eval expectation. Local case data is mounted with `AGENTM_RCA_DATA_DIR`;
the current clean dataset lives at:

```text
/Users/bytedance/workspace/AgentM/datasets/ops-lite-clean/cases/<case-name>
```

Current baseline/fork smoke test used:

```text
case: /Users/bytedance/workspace/AgentM/datasets/ops-lite-clean/cases/ts0-ts-seat-service-pod-failure-c87xdg
cwd:  /tmp/agentm-rca-verify-seat-pod
```

Baseline command:

```bash
AGENTM_RCA_DATA_DIR=/Users/bytedance/workspace/AgentM/datasets/ops-lite-clean/cases/ts0-ts-seat-service-pod-failure-c87xdg \
AGENTM_DUCKDB_THREADS=1 \
uv run agentm --cwd /tmp/agentm-rca-verify-seat-pod \
  --scenario rca:baseline \
  -p 'Investigate the incident using the RCA telemetry dataset mounted via AGENTM_RCA_DATA_DIR. Identify the root cause service and fault kind from the observability data, then submit the final RCA report via the required tool.'
```

Baseline session `32d9b6203aba4f7d82071070bddd149d` completed naturally:
24 turns, 32 tool calls, 599,421 input tokens, 10,761 output tokens. It
submitted `ts-seat-service` with `fault_kind: pod_failure`.

Fork smoke command:

```bash
AGENTM_RCA_DATA_DIR=/Users/bytedance/workspace/AgentM/datasets/ops-lite-clean/cases/ts0-ts-seat-service-pod-failure-c87xdg \
AGENTM_DUCKDB_THREADS=1 \
uv run agentm fork 32d9b6203aba4f7d82071070bddd149d \
  --cwd /tmp/agentm-rca-verify-seat-pod \
  --turn-index 22 \
  --prompt '<system_reminder>Before finalizing, do not mistake route-plan/travel/travel2 UI latency as the root. Re-check whether the direct seat-service pod/container evidence explains the downstream 503 and timeout pattern. If the evidence supports it, submit the final RCA report via submit_final_report.</system_reminder>'
```

Fork session `34ac1defdca948b0906bc2485a1d530b` continued from ClickHouse by
session id even though `/tmp/agentm-rca-verify-seat-pod/.agentm` had no local
session JSONL. It ran 3 new turns, submitted the same
`ts-seat-service:pod_failure` result, and is discoverable with:

```bash
uv run agentm trace index \
  --children-of 32d9b6203aba4f7d82071070bddd149d \
  --format ndjson --no-cache
```

`agentm trace info --session 34ac1defdca948b0906bc2485a1d530b --format ndjson`
shows:

```json
{
  "parent_session": "32d9b6203aba4f7d82071070bddd149d",
  "lineage": {
    "kind": "fork",
    "source_session_id": "32d9b6203aba4f7d82071070bddd149d",
    "fork_point": {"turn_index": 22}
  }
}
```

The reminder case-study exporter can start from only the baseline session and
discover fork children from ClickHouse lineage:

```bash
uv run python scripts/export_reminder_case_study.py \
  --baseline-session 32d9b6203aba4f7d82071070bddd149d \
  --out-prefix runs/rescue-window/verify-seat-pod-case-study
```

It wrote `runs/rescue-window/verify-seat-pod-case-study.csv` and `.md`
(ignored `runs/` artifacts). The exported row was `unchanged`: both baseline
and fork exact-matched `ts-seat-service:pod_failure`.

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
  and rescue-window reminder branch continuations.

This should let a ClickHouse view reconstruct a table like:
baseline session -> forked reminder session -> insertion point -> child
extractor/auditor sessions -> final outcome, without coupling that table to
llmharness internals.
