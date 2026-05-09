# Issue #87 — Builtin Atoms Consume ExtensionAPI C-1 Capabilities

## Scope

Rework five builtin atom files around the C-1 surfaces:

- `observability`: use `api.add_observer`, `ResourceWriter.current_version_for_path`, and `core.lib._to_jsonable`; document the `path` config placeholder.
- `sub_agent`: use `api.spawn_child_session(**kwargs)`, resolve `system_prompt` via `discover_builtin`, move default inherited extensions into scenario config, and use `core.lib._to_jsonable` for tool payloads.
- `artifact_store`: register the live store through `api.set_service("artifact_store", store)` and have tools resolve it through `api.get_service`.
- `micro_compact` / `llm_compaction`: rely on the shared JSON serializer instead of local ladders where applicable.

## C18 Decision

Keep `src/agentm/extensions/builtin/sub_agent.py` as a single atom file in this round.
The split threshold is **at least 1500 LOC**; issue #87 does not split the atom even though it is over 1000 LOC, preserving the §11 single-file builtin contract and avoiding an unrequested refactor.

## Verification Plan

- Grep gates from the issue acceptance criteria.
- `validate_atom_file` for `observability`, `sub_agent`, `artifact_store`, `micro_compact`, and `llm_compaction`.
- Regression tests for observability trace shape, CLI trajectory identity, observer reload cleanup, sub-agent spawn kwargs/config behavior, artifact-store per-session service isolation, and compaction atom serializer/summarizer behavior.
- `uv run ruff check src/`, `uv run mypy src/`, and `uv run pytest --tb=short`.

## Review Follow-up

- Observer registrations created through `api.add_observer` are tracked by the atom reloader with the same owner cleanup path as `api.on`, so reload/unload cannot leave stale trace observers attached.
- The CLI observability regression drives `agentm` through a sandbox repo and inspects `.agentm/observability/<session_id>.jsonl`, matching the identity E2E rule for trajectory-affecting atom changes.
- Compaction sibling coverage is intentionally focused on the shared serializer boundary and provider summarizer behavior; no extra scenario E2E is added because the C10 change is an atom-local serialization substitution, not a new fail-stop position beyond §11 validator and trace identity checks.

## Review Follow-up 2

- Restored `include_mutation_diff` compatibility without reintroducing `api.on` monkey-patching by adding an observer-side `on_handler_start` hook to snapshot mutable event payloads before handler invocation and emit `handler.mutated` spans from `on_handler_done`.
- Made artifact ID allocation safe for concurrent same-root session services by replacing per-process module registries with a filesystem `.next_id.lock` around `.next_id` updates.
- Added regression coverage for mutation-diff trace records and concurrent same-root artifact writes producing unique IDs.
