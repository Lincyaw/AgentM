# Task: observability-fingerprint — Active-set fingerprint + atom.reload markers

**Date**: 2026-05-01
**Status**: PENDING
**Plan**: [self-mod-mvp](../plans/2026-05-01-self-mod-mvp.md)
**Design**: [evolution-substrate](../designs/evolution-substrate.md) §4, §4.2
**Assignee**: implementer
**Wave**: 2 (parallel)
**Size**: S
**Depends on**: [catalog-storage](2026-05-01-catalog-storage.md) (for `compute_active_set_fingerprint`), [transactional-reload](2026-05-01-transactional-reload.md) (for `ExtensionReloadEvent`)

## Objective

Embed the active-set fingerprint into the observability JSONL stream so the indexer can attribute every event to the exact (core, scenario, atom_versions) tuple under which it ran. Add an `atom.reload` marker record when an atom is reloaded mid-session.

Per plan §4 R10: the **`session.start` record stays as-is** (anchor record before any extension is loaded). The fingerprint is emitted as a **separate `session.fingerprint` record** subscribed on `session_ready`, where every extension is guaranteed to be loaded. This is a documented deviation from evolution-substrate §4 and must be noted in the design doc post-MVP.

## Inputs to read

- `evolution-substrate.md` §4 (header structure), §4.2 (mid-session reload)
- `src/agentm/extensions/builtin/observability.py` (existing `_on_ready`, sink writers)
- `src/agentm/core/catalog/hashing.py` (`compute_active_set_fingerprint` — landed in `catalog-storage`)
- `src/agentm/harness/events.py` (`ExtensionReloadEvent` — landed in `transactional-reload`)

## Outputs

### Modified files

| Path | Change |
|---|---|
| `src/agentm/extensions/builtin/observability.py` | (a) Subscribe to `session_ready`. In addition to the existing `session.ready` record, compute fingerprint via `agentm.core.catalog.compute_active_set_fingerprint(...)` and write a `session.fingerprint` record. (b) Subscribe to `extension_reload`; on each event, write an `atom.reload` record carrying old/new hash, trigger, and the new fingerprint snapshot. (c) Update `MANIFEST.registers` to include `event:extension_reload`. |
| `tests/unit/extensions/builtin/observability/test_observability.py` (existing) | Add tests below. (Or, if cleaner, a new file `test_fingerprint.py` in the same dir.) |

## Concrete shape — `session.fingerprint` record

```json
{
  "schema": "otel/span/v0",
  "kind": "session.fingerprint",
  "trace_id": "<existing trace id>",
  "span_id": "<new>",
  "name": "session.fingerprint",
  "start_time_unix_nano": <now>,
  "attributes": {
    "core": "core@<hash or null>",
    "scenario": "<scenario_name>@<hash or null>",
    "atoms": {
      "tool_read": "tool_read@e5f6abc12345",
      ...
    },
    "task_meta": {
      "type": null,
      "difficulty": null,
      "external_id": null
    }
  },
  "status": {"code": "OK"}
}
```

`task_meta` is optional and not populated by the harness in MVP; the indexer reads `task_meta` if present (some test fixtures inject it). Default values are `null` — a subsequent task in Phase 2 (or a presenter feature) will write task metadata at `prompt()` boundaries.

## Concrete shape — `atom.reload` record

```json
{
  "schema": "otel/span/v0",
  "kind": "atom.reload",
  "trace_id": "...",
  "span_id": "...",
  "name": "atom.reload:<atom_name>",
  "start_time_unix_nano": <now>,
  "attributes": {
    "name": "<atom_name>",
    "old_hash": "<old hash>",
    "new_hash": "<new hash>",
    "trigger": "agent" | "human" | "propose_change_approved",
    "tier": <int>,
    "fingerprint_after": { ... full fingerprint after reload ... }
  },
  "status": {"code": "OK"}
}
```

The indexer uses `fingerprint_after` to attribute *subsequent* events in the trace. (Older events keep the original `session.fingerprint`.) The indexer also flags every event after the first `atom.reload` in a trace with `mid_session_reload=True` per evolution-substrate §4.2.

## Acquiring the loaded set inside observability

Observability at `session_ready` time has access to the `SessionReadyEvent.tool_names`/`command_names` but NOT directly to the loaded atom set. Two options:

1. (Chosen) Read `discover_builtin()` from `agentm.extensions.discover` — observability is allowed to import this per the §11 allow-list (`agentm.extensions` is on the list). Each discovered module's `inspect.getsource(module)` is hashed via `compute_atom_hash`. This gives the **discovered** set, which equals the loaded set when no recipe filtering happens (MVP).
2. (Rejected) Have the harness pass the loaded set explicitly via a new field on `SessionReadyEvent`. Rejected because changing the event payload affects every existing observer; defer to Phase 2 if recipe filtering becomes load-time.

**Recipe filtering note**: in MVP, scenarios load atoms by listing them in YAML. Atoms NOT in the YAML are NOT loaded — but they ARE discovered. The observability fingerprint should record only **loaded** atoms. We get this set by checking `api_register` events that have already fired by `session_ready` — observability already subscribes to these. So: track loaded module paths from `api_register` events (the `extension` attribute), intersect with `discover_builtin()` for source hashing. Document this in the function's comment.

## Test cases

| Test | Asserts |
|---|---|
| `test_M2_fingerprint_record_includes_all_loaded_atoms` | After session.create with N atoms loaded, the JSONL contains a `session.fingerprint` record with `attributes.atoms` containing exactly N entries |
| `test_fingerprint_record_atom_format` | Each entry's value matches `<name>@<hash>` (12 hex chars) |
| `test_fingerprint_record_includes_scenario_when_provided` | Scenario name + hash present when load came from `load_scenario` (best-effort: harness passes scenario hash via a future hook; for MVP, `null` is acceptable) |
| `test_atom_reload_record_emitted_on_extension_reload` | After `bus.emit("extension_reload", ExtensionReloadEvent(...))`, an `atom.reload` record appears |
| `test_atom_reload_record_carries_new_fingerprint` | The `fingerprint_after` field has the new hash for the reloaded atom |
| `test_session_start_record_unchanged` | The existing `session.start` record format is preserved (no breaking changes for the existing `test_observability.py` tests) |

## Acceptance Conditions

- [ ] `uv run pytest tests/unit/extensions/builtin/observability/ -v` all green
- [ ] Existing observability tests still pass
- [ ] `uv run ruff check src/agentm/extensions/builtin/observability.py` clean
- [ ] `uv run mypy src/agentm/extensions/builtin/observability.py` clean
- [ ] Validator gate `tests/unit/extensions/test_extension_contract.py` green (no new import violations — `agentm.core.catalog` already added to allow-list in `core-manifest`)
- [ ] No new third-party dependency

## Acceptance scenarios covered

- **M2** — `test_M2_fingerprint_record_includes_all_loaded_atoms`
- **E8** (partial) — `test_atom_reload_record_emitted_on_extension_reload`

## Notes

- **R10 deviation noted in code comments**: the `session.start` record still fires immediately at install time; the fingerprint lands in a follow-up `session.fingerprint` record at `session_ready`. The indexer (next task) reads `session.fingerprint` for attribution. Add a one-line comment in observability and one in the indexer pointing at this plan's R10.
- **No `task_meta.external_id`** — that requires a presenter-level concept of "task". The field is reserved in the schema; populated `null` in MVP.
- **No mid-session reload exclusion logic in observability** — that lives in the indexer. Observability is a pure subscriber.
- **Performance**: hashing all loaded modules at `session_ready` runs once per session and reads source via `inspect.getsource`. For ~25 atoms this is cheap (<10 ms). No need for caching in MVP; if it becomes hot, cache by module-id-and-mtime.
```

============ END FILE ============