# Task: indexer-mvp — Post-session attribution + CLI rebuild

**Date**: 2026-05-01
**Status**: PENDING
**Plan**: [self-mod-mvp](../plans/2026-05-01-self-mod-mvp.md)
**Design**: [evolution-substrate](../designs/evolution-substrate.md) §5
**Assignee**: implementer
**Wave**: 3 (parallel)
**Size**: M
**Depends on**: [catalog-storage](2026-05-01-catalog-storage.md), [observability-fingerprint](2026-05-01-observability-fingerprint.md)

## Objective

Read raw observability JSONL files, derive a minimal set of metrics (`n_runs`, `task.completion_rate`, `tokens_per_task` if available), and atomic-append them to `<atom>/<hash>/metrics.jsonl`. Symlink each contributing trace into `<atom>/<hash>/runs/<trace_id>`. Provide a CLI (`python -m agentm.core.catalog.indexer rebuild`) that re-derives idempotently from raw observability.

**MVP scope is deliberately narrow**: no `compare()`, no statistical confidence intervals, no `regressed` flag. Just counters that prove the substrate works end-to-end (E5).

## Inputs to read

- `evolution-substrate.md` §5 (pipeline), §5.2 (bucketing), §5.3 (what NOT to do)
- `src/agentm/core/catalog/_layout.py` (path constants — DO NOT recompute paths here)
- `src/agentm/core/catalog/hashing.py` (`compute_active_set_fingerprint` for verification, not derivation)
- `src/agentm/extensions/builtin/observability.py` — record formats: `session.fingerprint`, `atom.reload`, `agent_end` (carries `stop_reason`), `llm.request.end` (carries usage, optional)

## Outputs

### New files

| Path | Purpose |
|---|---|
| `src/agentm/core/catalog/indexer.py` | Indexer module + CLI entry point. Public function: `index_trace(trace_path: Path, *, root: Path | None = None) -> IndexerResult`. CLI: `python -m agentm.core.catalog.indexer rebuild [--root .agentm/catalog] [--observability .agentm/observability]`. |
| `tests/unit/core/catalog/test_indexer.py` | Tests below. |

### Modified files

| Path | Change |
|---|---|
| `src/agentm/harness/session.py` | In `AgentSession.shutdown()`, **after** `bus.clear()`, call `agentm.core.catalog.indexer.index_trace(<trace_path>)` in a `try/except` — failures log a warning but don't propagate. The trace path is supplied via the observability extension's known path; for MVP we agree on the convention `<cwd>/.agentm/observability/<trace_id>.jsonl` and the session knows its `session_id`. If the trace is missing (observability not loaded), the indexer call is skipped. |

## §1. Indexer pipeline (per design §5.1)

```
def index_trace(trace_path: Path, *, root: Path) -> IndexerResult:
    1. Open trace_path.
    2. Read records line-by-line; collect:
       - session.fingerprint (the canonical anchor)
       - atom.reload events (split the trace into "epochs" — events
         before the first reload attribute to the original fingerprint;
         events between reload[i] and reload[i+1] attribute to the
         post-reload fingerprint)
       - agent_end records (one per turn; final one's stop_reason
         drives task.completion_rate)
       - llm.request.end records (optional; sum up token usage if present)
    3. Derive metrics for the **final** fingerprint epoch:
       - n_runs: 1 (this trace contributes one run)
       - task.completion_rate: 1.0 if the LAST agent_end.stop_reason
         is "end_turn" or "stop", else 0.0
       - tokens_per_task: sum of (input_tokens + output_tokens) from
         llm.request.end if any; else null
    4. For each atom in fingerprint.atoms:
       - resolve <atom>/<hash>/ via _layout
       - if dir doesn't exist: skip (atom never frozen — possible
         for atoms that haven't been edited yet; their genesis version
         is recorded only after the first reload OR after first
         shutdown when we lazily-freeze per the genesis rule below)
       - atomic-append metrics row to metrics.jsonl
       - os.symlink trace_path → runs/<trace_id> (idempotent: skip if
         exists)
    5. Return IndexerResult(n_atoms_attributed=..., warnings=[...]).
```

### §1.1 Genesis-version rule

An atom that has never been reloaded has no entry in `.agentm/catalog/atoms/<name>/`. The indexer needs *some* hash to attribute to. Resolution:

- On first session shutdown that includes an atom in `session.fingerprint`, if `.agentm/catalog/atoms/<name>/` does not exist, the indexer **lazily freezes** the genesis version using the source it can read from disk via `inspect.getsourcefile(module)` → `Path.read_text()`. The hash should match the fingerprint's hash if the source hasn't changed since the fingerprint was computed (it shouldn't have).
- If the hash *doesn't* match (the source on disk drifted between fingerprint capture and indexer run): emit a warning record into the metrics.jsonl with `attributes.warning="source_drifted_post_fingerprint"`. This should be vanishingly rare in practice.

### §1.2 Bucketing per §5.2

The metrics row format (one line per index call per atom):

```jsonc
{
  "indexed_at": "<ISO8601 UTC>",
  "scenario": "<scenario name or null>",
  "task_type": null,             // reserved for future task_meta
  "n_runs": 1,
  "metrics": {
    "task.completion_rate": 1.0,
    "tokens_per_task": 12345     // or null
  },
  "trace_id": "<trace id>",      // for traceability; aggregator at Phase 2 collapses by bucket
  "regressed": false             // always false in MVP
}
```

Aggregation across multiple traces (the `(scenario, task_type, time_window=daily)` bucketing) is a **read-time** concern in MVP — `tool_catalog.list_versions` returns the raw rows; Phase 2 `compare()` will roll them up.

### §1.3 Mid-session reload handling

Per design §4.2: events after a mid-session reload are flagged. In MVP indexer:

- The indexer attributes the trace to the **post-reload** fingerprint (the one after the last `atom.reload` event).
- A `mid_session_reload: true` field is added to the metrics row when any `atom.reload` is present in the trace.
- `compare()` (Phase 2) excludes `mid_session_reload=true` rows by default.

## §2. CLI shape

```
python -m agentm.core.catalog.indexer rebuild [options]

Options:
  --root PATH        Catalog root (default: .agentm/catalog)
  --observability PATH   Observability dir (default: .agentm/observability)

Behavior:
  1. Wipe metrics.jsonl files under <root>/atoms/<name>/<hash>/.
     (Decisions.jsonl is NOT touched — it's not derivable.)
  2. Wipe runs/ symlinks (then re-create).
  3. For each *.jsonl in observability/, call index_trace.
  4. Print a summary: n_traces, n_atoms_attributed, n_warnings.

Exit codes:
  0 — clean rebuild
  1 — at least one trace failed to parse (warning, not fatal; rest still indexed)
```

Atomic-append uses an `O_APPEND | O_CREAT` open + single `write` of `json.dumps(row) + "\n"`. POSIX guarantees atomicity for writes < `PIPE_BUF` (4 KiB), which is enough for a metrics row.

## §3. Wiring into `AgentSession.shutdown`

```python
async def shutdown(self) -> None:
    await self._bus.emit("session_shutdown", ...)  # existing
    if self._parent_bus is not None: ...           # existing
    self._bus.clear()                               # existing

    # NEW: post-shutdown indexer (best-effort).
    try:
        from agentm.core.catalog.indexer import index_trace
        # The trace path is conventional; observability writes to
        # <cwd>/.agentm/observability/<session_id>.jsonl.
        trace = Path(self._cwd) / ".agentm" / "observability" / f"{self._session_id}.jsonl"
        if trace.is_file():
            index_trace(trace)
    except Exception as exc:
        logger.warning("agentm catalog indexer post-shutdown failed: %r", exc)
```

**Layer note**: `harness/session.py` is constitution layer. Importing from `agentm.core.catalog` is fine (constitution → constitution). The import is lazy to avoid a startup cost when the catalog module isn't needed.

## §4. Test cases

| Test | Asserts | Scenario |
|---|---|---|
| `test_E5_rebuild_is_idempotent` | Run a session, capture metrics.jsonl content (post-shutdown). Wipe it. Run `rebuild`. Resulting content equals the original modulo `indexed_at` timestamps (compare with timestamps stripped). | E5 |
| `test_index_trace_attributes_to_all_loaded_atoms` | A trace with `session.fingerprint` listing 3 atoms produces 3 metric rows | (mechanism) |
| `test_index_trace_marks_mid_session_reload` | A trace with one `atom.reload` event causes `mid_session_reload: true` in the row | E8 |
| `test_index_trace_lazily_freezes_genesis_version` | A first-time atom (no `.agentm/catalog/atoms/<name>/` dir) gets one created with the correct hash after indexing | (genesis rule) |
| `test_index_trace_handles_missing_fingerprint_record` | A pre-fingerprint trace (just `session.start` and no `session.fingerprint`) is skipped with a warning, not a crash | (resilience) |
| `test_completion_rate_one_for_end_turn` | A trace ending with `agent_end.stop_reason="end_turn"` produces `task.completion_rate=1.0` | (mechanism) |
| `test_completion_rate_zero_for_budget_stop` | A trace ending with `agent_end.stop_reason="budget"` produces `task.completion_rate=0.0` | (mechanism) |
| `test_runs_symlink_created_idempotently` | Re-running indexer does not duplicate the symlink | (idempotence) |
| `test_cli_rebuild_returns_zero_on_clean_run` | `python -m agentm.core.catalog.indexer rebuild` exits 0 | (CLI shape) |

## §5. Acceptance Conditions

- [ ] `uv run pytest tests/unit/core/catalog/test_indexer.py -v` all green
- [ ] `uv run python -m agentm.core.catalog.indexer rebuild --help` shows usage text
- [ ] `uv run ruff check src/agentm/core/catalog/indexer.py src/agentm/harness/session.py` clean
- [ ] `uv run mypy src/agentm/core/catalog/` clean
- [ ] No autonomy-layer file modified
- [ ] No new third-party dependency

## §6. Acceptance scenarios covered

- **E5** (rebuild idempotence) — `test_E5_rebuild_is_idempotent`
- **E8** (mid-session reload marker) — `test_index_trace_marks_mid_session_reload`

## §7. Notes

- **No statistics, no CIs, no comparison logic** in MVP. The indexer is a recorder, not a decision-maker. Phase 2 layers `compare()` on top of `metrics.jsonl`.
- **No `decisions.jsonl` writes** — that file is not even created in MVP. `tool_catalog.list_versions` returns whatever's there (an empty list).
- **No guard-metrics flag** — `regressed` is hard-coded `false`. Phase 2 introduces the actual computation.
- **Performance budget**: indexing one session post-shutdown should take <50 ms for 25 atoms × 1 trace. The trace file may grow large with handler records enabled; we read it line-by-line and stop at `agent_end` — for very large traces we accept the longer indexer time as a non-blocking warning.
- **The indexer NEVER writes to `decisions.jsonl`**. That file is reserved for `propose_change` (Phase 2). Document at the top of `indexer.py` so a Phase 2 author doesn't accidentally entangle the two.
```

============ END FILE ============