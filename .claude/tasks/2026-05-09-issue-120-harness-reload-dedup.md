# Issue #120 — Harness reload state preservation + auto-discovery dedup

Sub-issue of #73. Closes the last two follow-ups (B5 and B15) from the
atom-rework infra refactor.

## B5 — transactional reload snapshot

`src/agentm/harness/atom_reloader.py` previously claimed the reload was
"transactional" but rolled back by re-running `_activate_atom_install`
on the original atom. If that re-install itself raised, the recovery
path only reseated `_loaded_by_module` / `_loaded_by_name` / `apis` /
`sys.modules`, leaving the bus subscription map, `_handlers_by_atom`,
`_registrations_by_atom`, `owners_by_kind`, and the registries (tools,
commands, providers, renderers) half-new / half-old.

The fix introduces a single immutable pre-reload `_ReloadSnapshot`
captured once at the start of `reload_atom_async`. The contract is:

1. `_capture_snapshot(module_path)` — defensive shallow copy of every
   reloader-mutable structure plus the `sys.modules` entry plus a
   per-channel snapshot of bus subscriptions.
2. `apply` (`_activate_atom_install`) is free to mutate.
3. On apply failure, re-install the original atom in place (cheap
   path — works in the common case where `install()` of the previous
   source still succeeds).
4. On rollback failure, call `_restore_from_snapshot(snapshot)` — a
   pure write-back of the snapshot's pre-copied data into the live
   structures. Because the snapshot already holds private copies, the
   restore path cannot half-fail.

A new test
`test_reload_double_failure_preserves_bus_subscriptions_and_registrations`
makes both apply and rollback raise, then asserts handler tracking,
registration tracking, owners-by-kind, and the bus subscription list
are all bit-identical to the pre-reload state, and that emitting on
the channel hits the original handler exactly once (proves no
post-apply orphan handlers leaked through).

## B15 — auto-discovery dedup

`src/agentm/harness/session_factory.py:_resolve_extensions` had three
copy-pasted calls to `_iter_auto_discovered_atoms` (builtin, contrib,
user). Folded into a single call to a new
`collect_auto_discovered_atoms(bus, sources)` helper that takes an
iterable of `AtomSource(label, discover, skip_label)`. The original
`_iter_auto_discovered_atoms` is preserved as a thin shim for
backward compatibility but is no longer called from the factory.

## Verification

- `uv run pytest tests/unit/harness/test_transactional_reload.py` — 16/16 pass
- `uv run pytest --tb=short -q` — 186/187 (1 pre-existing failure in
  `tests/unit/llm/test_retry_policy.py::test_openai_stream_fn_retries_typed_rate_limit`
  that also fails on origin/main; unrelated to this change)
- `uv run ruff check src/` — clean
- `uv run mypy src/` — clean (no new `# type: ignore`)
