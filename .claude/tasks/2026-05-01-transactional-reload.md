# Task: transactional-reload — reload_atom + assert_active + new ExtensionAPI methods

**Date**: 2026-05-01
**Status**: PENDING
**Plan**: [self-mod-mvp](../plans/2026-05-01-self-mod-mvp.md)
**Design**: [self-modifiable-architecture](../designs/self-modifiable-architecture.md) §5, §6, §8
**Assignee**: implementer
**Wave**: 2 (parallel)
**Size**: M (largest of the wave)
**Depends on**: [core-manifest](2026-05-01-core-manifest.md), [manifest-schema](2026-05-01-manifest-schema.md), [catalog-storage](2026-05-01-catalog-storage.md) signature

## Objective

Land the four new `ExtensionAPI` methods (`reload_atom`, `freeze_current`, `list_atoms`, `is_constitution_path`), the transactional reload pipeline in `harness/session.py`, and the `assert_active()` stale-context guard. Refactor the orchestrator to maintain per-atom `_ExtensionAPIImpl` instances so individual stale flags are meaningful.

This is the single largest piece of constitution-layer code in the MVP. Internal split note: if the per-atom-API refactor (§3 below) bloats the PR, split into `transactional-reload-api-refactor` (the per-atom plumbing) + `transactional-reload-pipeline` (the actual `reload_atom` flow). Reviewer makes the call after the first draft.

## Inputs to read

- `self-modifiable-architecture.md` §5.1 (the flow), §5.3 (rollback details), §6 (assert_active mechanics)
- pi-mono `loader.ts:139-176, 196` (assertActive pattern, error message style)
- `src/agentm/harness/extension.py` — current `_ExtensionAPIImpl` shape, `_INSTALLING_EXTENSION` ContextVar, `load_extension`
- `src/agentm/harness/session.py` — current install loop in `AgentSession.create`
- `src/agentm/core/catalog/__init__.py` (signatures landed in `catalog-storage`)

## Outputs

### New files

| Path | Purpose |
|---|---|
| `tests/unit/harness/test_reload.py` | Tests for the reload pipeline + per-atom API + assert_active |

### Modified files

| Path | Change |
|---|---|
| `src/agentm/harness/extension.py` | Add `assert_active` infrastructure to `_ExtensionAPIImpl`; add the four new method **signatures** to the `ExtensionAPI` Protocol; add a constructor argument `_owner_name: str` so the impl knows which atom it belongs to. The impl's *concrete implementations* of the four new methods delegate to a `_SessionGateway` callable injected at construction (so the impl stays free of `harness/session.py` imports). |
| `src/agentm/harness/session.py` | (a) Maintain `dict[str, _ExtensionAPIImpl]` keyed by atom name. (b) Maintain `dict[str, list[Unsubscribe]]` for handler unsubscription. (c) Wrap `api.on` for each atom so each registration's Unsubscribe is recorded under the atom's name. (d) Implement `reload_atom`'s flow per design §5.1. (e) Implement the `_SessionGateway` callbacks the impls dispatch into. (f) Emit `ExtensionReloadEvent`. |
| `src/agentm/harness/events.py` | Add `ExtensionReloadEvent` dataclass with fields: `name`, `old_hash` (str or None), `new_hash` (str), `trigger: Literal["agent","human","propose_change_approved"]`, `tier: int`, `error: str | None = None`. |

## §1. The four new ExtensionAPI methods

```python
# In ExtensionAPI Protocol:

def reload_atom(
    self,
    name: str,
    new_source: str,
    *,
    agent_initiated: bool = True,
    rationale: str | None = None,
) -> ReloadResult: ...

def freeze_current(self, name: str) -> str: ...      # returns content hash

def list_atoms(self) -> list[AtomInfo]: ...          # name, current_hash, tier, api_version

def is_constitution_path(self, path: str) -> bool: ...
```

`ReloadResult` is a frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class ReloadResult:
    ok: bool
    name: str
    old_hash: str | None
    new_hash: str | None
    error: str | None = None
    rolled_back: bool = False

@dataclass(frozen=True, slots=True)
class AtomInfo:
    name: str
    current_hash: str | None
    tier: int
    api_version: int
```

Both live in `harness/extension.py` next to the Protocol.

## §2. assert_active mechanism

`_ExtensionAPIImpl` gains:

```python
class _ExtensionAPIImpl:
    def __init__(self, *, _owner_name: str, ...):
        self._owner_name = _owner_name
        self._stale = False

    def _assert_active(self) -> None:
        if self._stale:
            raise ExtensionStaleError(
                f"Extension {self._owner_name!r} was reloaded; this api/ctx "
                f"reference is stale. Re-acquire via the new install() call. "
                f"To exit gracefully on reload, catch ExtensionStaleError "
                f"around long-running operations that capture api or ctx."
            )
```

Every public method calls `self._assert_active()` before delegating. Call sites: `on`, `register_*`, `send_user_message`, the four new methods, **and** every property accessor (`cwd`, `tools`, `session`, `model`, `provider`, `events`). The `events` property is the most subtle — handlers captured `api.events` and might call `bus.emit` later; staleness on the bus is a no-op (bus is shared), so we permit `events` access even when stale, but we mark it with a docstring warning.

`ExtensionStaleError` is a new class in `harness/extension.py`:

```python
class ExtensionStaleError(RuntimeError):
    """Raised when a stale ExtensionAPI reference is used after reload."""
```

## §3. Per-atom API instance refactor

Today `AgentSession.create` builds **one** `_ExtensionAPIImpl` and passes it to every `load_extension` call. After this refactor:

- A new `_ExtensionAPIImpl` is built **per atom load** with `_owner_name=module_path` (or the manifest name).
- The session keeps `_apis: dict[str, _ExtensionAPIImpl]` so reload can locate and stale the prior instance.
- The provider extension also gets its own instance — symmetry over special-casing.

This is structural enough that the change MUST come with these tests:

| Test (in `test_reload.py`) | Asserts |
|---|---|
| `test_M4_per_atom_api_instances_distinct` | After loading two atoms, their captured `api` references are different objects |
| `test_per_atom_api_owner_name_set` | Each atom's `api._owner_name` matches its module path |

## §4. Handler unsubscription mechanism

Add to session:

```python
self._handlers_by_atom: dict[str, list[Unsubscribe]] = {}
```

Wrap each atom's `api.on` so the Unsubscribe returned by `EventBus.on` is recorded:

```python
# Wrap is applied around the per-atom api before passing to load_extension
def _wrap_on(api, atom_name):
    original = api.on
    def tracked(channel, handler):
        unsub = original(channel, handler)
        self._handlers_by_atom.setdefault(atom_name, []).append(unsub)
        return unsub
    api.on = tracked   # method assignment, allowed because impl is plain class
```

Note: this **interacts** with `observability.py`'s own `api.on` wrap. Order matters: the session's wrap must happen *before* observability's install (so observability sees the wrapped `on`); observability's wrap operates on top, and its `_HANDLER_OWNER_ATTR` continues to attribute correctly because `_INSTALLING_EXTENSION` is set during install. Confirmed by reading observability source — its wrap sets `setattr(wrapped, _HANDLER_OWNER_ATTR, ext)` and `original_on(channel, wrapped)`. The session's wrapper is the "outer" wrap; observability's calls go through it. Add a test:

| Test | Asserts |
|---|---|
| `test_observability_wrap_composes_with_session_wrap` | After loading observability + a tier-1 atom, reloading the tier-1 atom unsubscribes its handler AND observability still receives the same atom's events under the new install |

## §5. The `reload_atom` pipeline

Implement per design §5.1:

1. `is_constitution_path(target_file_for(name))` — reject early if the atom file is constitution.
2. Discover validation: write to a tmp file, point a fresh `ImportError`-safe `discover_one(name, source)` helper at it, run validator's checks (call into existing `validate.validate_builtin`-equivalent narrowed to the new module).
3. (MVP scope, R9): if `tier == 2`, log `WARNING` but proceed. Phase 2 will add the deferral.
4. Compute new hash via `compute_atom_hash(new_source)`.
5. Call `freeze_current(name, current_source, current_manifest)` to capture the *prior* state. This returns the **prior** hash for rollback.
6. `os.replace(tmp_file, atom_path)` — atomic write to the autonomy-layer atom file.
7. Try to install:
   - Stale the prior `_ExtensionAPIImpl` (set `_stale=True`).
   - Call every Unsubscribe in `_handlers_by_atom[name]`.
   - Build a fresh `_ExtensionAPIImpl(_owner_name=name)`.
   - Set `_INSTALLING_EXTENSION` token; call `load_extension(module_path, new_api, config)`.
8. On `Exception`: rollback — `os.replace(snapshot_source_path, atom_path)`, re-`load_extension` on the restored source. Log to catalog (post-MVP, just emits an event for now). Return `ReloadResult.ok=False, rolled_back=True, error=str(e)`.
9. On success: emit `ExtensionReloadEvent(name, old_hash=prior_hash, new_hash=new_hash, trigger=..., tier=manifest.tier)` on the bus and return `ReloadResult.ok=True`.

The "discovery cache invalidation" part is subtle: `discover_builtin` is memoized at process scope. After a successful reload, call `agentm.extensions.discover.reset_cache()` and re-run discovery so `list_atoms()` shows the new hash. This is mentioned in `discover.py:reset_cache` and is exactly the use case it was built for.

## §6. Test cases

In `tests/unit/harness/test_reload.py`:

| Test | Asserts | Scenario |
|---|---|---|
| `test_S1_reload_tool_atom_takes_effect_next_turn` | After reload of `tool_read`, the registered tool has the new behavior | S1 |
| `test_S4_syntax_error_rejected_no_write` | Submitting source with `def install(api):` (1-arg) is rejected; original file unchanged | S4 |
| `test_S5_install_failure_rolls_back` | An atom whose `install` raises causes `reload_atom` to restore the prior file content; `ReloadResult.rolled_back=True` | S5 |
| `test_S6_assert_active_raises_after_reload` | Capture `api` in atom A's `install`; reload A; later access on the captured `api` raises `ExtensionStaleError` | S6 |
| `test_M4_per_atom_api_instances_distinct` | Two atoms get different `_ExtensionAPIImpl` instances | M4 |
| `test_reload_emits_extension_reload_event` | `ExtensionReloadEvent` lands on the bus after a successful reload | E8 contributor |
| `test_reload_path_check_rejects_constitution` | `reload_atom("kernel_loop", ...)` (a faux name pointing at a constitution path) rejects without writing | S3, S8 contributor |
| `test_reload_invalidates_old_handlers` | After reload, old handlers attributed to the atom are unsubscribed; new handlers are subscribed | (mechanism) |

## §7. Acceptance Conditions

- [ ] `uv run pytest tests/unit/harness/test_reload.py -v` all green
- [ ] `uv run pytest tests/unit/extensions/ -q` still green (existing atoms still load with per-atom APIs)
- [ ] `uv run pytest tests/unit/extensions/builtin/observability/ -q` still green (handler attribution still works)
- [ ] `uv run ruff check src/agentm/harness/` clean
- [ ] `uv run mypy src/agentm/harness/` clean
- [ ] No constitution-layer file imports `agentm.extensions.builtin.*`
- [ ] `harness/extension.py` does not import `agentm.harness.session` (avoid circularity — `_SessionGateway` is the only injection point)

## §8. Acceptance scenarios covered

- **S1**, **S4**, **S5**, **S6** — direct
- **S3**, **S8** — partial (the constitution-path block; integration test in `acceptance-tests` exercises through `tool_edit`)
- **M4** — direct
- **E8** — partial (the event is emitted; observability records the marker in `observability-fingerprint`)

## §9. Notes

- **Tier-2 enforcement is deferred (R9)**: in MVP, tier-2 atoms reload normally. We log a `WARNING` so the deferral is visible. The S2 test in `acceptance-tests` is `pytest.skip("Phase 2: tier-2 deferral").
- **Rollback re-loads the snapshot**: per design §5.3, the rollback path itself runs `load_extension`. If *that* also fails, mark the atom unavailable (drop from `_apis`) and emit a second `ExtensionReloadEvent` with `error="rollback_failure"`. The session continues; the atom is gone. Add a test: `test_double_failure_marks_atom_unavailable`.
- **Symbol shape for `_owner_name`**: we use the **module path** (`agentm.extensions.builtin.tool_read`) not the manifest `name` (`tool_read`), because that is what `_INSTALLING_EXTENSION` already stores and what observability records. Consistency wins.
- **No `propose_change` in MVP**: the API is `reload_atom` with `agent_initiated: bool`. Phase 2 adds `propose_change` which routes to `reload_atom`. Document this in the docstring.
- **No `decisions.jsonl` write**: the reload event is emitted to the bus; nothing persists it in MVP. Phase 2's `propose_change` path will append to `decisions.jsonl`.
```

============ END FILE ============