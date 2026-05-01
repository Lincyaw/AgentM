# Plan: Self-Modifiable Architecture + Evolution Substrate — MVP

**Date**: 2026-05-01
**Status**: DRAFT
**Target designs**:
- [self-modifiable-architecture](../designs/self-modifiable-architecture.md)
- [evolution-substrate](../designs/evolution-substrate.md)

**Architectural context**:
- [pluggable-architecture](../designs/pluggable-architecture.md)
- [extension-as-scenario](../designs/extension-as-scenario.md) (esp. §11 single-file contract)
- [observability](../designs/observability.md)

**Reference codebase**: pi-mono at `/tmp/pi-analysis/pi-mono/packages/coding-agent/src/core/extensions/loader.ts:139-176, 196` for the `assertActive` pattern.

---

## 1. Requirements Restatement

Two sister designs have been approved. They are inseparable: self-mod without an evidence substrate degenerates to "react to install errors", and the substrate without safe transactional reload has no events to attribute. We plan **only the MVP slice from each**; Phase 2 (`compare`, `find_best`, `propose_change`, `decisions.jsonl`) and Phase 3 (replay cache, experiment-mode lock) are explicitly deferred.

The MVP delivers four integrated capabilities:

1. **Constitution boundary** — A `core-manifest.yaml` at the repo root declares which paths the agent may not self-modify. The validator gains `is_constitution_path()` and rejects agent-driven edits to those paths. The manifest is the closed list; everything outside it inside `extensions/builtin/`, `extensions/scenarios/`, `skills/`, `prompts/`, `.claude/`, or `.agentm/` (except catalog) is autonomy-layer.

2. **Versioned ExtensionAPI** — `ExtensionManifest` grows three fields with backward-compatible defaults: `api_version: int = 1`, `affects: tuple[str, ...] | dict[...] = ()`, `tier: int = 1`. The validator gains an `api_version` compatibility check (current=1, grace=1). The downgrade rule (tier 2 → tier 1 is human-only) is enforced syntactically by comparing against the prior manifest in the catalog.

3. **Transactional reload** — `reload_atom`, `freeze_current`, `list_atoms`, `is_constitution_path` land on `ExtensionAPI`. The flow validates new source, freezes the current version into `.agentm/catalog/`, atomically writes, re-loads, and rolls back on `install()` failure. Stale-context invalidation (`assert_active`) is ported from pi-mono so old `api`/`ctx` references trip with a precise error.

4. **Catalog substrate (read-only side)** — `.agentm/catalog/` directory layout is materialised; content-hash logic, `freeze_current()` writer, and a minimal indexer (post-session, attribution of `n_runs`, `task.completion_rate`, `cost_per_task`) all live in the constitution-layer module `agentm.core.catalog`. The `observability` atom adds the active-set fingerprint to its `session.start` header. A new tier-1 atom `tool_catalog` exposes only `list_versions`, `get_manifest`, `runs_for` to the agent.

The MVP is **read + freeze, no decide**: the agent can browse versions and see which traces ran under which fingerprint; it cannot yet ask `compare()` or `find_best()`. This shape is intentional — it locks the catalog contract before any decision-making logic depends on it.

### 1.1 Out of scope (deferred to Phase 2 / Phase 3)

| Deferred | Owner doc | Reason |
|---|---|---|
| `compare()` with confidence intervals | evolution-substrate §6.1 | Requires statistics layer (`scipy.stats` or in-house bootstrap) — a self-contained next step |
| `find_best()` with constraints | evolution-substrate §6.1 | Depends on `compare()` |
| `propose_change()` | evolution-substrate §6.1 + self-mod §7.2 | Depends on tier-2 enforcement and `decisions.jsonl` writer |
| `decisions.jsonl` writes | evolution-substrate §3.3 | Depends on `propose_change()` |
| Tier-2 deferred-approval gate | self-mod §7.2 | The `tier` field is **recorded** in the MVP manifest and frozen into the catalog; runtime enforcement (block reload pending approval) is Phase 2 |
| Goodhart guard metrics + `regressed` flag | evolution-substrate §7.3 | Requires baseline statistics |
| Replay cache + `experiment_mode_atom` lock | evolution-substrate §7.4 / §7.2 | Phase 3 |
| Scenario-level catalog (`.agentm/catalog/scenarios/`) | evolution-substrate §8 | Captured by fingerprint header and indexer (atoms only); scenario-version dirs deferred to Phase 2 |
| Dashboards / TUI | evolution-substrate §10 | Presenter, post-MVP |

---

## 2. Prerequisites

- Phase 2 (built-in atoms) and Phase 2.5 (legacy sweep) are landed (per `index.yaml` and recent commits).
- §11 single-file contract scaffolding (`extensions.discover`, `extensions.validate`) already exists and is tested.
- `observability` atom is landed and writes JSONL with `session.start` header. (Confirmed: `src/agentm/extensions/builtin/observability.py`.)
- `_INSTALLING_EXTENSION` ContextVar pattern exists in `harness/extension.py` and is used by observability for handler attribution. (Confirmed.)

No new third-party dependencies are required for the MVP. Hashing uses stdlib `hashlib`; YAML is already a project dependency.

---

## 3. Dependency Analysis (existing modules touched)

### Constitution-layer modules (modified)

| File | Change | Reason |
|---|---|---|
| `src/agentm/extensions/__init__.py` | Add 3 fields to `ExtensionManifest` (`api_version`, `affects`, `tier`) with backward-compatible defaults | self-mod §4.1 — manifest schema growth |
| `src/agentm/extensions/validate.py` | Add `is_constitution_path()` helper; add new validator check for `api_version` compatibility window; add tier-downgrade-vs-prior-manifest check; expose constitution-path API to atoms via `ExtensionAPI` | self-mod §3, §4.2, §7.3 |
| `src/agentm/extensions/discover.py` | None expected (the new manifest fields are read by the validator and the catalog freezer; discover keeps returning `BuiltinEntry`) | n/a |
| `src/agentm/harness/extension.py` | Add 4 new methods to `ExtensionAPI` Protocol and `_ExtensionAPIImpl` (`reload_atom`, `freeze_current`, `list_atoms`, `is_constitution_path`); add `assert_active`/stale-flag mechanism on `_ExtensionAPIImpl`; add per-extension API instance management (replace the single shared impl with per-atom instances so individual stale flags are meaningful — see §6 risk) | self-mod §6, §8 |
| `src/agentm/harness/session.py` | Wire `reload_atom` flow: validator call, snapshot via `freeze_current`, atomic write, install attempt, rollback on failure; emit `ExtensionReloadEvent`; track per-extension API instance map; clear handlers attributed to the reloaded atom and mark old API instance stale | self-mod §5.1 |
| `src/agentm/harness/events.py` | Add `ExtensionReloadEvent` (name, old_hash, new_hash, trigger, error?) + `AtomReloadEvent` for observability mid-session marker | self-mod §5.1, evolution-substrate §4.2 |
| `src/agentm/extensions/builtin/observability.py` | At `session.start` emission: compute active-set fingerprint and embed under `attributes.fingerprint`; subscribe to `extension_reload` and emit `atom.reload` records during the session | evolution-substrate §4.1, §4.2 |

### Constitution-layer modules (new)

| File | Purpose |
|---|---|
| `core-manifest.yaml` (repo root) | Closed list of constitution paths + `extension_api.current` / `grace` + `tier_2_atoms` |
| `src/agentm/core/catalog/__init__.py` | Public surface: `freeze_current`, `list_atoms`, `is_constitution_path` (re-export from `manifest`), `compute_atom_hash` |
| `src/agentm/core/catalog/manifest.py` | Parser + loader for `core-manifest.yaml`; `is_constitution_path(path) -> bool`; cached at module scope |
| `src/agentm/core/catalog/hashing.py` | `hash_atom_source(source: str) -> str` (stdlib `hashlib.sha256`, hex truncated to 12 chars to match design examples like `e5f6...`); helpers for hashing manifest payloads |
| `src/agentm/core/catalog/freeze.py` | `freeze_current(name, source, manifest) -> str`: writes `.agentm/catalog/atoms/<name>/<hash>/{source.py, manifest.yaml}`; updates the `current` symlink; idempotent (no rewrite when hash dir exists) |
| `src/agentm/core/catalog/indexer.py` | Post-session indexer: parses observability JSONL files, derives `n_runs`, `task.completion_rate`, `cost_per_task` (rough — token-count × pricing not in MVP, so cost is "n/a" unless the trace carries a cost record), atomic-appends to `metrics.jsonl`, symlinks runs; CLI `python -m agentm.core.catalog.indexer rebuild` re-derives from raw |
| `src/agentm/core/catalog/_layout.py` | Path constants for the `.agentm/catalog/` tree (single source of truth so freezer and indexer agree on layout) |

### Autonomy-layer modules (modified)

| File | Change |
|---|---|
| All ~25 atoms under `src/agentm/extensions/builtin/` | **Mechanical migration**: defaults `api_version=1`, `affects=()`, `tier=1` apply via the dataclass defaults — no atom file needs to change unless we want explicit tier-2 declarations. Tier-2 atoms (per `core-manifest.yaml`: `permission`, `cost_budget`, `tool_filter`, `llm_compaction`, `claude_agents`) get an explicit `tier=2` so the catalog records intent. We do **not** require `affects` declarations in the MVP — the indexer attributes universal counters that are unaware of `affects.primary` until Phase 2's `compare()`. |
| `src/agentm/extensions/scenarios/general_purpose.yaml` | Append `agentm.extensions.builtin.tool_catalog` entry (read-only browser) |

### Autonomy-layer modules (new)

| File | Purpose |
|---|---|
| `src/agentm/extensions/builtin/tool_catalog.py` | Tier-1 atom registering `list_versions`, `get_manifest`, `runs_for` tools; reaches into `agentm.core.catalog` for read-only data only |
| `tests/unit/core/catalog/{__init__.py, test_hashing.py, test_freeze.py, test_indexer.py, test_manifest.py}` | Catalog unit tests |
| `tests/unit/extensions/builtin/tool_catalog/{__init__.py, test_tool_catalog.py}` | Atom unit tests |
| `tests/unit/harness/test_reload.py` | Transactional reload + `assert_active` tests |
| `tests/unit/extensions/builtin/observability/test_fingerprint.py` | Fingerprint header tests (extend the existing observability test directory) |
| `tests/integration/test_self_mod_mvp.py` | End-to-end S1-S10 + E1-E10 acceptance tests (the subset reachable in MVP — see §6 acceptance map) |

---

## 4. Risk Register

| # | Risk | Severity | Likelihood | Mitigation / Position |
|---|------|----------|------------|-----------------------|
| R1 | **Per-extension API instance** — the current `_ExtensionAPIImpl` is one instance shared by all atoms. `assert_active` requires per-atom instances so a reload of atom X does not stale atom Y's references. The session needs a `dict[str, _ExtensionAPIImpl]` and `load_extension` must take the per-atom instance. | HIGH | HIGH | Position: this is unavoidable. Group `transactional-reload` includes the refactor. Without it the stale-flag is meaningless or globally destructive. The change is local — `_ExtensionAPIImpl` already holds no per-extension state besides identity, so a per-atom instance is cheap. Keep the *existing public* `ExtensionAPI` shape; the orchestrator owns the dispatch table and hands out a fresh instance for each `install()`. |
| R2 | **Layer purity violation**: `tool_catalog` (autonomy) reaching into `agentm.core.catalog` (constitution). The §11 import allow-list does NOT currently include `agentm.core.catalog`. | HIGH | CERTAIN | Add `agentm.core.catalog` to `_ALLOWED_PREFIXES` in `extensions/validate.py` as part of `core-manifest` task. The catalog package's public surface is intentionally narrow (`list_atoms`, `freeze_current`, `is_constitution_path`, `compute_atom_hash`, plus indexer-side helpers) — this is the stable contract autonomy may consume. Indexer internals stay private. |
| R3 | **Catalog layout is a contract** — once the MVP ships, the indexer (Phase 2 `compare()` consumer), the freezer (called from `reload_atom`), and `tool_catalog` (read API) all assume the same paths. Changing layout later breaks rebuild idempotence (E5). | HIGH | MEDIUM | Lock layout in `core/catalog/_layout.py` as path constants in the very first task (`catalog-storage`). Freezer and indexer import constants — neither hard-codes paths. The directory schema test (E5: rebuild yields byte-identical metrics) catches drift. |
| R4 | **Backward-compat MANIFEST migration**: defaults must permit ~25 existing atoms to load unchanged. Tier-2 atom list duplication risk: the `core-manifest.yaml` `tier_2_atoms` list and per-atom `tier=2` declarations could disagree. | MEDIUM | LOW | Position: per-atom `tier=2` is the source of truth (it ships with the source); `core-manifest.yaml::reload.tier_2_atoms` is informational. The validator emits a **warning** (not error) on disagreement and the MVP test asserts the lists match. Phase 2 will choose one canonical source when `propose_change` lands. |
| R5 | **Mid-session reload semantics + `_INSTALLING_EXTENSION` ContextVar**. The observability atom uses this ContextVar to attribute handlers. During reload, `_INSTALLING_EXTENSION` must be set to the reloaded module path so the new install's handlers attribute correctly; old handlers attributed to the same module path must be unsubscribed. | MEDIUM | MEDIUM | Reuse the existing `_INSTALLING_EXTENSION` ContextVar via `load_extension`. Handler unsubscription works through a side index: the session keeps `dict[str, list[Unsubscribe]]` (atom name → unsubscribes) populated via a wrapped `api.on`. On reload, call every Unsubscribe for that atom, drop from the table, then re-install. This same wrapper can set the per-atom `_stale` flag on the old `_ExtensionAPIImpl`. |
| R6 | **`freeze_current` race conditions** — concurrent reloads of the same atom could write to `<hash>/` simultaneously and corrupt the symlink. | LOW | LOW | `reload_atom` is intended to be invoked from the main asyncio loop; concurrent reloads of one atom are not a documented use case. We add a per-atom `asyncio.Lock` in the session and document the contract. Atomic write goes through `os.replace` (POSIX atomic). Symlink update is `os.symlink` to a temp name + `os.rename` (atomic on POSIX). |
| R7 | **Test infrastructure regression** — adding 3 new manifest fields could destabilise the `test_extension_contract.py` validator gate that all existing atoms must pass. | MEDIUM | MEDIUM | Tasks order: `core-manifest` → `manifest-schema` (single PR, mechanical, runs the contract test). Defaults preserve compat; the only risk is the new validator check (`api_version` compat) failing on an existing atom — it won't, because all existing atoms inherit `api_version=1` and core's `current=1`. |
| R8 | **Indexer running on autonomy thread** — design says indexer is constitution-owned and runs after each session ends. AgentM's `AgentSession.shutdown()` runs in the same event loop as autonomy code. | MEDIUM | LOW | Indexer runs as a sync function called from `shutdown()` after `bus.clear()` (post-shutdown — observability sink is closed; raw JSONL is on disk). It does not hold the loop for long (small read; small write). Failure does not propagate (logged-and-swallowed) — the user can rebuild via CLI later. |
| R9 | **Tier-2 atoms still auto-reload in MVP** — the `tier` field is recorded but enforcement is deferred. An agent reloading `permission` will succeed in the MVP. | MEDIUM (scope) | n/a (intentional) | Position: documented in the design's MVP scope. The reload path emits `ExtensionReloadEvent.tier=N` so future enforcement has the data point. We add a `WARNING` log when tier-2 atoms are reloaded to make the deferral visible during MVP usage. |
| R10 | **`session.start` is currently a direct sink write before any other extension is loaded**. The fingerprint must include all atoms, but at the time observability writes its anchor, no other extensions have called `install()` yet. | HIGH | CERTAIN | The fingerprint depends on the *full discovered set* and the *recipe-driven loaded set*. Two options: (a) defer fingerprint emission to `session_ready` (after every extension is installed); (b) compute the fingerprint from the recipe before installing any atom. Position: do (a) plus a stub anchor record: keep the existing immediate `session.start` write so observability still has its anchor (E.g. raw cwd/log_path), and emit a separate `session.fingerprint` record at `session_ready` carrying the active-set hash. The indexer uses `session.fingerprint`, not `session.start`, for attribution. This breaks the design wording of "first record" but preserves causal correctness. Document the deviation in `evolution-substrate.md` (one-line note) once landed. **This is the structural ambiguity the brief warned about.** Choosing (a) over (b) because (b) would require the observability atom to recompute the fingerprint based on recipe parsing (atoms-not-yet-installed), reaching into discovery / loader internals — that violates layer purity. (a) keeps observability a pure subscriber, at the cost of one extra record. |
| R11 | **Single task or two for `core-manifest` + `manifest-schema`?** They are interlocking: schema growth references `core-manifest.yaml::extension_api.current`, and `is_constitution_path` reads the manifest's `paths` list. | LOW (decision) | n/a | Position: **two tasks**, sequential. `core-manifest` lands first (manifest file + parser + `is_constitution_path` only — no atom changes). `manifest-schema` lands second (extends `ExtensionManifest`, runs validator's new check, optionally updates the 5 tier-2 atoms with explicit `tier=2`). Splitting limits blast radius — a contract-test break in `manifest-schema` does not block the `core-manifest` foundation. |
| R12 | **Indexer cost calculation requires LLM token-pricing data**, which AgentM does not currently surface. | LOW (scope) | n/a | Position: indexer attributes `n_runs` and `task.completion_rate` (from `agent_end.stop_reason`) in the MVP. Cost is recorded as `null` unless the trace carries a `llm.request.end` record with usage; if so, the indexer aggregates raw token counts (no pricing) into `tokens_per_task`. Phase 2 adds a guard-metrics file (`guard_metrics.yaml` per evolution-substrate §11) that turns tokens into dollars. |

---

## 5. Ordered Work Groups

The eight task groups below execute in three waves. Within a wave, listed groups are independent (parallelizable across implementer agents). Between waves, ordering is strict because each downstream group depends on the prior wave's contract.

### Wave 1 — Constitution foundation (sequential)

Wave 1 has two tasks; the second strictly depends on the first.

1. **`core-manifest`** — `core-manifest.yaml` + `core/catalog/manifest.py` parser + `is_constitution_path()` helper + validator allow-list extension for `agentm.core.catalog`. **Pre-req for everything.** Until the manifest exists, no other group can know what counts as a constitution path.
2. **`manifest-schema`** — `ExtensionManifest` grows `api_version`, `affects`, `tier`; validator gains `api_version` compatibility check; explicit `tier=2` declarations on the 5 named atoms; one mechanical patch landing all `MANIFEST = ExtensionManifest(...)` site updates that need them. Existing atoms that take defaults need no edit.

Wave 1 is sequential because the schema growth references the manifest's `extension_api.current` field. Splitting them lets the `core-manifest` task focus on YAML + parser without coupling to atom files.

### Wave 2 — Catalog primitives + reload mechanism (parallel)

After Wave 1 lands, three groups execute in parallel:

3. **`catalog-storage`** — `core/catalog/{_layout, hashing, freeze}.py` plus directory layout tests. **Locks the catalog layout contract** that the indexer (group 6) and `tool_catalog` (group 7) consume. No autonomy-layer code yet.
4. **`transactional-reload`** — `reload_atom`, `freeze_current` ExtensionAPI methods, snapshot/rollback flow in `harness/session.py`, per-atom `_ExtensionAPIImpl` refactor, `assert_active` stale-flag mechanism, `ExtensionReloadEvent` payload type. Imports `freeze_current` from group 3 — but the *signature* of `freeze_current` is locked at the start of group 3, so group 4 can stub against the signature and run its tests with a fake freezer until group 3 lands. **Contract-first.**
5. **`observability-fingerprint`** — `observability` atom emits `session.fingerprint` record at `session_ready` (per R10) and `atom.reload` records on `extension_reload`. Updates the existing observability test suite. Imports the fingerprint-computation helper from `agentm.core.catalog` (added in group 3).

Group 4 strictly depends on group 3's signature of `freeze_current` and on the `manifest.is_constitution_path` from Wave 1. Group 5 strictly depends on group 4's `ExtensionReloadEvent` payload but can develop against a stub event class until group 4 lands the real one.

**Coordination point**: at the start of Wave 2, `catalog-storage` author writes the `freeze_current` signature stub and the `_layout.py` constants in a single commit, *before* their tests are filled in. Groups 4 and 5 import from that stub. This pattern matches Phase 2.0b's contract-stub approach.

### Wave 3 — Indexer + agent surface + integration (parallel after Wave 2)

After Wave 2 lands, three groups execute in parallel:

6. **`indexer-mvp`** — `core/catalog/indexer.py` + CLI rebuild command. Wired into `AgentSession.shutdown()`. Reads observability JSONL, derives basic counters, atomic-appends to `<atom>/<hash>/metrics.jsonl`, symlinks `runs/`. Idempotent (E5).
7. **`tool-catalog-atom`** — New tier-1 atom `extensions/builtin/tool_catalog.py` registering `list_versions`, `get_manifest`, `runs_for`. Updates `general_purpose.yaml`. Read-only — no `propose_change` in MVP.
8. **`acceptance-tests`** — Integration test file covering S1-S10 (where reachable) and E1-E10 (where reachable). Some scenarios test Phase 2 features (S2, S9, E1, E2, E3, E6, E7, E10) and are documented as `pytest.skip("Phase 2: requires <feature>")` placeholders so the harness for them lands in MVP and the skip count is the regression marker for Phase 2 entry.

Group 6 has no code dependency on group 7; `tool_catalog` queries can succeed against a freshly-seeded catalog (one atom, one hash, no metrics yet) — `runs_for` returns `[]`, `list_versions` returns one entry. So 6 and 7 can land in either order.

Group 8 strictly depends on 6 and 7 (it asserts on their behavior); it is the gate for declaring MVP done.

### Ordering rationale at a glance

```
Wave 1 (sequential):    core-manifest → manifest-schema
Wave 2 (parallel):      catalog-storage  ┐
                        transactional-reload  ┤  (after Wave 1)
                        observability-fingerprint ┘
Wave 3 (parallel):      indexer-mvp  ┐
                        tool-catalog-atom  ┤  (after Wave 2)
                        acceptance-tests   ┘  (gates MVP done)
```

---

## 6. Acceptance Criteria — Scenario → Test mapping

Each scenario from the two designs is mapped to a concrete test target. Scenarios marked **(Phase 2)** ship in the test file as `pytest.skip("Phase 2: requires <feature>")` so the test infrastructure is in place when the feature lands.

### Self-modifiable architecture §9 (S1-S10)

| # | Scenario | MVP test target | Status |
|---|----------|-----------------|--------|
| S1 | Agent reloads `tool_read` with new arg; next turn uses new tool | `tests/integration/test_self_mod_mvp.py::test_S1_reload_tool_atom_takes_effect_next_turn` | MVP |
| S2 | Reload of tier-2 atom `permission` is deferred via `propose_change` | `tests/integration/test_self_mod_mvp.py::test_S2_tier2_reload_deferred_pending_approval` | **(Phase 2)** — skip |
| S3 | `tool_edit` on `core/kernel/loop.py` is rejected | `tests/integration/test_self_mod_mvp.py::test_S3_tool_edit_blocked_on_constitution_path` (assertion: `is_constitution_path` returns `True`) | MVP |
| S4 | Atom with syntax error rejected; no file written | `tests/integration/test_self_mod_mvp.py::test_S4_syntax_error_rejected_no_write` | MVP |
| S5 | Atom passes validator but `install()` raises → rollback | `tests/integration/test_self_mod_mvp.py::test_S5_install_failure_rolls_back` | MVP |
| S6 | Mid-turn reload: deferred coroutine on stale `api` raises `ExtensionStaleError` | `tests/unit/harness/test_reload.py::test_S6_assert_active_raises_after_reload` | MVP |
| S7 | Atom with `api_version=99` rejected | `tests/unit/extensions/test_extension_contract.py::test_S7_api_version_too_new_rejected` | MVP |
| S8 | `tool_edit` on `harness/extension.py` rejected | Same gate as S3 — covered by `test_S3_*` parameterized over `(constitution_path, expected=True)` cases | MVP |
| S9 | Tier downgrade rejected | `tests/integration/test_self_mod_mvp.py::test_S9_tier_downgrade_blocked_for_agent` | **(Phase 2)** — skip; requires propose_change author identification |
| S10 | After human edits `core-manifest.yaml` to remove a path, agent edit allowed | `tests/unit/core/catalog/test_manifest.py::test_S10_manifest_change_moves_constitution_boundary` (manifest reload + path check) | MVP |

### Evolution substrate §9 (E1-E10)

| # | Scenario | MVP test target | Status |
|---|----------|-----------------|--------|
| E1 | `compare()` returns numbers with CIs after enough runs | `compare_id` test target | **(Phase 2)** — skip |
| E2 | `compare()` returns `inconclusive: true` on small N | **(Phase 2)** — skip |
| E3 | `find_best` skips regressed candidate | **(Phase 2)** — skip |
| E4 | `tool_edit` on `.agentm/catalog/atoms/.../metrics.jsonl` rejected | `tests/integration/test_self_mod_mvp.py::test_E4_catalog_path_blocked` (extends S3 parameterization) | MVP |
| E5 | `python -m agentm.core.catalog.indexer rebuild` produces byte-identical `metrics.jsonl` | `tests/unit/core/catalog/test_indexer.py::test_E5_rebuild_is_idempotent` | MVP |
| E6 | Experiment-mode lock blocks parallel atom changes | **(Phase 2)** — skip |
| E7 | Reactivating regressed version blocked | **(Phase 2)** — skip |
| E8 | Mid-session reload triggers `atom.reload` event; downstream events flagged | `tests/integration/test_self_mod_mvp.py::test_E8_mid_session_reload_emits_marker` | MVP |
| E9 | Scenario-level `compare()` | **(Phase 2)** — skip; scenario-version dirs deferred |
| E10 | `find_best` with no winners returns `None` | **(Phase 2)** — skip |

### Additional MVP-only acceptance points (not in either §9)

| # | Acceptance | Test |
|---|------------|------|
| M1 | `freeze_current` is idempotent — second call with same source produces no rewrite | `tests/unit/core/catalog/test_freeze.py::test_idempotent_no_rewrite_when_hash_exists` |
| M2 | Active-set fingerprint is included in `session.fingerprint` record (per R10 deviation) | `tests/unit/extensions/builtin/observability/test_fingerprint.py::test_fingerprint_record_includes_all_loaded_atoms` |
| M3 | `tool_catalog.list_versions("tool_read")` returns at least the current version after first session | `tests/unit/extensions/builtin/tool_catalog/test_tool_catalog.py::test_list_versions_includes_current_after_seed` |
| M4 | Per-extension `_ExtensionAPIImpl` is created — atoms get distinct instances | `tests/unit/harness/test_reload.py::test_per_atom_api_instances_distinct` |
| M5 | `tool_catalog`'s manifest passes the §11 contract test | `tests/unit/extensions/test_extension_contract.py` (vacuous — runs as part of the existing gate) |

---

## 7. Definition of Done

The MVP is complete when **all** of the following hold:

1. **All MVP-tagged tests pass**: `uv run pytest -q` is green; the Phase-2-skipped tests appear as `s` (skipped) with an explicit reason.
2. **Code quality gates green** for every changed file:
   - `uv run ruff check src/agentm/`
   - `uv run mypy src/agentm/core/catalog/ src/agentm/extensions/ src/agentm/harness/`
3. **Layer purity**: `grep -rE 'from agentm\.harness\.session ' src/agentm/extensions/` returns empty; `grep -rE 'from agentm\.core\.catalog' src/agentm/extensions/builtin/` returns only `tool_catalog.py`.
4. **§11 contract gate green**: `uv run pytest tests/unit/extensions/test_extension_contract.py -q` passes (every atom's manifest validates including the new fields).
5. **Catalog rebuild idempotence**: after running a session and `python -m agentm.core.catalog.indexer rebuild`, `metrics.jsonl` content matches modulo timestamp ordering (E5).
6. **Observability schema**: a session.fingerprint record is present in every JSONL trace, listing every loaded atom by `<name>@<hash>`.
7. **`core-manifest.yaml` is the source of truth**: editing it moves the constitution boundary observably (S10 test green).
8. **Reviewer approval** on each task's PR per the standard `architect → planner → tdd → implementer → reviewer` flow.
9. **Index updated**: `index.yaml` carries the `plans:` and `tasks:` references for both `self_modifiable_architecture` and `evolution_substrate` concepts.

---

## 8. Tasks

In execution order (waves laid out in §5):

### Wave 1
- [core-manifest](../tasks/2026-05-01-core-manifest.md) — Size: M
- [manifest-schema](../tasks/2026-05-01-manifest-schema.md) — Size: M

### Wave 2 (parallel)
- [catalog-storage](../tasks/2026-05-01-catalog-storage.md) — Size: M
- [transactional-reload](../tasks/2026-05-01-transactional-reload.md) — Size: M (largest of the wave; consider splitting the per-atom-API refactor if it grows)
- [observability-fingerprint](../tasks/2026-05-01-observability-fingerprint.md) — Size: S

### Wave 3 (parallel)
- [indexer-mvp](../tasks/2026-05-01-indexer-mvp.md) — Size: M
- [tool-catalog-atom](../tasks/2026-05-01-tool-catalog-atom.md) — Size: S
- [acceptance-tests](../tasks/2026-05-01-acceptance-tests.md) — Size: M

---

## 9. Index.yaml updates required

After this plan is approved, `index.yaml` is amended:

```yaml
self_modifiable_architecture:
  plans:
    - "plans/2026-05-01-self-mod-mvp.md"
  tasks:
    - "tasks/2026-05-01-core-manifest.md"
    - "tasks/2026-05-01-manifest-schema.md"
    - "tasks/2026-05-01-transactional-reload.md"
    - "tasks/2026-05-01-acceptance-tests.md"

evolution_substrate:
  plans:
    - "plans/2026-05-01-self-mod-mvp.md"
  tasks:
    - "tasks/2026-05-01-catalog-storage.md"
    - "tasks/2026-05-01-observability-fingerprint.md"
    - "tasks/2026-05-01-indexer-mvp.md"
    - "tasks/2026-05-01-tool-catalog-atom.md"
    - "tasks/2026-05-01-acceptance-tests.md"
```

The `acceptance-tests` task is shared (it covers scenarios from both designs).
```

============ END FILE ============