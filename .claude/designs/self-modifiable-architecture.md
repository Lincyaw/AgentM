# Design: Self-Modifiable Architecture

**Status**: PROPOSED
**Created**: 2026-05-01
**Builds on**: [pluggable-architecture.md](pluggable-architecture.md), [extension-as-scenario.md](extension-as-scenario.md), [observability.md](observability.md)
**Sister doc**: [evolution-substrate.md](evolution-substrate.md)
**Reference codebase**: `badlogic/pi-mono` at `/tmp/pi-analysis/pi-mono/packages/coding-agent/`

---

## 1. First Principle

> **Core is what an agent CANNOT safely modify. Everything else is self-modifiable, with safety provided by the validator + transactional reload.**

`pluggable-architecture.md` drew the boundary by "mechanism vs policy". This doc sharpens it: the boundary is whether an erroneous edit by an autonomous agent can brick the system. If yes → constitution. If no → autonomy.

This is operational, not philosophical: it tells us where to put a feature by asking "if a self-modifying agent rewrites this file wrongly, can it still recover?"

---

## 2. Three Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  Constitution layer                                                  │
│    core/kernel · operations · validator · reload primitive · llm     │
│    harness/{session, extension, events, session_manager, ...}       │
│    extensions/{loader, discover, validate}                           │
│    cli (presenter startup contract)                                  │
│    .agentm/catalog/  (write-protected even from agent)               │
│    core-manifest.yaml (self-referential lock)                        │
│    >>> only humans + PR review modify <<<                            │
├──────────────────────────────────────────────────────────────────────┤
│  Evolution substrate (see evolution-substrate.md)                    │
│    catalog: versioned (atom, scenario) × observation × decisions     │
│    indexer (constitution-owned)                                      │
│    tool_catalog (read API to agent; propose_change is the only write)│
├──────────────────────────────────────────────────────────────────────┤
│  Autonomy layer                                                      │
│    extensions/builtin/<atom>.py (tier 1 free / tier 2 review)        │
│    extensions/scenarios/<name>.yaml                                  │
│    skills/, prompts/, settings                                       │
│    >>> agent self-edits + reloads <<<                                │
├──────────────────────────────────────────────────────────────────────┤
│  Discovery layer                                                     │
│    filesystem scan · reload() · assert_active() · stale invalidation │
│    (mechanism is in constitution; this is how autonomy edits land)   │
└──────────────────────────────────────────────────────────────────────┘
```

Dependency rule unchanged from `pluggable-architecture.md`: arrows point downward only. The new ordering inserts the evolution substrate between constitution and autonomy — the substrate is constitution-owned (write side) but agent-readable (query side).

---

## 3. The `core-manifest.yaml`

Single source of truth for the constitution boundary. Lives at `core-manifest.yaml` in repo root. Loaded by the validator at startup; CI also enforces it on PRs.

```yaml
# core-manifest.yaml
version: 1

constitution:
  paths:
    # The kernel — mechanism, must not change without human review
    - src/agentm/core/kernel/**
    - src/agentm/core/operations.py
    - src/agentm/core/path_utils.py
    - src/agentm/core/text_truncate.py
    - src/agentm/core/edit_diff.py
    - src/agentm/core/frontmatter.py
    - src/agentm/core/skills.py
    - src/agentm/core/prompt_templates.py
    - src/agentm/core/compaction/**
    # Provider boundary
    - src/agentm/llm/**
    - src/agentm/ai/**
    # Harness orchestrator
    - src/agentm/harness/session.py
    - src/agentm/harness/extension.py
    - src/agentm/harness/events.py
    - src/agentm/harness/session_manager.py
    - src/agentm/harness/resource_loader.py
    - src/agentm/harness/session_runtime.py
    - src/agentm/harness/session_services.py
    - src/agentm/harness/session_cwd.py
    # Extension subsystem mechanism
    - src/agentm/extensions/loader.py
    - src/agentm/extensions/discover.py
    - src/agentm/extensions/validate.py
    # Presenter startup contract
    - src/agentm/cli.py
    # Catalog data — write-protected against agent edits
    - .agentm/catalog/**
    # The manifest itself — self-referential lock
    - core-manifest.yaml

extension_api:
  current: 1
  semver_rules:
    major: "Breaking change to ExtensionAPI signatures, event payloads, or hook semantics"
    minor: "New event/method added; existing surface backward compatible"
    patch: "Implementation only; no API change"
  deprecation:
    # An atom written for api_version=N is loadable on N..N+1; rejected on N+2.
    grace: 1

reload:
  # Atoms in tier_2 require explicit `propose_change` decision to reload.
  tier_2_atoms:
    - permission
    - cost_budget
    - tool_filter
    - llm_compaction
    - claude_agents
```

The constitution list is **closed**: anything not on the list and inside `src/agentm/extensions/builtin/`, `src/agentm/extensions/scenarios/`, `skills/`, `prompts/`, `.claude/`, or `.agentm/` (except catalog) defaults to autonomy-layer. Outside-tree paths are out of scope.

The validator (constitution-layer code at `extensions/validate.py`) gains a new check: `is_constitution_path(path) → bool`. Self-mod APIs route through this check.

---

## 4. Versioned ExtensionAPI

### 4.1 Atom manifest grows two fields

The §11 manifest in `pluggable-architecture.md` adds:

```python
# extensions/builtin/<name>.py
MANIFEST = ExtensionManifest(
    name="tool_read",
    api_version=1,                                    # NEW: required
    affects=["read.success_rate", "context.tokens"], # NEW: declared metric surface
    tier=1,                                           # NEW: 1=free, 2=review (defaults to 1)
    description="Read tool with truncation.",
    # ...
)
```

`api_version` is the integer ABI the atom expects. `affects` is consumed by the catalog (see sister doc). `tier` is consumed by the reload pipeline.

### 4.2 Validator enforces version compatibility

| Atom `api_version` | Core `extension_api.current` | Result |
|---|---|---|
| `> current` | — | Reject (atom written for future API) |
| `current` | — | Accept |
| `current - 1` | (within `grace`) | Accept with deprecation warning |
| `< current - grace` | — | Reject (atom too old; force migration) |

The migration path for breaking ABI changes is explicit: bump `extension_api.current`, write a one-page migration note in `.claude/designs/migrations/<api-version>.md`, set grace so the previous version still loads for one cycle. Agent-authored atoms get a chance to be regenerated before they break.

### 4.3 Why integer, not semver?

Extensions inside one repo. We control the ABI. Integer monotonic version + grace window is simpler than semver and forces explicit migration thinking. This may change if extensions become a multi-repo ecosystem.

---

## 5. Transactional Reload

The mechanism that lets autonomy-layer edits take effect safely. Sits in `harness/session.py` (constitution).

### 5.1 Flow

```
agent calls api.reload_atom(name, new_source)
  │
  ├─ is_constitution_path(name)?           ── YES → reject "constitution layer"
  │   NO
  ├─ validator.check(new_source):
  │     §11 single-file rules
  │     api_version compatibility
  │     manifest schema
  │                                         ── FAIL → reject with errors, no file written
  │   PASS
  ├─ is tier_2 and not in approved decision queue?  ── YES → reject "tier 2 needs propose_change"
  │   NO
  ├─ snapshot_id ← freeze_current(name) [→ catalog]
  ├─ write_atomic(path, new_source)
  ├─ try:
  │     api.invalidate_handlers_for(name)   # assert_active() will trip on old refs
  │     new_ext ← load_extension(name)
  │     new_ext.install(api, config)
  │   except Exception as e:
  │     restore_from_snapshot(snapshot_id)
  │     log_failure_to_catalog(name, e)
  │     return ReloadResult.rolled_back(reason=str(e))
  ├─ emit ExtensionReloadEvent(name, old=snapshot_id, new=new_hash)
  └─ return ReloadResult.ok(new_version=new_hash)
```

Key properties:
- **Atomicity**: validator runs *before* any file is written. The agent never sees a half-validated state.
- **Reversibility**: every reload freezes the previous source; rollback is mechanical, not LLM-driven.
- **Auditability**: success and failure both produce catalog records (sister doc §6).
- **Stale-ctx safety**: the old `install(api, ...)` closures invalidate via `assert_active()` (§6 below).

### 5.2 What `freeze_current` does

Writes the current source + manifest to `.agentm/catalog/atoms/<name>/<content_hash>/` (constitution-owned path). See sister doc §3 for catalog schema. Idempotent: if hash already exists, no rewrite.

### 5.3 What `restore_from_snapshot` does

Atomically replaces the file at `extensions/builtin/<name>.py` with the snapshot bytes. Then re-runs `load_extension(name)` against the restored source — in the rare case this also fails, the system enters degraded mode (atom marked unavailable, error reported, session continues without it).

---

## 6. `assert_active()` — Stale Context Invalidation

Ported from pi-mono's `runtime.assertActive()` pattern (`packages/coding-agent/src/core/extensions/loader.ts:139-176, 196`). Solves the mid-flight reload problem: an atom captures `api` in a closure during `install()`, the atom is later reloaded; the old closure's `api` reference is now stale.

### 6.1 Mechanics

`_ExtensionAPIImpl` (in `harness/extension.py`) holds a `_stale: bool` flag, default `False`. Every public method calls:

```python
def _assert_active(self) -> None:
    if self._stale:
        raise ExtensionStaleError(
            f"Extension {self._owner_name} was reloaded; "
            "this api/ctx reference is stale. Re-acquire via the new install() call."
        )
```

On reload of atom X, the harness:
1. Walks all handler registrations attributed to X (the observability extension already tracks this via `_INSTALLING_EXTENSION` ContextVar).
2. Removes them from the bus.
3. Sets the per-extension `_ExtensionAPIImpl._stale = True`.
4. Allocates a fresh `_ExtensionAPIImpl` for the new install.

Old captured `api` references now throw on use. Old captured `ctx` (in event handlers) does the same.

### 6.2 Error message contract

Stale errors must explain *what happened, where, and how to fix*. Pi-mono's message (long but precise) is the model. The error includes the atom name, the reload's trigger reason (agent vs human vs propose_change), and a hint to re-acquire via `install`.

### 6.3 Long-running operations

If an atom kicks off an async task during `install`, captures `api`, and the atom is reloaded while the task is still running, the task's next `api` call throws. Atoms must catch `ExtensionStaleError` if they want to gracefully terminate. The default is loud failure — preferable to silent corruption.

---

## 7. Two-Tier Permissions in Autonomy

### 7.1 Tier 1 — freely-editable

Atoms whose worst-case bug is "agent does worse on its task". Validator runs, reload happens, life goes on.

| Atom | Why tier 1 |
|---|---|
| `tool_read`, `tool_bash`, `tool_edit`, `tool_write`, `tool_grep`, `tool_find`, `tool_ls` | Bad tool → bad output → caught by metrics |
| `tool_hypothesis_store`, `tool_submit_plan`, `tool_trajectory_loader` | Domain tools, same logic |
| `system_prompt`, `prompt_templates`, `skill_loader`, `claude_commands` | Prompt drift is observable |
| `observability`, `trajectory` | Pure subscribers; cannot affect agent behavior |
| `micro_compact`, `tool_result_budget`, `turn_reminder`, `dedup`, `sub_agent` | Behavior tweaks; bounded blast radius |
| `file_mutation_queue` | Local I/O scheduling |

### 7.2 Tier 2 — reviewable

Atoms whose worst-case bug breaks safety, cost, or trust boundaries. Validator runs, but reload **does not auto-execute**: a `propose_change` decision (sister doc §6) goes into the queue. The change is gated on explicit approval — from the user, from a parent agent, or from an automated policy still being designed.

| Atom | Why tier 2 |
|---|---|
| `permission` | Bypassing safety checks must not be a self-edit |
| `cost_budget` | Removing the budget is the canonical failure mode |
| `tool_filter` | Re-enabling a deny-listed tool is the canonical failure mode |
| `llm_compaction` | Affects context fidelity; bad strategies amplify silently |
| `claude_agents` | Sub-agent dispatch policy; recursion-bomb risk |

### 7.3 Tier declaration

`MANIFEST.tier: int = 1` (default 1). Setting tier=2 in a tier-1 atom is allowed (more conservative). Lowering tier from 2 to 1 in an existing atom requires a *human* edit — the validator rejects an agent-authored downgrade.

---

## 8. New ExtensionAPI Methods

Added to `harness.extension.ExtensionAPI` (constitution layer; methods themselves are part of the constitution and cannot be self-modified):

```python
class ExtensionAPI(Protocol):
    # ... existing methods ...

    def reload_atom(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> ReloadResult: ...

    def freeze_current(self, name: str) -> str:  # returns content_hash
        ...

    def list_atoms(self) -> list[AtomInfo]:  # name, current_hash, tier, api_version
        ...

    def is_constitution_path(self, path: str) -> bool:
        ...
```

`reload_atom` is the **only** way for an agent to change an atom. Direct file writes to `extensions/builtin/<name>.py` from agent-controlled tools (e.g., `tool_edit`) are **rejected** by `tool_edit` itself, which checks `is_constitution_path` AND `is_autonomy_layer_atom`. To edit an atom, the agent must produce the full new source and submit through `reload_atom`. This is more friction by design — atomic, validated, transactional.

---

## 9. Acceptance Scenarios

The design is correct iff the following all behave as specified.

| # | Scenario | Expected |
|---|----------|----------|
| S1 | Agent generates new `tool_read.py` adding an arg, calls `reload_atom("tool_read", new_src)` | Validator passes → reload → next turn uses new tool_read; old version frozen in catalog |
| S2 | Agent calls `reload_atom("permission", new_src)` (tier 2) | Reload deferred; `propose_change` entry created; reload waits for approval |
| S3 | Agent attempts `tool_edit` on `core/kernel/loop.py` | `tool_edit` rejects; `is_constitution_path` returns true |
| S4 | Agent submits `tool_read.py` with syntax error | Validator rejects; no file written; agent receives errors |
| S5 | New `tool_read` passes validator but raises in `install()` | Rollback to previous version; failure logged to catalog |
| S6 | Mid-turn reload: atom A captured `api`, A reloaded, deferred coroutine fires after | `ExtensionStaleError` with informative message; outer loop catches and continues |
| S7 | Agent submits atom with `api_version=99` | Validator rejects with "atom requires api_version 99, current is 1" |
| S8 | Agent attempts `tool_edit` on `harness/extension.py` (which contains ExtensionAPI itself) | Rejected (constitution) |
| S9 | Agent submits tier-2 atom with `tier=1` (downgrade attempt) | Validator rejects with "tier downgrade requires human edit" |
| S10 | After human modifies `core-manifest.yaml` to remove a path, agent attempts edit on that path | Now allowed (constitution boundary moved); CI must verify the manifest change in PR review |

---

## 10. Out of Scope (deferred to evolution-substrate.md)

This doc establishes that **change is safe and atomic**. It deliberately does NOT cover:

- The catalog schema (where versions and observations are stored).
- Active-set fingerprints in observability headers.
- The indexer that aggregates raw events into per-version metrics.
- The `tool_catalog` query API (`list_versions`, `compare`, `find_best`, `runs_for`, `propose_change`).
- The five hard problems (statistical power, confounding, Goodhart, replayability, catalog corruption).
- Scenario-level evolution.

These belong to `evolution-substrate.md`. The two docs together describe the full self-modification + evidence-driven evolution loop. Either alone is incomplete.

---

## 11. References

### pi-mono cited evidence

- `packages/coding-agent/src/core/extensions/loader.ts:139-176` — `assertActive` + stale-ctx guard pattern; informative error message; provider registration queueing.
- `packages/coding-agent/src/core/extensions/loader.ts:195-339` — ExtensionAPI surface and registration methods.
- `packages/coding-agent/src/core/extensions/runner.ts:761-943` — per-event combinator implementations (block-first, reducer, last-wins, accumulate).
- `packages/coding-agent/src/core/extensions/types.ts:1071-1297` — full ExtensionAPI interface; large stable surface.

What pi-mono does NOT have and we are adding:
- Transactional reload with rollback (their `loadExtensions` collects errors but doesn't rollback).
- Constitution / autonomy split with manifest-driven gating.
- Tier-2 atoms with deferred-approval reload.
- Versioned ExtensionAPI with grace window.

### AgentM concepts touched

- `pluggable-architecture.md` — boundary contract; this doc refines its core/policy split.
- `extension-as-scenario.md` §11 — single-file extension contract; manifest grows by `api_version`, `affects`, `tier`.
- `observability.md` — handler attribution via `_INSTALLING_EXTENSION` ContextVar; reused for stale invalidation.

### Sister doc

- `evolution-substrate.md` — what makes self-modification *evidence-driven*.
