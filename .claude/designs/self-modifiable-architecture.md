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

### Recovery-floor invariant

A stronger property follows from this boundary, and the constitution split is designed around it:

> **`core/abi/` + `core/lib/` + `llm/` + stdlib must be sufficient to launch a working agent loop.** No import from `agentm.harness` or `agentm.extensions` may be required.

This is not just "agent cannot break core" — it is "core alone yields a usable agent". When the autonomy layer is corrupted (a self-edited atom misbehaves, a scenario YAML is malformed, an extension reload leaves harness in an inconsistent state), the operator still has a path to launch an agent: invoke `agentm --no-extensions "<prompt>"` to bypass atom discovery entirely, leaving only the kernel + provider + loop. That floor has no tool environment, skills, or observability, but it launches, accepts a prompt, and returns — enough to inspect and repair whatever is broken above.

The invariant is enforced structurally (no `harness`/`extensions` import in `core/abi/`, `core/lib/`, or `llm/`) and is the load-bearing reason for keeping `core/abi/` and `core/lib/` import-closed. Future PRs that introduce a `harness` import into `core/abi/` or `core/lib/` should be rejected on this basis alone.

---

## 2. Three Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  Constitution layer  (write-protected; only humans + PR review)      │
│                                                                      │
│    core/abi/        types & Protocols (Tool, Message, StreamFn,      │
│                     events, FileOperations / BashOperations)         │
│                     → atom AND llm provider may import               │
│    core/lib/        pure-function utilities (edit_diff, frontmatter, │
│                     path_utils, text_truncate)                       │
│                     → atom may import (stdlib-style)                 │
│    core/_internal/  stateful subsystems & default impls              │
│                     (operations_impl, skills, prompt_templates,      │
│                     catalog/, compaction/ engine)                    │
│                     → atom should reach via ExtensionAPI             │
│                                                                      │
│    llm/, ai/                          (provider boundary)            │
│    harness/{session, extension, events, session_manager, ...}       │
│    extensions/{loader, discover, validate}                           │
│    cli                                (presenter startup contract)   │
│    .agentm/catalog/                   (write-protected from agent)   │
│    core-manifest.yaml                 (self-referential lock)        │
├──────────────────────────────────────────────────────────────────────┤
│  Evolution substrate (see evolution-substrate.md)                    │
│    catalog: versioned (atom, scenario) × observation × decisions     │
│    indexer (constitution-owned)                                      │
│    tool_catalog (read API to agent; propose_change is the only write)│
├──────────────────────────────────────────────────────────────────────┤
│  Autonomy layer  (agent self-edits + reloads)                        │
│    extensions/builtin/<atom>.py (tier 1 free / tier 2 review)        │
│    extensions/scenarios/<name>.yaml                                  │
│    skills/, prompts/, settings                                       │
├──────────────────────────────────────────────────────────────────────┤
│  Discovery layer                                                     │
│    filesystem scan · reload() · assert_active() · stale invalidation │
│    (mechanism is in constitution; this is how autonomy edits land)   │
└──────────────────────────────────────────────────────────────────────┘
```

Dependency rule unchanged from `pluggable-architecture.md`: arrows point downward only. The new ordering inserts the evolution substrate between constitution and autonomy — the substrate is constitution-owned (write side) but agent-readable (query side).

### 2.1 Why the constitution is split into abi / lib / _internal

The constitution is uniform on the **modifiability** axis — nothing in `core/` is agent-modifiable. But it is heterogeneous on the **visibility** axis — different parts have different rules for who may `import` them. The three subtrees are the visibility layer; they are not three privilege levels.

| Subtree | Modifiability | Atom may import? | Why |
|---|---|---|---|
| `core/abi/` | constitution | yes | the ABI surface — types and Protocols both atoms and the llm provider depend on |
| `core/lib/` | constitution | yes (stdlib-style) | pure functions with no state; routing through ExtensionAPI would only add boilerplate |
| `core/_internal/` | constitution | **no** — atoms use ExtensionAPI services | stateful subsystems and default impls. Atoms reach them via `api.get_operations()`, `api.skills`, `api.prompt_templates`, `api.catalog`, `api.compaction` — harness owns instance lifecycle and substitution |

Concretely, this means an atom like `tool_edit` can `from agentm.core.lib import edit_diff` (pure function) freely and `from agentm.core.abi.operations import FileOperations` for the type, but obtains the bound instance through `api.get_operations().file` — never `from agentm.core._internal.operations_impl import LocalFileOperations`. The validator forbids any import under `agentm.core._internal.*` from atom modules.

Pure data types that atoms exchange (e.g. `SkillRecord`, `PromptTemplateRecord`, `Operations` bundle, `CompactionSettings`) live in `core/abi/` so the atom can construct and pattern-match against them without traversing the API. Only behaviour (loaders, engines, fingerprint computation) hides behind the API.

### 2.2 Worked example: compaction prompts move out of constitution

The split forces a useful question on every constitution module: "is this mechanism, or has policy leaked in?" Compaction is the canonical case.

Before the split, `core/compaction/compaction.py` and `core/compaction/branch_summarization.py` carried two large hard-coded prompt strings (`_SUMMARIZATION_PROMPT`, `_BRANCH_SUMMARY_PROMPT`) inside what is otherwise pure mechanism (token estimation, splice points, message replacement). The prompts dictate **how aggressively** to summarize and **what shape** the summary takes — that is policy. Locking them in constitution made them un-evolvable: the agent could not A/B-test a tighter prompt, could not roll back a regression via `tool_catalog`, could not even propose a change.

After the split:

| Concern | Old location | New location | Layer |
|---|---|---|---|
| Splice / token math / message replacement | `core/compaction/` | `core/_internal/compaction/` | constitution (mechanism) |
| `_SUMMARIZATION_PROMPT`, `_BRANCH_SUMMARY_PROMPT` | hard-coded constant in core | module constant in `extensions/builtin/llm_compaction.py` | autonomy (tier-1, agent-modifiable) |
| Engine signature | `compact(messages, ...)` | `compact(messages, ..., summarization_prompt: str)` | constitution (mechanism takes policy as a parameter) |

The atom now passes its prompt into the engine on every call. The engine has no default — refusing to compact without an explicit prompt is the cleanest expression of "engine doesn't carry policy".

This is the pattern to apply elsewhere when in doubt: if a `core/_internal/` module contains a constant whose value is a judgment call (a prompt, a threshold, a strategy choice), that constant belongs in an autonomy-layer atom and should be passed in. Mechanism receives policy; mechanism does not own policy.

---

## 3. The `core-manifest.yaml`

Single source of truth for the constitution boundary. Lives at `core-manifest.yaml` in repo root. Loaded by the validator at startup; CI also enforces it on PRs.

```yaml
# core-manifest.yaml
version: 1

constitution:
  paths:
    # core/ — three visibility tiers, all write-protected
    - src/agentm/core/abi/**         # ABI surface (atom + llm import)
    - src/agentm/core/lib/**         # pure-function utility shelf (atom import)
    - src/agentm/core/_internal/**   # stateful subsystems & default impls
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
  # Claude Code compatibility atoms live under contrib/extensions/cc/
  # (opt-in package); scenarios that load them can re-list them in their
  # own scenario manifest's reload.tier_2_atoms.
  tier_2_atoms:
    - permission
    - cost_budget
    - tool_filter
    - llm_compaction
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
- **Reversibility**: every reload freezes the previous source; rollback is mechanical, not LLM-driven. If installing the new source fails and rollback activation also fails, the in-memory loaded-atom registries keep the previously live `LoadedAtom` instead of purging the operational state; the failure is surfaced as `rollback_failure_state_preserved`.
- **Auditability**: success and failure both produce catalog records (sister doc §6).
- **Stale-ctx safety**: the old `install(api, ...)` closures invalidate via `assert_active()` (§6 below).
- **Async boundary discipline**: reload/install/freeze are native async in the reloader; synchronous ExtensionAPI facades are only API-boundary adapters.
- **Lifecycle symmetry**: harness-owned EventBus subscriptions registered by the reloader are explicitly unsubscribed during `AgentSession.shutdown()` before the bus is cleared.

### 5.2 What `freeze_current` does

Writes the current source + manifest to `.agentm/catalog/atoms/<name>/<content_hash>/` (constitution-owned path). See sister doc §3 for catalog schema. Idempotent: if hash already exists, no rewrite.

> **Implementation note**: as of [git-backed-versioning.md](git-backed-versioning.md), this snapshot mechanism is provided by git plumbing through the harness `ResourceWriter` service. The catalog directory still exists for `metrics.jsonl` and `runs/`; source/manifest content moves into the git object store. The transactional reload flow simplifies to `writer.write(...) → on install failure: writer.restore(path, pre_sha)`.

### 5.3 What `restore_from_snapshot` does

Atomically replaces the file at `extensions/builtin/<name>.py` with the snapshot bytes. Then re-runs `load_extension(name)` against the restored source — in the rare case this also fails, the system enters degraded mode (atom marked unavailable, error reported, session continues without it).

> Per [git-backed-versioning.md](git-backed-versioning.md), this is `git restore --source=<pre_sha> -- <path>` followed by `git reset --hard <pre_sha>` on the auto-commit produced by the writer.

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
| `system_prompt`, `prompt_templates`, `skill_loader` | Prompt drift is observable |
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

Contrib atoms (e.g. `contrib.extensions.cc.agents` under the opt-in
`contrib/extensions/cc/` package) declare their own tier; scenarios opting in
are responsible for re-listing them in the scenario manifest's
`reload.tier_2_atoms`.

### 7.3 Tier declaration

`MANIFEST.tier: int = 1` (default 1). Setting tier=2 in a tier-1 atom is allowed (more conservative). Lowering tier from 2 to 1 in an existing atom requires a *human* edit — the validator rejects an agent-authored downgrade.

---

## 8. New ExtensionAPI Methods

Added to `harness.extension.ExtensionAPI` (constitution layer; methods themselves are part of the constitution and cannot be self-modified):

```python
class ExtensionAPI(Protocol):
    # ... existing methods ...

    def add_observer(self, callback: ObserverCallback) -> Unsubscribe:
        ...

    async def spawn_child_session(self, config: AgentSessionConfig | dict[str, Any]) -> Any:
        ...

    def set_service(self, name: str, obj: Any) -> None:
        ...

    def get_service(self, name: str) -> Any | None:
        ...

    def get_resource_writer(self) -> ResourceWriter:
        ...

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
| S3 | Agent attempts `tool_edit` on `core/abi/loop.py` | `tool_edit` rejects; `is_constitution_path` returns true |
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
