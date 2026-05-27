# Nested-Session Task Mechanism

> One mechanism for "spawn a nested `AgentSession`, drive it to completion,
> collect its result" — shared by the agent-driven dispatcher
> (`sub_agent`) and the framework-driven supervisor (`llmharness`). Today
> each rolls its own copy; this unifies the copies onto a single
> allow-listed helper without growing core policy.

Status: planned (2026-05-27)
Owner: subagent + orchestration + llmharness

Related: [sub-agent-lifecycle](sub-agent-lifecycle.md),
[harness-runner](harness-runner.md),
[llmharness-cognitive-audit](llmharness-cognitive-audit.md),
[pluggable-architecture](pluggable-architecture.md),
[agent-loop](agent-loop.md).

## 1. Problem

"Spawn a child `AgentSession`, send it one prompt, run it to completion,
scrape its result from the returned messages" is implemented **three
times** with no shared code:

| # | Where | Spawns via | Collects |
|---|---|---|---|
| 1 | `sub_agent._ChildTaskManager._run_child` (builtin) | `api.spawn_child_session` | child's final assistant text |
| 2 | `llmharness LiveChildRunner.run_{extractor,auditor}` (contrib atom) | `api.spawn_child_session` | a **terminal tool-call's args** (`finalize_extraction` / `submit_verdict`) |
| 3 | `llmharness tools.engine.run_phase_standalone` (contrib host-driver) | `create_agent_session` (no parent) | same terminal tool-call args |

The only real differences are **how the child is spawned** (live, via a
parent `ExtensionAPI`, vs. embedded, by building a top-level session) and
**what counts as the result** (final free text vs. a terminal tool-call's
arguments). Everything else — assemble `AgentSessionConfig`, prompt,
shut the child down, flatten assistant blocks — is duplicated.

`harness-runner.md` already collapsed llmharness's own three copies behind
the `ChildRunner` Protocol; this design takes the next step and shares the
mechanism across the `sub_agent`/`llmharness` package boundary too.

## 2. The §11 constraint decides the home (and it is not contrib)

The natural instinct is "shared helper → `contrib/`". The extension
import allow-list forbids it. `src/agentm/extensions/validate.py`
`_ALLOWED_PREFIXES` is exactly:

```
agentm.core.abi · agentm.core.lib · agentm.ai · agentm.extensions
```

Cross-contrib imports are forbidden (`_agentm_contrib__` rule:
"contrib atoms must stay decoupled"). So a helper that **two atoms in
different packages** (`sub_agent`, `llmharness`) both import **must live
in the agentm core tree**, not contrib.

This does not violate "no policy in core". The shared piece is pure
**mechanism** — compose `ExtensionAPI` + scrape messages. Policy (which
extensions/persona/provider, free-text vs. terminal-tool collect) stays
in the calling atoms.

Within the core tree there are two allow-listed homes; pick by leaf-purity
(see §3.1): `core.lib` is the *leaf* utility layer for helpers that reach
no further than stdlib/pydantic (cf. `pydantic_to_openai_tool_schema`);
`agentm.extensions` is for non-atom mechanism that composes `core.abi`.
Both new modules touch `core.abi`, so both live under `agentm.extensions`.

The boundary splits by **who is §11-constrained**, which inverts the
intuition:

| Path | Caller | §11? | Helper home |
|---|---|---|---|
| **live** (`spawn_child_session`) | an **atom** | yes — allow-list only | **agentm core tree** |
| **embedded** (`create_agent_session`) | a **host-driver** (CLI/eval) | no — not an atom | **contrib** is fine |

`create_agent_session` and `spawn_child_session` themselves both already
live in core and do not move.

## 3. Design

### 3.1 Pure collect helpers — `agentm.extensions.child_collect`

Pure functions over `list[AgentMessage]`, no I/O, atom-safe:

- `flatten_assistant_blocks(messages) -> list[dict]`
- `terminal_tool_arguments(messages, tool_name) -> dict | None`
- `final_assistant_text(messages) -> str | None`
- `serialize_block(block) -> dict | None` (the single block-shape
  definition; `flatten_assistant_blocks` and the llmharness trajectory
  serializer both use it, so the replay-sidecar block shape lives in one
  place)

These are lifted verbatim from llmharness
(`audit/seams/session.py`, `audit/runner/runner.py`) — already pure and
already shared inside llmharness; they move into the agentm core tree so
`sub_agent` can use them too.

**Home: `agentm.extensions`, not `core.lib`.** Both are allow-listed and
atom-safe, so either satisfies §11. The deciding factor is leaf-purity:
`core.lib` is the *leaf* utility layer (`redact`, `serialization`,
`tool_schema` reach no further than stdlib/pydantic), but these helpers
import `core.abi.messages`, which transitively runs the `core.abi`
package `__init__` (→ `core.runtime.otel_export` → back to
`core.lib.to_jsonable`). Putting them in `core.lib` made a bare
`import agentm.core.lib` race that re-entrant chain during package init.
`agentm.extensions` is the namespace where non-atom helpers that compose
`core.abi` already live (it sits *above* `core.lib`/`core.abi`), so
`child_collect` belongs next to `child_task` — no `core.lib` special-casing,
no import-order fragility.

### 3.2 Live orchestration helper — `agentm.extensions.child_task`

A non-atom module under the `agentm.extensions` namespace (allow-listed;
NOT under `builtin/`, so not auto-discovered as an atom and not subject to
the atom-to-atom ban). Atom-safe: imports only `core.abi`
(`ExtensionAPI`, `AgentSessionConfig`, messages) + `agentm.extensions.child_collect`.
It does **not** import `core.runtime`.

```python
@dataclass(frozen=True)
class ChildTaskResult:
    messages: list[AgentMessage]
    terminal_called: bool          # a terminal_tool call was found
    terminal_args: dict | None     # populated when terminal_tool is set
    final_text: str | None         # the child's last assistant text
    error: str | None
    latency_ms: int

async def run_child_task(
    api: ExtensionAPI,
    *,
    extensions: list[tuple[str, dict]],
    provider: tuple[str, dict] | None,
    prompt: str,
    purpose: str,
    terminal_tool: str | None = None,   # None → free-text collect; set → tool-arg collect
) -> ChildTaskResult: ...
```

Body = exactly what `LiveChildRunner.run_extractor` does today: build
`AgentSessionConfig(cwd=api.cwd, provider, extensions, purpose)`,
`spawn_child_session`, `prompt`, `safe_shutdown` (always), then collect.
`safe_shutdown` moves into this module.

`terminal_tool` is the one knob that unifies the two collect modes:
`sub_agent` passes `None` (wants the child's free-text answer);
`llmharness` passes `finalize_extraction` / `submit_verdict` (wants the
structured tool-call args). Both come back on the same `ChildTaskResult`.

### 3.3 Embedded helper stays in contrib

`llmharness tools.engine.run_phase_standalone` keeps using
`create_agent_session` (it is a host-driver, not an atom). It is refactored
only to **reuse the `agentm.extensions.child_collect` pure helpers** instead
of its private copies, so the collect contract is identical to the live path.

### 3.4 Consumers refactor onto the mechanism

- **llmharness `LiveChildRunner`** → thin wrapper over `run_child_task`
  (extractor: `terminal_tool=finalize_extraction`; auditor:
  `terminal_tool=submit_verdict`, mapping `ChildTaskResult` →
  `AuditorChildResult` + the failure-record side effects it owns).
- **`sub_agent._run_child`** → calls `run_child_task(terminal_tool=None)`;
  keeps its own task-manager, persona resolution (`ResolveSubagentEvent`),
  parallelism, `inject_instruction`, abort — that is its policy, unchanged.

Net core change: **two new allow-listed modules** (`agentm.extensions.child_collect`,
`agentm.extensions.child_task`), both pure mechanism. No change to
`core.runtime`, the `SUB_AGENT_RUNTIME` role, `ResolveSubagentEvent`, or the
session factory.

## 4. Non-goals / deferred

- **Moving `sub_agent` builtin → contrib.** Verified feasible (core's
  coupling is the provider-agnostic `SUB_AGENT_RUNTIME` role + the
  `ResolveSubagentEvent` ABI, not the builtin module path; scenarios
  already mount it explicitly). But it is **independent** of this
  unification and not required — decide separately.
- **Exposing structured (`terminal_tool`) output through `dispatch_agent`.**
  The mechanism supports it from day one; surfacing it as a
  `dispatch_agent` option (so a model can ask a subagent for typed output)
  is a follow-up once a consumer needs it.

## 5. Phasing (each phase is a dev-worker unit, independently verifiable)

1. **Foundation + first consumer.** Add `agentm.extensions.child_collect` +
   `agentm.extensions.child_task.run_child_task`. Refactor llmharness
   `LiveChildRunner` and `run_phase_standalone` onto them. Verify: llmharness
   `pytest`/`mypy`/`ruff` green; a real-provider live + replay run still
   produces a valid graph. (No `sub_agent` change yet — de-risks the
   mechanism in isolation.)
2. **Second consumer.** Refactor `sub_agent._run_child` onto
   `run_child_task`. Verify: sub_agent tests + an rca-scenario E2E (which
   mounts sub_agent) still dispatch + collect correctly.
3. **(Optional) structured dispatch.** Add the `terminal_tool` /
   `output_schema` option to `dispatch_agent`.

`harness-runner.md`'s `ChildRunner` Protocol becomes llmharness's thin
adapter over phase-1 helpers; revisit whether it still earns its keep after
phase 1.

## 6. Acceptance

- Boundary: `agentm.extensions.child_collect` and `child_task` pass the §11
  import validator (only `core.abi` / `core.lib` imports) AND a full
  `validate_atom_file` run (e.g. `__all__: Final`); both `sub_agent` and
  `llmharness` import them without a validator violation.
- Behaviour: llmharness sidecar `ReplayRecord` shape unchanged; the
  extractor directive stays byte-identical; sub_agent `dispatch_agent` /
  `check_tasks` / `wait_subagent` results unchanged.
- No duplication: `flatten_assistant_blocks`, `serialize_block`, and
  terminal-arg scraping exist in exactly one place (`child_collect`); the
  llmharness trajectory serializer imports `serialize_block` rather than
  copying it.
