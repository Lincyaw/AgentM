# Nested-Session Task Mechanism

> "Spawn a nested `AgentSession`, drive it to completion, scrape its
> result" — the duplicated copies of this dance inside **llmharness** are
> unified onto one convenience helper layered over the
> `api.spawn_child_session` ABI seam, kept in llmharness (contrib). The
> interactive `sub_agent` dispatcher is a genuinely different shape and is
> deliberately left alone.

Status: implemented — llmharness unified (PR #170 + follow-up); sub_agent
unification evaluated and **declined** (see §4).
Owner: llmharness

Related: [sub-agent-lifecycle](sub-agent-lifecycle.md),
[harness-runner](harness-runner.md),
[llmharness-cognitive-audit](llmharness-cognitive-audit.md),
[pluggable-architecture](pluggable-architecture.md),
[agent-loop](agent-loop.md).

## 1. Problem (and what it turned out NOT to be)

The opening hypothesis was that "spawn a child `AgentSession`, prompt it,
run to completion, scrape its result" was written **three** times and
should collapse to one shared mechanism across the `sub_agent` /
`llmharness` package boundary:

| # | Where | Spawns via | Drive | Collect |
|---|---|---|---|---|
| 1 | `sub_agent._ChildTaskManager._run_child` (builtin) | `api.spawn_child_session` (in `dispatch`) | **multi-turn loop** draining `inject_instruction` | `return_response` arg → else text; dict-tolerant; `\n`-joined |
| 2 | `llmharness LiveChildRunner.run_{extractor,auditor}` (contrib atom) | `api.spawn_child_session` | single prompt | terminal tool-call args (`finalize_extraction` / `submit_verdict`) |
| 3 | `llmharness tools.engine.run_phase_standalone` (contrib host-driver) | `create_agent_session` (no parent) | single prompt | same terminal tool-call args |

On close inspection (§4) the hypothesis was **half wrong**. Copies 2 and 3
are genuinely the same single-shot shape and *were* duplicated — that is
real, and it is now unified. Copy 1 only *looked* similar: `_run_child` is
an **interactive, multi-turn, injectable** driver over a *pre-spawned*
session with its own finalize/abort/artifact bookkeeping and a different
(dict-tolerant, `return_response`-aware) collect contract. It is not a
single-shot spawn-collect at all. Forcing it onto the shared helper would
mean pushing injection-loop policy into the helper — exactly the kind of
"abstraction serving five cases" the project rejects.

So the genuine duplication was **within llmharness**, not across the
package boundary.

## 2. Where the helper lives — and why it is contrib, not core

The §11 import allow-list (`src/agentm/extensions/validate.py`
`_ALLOWED_PREFIXES`) is `agentm.core.abi · agentm.core.lib · agentm.ai ·
agentm.extensions`; **cross-contrib imports are forbidden**. The rule this
implies:

> A helper imported by atoms in **two different packages** must live in the
> agentm core tree (a contrib package cannot import another contrib
> package). A helper with a **single** consuming package belongs **in that
> package**.

The original design placed `child_collect` + `child_task` in the core tree
(`agentm.extensions`) on the assumption that `sub_agent` *and* `llmharness`
would both consume them. Since `sub_agent` does not (§4), the sole consumer
is llmharness — so the two-package trigger never fires, and the
"keep core small; compose existing ABI in contrib" rule governs instead:

- `run_child_task` is composed entirely from the public `core.abi` surface
  (`spawn_child_session`, `AgentSessionConfig`, messages). It adds no new
  capability to core; it is a *convenience* over an existing ABI seam.
- Its only consumer is llmharness's `LiveChildRunner`.

→ Both modules live in **llmharness (contrib)** as internal helpers, not in
core. Core's contribution is unchanged: `api.spawn_child_session` (the
seam) and `create_agent_session` (the embedded host-driver entry) already
exist and do not move.

**Promotion rule.** If a *second package* later needs `run_child_task`,
do not move the function (a second contrib consumer still could not import
it cross-package). Instead promote the capability to a **registered
service** — an atom does `api.set_service(CHILD_TASK_RUNNER, runner)` and
consumers resolve it via `api.get_service(...)` (a string key, zero
cross-package import). That is the §11-clean way to share a stateful-ish
capability, and it is warranted only when a real second consumer exists.

## 3. Design — a layering, not one universal function

| Layer | What | Home | Form |
|---|---|---|---|
| ABI seam (the real pluggability point) | `api.spawn_child_session` | **core** (`ExtensionAPI`) | method |
| single-shot convenience | `run_child_task` + `ChildTaskResult` | **llmharness** (`child_task.py`) | plain function |
| live / offline swap | `ChildRunner` Protocol (`LiveChildRunner` / `StandaloneChildRunner`) | **llmharness** (`audit/`) | internal Protocol |
| pure collect | `flatten_assistant_blocks` / `terminal_tool_arguments` / `final_assistant_text` / `serialize_block` | **llmharness** (`child_collect.py`) | pure functions |

### 3.1 Pure collect helpers — `llmharness.child_collect`

Pure functions over `list[AgentMessage]`, no I/O, importing only
`agentm.core.abi.messages`:

- `flatten_assistant_blocks(messages) -> list[dict]`
- `terminal_tool_arguments(messages, tool_name) -> dict | None`
- `final_assistant_text(messages) -> str | None`
- `serialize_block(block) -> dict | None` — the single replay-sidecar
  block-shape definition; `flatten_assistant_blocks` and the runner's
  trajectory serializer both call it, so the shape lives in one place.

These were duplicated across `audit/seams/*` and `audit/runner/runner.py`;
they are now single-homed.

### 3.2 Single-shot orchestration — `llmharness.child_task`

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
    api, *, extensions, provider, prompt, purpose,
    terminal_tool: str | None = None,   # None → free-text; set → tool-arg collect
) -> ChildTaskResult: ...
```

Builds `AgentSessionConfig(cwd=api.cwd, provider, extensions, purpose)`,
`spawn_child_session`, one `prompt`, always `safe_shutdown`, then collect.
`terminal_tool` is the knob: llmharness passes `finalize_extraction` /
`submit_verdict`. It is **single-shot by design** — no injection loop; a
consumer that needs multi-turn drives `spawn_child_session` itself.

### 3.3 Consumers

- `LiveChildRunner.run_{extractor,auditor}` → thin wrappers over
  `run_child_task` (preserving `ExtractorSpawnError` routing, the auditor
  `_record_failure` side-effects, verdict parsing).
- `run_phase_standalone` (embedded host-driver) keeps `create_agent_session`
  and reuses `child_collect.terminal_tool_arguments` so its collect contract
  matches the live path.
- The runner's `_serialize_message_for_extractor` imports
  `child_collect.serialize_block` (no private copy).

The `ChildRunner` Protocol (`harness-runner.md`) **earns its keep**: it is
the genuine live/offline seam, with two real implementations
(`LiveChildRunner` over `spawn_child_session`; `StandaloneChildRunner` over
`create_agent_session`) and multiple consumers (the live adapter; the
replay / fork-tree / offline drivers). It stays.

## 4. Why `sub_agent` is out of scope

`sub_agent._run_child` is not a single-shot spawn-collect:

- **Multi-turn injection loop**: `while True: prompt(); next = _drain_instructions()` — feeds `inject_instruction` batches back into the same child. `run_child_task` is single-prompt and cannot host this without absorbing injection policy.
- **Pre-spawned session**: the child is spawned in `dispatch()` and stored on `state.session`; `_run_child` reuses it across turns. `run_child_task` spawns internally.
- **Custom finalize**: artifact collection, abort handling, error-vs-clean shutdown.
- **Different collect contract**: `_final_assistant_text` prefers the `return_response` tool-call arg, falls back to text, tolerates **dict-form** messages, and joins with `\n`. `child_collect`'s helpers are `isinstance`-strict and space-join — not byte-equivalent, so reuse would silently change worker output.

These collect helpers live only inside `sub_agent` (no cross-package
duplication), so there is nothing to deduplicate. `sub_agent` keeps its own
interactive driver. The shared seam it already uses is the right one:
`api.spawn_child_session`.

## 5. History

1. **llmharness unification (done).** `child_collect` + `child_task` added,
   `LiveChildRunner` / `run_phase_standalone` refactored onto them; verified
   green incl. a real-provider doubao live + replay run.
2. **Relocation (done).** After establishing `sub_agent` is not a consumer,
   the two modules moved from `agentm.extensions` (core) into
   `llmharness` (contrib), per §2. Core unchanged.
3. **sub_agent unification — declined** (§4).

## 6. Acceptance

- Boundary: no atom imports a non-existent core module; the llmharness
  adapter still validates; `child_task` imports only `core.abi` + the
  sibling `child_collect`, never `core.runtime`.
- No duplication: `flatten_assistant_blocks`, `serialize_block`, and
  terminal-arg scraping exist in exactly one place (`llmharness.child_collect`).
- Behaviour: llmharness sidecar `ReplayRecord` shape and the extractor
  directive are byte-identical; `sub_agent` is untouched.
- Core stays minimal: zero speculative generic child-task modules in the
  core tree; the seam (`spawn_child_session`) is the only core surface.
