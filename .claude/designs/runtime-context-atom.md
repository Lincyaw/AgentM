# Design: `runtime_context` builtin atom

**Status**: PROPOSED
**Created**: 2026-05-10
**Builds on**: [extension-as-scenario.md](extension-as-scenario.md), [pluggable-architecture.md](pluggable-architecture.md)

---

## 1. Rationale

A live Feishu test surfaced a bug: the agent answered "what is your cwd?" with
`/home/ddq` (operator's home) instead of the actual gateway workspace
`/tmp/agentm-gateway-test`. The scenario `system_prompt` never told it the cwd,
so it guessed.

The previous turn fixed this in the **wrong layer**: `contrib/channels/`
synthesised a runtime-identity preface and injected it through
`AgentSessionConfig.extra_extensions`. That makes the channel layer compose
prompt content, which violates the four-layer model:

| Layer | Owns | Forbidden |
|---|---|---|
| **atom** | a single capability/behavior unit | scenario composition |
| **scenario** | YAML composition of atoms = persona + tool stack | session orchestration |
| **channel** | I/O adapter (Feishu/Slack/…) ↔ MessageBus | composing prompts or atoms |
| **gateway** | MessageBus ↔ AgentSession routing + approval bridge | composing prompts or atoms |

Cwd hallucination is **not chat-specific**. CLI sessions, plan mode, RCA, and
trajectory_analysis are all equally vulnerable — the LLM has no other source of
truth about its working directory or runtime. The fix belongs in a generic atom
that any scenario can opt into.

The chat-format hint (Feishu prefers short paragraphs, no large headings, no
tables) is scenario-shaped, not channel-shaped: it is part of the agent's
persona on a chat surface. It belongs in the `chatbot` scenario's existing
`system_prompt` text, not in the channel layer or the runtime atom.

---

## 2. Atom contract

**File**: `src/agentm/extensions/builtin/runtime_context.py` (single file, §11).

**Behavior**: subscribes to `BeforeAgentStartEvent` and **prepends** a small
`<runtime_context>` block describing workspace + host platform to the assembled
system prompt. Same hook and same prepend semantics as
`extensions.builtin.system_prompt`, so order in the scenario YAML is what the
author wrote — when both are listed, whichever runs first sits closest to the
top.

**Inputs**:
- `api.cwd` — already exposed on `ExtensionAPI` (used by `skill_loader.py`).
- `platform.system()`, `platform.machine()`, `platform.python_version()` — stdlib.

**Output (injected, prepend)**:

```
<runtime_context>
workspace: /tmp/agentm-gateway-test
runtime: Linux x86_64, Python 3.12.5
</runtime_context>

When the user asks where you are, what you can see, or what your working
directory is, answer from the workspace path above. Do not guess and do not
fall back to a home directory. All shell commands run with this as cwd.
```

The XML-tagged block keeps the facts machine-greppable in traces and visually
separable from scenario persona text.

### 2.1 MANIFEST

```python
MANIFEST = ExtensionManifest(
    name="runtime_context",
    description="Injects workspace cwd + host runtime facts into the system prompt.",
    registers=("event:before_agent_start",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=(),  # Leaf atom: reads only api.cwd + stdlib platform.
)
```

`config_schema` is an empty object (not `None`) so the validator accepts a bare
`- module: ...` entry and any future config keys remain explicit additions.

### 2.2 §11 compliance

- One file under `src/agentm/extensions/builtin/`.
- Allowed imports only: stdlib (`platform`, `pathlib`), `agentm.extensions`
  (`ExtensionManifest`), `agentm.core.abi.events` (`BeforeAgentStartEvent`),
  `agentm.core.abi.extension` (`ExtensionAPI`).
- No atom-to-atom imports, no `harness.session`, no `core._internal`.
- Config schema declared.

### 2.3 Why a new atom and not a `system_prompt` enhancement

`system_prompt` already takes a static `prompt` string. The runtime block is
**dynamic** (resolved at install time from `api.cwd` + the live host). Pushing
runtime resolution into `system_prompt` would either (a) overload its config
shape with conditional templating, or (b) force every scenario author to
hand-build the workspace string. Two atoms with one responsibility each is
cleaner than one atom with two modes.

Both atoms compose by simple prepend on `BeforeAgentStartEvent`; ordering in
the scenario YAML controls which sits on top.

---

## 3. Scenario integration

`chatbot` and every other scenario that wants the fix opts in by adding
**one line** to `extensions:`, listed *after* the persona `system_prompt`
so its prepend runs last and lands at the top of the assembled prompt:

```yaml
extensions:
  - module: agentm.extensions.builtin.tool_read
  ...
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: |
        You are talking to a user inside a Feishu / Lark chat. Speak
        like a coworker: short, concrete, low ceremony. ...
  - module: agentm.extensions.builtin.runtime_context   # NEW — workspace + runtime facts
```

EventBus dispatches `BeforeAgentStartEvent` handlers in **registration order
(FIFO)**, so the *later* an atom is listed in `extensions:`, the *later* its
prepend runs, and the further toward the **top** of the assembled system
prompt its content lands. To put runtime facts at the top, list
`runtime_context` **after** `system_prompt` (and any other prompt-prepending
atom). Both atoms prepend; the last one to fire wins the top slot.

Other scenarios (`general_purpose`, `plan_mode`, `rca`, `trajectory_analysis`)
should add the same line. The atom is a no-op for callers that don't list it,
so adoption is opt-in per scenario.

---

## 4. Tests

No new fail-stop test. Per CLAUDE.md "Testing philosophy", this atom is in the
"single happy path / framework guarantee" zone:
- the §11 validator (`tests/unit/extensions/test_extension_contract.py`,
  per design §11.4) already mechanically gates the new file's imports, manifest,
  and signature — so the contract holds without hand-written assertions;
- the prepend semantics are identical to `system_prompt`, which already
  exercises `BeforeAgentStartEvent` end-to-end through the harness.

If a regression of the original cwd-hallucination bug needs locking down,
the right home is a stub-provider integration test under
`tests/integration/` that drives a real `AgentSession` with `runtime_context`
loaded and asserts the system prompt seen by the StreamFn contains the
workspace path. That is a follow-up, not a blocker for this design.

---

## 5. Index propagation

- New concept `runtime_context_atom` registered in `.claude/index.yaml`.
- `extension_as_scenario` and `pluggable_architecture` gain `runtime_context_atom`
  in their `related_concepts` lists (the atom is a leaf consumer of both).
- No design supersession; this is purely additive.
