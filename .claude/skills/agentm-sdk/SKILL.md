---
name: agentm-sdk
description: >
  AgentM SDK development guide — manifest-as-agent-unit philosophy, SDK
  programmatic invocation, atom contract, Operations abstraction, event
  system, service communication, CLI conventions, scenario authoring,
  logging, structured output, and config resolution. Use whenever writing,
  editing, or reviewing code under src/agentm/ (atoms, core, gateway),
  contrib/scenarios/ (manifests), or contrib/extensions/ (workspace-member
  atoms). Also trigger when creating new atoms, modifying MANIFEST
  declarations, registering tools or events, touching FileOperations /
  BashOperations / ResourceWriter, writing CLI subcommands, configuring
  model profiles, spawning child sessions, or when a code change looks
  like it might bypass the SDK's existing abstractions. If you catch
  yourself about to write raw os.stat / open() / subprocess.run in an
  atom, importing openai/anthropic directly instead of going through the
  provider layer, or using AgentSession.create() inside an atom instead of
  api.spawn_child_session(), stop and read this first.
---

# AgentM SDK Guide

This skill exists because agents working on this codebase tend to reinvent
mechanisms the SDK already provides. Before writing new infrastructure, check
whether the SDK already has it.

## Architecture

```
presenters: agentm.cli  /  embedded SDK (AgentSession.create)
atoms:      extensions/builtin/  +  contrib/scenarios/<name>/  +  contrib/extensions/
substrate:  core/ (abi + runtime + lib — write-protected)
```

- **Atoms** are plugins. Each gets an `AtomAPI` handle in its `install()`.
  Everything an atom needs is accessed through `api`. Importing from
  `core.runtime` is forbidden.
- **Manifests** (scenario YAML) compose atoms into single-purpose agents.
  One manifest = one role. Dynamic behavior comes from composing manifests
  in orchestration code, not from building mega-prompts.
- **Host programs** (CLI, gateway, eval harnesses) invoke manifests via
  `AgentSession.create()`. They own scheduling, parallelism, and retry.
  They do not own domain logic — that lives in the manifest's atoms.

---

## 1. Manifest = one agent, one purpose

A manifest is the **minimal unit of agent composition**. Each manifest
defines a single-purpose agent — not a multi-role agent switched by
prompt text.

**Why:** A manifest declares what tools the agent has, what system prompt
guides it, and what atoms shape its behavior. Mixing roles in one manifest
means the agent carries tools and guidance it doesn't need — the LLM sees
irrelevant options, and the prompt must work harder to constrain behavior.

```
# WRONG: one manifest, two roles via prompt
contrib/scenarios/verifier/manifest.yaml   <- registers both hop + judge tools
harness.py: agentm --scenario verifier -p "<hop prompt>"
harness.py: agentm --scenario verifier -p "<judge prompt>"

# RIGHT: one manifest per role
contrib/scenarios/verifier_hop/manifest.yaml    <- hop tools + hop guidance
contrib/scenarios/verifier_judge/manifest.yaml  <- judge tools + judge guidance
```

### Prompt construction co-locates with the manifest

The domain knowledge needed to build a good prompt (relationship
descriptions, fault reference docs, heuristics) belongs in the scenario
directory, co-located with the manifest it serves — not in external
orchestration code.

```python
# contrib/scenarios/verifier_hop/prompt.py
def build_hop_prompt(from_svc, to_svc, rel_type, faults, ...) -> str:
    ...
```

The orchestrator imports and calls it, then passes the result to
`agent()` or `session.prompt()`. The manifest's `system_prompt` handles
"how to think"; the built prompt handles "what to check".

### Policy lives in atoms, orchestration lives outside

**Atoms** own domain logic: prompt formatting, fault reference loading,
relationship descriptions, threshold decisions. **Orchestration code**
owns scheduling: BFS traversal, parallel dispatch, retry, result
collection, batch caching. If orchestration code is building 40-line
prompt strings or hardcoding domain thresholds (`if drop > 80%`), that
logic belongs in an atom or prompt-builder co-located with the manifest.

---

## 2. Session creation — who uses what

Your position in the architecture determines which API you use:

| You are writing… | Use this |
|------------------|----------|
| **Atom** code | `api.spawn_child_session(config)` — inherits trace + provider from parent |
| **Atom** code (lightweight child) | `api.spawn(purpose=..., tools=..., model=...)` — inherits everything, override only what differs |
| **Host program** (CLI, gateway, eval harness) | `AgentSession.create(AgentSessionConfig(...))` — cold-starts a full session |

The most common mistake is using `AgentSession.create()` inside an atom.
Atoms cannot import `core.runtime`; use `api.spawn_child_session()`
or `api.spawn()` instead — they give you trace and provider inheritance
for free.

For detailed API signatures, examples, threading patterns, and common
mistakes, read `references/session-creation.md`.

---

## 3. The atom contract

Every atom exports `MANIFEST` + `install()`. Builtin atoms are single
`.py` files; contrib atoms may be packages (directories with
`__init__.py`). The atom contract applies to all code **reachable from
install()** — including intra-package modules that `install()` imports.

```python
MANIFEST = ExtensionManifest(
    name="my_atom",
    description="What this atom does.",
    registers=("tool:my_tool",),  # what it provides
    requires=(),                   # what it needs from other atoms
)

def install(api: AtomAPI, config: Mapping[str, JsonValue]) -> None:
    # all wiring happens here
```

### Import rules (enforced by `extensions.validate`)

| Allowed | Forbidden |
|---------|-----------|
| `agentm.core.abi.*` | `agentm.core.runtime.*` |
| `agentm.core.lib.*` | other atoms (`agentm.extensions.builtin.X`) |
| `agentm.extensions.ExtensionManifest` | `agentm.core._internal` |
| `agentm.ai` (provider metadata) | cross-contrib imports |
| stdlib, third-party libs | `langchain` |

These rules are enforced at two levels:
1. **Static check** — `agentm lint` (run like ruff/mypy before execution)
2. **Runtime load-time** — `load_extension()` blocks non-compliant atoms

For package atoms, the validator builds an AST import-reachability graph
from `install()` to determine which files are atom code (contract
enforced) vs host-level code (not enforced). A file in the package that is not
reachable from `install()` is free to import `core.runtime` — it is a
host-level tool, not atom code.

### Config resolution

Atom config comes from three sources (highest wins):

```
CLI --set overrides  >  env AGENTM_<ATOM>_<KEY>  >  manifest config:
```

Declare accepted keys in `MANIFEST.config_schema`. Use
`config.get(key, default)` in `install()` — defaults live there, not in
the schema.

---

## 4. Scenario authoring

A scenario is a YAML manifest at `contrib/scenarios/<name>/manifest.yaml`:

```yaml
name: my_scenario
description: What this scenario does.
extensions:
  - module: agentm.extensions.builtin.operations
    config:
      backend: local
  - module: agentm.extensions.builtin.file_tools
  - module: agentm.extensions.builtin.observability
  - local: my_local_atom    # scenario-local atom
  - module: agentm.extensions.builtin.system_prompt
    config:
      prompt: "You are a helpful assistant."
```

Rules:
- `operations` atom listed **first** — other atoms depend on it
- Scenarios must explicitly list every atom they need; there is no auto-mount
- Scenario-specific logic in `contrib/scenarios/<name>/`, **never** in
  `src/agentm/core/`
- Local atoms referenced with `local:` resolve from the scenario directory

---

## 5. Quick reference

| I want to... | Use this |
|--------------|----------|
| Read a user file | `api.get_operations().file.read_file(path)` |
| Write a user file (git-tracked) | `api.get_resource_writer().write(path, data)` |
| Run a shell command | `api.get_operations().bash.exec(cmd, cwd=api.ctx.cwd)` |
| Register a tool | `api.register_tool(FunctionTool(...))` |
| Listen to events | `api.on(Event.CHANNEL, handler)` |
| Share state between atoms | `api.services.register(name, obj, protocol)` |
| Consume another atom's state | `api.services.get(name, protocol)` |
| Inject system prompt content | Handle `BeforeRunEvent` |
| Inject per-turn context | Handle `ContextEvent` |
| JSON Schema from Pydantic | `pydantic_to_tool_schema(Model)` from `agentm.core.lib` |
| Spawn a child agent (lightweight) | `api.spawn(purpose=..., tools=..., model=...)` |
| Spawn a child agent (full config) | `api.spawn_child_session(AgentSessionConfig(...))` |
| Boot a session from host code | `AgentSession.create(AgentSessionConfig(...))` |
| Validate atom contract compliance | `agentm lint` (CLI) |
| Emit user-visible diagnostic | Emit `DiagnosticEvent` |
| Log for debugging | `from loguru import logger` (stdlib logging is forbidden) |

For detailed API signatures, read `references/api.md`.
For session creation APIs, read `references/session-creation.md`.
For provider layer, CLI conventions, and logging, read
`references/provider-and-cli.md`.

---

## 6. Anti-patterns

### Boundary violations

- **Importing `core.runtime`** — Use `core.abi` only.
- **Importing another atom** — Use `api.services.get()`.
- **Scenario-specific logic in `src/agentm/core/`** — Belongs in
  `contrib/scenarios/<name>/`.
- **Direct `openai` / `anthropic` imports** — Use the provider layer.

### Abstraction bypasses

- **Direct filesystem I/O in tool handlers** — Use Operations.
- **`subprocess.run()` for shell commands** — Use `BashOperations.exec()`.
- **Hand-writing JSON Schema next to a Pydantic model** — Use
  `pydantic_to_tool_schema`.
- **`print()` or stdout writes in atoms** — Use `loguru.logger` or
  `DiagnosticEvent`.

### Composition mistakes

- **Multi-role manifest** — One manifest = one purpose. Split roles.
- **Hardcoded domain thresholds in orchestration** — Policy belongs in
  atoms, not in eval harnesses or host programs.
- **Prompt construction outside the scenario** — Domain knowledge needed
  to build prompts belongs in the scenario directory, co-located with the
  manifest, not in external orchestration code.

### Invocation mistakes

- **Shelling out to CLI from Python** — Use `AgentSession.create()`.
- **Parsing obs JSONL to extract results** — Use session return values.
- **Wrong session API for your layer** — See §2 and
  `references/session-creation.md` for the full decision table.

### Miscellaneous

- **Resolving Operations at install time** — Use lazy-resolve.
- **Raising exceptions from tool handlers** — Return
  `ToolResult(is_error=True)`.
- **Preset enums for subjective fields** — Use free-text + LLM-decided.
- **Config defaults in schema** — Defaults live in
  `config.get(key, default)`.
