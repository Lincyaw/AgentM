# AgentM

A pluggable agent framework in Python. The SDK is a **mechanism**; every policy
is a port; every port has a default; every default is a replaceable extension.

Inspired by [`badlogic/pi-mono`](https://github.com/badlogic/pi-mono). See
`.claude/designs/pluggable-architecture.md` for the boundary contract.

---

## Architecture at a glance

Four layers, dependency arrows point downward only.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  agentm.cli  /  embedded SDK  /  (future: HTTP, RPC)                     │  presenters
│  thin: load scenario → prompt → print                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  agentm.harness                                                          │
│    AgentSession  ── orchestrator façade (no business logic)              │  harness
│    EventBus      ── before_agent_start / tool events / observability     │
│    SessionManager · ResourceLoader · ExtensionAPI · load_extension       │
├──────────────────────────────────────────────────────────────────────────┤
│  agentm.core                                                             │
│    kernel/  AgentLoop · Tool · Message · StreamFn · LoopConfig           │  pure SDK
│    operations.py  FileOperations / BashOperations Protocols + locals     │
│    skills · prompt_templates · edit_diff · frontmatter · path_utils      │
│    compaction/  llm + branch summarization                               │
├──────────────────────────────────────────────────────────────────────────┤
│  agentm.llm  (provider layer, analogous to pi-ai)                        │  provider
│    anthropic StreamFn · ai/api_registry · OAuth / env keys               │
└──────────────────────────────────────────────────────────────────────────┘
```

`agentm.core` must be importable in a Jupyter notebook with no harness, no CLI,
no filesystem touched. The kernel knows nothing about sessions, files, or
scenarios; the harness knows nothing about concrete scenarios or UI.

---

## Five pluggability axes

Every axis is a `typing.Protocol` in `agentm.core`. The harness ships a default;
extensions or users can substitute without forking.

| # | Axis                | Protocol / Port          | Default impl                       |
|---|---------------------|--------------------------|------------------------------------|
| 1 | LLM stream          | `StreamFn`               | `agentm.llm.anthropic`             |
| 2 | Tool environment    | `Tool` + `*Operations`   | `LocalFileOperations`, `LocalBashOperations` |
| 3 | Session state       | `SessionManager`         | `InMemorySessionManager`           |
| 4 | Project context     | `ResourceLoader`         | `DefaultResourceLoader`            |
| 5 | Policy / cross-cut  | `EventBus` + `ExtensionAPI` | bus + per-extension install hook |

---

## Extension-as-Scenario

A *scenario* is a **composition of atomic extensions**, expressed as data (YAML),
not code. There is no privileged path between built-in and third-party scenarios.

```
                         ┌──────────────────────────────┐
   scenarios/*.yaml ───▶ │ extensions.loader            │ ──▶ list[(module, config)]
   (recipe of atoms)     │ load_scenario(name)          │
                         └──────────────────────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────────┐
                         │ AgentSession.create()        │
                         │ for each (module, cfg):      │
                         │   load_extension → install() │
                         │ register handlers on bus     │
                         └──────────────────────────────┘
```

Each atom is **one Python file** under `extensions/builtin/<name>.py` exporting:

```python
MANIFEST: ExtensionManifest
def install(api: ExtensionAPI, config: dict) -> None: ...
```

A mechanical validator (`extensions.validate`) enforces the §11 single-file
contract: no atom-to-atom imports, no `harness.session` import, allow-listed
imports only. This keeps the surface tiny enough that future agent self-edits
remain verifiable.

### Built-in atoms (`src/agentm/extensions/builtin/`)

| Group        | Atoms |
|--------------|-------|
| Tools        | `tool_read`, `tool_write`, `tool_edit`, `tool_bash`, `tool_grep`, `tool_find`, `tool_ls`, `tool_hypothesis_store`, `tool_submit_plan`, `tool_trajectory_loader` |
| Prompt/skill | `system_prompt`, `prompt_templates`, `skill_loader`, `claude_commands`, `claude_agents` |
| Compaction   | `micro_compact`, `llm_compaction`, `tool_result_budget` |
| Policy       | `permission`, `tool_filter`, `cost_budget`, `dedup`, `turn_reminder` |
| Observability| `observability`, `trajectory` |
| Misc         | `sub_agent`, `file_mutation_queue` |

### Built-in scenarios (`src/agentm/extensions/scenarios/`)

- `general_purpose.yaml` — read/bash/edit/write coding agent
- `plan_mode.yaml` — read-only planning before execution
- `trajectory_analysis.yaml` — analyze completed trajectories

---

## Session lifecycle

```
AgentSession.create(config)
    │
    ├─ build EventBus, SessionManager, ResourceLoader, ExtensionAPI
    ├─ load_extension(...)  for each (module, cfg)  → emit ExtensionInstallEvent
    ├─ load provider extension last (last registration wins)
    └─ append initial_messages

session.prompt(text)
    │
    ├─ append UserMessage
    ├─ assemble system prompt (context files + skills index)
    ├─ emit before_agent_start  ── handlers may rewrite system prompt
    ├─ AgentLoop.run            ── kernel: stream LLM → dispatch tools → repeat
    │     emits LlmRequestStart/End, ToolCall events on the bus
    ├─ append assistant + tool_result entries
    └─ return updated message list

session.shutdown()
    └─ emit SessionShutdownEvent
```

Every signal — install, LLM request, tool call, mutation, turn summary — flows
through the **same** `EventBus`. The `observability` builtin is a pure subscriber
that writes OTel-flavored JSONL to `<cwd>/.agentm/observability/<trace_id>.jsonl`.

---

## Quick start

```bash
uv sync                               # install
export ANTHROPIC_API_KEY="..."        # or use api_registry
uv run agentm "list files in src/"    # one-shot prompt via CLI
```

Programmatic use:

```python
from agentm.harness import AgentSession, AgentSessionConfig
from agentm.extensions.loader import load_scenario

extensions = load_scenario("general_purpose")
session = await AgentSession.create(AgentSessionConfig(
    cwd=".",
    extensions=extensions,
    provider=("agentm.llm.anthropic", {"model": "claude-sonnet-4-6"}),
))
final_messages = await session.prompt("explain core/kernel/loop.py")
await session.shutdown()
```

---

## Build & development

```bash
uv sync
uv run agentm "..."
uv run pytest
uv run ruff check src/
uv run mypy src/
```

- Python 3.12+, build backend `uv_build`.
- Source layout: `src/agentm/`.
- Entry point: `agentm:main`.

---

## Repository layout

```
src/agentm/
├── cli.py                    # thin presenter
├── core/                     # pure SDK (no I/O, no sessions)
│   ├── kernel/               # AgentLoop, Tool, Message, StreamFn, events
│   ├── compaction/           # llm + branch summarization
│   ├── operations.py         # FileOperations / BashOperations Protocols
│   ├── skills.py             # SKILL.md discovery
│   ├── prompt_templates.py   # /name args expansion
│   ├── edit_diff.py          # multi-edit application
│   ├── frontmatter.py        # YAML frontmatter parser
│   └── path_utils.py · text_truncate.py
├── llm/                      # StreamFn implementations (anthropic)
├── ai/                       # provider/api registry, OAuth, env keys
├── harness/                  # AgentSession, EventBus, SessionManager, ...
└── extensions/
    ├── loader.py · discover.py · validate.py
    ├── builtin/<atom>.py     # one file per atom (§11 contract)
    └── scenarios/<name>.yaml # composition recipes

.claude/
├── designs/                  # active design docs (continuously maintained)
├── plans/   · tasks/         # append-only history
└── index.yaml                # concept relationship graph
```

See `CLAUDE.md` for design-doc workflow and the architect / planner / tdd /
implementer / reviewer agent pipeline.
