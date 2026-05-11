# AgentM

A pluggable agent framework in Python. The SDK is a **mechanism**; every policy
is a port; every port has a default; every default is a replaceable extension.

See `.claude/designs/pluggable-architecture.md` for the boundary contract.

---

## Architecture at a glance

Three layers, dependency arrows point downward only.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  agentm.cli  /  embedded SDK  /  (future: HTTP, RPC)                     │  presenters
│  thin: load scenario → prompt → print                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  agentm.extensions.builtin / contrib/extensions                          │  atoms
│    one-file atoms — including default policies (operations_local,        │
│    llm_<provider>, resource_writer_git, skills, prompt assembly, …)      │
├──────────────────────────────────────────────────────────────────────────┤
│  agentm.core   (constitution — unreplaceable substrate)                  │
│    abi/      AgentLoop · Tool · Message · StreamFn · events ·            │  pure SDK
│              ExtensionAPI / ExtensionManifest · AgentSessionConfig ·     │
│              ResourceWriter / WriteResult · Catalog Protocols            │
│              → atoms speak in these types                                │
│    lib/      edit_diff · frontmatter · path_utils · text_truncate ·      │
│              stream accumulator                                          │
│              → atoms import as stdlib                                    │
│    runtime/  AgentSession · EventBus impl · SessionManager · catalog/ ·  │
│              extension loader · GitBackedResourceWriter                  │
│              → atoms reach via ExtensionAPI / `api.*` hooks only         │
│    _internal/  reload-time helpers; atoms never touch                    │
└──────────────────────────────────────────────────────────────────────────┘
```

`agentm.core` must be importable in a Jupyter notebook with no CLI and no
filesystem touched. The ABI knows nothing about sessions, files, or scenarios;
the runtime substrate knows nothing about concrete scenarios or UI.

The split inside `core/` is a **visibility** boundary, not a modifiability
boundary — none of `core/` is agent-modifiable. `abi/` is the typed contract
atoms speak in; `lib/` is a stdlib-style shelf of pure functions; `runtime/`
holds the stateful substrate (sessions, catalog, writers) that atoms reach
exclusively through ExtensionAPI services (`api.get_operations()`,
`api.skills`, `api.prompt_templates`, `api.catalog`, `api.compaction`). The
validator rejects any atom that imports `core.runtime.*` or `core._internal.*`
directly.

---

## Five pluggability axes

Every axis is a `typing.Protocol` in `agentm.core`. The harness ships a default;
extensions or users can substitute without forking. The Tool environment axis is
constitution-only in v0: the operations bundle is selected by the harness at
session construction and exposed to atoms via `api.get_operations()`, but atoms
do not register a replacement operations bundle at runtime.

| # | Axis                | Protocol / Port          | Default impl                       |
|---|---------------------|--------------------------|------------------------------------|
| 1 | LLM stream          | `StreamFn`               | `agentm.extensions.builtin.llm_anthropic` |
| 2 | Tool environment    | `Tool` + `*Operations`   | `LocalFileOperations`, `LocalBashOperations` (atoms obtain via `api.get_operations()`) |
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
contract: no atom-to-atom imports, no `harness.session` import, no
`core._internal` import. The allowlist is `core.abi` + `core.lib` +
`harness.{extension,events,session_manager,resource_loader}` + `extensions`
(public surface). Anything stateful is reached through ExtensionAPI services.
This keeps the surface tiny enough that future agent self-edits remain
verifiable.

### Built-in atoms (`src/agentm/extensions/builtin/`)

| Group        | Atoms |
|--------------|-------|
| Tools        | `tool_read`, `tool_write`, `tool_edit`, `tool_bash`, `tool_grep`, `tool_find`, `tool_ls`, `tool_hypothesis_store`, `tool_submit_plan`, `tool_trajectory_loader` |
| Prompt/skill | `system_prompt`, `prompt_templates`, `skill_loader` |
| Compaction   | `micro_compact`, `llm_compaction`, `tool_result_budget` |
| Policy       | `permission`, `tool_filter`, `cost_budget`, `dedup`, `turn_reminder` |
| Observability| `observability`, `trajectory` |
| Misc         | `sub_agent`, `file_mutation_queue` |

### Contrib atoms (`contrib/extensions/`)

Flat-file `*.py` atoms at the top level (e.g. `turn_reminder`) are
auto-discovered alongside builtins under the synthetic module prefix
`_agentm_contrib__<name>`. Nested packages (e.g. `tool_catalog/`, `cc/`)
are opt-in and are mounted explicitly by scenario manifests or
`agentm --extension <dotted.module.path>`.

- `contrib.extensions.cc` (package under `contrib/extensions/cc/`)
  — Claude Code compatibility atoms (`agents`, `commands`, `plugins`)
  (read `~/.claude/{agents,commands,plugins}` and surface them through
  the generic `resources_discover` event).

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
uv run agentm "list files in src/"    # full mode — loads general_purpose scenario
```

Programmatic use (full mode):

```python
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession
from agentm.extensions.loader import load_scenario

extensions = load_scenario("general_purpose")
session = await AgentSession.create(AgentSessionConfig(
    cwd=".",
    extensions=extensions,
    provider=("agentm.extensions.builtin.llm_anthropic", {"model": "claude-sonnet-4-6"}),
))
final_messages = await session.prompt("explain core/abi/loop.py")
await session.shutdown()
```

### Recovery floor

When the autonomy layer is broken (a corrupted atom, a regressing
scenario, a harness bug), use `--no-extensions` to drive the kernel
without any atoms loaded — the agent still launches with provider +
loop and can be steered toward diagnosis:

```bash
uv run agentm --no-extensions "explain core/abi/loop.py"
```

The dependency cone in this mode is `core/abi` + `core/lib` + `llm` +
harness. No tool environment, no skills, no observability — exactly the
"core alone yields a usable agent" floor described in
`.claude/designs/self-modifiable-architecture.md`.

Every full-mode feature (skills, prompt templates, compaction, observability,
permission gates, ...) is opt-in through a scenario recipe or extension config —
nothing is inlined into core. Minimal mode is the irreducible base; everything
else is a layer you choose.

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
├── core/                     # constitution — write-protected
│   ├── abi/                  # ABI surface (atom-facing Protocols + data types)
│   │   ├── loop.py · tool.py · messages.py · stream.py · events.py
│   │   ├── operations.py     # FileOperations / BashOperations Protocols + Operations bundle
│   │   ├── extension.py · session_config.py · resource.py · catalog.py
│   │   ├── skill.py · prompt_template.py · compaction.py
│   ├── lib/                  # pure-function utility shelf (atom imports as stdlib)
│   │   └── edit_diff.py · frontmatter.py · path_utils.py · text_truncate.py · stream.py
│   ├── runtime/              # stateful substrate — atoms reach via ExtensionAPI only
│   │   ├── session.py · session_manager.py · session_factory.py · session_bootstrap.py
│   │   ├── extension.py · atom_reloader.py · resource_writer.py · catalog/
│   └── _internal/            # reload-time helpers (atoms never touch)
├── ai/                       # provider/api registry, OAuth, env keys
└── extensions/
    ├── loader.py · discover.py · validate.py
    ├── builtin/<atom>.py     # one file per atom (§11 contract; includes llm_<provider>)
    └── scenarios/<name>.yaml # composition recipes

.claude/
├── designs/                  # active design docs (continuously maintained)
├── plans/   · tasks/         # append-only history
└── index.yaml                # concept relationship graph
```

See `CLAUDE.md` for design-doc workflow and the architect / planner / tdd /
implementer / reviewer agent pipeline.
