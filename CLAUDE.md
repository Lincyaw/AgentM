# CLAUDE.md

Index for Claude Code working in this repo. Pointers, not exhaustive docs —
follow links for detail.

## Project

AgentM — pluggable agent framework. Python 3.12+, `uv` only. The SDK is a
mechanism; every policy is a replaceable atom. Boundary contract in
`.claude/designs/pluggable-architecture.md`.

## CLI

- `agentm -p "<prompt>"` — one-shot prompt (default scenario `chatbot`).
- `agentm` (no args) — show help and subcommand list.
- `agentm trace …` — query session traces from ClickHouse (default
  backend) or local JSONL. Subcommands: `messages` · `turns` · `tools` ·
  `chats` · `info` · `index` · `spans` · `logs` · `stats` · `usage`.
  Always use `agentm trace` to inspect trajectories — never hand-parse
  JSONL or artifacts.
- `agentm gateway --bind …` — single-process gateway subcommand: holds
  all chat sessions in memory and serves chat-client peers over the v2
  wire protocol (`.claude/designs/single-process-gateway.md`).
- Chat-client peer CLIs (separate binaries, vendor-SDK isolation only):
  `agentm-terminal`, `agentm-feishu`, `agentm-weixin`.
- Shared `AGENTM_*` env namespace; `.env` autoloaded. Long-lived model
  settings can live in `~/.agentm/config.toml` instead of env vars
  (`$AGENTM_HOME/config.toml` overrides the directory). Precedence:
  CLI flag > shell env / `.env` > `config.toml` profile/default_model >
  provider default. Per-CLI prefixes (`AGENTM_GATEWAY_*`) are **not**
  supported. Model profiles live in `~/.agentm/config.toml`
  (`$AGENTM_HOME/config.toml` overrides the directory).
  Select a profile with `agentm --model <name>`; if `default_model` is set,
  no env vars are needed for the default provider/model/key. The
  `reasoning_effort` convenience knob also has a CLI flag
  `--reasoning-effort` and env `AGENTM_REASONING_EFFORT` (precedence: flag >
  env > config.toml profile); it maps into `extra_body` per provider, any
  user-set `extra_body` key winning. Run `<cli> --help` for flags.
- Optional extra: `uv sync --extra agent-env` installs `arl-env` for the
  `operations_agent_env` atom (ARL-sandboxed Operations).

### WeChat (微信) peer

Personal WeChat gateway peer via iLink Bot API
(`contrib/gateway-peers/weixin/`). Subcommands: `login` (QR scan),
`run` (connect to existing gateway), `serve` (supervisord: gateway +
adapter in one command), `list` (show accounts). `serve` uses
supervisord for auto-restart and log rotation (`~/.agentm/weixin/logs/`).

### Trace debugging combos

Default backend is ClickHouse; traces are queryable by `--session <id>`
across the full history. A logical trace spans multiple sessions — one
root + N spawned children. Composition pattern:

- `agentm trace index` — selection layer: emits one identity row per
  session (`{session_id, trace_id, parent_session_id, purpose, scenario}`).
- `agentm trace messages --session <id>` — full conversation trajectory.
- `agentm trace tools --session <id> [--tool <name>]` — tool calls with
  args + results joined.
- Idiom: find a child session from the workflow delivery artifact, then
  `agentm trace tools --session <child_id> --format ndjson | jq …`.

## Repo exploration

- `agentm list-extensions [--source builtin|contrib|home|user|all] [--filter X]`
- `ls contrib/scenarios/` — names usable as `--scenario <name>`
- `ls src/agentm/extensions/builtin/` — builtin atoms (one file per atom)
- `ls contrib/extensions/` — third-party atoms (flat files auto-discover;
  nested packages need `--extension <dotted.path>`)
- `.claude/index.yaml` — design-concept graph; `.claude/designs/` — concept docs
- `CONTEXT.md` — project glossary; check here when an unfamiliar term appears.
- `core-manifest.yaml` — constitution layer (kernel-singleton declarations); read-only.

### Finding existing capabilities

Before building something new, check what already exists. Read the
docstring at the top of each atom file — it describes what the atom does
and what tools/events it registers.

- **Tools**: `grep -rn "register_tool" src/agentm/extensions/builtin/` —
  every tool registration with its name.
- **Events**: `grep "CHANNEL.*=" src/agentm/core/abi/events.py` — all
  kernel event channels.
- **Key builtin atoms by capability area**:
  - Orchestration: `workflow.py` (Python script orchestration over child
    sessions — `agent()`, `parallel()`, `pipeline()`), `sub_agent.py`
    (spawn/check/wait/abort workers).
  - File ops: `file_tools.py` (`read`/`write`/`edit`/`glob`/`grep`).
  - Execution: `tool_bash.py`, `background_exec.py`
    (background process management).
  - Monitoring: `monitor.py` (wakeup scheduling, monitors).
  - Knowledge: `skill_loader.py` (SKILL.md discovery + `load_skill`
    tool), `memory.py` (persistent key-value memory).
  - Data: `query_tools.py` (DuckDB queries), `artifact_store.py`.
  - Self-modification: `atom_management.py` (`install_atom` /
    `reload_atom`), `structured_output.py` (`submit_result`).
  - Safety: `tool_bash_guard.py`, `tool_filter.py`, `tool_result_budget.py`,
    `tool_error_messages.py`, `permission.py`, `dedup.py`.
  - LLM: `llm_openai.py`, `llm_anthropic.py`, `llm_compaction.py`,
    `thinking_retry.py`, `retry_policy.py`.
  - Observability: `observability.py`, `otlp_export.py`, `cost_budget.py`,
    `loop_budget.py`.
  - UI: `wire_driver.py` (gateway wire protocol), `tui_snapshot.py`,
    `slash_commands.py`.

## Architecture

```
presenters: agentm.cli  /  embedded SDK
atoms:      src/agentm/extensions/builtin/  +  contrib/extensions/  +  ~/.agentm/contrib/extensions/
substrate:  agentm.core/  (abi · runtime · lib — write-protected)
```

- Atoms reach stateful subsystems only through `ExtensionAPI` services;
  `extensions.validate` rejects direct `core.runtime.*` imports.
- Five pluggability axes are `Protocol`s in `core.abi`, registered by atoms
  via `api.register_*`. See `.claude/designs/pluggable-architecture.md`.

## Extensions & scenarios

- **Builtin atom**: `src/agentm/extensions/builtin/<name>.py` — one file,
  exports `MANIFEST` + `install(api, config)`. §11 contract: no
  atom-to-atom imports, no `core.runtime.*`, no `core._internal`.
- **Scenario**: YAML at `contrib/scenarios/<name>/manifest.yaml`, selected
  via `--scenario <name>`. Default is `chatbot`.
- **contrib/extensions/**: flat `<name>.py` auto-discovers; nested packages
  mount via `--extension <dotted.path>` and are **not** scenarios.
- **Home contrib**: `~/.agentm/contrib/extensions/<name>.py` and
  `~/.agentm/contrib/scenarios/<name>/manifest.yaml` — user-installed
  atoms and scenarios that work from pip-installed wheels (similar to
  Claude Code plugins). Respects `$AGENTM_HOME` override.

## Design docs (`.claude/`)

- `index.yaml` — concept graph; keep in sync on every concept change.
- `designs/<concept>.md` — continuously maintained; one per live concept.
  Only current design is kept — superseded designs are deleted, not archived
  (git history is the archive).

Concept-change flow: update the design doc → check `index.yaml`
`related_concepts` → propagate → update `index.yaml`.

## Testing

Quality over quantity. A test exists only to protect a **fail-stop
position**:

| Position | Why load-bearing |
|---|---|
| Constitution boundary (`is_constitution_path`, manifest reload) | Agent self-modifies kernel |
| Atom hash determinism (`compute_atom_hash`) | Evidence attribution corrupt |
| Active-set fingerprint pairing | Observation can't link to atom version |
| Catalog freeze idempotence | Catalog state untrustworthy |
| Indexer rebuild idempotence | Evolution evidence drifts |
| Transactional reload atomicity | Live agent in inconsistent state |
| extension contract validator | Bad atoms slip into catalog |

`pytest` markers: `ui` (Textual TUI) and `slow` (real-LLM E2E,
minutes-long) — both opt-in.

**E2E** = drive `agentm` with NL prompts on a sandbox repo and verify
through `agentm trace`. Never substitute SDK-internal assertions
(`session._tools`, `session._apis`). Lock real bugs down with a
stub-provider integration test.

## Dev loop

After every change:

```bash
uv run ruff check <files>
uv run mypy <files>
uv run pytest --tb=short
```

Prefer targeted `# type: ignore[attr-defined]` over broad suppression.
For identity-affecting changes (atoms, kernel, catalog): also run an E2E
prompt against a sandbox repo and inspect the trace.

CI lints/types a broader scope — `src/` (gateway lives at
`src/agentm/gateway/`), `contrib/gateway-peers/{terminal,feishu,weixin}/src`,
`contrib/extensions/llmharness/src`, `contrib/scenarios/rca/src` — and
runs mypy per workspace from each member's root (per-package overrides).
For sweeping changes, mirror that scope locally.

## Iteration tracking

- `progress.tsv` — dev-loop keep/discard decisions + metric values.
- `decisions.md` — long-horizon autonomous decisions (L2+).

## Conventions

- **Language**: code, comments, commits, design docs in English;
  conversation in Chinese.
- **No SDK / scenario conflation**: scenario-specific logic never inside
  `agentm.core`.
- **atom contract**: enforced by `extensions.validate`.
- **No preset enums for subjective fields** — free-text + LLM-decided.
- **Auto-commit awareness**: `agentm` auto-commits during sessions; run
  E2E in a sandbox, never on `main`.
- **No destructive git**: `git reset --hard`, `git checkout -- .`,
  `git clean -f` and any other command that discards uncommitted work are
  **forbidden**. Always use recoverable alternatives (`git stash`,
  `git revert`, `git reset --soft`, worktree isolation). Uncommitted
  changes in the working tree may contain the user's in-progress work.
- **No `path.resolve().parent`** — chaining `.resolve()` before
  `.parent` obscures intent and silently follows symlinks. For
  `__file__` (absolute in Python 3.12+), drop `resolve()`:
  `Path(__file__).parent` / `Path(__file__).parents[N]`. For dynamic
  paths that genuinely need symlink resolution, split into two steps:
  `real = p.resolve(); real.parent / …`.

## Requirements index

`project-index.yaml` (repo root) is the single source of truth for product
requirements. Every code change keeps `code` / `tests` paths and `status`
in sync — many entries currently have stale paths from the harness-collapse
migration (e.g. `src/agentm/harness/`, `src/agentm/llm/`); fix them as you
touch the affected requirements. Validation runs through the autoharness
skill. Distinct from `.claude/index.yaml` (design-concept graph).

<!-- auto-harness:begin -->
## Core principles

1. **Quality over quantity** — a few things done well beats many done poorly.
2. **Surface problems early** — fail fast, validate before investing, outline before drafting.
3. **Deliberate execution** — every decision traceable to a reason.

Full text: `/autoharness:guide`.

## North-star targets

| Target | Measure |
|---|---|
| Spec coverage | fraction of `project-index.yaml` requirements at `status: tested` |
| Test health | `uv run pytest --tb=short` pass rate; every `implemented` requirement has tests |
| Index integrity | `validate_index.py` reports 0 violations |
| Code health | `uv run mypy src/` + `uv run ruff check src/` both clean |

Secondary: simple code mapping cleanly to requirements > clever abstractions
serving five.

## Active skills

| Skill | Purpose |
|---|---|
| `/autoharness:guide` | methodology briefing |
| `/autoharness:dev-loop` | implement → test → vibe → review → measure |
| `/autoharness:north-star` | define and track optimization targets |
| `/autoharness:long-horizon` | autonomous decisions with escalation ladder |
| `/autoharness:existing-project` | recover `project-index.yaml` from current code/docs |
| `/autoharness:notify` | push iteration reports (not yet configured) |
| `/autoharness:skill-feedback` | file issues back to the autoharness plugin |
<!-- auto-harness:end -->