# AGENTS.md

Index for Codex working in this repo. Pointers, not exhaustive docs —
follow links for detail.

## Project

AgentM — pluggable agent framework. Python 3.12+, `uv` only. The SDK is a
mechanism; every policy is a replaceable atom. Boundary contract in
`.claude/designs/pluggable-architecture.md`.

## CLI

- `agentm -p "<prompt>"` — one-shot prompt (default scenario `chatbot`).
- `agentm` (no args) — show help and subcommand list.
- `agentm contrib sync [--mode copy|symlink] [--overwrite]` — install bundled
  or checkout contrib scenarios/extensions into `~/.agentm/contrib/` so
  external `agentm -p` use does not depend on the source checkout.
- `agentm trace …` — query session traces from ClickHouse (default
  backend) or local JSONL. Subcommands: `messages` · `turns` · `tools` ·
  `chats` · `info` · `index` · `spans` · `logs` · `stats` · `usage`.
  Always use `agentm trace` to inspect trajectories — never hand-parse
  JSONL or artifacts. Local JSONL fallback lives in
  `$AGENTM_HOME/observability/` (default `~/.agentm/observability/`; override
  with `AGENTM_OBSERVABILITY_DIR`).
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
  Codex plugins). Respects `$AGENTM_HOME` override. Use
  `agentm contrib sync` to materialize bundled/source contrib resources there;
  packaged portable scenarios (`chatbot`, `local`, `minimal`) are also a
  fallback when no home/source checkout scenario exists.

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

- Do **not** add any new test case by default. Before writing a new test,
  first ask the user and receive explicit confirmation that the behavior is
  load-bearing enough to lock down. Trivial, ad hoc, or shape-only tests are
  forbidden.

| Position | Why load-bearing |
|---|---|
| Constitution boundary (`is_constitution_path`, manifest reload) | Agent self-modifies kernel |
| Atom hash determinism (`compute_atom_hash`) | Evidence attribution corrupt |
| Active-set fingerprint pairing | Observation can't link to atom version |
| Catalog freeze idempotence | Catalog state untrustworthy |
| Indexer rebuild idempotence | Evolution evidence drifts |
| Transactional reload atomicity | Live agent in inconsistent state |
| §11 extension contract validator | Bad atoms slip into catalog |

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
`src/agentm/gateway/`), `contrib/gateway-peers/{feishu,weixin}/src`,
`contrib/extensions/llmharness/src`, `contrib/scenarios/rca/src` — and
runs mypy per workspace from each member's root (per-package overrides).
The Go terminal peer lives at `contrib/gateway-peers/terminal-go/` and is
checked with `go test ./...`. For sweeping changes, mirror that scope locally.

## Iteration tracking

- `progress.tsv` — dev-loop keep/discard decisions + metric values.
- `decisions.md` — long-horizon autonomous decisions (L2+).

## Autonomy level: high

(Per `/autoharness:long-horizon`. Decide through L4 autonomously, log in
`decisions.md`, flag L4 entries with `[flagged]` for post-hoc review.
Self-merge small / low-risk PRs (docs, single-atom cleanups, polish) once
CI is green and any boundary review is clean. Escalate to the user only
for: large/architectural PRs (hand off for review before merge),
strategic drift (north-star changes, scope creep), access/credentials,
or genuinely ambiguous requirements that research cannot resolve.)

## Conventions

- **Language**: code, comments, commits, design docs in English;
  conversation in Chinese.
- **No SDK / scenario conflation**: scenario-specific logic never inside
  `agentm.core`.
- **§11 atom contract**: enforced by `extensions.validate`.
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

Three axioms govern all work. Fall back to these when a skill's instructions
don't cover a situation:

1. **Quality over quantity** — a few things done well beats many done poorly. Applies to tests, observations, skills, code, docs, experiments, ideas. If you can't say why each item exists, there are too many.
2. **Surface problems early** — fail fast, validate before investing, outline before drafting. Never hide complexity to make something look simpler.
3. **Deliberate execution** — every decision traceable to a reason. Understand before acting; validate manually before automating; measure before optimizing; consider removing before adding.

Full text: `/autoharness:guide`.

## North-star targets

1. **Spec coverage** — active requirements at `status: tested` (currently:
   7/26 = 26.9% on 2026-06-28 after stale path cleanup).
   Measure: `project-index.yaml` status counts.
   Mechanism: script.

2. **Test health** — Python suite and terminal-go suite pass; every
   `implemented` requirement should have tests.
   Measure: `uv run pytest --tb=short` and
   `cd contrib/gateway-peers/terminal-go && go test ./...`.
   Mechanism: script.

3. **Index integrity** — autoharness index validation reports 0 violations
   (currently: 0 violations on 2026-06-28).
   Measure: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/validate_index.py project-index.yaml`.
   Mechanism: script.

4. **Code health** — lint and type checks stay clean across the Python
   substrate and checked contrib peers.
   Measure: CI lint/type commands listed below.
   Mechanism: script.

Secondary: simple code mapping cleanly to requirements > clever abstractions
serving five.

## Dev-loop stages

| Stage | Command | Notes |
|---|---|---|
| Lint | `uv run ruff check src/ contrib/gateway-peers/feishu/src contrib/gateway-peers/weixin/src contrib/extensions/llmharness/src contrib/scenarios/rca/src` | Run after code changes in the touched scope; mirror CI for sweeping changes. |
| Type check | `uv run mypy src/` | Root package check. |
| Type check peers | `cd contrib/gateway-peers/feishu && uv run mypy src/agentm_feishu`; `cd contrib/gateway-peers/weixin && uv run mypy src/agentm_weixin`; `cd contrib/extensions/llmharness && uv run mypy src/llmharness`; `cd contrib/scenarios/rca && uv run mypy src/rca src/rca_eval` | Run from each workspace member root so package-local config applies. |
| Python tests | `uv run pytest --tb=short` | Default excludes `ui`; slow/real-provider tests are opt-in. |
| Terminal peer tests | `cd contrib/gateway-peers/terminal-go && go test ./...` | Go peer is separate from the Python workspace. |
| Index validation | `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/validate_index.py project-index.yaml` | Must stay clean; stale path references are regressions. |

## Observation setup

- **Automated**: CI-style lint, mypy, pytest, terminal-go `go test`, and
  project-index validation. Keep commands deterministic and run the narrowest
  touched scope first, then the broader gate before declaring done.
- **Agent**: periodically review boundary isolation, requirement-code mapping,
  test necessity, and whether new abstractions preserve the pluggability axes.
- **Human**: ask before adding new tests, before L4+ scope changes, and before
  strategic/north-star changes.
- **Priority**: boundary contract and CI health first; index integrity improves
  opportunistically as affected requirements are touched.

## Project conventions

- `uv` only for Python dependency and command execution.
- Python 3.12+ for the SDK and Python peers; terminal-go is a separate Go peer.
- Scenario-specific logic never goes into `agentm.core`.
- New tests require explicit confirmation that the behavior is load-bearing.
- For identity-affecting changes (atoms, kernel, catalog), run an E2E prompt
  in a sandbox repo and inspect it via `agentm trace`.

## Requirements index (MANDATORY)

This project uses `project-index.yaml` as the single source of truth for all
requirements. Every code change must keep the index synchronized:

1. **Before implementing**: find the matching requirement in `project-index.yaml`. If none exists, add one first.
2. **After implementing**: update the requirement's `code` paths and set `status: implemented`.
3. **After adding tests**: update the requirement's `tests` paths and set `status: tested`.
4. **After refactoring**: update any affected `code`/`tests` paths if files were moved or renamed.
5. **Never skip**: a code change without the corresponding index update is incomplete work.

Validate with: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/validate_index.py project-index.yaml`.

## Active skills

- `/autoharness:guide` — methodology briefing.
- `/autoharness:dev-loop` — implement → test → vibe → review → measure.
- `/autoharness:north-star` — define and track optimization targets.
- `/autoharness:long-horizon` — autonomous decisions with escalation ladder.
- `/autoharness:existing-project` — recover and maintain `project-index.yaml`.
- `/autoharness:notify` — push iteration reports when configured.
- `/autoharness:skill-feedback` — file issues back to the autoharness plugin.
<!-- auto-harness:end -->

## Related plugins

- **workbuddy** — pipeline monitoring / repo setup / incident handling.
  Repo carries `.github/workbuddy/`; install with
  `/plugin install workbuddy@workbuddy-local`.
