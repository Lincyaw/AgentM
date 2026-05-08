# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentM is a pluggable agent framework in Python (v0.1.0). The SDK is a **mechanism**;
every policy is a port; every port has a default; every default is a replaceable
extension. Inspired by [`badlogic/pi-mono`](https://github.com/badlogic/pi-mono);
boundary contract in `.claude/designs/pluggable-architecture.md`.

- **Language**: Python 3.12+
- **Package manager / build backend**: `uv` / `uv_build`
- **Source layout**: `src/agentm/`
- **Entry point**: `agentm:main` (console script)

## Build & Development Commands

```bash
uv sync                                # install deps
uv run agentm "<prompt>"               # full mode (loads general_purpose scenario)
uv run agentm --minimal "<prompt>"     # recovery floor (stdlib tools only)
uv run pytest                          # run tests (excludes nested workspaces, ui)
uv run pytest -m ui                    # Textual TUI tests (opt-in)
uv run ruff check src/                 # lint
uv run mypy src/                       # type check
uv add <package>                       # add a runtime dep
```

After modifying code, **always** lint and type-check the touched files:

```bash
uv run ruff check <changed-files>
uv run mypy <changed-files>
```

For `mypy` issues on dynamic/duck-typed parameters, prefer targeted
`# type: ignore[attr-defined]` over broad suppression.

## Architecture (four layers, dependency arrows down only)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  agentm.cli  /  embedded SDK  /  (future: HTTP, RPC)                     │  presenters
├──────────────────────────────────────────────────────────────────────────┤
│  agentm.harness — AgentSession · EventBus · SessionManager · Loader      │  harness
├──────────────────────────────────────────────────────────────────────────┤
│  agentm.core (constitution — write-protected)                            │  pure SDK
│    abi/        AgentLoop · Tool · Message · StreamFn · events · ops      │
│    lib/        edit_diff · frontmatter · path_utils · text_truncate      │
│    _internal/  default impls + loaders (atoms reach via ExtensionAPI)    │
├──────────────────────────────────────────────────────────────────────────┤
│  agentm.llm   (StreamFn implementations: anthropic, openai-compatible)   │  provider
└──────────────────────────────────────────────────────────────────────────┘
```

`agentm.core` must import in a Jupyter notebook with no harness, no CLI, no
filesystem touched. Atoms reach stateful subsystems exclusively through
ExtensionAPI services (`api.get_operations()`, `api.skills`,
`api.prompt_templates`, `api.catalog`, `api.compaction`). The `extensions.validate`
checker rejects atoms that import `core._internal.*` directly.

**Five pluggability axes** (each is a `typing.Protocol` with a default impl):
LLM stream · Tool environment · Session state · Project context · Policy / cross-cut.

## Extension-as-Scenario

A scenario is a *composition of atomic extensions* expressed as YAML data, not
code. There is no privileged path between built-in and third-party scenarios.

- **Built-in atoms**: `src/agentm/extensions/builtin/<name>.py` — one file per atom
  exporting `MANIFEST` and `install(api, config)`. §11 single-file contract:
  no atom-to-atom imports, no `harness.session` import, no `core._internal` import.
- **Built-in scenarios**: `src/agentm/extensions/scenarios/<name>.yaml` —
  `general_purpose`, `plan_mode`, `trajectory_analysis`.

## contrib/ layout

Everything that is **not** SDK core lives under `contrib/`. SDK builtins still
live under `src/agentm/extensions/builtin/` (auto-discovered); `contrib/` is for
opt-in, separately-maintained extras.

```
contrib/
├── extensions/        # third-party-maintained atoms (workspace members)
│   └── llmharness/    # cognitive-audit package: atoms + adapter + tests
└── scenarios/         # scenario manifests (loader entry point)
    ├── plan_mode/
    ├── rca/           # also a workspace member (agentm_rca/)
    └── trajectory_analysis/
```

- `contrib/scenarios/<name>/manifest.yaml` — resolved by `agentm --scenario <name>`.
  Loader at `src/agentm/extensions/loader.py` looks up
  `<cwd>/contrib/scenarios/<name>/manifest.yaml`.
- `contrib/extensions/<name>.py` — flat-file atoms, auto-discovered alongside
  `src/agentm/extensions/builtin/`, registered under synthetic module names
  `_agentm_contrib__<name>`.
- `contrib/extensions/<name>/` — Python packages whose `MANIFEST` makes them
  mountable via `agentm --extension <dotted.module.path>` (repeatable; stacks
  on top of `--scenario` or auto-discovery). **Not** a scenario — don't put
  `manifest.yaml` here.

Don't add scenario-loader logic that walks subdirs blindly — keep the layout
open to nested projects.

## Design Documentation System

All design documents live in `.claude/`. `index.yaml` is the relationship graph
and must always stay in sync.

```
.claude/
├── index.yaml          # concept index — must stay current
├── designs/            # high-level design docs (continuously maintained)
├── plans/              # YYYY-MM-DD-<plan>.md (append-only, never modify)
└── tasks/              # YYYY-MM-DD-<task>.md (append-only, never modify)
```

| Directory | Lifecycle | Naming |
|-----------|-----------|--------|
| `designs/` | continuously maintained | `<concept>.md` |
| `plans/` | append-only | `YYYY-MM-DD-<plan-name>.md` |
| `tasks/` | append-only | `YYYY-MM-DD-<task-name>.md` |

**Change propagation:** when any design concept changes, (1) update the design
doc, (2) query `index.yaml` for `related_concepts`, (3) update affected docs,
(4) update `index.yaml`, (5) if implementation needed, append new plan/task.

**Cross-references** use relative paths:
`[concept](other-concept.md)`, `[plan](../plans/YYYY-MM-DD-plan.md)`, etc.

## Custom agents (`.claude/agents/`)

| Agent | Role | Model |
|-------|------|-------|
| **architect** | Architecture design, design docs, `index.yaml` upkeep | opus |
| **planner** | Break designs into executable plans/tasks | sonnet |
| **tdd** | Write tests first (RED), guide TDD cycle | sonnet |
| **implementer** | Execute plans, write code following designs | sonnet |
| **reviewer** | Verify implementation matches design | opus |

Workflow: `architect → planner → tdd → implementer → reviewer` (with feedback
loops back to architect on design issues).

## Custom slash commands (`.claude/commands/`)

`/design` · `/plan` · `/index [show|check|fix]` · `/status` · `/tdd` ·
`/eval [define|check|report]` · `/checkpoint [create|verify|list]` · `/learn`

## Testing philosophy

**Quality over quantity. Only test positions where AgentM's value proposition
fails if broken.** Single-tool happy paths, vendor wiring, utility helpers, and
framework guarantees are NOT core. The fail-stop positions:

| Position | Why load-bearing |
|---|---|
| Constitution boundary (`is_constitution_path`, manifest reload) | Wrong → agent self-modifies kernel |
| Atom hash determinism (`compute_atom_hash`) | Wrong → evidence attribution corrupt |
| Active-set fingerprint pairing | Wrong → observation can't link to atom version |
| Catalog freeze idempotence | Wrong → catalog state untrustworthy |
| Indexer rebuild idempotence | Wrong → evolution evidence drifts |
| Transactional reload atomicity | Wrong → live agent in inconsistent state |
| §11 extension contract validator | Wrong → bad atoms slip into catalog |

New tests outside this list require explicit justification of which fail-stop
they protect. **Test behavior, not structure.** Don't test framework guarantees.

### E2E methodology (load-bearing)

E2E means: drive the agent with natural-language prompts and verify by
inspecting the trajectory. **Do not call SDK / harness internals to "shortcut"
the verification.** Identity bugs only show through `agentm` CLI in →
trajectory out.

Procedure: run `uv run agentm --cwd <sandbox> "<NL prompt>"` against a real
git sandbox; inspect `<sandbox>/.agentm/observability/<trace>.jsonl` for
`emit:tool_call`, `emit:tool_result`, `install:*`, `emit:diagnostic`;
cross-check on-disk state for writes. When a real bug surfaces, lock it down
with a stub-provider integration test so CI catches the regression without
API keys.

<!-- auto-harness:begin -->
## Core principles

Three axioms govern all work. Fall back to these when a skill's instructions don't cover a situation:

1. **Quality over quantity** — a few things done well beats many done poorly. Applies to tests, observations, skills, code, docs, experiments, ideas. If you can't say why each item exists, there are too many.
2. **Surface problems early** — fail fast, validate before investing, outline before drafting. Never hide complexity to make something look simpler.
3. **Deliberate execution** — every decision traceable to a reason. Understand before acting; validate manually before automating; measure before optimizing; consider removing before adding.

Full text: `/home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/references/principles.md`.

## North-star targets

1. **Spec coverage** — fraction of active requirements at `tested` status (currently: unmeasured — index not yet built)
   Measure: `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`
   Mechanism: script

2. **Test health** — pass rate + every implemented requirement has tests (currently: unmeasured)
   Measure: `uv run pytest --tb=short`
   Mechanism: script

3. **Index integrity** — `validate_index.py` reports 0 violations (currently: n/a — index not yet built)
   Measure: `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`
   Mechanism: script

4. **Code health** — 0 mypy errors, 0 critical ruff violations on `src/` (currently: unmeasured)
   Measure: `uv run mypy src/ && uv run ruff check src/`
   Mechanism: script

Secondary: simpler code that maps clearly to requirements > clever abstractions
that serve five requirements but belong to none.

## Dev-loop stages

| Stage | Command | Notes |
|-------|---------|-------|
| Test | `uv run pytest --tb=short` | Run after every change; fail-stop tests only (see "Testing philosophy" above) |
| Type check | `uv run mypy src/` | Targeted ignores OK on duck-typed args |
| Lint | `uv run ruff check src/` | |
| Index validate | `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml` | Run once `project-index.yaml` exists |
| E2E | `uv run agentm --cwd <sandbox> "<NL prompt>"` + trajectory inspection | For identity-affecting changes (atoms, kernel, catalog) |

## Iteration tracking

- Progress log: `progress.tsv` — dev-loop records keep/discard decisions and metric values
- Decision log: `decisions.md` — long-horizon logs autonomous decisions (L2+)
- Design log: `.claude/{plans,tasks}/` — append-only history of design-driven work

## Project conventions

- **Language rule**: code, comments, commits, design docs are **English**;
  conversation with the user is **Chinese**.
- **Package manager**: `uv` only — never `pip`, `poetry`, `pipenv`.
- **Python**: 3.12+ required; build backend `uv_build`.
- **No SDK / scenario conflation**: never put scenario-specific logic inside
  `agentm.core`. Core is the mechanism; scenarios are compositions of atoms.
- **§11 atom contract**: each builtin atom is one file, no atom-to-atom imports,
  no `harness.session` import, no `core._internal` import. Validator enforces.
- **contrib/ layout**: scenarios are manifest-only YAML; flat-file atoms
  auto-discover; nested packages mount via `--extension`. Don't extend the
  scenario loader to walk subdirs blindly.
- **Design doc workflow**: changes propagate via `index.yaml`. `designs/`
  is continuously updated; `plans/` and `tasks/` are append-only.
- **Tests gate on fail-stop positions** (see "Testing philosophy"). Reject
  PRs that add tests for framework guarantees.
- **E2E by trajectory** — never assert on `session._tools` / `session._apis`
  as a substitute for CLI + JSONL inspection.
- **No preset enums for subjective dimensions** — use free-text + LLM-decided
  for relationship/status/classification fields where reasonable
  interpretations differ.
- **Auto-commit awareness**: `agentm` auto-commits during sessions. Run E2E
  in a sandbox repo, never on `main`.

## Requirements index (MANDATORY once `project-index.yaml` exists)

This project will use `project-index.yaml` as the single source of truth for
all requirements. The index does **not** yet exist — bootstrap it via
`/autoharness:existing-project` (recover requirements from current code/docs).

Once it exists, every code change MUST keep the index synchronized:

1. **Before implementing**: find the matching requirement in `project-index.yaml`. If none exists, add one first.
2. **After implementing**: update the requirement's `code` paths and set `status: implemented`.
3. **After adding tests**: update the requirement's `tests` paths and set `status: tested`.
4. **After refactoring**: update any affected `code`/`tests` paths if files were moved or renamed.
5. **Never skip**: a code change without the corresponding index update is incomplete work.

Validate with: `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`

> Note: this index is at the repo root and tracks **product requirements**.
> Distinguish from `.claude/index.yaml`, which is the **design-concept**
> relationship graph. Both coexist.

## Active skills

- /autoharness:guide          — methodology briefing at session start
- /autoharness:dev-loop       — full dev cycle: implement → test → vibe → review → measure
- /autoharness:north-star     — define and track optimization targets
- /autoharness:long-horizon   — autonomous decision-making with escalation ladder
- /autoharness:notify         — push iteration reports to Telegram/Feishu/email (not yet configured)
- /autoharness:existing-project — recover `project-index.yaml` from current code/docs
- /autoharness:skill-feedback — file issues/PRs back to the autoharness plugin when a skill misfires
<!-- auto-harness:end -->

## Related plugins

- **workbuddy** — pipeline monitoring / repo setup / incident handling.
  This repo has `.github/workbuddy/` (config + agents + workflows), so
  workbuddy is recommended:
  `/plugin install workbuddy@workbuddy-local`
  (requires the marketplace registered first).
