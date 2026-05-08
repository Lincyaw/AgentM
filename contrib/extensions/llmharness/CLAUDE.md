# llmharness

Claude Code plugin + Python package + AgentM extension providing async,
non-blocking supervision for the main agent. Hook/event-driven turn stream +
AgentM-backed drift detection.

Lives at `<AgentM-root>/contrib/extensions/llmharness/`. Mounted onto a
running agent via `agentm --extension llmharness.adapters.agentm`
(repeatable; stacks on top of `--scenario` or auto-discovery).

Downstream consumer: **rca-autorl** installs this package via path, e.g.
`pip install -e <AgentM-root>/contrib/extensions/llmharness` (no AgentM Python
dependency required for that path).

Design reference: see `README.md` and (in the rca-autorl repo)
`.doc/designs/llm-harness.md`.

<!-- auto-harness:begin -->
## Core principles

Three axioms govern all work. Fall back to these when a skill's instructions don't cover a situation:

1. **Quality over quantity** — a few things done well beats many done poorly. Applies to tests, observations, skills, code, docs, experiments, ideas. If you can't say why each item exists, there are too many.
2. **Surface problems early** — fail fast, validate before investing, outline before drafting. Never hide complexity to make something look simpler.
3. **Deliberate execution** — every decision traceable to a reason. Understand before acting; validate manually before automating; measure before optimizing; consider removing before adding.

Full text: `/home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/references/principles.md`.

## North-star targets

1. **Spec coverage** — % of active requirements at `tested` status (currently: ~7%, only REQ-006 has a smoke test)
   Measure: `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`
   Mechanism: script

2. **Test health** — 100% pass rate; every `implemented` requirement has a test (currently: 2/2 passing, but most reqs lack tests)
   Measure: `uv run pytest --tb=short`
   Mechanism: script

3. **Index integrity** — 0 violations from `validate_index.py` (currently: 0)
   Measure: `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`
   Mechanism: script

4. **Type cleanliness** — `mypy --strict` clean on `src/llmharness` (currently: unmeasured, run once first)
   Measure: `uv run mypy src/llmharness`
   Mechanism: script

5. **Lint cleanliness** — `ruff check` clean on `src/` and `tests/` (currently: unmeasured)
   Measure: `uv run ruff check src tests`
   Mechanism: script

Secondary: simpler code that maps clearly to a requirement > clever abstractions. This package is depended on by other repos — interface stability matters more than cleverness.

## Dev-loop stages

| Stage | Command | Notes |
|-------|---------|-------|
| Lint | `uv run ruff check src tests` | Run before commit |
| Format | `uv run ruff format src tests` | Run before commit |
| Type check | `uv run mypy src/llmharness` | Strict mode |
| Test | `uv run pytest --tb=short` | Smoke tests must always pass |
| Index validate | `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml` | Run after every code change that touches `src/` |
| Hook smoke | `bash scripts/tick_worker.sh` (with `LLMHARNESS_PROVIDER=rule`) | Manual; verifies the file protocol end-to-end |

## Iteration tracking

- Progress log: `progress.tsv` — dev-loop records keep/discard decisions and metric values
- Decision log: `decisions.md` — long-horizon logs autonomous decisions (L2+)

## Project conventions

- **Package manager**: `uv` only. Never `pip install`. `uv sync --extra dev` for dev env; `uv run <cmd>` to execute.
- **Python**: 3.10+, `src/` layout, type hints required on every function in `src/`. `mypy --strict` is the gate.
- **Lint/format**: `ruff` (no black, no isort, no flake8 — single tool).
- **Hook scripts**: every `scripts/*.sh` must `set -euo pipefail` at the top. No bashisms beyond what shellcheck accepts.
- **AgentM mount point**: this package's adapter (`llmharness.adapters.agentm`) is loaded onto a session via `agentm --extension llmharness.adapters.agentm`. There is no scenario manifest under this directory — keep it as a Python package + extension only.
- **No commits to main**: feature branches only. Use `gh` (HTTPS) for any GitHub operations — never ssh URLs.
- **Schema stability**: `src/llmharness/schema.py` is a public contract for rca-autorl. Breaking changes require bumping `version` in `pyproject.toml` and a note in the requirement description.
- **No silent failures in hooks**: hook scripts may fail-open (return 0) on unrecognized payloads, but only after `parse_hook_payload` returns `None` — never via blanket `try/except`.

## Requirements index (MANDATORY)

This project uses `project-index.yaml` as the single source of truth for all requirements.
Every code change MUST keep the index synchronized:

1. **Before implementing**: find the matching requirement in `project-index.yaml`. If none exists, add one first.
2. **After implementing**: update the requirement's `code` paths and set `status: implemented`.
3. **After adding tests**: update the requirement's `tests` paths and set `status: tested`.
4. **After refactoring**: update any affected `code`/`tests` paths if files were moved or renamed.
5. **Never skip**: a code change without the corresponding index update is incomplete work.

Validate with: `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`

## Active skills

- /autoharness:guide — methodology briefing at session start
- /autoharness:dev-loop — full implement → test → vibe → review → measure cycle
- /autoharness:north-star — keeps targets honest and quantifiable
- /autoharness:new-project — SDD entry point for greenfield work; owns `project-index.yaml`
- /autoharness:long-horizon — autonomy/escalation ladder for self-directed work
<!-- auto-harness:end -->
