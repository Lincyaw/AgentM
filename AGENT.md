# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Verify

All commands go through `uv` — pip/poetry/pipenv are forbidden.

```bash
uv sync                           # install deps (add --all-packages for workspace members)
git config core.hooksPath .githooks  # enable tracked pre-commit hook once per clone/worktree
uv run ruff format src/ tests/     # auto-format
uv run ruff check src/ tests/      # lint
uv run mypy src/                   # type check
uv run pytest --tb=short           # test
agentm lint [paths...]             # project-specific AST linter (AM001-AM025)
```

Run single test: `uv run pytest tests/unit/core/test_foo.py -k 'test_name'`

CI runs on push/PR to `main`: frozen sync → ruff check → mypy → pytest.

## Architecture

The SDK is mechanism, not policy. Policy enters through atoms.

```
presenters   agentm.cli / AgentSession.create (SDK embedding)
atoms        extensions/builtin/ + contrib/scenarios/ + contrib/extensions/
substrate    core/ (abi + runtime + lib) — write-protected by core-manifest.yaml
```

Dependency arrows go down only. `core-manifest.yaml` guards constitutional paths — changes to `core/abi/`, `core/lib/`, `core/runtime/`, and `extensions/validate.py` require deliberate acknowledgment.

## Atom Contract

An atom is a single-file Python module with a `MANIFEST` and an `install(api, config)` function. Atoms may import from `agentm.core.abi`, `agentm.core.lib`, and `agentm.extensions`. They must NOT import from `agentm.core.runtime`, `agentm.core._internal`, or other atoms. The load-time validator rejects violations. Atoms reach runtime subsystems through `AtomAPI` methods and services.

## Code Style

Enforced rules that differ from Python defaults:

- **No stdlib logging** — use `from loguru import logger` (AM024)
- **No `typing.Any`** — use `object`, `JsonValue`, concrete DTOs, or Protocols; suppress with `# code-health: ignore[AM022]` at vendor boundaries (AM022)
- **No bare `dict`** — parameterize or use Mapping/dataclass/Protocol (AM023)
- **`@dataclass(slots=True)`** required on all dataclasses (AM002)
- **No silent exceptions** — every `except` must log or surface; `try/except/pass` and `try/except/continue` flagged by ruff S110/S112 (AM001)
- **No raw `open()` or `subprocess`** in atom files (AM004)
- **No `getattr`/`hasattr`/`setattr`/`delattr`** without ignore comment (AM021)
- **No `isinstance`** unless unavoidable; suppress with a precise `# code-health: ignore[AM025]` at validation/deserialization boundaries (AM025)
- **Cross-layer imports** enforced by AM010/AM017
- **Do not write test case unless user explicitly asks for it** — only write important test cases.

Inline suppression: `# code-health: ignore[AM022]` (line) or `# code-health: ignore-file[AM022]` (file).

Tool schemas: use `agentm.core.lib.pydantic_to_openai_tool_schema`, not hand-written dicts.

## Git Conventions

Branch naming: `prefix/description` (e.g. `feat/add-auth`, `fix/crash-on-empty`, `refactor/trajectory-model`).

Run `git config core.hooksPath .githooks` once per clone/worktree to enable the tracked pre-commit hook. The hook checks staged Python files with ruff format/check, mypy for staged `src/` files, and code-health for staged `src/`/`contrib/` files. Do not skip it with `--no-verify`.

## Workspace Members

This is a uv workspace. Members under `contrib/`:
- `contrib/extensions/policy` — policy-engine
- `contrib/scenarios/harbor` — Harbor external agent adapter

## Key Environment Variables

- `AGENTM_HOME` — runtime state directory (default `~/.agentm`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` — OTel collector (default `http://localhost:4317`)
- Feishu/Lark gateway: `LARK_APP_ID`, `LARK_APP_SECRET`, `LARK_ALLOW_FROM`
- Model API keys configured in `config.toml`, not env vars (see `config.toml.example`)
