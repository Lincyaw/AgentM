# devloop — DevOps control loop

Requirement → spec → design review → test writing → development →
test verification (with retry) → code review.

A self-contained AgentM workflow template. Fork this as a starting point
for your own development automation.

## Structure

```
devloop/
  pyproject.toml             ← uv project, agentm as library dependency
  agents/
    coder/                   ← coding agent: file_tools + bash + skills
      manifest.yaml
      devloop_context.py
  skills/                    ← domain skills (evolve per project)
    coding-guidelines/SKILL.md
  workflow/
    types.py                 ← Pydantic models (args + structured output)
    prompts.py               ← prompt construction
    devloop_workflow.py      ← pure orchestration
```

## Quick start

```bash
cd contrib/scenarios/devloop
uv sync

# Run against a target project directory
agentm workflow run workflow/devloop_workflow.py \
  --args '{"requirement": "Implement a Stack class with push, pop, peek, is_empty, and size methods."}' \
  --model doubao \
  --cwd /path/to/your/project
```

## Args

Defined in `workflow/types.py` as `DevloopArgs`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `requirement` | str | *required* | Natural-language feature requirement |
| `language` | str | `"python"` | Target language |
| `test_framework` | str | `"pytest"` | Test framework |
| `max_rounds` | int | `3` | Max develop+test iterations |
| `skip_review` | bool | `false` | Skip final code review stage |
| `agent_timeout_seconds` | float | `900` | Per-agent wall-clock timeout |

## Pipeline

1. **Spec** — generates a structured `ImplementationSpec` with interfaces and acceptance criteria
2. **Design review** — validates every AC is testable; revises if rejected
3. **Test writing** — writes test files from the spec (TDD: tests before implementation)
4. **Development** — implements the code to pass the tests
5. **Test verification** — runs the test suite; on failure, feeds errors back and retries
6. **Code review** — verifies implementation against spec (skippable)

## Skills

The `skills/` directory contains domain-specific knowledge that the coder
agent loads on demand. Skills evolve per project — add new ones as you
discover patterns, remove ones that don't help.

Example: after repeated Playwright failures, create
`skills/playwright-setup/SKILL.md` with the startup configuration that
works for your stack.

## Customization

- **Add agent types**: new directory under `agents/` with a `manifest.yaml`
- **Add skills**: new directory under `skills/` with a `SKILL.md`
- **Change prompts**: edit `workflow/prompts.py`
- **Change pipeline**: edit `workflow/devloop_workflow.py`
- **Change output schemas**: edit `workflow/types.py`
