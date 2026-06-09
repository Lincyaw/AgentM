# devloop — DevOps control loop

Requirement → spec → design review → test writing → development →
test verification (with retry) → code review.

## Structure

```
devloop/
  agents/                    ← agent unit definitions (extensible)
    coder/                   ← coding agent: file_tools + bash
      manifest.yaml
      devloop_context.py
  workflow/                  ← workflow definition
    types.py                 ← TypedDict + JSON Schemas
    prompts.py               ← prompt construction functions
    devloop_workflow.py      ← pure orchestration logic
```

## Quick start

```bash
cd /path/to/your/project

agentm workflow run /path/to/AgentM/contrib/scenarios/devloop/workflow/devloop_workflow.py \
  --args '{"requirement": "Implement a Stack class with push, pop, peek, is_empty, and size methods."}' \
  --model doubao
```

## Args

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `requirement` | str | *required* | Natural-language feature requirement |
| `language` | str | `"python"` | Target language |
| `test_framework` | str | `"pytest"` | Test framework |
| `max_rounds` | int | `3` | Max develop+test iterations |
| `skip_review` | bool | `false` | Skip final code review stage |

## Pipeline

1. **Spec** — generates a structured spec with interfaces and acceptance criteria
2. **Design review** — validates every AC is testable; revises if rejected
3. **Test writing** — writes test files from the spec (before implementation)
4. **Development** — implements the code to pass the tests
5. **Test verification** — runs the test suite; on failure, feeds errors back and retries (up to `max_rounds`)
6. **Code review** — verifies implementation against spec (skippable)

## Adding agent types

Add a new directory under `agents/` with a `manifest.yaml`. Reference it
from the workflow as `scenario="devloop/agents/<name>"`.

## Validate without running

```bash
agentm workflow validate contrib/scenarios/devloop/workflow/devloop_workflow.py
```
