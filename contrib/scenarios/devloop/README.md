# devloop — DevOps control loop

Requirement → spec → design review → test writing → development →
test verification (with retry) → code review.

## Quick start

```bash
# Run in the target repo directory
cd /path/to/your/project

agentm workflow run /path/to/AgentM/contrib/scenarios/devloop/eval/devloop_workflow.py \
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

## What it does

1. **Spec** — generates a structured spec with interfaces and acceptance criteria
2. **Design review** — validates every AC is testable; revises if rejected
3. **Test writing** — writes test files from the spec (before implementation)
4. **Development** — implements the code to pass the tests
5. **Test verification** — runs the test suite; on failure, feeds errors back to the developer and retries (up to `max_rounds`)
6. **Code review** — verifies implementation against spec (skippable)

## Agent units

- `coder/` — coding agent with file tools + bash, used for test writing, implementation, test running, and code review

## Validate without running

```bash
agentm workflow validate contrib/scenarios/devloop/eval/devloop_workflow.py
```
