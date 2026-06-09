# Workflow design guide

## Three-layer architecture

```
Layer 1: Data preparation (full Python)
    → DB queries, file I/O, API calls → JSON args dict

Layer 2: Workflow script (curated namespace)
    → agent() / parallel() / pipeline() / phase() / log()
    → passes structured data via atom_config

Layer 3: Agent units (scenario folders)
    → manifest.yaml + context atom + finalize atom
    → each agent does one thing, builds its own prompt
```

Layer 1 can't collapse into Layer 2 (sandboxed — no imports, no
filesystem). Layer 2 can't collapse into Layer 1 (loses
journal/resume/budget, requires manual session management).

## When to create an agent unit

Create a separate agent unit (manifest + atoms) when:

- The task needs its own tools or system prompt
- The output must be structured (needs a finalize tool)
- The same role is used across multiple workflows
- Domain logic (prompt construction, validation) is complex enough
  to warrant its own file

If `schema=` on `agent()` is sufficient and no custom tools are needed,
an ad-hoc call with an existing scenario works without a dedicated unit.

## Finalize tool vs schema=

Both produce `dict` returns from `agent()`. Choose based on complexity:

| | Finalize tool (ToolTerminate) | `schema=` |
|---|---|---|
| When | Agent needs validation, SQL checks, retries on bad input | Just need structured JSON output |
| Where | Dedicated finalize atom in the agent unit | Inline on the `agent()` call |
| Control | Full — custom validation logic, error messages | Schema-only — the engine handles it |

Don't use both on the same agent — they conflict (two terminal tools).

## Smells

- **Prompt strings in the workflow script** → move to a context atom
- **`json.loads()` on agent output** → `agent()` already auto-parses
- **Intermediate files alongside the main output** → return one dict
- **Domain thresholds in the workflow** → put policy in agent atoms
- **Env var fallbacks for child config** → pass via `atom_config`

## Programmatic invocation

External code (eval harnesses, CLI tools) runs workflows via the
`WorkflowRunner` service:

```python
session = await AgentSession.create(config)
runner = session.get_service("workflow_runner")
result = await runner.run_file("path/to/workflow.py", args)
```
