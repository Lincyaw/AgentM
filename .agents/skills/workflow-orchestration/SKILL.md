---
name: workflow-orchestration
description: >
  How to write workflow scripts that orchestrate multiple agents in AgentM.
  Covers agent(), parallel(), pipeline(), phase(), return values, and
  atom_config data passing. Trigger when writing a workflow script, choosing
  an orchestration pattern, or when the user says 编排 / workflow / 多 agent /
  并行 / 扇出 / 调度 / 写个 workflow. Also trigger when you catch yourself
  building prompts inside a workflow script — pass data via atom_config
  instead and let the agent build its own prompt.
---

# Writing workflow scripts

A workflow script is async Python in a sandboxed namespace. Available
names: `agent`, `parallel`, `pipeline`, `phase`, `log`, `args`,
`budget`, `json`, plus data builtins (`set`, `dict`, `list`, `sorted`,
`len`, `range`, `str`, `int`, `isinstance`, `enumerate`, `zip`).
No `import`, `open`, `time`, or `random`.

End with `return <value>` — that becomes the tool result.

## agent()

```python
result = await agent(
    "One-line task description.",
    scenario="path/to/agent",
    atom_config={"atom_name": {"key": value}},
)
```

- `scenario` — which manifest to load (e.g. `"verifier/hop"`)
- `atom_config` — structured data passed to the agent's atoms;
  configure **every** atom that needs runtime data, not just your own
- `schema` — JSON schema; synthesizes a finalize tool if the agent
  doesn't have one
- Returns `dict` if the agent has a finalize tool or `schema=`;
  returns `str` otherwise

## parallel()

Run coroutines concurrently, return results in order (barrier):

```python
results = await parallel([check(item) for item in items])
```

## pipeline()

Multi-stage per item, **no barrier** between items (item A can be at
stage 2 while B is still at stage 1):

```python
results = await pipeline(items, stage1_fn, stage2_fn, stage3_fn)
```

## Choosing a pattern

| Shape | Use |
|-------|-----|
| Same operation × N items | `parallel([agent(...) for x in items])` |
| Multi-stage per item | `pipeline(items, s1, s2, ...)` |
| Each round discovers new work | `while queue:` loop + `parallel` |
| Sequential phases | `phase("A"); ...; phase("B"); ...` |
| Re-run later stage on existing data | `if args.get("skip_to_B"):` guard |

## progress

```python
phase("review")       # groups subsequent agents under this label
log("3 items left")   # free-text progress line
```

## budget

```python
budget.total          # token ceiling (None if unset)
budget.spent()        # tokens used so far
budget.remaining()    # tokens left (Infinity if no ceiling)
```

## Key rules

1. **Don't build prompts here.** Pass structured data via `atom_config`.
   The agent's context atom builds the prompt.
2. **`atom_config` is the only data channel.** Don't rely on env vars
   — pass every runtime config explicitly, including for third-party
   atoms.
3. **Return one dict.** No intermediate files. The return value is the
   single output.
4. **Preserve all results.** Keep failures alongside successes.
5. **No `json.loads()`.** `agent()` auto-parses finalize-tool output.
   Check `isinstance(result, dict)`.

## References

For deeper topics, read these when needed:

- `references/patterns.md` — full code examples, agent unit structure
  (context atom + finalize atom + manifest), WorkflowRunner programmatic
  invocation
- `references/design-guide.md` — three-layer architecture, when to
  create agent units, finalize tool vs `schema=`, design smells
