---
name: workflow-orchestration
description: >
  Use when a task is a KNOWN-SHAPE fan-out over many items and you want it run
  as deterministic code in one tool call instead of many turns — e.g. review N
  files, research M sub-questions in parallel, judge/verify K candidates, sweep
  a dataset, map-then-reduce. Trigger when you catch yourself about to dispatch
  several near-identical sub-agents by hand, or when the user says "research X"
  / "compare these" / "go through all of these" / 调研 / 批量 / 并行 / 扇出 /
  写个 workflow / 跑一遍. Do NOT use for exploratory work whose next step depends
  on the previous result (use turn-by-turn tools or dispatch_agent for that).
---

# Workflow orchestration (the `workflow` tool)

The `workflow` tool runs an **async Python** script you write. The script
fans out child agent sessions deterministically; only its return value comes
back to you. Use it to push a repetitive multi-agent plan into code so it runs
in one tool call — cheaply, reproducibly, without polluting your context with
every intermediate result.

## When to reach for it (and when not)

Use `workflow` when the **shape is known before the work starts**:
- fan-out the same operation over a list (review each file, research each
  sub-question, grade each candidate);
- a fixed pipeline (search → read → summarize per item);
- you'd otherwise call `dispatch_agent` 3+ times with near-identical prompts.

Do **not** use it when you must **see each result before deciding the next
step** (open-ended debugging, "investigate and follow the trail"). That is
model-driven work — stay turn-by-turn or use `dispatch_agent`. Common hybrid:
scout inline first to discover the work-list, *then* run a `workflow` over it.

## The SDK (names already in scope — do not import anything)

- `await agent(prompt, *, scenario=None, isolation=None, tool_allowlist=None)`
  → the child's final text. Each call is journaled by `hash(prompt, args)`, so
  re-running the same workflow resumes from cache instead of re-spawning.
- `await parallel([agent(p) for p in items])` → list of results (concurrent;
  a shared semaphore caps real parallelism).
- `await pipeline(items, stage1, stage2, ...)` → run each item through the
  stages independently, **no barrier between items** (item A can be at stage 2
  while B is still at stage 1). Stages are callables (sync or async).
- `budget.total` / `budget.spent()` / `budget.remaining()` → token spend so far
  (a ceiling, when configured, lets you scale depth: loop until budget is low).
- `args` → the JSON payload passed alongside the script (a dict).
- `log(msg)` / `phase(name)` → surface progress (fire-and-forget).

Write normal Python control flow (`for`, `while`, `if`, comprehensions) and
`await` the SDK calls. End the script with `return <value>` — that becomes the
tool result. The namespace has **data builtins only**: no `import`, `open`,
`eval`, `time`, or `random` (the last two are withheld so resume stays
deterministic — don't rely on wall-clock or randomness in the script).

## Minimal examples

Fan-out (map):
```python
findings = await parallel([
    agent(f"Research this sub-question and return 5 cited bullets: {q}")
    for q in args["subquestions"]
])
return findings
```

Pipeline (search → summarize, no barrier):
```python
async def research(q):
    return await agent(f"Use the tinyfish CLI via bash to search the web for "
                       f"'{q}', fetch the top 1-2 sources, and return cited "
                       f"bullet findings. Only state facts grounded in fetched "
                       f"content; do not invent ids or numbers.")
return await pipeline(args["topics"], research)
```

## Workers: what tools they have

A worker (`agent(...)`) defaults to **your own scenario** — a clean, slim
reload of it. So a worker has roughly the tools you have, *minus* the
`workflow` tool itself (no nested workflows). Consequences:
- If workers need to **search the web**, tell them in the prompt to use the
  `tinyfish` CLI through their `bash` tool (it is not a dedicated tool).
- If a worker needs a different toolset, pass `scenario="..."`. Do **not** try
  to slim a worker with `tool_allowlist=[]` — an empty allowlist can starve
  extensions that need their tools (e.g. file editing).
- For untrusted/destructive worker work, pass `isolation="agent_env"` (requires
  the `operations_agent_env` atom in the scenario).

## Make the output trustworthy

Worker models hallucinate — especially specific numbers, ids, and URLs. Two
cheap defenses, both worth adding for research/judging tasks:
- **Constrain the worker prompt**: "only state facts grounded in fetched
  content; cite source URLs; do not invent arxiv ids, stats, or numbers."
- **Adversarial verify stage**: after gathering findings, `parallel` a second
  pass of skeptic agents, each asked to *refute* a finding against its source;
  drop findings the majority cannot confirm. Then synthesize from survivors.

## Scaling depth to budget

```python
results = []
while budget.total and budget.remaining() > 50_000:
    results.append(await agent(next_prompt(results)))
return results
```

## Gotchas

- `await` every `agent` / `parallel` / `pipeline` call — they are coroutines.
- The script's `return` value is the result; a bare last expression is not.
- Re-running an identical workflow hits the journal and spawns nothing — change
  a prompt to force fresh work.
- Errors in the script surface as the tool error; fix and call again.
