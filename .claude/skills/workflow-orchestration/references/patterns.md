# Workflow patterns and examples

## Agent unit structure

An agent unit is a scenario folder with three files:

```
my_workflow/reviewer/
  manifest.yaml        ← atoms, system prompt, tool budget
  reviewer_context.py  ← reads atom_config → builds prompt → injects via BeforeAgentStartEvent
  reviewer_finalize.py ← registers the finalize tool (ToolTerminate)
```

### Context atom

Reads structured config at install time, builds the domain-specific
prompt, injects it into the session before the agent starts:

```python
def install(api, config):
    context = _build_prompt(
        target=config["target"],
        criteria=config["criteria"],
        prior=config.get("prior_results", []),
    )
    def before_start(event):
        event.system = f"{event.system}\n\n{context}"
    api.on(BeforeAgentStartEvent.CHANNEL, before_start)
```

### Finalize atom

Registers a terminal tool that validates and returns structured output:

```python
return ToolTerminate(
    result=ToolResult(content=[TextContent(
        type="text", text=result.model_dump_json(),
    )]),
    reason="result-submitted",
)
```

### Calling from a workflow script

```python
result = await agent(
    "Review this item.",
    scenario="my_workflow/reviewer",
    atom_config={
        "reviewer_context": {
            "target": item_name,
            "criteria": criteria,
        },
        "data_source": {"data_dir": data_dir},  # third-party atoms need config too
        "reviewer_finalize": {"output_dir": out_dir},
    },
)
# result is a dict (auto-parsed from the finalize tool)
```

## Orchestration patterns

### Fan-out

Same operation across N independent items:

```python
phase("review")
results = await parallel([
    review_item(item) for item in args["items"]
])
passed = [r for r in results if r and r.get("quality") == "pass"]
```

### Pipeline

Multi-stage per item, no cross-item barrier. Item A can be at stage 2
while item B is still at stage 1:

```python
results = await pipeline(
    args["documents"],
    lambda doc: agent("Extract claims.", scenario="extractor", atom_config={...}),
    lambda claims: agent("Verify claims.", scenario="verifier", atom_config={...}),
)
```

### Iterative expansion

Each round discovers new work (graph walk, dependency resolution,
cascading checks):

```python
queue = list(args["seeds"])
visited = set(queue)
while queue:
    batch, queue = queue, []
    results = await parallel([explore(node) for node in batch])
    for node, result in zip(batch, results):
        for neighbor in result.get("discovered", []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### Multi-phase

Sequential stages where later phases consume earlier results:

```python
phase("produce")
drafts = await parallel([draft(topic) for topic in args["topics"]])

phase("review")
reviewed = await parallel([
    agent("Review this draft.", scenario="reviewer", atom_config={...})
    for d in drafts if d
])
```

### Skip-to-phase

Re-run a later stage on existing earlier results (useful for iterating
on the review without re-running production):

```python
if args.get("skip_to_review"):
    drafts = args["existing_drafts"]
else:
    phase("produce")
    drafts = await parallel([...])

phase("review")
# ... review drafts ...
```

## Programmatic invocation

External code (eval harnesses, CLI tools) runs workflows via the
`WorkflowRunner` service:

```python
session = await AgentSession.create(config)
runner = session.get_service("workflow_runner")
result = await runner.run_file("path/to/workflow.py", args)
# result is a dict (auto-parsed from finalize tool) or str
```

## Output structure

Return everything in one dict — no intermediate files:

```python
return {
    "results": results,
    "summary": summary,
    "metadata": {"rounds": round_n, "items_processed": len(items)},
}
```
