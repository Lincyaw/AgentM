---
description: Archive experiment results to the Notes repo (log entry + experiment data).
argument-hint: "<topic>   e.g. 'goal oversight cow pass@8 results'"
---

Record the current experiment session to the Notes repo at
`../Notes/research/ongoing/llm-as-harness/`.

## What to write

Create or update two things:

### 1. Log entry

File: `log/YYYY-MM-DD-<topic-slug>.md` (use today's date).

Required sections — every log entry MUST include ALL of these:

```markdown
---
title: "<descriptive title>"
date: YYYY-MM-DD
tags: [research, llm-as-harness, log, ...]
---

# <Title>

## Context
Why this experiment was run. Link to prior entries that motivated it.

## Environment
- **Machine**: hostname / alias (e.g. `exp-remote`)
- **Cluster**: which ARL cluster, namespace, gateway URL
- **Registry**: which container registry was used
- **Model**: model name and provider (e.g. `doubao-seed-2.0-pro via litellm`)
- **Repo commit**: AgentM commit hash at time of experiment
- **Working directories**: relative paths for results, logs, and configs

## Method
What was done. Be specific enough that someone on a different machine
can reproduce from scratch.

## Results
Tables, numbers, comparisons. Raw data, not just conclusions.

## Reproduction steps
Step-by-step commands to reproduce this experiment from a clean state.
Include:
- Environment variables needed
- Exact CLI commands (copy-pasteable)
- Expected output / how to verify success
- Any prerequisites (images to pull, clusters to configure, data to download)

## Data
Where to find raw outputs:
- Score files, logs, session IDs
- Use relative paths from the relevant repo root or experiment directory
- Note which machine the paths are on if not obvious

## Findings
Interpretation of the results. What was learned, what's next.
```

### 2. Experiment data (if new experiment)

If this is a new experiment (not updating an existing one), also create or
update files under `experiments/<experiment-name>/`:
- `data/<descriptive-name>.json` — structured results summary
- `protocol.md` — update with new results if the experiment already exists

### 3. Update CONTEXT.md

Add or update the status line in `CONTEXT.md` for this experiment.

## Guidelines

- ***REMOVED***
**Relative paths**: use paths relative to the relevant repo root or experiment directory, prefixed with the machine alias only when the data lives on a remote host.
- **Session IDs**: record AgentM session IDs for trajectory queries
  (`agentm trace messages --session <id>`).
- **No stale references**: if prior log entries reference data that has moved
  or been superseded, note that in the new entry.
- **Commit and push**: git add + commit + push to the Notes repo after writing.
- **Cross-reference**: use `[[slug]]` wiki-links to reference other entries
  in the same research project.
- **Machine context**: different experiments run on different machines.
  Always note which machine, how to SSH to it, and where the data lives.
  Common machines:
  - Local dev: `../AgentM`
  - Exp remote: `exp-remote`
  - ARL cluster gateway: configured gateway URL, not a literal IP

## Input

The `$ARGUMENTS` is a short topic description. Use it to:
1. Derive the log filename slug
2. Understand what experiment to document
3. Look at the current conversation context for data, commands, and results
   to include

If the argument is empty, ask what to document.
