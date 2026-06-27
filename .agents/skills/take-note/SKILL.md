---
name: take-note
description: >
  Archive experiment results to the Notes repo with a structured research
  log entry, experiment data, reproduction steps, and context updates. Use
  when asked to take a note, archive experiment results, record a run,
  document llm-as-harness results, or save an experiment report.
---

# take-note

Record the current experiment session to the Notes repo at:

`../Notes/research/ongoing/llm-as-harness/`

## Input

Treat the user's text after the skill name as a short topic description.
Use it to:

1. Derive the log filename slug.
2. Understand what experiment to document.
3. Look at the current conversation context for data, commands, and
   results to include.

If the topic is empty, ask what to document.

## What To Write

Create or update two things.

### 1. Log Entry

File: `log/YYYY-MM-DD-<topic-slug>.md`, using today's date.

Every log entry must include all of these sections:

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
- **Machine**: hostname / alias, such as `exp-remote`
- **Cluster**: which ARL cluster, namespace, gateway URL
- **Registry**: which container registry was used
- **Model**: model name and provider, such as `doubao-seed-2.0-pro via litellm`
- **Repo commit**: AgentM commit hash at time of experiment
- **Working directories**: relative paths for results, logs, and configs

## Method
What was done. Be specific enough that someone on a different machine can
reproduce from scratch.

## Results
Tables, numbers, comparisons. Raw data, not just conclusions.

## Reproduction steps
Step-by-step commands to reproduce this experiment from a clean state.
Include:
- Environment variables needed
- Exact CLI commands
- Expected output / how to verify success
- Any prerequisites such as images, clusters, or datasets

## Data
Where to find raw outputs:
- Score files, logs, session IDs
- Use relative paths from the relevant repo root or experiment directory
- Note which machine the paths are on if not obvious

## Findings
Interpretation of the results. What was learned, what's next.
```

### 2. Experiment Data

If this is a new experiment, also create or update files under
`experiments/<experiment-name>/`:

- `data/<descriptive-name>.json`: structured results summary.
- `protocol.md`: update with new results if the experiment already exists.

### 3. Context

Add or update the status line in `CONTEXT.md` for this experiment.

## Guidelines

- Use relative paths from the relevant repo root or experiment directory.
  Prefix with the machine alias only when the data lives on a remote host.
- Record AgentM session IDs for trajectory queries, such as
  `agentm trace messages --session <id>`.
- If prior log entries reference moved or superseded data, note that in
  the new entry.
- Cross-reference related entries with `[[slug]]` wiki-links.
- Always note machine context, SSH route when relevant, and where the data
  lives.
- Common machine labels:
  - Local dev: `../AgentM`
  - Exp remote: `exp-remote`
  - ARL cluster gateway: configured gateway URL, not a literal IP
- Commit and push the Notes repo after writing.
