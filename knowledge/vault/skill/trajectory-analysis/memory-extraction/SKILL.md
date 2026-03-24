---
name: memory-extraction
description: >
  Extract reusable diagnostic knowledge from completed RCA trajectories into the
  vault. Use when processing completed RCA runs to build or improve the knowledge
  store, even if the user just says "learn from these runs." Produces principle-based
  entries that teach reasoning approaches, not brittle rules.
---

# Memory Extraction

## Goal

Extract **transferable diagnostic wisdom** from RCA trajectories. Not case facts,
not threshold rules -- reasoning patterns, signal interpretation principles, and
pitfalls that make future investigations better.

What you write will be loaded into future agent prompts. Every entry must be:
- **Principle-based** -- causal relationships, not "if X > 3x then Y"
- **Transferable** -- applies beyond the specific services in this trajectory
- **Actionable but not prescriptive** -- teaches what to look for and why

## Workflow

### 1. Survey existing knowledge

Call `vault_list("/")` to see what already exists. Never skip this step.

### 2. Choose approach based on trajectory count

**3 or fewer trajectories** -- do it yourself:
1. `read_trajectory(thread_id)` for each
2. Extract principles, deduplicate against existing entries
3. `vault_write` each entry

**More than 3 trajectories** -- dispatch workers:
1. `dispatch_agent` one worker per trajectory (task_type="analyze")
   - Workers read, search for similar entries, return candidates. They cannot write.
2. `check_tasks` to collect all candidate lists
3. Deduplicate and merge across workers -- prioritize principles appearing in multiple
   trajectories (higher confidence) over single-trajectory observations
4. `vault_write` final entries yourself

### 3. Deduplicate before writing

For every candidate entry, `vault_search(query=<description>, mode="hybrid", limit=3)`.
If a similar entry exists, `vault_read` it, then either:
- **Merge**: `vault_edit` to strengthen the existing entry with new evidence
- **Skip**: if the existing entry already covers the principle
- **Create new**: only if genuinely distinct

### 4. Write entries

Follow the entry format in [references/entry-format.md](references/entry-format.md).
Only write entries with confidence >= medium. Low-confidence observations can be noted
in existing entries as "possible related pattern" but not as standalone entries.

For confidence level definitions, see [references/confidence-levels.md](references/confidence-levels.md).

## Decision Tree: What to Extract

Read each trajectory through four lenses. Detailed guidance:
[references/what-to-extract.md](references/what-to-extract.md)

1. **Where did the investigation go wrong?** -- misread signals, dead ends, reasoning errors
2. **Where did it go right?** -- non-obvious connections, correct rejections
3. **What diagnostic principle is this an instance of?** -- the transferable pattern
4. **What anti-pattern did agents fall into?** -- anchoring, confirmation bias, etc.

## Entry Categories

Five categories for organizing entries. Full definitions:
[references/categories.md](references/categories.md)

- `diagnostic-principles` -- transferable reasoning patterns
- `signal-interpretation` -- reading and cross-referencing observability data
- `anti-patterns` -- reasoning traps that derail investigations
- `failure-mechanisms` -- recurring fault types and their signatures
- `investigation-strategy` -- meta-patterns about structuring investigations

## Entry Quality

Before writing, check your entry against the quality bar. Good/bad examples:
[references/entry-quality.md](references/entry-quality.md)

Quick test: does your entry teach a **reasoning approach** or state a **rule**?
If it is a rule, rewrite it as the causal relationship behind the rule.

## Gotchas

| Mistake | Correction |
|---------|------------|
| Recording case-specific facts ("ts-auth CPU was 6.83x") | Extract the transferable principle ("resource anomaly without latency anomaly is still a root cause candidate") |
| Writing threshold-based rules ("if X > 3x then Y") | Describe causal relationships ("X and Y are linked through mechanism Z") |
| Writing without checking duplicates | `vault_search` first. If similar exists, `vault_read` + merge, do not create a near-duplicate |
| Writing vague platitudes ("always check everything") | Be concrete: what signal pattern, what it means, what diagnostic move it suggests |
| Finalizing before all writes complete | Only finalize after ALL `vault_write` calls have returned |
| Guessing what exists in the store | `vault_list("/")` at least once before writing anything |
| Treating each trajectory in isolation | Cross-reference: does this trajectory confirm, contradict, or extend an existing entry? |
| Including case facts as entries | Generalize into transferable principles |
| Only extracting failures | Also extract what went RIGHT -- successful reasoning patterns are high value |
| Skipping duplicate check for worker candidates | Workers must `vault_search` for each candidate and populate `existing_similar` |
