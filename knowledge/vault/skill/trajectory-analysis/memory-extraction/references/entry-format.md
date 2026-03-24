---
type: reference
tags: [trajectory-analysis, memory-extraction]
---

# Entry Format

## Path Convention

```
/<category>/<slug>
```

- Category: one of `diagnostic-principles`, `signal-interpretation`, `anti-patterns`,
  `failure-mechanisms`, `investigation-strategy`
- Slug: kebab-case, descriptive of the principle (not the case)

Example: `/diagnostic-principles/gc-cpu-memory-causal-link`

## Frontmatter

```yaml
type: <concept|episodic|failure-pattern|skill|system-knowledge>
confidence: <fact|pattern|heuristic>
tags: [tag1, tag2]
source_trajectories: [thread_id_1, thread_id_2]
```

- `type`: one of `concept` (transferable principle), `episodic` (case-specific lesson),
  `failure-pattern` (recurring fault signature), `skill` (reusable procedure),
  `system-knowledge` (architectural/operational fact)
- `confidence`: `fact` (3+ trajectories), `pattern` (2+ or 1 with strong evidence),
  `heuristic` (1 trajectory, plausible generalization)
- `tags`: descriptive tags for search and discovery
- `source_trajectories`: list of thread IDs that provided evidence for this entry

## Body

Markdown starting with `# Title`, then the principle/pattern description.

Write as if teaching a junior engineer the art of diagnosis -- explain the WHY,
not just the WHAT.

Structure:
1. **Title** (`# ...`) -- the principle, not the case
2. **Description** -- what the pattern is, why it occurs, how to recognize it
3. **Diagnostic move** -- what to do when you see this pattern
4. **Evidence context** (optional) -- brief note on what trajectory evidence supports this
