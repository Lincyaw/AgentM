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
type: knowledge
confidence: <fact|high|medium|low>
tags: [tag1, tag2]
source_trajectories: [thread_id_1, thread_id_2]
```

- `type`: always `knowledge`
- `confidence`: see confidence-levels.md for definitions
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
