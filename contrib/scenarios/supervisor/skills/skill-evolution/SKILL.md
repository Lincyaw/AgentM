---
name: skill-evolution
description: >
  How to write, evaluate, and evolve skills based on workflow execution
  experience. Load this skill after a workflow completes (success or
  failure) when you want to distill what happened into reusable knowledge.
---

# Skill Evolution

## What is a skill

A skill is a markdown file (`SKILL.md`) in the workflow's `skills/`
directory. It contains domain knowledge that gets injected into agent
context via skill_loader. Skills are the primary mechanism for
cross-task learning — they persist on disk and improve future runs.

## When to create a skill

Create a skill when you observe a pattern that will recur:
- The coder made a mistake that any reasonable model would make without
  domain-specific guidance
- The coder spent many turns discovering something that could have been
  stated upfront
- A specific technique or pattern worked well and should be reused

Do NOT create a skill for:
- One-off fixes (just fix the code)
- Model-specific workarounds (these are fragile)
- Things that are obvious from the spec alone

## Writing effective skills

A skill should explain WHY, not just WHAT. Models are capable reasoners —
they apply principles better than they follow checklists.

Structure:
```markdown
---
name: kebab-case-name
description: >
  When to load this skill — be specific enough that the loader can
  decide relevance. One or two sentences.
---

# Title

## Context (why this matters)
Brief explanation of the problem this skill addresses.

## Guidance
The actual knowledge. Principles over procedures.

## Common pitfalls
What goes wrong if you ignore this.
```

Keep skills under 60 lines. If it's longer, the knowledge is too broad —
split it into multiple focused skills.

## Evaluating skill effectiveness

After creating or modifying a skill, re-run the workflow and compare:
- Did the failure it was meant to prevent actually not recur?
- Did it cause any new problems (over-constraining the agent)?
- Did the agent actually load and reference the skill?

If a skill doesn't help after two iterations, delete it. Dead skills
add noise.

## Skill hygiene

- One concept per skill. "Testing guidelines" is too broad;
  "thread-safety testing with barriers" is right.
- No overlap between skills. If two skills give conflicting advice,
  the agent gets confused. Merge or delete one.
- Review all skills periodically. After several tasks, some skills
  will be stale or superseded. Prune them.

## Proposing tools

Sometimes the right response to a repeated pattern is not a skill but a
tool — something the agent can call rather than something it reads.

Signs that a tool is needed:
- Agents repeatedly write the same boilerplate script
- A manual multi-step process could be a single tool call
- The task requires information that's expensive to discover each time

Record tool proposals as a file in the workflow's directory (e.g.,
`tool-proposals.md`) with: what the tool does, why it's needed,
what agent behavior it replaces. Don't implement — propose.
