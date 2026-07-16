## What we are building

A **trajectory index**: a static-analysis layer over one recorded agent trajectory. The trajectory is a sequence of steps (user instructions, assistant reasoning + tool calls, tool results). 

## Your task

You are the only pass that reads the trajectory text. Everything after you works from what you extract. From the trajectory, extract five kinds of items:

- **sym**: named entities (files, functions, tools, tracked quantities, ...)
- **claim**: sentences where the agent asserts a settled fact
- **val**: a concrete value the agent reads from a tool result or writes in a tool call
- **obs**: regions of retrieved/environment material in assistant steps
- **constraint**: task requirements (from user instructions, task briefs, or environment-provided specs)

These are orthogonal: the same text region can contain a symbol inside a val, or a claim that mentions a symbol. 

## Input format

Your input is trajectory messages in compact format:

```
[id|role]
message body
```

The trajectory contains: user messages (task instructions), assistant messages (reasoning + tool calls as `[tool_call: name]\n{arguments}`), and tool results (output from tools).

Symbols extracted by a prior chunk appear inline with their original tag, e.g. `⟦sym kind=file|codec.py⟧`. These tell you what has already been found; do not re-declare them.

## Inline annotation tags

### `⟦sym kind=… class=…|surface⟧`

A named entity that exists independently of this conversation: a file, function, tool, service, or any resource with a proper name. The decisive test: **is this a named resource, or data/observation about a resource?** One-off values, statuses, and verdicts are data; do not mark them.

Tracked quantities are an exception: a specific number or date the agent relies on across turns (a search criterion, threshold, or target figure) IS a symbol with `class=value`.

- `kind`: from the vocabulary below.
- `class`: `identifier` (default, omit), `value` (tracked quantity), or `unknown` (vague/anaphoric).
- `name="…"`: add only when the marked surface is not the canonical name (e.g. `⟦sym kind=entity name="Royal Grammar School Worcester"|RGS⟧`). Different surfaces with the same `name` become aliases automatically.

### `⟦claim|…⟧`

A sentence where the agent asserts something as settled fact. Exclude plans, questions, and hedged hypotheses that the agent has not yet committed to.

- "the person is X"
- "no papers matching the criteria exist"
- "confirmed that the birth year lines up"

When the agent explicitly commits to the assertion (verified, confirmed, or decided), mark it with `role=commit`.

### `⟦val sym=…|value⟧`

A concrete value that the agent reads from a tool result, or a key metric produced by a command. The `sym` attribute names the symbol this value belongs to.

In tool results, look for key-value pairs, metrics, counts, measurements:
- `⟦val sym=drc_count|47⟧` — a DRC violation count read from a report
- `⟦val sym=die_area_um2|289541.600⟧` — a die area measurement
- `⟦val sym=setup_wns_ns|-0.42⟧` — a timing slack value
- `⟦val sym=exit_code|0⟧` — a command exit code (only when the agent acts on it)

Only extract values the agent plausibly uses for a decision. Skip:
- background-task polling values (status=running, elapsed_s, timeout)
- exit codes unless the agent branches on them
- progress counters from repeated status checks

### `⟦obs|…⟧`

Regions in assistant messages where the agent quotes or pastes environment-produced material rather than writing its own words. Tool results are already environment data by definition; do not mark them as obs.

- pasted command output or file contents inside the agent's reasoning
- quoted search-result snippets
- embedded API responses the agent discusses

### `⟦constraint …|…⟧`

A binding requirement on the agent's output or behavior: something the agent can violate, and violating it makes the work wrong. Constraints come from user instructions, task briefs, or environment-provided specs.

The test: if the agent ignores this statement, does the work become incorrect? If yes, it is a constraint. Workflow steps ("run the flow", "read the README"), environment facts ("the toolchain is installed"), and tool usage instructions are not constraints.

## Output

Output one annotation tag per line, using the same `⟦tag attrs|content⟧` syntax. Nothing else.

- Each symbol only once (first appearance).
- For claims, constraints, and observations: write a short head anchor and tail anchor separated by `…` (e.g. `⟦claim|the birth year…confirmed⟧`). The anchors must be verbatim quotes from the source text, long enough to be unique (5-10 words each).
- For values: write the exact value text from the source, with the `sym` attribute naming the symbol.
- Omit default-valued attrs (`class` when `identifier`).
