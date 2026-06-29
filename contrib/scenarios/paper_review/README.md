# paper_review - paper review workflow

Minimal paper review workflow adapted from the `paper-review` skill:

```text
path
  -> fixed reviewer roles
  -> Markdown artifacts
  -> final Markdown report
```

This scenario follows the same shape as `contrib/scenarios/devloop`: a
checked-in workflow script coordinates a small child agent scenario. The
workflow is deliberately thin: it validates the input path, runs the roles in
order, passes prior Markdown forward, and writes the final report. The child
agent receives the input path and discovers the relevant paper files itself
with file tools and bash. It can load only these paper skills:

- `paper-review`
- `paper-reader`
- `paper-prose`
- `paper-consistency`
- `paper-evidence`
- `paper-structure`

## Structure

```text
paper_review/
  agents/
    reviewer/
      manifest.yaml
      paper_review_context.py
  workflow/
    __init__.py
    paper_review_workflow.py
    types.py
```

## Quick start

```bash
agentm workflow run contrib/scenarios/paper_review/workflow/paper_review_workflow.py \
  --args '{"path": "/path/to/paper-or-directory"}'
```

`path` may be a paper file or a directory. If it is a directory, reviewer
workers decide which files are relevant, for example by finding a LaTeX entry
point and following `\input` / `\include` references.

By default the report is written to `paper-review-report.md` inside the input
directory, or next to the input file. Override it with:

```bash
agentm workflow run contrib/scenarios/paper_review/workflow/paper_review_workflow.py \
  --args '{"path": "/path/to/paper.tex", "output_path": "review.md"}'
```

## Args

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `path` | str | required | Paper file or directory. |
| `output_path` | str | `paper-review-report.md` next to the paper | Report path. |
| `agent_timeout_seconds` | float | `1200` | Per-reviewer timeout. |
| `agent_retries` | int | `2` | Child-agent retry count. |

## Notes

The workflow deliberately does not copy the paper skill prompts into code.
The reviewer loads the relevant skill through `load_skill` at the start of
each pass. Skill paths are listed explicitly in the reviewer manifest so
default/user skills are not visible during the review.

The workflow does not pre-read the paper, scan `.tex` files, build line
indexes, maintain a structured notes object, or render a structured report.
Each worker discovers and reads the paper files it needs and returns Markdown.
The last worker loads
`paper-review`, receives the prior Markdown artifacts, and returns the final
report; the workflow writes that Markdown verbatim.
