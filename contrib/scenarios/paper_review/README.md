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
order, assigns artifact paths, and checks that each artifact file exists. The
child agent receives the input path and discovers the relevant paper files
itself with file tools and bash. It writes its own Markdown artifact with the
file tools and can load only these paper skills:

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
  agentm_paper_review/
    workflow/
      __init__.py
      paper_review_workflow.py
      types.py
  workflow/                  # compatibility wrappers for path-based runs
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

## Artifacts

Intermediate pass artifacts are written by the reviewer workers under a
sibling artifact directory named after the report. For example, the default
report path `paper-review-report.md` uses:

```text
paper-review-report.artifacts/
  01-paper-reader.md
  02-paper-prose.md
  03-paper-consistency.md
  04-paper-evidence.md
  05-paper-structure.md
paper-review-report.md
```

The final `paper-review` worker writes the final report directly to
`output_path`. Earlier pass artifact paths are passed to later workers; the
workflow does not inline or parse their Markdown.

The `paper-reader` pass is stricter than the later review passes: it must
simulate a linear first read by updating `Reading Notes` in the artifact after
each reading unit. A healthy trace should show multiple `write`/`edit` updates
before the pass finishes.

The final `paper-review` pass must synthesize rather than concatenate prior
artifacts: it de-duplicates overlapping findings, groups them by aspect, and
sorts each group by severity and revision impact.

## Notes

The workflow deliberately does not copy the paper skill prompts into code.
The reviewer loads the relevant skill through `load_skill` at the start of
each pass. Skill paths are listed explicitly in the reviewer manifest so
default/user skills are not visible during the review.

The workflow does not pre-read the paper, scan `.tex` files, build line
indexes, maintain a structured notes object, or render a structured report.
Each worker discovers and reads the paper files it needs, writes its Markdown
artifact to the path supplied by the workflow, and returns only a short status.
The last worker loads `paper-review`, reads the prior artifact files, and
writes the final report.
