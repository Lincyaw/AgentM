# Issue 99 — Presenter Seams and Error Visibility

## Requirement

Presenters must consume the harness public surface, expose testable output and
Textual presentation seams, report recoverable presenter failures visibly, and
remove legacy tool-argument preview plumbing.

## Implementation Notes

- Route CLI and Textual presenter imports through `agentm.harness` public
  exports.
- Add `cli.run(..., output: TextIO = sys.stdout)` and write final CLI output to
  the injected stream.
- Add Textual `css_path`, `theme`, and `keymap` seams while keeping existing
  defaults.
- Replace silent theme/clipboard fallbacks with diagnostics or toast-visible
  `ClipboardStatus` results.
- Delete the legacy `args_preview` path.

## Verification

- `uv run ruff check src/`
- `uv run mypy src/`
- `uv run pytest --tb=short`
- `python /home/ddq/.claude/plugins/cache/autoharness/autoharness/1.1.3/scripts/validate_index.py project-index.yaml`
