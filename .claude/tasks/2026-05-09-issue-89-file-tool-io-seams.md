# Issue #89 — File Tool IO Seam Decision

Chosen option: **Option C — Hybrid: read via Operations, write via ResourceWriter**.

Rationale:
- This formalizes the current architecture with the smallest ABI change.
- `FileOperations` remains the read-only tool-environment port for `read`, `grep`, `find`, and `ls`.
- `ResourceWriter` remains the single mutation chokepoint for `write` and `edit`, preserving managed-resource git commits, constitution-path rejection, and write audit events.
- Scenario authors now have explicit guidance: override `FileOperations` to redirect reads/listing, and override `ResourceWriter` to redirect/audit writes.

Design docs updated:
- `.claude/designs/pluggable-architecture.md`
- `.claude/designs/extension-as-scenario.md`

Implementation notes:
- Removed `write_file` from the public `FileOperations` Protocol and default local implementation.
- Added `is_dir` / `list_dir` to `FileOperations` so read-only directory tools no longer need per-atom local filesystem operations.
- Routed `tool_find` / `tool_grep` `.gitignore` discovery through `FileOperations` so read seam overrides intercept directory traversal and ignore-file reads.
- Added `ResourceWriter.read` so `tool_edit` can read expected bytes and write through one write seam without touching `FileOperations`.
- Updated `tool_write` and `tool_edit` to use `ResourceWriter` exclusively.

## Review follow-up

- Synchronized the active `search_tools` design concept with Option C: `tool_grep`, `tool_find`, and `tool_ls` use the shared `FileOperations` read seam instead of bespoke per-tool Operations protocols or direct external-search subprocesses.
- Added the issue #89 task note to the `search_tools` concept in `.claude/index.yaml` and added `.claude/designs/search-tools.md` to `REQ-089-file-tool-io-seams` docs.
