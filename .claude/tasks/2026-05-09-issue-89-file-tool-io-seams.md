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
