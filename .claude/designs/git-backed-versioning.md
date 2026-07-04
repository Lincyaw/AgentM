# Design: Git-Backed Versioning

**Status**: DRAFT
**Created**: 2026-05-01
**Last Updated**: 2026-05-01
**Builds on**: [self-modifiable-architecture.md](self-modifiable-architecture.md), [evolution-substrate.md](evolution-substrate.md), [pluggable-architecture.md](pluggable-architecture.md)

---

## 1. Overview

Replace AgentM's bespoke per-atom version store (`.agentm/catalog/atoms/<name>/<content_hash>/source.py + manifest.yaml`) with a git-backed version layer. The catalog keeps only the data git does **not** provide: `metrics.jsonl` and `runs/` (run-id ↔ commit-sha attribution). All version content, lineage, authorship, message, and rollback flow through git plumbing.

## 2. Motivation

The current scheme reinvents five things git already ships:

| Need | Bespoke catalog today | Git equivalent |
|---|---|---|
| Content-addressed object store | `<content_hash>/source.py` written by `freeze.py` | `git hash-object` / blob store |
| Parent links | `parent_hash` field in `manifest.yaml` (currently hardcoded `None` — never populated) | commit's `parent` |
| Version log | `list_versions` walks `<name>/` directory | `git log -- <path>` |
| Active pointer | `current` symlink, written by `_replace_current_pointer` | the working tree itself (`HEAD`) |
| Authorship + rationale | `author`, `authored_at` fields; no message | committer + commit message |

Plus, three failure modes of the current design vanish:

1. **`parent_hash` is dead code.** Every frozen manifest stores `parent_hash: None`; lineage only exists implicitly in the symlink semantics. Git commits *are* the lineage.
2. **Direct `tool_edit`/`tool_write` writes bypass freeze entirely.** Today only `reload_atom` and the post-shutdown indexer touch `freeze_current`. A scenario YAML edited via `tool_write` leaves no version record. Routing every mutating write through a `ResourceWriter` that git-commits before mutation closes this gap.
3. **Skills, prompts, scenarios, `.claude/` are not catalogued at all.** Git treats them identically to atoms — versioning is free for any path under the working tree.

Net concrete payoff:

- Roughly **300 lines deleted**: `freeze.py` shrinks to a `metrics.jsonl` appender; `_layout.py` loses `<content_hash>` directory math; `browse.py`'s manifest/source readers become git plumbing wrappers.
- Cross-atom related changes (e.g. `tool_read` and its companion prompt) become **one commit**, not two unrelated frozen blobs.
- Any human reading the project's git history sees agent edits inline, attributed by author (`agent <session-id>@agentm`) and message (rationale).

## 3. Design Details

### 3.1 Boundary: catalog keeps only the unique part

```
.agentm/catalog/
├── atoms/
│   └── <atom_name>/
│       └── <git_sha>/                 # was <content_hash>
│           ├── metrics.jsonl          # KEPT — derived by indexer
│           └── runs/                  # KEPT — symlinks to traces
│               └── <trace_id> → ../../../../observability/<trace_id>.jsonl
└── scenarios/<name>/<git_sha>/...    # same shape
```

Removed from catalog:

- `source.py` (use `git show <sha>:<path>`)
- `manifest.yaml` (parse `MANIFEST` from `git show <sha>:<path>` if needed; or store derived manifest summary inside `metrics.jsonl`'s first row per SHA)
- `current` symlink (the working tree IS the current state)
- `core/<core_hash>/` directory (the repo HEAD commit IS the core version)

### 3.2 ResourceWriter — the chokepoint for all managed writes

A new harness service (per `pluggable-architecture.md` §7 service-style components) sitting in `agentm/core/runtime/resource_writer.py`. It is the **single API** for autonomy-layer writes to managed resources, and lives at the constitution layer (cannot be self-modified).

```python
WriterAuthor = Literal["agent", "human", "indexer"]
PathClass    = Literal["managed", "unmanaged", "constitution"]

@dataclass(frozen=True, slots=True)
class WriteResult:
    path: str
    path_class: PathClass
    committed: bool
    commit_sha_before: str | None     # None if not in repo / advisory mode
    commit_sha_after:  str | None
    error: str | None = None

class ResourceWriter(Protocol):
    async def write(self, path: str, content: bytes, *, rationale: str,
                    author: WriterAuthor = "agent") -> WriteResult: ...
    async def replace(self, path: str, old: bytes, new: bytes, *, rationale: str,
                      author: WriterAuthor = "agent") -> WriteResult: ...
    async def delete(self, path: str, *, rationale: str,
                     author: WriterAuthor = "agent") -> WriteResult: ...
    def classify(self, path: str) -> PathClass: ...
    def restore(self, path: Path, version: str) -> None: ...
    def batch(self, *, rationale: str,
              author: WriterAuthor = "agent") -> AbstractAsyncContextManager[BatchHandle]: ...
```

The bus emits `ResourceWriteEvent` so observability can attribute mutations.

#### Pre-mutation transactional flow

```
1. cls = self.classify(path)
2. if cls == "constitution": reject
3. if cls == "unmanaged":     plain FS write (no commit, no log)  -- e.g. tmp/
4. if cls == "managed":
     a. pre_sha = git rev-parse HEAD
     b. if working tree at <path> dirty (uncommitted human edits):
          → commit those first under author="human", message="auto: pre-agent snapshot"
            (so agent edits never silently absorb human WIP)
     c. write new bytes to <path>
     d. git add <path> && git commit -m "<rationale>" --author="<author>"
     e. post_sha = git rev-parse HEAD
     f. emit ResourceWriteEvent(path, pre_sha, post_sha, rationale, author)
     g. on FS write or commit failure:
          git restore --source=<pre_sha> -- <path>
          if the failed commit landed: git reset --hard <pre_sha>
```

Multiple `replace` calls in the same logical change can opt into a `with writer.batch(rationale=...) as b: ...` context that produces a single commit covering all paths.

#### Path classification rule

Drives off `core-manifest.yaml` (already the constitution registry) plus a managed-set rule:

| Class | Predicate | Example |
|---|---|---|
| `constitution` | `is_constitution_path(path)` returns True | `src/agentm/core/abi/...`, `core-manifest.yaml`, `.agentm/catalog/**` |
| `managed` | inside repo working tree AND not constitution AND matches `MANAGED_GLOBS` | `src/agentm/extensions/builtin/*.py`, `scenarios/**`, `skills/**`, `prompts/**`, `.claude/**`, `*.md` at repo root |
| `unmanaged` | inside repo but neither managed nor constitution | runtime artifacts: `.agentm/replay_cache/**`, build outputs; or paths outside the working tree such as `$AGENTM_HOME/observability/**` |

`MANAGED_GLOBS` is a new section in `core-manifest.yaml`:

```yaml
managed:
  globs:
    - src/agentm/extensions/builtin/**.py
    - scenarios/**
    - skills/**
    - prompts/**
    - .claude/**.md
    - "*.md"      # top-level docs
```

Outside-the-repo paths are always `unmanaged` (writer cannot version them; agent gets a result with `committed: false`, no error). Writes to `$AGENTM_HOME/observability/**` or legacy `.agentm/observability/**` are `unmanaged` because they are derived data, not source.

### 3.3 Wiring: tool_edit / tool_write / reload_atom

| Caller today | Becomes |
|---|---|
| `tool_edit` calls `file_ops.write_file(path, bytes)` directly | `tool_edit` calls `api.get_resource_writer().replace(path, old, new, rationale=...)` |
| `tool_write` calls `file_ops.write_file(...)` | `tool_write` calls `api.get_resource_writer().write(path, content, rationale=...)` |
| `reload_atom` does `freeze_atom_snapshot → tempfile + os.replace → on fail rollback temp file` | `reload_atom` does `writer.write(<atom path>, new_source, rationale=rationale, author="agent")` — git is the snapshot, `git restore` is the rollback |

`tool_edit`/`tool_write` no longer need their own constitution-path check; the writer rejects centrally. They still need a friendly `rationale` parameter — added to the tool schema with a sensible default ("agent edit via tool_edit") so the LLM can pass through a reason from its scratchpad.

### 3.4 Reload flow rewrite

**Before** (current `harness/atom_reloader.py:reload_atom`, ~150 lines around the transactional dance):

```
freeze_atom_snapshot(name, current_source, manifest, root=cwd)
  → writes <catalog>/atoms/<name>/<old_hash>/{source.py,manifest.yaml}
  → updates `current` symlink
mkstemp(...)
write new_source to tempfile
mark previous api stale; remove handlers; remove registrations
os.replace(tempfile, atom.file_path)
clear bytecode; importlib.invalidate_caches()
load_extension(...) ; install(api, config)
on Exception:
   mkstemp again with snapshot_source
   os.replace(rollback_tempfile, atom.file_path)
   re-import; re-install
   if THAT fails too: drop the atom; emit reload event with error="rollback_failure"
```

**After**:

```python
def reload_atom(name, new_source, *, rationale, agent_initiated):
    atom = self._loaded_by_name[name]
    if writer.classify(atom.file_path) == "constitution":
        return ReloadResult.fail("constitution path")

    manifest = self._validate_reload_source(name, atom.module_path, new_source)
    # ... tier-2 gating, identical to today ...

    write_result = writer.write(
        atom.file_path,
        new_source.encode("utf-8"),
        rationale=rationale or f"reload {name}",
        author="agent" if agent_initiated else "human",
    )
    if not write_result.committed:
        return ReloadResult.fail(write_result.error)

    try:
        previous_api.mark_stale()
        remove_handlers/registrations(atom.module_path)
        sys.modules.pop(atom.module_path, None)
        clear_bytecode; importlib.invalidate_caches()
        load_extension + install
        emit ExtensionReloadEvent(old=write_result.commit_sha_before,
                                  new=write_result.commit_sha_after, ...)
        return ReloadResult.ok(old_hash=..., new_hash=...)
    except Exception as exc:
        # git is our snapshot; one command rolls source back.
        run(["git", "restore", "--source", write_result.commit_sha_before,
             "--", str(atom.file_path)], check=True)
        run(["git", "reset", "--hard", write_result.commit_sha_before], check=True)
        re-import; re-install
        return ReloadResult.fail(str(exc), rolled_back=True)
```

Net deletions: the temp-file mkstemp dance (both happy and rollback paths), the snapshot-source read-back, all the `<content_hash>` symlink update logic in `freeze.py`. Git owns the snapshot; the rollback is `git restore` + `git reset --hard <pre_sha>`.

### 3.5 Browse API rewrite

`agentm.core._internal.catalog.browse` becomes a thin shell over git plumbing.

| Function | Today | After |
|---|---|---|
| `list_versions(name)` | scan `<name>/` for hash dirs | `git log --format=%H -- <atom_path>` (returns SHAs newest-first) |
| `get_source_at(path, sha)` | read `<hash>/source.py` | `git cat-file -p <sha>:<path>` |
| `get_manifest_at(name, sha)` | read `<hash>/manifest.yaml` | parse `MANIFEST` from the source blob via AST — see Open Question §8.3 |
| `runs_for(fingerprint)` | intersect `runs/` symlink sets across atoms | unchanged — `runs/` is still the catalog's responsibility, just keyed by SHA |
| `current_version(name)` | read `current` symlink | `git log -n 1 --format=%H -- <path>` |

**Key change**: version identifiers throughout the system become git SHAs (40-char hex), not the existing 8/16-char `compute_atom_hash` content hashes. The fingerprint format in `observability` `session.start` records changes from `tool_read@e5f6` to `tool_read@<40-char sha>` (or its short form). This is a wire-format break — see Migration §3.7.

### 3.6 Rollback API

A new tier-1 verb on `tool_catalog`:

```python
rollback_resource(
    path: str,
    target_sha: str,
    *,
    rationale: str,
) -> ReloadResult | WriteResult
```

Implementation: read `git show <target_sha>:<path>` → call `writer.write(path, contents, rationale=f"rollback to {target_sha[:8]}: {rationale}", author="agent")`. If `path` is an atom file, also run the reload flow on the new contents. Otherwise (skill, prompt, .claude doc) the writer commit is sufficient — next read picks up the rolled-back content.

Rollback respects the `decisions.jsonl` regression contract from `evolution-substrate.md` §6.3: rolling forward to a `regressed: true` SHA requires explicit override.

### 3.7 Repository assumption

ResourceWriter requires a git working tree. Two scenarios where one is absent:

1. **Tarball / zip distribution.** User downloads a release archive without `.git/`.
2. **Non-git project root.** User runs `agentm` in a fresh directory that is not a git repo.

**Decision: auto-init a private worktree at `.agentm/repo/`, fall back to "advisory mode" only if even that fails.**

Rationale (chosen over "fail loudly"):

- The whole point of self-modifiability is the agent can run anywhere. Bricking the agent on absent `.git/` punishes the most fragile users.
- Auto-init is cheap and side-effect-light: `git -C .agentm/repo init -q --bare` is silent, recoverable, and discoverable (the user can `cd .agentm/repo && git log` to inspect).
- The private worktree uses `--git-dir=.agentm/repo --work-tree=<cwd>` so it shadows the user's main tree without polluting it.

Decision tree on writer construction:

```
if repo_root_has_git_dir(cwd):
    use_repo = cwd                         # primary case
elif (cwd / ".agentm" / "repo").exists():
    use_repo = (cwd / ".agentm" / "repo")  # already auto-inited
else:
    git -C .agentm/repo init -q
    git --git-dir=.agentm/repo --work-tree=cwd add -A
    git --git-dir=.agentm/repo --work-tree=cwd commit -m "agentm: initial snapshot"
    use_repo = .agentm/repo
```

If git itself is not installed on the system, ResourceWriter logs a structured warning and degrades to **advisory mode**: writes go through, but `committed: false` in every WriteResult, version history is unavailable, and `tool_catalog.list_versions` returns a single synthetic entry. Self-modification still works; rollback does not.

### 3.8 Migration of existing catalog data

**Throw away.** We are MVP; only 5 of 20+ atoms have any history; no observability data depends on the existing content-hash IDs in a way that survives a re-index.

Concrete migration:

1. `python -m agentm.core._internal.catalog.indexer rebuild` is updated to:
   - delete every `<content_hash>/source.py` and `<content_hash>/manifest.yaml` on first run
   - rename the `<content_hash>/` directory to `<git_sha_of_HEAD_for_that_path>/` if computable, otherwise leave under the old name with a `.legacy` marker file (still readable, just orphaned from new traces).
2. `observability/<trace_id>.jsonl` files written before the migration retain their old `tool_read@<8-char>` fingerprints. The indexer skips them with a warning. We accept a one-shot history loss; everything from the migration commit forward is fully indexed.

This is acceptable because the catalog is a derived index, not a system of record. The system of record is `git log` (after migration) plus raw `observability/*.jsonl`.

---

## 4. Interface Definition

```python
# agentm/core/runtime/resource_writer.py

WriterAuthor = Literal["agent", "human", "indexer"]
PathClass    = Literal["managed", "unmanaged", "constitution"]

@dataclass(frozen=True, slots=True)
class WriteResult:
    path: str
    path_class: PathClass
    committed: bool
    commit_sha_before: str | None
    commit_sha_after:  str | None
    error: str | None = None

class ResourceWriter(Protocol):
    async def write(self, path: str, content: bytes, *, rationale: str,
                    author: WriterAuthor = "agent") -> WriteResult: ...
    async def replace(self, path: str, old: bytes, new: bytes, *, rationale: str,
                      author: WriterAuthor = "agent") -> WriteResult: ...
    async def delete(self, path: str, *, rationale: str,
                     author: WriterAuthor = "agent") -> WriteResult: ...
    def classify(self, path: str) -> PathClass: ...
    def restore(self, path: Path, version: str) -> None: ...
    def batch(self, *, rationale: str,
              author: WriterAuthor = "agent") -> AbstractAsyncContextManager[BatchHandle]: ...
```

ExtensionAPI gains:

```python
def get_resource_writer(self) -> ResourceWriter: ...
```

`tool_catalog` is split into two contrib atoms so scenarios can mount introspection without mutation:

```python
# contrib.extensions.tool_catalog.browse
def get_source_at(path: str, sha: str) -> bytes: ...
def list_history(path: str, *, limit: int = 20) -> list[dict[str, str]]: ...
def list_atoms() -> list[dict[str, object]]: ...

# contrib.extensions.tool_catalog.mutate
def rollback_resource(path: str, target_sha: str, *, rationale: str) -> ReloadResult | WriteResult: ...
def install_atom(name: str, source: str, ...) -> InstallAtomResult: ...
def unload_atom(name: str, ...) -> UnloadAtomResult: ...
def reload_atom(name: str, source: str, ...) -> ReloadResult: ...
```

The existing `freeze_current(name) -> str` is retained for backward compatibility but its body changes to: read current bytes → call `writer.write(... rationale="freeze_current snapshot")` → return resulting commit SHA. Callers (the indexer) keep working without code changes.

---

## 5. Related Concepts

- [self-modifiable-architecture.md](self-modifiable-architecture.md) — defines what is/isn't agent-modifiable; this design replaces the freeze/snapshot mechanism in §5 of that doc with git plumbing.
- [evolution-substrate.md](evolution-substrate.md) — defines the catalog as `versions × observations × decisions`; this design moves the "version" half to git, leaves observation+decision attribution in the catalog. Active-set fingerprint version IDs change format.
- [pluggable-architecture.md](pluggable-architecture.md) — `ResourceWriter` is a new harness-layer service in the §3.2 environment-boundary family (alongside FileOperations / BashOperations).

---

## 6. Constraints and Decisions

| Decision | Rationale | Alternative |
|---|---|---|
| Use git, not bespoke version store | Five built-in primitives (object store, parent links, log, refs, blame); zero new code to maintain | Keep current store + fix `parent_hash`. Rejected: still 300 lines of redundant infra. |
| Catalog stores only metrics + runs | Boundary follows "what does git not give us": cardinality-1 file per version (manifest, source) → git; aggregate over many runs (metrics) → catalog | Mirror everything in catalog AND git. Rejected: two sources of truth. |
| All managed writes go through ResourceWriter | Closes the `tool_edit` / `tool_write` bypass; single chokepoint for path classification | Keep tool_edit unchanged, add a post-write hook. Rejected: hook order races; no atomic rollback. |
| Auto-init `.agentm/repo` on absent `.git/` | Self-mod must work in any cwd; zero-config principle from `pluggable-architecture.md` §1 | Fail loudly. Rejected: punishes tarball users, contradicts default-everywhere. |
| Throw away old catalog content | MVP, 5 atoms, derived data | One-shot importer. Rejected: not worth 100 LOC for 5 atoms with one parent_hash field. |
| Version IDs become full git SHAs | Match git's identifier; no reinvention | Keep `compute_atom_hash` as the public ID, add SHA as metadata. Rejected: two IDs is two truths. |

---

## 7. Acceptance Scenarios

| # | Scenario | Expected |
|---|----------|----------|
| G1 | Agent edits `tool_read.py` via `reload_atom(rationale="add limit param")` | Commit lands on HEAD, author=`agent`, message=`add limit param`; `git log -- src/agentm/extensions/builtin/tool_read.py` shows it |
| G2 | New `tool_read.py` raises in `install()` | `git reset --hard <pre_sha>` runs; HEAD matches pre-state; ReloadResult.rolled_back=True |
| G3 | Agent calls `tool_edit` on `core/abi/loop.py` | ResourceWriter classifies as `constitution`, rejects before any git/FS op |
| G4 | Agent calls `tool_edit` on `skills/foo/SKILL.md` | Classified `managed`; commit lands; `tool_catalog.list_versions("skills/foo/SKILL.md")` returns the SHA |
| G5 | Agent calls `tool_write` to `/tmp/scratch.txt` (outside repo) | Classified `unmanaged`; FS write happens; `committed=false` returned |
| G6 | Agent calls `rollback_resource("tool_read.py", <prior_sha>, rationale="...")` | Source at SHA written via writer; reload runs; new commit recorded with rollback rationale |
| G7 | User runs `agentm` in fresh non-git directory | `.agentm/repo/` auto-inited; first call records initial-snapshot commit; subsequent agent edits commit normally |
| G8 | git binary missing on host | Writer enters advisory mode; writes succeed; `committed=false`; warning logged once |
| G9 | Two agent edits in one logical change via `writer.batch(rationale="refactor read+search")` | Single commit covering both files lands |
| G10 | Human dirties `tool_read.py` in working tree, agent then edits via `reload_atom` | Pre-commit "auto: pre-agent snapshot" lands first under author=`human`; agent commit lands second |

---

## 8. Open Questions

1. **Where do agent commits go: working branch or dedicated ref?**
   - Option A: HEAD on the human's working branch. Simple, discoverable (`git log` just works), but pollutes the human's history with potentially noisy agent edits — a 50-turn session can produce 50 commits.
   - Option B: dedicated branch `refs/agentm/history` (or `agentm/work`). Keeps the human's mainline clean; requires the human to `git merge agentm/history` to adopt agent changes. Risk: agents see their writes "land" but a `git pull` from collaborators won't include them; mental model mismatch.
   - Option C: orphan ref `refs/agentm/journal` (not reachable from HEAD), with HEAD updated only on a `propose_change`-style approval. Cleanest separation, but adds a workflow step and requires the working tree to track HEAD-but-also-some-other-ref state.
   - **Tentative recommendation**: Option A for MVP; revisit if commit volume becomes a real complaint. Add a config knob `resource_writer.commit_target: "head" | "ref:agentm/history"` so users who care can opt out.

2. **Should ResourceWriter migrate `.agentm/repo` → user's `.git/` if the user later inits one?** An auto-detected switch sounds elegant but loses the agent-only history (the new `.git/` doesn't have those commits). Cleanest answer: don't migrate; `.agentm/repo` becomes orphan and stays as audit trail. Needs explicit user-facing doc.

3. **Manifest reconstruction for `get_manifest_at(sha)`.** Executing `MANIFEST = ExtensionManifest(...)` from a historical SHA's source means importing arbitrary code. For trusted local commits this is fine; for a future federated catalog it is not. Defer until federation is on the table; for now, parse `MANIFEST` via AST visitor (no code execution) and accept that exotic manifests with computed values won't round-trip.

4. **Performance: every agent edit forks a `git` subprocess.** A 100-tool-call session is 100 commits. For MVP this is acceptable (git is fast at small commits). If it becomes a bottleneck, switch to `pygit2` or `dulwich` for in-process commits — neither blocks the design.

5. **Multi-process safety.** Two agent sessions writing concurrently to the same repo will race on `HEAD`. Git's lock file (`.git/index.lock`) provides serialization but not fairness. Out of scope for MVP (single-session assumption); revisit when sub-agent system needs to mutate shared resources.
