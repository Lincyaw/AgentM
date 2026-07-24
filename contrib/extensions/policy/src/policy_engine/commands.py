# code-health: ignore-file[AM025] -- shell tokens are untyped at the boundary
"""Command analysis shared by the live watcher and offline replay.

One definition of "what did this bash call do" for every consumer:
segmentation, wrapper unwrapping, validation-shape classification, scope
matching, selector extraction and red/green supersession. All of it is
token-shape algebra over the agent's own argv history; the lexical
heuristics carry their calibration provenance inline.

Audit lineage (2026-07-24 emission audit over 28 sessions): every heuristic
here was corrected against per-emission evidence — see the fixes noted on
``is_validation``, ``scope_refs``, ``selector_tokens`` and
``supersession_key``.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

# Tokens that split a bash command line into independent segments.
_LINE_REF = re.compile(r":\d+$")

_CONNECTORS = frozenset({"&&", ";", "||", "|"})

# Wrappers stripped before comparing commands, so a red under one wrapper
# can be superseded by a green under another (`uv run pytest` vs `pytest`).
_WRAPPERS = frozenset({"uv", "run", "poetry", "exec", "pnpm", "npx", "env"})

# Heads that read or format rather than execute. The 2026-07-24 audit caught
# `cat > x_test.go`, `ls *_test.go`, `which pytest`, `gofmt -w *_test.py`,
# `find -name '*test*'` and failed `rg` calls all being counted as
# validation runs; every downstream signal inherited that noise.
_NON_EXEC_HEADS = frozenset(
    {
        "cat",
        "ls",
        "find",
        "rg",
        "grep",
        "sed",
        "awk",
        "head",
        "tail",
        "which",
        "echo",
        "wc",
        "sort",
        "cut",
        "tr",
        "stat",
        "file",
        "touch",
        "mkdir",
        "cp",
        "mv",
        "rm",
        "gofmt",
        "diff",
        "tee",
        "printf",
        "chmod",
    }
)

# Subcommands that groom rather than validate (`mix format x_test.exs`,
# `cargo fmt`, `go vet` — the audit caught format calls counted as green
# validations because a *file argument* contained `test`).
_HYGIENE_SUBCOMMANDS = frozenset({"format", "fmt", "lint", "vet", "fix"})

# Breadth markers: positional tokens that widen rather than narrow a run.
# The audit caught `go test ./...` being read as selector-narrowed.
_BREADTH_MARKERS = frozenset({".", "./...", "...", "./."})

# Path segments too generic to serve as scope names on their own.
# CALIBRATION DEBT (docs/policy-anomaly-detection.md): hand-picked stoplist;
# a repository whose real package is literally named `lib` loses its scope.
GENERIC_SEGMENTS = frozenset(
    {"src", "lib", "test", "tests", "crates", "packages", "apps", "internal", "pkg"}
)

# Environment-shaped failures, not red suites (2026-07-22 calibration:
# removed the Prefect/Electric false fires). Pytest exit 4 is a mistyped
# selector (Paperless lesson), handled in ``usage_error``.
ENV_PROBE_EXITS = frozenset({126, 127})


@dataclass(slots=True, frozen=True)
class Segment:
    """One shell segment of a bash call (split on ``&&``, ``;``, ``||``, ``|``)."""

    ordered: tuple[str, ...]
    norm: tuple[str, ...]

    @property
    def raw(self) -> str:
        return " ".join(self.ordered)

    @property
    def head(self) -> str:
        return self.norm[0] if self.norm else ""


@dataclass(slots=True, frozen=True)
class ExecRecord:
    """One bash tool call: raw text, segments, exit code, turn."""

    raw: str
    segments: tuple[Segment, ...]
    exit_code: int | None
    turn: int = 0


def template_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Strip leading wrappers and env assignments: the comparable command."""

    out = list(tokens)
    while out and ("=" in out[0] and not out[0].startswith("-")):
        out.pop(0)
    while len(out) > 1 and out[0] in _WRAPPERS:
        out.pop(0)
    if len(out) > 2 and out[0] == "python" and out[1] == "-m":
        out = out[2:]
    return tuple(out)


def split_segments(raw: str) -> tuple[Segment, ...]:
    """Split a bash command into logical segments; drop bare ``cd`` hops."""

    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError:
        tokens = raw.split()
    segments: list[Segment] = []
    current: list[str] = []

    def flush() -> None:
        if current and not (current[0] == "cd" and len(current) <= 2):
            segments.append(
                Segment(ordered=tuple(current), norm=template_tokens(tuple(current)))
            )

    for token in tokens:
        if token in _CONNECTORS:
            flush()
            current = []
        else:
            current.append(token)
    flush()
    return tuple(segments)


def is_validation(segment: Segment) -> bool:
    """Execution-shaped run whose head area names a test.

    Audit fix: the previous form ("`test` anywhere in the first four raw
    tokens") labeled `cat`/`ls`/`find`/`gofmt`/`rg` calls as validation
    because a *filename argument* contained ``_test``. Now the head must
    not be a read/format tool and ``test`` must appear within the first
    three unwrapped tokens (covers `go test`, `pytest`, `mix test`,
    `pnpm --filter x test`, `pnpm exec vitest run`).
    """

    if not segment.norm or segment.head in _NON_EXEC_HEADS:
        return False
    if len(segment.norm) > 1 and segment.norm[1] in _HYGIENE_SUBCOMMANDS:
        return False
    return any(
        "test" in token.lower()
        for token in segment.norm[:4]
        if "/" not in token and not token.startswith("-")
    )


def scope_tokens(mutated_paths: list[str]) -> frozenset[str]:
    """Candidate scope names derived from mutated paths."""

    tokens: set[str] = set()
    for raw_path in mutated_paths:
        parts = Path(raw_path).parts
        for part in parts[:-1]:
            if part and part not in GENERIC_SEGMENTS and not part.startswith("/"):
                tokens.add(part)
        stem = Path(raw_path).stem
        if stem:
            tokens.add(stem)
    return frozenset(tokens)


def _token_mentions_scope(token: str, scope: str) -> bool:
    if token == scope:
        return True
    # Path-shaped mention: `./services/gitdiff`, `src/api/x_test.go`.
    if "/" in token:
        return scope in token.split("/")
    # Filename-shaped mention: `poll_test.exs` mentions scope `poll_test`.
    return Path(token).stem == scope


def scope_refs(segment: Segment, scopes: frozenset[str]) -> frozenset[str]:
    """Scopes a segment references, path-aware.

    Audit fix: exact token-set intersection only ever matched cargo-style
    CLIs where the package name stands alone (`-p turborepo-scm`); in
    go/mix/pnpm worlds the scope lives inside path tokens
    (`./services/gitdiff`) and never matched — the gitea firing worked only
    because a heredoc body happened to contain a bare `gitdiff` token.
    """

    referenced: set[str] = set()
    for token in segment.ordered:
        for scope in scopes:
            if _token_mentions_scope(token, scope):
                referenced.add(scope)
    return frozenset(referenced)


def selector_tokens(segment: Segment, scope: str) -> tuple[str, ...]:
    """Tokens that narrow a validation run below scope level.

    Shape rules only (no runner lexicon), rebuilt from the 2026-07-24
    emission audit which showed the old positional rule counted wrapper
    heads (`uv run pytest` -> `pytest`) and heredoc bodies as selectors
    while missing every go-style `-run TestX` flag selector:

    - a token containing ``::`` or a trailing ``:<line>`` narrows the run
      to named cases;
    - a bare identifier past the command head that mentions no scope
      narrows the run (cargo-style positional filters);
    - a bare-identifier value following a flag narrows the run unless it
      mentions a scope (`-run TestX` narrows; `-p <scope>` does not).
    """

    selectors: list[str] = []
    previous = ""
    for position, token in enumerate(segment.norm):
        mentions_scope = _token_mentions_scope(token, scope) or scope in token
        if position >= 1 and not mentions_scope:
            if "::" in token or _LINE_REF.search(token):
                selectors.append(token)
                previous = token
                continue
            bare_identifier = (
                not token.startswith("-")
                and "=" not in token
                and "/" not in token
                and " " not in token
                and token not in _BREADTH_MARKERS
                and token != "test"
            )
            if bare_identifier and (position >= 2 or not previous.startswith("-")):
                selectors.append(token)
        previous = token
    return tuple(selectors)


def usage_error(record: ExecRecord) -> bool:
    """Pytest exit 4 is a mistyped selector, not a red suite."""

    return record.exit_code == 4 and any(
        segment.norm[:1] == ("pytest",) for segment in record.segments
    )


def supersession_key(
    segment: Segment, scopes: frozenset[str]
) -> tuple[str, frozenset[str]]:
    """Comparable identity of a validation run: head plus referenced scopes.

    Audit fix: full token-subset comparison could not see that a green
    `cargo test -p x --no-default-features` covers a red
    `cargo test -p x --lib` (flag variants are disjoint token sets). Two
    runs validate the same thing when they share the command head and the
    green references every scope the red referenced.
    """

    return (segment.head, scope_refs(segment, scopes))


def supersedes(green: Segment, red: Segment, scopes: frozenset[str]) -> bool:
    """A later green supersedes a red when it runs the same head over an
    equal-or-wider scope set, and is not narrower via selectors."""

    green_head, green_scopes = supersession_key(green, scopes)
    red_head, red_scopes = supersession_key(red, scopes)
    if green_head != red_head:
        return False
    if not red_scopes <= green_scopes:
        return False
    # A selector-narrowed green does not clear a broader red.
    red_sel = (
        {s for sc in red_scopes for s in selector_tokens(red, sc)}
        if red_scopes
        else set(selector_tokens(red, ""))
    )
    green_sel = (
        {s for sc in green_scopes for s in selector_tokens(green, sc)}
        if green_scopes
        else set(selector_tokens(green, ""))
    )
    return green_sel <= red_sel or not green_sel
