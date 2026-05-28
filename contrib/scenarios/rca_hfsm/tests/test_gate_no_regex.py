"""Phase-2 acceptance §8.1 — gate file contains zero regex / lemma matching.

The gate's job after C2 is to be a dispatcher over the 4 ``rca.judge.*``
services. Any semantic decision about a free-text field
(``verdict_proposal``, ``interpretation.reasoning``, ``rationale``, ...)
goes through a judge, not through a substring check in the gate source.
This test grep s the source for the banned constructs and fails if any
appear.

The banned substrings have two flavours:

* Regex machinery — ``import re``, ``re.<fn>``, ``\\b`` (word boundary).
  Pure-Python regex use would smuggle the Phase-1 vocabulary back in via
  pattern compilation.
* Phase-1 vocabulary lemmas — ``"triggered"``, ``"supports"``,
  ``"steelman"``. Even without regex, equality / membership checks
  against these literals would resurrect the structural rule the
  refactor exists to remove.

The same checks run against ``updates.py`` to catch any rule that leaked
back into the shared pure-data module (the C2 refactor stripped that
module down to data types + light shape preconditions).

See ``.claude/designs/llm-native-judges.md`` §8.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_GATE_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "agentm_rca_hfsm"
    / "atoms"
    / "rca_falsification_gate.py"
)
_UPDATES_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "agentm_rca_hfsm"
    / "updates.py"
)
_FINALIZE_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "agentm_rca_hfsm"
    / "atoms"
    / "rca_finalize.py"
)


_BANNED_REGEX_TOKENS = (
    "import re",
    "re.match",
    "re.search",
    "re.fullmatch",
    "re.compile",
    "re.finditer",
    "\\b",
)


_BANNED_LEMMA_TOKENS = (
    '"triggered"',
    "'triggered'",
    '"supports"',
    "'supports'",
    '"steelman"',
    "'steelman'",
)


_BANNED_MEMBERSHIP_TOKENS = (
    ".startswith(",
    ".endswith(",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _gate_body() -> str:
    """Return the gate file source with its module docstring stripped.

    The docstring legitimately mentions the banned vocabulary while
    explaining what the refactor removed — that mention is meta, not
    behavior. Stripping it isolates the assertion to executable Python.
    """

    source = _read(_GATE_FILE)
    # The module docstring is the first triple-quoted string. Locate the
    # closing triple-quote and discard everything up to and including it.
    if source.startswith('"""'):
        end = source.find('"""', 3)
        if end != -1:
            return source[end + 3 :]
    return source


def _updates_body() -> str:
    source = _read(_UPDATES_FILE)
    if source.startswith('"""'):
        end = source.find('"""', 3)
        if end != -1:
            return source[end + 3 :]
    return source


def _finalize_body() -> str:
    source = _read(_FINALIZE_FILE)
    if source.startswith('"""'):
        end = source.find('"""', 3)
        if end != -1:
            return source[end + 3 :]
    return source


@pytest.mark.parametrize("token", _BANNED_REGEX_TOKENS)
def test_gate_source_contains_no_regex_machinery(token: str) -> None:
    body = _gate_body()
    assert token not in body, (
        f"gate file contains banned regex token {token!r}; design §8.1 "
        f"requires zero regex / lemma matching in the gate"
    )












