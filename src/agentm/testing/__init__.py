"""Test-facing helpers for proving properties of a composition.

This lives in ``src/`` rather than under ``tests/`` for two reasons. The
contrib workspace members ship their own atoms and their own test suites, and a
helper that can only be imported from the SDK's ``tests/`` package is not
available to them. And the obligation ``assert_revertible`` discharges is part
of the atom contract, not part of one test file: every step of the
composability work is verified by installing an atom, uninstalling it, and
finding the session unchanged.

Nothing here imports pytest — the assertions are plain ``AssertionError``s, so
the helpers work from any runner or from a script.
"""

from agentm.core.runtime.composition_digest import (
    CompositionDigest,
    composition_digest,
)
from agentm.testing.revertibility import (
    FieldDifference,
    NeverStreams,
    assert_revertible,
    digest_differences,
    probe_session,
)

__all__ = [
    "CompositionDigest",
    "FieldDifference",
    "NeverStreams",
    "assert_revertible",
    "composition_digest",
    "digest_differences",
    "probe_session",
]
