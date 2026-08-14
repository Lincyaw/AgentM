"""Witness that an atom's installation can be undone.

The composability programme this supports rests on one obligation that nothing
in the repository could previously check: an inverse really does undo its
effect.  ``assert_revertible`` discharges it the only way a running system can
— install the atom into a probe session, uninstall it, and compare what the
session holds before and after.

Recovery is only ever "up to an equivalence", and an equivalence that is not
written down is a wish.  Two things make it explicit here:

* ``composition_digest`` states the equivalence positively — everything it
  covers must come back, and its module docstring justifies everything it
  leaves out.
* ``residue`` states the exceptions negatively.  A caller names the digest
  fields it expects *not* to come back, and the assertion fails both when an
  unnamed field differs and when a named one does not.  A residue that gets
  fixed therefore breaks the test that tolerated it, which is what stops the
  list quietly outliving the thing it excuses.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.session_api import AgentSessionConfig, ExtensionSpec
from agentm.core.abi.stream import (
    AssistantStreamEvent,
    Model,
    ThinkingLevel,
)
from agentm.core.abi.tool import Tool
from agentm.core.runtime.composition_digest import CompositionDigest, composition_digest
from agentm.core.runtime.session_core import SessionRuntime
from agentm.sdk import AgentSession

_PROBE_MODEL = Model(
    id="probe-model",
    provider="probe",
    context_window=128_000,
    max_output_tokens=4_096,
)


class NeverStreams:
    """The stream function of a session that exists only to be inspected."""

    def __call__(
        self,
        *,
        messages: list[AgentMessage],
        model: Model,
        tools: list[Tool],
        system: str | None = None,
        signal: CancelSignal | None = None,
        thinking: ThinkingLevel = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        raise RuntimeError("a probe session never streams")


@dataclass(frozen=True, slots=True)
class FieldDifference:
    """One digest field that did not come back, and how it differs."""

    field: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    before: str
    after: str

    def describe(self) -> str:
        if self.added or self.removed:
            parts = []
            if self.added:
                parts.append("left behind: " + ", ".join(self.added))
            if self.removed:
                parts.append("failed to restore: " + ", ".join(self.removed))
            return f"{self.field}: " + "; ".join(parts)
        return f"{self.field}: reordered\n  before {self.before}\n  after  {self.after}"


def digest_differences(
    before: CompositionDigest,
    after: CompositionDigest,
) -> tuple[FieldDifference, ...]:
    """Field-by-field difference between two digests, ignoring nothing."""

    differences: list[FieldDifference] = []
    for field in dataclasses.fields(CompositionDigest):
        # A structural diff over the digest must read fields generically:
        # naming them here as well would let a field added to the digest escape
        # every revertibility check in the repo.
        before_value = getattr(before, field.name)  # code-health: ignore[AM021]
        after_value = getattr(after, field.name)  # code-health: ignore[AM021]
        if before_value == after_value:
            continue
        added, removed = _sequence_delta(before_value, after_value)
        differences.append(
            FieldDifference(
                field=field.name,
                added=added,
                removed=removed,
                before=repr(before_value),
                after=repr(after_value),
            )
        )
    return tuple(differences)


def _sequence_delta(
    before: object,
    after: object,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Multiset delta of two digest tuples; empty for scalars and reorderings."""

    # Every digest field is either a tuple of frozen entries or a scalar, and
    # only the tuples have an element-wise delta worth reporting.
    if type(before) is tuple and type(after) is tuple:
        surviving = list(after)
        removed: list[object] = []
        for item in before:
            if item in surviving:
                surviving.remove(item)
            else:
                removed.append(item)
        return tuple(repr(item) for item in surviving), tuple(
            repr(item) for item in removed
        )
    return (), ()


def _label(atom: ExtensionSpec | str) -> str:
    """Name an atom the way a failure message should read it.

    A spec's ``repr`` is its source, its digest and its config; a failure that
    leads with sixty characters of sha256 buries the field that differed.
    """

    if type(atom) is ExtensionSpec:
        return atom.module_path
    return str(atom)


@asynccontextmanager
async def probe_session(
    cwd: str,
    *,
    extensions: Sequence[ExtensionSpec] = (),
) -> AsyncIterator[AgentSession]:
    """A session that composes atoms and never runs a turn.

    The driver is deliberately left unstarted: starting it binds context
    policies and subscribes the provider's turn hook, both of which belong to
    the running session rather than to the atom under test.
    """

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=cwd,
            extensions=list(extensions),
            stream_fn=NeverStreams(),
            model=_PROBE_MODEL,
        )
    )
    try:
        yield session
    finally:
        await session.shutdown()


async def assert_revertible(
    session: SessionRuntime,
    atom: ExtensionSpec | str,
    config: Mapping[str, object] | None = None,
    *,
    residue: Sequence[str] = (),
) -> None:
    """Install ``atom`` into ``session``, uninstall it, and diff the digest.

    ``residue`` names the digest fields this atom is known not to restore, each
    of which must be justified where it is passed. Raises ``AssertionError``
    with a field-by-field account of anything that does not match.
    """

    label = _label(atom)
    before = composition_digest(session)
    await session.install_extension(atom, None if config is None else dict(config))
    installed = composition_digest(session)
    # Bookkeeping every install writes by definition. An atom that moves only
    # these has registered nothing, and asserting that nothing reverts proves
    # nothing about the inverse.
    bookkeeping = {"atoms", "composition_specs"}
    touched = {difference.field for difference in digest_differences(before, installed)}
    if not touched - bookkeeping:
        raise AssertionError(
            f"{label} registered nothing the digest can see, so asserting it "
            "reverts witnesses nothing; either the atom is inert under this "
            "config or the digest is missing a kind of registration"
        )

    if not session.uninstall_extension(atom):
        raise AssertionError(f"{label} reported nothing to uninstall")

    differences = digest_differences(before, composition_digest(session))
    by_field = {difference.field: difference for difference in differences}
    expected = set(residue)
    unexpected = [
        difference for difference in differences if difference.field not in expected
    ]
    if unexpected:
        raise AssertionError(
            f"{label} did not revert:\n"
            + "\n".join(difference.describe() for difference in unexpected)
        )
    stale = sorted(expected - by_field.keys())
    if stale:
        raise AssertionError(
            f"{label} now reverts {', '.join(stale)}, which is declared as "
            "residue; the declaration is stale and must be removed"
        )


__all__ = [
    "FieldDifference",
    "NeverStreams",
    "assert_revertible",
    "digest_differences",
    "probe_session",
]
