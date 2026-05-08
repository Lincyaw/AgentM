"""Typed coercion of the ``submit_verdict`` tool-call arguments.

The auditor terminates by calling ``submit_verdict(verdict=...)``. The
kernel records that call as a :class:`ToolCallBlock` whose ``arguments``
is a ``dict[str, Any]`` validated against
:data:`llmharness.audit.auditor.submit_tool.SUBMIT_VERDICT_PARAMETERS`.
This module gives the adapter a typed view over that dict — coercing it
to a :class:`llmharness.schema.Verdict` so downstream code never sees
``Any``.

A malformed payload (missing ``verdict`` object, missing ``drift``, or
``drift=true`` with no resolvable ``type``) raises
:class:`AuditorOutputError`. The adapter catches that and writes an
``audit_no_call`` entry — making the failure visible in the entry stream
rather than silently downgrading to a no-drift verdict.

The ``if/then`` block in the tool schema is supposed to reject
``drift=true && type=null`` at the provider edge, but we still validate
defensively here in case a provider strips the ``if/then`` clause or a
test path bypasses provider-side validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...schema import DriftType, Verdict


class AuditorOutputError(Exception):
    """Raised when the auditor's ``submit_verdict`` payload is malformed.

    The adapter records this as an ``audit_error`` entry on the session
    entry tree (see design §8) so the failure is visible to operators
    rather than silently collapsed to a no-drift verdict.
    """


def _coerce_str_list_or_none(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    return [item for item in raw if isinstance(item, str)]


@dataclass(frozen=True)
class RawVerdictOutput:
    """Typed view over the ``{verdict: {...}}`` payload of submit_verdict.

    Use :meth:`from_dict` to parse and :meth:`to_verdict` to materialize
    a :class:`Verdict` for the adapter. Both methods raise
    :class:`AuditorOutputError` on shape violations.
    """

    drift: bool
    type: DriftType | None
    reminder: dict[str, Any] | None
    cited_cards: list[str] | None
    downstream_reaction: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RawVerdictOutput:
        """Parse the tool-call ``arguments`` dict into a typed view.

        Raises :class:`AuditorOutputError` on:
        - missing or non-dict ``verdict`` key,
        - missing or non-bool ``drift`` field,
        - ``drift=true`` with no resolvable ``type`` (closes the V0
          silent-drop bug as a defensive backstop to the schema's
          ``if/then`` clause).
        """

        verdict_raw = raw.get("verdict")
        if not isinstance(verdict_raw, dict):
            raise AuditorOutputError(
                "submit_verdict payload missing object-typed 'verdict' field"
            )

        if "drift" not in verdict_raw:
            raise AuditorOutputError(
                "submit_verdict.verdict missing required 'drift' field"
            )
        drift_raw = verdict_raw["drift"]
        if not isinstance(drift_raw, bool):
            raise AuditorOutputError(
                f"submit_verdict.verdict.drift must be bool, got {type(drift_raw).__name__}"
            )

        type_raw = verdict_raw.get("type")
        drift_type: DriftType | None = None
        if isinstance(type_raw, str) and type_raw:
            try:
                drift_type = DriftType(type_raw)
            except ValueError as exc:
                raise AuditorOutputError(
                    f"submit_verdict.verdict.type {type_raw!r} is not a valid DriftType"
                ) from exc
        elif type_raw is not None:
            raise AuditorOutputError(
                "submit_verdict.verdict.type must be a string or null"
            )

        if drift_raw and drift_type is None:
            # Defensive backstop to the schema's if/then clause: a model
            # that returns drift=true with type=null should have been
            # rejected by the provider already, but if it leaks through
            # we surface it as a named failure rather than silently
            # downgrading the reminder.
            raise AuditorOutputError(
                "submit_verdict.verdict has drift=true but no resolvable 'type' "
                "(violates if/then schema clause)"
            )

        reminder_raw = verdict_raw.get("reminder")
        reminder: dict[str, Any] | None
        if reminder_raw is None:
            reminder = None
        elif isinstance(reminder_raw, dict):
            reminder = dict(reminder_raw)
        else:
            raise AuditorOutputError(
                "submit_verdict.verdict.reminder must be an object or null"
            )

        downstream_raw = verdict_raw.get("downstream_reaction")
        downstream: str | None
        if downstream_raw is None:
            downstream = None
        elif isinstance(downstream_raw, str):
            downstream = downstream_raw
        else:
            raise AuditorOutputError(
                "submit_verdict.verdict.downstream_reaction must be string or null"
            )

        return cls(
            drift=drift_raw,
            type=drift_type,
            reminder=reminder,
            cited_cards=_coerce_str_list_or_none(verdict_raw.get("cited_cards")),
            downstream_reaction=downstream,
        )

    def to_verdict(self) -> Verdict | None:
        """Materialize a :class:`Verdict` for the adapter.

        Returns ``None`` when the payload encoded a silent verdict
        (``drift=false``) — the adapter writes a verdict entry but does
        not arm a pending reminder. Returns a populated :class:`Verdict`
        when ``drift=true``; the reminder body is extracted from the
        ``reminder`` dict's ``text`` / ``body`` field if present.
        """

        if not self.drift:
            return Verdict(
                drift=False,
                type=None,
                reminder="",
                cited_cards=list(self.cited_cards or []),
                downstream_reaction=self.downstream_reaction,
            )

        # Phase-2 reminder is structured (object) rather than free string;
        # we accept either ``text`` or ``body`` as the rendered advisory.
        reminder_text = ""
        if self.reminder is not None:
            for key in ("text", "body"):
                value = self.reminder.get(key)
                if isinstance(value, str) and value:
                    reminder_text = value
                    break

        # ``self.type`` is guaranteed non-None when drift=True (from_dict
        # raises otherwise); assert for the type checker.
        assert self.type is not None
        return Verdict(
            drift=True,
            type=self.type,
            reminder=reminder_text,
            cited_cards=list(self.cited_cards or []),
            downstream_reaction=self.downstream_reaction,
        )


__all__ = ["AuditorOutputError", "RawVerdictOutput"]
