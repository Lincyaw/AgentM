"""V2 fail-stop integration test for the two-phase cognitive audit.

Each scenario is tied to a realistic disaster:

1. **Scenario A** (happy path, ``k=3``, 4 turns) — protects against the
   *Phase trigger regression*. If Phase 2 fired every turn we would
   see token blowup in production; if it never fired the watchdog is
   silently dead. We assert exactly 1 verdict over 4 turns. It also
   pins *cursor monotonicity* — extractor re-extracting the same
   window or skipping a window manifests as a non-monotonic
   ``last_turn_index`` series.

2. **Scenario B** (extractor declines to call ``submit_events`` on
   turn 2) — protects against V0's *silent-fallback regression*.
   V0 collapsed extraction outage to a clean ``Verdict(drift=False)``;
   V2 must surface it as exactly one ``llmharness.extractor_no_call``
   entry, with no synthesized silent verdict from the extractor failure
   itself.

3. **Scenario C** (pure imports + JSON-Schema introspection) —
   protects against *schema contract drift* between ``submit_events`` /
   ``submit_verdict`` and the typed payloads in ``llmharness.schema``.
   V2: no ``if/then`` clause; V2 required fields are ``surface_reminder``,
   ``reminder_text``, ``continuation_notes``, ``matched_event_ids``,
   ``cited_cards``.

4. **Scenario D** (continuation_notes forwarding) — protects against the
   new plumbing that carries the auditor's ``continuation_notes`` from
   firing N into the payload for firing N+1. If this breaks the auditor
   loses its cross-firing memory.

The test uses a stub provider routed through AgentM's ``inherit_provider``
builtin. The stub branches on the system prompt to decide which canned
response to emit.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from agentm.core.abi import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
)
from agentm.harness.extension import ProviderConfig
from agentm.harness.session import AgentSession, AgentSessionConfig

# Public-surface imports the schema-contract probe (Scenario C) pins:
from llmharness.audit._enum_schema import EVENT_KIND_VALUES
from llmharness.audit.auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    compose_auditor_extensions,
)
from llmharness.audit.auditor.submit_tool import SUBMIT_VERDICT_PARAMETERS
from llmharness.audit.entry_types import EXTRACTOR_INVALID
from llmharness.audit.extractor import (
    SUBMIT_EVENTS_TOOL_NAME,
    compose_extractor_extensions,
)
from llmharness.audit.extractor.submit_tool import SUBMIT_EVENTS_PARAMETERS

# --- shared constants -------------------------------------------------------

_AUDIT_EVENT_ENTRY = "llmharness.audit_event"
_VERDICT_ENTRY = "llmharness.verdict"
_EXTRACTOR_CURSOR_ENTRY = "llmharness.extractor_cursor"
_EXTRACTOR_NO_CALL_ENTRY = "llmharness.extractor_no_call"
_EXTRACTOR_INVALID_ENTRY = EXTRACTOR_INVALID  # "llmharness.extractor_invalid"

_EXTRACTOR_PROMPT_NEEDLE = "cognitive-audit **extractor**"
_AUDITOR_PROMPT_NEEDLE = "cognitive-audit *auditor*"

# V2 silent verdict payload the stub auditor emits.
_V2_SILENT_VERDICT = {
    "verdict": {
        "surface_reminder": False,
        "reminder_text": "",
        "continuation_notes": ["recheck event #1 next firing"],
        "matched_event_ids": [],
        "cited_cards": [],
    }
}

# V2 active verdict (surface_reminder=True) for Scenario D.
_V2_ACTIVE_VERDICT = {
    "verdict": {
        "surface_reminder": True,
        "reminder_text": "consider whether the dropped branch is still open",
        "continuation_notes": ["verify event #5 was resolved"],
        "matched_event_ids": [1],
        "cited_cards": [],
    }
}


# --- stub provider ----------------------------------------------------------


class _TwoPhaseStubProvider:
    """Routes per-session canned responses by inspecting ``system``.

    The audit adapter spawns child sessions with ``provider=None`` so the
    ``inherit_provider`` builtin re-publishes the parent's ProviderConfig.
    That means the parent's ``stream_fn`` is invoked for the parent AND
    every extractor / auditor child.

    Disambiguation is by system-prompt needle. Extractor behaviour is keyed
    off the current turn count so Scenario B can selectively drop
    ``submit_events`` on turn 2.
    """

    def __init__(self, *, mode: str) -> None:
        self.mode = mode  # "happy", "drop_extractor_turn_2", "active_verdict"
        self.parent_calls = 0
        self.extractor_calls = 0
        self.auditor_calls = 0
        # Capture the payload the auditor child received so tests can inspect it.
        self.auditor_payloads: list[Any] = []

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del model, tools, signal, thinking
        sys_text = system or ""
        if _EXTRACTOR_PROMPT_NEEDLE in sys_text:
            self.extractor_calls += 1
            return self._extractor_iter(self.extractor_calls)
        if _AUDITOR_PROMPT_NEEDLE in sys_text:
            self.auditor_calls += 1
            # Capture the prompt message so tests can inspect the payload.
            if messages:
                self.auditor_payloads.append(messages[-1] if messages else None)
            return self._auditor_iter()
        # Parent main agent: emit a plain assistant text and end the turn.
        self.parent_calls += 1
        return self._parent_iter(self.parent_calls)

    async def _parent_iter(self, n: int) -> AsyncIterator[AssistantStreamEvent]:
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=f"main-turn-{n}")],
            timestamp=float(n),
            stop_reason="end_turn",
        )
        yield MessageEnd(message=msg)

    async def _extractor_iter(self, n: int) -> AsyncIterator[AssistantStreamEvent]:
        if self.mode == "drop_extractor_turn_2" and n == 2:
            # Skip the terminal tool call → adapter records extractor_no_call.
            msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="declining to submit")],
                timestamp=float(100 + n),
                stop_reason="end_turn",
            )
            yield MessageEnd(message=msg)
            return
        if self.mode == "invalid_graph":
            # Submit an event with an unresolved ref (id=9999 does not exist).
            # The Phase 1 validator must reject this batch.
            msg = AssistantMessage(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id=f"submit-events-{n}",
                        name=SUBMIT_EVENTS_TOOL_NAME,
                        arguments={
                            "events": [
                                {
                                    "kind": EVENT_KIND_VALUES[0],  # task
                                    "summary": f"task event turn {n}",
                                    "source_turns": [],
                                    "refs": [],
                                },
                                {
                                    "kind": EVENT_KIND_VALUES[1],  # hypothesis
                                    "summary": f"hypothesis with dangling ref turn {n}",
                                    "source_turns": [],
                                    "refs": [9999],  # unresolved — triggers violation
                                },
                            ]
                        },
                    )
                ],
                timestamp=float(100 + n),
                stop_reason="tool_use",
            )
            yield MessageEnd(message=msg)
            return
        msg = AssistantMessage(
            role="assistant",
            content=[
                ToolCallBlock(
                    type="tool_call",
                    id=f"submit-events-{n}",
                    name=SUBMIT_EVENTS_TOOL_NAME,
                    arguments={
                        "events": [
                            {
                                "kind": EVENT_KIND_VALUES[0],
                                "summary": f"synthetic event for turn {n}",
                                "source_turns": [n - 1],
                                "refs": [],
                            }
                        ]
                    },
                )
            ],
            timestamp=float(100 + n),
            stop_reason="tool_use",
        )
        yield MessageEnd(message=msg)

    async def _auditor_iter(self) -> AsyncIterator[AssistantStreamEvent]:
        # For "active_verdict" mode emit an active reminder on the first firing,
        # silent on subsequent.
        if self.mode == "active_verdict" and self.auditor_calls == 1:
            verdict_payload = _V2_ACTIVE_VERDICT
        else:
            verdict_payload = _V2_SILENT_VERDICT
        msg = AssistantMessage(
            role="assistant",
            content=[
                ToolCallBlock(
                    type="tool_call",
                    id=f"submit-verdict-{self.auditor_calls}",
                    name=SUBMIT_VERDICT_TOOL_NAME,
                    arguments=verdict_payload,
                )
            ],
            timestamp=float(200 + self.auditor_calls),
            stop_reason="tool_use",
        )
        yield MessageEnd(message=msg)


def _install_provider_module(name: str, provider: _TwoPhaseStubProvider) -> str:
    """Register the stub as an AgentM provider extension module."""
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-twophase",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-twophase",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-twophase",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


def _build_session_config(
    *, cwd: str, provider_module: str, mode: str = "async"
) -> AgentSessionConfig:
    """Build an AgentSessionConfig wired with the V2 audit adapter."""
    return AgentSessionConfig(
        cwd=cwd,
        provider=(provider_module, {}),
        extensions=[
            (
                "llmharness.adapters.agentm",
                {
                    "mode": mode,
                    "audit_interval_turns": 3,
                    "cards_tools_config": None,
                    "observability_config": None,
                },
            ),
        ],
    )


def _entries(session: AgentSession, entry_type: str) -> list[Any]:
    """Return all branch entries with the given namespaced type."""
    return [e for e in session.session_manager.get_active_branch() if e.type == entry_type]


# --- Scenario A: happy-path 4-turn dialog at k=3 ----------------------------


@pytest.mark.asyncio
async def test_happy_path_4_turns_at_k3_fires_phase2_exactly_once(
    tmp_path: Path,
) -> None:
    """4 user turns, k=3 -> Phase 1 fires 4 times, Phase 2 once at turn 3.

    Disaster guarded: Phase trigger regression. ``(turn_count % k) == 0``
    must select exactly turn 3 in a 4-turn run; firing every turn would
    blow up token budget, firing zero times silently kills the watchdog.
    """
    provider = _TwoPhaseStubProvider(mode="happy")
    provider_module = _install_provider_module("tests._fake_twophase_happy_provider", provider)

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module)
    )

    for i in range(4):
        await session.prompt(f"user turn {i + 1}")

    await session.shutdown()

    # Phase-1 cursor: one per TurnEndEvent firing → exactly 4.
    cursors = _entries(session, _EXTRACTOR_CURSOR_ENTRY)
    assert len(cursors) == 4

    # Cursor monotonicity: last_turn_index non-decreasing.
    last_turn_indices = [int(c.payload["last_turn_index"]) for c in cursors]
    assert last_turn_indices == sorted(last_turn_indices), f"cursor drift: {last_turn_indices}"

    # Audit events: stub emits one per Phase-1 firing → at least 4.
    audit_events = _entries(session, _AUDIT_EVENT_ENTRY)
    assert len(audit_events) >= 4

    # Phase-2 verdict: only turn 3 satisfies % k == 0 in a 4-turn run.
    verdicts = _entries(session, _VERDICT_ENTRY)
    assert len(verdicts) == 1, f"expected exactly 1 verdict, got {len(verdicts)}"

    # Cross-check: auditor child invoked exactly once.
    assert provider.auditor_calls == 1

    # V2 shape check: verdict payload uses surface_reminder, not drift.
    v_payload = verdicts[0].payload
    assert "surface_reminder" in v_payload, "V2 verdict must have surface_reminder"
    assert "drift" not in v_payload, "V1 drift field must be absent in V2 verdict"
    assert "continuation_notes" in v_payload, "V2 verdict must have continuation_notes"


# --- Scenario B: extractor failure-entry probe ------------------------------


@pytest.mark.asyncio
async def test_extractor_no_call_on_turn_2_does_not_synthesize_silent_verdict(
    tmp_path: Path,
) -> None:
    """Extractor declines to call submit_events on turn 2.

    Disaster guarded: V0's silent-fallback bug. When the extractor exits
    without calling submit_events, V2 must record an
    ``llmharness.extractor_no_call`` entry and must NOT synthesize a
    silent verdict to fill the gap.
    """
    provider = _TwoPhaseStubProvider(mode="drop_extractor_turn_2")
    provider_module = _install_provider_module("tests._fake_twophase_drop_provider", provider)

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module)
    )

    for i in range(4):
        await session.prompt(f"user turn {i + 1}")

    await session.shutdown()

    # Exactly one extractor_no_call entry, scoped to turn 2.
    no_call_entries = _entries(session, _EXTRACTOR_NO_CALL_ENTRY)
    assert len(no_call_entries) == 1
    payload = no_call_entries[0].payload
    assert "turn_window" in payload

    # Other turns still produce audit events (3 of 4 firings succeed).
    audit_events = _entries(session, _AUDIT_EVENT_ENTRY)
    assert len(audit_events) >= 3

    # CRITICAL: no silent verdict synthesized from the extractor outage.
    silent_fallbacks = [
        v
        for v in _entries(session, _VERDICT_ENTRY)
        if isinstance(v.payload, dict)
        and v.payload.get("surface_reminder") is False
        and not v.payload.get("reminder_text")
        and not v.payload.get("matched_event_ids")
    ]
    # The stub emits surface_reminder=False as its real auditor output,
    # so the *count* is allowed to be 1 — but only if an auditor child
    # was actually called that many times.
    assert len(silent_fallbacks) <= provider.auditor_calls


# --- Scenario B2: sync mode -------------------------------------------------


@pytest.mark.asyncio
async def test_sync_mode_persists_audit_inline_without_shutdown(
    tmp_path: Path,
) -> None:
    """``mode='sync'`` -> every turn's audit is fully persisted before
    ``session.prompt()`` returns.

    Disaster guarded: a regression that re-routes sync mode through the
    async worker would silently break the data-collection contract.
    """
    provider = _TwoPhaseStubProvider(mode="happy")
    provider_module = _install_provider_module("tests._fake_twophase_sync_provider", provider)

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module, mode="sync")
    )

    for i in range(4):
        await session.prompt(f"user turn {i + 1}")
        cursors = _entries(session, _EXTRACTOR_CURSOR_ENTRY)
        assert len(cursors) == i + 1, (
            f"after prompt #{i + 1}, expected {i + 1} cursors, got {len(cursors)}"
        )

    # Phase-2 verdict at turn 3 — must also be persisted inline.
    verdicts = _entries(session, _VERDICT_ENTRY)
    assert len(verdicts) == 1
    assert provider.auditor_calls == 1

    await session.shutdown()


# --- Scenario C: schema contract probe (V2) ---------------------------------


def test_submit_events_schema_kind_enum_matches_event_kind_values() -> None:
    """``submit_events`` ``kind`` enum tracks ``EventKind`` exactly.

    Disaster guarded: schema contract drift between the JSON Schema
    embedded in the tool registration and the ``EventKind`` enum.
    """
    enum_in_schema = SUBMIT_EVENTS_PARAMETERS["properties"]["events"]["items"]["properties"][
        "kind"
    ]["enum"]
    assert enum_in_schema == EVENT_KIND_VALUES


def test_submit_verdict_v2_schema_has_required_fields() -> None:
    """V2 ``submit_verdict`` schema requires all five V2 fields.

    Disaster guarded: V2 schema drift — if a field is accidentally
    removed from ``required``, the LLM may omit it and the adapter
    silently loses continuity_notes or reminder_text.
    """
    verdict_schema = SUBMIT_VERDICT_PARAMETERS["properties"]["verdict"]

    # V2 must have no if/then clause.
    assert "if" not in verdict_schema, (
        "V2 submit_verdict schema must NOT have an 'if' clause; "
        "that was a V1 pattern for DriftType enforcement"
    )
    assert "then" not in verdict_schema, "V2 submit_verdict schema must NOT have a 'then' clause"

    # V2 required fields.
    required = verdict_schema.get("required", [])
    for field_name in (
        "surface_reminder",
        "reminder_text",
        "continuation_notes",
        "matched_event_ids",
        "cited_cards",
    ):
        assert field_name in required, (
            f"V2 submit_verdict 'required' must include {field_name!r}; got {required}"
        )

    # V1 fields must be absent.
    props = verdict_schema.get("properties", {})
    for v1_field in ("drift", "type", "reminder", "downstream_reaction"):
        assert v1_field not in props, (
            f"V1 field {v1_field!r} must be removed from V2 submit_verdict schema"
        )


def test_submit_verdict_tool_rejects_surface_reminder_true_without_text() -> None:
    """Auditor child loop must let the LLM retry when surface_reminder=True
    but reminder_text is empty.

    Disaster guarded: the V2 analog of the V1 silent-drop bug — the
    adapter must not silently accept an active verdict with no text.
    The tool must return an is_error ToolResult so the LLM retries.
    """
    from llmharness.audit.auditor.submit_tool import install as _submit_install

    captured: list[Any] = []

    class _Capture:
        def register_tool(self, tool: Any) -> None:
            captured.append(tool)

        @property
        def cwd(self) -> str:
            return "/tmp"

    _submit_install(_Capture(), {})  # type: ignore[arg-type]
    assert len(captured) == 1
    submit = captured[0]

    import asyncio

    from agentm.core.abi import ToolResult, ToolTerminate

    async def _drive() -> tuple[Any, Any]:
        # Bad: surface_reminder=True but reminder_text is empty.
        bad = await submit.fn(
            {
                "verdict": {
                    "surface_reminder": True,
                    "reminder_text": "",
                    "continuation_notes": [],
                    "matched_event_ids": [1],
                    "cited_cards": [],
                }
            }
        )
        # Good: surface_reminder=False with empty text — silent verdict.
        good = await submit.fn(
            {
                "verdict": {
                    "surface_reminder": False,
                    "reminder_text": "",
                    "continuation_notes": [],
                    "matched_event_ids": [],
                    "cited_cards": [],
                }
            }
        )
        return bad, good

    bad, good = asyncio.run(_drive())

    # Bad call: continues the loop with an is_error ToolResult.
    assert isinstance(bad, ToolResult), (
        f"malformed submit_verdict must return a bare ToolResult, got {type(bad).__name__}"
    )
    assert bad.is_error is True
    body = "".join(block.text for block in bad.content if hasattr(block, "text"))
    assert "submit_verdict rejected" in body
    assert "Reissue submit_verdict" in body

    # Good call: terminates the child loop.
    assert isinstance(good, ToolTerminate)


def test_compose_factories_are_importable_and_callable() -> None:
    """The two phase-extension factories survive at the documented import paths."""
    extractor_exts = compose_extractor_extensions(
        cards_tools_config=None, observability_config=None
    )
    auditor_exts = compose_auditor_extensions(cards_tools_config=None, observability_config=None)

    extractor_modules = {mod for mod, _cfg in extractor_exts}
    auditor_modules = {mod for mod, _cfg in auditor_exts}
    assert "llmharness.audit.extractor.submit_tool" in extractor_modules
    assert "agentm.extensions.builtin.system_prompt" in extractor_modules
    assert "llmharness.audit.auditor.submit_tool" in auditor_modules
    assert "agentm.extensions.builtin.system_prompt" in auditor_modules


# --- Scenario D: continuation_notes forwarding ------------------------------


@pytest.mark.asyncio
async def test_continuation_notes_reach_next_auditor_firing(
    tmp_path: Path,
) -> None:
    """continuation_notes from firing N are present in the auditor payload
    at firing N+1.

    Disaster guarded: the new continuation_notes plumbing. If
    _drain_auditor fails to pass last_continuation_notes into the
    _run_auditor payload, the auditor loses its cross-firing memory
    silently — it just receives an empty list where its own notes
    should be.

    We run 6 turns with k=3 so the auditor fires twice (at turns 3 and 6).
    The first firing emits active_verdict whose continuation_notes =
    ["verify event #5 was resolved"]. The second firing must see that
    list in its input payload as continuation_notes_from_prior_firing.
    """
    import json

    provider = _TwoPhaseStubProvider(mode="active_verdict")
    provider_module = _install_provider_module(
        "tests._fake_twophase_continuation_provider", provider
    )

    session = await AgentSession.create(
        _build_session_config(cwd=str(tmp_path), provider_module=provider_module)
    )

    for i in range(6):
        await session.prompt(f"user turn {i + 1}")

    await session.shutdown()

    # Two auditor firings expected.
    assert provider.auditor_calls == 2, f"expected 2 auditor firings, got {provider.auditor_calls}"

    # The second auditor firing's payload must contain continuation_notes
    # from the first verdict.
    assert len(provider.auditor_payloads) >= 2, "expected at least 2 captured auditor payloads"
    second_payload_raw = provider.auditor_payloads[1]
    # The payload is the last message object, whose content[0].text is the
    # JSON string sent to the child session.
    content_blocks = getattr(second_payload_raw, "content", None) or []
    payload_text = ""
    for block in content_blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            payload_text = text
            break

    assert payload_text, "second auditor payload must have text content"
    payload_dict = json.loads(payload_text)
    prior_notes = payload_dict.get("continuation_notes_from_prior_firing")
    assert isinstance(prior_notes, list), (
        "continuation_notes_from_prior_firing must be a list in the auditor payload"
    )
    assert "verify event #5 was resolved" in prior_notes, (
        f"expected first-firing continuation_notes in second payload, got {prior_notes}"
    )

    # V2 shape: the verdict entries also carry continuation_notes.
    verdicts = _entries(session, _VERDICT_ENTRY)
    assert len(verdicts) == 2
    first_v = verdicts[0].payload
    assert first_v.get("surface_reminder") is True
    assert first_v.get("reminder_text") == ("consider whether the dropped branch is still open")
    assert "verify event #5 was resolved" in first_v.get("continuation_notes", [])


# --- Scenario E: extractor_invalid on graph-validation failure --------------


@pytest.mark.asyncio
async def test_extractor_invalid_skips_auditor_and_holds_cursor(
    tmp_path: Path,
) -> None:
    """Stub extractor returns events with an unresolved ref.

    Assertions:
    - One ``llmharness.extractor_invalid`` entry written with non-empty
      ``violations`` list.
    - Zero ``llmharness.audit_event`` entries for that (only) firing.
    - The extractor cursor is NOT advanced (last_turn_index stays at -1
      / no cursor entries, since each firing fails validation).
    - The auditor is NOT spawned (auditor_calls == 0 across all turns,
      even at the k-turn boundary).

    Disaster guarded: Phase 1 graph-validation regression. If bad events
    were committed to the graph, the auditor would reason over incoherent
    input. If the auditor still ran on a failed extraction batch it would
    be judging a stale graph without the new window's events — the
    extractor_invalid entry signals that the window must be retried.
    """
    provider = _TwoPhaseStubProvider(mode="invalid_graph")
    provider_module = _install_provider_module(
        "tests._fake_twophase_invalid_graph_provider", provider
    )

    # Use k=1 so the auditor would fire every turn if validation succeeded.
    # This makes it easy to assert it was NOT spawned.
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[
                (
                    "llmharness.adapters.agentm",
                    {
                        "mode": "sync",  # sync so assertions hold without shutdown wait
                        "audit_interval_turns": 1,
                        "cards_tools_config": None,
                        "observability_config": None,
                    },
                ),
            ],
        )
    )

    # Drive one turn end.
    await session.prompt("user turn 1")

    await session.shutdown()

    # Must have exactly one extractor_invalid entry.
    invalid_entries = _entries(session, _EXTRACTOR_INVALID_ENTRY)
    assert len(invalid_entries) == 1, (
        f"expected 1 extractor_invalid entry, got {len(invalid_entries)}"
    )
    payload = invalid_entries[0].payload
    assert isinstance(payload, dict)
    violations = payload.get("violations")
    assert isinstance(violations, list) and len(violations) > 0, (
        f"extractor_invalid payload must have non-empty violations: {payload}"
    )
    assert "turn_window" in payload, "extractor_invalid payload must include turn_window"

    # No audit_event entries must be written (graph was rejected).
    audit_events = _entries(session, _AUDIT_EVENT_ENTRY)
    assert len(audit_events) == 0, (
        f"expected 0 audit_event entries after validation failure, got {len(audit_events)}"
    )

    # Cursor must NOT be advanced — extraction failed, window must be retried.
    cursors = _entries(session, _EXTRACTOR_CURSOR_ENTRY)
    assert len(cursors) == 0, (
        f"cursor must not advance on validation failure, got {len(cursors)} cursor entries"
    )

    # Auditor must NOT have been spawned.
    assert provider.auditor_calls == 0, (
        f"auditor must not run when extractor validation fails, "
        f"got auditor_calls={provider.auditor_calls}"
    )
