"""V1 fail-stop integration test for the two-phase cognitive audit.

This is the **single fail-stop** for the V1 two-phase audit per
``.claude/designs/llmharness-cognitive-audit.md`` §11 and the AgentM
top-level CLAUDE.md "Core test positions" rule. It replaces the V0
schema-pinning smoke tests deleted in the 2026-05-08 hard cut.

Each scenario in this file is tied to a realistic disaster:

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
   V1 must surface it as exactly one ``llmharness.extractor_no_call``
   entry, with *no* synthesized ``drift=False`` verdict.

3. **Scenario C** (pure imports + JSON-Schema introspection) —
   protects against *schema contract drift* between
   ``submit_events`` / ``submit_verdict`` and the typed payloads in
   ``llmharness.schema``. Changes to ``EventKind`` / ``DriftType``
   that don't propagate to the tool schemas would let bad atoms
   slip through provider-side validation.

The test uses a stub provider routed through AgentM's
``inherit_provider`` builtin — child extractor / auditor sessions are
spawned with ``provider=None``, so the parent's stub serves all three
session purposes. The stub branches on the system prompt to decide
which canned response to emit (parent text vs ``submit_events`` call
vs ``submit_verdict`` call).

Pattern reused from
``tests/integration/test_sub_agent_lifecycle.py`` (parent repo) — the
``_install_provider_module`` / ``ProviderConfig`` shape is the
established cross-session stub-provider convention.
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

_EXTRACTOR_PROMPT_NEEDLE = "cognitive-audit **extractor**"
_AUDITOR_PROMPT_NEEDLE = "cognitive-audit *auditor*"


# --- stub provider ----------------------------------------------------------


class _TwoPhaseStubProvider:
    """Routes per-session canned responses by inspecting ``system``.

    The audit adapter spawns child sessions with ``provider=None`` so the
    ``inherit_provider`` builtin re-publishes the parent's
    :class:`ProviderConfig`. That means the parent's ``stream_fn`` is
    invoked for the parent **and** every extractor / auditor child.

    Disambiguation is by system-prompt needle (extractor / auditor have
    distinct system-prompt bodies; the parent has neither). Extractor
    behaviour is keyed off the current turn count so Scenario B can
    selectively drop ``submit_events`` on turn 2.
    """

    def __init__(self, *, mode: str) -> None:
        self.mode = mode  # "happy" or "drop_extractor_turn_2"
        self.parent_calls = 0
        self.extractor_calls = 0
        self.auditor_calls = 0

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
        del messages, model, tools, signal, thinking
        sys_text = system or ""
        if _EXTRACTOR_PROMPT_NEEDLE in sys_text:
            self.extractor_calls += 1
            return self._extractor_iter(self.extractor_calls)
        if _AUDITOR_PROMPT_NEEDLE in sys_text:
            self.auditor_calls += 1
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
            # Skip the terminal tool call → adapter must record
            # `extractor_no_call`. This is the V0-regression probe.
            msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="declining to submit")],
                timestamp=float(100 + n),
                stop_reason="end_turn",
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
        msg = AssistantMessage(
            role="assistant",
            content=[
                ToolCallBlock(
                    type="tool_call",
                    id=f"submit-verdict-{self.auditor_calls}",
                    name=SUBMIT_VERDICT_TOOL_NAME,
                    arguments={
                        "verdict": {
                            "drift": False,
                            "type": None,
                            "reminder": None,
                            "cited_cards": None,
                            "downstream_reaction": None,
                        }
                    },
                )
            ],
            timestamp=float(200 + self.auditor_calls),
            stop_reason="tool_use",
        )
        yield MessageEnd(message=msg)


def _install_provider_module(
    name: str, provider: _TwoPhaseStubProvider
) -> str:
    """Register the stub as an AgentM provider extension module.

    Mirrors the pattern in
    ``tests/integration/test_sub_agent_lifecycle.py``: a synthetic module
    is installed into ``sys.modules`` with an ``install`` callable that
    registers the stub via ``api.register_provider``.
    """
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
    """Build an ``AgentSessionConfig`` wired with the V1 audit adapter.

    Cards-tools and observability are explicitly disabled (config=None)
    to keep the child-session machinery minimal — the test only exercises
    the orchestration loop. ``mode`` selects sync vs async dispatch.
    """

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
    return [
        e
        for e in session.session_manager.get_active_branch()
        if e.type == entry_type
    ]


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
    provider_module = _install_provider_module(
        "tests._fake_twophase_happy_provider", provider
    )

    session = await AgentSession.create(
        _build_session_config(
            cwd=str(tmp_path), provider_module=provider_module
        )
    )

    for i in range(4):
        await session.prompt(f"user turn {i + 1}")

    # Audit work runs on a background worker; persisted state is settled
    # only after ``session.shutdown()`` drains the queue.
    await session.shutdown()

    # Phase-1 cursor: one per TurnEndEvent firing → exactly 4.
    cursors = _entries(session, _EXTRACTOR_CURSOR_ENTRY)
    assert len(cursors) == 4

    # Cursor monotonicity: ``last_turn_index`` non-decreasing across firings.
    last_turn_indices = [int(c.payload["last_turn_index"]) for c in cursors]
    assert last_turn_indices == sorted(last_turn_indices), (
        f"cursor drift: {last_turn_indices}"
    )

    # Audit events: stub emits one per Phase-1 firing → at least 4.
    audit_events = _entries(session, _AUDIT_EVENT_ENTRY)
    assert len(audit_events) >= 4

    # Phase-2 verdict: only ``turn_count == 3`` satisfies ``% k == 0``
    # in a 4-turn run with k=3. Exactly one verdict entry.
    verdicts = _entries(session, _VERDICT_ENTRY)
    assert len(verdicts) == 1, (
        f"expected exactly 1 verdict, got {len(verdicts)}"
    )

    # Auditor child should also have been invoked exactly once on the
    # provider — cheap cross-check that the verdict count above isn't
    # an entry-tree artifact.
    assert provider.auditor_calls == 1


# --- Scenario B: extractor failure-entry probe ------------------------------


@pytest.mark.asyncio
async def test_extractor_no_call_on_turn_2_does_not_synthesize_silent_verdict(
    tmp_path: Path,
) -> None:
    """Extractor declines to call submit_events on turn 2.

    Disaster guarded: V0's silent-fallback bug returning under a different
    name. When the extractor child exits without calling
    ``submit_events``, V1 must record an ``llmharness.extractor_no_call``
    entry and **must not** synthesize a ``drift=False`` verdict to fill
    the gap. Other turns must still produce ``audit_event`` entries
    normally so the failure is localized.
    """

    provider = _TwoPhaseStubProvider(mode="drop_extractor_turn_2")
    provider_module = _install_provider_module(
        "tests._fake_twophase_drop_provider", provider
    )

    session = await AgentSession.create(
        _build_session_config(
            cwd=str(tmp_path), provider_module=provider_module
        )
    )

    for i in range(4):
        await session.prompt(f"user turn {i + 1}")

    # Audit work runs on a background worker; persisted state is settled
    # only after ``session.shutdown()`` drains the queue.
    await session.shutdown()

    # Exactly one extractor_no_call entry, scoped to turn 2.
    no_call_entries = _entries(session, _EXTRACTOR_NO_CALL_ENTRY)
    assert len(no_call_entries) == 1
    payload = no_call_entries[0].payload
    assert "turn_window" in payload

    # Other turns still produce audit events: the stub emits one event per
    # successful extractor firing, and 3 of 4 firings succeed.
    audit_events = _entries(session, _AUDIT_EVENT_ENTRY)
    assert len(audit_events) >= 3

    # CRITICAL: no silent ``drift=False`` verdict synthesized to paper
    # over the extractor outage. Phase 2 *may* still fire at turn 3 with
    # whatever events were extracted; what we lock down here is that the
    # *extractor failure itself* does not produce a verdict entry.
    silent_fallbacks = [
        v
        for v in _entries(session, _VERDICT_ENTRY)
        if isinstance(v.payload, dict)
        and v.payload.get("drift") is False
        and v.payload.get("reminder") == ""
        and not v.payload.get("matched_event_ids")
        # A genuine auditor verdict reaches the auditor child; the stub
        # records one auditor_calls per such firing. We can therefore
        # cross-check that every verdict is paired with an actual auditor
        # invocation rather than being a silent synthesis.
    ]
    # The stub emits drift=False as its real auditor output, so the
    # *count* is allowed to be 1 — but only if the auditor child was
    # actually called that many times.
    assert len(silent_fallbacks) <= provider.auditor_calls


# --- Scenario B2: sync mode runs audit inline (dataset-collection use) ------


@pytest.mark.asyncio
async def test_sync_mode_persists_audit_inline_without_shutdown(
    tmp_path: Path,
) -> None:
    """``mode='sync'`` -> every turn's audit is fully persisted before
    ``session.prompt()`` returns, so dataset-collection runs can read the
    paired audit data immediately without awaiting shutdown.

    Disaster guarded: a regression that re-routes sync mode through the
    async worker would silently break the data-collection contract — the
    main agent would race ahead and the harness would lose pairings.
    """

    provider = _TwoPhaseStubProvider(mode="happy")
    provider_module = _install_provider_module(
        "tests._fake_twophase_sync_provider", provider
    )

    session = await AgentSession.create(
        _build_session_config(
            cwd=str(tmp_path), provider_module=provider_module, mode="sync"
        )
    )

    for i in range(4):
        await session.prompt(f"user turn {i + 1}")
        # Sync contract: after each prompt() returns, that turn's extractor
        # entry MUST already be persisted. We assert mid-loop so a
        # regression to async-via-shutdown-drain would be caught here.
        cursors = _entries(session, _EXTRACTOR_CURSOR_ENTRY)
        assert len(cursors) == i + 1, (
            f"after prompt #{i + 1}, expected {i + 1} cursors, got "
            f"{len(cursors)}"
        )

    # Phase-2 verdict at turn 3 — must also be persisted inline.
    verdicts = _entries(session, _VERDICT_ENTRY)
    assert len(verdicts) == 1
    assert provider.auditor_calls == 1

    await session.shutdown()


# --- Scenario C: schema contract probe --------------------------------------


def test_submit_events_schema_kind_enum_matches_event_kind_values() -> None:
    """``submit_events`` ``kind`` enum tracks ``EventKind`` exactly.

    Disaster guarded: schema contract drift between the JSON Schema
    embedded in the tool registration and the ``EventKind`` enum used
    by the typed payloads. V0 had two hand-listed copies that drifted
    silently; the V1 ``_enum_schema`` derivation must keep them
    locked.
    """

    enum_in_schema = SUBMIT_EVENTS_PARAMETERS["properties"]["events"]["items"][
        "properties"
    ]["kind"]["enum"]
    assert enum_in_schema == EVENT_KIND_VALUES


def test_submit_verdict_schema_enforces_drift_true_requires_type() -> None:
    """``submit_verdict`` schema embeds an ``if/then`` clause requiring
    ``type`` when ``drift=true``.

    Disaster guarded: V0's silent-drop of ``drift=true && type=null``.
    The provider edge must reject non-conforming payloads via
    ``if/then``; if that clause vanishes the failure mode resurfaces
    silently inside the adapter.
    """

    verdict_schema = SUBMIT_VERDICT_PARAMETERS["properties"]["verdict"]
    if_clause = verdict_schema.get("if")
    then_clause = verdict_schema.get("then")
    assert isinstance(if_clause, dict), "submit_verdict missing 'if' clause"
    assert isinstance(then_clause, dict), "submit_verdict missing 'then' clause"

    # ``if`` keys off ``drift == True``.
    if_drift = if_clause.get("properties", {}).get("drift", {})
    assert if_drift.get("const") is True

    # ``then`` requires ``type``, ``reminder``, and ``matched_event_ids``
    # — protects the live failure observed 2026-05-08 where
    # ``drift=true`` shipped with ``reminder.message`` (wrong key) and
    # empty ``matched_event_ids`` got silently dropped to ``reminder=""``
    # in the persisted verdict. Tightening here means the provider edge
    # rejects the call before the LLM even returns.
    then_required = then_clause.get("required", [])
    for field_name in ("type", "reminder", "matched_event_ids"):
        assert field_name in then_required, (
            f"submit_verdict 'then' clause must require {field_name!r} "
            f"when drift=true; got {then_required}"
        )

    then_props = then_clause.get("properties", {})
    reminder_then = then_props.get("reminder", {})
    assert reminder_then.get("type") == "object"
    assert "text" in reminder_then.get("required", []), (
        "submit_verdict reminder must require a 'text' field when "
        "drift=true — without it providers accepted reminders under "
        "arbitrary keys (message/body/...) and the adapter silently "
        "dropped them."
    )
    matched_then = then_props.get("matched_event_ids", {})
    assert matched_then.get("minItems") == 1


def test_submit_verdict_tool_rejects_drift_true_without_reminder_text() -> None:
    """Auditor child loop must let the LLM retry on a malformed verdict.

    Disaster guarded: the live 2026-05-08 incident where the auditor
    submitted ``drift=true`` with ``reminder.message`` (wrong key) and
    no ``matched_event_ids``. The schema's ``if/then`` is the first
    line of defense, but providers vary in how strictly they enforce
    it; the tool must validate again and return an ``is_error``
    ``ToolResult`` so the child loop continues, the LLM sees the error
    text, and retries — instead of ``ToolTerminate``-ing the child
    loop and shipping a silent ``reminder=""`` verdict.
    """

    from agentm.core.abi import FunctionTool, ToolContinue, ToolResult
    from agentm.harness.session import AgentSession

    # The tool function isn't directly exposed; spin up a tiny session
    # that has only the submit_verdict tool installed and inspect the
    # registered FunctionTool's fn.
    captured: list[FunctionTool] = []

    class _Capture:
        def register_tool(self, tool: FunctionTool) -> None:
            captured.append(tool)

        @property
        def cwd(self) -> str:
            return "/tmp"

    from llmharness.audit.auditor.submit_tool import install as _submit_install

    _submit_install(_Capture(), {})  # type: ignore[arg-type]
    assert len(captured) == 1
    submit = captured[0]

    import asyncio

    async def _drive() -> tuple[Any, Any]:
        # Bad: drift=true but reminder uses 'message' key + no matched ids.
        bad = await submit.fn(
            {
                "verdict": {
                    "drift": True,
                    "type": "task_drift",
                    "reminder": {"message": "you are off-track"},
                }
            }
        )
        # Good: drift=false; tool should ToolTerminate normally.
        good = await submit.fn(
            {
                "verdict": {
                    "drift": False,
                    "type": None,
                }
            }
        )
        return bad, good

    bad, good = asyncio.run(_drive())

    # Bad call: continues the loop with an is_error ToolResult so the
    # auditor LLM gets another chance.
    assert isinstance(bad, ToolResult), (
        f"malformed submit_verdict must return a bare ToolResult, got {type(bad).__name__}"
    )
    assert bad.is_error is True
    body = "".join(
        block.text for block in bad.content if hasattr(block, "text")
    )
    assert "submit_verdict rejected" in body
    assert "Reissue submit_verdict" in body

    # Good call: terminates the child loop normally.
    from agentm.core.abi import ToolTerminate
    assert isinstance(good, ToolTerminate)
    del AgentSession, ToolContinue  # silence unused-import lint


def test_compose_factories_are_importable_and_callable() -> None:
    """The two phase-extension factories survive at the documented import
    paths.

    Disaster guarded: the V0 single ``compose_extensions`` factory was
    renamed; downstream imports must use the V1 split names. A rename
    or path move would break adapter wiring at runtime.
    """

    extractor_exts = compose_extractor_extensions(
        cards_tools_config=None, observability_config=None
    )
    auditor_exts = compose_auditor_extensions(
        cards_tools_config=None, observability_config=None
    )

    # The submit-tool extension and the system-prompt extension are the
    # invariant pair; cards / observability are knob-controlled and can
    # be dropped. We only assert the invariant pieces.
    extractor_modules = {mod for mod, _cfg in extractor_exts}
    auditor_modules = {mod for mod, _cfg in auditor_exts}
    assert "llmharness.audit.extractor.submit_tool" in extractor_modules
    assert "agentm.extensions.builtin.system_prompt" in extractor_modules
    assert "llmharness.audit.auditor.submit_tool" in auditor_modules
    assert "agentm.extensions.builtin.system_prompt" in auditor_modules
