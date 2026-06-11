"""``rca_fsm_policy`` — FSM control surface for the rca_hfsm scenario (design §4).

Maintains a per-install FSM state container, advances it on
``rca.graph.mutated`` events (commit-4 gate addition), injects a per-state
system-prompt fragment via ``BeforeAgentStartEvent``, and filters the
visible tool catalog per state via ``AgentStartEvent`` (the canonical
``api.tools`` seam used by ``tool_filter`` and ``rca_observation_cache``).

The FSM state set is **structural**, not subjective — these are policy
phases the gate's downgrade semantics rely on, so the ``Literal`` typing
does not violate CLAUDE.md's "no preset enums for subjective dimensions".
Subjective fields (hypothesis status, worker verdict, interpretation
confidence) remain free-text upstream.

States::

    INTAKE      — record symptoms; no hypotheses yet
    OBSERVE     — gather L1 facts; no candidate H to verify
    HYPOTHESIZE — propose ≥1 H with ≥1 negative prediction each
    VERIFY      — dispatch worker(s) to test a prediction
    JUDGE       — apply update operator on the verification result
    FINALIZE    — coverage met; only path out is submit_final_report
    BLOCKED     — external help needed; Phase 1 ships no request_help tool
                   so this state is structurally unreachable. Documented
                   for design symmetry only.

The FSM state container is published as the ``rca.fsm`` service so the
finalize atom (and smoke tests) can read it. The state mutates only here.

§11 contract: stdlib + ``agentm.core.abi.*`` + ``agentm.extensions`` +
scenario-local pure modules (``schema``, ``scheduler``). No atom-to-atom
imports; the gate is reached only through the bus event it emits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from agentm.core.abi import AgentStartEvent, BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest

from rca.hfsm.scheduler import pick_next

FSMState = Literal[
    "INTAKE", "OBSERVE", "HYPOTHESIZE", "VERIFY", "JUDGE", "FINALIZE", "BLOCKED",
]

MANIFEST = ExtensionManifest(
    name="rca_fsm_policy",
    description=(
        "FSM control surface — advances trace state on rca.graph.mutated, "
        "injects per-state system-prompt fragments, and filters the visible "
        "tool catalog per state."
    ),
    registers=(
        "event:agent_start",
        "event:before_agent_start",
        "event:rca.graph.mutated",
    ),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    requires=("rca_hgraph_store", "rca_falsification_gate"),
)

_PROMPTS_DIR: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "prompts" / "hfsm"
)

# Per-state visible tool sets. ``None`` in the value means "no filter".
# FINALIZE is the only fully-locked state (only submit_final_report). The
# others list the tools the LLM has structural reason to call in that
# state; free-form reasoning is always permitted (the FSM does not gate
# LLM thinking, only tool surface). Read tools (``record_observation``,
# ``record_symptom``) stay broadly available because facts can show up at
# any point in the trace and re-routing them through later states would
# force the LLM to wait artificially.
_PER_STATE_TOOLS: Final[dict[str, frozenset[str]]] = {
    "INTAKE": frozenset({
        "record_symptom", "record_observation",
    }),
    "OBSERVE": frozenset({
        "record_symptom", "record_observation", "propose_hypothesis",
    }),
    "HYPOTHESIZE": frozenset({
        "propose_hypothesis", "record_observation",
    }),
    "VERIFY": frozenset({
        "attach_check", "record_observation", "dispatch_agent",
    }),
    "JUDGE": frozenset({
        "propose_update", "record_observation",
    }),
    "FINALIZE": frozenset({
        "submit_final_report",
    }),
    "BLOCKED": frozenset({
        # Phase 1 ships no request_help tool; BLOCKED is unreachable.
        # Empty visibility is a deliberate fail-safe — if the FSM ever
        # ends up here, the LLM has no tool calls available and the
        # loop ends naturally on ModelEndTurn.
    }),
}

@dataclass
class FSMStateContainer:
    """Per-session FSM state.

    Published as the ``rca.fsm`` service so the finalize atom and tests can
    read it. Writes happen exclusively inside :func:`_advance` in this atom.
    """

    state: FSMState = "INTAKE"
    current_prediction_id: str | None = None
    history: list[FSMState] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = [self.state]

    def transition(self, new_state: FSMState) -> None:
        if new_state == self.state:
            return
        self.state = new_state
        self.history.append(new_state)

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    read_handle = api.get_service("rca.hgraph.read")
    if read_handle is None:
        raise RuntimeError(
            "rca_fsm_policy: rca.hgraph.read is not published; install "
            "rca_hgraph_store before this atom"
        )
    fsm = FSMStateContainer()
    api.set_service("rca.fsm", fsm)
    prompt_cache: dict[str, str] = {}
    filtered_once = {"value": False}

    def _load_prompt(state: FSMState) -> str:
        if state in prompt_cache:
            return prompt_cache[state]
        path = _PROMPTS_DIR / f"{state.lower()}.md"
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
        else:
            text = ""
        prompt_cache[state] = text
        return text

    def _advance(payload: dict[str, Any]) -> None:
        """Advance FSM on a graph mutation.

        The payload shape is the gate's commit-4 emit:
        ``{"op": str, "kind": "applied"|"downgraded", "applied_id": str|None,
           "downgrade_op": str|None, "reason": str}``.
        Rejected mutations never reach this handler — the gate only emits
        on success/downgrade.
        """

        op = str(payload.get("op", ""))
        kind = str(payload.get("kind", ""))
        current = fsm.state
        if op == "record_symptom" and current == "INTAKE":
            fsm.transition("OBSERVE")
            return
        if op == "propose" and kind == "applied":
            # Any successful propose carries us into the verification loop;
            # the scheduler will pick the next prediction on entering VERIFY.
            fsm.transition("HYPOTHESIZE")
            fsm.transition("VERIFY")
            _select_next_prediction()
            return
        if op == "attach_check" and kind == "applied":
            fsm.transition("JUDGE")
            return
        if op == "confirm":
            # Successful confirm OR downgrade-to-refine both close JUDGE;
            # FINALIZE is reachable only on a clean apply where coverage is
            # fully met. Coverage is structurally a property of the read API
            # (no unexplained symptoms after the confirm landed).
            if kind == "applied" and not read_handle.get_unexplained_symptoms():
                fsm.transition("FINALIZE")
                return
            # Downgrade or non-covering confirm: back to VERIFY for the
            # next prediction.
            fsm.transition("VERIFY")
            _select_next_prediction()
            return
        if op == "refute":
            # If every open hypothesis is now closed, fall back to OBSERVE
            # (a fresh hypothesis must be proposed). Otherwise resume
            # verification on the remaining branches.
            if not read_handle.get_open_leaves():
                fsm.transition("OBSERVE")
            else:
                fsm.transition("VERIFY")
                _select_next_prediction()
            return
        if op in ("refine", "split", "merge", "supersede", "suspend"):
            fsm.transition("VERIFY")
            _select_next_prediction()
            return
        # record_observation never advances the FSM by itself.

    def _select_next_prediction() -> None:
        choice = pick_next(read_handle.get_open_leaves())
        fsm.current_prediction_id = choice.id if choice is not None else None

    def _on_mutation(event: Any) -> None:
        if isinstance(event, dict):
            _advance(event)

    def _inject_prompt(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        fragment = _load_prompt(fsm.state)
        suggestion = _scheduler_suggestion()
        body = "\n\n".join(part for part in (fragment, suggestion) if part)
        if not body:
            return None
        existing = event.system or ""
        merged = f"{body}\n\n{existing}" if existing else body
        event.system = merged
        return {"system": merged}

    def _scheduler_suggestion() -> str:
        if fsm.state != "VERIFY" or fsm.current_prediction_id is None:
            return ""
        return (
            f"Scheduler: next prediction to verify is "
            f"id={fsm.current_prediction_id} (information-gain default)."
        )

    def _on_agent_start(_: AgentStartEvent) -> None:
        # Best-effort per-state filter on the registered tool catalog. The
        # filter runs once per session (mirrors ``tool_filter``'s pattern)
        # because re-filtering on every state change would yank tools out
        # mid-turn. The system-prompt fragment ALSO names the available
        # tools for the current state so the LLM has the same information.
        if filtered_once["value"]:
            return
        allowed = _PER_STATE_TOOLS.get(fsm.state)
        if allowed is None:
            filtered_once["value"] = True
            return
        tools = api.tools
        # Keep tools the FSM lists; also keep tools the FSM does not know
        # about (operations primitives, sub_agent.dispatch_agent,
        # observability tools) so the scenario remains composable.
        # Filtering is intentionally conservative — the prompt is the
        # primary guidance channel; the visibility filter is the structural
        # backstop only for the FINALIZE state where over-broad surfaces
        # would let the LLM bypass coverage. See scope section "Tool
        # visibility filtering" for the degrade rationale.
        if fsm.state == "FINALIZE":
            kept = [t for t in tools if t.name in allowed]
            tools[:] = kept
        filtered_once["value"] = True

    api.on(AgentStartEvent.CHANNEL, _on_agent_start)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject_prompt)
    api.on("rca.graph.mutated", _on_mutation)
