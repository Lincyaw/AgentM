"""Extractor context atom — creates state and injects the system prompt."""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

from llmharness.schema import Edge, Event

from .prompt import load_extractor_prompt
from .tools import ExtractionState


class ExtractorContextConfig(BaseModel):
    turn_texts: dict[str, str] = {}
    recent_graph: list[dict[str, Any]] = []
    recent_edges: list[dict[str, Any]] = []
    next_event_id: int = 1
    new_turns: list[dict[str, Any]] = []
    tool_call_budget: int | None = None
    window_hi: int = 0
    ops_file: str | None = None
    prompt_name: str = "default"
    prompt_text: str | None = None

STATE_SERVICE_KEY: Final = "llmharness.extractor_state"

MANIFEST = ExtensionManifest(
    name="extractor_context",
    description="Create extraction state from config data and inject system prompt.",
    registers=("event:before_agent_start",),
    config_schema=ExtractorContextConfig,
)

def _build_directive(
    next_event_id: int,
    recent_graph_count: int,
    tool_call_budget: int | None,
) -> str:
    steps: list[str] = [
        "Build the graph incrementally with upsert_node / "
        "upsert_edge (and delete_node / delete_edge as needed). Every "
        "edit is validated immediately for witness + id rules. The "
        "validator may flag chain-link events (in=1, out=1) as a "
        "SOFT warning attached to a successful finalize — aim for "
        "compact graphs but do NOT fabricate refs just to satisfy "
        "the heuristic.\n",
        "Call finalize_extraction (no payload) when you are done. "
        "Finalize commits the witness-valid graph and ends the "
        "firing; any chain-link advisory comes back as part of the "
        "success result so the next firing can apply the hint.\n",
    ]
    if tool_call_budget is not None:
        edit_budget = max(tool_call_budget - 1, 0)
        steps.append(
            f"Tool-call budget: this extractor firing has at most "
            f"{tool_call_budget} total tool calls, including "
            "finalize_extraction. You MUST reserve the final tool call "
            "for finalize_extraction; do not spend the last call on a "
            "graph edit. Spend at most "
            f"{edit_budget} calls on graph edits, then call "
            "finalize_extraction immediately. If the graph is already "
            "coherent enough, prefer a smaller truthful graph and call "
            "finalize_extraction early instead of adding or revising more "
            "edges.\n"
        )
    steps.append(
        f"Start event ids at {next_event_id} and increment strictly — "
        "do NOT restart at 1 and do NOT reuse any id from recent_graph.\n"
    )
    steps.append(
        f"Cross-firing references: recent_graph has {recent_graph_count} "
        "entries. To link this firing's events to prior firings, emit "
        "upsert_edge with src/dst spanning the boundary — the folded "
        "view already contains prior-firing nodes by id. Most act "
        "events in this firing answer a hyp/act from earlier firings; "
        "linking them is what turns a single firing into a connected "
        "investigation.\n\n"
    )
    numbered = "".join(f"({i}) {step}" for i, step in enumerate(steps, start=1))
    return "Below is the firing input. Workflow:\n" + numbered

def install(api: ExtensionAPI, config: ExtractorContextConfig) -> None:
    turn_texts: dict[int, str] = {int(k): str(v) for k, v in config.turn_texts.items()}

    recent_events = tuple(Event.from_dict(e) for e in config.recent_graph)
    recent_edges = tuple(Edge.from_dict(e) for e in config.recent_edges)

    state = ExtractionState(
        turn_texts=turn_texts,
        recent_graph=recent_events,
        recent_graph_dict={e.id: e for e in recent_events},
        recent_edges_dict={(ed.src, ed.dst, ed.kind.value): ed for ed in recent_edges},
        next_event_id=config.next_event_id,
        ops_file=config.ops_file,
    )
    api.set_service(STATE_SERVICE_KEY, state)

    prompt_text = config.prompt_text or load_extractor_prompt(config.prompt_name)
    directive = _build_directive(config.next_event_id, len(recent_events), config.tool_call_budget)
    system = f"{prompt_text}\n\n{directive}" if prompt_text else directive

    def _before_start(event: BeforeAgentStartEvent) -> None:
        event.system = system

    api.on(BeforeAgentStartEvent.CHANNEL, _before_start)
