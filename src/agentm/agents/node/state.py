"""State schemas for the node-based worker subgraph.

These TypedDicts define the input/output contract between the worker
subgraph and the TaskManager.  They are intentionally minimal — the
worker owns its own message history and returns only the fields that
the Orchestrator needs to consume.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agentm.models.types import TaskType


class WorkerResult(TypedDict):
    """Structured result returned by collect_and_compress.

    Mirrors the fields in react/sub_agent.py ANSWER_SCHEMA so that
    TaskManager._extract_structured_response can handle both implementations
    with a single lookup.
    """

    findings: str

    # scout / deep_analyze: list of investigation leads
    # verify: omitted (use verdict instead)
    leads: list[str]

    # verify only
    verdict: str


class WorkerState(TypedDict):
    """Full mutable state inside the worker subgraph.

    ``messages`` uses the ``add_messages`` reducer so that each node
    appends to the history without overwriting it.  All other fields
    are set once by ``dispatch`` and read by subsequent nodes.

    ``structured_response`` is written by ``collect_and_compress`` and
    read by TaskManager._extract_structured_response via the events buffer.
    It must be declared here so LangGraph does not discard the update.
    """

    messages: Annotated[list[BaseMessage], add_messages]

    # Set by dispatch, read-only for llm_call / tool_node / collect_and_compress
    task_id: str
    task_type: TaskType
    instruction: str
    hypothesis_id: str | None

    # Injected by TaskManager before dispatch (optional cross-worker tips)
    tool_tips: list[dict[str, Any]]

    # Written by collect_and_compress — must be a declared field
    structured_response: WorkerResult
