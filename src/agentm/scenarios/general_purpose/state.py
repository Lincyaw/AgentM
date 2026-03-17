"""General-purpose-specific state schema."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from agentm.models.state import BaseExecutorState


class GeneralPurposeState(BaseExecutorState):
    """General-purpose task execution state.

    Extends BaseExecutorState with skill management, accumulated
    conversation facts, and an optional structured response slot.
    Inherits messages, task_id, task_description, current_phase
    from BaseExecutorState.
    """

    active_skills: list[str]
    skill_cache: dict[str, str]
    conversation_facts: Annotated[list[str], operator.add]
    structured_response: Optional[Any]
