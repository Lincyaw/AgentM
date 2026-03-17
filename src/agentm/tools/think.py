"""Think tool — a scratchpad for structured reasoning.

This tool performs NO I/O. It simply returns the agent's thought back,
giving the LLM a designated space to reason without polluting the final
structured output (findings/leads/verdict).

Inspired by Anthropic's "think" tool pattern for extended thinking in
tool-use contexts.
"""

from langchain_core.tools import tool


@tool
def think(thought: str) -> str:
    """A scratchpad for recording your reasoning process. This tool has NO
    side effects — it simply stores your thought and returns it.

    Use this to record analysis, compare data, or track hypothesis status.

    IMPORTANT: Think does NOT advance the investigation. After thinking,
    you MUST call an action tool (dispatch_agent, update_hypothesis, etc.)
    in the SAME response or the very next response. Never call think
    multiple times in a row without an action in between.

    Your thought is private — it will NOT appear in your final output.
    """
    return thought
