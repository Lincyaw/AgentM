"""Think tool — a scratchpad for structured reasoning between tool calls.

This tool performs NO I/O. It simply returns the agent's thought back,
giving the LLM a designated space to reason without polluting the final
structured output (findings/leads/verdict).

Inspired by Anthropic's "think" tool pattern for extended thinking in
tool-use contexts.
"""

from langchain_core.tools import tool


@tool
def think(thought: str) -> str:
    """Use this tool to think step-by-step between data queries.

    Call this BEFORE making tool calls to plan your next move, and AFTER
    receiving results to analyze what you learned. This helps you:
    - Decide which tools to call next and with what parameters
    - Compare abnormal vs normal data and assess significance
    - Track which hypotheses are supported or contradicted
    - Avoid wasting tool budget on redundant queries

    Your thought is private — it will NOT appear in your final output.
    """
    return thought
