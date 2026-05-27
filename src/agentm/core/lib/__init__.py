"""Pure helper utilities available to atoms and core code."""

# NB: ``child_collect`` is intentionally NOT re-exported here. Its module
# imports ``agentm.core.abi.messages``, which eagerly pulls
# ``agentm.core.runtime`` → ``otel_export`` → ``from agentm.core.lib import
# to_jsonable``. Re-exporting it at package import time runs that chain
# before ``to_jsonable`` is bound below, so a bare ``import agentm.core.lib``
# (or any submodule imported first) hits a partially-initialised-module
# circular import. Consumers import the submodule directly
# (``from agentm.core.lib.child_collect import ...``); by then this package
# has finished initialising and the cycle does not arise.
from agentm.core.lib.redact import redact_headers, redact_messages
from agentm.core.lib.serialization import to_jsonable
from agentm.core.lib.tool_schema import pydantic_to_openai_tool_schema

__all__ = [
    "pydantic_to_openai_tool_schema",
    "redact_headers",
    "redact_messages",
    "to_jsonable",
]
