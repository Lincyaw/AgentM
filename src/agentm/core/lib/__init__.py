"""Pure helper utilities — single import surface for atoms.

Atoms import from ``agentm.core.lib`` only; direct sub-module imports
(``from agentm.core.lib.stream import ...``) are forbidden by the §11
validator.
"""

from agentm.core.lib.artifact_files import (
    ArtifactCreator,
    ArtifactMetadata,
    artifacts_dir_for,
    find_metadata_files,
    list_artifacts_for_task,
    scan_artifact_metadata,
)
from agentm.core.lib.background_tasks import (
    BackgroundTask,
    BackgroundTaskRegistry,
    SlotLimitReached,
)
from agentm.core.lib.frontmatter import parse_frontmatter
from agentm.core.lib.observability_dir import file_export_requested, resolve_observability_dir
from agentm.core.lib.paths import (
    expand_optional_path_text,
    expand_path,
    expand_path_text,
    parsed_unix_socket_path,
)
from agentm.core.lib.read_state import (
    FileReadState,
    bind_session as bind_read_state_session,
    clear as clear_read_state,
    content_hash_for,
    file_modified_since_read,
    get_read_state,
    record_read,
)
from agentm.core.observability.redact import redact_headers, redact_messages
from agentm.core.lib.ref import Ref
from agentm.core.lib.render import assistant_text, final_summary
from agentm.core.lib.serialization import to_jsonable
from agentm.core.lib.shutdown import DEFAULT_SHUTDOWN_GRACE_SECONDS
from agentm.core.lib.stream import StreamAccumulator, ToolSpecAdapter, encode_tool_args
from agentm.core.lib.child_wire import forward_child_to_wire
from agentm.core.lib.tool_result import with_model_note
from agentm.core.lib.tool_schema import pydantic_to_openai_tool_schema, pydantic_to_tool_schema
from agentm.core.lib.tokens import (
    TokenTruncation,
    count_text_tokens,
    truncate_text_tokens,
    truncate_text_tokens_middle,
)
from agentm.core.lib.turns import Turn, enumerate_turns
from agentm.core.lib.user_config import agentm_home_dir, resolve_model_profile

__all__ = [
    "ArtifactCreator", "ArtifactMetadata",
    "BackgroundTask", "BackgroundTaskRegistry",
    "DEFAULT_SHUTDOWN_GRACE_SECONDS",
    "FileReadState",
    "Ref",
    "SlotLimitReached",
    "bind_read_state_session",
    "StreamAccumulator",
    "ToolSpecAdapter",
    "TokenTruncation",
    "Turn",
    "agentm_home_dir",
    "assistant_text",
    "artifacts_dir_for",
    "clear_read_state",
    "content_hash_for",
    "count_text_tokens",
    "encode_tool_args",
    "enumerate_turns",
    "expand_optional_path_text",
    "expand_path",
    "expand_path_text",
    "file_export_requested",
    "file_modified_since_read",
    "final_summary",
    "find_metadata_files",
    "get_read_state",
    "list_artifacts_for_task",
    "parse_frontmatter",
    "parsed_unix_socket_path",
    "pydantic_to_openai_tool_schema",
    "pydantic_to_tool_schema",
    "record_read",
    "redact_headers",
    "redact_messages",
    "resolve_model_profile",
    "resolve_observability_dir",
    "scan_artifact_metadata",
    "forward_child_to_wire",
    "to_jsonable",
    "truncate_text_tokens",
    "truncate_text_tokens_middle",
    "with_model_note",
]
