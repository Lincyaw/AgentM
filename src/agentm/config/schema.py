"""Configuration schema definitions (Pydantic models).

Fully implemented — these are value objects validated at startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, cast

from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel

if TYPE_CHECKING:
    from agentm.harness.types import ModelProtocol


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for a model."""

    requests_per_second: float
    max_bucket_size: int

    def create_rate_limiter(self) -> InMemoryRateLimiter:
        """Create a LangChain InMemoryRateLimiter from this config."""

        return InMemoryRateLimiter(
            requests_per_second=self.requests_per_second,
            max_bucket_size=self.max_bucket_size,
        )


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    api_key: str
    base_url: Optional[str] = None
    rate_limit: Optional[RateLimitConfig] = None
    max_retries: int = 20
    provider: Literal["openai", "anthropic"] = "openai"


def create_chat_model(
    model: str,
    temperature: float = 0,
    model_config: ModelConfig | None = None,
    **kwargs: Any,
) -> ModelProtocol:
    """Create a ChatOpenAI or ChatAnthropic instance based on model_config.provider.

    Centralizes all LLM client instantiation so that switching between
    OpenAI-compatible and Anthropic endpoints requires only a config change.

    Returns:
        BaseChatModel instance (ChatOpenAI or ChatAnthropic).
    """
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        **kwargs,
    }

    provider = "openai"
    if model_config is not None:
        provider = model_config.provider
        if model_config.api_key:
            llm_kwargs["api_key"] = model_config.api_key
        if model_config.base_url:
            llm_kwargs["base_url"] = model_config.base_url
        if model_config.rate_limit is not None:
            llm_kwargs["rate_limiter"] = model_config.rate_limit.create_rate_limiter()
        llm_kwargs["max_retries"] = model_config.max_retries

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # ChatAnthropic uses 'anthropic_api_key', not 'api_key'
        if "api_key" in llm_kwargs:
            llm_kwargs["anthropic_api_key"] = llm_kwargs.pop("api_key")
        # ChatAnthropic uses 'anthropic_api_url', not 'base_url'
        if "base_url" in llm_kwargs:
            llm_kwargs["anthropic_api_url"] = llm_kwargs.pop("base_url")
        return cast("ModelProtocol", ChatAnthropic(**llm_kwargs))

    from langchain_openai import ChatOpenAI

    return cast("ModelProtocol", ChatOpenAI(**llm_kwargs))


class StorageBackendConfig(BaseModel):
    """Configuration for a storage backend (checkpointer or store)."""

    backend: str
    url: str
    index: Optional[dict[str, Any]] = None


class StorageConfig(BaseModel):
    """Storage configuration for checkpointer and store."""

    checkpointer: StorageBackendConfig
    store: StorageBackendConfig


class RecoveryConfig(BaseModel):
    """System recovery configuration."""

    mode: str = "manual"
    expose_api: bool = True


class TrajectoryConfig(BaseModel):
    """Trajectory export configuration."""

    enabled: bool = True
    output_dir: str = "./trajectories"


class DebugConfig(BaseModel):
    """Debug infrastructure configuration."""

    trajectory: TrajectoryConfig = TrajectoryConfig()
    console_live: bool = False
    verbose: bool = False


class SystemConfig(BaseModel):
    """Global system configuration (system.yaml)."""

    models: dict[str, ModelConfig]
    storage: StorageConfig
    recovery: RecoveryConfig
    debug: DebugConfig = DebugConfig()


class RetryConfig(BaseModel):
    """Retry configuration for agent execution."""

    max_attempts: int = 3
    initial_interval: float = 1.0
    backoff_factor: float = 2.0


class DedupConfig(BaseModel):
    """Tool call deduplication configuration.

    Ref: designs/tool-dedup.md
    """

    enabled: bool = True
    max_cache_size: int = 50


class LoopDetectionConfig(BaseModel):
    """Loop detection configuration for an agent or orchestrator."""

    threshold: int = 5
    window_size: int = 15
    think_stall_limit: int = 3


class ExecutionConfig(BaseModel):
    """Execution configuration for an agent."""

    max_steps: int = 20
    timeout: int = 120
    tool_call_budget: Optional[int] = None
    dedup: Optional[DedupConfig] = None
    retry: RetryConfig = RetryConfig()
    loop_detection: LoopDetectionConfig = LoopDetectionConfig()
    max_concurrent_workers: Optional[int] = None


class CompressionConfig(BaseModel):
    """Compression configuration for an agent or orchestrator."""

    enabled: bool = True
    compression_threshold: float = 0.8
    compression_model: str = "gpt-4o-mini"
    context_window: int = 128_000
    preserve_latest_n: int = 2
    prompt: Optional[str] = None
    recall: Optional[dict[str, Any]] = None


class LLMConfig(BaseModel):
    """Shared configuration for LLM-backed components (agents and orchestrator).

    Fields common to both AgentConfig and OrchestratorConfig are defined here
    to avoid duplication.
    """

    model: str
    temperature: float
    compression: Optional[CompressionConfig] = None
    skills: list[str] = []
    include_think_tool: bool = True


class AgentConfig(LLMConfig):
    """Configuration for a single Sub-Agent."""

    prompt: Optional[str] = None
    tools: list[str]
    tool_settings: dict[str, dict[str, Any]] = {}
    task_type_prompts: Optional[dict[str, str]] = None
    execution: ExecutionConfig = ExecutionConfig()


class FeatureGatesConfig(BaseModel):
    """Feature gates for the Orchestrator.

    This base class is intentionally empty — domain-specific gates belong in
    scenario subclasses (e.g. ``RCAFeatureGates``, ``MemoryFeatureGates``).
    ``extra="allow"`` lets scenario configs pass through without schema errors.
    """

    model_config = {"extra": "allow"}


class OutputConfig(BaseModel):
    """Structured output configuration for the Orchestrator.

    When set, create_react_agent receives response_format=(prompt, schema)
    so the framework appends a generate_structured_response node after the
    ReAct loop finishes.
    """

    prompt: str
    schema_name: str = "CausalGraph"


class SkeletonConfig(BaseModel):
    """Configuration for trajectory skeleton injection into judge prompts.

    The skeleton gives the LLM an at-a-glance view of all tool calls
    (with args but without full responses), so it can reason about the
    agent's behaviour and selectively query responses via jq_query.
    """

    exclude_tools: list[str] = []
    include_tools: list[str] = []
    max_args_length: int = 500
    response_preview_length: int = 100


class OrchestratorConfig(LLMConfig):
    """Orchestrator configuration within a scenario."""

    temperature: float = 0.7
    orchestrator_mode: str = "node"  # "react" | "node"
    prompts: dict[str, str] = {}
    tools: list[str] = []
    feature_gates: FeatureGatesConfig = FeatureGatesConfig()
    monitoring: Optional[dict[str, Any]] = None
    output: Optional[OutputConfig] = None
    max_rounds: int = 20  # node mode only: max LLM rounds before forced synthesize
    disable_tool_binding: bool = (
        False  # set True for models that don't support bind_tools (e.g. MiniMax)
    )
    retry: RetryConfig = RetryConfig()
    loop_detection: LoopDetectionConfig = LoopDetectionConfig()
    skeleton: Optional[SkeletonConfig] = None


class SystemTypeConfig(BaseModel):
    """System type declaration within a scenario."""

    type: str


class ScenarioConfig(BaseModel):
    """Complete scenario configuration (scenario.yaml)."""

    system: SystemTypeConfig
    orchestrator: OrchestratorConfig
    agents: dict[str, AgentConfig]
