"""Configuration schema definitions (Pydantic models).

Fully implemented — these are value objects validated at startup.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for a model."""

    requests_per_second: float
    max_bucket_size: int


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    api_key: str
    base_url: Optional[str] = None
    rate_limit: Optional[RateLimitConfig] = None


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


class ExecutionConfig(BaseModel):
    """Execution configuration for an agent."""

    max_steps: int = 20
    timeout: int = 120
    interrupt_before: Optional[list[str]] = None
    retry: RetryConfig = RetryConfig()
    tool_call_budget: Optional[int] = None


class CompressionConfig(BaseModel):
    """Compression configuration for an agent or orchestrator."""

    enabled: bool = True
    compression_threshold: float = 0.8
    compression_model: str = "gpt-4o-mini"
    preserve_latest_n: int = 2
    prompt: Optional[str] = None
    notebook: Optional[dict[str, Any]] = None
    recall: Optional[dict[str, Any]] = None


class AgentConfig(BaseModel):
    """Configuration for a single Sub-Agent."""

    model: str
    temperature: float
    prompt: Optional[str] = None
    tools: list[str]
    tool_settings: dict[str, dict[str, Any]] = {}
    task_type_prompts: Optional[dict[str, str]] = None
    execution: ExecutionConfig = ExecutionConfig()
    compression: Optional[CompressionConfig] = None


class FeatureGatesConfig(BaseModel):
    """Feature gates for the Orchestrator."""

    adversarial_review: bool = False
    parallel_verification: bool = False
    auto_refine_partial: bool = False
    min_verifications_before_confirm: int = 1
    deep_exploration: bool = False
    dedup_against_existing: bool = False
    auto_merge_similar: bool = False
    min_evidence_for_pattern: int = 2


class OutputConfig(BaseModel):
    """Structured output configuration for the Orchestrator.

    When set, create_react_agent receives response_format=(prompt, schema)
    so the framework appends a generate_structured_response node after the
    ReAct loop finishes.
    """

    prompt: str
    schema_name: str = "CausalGraph"


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration within a scenario."""

    model: str
    temperature: float = 0.7
    orchestrator_mode: str = "react"
    prompts: dict[str, str] = {}
    tools: list[str] = []
    feature_gates: FeatureGatesConfig = FeatureGatesConfig()
    compression: Optional[CompressionConfig] = None
    monitoring: Optional[dict[str, Any]] = None
    output: Optional[OutputConfig] = None


class SystemTypeConfig(BaseModel):
    """System type declaration within a scenario."""

    type: str


class ScenarioConfig(BaseModel):
    """Complete scenario configuration (scenario.yaml)."""

    system: SystemTypeConfig
    orchestrator: OrchestratorConfig
    agents: dict[str, AgentConfig]
    phases: Optional[dict[str, dict[str, Any]]] = None
