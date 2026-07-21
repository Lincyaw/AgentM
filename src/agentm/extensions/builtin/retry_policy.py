"""Provider retry-policy service."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    RETRY_POLICY_SERVICE,
    AtomAPI,
    AtomInstallPriority,
)
from agentm.extensions import ExtensionManifest

T = TypeVar("T")


class RetryPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=7, ge=0)
    base_delay: float = Field(default=5.0, ge=0)
    factor: float = Field(default=2.0, ge=1)
    jitter: float = Field(default=0.0, ge=0)


MANIFEST = ExtensionManifest(
    name="retry_policy",
    description="Register the retry policy consumed during provider construction.",
    registers=("service:retry_policy",),
    config_schema=RetryPolicyConfig,
    requires=(),
    priority=AtomInstallPriority.POLICY,
)


@dataclass(slots=True)
class ExponentialBackoffRetry:
    max_retries: int
    base_delay: float
    factor: float
    jitter: float

    async def run(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        is_retryable: Callable[[BaseException], bool],
    ) -> T:
        attempt = 0
        delay = self.base_delay
        while True:
            try:
                return await fn()
            except Exception as exc:
                if not is_retryable(exc) or attempt >= self.max_retries:
                    raise
                attempt += 1
                sleep_for = delay + (
                    random.uniform(0.0, self.jitter) if self.jitter else 0.0
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                delay *= self.factor


def install(api: AtomAPI, config: RetryPolicyConfig) -> None:
    api.services.register(
        RETRY_POLICY_SERVICE,
        ExponentialBackoffRetry(
            max_retries=config.max_retries,
            base_delay=config.base_delay,
            factor=config.factor,
            jitter=config.jitter,
        ),
        scope="session",
    )


__all__ = ("ExponentialBackoffRetry", "MANIFEST", "RetryPolicyConfig", "install")
