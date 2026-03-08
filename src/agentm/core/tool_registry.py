"""Tool registry for dynamic tool binding."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any, Callable

import yaml
from langchain_core.tools import StructuredTool


class ToolDefinition:
    """A registered tool definition with config-based instantiation."""

    def __init__(self, name: str, func: Callable[..., Any], config_schema: dict[str, Any]) -> None:
        self.name = name
        self.func = func
        self.config_schema = config_schema
        self.parameters: dict[str, Any] = config_schema

    def create_with_config(self, **config: Any) -> StructuredTool:
        """Create a LangChain StructuredTool instance with bound config parameters."""
        description = config.pop("description", self.func.__doc__ or self.name)

        if asyncio.iscoroutinefunction(self.func):
            return StructuredTool.from_function(
                coroutine=self.func,
                name=self.name,
                description=description,
            )
        return StructuredTool.from_function(
            func=self.func,
            name=self.name,
            description=description,
        )


class ToolRegistry:
    """Central registry for tool definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, func: Callable[..., Any], config_schema: dict[str, Any]) -> None:
        """Register a tool definition."""
        self._tools[name] = ToolDefinition(name, func, config_schema)

    def get(self, name: str) -> ToolDefinition:
        """Get a tool definition by name."""
        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def load_from_yaml(self, path: Path | str) -> None:
        """Load tool definitions from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        for tool in data.get("tools", []):
            module = importlib.import_module(tool["module"])
            func = getattr(module, tool["function"])
            self.register(tool["name"], func, tool.get("params", {}))
