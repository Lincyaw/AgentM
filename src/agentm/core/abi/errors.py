"""Public SDK error types."""

from __future__ import annotations


class ExtensionLoadError(Exception):
    """Raised when an extension module fails to load or install."""

    def __init__(self, module_path: str, cause: BaseException | None = None) -> None:
        self.module_path = module_path
        self.cause = cause
        super().__init__(f"failed to load extension {module_path!r}: {cause}")


__all__ = ["ExtensionLoadError"]
