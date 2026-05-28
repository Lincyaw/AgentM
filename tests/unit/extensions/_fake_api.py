"""Shared fake ``ExtensionAPI`` for producer-atom unit tests.

Extracted from the duplicated ``_FakeApi`` stubs in ``test_background_exec.py``
and ``test_monitor.py`` (B7 boundary-review fix). Covers the minimal
``ExtensionAPI`` surface the producer atoms touch:

* :meth:`post_inbox` — delegates to a real :class:`SessionInbox` so the
  dedup-replace contract is the genuine one (the runtime impl does exactly
  this); tests can drain the inbox to assert pushed items.
* :meth:`register_tool` — collects registered tools on ``self.tools``.
* :meth:`on` — records handlers per channel; returns a real
  ``Unsubscribe``-shaped callable so subscribe/unsubscribe semantics survive.
* :meth:`fire` — test helper to invoke every handler on a channel (mirrors
  what the kernel bus does for the bits these atoms exercise).

NOT included: any provider/extension-management surface (``provider``,
``register_provider``, ``list_atoms``, ``install_atom``, ...). Producer
atoms don't touch those; the integration-test layer covers the full API.
"""

from __future__ import annotations

from typing import Any

from agentm.core.runtime.session_inbox import InboxItem, SessionInbox


class FakeExtensionAPI:
    """Minimal pub-sub + inbox + tools list ExtensionAPI shim for unit tests.

    Cast to :class:`agentm.core.abi.extension.ExtensionAPI` at the test
    boundary — none of the producer-atom code paths these tests exercise
    actually require the full Protocol surface, so ``cast(ExtensionAPI, ...)``
    is the cheapest way to keep mypy happy.
    """

    def __init__(self) -> None:
        self.tools: list[Any] = []
        self.inbox = SessionInbox()
        self._handlers: dict[str, list[Any]] = {}

    def post_inbox(
        self, *, source: str, payload: Any, dedup_key: str | None = None
    ) -> None:
        self.inbox.push(
            InboxItem(source=source, payload=payload, dedup_key=dedup_key)
        )

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def on(self, channel: str, handler: Any, *, priority: int = 500) -> Any:
        bucket = self._handlers.setdefault(channel, [])
        bucket.append(handler)

        def _unsub() -> None:
            if handler in bucket:
                bucket.remove(handler)

        return _unsub

    def fire(self, channel: str, event: Any) -> None:
        """Invoke every subscribed handler for ``channel`` (test helper).

        Mirrors the kernel-bus dispatch for the sync-handler subset these
        producer-atom tests use. Async handlers (none in this fake's
        callers) would be ignored — by design; tests that need async
        dispatch should use the real ``EventBus``.
        """

        for h in list(self._handlers.get(channel, [])):
            h(event)


__all__ = ["FakeExtensionAPI"]
