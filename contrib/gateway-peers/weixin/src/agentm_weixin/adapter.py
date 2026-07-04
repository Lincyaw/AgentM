"""WeChat adapter for the ``agentm-weixin`` chat-client peer (v2).

Owns a :class:`WireClient`, runs a long-poll loop against the iLink Bot API
(``getupdates``), and renders outbound envelopes as WeChat messages.

WeChat personal chat is text-only (no cards / interactive elements), so the
outbound rendering is simpler than Feishu: durable kinds (assistant_text,
approval_request, diagnostic) become plain text messages; ephemeral kinds
(tool_call, stream_text, etc.) are silently consumed. A typing indicator
is sent while the agent is working.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from agentm.gateway.client import WireClient
from agentm.gateway.wire import KIND_INBOUND, WIRE_VERSION, Envelope

from . import api
from .markdown_filter import filter_markdown
from .media import (
    build_file_send_item,
    build_image_send_item,
    build_video_send_item,
    download_media,
    upload_file,
)
from .state import ContextTokenStore, load_sync_buf, save_sync_buf
from .types import (
    CDNMedia,
    MessageItemType,
    MessageType,
    TypingStatus,
    WeixinMessage,
)

STALE_TOKEN_ERRCODE = -14
LONG_POLL_TIMEOUT = 35.0
MAX_CONSECUTIVE_FAILURES = 3
BACKOFF_DELAY = 30.0
RETRY_DELAY = 2.0
TYPING_KEEPALIVE = 5.0

# Workspace subdirectory for inbound media files the agent can access.
MEDIA_INBOX_DIR = ".agentm/weixin/media/inbox"

# Regex to detect MEDIA: directives in outbound text.
# Must be on its own line: ``MEDIA:/abs/path/to/file.png``
_MEDIA_LINE_RE = re.compile(r"^MEDIA:(.+)$", re.MULTILINE)

# Injected on the first inbound per session so the agent knows how to
# send media back to WeChat.
_CHANNEL_HINT = (
    "\n\n<system-reminder>"
    "You are chatting via WeChat. To send a file, image, or video to the "
    "user, put MEDIA:/absolute/path on its own line in your reply. "
    "Example:\nHere is the chart you asked for.\nMEDIA:/tmp/chart.png\n"
    "The MEDIA: tag MUST be on its own line with an absolute path. "
    "When the user sends you media, it is saved to disk and the path is "
    "shown in the message — use file tools to read it."
    "</system-reminder>"
)


@dataclass(slots=True)
class WeixinConfig:
    account_id: str
    token: str
    base_url: str = api.DEFAULT_BASE_URL
    cdn_base_url: str = api.CDN_BASE_URL
    channel_name: str = "weixin"
    scenario: str | None = "chatbot"
    session_scope: str = "user"


class WeixinAdapter:
    """Bridges the iLink Bot API to a :class:`WireClient`."""

    def __init__(
        self,
        client: WireClient,
        config: WeixinConfig,
        http_session: aiohttp.ClientSession,
    ) -> None:
        self._client = client
        self._config = config
        self._http = http_session
        self._ctx_tokens = ContextTokenStore()
        self._scenario_sent: set[str] = set()
        self._typing_tickets: dict[str, str] = {}
        self._typing_tasks: dict[str, asyncio.Task[Any]] = {}
        self._active_turns: set[str] = set()
        self._stop_event = asyncio.Event()
        self._running = False

    # -- session_key (§3.4) -------------------------------------------

    def _session_key(self, user_id: str) -> str:
        base = f"{self._config.channel_name}:{self._config.account_id}"
        if self._config.session_scope == "user":
            return f"{base}:{user_id}"
        return base

    # -- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        cfg = self._config

        # Restore context tokens from disk
        self._ctx_tokens.restore(cfg.account_id)

        # Notify server we're starting
        try:
            await api.notify_start(self._http, base_url=cfg.base_url, token=cfg.token)
        except Exception:
            logger.warning("notify_start failed (ignored)")

        logger.info(
            f"weixin monitor started (base_url={cfg.base_url}, "
            f"account={cfg.account_id})"
        )

        # Run the long-poll loop
        try:
            await self._poll_loop()
        finally:
            self._running = False

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()

        # Cancel typing tasks
        for task in self._typing_tasks.values():
            if not task.done():
                task.cancel()
        self._typing_tasks.clear()

        # Notify server we're stopping
        try:
            await api.notify_stop(
                self._http,
                base_url=self._config.base_url,
                token=self._config.token,
            )
        except Exception:
            logger.warning("notify_stop failed (ignored)")

    # -- long-poll loop -----------------------------------------------

    async def _poll_loop(self) -> None:
        cfg = self._config
        sync_buf = load_sync_buf(cfg.account_id)
        next_timeout = LONG_POLL_TIMEOUT
        consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                resp = await api.get_updates(
                    self._http,
                    base_url=cfg.base_url,
                    token=cfg.token,
                    get_updates_buf=sync_buf,
                    timeout=next_timeout,
                )

                if resp.longpolling_timeout_ms > 0:
                    next_timeout = resp.longpolling_timeout_ms / 1000.0

                is_error = (resp.ret != 0) or (resp.errcode != 0)
                if is_error:
                    if resp.errcode == STALE_TOKEN_ERRCODE or resp.ret == STALE_TOKEN_ERRCODE:
                        logger.error(
                            f"token stale for {cfg.account_id}, pausing 5 min"
                        )
                        consecutive_failures = 0
                        await self._sleep(300)
                        continue

                    consecutive_failures += 1
                    logger.error(
                        f"getupdates failed: ret={resp.ret} errcode={resp.errcode} "
                        f"errmsg={resp.errmsg} ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0
                        await self._sleep(BACKOFF_DELAY)
                    else:
                        await self._sleep(RETRY_DELAY)
                    continue

                consecutive_failures = 0

                # Persist sync buffer
                if resp.get_updates_buf:
                    save_sync_buf(cfg.account_id, resp.get_updates_buf)
                    sync_buf = resp.get_updates_buf

                # Process messages
                for msg in resp.msgs:
                    if msg.message_type != MessageType.USER:
                        continue
                    try:
                        await self._process_inbound(msg)
                    except Exception:
                        logger.exception(f"process_inbound failed for msg seq={msg.seq}")

            except asyncio.CancelledError:
                return
            except Exception:
                if self._stop_event.is_set():
                    return
                consecutive_failures += 1
                logger.exception(
                    f"getupdates error ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0
                    await self._sleep(BACKOFF_DELAY)
                else:
                    await self._sleep(RETRY_DELAY)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except (TimeoutError, asyncio.CancelledError):
            pass

    # -- inbound (WeChat -> gateway) ----------------------------------

    def _media_inbox(self) -> Path:
        """Return the workspace-relative media inbox directory, creating it."""
        d = Path(MEDIA_INBOX_DIR)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_to_inbox(self, tmp_path: str) -> str:
        """Move a temp media file into the workspace inbox. Returns new path."""
        inbox = self._media_inbox()
        src = Path(tmp_path)
        dst = inbox / src.name
        # Avoid name collisions
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            dst = inbox / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
        src.rename(dst)
        return str(dst.resolve())

    async def _process_inbound(self, msg: WeixinMessage) -> None:
        user_id = msg.from_user_id
        if not user_id:
            return

        # Cache context token
        if msg.context_token:
            self._ctx_tokens.set(self._config.account_id, user_id, msg.context_token)

        # Fetch typing ticket if not cached
        if user_id not in self._typing_tickets:
            try:
                cfg_resp = await api.get_config(
                    self._http,
                    base_url=self._config.base_url,
                    token=self._config.token,
                    user_id=user_id,
                    context_token=msg.context_token,
                )
                ticket = cfg_resp.get("typing_ticket", "")
                if ticket:
                    self._typing_tickets[user_id] = ticket
            except Exception:
                logger.debug(f"get_config failed for {user_id}")

        # Extract text content
        text = self._extract_text(msg)

        # Extract and save media to workspace
        media_path = await self._download_inbound_media(msg)
        saved_path: str | None = None
        if media_path:
            try:
                saved_path = self._save_to_inbox(media_path)
            except OSError:
                logger.exception("failed to save media to inbox")
                saved_path = media_path

        logger.info(
            f"[weixin] rx from={user_id} text={text[:80]!r} "
            f"media={saved_path or 'no'}"
        )

        # Build content: combine text + media path so the agent can read
        content_parts: list[str] = []
        if text:
            content_parts.append(text)
        if saved_path:
            media_type = self._describe_media_type(saved_path)
            content_parts.append(
                f"[用户发送了{media_type}，已保存到 {saved_path}]"
            )
        content = "\n".join(content_parts) if content_parts else ""

        await self._forward_inbound(
            sender_id=user_id,
            content=content,
        )

    @staticmethod
    def _describe_media_type(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
            return "图片"
        if suffix in (".mp4", ".mov", ".avi", ".mkv"):
            return "视频"
        if suffix in (".silk", ".wav", ".mp3", ".ogg", ".amr"):
            return "语音"
        return "文件"

    def _extract_text(self, msg: WeixinMessage) -> str:
        for item in msg.item_list:
            if item.type == MessageItemType.TEXT and item.text_item:
                return item.text_item.text
            if item.type == MessageItemType.VOICE and item.voice_item and item.voice_item.text:
                return item.voice_item.text
        return ""

    async def _download_inbound_media(self, msg: WeixinMessage) -> str | None:
        for item in msg.item_list:
            if item.type == MessageItemType.IMAGE and item.image_item:
                media = item.image_item.media
                if not media:
                    continue
                cdn = CDNMedia(
                    encrypt_query_param=media.encrypt_query_param,
                    aes_key=media.aes_key,
                    full_url=media.full_url,
                )
                aes_key = item.image_item.aeskey
                return await download_media(
                    self._http,
                    cdn_media=cdn,
                    aes_key_hex=aes_key,
                    cdn_base_url=self._config.cdn_base_url,
                    suffix=".jpg",
                )
            if item.type == MessageItemType.VOICE and item.voice_item:
                media = item.voice_item.media
                if not media or item.voice_item.text:
                    continue
                cdn = CDNMedia(
                    encrypt_query_param=media.encrypt_query_param,
                    aes_key=media.aes_key,
                    full_url=media.full_url,
                )
                return await download_media(
                    self._http,
                    cdn_media=cdn,
                    cdn_base_url=self._config.cdn_base_url,
                    suffix=".silk",
                )
            if item.type == MessageItemType.FILE and item.file_item:
                media = item.file_item.media
                if not media:
                    continue
                cdn = CDNMedia(
                    encrypt_query_param=media.encrypt_query_param,
                    aes_key=media.aes_key,
                    full_url=media.full_url,
                )
                ext = Path(item.file_item.file_name).suffix or ".bin"
                return await download_media(
                    self._http,
                    cdn_media=cdn,
                    cdn_base_url=self._config.cdn_base_url,
                    suffix=ext,
                )
            if item.type == MessageItemType.VIDEO and item.video_item:
                media = item.video_item.media
                if not media:
                    continue
                cdn = CDNMedia(
                    encrypt_query_param=media.encrypt_query_param,
                    aes_key=media.aes_key,
                    full_url=media.full_url,
                )
                return await download_media(
                    self._http,
                    cdn_media=cdn,
                    cdn_base_url=self._config.cdn_base_url,
                    suffix=".mp4",
                )
        return None

    async def _forward_inbound(
        self,
        *,
        sender_id: str,
        content: str,
        button_value: str | None = None,
        control: str | None = None,
    ) -> None:
        session_key = self._session_key(sender_id)
        body: dict[str, Any] = {
            "channel": self._config.channel_name,
            "sender_id": sender_id,
            "chat_id": sender_id,
            "content": content,
        }
        if button_value is not None:
            body["button_value"] = button_value
        if control is not None:
            body["control"] = control

        scenario = None
        first_message = session_key not in self._scenario_sent
        if first_message:
            scenario = self._config.scenario
            self._scenario_sent.add(session_key)
            content += _CHANNEL_HINT

        env = Envelope(
            v=WIRE_VERSION,
            id=f"in-weixin-{int(time.time() * 1_000_000)}",
            kind=KIND_INBOUND,
            ts=time.time(),
            session_key=session_key,
            scenario=scenario,
            body=body,
        )
        try:
            await self._client.send(env)
        except Exception:
            logger.exception("forward_inbound failed; dropping envelope")

    # -- outbound (gateway -> WeChat) ---------------------------------

    async def handle_outbound(self, env: Envelope) -> None:
        """Dispatch one v2 outbound envelope to its render path."""
        body = env.body if isinstance(env.body, dict) else {}
        out_channel = str(body.get("channel") or "")
        if out_channel and out_channel != self._config.channel_name:
            return

        chat_id = str(body.get("chat_id") or "")
        if not chat_id:
            logger.warning(f"outbound dropped: empty chat_id (env id={env.id})")
            return

        raw_meta = body.get("metadata")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        meta_kind = str(meta.get("kind") or "assistant_text")

        if meta_kind == "turn_start":
            self._start_typing(chat_id)
            self._active_turns.add(chat_id)
            return

        if meta_kind == "assistant_text":
            self._stop_typing(chat_id)
            self._active_turns.discard(chat_id)
            text = str(body.get("content") or "")
            if text.strip():
                await self._send_text_with_media(chat_id, text)
            return

        if meta_kind == "agent_end":
            self._stop_typing(chat_id)
            self._active_turns.discard(chat_id)
            return

        if meta_kind == "approval_request":
            self._stop_typing(chat_id)
            await self._send_approval(chat_id, body)
            return

        if meta_kind in ("diagnostic_error", "diagnostic_warning"):
            self._stop_typing(chat_id)
            icon = "⚠️" if meta_kind == "diagnostic_warning" else "🛑"
            content = str(body.get("content") or "")
            await self._send_text(chat_id, f"{icon} {content}")
            return

        if meta_kind == "command_result":
            text = str(body.get("content") or "")
            if text.strip():
                await self._send_text(chat_id, text)
            return

        # Ephemeral kinds (tool_call, stream_text, etc.) → silent

    async def _send_text(self, user_id: str, text: str) -> None:
        filtered = filter_markdown(text)
        if not filtered:
            return
        ctx_token = self._ctx_tokens.get(self._config.account_id, user_id)
        client_id = f"agentm-{uuid.uuid4().hex[:12]}"
        try:
            await api.send_message(
                self._http,
                base_url=self._config.base_url,
                token=self._config.token,
                to=user_id,
                text=filtered,
                context_token=ctx_token,
                client_id=client_id,
            )
            logger.info(f"[weixin] sent text to={user_id} len={len(filtered)}")
        except Exception:
            logger.exception(f"[weixin] send_message failed to={user_id}")

    async def _send_text_with_media(self, user_id: str, text: str) -> None:
        """Send text, extracting and sending any MEDIA: lines as files.

        The agent signals a file send by putting ``MEDIA:/abs/path``
        on its own line. This method splits the text, sends text chunks
        as messages, and uploads+sends each media file in order.
        """
        media_matches = list(_MEDIA_LINE_RE.finditer(text))
        if not media_matches:
            await self._send_text(user_id, text)
            return

        # Split text around MEDIA: lines and interleave
        last_end = 0
        for match in media_matches:
            # Send text before this MEDIA: line
            preceding = text[last_end:match.start()].strip()
            if preceding:
                await self._send_text(user_id, preceding)

            # Send the media file
            file_path = match.group(1).strip()
            if Path(file_path).is_file():
                await self._send_media(user_id, file_path)
            else:
                logger.warning(f"[weixin] MEDIA file not found: {file_path}")
                await self._send_text(user_id, f"[文件未找到: {file_path}]")

            last_end = match.end()

        # Send any trailing text after the last MEDIA: line
        trailing = text[last_end:].strip()
        if trailing:
            await self._send_text(user_id, trailing)

    async def _send_media(self, user_id: str, file_path: str, caption: str = "") -> None:
        ctx_token = self._ctx_tokens.get(self._config.account_id, user_id)
        client_id = f"agentm-{uuid.uuid4().hex[:12]}"
        try:
            uploaded = await upload_file(
                self._http,
                file_path=file_path,
                to_user_id=user_id,
                base_url=self._config.base_url,
                token=self._config.token,
            )
            from .types import UploadMediaType
            mt = uploaded.get("media_type", UploadMediaType.FILE)
            if mt == UploadMediaType.IMAGE:
                item = build_image_send_item(uploaded)
            elif mt == UploadMediaType.VIDEO:
                item = build_video_send_item(uploaded)
            else:
                item = build_file_send_item(uploaded)

            # Send caption first if present
            if caption:
                await api.send_message(
                    self._http,
                    base_url=self._config.base_url,
                    token=self._config.token,
                    to=user_id,
                    text=filter_markdown(caption),
                    context_token=ctx_token,
                    client_id=f"agentm-{uuid.uuid4().hex[:12]}",
                )

            await api.send_message_item(
                self._http,
                base_url=self._config.base_url,
                token=self._config.token,
                to=user_id,
                item=item,
                context_token=ctx_token,
                client_id=client_id,
            )
            logger.info(f"[weixin] sent media to={user_id} type={mt}")
        except Exception:
            logger.exception(f"[weixin] send_media failed to={user_id}")

    async def _send_approval(self, user_id: str, body: dict[str, Any]) -> None:
        content = str(body.get("content") or "")
        buttons = body.get("buttons") or []
        lines = [f"⚡ {content}", ""]
        for btn in buttons:
            if not isinstance(btn, dict):
                continue
            label = btn.get("label", "")
            value = btn.get("value", "")
            lines.append(f"  回复 \"{value}\" → {label}")
        await self._send_text(user_id, "\n".join(lines))

    # -- typing indicator ---------------------------------------------

    def _start_typing(self, user_id: str) -> None:
        if user_id in self._typing_tasks:
            return
        ticket = self._typing_tickets.get(user_id)
        if not ticket:
            return
        task = asyncio.create_task(
            self._typing_loop(user_id, ticket), name=f"typing-{user_id}"
        )
        self._typing_tasks[user_id] = task

    def _stop_typing(self, user_id: str) -> None:
        task = self._typing_tasks.pop(user_id, None)
        if task and not task.done():
            task.cancel()
        ticket = self._typing_tickets.get(user_id)
        if ticket:
            asyncio.create_task(
                api.send_typing(
                    self._http,
                    base_url=self._config.base_url,
                    token=self._config.token,
                    user_id=user_id,
                    typing_ticket=ticket,
                    status=TypingStatus.CANCEL,
                ),
                name=f"typing-cancel-{user_id}",
            )

    async def _typing_loop(self, user_id: str, ticket: str) -> None:
        try:
            while True:
                await api.send_typing(
                    self._http,
                    base_url=self._config.base_url,
                    token=self._config.token,
                    user_id=user_id,
                    typing_ticket=ticket,
                    status=TypingStatus.TYPING,
                )
                await asyncio.sleep(TYPING_KEEPALIVE)
        except asyncio.CancelledError:
            pass


__all__ = ["WeixinAdapter", "WeixinConfig"]
