"""iLink Bot API HTTP client.

All communication with ``ilinkai.weixin.qq.com`` goes through this module.
Uses ``aiohttp`` for async HTTP; each API call is a standalone coroutine.
"""

from __future__ import annotations

import base64
import json
import os
import struct
from typing import Any

import aiohttp
from loguru import logger

from . import __version__
from .types import (
    GetUpdatesResp,
    MessageItemType,
    MessageState,
    MessageType,
)

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"

ILINK_APP_ID = "bot"
CHANNEL_VERSION = __version__

DEFAULT_LONG_POLL_TIMEOUT = 35.0
DEFAULT_API_TIMEOUT = 15.0
DEFAULT_CONFIG_TIMEOUT = 10.0


def _client_version_int(version: str) -> int:
    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return ((major & 0xFF) << 16) | ((minor & 0xFF) << 8) | (patch & 0xFF)


def _random_wechat_uin() -> str:
    raw = struct.pack(">I", int.from_bytes(os.urandom(4), "big"))
    return base64.b64encode(str(int.from_bytes(raw, "big")).encode()).decode()


def _build_base_info() -> dict[str, Any]:
    return {
        "channel_version": CHANNEL_VERSION,
        "bot_agent": "AgentM",
    }


def _common_headers() -> dict[str, str]:
    return {
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(_client_version_int(__version__)),
    }


def _auth_headers(token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": _random_wechat_uin(),
        **_common_headers(),
    }
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else f"{url}/"


async def _post(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    body: dict[str, Any],
    token: str | None = None,
    timeout: float = DEFAULT_API_TIMEOUT,
) -> dict[str, Any]:
    url = _ensure_trailing_slash(base_url) + endpoint
    headers = _auth_headers(token)
    try:
        async with session.post(
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            text = await resp.text()
            if not resp.ok:
                raise RuntimeError(f"{endpoint} {resp.status}: {text[:200]}")
            return json.loads(text)  # type: ignore[no-any-return]
    except aiohttp.ClientError as exc:
        logger.error(f"{endpoint}: request failed: {exc}")
        raise


async def _get(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    timeout: float = DEFAULT_API_TIMEOUT,
) -> dict[str, Any]:
    url = _ensure_trailing_slash(base_url) + endpoint
    headers = _common_headers()
    try:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            text = await resp.text()
            if not resp.ok:
                raise RuntimeError(f"{endpoint} {resp.status}: {text[:200]}")
            return json.loads(text)  # type: ignore[no-any-return]
    except aiohttp.ClientError as exc:
        logger.error(f"{endpoint}: request failed: {exc}")
        raise


# -- Public API -----------------------------------------------------------


async def get_updates(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str | None,
    get_updates_buf: str = "",
    timeout: float = DEFAULT_LONG_POLL_TIMEOUT,
) -> GetUpdatesResp:
    body = {
        "get_updates_buf": get_updates_buf,
        "base_info": _build_base_info(),
    }
    try:
        data = await _post(session, base_url, "ilink/bot/getupdates", body, token, timeout)
    except (aiohttp.ClientError, TimeoutError, RuntimeError) as exc:
        if "timeout" in str(exc).lower() or isinstance(exc, TimeoutError):
            logger.debug(f"getupdates: timeout after {timeout}s, returning empty")
            return GetUpdatesResp(ret=0, get_updates_buf=get_updates_buf)
        raise
    return GetUpdatesResp.from_dict(data)


async def send_message(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str | None,
    to: str,
    text: str,
    context_token: str | None = None,
    run_id: str | None = None,
    client_id: str = "",
) -> dict[str, Any]:
    item_list = []
    if text:
        item_list.append({
            "type": MessageItemType.TEXT,
            "text_item": {"text": text},
        })
    body = {
        "msg": {
            "from_user_id": "",
            "to_user_id": to,
            "client_id": client_id,
            "message_type": MessageType.BOT,
            "message_state": MessageState.FINISH,
            "item_list": item_list or None,
            "context_token": context_token or None,
            "run_id": run_id or None,
        },
        "base_info": _build_base_info(),
    }
    return await _post(session, base_url, "ilink/bot/sendmessage", body, token)


async def send_message_item(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str | None,
    to: str,
    item: dict[str, Any],
    context_token: str | None = None,
    run_id: str | None = None,
    client_id: str = "",
) -> dict[str, Any]:
    body = {
        "msg": {
            "from_user_id": "",
            "to_user_id": to,
            "client_id": client_id,
            "message_type": MessageType.BOT,
            "message_state": MessageState.FINISH,
            "item_list": [item],
            "context_token": context_token or None,
            "run_id": run_id or None,
        },
        "base_info": _build_base_info(),
    }
    return await _post(session, base_url, "ilink/bot/sendmessage", body, token)


async def send_typing(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str | None,
    user_id: str,
    typing_ticket: str,
    status: int = 1,
) -> None:
    body = {
        "ilink_user_id": user_id,
        "typing_ticket": typing_ticket,
        "status": status,
        "base_info": _build_base_info(),
    }
    try:
        await _post(session, base_url, "ilink/bot/sendtyping", body, token, DEFAULT_CONFIG_TIMEOUT)
    except Exception:
        logger.debug(f"sendtyping failed for {user_id}")


async def get_config(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str | None,
    user_id: str,
    context_token: str | None = None,
) -> dict[str, Any]:
    body = {
        "ilink_user_id": user_id,
        "context_token": context_token,
        "base_info": _build_base_info(),
    }
    return await _post(session, base_url, "ilink/bot/getconfig", body, token, DEFAULT_CONFIG_TIMEOUT)


async def notify_start(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str | None,
) -> dict[str, Any]:
    body = {"base_info": _build_base_info()}
    return await _post(session, base_url, "ilink/bot/msg/notifystart", body, token, DEFAULT_CONFIG_TIMEOUT)


async def notify_stop(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str | None,
) -> dict[str, Any]:
    body = {"base_info": _build_base_info()}
    return await _post(session, base_url, "ilink/bot/msg/notifystop", body, token, DEFAULT_CONFIG_TIMEOUT)


async def get_upload_url(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str | None,
    filekey: str,
    media_type: int,
    to_user_id: str,
    rawsize: int,
    rawfilemd5: str,
    filesize: int,
    aeskey: str,
    thumb_rawsize: int = 0,
    thumb_rawfilemd5: str = "",
    thumb_filesize: int = 0,
    no_need_thumb: bool = True,
) -> dict[str, Any]:
    body = {
        "filekey": filekey,
        "media_type": media_type,
        "to_user_id": to_user_id,
        "rawsize": rawsize,
        "rawfilemd5": rawfilemd5,
        "filesize": filesize,
        "aeskey": aeskey,
        "no_need_thumb": no_need_thumb,
        "base_info": _build_base_info(),
    }
    if thumb_rawsize:
        body["thumb_rawsize"] = thumb_rawsize
        body["thumb_rawfilemd5"] = thumb_rawfilemd5
        body["thumb_filesize"] = thumb_filesize
    return await _post(session, base_url, "ilink/bot/getuploadurl", body, token)


# -- QR Login API ---------------------------------------------------------

QR_BASE_URL = "https://ilinkai.weixin.qq.com"
DEFAULT_BOT_TYPE = "3"
QR_POLL_TIMEOUT = 35.0


async def fetch_qr_code(
    session: aiohttp.ClientSession,
    *,
    bot_type: str = DEFAULT_BOT_TYPE,
    local_token_list: list[str] | None = None,
) -> dict[str, Any]:
    endpoint = f"ilink/bot/get_bot_qrcode?bot_type={bot_type}"
    body = {"local_token_list": local_token_list or []}
    return await _post(session, QR_BASE_URL, endpoint, body)


async def poll_qr_status(
    session: aiohttp.ClientSession,
    *,
    qrcode: str,
    verify_code: str | None = None,
) -> dict[str, Any]:
    endpoint = f"ilink/bot/get_qrcode_status?qrcode={qrcode}"
    if verify_code:
        endpoint += f"&verify_code={verify_code}"
    try:
        return await _get(session, QR_BASE_URL, endpoint, QR_POLL_TIMEOUT)
    except Exception:
        logger.debug("qr poll timeout or error, returning wait")
        return {"status": "wait"}
