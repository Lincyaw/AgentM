"""WeChat QR-code login flow.

Generates a QR code, displays it in the terminal, polls for scan
confirmation, and persists the resulting bot_token + account metadata.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

import aiohttp
from loguru import logger

from .api import fetch_qr_code, poll_qr_status
from .state import AccountData, register_account_id, save_account


@dataclass(slots=True)
class LoginResult:
    connected: bool = False
    already_connected: bool = False
    account_id: str = ""
    bot_token: str = ""
    base_url: str = ""
    user_id: str = ""
    message: str = ""


def _normalize_account_id(raw: str) -> str:
    return raw.replace("@", "-").replace(".", "-")


def _display_qr_terminal(url: str) -> None:
    """Render a QR code in the terminal using the qrcode library."""
    try:
        import qrcode  # noqa: PLC0415

        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(out=sys.stderr, invert=True)
    except ImportError:
        pass
    sys.stderr.write(f"\n若二维码未能显示，请访问以下链接:\n{url}\n\n")
    sys.stderr.flush()


async def _read_verify_code() -> str:
    """Read a verification code from stdin (blocking, wrapped in executor)."""
    sys.stderr.write("输入手机微信显示的数字，以继续连接: ")
    sys.stderr.flush()
    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    return line.strip()


async def login_qr(
    session: aiohttp.ClientSession,
    *,
    account_id: str | None = None,
    timeout_s: float = 480.0,
) -> LoginResult:
    """Run the full QR login flow. Returns a LoginResult."""

    # Get QR code
    logger.info("requesting QR code from iLink...")
    try:
        qr_resp = await fetch_qr_code(session)
    except Exception as exc:
        logger.debug("weixin login: failed to fetch QR code: {}", exc)
        return LoginResult(message=f"获取二维码失败: {exc}")

    qrcode_key = qr_resp.get("qrcode", "")
    qrcode_url = qr_resp.get("qrcode_img_content", "")
    if not qrcode_key or not qrcode_url:
        return LoginResult(message="获取二维码失败: 服务器返回为空")

    sys.stderr.write("\n用手机微信扫描以下二维码，以继续连接:\n\n")
    _display_qr_terminal(qrcode_url)

    # Poll for scan
    deadline = asyncio.get_event_loop().time() + timeout_s
    scanned_printed = False
    pending_verify_code: str | None = None
    current_base_url = "https://ilinkai.weixin.qq.com"

    while asyncio.get_event_loop().time() < deadline:
        status_resp = await poll_qr_status(
            session,
            qrcode=qrcode_key,
            verify_code=pending_verify_code,
        )
        status = status_resp.get("status", "wait")

        if status == "wait":
            await asyncio.sleep(1.0)
            continue

        if status == "scaned":
            if pending_verify_code:
                pending_verify_code = None
            if not scanned_printed:
                sys.stderr.write("\n正在验证...\n")
                sys.stderr.flush()
                scanned_printed = True
            await asyncio.sleep(1.0)
            continue

        if status == "need_verifycode":
            prompt = (
                "❌ 数字不匹配，请重新输入: "
                if pending_verify_code
                else "输入手机微信显示的数字: "
            )
            sys.stderr.write(prompt)
            sys.stderr.flush()
            pending_verify_code = await _read_verify_code()
            continue

        if status == "verify_code_blocked":
            sys.stderr.write("\n⛔ 多次输入错误，请稍后再试。\n")
            sys.stderr.flush()
            return LoginResult(message="多次输入错误，连接流程已停止。")

        if status == "expired":
            return LoginResult(message="二维码已过期，请重试。")

        if status == "binded_redirect":
            sys.stderr.write("\n✅ 已连接过，无需重复连接。\n")
            sys.stderr.flush()
            return LoginResult(already_connected=True, message="已连接过，无需重复连接。")

        if status == "scaned_but_redirect":
            redirect_host = status_resp.get("redirect_host", "")
            if redirect_host:
                current_base_url = f"https://{redirect_host}"
                logger.info(f"IDC redirect to {current_base_url}")
            await asyncio.sleep(1.0)
            continue

        if status == "confirmed":
            bot_token = status_resp.get("bot_token", "")
            raw_account_id = status_resp.get("ilink_bot_id", "")
            base_url = status_resp.get("baseurl", "") or current_base_url
            user_id = status_resp.get("ilink_user_id", "")

            if not raw_account_id:
                return LoginResult(message="登录失败: 服务器未返回 bot ID")

            normalized_id = _normalize_account_id(raw_account_id)

            save_account(normalized_id, AccountData(
                token=bot_token,
                base_url=base_url,
                user_id=user_id,
            ))
            register_account_id(normalized_id)

            sys.stderr.write(f"\n✅ 已连接到微信 (account={normalized_id})\n")
            sys.stderr.flush()

            return LoginResult(
                connected=True,
                account_id=normalized_id,
                bot_token=bot_token,
                base_url=base_url,
                user_id=user_id,
                message="已将此 AgentM 连接到微信。",
            )

        await asyncio.sleep(1.0)

    return LoginResult(message="登录超时，请重试。")
