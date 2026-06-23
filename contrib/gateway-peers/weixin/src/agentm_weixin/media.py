"""CDN media upload/download with AES-128-ECB encryption.

The iLink Bot API requires:
  Upload:  generate AES key → encrypt file → getuploadurl → PUT ciphertext
  Download: get URL from CDNMedia → fetch ciphertext → AES decrypt
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

import aiohttp
from Crypto.Cipher import AES
from loguru import logger

from . import api
from .types import (
    CDNMedia,
    MessageItemType,
    UploadMediaType,
)


AES_BLOCK_SIZE = 16


def _pad_pkcs7(data: bytes) -> bytes:
    pad_len = AES_BLOCK_SIZE - (len(data) % AES_BLOCK_SIZE)
    return data + bytes([pad_len] * pad_len)


def _unpad_pkcs7(data: bytes) -> bytes:
    if not data:
        return data
    pad_len = data[-1]
    if pad_len < 1 or pad_len > AES_BLOCK_SIZE:
        return data
    if data[-pad_len:] != bytes([pad_len] * pad_len):
        return data
    return data[:-pad_len]


def aes_ecb_encrypt(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(_pad_pkcs7(data))


def aes_ecb_decrypt(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return _unpad_pkcs7(cipher.decrypt(data))


def _guess_media_type(file_path: str) -> int:
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        if mime.startswith("image/"):
            return UploadMediaType.IMAGE
        if mime.startswith("video/"):
            return UploadMediaType.VIDEO
        if mime.startswith("audio/"):
            return UploadMediaType.VOICE
    return UploadMediaType.FILE


def _file_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


async def upload_file(
    session: aiohttp.ClientSession,
    *,
    file_path: str,
    to_user_id: str,
    base_url: str,
    token: str | None,
) -> dict[str, Any]:
    """Upload a local file to the WeChat CDN.

    Returns a dict with keys needed to build the send message item:
    ``filekey``, ``download_encrypted_query_param``, ``aeskey``,
    ``file_size`` (plaintext), ``file_size_ciphertext``, ``media_type``.
    """
    raw_data = Path(file_path).read_bytes()
    raw_size = len(raw_data)
    raw_md5 = _file_md5(raw_data)
    media_type = _guess_media_type(file_path)

    # Generate random AES-128 key
    aes_key = os.urandom(16)
    aes_key_hex = aes_key.hex()
    ciphertext = aes_ecb_encrypt(raw_data, aes_key)
    cipher_size = len(ciphertext)

    filekey = f"agentm_{os.urandom(8).hex()}"

    # Get pre-signed upload URL
    resp = await api.get_upload_url(
        session,
        base_url=base_url,
        token=token,
        filekey=filekey,
        media_type=media_type,
        to_user_id=to_user_id,
        rawsize=raw_size,
        rawfilemd5=raw_md5,
        filesize=cipher_size,
        aeskey=aes_key_hex,
    )

    upload_url = resp.get("upload_full_url", "")
    if not upload_url:
        upload_param = resp.get("upload_param", "")
        if not upload_param:
            raise RuntimeError("getuploadurl returned no upload URL or param")
        upload_url = upload_param

    # PUT encrypted data to CDN
    async with session.put(
        upload_url,
        data=ciphertext,
        headers={"Content-Type": "application/octet-stream"},
        timeout=aiohttp.ClientTimeout(total=60),
    ) as put_resp:
        if not put_resp.ok:
            body = await put_resp.text()
            raise RuntimeError(f"CDN upload failed {put_resp.status}: {body[:200]}")

    return {
        "filekey": filekey,
        "download_encrypted_query_param": resp.get("upload_param", ""),
        "aeskey": aes_key_hex,
        "file_size": raw_size,
        "file_size_ciphertext": cipher_size,
        "media_type": media_type,
        "file_name": Path(file_path).name,
    }


async def download_media(
    session: aiohttp.ClientSession,
    *,
    cdn_media: CDNMedia,
    aes_key_hex: str = "",
    cdn_base_url: str = "",
    out_dir: str = "",
    suffix: str = "",
) -> str | None:
    """Download and decrypt a media file from the WeChat CDN.

    Returns the path to the decrypted file, or None on failure.
    """
    url = cdn_media.full_url
    if not url and cdn_media.encrypt_query_param:
        base = cdn_base_url or api.CDN_BASE_URL
        url = f"{base}?{cdn_media.encrypt_query_param}"
    if not url:
        logger.warning("download_media: no URL available")
        return None

    # Resolve AES key
    key_hex = aes_key_hex
    if not key_hex and cdn_media.aes_key:
        try:
            key_hex = base64.b64decode(cdn_media.aes_key).hex()
        except Exception:
            key_hex = cdn_media.aes_key
    if not key_hex:
        logger.warning("download_media: no AES key available")
        return None

    try:
        aes_key = bytes.fromhex(key_hex)
    except ValueError:
        logger.warning(f"download_media: invalid AES key hex: {key_hex[:8]}...")
        return None

    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if not resp.ok:
                logger.warning(f"download_media: CDN returned {resp.status}")
                return None
            encrypted = await resp.read()
    except Exception as exc:
        logger.warning(f"download_media: fetch failed: {exc}")
        return None

    decrypted = aes_ecb_decrypt(encrypted, aes_key)

    if not out_dir:
        out_dir = tempfile.mkdtemp(prefix="agentm-weixin-media-")
    os.makedirs(out_dir, exist_ok=True)

    if not suffix:
        suffix = ".bin"
    out_path = os.path.join(out_dir, f"media_{os.urandom(4).hex()}{suffix}")
    Path(out_path).write_bytes(decrypted)
    return out_path


async def download_remote_image(
    session: aiohttp.ClientSession,
    url: str,
    out_dir: str = "",
) -> str:
    """Download a remote HTTP(S) image to a local temp file."""
    if not out_dir:
        out_dir = tempfile.mkdtemp(prefix="agentm-weixin-remote-")
    os.makedirs(out_dir, exist_ok=True)

    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        if not resp.ok:
            raise RuntimeError(f"remote download failed {resp.status}: {url[:80]}")
        data = await resp.read()

    content_type = resp.headers.get("Content-Type", "")
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".bin"
    out_path = os.path.join(out_dir, f"remote_{os.urandom(4).hex()}{ext}")
    Path(out_path).write_bytes(data)
    return out_path


def build_image_send_item(uploaded: dict[str, Any]) -> dict[str, Any]:
    """Build a sendmessage item_list entry for an uploaded image."""
    return {
        "type": MessageItemType.IMAGE,
        "image_item": {
            "media": {
                "encrypt_query_param": uploaded.get("download_encrypted_query_param", ""),
                "aes_key": base64.b64encode(
                    bytes.fromhex(uploaded["aeskey"])
                ).decode(),
                "encrypt_type": 1,
            },
            "mid_size": uploaded.get("file_size_ciphertext", 0),
        },
    }


def build_video_send_item(uploaded: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": MessageItemType.VIDEO,
        "video_item": {
            "media": {
                "encrypt_query_param": uploaded.get("download_encrypted_query_param", ""),
                "aes_key": base64.b64encode(
                    bytes.fromhex(uploaded["aeskey"])
                ).decode(),
                "encrypt_type": 1,
            },
            "video_size": uploaded.get("file_size_ciphertext", 0),
        },
    }


def build_file_send_item(uploaded: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": MessageItemType.FILE,
        "file_item": {
            "media": {
                "encrypt_query_param": uploaded.get("download_encrypted_query_param", ""),
                "aes_key": base64.b64encode(
                    bytes.fromhex(uploaded["aeskey"])
                ).decode(),
                "encrypt_type": 1,
            },
            "file_name": uploaded.get("file_name", "file"),
            "len": str(uploaded.get("file_size", 0)),
        },
    }
