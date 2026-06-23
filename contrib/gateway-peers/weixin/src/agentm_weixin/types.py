"""iLink Bot API protocol types.

Mirrors the Weixin proto: GetUpdatesReq/Resp, WeixinMessage, SendMessageReq.
API uses JSON over HTTP; bytes fields are base64 strings in JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MessageType(IntEnum):
    NONE = 0
    USER = 1
    BOT = 2


class MessageItemType(IntEnum):
    NONE = 0
    TEXT = 1
    IMAGE = 2
    VOICE = 3
    FILE = 4
    VIDEO = 5
    TOOL_CALL_START = 11
    TOOL_CALL_RESULT = 12


class MessageState(IntEnum):
    NEW = 0
    GENERATING = 1
    FINISH = 2


class UploadMediaType(IntEnum):
    IMAGE = 1
    VIDEO = 2
    FILE = 3
    VOICE = 4


class TypingStatus(IntEnum):
    TYPING = 1
    CANCEL = 2


@dataclass(slots=True)
class CDNMedia:
    encrypt_query_param: str = ""
    aes_key: str = ""
    encrypt_type: int = 0
    full_url: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> CDNMedia | None:
        if not d:
            return None
        return cls(
            encrypt_query_param=d.get("encrypt_query_param", ""),
            aes_key=d.get("aes_key", ""),
            encrypt_type=d.get("encrypt_type", 0),
            full_url=d.get("full_url", ""),
        )


@dataclass(slots=True)
class TextItem:
    text: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> TextItem | None:
        if not d:
            return None
        return cls(text=d.get("text", ""))


@dataclass(slots=True)
class ImageItem:
    media: CDNMedia | None = None
    thumb_media: CDNMedia | None = None
    aeskey: str = ""
    url: str = ""
    mid_size: int = 0
    thumb_size: int = 0
    hd_size: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ImageItem | None:
        if not d:
            return None
        return cls(
            media=CDNMedia.from_dict(d.get("media")),
            thumb_media=CDNMedia.from_dict(d.get("thumb_media")),
            aeskey=d.get("aeskey", ""),
            url=d.get("url", ""),
            mid_size=d.get("mid_size", 0),
            thumb_size=d.get("thumb_size", 0),
            hd_size=d.get("hd_size", 0),
        )


@dataclass(slots=True)
class VoiceItem:
    media: CDNMedia | None = None
    encode_type: int = 0
    sample_rate: int = 0
    playtime: int = 0
    text: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> VoiceItem | None:
        if not d:
            return None
        return cls(
            media=CDNMedia.from_dict(d.get("media")),
            encode_type=d.get("encode_type", 0),
            sample_rate=d.get("sample_rate", 0),
            playtime=d.get("playtime", 0),
            text=d.get("text", ""),
        )


@dataclass(slots=True)
class FileItem:
    media: CDNMedia | None = None
    file_name: str = ""
    md5: str = ""
    len: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FileItem | None:
        if not d:
            return None
        return cls(
            media=CDNMedia.from_dict(d.get("media")),
            file_name=d.get("file_name", ""),
            md5=d.get("md5", ""),
            len=d.get("len", ""),
        )


@dataclass(slots=True)
class VideoItem:
    media: CDNMedia | None = None
    video_size: int = 0
    play_length: int = 0
    video_md5: str = ""
    thumb_media: CDNMedia | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> VideoItem | None:
        if not d:
            return None
        return cls(
            media=CDNMedia.from_dict(d.get("media")),
            video_size=d.get("video_size", 0),
            play_length=d.get("play_length", 0),
            video_md5=d.get("video_md5", ""),
            thumb_media=CDNMedia.from_dict(d.get("thumb_media")),
        )


@dataclass(slots=True)
class MessageItem:
    type: int = 0
    create_time_ms: int = 0
    update_time_ms: int = 0
    is_completed: bool = False
    msg_id: str = ""
    text_item: TextItem | None = None
    image_item: ImageItem | None = None
    voice_item: VoiceItem | None = None
    file_item: FileItem | None = None
    video_item: VideoItem | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MessageItem:
        return cls(
            type=d.get("type", 0),
            create_time_ms=d.get("create_time_ms", 0),
            update_time_ms=d.get("update_time_ms", 0),
            is_completed=d.get("is_completed", False),
            msg_id=d.get("msg_id", ""),
            text_item=TextItem.from_dict(d.get("text_item")),
            image_item=ImageItem.from_dict(d.get("image_item")),
            voice_item=VoiceItem.from_dict(d.get("voice_item")),
            file_item=FileItem.from_dict(d.get("file_item")),
            video_item=VideoItem.from_dict(d.get("video_item")),
        )


@dataclass(slots=True)
class WeixinMessage:
    seq: int = 0
    message_id: int = 0
    from_user_id: str = ""
    to_user_id: str = ""
    client_id: str = ""
    create_time_ms: int = 0
    update_time_ms: int = 0
    session_id: str = ""
    group_id: str = ""
    message_type: int = 0
    message_state: int = 0
    item_list: list[MessageItem] = field(default_factory=list)
    context_token: str = ""
    run_id: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WeixinMessage:
        items = [MessageItem.from_dict(i) for i in d.get("item_list", [])]
        return cls(
            seq=d.get("seq", 0),
            message_id=d.get("message_id", 0),
            from_user_id=d.get("from_user_id", ""),
            to_user_id=d.get("to_user_id", ""),
            client_id=d.get("client_id", ""),
            create_time_ms=d.get("create_time_ms", 0),
            update_time_ms=d.get("update_time_ms", 0),
            session_id=d.get("session_id", ""),
            group_id=d.get("group_id", ""),
            message_type=d.get("message_type", 0),
            message_state=d.get("message_state", 0),
            item_list=items,
            context_token=d.get("context_token", ""),
            run_id=d.get("run_id", ""),
        )


@dataclass(slots=True)
class GetUpdatesResp:
    ret: int = 0
    errcode: int = 0
    errmsg: str = ""
    msgs: list[WeixinMessage] = field(default_factory=list)
    get_updates_buf: str = ""
    longpolling_timeout_ms: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GetUpdatesResp:
        msgs = [WeixinMessage.from_dict(m) for m in d.get("msgs", [])]
        return cls(
            ret=d.get("ret", 0),
            errcode=d.get("errcode", 0),
            errmsg=d.get("errmsg", ""),
            msgs=msgs,
            get_updates_buf=d.get("get_updates_buf", ""),
            longpolling_timeout_ms=d.get("longpolling_timeout_ms", 0),
        )
