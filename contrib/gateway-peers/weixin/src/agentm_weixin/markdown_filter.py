"""Markdown-to-WeChat text filter.

WeChat personal chat has limited rich-text support (no markdown rendering).
This filter converts common markdown patterns into plain-text equivalents
that read well in the WeChat message bubble.
"""

from __future__ import annotations

import re


# Bold: **text** or __text__ → text
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
# Italic: *text* or _text_ → text
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
# Strikethrough: ~~text~~ → text
_STRIKE_RE = re.compile(r"~~(.+?)~~")
# Inline code: `code` → code
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
# Links: [text](url) → text (url)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Images: ![alt](url) → [图片: alt]
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
# Headers: # Header → 【Header】
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Horizontal rule: --- or *** or ___ → ————
_HR_RE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
# Unordered list: - item or * item → • item
_UL_RE = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)
# Ordered list: 1. item → 1. item (keep as-is)
# Code blocks: ```...``` → preserve content, strip fences
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
# Blockquote: > text → ｜text
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$", re.MULTILINE)


def filter_markdown(text: str) -> str:
    """Convert markdown to WeChat-friendly plain text."""
    if not text:
        return text

    # Code blocks first (preserve content, strip fences)
    text = _CODE_BLOCK_RE.sub(r"\1", text)

    # Images before links (images are a superset pattern)
    text = _IMAGE_RE.sub(lambda m: f"[图片: {m.group(1)}]" if m.group(1) else "[图片]", text)

    # Links
    text = _LINK_RE.sub(r"\1 (\2)", text)

    # Headers
    text = _HEADER_RE.sub(lambda m: f"【{m.group(2)}】", text)

    # HR
    text = _HR_RE.sub("————————", text)

    # Blockquotes
    text = _BLOCKQUOTE_RE.sub(r"｜\1", text)

    # Unordered lists
    text = _UL_RE.sub(r"\1• ", text)

    # Bold (before italic, since bold uses double markers)
    text = _BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)

    # Strikethrough
    text = _STRIKE_RE.sub(r"\1", text)

    # Inline code
    text = _INLINE_CODE_RE.sub(r"\1", text)

    # Clean up excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
