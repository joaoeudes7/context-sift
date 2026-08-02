"""Compact opaque payloads before prompt scoring."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re


_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/]{128,}={0,2}(?![A-Za-z0-9+/=_-])")


def compact_base64(text: str) -> str:
    """Replace valid long base64 with size and stable identity metadata."""
    def replacement(match: re.Match[str]) -> str:
        encoded = match.group()
        if len(encoded) % 4:
            return encoded
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return encoded
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return f"[base64 bytes={len(payload)} sha256={digest}]"

    return _BASE64_RE.sub(replacement, text)
