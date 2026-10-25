# SPDX-License-Identifier: Apache-2.0
"""Stable media-content digests for image-aware KV cache keying (issue #56).

Disk KV checkpoints are keyed on the round-trip-stable prompt token prefix
(#45). For a multimodal prompt the image *content* does not survive in the
token stream — the tokenizer expands every image into a run of the same
``image_token_id`` placeholder(s), and two different images that produce the
same token count are token-identical. Keying on tokens alone would let a
changed image reuse the stale image KV of the previous one.

This module closes the gap with a *stable serialization* of each media item:

- ``data:`` URIs digest the decoded payload bytes — the same image bytes
  always produce the same digest, changed bytes a different one;
- local paths digest the file bytes;
- ``http(s)`` URIs digest the fetched bytes, falling back to a digest of the
  URL string when the fetch fails (conservative: same URL still matches,
  a different URL never does);
- anything else digests the reference string itself.

The per-request key is the join of the ordered digests
(:func:`media_key_for_media`). An empty media list yields ``""`` so
text-only checkpoints keep their pre-#56 hash bytes exactly (backward
compatible with checkpoints already on disk).
"""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from typing import Any

# Cap on the bytes fetched/digested for a single remote media item. Large
# enough for any realistic photo/video frame reference, small enough that a
# pathological URL can't hold the keying path hostage.
_REMOTE_FETCH_LIMIT = 64 * 1024 * 1024  # 64 MiB
_REMOTE_TIMEOUT = 10.0  # seconds


def _digest_bytes(tag: str, data: bytes) -> str:
    """Digest ``data`` under source tag ``tag`` (``data``/``file``/``http``/
    ``url``/``ref``) — the tag is part of the key so a byte-identical file
    and a byte-identical URL payload can never collide."""
    h = hashlib.sha256(data)
    return f"{tag}:{h.hexdigest()[:40]}"


def _digest_data_uri(ref: str) -> str:
    """Digest the decoded payload of a ``data:[<mediatype>][;base64],<payload>``
    URI. Non-base64 data URIs digest the percent-decoded payload directly."""
    head, _, payload = ref.partition(",")
    if not payload:
        return _digest_bytes("ref", ref.encode("utf-8", "replace"))
    if ";base64" in head:
        import base64

        try:
            data = base64.b64decode(payload, validate=True)
        except Exception:
            # Not actually base64 — fall back to digesting the raw payload
            # so the digest is still stable for identical refs.
            data = payload.encode("utf-8", "replace")
    else:
        data = urllib.parse.unquote_to_bytes(payload)
    return _digest_bytes("data", data)


def media_digest_for_ref(ref: Any) -> str | None:
    """Stable content digest for one media reference.

    Returns ``None`` only for non-string refs (callers drop those from the
    key — a non-string ref is not addressable by a client round-trip anyway).
    """
    if not isinstance(ref, str) or not ref:
        return None
    low = ref.lower()
    if low.startswith("data:"):
        return _digest_data_uri(ref)

    url = urllib.parse.urlparse(ref)
    if url.scheme in ("http", "https"):
        try:
            req = urllib.request.Request(
                ref, headers={"User-Agent": "qmlx-media-digest/1"}
            )
            with urllib.request.urlopen(req, timeout=_REMOTE_TIMEOUT) as resp:
                data = resp.read(_REMOTE_FETCH_LIMIT + 1)
            if len(data) > _REMOTE_FETCH_LIMIT:
                # Truncated read — still digest what we got, but tag it so a
                # later full fetch of the same content can't silently collide.
                data = data[:_REMOTE_FETCH_LIMIT]
            return _digest_bytes("http", data)
        except Exception:
            # Fetch failed (offline, slow network, 404): fall back to a
            # digest of the URL string itself. Same URL still matches across
            # turns; a changed URL (and almost always changed bytes) never
            # reuses the old image's KV.
            normalized = ref.split("#", 1)[0].rstrip("/")
            return _digest_bytes("url", normalized.encode("utf-8", "replace"))

    # Local path (absolute, relative, or file://). file:// URLs parse to a
    # path under scheme "file"; try the raw ref as a path first so both forms
    # digest the same file.
    path = urllib.parse.unquote(url.path) if url.scheme == "file" else ref
    try:
        with open(path, "rb") as fh:
            data = fh.read(_REMOTE_FETCH_LIMIT + 1)
        if len(data) > _REMOTE_FETCH_LIMIT:
            data = data[:_REMOTE_FETCH_LIMIT]
        return _digest_bytes("file", data)
    except OSError:
        # Unreadable / missing — digest the reference string so the key is
        # stable for the same ref across turns.
        return _digest_bytes("ref", ref.encode("utf-8", "replace"))


def media_digests_for_media(
    images: list[Any] | tuple[Any, ...] | None,
    videos: list[Any] | tuple[Any, ...] | None = None,
) -> tuple[str, ...]:
    """Ordered content digests for a request's media list (images then
    videos, document order). Non-string entries are dropped; an all-empty
    result is ``()``."""
    out: list[str] = []
    for ref in list(images or ()) + list(videos or ()):
        d = media_digest_for_ref(ref)
        if d is not None:
            out.append(d)
    return tuple(out)


def media_key_for_media(
    images: list[Any] | tuple[Any, ...] | None,
    videos: list[Any] | tuple[Any, ...] | None = None,
) -> str:
    """Canonical per-request media key for KV cache keying.

    ``""`` for text-only requests (the pre-#56 behaviour: checkpoint hash
    bytes are unchanged and text-only prompts never match image checkpoints
    and vice versa). For media requests, a stable joined string that changes
    when any image/video changes and is identical for identical content.
    """
    digests = media_digests_for_media(images, videos)
    if not digests:
        return ""
    return "\x1f".join(digests)
