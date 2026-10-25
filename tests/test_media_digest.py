# SPDX-License-Identifier: Apache-2.0
"""Stable media-content digests for image-aware KV cache keying (issue #56).

Pins the serialization contract the disk KV checkpoint keying relies on:

* the SAME image content always produces the SAME digest (repeated
  multimodal turns keep their prefix hits);
* CHANGED image content produces a DIFFERENT digest (a new image never
  reuses the previous image's KV);
* text-only requests yield the empty media key so pre-#56 checkpoint
  hash bytes are unchanged.
"""

from __future__ import annotations

import base64

from vllm_mlx import media_digest as md

# ---------------------------------------------------------------------------
# data: URIs
# ---------------------------------------------------------------------------


def test_data_uri_digest_is_deterministic():
    img = b"fake-png-bytes-v1"
    ref = "data:image/png;base64," + base64.b64encode(img).decode()
    d1 = md.media_digest_for_ref(ref)
    d2 = md.media_digest_for_ref(ref)
    assert d1 == d2
    assert d1.startswith("data:")
    assert len(d1.split(":", 1)[1]) == 40


def test_data_uri_digest_changes_with_content():
    a = "data:image/png;base64," + base64.b64encode(b"image-A").decode()
    b = "data:image/png;base64," + base64.b64encode(b"image-B").decode()
    assert md.media_digest_for_ref(a) != md.media_digest_for_ref(b)


def test_data_uri_digest_ignores_mime_label_but_not_bytes():
    # The same payload under a different mime label is the same content.
    a = "data:image/png;base64," + base64.b64encode(b"xyz").decode()
    b = "data:image/jpeg;base64," + base64.b64encode(b"xyz").decode()
    assert md.media_digest_for_ref(a) == md.media_digest_for_ref(b)
    # Different payload, same label, is not.
    c = "data:image/png;base64," + base64.b64encode(b"xyz2").decode()
    assert md.media_digest_for_ref(a) != md.media_digest_for_ref(c)


def test_non_base64_data_uri_falls_back_to_payload_bytes():
    ref = "data:text/plain,hello%20world"
    d = md.media_digest_for_ref(ref)
    assert d is not None and d.startswith("data:")


def test_media_key_empty_for_text_only():
    assert md.media_key_for_media(None, None) == ""
    assert md.media_key_for_media([], []) == ""
    assert md.media_key_for_media((), ()) == ""


def test_media_key_order_is_document_order():
    a = "data:image/png;base64," + base64.b64encode(b"a").decode()
    b = "data:image/png;base64," + base64.b64encode(b"b").decode()
    key_ab = md.media_key_for_media([a, b])
    key_ba = md.media_key_for_media([b, a])
    assert key_ab != key_ba
    # Stable across calls.
    assert key_ab == md.media_key_for_media([a, b])


def test_media_key_includes_videos():
    v = "data:video/mp4;base64," + base64.b64encode(b"vid").decode()
    assert md.media_key_for_media(None, [v]) != ""


# ---------------------------------------------------------------------------
# local files
# ---------------------------------------------------------------------------


def test_file_digest_deterministic_and_content_sensitive(tmp_path):
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    p1.write_bytes(b"content-one")
    p2.write_bytes(b"content-two")
    d1 = md.media_digest_for_ref(str(p1))
    assert d1 == md.media_digest_for_ref(str(p1))
    assert d1.startswith("file:")
    assert d1 != md.media_digest_for_ref(str(p2))

    # Same bytes under a different path name are the same content.
    p3 = tmp_path / "renamed.png"
    p3.write_bytes(b"content-one")
    assert md.media_digest_for_ref(str(p3)) == d1


def test_missing_file_digests_the_reference():
    d = md.media_digest_for_ref("/nonexistent/definitely/missing.png")
    assert d is not None and d.startswith("ref:")
    assert d == md.media_digest_for_ref("/nonexistent/definitely/missing.png")


# ---------------------------------------------------------------------------
# http(s) refs
# ---------------------------------------------------------------------------


def test_http_fetch_failure_falls_back_to_url_digest(monkeypatch):
    import urllib.request

    def _raise(*a, **kw):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    d = md.media_digest_for_ref("https://cdn.example.com/pic/1.png")
    assert d.startswith("url:")
    assert d == md.media_digest_for_ref("https://cdn.example.com/pic/1.png")
    # A different URL never collides.
    assert d != md.media_digest_for_ref("https://cdn.example.com/pic/2.png")


def test_http_fetch_success_digests_bytes(monkeypatch):
    import urllib.request

    class _Resp:
        def read(self, n=-1):
            return b"http-image-bytes"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _Resp())
    d = md.media_digest_for_ref("https://cdn.example.com/pic/1.png")
    assert d.startswith("http:")
    assert d == md.media_digest_for_ref("https://cdn.example.com/pic/1.png")


# ---------------------------------------------------------------------------
# mixed lists
# ---------------------------------------------------------------------------


def test_digests_for_media_drops_non_strings_and_keeps_order():
    a = "data:image/png;base64," + base64.b64encode(b"a").decode()
    ref_b = "data:image/png;base64," + base64.b64encode(b"b").decode()
    digs = md.media_digests_for_media([None, a, "", 42, ref_b])  # type: ignore[list-item]
    assert digs == (md.media_digest_for_ref(a), md.media_digest_for_ref(ref_b))


def test_non_string_ref_is_none():
    assert md.media_digest_for_ref(12) is None  # type: ignore[arg-type]
    assert md.media_digest_for_ref(None) is None  # type: ignore[arg-type]
    assert md.media_digest_for_ref("") is None
