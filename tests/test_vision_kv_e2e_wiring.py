# SPDX-License-Identifier: Apache-2.0
"""E2E acceptance tests for vision x KV (issue #56, PR #60).

The PR #60 acceptance draft (``docs/E2E-VISION.md``) defines four
acceptance items. This file covers what CI (no Metal — hydrogen) can
verify; the Metal-only items (real VLM forward pass, real Metal KV
allocations) stay manual gates in ``docs/E2E-VISION.md``.

* (1) serve-path wiring — ``qmlx serve <vlm-mlx> --enable-prefix-cache``
  (without ``--text-only``) accepts and routes image+text chat
  requests: CLI flag parsing (``--text-only`` is opt-in, not default),
  ``BatchedEngine`` auto-detection of vision-capable builds,
  ``start()`` dispatch (the mlx_vlm MLLM scheduler was removed in the
  Qwen-only build #20 — issue #56 serves vision through the LLM
  engine path + image-aware KV wiring), the chat route's image gate
  (``allow_image=engine.is_mllm``), and ``media_key`` threading through
  ``generate()`` / ``stream_generate()`` → ``add_request``.
* (2) prefix-cache hits on repeated multimodal turns; no stale reuse on
  image change (content-index + scheduler persist level).
* (3) disk KV checkpoint round-trip with image-aware keys (mock frames —
  no real model weights; the ``_SeededMockModel``-style pattern of
  ``tests/test_mtp_spec_decode.py``).
* (4) MTP interaction: MTP-off with images (``media_key`` threads
  through with ``spec_decode="none"``; the MTP-on path keys the same
  media bucket).

Model loading is mocked throughout — the tests assert the WIRING
(flags → engine → request → scheduler → disk), not the numerics of a
real VLM forward (that is the Metal manual gate).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402

from vllm_mlx.api.utils import (  # noqa: E402
    is_mllm_model,
    validate_content_blocks_for_capabilities,
)
from vllm_mlx.engine.batched import BatchedEngine  # noqa: E402
from vllm_mlx.engine_core import AsyncEngineCore  # noqa: E402
from vllm_mlx.media_digest import media_key_for_media  # noqa: E402
from vllm_mlx.request import Request, RequestOutput, SamplingParams  # noqa: E402
from vllm_mlx.runtime import disk_kv_checkpoint as _dkc  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

# Two distinct image contents -> two distinct stable media keys.
IMG_A = "data:image/png;base64,aW1nLWEtYnl0ZXM="  # base64 of "img-a-bytes"
IMG_B = "data:image/png;base64,aW1nLWItdHJ1ZQ=="  # base64 of "img-b-true"


def _media_key(ref: str) -> str:
    return media_key_for_media([ref])


def _seed_kv_cache(num_tokens: int = 256) -> list[KVCache]:
    cache = KVCache()
    k = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(0))
    v = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(1))
    cache.update_and_fetch(k, v)
    return [cache]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    root = tmp_path / "kv-ckpt"
    root.mkdir()
    monkeypatch.setattr(_dkc, "get_default_root", lambda: str(root))
    _dkc.reset_stats_for_tests()
    _dkc.reset_content_index_for_tests()
    return str(root)


# ---------------------------------------------------------------------------
# (1) Serve-path wiring — CLI
# ---------------------------------------------------------------------------


def test_cli_serve_prefix_cache_without_text_only():
    """Acceptance (1): ``qmlx serve <vlm-mlx> --enable-prefix-cache`` —
    WITHOUT ``--text-only`` — parses and keeps vision routing on
    (``no_mllm=False``)."""
    # Patch serve_command so the parse test doesn't boot a server.
    snippet = (
        "import sys\n"
        "import vllm_mlx.cli as cli\n"
        "captured = {}\n"
        "def fake_serve(a): captured['a'] = a\n"
        "cli.serve_command = fake_serve\n"
        "sys.argv = ['qmlx', 'serve', 'qwen3-vl-2b-4bit', '--enable-prefix-cache']\n"
        "cli.main()\n"
        "a = captured['a']\n"
        "assert a.command == 'serve', a.command\n"
        "assert a.enable_prefix_cache is True\n"
        "assert a.no_mllm is False, 'vision routing must stay ON without --text-only'\n"
        "assert a.mllm is False\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_cli_text_only_is_opt_in_flag():
    """``--text-only`` is an OPT-IN escape hatch (#393): with it,
    ``no_mllm=True`` forces the text path even for vision builds."""
    snippet = (
        "import sys\n"
        "import vllm_mlx.cli as cli\n"
        "captured = {}\n"
        "def fake_serve(a): captured['a'] = a\n"
        "cli.serve_command = fake_serve\n"
        "sys.argv = ['qmlx', 'serve', 'qwen3-vl-2b-4bit', '--text-only']\n"
        "cli.main()\n"
        "a = captured['a']\n"
        "assert a.no_mllm is True\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_mllm_and_text_only_mutually_exclusive():
    """``--mllm`` + ``--text-only`` together is a user error: the loader
    rejects it (ValueError with an actionable message) instead of
    silently picking one."""
    from vllm_mlx.server import load_model

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_model("qwen3-vl-2b-4bit", force_mllm=True, force_text=True)


# ---------------------------------------------------------------------------
# (1) Serve-path wiring — engine auto-detection + start() dispatch
# ---------------------------------------------------------------------------


def _vision_build_dir(tmp_path: Path) -> str:
    """A vision-capable MLX build (post ``scripts/tmax_merge_vision.py``):
    ConditionalGeneration architecture + vision_config + image_token_id."""
    d = tmp_path / "vlm-build"
    d.mkdir()
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "vision_config": {"hidden_size": 1152},
        "image_token_id": 276,
    }
    json.dump(cfg, open(d / "config.json", "w"))
    return str(d)


def test_vision_capable_build_auto_detects_mllm(tmp_path: Path):
    """Acceptance (1): ``qmlx serve <vlm-mlx>`` WITHOUT ``--text-only``
    auto-routes a vision-capable build as MLLM — the chat route keeps
    accepting image blocks."""
    path = _vision_build_dir(tmp_path)
    assert is_mllm_model(path) is True
    engine = BatchedEngine(path)
    assert engine._is_mllm is True


def test_text_only_flag_forces_text_path(tmp_path: Path):
    """The #393 escape hatch: ``force_text=True`` overrides
    auto-detection (the documented way to serve a vision build as
    text-only)."""
    path = _vision_build_dir(tmp_path)
    engine = BatchedEngine(path, force_text=True)
    assert engine._is_mllm is False


def test_start_dispatch_mllm_serves_via_llm_path():
    """The mlx_vlm MLLM scheduler was removed in the Qwen-only build
    (#20). Issue #56 serves vision-capable models through the LLM
    engine path instead — ``_start_mllm`` must NOT raise the old
    'not available in this text-only build' RuntimeError; it delegates
    to ``_start_llm`` (model loads via mlx_lm; the #56 image-aware KV
    wiring handles the vision forward)."""
    engine = BatchedEngine("qwen3-vl-2b")
    engine._is_mllm = True
    calls: list[str] = []

    async def fake_start_llm() -> None:
        calls.append("llm")

    async def main() -> None:
        await engine._start_mllm()

    engine._start_llm = fake_start_llm  # type: ignore[method-assign]
    asyncio.run(main())
    assert calls == ["llm"]


def test_start_dispatch_routes_by_is_mllm():
    """``start()`` dispatch: is_mllm=True → ``_start_mllm`` (which
    delegates to the LLM path); is_mllm=False → ``_start_llm``
    directly."""
    engine = BatchedEngine("qwen3-vl-2b")
    engine._is_mllm = True
    calls: list[str] = []

    async def fake_start_llm() -> None:
        calls.append("llm")

    async def fake_start_mllm() -> None:
        calls.append("mllm")
        await fake_start_llm()

    engine._start_llm = fake_start_llm  # type: ignore[method-assign]
    engine._start_mllm = fake_start_mllm  # type: ignore[method-assign]
    asyncio.run(engine.start())
    assert calls == ["mllm", "llm"]

    engine2 = BatchedEngine("qwen3.5-4b-4bit")
    engine2._is_mllm = False
    calls2: list[str] = []

    async def fake_start_llm2() -> None:
        calls2.append("llm")

    engine2._start_llm = fake_start_llm2  # type: ignore[method-assign]
    asyncio.run(engine2.start())
    assert calls2 == ["llm"]


# ---------------------------------------------------------------------------
# (1) Serve-path wiring — chat route image gate
# ---------------------------------------------------------------------------

_IMAGE_MSG = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.png"},
            },
        ],
    }
]


def test_chat_route_image_gate_accepts_when_engine_is_mllm():
    """The route's gate is ``allow_image=engine.is_mllm``: a vision
    build (is_mllm=True) accepts image content blocks."""
    engine = BatchedEngine("qwen3-vl-2b")
    engine._is_mllm = True
    validate_content_blocks_for_capabilities(
        _IMAGE_MSG,
        model_name="qwen3-vl-2b",
        allow_image=engine.is_mllm,
        allow_video=engine.is_mllm,
    )


def test_chat_route_image_gate_rejects_when_text_only():
    """A text-only engine (is_mllm=False) rejects image blocks with a
    400 (ValueError → HTTPException at the route)."""
    engine = BatchedEngine("qwen3.5-4b-4bit")
    engine._is_mllm = False
    with pytest.raises(ValueError, match="image inputs"):
        validate_content_blocks_for_capabilities(
            _IMAGE_MSG,
            model_name="qwen3.5-4b-4bit",
            allow_image=engine.is_mllm,
            allow_video=engine.is_mllm,
        )


def test_chat_route_audio_still_rejected_for_vision():
    """The server has no chat audio lane — audio is a request-time 400
    even for vision builds (allow_audio=False at the route)."""
    engine = BatchedEngine("qwen3-vl-2b")
    engine._is_mllm = True
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {"type": "input_audio", "input_audio": {"data": "x"}},
            ],
        }
    ]
    with pytest.raises(ValueError, match="audio"):
        validate_content_blocks_for_capabilities(
            msg,
            model_name="qwen3-vl-2b",
            allow_image=engine.is_mllm,
            allow_video=engine.is_mllm,
            allow_audio=False,
        )


# ---------------------------------------------------------------------------
# (1) Serve-path wiring — media_key threading (engine → request)
# ---------------------------------------------------------------------------


class _StubEngineCore:
    """Records add_request/generate kwargs; yields one finished output."""

    def __init__(self) -> None:
        self.add_request_kwargs: dict[str, Any] = {}
        self.generate_kwargs: dict[str, Any] = {}

    async def add_request(self, **kwargs: Any) -> str:
        self.add_request_kwargs = kwargs
        return "req-stub"

    async def generate(self, **kwargs: Any) -> RequestOutput:
        self.generate_kwargs = kwargs
        return RequestOutput(
            request_id="req-stub",
            output_text="ok",
            output_token_ids=[1],
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=1,
        )

    async def stream_outputs(self, request_id: str):
        yield RequestOutput(
            request_id=request_id,
            new_text="ok",
            new_token_ids=[1],
            output_text="ok",
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=1,
        )


def _make_batched_engine(is_mllm: bool) -> BatchedEngine:
    engine = BatchedEngine("qwen3-vl-2b")
    engine._loaded = True
    engine._is_mllm = is_mllm
    engine._mllm_scheduler = None  # Qwen-only build: LLM path
    engine._engine = _StubEngineCore()  # type: ignore[assignment]
    return engine


def test_engine_generate_threads_media_key_to_engine_core():
    """Acceptance (1): an image+text chat request on a vision build
    threads the stable media key down to ``add_request`` — the
    scheduler/disk tiers key the checkpoint on (prompt, media)."""
    engine = _make_batched_engine(is_mllm=True)
    out = asyncio.run(
        engine.generate(prompt="what is this?", images=[IMG_A], max_tokens=8)
    )
    assert out.text == "ok"
    assert (
        engine._engine.generate_kwargs["media_key"]  # type: ignore[union-attr]
        == _media_key(IMG_A)
    )


def test_engine_generate_text_only_keeps_empty_media_key():
    """No images → media_key stays "" — text-only checkpointing is
    byte-identical to pre-#56."""
    engine = _make_batched_engine(is_mllm=True)
    asyncio.run(engine.generate(prompt="hello", max_tokens=8))
    assert engine._engine.generate_kwargs["media_key"] == ""  # type: ignore[union-attr]


def test_engine_stream_generate_threads_media_key():
    """The streaming path must key checkpoints identically to the
    non-streaming path (a streamed turn 1 and a non-streamed turn 2
    must not diverge)."""
    engine = _make_batched_engine(is_mllm=True)

    async def collect() -> list[str]:
        texts = []
        async for out in engine.stream_generate(
            prompt="what is this?", images=[IMG_A], max_tokens=8
        ):
            texts.append(out.text)
        return texts

    asyncio.run(collect())
    assert (
        engine._engine.add_request_kwargs["media_key"]  # type: ignore[union-attr]
        == _media_key(IMG_A)
    )


def test_engine_core_add_request_stores_media_key_and_pixel_values():
    """``add_request`` stores both the media key and the processed
    vision tensors on the Request (the scheduler's chunked-prefill
    vision kwargs read ``request.pixel_values``)."""
    sched = _make_scheduler()
    core = AsyncEngineCore(object(), _StubTok(), executor=None)
    core.engine.scheduler = sched  # type: ignore[assignment]

    captured: dict[str, Any] = {}

    def fake_add_request(req: Request) -> None:
        captured["req"] = req

    sched.add_request = fake_add_request  # type: ignore[method-assign]

    pv = mx.random.normal((1, 8), key=mx.random.key(2))
    asyncio.run(
        core.add_request(
            prompt="what is this?",
            images=[IMG_A],
            media_key=_media_key(IMG_A),
            pixel_values=pv,
        )
    )
    assert captured["req"].media_key == _media_key(IMG_A)
    assert captured["req"].pixel_values is pv


# ---------------------------------------------------------------------------
# (2) Repeated multimodal turns — hit on same image, no stale reuse
# ---------------------------------------------------------------------------


def _make_scheduler(interval: int = 256) -> Scheduler:
    cfg = SchedulerConfig()
    cfg.kv_disk_checkpoint_interval = interval
    cfg.kv_cache_dtype = "bf16"
    cfg.kv_disk_restore_enabled = False
    cfg.enable_prefix_cache = False

    sched = Scheduler(model=object(), tokenizer=_StubTok(), config=cfg)
    sched._model_name = "test-model"
    return sched


class _StubTok:
    eos_token_id = 0
    pad_token_id = 0
    special_tokens_map: dict[str, Any] = {}

    def decode(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        return ""


def test_repeated_multimodal_turn_prefix_hit(root: str):
    """Acceptance (2): a repeated multimodal turn (same image, extended
    conversation) is a prefix hit — the turn-1 checkpoint (prompt +
    media) is reused for the turn-2 prefix."""
    sched = _make_scheduler()
    tokens = list(range(1, 257))  # turn 1: 256 prompt tokens (image expanded)
    cache = _seed_kv_cache(num_tokens=len(tokens))
    mk = _media_key(IMG_A)
    sched._disk_persist_boundary(tokens, cache, media_key=mk)

    # Turn 2: the SAME image, the conversation extended with new tokens.
    turn2 = tokens + [900, 901, 902]
    loaded = _dkc.get_content_index().lookup(turn2, mk)
    assert loaded is not None
    assert loaded.token_offset == len(tokens)


def test_image_change_no_stale_reuse(root: str):
    """Acceptance (2): a CHANGED image never reuses the previous
    image's KV, even at identical tokens."""
    sched = _make_scheduler()
    tokens = list(range(1, 257))
    cache = _seed_kv_cache(num_tokens=len(tokens))
    sched._disk_persist_boundary(tokens, cache, media_key=_media_key(IMG_A))

    assert _dkc.get_content_index().lookup(tokens, _media_key(IMG_B)) is None
    # ...and a text-only query for the same tokens must not match either.
    assert _dkc.get_content_index().lookup(tokens) is None
    # ...while the matching image still hits.
    assert _dkc.get_content_index().lookup(tokens, _media_key(IMG_A)) is not None


# ---------------------------------------------------------------------------
# (3) Disk KV checkpoint round-trip with image-aware keys
# ---------------------------------------------------------------------------


def test_disk_kv_roundtrip_image_aware_keys(root: str):
    """Acceptance (3): a checkpoint written by the scheduler store site
    round-trips through a process restart — the sidecar's media key
    re-adds the checkpoint to the SAME media bucket, so the same image
    still hits and a changed image still misses after the rebuild."""
    sched = _make_scheduler()
    tokens = list(range(1, 257))
    cache = _seed_kv_cache(num_tokens=len(tokens))
    mk_a = _media_key(IMG_A)
    sched._disk_persist_boundary(tokens, cache, media_key=mk_a)

    # The on-disk sidecar carries the media key.
    import hashlib

    raw = b"".join(int(t).to_bytes(4, "little", signed=True) for t in tokens)
    media_hash = hashlib.sha256(
        b"test-model" + raw + b"\x00" + mk_a.encode()
    ).hexdigest()[:16]
    sidecar = _dkc.read_sidecar(root, media_hash, len(tokens))
    assert sidecar is not None
    assert sidecar.get("media_key") == mk_a

    # Simulate a restart: wipe the in-memory index, rebuild from disk.
    _dkc.reset_content_index_for_tests()
    n = _dkc.build_content_index(root)
    assert n == 1

    idx = _dkc.get_content_index()
    # Same image → restored from the rebuilt index (round-trip hit).
    assert idx.lookup(tokens, mk_a) is not None
    # Changed image → still no stale reuse after the rebuild.
    assert idx.lookup(tokens, _media_key(IMG_B)) is None
    # Text-only query for image tokens → no cross-match after rebuild.
    assert idx.lookup(tokens) is None


# ---------------------------------------------------------------------------
# (4) MTP interaction — MTP-off with images
# ---------------------------------------------------------------------------


def test_mtp_off_with_images_threads_media_key(root: str):
    """Acceptance (4): with MTP off (``spec_decode="none"``) an image
    request threads its media key through the disk persist path; the
    model mock carries NO MTP attributes (no mtp_forward / mtp_head),
    proving the MTP-off path needs nothing MTP-shaped."""

    class _NoMtpModel:
        pass  # deliberately no mtp_* attributes

    cfg = SchedulerConfig()
    cfg.spec_decode = "none"
    cfg.kv_disk_checkpoint_interval = 256
    cfg.kv_cache_dtype = "bf16"
    cfg.kv_disk_restore_enabled = False
    cfg.enable_prefix_cache = False

    sched = Scheduler(model=_NoMtpModel(), tokenizer=_StubTok(), config=cfg)
    sched._model_name = "test-model"

    tokens = list(range(1, 257))
    cache = _seed_kv_cache(num_tokens=len(tokens))
    req = Request(
        request_id="req-mm",
        prompt="what is this?",
        sampling_params=SamplingParams(max_tokens=32),
        images=[IMG_A],
        media_key=_media_key(IMG_A),
    )
    sched._disk_persist_boundary(
        tokens, cache, media_key=getattr(req, "media_key", "") or ""
    )
    idx = _dkc.get_content_index()
    assert idx.lookup(tokens, req.media_key) is not None
    assert idx.lookup(tokens, _media_key(IMG_B)) is None


def test_mtp_on_with_images_keys_same_media_bucket(root: str):
    """MTP-on (``spec_decode="mtp"``) does not change the media bucket
    keying — the same (tokens, image) pair resolves to the same
    checkpoint whether MTP is on or off (the media key is orthogonal
    to the spec-decode mode)."""
    cfg = SchedulerConfig()
    cfg.spec_decode = "mtp"
    cfg.kv_disk_checkpoint_interval = 256
    cfg.kv_cache_dtype = "bf16"
    cfg.kv_disk_restore_enabled = False
    cfg.enable_prefix_cache = False

    sched = Scheduler(model=object(), tokenizer=_StubTok(), config=cfg)
    sched._model_name = "test-model"

    tokens = list(range(1, 257))
    cache = _seed_kv_cache(num_tokens=len(tokens))
    mk = _media_key(IMG_A)
    sched._disk_persist_boundary(tokens, cache, media_key=mk)
    assert _dkc.get_content_index().lookup(tokens, mk) is not None
    assert _dkc.get_content_index().lookup(tokens, _media_key(IMG_B)) is None
