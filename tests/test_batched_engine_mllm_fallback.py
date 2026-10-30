# SPDX-License-Identifier: Apache-2.0
"""Regression tests: MLLM auto-routing dispatch (issue #56, #71).

The Qwen-only simplification (#20) stripped the mlx_vlm MLLM scheduler;
the 2026-10-28 dogfood added a text-only fallback for auto-detected VLMs
(#78). #71 restores vision serving through the LLM engine path
(image-aware KV wiring), so the dispatch is back to the simple contract:
any MLLM verdict (auto-detected or explicit ``--mllm``) routes to
``_start_mllm``; only ``--text-only`` (force_text, the #393 escape
hatch) forces the text path.
"""

from __future__ import annotations

import vllm_mlx.engine.batched as _batched


class TestStartMllmRouting:
    async def test_auto_mllm_routes_to_mllm_path(self):
        """Auto-detected VLM → ``_start_mllm`` runs (vision serving
        restored in #71) and ``_is_mllm`` stays True so the chat route
        keeps accepting image/video content blocks."""
        engine = _batched.BatchedEngine("some-vl-model")
        engine._is_mllm = True  # auto-detect verdict
        engine._mllm_explicit = False

        calls: dict = {}

        async def _fake_start_llm():
            calls["llm"] = True

        async def _fake_start_mllm():
            calls["mllm"] = True

        engine._start_llm = _fake_start_llm
        engine._start_mllm = _fake_start_mllm

        await engine.start()

        assert calls == {"mllm": True}
        assert engine._is_mllm is True
        assert engine._loaded is True

    async def test_explicit_mllm_routes_to_mllm_path(self):
        """``--mllm`` (force_mllm=True) routes to ``_start_mllm`` like
        any other MLLM verdict — the pre-#71 hard-fail stub is gone
        (vision-capable builds serve images through the LLM engine
        path, issue #56)."""
        engine = _batched.BatchedEngine("some-vl-model", force_mllm=True)
        assert engine._mllm_explicit is True

        calls: dict = {}

        async def _fake_start_llm():
            calls["llm"] = True

        async def _fake_start_mllm():
            calls["mllm"] = True

        engine._start_llm = _fake_start_llm
        engine._start_mllm = _fake_start_mllm

        await engine.start()
        assert calls == {"mllm": True}
        assert engine._loaded is True

    async def test_force_text_never_routes_mllm(self):
        """--text-only forces the text path regardless of auto-detect
        (the pre-existing #393 escape hatch, unchanged)."""
        engine = _batched.BatchedEngine("some-vl-model", force_text=True)
        assert engine._is_mllm is False

        calls: dict = {}

        async def _fake_start_llm():
            calls["llm"] = True

        async def _fake_start_mllm():
            calls["mllm"] = True

        engine._start_llm = _fake_start_llm
        engine._start_mllm = _fake_start_mllm

        await engine.start()

        assert calls == {"llm": True}
