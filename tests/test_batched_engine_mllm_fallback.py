# SPDX-License-Identifier: Apache-2.0
"""Regression tests: MLLM auto-routing fallback in a build without
MLLM/vision serving (stripped in the Qwen-only simplification, #20).

Dogfood 2026-10-28 (oQ4e vision-capable model): auto-detection correctly
routes a vision-capable checkpoint to the MLLM path, but the stripped
build's ``_start_mllm`` is a raising stub — so the model hard-failed at
boot with no way to serve its text + MTP path. Auto-detected (non
explicit) MLLM routing must now fall back to the text path with a
warning; only an explicit ``--mllm`` still hard-fails.
"""

from __future__ import annotations

import vllm_mlx.engine.batched as _batched


class TestStartMllmFallback:
    async def test_auto_mllm_falls_back_to_text_path(self):
        """Auto-detected VLM + no MLLM build → ``_start_llm`` runs, the
        stub is not raised, and ``_is_mllm`` clears so the vision-gated
        route/service code treats the engine as text-active."""
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

        assert calls == {"llm": True}
        assert engine._is_mllm is False
        assert engine._loaded is True

    async def test_explicit_mllm_still_hard_fails(self):
        """``--mllm`` (force_mllm=True) on a build without MLLM serving
        keeps the hard fail — the operator asked for vision and gets a
        clear refusal naming --text-only / a vision-capable build."""
        engine = _batched.BatchedEngine("some-vl-model", force_mllm=True)
        assert engine._mllm_explicit is True

        calls: dict = {}

        async def _fake_start_llm():
            calls["llm"] = True

        engine._start_llm = _fake_start_llm

        try:
            await engine.start()
            raised = False
        except RuntimeError as exc:
            raised = True
            assert "text-only build" in str(exc)
        assert raised, "explicit --mllm must hard-fail on the stub"
        assert calls == {}
        assert engine._loaded is False

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
