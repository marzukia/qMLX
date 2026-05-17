# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MLLMBatchGenerator model-call kwargs.

Some mlx-vlm model classes (notably ``Gemma3ForConditionalGeneration``)
declare ``pixel_values`` as a *required* positional kwarg in ``__call__``,
even though the inner ``get_input_embeddings`` already handles ``None`` for
the text-only path. Omitting the kwarg raises ``TypeError`` for every
text-only request to those models, so ``_run_vision_encoding`` must always
pass it through — including when it's ``None``.
"""

import mlx.core as mx

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest


class _RecordingModel:
    """VLM model stub that captures kwargs from its ``__call__``."""

    def __init__(self):
        self.last_call_kwargs = None
        self.last_input_ids = None
        # Provide a language_model attribute so the generator's
        # is_vlm branch picks it up without warnings.
        self.language_model = object()

    def __call__(self, input_ids, cache=None, **kwargs):
        self.last_input_ids = input_ids
        self.last_call_kwargs = kwargs
        # Return a dummy logits tensor — generator only inspects shape via
        # ``hasattr(output, "logits")``; the value is irrelevant for this test.
        return mx.zeros((1, 1, 8))


def _make_generator(model: _RecordingModel) -> MLLMBatchGenerator:
    """Construct a generator without booting Metal / vision cache plumbing."""
    return MLLMBatchGenerator(
        model=model,
        processor=object(),
        mm_processor=None,
        enable_vision_cache=False,
    )


def _make_request(*, pixel_values, extra_kwargs=None) -> MLLMBatchRequest:
    return MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="hello",
        max_tokens=8,
        input_ids=mx.array([1, 2, 3], dtype=mx.int32),
        pixel_values=pixel_values,
        extra_kwargs=extra_kwargs or {},
    )


def test_run_vision_encoding_passes_pixel_values_none_for_text_only_request():
    """Text-only request still includes pixel_values=None in kwargs.

    Gemma3ForConditionalGeneration's ``__call__`` declares ``pixel_values``
    as a required kwarg, so we must always forward it — even when None.
    """
    model = _RecordingModel()
    gen = _make_generator(model)
    request = _make_request(pixel_values=None)

    gen._run_vision_encoding(request, cache=None)

    assert "pixel_values" in model.last_call_kwargs
    assert model.last_call_kwargs["pixel_values"] is None


def test_run_vision_encoding_forwards_pixel_values_when_set():
    """Multimodal request keeps forwarding the real pixel tensor."""
    model = _RecordingModel()
    gen = _make_generator(model)
    pixels = mx.zeros((1, 3, 4, 4))
    request = _make_request(pixel_values=pixels)

    gen._run_vision_encoding(request, cache=None)

    assert "pixel_values" in model.last_call_kwargs
    # Must be the same object we put in — generator should not silently copy
    # or downcast pixel_values before the forward pass.
    assert model.last_call_kwargs["pixel_values"] is pixels


def test_run_vision_encoding_preserves_extra_kwargs_alongside_pixel_values():
    """Extra processor kwargs (e.g. token_type_ids) survive alongside pixel_values."""
    model = _RecordingModel()
    gen = _make_generator(model)
    request = _make_request(
        pixel_values=None,
        extra_kwargs={"token_type_ids": mx.array([0, 0, 1])},
    )

    gen._run_vision_encoding(request, cache=None)

    assert "pixel_values" in model.last_call_kwargs
    assert model.last_call_kwargs["pixel_values"] is None
    assert "token_type_ids" in model.last_call_kwargs


# ---------------------------------------------------------------------------
# Shutdown — mx.synchronize must not propagate cross-thread errors
# ---------------------------------------------------------------------------


def test_close_swallows_synchronize_thread_error(monkeypatch):
    """`close()` must not propagate RuntimeError from mx.synchronize.

    mlx-lm 0.31.3+ streams are thread-local. When the engine is torn down
    from a thread that isn't the one that owns MLLMBatchGenerator._stream,
    mx.synchronize raises `There is no Stream(gpu, N) in current thread`.
    Pre-fix this propagated out of the lifespan shutdown and produced a
    scary traceback (Persona E v0.6.51 onboarding finding). The sync is
    best-effort on shutdown; the wired-limit reset is what matters.
    """
    import mlx.core as mx

    # Construct a generator and force the wired-limit branch to execute.
    gen = _make_generator(_RecordingModel())
    gen._old_wired_limit = 1234  # any sentinel triggers the close path

    sync_calls: list[object] = []
    set_limit_calls: list[int] = []

    def _raising_sync(stream):
        sync_calls.append(stream)
        raise RuntimeError("There is no Stream(gpu, 2) in current thread")

    def _record_set_limit(value):
        set_limit_calls.append(value)
        return value

    monkeypatch.setattr(mx, "synchronize", _raising_sync)
    monkeypatch.setattr(mx, "set_wired_limit", _record_set_limit)

    # Must not raise.
    gen.close()

    # Best-effort sync attempted exactly once.
    assert len(sync_calls) == 1
    # Wired limit was still reset to the original value — the important
    # cleanup is not skipped just because the cross-thread sync failed.
    assert set_limit_calls == [1234]
    # State is cleared so __del__ is a no-op afterward.
    assert gen._old_wired_limit is None


def test_close_propagates_non_runtime_errors_from_set_wired_limit(monkeypatch):
    """Errors from set_wired_limit are unrelated to the thread bug — keep
    propagating them so a real OS-level failure isn't silently swallowed.
    """
    import mlx.core as mx

    gen = _make_generator(_RecordingModel())
    gen._old_wired_limit = 999

    monkeypatch.setattr(mx, "synchronize", lambda _s: None)

    def _boom(value):
        raise OSError("metal API call failed")

    monkeypatch.setattr(mx, "set_wired_limit", _boom)

    import pytest

    with pytest.raises(OSError, match="metal API call failed"):
        gen.close()
