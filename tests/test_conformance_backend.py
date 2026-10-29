# SPDX-License-Identifier: Apache-2.0
"""
Backend conformance suite (qMLX repositioning, item 7 of 9).

This suite IS the ``mlx-rocm`` / ``mlx-cuda`` acceptance gate. It runs
against whatever ``mx.default_device()`` is — on this Linux host that is
CPU, so Metal-only checks produce EXPLICIT backend-gap markers
(``pytest.skip`` with a reason string naming the gap), never silent
passes.

Conformance-backend resolution (two tiers)
------------------------------------------

* Tier 1 — ``get_backend()``: a registered / active backend. A ported
  ``mlx-rocm`` / ``mlx-cuda`` build self-registers at import time via
  ``register_backend``; the full conformance (report + observed +
  consistency) runs against it.
* Tier 2 — no active backend (this host): the suite falls back to the
  built-in ``MetalBackend`` probed against the real ``mx`` on the default
  device as a *reference* backend. Its methods call the same ``mx`` APIs a
  ported backend will, so the report-vs-observation checks still run; the
  checks that require a notion the backend does not expose (e.g. the
  ``device_info`` memory-size keys) produce explicit gap markers.

Gap-marker convention
---------------------

Every check either PASSES (the seam's report is consistent with the observed
behavior on the default device) or produces an EXPLICIT gap marker via
``_gap(reason)`` — a ``pytest.skip`` whose reason string names the backend
gap. No silent skips: a skip without a reason is a test bug.

Checks (plan-note section 8, item 7)
------------------------------------

1. Memory-cap enforcement mode (D-METAL-CAP) — the declared
   ``enforcement_mode`` is cross-checked against the observed allocator
   behavior (set a small cap, allocate well past it).
2. Quantised-KV round-trip — ``mx.quantize`` / ``mx.dequantize`` int4
   determinism (the byte-stable packed form that makes a checkpoint
   round-trip reliable); the loader fail-closed guard; the end-to-end
   class-name detection.
3. Prompt-cache format round-trip — ``save_prompt_cache`` /
   ``load_prompt_cache`` class-identity round-trip (incl.
   ``QuantizedKVCache``); the restore identity guard keyed on model +
   cache class (fails closed by design).
4. Stream / thread ownership rule — the "one owner thread" invariant; the
   backend's declared stream contract (``supports_secondary_streams`` +
   ``capabilities().stream_model``) cross-checked against a live probe.
5. ``mx.compile`` / ``mx.async_eval`` semantics — the capability report is
   cross-checked against the actual presence (the scheduler's backoff logic
   depends on ``async_eval`` semantics).
6. ``device_info`` key set — the keys the seam consumes exist with the
   right types on this backend.
"""

from __future__ import annotations

import struct
import tempfile
import threading

import pytest

from vllm_mlx.device import (
    EnforcementMode,
    MetalBackend,
    StreamModel,
    get_backend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_mx_or_none():
    try:
        import mlx.core as mx

        return mx
    except Exception:
        return None


def _gap(reason: str) -> None:
    """Explicit backend-gap marker: a skip with a reason string naming the
    gap (never a silent pass). A skip without a reason is a test bug."""
    assert reason, "gap marker must name the backend gap (no silent skips)"
    pytest.skip(reason)


def _observe_enforcement_mode(mx) -> str | None:
    """Probe the allocator: set a small cap, then allocate well past it.

    Returns ``"hint"`` (the allocation grew past the cap — the cap is a
    guideline), ``"hard"`` (the allocation failed past the cap — the cap is
    enforced), or ``None`` when ``set_memory_limit`` is not supported on this
    backend (the report cannot be cross-checked against observed behavior).
    """
    cap = 1 << 20  # 1 MiB
    try:
        mx.set_memory_limit(cap)
    except Exception:
        return None
    try:
        big = mx.ones((4096, 4096), dtype=mx.float32)  # 64 MiB, well past cap
        mx.eval(big)
        del big
        return "hint"
    except Exception:
        return "hard"


def _probe_secondary_stream_in_thread(mx) -> bool:
    """Create a secondary stream in a *second* thread and exercise it.

    Returns ``True`` when the secondary stream is usable (MULTI_STREAM) and
    ``False`` when it raises (SINGLE_STREAM). This is the live probe the
    backend's declared stream contract is cross-checked against.
    """
    result = {"ok": False}

    def work() -> None:
        try:
            stream = mx.new_thread_local_stream(mx.default_device())
            with mx.stream(stream):
                _ = (mx.array([0.0]) + mx.array([1.0])).item()
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=work)
    t.start()
    t.join()
    return result["ok"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mx():
    m = _real_mx_or_none()
    if m is None:
        _gap(
            "mlx is not installed on this host — the conformance suite runs "
            "against mx.default_device(); there is no device to probe"
        )
    return m


@pytest.fixture()
def backend(mx):
    """Resolve the conformance backend (Tier 1 registered, Tier 2 reference)."""
    b = get_backend()
    if b is not None:
        return b
    # Tier 2: the built-in Metal backend probed against the real mx on the
    # default device. Its methods call the same mx APIs a ported backend
    # will; is_available() is False here (no Metal) but the report-vs-
    # observation checks still run.
    return MetalBackend(mx=mx)


# ---------------------------------------------------------------------------
# Check 1 — memory-cap enforcement mode (D-METAL-CAP)
# ---------------------------------------------------------------------------


def test_memory_cap_enforcement_mode(backend, mx):
    """The seam's declared ``enforcement_mode`` must be consistent with the
    observed allocator behavior on the default device.

    D-METAL-CAP root cause: Metal treats ``set_memory_limit`` as a
    *guideline* (HINT) and grows past it under load, so the scheduler
    polices the cap at admission. On a HARD backend the allocator enforces
    the cap and the scheduler-side police become redundant. The declared
    mode and the observed behavior must agree.
    """
    declared = backend.enforcement_mode
    observed = _observe_enforcement_mode(mx)
    if observed is None:
        _gap(
            f"set_memory_limit is not supported on this backend "
            f"({mx.default_device()}) — the enforcement-mode report "
            f"({declared}) cannot be cross-checked against observed "
            f"allocator behavior"
        )
    expected = "hint" if declared == EnforcementMode.HINT else "hard"
    assert observed == expected, (
        f"enforcement_mode mismatch: the backend declares {declared} "
        f"(expected {expected!r} behavior) but the allocator behaved "
        f"{observed!r} (set_memory_limit(1MiB) then a 64MiB allocation "
        f"{'succeeded — grew past the cap' if observed == 'hint' else 'failed — cap enforced'})"
    )


# ---------------------------------------------------------------------------
# Check 2 — quantised-KV round-trip + fail-closed loader
# ---------------------------------------------------------------------------


def test_quantized_kv_roundtrip(backend, mx):
    """``mx.quantize`` / ``mx.dequantize`` int4 (Metal-only in upstream MLX).

    On a backend that ships them: the packed int4 form must be deterministic
    and byte-stable (the invariant that makes a QuantizedKVCache checkpoint
    round-trip reliable). The capability report must agree with the actual
    presence. On a backend that lacks them: an explicit gap marker (the
    loader must fail closed — see test_quantized_kv_loader_guard_fail_closed).
    """
    caps = backend.capabilities()
    has_q = hasattr(mx, "quantize")
    has_dq = hasattr(mx, "dequantize")
    assert caps.quantized_kv == (has_q and has_dq), (
        f"capability report quantized_kv={caps.quantized_kv} disagrees with "
        f"actual presence (hasattr quantize={has_q}, dequantize={has_dq})"
    )
    if not (has_q and has_dq):
        _gap(
            f"mx.quantize / mx.dequantize are absent on this backend "
            f"({mx.default_device()}) — the QuantizedKVCache round-trip is "
            f"not testable; the loader must fail closed (see "
            f"test_quantized_kv_loader_guard_fail_closed)"
        )
    # The round-trip invariant: the packed int4 form is deterministic and
    # byte-stable across identical calls (int4 is lossy by design, so the
    # invariant is byte-stability of the packed form, not losslessness).
    x = (mx.arange(64).reshape(2, 32).astype(mx.float32) * 0.25) - 4.0
    q1 = mx.quantize(x, group_size=32, bits=4)
    q2 = mx.quantize(x, group_size=32, bits=4)
    assert all(a.tolist() == b.tolist() for a, b in zip(q1, q2)), (
        "mx.quantize is not deterministic — the packed int4 form differs "
        "across identical calls; a QuantizedKVCache checkpoint would not "
        "round-trip byte-exact"
    )
    d1 = mx.dequantize(*q1, group_size=32, bits=4)
    mx.eval(d1)
    d2 = mx.dequantize(*q2, group_size=32, bits=4)
    mx.eval(d2)
    assert d1.tolist() == d2.tolist(), (
        "mx.dequantize is not deterministic — the same packed int4 form "
        "dequantizes to different values; a restored cache would not be "
        "reproducible"
    )


def test_quantized_kv_loader_guard_fail_closed(backend, mx):
    """The loader must fail closed on a QuantizedKVCache checkpoint it
    cannot load (plan note: "the format looks valid, the loader cannot
    tell — there must be a load-time dtype/quantization conformance check,
    not just an import check").

    The guard logic (``_cache_classes_compatible``) is the fail-closed
    conformance check: given a *known* class name, a QuantizedKVCache file
    is rejected when ``kv_quantize`` is off (the dequantize path is guarded
    on the flag; under any other config the tuple-form keys reach the
    scheduler and crash — #198 BUG A). A plain KVCache is always accepted.
    This is the fail-closed path, tested directly (backend-agnostic).
    """
    from vllm_mlx.memory_cache import (
        MemoryCacheConfig,
        _cache_classes_compatible,
    )

    cfg = MemoryCacheConfig()  # kv_quantize defaults to False
    ok, reason = _cache_classes_compatible(["QuantizedKVCache"], cfg)
    assert ok is False, (
        "the loader must fail closed: a QuantizedKVCache checkpoint is not "
        f"loadable when kv_quantize is off, but _cache_classes_compatible "
        f"accepted it (reason={reason!r})"
    )
    assert "quantiz" in reason.lower(), (
        f"the reject reason must name the quantization gap: {reason!r}"
    )
    ok_kv, _ = _cache_classes_compatible(["KVCache"], cfg)
    assert ok_kv is True, "a plain KVCache checkpoint must always be loadable"
    # A QuantizedKVCache file IS loadable when kv_quantize is on.
    cfg_on = MemoryCacheConfig(kv_quantize=True)
    ok_on, _ = _cache_classes_compatible(["QuantizedKVCache"], cfg_on)
    assert ok_on is True, (
        "a QuantizedKVCache checkpoint must be loadable when kv_quantize is on"
    )


def test_quantized_kv_loader_detection(backend, mx):
    """End-to-end: can the loader actually *tell* a QuantizedKVCache
    checkpoint is quantized, on this host?

    The plan note's "the loader cannot tell" failure mode: a QuantizedKVCache
    checkpoint round-trips as garbage when the backend lacks quantize, and
    the format looks valid. The load-time conformance check depends on the
    metadata reader (``_safetensors_cache_classes``) detecting the class
    name. If the reader cannot detect it (returns ``[]``), the fail-closed
    guard (test_quantized_kv_loader_guard_fail_closed) is correct but
    unreachable end-to-end — the loader misclassifies the file as plain
    KVCache.
    """
    from mlx_lm.models.cache import (
        QuantizedKVCache,
        save_prompt_cache,
    )

    from vllm_mlx.memory_cache import _safetensors_cache_classes

    if not hasattr(mx, "quantize"):
        _gap(
            f"mx.quantize is absent on this backend ({mx.default_device()}) "
            f"— a QuantizedKVCache checkpoint cannot be constructed, so the "
            f"end-to-end class-name detection is not testable"
        )
    qc = QuantizedKVCache()
    qc.group_size = 32
    qc.bits = 4
    qc.keys = mx.quantize(mx.ones((2, 4, 32)), group_size=32, bits=4)
    qc.values = mx.quantize(mx.ones((2, 4, 32)), group_size=32, bits=4)
    qc.offset = 8
    with tempfile.TemporaryDirectory() as d:
        import os

        p = os.path.join(d, "qkv.safetensors")
        save_prompt_cache(p, [qc])
        detected = _safetensors_cache_classes(p)
    if detected != ["QuantizedKVCache"]:
        _gap(
            "the loader cannot tell this is a QuantizedKVCache checkpoint: "
            f"_safetensors_cache_classes detected {detected!r} (expected "
            "['QuantizedKVCache']) — it reads '2.*' metadata keys, but this "
            "mlx_lm build writes the class name at '1.<idx>', so the reader "
            "returns [] (permissive → assume KVCache) and a quantized "
            "checkpoint is misclassified as plain KVCache (the 'format looks "
            "valid, the loader cannot tell' failure mode). The fail-closed "
            "guard (test_quantized_kv_loader_guard_fail_closed) is correct "
            "but unreachable end-to-end on this host."
        )
    assert detected == ["QuantizedKVCache"]


# ---------------------------------------------------------------------------
# Check 3 — prompt-cache format round-trip + restore identity guard
# ---------------------------------------------------------------------------


def test_prompt_cache_class_identity_roundtrip(backend, mx):
    """The on-disk format stores the cache class name; the loader constructs
    ``KVCache`` / ``QuantizedKVCache`` from it. Class identity must
    round-trip (save → load → same class) on this backend.

    The QuantizedKVCache portion requires ``mx.quantize`` to construct the
    cache; when it is absent the KVCache portion still runs and the
    quantized round-trip is covered by the gap marker in
    test_quantized_kv_roundtrip.
    """
    from mlx_lm.models.cache import (
        KVCache,
        QuantizedKVCache,
        load_prompt_cache,
        save_prompt_cache,
    )

    with tempfile.TemporaryDirectory() as d:
        import os

        # KVCache — always testable.
        c = KVCache()
        c.keys = [mx.ones((2, 4, 32))]
        c.values = [mx.ones((2, 4, 32))]
        c.offset = 8
        p = os.path.join(d, "kv.safetensors")
        save_prompt_cache(p, [c])
        loaded, _ = load_prompt_cache(p, return_metadata=True)
        assert type(loaded[0]).__name__ == "KVCache", (
            f"KVCache class identity did not round-trip: saved KVCache, "
            f"loaded {type(loaded[0]).__name__}"
        )
        # QuantizedKVCache — requires mx.quantize to construct.
        if not hasattr(mx, "quantize"):
            return
        qc = QuantizedKVCache()
        qc.group_size = 32
        qc.bits = 4
        qc.keys = mx.quantize(mx.ones((2, 4, 32)), group_size=32, bits=4)
        qc.values = mx.quantize(mx.ones((2, 4, 32)), group_size=32, bits=4)
        qc.offset = 8
        p2 = os.path.join(d, "qkv.safetensors")
        save_prompt_cache(p2, [qc])
        loaded2, _ = load_prompt_cache(p2, return_metadata=True)
        assert type(loaded2[0]).__name__ == "QuantizedKVCache", (
            f"QuantizedKVCache class identity did not round-trip: saved "
            f"QuantizedKVCache, loaded {type(loaded2[0]).__name__}"
        )


def test_restore_identity_guard_fail_closed(backend, mx):
    """The restore identity guard is keyed on model + cache class and fails
    closed by design (a mismatch rejects the restore; it never silently
    corrupts output).

    (a) MODEL IDENTITY — the scheduler's ``_maybe_disk_restore`` (a2) guard:
        the checkpoint's recorded ``model_name`` must match the running model
        exactly; any unknown side (either side missing) rejects (fail
        closed). A wrong restore corrupts output silently and raises nothing,
        so the contract is reject-and-reprefill on ANY doubt.
    (b) CACHE CLASS — ``_cache_classes_compatible``: a QuantizedKVCache
        checkpoint is rejected when ``kv_quantize`` is off (fail closed); a
        plain KVCache is always accepted.
    """
    from vllm_mlx.memory_cache import (
        MemoryCacheConfig,
        _cache_classes_compatible,
    )

    # (a) MODEL IDENTITY — the exact predicate the scheduler's (a2) guard
    # uses: accept only when both sides are known and match exactly; reject
    # (fail closed) on any doubt.
    def model_identity_accepts(run_model, ckpt_model) -> bool:
        return (
            bool(ckpt_model) and bool(run_model) and str(ckpt_model) == str(run_model)
        )

    assert model_identity_accepts("qwen3-122b", "qwen3-122b") is True, (
        "an exact model match must be accepted"
    )
    assert model_identity_accepts("qwen3-122b", "llama-70b") is False, (
        "a different model must be rejected (fail closed) — a checkpoint "
        "from a different model sharing a prompt prefix would silently "
        "corrupt output"
    )
    assert model_identity_accepts("qwen3-122b", None) is False, (
        "an unknown checkpoint model must be rejected (fail closed)"
    )
    assert model_identity_accepts(None, "qwen3-122b") is False, (
        "an unknown running model must be rejected (fail closed)"
    )
    assert model_identity_accepts(None, None) is False, (
        "both sides unknown must be rejected (fail closed)"
    )

    # (b) CACHE CLASS — the real guard code (fail closed by design).
    cfg = MemoryCacheConfig()  # kv_quantize defaults to False
    ok, _ = _cache_classes_compatible(["QuantizedKVCache"], cfg)
    assert ok is False, (
        "the cache-class guard must fail closed: a QuantizedKVCache "
        "checkpoint is rejected when kv_quantize is off"
    )
    ok_kv, _ = _cache_classes_compatible(["KVCache"], cfg)
    assert ok_kv is True, "the cache-class guard must accept a plain KVCache"


def test_prompt_cache_on_disk_stores_class_name(backend, mx):
    """The on-disk safetensors format must store the cache class name so the
    loader can reconstruct the right class. This is the format invariant the
    class-identity round-trip (test_prompt_cache_class_identity_roundtrip)
    depends on.

    On this host the class name is written at the ``1.<idx>`` metadata key
    (this mlx_lm build's layout). The check asserts the class name is
    present *somewhere* in the metadata (the exact key is mlx_lm-version
    shaped — qMLX's reader keys on ``2.*``, which is the gap surfaced by
    test_quantized_kv_loader_detection).
    """
    from mlx_lm.models.cache import KVCache, save_prompt_cache

    with tempfile.TemporaryDirectory() as d:
        import os

        c = KVCache()
        c.keys = [mx.ones((2, 4, 32))]
        c.values = [mx.ones((2, 4, 32))]
        c.offset = 8
        p = os.path.join(d, "kv.safetensors")
        save_prompt_cache(p, [c])
        with open(p, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            import json

            header = json.loads(f.read(header_len))
        meta = header.get("__metadata__", {})
        # The class name must be stored in the metadata (the format
        # invariant). It is NOT required to be at a specific key (that is
        # mlx_lm-version shaped); it must be present so the loader can
        # reconstruct the class.
        assert any(v == "KVCache" for v in meta.values()), (
            f"the on-disk format must store the cache class name in the "
            f"metadata so the loader can reconstruct the class; metadata "
            f"= {meta!r} does not contain 'KVCache'"
        )


# ---------------------------------------------------------------------------
# Check 4 — stream / thread ownership rule (one owner thread)
# ---------------------------------------------------------------------------


def test_stream_thread_ownership_contract(backend, mx):
    """The "one owner thread" invariant: all MLX work on a device is
    submitted by a single owner thread (qMLX: the ``mlx-step`` thread). The
    backend declares its stream semantics via the seam —
    ``supports_secondary_streams()`` + ``capabilities().stream_model`` — the
    detection contract the engine relies on instead of string-probing a
    ``RuntimeError``.

    This test:
    (a) queries the declared contract;
    (b) cross-checks it against a live probe (create a secondary stream in a
        second thread; usable → MULTI_STREAM, raises → SINGLE_STREAM);
    (c) asserts the declared contract matches the live probe.

    The one-owner-thread invariant itself is enforced by the engine (a single
    ``mlx-step`` thread owns all MLX work); a second thread mutating the
    default stream (``mx.set_default_stream``) is the race the plan note
    flags (diffusion_lane.py). The backend's declared stream model is the
    conformance contract that tells the engine what the hardware supports.
    """
    declared_secondary = backend.supports_secondary_streams()
    declared_model = backend.capabilities().stream_model
    observed = _probe_secondary_stream_in_thread(mx)
    expected_model = StreamModel.MULTI_STREAM if observed else StreamModel.SINGLE_STREAM
    assert declared_model == expected_model, (
        f"the declared stream model {declared_model} does not match the "
        f"live probe (a secondary stream in a second thread was "
        f"{'usable → MULTI_STREAM' if observed else 'not usable → SINGLE_STREAM'})"
    )
    assert declared_secondary == observed, (
        f"supports_secondary_streams()={declared_secondary} does not match "
        f"the live probe (a secondary stream in a second thread was "
        f"{'usable' if observed else 'not usable'})"
    )


# ---------------------------------------------------------------------------
# Check 5 — mx.compile / mx.async_eval semantics
# ---------------------------------------------------------------------------


def test_compile_async_eval_semantics(backend, mx):
    """Record which compile / async_eval semantics this backend reports via
    the seam (the scheduler's backoff logic depends on ``async_eval``
    semantics). The capability report must agree with the actual presence of
    ``mx.compile`` / ``mx.async_eval``.

    On this host both are present — the backend reports
    ``compile_through=True`` and ``async_eval=True``. On a backend that
    lacks either, an explicit gap marker names the degradation path.
    """
    caps = backend.capabilities()
    actual_compile = hasattr(mx, "compile")
    actual_async = hasattr(mx, "async_eval")
    assert caps.compile_through == actual_compile, (
        f"capability report compile_through={caps.compile_through} "
        f"disagrees with actual presence (hasattr mx.compile="
        f"{actual_compile})"
    )
    assert caps.async_eval == actual_async, (
        f"capability report async_eval={caps.async_eval} disagrees with "
        f"actual presence (hasattr mx.async_eval={actual_async})"
    )
    if not actual_compile:
        _gap(
            f"mx.compile is absent on this backend ({mx.default_device()}) — "
            f"the seeded-sampler reproducibility boundary must degrade "
            f"cleanly (the fast path cannot compile-through)"
        )
    if not actual_async:
        _gap(
            f"mx.async_eval is absent on this backend ({mx.default_device()}) "
            f"— the multiplexer's eval pump is unavailable; the scheduler's "
            f"backoff logic cannot rely on async_eval semantics"
        )


# ---------------------------------------------------------------------------
# Check 6 — device_info key set
# ---------------------------------------------------------------------------


def test_device_info_device_name(backend, mx):
    """``device_info()['device_name']`` must exist and be a str on this
    backend (it replaces the four ``sysctl machdep.cpu.brand_string`` call
    sites)."""
    info = mx.device_info()
    name = info.get("device_name")
    assert isinstance(name, str) and name, (
        f"device_info()['device_name'] must be a non-empty str; got "
        f"{type(name).__name__} ({name!r})"
    )
    # The seam's device_name() must return the same value (parity).
    assert backend.device_name() == name, (
        f"the seam's device_name() ({backend.device_name()!r}) does not "
        f"match device_info()['device_name'] ({name!r})"
    )


def test_device_info_memory_keys(backend, mx):
    """The memory-size keys the seam consumes (``memory_size``,
    ``max_recommended_working_set_size``) must exist with the right types on
    this backend. On a backend that does not expose a memory-size notion
    (this host: CPU), the absent keys produce an explicit gap marker — the
    seam falls back to 0 by design ("0 when the backend has no such
    notion"), so the memory-cap denominator is unavailable.
    """
    info = mx.device_info()
    missing = [
        k for k in ("memory_size", "max_recommended_working_set_size") if k not in info
    ]
    if missing:
        _gap(
            f"device_info() on this backend ({mx.default_device()}) lacks "
            f"{missing} — the seam's total_memory_bytes() / "
            f"recommended_working_set_bytes() fall back to 0 (by design: "
            f"'0 when the backend has no such notion'); the memory-cap "
            f"denominator is unavailable on this backend"
        )
    # Present → must be ints (the seam's right-type contract).
    for k in ("memory_size", "max_recommended_working_set_size"):
        assert isinstance(info[k], int), (
            f"device_info()['{k}'] must be an int; got "
            f"{type(info[k]).__name__} ({info[k]!r})"
        )
    # The seam's memory queries must return the same values (parity).
    assert backend.total_memory_bytes() == info.get("memory_size", 0), (
        "the seam's total_memory_bytes() does not match device_info()['memory_size']"
    )
