# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the backend conformance suite (repositioning item 7 of 9).

The report assembly (``ConformanceReport``, documented gaps, ``to_dict``
/ ``format_table``) is pure Python — testable without mlx. The device
probes are exercised against a fake ``mlx.core`` (``FakeMx``) and a
fake ``Backend`` (``FakeBackend``) so the counterfactuals (missing
quantizer, broken round-trip, declaration mismatches) are pinned
without a real device. A fake ``mlx_lm`` is injected via ``sys.modules``
for the prompt-cache format probes.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import types
from typing import Any

import pytest

from vllm_mlx.device.backend import (
    BackendKind,
    CapabilityFlags,
    EnforcementMode,
    MemoryStats,
    StreamModel,
    reset_registry,
)
from vllm_mlx.device.conformance import (
    STATUS_CAPABILITY_MISSING,
    STATUS_FAIL,
    STATUS_PASS,
    CheckResult,
    ConformanceReport,
    documented_gap_names,
    run_conformance,
)
from vllm_mlx.runtime.disk_kv_checkpoint import model_identity_matches

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _infer_shape(v: Any) -> tuple:
    if isinstance(v, (list, tuple)):
        if not v:
            return (len(v),)
        return (len(v),) + _infer_shape(v[0])
    return ()


def _zip_op(a: list, b: Any, fn) -> list:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [_zip_op(x, y, fn) for x, y in zip(a, b)]
    if isinstance(a, (list, tuple)):
        return [_zip_op(x, b, fn) for x in a]
    if isinstance(b, (list, tuple)):
        return [_zip_op(a, x, fn) for x in b]
    return fn(a, b)


class _FakeArray:
    """A minimal ``mlx.array`` stand-in (nested-list backed)."""

    def __init__(self, values: Any, shape: tuple | None = None) -> None:
        self._values = values
        self._shape = tuple(shape) if shape is not None else _infer_shape(values)
        self._source: Any = None

    @property
    def shape(self) -> tuple:
        return self._shape

    def tolist(self) -> Any:
        return self._values

    def flatten(self) -> _FakeArray:
        def flat(v):
            if isinstance(v, (list, tuple)):
                for x in v:
                    yield from flat(x)
            else:
                yield v

        return _FakeArray(list(flat(self._values)), self._shape)

    def item(self) -> float:
        v = self._values
        while isinstance(v, (list, tuple)):
            v = v[0]
        return v

    def __float__(self) -> float:
        return float(self.item())

    def reshape(self, *shape: int) -> _FakeArray:
        def _chunk(vals, dims):
            if len(dims) == 1:
                return list(vals)  # dims[0] chunks of size prod([]) = 1
            size = 1
            for d in dims[1:]:
                size *= d
            return [vals[i : i + size] for i in range(0, len(vals), size)]

        def nest(vals, dims):
            if not dims:
                return vals
            return [nest(c, dims[1:]) for c in _chunk(vals, dims)]

        flat = list(self.flatten().tolist())
        return _FakeArray(nest(flat, list(shape)), tuple(shape))

    def astype(self, dtype: Any) -> _FakeArray:
        return self

    def abs(self) -> _FakeArray:
        return _FakeArray(_zip_op(self._values, 0, lambda a, _b: abs(a)), self._shape)

    def max(self) -> _FakeArray:
        def _max(v):
            if isinstance(v, (list, tuple)):
                return max(_max(x) for x in v)
            return v

        return _FakeArray(_max(self._values))

    def __add__(self, other: Any) -> _FakeArray:
        o = other._values if isinstance(other, _FakeArray) else other
        return _FakeArray(_zip_op(self._values, o, lambda a, b: a + b), self._shape)

    def __radd__(self, other: Any) -> _FakeArray:
        return self.__add__(other)

    def __sub__(self, other: Any) -> _FakeArray:
        o = other._values if isinstance(other, _FakeArray) else other
        return _FakeArray(_zip_op(self._values, o, lambda a, b: a - b), self._shape)

    def __mul__(self, other: Any) -> _FakeArray:
        o = other._values if isinstance(other, _FakeArray) else other
        return _FakeArray(_zip_op(self._values, o, lambda a, b: a * b), self._shape)

    def __eq__(self, other: Any) -> _FakeArray:  # noqa: D105
        return _FakeArray(
            _zip_op(self._values, other, lambda a, b: a == b), self._shape
        )


class FakeMx:
    """A minimal ``mlx.core`` stand-in for the conformance probes."""

    __version__ = "0.32.4-fake"
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __init__(
        self,
        device: str = "cpu",
        device_info: dict | None = None,
        with_quantizer: bool = True,
        broken_dequantize: bool = False,
        set_memory_limit_raises: bool = False,
        allocation_fails: bool = False,
        with_compile: bool = True,
        compile_raises: bool = False,
        with_async_eval: bool = True,
        multi_stream: bool = True,
        stream_error: Exception | None = None,
        set_default_stream_raises: bool = False,
        fast_kernels: bool = True,
    ) -> None:
        self._device = device
        self._device_info = (
            device_info if device_info is not None else {"device_name": "FakeDevice"}
        )
        self._set_memory_limit_raises = set_memory_limit_raises
        self._allocation_fails = allocation_fails
        self._multi_stream = multi_stream
        self._stream_error = stream_error
        self._set_default_stream_raises = set_default_stream_raises
        self._compile_raises = compile_raises
        self._memory_limit = None
        self._cache_limit = None
        if with_quantizer:
            self.quantize = self._quantize
            self.dequantize = (
                self._broken_dequantize if broken_dequantize else self._dequantize
            )
        if with_compile:
            self.compile = self._compile
        if with_async_eval:
            self.async_eval = lambda *a, **k: None
        if fast_kernels:
            fast = type(
                "_Fast",
                (),
                {
                    "rms_norm": staticmethod(lambda *a, **k: None),
                    "scaled_dot_product_attention": staticmethod(lambda *a, **k: None),
                },
            )
            self.fast = fast()

    # -- quantizer -----------------------------------------------------

    def _quantize(self, x, group_size=64, bits=8):
        n = 1
        for s in x.shape:
            n *= s
        packed = _FakeArray([0.0] * n, x.shape)
        packed._source = x
        return [packed, _FakeArray([0.0]), _FakeArray([0.0])]

    def _dequantize(self, *packed, group_size=64, bits=8):
        return packed[0]._source  # identity: deviation 0

    def _broken_dequantize(self, *packed, group_size=64, bits=8):
        return _FakeArray([float("nan")], (1,))  # wrong shape + non-finite

    # -- memory --------------------------------------------------------

    def set_memory_limit(self, n: int) -> None:
        if self._set_memory_limit_raises:
            raise RuntimeError("set_memory_limit not supported on this device")
        self._memory_limit = n

    def set_cache_limit(self, n: int) -> None:
        self._cache_limit = n

    def ones(self, shape: tuple, dtype: Any = None) -> _FakeArray:
        n = 1
        for s in shape:
            n *= s
        if self._allocation_fails and n > 10**6:
            raise MemoryError("allocation failed past the memory cap")
        return _FakeArray([1.0] * n, tuple(shape))

    # -- device / streams ----------------------------------------------

    def default_device(self):
        return self._device

    def device_info(self) -> dict:
        return dict(self._device_info)

    def new_thread_local_stream(self, device):
        if not self._multi_stream:
            raise self._stream_error or RuntimeError(
                "There is no Stream(gpu, 1) in current thread."
            )
        return ("thread_local_stream", device)

    def default_stream(self, device):
        return ("default_stream", device)

    def set_default_stream(self, stream) -> None:
        if self._set_default_stream_raises:
            raise RuntimeError("no Stream in current thread")

    def stream(self, stream):
        return contextlib.nullcontext()

    # -- arrays / eval --------------------------------------------------

    def array(self, values: Any) -> _FakeArray:
        return _FakeArray(values)

    def arange(self, n: int) -> _FakeArray:
        return _FakeArray(list(range(n)))

    def eval(self, *a, **k) -> None:
        pass

    def all(self, arr: _FakeArray) -> bool:
        def _all(v):
            if isinstance(v, (list, tuple)):
                return all(_all(x) for x in v)
            return bool(v)

        return _all(arr.tolist())

    def isfinite(self, arr: _FakeArray) -> _FakeArray:
        def _fin(v):
            if isinstance(v, (list, tuple)):
                return [_fin(x) for x in v]
            return math.isfinite(v)

        return _FakeArray(_fin(arr.tolist()), arr.shape)

    # -- compile -------------------------------------------------------

    def _compile(self, f):
        if self._compile_raises:
            raise RuntimeError("compile not supported on this device")
        return f

    # -- safetensors ----------------------------------------------------

    def save_safetensors(self, path, tensors, metadata) -> None:
        header: dict = {}
        offset = 0
        for k, v in tensors.items():
            header[k] = {"dtype": "BF16", "shape": list(v.shape), "offset": offset}
            offset += 16
        if metadata:
            header["__metadata__"] = dict(metadata)
        h = json.dumps(header).encode()
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(h)))
            fh.write(h)
            fh.write(b"\x00" * offset)


class FakeBackend:
    """A fake ``Backend`` seam with declaration toggles."""

    def __init__(
        self,
        mx: FakeMx,
        kind: BackendKind = BackendKind.METAL,
        mode: EnforcementMode = EnforcementMode.HINT,
        multi_stream: bool = True,
        caps: CapabilityFlags | None = None,
        available: bool = True,
    ) -> None:
        self._mx = mx
        self.kind = kind
        self.enforcement_mode = mode
        self._multi_stream = multi_stream
        self._caps = caps
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def capabilities(self) -> CapabilityFlags:
        if self._caps is not None:
            return self._caps
        return CapabilityFlags(
            fast_kernels=hasattr(self._mx, "fast"),
            flash_attention=hasattr(self._mx, "fast")
            and hasattr(self._mx.fast, "scaled_dot_product_attention"),
            quantized_kv=hasattr(self._mx, "quantize")
            and hasattr(self._mx, "dequantize"),
            compile_through=hasattr(self._mx, "compile"),
            async_eval=hasattr(self._mx, "async_eval"),
            stream_model=(
                StreamModel.MULTI_STREAM
                if self._multi_stream
                else StreamModel.SINGLE_STREAM
            ),
        )

    def supports_secondary_streams(self) -> bool:
        return self._multi_stream

    def device(self):
        return self._mx.default_device()

    def device_name(self) -> str:
        return self._mx.device_info().get("device_name", "")

    def total_memory_bytes(self) -> int:
        return self._mx.device_info().get("memory_size", 0)

    def recommended_working_set_bytes(self) -> int:
        return self._mx.device_info().get(
            "max_recommended_working_set_size",
            self._mx.device_info().get("memory_size", 0),
        )

    def set_memory_limit(self, n: int) -> None:
        self._mx.set_memory_limit(n)

    def set_cache_limit(self, n: int) -> None:
        self._mx.set_cache_limit(n)

    def memory_stats(self) -> MemoryStats:
        return MemoryStats(1, 2, 3)

    def default_stream(self):
        return self._mx.default_stream(self._mx.default_device())


def _make_fake_mlx_lm(monkeypatch, mx: FakeMx) -> None:
    """Inject a fake ``mlx_lm`` (the prompt-cache format API) into
    ``sys.modules``."""
    mlx_lm_mod = types.ModuleType("mlx_lm")
    models_mod = types.ModuleType("mlx_lm.models")
    cache_mod = types.ModuleType("mlx_lm.models.cache")

    class KVCache:
        def __init__(self) -> None:
            self.keys: list = []
            self.values: list = []
            self.offset = 0

    class QuantizedKVCache(KVCache):
        def __init__(self) -> None:
            super().__init__()
            self.group_size = 64
            self.bits = 8

    store: dict[str, list] = {}

    def save_prompt_cache(file_name, cache_list, metadata=None) -> None:
        tensors: dict = {}
        meta: dict = {}
        for i, c in enumerate(cache_list):
            tensors[f"{i}.0.0"] = c.keys[0] if c.keys else mx.array([])
            tensors[f"{i}.1.0"] = c.values[0] if c.values else mx.array([])
            tensors[f"{i}.2"] = mx.array(c.offset)
            meta[f"1.{i}"] = type(c).__name__
        meta["2.0"] = ""
        meta["2.1.0"] = "0.2"
        meta["2.1.1"] = "scalar"
        mx.save_safetensors(file_name, tensors, meta)
        store[file_name] = list(cache_list)

    def load_prompt_cache(file_name, return_metadata=False):
        # Fail-closed: read the raw header, take the class name, and
        # refuse an unknown class (mirrors load_prompt_cache's
        # globals()[c] KeyError).
        with open(file_name, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen))
        meta = header.get("__metadata__") or {}
        classes = [meta[k] for k in sorted(meta) if k.startswith("1.")]
        for c in classes:
            if c not in ("KVCache", "QuantizedKVCache"):
                raise KeyError(c)
        objs = store.get(file_name)
        if objs is None:
            objs = [
                KVCache() if c == "KVCache" else QuantizedKVCache() for c in classes
            ]
        if return_metadata:
            return objs, {}
        return objs

    cache_mod.KVCache = KVCache
    cache_mod.QuantizedKVCache = QuantizedKVCache
    cache_mod.save_prompt_cache = save_prompt_cache
    cache_mod.load_prompt_cache = load_prompt_cache
    models_mod.cache = cache_mod
    mlx_lm_mod.models = models_mod
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)


def _status(report: ConformanceReport, name: str) -> str | None:
    for c in report.checks:
        if c.name == name:
            return c.status
    return None


def _detail(report: ConformanceReport, name: str) -> str | None:
    for c in report.checks:
        if c.name == name:
            return c.detail
    return None


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


# ---------------------------------------------------------------------------
# Report assembly (pure — no mlx needed)
# ---------------------------------------------------------------------------


def test_report_acceptance_rules():
    env = {"probes_ran": True}
    # All pass → accepted.
    r = ConformanceReport(checks=[CheckResult("a", STATUS_PASS, "ok")], environment=env)
    assert r.accepted
    # One fail → not accepted.
    r = ConformanceReport(
        checks=[CheckResult("a", STATUS_FAIL, "broken")], environment=env
    )
    assert not r.accepted
    # capability-missing + documented → accepted.
    r = ConformanceReport(
        checks=[CheckResult("a", STATUS_CAPABILITY_MISSING, "gap")],
        environment=env,
        documented_gaps=frozenset({"a"}),
    )
    assert r.accepted
    # capability-missing + undocumented → not accepted.
    r = ConformanceReport(
        checks=[CheckResult("a", STATUS_CAPABILITY_MISSING, "gap")],
        environment=env,
    )
    assert not r.accepted
    # probes did not run → never accepted (metadata-only is not a
    # verdict).
    r = ConformanceReport(
        checks=[CheckResult("a", STATUS_PASS, "ok")],
        environment={"probes_ran": False},
    )
    assert not r.accepted


def test_report_to_dict_structure():
    r = ConformanceReport(
        checks=[
            CheckResult("a", STATUS_PASS, "ok"),
            CheckResult("b", STATUS_CAPABILITY_MISSING, "gap"),
        ],
        environment={"probes_ran": True, "python": "3.12"},
        documented_gaps=frozenset({"b"}),
    )
    d = r.to_dict()
    assert d["accepted"] is True
    assert d["fails"] == 0
    assert d["capability_missing"] == 1
    assert d["undocumented_gaps"] == 0
    assert d["documented_gaps"] == ["b"]
    assert d["environment"]["python"] == "3.12"
    assert [c["name"] for c in d["checks"]] == ["a", "b"]
    assert d["checks"][1]["status"] == "capability-missing"
    # JSON-serializable.
    json.dumps(d)


def test_report_format_table():
    r = ConformanceReport(
        checks=[
            CheckResult("a", STATUS_PASS, "ok"),
            CheckResult("b", STATUS_FAIL, "broken"),
        ],
        environment={
            "probes_ran": True,
            "python": "3.12",
            "mlx": "0.32.4",
            "device": "Device(cpu, 0)",
            "device_class": "cpu",
        },
    )
    t = r.format_table()
    assert "qMLX backend conformance report" in t
    assert "a" in t and "b" in t
    assert "pass" in t and "fail" in t
    assert "NOT ACCEPTED" in t
    assert "1 fail(s)" in t


def test_documented_gap_registry():
    # CPU profile: the metal-specific absences are documented gaps.
    cpu = documented_gap_names("cpu")
    assert "memory_cap_enforcement" in cpu
    assert "stream_ownership" in cpu
    assert "device_info.memory_size" in cpu
    assert "device_info.max_recommended_working_set_size" in cpu
    # GPU profile: nothing documented by default — the port documents its
    # own gaps via the gap register.
    gpu = documented_gap_names("gpu")
    assert gpu == set()
    # Gap register adds names.
    gpu2 = documented_gap_names("gpu", ["stream_ownership"])
    assert "stream_ownership" in gpu2
    # No mlx: every check is documented (the whole suite is skipped).
    none = documented_gap_names(None)
    assert "environment.mlx" in none
    assert "prompt_cache_format" in none


# ---------------------------------------------------------------------------
# Metadata-only mode (no mlx)
# ---------------------------------------------------------------------------


def test_metadata_only_report():
    r = run_conformance(mx=None, backend=None)
    assert r.environment["probes_ran"] is False
    assert not r.accepted
    # One report line: the skipped probes.
    assert len(r.checks) == 1
    assert r.checks[0].name == "environment.mlx"
    assert r.checks[0].status == STATUS_CAPABILITY_MISSING
    # Every capability-missing line is documented (the no-mlx
    # environment documents the whole suite).
    assert r.undocumented_gaps == []
    # The table renders (metadata-only verdict).
    assert "NOT ACCEPTED" in r.format_table()


# ---------------------------------------------------------------------------
# The model identity guard (pure logic)
# ---------------------------------------------------------------------------


def test_model_identity_guard_fail_closed():
    assert model_identity_matches("A", "A") is True
    assert model_identity_matches("A", "B") is False
    assert model_identity_matches(None, "A") is False
    assert model_identity_matches("A", None) is False
    assert model_identity_matches(None, None) is False
    assert model_identity_matches("", "A") is False
    assert model_identity_matches("A", "") is False
    # str() comparison (the scheduler's original semantics).
    assert model_identity_matches(5, "5") is True


# ---------------------------------------------------------------------------
# The probes against FakeMx / FakeBackend
# ---------------------------------------------------------------------------


def test_full_fake_mlx_accepted(monkeypatch):
    mx = FakeMx(
        device_info={
            "device_name": "FakeDevice",
            "memory_size": 32 * 2**30,
            "max_recommended_working_set_size": 16 * 2**30,
        }
    )
    _make_fake_mlx_lm(monkeypatch, mx)
    backend = FakeBackend(mx)
    r = run_conformance(mx=mx, backend=backend)
    assert r.environment["probes_ran"] is True
    # No fails.
    assert r.fails == [], [c.detail for c in r.fails]
    assert r.accepted, [f"{c.name}: {c.status} — {c.detail}" for c in r.checks]
    # The key lines.
    assert _status(r, "seam_contract") == STATUS_PASS
    assert _status(r, "memory_cap_enforcement") == STATUS_PASS
    assert _status(r, "quantized_kv_roundtrip") == STATUS_PASS
    assert _status(r, "prompt_cache_format") == STATUS_PASS
    assert _status(r, "stream_ownership") == STATUS_PASS
    assert _status(r, "compile_through") == STATUS_PASS
    assert _status(r, "async_eval") == STATUS_PASS
    assert _status(r, "device_info.memory_size") == STATUS_PASS


def test_missing_quantizer_is_loud(monkeypatch):
    mx = FakeMx(with_quantizer=False)
    _make_fake_mlx_lm(monkeypatch, mx)
    r = run_conformance(mx=mx, backend=FakeBackend(mx))
    assert _status(r, "quantized_kv_roundtrip") == STATUS_CAPABILITY_MISSING
    d = _detail(r, "quantized_kv_roundtrip")
    assert "mx.quantize" in d
    assert "mx.dequantize" in d
    # LOUD: the detail names the danger (garbage round-trip, the loader
    # cannot tell).
    assert "garbage" in d
    # Not a fail, not a pass.
    assert _status(r, "quantized_kv_roundtrip") != STATUS_FAIL


def test_broken_quantize_roundtrip_fails(monkeypatch):
    mx = FakeMx(broken_dequantize=True)
    _make_fake_mlx_lm(monkeypatch, mx)
    r = run_conformance(mx=mx, backend=FakeBackend(mx))
    assert _status(r, "quantized_kv_roundtrip") == STATUS_FAIL
    d = _detail(r, "quantized_kv_roundtrip")
    assert "restore bug" in d
    # The broken dequantize returns the wrong shape (and non-finite).
    assert "shape" in d or "finite" in d


def test_enforcement_mode_mismatch_fails():
    # Declared HARD, but the allocator grows past the cap (hint
    # behaviour) → fail.
    mx = FakeMx()
    r = run_conformance(
        mx=mx,
        backend=FakeBackend(mx, mode=EnforcementMode.HARD),
    )
    assert _status(r, "memory_cap_enforcement") == STATUS_FAIL
    assert "declares enforcement_mode=hard" in _detail(r, "memory_cap_enforcement")
    # Declared HARD + the allocator enforces (allocation fails past the
    # cap) → pass.
    mx_hard = FakeMx(allocation_fails=True)
    r2 = run_conformance(
        mx=mx_hard,
        backend=FakeBackend(mx_hard, mode=EnforcementMode.HARD),
    )
    assert _status(r2, "memory_cap_enforcement") == STATUS_PASS


def test_stream_declaration_mismatch_fails():
    # Declares multi-stream, but the probe raises the single-stream
    # error → fail (the seam report is wrong).
    mx = FakeMx(multi_stream=False)
    r = run_conformance(
        mx=mx,
        backend=FakeBackend(mx, multi_stream=True),
    )
    assert _status(r, "stream_ownership") == STATUS_FAIL
    assert "supports_secondary_streams=True" in _detail(r, "stream_ownership")
    # Declares single-stream + the probe raises the single-stream error
    # → pass.
    mx_single = FakeMx(multi_stream=False)
    r2 = run_conformance(
        mx=mx_single,
        backend=FakeBackend(mx_single, multi_stream=False),
    )
    assert _status(r2, "stream_ownership") == STATUS_PASS


def test_compile_absent_is_capability_missing():
    mx = FakeMx(with_compile=False)
    r = run_conformance(mx=mx, backend=FakeBackend(mx))
    assert _status(r, "compile_through") == STATUS_CAPABILITY_MISSING
    assert "mx.compile" in _detail(r, "compile_through")
    # The seeded-sampler degradation is named.
    assert "seeded-sampler" in _detail(r, "compile_through")


def test_async_eval_absent_is_capability_missing():
    mx = FakeMx(with_async_eval=False)
    r = run_conformance(mx=mx, backend=FakeBackend(mx))
    assert _status(r, "async_eval") == STATUS_CAPABILITY_MISSING
    assert "mx.async_eval" in _detail(r, "async_eval")


def test_device_info_wrong_type_fails():
    mx = FakeMx(
        device_info={
            "device_name": "FakeDevice",
            "memory_size": "32 GiB",  # str, not int
            "max_recommended_working_set_size": 16 * 2**30,
        }
    )
    r = run_conformance(mx=mx, backend=FakeBackend(mx))
    assert _status(r, "device_info.memory_size") == STATUS_FAIL
    assert "str" in _detail(r, "device_info.memory_size")
    # The well-typed key still passes.
    assert _status(r, "device_info.max_recommended_working_set_size") == STATUS_PASS


def test_device_info_seam_parity_fails():
    # The backend accessor disagrees with the raw device_info key →
    # fail (the seam report is wrong).
    mx = FakeMx(
        device_info={
            "device_name": "FakeDevice",
            "memory_size": 32 * 2**30,
            "max_recommended_working_set_size": 16 * 2**30,
        }
    )
    backend = FakeBackend(mx)
    backend.total_memory_bytes = lambda: 64 * 2**30  # noqa: B023
    r = run_conformance(mx=mx, backend=backend)
    assert _status(r, "device_info.memory_size") == STATUS_FAIL
    assert "seam report is wrong" in _detail(r, "device_info.memory_size")


def test_prompt_cache_format_unknown_class_fails_closed(monkeypatch):
    # The fake load_prompt_cache refuses an unknown class (KeyError) —
    # the probe must observe that as fail-closed (pass). A load that
    # SUCCEEDS on an unknown class would be a fail; here the fake
    # behaves like the real loader, so the check passes.
    mx = FakeMx()
    _make_fake_mlx_lm(monkeypatch, mx)
    r = run_conformance(mx=mx, backend=FakeBackend(mx))
    assert _status(r, "prompt_cache_format") == STATUS_PASS
    d = _detail(r, "prompt_cache_format")
    assert "fails closed" in d
    assert "model identity guard" in d


def test_no_backend_resolves_is_fine(monkeypatch):
    # No backend on this device (CPU profile) → the seam contract
    # passes with a note; the raw probes run directly.
    mx = FakeMx()
    _make_fake_mlx_lm(monkeypatch, mx)
    r = run_conformance(mx=mx, backend=None)
    assert _status(r, "seam_contract") == STATUS_PASS
    assert "no backend resolved" in _detail(r, "seam_contract")
    # The suite still runs (raw API probes).
    assert _status(r, "quantized_kv_roundtrip") == STATUS_PASS


def _real_mx():
    """The real ``mlx.core`` (skip on no-mlx hosts)."""
    try:
        import mlx.core as mx
    except ImportError as e:
        pytest.skip(f"mlx is not usable on this host ({e})")
    return mx


# ---------------------------------------------------------------------------
# The disk-KV restore guards (real vllm_mlx guard code)
# ---------------------------------------------------------------------------


def test_quantized_kv_loader_guard_fail_closed():
    """The fail-closed ``_cache_classes_compatible`` guard: a QuantizedKVCache
    checkpoint is rejected when ``kv_quantize`` is off; a plain KVCache is
    always accepted. Backend-agnostic: the guard needs no mlx, so this runs
    unskipped on no-mlx hosts."""
    from vllm_mlx.memory_cache import (
        MemoryCacheConfig,
        _cache_classes_compatible,
    )

    cfg = MemoryCacheConfig()  # kv_quantize defaults to False
    ok, reason = _cache_classes_compatible(["QuantizedKVCache"], cfg)
    assert ok is False, (
        "the loader must fail closed: a QuantizedKVCache checkpoint is not "
        f"loadable when kv_quantize is off, but the guard accepted it "
        f"(reason={reason!r})"
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


def test_restore_identity_guard_fail_closed():
    """The restore identity guard is keyed on model + cache class and fails
    closed by design (a mismatch rejects the restore; it never silently
    corrupts output). The model-identity half pins the real
    ``model_identity_matches`` (the full truth table is covered by
    test_model_identity_guard_fail_closed); the cache-class half pins the
    real ``_cache_classes_compatible`` guard. Backend-agnostic: no mlx
    needed, so this runs unskipped on no-mlx hosts."""
    from vllm_mlx.memory_cache import MemoryCacheConfig, _cache_classes_compatible

    # (a) MODEL IDENTITY, via the real predicate the scheduler's guard uses:
    # accept only an exact match; reject on a mismatch or an unknown side.
    assert model_identity_matches("qwen3-122b", "qwen3-122b") is True
    assert model_identity_matches("qwen3-122b", "llama-70b") is False
    assert model_identity_matches(None, "qwen3-122b") is False
    assert model_identity_matches("qwen3-122b", None) is False

    # (b) CACHE CLASS, the real guard code (fail closed by design).
    cfg = MemoryCacheConfig()  # kv_quantize defaults to False
    ok, _ = _cache_classes_compatible(["QuantizedKVCache"], cfg)
    assert ok is False, (
        "the cache-class guard must fail closed: a QuantizedKVCache "
        "checkpoint is rejected when kv_quantize is off"
    )
    ok_kv, _ = _cache_classes_compatible(["KVCache"], cfg)
    assert ok_kv is True, "the cache-class guard must accept a plain KVCache"


def test_quantized_kv_loader_detection():
    """End-to-end ``_safetensors_cache_classes`` class-name detection: the
    reader must tell a saved QuantizedKVCache checkpoint is quantized
    (requires real mlx + mlx_lm; skips on no-mlx hosts)."""
    mx = _real_mx()
    from mlx_lm.models.cache import QuantizedKVCache, save_prompt_cache

    from vllm_mlx.memory_cache import _safetensors_cache_classes

    if not hasattr(mx, "quantize"):
        pytest.skip(
            f"mx.quantize is absent on this device ({mx.default_device()}); a "
            "QuantizedKVCache cannot be constructed"
        )
    qc = QuantizedKVCache()
    qc.group_size = 32
    qc.bits = 4
    qc.keys = mx.quantize(mx.ones((2, 4, 32)), group_size=32, bits=4)
    qc.values = mx.quantize(mx.ones((2, 4, 32)), group_size=32, bits=4)
    qc.offset = 8
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "qkv.safetensors")
        save_prompt_cache(p, [qc])
        detected = _safetensors_cache_classes(p)
    assert detected == ["QuantizedKVCache"], (
        f"the reader cannot detect a QuantizedKVCache checkpoint: "
        f"detected {detected!r} (expected ['QuantizedKVCache'])"
    )


def test_prompt_cache_on_disk_stores_class_name():
    """The on-disk safetensors format stores the cache class name so the
    loader can reconstruct the right class (the format invariant the
    class-identity round-trip depends on; requires real mlx + mlx_lm; skips
    on no-mlx hosts)."""
    mx = _real_mx()
    from mlx_lm.models.cache import KVCache, save_prompt_cache

    c = KVCache()
    c.keys = [mx.ones((2, 4, 32))]
    c.values = [mx.ones((2, 4, 32))]
    c.offset = 8
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "kv.safetensors")
        save_prompt_cache(p, [c])
        with open(p, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
    meta = header.get("__metadata__", {})
    assert any(v == "KVCache" for v in meta.values()), (
        f"the on-disk format must store the cache class name in the "
        f"metadata so the loader can reconstruct the class; metadata "
        f"= {meta!r} does not contain 'KVCache'"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_json_and_exit_code(tmp_path):
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.device.conformance", "--json", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode in (0, 1), proc.stderr
    d = json.loads(out.read_text())
    assert "environment" in d
    assert "checks" in d
    assert "accepted" in d
    assert "documented_gaps" in d
    # The table rendered on stdout.
    assert "qMLX backend conformance report" in proc.stdout


def test_cli_gap_register(tmp_path):
    gaps = tmp_path / "gaps.json"
    gaps.write_text(json.dumps(["stream_ownership"]))
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.device.conformance",
            "--gaps",
            str(gaps),
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode in (0, 1), proc.stderr
    d = json.loads(out.read_text())
    assert "stream_ownership" in d["documented_gaps"]


def test_cli_bad_gap_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.device.conformance",
            "--gaps",
            str(bad),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2
    assert "JSON list" in proc.stderr
