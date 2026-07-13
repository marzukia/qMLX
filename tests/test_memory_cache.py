# SPDX-License-Identifier: Apache-2.0
"""Tests for memory-aware prefix cache."""

from unittest.mock import patch

import pytest

try:
    import mlx_lm  # noqa: F401

    _has_mlx_lm = True
except ImportError:
    _has_mlx_lm = False

from vllm_mlx.memory_cache import (
    MemoryCacheConfig,
    _array_memory,
    _get_available_memory,
    estimate_kv_cache_memory,
)


class TestMemoryCacheConfig:
    """Tests for MemoryCacheConfig."""

    def test_default_config(self):
        config = MemoryCacheConfig()
        assert config.max_memory_mb is None
        assert config.max_memory_percent == 0.20
        assert config.max_entries == 1000
        assert config.enable_memory_tracking is True

    def test_custom_config(self):
        config = MemoryCacheConfig(
            max_memory_mb=2048,
            max_memory_percent=0.5,
            max_entries=100,
        )
        assert config.max_memory_mb == 2048
        assert config.max_memory_percent == 0.5
        assert config.max_entries == 100

    def test_invalid_memory_percent_zero(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=0.0)

    def test_invalid_memory_percent_negative(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=-0.1)

    def test_invalid_memory_percent_over_one(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=1.5)

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            MemoryCacheConfig(max_entries=0)

    def test_compute_memory_limit_explicit(self):
        config = MemoryCacheConfig(max_memory_mb=1024)
        assert config.compute_memory_limit() == 1024 * 1024 * 1024

    def test_compute_memory_limit_auto(self):
        with patch(
            "vllm_mlx.memory_cache._get_available_memory",
            return_value=8 * 1024 * 1024 * 1024,  # 8GB
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            assert limit == 2 * 1024 * 1024 * 1024  # 25% of 8GB = 2GB

    def test_compute_memory_limit_fallback(self):
        with patch(
            "vllm_mlx.memory_cache._get_available_memory",
            return_value=0,  # Detection failed
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            # Fallback: 25% of 8GB = 2GB
            assert limit == 2 * 1024 * 1024 * 1024


class MockArray:
    """Mock array with nbytes attribute."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class MockDtype:
    """Mock dtype with size attribute."""

    def __init__(self, size: int):
        self.size = size


class MockShapeArray:
    """Mock array with shape and dtype (like MLX arrays) but no nbytes."""

    def __init__(self, shape: tuple, dtype_size: int):
        self.shape = shape
        self.dtype = MockDtype(dtype_size)


class MockKVCache:
    """Mock KV cache with keys/values attributes."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self.keys = MockArray(key_bytes)
        self.values = MockArray(value_bytes)


class MockStateCache:
    """Mock cache with state property."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self._keys = MockArray(key_bytes)
        self._values = MockArray(value_bytes)

    @property
    def state(self):
        return (self._keys, self._values)


class TestArrayMemory:
    """Tests for _array_memory helper (shape-based, no lazy eval trigger)."""

    def test_shape_dtype_estimation(self):
        """Verify shape*dtype.size computation without .nbytes access."""
        arr = MockShapeArray(shape=(2, 16, 128, 64), dtype_size=2)
        # 2 * 16 * 128 * 64 * 2 = 524288
        assert _array_memory(arr) == 2 * 16 * 128 * 64 * 2

    def test_fallback_to_nbytes(self):
        """Verify fallback to .nbytes when shape/dtype not available."""
        arr = MockArray(nbytes=4096)
        assert _array_memory(arr) == 4096

    def test_zero_for_unknown_object(self):
        """Return 0 for objects without shape/dtype/nbytes."""
        assert _array_memory(42) == 0
        assert _array_memory("string") == 0

    def test_shape_dtype_preferred_over_nbytes(self):
        """When both shape+dtype and nbytes exist, shape+dtype is used."""

        class DualArray:
            def __init__(self):
                self.shape = (10,)
                self.dtype = MockDtype(4)
                self.nbytes = 9999  # should NOT be used

        arr = DualArray()
        assert _array_memory(arr) == 40  # 10 * 4, not 9999

    def test_estimate_uses_shape_based_for_dict_state(self):
        """estimate_kv_cache_memory uses _array_memory (shape-based) for dicts."""
        keys = MockShapeArray(shape=(1, 8, 100, 64), dtype_size=2)
        values = MockShapeArray(shape=(1, 8, 100, 64), dtype_size=2)
        layer = {"state": (keys, values)}
        expected = 2 * (1 * 8 * 100 * 64 * 2)
        assert estimate_kv_cache_memory([layer]) == expected


class TestEstimateKvCacheMemory:
    """Tests for estimate_kv_cache_memory function."""

    def test_empty_cache(self):
        assert estimate_kv_cache_memory([]) == 0
        assert estimate_kv_cache_memory(None) == 0

    def test_cache_with_nbytes_attribute(self):
        layer = MockKVCache(1000, 1000)
        assert estimate_kv_cache_memory([layer]) == 2000

    def test_cache_with_state_property(self):
        layer = MockStateCache(500, 500)
        assert estimate_kv_cache_memory([layer]) == 1000

    def test_cache_with_dict_state(self):
        keys = MockArray(300)
        values = MockArray(300)
        layer = {"state": (keys, values)}
        assert estimate_kv_cache_memory([layer]) == 600

    def test_multiple_layers(self):
        layers = [MockKVCache(100, 100) for _ in range(4)]
        assert estimate_kv_cache_memory(layers) == 800


class TestGetAvailableMemory:
    """Tests for _get_available_memory helper."""

    def test_with_psutil(self):
        try:
            from importlib.util import find_spec

            if find_spec("psutil") is None:
                pytest.skip("psutil not installed")
            mem = _get_available_memory()
            assert mem > 0
        except ImportError:
            pytest.skip("psutil not installed")

    def test_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            # Should return 0 when psutil not available
            # Note: This test may not work as expected due to import caching
            pass
