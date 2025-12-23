# SPDX-License-Identifier: Apache-2.0
"""
Tests for prefix cache functionality.

These tests verify the PrefixCacheManager for KV cache reuse
to speed up inference with repeated prompts.
"""

import pytest
from unittest.mock import MagicMock

from vllm_mlx.prefix_cache import (
    CacheEntry,
    PrefixCacheManager,
    PrefixCacheStats,
)


class TestPrefixCacheStats:
    """Tests for PrefixCacheStats class."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        stats = PrefixCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0
        assert stats.total_queries == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = PrefixCacheStats(hits=3, misses=7, total_queries=10)
        assert stats.hit_rate == 0.3

    def test_hit_rate_zero_queries(self):
        """Test hit rate with zero queries."""
        stats = PrefixCacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = PrefixCacheStats(hits=5, misses=5, tokens_saved=100, total_queries=10)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5
        assert d["tokens_saved"] == 100
        assert d["total_queries"] == 10


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        cache = ["mock_kv_cache"]
        entry = CacheEntry(prompt_cache=cache, count=1)
        assert entry.prompt_cache == ["mock_kv_cache"]
        assert entry.count == 1

    def test_cache_entry_count_increment(self):
        """Test incrementing reference count."""
        entry = CacheEntry(prompt_cache=["cache"], count=1)
        entry.count += 1
        assert entry.count == 2


class TestPrefixCacheManager:
    """Tests for PrefixCacheManager class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MagicMock()

    @pytest.fixture
    def cache_manager(self, mock_model):
        """Create a cache manager with default settings."""
        return PrefixCacheManager(mock_model, max_entries=10)

    def test_initialization(self, mock_model):
        """Test cache manager initialization."""
        manager = PrefixCacheManager(mock_model, max_entries=50)
        assert manager.max_size == 50
        assert manager.model_key == id(mock_model)
        assert len(manager) == 0

    def test_fetch_empty_cache(self, cache_manager):
        """Test fetching from empty cache returns miss."""
        tokens = [1, 2, 3, 4, 5]
        cache, remaining = cache_manager.fetch_cache(tokens)

        assert cache is None
        assert remaining == tokens
        assert cache_manager.stats.misses == 1
        assert cache_manager.stats.hits == 0

    def test_store_and_fetch_exact_match(self, cache_manager):
        """Test storing and fetching exact match."""
        tokens = [1, 2, 3, 4, 5]
        mock_cache = ["kv_layer_1", "kv_layer_2"]

        # Store cache
        cache_manager.store_cache(tokens, mock_cache)
        assert len(cache_manager) == 1

        # Fetch exact match
        cache, remaining = cache_manager.fetch_cache(tokens)

        assert cache is not None
        assert remaining == []
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == len(tokens)

    def test_fetch_shorter_prefix(self, cache_manager):
        """Test fetching when a shorter prefix is cached."""
        # Store short prefix
        short_tokens = [1, 2, 3]
        mock_cache = ["short_cache"]
        cache_manager.store_cache(short_tokens, mock_cache)

        # Fetch longer sequence
        long_tokens = [1, 2, 3, 4, 5, 6]
        cache, remaining = cache_manager.fetch_cache(long_tokens)

        assert cache is not None
        assert remaining == [4, 5, 6]
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == len(short_tokens)

    def test_lru_eviction(self, mock_model):
        """Test LRU eviction when cache is full."""
        manager = PrefixCacheManager(mock_model, max_entries=3)

        # Fill cache
        manager.store_cache([1], ["cache1"])
        manager.store_cache([2], ["cache2"])
        manager.store_cache([3], ["cache3"])
        assert len(manager) == 3

        # Add one more - should evict oldest
        manager.store_cache([4], ["cache4"])
        assert len(manager) == 3
        assert manager.stats.evictions == 1

        # Token [1] should be evicted
        cache, _ = manager.fetch_cache([1])
        assert cache is None

    def test_lru_touch_on_access(self, mock_model):
        """Test that accessing a cache updates LRU order."""
        manager = PrefixCacheManager(mock_model, max_entries=3)

        # Fill cache
        manager.store_cache([1], ["cache1"])
        manager.store_cache([2], ["cache2"])
        manager.store_cache([3], ["cache3"])

        # Access [1] to make it most recently used
        manager.fetch_cache([1])

        # Add new entry - should evict [2] (oldest untouched)
        manager.store_cache([4], ["cache4"])

        # [1] should still be there
        cache, _ = manager.fetch_cache([1])
        assert cache is not None

        # [2] should be evicted
        cache, _ = manager.fetch_cache([2])
        assert cache is None

    def test_store_empty_tokens(self, cache_manager):
        """Test that empty tokens are not stored."""
        cache_manager.store_cache([], ["empty_cache"])
        assert len(cache_manager) == 0

    def test_get_stats(self, cache_manager):
        """Test getting statistics."""
        # Generate some activity
        cache_manager.store_cache([1, 2, 3], ["cache1"])
        cache_manager.fetch_cache([1, 2, 3])  # Hit
        cache_manager.fetch_cache([4, 5, 6])  # Miss

        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["total_queries"] == 2

    def test_reset_stats(self, cache_manager):
        """Test resetting statistics."""
        cache_manager.stats.hits = 10
        cache_manager.stats.misses = 5
        cache_manager.reset_stats()

        assert cache_manager.stats.hits == 0
        assert cache_manager.stats.misses == 0

    def test_clear(self, cache_manager):
        """Test clearing the cache."""
        cache_manager.store_cache([1, 2], ["cache1"])
        cache_manager.store_cache([3, 4], ["cache2"])
        assert len(cache_manager) == 2

        cache_manager.clear()
        assert len(cache_manager) == 0

        # Stats should also be reset
        assert cache_manager.stats.hits == 0

    def test_cache_deep_copy(self, cache_manager):
        """Test that fetched cache is a deep copy."""
        original = [[1, 2, 3]]
        cache_manager.store_cache([1, 2], original)

        cache, _ = cache_manager.fetch_cache([1, 2])

        # Modify returned cache
        cache[0].append(4)

        # Original should be unchanged
        cache2, _ = cache_manager.fetch_cache([1, 2])
        assert cache2[0] == [1, 2, 3]

    def test_multiple_prefixes(self, cache_manager):
        """Test multiple different prefixes."""
        cache_manager.store_cache([1, 2], ["cache_a"])
        cache_manager.store_cache([3, 4], ["cache_b"])
        cache_manager.store_cache([1, 2, 3], ["cache_c"])

        # Fetch each
        cache_a, _ = cache_manager.fetch_cache([1, 2])
        cache_b, _ = cache_manager.fetch_cache([3, 4])
        cache_c, _ = cache_manager.fetch_cache([1, 2, 3])

        assert cache_a == ["cache_a"]
        assert cache_b == ["cache_b"]
        assert cache_c == ["cache_c"]

    def test_trie_structure(self, cache_manager):
        """Test trie correctly handles branching prefixes."""
        # Store two prefixes with common start
        cache_manager.store_cache([1, 2, 3], ["cache_123"])
        cache_manager.store_cache([1, 2, 4], ["cache_124"])

        # Fetch the common prefix should return shorter match
        cache, remaining = cache_manager.fetch_cache([1, 2])
        # No exact match for [1,2], but [1,2,3] is longer - behavior depends on implementation
        # In our implementation, we find shorter prefix if available, otherwise return miss

        # Fetch exact matches
        cache_123, _ = cache_manager.fetch_cache([1, 2, 3])
        cache_124, _ = cache_manager.fetch_cache([1, 2, 4])

        assert cache_123 == ["cache_123"]
        assert cache_124 == ["cache_124"]


class TestSchedulerIntegration:
    """Test integration with scheduler."""

    def test_request_cache_fields(self):
        """Test that Request has cache fields."""
        from vllm_mlx.request import Request, SamplingParams

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        # Check cache fields exist
        assert hasattr(request, 'prompt_cache')
        assert hasattr(request, 'cached_tokens')
        assert hasattr(request, 'remaining_tokens')

        # Check defaults
        assert request.prompt_cache is None
        assert request.cached_tokens == 0
        assert request.remaining_tokens is None

    def test_scheduler_config_cache_options(self):
        """Test scheduler config has cache options."""
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig(
            enable_prefix_cache=True,
            prefix_cache_size=200,
        )

        assert config.enable_prefix_cache is True
        assert config.prefix_cache_size == 200

        # Test defaults
        default_config = SchedulerConfig()
        assert default_config.enable_prefix_cache is True
        assert default_config.prefix_cache_size == 100


if __name__ == "__main__":
    # Quick standalone test with real model
    import asyncio
    import time

    MODEL_NAME = "mlx-community/Qwen3-0.6B-8bit"

    async def run_cache_test():
        from mlx_lm import load
        from vllm_mlx import AsyncEngineCore, EngineConfig, SamplingParams, SchedulerConfig

        print("=" * 60)
        print("Prefix Cache Test")
        print("=" * 60)
        print(f"Model: {MODEL_NAME}")

        print("\nLoading model...")
        model, tokenizer = load(MODEL_NAME)

        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                enable_prefix_cache=True,
                prefix_cache_size=100,
            ),
        )

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            # Same prompt twice to test caching
            prompt = "What is 2+2?"
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            params = SamplingParams(max_tokens=20, temperature=0.0)

            print("\n[1] First request (cache miss expected)...")
            start = time.perf_counter()
            rid1 = await engine.add_request(formatted, params)
            async for out in engine.stream_outputs(rid1, timeout=30):
                if out.finished:
                    break
            t1 = time.perf_counter() - start
            stats1 = engine.get_cache_stats()
            print(f"    Time: {t1*1000:.1f}ms | Stats: hits={stats1['hits']}, misses={stats1['misses']}")

            print("\n[2] Second request SAME prompt (cache hit expected)...")
            start = time.perf_counter()
            rid2 = await engine.add_request(formatted, params)
            async for out in engine.stream_outputs(rid2, timeout=30):
                if out.finished:
                    break
            t2 = time.perf_counter() - start
            stats2 = engine.get_cache_stats()
            print(f"    Time: {t2*1000:.1f}ms | Stats: hits={stats2['hits']}, misses={stats2['misses']}, tokens_saved={stats2['tokens_saved']}")

            print("\n[3] Third request DIFFERENT prompt (cache miss expected)...")
            different_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is the capital of France?"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            start = time.perf_counter()
            rid3 = await engine.add_request(different_prompt, params)
            async for out in engine.stream_outputs(rid3, timeout=30):
                if out.finished:
                    break
            t3 = time.perf_counter() - start
            stats3 = engine.get_cache_stats()
            print(f"    Time: {t3*1000:.1f}ms | Stats: hits={stats3['hits']}, misses={stats3['misses']}")

            print("\n" + "=" * 60)
            print("Final Cache Stats")
            print("=" * 60)
            final_stats = engine.get_cache_stats()
            print(f"Hit rate: {final_stats['hit_rate']*100:.1f}%")
            print(f"Tokens saved: {final_stats['tokens_saved']}")
            print("=" * 60)

    asyncio.run(run_cache_test())
