# SPDX-License-Identifier: Apache-2.0
"""
Tests for VLM (Vision Language Model) KV cache functionality.

These tests verify the VLMCacheManager for caching KV states
when the same image/video + prompt combination is requested.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vllm_mlx.vlm_cache import (
    VLMCacheEntry,
    VLMCacheManager,
    VLMCacheStats,
    compute_image_hash,
    compute_images_hash,
)


class TestVLMCacheStats:
    """Tests for VLMCacheStats class."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        stats = VLMCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0
        assert stats.image_cache_hits == 0
        assert stats.total_queries == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = VLMCacheStats(hits=3, misses=7, total_queries=10)
        assert stats.hit_rate == 0.3

    def test_hit_rate_zero_queries(self):
        """Test hit rate with zero queries."""
        stats = VLMCacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = VLMCacheStats(
            hits=5,
            misses=5,
            tokens_saved=100,
            image_cache_hits=3,
            total_queries=10,
            evictions=2,
        )
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5
        assert d["tokens_saved"] == 100
        assert d["image_cache_hits"] == 3
        assert d["total_queries"] == 10
        assert d["evictions"] == 2


class TestImageHashing:
    """Tests for image hashing functions."""

    def test_compute_image_hash_file(self):
        """Test hashing a real image file."""
        # Create a temp file with some content
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image content for testing")
            temp_path = f.name

        try:
            hash1 = compute_image_hash(temp_path)
            hash2 = compute_image_hash(temp_path)

            # Same file should give same hash
            assert hash1 == hash2
            assert len(hash1) == 16  # First 16 chars of SHA256
        finally:
            os.unlink(temp_path)

    def test_compute_image_hash_different_content(self):
        """Test that different content gives different hash."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"content 1")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"content 2")
            path2 = f2.name

        try:
            hash1 = compute_image_hash(path1)
            hash2 = compute_image_hash(path2)
            assert hash1 != hash2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_compute_image_hash_url(self):
        """Test hashing a URL (non-existent file path)."""
        url = "https://example.com/image.jpg"
        hash1 = compute_image_hash(url)
        hash2 = compute_image_hash(url)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_images_hash_empty(self):
        """Test hashing empty image list."""
        result = compute_images_hash([])
        assert result == "no_images"

    def test_compute_images_hash_single(self):
        """Test hashing single image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test content")
            path = f.name

        try:
            hash_single = compute_images_hash([path])
            assert len(hash_single) == 16
        finally:
            os.unlink(path)

    def test_compute_images_hash_multiple(self):
        """Test hashing multiple images."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"image 1")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"image 2")
            path2 = f2.name

        try:
            # Order shouldn't matter (sorted internally)
            hash_a = compute_images_hash([path1, path2])
            hash_b = compute_images_hash([path2, path1])
            assert hash_a == hash_b
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestVLMCacheEntry:
    """Tests for VLMCacheEntry class."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        cache = ["mock_kv_cache"]
        entry = VLMCacheEntry(
            prompt_cache=cache,
            image_hash="abc123",
            prompt_tokens=50,
        )
        assert entry.prompt_cache == ["mock_kv_cache"]
        assert entry.image_hash == "abc123"
        assert entry.prompt_tokens == 50
        assert entry.count == 1

    def test_cache_entry_count_increment(self):
        """Test incrementing reference count."""
        entry = VLMCacheEntry(
            prompt_cache=["cache"],
            image_hash="xyz",
            prompt_tokens=10,
        )
        entry.count += 1
        assert entry.count == 2


class TestVLMCacheManager:
    """Tests for VLMCacheManager class."""

    @pytest.fixture
    def cache_manager(self):
        """Create a cache manager with default settings."""
        return VLMCacheManager(max_entries=10)

    @pytest.fixture
    def temp_image(self):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test image content")
            path = f.name
        yield path
        os.unlink(path)

    def test_initialization(self):
        """Test cache manager initialization."""
        manager = VLMCacheManager(max_entries=50)
        assert manager.max_size == 50
        assert len(manager) == 0

    def test_fetch_empty_cache(self, cache_manager):
        """Test fetching from empty cache returns miss."""
        cache, hit = cache_manager.fetch_cache(["image.jpg"], "Describe this")

        assert cache is None
        assert hit is False
        assert cache_manager.stats.misses == 1
        assert cache_manager.stats.hits == 0

    def test_store_and_fetch_exact_match(self, cache_manager, temp_image):
        """Test storing and fetching exact match."""
        images = [temp_image]
        prompt = "Describe this image"
        mock_cache = ["kv_layer_1", "kv_layer_2"]

        # Store cache
        cache_manager.store_cache(images, prompt, mock_cache, num_tokens=100)
        assert len(cache_manager) == 1

        # Fetch exact match
        cache, hit = cache_manager.fetch_cache(images, prompt)

        assert cache is not None
        assert hit is True
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == 100

    def test_different_prompt_different_cache(self, cache_manager, temp_image):
        """Test that different prompts get different cache entries."""
        images = [temp_image]

        # Store with first prompt
        cache_manager.store_cache(images, "Describe this", ["cache1"], num_tokens=50)

        # Fetch with different prompt
        cache, hit = cache_manager.fetch_cache(images, "What is in this image?")

        assert cache is None
        assert hit is False
        assert cache_manager.stats.misses == 1

    def test_different_image_different_cache(self, cache_manager):
        """Test that different images get different cache entries."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"image 1 content")
            img1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"image 2 content")
            img2 = f2.name

        try:
            prompt = "Describe this"

            # Store with first image
            cache_manager.store_cache([img1], prompt, ["cache1"], num_tokens=50)

            # Fetch with different image - should miss
            cache, hit = cache_manager.fetch_cache([img2], prompt)
            assert cache is None
            assert hit is False
        finally:
            os.unlink(img1)
            os.unlink(img2)

    def test_video_cache_key(self, cache_manager):
        """Test that video parameters affect cache key."""
        video_source_1 = "video:test.mp4:fps2.0:max32"
        video_source_2 = "video:test.mp4:fps1.0:max64"
        prompt = "Describe this video"

        # Store with first video params
        cache_manager.store_cache([video_source_1], prompt, ["cache1"], num_tokens=100)

        # Fetch with same params - should hit
        cache, hit = cache_manager.fetch_cache([video_source_1], prompt)
        assert hit is True

        # Fetch with different params - should miss
        cache, hit = cache_manager.fetch_cache([video_source_2], prompt)
        assert hit is False

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        manager = VLMCacheManager(max_entries=3)

        # Fill cache
        manager.store_cache(["img1.jpg"], "prompt1", ["cache1"])
        manager.store_cache(["img2.jpg"], "prompt2", ["cache2"])
        manager.store_cache(["img3.jpg"], "prompt3", ["cache3"])
        assert len(manager) == 3

        # Add one more - should evict oldest
        manager.store_cache(["img4.jpg"], "prompt4", ["cache4"])
        assert len(manager) == 3
        assert manager.stats.evictions == 1

        # img1 should be evicted
        cache, hit = manager.fetch_cache(["img1.jpg"], "prompt1")
        assert cache is None
        assert hit is False

    def test_lru_touch_on_access(self):
        """Test that accessing a cache updates LRU order."""
        manager = VLMCacheManager(max_entries=3)

        # Fill cache
        manager.store_cache(["img1.jpg"], "p1", ["cache1"])
        manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        manager.store_cache(["img3.jpg"], "p3", ["cache3"])

        # Access img1 to make it most recently used
        manager.fetch_cache(["img1.jpg"], "p1")

        # Add new entry - should evict img2 (oldest untouched)
        manager.store_cache(["img4.jpg"], "p4", ["cache4"])

        # img1 should still be there
        cache, hit = manager.fetch_cache(["img1.jpg"], "p1")
        assert hit is True

        # img2 should be evicted
        cache, hit = manager.fetch_cache(["img2.jpg"], "p2")
        assert hit is False

    def test_store_empty_cache(self, cache_manager):
        """Test that empty cache is not stored."""
        cache_manager.store_cache(["img.jpg"], "prompt", [])
        assert len(cache_manager) == 0

    def test_store_none_cache(self, cache_manager):
        """Test that None cache is not stored."""
        cache_manager.store_cache(["img.jpg"], "prompt", None)
        assert len(cache_manager) == 0

    def test_get_stats(self, cache_manager, temp_image):
        """Test getting statistics."""
        # Generate some activity
        cache_manager.store_cache([temp_image], "Describe", ["cache1"], num_tokens=50)
        cache_manager.fetch_cache([temp_image], "Describe")  # Hit
        cache_manager.fetch_cache(["other.jpg"], "Describe")  # Miss

        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["total_queries"] == 2
        assert stats["image_cache_hits"] == 1

    def test_reset_stats(self, cache_manager):
        """Test resetting statistics."""
        cache_manager.stats.hits = 10
        cache_manager.stats.misses = 5
        cache_manager.reset_stats()

        assert cache_manager.stats.hits == 0
        assert cache_manager.stats.misses == 0

    def test_clear(self, cache_manager, temp_image):
        """Test clearing the cache."""
        cache_manager.store_cache([temp_image], "p1", ["cache1"])
        cache_manager.store_cache(["img2.jpg"], "p2", ["cache2"])
        assert len(cache_manager) == 2

        cache_manager.clear()
        assert len(cache_manager) == 0

        # Stats should also be reset
        assert cache_manager.stats.hits == 0

    def test_cache_deep_copy(self, cache_manager, temp_image):
        """Test that fetched cache is a deep copy."""
        original = [[1, 2, 3]]
        cache_manager.store_cache([temp_image], "prompt", original)

        cache, _ = cache_manager.fetch_cache([temp_image], "prompt")

        # Modify returned cache
        cache[0].append(4)

        # Original should be unchanged
        cache2, _ = cache_manager.fetch_cache([temp_image], "prompt")
        assert cache2[0] == [1, 2, 3]

    def test_repr(self, cache_manager):
        """Test string representation."""
        repr_str = repr(cache_manager)
        assert "VLMCacheManager" in repr_str
        assert "entries=0" in repr_str
        assert "max=10" in repr_str

    def test_multi_image_cache(self, cache_manager):
        """Test caching with multiple images."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"img1")
            img1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            f2.write(b"img2")
            img2 = f2.name

        try:
            prompt = "Compare these images"
            images = [img1, img2]

            # Store
            cache_manager.store_cache(images, prompt, ["multi_cache"], num_tokens=200)

            # Fetch same images in same order
            cache, hit = cache_manager.fetch_cache(images, prompt)
            assert hit is True

            # Fetch same images in different order - should still hit (sorted internally)
            cache, hit = cache_manager.fetch_cache([img2, img1], prompt)
            assert hit is True
        finally:
            os.unlink(img1)
            os.unlink(img2)


class TestMLXMultimodalLMCache:
    """Tests for cache integration with MLXMultimodalLM."""

    def test_mllm_cache_enabled_by_default(self):
        """Test that cache is enabled by default in MLXMultimodalLM."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        assert model.enable_cache is True
        assert model._cache_manager is not None

    def test_mllm_cache_disabled(self):
        """Test disabling cache in MLXMultimodalLM."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=False)
        assert model.enable_cache is False
        assert model._cache_manager is None

    def test_mllm_cache_custom_size(self):
        """Test custom cache size in MLXMultimodalLM."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", cache_size=100)
        assert model._cache_manager.max_size == 100

    def test_mllm_get_cache_stats_disabled(self):
        """Test get_cache_stats when cache is disabled."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=False)
        stats = model.get_cache_stats()
        assert stats["enabled"] is False

    def test_mllm_get_cache_stats_enabled(self):
        """Test get_cache_stats when cache is enabled."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=True)
        stats = model.get_cache_stats()
        assert stats["enabled"] is True
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "cache_entries" in stats
        assert "max_entries" in stats

    def test_mllm_clear_cache(self):
        """Test clearing cache in MLXMultimodalLM."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model", enable_cache=True)

        # Add some entries manually
        model._cache_manager.store_cache(["img.jpg"], "prompt", ["cache"])
        assert len(model._cache_manager) == 1

        # Clear
        model.clear_cache()
        assert len(model._cache_manager) == 0


if __name__ == "__main__":
    # Test VLM cache with real model using mlx-vlm directly (no transformers processor)
    # Reuses image/video loading from benchmark module
    import time
    from pathlib import Path

    VLM_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-3bit"

    def run_vlm_cache_test():
        """
        Test VLM cache with real model's KV cache and real images/videos.

        Uses mlx-vlm's load_model directly to avoid transformers processor bugs.
        Reuses image/video utilities from benchmark module.
        """
        from huggingface_hub import snapshot_download
        from mlx_vlm.utils import load_model, load_config
        from mlx_vlm.models import cache as vlm_cache

        # Reuse benchmark utilities for image/video loading
        from vllm_mlx.benchmark import (
            download_test_image,
            download_video,
            get_video_info,
            MLLM_TEST_IMAGE_URLS,
            VLM_TEST_VIDEO_URLS,
        )

        def print_cache_stats(label, manager):
            stats = manager.get_stats()
            print(
                f"    {label}: hits={stats['hits']}, misses={stats['misses']}, "
                f"hit_rate={stats['hit_rate']*100:.1f}%, "
                f"tokens_saved={stats['tokens_saved']}, "
                f"image_cache_hits={stats['image_cache_hits']}, "
                f"evictions={stats['evictions']}"
            )

        print("=" * 60)
        print("VLM KV Cache Test with Real Model")
        print("=" * 60)
        print(f"Model: {VLM_MODEL}")
        print()

        # Download and load model (without processor to avoid transformers bugs)
        print("Downloading model...")
        model_path = Path(snapshot_download(
            VLM_MODEL,
            allow_patterns=["*.safetensors", "*.json"]
        ))

        print("Loading model...")
        start = time.perf_counter()
        model = load_model(model_path)
        config = load_config(model_path)
        load_time = time.perf_counter() - start
        print(f"Model loaded in {load_time:.2f}s")
        print(f"Model type: {config.get('model_type', 'unknown')}")
        print()

        # Create real KV cache from model
        print("Creating real KV cache from model.language_model...")
        real_kv_cache = vlm_cache.make_prompt_cache(model.language_model)
        print(f"KV cache: {len(real_kv_cache)} layers of {type(real_kv_cache[0]).__name__}")
        print()

        # Download real test images (from benchmark list)
        print("Downloading test images...")
        image_paths = []
        resized_image_entries = []
        base_image = None
        import tempfile
        from PIL import Image
        for idx, url in enumerate(MLLM_TEST_IMAGE_URLS, start=1):
            try:
                test_image = download_test_image(url)
                temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_path = temp_img.name
                temp_img.close()
                test_image.save(temp_path, "JPEG")
                image_paths.append(temp_path)
                if base_image is None:
                    base_image = test_image.copy()
                print(f"Image {idx}: {test_image.size[0]}x{test_image.size[1]} ({url})")
            except Exception as exc:
                print(f"Image {idx}: failed to download ({url}): {exc}")
        if not image_paths:
            raise RuntimeError("No test images could be downloaded.")
        print()

        if base_image is not None:
            print("Creating resized image variants...")
            resize_sizes = [(224, 224), (336, 336), (512, 512), (768, 768)]
            for width, height in resize_sizes:
                resized = base_image.resize((width, height), Image.Resampling.LANCZOS)
                temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_path = temp_img.name
                temp_img.close()
                resized.save(temp_path, "JPEG")
                resized_image_entries.append((temp_path, width, height))
                print(f"Resized: {width}x{height}")
            print()

        # Download real test videos (from benchmark list)
        print("Downloading test videos...")
        video_paths = []
        for idx, url in enumerate(VLM_TEST_VIDEO_URLS, start=1):
            try:
                path = download_video(url)
                video_paths.append(path)
                video_info = get_video_info(path)
                print(f"Video {idx}: {video_info['width']}x{video_info['height']}, "
                      f"{video_info['duration']:.1f}s, {video_info['fps']:.1f} fps ({url})")
            except Exception as exc:
                print(f"Video {idx}: failed to download ({url}): {exc}")
        if not video_paths:
            raise RuntimeError("No test videos could be downloaded.")
        print()

        primary_image_path = image_paths[0]
        primary_video_path = video_paths[0]

        # Initialize VLM Cache Manager
        cache_manager = VLMCacheManager(max_entries=50)

        # Test 1: Image cache miss then hit
        print("[1] Testing IMAGE cache...")
        test_prompt = "Describe this image in detail"

        # First request - miss
        cached, hit = cache_manager.fetch_cache([primary_image_path], test_prompt)
        print(f"    First request: hit={hit} (expected: False)")
        assert not hit, "Expected cache miss"
        print_cache_stats("After image miss", cache_manager)

        # Store cache
        cache_manager.store_cache([primary_image_path], test_prompt, real_kv_cache, num_tokens=500)
        print(f"    Stored cache for image")
        print_cache_stats("After image store", cache_manager)

        # Second request - hit
        cached, hit = cache_manager.fetch_cache([primary_image_path], test_prompt)
        print(f"    Second request: hit={hit} (expected: True)")
        assert hit, "Expected cache hit"
        print(f"    Retrieved cache layers: {len(cached)}")
        print_cache_stats("After image hit", cache_manager)

        # Different prompt - miss
        cached, hit = cache_manager.fetch_cache([primary_image_path], "Different question")
        print(f"    Different prompt: hit={hit} (expected: False)")
        assert not hit, "Expected cache miss for different prompt"
        print_cache_stats("After image different prompt", cache_manager)

        # Extra images - independent cache entries
        if len(image_paths) > 1:
            print("\n[1b] Testing ADDITIONAL image cache entries...")
            for idx, image_path in enumerate(image_paths[1:], start=2):
                extra_prompt = f"Describe image {idx}"
                cached, hit = cache_manager.fetch_cache([image_path], extra_prompt)
                print(f"    Image {idx} first request: hit={hit} (expected: False)")
                assert not hit
                cache_manager.store_cache([image_path], extra_prompt, real_kv_cache, num_tokens=300 + idx * 10)
                cached, hit = cache_manager.fetch_cache([image_path], extra_prompt)
                print(f"    Image {idx} second request: hit={hit} (expected: True)")
                assert hit
            print_cache_stats("After additional images", cache_manager)

        if resized_image_entries:
            print("\n[1c] Testing RESIZED image cache entries...")
            for idx, (image_path, width, height) in enumerate(resized_image_entries, start=1):
                extra_prompt = f"Describe resized image {idx}"
                cached, hit = cache_manager.fetch_cache([image_path], extra_prompt)
                print(f"    Resized {width}x{height} first request: hit={hit} (expected: False)")
                assert not hit
                cache_manager.store_cache([image_path], extra_prompt, real_kv_cache, num_tokens=200 + idx * 5)
                cached, hit = cache_manager.fetch_cache([image_path], extra_prompt)
                print(f"    Resized {width}x{height} second request: hit={hit} (expected: True)")
                assert hit
            print_cache_stats("After resized images", cache_manager)

        # Test 2: Video cache with fps/max_frames
        print("\n[2] Testing VIDEO cache with fps/max_frames...")
        video_fps = 2.0
        video_max_frames = 16

        # Video cache key includes fps and max_frames
        video_key = f"video:{primary_video_path}:fps{video_fps}:max{video_max_frames}"
        video_prompt = "Describe what happens in this video"

        # First request - miss
        cached, hit = cache_manager.fetch_cache([video_key], video_prompt)
        print(f"    First request: hit={hit} (expected: False)")
        assert not hit
        print_cache_stats("After video miss", cache_manager)

        # Store cache
        cache_manager.store_cache([video_key], video_prompt, real_kv_cache, num_tokens=800)
        print(f"    Stored cache for video (fps={video_fps}, max_frames={video_max_frames})")
        print_cache_stats("After video store", cache_manager)

        # Same params - hit
        cached, hit = cache_manager.fetch_cache([video_key], video_prompt)
        print(f"    Same params: hit={hit} (expected: True)")
        assert hit
        print_cache_stats("After video hit", cache_manager)

        # Different fps - miss (important for video!)
        video_key_diff_fps = f"video:{primary_video_path}:fps4.0:max{video_max_frames}"
        cached, hit = cache_manager.fetch_cache([video_key_diff_fps], video_prompt)
        print(f"    Different fps (4.0): hit={hit} (expected: False)")
        assert not hit
        print_cache_stats("After video different fps", cache_manager)

        # Different max_frames - miss
        video_key_diff_frames = f"video:{primary_video_path}:fps{video_fps}:max32"
        cached, hit = cache_manager.fetch_cache([video_key_diff_frames], video_prompt)
        print(f"    Different max_frames (32): hit={hit} (expected: False)")
        assert not hit
        print_cache_stats("After video different max_frames", cache_manager)

        print("\n[2a] Testing multiple fps/max_frames combinations...")
        video_configs = [
            (0.5, 2),
            (1.0, 4),
            (2.0, 8),
            (4.0, 16),
        ]
        for fps_value, max_frames in video_configs:
            extra_key = f"video:{primary_video_path}:fps{fps_value}:max{max_frames}"
            extra_prompt = f"Describe video at {fps_value} fps"
            cached, hit = cache_manager.fetch_cache([extra_key], extra_prompt)
            print(f"    fps={fps_value}, max_frames={max_frames} first: hit={hit} (expected: False)")
            assert not hit
            cache_manager.store_cache(
                [extra_key],
                extra_prompt,
                real_kv_cache,
                num_tokens=600 + int(fps_value * 100) + max_frames,
            )
            cached, hit = cache_manager.fetch_cache([extra_key], extra_prompt)
            print(f"    fps={fps_value}, max_frames={max_frames} second: hit={hit} (expected: True)")
            assert hit
        print_cache_stats("After multiple fps/max_frames", cache_manager)

        # Extra videos - independent cache entries
        if len(video_paths) > 1:
            print("\n[2b] Testing ADDITIONAL video cache entries...")
            for idx, path in enumerate(video_paths[1:], start=2):
                extra_video_key = f"video:{path}:fps{video_fps}:max{video_max_frames}"
                extra_prompt = f"Describe video {idx}"
                cached, hit = cache_manager.fetch_cache([extra_video_key], extra_prompt)
                print(f"    Video {idx} first request: hit={hit} (expected: False)")
                assert not hit
                cache_manager.store_cache([extra_video_key], extra_prompt, real_kv_cache, num_tokens=700 + idx * 10)
                cached, hit = cache_manager.fetch_cache([extra_video_key], extra_prompt)
                print(f"    Video {idx} second request: hit={hit} (expected: True)")
                assert hit
            print_cache_stats("After additional videos", cache_manager)

        # Test 3: LRU eviction
        print("\n[3] Testing LRU eviction...")
        small_cache = VLMCacheManager(max_entries=2)
        small_cache.store_cache(["img1.jpg"], "p1", real_kv_cache)
        small_cache.store_cache(["img2.jpg"], "p2", real_kv_cache)
        print(f"    Cache full: {len(small_cache)}/2 entries")

        # Access img1 to make it recently used
        small_cache.fetch_cache(["img1.jpg"], "p1")

        # Add new entry - should evict img2
        small_cache.store_cache(["img3.jpg"], "p3", real_kv_cache)
        print(f"    Added img3, evictions: {small_cache.stats.evictions}")
        print_cache_stats("Small cache after eviction", small_cache)

        # img2 should be evicted
        _, hit = small_cache.fetch_cache(["img2.jpg"], "p2")
        print(f"    img2 (oldest): hit={hit} (expected: False - evicted)")
        assert not hit

        # img1 should still be there
        _, hit = small_cache.fetch_cache(["img1.jpg"], "p1")
        print(f"    img1 (recently used): hit={hit} (expected: True)")
        assert hit

        # Print final stats
        print("\n" + "=" * 60)
        print("Final Cache Statistics")
        print("=" * 60)
        stats = cache_manager.get_stats()
        print(f"hits: {stats['hits']}")
        print(f"misses: {stats['misses']}")
        print(f"hit_rate: {stats['hit_rate']*100:.1f}%")
        print(f"tokens_saved: {stats['tokens_saved']}")
        print(f"image_cache_hits: {stats['image_cache_hits']}")
        print(f"evictions: {stats['evictions']}")
        print("=" * 60)
        print("\nALL TESTS PASSED!")

        # Cleanup temp files
        for path in image_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        for path, _, _ in resized_image_entries:
            try:
                os.unlink(path)
            except OSError:
                pass
        for path in video_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    run_vlm_cache_test()
