# SPDX-License-Identifier: Apache-2.0
"""
Prefix Cache Manager for vllm-mlx.

Wraps mlx-lm's LRUPromptCache to provide prefix caching functionality,
allowing reuse of computed KV cache for common prompt prefixes.
"""

import copy
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CacheEntry:
    """Entry in the prefix cache."""

    prompt_cache: List[Any]  # The cached KV state
    count: int  # Reference count for sharing


@dataclass
class PrefixCacheStats:
    """Statistics for prefix cache performance."""

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


class PrefixCacheManager:
    """
    Manages prefix caching for vllm-mlx using a trie-based LRU cache.

    This implementation is inspired by mlx-lm's LRUPromptCache but adapted
    for vllm-mlx's batching architecture.

    The cache stores KV states keyed by token sequences, allowing:
    - Exact match: Full prompt found in cache
    - Shorter match: Partial prefix found, process remaining tokens
    - Longer match: Cached prefix longer than request, trim excess

    Example:
        cache_manager = PrefixCacheManager(model, max_entries=100)

        # Check for cached prefix
        cache, remaining_tokens = cache_manager.fetch_cache(tokens)
        if cache:
            # Use cached KV, only process remaining_tokens
            pass

        # After generation, store cache for reuse
        cache_manager.store_cache(full_tokens, prompt_cache)
    """

    def __init__(self, model: Any, max_entries: int = 100):
        """
        Initialize the prefix cache manager.

        Args:
            model: The MLX model (used for cache key identification)
            max_entries: Maximum number of cached entries before LRU eviction
        """
        self.model = model
        self.model_key = id(model)
        self.max_size = max_entries

        # Trie-based cache: nested dicts with token keys
        # Structure: {model_key: {token1: {token2: {..., "cache": CacheEntry}}}}
        self._cache: Dict[Any, Dict] = {}

        # LRU tracking: (model_key, tuple(tokens)) ordered by access time
        self._lru: deque = deque()

        # Statistics
        self.stats = PrefixCacheStats()

    def _search(
        self, tokens: List[int]
    ) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]], int]:
        """
        Search for cached prefix matching tokens.

        Returns:
            Tuple of (exact, shorter, longer, common_prefix_len)
            - exact: Tokens if exact match found
            - shorter: Tokens of shorter cached prefix
            - longer: Tokens of longer cached prefix
            - common_prefix_len: Length of common prefix with longer match
        """
        if self.model_key not in self._cache:
            return None, None, None, 0

        current = self._cache[self.model_key]
        path = []

        # Traverse trie following token sequence
        for i, tok in enumerate(tokens):
            if tok not in current:
                # No match for this token
                # Check if we have a shorter prefix with cache
                if "cache" in current:
                    return None, list(path), None, 0
                return None, None, None, 0

            path.append(tok)
            current = current[tok]

        # Reached end of tokens
        if "cache" in current:
            # Exact match
            return list(tokens), None, None, 0

        # Check for longer cached prefix
        # DFS to find shortest extension with cache
        stack = [(current, list(path))]
        while stack:
            node, node_path = stack.pop()
            if "cache" in node:
                return None, None, node_path, len(tokens)
            for tok, child in node.items():
                if tok != "cache":
                    stack.append((child, node_path + [tok]))

        return None, None, None, 0

    def fetch_cache(
        self, tokens: List[int]
    ) -> Tuple[Optional[List[Any]], List[int]]:
        """
        Find cached prefix for the given tokens.

        Args:
            tokens: Input token sequence

        Returns:
            Tuple of (cache, remaining_tokens)
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        self.stats.total_queries += 1
        tokens_tuple = tuple(tokens)

        exact, shorter, longer, common_len = self._search(tokens)

        if exact:
            # Exact match - return full cache
            cache_entry = self._get_cache_entry(exact)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(tokens)
                self._touch_lru(tokens_tuple)
                # Deep copy to prevent mutation
                return copy.deepcopy(cache_entry.prompt_cache), []

        if shorter:
            # Shorter prefix cached - return cache and remaining tokens
            cache_entry = self._get_cache_entry(shorter)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(shorter)
                self._touch_lru(tuple(shorter))
                remaining = tokens[len(shorter) :]
                return copy.deepcopy(cache_entry.prompt_cache), remaining

        if longer:
            # Longer prefix cached - trim to match and return
            cache_entry = self._get_cache_entry(longer)
            if cache_entry:
                # Check if cache supports trimming
                prompt_cache = cache_entry.prompt_cache
                if self._can_trim_cache(prompt_cache):
                    trim_amount = len(longer) - len(tokens)
                    trimmed_cache = self._trim_cache(
                        copy.deepcopy(prompt_cache), trim_amount
                    )
                    self.stats.hits += 1
                    self.stats.tokens_saved += len(tokens)
                    return trimmed_cache, []

        # No cache hit
        self.stats.misses += 1
        return None, tokens

    def store_cache(self, tokens: List[int], prompt_cache: List[Any]) -> None:
        """
        Store computed cache for future reuse.

        Args:
            tokens: Token sequence that was processed
            prompt_cache: The computed KV cache to store
        """
        if not tokens:
            return

        tokens_tuple = tuple(tokens)

        # Build trie path
        if self.model_key not in self._cache:
            self._cache[self.model_key] = {}

        current = self._cache[self.model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        # Store or update cache entry
        if "cache" in current:
            current["cache"].count += 1
            # Update LRU position
            try:
                self._lru.remove((self.model_key, tokens_tuple))
            except ValueError:
                pass
        else:
            current["cache"] = CacheEntry(prompt_cache, 1)

        self._lru.append((self.model_key, tokens_tuple))

        # Evict if over capacity
        while len(self._lru) > self.max_size:
            self._evict_lru()

    def _get_cache_entry(self, tokens: List[int]) -> Optional[CacheEntry]:
        """Get cache entry for given tokens."""
        if self.model_key not in self._cache:
            return None

        current = self._cache[self.model_key]
        for tok in tokens:
            if tok not in current:
                return None
            current = current[tok]

        return current.get("cache")

    def _touch_lru(self, tokens_tuple: tuple) -> None:
        """Move entry to end of LRU queue (most recently used)."""
        key = (self.model_key, tokens_tuple)
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._lru:
            return

        model_key, tokens_tuple = self._lru.popleft()
        self._delete_cache(model_key, list(tokens_tuple))
        self.stats.evictions += 1

    def _delete_cache(self, model_key: Any, tokens: List[int]) -> None:
        """Delete cache entry and clean up empty trie branches."""
        if model_key not in self._cache:
            return

        # Navigate to entry
        path = [(self._cache[model_key], None)]
        current = self._cache[model_key]

        for tok in tokens:
            if tok not in current:
                return
            path.append((current[tok], tok))
            current = current[tok]

        # Delete cache entry
        if "cache" in current:
            del current["cache"]

        # Clean up empty branches (bottom-up)
        for i in range(len(path) - 1, 0, -1):
            node, tok = path[i]
            parent, _ = path[i - 1]
            if not node:  # Empty dict
                del parent[tok]

    def _can_trim_cache(self, prompt_cache: List[Any]) -> bool:
        """Check if cache can be trimmed."""
        if not prompt_cache:
            return False
        # Check if first cache layer has is_trimmable method
        first_cache = prompt_cache[0]
        if hasattr(first_cache, "is_trimmable"):
            return first_cache.is_trimmable()
        return hasattr(first_cache, "trim")

    def _trim_cache(self, prompt_cache: List[Any], num_tokens: int) -> List[Any]:
        """Trim cache by removing num_tokens from the end."""
        for cache in prompt_cache:
            if hasattr(cache, "trim"):
                cache.trim(num_tokens)
        return prompt_cache

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = PrefixCacheStats()

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._lru.clear()
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._lru)
