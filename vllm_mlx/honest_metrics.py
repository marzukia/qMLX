# SPDX-License-Identifier: Apache-2.0
"""Honest per-request reuse / latency accounting (issues #10, #2).

The pre-existing ``rapid_mlx_prompt_tokens_total`` /
``rapid_mlx_completion_tokens_total`` counters mix genuinely-computed
prompt tokens with tokens that were reused from a KV cache, and the
``/v1/status`` ``tokens_per_second`` divides output tokens by wall time
that includes the prefill. Both overstate throughput on cache hits — the
"amortized lie" the audit calls out. This module holds the honest split:

* ``offered``   — Σ ``num_prompt_tokens`` over admitted requests.
* ``computed``  — Σ ``num_prompt_tokens - cached_tokens`` (tokens the
  model actually forwarded through prefill).
* ``reused``    — Σ ``cached_tokens`` split by the source the KV was
  actually installed from (``memory`` vs ``disk``). Captured at the
  point the request is inserted into the running batch with its cache,
  never at lookup time — a lookup that reports a hit but then fails
  reconstruction / insert has ``cached_tokens`` scrubbed back to 0 by
  the scheduler before this records anything, so a failed install
  contributes nothing.
* ``prefill_kind`` — one of ``cold`` (no cache), ``extend`` (partial
  prefix reused, tail re-prefilled), ``exact`` (whole prompt reused,
  only the generation-kickoff token fed).
* ``prefix_cache_match`` — the in-memory prefix cache's match-type
  distribution (``exact``/``prefix``/``supersequence``/``lcp``/``miss``),
  taken straight off the cache's ``_last_match_type``.
* ``ttft`` histogram — ``first_token_time - arrival_time`` seconds.
* ``decode_tps`` histogram — ``(num_output_tokens - 1) / (t_last_token
  - first_token_time)``, i.e. inter-token gaps divided by the pure decode
  window. Prompt tokens and the arrival→first-token latency are excluded
  by construction. Only outputs of ≥2 tokens contribute (a single output
  token spans zero decode gaps).

Guardrail (see the module owner's commit message): every counter here
that can move with prompt length is a prefill/reuse counter by design.
Nothing in this module divides ``(prompt + generated) / wall`` — that is
the amortized number the honest-metrics pass exists to kill.

Thread-safety: the scheduler step loop records from a single thread while
``/metrics`` and ``/v1/status`` snapshot from request-handler threads. A
lock guards every mutation and the snapshot copies out, so a scrape never
observes a torn dict or a dict mutated mid-iteration.
"""

from __future__ import annotations

import threading

# Fixed histogram bucket upper bounds. Chosen once here (not per-deploy)
# so the exposed ``le`` set is stable across restarts — Prometheus treats
# a histogram whose bucket layout changes as a different series. The
# ``+Inf`` bucket is implicit and always emitted last.
TTFT_BUCKET_BOUNDS: tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
)
DECODE_TPS_BUCKET_BOUNDS: tuple[float, ...] = (
    5.0,
    10.0,
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    150.0,
    200.0,
)

# Canonical in-memory prefix-cache match types (mirrors
# ``MemoryAwarePrefixCache._last_match_type`` and the ``Request``
# ``cache_hit_type`` docstring). Seeded at zero so every series is always
# present in the exposition even before the first request lands.
PREFIX_MATCH_TYPES: tuple[str, ...] = (
    "exact",
    "prefix",
    "supersequence",
    "lcp",
    "miss",
)


class FixedBucketHistogram:
    """Minimal cumulative-bucket histogram (no prometheus_client dep).

    Stores cumulative counts directly: ``_cumulative[i]`` is the number of
    observations ``<= bounds[i]``. ``snapshot`` appends the implicit
    ``+Inf`` bucket (== total count) so the caller can render a spec-valid
    Prometheus histogram (monotonic ``_bucket`` series + ``_sum`` +
    ``_count``).
    """

    __slots__ = ("_bounds", "_cumulative", "_sum", "_count")

    def __init__(self, bounds: tuple[float, ...]) -> None:
        self._bounds = tuple(float(b) for b in bounds)
        self._cumulative = [0] * len(self._bounds)
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float) -> None:
        """Record one observation. O(len(bounds)); called at request finish only."""
        v = float(value)
        self._sum += v
        self._count += 1
        for i, bound in enumerate(self._bounds):
            if v <= bound:
                self._cumulative[i] += 1

    def snapshot(self) -> dict[str, object]:
        """Copy out ``{buckets: [(le_str, cum_count)...], sum, count}``."""
        buckets: list[tuple[str, int]] = [
            (repr(bound), self._cumulative[i]) for i, bound in enumerate(self._bounds)
        ]
        buckets.append(("+Inf", self._count))
        return {"buckets": buckets, "sum": self._sum, "count": self._count}


class HonestMetrics:
    """Process-lifetime accumulator for the honest reuse / latency counters.

    One instance lives on the scheduler. It is never reset by
    ``cache.clear()`` / ``Scheduler.reset()`` — those clear cache contents
    and per-request ledgers, not lifetime observability counters — so the
    exposed series stay monotonic for the life of the process (matching
    every other ``rapid_mlx_*_total`` counter).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens_offered = 0
        self.prompt_tokens_computed = 0
        self.prompt_tokens_reused: dict[str, int] = {"memory": 0, "disk": 0}
        self.prefill_kind: dict[str, int] = {"cold": 0, "extend": 0, "exact": 0}
        self.prefix_cache_match: dict[str, int] = {t: 0 for t in PREFIX_MATCH_TYPES}
        # Disk (SSD) KV-restore attempts by result. Request-level: exactly
        # one increment per request that actually ran a disk-checkpoint
        # lookup (hit == a checkpoint prefix was found, verified, AND
        # installed; miss == the lookup found nothing or a candidate was
        # rejected before install). Requests that never engaged the
        # disk-restore path are counted as neither, so hit + miss ==
        # disk-restore attempts and hit / (hit + miss) is the true SSD
        # hit rate. On this hybrid model the in-memory prefix cache always
        # misses and the reuse rides on disk restore, so this — not
        # prefix_cache_match — is the honest reuse-hit-rate surface.
        self.kv_restore_result: dict[str, int] = {"hit": 0, "miss": 0}
        self._ttft = FixedBucketHistogram(TTFT_BUCKET_BOUNDS)
        self._decode_tps = FixedBucketHistogram(DECODE_TPS_BUCKET_BOUNDS)

    def record_prefill(
        self,
        num_prompt_tokens: int,
        cached_tokens: int,
        cache_hit_type: str | None,
        remaining_tokens: list[int] | None,
    ) -> None:
        """Record offered / computed / reused / prefill-kind for one admitted request.

        Call site: the moment the request is inserted into the running
        batch with its (already-validated) cache — ``cached_tokens`` at
        that point is exactly the KV that was installed, scrubbed to 0 by
        the scheduler if reconstruction / insert failed. So ``reused``
        never over-counts a lookup that didn't turn into a real install.
        """
        num_prompt_tokens = max(0, int(num_prompt_tokens))
        cached_tokens = max(0, int(cached_tokens))
        # cached can never legitimately exceed the prompt; floor computed at 0.
        computed = max(0, num_prompt_tokens - cached_tokens)
        # Exact match is signalled by an empty (but not None) remaining list:
        # the whole prompt was reused and only the kickoff token is fed.
        is_exact = remaining_tokens is not None and len(remaining_tokens) == 0
        with self._lock:
            self.prompt_tokens_offered += num_prompt_tokens
            self.prompt_tokens_computed += computed
            if cached_tokens > 0:
                source = "disk" if cache_hit_type == "disk" else "memory"
                self.prompt_tokens_reused[source] += cached_tokens
            if is_exact:
                kind = "exact"
            elif cached_tokens == 0:
                kind = "cold"
            else:
                kind = "extend"
            self.prefill_kind[kind] += 1

    def record_prefix_match(self, match_type: str | None) -> None:
        """Record one in-memory prefix-cache match-type observation.

        Only the five canonical memory-cache match types are tracked; the
        paged/legacy caches' coarse ``hit`` and the disk-restore ``disk``
        tag are not prefix-cache match types and are intentionally not
        folded in (they would need their own, differently-defined series).
        """
        if match_type in self.prefix_cache_match:
            with self._lock:
                self.prefix_cache_match[match_type] += 1

    def record_disk_restore(self, hit: bool) -> None:
        """Record one disk (SSD) KV-restore attempt as hit or miss.

        Call site: the single accounting point at the end of the
        scheduler's ``_maybe_disk_restore``, reached only when a disk
        checkpoint lookup actually ran (all the earlier gate returns —
        feature off, PFlash, an in-memory hit, empty prompt — happen
        before the lookup and are NOT attempts). ``hit`` is true only when
        a checkpoint was found, verified, and installed onto the request;
        every lookup-miss / validation-reject path is a miss.
        """
        with self._lock:
            self.kv_restore_result["hit" if hit else "miss"] += 1

    def record_finish(
        self,
        arrival_time: float | None,
        first_token_time: float | None,
        t_last_token: float | None,
        num_output_tokens: int,
    ) -> None:
        """Record TTFT and pure-decode throughput for one finished request.

        ``arrival_time`` / ``first_token_time`` / ``t_last_token`` must all
        be on the SAME clock (the scheduler stamps all three off
        ``time.time()``); a mixed wall/monotonic pair would corrupt the
        interval. TTFT is recorded whenever a first token was produced.
        Decode throughput needs ≥2 output tokens (one inter-token gap) and
        a strictly-positive decode window.
        """
        with self._lock:
            if (
                arrival_time is not None
                and first_token_time is not None
                and first_token_time >= arrival_time
            ):
                self._ttft.observe(first_token_time - arrival_time)
            if (
                num_output_tokens >= 2
                and first_token_time is not None
                and t_last_token is not None
            ):
                window = t_last_token - first_token_time
                if window > 0:
                    self._decode_tps.observe((num_output_tokens - 1) / window)

    def snapshot(self) -> dict[str, object]:
        """Copy out every counter / histogram for ``Scheduler.get_stats()``."""
        with self._lock:
            return {
                "prompt_tokens_offered": self.prompt_tokens_offered,
                "prompt_tokens_computed": self.prompt_tokens_computed,
                "prompt_tokens_reused": dict(self.prompt_tokens_reused),
                "prefill_kind": dict(self.prefill_kind),
                "prefix_cache_match": dict(self.prefix_cache_match),
                "kv_restore_result": dict(self.kv_restore_result),
                "ttft_seconds": self._ttft.snapshot(),
                "decode_tokens_per_second": self._decode_tps.snapshot(),
            }
