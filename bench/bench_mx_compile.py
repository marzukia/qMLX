#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark mx.compile() on the MTP verify forward pass.

The verify forward (``model(y, cache=..., return_hidden=True, n_confirmed=1)``)
runs every speculative-decode cycle.  If ``mx.compile()`` can cache the
computation graph it could save per-cycle compilation overhead.

Tests two paths:
  1. Full verify forward — ``model(...)`` with KV cache (may fail if
     ``mx.compile`` can't trace through mutable cache objects).
  2. MTP head only — ``model.mtp_forward(hidden, token_ids, mtp_cache)``
     (smaller scope, more likely to compile).

Usage::

    python bench/bench_mx_compile.py
    python bench/bench_mx_compile.py --model mlx-community/Qwen3.5-9B-4bit --n-runs 100
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as _cache_module

from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
    inject_mtp_support,
    validate_mtp_support,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-9B-4bit",
        help="Base model alias or HF path.",
    )
    p.add_argument(
        "--sidecar",
        default="mlx-community/Qwen3.5-9B-MTP-4bit",
        help="MTP head sidecar repo or local path.",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Timed iterations per condition (default: 50).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations before timing (default: 5).",
    )
    return p.parse_args()


def _time_fn(fn, args, *, n_runs: int, warmup: int) -> list[float]:
    """Run ``fn(*args)`` with warmup, return per-iteration ms list."""
    for _ in range(warmup):
        out = fn(*args)
        if isinstance(out, tuple):
            mx.eval(out[0])
        else:
            mx.eval(out)
    mx.synchronize(None)

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            mx.eval(out[0])
        else:
            mx.eval(out)
        mx.synchronize(None)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _fmt_stats(times: list[float]) -> str:
    avg = statistics.mean(times)
    med = statistics.median(times)
    lo, hi = min(times), max(times)
    return f"avg {avg:.2f} ms  |  med {med:.2f} ms  |  [{lo:.2f}, {hi:.2f}]"


def main() -> int:
    args = _parse_args()

    print("=== mx.compile Verify Forward Benchmark ===")
    print(f"Model:   {args.model}")
    print(f"Sidecar: {args.sidecar}")
    print(f"Runs:    {args.n_runs}, Warmup: {args.warmup}")
    print()

    # ------------------------------------------------------------------
    # Load model + inject MTP
    # ------------------------------------------------------------------
    print("Loading model ...", file=sys.stderr)
    model, tokenizer = load(args.model)
    inject_mtp_support(model, mtp_sidecar=args.sidecar)
    validate_mtp_support(model)

    if hasattr(model, "language_model"):
        model = model.language_model

    model_cache = _cache_module.make_prompt_cache(model)
    mtp_cache = model.make_mtp_cache()

    # ------------------------------------------------------------------
    # Prepare sample input
    # ------------------------------------------------------------------
    prompt = "Explain how a Bloom filter works in three sentences."
    prompt_ids = mx.array(tokenizer.encode(prompt), mx.uint32)
    if len(prompt_ids) < 10:
        prompt_ids = mx.array(
            tokenizer.encode(prompt * 3), mx.uint32
        )

    # Prefill to populate cache
    print("Running prefill ...", file=sys.stderr)
    logits, hidden = model(
        prompt_ids[:10][None],
        cache=model_cache,
        return_hidden=True,
    )
    mx.eval(logits, hidden)

    # Build 2-token input: main token + 1 draft token
    y = mx.array(
        [prompt_ids[9].item(), prompt_ids[8].item()], mx.uint32
    )
    hidden_last = hidden[:, -1:, :]

    # ==================================================================
    # BENCHMARK 1: Full verify forward
    # ==================================================================
    print("-" * 58)
    print("Benchmark 1: Full verify forward  model(y[None], ..., n_confirmed=1)")
    print("-" * 58)

    # -- Baseline (uncached) --
    def verify_uncached(input_ids):
        fresh = copy.deepcopy(model_cache)
        logits, _h = model(
            input_ids[None],
            cache=fresh,
            return_hidden=True,
            n_confirmed=1,
        )
        return logits

    print("Running baseline (uncached, deepcopy cache each call) ...",
          file=sys.stderr)
    baseline_times = _time_fn(
        verify_uncached, (y,), n_runs=args.n_runs, warmup=args.warmup
    )
    print(f"  Baseline:  {_fmt_stats(baseline_times)}")

    # -- Compiled --
    compiled_ok = False
    compiled_times: list[float] = []

    def verify_cached(input_ids):
        logits, _h = model(
            input_ids[None],
            cache=model_cache,
            return_hidden=True,
            n_confirmed=1,
        )
        return logits

    try:
        print("Compiling verify forward with mx.compile ...",
              file=sys.stderr)
        compiled_fn = mx.compile(verify_cached)
        # Warm the compiled graph once (triggers trace)
        _out = compiled_fn(y)
        mx.eval(_out)
        mx.synchronize(None)

        compiled_times = _time_fn(
            compiled_fn, (y,), n_runs=args.n_runs, warmup=args.warmup
        )
        compiled_ok = True
        print(f"  Compiled:  {_fmt_stats(compiled_times)}")
    except Exception as exc:
        print(f"  Compiled:  FAILED — {exc}")
        print("  (Cache objects are likely not traceable by mx.compile)")

    if compiled_ok:
        b_avg = statistics.mean(baseline_times)
        c_avg = statistics.mean(compiled_times)
        speedup = b_avg / c_avg if c_avg > 0 else float("inf")
        delta = b_avg - c_avg
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Delta:     {delta:+.2f} ms  "
              f"({'faster' if delta > 0 else 'slower'})")
    print()

    # ==================================================================
    # BENCHMARK 2: MTP head only
    # ==================================================================
    print("-" * 58)
    print("Benchmark 2: MTP head only  model.mtp_forward(hidden, ids, cache)")
    print("-" * 58)

    # MTP head expects (hidden, token_ids) where token_ids is (1, 1)
    mtp_tok = y[-1:].reshape(1, 1).astype(mx.uint32)

    def mtp_uncached(h, tok_ids):
        fresh_mtp = model.make_mtp_cache()
        return model.mtp_forward(h, tok_ids, fresh_mtp)

    print("Running MTP head baseline (uncached) ...", file=sys.stderr)
    mtp_baseline_times = _time_fn(
        mtp_uncached,
        (hidden_last, mtp_tok),
        n_runs=args.n_runs,
        warmup=args.warmup,
    )
    print(f"  MTP Baseline:  {_fmt_stats(mtp_baseline_times)}")

    def mtp_cached(h, tok_ids):
        return model.mtp_forward(h, tok_ids, mtp_cache)

    mtp_compiled_ok = False
    mtp_compiled_times: list[float] = []

    try:
        print("Compiling mtp_forward with mx.compile ...",
              file=sys.stderr)
        compiled_mtp = mx.compile(mtp_cached)
        _out = compiled_mtp(hidden_last, mtp_tok)
        mx.eval(_out)
        mx.synchronize(None)

        mtp_compiled_times = _time_fn(
            compiled_mtp,
            (hidden_last, mtp_tok),
            n_runs=args.n_runs,
            warmup=args.warmup,
        )
        mtp_compiled_ok = True
        print(f"  MTP Compiled:  {_fmt_stats(mtp_compiled_times)}")
    except Exception as exc:
        print(f"  MTP Compiled:  FAILED — {exc}")

    if mtp_compiled_ok:
        b_avg = statistics.mean(mtp_baseline_times)
        c_avg = statistics.mean(mtp_compiled_times)
        speedup = b_avg / c_avg if c_avg > 0 else float("inf")
        delta = b_avg - c_avg
        print(f"  MTP Speedup:   {speedup:.2f}x")
        print(f"  MTP Delta:     {delta:+.2f} ms  "
              f"({'faster' if delta > 0 else 'slower'})")
    print()

    # ==================================================================
    # Summary
    # ==================================================================
    print("=" * 58)
    print("Summary")
    print("=" * 58)
    print(f"  Verify Baseline:       {_fmt_stats(baseline_times)}")
    if compiled_ok:
        print(f"  Verify Compiled:       {_fmt_stats(compiled_times)}")
    else:
        print(f"  Verify Compiled:       FAILED (not traceable)")
    print(f"  MTP Head Baseline:     {_fmt_stats(mtp_baseline_times)}")
    if mtp_compiled_ok:
        print(f"  MTP Head Compiled:     {_fmt_stats(mtp_compiled_times)}")
    else:
        print(f"  MTP Head Compiled:     FAILED (not traceable)")

    if not compiled_ok and not mtp_compiled_ok:
        print()
        print("Note: mx.compile could not trace either function.")
        print("KVCache / ArraysCache mutation may prevent graph capture.")
        print("Consider pinning caches as static inputs or using a "
              "functional forward that returns new cache state.")

    return 0


if __name__ == "__main__":
    sys.exit(main())