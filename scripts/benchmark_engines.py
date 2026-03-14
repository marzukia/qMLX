#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark script to compare inference engines on a given model.

Tests:
1. Decode speed (tok/s) - generate 200 tokens from a short prompt
2. TTFT (time to first token) - cold and cached
3. Long generation (tok/s) - generate 500 tokens

Usage:
    # Benchmark vllm-mlx (must have server running on port 8000)
    python scripts/benchmark_engines.py --engine vllm-mlx --port 8000

    # Benchmark Ollama (must have ollama running with model loaded)
    python scripts/benchmark_engines.py --engine ollama --model qwen3.5:9b-instruct-q4_K_M

    # Benchmark mlx-lm directly (no server needed)
    python scripts/benchmark_engines.py --engine mlx-lm --model mlx-community/Qwen3.5-9B-MLX-4bit

    # Run all engines
    python scripts/benchmark_engines.py --engine all --model-name "Qwen3.5-9B-4bit"
"""

import argparse
import json
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SHORT_PROMPT = "/no_think Explain how a CPU works in detail."
LONG_PROMPT = (
    "/no_think Write a comprehensive guide to building a web application with Python. "
    "Cover the following topics in detail: 1) Choosing a framework (Django vs Flask vs FastAPI), "
    "2) Setting up the project structure, 3) Database design and ORM usage, "
    "4) Authentication and authorization, 5) RESTful API design, "
    "6) Testing strategies, 7) Deployment options. "
    "For each topic, provide code examples and best practices."
)
MULTI_TURN_SYSTEM = "You are a helpful coding assistant."
MULTI_TURN_MESSAGES = [
    {"role": "user", "content": "/no_think What is a binary search tree?"},
    {
        "role": "assistant",
        "content": "A binary search tree (BST) is a data structure where each node has at most two children. The left child contains values less than the parent, and the right child contains values greater than the parent.",
    },
    {
        "role": "user",
        "content": "/no_think Now implement one in Python with insert, search, and delete operations.",
    },
]

# ---------------------------------------------------------------------------
# OpenAI-compatible benchmark (vllm-mlx, Ollama)
# ---------------------------------------------------------------------------


def benchmark_openai_compatible(
    base_url: str, model: str, num_runs: int = 3, max_tokens_short: int = 200, max_tokens_long: int = 500
):
    """Benchmark any OpenAI-compatible server."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. pip install openai")
        return None

    client = OpenAI(base_url=base_url, api_key="not-needed")
    results = {
        "engine": base_url,
        "model": model,
        "short_gen": [],
        "long_gen": [],
        "ttft_cold": [],
        "ttft_cached": [],
        "multi_turn_ttft": [],
    }

    # --- Short generation (decode speed) ---
    print(f"  Short generation ({max_tokens_short} tokens, {num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()
        tokens_received = 0
        first_token_time = None

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SHORT_PROMPT}],
            max_tokens=max_tokens_short,
            stream=True,
            temperature=0.7,

        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                has_token = bool(delta.content) or bool(getattr(delta, "reasoning", None))
                if has_token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens_received += 1

        elapsed = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else elapsed
        decode_time = elapsed - ttft if first_token_time else elapsed
        tps = tokens_received / decode_time if decode_time > 0 else 0

        results["short_gen"].append(
            {"tokens": tokens_received, "elapsed": elapsed, "ttft": ttft, "tps": tps}
        )

        if i == 0:
            results["ttft_cold"].append(ttft)
        else:
            results["ttft_cached"].append(ttft)

        print(f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s, {tokens_received} tokens")

    # --- Long generation (sustained decode speed) ---
    print(f"  Long generation ({max_tokens_long} tokens, {num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()
        tokens_received = 0
        first_token_time = None

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": LONG_PROMPT}],
            max_tokens=max_tokens_long,
            stream=True,
            temperature=0.7,

        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                has_token = bool(delta.content) or bool(getattr(delta, "reasoning", None))
                if has_token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens_received += 1

        elapsed = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else elapsed
        decode_time = elapsed - ttft if first_token_time else elapsed
        tps = tokens_received / decode_time if decode_time > 0 else 0

        results["long_gen"].append(
            {"tokens": tokens_received, "elapsed": elapsed, "ttft": ttft, "tps": tps}
        )
        print(f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s, {tokens_received} tokens")

    # --- Multi-turn TTFT (tests prompt cache) ---
    print(f"  Multi-turn TTFT ({num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()
        first_token_time = None
        tokens_received = 0

        messages = [{"role": "system", "content": MULTI_TURN_SYSTEM}] + MULTI_TURN_MESSAGES
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=100,
            stream=True,
            temperature=0.7,

        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                has_token = bool(delta.content) or bool(getattr(delta, "reasoning", None))
                if has_token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens_received += 1

        ttft = first_token_time - start if first_token_time else time.perf_counter() - start
        results["multi_turn_ttft"].append(ttft)
        print(f"    Run {i + 1}: TTFT {ttft:.3f}s")

    return results


# ---------------------------------------------------------------------------
# mlx-lm direct benchmark (no server)
# ---------------------------------------------------------------------------


def benchmark_mlx_lm_direct(model_path: str, num_runs: int = 3, max_tokens_short: int = 200, max_tokens_long: int = 500):
    """Benchmark mlx-lm directly without a server."""
    try:
        import mlx_lm
    except ImportError:
        print("ERROR: mlx-lm not installed. pip install mlx-lm")
        return None

    print(f"  Loading model {model_path}...")
    try:
        model, tokenizer = mlx_lm.load(model_path)
    except ValueError:
        # VLM models have extra weights, retry with strict=False
        print("  Retrying with strict=False (VLM model)...")
        model, tokenizer = mlx_lm.load(model_path, strict=False)

    results = {
        "engine": "mlx-lm (direct)",
        "model": model_path,
        "short_gen": [],
        "long_gen": [],
    }

    # --- Short generation ---
    print(f"  Short generation ({max_tokens_short} tokens, {num_runs} runs)...")
    for i in range(num_runs):
        prompt_tokens = tokenizer.encode(SHORT_PROMPT)
        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        for token in mlx_lm.stream_generate(
            model, tokenizer, prompt=SHORT_PROMPT, max_tokens=max_tokens_short
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1

        elapsed = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else elapsed
        decode_time = elapsed - ttft if first_token_time else elapsed
        tps = token_count / decode_time if decode_time > 0 else 0

        results["short_gen"].append(
            {"tokens": token_count, "elapsed": elapsed, "ttft": ttft, "tps": tps}
        )
        print(f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s, {token_count} tokens")

    # --- Long generation ---
    print(f"  Long generation ({max_tokens_long} tokens, {num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        for token in mlx_lm.stream_generate(
            model, tokenizer, prompt=LONG_PROMPT, max_tokens=max_tokens_long
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1

        elapsed = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else elapsed
        decode_time = elapsed - ttft if first_token_time else elapsed
        tps = token_count / decode_time if decode_time > 0 else 0

        results["long_gen"].append(
            {"tokens": token_count, "elapsed": elapsed, "ttft": ttft, "tps": tps}
        )
        print(f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s, {token_count} tokens")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(results: dict) -> dict:
    """Compute summary statistics from benchmark results."""
    summary = {"engine": results["engine"], "model": results["model"]}

    if results.get("short_gen"):
        tps_vals = [r["tps"] for r in results["short_gen"]]
        summary["short_decode_tps"] = {
            "mean": statistics.mean(tps_vals),
            "median": statistics.median(tps_vals),
            "min": min(tps_vals),
            "max": max(tps_vals),
        }

    if results.get("long_gen"):
        tps_vals = [r["tps"] for r in results["long_gen"]]
        summary["long_decode_tps"] = {
            "mean": statistics.mean(tps_vals),
            "median": statistics.median(tps_vals),
            "min": min(tps_vals),
            "max": max(tps_vals),
        }

    if results.get("ttft_cold"):
        summary["ttft_cold_s"] = statistics.mean(results["ttft_cold"])
    if results.get("ttft_cached"):
        summary["ttft_cached_s"] = statistics.mean(results["ttft_cached"])
    if results.get("multi_turn_ttft"):
        vals = results["multi_turn_ttft"]
        summary["multi_turn_ttft_cold_s"] = vals[0]
        if len(vals) > 1:
            summary["multi_turn_ttft_cached_s"] = statistics.mean(vals[1:])

    return summary


def print_summary(summary: dict):
    """Pretty-print benchmark summary."""
    print(f"\n{'=' * 60}")
    print(f"  {summary['engine']} — {summary['model']}")
    print(f"{'=' * 60}")

    if "short_decode_tps" in summary:
        s = summary["short_decode_tps"]
        print(f"  Short decode:  {s['median']:.1f} tok/s (median), range {s['min']:.1f}-{s['max']:.1f}")

    if "long_decode_tps" in summary:
        s = summary["long_decode_tps"]
        print(f"  Long decode:   {s['median']:.1f} tok/s (median), range {s['min']:.1f}-{s['max']:.1f}")

    if "ttft_cold_s" in summary:
        print(f"  TTFT (cold):   {summary['ttft_cold_s']:.3f}s")
    if "ttft_cached_s" in summary:
        print(f"  TTFT (cached): {summary['ttft_cached_s']:.3f}s")
    if "multi_turn_ttft_cold_s" in summary:
        print(f"  Multi-turn TTFT (cold):   {summary['multi_turn_ttft_cold_s']:.3f}s")
    if "multi_turn_ttft_cached_s" in summary:
        print(f"  Multi-turn TTFT (cached): {summary['multi_turn_ttft_cached_s']:.3f}s")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference engines")
    parser.add_argument(
        "--engine",
        choices=["vllm-mlx", "ollama", "mlx-lm", "all"],
        required=True,
        help="Engine to benchmark",
    )
    parser.add_argument("--model", default="default", help="Model name for OpenAI API or path for mlx-lm")
    parser.add_argument("--port", type=int, default=8000, help="Port for vllm-mlx server")
    parser.add_argument("--ollama-port", type=int, default=11434, help="Port for Ollama server")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--max-tokens-short", type=int, default=200, help="Max tokens for short gen")
    parser.add_argument("--max-tokens-long", type=int, default=500, help="Max tokens for long gen")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    all_summaries = []

    if args.engine in ("vllm-mlx", "all"):
        print(f"\n>>> Benchmarking vLLM-MLX (port {args.port})...")
        results = benchmark_openai_compatible(
            f"http://localhost:{args.port}/v1",
            args.model,
            num_runs=args.runs,
            max_tokens_short=args.max_tokens_short,
            max_tokens_long=args.max_tokens_long,
        )
        if results:
            results["engine"] = "vLLM-MLX"
            s = summarize(results)
            print_summary(s)
            all_summaries.append(s)

    if args.engine in ("ollama", "all"):
        print(f"\n>>> Benchmarking Ollama (port {args.ollama_port})...")
        results = benchmark_openai_compatible(
            f"http://localhost:{args.ollama_port}/v1",
            args.model,
            num_runs=args.runs,
            max_tokens_short=args.max_tokens_short,
            max_tokens_long=args.max_tokens_long,
        )
        if results:
            results["engine"] = "Ollama"
            s = summarize(results)
            print_summary(s)
            all_summaries.append(s)

    if args.engine in ("mlx-lm", "all"):
        print(f"\n>>> Benchmarking mlx-lm (direct)...")
        results = benchmark_mlx_lm_direct(
            args.model,
            num_runs=args.runs,
            max_tokens_short=args.max_tokens_short,
            max_tokens_long=args.max_tokens_long,
        )
        if results:
            s = summarize(results)
            print_summary(s)
            all_summaries.append(s)

    # --- Comparison table ---
    if len(all_summaries) > 1:
        print(f"\n{'=' * 70}")
        print("  COMPARISON")
        print(f"{'=' * 70}")
        header = f"{'Engine':<20} {'Short tok/s':>12} {'Long tok/s':>12} {'TTFT cold':>10} {'TTFT cached':>12}"
        print(header)
        print("-" * 70)
        for s in all_summaries:
            short = s.get("short_decode_tps", {}).get("median", 0)
            long_ = s.get("long_decode_tps", {}).get("median", 0)
            cold = s.get("ttft_cold_s", 0)
            cached = s.get("ttft_cached_s", 0)
            print(f"{s['engine']:<20} {short:>12.1f} {long_:>12.1f} {cold:>10.3f} {cached:>12.3f}")

    # --- Save results ---
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
