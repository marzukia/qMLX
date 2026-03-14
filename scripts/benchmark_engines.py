#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark script to compare inference engines on a given model.

Metrics measured:
  1. Decode TPS      — token generation speed (tok/s)
  2. TTFT cold       — time to first token on first request
  3. TTFT cached     — time to first token on subsequent requests (prompt cache)
  4. Prefill TPS     — prompt processing speed (prompt_tokens / TTFT)
  5. Multi-turn TTFT — conversation continuation latency (cold + cached)
  6. Peak RAM        — memory usage during inference (macOS only)

Engines supported:
  - rapid-mlx   (OpenAI API, default port 8000)
  - ollama      (OpenAI API, default port 11434)
  - llama-cpp   (llama-server OpenAI API, default port 8080)
  - mlx-lm      (direct Python, no server)

Usage:
    # Single engine
    python scripts/benchmark_engines.py --engine rapid-mlx --port 8000

    # Multiple engines
    python scripts/benchmark_engines.py --engine rapid-mlx ollama llama-cpp

    # All engines
    python scripts/benchmark_engines.py --engine all --model default

    # Custom ports
    python scripts/benchmark_engines.py --engine rapid-mlx llama-cpp \
        --rapid-mlx-port 8000 --llama-cpp-port 8080

    # Save results
    python scripts/benchmark_engines.py --engine all --output results.json
"""

import argparse
import json
import os
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
# Memory measurement (macOS)
# ---------------------------------------------------------------------------


def get_process_memory_mb(port: int) -> float | None:
    """Get RSS memory of the process listening on a port (macOS only)."""
    try:
        # Find PID listening on port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        # May return multiple PIDs, take the first
        pid = result.stdout.strip().split("\n")[0]
        # Get RSS in KB via ps
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", pid],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        rss_kb = int(result.stdout.strip())
        return rss_kb / 1024  # MB
    except Exception:
        return None


def get_system_memory_pressure() -> dict | None:
    """Get macOS memory pressure info."""
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        # Parse vm_stat output
        info = {}
        for line in result.stdout.strip().split("\n"):
            if "Pages free" in line:
                pages = int(line.split(":")[1].strip().rstrip("."))
                info["free_mb"] = pages * 16384 / (1024 * 1024)  # 16KB pages on Apple Silicon
            elif "Pages active" in line:
                pages = int(line.split(":")[1].strip().rstrip("."))
                info["active_mb"] = pages * 16384 / (1024 * 1024)
        return info if info else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OpenAI-compatible benchmark (rapid-mlx, ollama, llama-cpp)
# ---------------------------------------------------------------------------


def _count_stream_tokens(stream):
    """Consume an OpenAI streaming response, counting tokens.

    Returns (tokens_received, first_token_time, prompt_tokens).
    Handles both content and reasoning (Ollama Qwen3 quirk).
    """
    tokens_received = 0
    first_token_time = None
    prompt_tokens = 0

    for chunk in stream:
        # Try to extract prompt_tokens from usage (some servers report it)
        if hasattr(chunk, "usage") and chunk.usage:
            pt = getattr(chunk.usage, "prompt_tokens", None)
            if pt:
                prompt_tokens = pt

        if chunk.choices:
            delta = chunk.choices[0].delta
            has_token = bool(delta.content) or bool(getattr(delta, "reasoning", None))
            if has_token:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens_received += 1

    return tokens_received, first_token_time, prompt_tokens


def benchmark_openai_compatible(
    base_url: str,
    model: str,
    engine_name: str,
    num_runs: int = 3,
    max_tokens_short: int = 200,
    max_tokens_long: int = 500,
    port: int | None = None,
):
    """Benchmark any OpenAI-compatible server."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. pip install openai")
        return None

    # Check server is reachable
    client = OpenAI(base_url=base_url, api_key="not-needed")
    try:
        client.models.list()
    except Exception as e:
        print(f"  ERROR: Cannot reach {base_url} — {e}")
        return None

    results = {
        "engine": engine_name,
        "model": model,
        "short_gen": [],
        "long_gen": [],
        "ttft_cold": [],
        "ttft_cached": [],
        "multi_turn_ttft": [],
        "peak_ram_mb": None,
    }

    # Measure RAM before
    if port:
        ram_before = get_process_memory_mb(port)

    # --- Short generation (decode speed) ---
    print(f"  Short generation ({max_tokens_short} tokens, {num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SHORT_PROMPT}],
            max_tokens=max_tokens_short,
            stream=True,
            temperature=0.7,
        )
        tokens_received, first_token_time, prompt_tokens = _count_stream_tokens(stream)

        elapsed = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else elapsed
        decode_time = elapsed - ttft if first_token_time else elapsed
        tps = tokens_received / decode_time if decode_time > 0 else 0
        prefill_tps = prompt_tokens / ttft if (prompt_tokens and ttft > 0) else None

        results["short_gen"].append({
            "tokens": tokens_received,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "ttft": ttft,
            "tps": tps,
            "prefill_tps": prefill_tps,
        })

        if i == 0:
            results["ttft_cold"].append(ttft)
        else:
            results["ttft_cached"].append(ttft)

        prefill_str = f", prefill {prefill_tps:.0f} tok/s" if prefill_tps else ""
        print(f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s{prefill_str}, {tokens_received} tokens")

    # --- Long generation (sustained decode speed) ---
    print(f"  Long generation ({max_tokens_long} tokens, {num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": LONG_PROMPT}],
            max_tokens=max_tokens_long,
            stream=True,
            temperature=0.7,
        )
        tokens_received, first_token_time, prompt_tokens = _count_stream_tokens(stream)

        elapsed = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else elapsed
        decode_time = elapsed - ttft if first_token_time else elapsed
        tps = tokens_received / decode_time if decode_time > 0 else 0
        prefill_tps = prompt_tokens / ttft if (prompt_tokens and ttft > 0) else None

        results["long_gen"].append({
            "tokens": tokens_received,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "ttft": ttft,
            "tps": tps,
            "prefill_tps": prefill_tps,
        })
        prefill_str = f", prefill {prefill_tps:.0f} tok/s" if prefill_tps else ""
        print(f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s{prefill_str}, {tokens_received} tokens")

    # --- Multi-turn TTFT (tests prompt cache) ---
    print(f"  Multi-turn TTFT ({num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()

        messages = [{"role": "system", "content": MULTI_TURN_SYSTEM}] + MULTI_TURN_MESSAGES
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=100,
            stream=True,
            temperature=0.7,
        )
        tokens_received, first_token_time, prompt_tokens = _count_stream_tokens(stream)

        ttft = first_token_time - start if first_token_time else time.perf_counter() - start
        results["multi_turn_ttft"].append(ttft)
        print(f"    Run {i + 1}: TTFT {ttft:.3f}s")

    # Measure RAM after
    if port:
        ram_after = get_process_memory_mb(port)
        if ram_after:
            results["peak_ram_mb"] = ram_after
            print(f"  Process RAM: {ram_after:.0f} MB")

    return results


# ---------------------------------------------------------------------------
# mlx-lm direct benchmark (no server)
# ---------------------------------------------------------------------------


def benchmark_mlx_lm_direct(
    model_path: str,
    num_runs: int = 3,
    max_tokens_short: int = 200,
    max_tokens_long: int = 500,
):
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
        print("  Retrying with strict=False (VLM model)...")
        try:
            model, tokenizer = mlx_lm.load(model_path, strict=False)
        except TypeError:
            print("  ERROR: mlx_lm.load() does not support strict=False. Skipping.")
            return None

    results = {
        "engine": "mlx-lm (direct)",
        "model": model_path,
        "short_gen": [],
        "long_gen": [],
    }

    # --- Short generation ---
    print(f"  Short generation ({max_tokens_short} tokens, {num_runs} runs)...")
    for i in range(num_runs):
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

    for key, label in [("short_gen", "short_decode_tps"), ("long_gen", "long_decode_tps")]:
        if results.get(key):
            tps_vals = [r["tps"] for r in results[key]]
            summary[label] = {
                "mean": statistics.mean(tps_vals),
                "median": statistics.median(tps_vals),
                "min": min(tps_vals),
                "max": max(tps_vals),
            }
            # Prefill TPS (if available)
            prefill_vals = [r["prefill_tps"] for r in results[key] if r.get("prefill_tps")]
            if prefill_vals:
                summary[f"{label.replace('decode', 'prefill')}"] = {
                    "mean": statistics.mean(prefill_vals),
                    "median": statistics.median(prefill_vals),
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

    if results.get("peak_ram_mb"):
        summary["peak_ram_mb"] = results["peak_ram_mb"]

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

    if "short_prefill_tps" in summary:
        s = summary["short_prefill_tps"]
        print(f"  Prefill:       {s['median']:.0f} tok/s (median)")

    if "ttft_cold_s" in summary:
        print(f"  TTFT (cold):   {summary['ttft_cold_s']:.3f}s")
    if "ttft_cached_s" in summary:
        print(f"  TTFT (cached): {summary['ttft_cached_s']:.3f}s")
    if "multi_turn_ttft_cold_s" in summary:
        print(f"  Multi-turn TTFT (cold):   {summary['multi_turn_ttft_cold_s']:.3f}s")
    if "multi_turn_ttft_cached_s" in summary:
        print(f"  Multi-turn TTFT (cached): {summary['multi_turn_ttft_cached_s']:.3f}s")
    if "peak_ram_mb" in summary:
        ram = summary["peak_ram_mb"]
        print(f"  Peak RAM:      {ram:.0f} MB ({ram / 1024:.1f} GB)")

    print()


def print_comparison(all_summaries: list[dict]):
    """Print a comparison table across all engines."""
    print(f"\n{'=' * 100}")
    print("  COMPARISON")
    print(f"{'=' * 100}")

    header = (
        f"{'Engine':<16} {'Decode':>10} {'Decode':>10} {'Prefill':>10}"
        f" {'TTFT cold':>10} {'TTFT hit':>10} {'MT TTFT':>10} {'RAM':>10}"
    )
    subheader = (
        f"{'':.<16} {'short t/s':>10} {'long t/s':>10} {'tok/s':>10}"
        f" {'(s)':>10} {'(s)':>10} {'cached':>10} {'(GB)':>10}"
    )
    print(header)
    print(subheader)
    print("-" * 100)

    for s in all_summaries:
        short = s.get("short_decode_tps", {}).get("median", 0)
        long_ = s.get("long_decode_tps", {}).get("median", 0)
        prefill = s.get("short_prefill_tps", {}).get("median", 0)
        cold = s.get("ttft_cold_s", 0)
        cached = s.get("ttft_cached_s", 0)
        mt_cached = s.get("multi_turn_ttft_cached_s", 0)
        ram = s.get("peak_ram_mb", 0) / 1024 if s.get("peak_ram_mb") else 0

        prefill_str = f"{prefill:>10.0f}" if prefill else f"{'—':>10}"
        ram_str = f"{ram:>10.1f}" if ram else f"{'—':>10}"

        print(
            f"{s['engine']:<16} {short:>10.1f} {long_:>10.1f} {prefill_str}"
            f" {cold:>10.3f} {cached:>10.3f} {mt_cached:>10.3f} {ram_str}"
        )

    # Speedup row (first engine as baseline)
    if len(all_summaries) >= 2:
        base = all_summaries[0]
        print("-" * 100)
        for s in all_summaries[1:]:
            b_short = base.get("short_decode_tps", {}).get("median", 1)
            s_short = s.get("short_decode_tps", {}).get("median", 1)
            b_long = base.get("long_decode_tps", {}).get("median", 1)
            s_long = s.get("long_decode_tps", {}).get("median", 1)

            short_x = b_short / s_short if s_short > 0 else 0
            long_x = b_long / s_long if s_long > 0 else 0

            b_mt = base.get("multi_turn_ttft_cached_s", 1)
            s_mt = s.get("multi_turn_ttft_cached_s", 1)
            mt_x = s_mt / b_mt if b_mt > 0 else 0

            label = f"{base['engine']} speedup"
            print(
                f"{label:<16} {short_x:>9.1f}x {long_x:>9.1f}x {'':>10}"
                f" {'':>10} {'':>10} {mt_x:>9.1f}x {'':>10}"
            )


# ---------------------------------------------------------------------------
# Engine configs
# ---------------------------------------------------------------------------

ENGINE_CONFIGS = {
    "rapid-mlx": {"display": "Rapid-MLX", "default_port": 8000},
    "ollama": {"display": "Ollama", "default_port": 11434},
    "llama-cpp": {"display": "llama.cpp", "default_port": 8080},
    "mlx-lm": {"display": "mlx-lm", "default_port": None},
}

ALL_OPENAI_ENGINES = ["rapid-mlx", "ollama", "llama-cpp"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark Rapid-MLX (server must be running on port 8000)
  python scripts/benchmark_engines.py --engine rapid-mlx

  # Benchmark multiple engines
  python scripts/benchmark_engines.py --engine rapid-mlx ollama llama-cpp

  # All engines with custom model name
  python scripts/benchmark_engines.py --engine all --model default

  # Custom ports
  python scripts/benchmark_engines.py --engine rapid-mlx llama-cpp \\
      --rapid-mlx-port 8100 --llama-cpp-port 8080

  # Save results to JSON
  python scripts/benchmark_engines.py --engine all --output results.json
""",
    )
    parser.add_argument(
        "--engine",
        nargs="+",
        choices=["rapid-mlx", "ollama", "llama-cpp", "mlx-lm", "all"],
        required=True,
        help="Engine(s) to benchmark",
    )
    parser.add_argument("--model", default="default", help="Model name for OpenAI API or path for mlx-lm")
    parser.add_argument("--rapid-mlx-port", type=int, default=8000, help="Port for Rapid-MLX server (default: 8000)")
    parser.add_argument("--ollama-port", type=int, default=11434, help="Port for Ollama server (default: 11434)")
    parser.add_argument("--llama-cpp-port", type=int, default=8080, help="Port for llama-server (default: 8080)")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs (default: 3)")
    parser.add_argument("--max-tokens-short", type=int, default=200, help="Max tokens for short gen (default: 200)")
    parser.add_argument("--max-tokens-long", type=int, default=500, help="Max tokens for long gen (default: 500)")
    parser.add_argument("--output", help="Save results to JSON file")
    # Legacy compat
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Expand "all"
    engines = args.engine
    if "all" in engines:
        engines = list(ALL_OPENAI_ENGINES) + ["mlx-lm"]

    # Legacy --port fallback
    if args.port and args.rapid_mlx_port == 8000:
        args.rapid_mlx_port = args.port

    port_map = {
        "rapid-mlx": args.rapid_mlx_port,
        "ollama": args.ollama_port,
        "llama-cpp": args.llama_cpp_port,
    }

    all_summaries = []

    for engine in engines:
        config = ENGINE_CONFIGS[engine]

        if engine == "mlx-lm":
            print(f"\n>>> Benchmarking mlx-lm (direct)...")
            results = benchmark_mlx_lm_direct(
                args.model,
                num_runs=args.runs,
                max_tokens_short=args.max_tokens_short,
                max_tokens_long=args.max_tokens_long,
            )
        else:
            port = port_map[engine]
            print(f"\n>>> Benchmarking {config['display']} (port {port})...")
            results = benchmark_openai_compatible(
                f"http://localhost:{port}/v1",
                args.model,
                engine_name=config["display"],
                num_runs=args.runs,
                max_tokens_short=args.max_tokens_short,
                max_tokens_long=args.max_tokens_long,
                port=port,
            )

        if results:
            if engine == "mlx-lm":
                results["engine"] = config["display"]
            s = summarize(results)
            print_summary(s)
            all_summaries.append(s)

    # --- Comparison table ---
    if len(all_summaries) > 1:
        print_comparison(all_summaries)

    # --- Save results ---
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
