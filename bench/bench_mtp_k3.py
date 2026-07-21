#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K=3 MTP acceptance rate benchmark (tape rollback).

Tests the new TAPE rollback implementation that unlocks K>=2 on SSM-hybrid
targets. This script measures:

1. Acceptance rates at K=1, K=2, K=3
2. Position-wise acceptance (pos 1, pos 2, pos 3)
3. Overall speedup compared to baseline

Expected results (from oQ4-mtp 122B on marzuki-helium):
- K=1: ~93% acceptance, ~1.6x speedup
- K=2: ~80% acceptance (pos1: 93%, pos2: 67%)
- K=3: ~81% acceptance (pos1: 93%, pos2: 96%, pos3: 81%), ~2x+ speedup

Usage::

    python bench/bench_mtp_k3.py --model mlx-community/Qwen3.5-122B-A10B-oQ4-mtp --max-k 3

Requires the tape rollback patch in cache_patch.py to be active.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from mlx_lm.load import load

from vllm_mlx.spec_decode.mtp.detect import detect_mtp_support
from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step


@dataclass
class PositionStats:
    """Acceptance stats for a specific draft position."""

    attempts: int = 0
    accepted: int = 0

    @property
    def rate(self) -> float:
        return self.accepted / self.attempts if self.attempts > 0 else 0.0


@dataclass
class KRunStats:
    """Stats for a specific K value."""

    k: int
    total_attempts: int = 0
    total_accepted: int = 0
    position_stats: dict[int, PositionStats] = field(default_factory=dict)
    total_tokens: int = 0
    total_time: float = 0.0
    decode_tok_per_sec: float = 0.0

    @property
    def accept_rate(self) -> float:
        return (
            self.total_accepted / self.total_attempts
            if self.total_attempts > 0
            else 0.0
        )

    def record_position_accept(self, pos: int, accepted: bool):
        if pos not in self.position_stats:
            self.position_stats[pos] = PositionStats()
        self.position_stats[pos].attempts += 1
        if accepted:
            self.position_stats[pos].accepted += 1


def run_mtp_benchmark(
    model_path: str,
    max_k: int,
    max_tokens: int = 200,
    num_prompts: int = 3,
):
    """Run MTP benchmark with tape rollback."""

    print(f"Loading model: {model_path}")
    print(f"Max K: {max_k}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Load model
    model, tokenizer = load(model_path)

    # Detect MTP support
    has_mtp = detect_mtp_support(model)
    if not has_mtp:
        print("ERROR: MTP support not detected in model")
        return

    print("MTP support detected ✓")
    print()

    # Test prompts (short, varied)
    prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Explain how attention mechanisms work in transformers",
        "The quick brown fox jumps over the lazy dog",
    ][:num_prompts]

    # Run for each K value
    all_stats: dict[int, KRunStats] = {}

    for k in range(1, max_k + 1):
        print(f"{'=' * 60}")
        print(f"Testing K={k}")
        print(f"{'=' * 60}")

        stats = KRunStats(k=k)
        all_stats[k] = stats

        total_generated = 0
        start_time = time.perf_counter()

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")

            # Generate with MTP
            tokens_generated = 0

            try:
                for token_id, logprob, is_draft in mtp_generate_step(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_tokens // num_prompts,
                    temp=0.0,
                    print_progress=False,
                    # Note: callback tracking needs to be wired through the generator
                ):
                    tokens_generated += 1

            except Exception as e:
                print(f"  Error during generation: {e}")
                continue

            total_generated += tokens_generated

            print(f"  Generated {tokens_generated} tokens")

        # Calculate final stats
        total_time = time.perf_counter() - start_time
        stats.total_tokens = total_generated
        stats.total_time = total_time
        stats.decode_tok_per_sec = total_generated / total_time if total_time > 0 else 0

        print(f"\nK={k} Summary:")
        print(f"  Total tokens: {stats.total_tokens}")
        print(f"  Total time: {stats.total_time:.2f}s")
        print(f"  Decode speed: {stats.decode_tok_per_sec:.2f} tok/s")
        print(f"  Overall accept rate: {stats.accept_rate * 100:.1f}%")

    # Print comparison table
    print(f"\n{'=' * 60}")
    print("K=1 vs K=2 vs K=3 Comparison")
    print(f"{'=' * 60}")

    print(f"\n{'K':<4} {'Speed (tok/s)':<16} {'Accept Rate':<14} {'Speedup':<10}")
    print("-" * 50)

    base_speed = all_stats[1].decode_tok_per_sec if 1 in all_stats else 1.0

    for k in sorted(all_stats.keys()):
        stats = all_stats[k]
        speedup = stats.decode_tok_per_sec / base_speed if base_speed > 0 else 1.0
        print(
            f"{k:<4} {stats.decode_tok_per_sec:<16.2f} {stats.accept_rate * 100:<14.1f}% {speedup:<10.2f}x"
        )

    # Save results
    results = {
        k: {
            "k": stats.k,
            "total_tokens": stats.total_tokens,
            "total_time": stats.total_time,
            "decode_tok_per_sec": stats.decode_tok_per_sec,
            "accept_rate": stats.accept_rate,
            "total_attempts": stats.total_attempts,
            "total_accepted": stats.total_accepted,
        }
        for k, stats in all_stats.items()
    }

    output_file = f"mtp_k{max_k}_bench_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="K=3 MTP acceptance rate benchmark")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-122B-A10B-oQ4-mtp",
        help="Model path or HF repo ID",
    )
    parser.add_argument(
        "--max-k", type=int, default=3, help="Maximum K value to test (1, 2, or 3)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Maximum tokens to generate per run"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=3, help="Number of test prompts"
    )

    args = parser.parse_args()

    run_mtp_benchmark(
        model_path=args.model,
        max_k=args.max_k,
        max_tokens=args.max_tokens,
        num_prompts=args.num_prompts,
    )


if __name__ == "__main__":
    main()
