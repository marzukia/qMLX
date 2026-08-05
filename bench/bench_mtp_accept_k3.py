"""
Benchmark MTP draft acceptance rates at K=1, K=2, K=3 with cache trimming.

This is a STANDALONE measurement — it does NOT implement SSM rollback.
For ArraysCache (GatedDeltaNet), we do nothing on trim; for KVCache we trim.
This means accept rates at K>1 are slightly pessimistic (stale SSM state),
but still directional and useful for deciding whether to invest in tape rollback.

Accept rates at K=1 are baseline-accurate (no multi-step SSM divergence).
Accept rates at K=2 are slightly pessimistic (SSM state after trim is stale).
Accept rates at K=3 are more pessimistic (compounding staleness).

If K=3 accept > 50% even with degraded state, tape rollback is DEFINITELY worth building.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as _cache_module

from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
    inject_mtp_support,
    validate_mtp_support,
)


def measure_accept_at_k(model, tokenizer, prompt_ids, depth=1, max_rounds=50):
    """Measure MTP acceptance rates at depth K.

    Output is NOT correct (no proper SSM rollback) — we only care about accept rates.
    """

    model_cache = _cache_module.make_prompt_cache(model)
    mtp_cache = model.make_mtp_cache()

    # Prefill: run entire prompt through backbone
    prefill_size = 2048
    for i in range(0, len(prompt_ids), prefill_size):
        chunk = prompt_ids[i : i + prefill_size]
        if len(chunk) <= 1:
            break
        # Leave last token for the decode loop
        if i + prefill_size >= len(prompt_ids):
            yy = chunk[:-1]
        else:
            yy = chunk
        if len(yy) == 0:
            break
        logits, hidden = model(yy[None], cache=model_cache, return_hidden=True)
        # Also run MTP forward to populate MTP cache
        if len(chunk) > 1:
            model.mtp_forward(hidden[:, :-1, :], yy[1:][None], mtp_cache)
        mx.eval(hidden)

    # Start decode from the last prompt token
    y = prompt_ids[-1:].astype(mx.uint32)

    accepted_total = 0
    drafted_total = 0
    per_position = {}  # position -> [accepted, total]

    for round_idx in range(max_rounds):
        # Step 1: Backbone forward on current token
        logits, hidden = model(y[None], cache=model_cache, return_hidden=True)
        main_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(main_tok)

        hidden_last = hidden[:, -1:, :]

        # Step 2: Draft K tokens via MTP head (cascade with drafter hidden)
        draft_ids = []
        prev_tok = main_tok
        cur_hidden = hidden_last

        for d in range(depth):
            mtp_logits, mtp_hidden = model.mtp_forward(
                cur_hidden, prev_tok.reshape(1, 1).astype(mx.uint32), mtp_cache,
                return_hidden=True,
            )
            draft_tok = mx.argmax(mtp_logits[:, -1, :], axis=-1)
            mx.eval(draft_tok, mtp_hidden)
            draft_ids.append(draft_tok.item())

            # Cascade: feed drafter's own hidden into next iteration
            prev_tok = draft_tok
            cur_hidden = mtp_hidden  # drafter's representation, not backbone's

        # Step 3: Verify — run backbone on [main_tok, draft_1, ..., draft_K]
        verify_ids = mx.array([main_tok.item()] + draft_ids, mx.uint32)

        v_logits, v_hidden = model(
            verify_ids[None],
            cache=model_cache,
            return_hidden=True,
            n_confirmed=depth,
        )
        mx.eval(v_logits)

        # Step 4: Check acceptance (greedy, temp=0)
        # v_logits shape: (1, K+1, vocab)
        # position i (0..K-1): target's prediction after seeing verify_ids[0:i+1]
        #   draft_ids[i] is accepted iff argmax(v_logits[0, i, :]) == draft_ids[i]
        n_accepted = 0
        for i in range(depth):
            target_tok = mx.argmax(v_logits[0, i, :], axis=-1)
            mx.eval(target_tok)
            target_id = target_tok.item()
            pos_key = i + 1  # 1-indexed position
            if pos_key not in per_position:
                per_position[pos_key] = [0, 0]
            if target_id == draft_ids[i]:
                n_accepted += 1
                per_position[pos_key][0] += 1
                per_position[pos_key][1] += 1
            else:
                per_position[pos_key][1] += 1
                break

        accepted_total += n_accepted
        drafted_total += depth

        # Step 5: Trim cache for rejected drafts (KVCache only)
        n_to_trim = depth - n_accepted
        if n_to_trim > 0:
            for c in model_cache:
                if hasattr(c, "is_trimmable") and c.is_trimmable():
                    c.trim(n_to_trim)
            for c in mtp_cache:
                if hasattr(c, "is_trimmable") and c.is_trimmable():
                    c.trim(n_to_trim)

        # Next backbone input
        if n_accepted == depth:
            # All accepted — use bonus token from position K
            bonus_tok = mx.argmax(v_logits[0, depth, :], axis=-1)
            mx.eval(bonus_tok)
            y = bonus_tok.reshape(1).astype(mx.uint32)
        else:
            # Rejected — use target's prediction at rejection point
            residual_tok = mx.argmax(v_logits[0, n_accepted, :], axis=-1)
            mx.eval(residual_tok)
            y = residual_tok.reshape(1).astype(mx.uint32)

    return accepted_total, drafted_total, per_position


def main():
    parser = argparse.ArgumentParser(description="MTP acceptance rate benchmark")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-9B-4bit",
        help="Base model alias",
    )
    parser.add_argument(
        "--sidecar",
        default="mlx-community/Qwen3.5-9B-MTP-4bit",
        help="MTP sidecar model alias",
    )
    parser.add_argument(
        "--model-repo",
        default=None,
        help="HF repo for auto-detecting embedded MTP weights (oQ4-mtp etc.)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=50,
        help="Number of decode rounds per K value",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt (default: Python code generation prompt)",
    )
    args = parser.parse_args()

    if args.prompt is None:
        args.prompt = (
            "Write a Python function that implements a LRU cache with a max size. "
            "Include type hints, docstrings, and handle edge cases. "
            "Also write comprehensive unit tests for it.\n\n"
            "```python\n"
        )

    print("=== MTP Acceptance Rate Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Sidecar: {args.sidecar}")
    print(f"Max rounds: {args.max_rounds}")
    print(
        "Note: SSM rollback is approximate at K>1 "
        "(KVCache trimmed, ArraysCache not restored)."
    )
    print(
        "Accept rates at K>1 are lower bounds — "
        "true rates may be slightly higher."
    )
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    sidecar = args.sidecar if args.sidecar else None
    inject_mtp_support(
        model,
        mtp_sidecar=sidecar,
        model_repo=args.model_repo or args.model,
    )
    validate_mtp_support(model)
    if hasattr(model, "language_model"):
        model = model.language_model
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print()

    # Encode prompt
    prompt_ids = mx.array(tokenizer.encode(args.prompt), mx.uint32)
    print(f"Prompt tokens: {len(prompt_ids)}")
    print()

    # Run benchmarks for K=1, K=2, K=3
    results = {}
    for k_val in [1, 2, 3]:
        print(f"--- K={k_val} ---")
        t0 = time.time()
        accepted, drafted, per_pos = measure_accept_at_k(
            model, tokenizer, prompt_ids, depth=k_val, max_rounds=args.max_rounds
        )
        elapsed = time.time() - t0
        rate = accepted / drafted * 100 if drafted > 0 else 0.0
        results[k_val] = {
            "accepted": accepted,
            "drafted": drafted,
            "rate": rate,
            "per_position": per_pos,
            "elapsed": elapsed,
        }
        print(f"  Accept rate: {rate:.1f}%  ({accepted}/{drafted})  [{elapsed:.1f}s]")
        print()

    # Print comparison table
    print()
    print("| K | Accept Rate | Accepted | Drafted |")
    print("|---|------------|----------|---------|")
    for k_val in [1, 2, 3]:
        r = results[k_val]
        print(f"| {k_val} | {r['rate']:>6.1f}%     | {r['accepted']:>6}   | {r['drafted']:>6}  |")

    # Per-position breakdown for K=3
    print()
    print("Per-position acceptance (K=3):")
    per_pos = results[3]["per_position"]
    for pos in sorted(per_pos.keys()):
        acc, total = per_pos[pos]
        pct = acc / total * 100 if total > 0 else 0.0
        print(f"  Position {pos}: {pct:.1f}% ({acc}/{total})")

    # Decision guidance
    print()
    print("Decision guidance:")
    r1 = results[1]["rate"]
    r3 = results[3]["rate"]
    if r1 > 75:
        print(f"  K=1 accept {r1:.1f}% > 75% → Phase 2 (verify_qmv) likely sufficient")
    else:
        print(f"  K=1 accept {r1:.1f}% < 75% → MTP sidecar quality needs investigation")
    if r3 > 50:
        print(f"  K=3 accept {r3:.1f}% > 50% → Phase 3 (tape rollback) is high-value investment")
    elif r3 > 30:
        print(f"  K=3 accept {r3:.1f}% in 30-50% → tape rollback worth exploring, sidecar calibration may help")
    else:
        print(f"  K=3 accept {r3:.1f}% < 30% → Calibrated INT4 MTP sidecar is critical before any other work")


if __name__ == "__main__":
    main()
