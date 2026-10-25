#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Issue #53 d=3 (K=3) gate harness — byte-exact MTP spec decode through
tape rollback.

This is the gating harness for byte-exact d=3 speculative decoding on
SSM-hybrid targets (Qwen3.5/3.6). It exercises the three contracts the
issue calls out:

* **G1 LOSSLESS-D3** — at ``temp=0``, a K=0 (no drafter) run and a
  K=3 (chain-of-3 through tape rollback) run must emit
  byte-identical token streams on the same prompt. Any SSM-state
  mis-restoration on a rejected draft shows up as a divergence.
* **G2 SEEDED-TEMP** — the OpenAI ``seed`` parameter must reach
  ``mtp_generate_step`` and drive every PRNG draw: two K=3 ``temp>0``
  runs with the same seed produce byte-identical streams, and a
  different seed produces a different stream.
* **G2b K0-VS-K3-DIST** (record-only) — issue #53 §3 option A:
  same-seed K=0 vs K=3 per-position distributional comparison at
  ``temp>0``. Records the per-position stream diff and the token
  multiset overlap WITHOUT a pass/fail tolerance gate (Phase 0 of the
  issue: record-only satisfies option A), and runs on the
  deterministic mocks so it works on hydrogen without real weights.
* **G3 ACCEPT TABLE** — per-position acceptance (positions 1..3) and
  decode tok/s at K=1/2/3, derived from the flat generator yield
  stream (the harness does not instrument the loop). Informational by
  default; gate with ``--min-accept-rate`` if desired.

Two modes:

* **real** (default): loads a Qwen3.5 checkpoint + MTP sidecar via
  ``inject_mtp_support`` and runs the gates on the injected inner
  ``TextModel``::

      python bench/bench_mtp_k3.py \\
          --model mlx-community/Qwen3.5-9B-4bit \\
          --sidecar mlx-community/Qwen3.5-9B-MTP-4bit

* **self-test**: runs the gates against inlined deterministic mock
  models (only ``mlx`` required, no download, no pytest) — validates
  the harness itself (and the gate machinery) on any machine with
  mlx::

      python bench/bench_mtp_k3.py --self-test

Exit status is 0 when every gate passes and 1 otherwise, so the
harness can be wired into a release checklist.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx  # noqa: E402

from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter  # noqa: E402
from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import (  # noqa: E402
    reset_controllers,
)
from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step  # noqa: E402

_DEFAULT_PROMPT = (
    "Write a Python function that implements an LRU cache with a max size. "
    "Include type hints, docstrings, and handle edge cases. "
    "Also write comprehensive unit tests for it.\n\n"
    "```python\n"
)

# (name, model, prompt_ids, max_tokens, temp, seed) run spec
_RUN = tuple


def run_stream(
    model,
    prompt_ids: mx.array,
    max_tokens: int,
    k: int,
    temp: float,
    seed: int | None = None,
    top_p: float | None = None,
) -> tuple[list[int], list[bool], float]:
    """One ``mtp_generate_step`` run at fixed depth ``k``.

    Returns ``(token_ids, from_draft_flags, wall_seconds)``. Each run
    gets a fresh accept counter and a cleared controller registry so
    runs never inherit acceptance-history state from a previous run.
    """
    counter = MTPAcceptCounter()
    reset_controllers()
    start = time.perf_counter()
    toks: list[int] = []
    flags: list[bool] = []
    kwargs: dict = dict(
        max_tokens=max_tokens,
        temp=temp,
        accept_counter=counter,
        max_k=k if k > 0 else 3,
        fixed_k=k,
        seed=seed,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p
    for tok, _lp, from_draft in mtp_generate_step(prompt_ids, model, **kwargs):
        toks.append(int(tok))
        flags.append(bool(from_draft))
        if len(toks) >= max_tokens:
            break
    wall = time.perf_counter() - start
    return toks, flags, wall


def parse_rounds(flags: list[bool], k: int) -> dict[int, tuple[int, int]]:
    """Derive per-position accept stats from the flat yield stream.

    Round structure of the generator's yield stream: a K=0 round emits
    one ``from_draft=False``; a verify round emits ``accepted_count``
    consecutive ``from_draft=True`` entries followed by exactly one
    ``from_draft=False`` (bonus on all-accept, residual on reject).
    A trailing ``True`` run without a closing ``False`` means the run
    hit ``max_tokens`` mid-round — it still counts as accepts.

    Returns ``{position: (attempts, accepted)}`` for positions
    ``1..k``: position ``p`` was attempted in every verify round
    (``k`` drafts each) and accepted iff the round's accept run length
    reached ``p``.
    """
    attempts = {p: 0 for p in range(1, k + 1)}
    accepted = {p: 0 for p in range(1, k + 1)}

    def count_run(run_len: int) -> None:
        if run_len <= 0:
            return
        for p in range(1, k + 1):
            attempts[p] += 1
            if run_len >= p:
                accepted[p] += 1

    run_len = 0
    for from_draft in flags:
        if from_draft:
            run_len += 1
        else:
            count_run(run_len)
            run_len = 0
    count_run(run_len)  # stream ended mid-round at max_tokens
    return {p: (attempts[p], accepted[p]) for p in range(1, k + 1)}


def gate_g1_lossless_d3(
    make_model, prompt_ids: mx.array, max_tokens: int
) -> tuple[bool, str, list[int], list[int]]:
    """G1: K=0 vs K=3 byte identity at temp=0."""
    base, _, _ = run_stream(
        make_model(), prompt_ids, max_tokens, k=0, temp=0.0
    )
    spec, _, _ = run_stream(
        make_model(), prompt_ids, max_tokens, k=3, temp=0.0
    )
    ok = base == spec
    msg = (
        f"G1 LOSSLESS-D3: {'PASS' if ok else 'FAIL'} — K=0 emitted "
        f"{len(base)} tokens, K=3 emitted {len(spec)}; "
        + (
            "streams byte-identical."
            if ok
            else f"first divergence: K0={base} K3={spec}"
        )
    )
    return ok, msg, base, spec


def gate_g2_seeded_temp(
    make_model, prompt_ids: mx.array, max_tokens: int, temp: float
) -> tuple[bool, str]:
    """G2: seeded temp>0 determinism (same seed ⇒ same stream; a
    different seed ⇒ different stream)."""
    a, _, _ = run_stream(
        make_model(), prompt_ids, max_tokens, k=3, temp=temp, seed=42, top_p=0.9
    )
    b, _, _ = run_stream(
        make_model(), prompt_ids, max_tokens, k=3, temp=temp, seed=42, top_p=0.9
    )
    c, _, _ = run_stream(
        make_model(), prompt_ids, max_tokens, k=3, temp=temp, seed=7, top_p=0.9
    )
    reproducible = a == b
    divergent = c != a
    ok = reproducible and divergent and len(a) > 0
    msg = (
        f"G2 SEEDED-TEMP: {'PASS' if ok else 'FAIL'} — "
        f"seed=42 reproducible: {reproducible}; "
        f"seed=7 diverges: {divergent}; stream len: {len(a)}"
    )
    if not reproducible:
        msg += f"\n  seed=42 run1: {a}\n  seed=42 run2: {b}"
    if not divergent:
        msg += f"\n  seed=42: {a}\n  seed=7:    {c}"
    return ok, msg


def gate_g2b_k0_vs_k3_distribution(
    make_model, prompt_ids: mx.array, max_tokens: int, temp: float, seed: int = 42
) -> tuple[bool, str]:
    """G2b (issue #53 §3 option A, record-only): same-seed K=0 vs K=3
    per-position distributional comparison at temp>0.

    Records — with NO pass/fail tolerance gate, per Phase 0 of the
    issue where record-only satisfies option A — the per-position
    token diff between a K=0 (no drafter) run and a K=3 (tape
    rollback) run at identical seed/temperature/top_p, plus the token
    multiset overlap. Works on the deterministic mock models, so it
    runs on hydrogen without real weights (real mode compares the
    injected inner model against itself at two depths).
    """
    from collections import Counter

    k0, _, _ = run_stream(
        make_model(), prompt_ids, max_tokens, k=0, temp=temp, seed=seed, top_p=0.9
    )
    k3, _, _ = run_stream(
        make_model(), prompt_ids, max_tokens, k=3, temp=temp, seed=seed, top_p=0.9
    )
    n = min(len(k0), len(k3))
    diff_positions = [i for i in range(n) if k0[i] != k3[i]]
    match = n - len(diff_positions)
    c0, c3 = Counter(k0[:n]), Counter(k3[:n])
    overlap = sum((c0 & c3).values())
    pct = lambda m: f"{m / n * 100:.1f}%" if n else "n/a"  # noqa: E731
    msg = (
        f"G2b K0-VS-K3-DIST (record-only, option A): seed={seed} "
        f"temp={temp} — K=0 len={len(k0)}, K=3 len={len(k3)}; "
        f"per-position identity: {match}/{n} ({pct(match)}); "
        f"token multiset overlap: {overlap}/{n} ({pct(overlap)}) "
        "(no tolerance gate — Phase 0 record-only)"
    )
    if diff_positions:
        first = ", ".join(f"pos={i} K0={k0[i]} K3={k3[i]}" for i in diff_positions[:5])
        msg += f"\n  first diffs: {first}" + (
            f" (+{len(diff_positions) - 5} more)" if len(diff_positions) > 5 else ""
        )
    else:
        msg += "\n  streams are per-position identical at this seed"
    return True, msg


def gate_g3_accept_table(
    make_model, prompt_ids: mx.array, max_tokens: int, min_accept_rate: float | None
) -> tuple[bool, str, dict[int, dict]]:
    """G3: per-position acceptance + tok/s at K=1/2/3 (report table)."""
    results: dict[int, dict] = {}
    for k in (1, 2, 3):
        toks, flags, wall = run_stream(
            make_model(), prompt_ids, max_tokens, k=k, temp=0.0
        )
        stats = parse_rounds(flags, k)
        n_tokens = len(toks)
        results[k] = {
            "tokens": n_tokens,
            "tok_per_sec": n_tokens / wall if wall > 0 else 0.0,
            "wall_seconds": wall,
            "per_position": {
                p: {
                    "attempts": a,
                    "accepted": c,
                    "rate": (c / a if a else 0.0),
                }
                for p, (a, c) in stats.items()
            },
        }
    # Overall accept rate across positions per K (attempts-weighted).
    for k, r in results.items():
        att = sum(v["attempts"] for v in r["per_position"].values())
        acc = sum(v["accepted"] for v in r["per_position"].values())
        r["overall_accept_rate"] = acc / att if att else 0.0

    ok = True
    if min_accept_rate is not None:
        for k, r in results.items():
            if r["overall_accept_rate"] < min_accept_rate:
                ok = False
    return ok, "", results


def render_g3(results: dict[int, dict]) -> str:
    lines = [
        "",
        "G3 ACCEPT TABLE (temp=0, from flat yield stream):",
        "",
        "| K | tok/s   | overall accept | pos 1    | pos 2    | pos 3    |",
        "|---|---------|----------------|----------|----------|----------|",
    ]
    for k in sorted(results):
        r = results[k]
        pp = r["per_position"]
        cells = []
        for p in range(1, k + 1):
            s = pp.get(p)
            if s is None or s["attempts"] == 0:
                cells.append(f"{'—':<8}")
            else:
                cells.append(f"{s['rate'] * 100:5.1f}%  ")
        lines.append(
            f"| {k} | {r['tok_per_sec']:>7.2f} | "
            f"{r['overall_accept_rate'] * 100:>14.1f}% | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------


def load_real_model(model: str, sidecar: str | None, model_repo: str | None):
    """Load base + inject MTP sidecar; return the injected inner TextModel."""
    from mlx_lm import load as mlx_load

    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    print(f"Loading model: {model}")
    model_obj, tokenizer = mlx_load(model)
    inject_mtp_support(
        model_obj,
        mtp_sidecar=sidecar,
        model_repo=model_repo or model,
    )
    validate_mtp_support(model_obj)
    inner = model_obj.language_model if hasattr(model_obj, "language_model") else model_obj
    return inner, tokenizer


def load_self_test_model(max_tokens: int):
    """Deterministic mock model FACTORIES (no download) for the
    self-test mode.

    The mocks are inlined here (not imported from the test suite) so
    the harness needs only ``mlx`` — no pytest. They mirror
    ``_MockedQwen35Model`` / ``_SeededMockModel`` in
    ``tests/test_mtp_spec_decode.py``. The mock models are stateful
    (script cursors), so each gate run gets a fresh instance — a
    factory per run keeps the K=0 and K=3 streams comparable.
    """

    class _MockedQwen35Model:
        """Greedy-capable mock: per-position scripted token ids."""

        def __init__(
            self, backbone_outputs, mtp_outputs, vocab=512, hidden_size=8
        ):
            self._backbone = list(backbone_outputs)
            self._mtp = list(mtp_outputs)
            self._backbone_cursor = 0
            self._mtp_cursor = 0
            self.vocab = vocab
            self.hidden_size = hidden_size
            self.layers = []

        def _logits_for_positions(self, target_ids, batch):
            out_rows = []
            for tid in target_ids:
                row = mx.zeros((batch, self.vocab))
                row = row + mx.where(
                    mx.arange(self.vocab)[None, :] == tid,
                    mx.array(50.0),
                    mx.array(0.0),
                )
                out_rows.append(row)
            return mx.stack(out_rows, axis=1)

        def __call__(
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden=False,
            n_confirmed=0,
        ):
            B, S = inputs.shape
            targets = []
            for _ in range(S):
                if self._backbone_cursor < len(self._backbone):
                    targets.append(self._backbone[self._backbone_cursor])
                    self._backbone_cursor += 1
                else:
                    targets.append(0)
            logits = self._logits_for_positions(targets, B)
            hidden = mx.zeros((B, S, self.hidden_size))
            if return_hidden:
                return logits, hidden
            return logits

        def mtp_forward(self, hidden, next_token_ids, mtp_cache):
            B = next_token_ids.shape[0]
            S = next_token_ids.shape[1]
            targets = []
            for _ in range(S):
                if self._mtp_cursor < len(self._mtp):
                    targets.append(self._mtp[self._mtp_cursor])
                    self._mtp_cursor += 1
                else:
                    targets.append(0)
            return self._logits_for_positions(targets, B)

        def make_mtp_cache(self):
            return []

    class _SeededMockModel:
        """Soft (biased) logits so temp>0 sampling is stochastic."""

        def __init__(self, backbone_outputs, mtp_outputs, vocab=512, hidden_size=8):
            self._backbone = list(backbone_outputs)
            self._mtp = list(mtp_outputs)
            self._backbone_cursor = 0
            self._mtp_cursor = 0
            self.vocab = vocab
            self.hidden_size = hidden_size
            self.layers = []

        def _soft_logits(self, entries, batch):
            V = self.vocab
            rows = []
            for tid, p in entries:
                row = mx.full((V,), float((1.0 - p) / (V - 1))).astype(mx.float32)
                row = row.at[tid].add(mx.log(float(p)))
                rows.append(row[None, :])
            return mx.stack(rows, axis=1)

        def _consume(self, script, cursor, n):
            out = []
            for _ in range(n):
                if cursor < len(script):
                    out.append(script[cursor])
                else:
                    out.append((0, 0.5))
            return out

        def __call__(
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden=False,
            n_confirmed=0,
        ):
            B, S = inputs.shape
            entries = self._consume(self._backbone, self._backbone_cursor, S)
            self._backbone_cursor += S
            logits = self._soft_logits(entries, B)
            hidden = mx.zeros((B, S, self.hidden_size))
            if return_hidden:
                return logits, hidden
            return logits

        def mtp_forward(self, hidden, next_token_ids, mtp_cache):
            B = next_token_ids.shape[0]
            S = next_token_ids.shape[1]
            entries = self._consume(self._mtp, self._mtp_cursor, S)
            self._mtp_cursor += S
            return self._soft_logits(entries, B)

        def make_mtp_cache(self):
            return []

    # G1 mock: a long backbone script; the MTP script is the backbone
    # tail, so every chain-of-3 verify all-accepts and the K=3 stream
    # collapses onto the K=0 stream (byte identity by construction —
    # the gate still validates the loop, the rollback, and the tape).
    backbone = list(range(10, 10 + max_tokens + 8))

    def make_greedy():
        return _MockedQwen35Model(
            backbone_outputs=list(backbone), mtp_outputs=list(backbone[1:])
        )

    # G2 mock: soft (stochastic) logits — same shape as the seeded
    # contract test in tests/test_mtp_spec_decode.py.
    backbone_soft = [(0, 0.5)] + [(t, 0.5) for t in range(21, 21 + max_tokens + 8)]
    mtp_soft = [(t, 0.9) for t in range(31, 31 + max_tokens + 12)]

    def make_seeded():
        return _SeededMockModel(list(backbone_soft), list(mtp_soft))

    return make_greedy, make_seeded, mx.array([1], dtype=mx.uint32)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Issue #53 d=3 (K=3) gate harness"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-9B-4bit",
        help="Base model alias or path (real mode).",
    )
    parser.add_argument(
        "--sidecar",
        default="mlx-community/Qwen3.5-9B-MTP-4bit",
        help="MTP sidecar alias or path (real mode).",
    )
    parser.add_argument(
        "--model-repo",
        default=None,
        help="HF repo for embedded-MTP auto-detection (oQ4-mtp etc.).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the gates against deterministic mock models "
        "(no model download).",
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_PROMPT,
        help="Prompt text (real mode).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Tokens to generate per run.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Temperature for the G2 seeded-temp runs.",
    )
    parser.add_argument(
        "--min-accept-rate",
        type=float,
        default=None,
        help="If set, G3 gates on overall accept rate at every K "
        "(0.0–1.0). Informational by default.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        make_greedy, make_seeded, prompt_ids = load_self_test_model(
            args.max_tokens
        )
        prompt_desc = "mock prompt (1 token)"
    else:
        inner, tokenizer = load_real_model(
            args.model, args.sidecar or None, args.model_repo
        )

        def make_greedy():  # noqa: E306
            return inner

        def make_seeded():  # noqa: E306
            return inner

        prompt_ids = mx.array(tokenizer.encode(args.prompt), mx.uint32)
        prompt_desc = f"{len(prompt_ids)} prompt tokens"

    print(f"Prompt: {prompt_desc}")
    print(f"Max tokens per run: {args.max_tokens}")
    print()

    failures = 0

    # G1 on the greedy-capable model (mock: _MockedQwen35Model; real:
    # the injected inner model).
    ok, msg, _, _ = gate_g1_lossless_d3(make_greedy, prompt_ids, args.max_tokens)
    print(msg)
    failures += 0 if ok else 1

    # G2 on the stochastic-capable model (mock: _SeededMockModel;
    # real: the same injected inner model).
    ok, msg = gate_g2_seeded_temp(make_seeded, prompt_ids, args.max_tokens, args.temp)
    print(msg)
    failures += 0 if ok else 1

    # G2b (record-only, issue #53 §3 option A) on the stochastic
    # model — always passes; the record is the point.
    _ok, msg = gate_g2b_k0_vs_k3_distribution(
        make_seeded, prompt_ids, args.max_tokens, args.temp
    )
    print(msg)

    # G3 on the greedy model.
    ok, _msg, results = gate_g3_accept_table(
        make_greedy, prompt_ids, args.max_tokens, args.min_accept_rate
    )
    print(render_g3(results))
    if args.min_accept_rate is not None and not ok:
        print(
            f"G3 ACCEPT TABLE: FAIL — overall accept rate below "
            f"{args.min_accept_rate:.2f} at one or more K."
        )
        failures += 1

    print()
    if failures == 0:
        print("ALL d=3 GATES PASSED")
        return 0
    print(f"{failures} d=3 GATE(S) FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
