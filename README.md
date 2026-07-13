<div align="center">
  <a href="https://qmlx.mrzk.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/marzukia/qMLX/main/assets/qmlx-dark.png">
      <img alt="qMLX" src="https://raw.githubusercontent.com/marzukia/qMLX/main/assets/qmlx-light.png" width="260">
    </picture>
  </a>
</div>

<p align="center">
  <strong>Keeping a hybrid 122B warm on a Mac.</strong>
  <br>
  <em>A Qwen-specialised fork of <a href="https://github.com/raullenchai/Rapid-MLX">qMLX</a> for long-context serving of hybrid MoE models on Apple Silicon.</em>
</p>

<p align="center">
  <a href="https://qmlx.mrzk.io">Website</a> &middot;
  <a href="https://github.com/marzukia/qMLX#readme">Docs</a> &middot;
  <a href="https://mrzk.io">Blog</a> &middot;
  <a href="https://charted.mrzk.io">charted</a>
</p>

---

## Why this exists

Qwen3.5-122B-A10B is a hybrid: about 75% of its layers are DeltaNet (recurrent, linear-attention) and 25% are full attention. The recurrent state cannot be rewound to an earlier position, so the standard in-memory prefix cache drops every entry that contains those layers. On this model it misses 100% of the time. In a normal window we measured zero in-memory hits against 109 disk hits.

So the only thing that keeps the model warm is disk KV restore: checkpoint the attention KV to SSD, page it back on the next turn. It is not a fallback here, it is the entire cache. qMLX is that subsystem built properly, plus the fixes needed to make it hold on real agentic-coding traffic.

The result: a follow-up question on a 130,000-token conversation goes from a multi-minute cold prefill to a sub-second restore. Measured on an M3 Ultra, a repeated 32k prompt drops from 88 seconds of prefill to 0.64 seconds, 137x faster.

## What is in it

- **Disk KV checkpoint and restore** for hybrid recurrent + attention MoE caches, with int4 checkpoints dequantised on restore.
- **Matchable-aware disk-cap eviction** so the checkpoint the next turn needs never gets evicted by unmatchable interval writes.
- **Honest, phase-split metrics**: real decode tok/s (decode window only), real prefill throughput (excludes cached tokens), disk-restore hit rate, TTFT. No amortised (prompt+gen)/wall throughput lie.
- **Live divergence logging** that pinpoints the exact token where a prefix-cache match broke, so this class of bug is diagnosable in minutes.

## Design principles

- **Built for the Mac Studio, not portability.** Optimise for Apple Silicon and unified memory. No abstraction tax to keep a CUDA path alive.
- **Hybrid attention and DeltaNet are first-class.** Recurrent state cannot be trimmed like a KV block, so the cache path branches on it explicitly instead of pretending it is KV-only.
- **SSD cache streaming is a first-class tier, not a fallback.** Unified memory is scarce. Reusable context lives on NVMe and streams back, rather than being hoarded in RAM.
- **Specialise for the models you run.** Qwen-first. Breadth is a cost, not a feature.
- **Honest about the concurrency profile.** Single-user, `--max-num-seqs 1`. A component that earns zero hits gets deleted, not tuned.
- **Correctness beats cleverness on the cache path.** A wrong restore does not throw, it corrupts. Verify the token blob byte-for-byte, quarantine bad checkpoints, prove changes on real traffic.
- **Measure on the real box.** Numbers come from an M3 Ultra with real models, not CI that cannot load a 122B.
- **Lean by default.** Minimal dependencies, no cruft.

## Status

Alpha. It runs one model (Qwen3.5-122B-A10B) on one class of machine (M3 Ultra, 96GB+ unified). Qwen-first, and honest about what is built and what is not. Decode slows gradually with context because the dense-attention layers re-read a growing KV each token, but there is no cliff: it stays usable well past 100k tokens on this hardware. Windowed attention to flatten that curve further is on the roadmap.

## Known limitations

- **Interrupting a cold prefill discards it.** A client disconnect or cancel during a long cold prefill (before the first generated token) aborts the request at 0 tokens and throws the prefill work away, so re-sending the same prompt cold-prefills again. Disk restore only helps once a prompt boundary has been checkpointed, so an interrupt-heavy workload pays a full re-prefill per interrupt. Checkpointing partial prefills at chunk boundaries so interrupted prefills retry warm is tracked in [#12](https://github.com/marzukia/qMLX/issues/12).

## Install

```sh
uv add qmlx-serve
```

Or `pip install qmlx-serve`. The PyPI name is `qmlx-serve` because the exact
`qmlx` is blocked as too similar to `mlx`; the import package is still
`vllm_mlx` and the CLI is still `qmlx`.

From source:

```sh
git clone https://github.com/marzukia/qMLX.git
cd qMLX
pip install -e .
```

## Serving

```sh
qmlx serve mlx-community/Qwen3.5-122B-A10B-4bit \
  --text-only --host 0.0.0.0 --port 8095 --max-num-seqs 1 \
  --enable-prefix-cache --kv-disk-checkpoint-interval 256
```

Drop-in OpenAI / Anthropic API, same as upstream. `--text-only` is required: the vision path is incompatible with the hybrid continuous-batching that the cache work depends on.

## Disk KV cache size

The disk KV checkpoint store is the only cache tier, so its size cap sets how much cross-turn reuse survives. It defaults to **100 GiB** and evicts oldest-first once the total crosses the cap, draining to 80% of it before stopping (a high/low-water scheme that avoids thrashing a single eviction at the boundary).

Two environment variables tune it. Both are read at scan time, so you can change them without touching code or restarting for the value to take:

- `QMLX_KV_CHECKPOINT_MAX_BYTES` sets the cap in bytes. Default `107374182400` (100 GiB). Use `214748364800` for 200 GiB, and so on. `0` disables cap eviction entirely (unbounded, only sane if you manage disk yourself).
- `QMLX_KV_CHECKPOINT_LOW_WATER` sets the low-water fraction eviction drains to, between 0 and 1. Default `0.80`.

Checkpoints live under `~/.cache/qmlx/kv_checkpoints/`. The store grows to the cap then holds steady near the low-water mark, so pick a cap that leaves headroom on that volume. The old 20 GiB default was far too small for agentic traffic: it evicted nearly every checkpoint it wrote, so most turns fell back to a cold prefill.

## Recommended sampling

Qwen3.5-122B-A10B ships a `generation_config` of temperature 0.6, top_p 0.95, top_k 20, but no repetition penalty (it defaults to 1.0). With no penalty the model can loop on long generations. Add these server defaults for Qwen's recommended thinking-mode profile plus a mild repetition penalty:

```sh
  --default-temperature 0.6 \
  --default-top-p 0.95 \
  --default-top-k 20 \
  --default-repetition-penalty 1.05
```

These are `--default-*`, so a client can still override any of them per request. Keep the repetition penalty mild (1.05) so it does not degrade code output.

## Credit

Forked from [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). The base engine, the OpenAI/Anthropic API surface, and the MLX serving path are theirs. qMLX adds the hybrid-aware disk restore, the eviction and metrics work, and the Qwen specialisation. We went a different direction on hybrid attention, too fundamental to reconcile in a PR, hence the fork.

## Notes

The package is still imported as `vllm_mlx` and the CLI is still `qmlx`; those are kept as functional identifiers for compatibility. `qmlx_*` metric names, `QMLX_*` env vars, and the `~/.cache/qmlx/` cache path are unchanged for the same reason.
