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
  <em>A Qwen-specialised fork of <a href="https://github.com/raullenchai/Rapid-MLX">Rapid-MLX</a> for long-context serving of hybrid MoE models on Apple Silicon.</em>
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

## Status

Alpha. It runs one model (Qwen3.5-122B-A10B) on one class of machine (M3 Ultra, 96GB+ unified). Qwen-first, and honest about what is built and what is not. Decode slows gradually with context because the dense-attention layers re-read a growing KV each token, but there is no cliff: it stays usable well past 100k tokens on this hardware. Windowed attention to flatten that curve further is on the roadmap.

## Install

```sh
uv add qmlx-serve
```

Or `pip install qmlx-serve`. The PyPI name is `qmlx-serve` because the exact
`qmlx` is blocked as too similar to `mlx`; the import package is still
`vllm_mlx` and the CLI is still `rapid-mlx`.

From source:

```sh
git clone https://github.com/marzukia/qMLX.git
cd qMLX
pip install -e .
```

## Serving

```sh
rapid-mlx serve mlx-community/Qwen3.5-122B-A10B-4bit \
  --text-only --host 0.0.0.0 --port 8095 --max-num-seqs 1 \
  --enable-prefix-cache --prefix-cache-index radix \
  --enable-disk-kv-restore --kv-disk-checkpoint-interval 256
```

Drop-in OpenAI / Anthropic API, same as upstream. `--text-only` is required: the vision path is incompatible with the hybrid continuous-batching that the cache work depends on.

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

The package is still imported as `vllm_mlx` and the CLI is still `rapid-mlx`; those are kept as functional identifiers for compatibility. `rapid_mlx_*` metric names, `RAPID_MLX_*` env vars, and the `~/.cache/rapid-mlx/` cache path are unchanged for the same reason.
