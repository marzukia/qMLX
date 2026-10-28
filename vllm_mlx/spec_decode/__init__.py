# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding bundle (R15-P1, task #302).

This package hosts ALL of qmlx's speculative-decoding paths — the
model-side vendored backends plus the mlx-vlm bridge, the dedicated
single-user servers, and the model-free suffix drafter. The former
``vllm_mlx.speculative`` package (prompt-lookup, suffix-decoding,
dflash, ddtree) was consolidated here in the 2026-10-28
repositioning wave; ``vllm_mlx.speculative`` no longer exists.

Currently bundled:

* :mod:`vllm_mlx.spec_decode.mtp` — Native Multi-Token Prediction
  speculative decoding for Qwen3.5 / Qwen3.6 family models, vendored
  from mlx-lm PR #990 (`Native MTP speculative decoding
  (Qwen3.5/3.6 reference implementation)
  <https://github.com/ml-explore/mlx-lm/pull/990>`_, head
  ``50c164fb82bf50d89ec4eb30fc5dc33820b4540f``). The PR is still
  ``OPEN`` upstream as of the vendoring date — we ship the code under
  this namespace rather than waiting on mlx-lm merge so the rest of
  R15-P1 (DFlash-MLX, DDTree-MLX) can build on top of the same accept-
  rate counter / lossless-contract scaffolding.
* :mod:`vllm_mlx.spec_decode.dflash` — Block-diffusion drafter
  speculative decoding for Qwen3.5 / Qwen3.6 (R15-P1 task #313, arxiv
  2410.04097). Builds on the same accept-counter / detect / metrics
  scaffolding as MTP but with a SEPARATE small drafter that emits
  16-token blocks per forward instead of the single-token chain MTP
  proposes. The verifier owns the cache write via the
  :func:`vllm_mlx.positioned_kv_cache.positioned_update_and_fetch`
  HELPER (NOT the subclass — the subclass breaks
  ``mlx_lm.save_prompt_cache``). Also hosts the dedicated single-user
  DFlash server (``run_dflash_server``) that the CLI forks to on
  ``--speculative-config '{"method":"dflash"}'`` — it bypasses
  BatchedEngine and owns a process-lifetime single-thread executor
  (documented in the module).
* :mod:`vllm_mlx.spec_decode.ddtree` — DDTree draft-tree speculative
  decoding (issue #879), kept behind explicit
  ``--speculative-config {"method":"ddtree"}`` opt-in and a dedicated
  single-user server, using the external ``dtree_mlx`` package as the
  runtime boundary. The CLI forks to ``run_ddtree_server`` the same
  way it does for DFlash; the server owns two process-lifetime
  single-thread executors (documented in the module).
* :mod:`vllm_mlx.spec_decode.suffix_decoding` — SuffixDecoding
  (NeurIPS 2025, arxiv 2411.04975): adaptive suffix-tree, draft-model-
  free speculative drafter wired into BatchedEngine's verify loop via
  the scheduler (``--speculative-config {"method":"suffix"}``).
* :mod:`vllm_mlx.spec_decode.config` and
  :mod:`vllm_mlx.spec_decode.registry` — vLLM-style
  ``--speculative-config`` parsing plus the small method registry used
  to migrate DFlash / MTP / SuffixDecoding behind one frontend surface.

Each backend exposes a thin public API the CLI / scheduler can call after
parsing ``--speculative-config`` at boot. The model-side patches that make
a particular family eligible stay private to the sub-package.
"""

from __future__ import annotations

__all__ = ["config", "ddtree", "dflash", "mtp", "registry", "suffix_decoding"]
