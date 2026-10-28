# SPDX-License-Identifier: Apache-2.0
"""DFlash block-diffusion speculative decoding for Qwen3.5 / Qwen3.6.

This package wires the **block-diffusion drafter** family (arxiv 2410.04097
"Block Diffusion-based Speculative Decoding") into qmlx's standardized
``--speculative-config`` interface. It sits next to :mod:`vllm_mlx.spec_decode.mtp`
as the second model-side speculative-decode backend the CLI knows about.

Background
----------

DFlash differs from MTP in two ways:

1. **Drafter is a SEPARATE small model** (~0.5 B params) rather than a
   head built into the target — the same target weights can be paired
   with different drafter checkpoints, and the drafter loads through
   its own HF path.
2. **Drafter emits a BLOCK of tokens per forward** (default 16) using
   block diffusion / parallel denoising, not a single token. The
   verifier confirms the whole block in 1 target forward pass with
   arbitrary ``position_ids``. The longest accepted prefix wins; a
   rejection at position ``k`` collapses to a single corrective token
   at position ``k`` (lossless to the no-spec-decode path).

Public surface
--------------

* :class:`DFlashAcceptCounter` — process-local attempts/accepts/
  tokens-saved counter. Surfaced via
  ``qmlx_spec_decode_dflash_*_total`` in
  :mod:`vllm_mlx.routes.metrics`.
* :func:`detect_dflash_eligibility` — returns
  :class:`DFlashEligibility` for a (model_type, alias_name) pair.
* :func:`get_dflash_drafter_path` — resolves the drafter HF path for
  an alias from the side-registry (kept OUT of ``aliases.json`` because
  the closed-key schema reserves :attr:`AliasProfile.dflash_draft_model`
  for the mlx-vlm DFlash bridge in :mod:`.runtime` /
  :mod:`.eligibility`).
* :func:`load_runtime` / :class:`DFlashRuntime` — the lazy mlx-vlm
  drafter-bridge handle (``pip install 'qmlx-serve[dflash]'``).
* :func:`check` / :class:`DFlashUnavailable` — alias-level eligibility
  gates (declared support, not-MoE, ≥8-bit, drafter declared).

Architecture note (two DFlash code paths, one package)
-----------------------------------------------------

Two DFlash code paths coexist in this package and target different
runtimes (the former ``vllm_mlx.speculative.dflash`` tree was
consolidated here in the 2026-10-28 repositioning wave):

* :mod:`.server` — the dedicated single-user DFlash server (the original
  PoC bridge) that calls into ``mlx-vlm 0.5.0+``'s built-in drafter
  loader via :func:`load_runtime`. It is the LIVE serve path: ``qmlx
  serve <alias> --speculative-config '{"method":"dflash"}'`` forks to
  :func:`.server.run_dflash_server` in the CLI before BatchedEngine
  starts. Its single-thread executor is documented in
  :mod:`.server` (executor ownership: process-lifetime, server-module
  owned, thread-local GPU-stream affinity).
* :mod:`.drafter` / :mod:`.verifier` / :mod:`.generator` — the shared
  eligibility/counter scaffolding used by the standardized config
  surface. BatchedEngine DFlash generation remains deferred; production
  DFlash still uses the single-user server above. The engine-integrated
  :class:`.drafter.MlxVlmDFlashDriver` loads its runtime through the
  SAME :func:`load_runtime` bridge — the two paths share one
  mlx-vlm boundary.

Both paths share the lossless contract: byte-identical output to the
no-spec-decode path (rejection always emits the verify model's pred at
the divergent position, not the drafter's).
"""

from __future__ import annotations

from .accept_counter import (
    DFlashAcceptCounter,
    DFlashAcceptSnapshot,
    get_global_counter,
    reset_global_counter_for_tests,
)
from .detect import DFlashEligibility, detect_dflash_eligibility
from .drafter_registry import (
    clear_drafter_registry_for_tests,
    get_dflash_drafter_path,
    register_dflash_drafter,
)

# Consolidated from the former ``vllm_mlx.speculative.dflash`` package
# (2026-10-28 repositioning): the alias-level eligibility gates and the
# lazy mlx-vlm runtime bridge. Kept at package level so
# ``from vllm_mlx.spec_decode.dflash import DFlashUnavailable, check,
# load_runtime`` works exactly like the old
# ``from vllm_mlx.speculative.dflash import ...`` did.
from .eligibility import DFlashUnavailable, check
from .runtime import DFlashRuntime, load_runtime

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DFlashAcceptCounter",
    "DFlashAcceptSnapshot",
    "DFlashEligibility",
    "DFlashRuntime",
    "DFlashUnavailable",
    "check",
    "clear_drafter_registry_for_tests",
    "detect_dflash_eligibility",
    "get_dflash_drafter_path",
    "get_global_counter",
    "load_runtime",
    "register_dflash_drafter",
    "reset_global_counter_for_tests",
]

# Default block size — the value the paper bench uses (16). Surfaced
# as a module constant so the CLI banner, the metrics gauge, and the
# verifier can all read from a single SSOT. Operators that want a
# smaller "prototype block size 4" can override at SchedulerConfig
# construction; the verifier validates positive int >= 1.
DEFAULT_BLOCK_SIZE = 16
