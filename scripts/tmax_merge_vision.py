#!/usr/bin/env python3
"""
Vision-tower merge tool for Tmax (Qwen3.5 multimodal) MLX snapshots.

Inverse of the removed vision-strip script: takes a text-only (or
vision-stripped) MLX snapshot plus an HF base that carries the vision
tower, and merges the vision-tower tensors + the multimodal config
entries back into the snapshot so ``mlx_lm`` can load it as a
vision-capable multimodal model (issue #56, scope 1: vision-capable
MLX builds).

Operations:

1. Collect every tensor whose key starts with a probed vision prefix
   (default: ``vision_tower.``, ``model.visual.``, ``visual.``) from the
   HF base's safetensors shards.
2. Append them to the MLX snapshot as a dedicated
   ``model-vision.safetensors`` shard (a new shard is always valid for
   ``mlx_lm``'s shard-globbing loader; existing shards are left untouched).
3. If ``model.safetensors.index.json`` exists, add the new keys to the
   weight_map and recompute ``metadata.total_size``.
4. Restore the multimodal config from the HF base's ``config.json``:
   ``vision_config``, ``image_token_id``, ``video_token_id``,
   ``vision_start_token_id``, ``vision_end_token_id``, and rewrite
   ``architectures`` ``Qwen3_5ForCausalLM`` ->
   ``Qwen3_5ForConditionalGeneration`` (the multimodal form). A stray
   ``language_model_only`` hint is dropped.

Usage:
    python scripts/tmax_merge_vision.py <mlx_snapshot_dir> <hf_base_dir>
        [--prefix vision_tower. ...] [--force]

Notes:
- Vision tensors are copied verbatim from the HF base (bf16/fp16). The
  text-LLM side of the snapshot keeps its MLX-quantised form; the vision
  tower loads unquantised, matching how mlx_lm serves VLM variants.
- Idempotent: re-running is a no-op (the existing vision keys win) unless
  ``--force`` rewrites the dedicated vision shard with the HF base's copy.

The script is safe to run on a snapshot that already carries a vision
tower: the default mode skips the shard write and only tops up missing
config keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as save_numpy  # type: ignore
except ImportError:  # pragma: no cover
    safe_open = None
    save_numpy = None

DEFAULT_VISION_PREFIXES = ("vision_tower.", "model.visual.", "visual.")
VISION_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)
TEXT_ARCH = "Qwen3_5ForCausalLM"
VISION_ARCH = "Qwen3_5ForConditionalGeneration"
VISION_SHARD_NAME = "model-vision.safetensors"


def _key_has_vision_prefix(key: str, prefixes: Iterable[str]) -> bool:
    return any(key.startswith(p) for p in prefixes)


def _iter_shard_keys(shard_path: Path) -> list[str]:
    with safe_open(str(shard_path), framework="numpy") as f:
        return list(f.keys())


def _snapshot_vision_keys(mlx_dir: Path, prefixes: Iterable[str]) -> set[str]:
    """Vision-prefixed keys already present in the MLX snapshot."""
    found: set[str] = set()
    for shard in sorted(mlx_dir.glob("*.safetensors")):
        if VISION_SHARD_NAME in shard.name:
            continue  # our own output shard; handled separately
        for k in _iter_shard_keys(shard):
            if _key_has_vision_prefix(k, prefixes):
                found.add(k)
    own = mlx_dir / VISION_SHARD_NAME
    if own.exists():
        for k in _iter_shard_keys(own):
            if _key_has_vision_prefix(k, prefixes):
                found.add(k)
    return found


def _collect_vision(hf_dir: Path, prefixes: Iterable[str]) -> dict[str, object]:
    """Collect all vision-prefixed tensors from the HF base's shards."""
    out: dict[str, object] = {}
    for shard in sorted(hf_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="numpy") as f:
            for k in f.keys():  # noqa: SIM118 - safe_open is not iterable
                if _key_has_vision_prefix(k, prefixes):
                    out[k] = f.get_tensor(k)
    return out


def _restore_config(mlx_cfg: Path, hf_cfg: dict) -> bool:
    """Copy the multimodal config entries from the HF base's config."""
    if not mlx_cfg.exists():
        return False
    cfg = json.loads(mlx_cfg.read_text())
    changed = False
    for k in VISION_CONFIG_KEYS:
        if k in hf_cfg and cfg.get(k) != hf_cfg[k]:
            cfg[k] = hf_cfg[k]
            changed = True
    archs = cfg.get("architectures") or []
    new_archs = [VISION_ARCH if a == TEXT_ARCH else a for a in archs]
    if not new_archs and TEXT_ARCH not in archs:
        # No architectures declared at all: the snapshot is the text form
        # by construction, so declare the multimodal architecture.
        new_archs = [VISION_ARCH]
    elif not new_archs:
        new_archs = [VISION_ARCH]
    if new_archs != archs:
        cfg["architectures"] = new_archs
        changed = True
    if "language_model_only" in cfg:
        cfg.pop("language_model_only")
        changed = True
    if changed:
        mlx_cfg.write_text(json.dumps(cfg, indent=2))
    return changed


def _update_index(
    index_path: Path, mlx_dir: Path, keys: list[str], shard_name: str
) -> bool:
    """Add the merged keys to the weight_map and recompute total_size."""
    idx = json.loads(index_path.read_text())
    wm = idx.setdefault("weight_map", {})
    changed = False
    for k in keys:
        if wm.get(k) != shard_name:
            wm[k] = shard_name
            changed = True
    # Recompute total_size from on-disk shard sizes (mirrors the strip
    # tool's bookkeeping so both directions stay consistent).
    total = 0
    seen: set[str] = set()
    for shard in wm.values():
        if shard in seen:
            continue
        seen.add(shard)
        total += (mlx_dir / shard).stat().st_size
    if "metadata" not in idx or idx.get("metadata", {}).get("total_size") != total:
        idx.setdefault("metadata", {})["total_size"] = total
        changed = True
    if changed:
        index_path.write_text(json.dumps(idx, indent=2))
    return changed


def merge(
    mlx_dir: Path,
    hf_dir: Path,
    prefixes: Iterable[str] = DEFAULT_VISION_PREFIXES,
    force: bool = False,
) -> dict:
    """Merge the HF base's vision tower into the MLX snapshot."""
    mlx_dir = Path(mlx_dir)
    hf_dir = Path(hf_dir)
    if safe_open is None or save_numpy is None:
        raise RuntimeError("safetensors (with numpy) not installed")
    prefixes = tuple(prefixes)

    report: dict = {"mlx": str(mlx_dir), "hf": str(hf_dir), "shard": None}

    vision = _collect_vision(hf_dir, prefixes)
    report["hf_vision_keys"] = len(vision)
    if not vision:
        report["status"] = "no-vision-tensors-in-hf-base"
        return report

    existing = _snapshot_vision_keys(mlx_dir, prefixes)
    report["snapshot_had_vision"] = len(existing)
    missing = set(vision) - existing

    need_shard_write = force or bool(missing)
    if need_shard_write:
        # In force mode the dedicated shard is authoritative: only the
        # HF-base's copy is kept, so drop any snapshot-native vision keys
        # from the dedicated shard's key set (they were written by an
        # earlier merge and are about to be overwritten).
        if force:
            to_write = dict(vision)
        else:
            # Keep snapshot-native vision keys (if any) + add the missing
            # ones from the HF base.
            to_write = dict(vision)
            own = mlx_dir / VISION_SHARD_NAME
            if own.exists() and existing:
                with safe_open(str(own), framework="numpy") as f:
                    for k in f.keys():  # noqa: SIM118 - safe_open is not iterable
                        if _key_has_vision_prefix(k, prefixes) and k not in to_write:
                            to_write[k] = f.get_tensor(k)
        tmp = mlx_dir / (VISION_SHARD_NAME + ".tmp")
        save_numpy(to_write, str(tmp))
        shutil.move(str(tmp), str(mlx_dir / VISION_SHARD_NAME))
        report["shard"] = VISION_SHARD_NAME
        report["shard_keys"] = len(to_write)
    else:
        report["status"] = "already-merged"

    # Config: always top up (cheap, idempotent).
    hf_cfg_path = hf_dir / "config.json"
    report["config_changed"] = False
    if hf_cfg_path.exists():
        hf_cfg = json.loads(hf_cfg_path.read_text())
        report["config_changed"] = _restore_config(mlx_dir / "config.json", hf_cfg)

    idx = mlx_dir / "model.safetensors.index.json"
    report["index_changed"] = False
    if idx.exists() and report.get("shard"):
        report["index_changed"] = _update_index(
            idx, mlx_dir, list(vision), VISION_SHARD_NAME
        )

    report.setdefault("status", "merged" if need_shard_write else "already-merged")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("mlx_snapshot_dir")
    p.add_argument("hf_base_dir")
    p.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Vision tensor key prefix (repeatable). Default: vision_tower. model.visual. visual.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing vision shard with the HF base's copy.",
    )
    args = p.parse_args(argv)
    prefixes = args.prefix or list(DEFAULT_VISION_PREFIXES)
    rep = merge(
        Path(args.mlx_snapshot_dir), Path(args.hf_base_dir), prefixes, args.force
    )
    print(json.dumps(rep, indent=2))
    return 0 if rep.get("hf_vision_keys") else 1


if __name__ == "__main__":
    sys.exit(main())
