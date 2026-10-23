# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``scripts/tmax_merge_vision.py`` (issue #56, scope 1).

The merge tool is the inverse of ``tmax_strip_vision.py``: it takes a
text-only MLX snapshot plus an HF base carrying the vision tower and
produces a vision-capable MLX snapshot. These tests pin the shard write,
the config restoration, the index bookkeeping, and the idempotency /
force semantics.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file as save_numpy

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "tmax_merge_vision.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tmax_merge_vision", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def merge_mod():
    return _load_script()


def _write_shard(path: Path, tensors: dict[str, np.ndarray]) -> None:
    save_numpy(tensors, str(path))


def _shard_keys(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():  # noqa: SIM118 - safe_open is not iterable
            out[k] = f.get_tensor(k)
    return out


def _make_mlx_snapshot(dirpath: Path) -> Path:
    """A text-only MLX snapshot: one shard + index + text-form config."""
    dirpath.mkdir(parents=True, exist_ok=True)
    text_w = np.ones((4, 4), dtype=np.float16)
    _write_shard(dirpath / "model-00001.safetensors", {"lm_head.weight": text_w})
    (dirpath / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForCausalLM"],
                "language_model_only": True,
                "model_type": "qwen3_5",
            }
        )
    )
    (dirpath / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"lm_head.weight": "model-00001.safetensors"},
                "metadata": {"total_size": 0},
            }
        )
    )
    return dirpath


def _make_hf_base(dirpath: Path, vision_values: np.ndarray) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    _write_shard(
        dirpath / "base.safetensors",
        {
            "model.visual.blocks.0.attn.qkv.weight": vision_values,
            "lm.weight": np.ones((2, 2), dtype=np.float16),  # non-vision noise
        },
    )
    (dirpath / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "vision_config": {"depth": 2, "hidden_size": 8},
                "image_token_id": 123,
                "video_token_id": 456,
                "vision_start_token_id": 1,
                "vision_end_token_id": 2,
            }
        )
    )
    return dirpath


VISION_KEY = "model.visual.blocks.0.attn.qkv.weight"


def test_merge_writes_vision_shard_and_config(merge_mod, tmp_path):
    mlx = _make_mlx_snapshot(tmp_path / "mlx")
    vision = np.arange(16, dtype=np.float32).reshape(4, 4).astype(np.float16)
    hf = _make_hf_base(tmp_path / "hf", vision)

    rep = merge_mod.merge(mlx, hf)

    assert rep["hf_vision_keys"] == 1
    shard = mlx / "model-vision.safetensors"
    assert shard.exists()
    got = _shard_keys(shard)
    assert set(got) == {VISION_KEY}
    assert np.array_equal(got[VISION_KEY], vision)

    cfg = json.loads((mlx / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert cfg["vision_config"] == {"depth": 2, "hidden_size": 8}
    assert cfg["image_token_id"] == 123
    assert cfg["video_token_id"] == 456
    assert cfg["vision_start_token_id"] == 1
    assert cfg["vision_end_token_id"] == 2
    assert "language_model_only" not in cfg

    idx = json.loads((mlx / "model.safetensors.index.json").read_text())
    assert idx["weight_map"][VISION_KEY] == "model-vision.safetensors"
    assert idx["weight_map"]["lm_head.weight"] == "model-00001.safetensors"
    expected_total = (mlx / "model-00001.safetensors").stat().st_size + (
        mlx / "model-vision.safetensors"
    ).stat().st_size
    assert idx["metadata"]["total_size"] == expected_total
    # The text shard is untouched.
    assert np.array_equal(
        _shard_keys(mlx / "model-00001.safetensors")["lm_head.weight"],
        np.ones((4, 4), dtype=np.float16),
    )


def test_merge_is_idempotent(merge_mod, tmp_path):
    mlx = _make_mlx_snapshot(tmp_path / "mlx")
    hf = _make_hf_base(tmp_path / "hf", np.ones((4, 4), dtype=np.float16))

    rep1 = merge_mod.merge(mlx, hf)
    assert rep1["status"] == "merged"
    shard_bytes = (mlx / "model-vision.safetensors").read_bytes()
    cfg_bytes = (mlx / "config.json").read_bytes()

    rep2 = merge_mod.merge(mlx, hf)
    assert rep2["status"] == "already-merged"
    assert (mlx / "model-vision.safetensors").read_bytes() == shard_bytes
    assert (mlx / "config.json").read_bytes() == cfg_bytes


def test_force_overwrites_with_hf_copy(merge_mod, tmp_path):
    mlx = _make_mlx_snapshot(tmp_path / "mlx")
    v1 = np.ones((4, 4), dtype=np.float16)
    hf = _make_hf_base(tmp_path / "hf", v1)
    merge_mod.merge(mlx, hf)

    v2 = np.full((4, 4), 7, dtype=np.float16)
    (tmp_path / "hf" / "base.safetensors").unlink()
    _write_shard(
        tmp_path / "hf" / "base.safetensors",
        {VISION_KEY: v2, "lm.weight": np.ones((2, 2), dtype=np.float16)},
    )
    rep = merge_mod.merge(mlx, tmp_path / "hf", force=True)
    assert rep["status"] == "merged"
    assert np.array_equal(_shard_keys(mlx / "model-vision.safetensors")[VISION_KEY], v2)


def test_no_vision_tensors_in_hf_base(merge_mod, tmp_path):
    mlx = _make_mlx_snapshot(tmp_path / "mlx")
    hf = tmp_path / "hf"
    hf.mkdir()
    _write_shard(
        hf / "base.safetensors", {"lm.weight": np.ones((2, 2), dtype=np.float16)}
    )
    (hf / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    rep = merge_mod.merge(mlx, hf)
    assert rep["hf_vision_keys"] == 0
    assert rep["status"] == "no-vision-tensors-in-hf-base"
    assert not (mlx / "model-vision.safetensors").exists()
    assert merge_mod.main([str(mlx), str(hf)]) == 1


def test_prefix_selection(merge_mod, tmp_path):
    mlx = _make_mlx_snapshot(tmp_path / "mlx")
    hf = tmp_path / "hf"
    hf.mkdir()
    _write_shard(
        hf / "base.safetensors",
        {
            "vision_tower.blocks.0.weight": np.ones((2, 2), dtype=np.float16),
            "model.visual.blocks.0.weight": np.ones((2, 2), dtype=np.float16),
            "lm.weight": np.ones((2, 2), dtype=np.float16),
        },
    )
    (hf / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    rep = merge_mod.merge(mlx, hf, prefixes=("vision_tower.",))
    assert rep["hf_vision_keys"] == 1
    got = _shard_keys(mlx / "model-vision.safetensors")
    assert set(got) == {"vision_tower.blocks.0.weight"}
