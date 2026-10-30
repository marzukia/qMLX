# E2E Vision × KV Acceptance (issue #56)

Manual-gate checklist for the PR #60 acceptance draft. The core (media_digest
cache keying, vision-capable MLX builds, image-aware disk KV, engine wiring)
lives on `feat/vision-kv-cache-56`; the verification split below is what a
**hydrogen CI host (no Metal)** can prove vs. what needs a **Metal (or
mlx-rocm) host**.

## The four acceptance items

| # | Item | CI-verifiable (hydrogen) | Manual gate (Metal) |
|---|------|-------------------------|--------------------|
| 1 | `qmlx serve <vlm-mlx> --enable-prefix-cache` (no `--text-only`) handles image+text chat E2E | CLI flag parsing, vision-build auto-detection, `start()` dispatch, chat-route image gate, `media_key` threading `generate()`/`stream_generate()` → `add_request` | real VLM forward + chat completion |
| 2 | prefix-cache hits on repeated multimodal turns; no stale reuse on image change | content-index media bucketing (same image + extended prompt = hit; changed image = miss; text/image buckets never cross-match) | real multi-turn conversation, `qmlx_prefix_cache_*` / `cached_tokens` |
| 3 | disk KV checkpoints round-trip with image-aware keys | sidecar carries `media_key`; restart `build_content_index` re-adds to the same bucket (mock frames, no real weights) | real checkpoint write/restore, `qmlx_kv_restore_total{result="hit"}` |
| 4 | MTP interaction: at minimum MTP-off with images | `spec_decode="none"` + image request threads `media_key`; MTP-on keys the same bucket | MTP-on + image (stretch), MTP-off + image (required) |

## What CI verifies (no Metal required)

Run on any host (hydrogen passes these with the CPU `mlx` build):

```bash
python -m pytest \
  tests/test_media_digest.py \
  tests/test_image_aware_cache_keys.py \
  tests/test_vision_kv_e2e_wiring.py \
  tests/test_tmax_merge_vision.py \
  tests/test_disk_kv_checkpoint.py \
  tests/test_text_only_media_gate.py \
  tests/test_image_block_requires_vision_model.py \
  -q
```

Coverage map:

* `test_media_digest.py` — `media_key_for_media` determinism (same image
  bytes → same key across processes; changed bytes → different key; video
  list ordering; empty list → `""`).
* `test_image_aware_cache_keys.py` — the content index is media-bucketed:
  same tokens + same image hit, same tokens + changed image never match,
  text-only and image prompts never cross-match; the on-disk sidecar
  carries the media key so a restart's `build_content_index` re-adds image
  checkpoints to the SAME bucket (acceptance 3 at the index level); the
  scheduler store site threads `request.media_key` end to end.
* `test_vision_kv_e2e_wiring.py` — the serve-path wiring (acceptance 1):
  `--enable-prefix-cache` without `--text-only` parses with vision routing
  ON (`no_mllm=False`); `--text-only` is opt-in; `--mllm`+`--text-only`
  is rejected by the loader; a vision-capable build (local path with
  `vision_config` + `image_token_id`) auto-routes as MLLM; `start()`
  dispatches MLLM builds through the LLM engine path (the mlx_vlm MLLM
  scheduler was removed in the Qwen-only build #20 — issue #56 serves
  vision via the LLM path + image-aware KV wiring); the chat route's
  image gate follows `engine.is_mllm`; `generate()` / `stream_generate()`
  thread `media_key_for_media(images, videos)` into `add_request`;
  repeated-turn hit / changed-image miss at the scheduler persist level
  (acceptance 2); scheduler-write → restart → rebuild round-trip
  (acceptance 3); MTP-off with images and MTP-on same-bucket keying
  (acceptance 4).
* `test_tmax_merge_vision.py` — the vision-tower merge tool
  (`scripts/tmax_merge_vision.py`) produces a vision-capable snapshot
  (vision shard appended, config restored, idempotent).
* `test_disk_kv_checkpoint.py` — the disk KV checkpoint tier (write,
  verify, quarantine, soft-evict, content index) — including the
  empty-media-bucket cleanup so evicted keys don't accumulate.

## Manual gates (Metal / mlx-rocm host)

### Gate 0 — build a vision-capable MLX snapshot

```bash
# From a text-only (or vision-stripped) MLX snapshot + an HF base that
# carries the vision tower:
python scripts/tmax_merge_vision.py <mlx_snapshot_dir> <hf_base_dir>
# Verify:
python -c "import json,sys; c=json.load(open(sys.argv[1]+'/config.json')); \
  assert 'vision_config' in c and c['architectures'][0].endswith('ForConditionalGeneration'), c['architectures']; \
  print('vision-capable OK')" <mlx_snapshot_dir>
```

### Gate 1 — serve + image+text chat E2E (acceptance 1)

```bash
qmlx serve <vlm-mlx-snapshot> --enable-prefix-cache
# NOTE: no --text-only. The build auto-routes as MLLM; the engine loads
# through the LLM path (mlx_lm) and the #56 wiring handles the vision
# forward. Watch the startup log for:
#   "MLLM model detected (...) — serving via the LLM engine path"
```

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "<served-name>",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "file:///path/to/cat.png"}}
  ]}],
  "max_tokens": 64
}'
# PASS: HTTP 200, a coherent answer about the image (not a 400
# "does not support image inputs", not a crash at model load).
```

### Gate 2 — repeated multimodal turns: hit, then no stale reuse (acceptance 2)

```bash
# Turn 1: image A + question.
# Turn 2: SAME image A, extended conversation (append an assistant turn +
#         a new user question).
# Turn 3: image B (different content, same token shape), same question.
# After each turn, check:
curl -s localhost:8000/metrics | grep -E 'qmlx_prefix_cache|qmlx_kv_restore'
# PASS: turn 2 shows a prefix-cache hit (qmlx_prefix_cache hit counter
#       increments / response reports cached_tokens > 0);
#       turn 3 does NOT reuse turn 1/2 KV (no hit for the changed
#       image — a fresh prefill runs).
```

### Gate 3 — disk KV checkpoint round-trip (acceptance 3)

```bash
# Disk checkpointing is on by default (--kv-disk-checkpoint-interval 256,
# --enable-disk-kv-restore). After the Gate 2 turns, restart the server:
#   (Ctrl-C, then re-run the same qmlx serve command)
# Re-send turn 2's prompt (same image A).
curl -s localhost:8000/metrics | grep 'qmlx_kv_restore_total'
# PASS: qmlx_kv_restore_total{result="hit"} >= 1 after the restart
#       (the checkpoint written before the restart is byte-verified and
#       restored from ~/.cache/qmlx/kv_checkpoints/); the changed-image
#       prompt still misses (no stale reuse across the restart).
```

### Gate 4 — MTP-off with images (acceptance 4)

```bash
# MTP off is the default for non-MTP builds; make it explicit:
qmlx serve <vlm-mlx-snapshot> --enable-prefix-cache --spec-decode none
# Repeat the Gate 1 chat request.
# PASS: the image request completes and its disk checkpoint lands in the
#       image's media bucket (visible in the qmlx_kv_restore_total
#       hit/miss split across changed vs. repeated images).
# Stretch (MTP-on): --spec-decode mtp on an MTP-capable VLM build — the
#       media bucket keying is orthogonal to the spec-decode mode.
```

## Reporting

When running the manual gates, record: host (chip + mlx build), model
snapshot, the four gate PASS/FAIL lines, and the
`qmlx_kv_restore_total` / `qmlx_prefix_cache_*` counter deltas. A gate
fails if: the server crashes at load, an image request 400s on a vision
build, a repeated turn misses, a changed turn hits, or a restart loses
the checkpoint (no hit after rebuild).
